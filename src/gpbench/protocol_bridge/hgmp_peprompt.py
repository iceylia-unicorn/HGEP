from __future__ import annotations
import torch.optim as optim
from protocols.hgmp.pretrain_legacy import EarlyStopping
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import dgl
import torch
import torch.nn as nn

from protocols.hgmp.pretrain_legacy import (
    GraphCL,
    HDGI,
    PreTrain as LegacyPreTrain,
)
from protocols.hgmp.prompt_legacy import HGNN
from protocols.hgmp.utils_legacy import load_data4pretrain

from pathlib import Path

PEPROMPT_PRETRAIN_DIR = Path("artifacts/checkpoints/peprompt_hgmp/pretrain")
PEPROMPT_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PEPromptRelationConfig:
    mode: str = "mul"          # 边提示注入方式："mul" 表示逐元素乘，"add" 表示逐元素加
    alpha: float = 0.5         # 关系提示聚合后的残差注入强度
    dropout: float = 0.1       # 关系提示层输出后的 dropout 比例
    use_ln: bool = True        # 是否对每种节点类型的输出使用 LayerNorm
    aggr: str = "mean"         # 边消息聚合方式："mean" 或 "sum"
    edge_feature_dim: int = 0  # 输入边特征维度，这里对应 Laplacian PE 差值维度
    edge_feature_name: str = "peprompt_edge_feat"  # 图中边特征保存时使用的字段名
    edge_prompt_hidden: int = 128  # 将边 PE 映射到提示向量时，中间 MLP 的隐藏层维度
    # ---- hop-decomposed compensation (Module 1→2 bridge) ----
    fusion_mode: str = "none"  # "none" = PE-only; "edge_type" = PE + canonical edge type embedding
    ctx_dim: int = 0           # 丢弃邻居上下文的特征维度，必须 == 全局节点特征维度
    generator_hidden: int = 128  # hop_decoupled 模式下融合 MLP 的隐藏层维度
    metapath_count: int = 0
    metapath_embed_dim: int = 16
    graph_summary_dim: int = 0
    metapath_pos_dim: int = 0
    metapath_pos_feature_name: str = "peprompt_metapath_pos_feat"
    basis_count: int = 4
    onehop_center_fusion: bool = False


class PEPromptRelation(nn.Module):
    """
    Spectral-only relation prompt:
    - no per-type-pair global prompt parameters
    - one edge prompt per edge, produced directly from Laplacian PE features
    - aggregate prompted messages to dst nodes and inject them residually
    """

    def __init__(
        self,
        metadata: Tuple[Iterable[str], Iterable[Tuple[str, str, str]]],
        dim: int,
        mode: str = "mul",
        alpha: float = 0.5,
        dropout: float = 0.1,
        use_ln: bool = True,
        aggr: str = "mean",
        edge_feature_dim: int = 0,
        edge_feature_name: str = "peprompt_edge_feat",
        edge_prompt_hidden: int = 128,
        fusion_mode: str = "none",
        ctx_dim: int = 0,
        generator_hidden: int = 128,
        metapath_count: int = 0,
        metapath_embed_dim: int = 16,
        graph_summary_dim: int = 0,
        metapath_pos_dim: int = 0,
        metapath_pos_feature_name: str = "peprompt_metapath_pos_feat",
        basis_count: int = 4,
        onehop_center_fusion: bool = False,
    ):
        """
        Args:
            metadata: 异构图元信息，格式为 (节点类型列表, 规范边类型列表)。
            dim: 目标提示向量维度，通常与节点隐藏维度一致。
            mode: 边提示注入方式；"mul" 为 `h_j * P_edge`，"add" 为 `h_j + P_edge`。
            alpha: 聚合后的关系提示残差系数，控制注入强度。
            dropout: 关系提示输出后的 dropout 比例。
            use_ln: 是否按节点类型对输出施加 LayerNorm。
            aggr: 对入边消息的聚合方式；支持 "mean" 和 "sum"。
            edge_feature_dim: 输入边特征维度，即 Laplacian PE 差值的维度。
            edge_feature_name: 从 DGL 边数据中读取 PE 边特征时使用的键名。
            edge_prompt_hidden: 边提示 MLP 的隐藏层维度。
            fusion_mode: "none" = PE-only; "edge_type" = PE + canonical edge type embedding;
                "hop_decoupled" = PE + dropped ctx + h_src + h_dst;
                "onehop_ctx" = 1-hop type-pooled dropped ctx + h_src + h_dst;
                "type_ctx" = one pooled dropped ctx per destination type per subgraph;
                "graph_summary" = one shared graph-level metapath summary for all edges;
                "graph_summary_basis" = use graph summary as selector over prompt bases.
                "metapath_pos" = PE + per-edge metapath-position support.
            ctx_dim: 丢弃邻居上下文的特征维度，仅 hop_decoupled 模式下使用。
            generator_hidden: hop_decoupled 模式下融合 MLP 的隐藏层维度。
        """
        super().__init__()
        if mode not in {"mul", "add"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if aggr not in {"mean", "sum"}:
            raise ValueError(f"Unsupported aggr: {aggr}")
        if fusion_mode not in {"none", "edge_type", "metapath_pos", "hop_decoupled", "onehop_ctx", "type_ctx", "graph_summary", "graph_summary_basis"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if int(edge_feature_dim or 0) <= 0:
            raise ValueError("PEPromptRelation requires positive edge_feature_dim.")

        node_types, edge_types = metadata
        self.mode = mode
        self.alpha = alpha
        self.aggr = aggr
        self.edge_feature_dim = int(edge_feature_dim or 0)
        self.edge_feature_name = edge_feature_name
        self.fusion_mode = fusion_mode
        self.ctx_dim = int(ctx_dim or 0)
        self.ctx_stats_dim = 4
        self.metapath_count = int(metapath_count or 0)
        self.metapath_embed_dim = int(metapath_embed_dim or 0)
        self.graph_summary_dim = int(graph_summary_dim or 0)
        self.metapath_pos_dim = int(metapath_pos_dim or 0)
        self.metapath_pos_feature_name = str(metapath_pos_feature_name)
        self.basis_count = int(basis_count or 0)
        self.onehop_center_fusion = bool(onehop_center_fusion)
        self.edge_type_to_id = {tuple(etype): idx for idx, etype in enumerate(edge_types)}
        self.edge_type_embed_dim = int(self.metapath_embed_dim or 8)

        self.drop = nn.Dropout(dropout)
        self.ln = nn.ModuleDict(
            {
                nt: (nn.LayerNorm(dim) if use_ln else nn.Identity())
                for nt in node_types
            }
        )
        self.edge_prompt_selector = None
        self.prompt_basis = None

        # Build MLP with appropriate input dimension based on fusion mode.
        if fusion_mode == "hop_decoupled" and self.ctx_dim > 0:
            if self.metapath_count > 0 and self.metapath_embed_dim > 0:
                self.metapath_embedding = nn.Embedding(self.metapath_count, self.metapath_embed_dim)
            else:
                self.metapath_embedding = None
                self.metapath_embed_dim = 0
            if self.edge_type_to_id:
                self.edge_type_embedding = nn.Embedding(len(self.edge_type_to_id), self.edge_type_embed_dim)
            else:
                self.edge_type_embedding = None
                self.edge_type_embed_dim = 0
            ctx_token_dim = self.ctx_dim + self.ctx_stats_dim + self.metapath_embed_dim + self.edge_type_embed_dim
            self.ctx_gate_mlp = nn.Sequential(
                nn.LayerNorm(ctx_token_dim),
                nn.Linear(ctx_token_dim, 1),
            )
            self.ctx_value_mlp = nn.Sequential(
                nn.LayerNorm(ctx_token_dim),
                nn.Linear(ctx_token_dim, generator_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(generator_hidden, self.ctx_dim),
            )
            # Input: [h_src || h_dst || pe || dropped_ctx]
            mlp_input_dim = 2 * dim + self.edge_feature_dim + self.ctx_dim
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, generator_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(generator_hidden, dim),
            )
        elif fusion_mode in {"onehop_ctx", "type_ctx"} and self.ctx_dim > 0:
            mlp_input_dim = 2 * dim + self.edge_feature_dim + self.ctx_dim
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, generator_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(generator_hidden, dim),
            )
            self.metapath_embedding = None
            self.edge_type_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            if fusion_mode == "onehop_ctx" and self.onehop_center_fusion:
                self.center_ctx_proj = nn.Sequential(
                    nn.LayerNorm(self.ctx_dim),
                    nn.Linear(self.ctx_dim, generator_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(generator_hidden, dim),
                )
                self.center_ctx_gate = nn.Sequential(
                    nn.LayerNorm(2 * dim),
                    nn.Linear(2 * dim, dim),
                    nn.Sigmoid(),
                )
            else:
                self.center_ctx_proj = None
                self.center_ctx_gate = None
        elif fusion_mode == "graph_summary" and self.graph_summary_dim > 0:
            mlp_input_dim = 2 * dim + self.edge_feature_dim + self.graph_summary_dim
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, generator_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(generator_hidden, dim),
            )
            self.metapath_embedding = None
            self.edge_type_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            self.center_ctx_proj = None
            self.center_ctx_gate = None
        elif fusion_mode == "graph_summary_basis" and self.graph_summary_dim > 0:
            selector_input_dim = 2 * dim + self.edge_feature_dim + self.graph_summary_dim
            self.edge_prompt_mlp = None
            self.edge_prompt_selector = nn.Sequential(
                nn.LayerNorm(selector_input_dim),
                nn.Linear(selector_input_dim, generator_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(generator_hidden, self.basis_count),
            )
            self.prompt_basis = nn.Parameter(torch.empty(self.basis_count, dim))
            nn.init.xavier_uniform_(self.prompt_basis)
            self.metapath_embedding = None
            self.edge_type_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            self.center_ctx_proj = None
            self.center_ctx_gate = None
        elif fusion_mode == "edge_type":
            if self.edge_type_to_id:
                self.edge_type_embedding = nn.Embedding(len(self.edge_type_to_id), self.edge_type_embed_dim)
            else:
                self.edge_type_embedding = None
                self.edge_type_embed_dim = 0
            mlp_input_dim = self.edge_feature_dim + self.edge_type_embed_dim
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, edge_prompt_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_prompt_hidden, dim),
            )
            self.metapath_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            self.center_ctx_proj = None
            self.center_ctx_gate = None
            self.edge_prompt_selector = None
            self.prompt_basis = None
        elif fusion_mode == "metapath_pos" and self.metapath_pos_dim > 0:
            mlp_input_dim = self.edge_feature_dim + self.metapath_pos_dim
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, edge_prompt_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_prompt_hidden, dim),
            )
            self.metapath_embedding = None
            self.edge_type_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            self.center_ctx_proj = None
            self.center_ctx_gate = None
            self.edge_prompt_selector = None
            self.prompt_basis = None
        elif self.edge_feature_dim > 0:
            # PE-only mode (original)
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(self.edge_feature_dim),
                nn.Linear(self.edge_feature_dim, edge_prompt_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_prompt_hidden, dim),
            )
            self.metapath_embedding = None
            self.edge_type_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            self.center_ctx_proj = None
            self.center_ctx_gate = None
            self.edge_prompt_selector = None
            self.prompt_basis = None
        else:
            self.edge_prompt_mlp = None
            self.metapath_embedding = None
            self.edge_type_embedding = None
            self.ctx_gate_mlp = None
            self.ctx_value_mlp = None
            self.center_ctx_proj = None
            self.center_ctx_gate = None
            self.edge_prompt_selector = None
            self.prompt_basis = None

    @property
    def uses_edge_features(self) -> bool:
        return self.edge_prompt_mlp is not None or self.edge_prompt_selector is not None

    def _match_last_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.size(-1) == dim:
            return x
        if x.size(-1) > dim:
            return x[..., :dim]
        pad_shape = list(x.shape)
        pad_shape[-1] = dim - x.size(-1)
        return torch.cat([x, x.new_zeros(pad_shape)], dim=-1)

    def _node_graph_ids(self, graph, ntype: str, device) -> torch.Tensor:
        counts = graph.batch_num_nodes(ntype)
        if not isinstance(counts, torch.Tensor):
            counts = torch.as_tensor(counts, dtype=torch.long)
        counts = counts.to(device=device, dtype=torch.long).view(-1)
        graph_ids = torch.arange(int(counts.numel()), device=device, dtype=torch.long)
        return torch.repeat_interleave(graph_ids, counts)

    def _batched_virtual_metapath_state(self, graph, device, dtype):
        if graph is None:
            return None

        for ntype in graph.ntypes:
            ctx_node_data = graph.nodes[ntype].data.get("dropped_metapath_ctx")
            stats_node_data = graph.nodes[ntype].data.get("dropped_metapath_stats")
            if ctx_node_data is None or stats_node_data is None:
                continue

            ctx = ctx_node_data.to(device=device, dtype=dtype)
            stats = stats_node_data.to(device=device, dtype=dtype)
            if ctx.dim() != 3 or stats.dim() != 3 or ctx.size(0) == 0:
                continue

            owner_graph_ids = self._node_graph_ids(graph, ntype, device)
            center_mask = stats.abs().sum(dim=(-1, -2)) > 0
            if not center_mask.any():
                continue

            num_graphs = int(owner_graph_ids.max().item()) + 1 if owner_graph_ids.numel() > 0 else 0
            ctx_by_graph = ctx.new_zeros((num_graphs, ctx.size(1), ctx.size(2)))
            stats_by_graph = stats.new_zeros((num_graphs, stats.size(1), stats.size(2)))
            ctx_by_graph[owner_graph_ids[center_mask]] = ctx[center_mask]
            stats_by_graph[owner_graph_ids[center_mask]] = stats[center_mask]
            return {
                "owner_ntype": ntype,
                "ctx_by_graph": ctx_by_graph,
                "stats_by_graph": stats_by_graph,
            }
        return None

    def _metapath_ctx_per_edge(self, graph, virtual_state, etype, src_t: str, src: torch.Tensor, device, dtype):
        if graph is None or self.ctx_gate_mlp is None or self.ctx_value_mlp is None:
            return None
        if virtual_state is None:
            return None

        keep_mask_node_data = graph.nodes[src_t].data.get("dropped_metapath_keep_mask")
        if keep_mask_node_data is None:
            return None

        unique_src, inverse = torch.unique(src, sorted=False, return_inverse=True)
        src_graph_ids = self._node_graph_ids(graph, src_t, device)[unique_src]
        ctx_mp = virtual_state["ctx_by_graph"][src_graph_ids]
        stats = virtual_state["stats_by_graph"][src_graph_ids]
        if ctx_mp.dim() != 3 or ctx_mp.size(1) == 0:
            return None
        ctx_mp = self._match_last_dim(ctx_mp, self.ctx_dim)
        stats = self._match_last_dim(stats, self.ctx_stats_dim)
        keep_mask = keep_mask_node_data[unique_src].to(device=device, dtype=torch.bool)
        chunk_size = 256
        ctx_chunks = []
        mp_emb_base = None
        if self.metapath_embedding is not None:
            if ctx_mp.size(1) <= self.metapath_embedding.num_embeddings:
                mp_ids = torch.arange(ctx_mp.size(1), device=device)
                mp_emb_base = self.metapath_embedding(mp_ids).to(dtype=dtype)
            else:
                mp_emb_base = None

        etype_emb_base = None
        if self.edge_type_embedding is not None:
            etype_id = self.edge_type_to_id.get(tuple(etype))
            if etype_id is not None:
                etype_ids = torch.full(
                    (ctx_mp.size(1),),
                    int(etype_id),
                    device=device,
                    dtype=torch.long,
                )
                etype_emb_base = self.edge_type_embedding(etype_ids).to(dtype=dtype)

        for start in range(0, ctx_mp.size(0), chunk_size):
            end = min(start + chunk_size, ctx_mp.size(0))
            ctx_chunk = ctx_mp[start:end]
            stats_chunk = stats[start:end]
            keep_mask_chunk = keep_mask[start:end]

            token_parts = [ctx_chunk, stats_chunk]
            if self.metapath_embedding is not None:
                if mp_emb_base is not None:
                    mp_emb = mp_emb_base.unsqueeze(0).expand(ctx_chunk.size(0), -1, -1)
                else:
                    mp_emb = ctx_chunk.new_zeros(ctx_chunk.size(0), ctx_chunk.size(1), self.metapath_embed_dim)
                token_parts.append(mp_emb)
            if self.edge_type_embedding is not None:
                if etype_emb_base is not None:
                    etype_emb = etype_emb_base.unsqueeze(0).expand(ctx_chunk.size(0), -1, -1)
                else:
                    etype_emb = ctx_chunk.new_zeros(ctx_chunk.size(0), ctx_chunk.size(1), self.edge_type_embed_dim)
                token_parts.append(etype_emb)

            token = torch.cat(token_parts, dim=-1)
            has_state = stats_chunk.abs().sum(dim=-1) > 0
            valid = keep_mask_chunk & has_state
            gate_logits = self.ctx_gate_mlp(token).squeeze(-1)
            gate_logits = gate_logits.masked_fill(~valid, torch.finfo(gate_logits.dtype).min)
            alpha = torch.softmax(gate_logits, dim=-1)
            has_valid = valid.any(dim=-1, keepdim=True)
            alpha = torch.where(has_valid, alpha, torch.zeros_like(alpha))

            values = self.ctx_value_mlp(token)
            ctx_chunks.append((alpha.unsqueeze(-1) * values).sum(dim=1))

        ctx_per_src = torch.cat(ctx_chunks, dim=0)
        return ctx_per_src[inverse]

    def _onehop_ctx_per_edge(self, graph, src_t: str, dst_t: str, src: torch.Tensor, device, dtype):
        if graph is None or self.ctx_dim <= 0:
            return None
        ctx_key = f"dropped_onehop_ctx_{dst_t}"
        ctx_node_data = graph.nodes[src_t].data.get(ctx_key)
        if ctx_node_data is None:
            return None
        ctx_per_edge = ctx_node_data[src].to(device=device, dtype=dtype)
        return self._match_last_dim(ctx_per_edge, self.ctx_dim)

    def _batched_type_ctx_state(self, graph, device, dtype):
        if graph is None:
            return None

        state = {}
        for ntype in graph.ntypes:
            owner_graph_ids = self._node_graph_ids(graph, ntype, device)
            if owner_graph_ids.numel() == 0:
                continue
            num_graphs = int(owner_graph_ids.max().item()) + 1
            for key, value in graph.nodes[ntype].data.items():
                if not key.startswith("dropped_ctx_"):
                    continue
                dst_t = key[len("dropped_ctx_"):]
                ctx = value.to(device=device, dtype=dtype)
                if ctx.dim() != 2 or ctx.size(0) == 0:
                    continue
                center_mask = ctx.abs().sum(dim=-1) > 0
                if not center_mask.any():
                    continue
                ctx_by_graph = ctx.new_zeros((num_graphs, ctx.size(-1)))
                ctx_by_graph[owner_graph_ids[center_mask]] = ctx[center_mask]
                state[dst_t] = ctx_by_graph
        return state or None

    def _type_ctx_per_edge(self, type_ctx_state, graph, src_t: str, dst_t: str, src: torch.Tensor, device, dtype):
        if graph is None or self.ctx_dim <= 0 or not type_ctx_state or dst_t not in type_ctx_state:
            return None
        src_graph_ids = self._node_graph_ids(graph, src_t, device)[src]
        ctx_per_edge = type_ctx_state[dst_t][src_graph_ids]
        return self._match_last_dim(ctx_per_edge, self.ctx_dim)

    def _batched_graph_summary_state(self, graph, device, dtype):
        if graph is None:
            return None

        for ntype in graph.ntypes:
            summary_node_data = graph.nodes[ntype].data.get("graph_metapath_summary")
            if summary_node_data is None:
                continue
            summary = summary_node_data.to(device=device, dtype=dtype)
            if summary.dim() != 2 or summary.size(0) == 0:
                continue
            owner_graph_ids = self._node_graph_ids(graph, ntype, device)
            center_mask = summary.abs().sum(dim=-1) > 0
            if not center_mask.any():
                continue
            num_graphs = int(owner_graph_ids.max().item()) + 1 if owner_graph_ids.numel() > 0 else 0
            summary_by_graph = summary.new_zeros((num_graphs, summary.size(-1)))
            summary_by_graph[owner_graph_ids[center_mask]] = summary[center_mask]
            return summary_by_graph
        return None

    def _graph_summary_per_edge(self, graph_summary_state, graph, src_t: str, src: torch.Tensor, device, dtype):
        if graph is None or self.graph_summary_dim <= 0 or graph_summary_state is None:
            return None
        src_graph_ids = self._node_graph_ids(graph, src_t, device)[src]
        summary_per_edge = graph_summary_state[src_graph_ids].to(device=device, dtype=dtype)
        return self._match_last_dim(summary_per_edge, self.graph_summary_dim)

    def _metapath_pos_per_edge(self, graph, etype, device, dtype):
        if graph is None or self.metapath_pos_dim <= 0:
            return None
        edge_data = graph.edges[etype].data.get(self.metapath_pos_feature_name)
        if edge_data is None:
            return None
        edge_data = edge_data.to(device=device, dtype=dtype)
        return self._match_last_dim(edge_data, self.metapath_pos_dim)

    def _apply_onehop_center_fusion(self, x_dict: Dict[str, torch.Tensor], graph) -> Dict[str, torch.Tensor]:
        if (
            graph is None
            or not self.onehop_center_fusion
            or self.center_ctx_proj is None
            or self.center_ctx_gate is None
        ):
            return x_dict

        out_dict = dict(x_dict)
        for ntype, x in x_dict.items():
            ctx_keys = sorted(
                key for key in graph.nodes[ntype].data.keys()
                if key.startswith("dropped_onehop_ctx_")
            )
            if not ctx_keys:
                continue

            ctx_list = []
            weight_list = []
            for ctx_key in ctx_keys:
                stats_key = ctx_key.replace("dropped_onehop_ctx_", "dropped_onehop_stats_")
                ctx_mat = graph.nodes[ntype].data.get(ctx_key)
                stats_mat = graph.nodes[ntype].data.get(stats_key)
                if ctx_mat is None or stats_mat is None:
                    continue
                ctx_list.append(self._match_last_dim(ctx_mat.to(device=x.device, dtype=x.dtype), self.ctx_dim))
                stats = stats_mat.to(device=x.device, dtype=x.dtype)
                if stats.dim() != 2 or stats.size(-1) == 0:
                    weight_list.append(torch.zeros((x.size(0),), device=x.device, dtype=x.dtype))
                else:
                    weight_list.append(stats[:, 1])

            if not ctx_list:
                continue

            ctx_stack = torch.stack(ctx_list, dim=1)  # [N, T, ctx_dim]
            weight_stack = torch.stack(weight_list, dim=1)  # [N, T]
            valid = ctx_stack.abs().sum(dim=-1) > 0
            logits = weight_stack.masked_fill(~valid, torch.finfo(weight_stack.dtype).min)
            alpha = torch.softmax(logits, dim=-1)
            has_valid = valid.any(dim=-1, keepdim=True)
            alpha = torch.where(has_valid, alpha, torch.zeros_like(alpha))
            pooled_ctx = (alpha.unsqueeze(-1) * ctx_stack).sum(dim=1)

            proj = self.center_ctx_proj(pooled_ctx)
            gate = self.center_ctx_gate(torch.cat([x, proj], dim=-1))
            out_dict[ntype] = x + gate * proj

        return out_dict

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_feature_dict: Dict[Tuple[str, str, str], torch.Tensor] | None = None,
        graph=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: 各节点类型的节点表示字典，键为节点类型，值为节点特征矩阵。
            edge_index_dict: 各规范边类型的边索引字典，形状为 [2, num_edges]。
            edge_feature_dict: 各规范边类型对应的边特征字典，这里应为 Laplacian PE 差值。
            graph: (hop_decoupled only) DGL heterograph used to read ``dropped_ctx_*`` from nodes.

        Returns:
            注入 PE 边提示后的各节点类型表示。
        """
        device = next(iter(x_dict.values())).device

        agg_dict = {
            ntype: torch.zeros_like(x)
            for ntype, x in x_dict.items()
        }
        deg_dict = {
            ntype: torch.zeros((x.size(0), 1), device=device, dtype=x.dtype)
            for ntype, x in x_dict.items()
        }

        use_onehop_ctx = (
            self.fusion_mode == "onehop_ctx"
            and self.ctx_dim > 0
            and graph is not None
        )
        use_type_ctx = (
            self.fusion_mode == "type_ctx"
            and self.ctx_dim > 0
            and graph is not None
        )
        use_graph_summary = (
            self.fusion_mode == "graph_summary"
            and self.graph_summary_dim > 0
            and graph is not None
        )
        use_graph_summary_basis = (
            self.fusion_mode == "graph_summary_basis"
            and self.graph_summary_dim > 0
            and self.edge_prompt_selector is not None
            and self.prompt_basis is not None
            and graph is not None
        )
        use_metapath_pos = (
            self.fusion_mode == "metapath_pos"
            and self.metapath_pos_dim > 0
            and graph is not None
        )
        use_hop_decoupled = (
            self.fusion_mode == "hop_decoupled"
            and self.ctx_dim > 0
            and graph is not None
        )
        if use_onehop_ctx:
            x_dict = self._apply_onehop_center_fusion(x_dict, graph)
        virtual_state = None
        if use_hop_decoupled:
            virtual_state = self._batched_virtual_metapath_state(graph, device, next(iter(x_dict.values())).dtype)
        type_ctx_state = None
        if use_type_ctx:
            type_ctx_state = self._batched_type_ctx_state(graph, device, next(iter(x_dict.values())).dtype)
        graph_summary_state = None
        if use_graph_summary or use_graph_summary_basis:
            graph_summary_state = self._batched_graph_summary_state(graph, device, next(iter(x_dict.values())).dtype)

        for (src_t, rel_t, dst_t), edge_index in edge_index_dict.items():
            src, dst = edge_index
            if src.numel() == 0:
                continue

            if edge_feature_dict is None:
                raise ValueError("PEPromptRelation requires edge_feature_dict, but received None.")

            edge_features = edge_feature_dict.get((src_t, rel_t, dst_t))
            if edge_features is None:
                raise ValueError(
                    f"PEPromptRelation missing edge features for etype={(src_t, rel_t, dst_t)}."
                )

            if use_hop_decoupled:
                # ---- hop-decomposed compensation ----
                # Gather [h_src || h_dst || PE || dropped_ctx] per edge
                h_src = x_dict[src_t][src]                              # [E, dim]
                h_dst = x_dict[dst_t][dst]                              # [E, dim]
                pe = edge_features.to(device)                            # [E, pe_dim]

                ctx_per_edge = self._metapath_ctx_per_edge(
                    graph,
                    virtual_state,
                    (src_t, rel_t, dst_t),
                    src_t,
                    src,
                    device,
                    pe.dtype,
                )
                if ctx_per_edge is None:
                    ctx_key = f"dropped_ctx_{dst_t}"
                    ctx_node_data = graph.nodes[src_t].data.get(ctx_key)
                    if ctx_node_data is not None:
                        ctx_per_edge = ctx_node_data[src].to(device=device, dtype=pe.dtype)
                        ctx_per_edge = self._match_last_dim(ctx_per_edge, self.ctx_dim)
                    else:
                        ctx_per_edge = torch.zeros(
                            pe.shape[0], self.ctx_dim, device=device, dtype=pe.dtype,
                        )

                mlp_input = torch.cat([h_src, h_dst, pe, ctx_per_edge], dim=-1)
                p = self.edge_prompt_mlp(mlp_input)
            elif use_onehop_ctx:
                h_src = x_dict[src_t][src]
                h_dst = x_dict[dst_t][dst]
                pe = edge_features.to(device)
                ctx_per_edge = self._onehop_ctx_per_edge(graph, src_t, dst_t, src, device, pe.dtype)
                if ctx_per_edge is None:
                    ctx_per_edge = torch.zeros(
                        pe.shape[0], self.ctx_dim, device=device, dtype=pe.dtype,
                    )
                mlp_input = torch.cat([h_src, h_dst, pe, ctx_per_edge], dim=-1)
                p = self.edge_prompt_mlp(mlp_input)
            elif use_type_ctx:
                h_src = x_dict[src_t][src]
                h_dst = x_dict[dst_t][dst]
                pe = edge_features.to(device)
                ctx_per_edge = self._type_ctx_per_edge(type_ctx_state, graph, src_t, dst_t, src, device, pe.dtype)
                if ctx_per_edge is None:
                    ctx_per_edge = torch.zeros(
                        pe.shape[0], self.ctx_dim, device=device, dtype=pe.dtype,
                    )
                mlp_input = torch.cat([h_src, h_dst, pe, ctx_per_edge], dim=-1)
                p = self.edge_prompt_mlp(mlp_input)
            elif use_graph_summary:
                h_src = x_dict[src_t][src]
                h_dst = x_dict[dst_t][dst]
                pe = edge_features.to(device)
                summary_per_edge = self._graph_summary_per_edge(
                    graph_summary_state, graph, src_t, src, device, pe.dtype
                )
                if summary_per_edge is None:
                    summary_per_edge = torch.zeros(
                        pe.shape[0], self.graph_summary_dim, device=device, dtype=pe.dtype,
                    )
                mlp_input = torch.cat([h_src, h_dst, pe, summary_per_edge], dim=-1)
                p = self.edge_prompt_mlp(mlp_input)
            elif use_graph_summary_basis:
                h_src = x_dict[src_t][src]
                h_dst = x_dict[dst_t][dst]
                pe = edge_features.to(device)
                summary_per_edge = self._graph_summary_per_edge(
                    graph_summary_state, graph, src_t, src, device, pe.dtype
                )
                if summary_per_edge is None:
                    summary_per_edge = torch.zeros(
                        pe.shape[0], self.graph_summary_dim, device=device, dtype=pe.dtype,
                    )
                selector_input = torch.cat([h_src, h_dst, pe, summary_per_edge], dim=-1)
                alpha = torch.softmax(self.edge_prompt_selector(selector_input), dim=-1)
                p = alpha @ self.prompt_basis.to(device=device, dtype=pe.dtype)
            elif self.fusion_mode == "edge_type":
                pe = edge_features.to(device)
                if self.edge_type_embedding is not None:
                    etype_id = self.edge_type_to_id.get((src_t, rel_t, dst_t))
                    if etype_id is not None:
                        etype_ids = torch.full(
                            (pe.size(0),),
                            int(etype_id),
                            device=device,
                            dtype=torch.long,
                        )
                        etype_emb = self.edge_type_embedding(etype_ids).to(dtype=pe.dtype)
                    else:
                        etype_emb = pe.new_zeros(pe.size(0), self.edge_type_embed_dim)
                else:
                    etype_emb = pe.new_zeros(pe.size(0), self.edge_type_embed_dim)
                p = self.edge_prompt_mlp(torch.cat([pe, etype_emb], dim=-1))
            elif use_metapath_pos:
                pe = edge_features.to(device)
                metapath_pos = self._metapath_pos_per_edge(
                    graph,
                    (src_t, rel_t, dst_t),
                    device,
                    pe.dtype,
                )
                if metapath_pos is None:
                    metapath_pos = pe.new_zeros(pe.size(0), self.metapath_pos_dim)
                p = self.edge_prompt_mlp(torch.cat([pe, metapath_pos], dim=-1))
            else:
                p = self.edge_prompt_mlp(edge_features.to(x_dict[src_t].device))

            msg = x_dict[src_t][src]
            if self.mode == "mul":
                msg = msg * p
            else:
                msg = msg + p

            agg_dict[dst_t].index_add_(0, dst, msg)

            ones = torch.ones(
                (dst.size(0), 1),
                device=device,
                dtype=x_dict[dst_t].dtype,
            )
            deg_dict[dst_t].index_add_(0, dst, ones)

        out_dict: Dict[str, torch.Tensor] = {}
        for ntype, x in x_dict.items():
            agg = agg_dict[ntype]
            if self.aggr == "mean":
                agg = agg / deg_dict[ntype].clamp_min(1.0)

            h = x + self.alpha * agg
            h = self.drop(h)
            h = self.ln[ntype](h)
            out_dict[ntype] = h

        return out_dict


def _build_edge_index_dict(graph) -> Dict[Tuple[str, str, str], torch.Tensor]:
    edge_index_dict = {}
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype)
        edge_index_dict[etype] = torch.stack((src, dst), dim=0)
    return edge_index_dict


def _split_h_by_keys(
    h: torch.Tensor,
    keys: Iterable[str],
    sizes: Iterable[int],
) -> Dict[str, torch.Tensor]:
    out = {}
    start = 0
    for key, size in zip(keys, sizes):
        out[key] = h[start:start + size]
        start += size
    return out


class RelationInjectedPEPromptLegacyHGT(nn.Module):
    """
    Wrap protocols.hgmp.prompt_legacy.HGT without modifying it.
    Injection point:
      after each HGTConv block, before the final output projection.

    Keep legacy parameter names (lin_dict / convs / lin) so a plain HGMP-HGT
    checkpoint can still load into this wrapper with only relation prompt keys missing.
    """

    def __init__(self, base_hgt: nn.Module, relation_prompt: PEPromptRelation):
        super().__init__()
        self.lin_dict = base_hgt.lin_dict
        self.convs = base_hgt.convs
        self.lin = base_hgt.lin
        self.relation_prompt = relation_prompt

    def forward(
        self,
        targetnode: str,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_feature_dict: Dict[Tuple[str, str, str], torch.Tensor] | None = None,
        graph=None,
    ) -> Dict[str, torch.Tensor]:
        del targetnode

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = self.relation_prompt(x_dict, edge_index_dict, edge_feature_dict, graph=graph)

        x_dict = {
            node_type: self.lin(x)
            for node_type, x in x_dict.items()
        }
        return x_dict


class RelationInjectedPEPromptLegacyGCN(nn.Module):
    """
    Wrap protocols.hgmp.prompt_legacy.GCL_GCN without modifying it.

    Injection point:
      after each GraphConv block on the hidden representation.

    Keep legacy parameter names (fc_list / layers / dropout) so a plain HGMP-GCN
    checkpoint can still load into this wrapper with only relation prompt keys missing.
    """

    def __init__(self, base_gcn: nn.Module, relation_prompt: PEPromptRelation):
        super().__init__()
        self.fc_list = base_gcn.fc_list
        self.layers = base_gcn.layers
        self.dropout = base_gcn.dropout
        self.relation_prompt = relation_prompt

    def forward(
        self,
        graph,
        x_dict: Dict[str, torch.Tensor],
        edge_feature_dict: Dict[Tuple[str, str, str], torch.Tensor] | None = None,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] | None = None,
        homo_graph=None,
    ) -> Dict[str, torch.Tensor]:
        keys = list(x_dict.keys())
        sizes = [x_dict[key].shape[0] for key in keys]

        feats_emd = [x_dict[key] for key in keys]
        h_list = []
        for fc, feature in zip(self.fc_list, feats_emd):
            h_list.append(fc(feature))
        h = torch.cat(h_list, dim=0)

        if edge_index_dict is None:
            edge_index_dict = _build_edge_index_dict(graph)
        if homo_graph is None:
            homo_graph = dgl.to_homogeneous(graph)
            homo_graph = dgl.remove_self_loop(homo_graph)
            homo_graph = dgl.add_self_loop(homo_graph)

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(homo_graph, h)

            hidden_dict = _split_h_by_keys(h, keys, sizes)
            hidden_dict = self.relation_prompt(hidden_dict, edge_index_dict, edge_feature_dict, graph=graph)
            h = torch.cat([hidden_dict[key] for key in keys], dim=0)

        return _split_h_by_keys(h, keys, sizes)


class HGMPPEPromptHGNN(nn.Module):
    """
    New-file-only wrapper around protocols.hgmp.prompt_legacy.HGNN.

    We do NOT modify legacy HGNN modules. Instead, we instantiate legacy HGNN,
    then replace its encoder path with a relation-injected wrapper.

    Supported now:
    - HGT: inject after each HGTConv block
    - GCN: inject after each GraphConv block
    """

    def __init__(
        self,
        ntypes,
        metadata,
        hid_dim=None,
        out_dim=None,
        num_layer=2,
        pool=None,
        hgnn_type="HGT",
        num_heads=8,
        device=None,
        dropout=0.2,
        norm=False,
        num_etypes=None,
        input_dims=None,
        args=None,
        relation_cfg: PEPromptRelationConfig | None = None,
    ):
        """
        Args:
            ntypes: 图中所有节点类型列表。
            metadata: 异构图元信息 `(节点类型列表, 规范边类型列表)`。
            hid_dim: 编码器隐藏维度，也是关系提示维度。
            out_dim: 编码器输出维度。
            num_layer: HGNN 编码层数。
            pool: 图级池化配置，保持与底层 HGNN 接口一致。
            hgnn_type: 编码器类型，目前支持 "HGT" 和 "GCN"。
            num_heads: HGT 使用的注意力头数。
            device: 运行设备。
            dropout: 编码器内部使用的 dropout。
            norm: 是否启用底层 HGNN 的归一化配置。
            num_etypes: 边类型数量。
            input_dims: 各节点类型的输入特征维度。
            args: 原始实验参数对象，透传给底层 HGNN。
            relation_cfg: PEPrompt 的配置对象，控制纯 PE 边提示注入方式。
        """
        super().__init__()
        if hgnn_type not in {"HGT", "GCN"}:
            raise NotImplementedError("HGMPPEPromptHGNN currently supports only HGT and GCN.")

        if relation_cfg is None:
            relation_cfg = PEPromptRelationConfig()

        base_hgnn = HGNN(
            ntypes=ntypes,
            metadata=metadata,
            hid_dim=hid_dim,
            out_dim=out_dim,
            num_layer=num_layer,
            pool=pool,
            hgnn_type=hgnn_type,
            num_heads=num_heads,
            device=device,
            dropout=dropout,
            norm=norm,
            num_etypes=num_etypes,
            input_dims=input_dims,
            args=args,
        )

        self.hgnn_type = base_hgnn.hgnn_type
        relation_prompt = PEPromptRelation(
            metadata=metadata,
            dim=hid_dim,
            mode=relation_cfg.mode,
            alpha=relation_cfg.alpha,
            dropout=relation_cfg.dropout,
            use_ln=relation_cfg.use_ln,
            aggr=relation_cfg.aggr,
            edge_feature_dim=relation_cfg.edge_feature_dim,
            edge_feature_name=relation_cfg.edge_feature_name,
            edge_prompt_hidden=relation_cfg.edge_prompt_hidden,
            fusion_mode=relation_cfg.fusion_mode,
            ctx_dim=relation_cfg.ctx_dim,
            generator_hidden=relation_cfg.generator_hidden,
            metapath_count=relation_cfg.metapath_count,
            metapath_embed_dim=relation_cfg.metapath_embed_dim,
            graph_summary_dim=relation_cfg.graph_summary_dim,
            metapath_pos_dim=relation_cfg.metapath_pos_dim,
            metapath_pos_feature_name=relation_cfg.metapath_pos_feature_name,
            basis_count=relation_cfg.basis_count,
            onehop_center_fusion=relation_cfg.onehop_center_fusion,
        )

        if self.hgnn_type == "HGT":
            self.GraphConv = RelationInjectedPEPromptLegacyHGT(
                base_hgnn.GraphConv,
                relation_prompt,
            )
        elif self.hgnn_type == "GCN":
            self.GraphConv = RelationInjectedPEPromptLegacyGCN(
                base_hgnn.GraphConv,
                relation_prompt,
            )
        else:
            raise NotImplementedError(
                f"Unsupported hgnn_type in HGMPPEPromptHGNN: {self.hgnn_type}"
            )

    @property
    def relation_prompt(self) -> PEPromptRelation:
        return self.GraphConv.relation_prompt

    def forward(self, targetnode, x, edge_index=None, edge_feature_dict=None, homo_graph=None, graph=None):
        """
        Args:
            targetnode: HGT 路径下的目标节点类型；GCN 路径下复用该位置传入 graph。
            x: 节点特征字典。
            edge_index: HGT 路径下的异构边索引字典。
            edge_feature_dict: 各边类型的 PE 边特征字典。
            homo_graph: GCN 路径下的同构图，可选。
            graph: (hop_decoupled) original heterograph for reading dropped_ctx node data.
        """
        if self.hgnn_type == "HGT":
            return self.GraphConv(targetnode, x, edge_index, edge_feature_dict, graph=graph)
        if self.hgnn_type == "GCN":
            g = targetnode  # targetnode IS the graph for GCN
            x_dict = x
            return self.GraphConv(
                g,
                x_dict,
                edge_feature_dict,
                edge_index_dict=edge_index,
                homo_graph=homo_graph,
            )
        raise NotImplementedError(
            f"Unsupported hgnn_type in HGMPPEPromptHGNN.forward: {self.hgnn_type}"
        )


class PEPromptLegacyPreTrain(LegacyPreTrain):
    """
    Reuse all training / loader / GraphCL logic from protocols.hgmp.pretrain_legacy.PreTrain,
    but swap in HGMPPEPromptHGNN at construction time.
    """

    def __init__(
        self,
        args,
        ntypes,
        metadata,
        num_class,
        num_etypes,
        input_dims,
        relation_cfg: PEPromptRelationConfig | None = None,
    ):
        """
        Args:
            args: 预训练总配置，包含数据集、模型和优化超参数。
            ntypes: 图中所有节点类型列表。
            metadata: 异构图元信息 `(节点类型列表, 规范边类型列表)`。
            num_class: 下游类别数，占位保持与旧接口兼容。
            num_etypes: 规范边类型数量。
            input_dims: 各节点类型输入维度。
            relation_cfg: PEPrompt 关系提示配置。
        """
        nn.Module.__init__(self)

        self.pretext = args.pretext
        self.hgnn_type = args.hgnn_type
        self.device = args.device
        self.args = args

        if args.hgnn_type not in {"HGT", "GCN"}:
            raise NotImplementedError("PEPrompt bridge currently supports only HGT and GCN.")

        self.hgnn = HGMPPEPromptHGNN(
            ntypes=ntypes,
            metadata=metadata,
            hid_dim=args.hidden_dim,
            out_dim=args.hidden_dim,
            hgnn_type=args.hgnn_type,
            num_layer=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_etypes=num_etypes,
            input_dims=input_dims,
            args=args,
            relation_cfg=relation_cfg,
        )

        if self.pretext == 'HDGI':
            raise NotImplementedError("PEPrompt bridge has not enabled HDGI yet.")
        elif args.pretext in ["GraphCL", "SimGRACE"]:
            self.model = GraphCL(self.hgnn, hid_dim=args.hidden_dim)
        else:
            raise ValueError("pretext should be HDGI, GraphCL, or SimGRACE")
        
    def train(
        self,
        graph_batch_size,
        node_batch_size,
        dataname,
        graph_list,
        lr=0.01,
        decay=0.0001,
        epochs=100,
        aug1='dropN',
        aug2='permE',
        seed=None,
        aug_ration=None,
    ):
        """
        Args:
            graph_batch_size: 图级预训练 dataloader 的 batch size。
            node_batch_size: 节点级对比学习或编码时使用的 batch size。
            dataname: 数据集名称，用于命名 checkpoint。
            graph_list: 参与预训练的图列表。
            lr: 优化器学习率。
            decay: 权重衰减系数。
            epochs: 最大训练轮数。
            aug1: 第一种图增广方式。
            aug2: 第二种图增广方式。
            seed: 随机种子，当前接口保留以兼容旧流程。
            aug_ration: 图增广强度配置，沿用旧代码命名。
        """
        loader1, loader2 = self.get_loader(
            graph_list,
            graph_batch_size,
            aug1=aug1,
            aug2=aug2,
            aug_ratio=aug_ration,
            pretext=self.pretext,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        save_path = PEPROMPT_PRETRAIN_DIR / (
            f"{dataname}.{self.pretext}.{self.hgnn_type}"
            f".peprompt.hid{self.args.hidden_dim}.np{self.args.num_samples}.pth"
        )

        early_stopping = EarlyStopping(
            patience=30,
            verbose=True,
            save_path=str(save_path),
        )

        graph = graph_list[0]

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            if self.pretext == 'HDGI':
                train_loss = self.train_hdgi(self.model, graph, optimizer)
            elif self.pretext == 'GraphCL':
                train_loss = self.train_graphcl(
                    self.model,
                    loader1,
                    loader2,
                    optimizer,
                    node_batch_size,
                    self.args.device,
                )
            elif self.pretext == 'SimGRACE':
                train_loss = self.train_simgrace(
                    self.model,
                    loader1,
                    optimizer,
                    self.args.device,
                )
            else:
                raise ValueError("pretext should be HDGI, GraphCL, SimGRACE")

            improved = train_loss < best_loss
            if improved:
                best_loss = train_loss

            print(
                f"*** epoch: {epoch}/{epochs} | "
                f"train_loss: {train_loss:.6f} | "
                f"best_loss: {best_loss:.6f}"
                + (" | saved_best" if improved else "")
            )

            early_stopping(train_loss, self.model.hgnn)
            if early_stopping.early_stop:
                print("Early stopping!")
                break
        print(f"+++ best checkpoint path: {save_path}")


def build_peprompt_relation_cfg_from_args(args) -> PEPromptRelationConfig:
    """从命令行或实验配置对象中抽取 PEPrompt 相关参数。"""
    return PEPromptRelationConfig(
        mode=getattr(args, "relation_prompt_mode", "mul"),
        alpha=getattr(args, "relation_prompt_alpha", 0.5),
        dropout=getattr(args, "relation_prompt_dropout", 0.1),
        use_ln=getattr(args, "relation_prompt_use_ln", True),
        aggr=getattr(args, "relation_prompt_aggr", "mean"),
        edge_feature_dim=getattr(args, "peprompt_edge_feature_dim", 0),
        edge_feature_name=getattr(args, "peprompt_edge_feature_name", "peprompt_edge_feat"),
        edge_prompt_hidden=getattr(args, "peprompt_edge_prompt_hidden", 128),
        fusion_mode=getattr(args, "peprompt_fusion_mode", "none"),
        ctx_dim=getattr(args, "peprompt_ctx_dim", 0),
        generator_hidden=getattr(args, "peprompt_generator_hidden", 128),
        metapath_count=getattr(args, "peprompt_metapath_count", 0),
        metapath_embed_dim=getattr(args, "peprompt_metapath_embed_dim", 16),
        graph_summary_dim=getattr(args, "peprompt_graph_summary_dim", 0),
        metapath_pos_dim=getattr(args, "peprompt_metapath_pos_dim", 0),
        metapath_pos_feature_name=getattr(args, "peprompt_metapath_pos_feature_name", "peprompt_metapath_pos_feat"),
        basis_count=getattr(args, "peprompt_basis_count", 4),
        onehop_center_fusion=getattr(args, "peprompt_onehop_center_fusion", False),
    )


def pretrain_peprompt_legacy(args):
    """
    HGMP-aligned pretraining entry for PEPromptRelation.
    This mirrors protocols.hgmp.run_legacy.pretrain(), but swaps the model class.

    Args:
        args: 预训练入口参数，需包含数据集、模型结构、预训练轮数和学习率等配置。
    """
    batch_size = 64
    num_sample = args.num_samples

    graph_list, in_dims, num_class = load_data4pretrain(
        args.feats_type,
        args.device,
        args.dataset,
        batch_size,
        num_sample,
    )

    graph = graph_list[0]
    metadata, ntypes = graph.canonical_etypes, graph.ntypes
    num_etypes = len(metadata) + 1
    metadata = (ntypes, metadata)

    # keep the same behavior as legacy HGMP runner
    num_class = args.num_class

    relation_cfg = build_peprompt_relation_cfg_from_args(args)
    pt = PEPromptLegacyPreTrain(
        ntypes=ntypes,
        args=args,
        metadata=metadata,
        num_class=num_class,
        num_etypes=num_etypes,
        input_dims=in_dims,
        relation_cfg=relation_cfg,
    )

    pt.model.to(args.device)
    pt.train(
        dataname=args.dataset,
        graph_list=graph_list,
        graph_batch_size=10,
        lr=args.pre_lr,
        decay=0.0001,
        epochs=args.pre_epoch,
        aug1="maskN",
        aug2="permE",
        node_batch_size=batch_size,
        seed=args.seed,
        aug_ration=args.aug_ration,
    )
