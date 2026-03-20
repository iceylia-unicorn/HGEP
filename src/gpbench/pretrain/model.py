# src/gpbench/pretrain/model.py
from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, to_hetero, GCNConv, GATConv


@dataclass
class PretrainModelConfig:
    backbone: str = "hgt"            # "hgt" | "to_hetero_gcn" | "to_hetero_gat"
    hidden_dim: int = 128
    proj_dim: int = 128
    num_layers: int = 2
    num_heads: int = 2              # only for HGT / GAT
    dropout: float = 0.2
    use_node_id_emb_if_missing_x: bool = True


# 不同类型节点特征映射到相同维度
class HeteroFeatureAdapter(nn.Module):
    """
    Map per-type x -> hidden_dim with per-type Linear.
    If a node type has no x, optionally use nn.Embedding(num_nodes, hidden_dim) with global n_id.
    """
    def __init__(self, data: HeteroData, hidden_dim: int, use_node_id_emb_if_missing_x: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_node_id_emb_if_missing_x = use_node_id_emb_if_missing_x # 如果节点类型没有特征，是否使用节点ID嵌入作为特征

        self.proj = nn.ModuleDict() # 与ModuleList不同，ModuleDict是字典形式，可以通过键访问对应的模块
        self.emb = nn.ModuleDict() # 用于存储节点ID嵌入模块，如果节点没有特征

        for ntype in data.node_types:
            x = getattr(data[ntype], "x", None) # [节点数，特征维度] 获取节点特征，如果没有则返回 None
            if x is not None:
                self.proj[ntype] = nn.Linear(int(x.size(-1)), hidden_dim) # size(-1) 获取最后一个维度的大小，即特征维度
            else:
                if not use_node_id_emb_if_missing_x:
                    raise ValueError(f"Node type {ntype} has no x. Set use_node_id_emb_if_missing_x=True.")
                self.emb[ntype] = nn.Embedding(int(data[ntype].num_nodes), hidden_dim)

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {} #键：节点类型，值：映射后的特征张量
        for ntype in batch.node_types:
            x = getattr(batch[ntype], "x", None)
            if x is not None:
                out[ntype] = self.proj[ntype](x) # 线性映射到 hidden_dim
            else:
                # NeighborLoader provides n_id (global ids); fallback to arange if absent
                if hasattr(batch[ntype], "n_id") and batch[ntype].n_id is not None:
                    nid = batch[ntype].n_id
                else:
                    nid = torch.arange(batch[ntype].num_nodes, device=next(self.parameters()).device)
                out[ntype] = self.emb[ntype](nid)
        return out


class HomoGCN(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False)) #必须显式关闭自环，否则会和HGT冲突
            # self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class HomogenizedGCNBackbone(nn.Module):
    """
    Apply a true GCN backbone on a per-batch homogeneous graph converted from hetero data.
    This keeps the backbone strictly GCNConv-based while remaining compatible with hetero inputs.
    """
    def __init__(self, node_types, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_types = list(node_types)
        self.ntype_to_id = {nt: i for i, nt in enumerate(self.node_types)}
        self.base = HomoGCN(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, x_dict, edge_index_dict):
        data = HeteroData()
        for ntype, x in x_dict.items():
            data[ntype].x = x
        for etype, ei in edge_index_dict.items():
            data[etype].edge_index = ei

        homo = data.to_homogeneous(node_attrs=["x"])
        x_h = self.base(homo.x, homo.edge_index)

        node_type = homo.node_type
        out = {}
        for ntype, tid in self.ntype_to_id.items():
            out[ntype] = x_h[node_type == tid]
        return out



class HomoGAT(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, heads: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # keep output dim = hidden_dim, so set out_channels = hidden_dim // heads
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim must be divisible by heads, got {hidden_dim} % {heads} != 0")
        out_ch = hidden_dim // heads

        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, out_ch, heads=heads, dropout=dropout, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns): #zip相当于压缩，将两个列表对应位置的元素配对成一个元组，for循环遍历这些元组
            x = conv(x, edge_index)              # [N, hidden_dim]
            x = bn(x)
            x = F.elu(x) #类似leaky relu是Relu的变种
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class TypePairRelationPrompt(nn.Module):
    """
    静态 type-pair relation prompt:
    - 每个 (src_type, dst_type) 一个可训练向量
    - 在每层 backbone 后，额外做一次 relation-aware residual propagation
    - 这是最小可跑版，不引入条件网
    """

    def __init__(
        self,
        metadata: Tuple[list, list],
        dim: int,
        mode: str = "mul",          # "mul" or "add"
        alpha: float = 0.5,         # residual strength
        dropout: float = 0.1,
        use_ln: bool = True,
        aggr: str = "mean",         # "mean" or "sum"
    ):
        super().__init__()
        assert mode in ("mul", "add")
        assert aggr in ("mean", "sum")

        node_types, edge_types = metadata
        self.mode = mode
        self.alpha = alpha
        self.aggr = aggr

        pair_keys = []
        for src_t, _, dst_t in edge_types:
            key = self.make_pair_key(src_t, dst_t)
            if key not in pair_keys:
                pair_keys.append(key)

        init_value = 1.0 if mode == "mul" else 0.0
        self.prompt = nn.ParameterDict({
            k: nn.Parameter(torch.full((dim,), init_value))
            for k in pair_keys
        })

        self.drop = nn.Dropout(dropout)
        self.ln = nn.ModuleDict({
            nt: (nn.LayerNorm(dim) if use_ln else nn.Identity())
            for nt in node_types
        })

    @staticmethod
    def make_pair_key(src_t: str, dst_t: str) -> str:
        return f"{src_t}__TO__{dst_t}"

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict) -> Dict[str, torch.Tensor]:
        device = next(iter(x_dict.values())).device

        agg_dict = {
            ntype: torch.zeros_like(x)
            for ntype, x in x_dict.items()
        }
        deg_dict = {
            ntype: torch.zeros((x.size(0), 1), device=device, dtype=x.dtype)
            for ntype, x in x_dict.items()
        }

        for (src_t, _, dst_t), edge_index in edge_index_dict.items():
            src, dst = edge_index
            if src.numel() == 0:
                continue

            key = self.make_pair_key(src_t, dst_t)
            p = self.prompt[key].unsqueeze(0)  # [1, d]

            msg = x_dict[src_t][src]  # [E, d]
            if self.mode == "mul":
                msg = msg * p
            else:
                msg = msg + p

            agg_dict[dst_t].index_add_(0, dst, msg)

            ones = torch.ones((dst.size(0), 1), device=device, dtype=x_dict[dst_t].dtype)
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


class HGTBackbone(nn.Module):
    def __init__(self, metadata, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))

    def forward(self, x_dict, edge_index_dict, relation_prompt: Optional[nn.Module] = None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

            # 新增：每层后做一次 type-pair-aware residual propagation
            if relation_prompt is not None:
                x_dict = relation_prompt(x_dict, edge_index_dict)

        return x_dict


class HeteroReadout(nn.Module):
    """
    Robust graph readout for NeighborLoader batches.

    If `paper.batch` exists -> do mean-pool per seed-subgraph (multi-graph pooling).
    Else -> fall back to "seed-node readout": use the embeddings of the seed nodes of input_ntype.
    This makes pretraining stable across PyG versions.
    """

    # input_ntype: 目标节点类型 x_dict[input_ntype] 是种子节点的特征张量
    def forward(self, x_dict: Dict[str, torch.Tensor], batch: HeteroData, input_ntype: str) -> torch.Tensor:
        store = batch[input_ntype] # 获取目标节点类型的存储对象 store.batch表示节点属于当前batch中哪个子图，input_id全图索引而非子图索引
        # print({store})

        # Case A: 将种子节点及其邻居节点组成的子图进行池化，在拥有batch属性的情况下（即新版PyG）
        if hasattr(store, "batch") and store.batch is not None:
            b = store.batch
            num_graphs = int(b.max().item()) + 1 if b.numel() > 0 else 1
            pooled_sum: Optional[torch.Tensor] = None

            for ntype, x in x_dict.items():
                s = batch[ntype]
                if hasattr(s, "batch") and s.batch is not None:
                    bn = s.batch
                else:
                    # if missing, skip that type (or could map to input batch; skipping is safer)
                    if ntype != input_ntype:
                        continue
                    bn = b

                out = torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
                out.index_add_(0, bn, x)
                cnt = torch.bincount(bn, minlength=num_graphs).clamp_min(1).to(x.device).unsqueeze(-1)
                out = out / cnt
                pooled_sum = out if pooled_sum is None else pooled_sum + out

            if pooled_sum is None:
                raise RuntimeError("Readout produced None (no valid node types pooled).")
            return pooled_sum

        # Case B: 默认信息已经聚合到种子节点上（即旧版PyG），种子节点在x_dict的最前面 
        # PyG often exposes `store.batch_size` for the number of seed nodes in this mini-batch.
        if hasattr(store, "batch_size") and store.batch_size is not None:
            bs = int(store.batch_size)
            return x_dict[input_ntype][:bs]  # [B, hidden_dim]

        # If even batch_size is missing, try `input_id` (seed node local indices in this subgraph)
        if hasattr(store, "input_id") and store.input_id is not None:
            idx = store.input_id
            return x_dict[input_ntype][idx]

        # Last resort: cannot infer seeds
        raise ValueError(
            f"Cannot infer seed nodes for readout: missing {input_ntype}.batch, "
            f"{input_ntype}.batch_size, and {input_ntype}.input_id"
        )



class ProjectionHead(nn.Module):
    """2-layer MLP projection head (HGMP pretraining)."""
    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HGMPPretrainModel(nn.Module):
    def __init__(self, full_data: HeteroData, cfg: PretrainModelConfig):
        super().__init__()
        self.cfg = cfg
        self.adapter = HeteroFeatureAdapter(
            full_data,
            hidden_dim=cfg.hidden_dim,
            use_node_id_emb_if_missing_x=cfg.use_node_id_emb_if_missing_x,
        )

        if cfg.backbone == "hgt":
            self.backbone = HGTBackbone(
                metadata=full_data.metadata(),
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
        elif cfg.backbone == "to_hetero_gcn":
            base = HomoGCN(cfg.hidden_dim, cfg.num_layers, cfg.dropout)
            self.backbone = to_hetero(base, full_data.metadata(), aggr="sum")
        elif cfg.backbone == "to_hetero_gat":
            base = HomoGAT(cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.dropout)
            self.backbone = to_hetero(base, full_data.metadata(), aggr="sum")
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        self.readout = HeteroReadout()
        self.proj = ProjectionHead(cfg.hidden_dim, cfg.proj_dim)

    def encode_graphs(
        self,
        batch: HeteroData,
        input_ntype: str,
        prompt=None,
        relation_prompt=None,
    ) -> torch.Tensor:
        # 1) feature adaptation
        x_dict = self.adapter(batch)

        # 2) optional node-type feature prompt
        if prompt is not None:
            x_dict = prompt(x_dict)

        # 3) backbone encode
        if self.cfg.backbone == "hgt":
            x_dict = self.backbone(
                x_dict,
                batch.edge_index_dict,
                relation_prompt=relation_prompt,
            )
        else:
            x_dict = self.backbone(x_dict, batch.edge_index_dict)
            if relation_prompt is not None:
                x_dict = relation_prompt(x_dict, batch.edge_index_dict)

        # 4) graph readout
        z = self.readout(x_dict, batch, input_ntype)
        return z


    def forward(
        self,
        batch: HeteroData,
        input_ntype: str,
        prompt=None,
        relation_prompt=None,
        return_proj: bool = True,
    ) -> torch.Tensor:
        z = self.encode_graphs(
            batch,
            input_ntype=input_ntype,
            prompt=prompt,
            relation_prompt=relation_prompt,
        )
        if return_proj:
            return self.proj(z)
        return z
    # def forward(self, batch: HeteroData, input_ntype: str) -> torch.Tensor:
    #     # 1) 特征对齐
    #     x_dict = self.adapter(batch)

    #     # 2) backbone encode
    #     x_dict = self.backbone(x_dict, batch.edge_index_dict)

    #     # 3) graph pooling (per seed-subgraph)
    #     z = self.readout(x_dict, batch, input_ntype)   # [B, hidden_dim]

    #     # 4) projection head
    #     return self.proj(z)                            # [B, proj_dim]
