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
        """
        super().__init__()
        if mode not in {"mul", "add"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if aggr not in {"mean", "sum"}:
            raise ValueError(f"Unsupported aggr: {aggr}")
        if int(edge_feature_dim or 0) <= 0:
            raise ValueError("PEPromptRelation requires positive edge_feature_dim.")

        node_types, edge_types = metadata
        del edge_types
        self.mode = mode
        self.alpha = alpha
        self.aggr = aggr
        self.edge_feature_dim = int(edge_feature_dim or 0)
        self.edge_feature_name = edge_feature_name

        self.drop = nn.Dropout(dropout)
        self.ln = nn.ModuleDict(
            {
                nt: (nn.LayerNorm(dim) if use_ln else nn.Identity())
                for nt in node_types
            }
        )
        if self.edge_feature_dim > 0:
            self.edge_prompt_mlp = nn.Sequential(
                nn.LayerNorm(self.edge_feature_dim),
                nn.Linear(self.edge_feature_dim, edge_prompt_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_prompt_hidden, dim),
            )
        else:
            self.edge_prompt_mlp = None

    @property
    def uses_edge_features(self) -> bool:
        return self.edge_prompt_mlp is not None

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_feature_dict: Dict[Tuple[str, str, str], torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: 各节点类型的节点表示字典，键为节点类型，值为节点特征矩阵。
            edge_index_dict: 各规范边类型的边索引字典，形状为 [2, num_edges]。
            edge_feature_dict: 各规范边类型对应的边特征字典，这里应为 Laplacian PE 差值。

        Returns:
            注入纯 PE 边提示后的各节点类型表示。
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
    ) -> Dict[str, torch.Tensor]:
        del targetnode

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = self.relation_prompt(x_dict, edge_index_dict, edge_feature_dict)

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
            hidden_dict = self.relation_prompt(hidden_dict, edge_index_dict, edge_feature_dict)
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

    def forward(self, targetnode, x, edge_index=None, edge_feature_dict=None, homo_graph=None):
        """
        Args:
            targetnode: HGT 路径下的目标节点类型；GCN 路径下复用该位置传入 graph。
            x: 节点特征字典。
            edge_index: HGT 路径下的异构边索引字典。
            edge_feature_dict: 各边类型的 PE 边特征字典。
        """
        if self.hgnn_type == "HGT":
            return self.GraphConv(targetnode, x, edge_index, edge_feature_dict)
        if self.hgnn_type == "GCN":
            graph = targetnode
            x_dict = x
            return self.GraphConv(
                graph,
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
