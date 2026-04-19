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

TYPEPAIR_PRETRAIN_DIR = Path("artifacts/checkpoints/typepair_hgmp/pretrain")
TYPEPAIR_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TypePairRelationPromptConfig:
    mode: str = "mul"          # "mul" or "add"
    alpha: float = 0.5         # residual strength
    dropout: float = 0.1
    use_ln: bool = True
    aggr: str = "mean"         # "mean" or "sum"
    edge_feature_dim: int = 0
    edge_feature_name: str = "typepair_edge_feat"
    edge_prompt_hidden: int = 128
    edge_prompt_alpha: float = 0.5
    edge_prompt_fusion: str = "add"  # "add", "mul", or "gate"


class TypePairRelationPrompt(nn.Module):
    """
    Static type-pair relation prompt:
    - one trainable vector for each (src_type, dst_type)
    - aggregate relation-aware messages to dst nodes
    - residual injection back into per-type node states
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
        edge_feature_name: str = "typepair_edge_feat",
        edge_prompt_hidden: int = 128,
        edge_prompt_alpha: float = 0.5,
        edge_prompt_fusion: str = "add",
    ):
        super().__init__()
        if mode not in {"mul", "add"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if aggr not in {"mean", "sum"}:
            raise ValueError(f"Unsupported aggr: {aggr}")
        if edge_prompt_fusion not in {"add", "mul", "gate"}:
            raise ValueError(f"Unsupported edge_prompt_fusion: {edge_prompt_fusion}")

        node_types, edge_types = metadata
        self.mode = mode
        self.alpha = alpha
        self.aggr = aggr
        self.edge_feature_dim = int(edge_feature_dim or 0)
        self.edge_feature_name = edge_feature_name
        self.edge_prompt_alpha = edge_prompt_alpha
        self.edge_prompt_fusion = edge_prompt_fusion

        pair_keys = []
        for src_t, _, dst_t in edge_types:
            key = self.make_pair_key(src_t, dst_t)
            if key not in pair_keys:
                pair_keys.append(key)

        init_value = 1.0 if mode == "mul" else 0.0
        self.prompt = nn.ParameterDict(
            {
                k: nn.Parameter(torch.full((dim,), init_value))
                for k in pair_keys
            }
        )

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

    @staticmethod
    def make_pair_key(src_t: str, dst_t: str) -> str:
        return f"{src_t}__TO__{dst_t}"

    @property
    def uses_edge_features(self) -> bool:
        return self.edge_prompt_mlp is not None

    def _fuse_edge_prompt(
        self,
        type_prompt: torch.Tensor,
        edge_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.edge_prompt_mlp is None or edge_features is None:
            return type_prompt.unsqueeze(0)

        edge_prompt = self.edge_prompt_mlp(edge_features.to(type_prompt.device))
        if self.edge_prompt_fusion == "add":
            return type_prompt.unsqueeze(0) + self.edge_prompt_alpha * edge_prompt
        if self.edge_prompt_fusion == "mul":
            scale = 1.0 + self.edge_prompt_alpha * torch.tanh(edge_prompt)
            return type_prompt.unsqueeze(0) * scale

        gate = torch.sigmoid(edge_prompt)
        return gate * type_prompt.unsqueeze(0) + (1.0 - gate) * edge_prompt

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_feature_dict: Dict[Tuple[str, str, str], torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
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

            key = self.make_pair_key(src_t, dst_t)
            if key not in self.prompt:
                continue

            edge_features = None
            if edge_feature_dict is not None:
                edge_features = edge_feature_dict.get((src_t, rel_t, dst_t))

            p = self._fuse_edge_prompt(self.prompt[key], edge_features)

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


class RelationInjectedLegacyHGT(nn.Module):
    """
    Wrap protocols.hgmp.prompt_legacy.HGT without modifying it.
    Injection point:
      after each HGTConv block, before the final output projection.

    Keep legacy parameter names (lin_dict / convs / lin) so a plain HGMP-HGT
    checkpoint can still load into this wrapper with only relation prompt keys missing.
    """

    def __init__(self, base_hgt: nn.Module, relation_prompt: TypePairRelationPrompt):
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


class RelationInjectedLegacyGCN(nn.Module):
    """
    Wrap protocols.hgmp.prompt_legacy.GCL_GCN without modifying it.

    Injection point:
      after each GraphConv block on the hidden representation.

    Keep legacy parameter names (fc_list / layers / dropout) so a plain HGMP-GCN
    checkpoint can still load into this wrapper with only relation prompt keys missing.
    """

    def __init__(self, base_gcn: nn.Module, relation_prompt: TypePairRelationPrompt):
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
    ) -> Dict[str, torch.Tensor]:
        keys = list(x_dict.keys())
        sizes = [x_dict[key].shape[0] for key in keys]

        feats_emd = [x_dict[key] for key in keys]
        h_list = []
        for fc, feature in zip(self.fc_list, feats_emd):
            h_list.append(fc(feature))
        h = torch.cat(h_list, dim=0)

        edge_index_dict = _build_edge_index_dict(graph)
        homo_g = dgl.to_homogeneous(graph)
        homo_g = dgl.remove_self_loop(homo_g)
        homo_g = dgl.add_self_loop(homo_g)

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(homo_g, h)

            hidden_dict = _split_h_by_keys(h, keys, sizes)
            hidden_dict = self.relation_prompt(hidden_dict, edge_index_dict, edge_feature_dict)
            h = torch.cat([hidden_dict[key] for key in keys], dim=0)

        return _split_h_by_keys(h, keys, sizes)


class HGMPTypePairHGNN(nn.Module):
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
        relation_cfg: TypePairRelationPromptConfig | None = None,
    ):
        super().__init__()
        if hgnn_type not in {"HGT", "GCN"}:
            raise NotImplementedError(
                "HGMPTypePairHGNN currently supports only HGT and GCN."
            )

        if relation_cfg is None:
            relation_cfg = TypePairRelationPromptConfig()

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
        relation_prompt = TypePairRelationPrompt(
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
            edge_prompt_alpha=relation_cfg.edge_prompt_alpha,
            edge_prompt_fusion=relation_cfg.edge_prompt_fusion,
        )

        if self.hgnn_type == "HGT":
            self.GraphConv = RelationInjectedLegacyHGT(
                base_hgnn.GraphConv,
                relation_prompt,
            )
        elif self.hgnn_type == "GCN":
            self.GraphConv = RelationInjectedLegacyGCN(
                base_hgnn.GraphConv,
                relation_prompt,
            )
        else:
            raise NotImplementedError(
                f"Unsupported hgnn_type in HGMPTypePairHGNN: {self.hgnn_type}"
            )

    @property
    def relation_prompt(self) -> TypePairRelationPrompt:
        return self.GraphConv.relation_prompt

    def forward(self, targetnode, x, edge_index=None, edge_feature_dict=None):
        if self.hgnn_type == "HGT":
            return self.GraphConv(targetnode, x, edge_index, edge_feature_dict)
        if self.hgnn_type == "GCN":
            graph = targetnode
            x_dict = x
            return self.GraphConv(graph, x_dict, edge_feature_dict)
        raise NotImplementedError(
            f"Unsupported hgnn_type in HGMPTypePairHGNN.forward: {self.hgnn_type}"
        )


class TypePairLegacyPreTrain(LegacyPreTrain):
    """
    Reuse all training / loader / GraphCL logic from protocols.hgmp.pretrain_legacy.PreTrain,
    but swap in HGMPTypePairHGNN at construction time.
    """

    def __init__(
        self,
        args,
        ntypes,
        metadata,
        num_class,
        num_etypes,
        input_dims,
        relation_cfg: TypePairRelationPromptConfig | None = None,
    ):
        nn.Module.__init__(self)

        self.pretext = args.pretext
        self.hgnn_type = args.hgnn_type
        self.device = args.device
        self.args = args

        if args.hgnn_type not in {"HGT", "GCN"}:
            raise NotImplementedError(
                "TypePair bridge currently supports only HGT and GCN."
            )

        self.hgnn = HGMPTypePairHGNN(
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
            raise NotImplementedError("TypePair bridge has not enabled HDGI yet.")
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
        loader1, loader2 = self.get_loader(
            graph_list,
            graph_batch_size,
            aug1=aug1,
            aug2=aug2,
            aug_ratio=aug_ration,
            pretext=self.pretext,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        save_path = TYPEPAIR_PRETRAIN_DIR / (
            f"{dataname}.{self.pretext}.{self.hgnn_type}"
            f".typepair.hid{self.args.hidden_dim}.np{self.args.num_samples}.pth"
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


def build_typepair_relation_cfg_from_args(args) -> TypePairRelationPromptConfig:
    return TypePairRelationPromptConfig(
        mode=getattr(args, "relation_prompt_mode", "mul"),
        alpha=getattr(args, "relation_prompt_alpha", 0.5),
        dropout=getattr(args, "relation_prompt_dropout", 0.1),
        use_ln=getattr(args, "relation_prompt_use_ln", True),
        aggr=getattr(args, "relation_prompt_aggr", "mean"),
        edge_feature_dim=getattr(args, "typepair_edge_feature_dim", 0),
        edge_feature_name=getattr(args, "typepair_edge_feature_name", "typepair_edge_feat"),
        edge_prompt_hidden=getattr(args, "typepair_edge_prompt_hidden", 128),
        edge_prompt_alpha=getattr(args, "typepair_edge_prompt_alpha", 0.5),
        edge_prompt_fusion=getattr(args, "typepair_edge_prompt_fusion", "add"),
    )


def pretrain_typepair_legacy(args):
    """
    HGMP-aligned pretraining entry for TypePairRelationPrompt.
    This mirrors protocols.hgmp.run_legacy.pretrain(), but swaps the model class.
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

    relation_cfg = build_typepair_relation_cfg_from_args(args)
    pt = TypePairLegacyPreTrain(
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
