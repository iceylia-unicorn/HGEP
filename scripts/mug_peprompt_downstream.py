from __future__ import annotations

import argparse
import csv
import json
import pickle as pk
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpbench.protocol_bridge.downstream_legacy import _prepare_labels_for_task, _unpack_batch
from gpbench.protocol_bridge.hgmp_peprompt import PEPromptRelation
from scripts.hgmae_bridge_utils import train_hgmae_embeddings
from scripts.mug_bridge_utils import set_seed, summarize, train_mug_embeddings


DATASET_ALIASES = {
    "ACM": "acm",
    "DBLP": "dblp",
    "Freebase": "freebase",
}

TARGET_NODETYPE = {
    "ACM": "paper",
    "DBLP": "author",
    "Freebase": "book",
}

DATASET_NUM_CLASS = {
    "ACM": 3,
    "DBLP": 4,
    "Freebase": 7,
}

PEPROMPT_EDGE_FEATURE_NAME = "peprompt_edge_feat"


class MUGPEPromptClassifier(nn.Module):
    def __init__(
        self,
        *,
        embeds: torch.Tensor,
        ntypes: list[str],
        canonical_etypes: list[tuple[str, str, str]],
        targetnode: str,
        edge_feature_dim: int,
        num_classes: int,
        feature_name: str,
        node_init: str,
        pool_scope: str,
        prompt_mode: str,
        prompt_alpha: float,
        prompt_dropout: float,
        prompt_aggr: str,
        prompt_use_ln: bool,
        prompt_hidden: int,
        prompt_edge_chunk_size: int,
        prompt_constraint: str,
        prompt_constraint_scale: float,
        head_hidden: int,
        head_dropout: float,
    ):
        super().__init__()
        self.register_buffer("embeds", embeds.float())
        self.ntypes = list(ntypes)
        self.targetnode = targetnode
        self.feature_name = feature_name
        self.node_init = node_init
        self.pool_scope = pool_scope
        dim = int(embeds.size(1))

        self.relation_prompt = PEPromptRelation(
            metadata=(self.ntypes, canonical_etypes),
            dim=dim,
            mode=prompt_mode,
            alpha=prompt_alpha,
            dropout=prompt_dropout,
            use_ln=prompt_use_ln,
            aggr=prompt_aggr,
            edge_feature_dim=edge_feature_dim,
            edge_feature_name=feature_name,
            edge_prompt_hidden=prompt_hidden,
            edge_chunk_size=prompt_edge_chunk_size,
            prompt_constraint=prompt_constraint,
            prompt_constraint_scale=prompt_constraint_scale,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _target_x(self, graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        nids = graph.nodes[self.targetnode].data[dgl.NID].long().to(self.embeds.device)
        max_nid = int(nids.max().item()) if nids.numel() else -1
        if max_nid >= self.embeds.size(0):
            raise RuntimeError(
                f"MUG embedding table has {self.embeds.size(0)} target rows, "
                f"but the PEPrompt batch references {self.targetnode} id {max_nid}."
            )
        return self.embeds[nids]

    def _graph_target_means(self, graph: dgl.DGLHeteroGraph, target_x: torch.Tensor) -> torch.Tensor:
        seg = graph.batch_num_nodes(self.targetnode).to(target_x.device)
        return dgl.ops.segment_reduce(seg, target_x, "mean")

    def _build_x_dict(self, graph: dgl.DGLHeteroGraph) -> dict[str, torch.Tensor]:
        target_x = self._target_x(graph)
        x_dict = {self.targetnode: target_x}
        dim = int(self.embeds.size(1))

        if self.node_init == "zero":
            for ntype in self.ntypes:
                if ntype == self.targetnode:
                    continue
                x_dict[ntype] = torch.zeros(
                    (graph.num_nodes(ntype), dim),
                    device=self.embeds.device,
                    dtype=target_x.dtype,
                )
            return x_dict

        graph_means = self._graph_target_means(graph, target_x)
        for ntype in self.ntypes:
            if ntype == self.targetnode:
                continue
            seg = graph.batch_num_nodes(ntype).to(self.embeds.device)
            x_dict[ntype] = torch.repeat_interleave(graph_means, seg, dim=0)
        return x_dict

    def _edge_dicts(self, graph: dgl.DGLHeteroGraph):
        edge_index_dict = {}
        edge_feature_dict = {}
        for etype in graph.canonical_etypes:
            src, dst = graph.edges(etype=etype, order="eid")
            edge_index_dict[etype] = torch.stack([src.long(), dst.long()], dim=0).to(self.embeds.device)
            if self.feature_name not in graph.edges[etype].data:
                raise RuntimeError(f"Missing edge feature {self.feature_name!r} for etype={etype}.")
            edge_feature_dict[etype] = graph.edges[etype].data[self.feature_name].float().to(self.embeds.device)
        return edge_index_dict, edge_feature_dict

    def _pool(self, h_dict: dict[str, torch.Tensor], graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        if self.pool_scope == "target":
            seg = graph.batch_num_nodes(self.targetnode).to(h_dict[self.targetnode].device)
            return dgl.ops.segment_reduce(seg, h_dict[self.targetnode], "mean")

        pooled = []
        for ntype in self.ntypes:
            if graph.num_nodes(ntype) == 0:
                continue
            seg = graph.batch_num_nodes(ntype).to(h_dict[ntype].device)
            pooled.append(dgl.ops.segment_reduce(seg, h_dict[ntype], "mean"))
        return torch.stack(pooled, dim=0).mean(dim=0)

    def forward(self, graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        x_dict = self._build_x_dict(graph)
        edge_index_dict, edge_feature_dict = self._edge_dicts(graph)
        h_dict = self.relation_prompt(x_dict, edge_index_dict, edge_feature_dict, graph=graph)
        return self.head(self._pool(h_dict, graph))


class MUGTargetMLP(nn.Module):
    def __init__(self, embeds: torch.Tensor, targetnode: str, num_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.register_buffer("embeds", embeds.float())
        self.targetnode = targetnode
        dim = int(embeds.size(1))
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        nids = graph.nodes[self.targetnode].data[dgl.NID].long().to(self.embeds.device)
        max_nid = int(nids.max().item()) if nids.numel() else -1
        if max_nid >= self.embeds.size(0):
            raise RuntimeError(
                f"MUG embedding table has {self.embeds.size(0)} target rows, "
                f"but the PEPrompt batch references {self.targetnode} id {max_nid}."
            )
        x = self.embeds[nids]
        seg = graph.batch_num_nodes(self.targetnode).to(x.device)
        return self.head(dgl.ops.segment_reduce(seg, x, "mean"))


def _format_float_for_key(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _subgraph_cache_key(args) -> str:
    if args.subgraph_cache_key:
        return args.subgraph_cache_key
    subgraph_type = str(args.subgraph_type)
    if subgraph_type not in {"metapath_topk", "metapath_topk_path", "metapath_topk_adapt", "metapath_topk_path_adapt"}:
        return subgraph_type

    metric = str(args.metapath_rank_metric)
    if subgraph_type in {"metapath_topk_adapt", "metapath_topk_path_adapt"}:
        base = (
            f"{subgraph_type}_m{args.metapath_max_hop}_"
            f"k{args.metapath_min_topk}-{args.metapath_max_topk}_"
            f"a{_format_float_for_key(args.metapath_rel_threshold)}_{metric}"
        )
    else:
        base = f"{subgraph_type}_m{args.metapath_max_hop}_k{args.metapath_topk}_{metric}"

    if args.metapath_endpoint_mode != "all":
        base += f"_{args.metapath_endpoint_mode}"
    support_mode = args.metapath_support_mode
    if support_mode == "auto" and args.metapath_endpoint_mode == "target_closed":
        support_mode = "count"
    if support_mode not in {"auto", "none"}:
        base += f"_support{support_mode}"
        if int(args.metapath_support_topk) > 0:
            base += f"_sk{int(args.metapath_support_topk)}"
    return base


def _load_cache(args, split_seed: int):
    path = (
        Path(args.peprompt_offline_cache_dir)
        / args.dataset
        / f"{_subgraph_cache_key(args)}_{args.shot}-shot"
        / f"seed{int(split_seed)}"
        / f"ft{int(args.feats_type)}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(f"PEPrompt offline cache not found: {path}")
    with open(path, "rb") as f:
        payload = pk.load(f)
    return payload, path


def _load_embedding_payload(args, device: torch.device):
    if args.embedding_path is not None and Path(args.embedding_path).exists():
        return torch.load(args.embedding_path, map_location="cpu", weights_only=False)

    if args.pretrain_source == "hgmae":
        payload = train_hgmae_embeddings(args, device)
        if args.save_embedding_path is not None:
            Path(args.save_embedding_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.save_embedding_path)
        return payload

    if args.pretrain_source == "embedding":
        raise FileNotFoundError("--pretrain_source embedding requires an existing --embedding_path payload.")

    mug_args = argparse.Namespace(**vars(args))
    mug_data_source = args.mug_data_source
    if mug_data_source == "auto":
        mug_data_source = "hgb" if args.dataset == "Freebase" else "native"
    mug_args.mug_data_source = mug_data_source
    mug_args.dataset = DATASET_ALIASES[args.dataset]
    payload = train_mug_embeddings(mug_args, device)
    if args.save_embedding_path is not None:
        Path(args.save_embedding_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.save_embedding_path)
    return payload


def _validate_payload(args, mug_payload: dict, cache_payload: dict):
    embeds = mug_payload["embeds"]
    labels = mug_payload["labels"]
    target_ids = []
    for key in ("train_ids", "val_ids", "test_ids"):
        if key in cache_payload:
            target_ids.extend(np.asarray(cache_payload[key], dtype=np.int64).tolist())
    max_cache_id = max(target_ids) if target_ids else -1
    if max_cache_id >= int(embeds.size(0)):
        raise RuntimeError(
            "Embedding/PEPrompt target-id mismatch. "
            f"pretrain_source={args.pretrain_source} has {int(embeds.size(0))} target embeddings, "
            f"but PEPrompt {args.dataset} cache references target id {max_cache_id}. "
            "This usually means the two datasets are not the same task definition."
        )
    if int(labels.max().item()) + 1 != DATASET_NUM_CLASS[args.dataset]:
        raise RuntimeError(
            f"MUG labels imply {int(labels.max().item()) + 1} classes, "
            f"but PEPrompt {args.dataset} expects {DATASET_NUM_CLASS[args.dataset]}."
        )


def _make_loader(samples, batch_size: int, shuffle: bool):
    return dgl.dataloading.GraphDataLoader(samples, batch_size=batch_size, shuffle=shuffle)


def _f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    pred = logits.argmax(dim=-1)
    micro = float((pred == labels).float().mean().item())
    vals = []
    for cls in range(num_classes):
        yt = labels == cls
        if int(yt.sum().item()) == 0:
            continue
        yp = pred == cls
        tp = (yt & yp).sum().float()
        fp = ((~yt) & yp).sum().float()
        fn = (yt & (~yp)).sum().float()
        vals.append(float((2 * tp / (2 * tp + fp + fn + 1e-12)).item()))
    return micro, float(np.mean(vals)) if vals else 0.0


def _evaluate(model, samples, args, device: torch.device):
    logits_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for batch in _make_loader(samples, args.batch_size, shuffle=False):
            graph, labels = _unpack_batch(batch, "NIG")
            graph = graph.to(device)
            labels = _prepare_labels_for_task(labels, device, args.dataset, "NIG")
            logits_all.append(model(graph))
            labels_all.append(labels)
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0).long()
    return _f1(logits, labels, args.num_class), float(F.cross_entropy(logits, labels).item())


def _train_once(model, cache_payload: dict, args, run_seed: int, device: torch.device):
    set_seed(run_seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_micro = 0.0
    best_val_macro = 0.0
    bad_epochs = 0

    train_loader = _make_loader(cache_payload["train"], args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            graph, labels = _unpack_batch(batch, "NIG")
            graph = graph.to(device)
            labels = _prepare_labels_for_task(labels, device, args.dataset, "NIG")
            logits = model(graph)
            loss = F.cross_entropy(logits, labels.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_items += int(labels.numel())

        (_val_micro, _val_macro), val_loss = _evaluate(model, cache_payload["val"], args, device)
        improved = val_loss <= best_val_loss
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_micro = _val_micro
            best_val_macro = _val_macro
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % max(1, args.log_interval) == 0:
            train_loss = total_loss / max(total_items, 1)
            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1={_val_micro:.4f}/{_val_macro:.4f} | "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    (test_micro, test_macro), _test_loss = _evaluate(model, cache_payload["test"], args, device)
    return {
        "micro": float(test_micro),
        "macro": float(test_macro),
        "best_epoch": int(best_epoch),
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "best_val_micro": float(best_val_micro),
        "best_val_macro": float(best_val_macro),
    }


def _make_model(method: str, embeds: torch.Tensor, cache_payload: dict, args):
    sample_graph = cache_payload["train"][0][0]
    ntypes = list(sample_graph.ntypes)
    etypes = list(sample_graph.canonical_etypes)
    if method in {"mug_target_mlp", "target_mlp", "mlp"}:
        return MUGTargetMLP(
            embeds=embeds,
            targetnode=cache_payload["targetnode"],
            num_classes=args.num_class,
            hidden=args.head_hidden,
            dropout=args.head_dropout,
        )
    return MUGPEPromptClassifier(
        embeds=embeds,
        ntypes=ntypes,
        canonical_etypes=etypes,
        targetnode=cache_payload["targetnode"],
        edge_feature_dim=int(cache_payload["peprompt_edge_feature_dim"]),
        num_classes=args.num_class,
        feature_name=str(cache_payload.get("peprompt_edge_feature_name", PEPROMPT_EDGE_FEATURE_NAME)),
        node_init=args.node_init,
        pool_scope=args.pool_scope,
        prompt_mode=args.relation_prompt_mode,
        prompt_alpha=args.relation_prompt_alpha,
        prompt_dropout=args.relation_prompt_dropout,
        prompt_aggr=args.relation_prompt_aggr,
        prompt_use_ln=args.relation_prompt_use_ln,
        prompt_hidden=args.peprompt_edge_prompt_hidden,
        prompt_edge_chunk_size=args.peprompt_edge_chunk_size,
        prompt_constraint=args.relation_prompt_constraint,
        prompt_constraint_scale=args.relation_prompt_constraint_scale,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    )


def main():
    ap = argparse.ArgumentParser("Embedding pretraining with PEPrompt offline-cache downstream")
    ap.add_argument("--dataset", choices=sorted(DATASET_ALIASES), default="DBLP")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--pretrain_source", choices=["mug", "hgmae", "embedding"], default="mug")
    ap.add_argument("--mug_epochs", type=int, default=50)
    ap.add_argument("--mug_ratio", type=int, default=60)
    ap.add_argument("--mug_data_source", choices=["auto", "native", "hgb"], default="auto")
    ap.add_argument("--hgmae_root", type=Path, default=None)
    ap.add_argument("--hgmae_epochs", type=int, default=200)
    ap.add_argument("--hgmae_patience", type=int, default=20)
    ap.add_argument("--hgmae_hidden_dim", type=int, default=256)
    ap.add_argument("--hgmae_num_layers", type=int, default=2)
    ap.add_argument("--hgmae_num_heads", type=int, default=4)
    ap.add_argument("--hgmae_lr", type=float, default=1e-3)
    ap.add_argument("--hgmae_feature_dim", type=int, default=256)
    ap.add_argument("--hgmae_feature_mode", choices=["auto", "target_x", "target_x_degree", "degree"], default="auto")
    ap.add_argument("--hgmae_max_edges_per_metapath", type=int, default=500000)
    ap.add_argument("--hgmae_max_metapaths", type=int, default=8)
    ap.add_argument("--hgmae_use_mp_edge_recon", action="store_true")
    ap.add_argument("--hgmae_recon_edges_per_metapath", type=int, default=200000)
    ap.add_argument("--hgmae_dense_recon_max_nodes", type=int, default=12000)
    ap.add_argument("--hgmae_log_interval", type=int, default=10)
    ap.add_argument("--embedding_path", type=Path, default=None)
    ap.add_argument("--save_embedding_path", type=Path, default=None)
    ap.add_argument("--no_unified_feature", action="store_true")
    ap.add_argument("--hgb_mug_feature_dim", type=int, default=256)
    ap.add_argument("--hgb_mug_hidden_dim", type=int, default=512)
    ap.add_argument("--hgb_mug_feature_signal_dim", type=int, default=1024)
    ap.add_argument("--hgb_mug_sample_size", type=int, default=1024)
    ap.add_argument("--hgb_mug_max_edges_per_metapath", type=int, default=500000)
    ap.add_argument("--hgb_mug_recon_edges_per_metapath", type=int, default=200000)
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--split_seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--peprompt_offline_cache_dir", type=Path, default=ROOT / "artifacts" / "cache" / "peprompt_offline_splits")
    ap.add_argument("--subgraph_cache_key", type=str, default=None)
    ap.add_argument("--subgraph_type", choices=["khop", "metapath_topk", "metapath_topk_path", "metapath_topk_adapt", "metapath_topk_path_adapt"], default="metapath_topk_adapt")
    ap.add_argument("--metapath_max_hop", type=int, default=3)
    ap.add_argument("--metapath_topk", type=int, default=3)
    ap.add_argument("--metapath_min_topk", type=int, default=1)
    ap.add_argument("--metapath_max_topk", type=int, default=8)
    ap.add_argument("--metapath_rel_threshold", type=float, default=0.5)
    ap.add_argument("--metapath_rank_metric", choices=["count", "degree_norm", "count_idf", "random"], default="count")
    ap.add_argument("--metapath_endpoint_mode", choices=["all", "target_closed"], default="all")
    ap.add_argument("--metapath_support_mode", choices=["auto", "none", "one_path", "count"], default="auto")
    ap.add_argument("--metapath_support_topk", type=int, default=0)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["target_mlp", "peprompt"],
        choices=["mug_target_mlp", "mug_peprompt", "target_mlp", "mlp", "peprompt", "type_neighborhood_edge"],
    )
    ap.add_argument("--node_init", choices=["zero", "graph_target_mean"], default="graph_target_mean")
    ap.add_argument("--pool_scope", choices=["target", "all"], default="target")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--relation_prompt_mode", choices=["mul", "add"], default="mul")
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", choices=["mean", "sum"], default="mean")
    ap.add_argument("--relation_prompt_use_ln", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--relation_prompt_constraint", choices=["none", "identity_tanh", "positive_sigmoid", "identity_l2norm"], default="identity_tanh")
    ap.add_argument("--relation_prompt_constraint_scale", type=float, default=0.5)
    ap.add_argument("--peprompt_edge_prompt_hidden", type=int, default=128)
    ap.add_argument("--peprompt_edge_chunk_size", type=int, default=50000)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "artifacts" / "results" / "mug_peprompt_downstream")
    args = ap.parse_args()

    if args.dataset == "Freebase" and "--feats_type" not in sys.argv:
        args.feats_type = 1
    args.num_class = DATASET_NUM_CLASS[args.dataset]
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    print(
        f"[Device] torch={torch.__version__} cuda_available={torch.cuda.is_available()} selected={device}",
        flush=True,
    )
    print(
        f"[Config] dataset={args.dataset} pretrain_source={args.pretrain_source} "
        f"mug_dataset={DATASET_ALIASES[args.dataset]} mug_data_source={args.mug_data_source} "
        f"cache_key={_subgraph_cache_key(args)} "
        f"shot={args.shot} splits={args.split_seeds}",
        flush=True,
    )

    mug_payload = _load_embedding_payload(args, device)
    rows = []
    cache_paths = {}
    for split_seed in args.split_seeds:
        cache_payload, cache_path = _load_cache(args, split_seed)
        cache_paths[int(split_seed)] = str(cache_path)
        if cache_payload.get("targetnode") != TARGET_NODETYPE[args.dataset]:
            raise RuntimeError(
                f"Unexpected targetnode in cache: {cache_payload.get('targetnode')} "
                f"for dataset={args.dataset}."
            )
        _validate_payload(args, mug_payload, cache_payload)
        embeds = mug_payload["embeds"].float()
        for repeat in range(args.repeats):
            run_seed = int(args.seed) * 100000 + int(split_seed) * 1000 + repeat
            for method in args.methods:
                print(f"[Downstream] method={method} split={split_seed} repeat={repeat}", flush=True)
                model = _make_model(method, embeds, cache_payload, args)
                result = _train_once(model, cache_payload, args, run_seed, device)
                result.update({"method": method, "split_seed": int(split_seed), "repeat": int(repeat), "run_seed": int(run_seed)})
                rows.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_key = "-".join(str(seed) for seed in args.split_seeds)
    pretrain_tag = f"{args.pretrain_source}{args.hgmae_epochs if args.pretrain_source == 'hgmae' else args.mug_epochs}"
    stem = (
        f"{args.dataset}_{args.shot}shot_{_subgraph_cache_key(args)}_"
        f"{pretrain_tag}_s{split_key}_r{args.repeats}"
    )
    stem = re.sub(r"[^A-Za-z0-9_.=-]+", "_", stem)
    per_run_path = args.out_dir / f"{stem}.per_run.csv"
    with open(per_run_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "split_seed",
                "repeat",
                "run_seed",
                "micro",
                "macro",
                "best_epoch",
                "best_val_loss",
                "best_val_micro",
                "best_val_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cache_paths": cache_paths,
        "embedding_path": None if args.embedding_path is None else str(args.embedding_path),
        "pretrain_source": args.pretrain_source,
        "target_embeddings": int(mug_payload["embeds"].size(0)),
        "embedding_dim": int(mug_payload["embeds"].size(1)),
        "summary": summarize(rows),
        "per_run": str(per_run_path),
    }
    summary_path = args.out_dir / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["summary"], indent=2), flush=True)
    print(f"[DONE] per_run={per_run_path}", flush=True)
    print(f"[DONE] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
