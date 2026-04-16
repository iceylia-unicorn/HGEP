
# scripts/protocol_benchmark_v2.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import argparse
import csv
import json
import pickle as pk
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import dgl
import numpy as np
import torch
from torch_geometric.datasets import HGBDataset
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_dgl

from gpbench.protocol_bridge import downstream_legacy as legacy_bridge
from gpbench.protocol_bridge.downstream_legacy import (
    build_legacy_fewshot_embeddings,
    train_hgmp_heteroprompt_probe,
    train_mlp_probe,
    train_typepair_prompt_probe,
)
from gpbench.downstream.fewshot import load_split_file
from gpbench.utils.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_nested_summary,
    log_metrics,
    log_run_record,
    log_table,
    maybe_configure_wandb_env,
    upload_dir_artifact,
)
from protocols.hgmp.utils_legacy import create_matrix, seed_everything
from protocols.hgprompt.runner import _run_once as hgprompt_run_once
from protocols.hgprompt.adapter import load_hgprompt_downstream_bundle


HOP_NUM = {
    "ACM": 1,
    "DBLP": 2,
    "IMDB": 2,
    "Freebase": 1,
}

TARGET_NODETYPE = {
    "ACM": "paper",
    "DBLP": "author",
    "IMDB": "movie",
    "Freebase": "book",
}


@dataclass
class RunRecord:
    method: str
    split_seed: int
    repeat_id: int
    run_seed: int
    ckpt_path: str
    test_micro: float
    test_macro: float
    best_epoch: int


@dataclass
class SeedAggregate:
    method: str
    split_seed: int
    count: int
    micro_mean: float
    micro_std: float
    macro_mean: float
    macro_std: float


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_global_seed(seed: int):
    seed_everything(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _apply_feats_type(data, feats_type: int):
    features = data.x_dict
    features_list = []
    for value in features.values():
        features_list.append(value)

    if feats_type in (0, 6, 7, -1, 8, 9):
        pass
    elif feats_type in (1, 5):
        save = 0 if feats_type == 1 else 2
        for i in range(len(features_list)):
            if i != save:
                features_list[i] = torch.zeros((features_list[i].shape[0], 10))
    elif feats_type in (2, 4):
        save = feats_type - 2
        for i in range(len(features_list)):
            if i == save:
                continue
            dim = features_list[i].shape[0]
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            idx = torch.LongTensor(idx)
            val = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(idx, val, torch.Size([dim, dim])).to_dense()
    elif feats_type == 3:
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            idx = torch.LongTensor(idx)
            val = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(idx, val, torch.Size([dim, dim])).to_dense()
    else:
        raise ValueError(f"Unsupported feats_type={feats_type}")

    value_dict = {}
    for i, ntype in enumerate(data.node_types):
        value_dict[ntype] = features_list[i]
    data.set_value_dict("x", value_dict)
    return data


def _load_raw_heterograph(root: str, dataset: str, feats_type: int):
    if dataset == "Freebase" and feats_type == -1:
        dataset_obj = HGBDataset(root=root, name=dataset, transform=ToUndirected(merge=False))
    else:
        dataset_obj = HGBDataset(root=root, name=dataset)
    data = dataset_obj[0]

    for node_type, node_store in data.node_items():
        for attr, value in list(node_store.items()):
            if attr == "num_nodes":
                data[node_type]["x"] = create_matrix(value, 0.01)
                del data[node_type][attr]

    if dataset == "IMDB":
        targetnode = TARGET_NODETYPE[dataset]
        oldy = data[targetnode]["y"]
        newy = []
        for arr in oldy:
            one_indices = np.where(arr.cpu().numpy() == 1)[0]
            if one_indices.size > 0:
                newy.append(int(np.random.choice(one_indices)))
            else:
                newy.append(-1)
        data[targetnode]["oldy"] = oldy
        data[targetnode]["y"] = torch.tensor(newy)

    data = _apply_feats_type(data, feats_type)
    graph = to_dgl(data)
    return graph, TARGET_NODETYPE[dataset]


def _build_single_sample(graph, targetnode: str, node_id: int, label: int, dataset: str):
    subgraph, inverse_indices = dgl.khop_in_subgraph(
        graph,
        {targetnode: int(node_id)},
        k=HOP_NUM[dataset],
    )
    return (subgraph, inverse_indices, torch.tensor(int(label)))


def _aligned_cache_path(root: str, dataset: str, shot: int, split_seed: int, feats_type: int) -> Path:
    return (
        ROOT
        / "artifacts"
        / "cache"
        / "protocol_aligned_splits"
        / dataset
        / f"{shot}-shot"
        / f"seed{split_seed}"
        / f"ft{feats_type}.pkl"
    )


def load_aligned_legacy_splits(args):
    if args.dataset not in HOP_NUM:
        raise ValueError(f"Unsupported dataset for aligned legacy splits: {args.dataset}")
    if args.dataset == "IMDB":
        raise NotImplementedError("Aligned legacy split builder currently targets ACM/DBLP/Freebase first.")

    cache_path = _aligned_cache_path(args.root, args.dataset, args.shot, args.split_seed, args.feats_type)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            payload = pk.load(f)
        return payload["train"], payload["val"], payload["test"], payload["targetnode"]

    _ = load_split_file(args.splits, args.dataset, args.shot, args.split_seed)
    bundle = load_hgprompt_downstream_bundle(
        root=args.root,
        dataset=args.dataset,
        splits=args.splits,
        shot=args.shot,
        seed=args.split_seed,
    )

    train_idx = np.asarray(bundle.train_val_test_idx["train_idx_local"][0], dtype=np.int64)
    val_idx = np.asarray(bundle.train_val_test_idx["val_idx_local"][0], dtype=np.int64)
    test_idx = np.asarray(bundle.train_val_test_idx["test_idx_local"], dtype=np.int64)

    train_y = np.asarray(bundle.labels["train"][0], dtype=np.int64)
    val_y = np.asarray(bundle.labels["val"][0], dtype=np.int64)
    test_y = np.asarray(bundle.labels["test"], dtype=np.int64)

    graph, targetnode = _load_raw_heterograph(args.root, args.dataset, args.feats_type)

    train_list = [
        _build_single_sample(graph, targetnode, int(node_id), int(label), args.dataset)
        for node_id, label in zip(train_idx, train_y)
    ]
    valid_list = [
        _build_single_sample(graph, targetnode, int(node_id), int(label), args.dataset)
        for node_id, label in zip(val_idx, val_y)
    ]
    test_list = [
        _build_single_sample(graph, targetnode, int(node_id), int(label), args.dataset)
        for node_id, label in zip(test_idx, test_y)
    ]

    _ensure_dir(cache_path.parent)
    with open(cache_path, "wb") as f:
        pk.dump(
            {
                "train": train_list,
                "val": valid_list,
                "test": test_list,
                "targetnode": targetnode,
            },
            f,
        )
    return train_list, valid_list, test_list, targetnode


def _patched_legacy_split_loader(args):
    return load_aligned_legacy_splits(args)


def _resolve_ckpt(cli_args, method: str) -> str:
    def _maybe_format(pattern: str | None):
        if not pattern:
            return None
        return pattern.format(
            seed=cli_args.pretrain_seed,
            pretrain_seed=cli_args.pretrain_seed,
            dataset=cli_args.dataset,
            shot=cli_args.shot,
            method=method,
        )

    if method == "hgmp":
        candidate = _maybe_format(cli_args.hgmp_ckpt_pattern) or cli_args.hgmp_ckpt
    elif method == "typepair":
        candidate = (
            _maybe_format(cli_args.typepair_ckpt_pattern)
            or cli_args.typepair_ckpt
            or _maybe_format(cli_args.hgmp_ckpt_pattern)
            or cli_args.hgmp_ckpt
        )
    elif method == "hgmp_prompt":
        candidate = _maybe_format(cli_args.hgmp_ckpt_pattern) or cli_args.hgmp_ckpt
    elif method == "hgprompt":
        candidate = _maybe_format(cli_args.hgprompt_ckpt_pattern) or cli_args.hgprompt_ckpt
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not candidate:
        raise ValueError(f"No checkpoint configured for method={method}")
    return candidate


def _make_run_seed(split_seed: int, repeat_id: int, run_seed_base: int) -> int:
    return int(run_seed_base) + int(split_seed) * 1000 + int(repeat_id)


def _make_legacy_args(cli_args, method: str, ckpt_path: str, split_seed: int, repeat_id: int):
    run_seed = _make_run_seed(split_seed, repeat_id, cli_args.run_seed_base)
    return SimpleNamespace(
        method=method,
        ckpt=ckpt_path,
        dataset=cli_args.dataset,
        device=torch.device(cli_args.device),
        seed=run_seed,
        split_seed=split_seed,
        shot=cli_args.shot,
        feats_type=cli_args.feats_type,
        hidden_dim=cli_args.hidden_dim,
        num_heads=cli_args.num_heads,
        num_layers=cli_args.num_layers,
        dropout=cli_args.dropout,
        hgnn_type=cli_args.hgnn_type,
        num_samples=cli_args.num_samples,
        num_class=cli_args.num_class,
        classification_type=cli_args.classification_type,
        relation_prompt_mode=cli_args.relation_prompt_mode,
        relation_prompt_alpha=cli_args.relation_prompt_alpha,
        relation_prompt_dropout=cli_args.relation_prompt_dropout,
        relation_prompt_aggr=cli_args.relation_prompt_aggr,
        relation_prompt_use_ln=cli_args.relation_prompt_use_ln,
        embed_batch_size=cli_args.embed_batch_size,
        head_hidden=cli_args.head_hidden,
        head_dropout=cli_args.head_dropout,
        epochs=cli_args.epochs,
        patience=cli_args.patience,
        lr=cli_args.lr,
        prompt_lr=cli_args.prompt_lr,
        weight_decay=cli_args.weight_decay,
        early_stop_metric=cli_args.early_stop_metric,
        save_dir=str(cli_args.save_dir),
        root=cli_args.root,
        splits=cli_args.splits,
    )


def run_legacy_method_once(cli_args, method: str, ckpt_path: str, split_seed: int, repeat_id: int, epoch_callback=None) -> RunRecord:
    args = _make_legacy_args(cli_args, method, ckpt_path, split_seed, repeat_id)
    _set_global_seed(args.seed)

    orig_loader = legacy_bridge._load_legacy_fewshot_splits
    legacy_bridge._load_legacy_fewshot_splits = _patched_legacy_split_loader
    try:
        save_dir = (
            Path(args.save_dir)
            / "aligned_protocol"
            / args.dataset
            / method
            / f"{args.shot}-shot"
            / f"pretrainseed{cli_args.pretrain_seed}"
            / f"splitseed{split_seed}"
            / f"repeat{repeat_id}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = str(save_dir / "best.pt")

        if method == "typepair":
            res = train_typepair_prompt_probe(
                args=args,
                batch_size=args.embed_batch_size,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                head_lr=args.lr,
                prompt_lr=args.prompt_lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=best_path,
                epoch_callback=epoch_callback,
            )
        elif method == "hgmp_prompt":
            res = train_hgmp_heteroprompt_probe(
                args=args,
                batch_size=args.embed_batch_size,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=best_path,
                epoch_callback=epoch_callback,
            )
        else:
            emb = build_legacy_fewshot_embeddings(args=args, batch_size=args.embed_batch_size)
            res = train_mlp_probe(
                x_train=emb.x_train,
                y_train=emb.y_train,
                x_val=emb.x_val,
                y_val=emb.y_val,
                x_test=emb.x_test,
                y_test=emb.y_test,
                in_dim=emb.x_train.size(-1),
                num_classes=args.num_class,
                device=args.device,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=best_path,
                epoch_callback=epoch_callback,
            )
    finally:
        legacy_bridge._load_legacy_fewshot_splits = orig_loader

    return RunRecord(
        method=method,
        split_seed=split_seed,
        repeat_id=repeat_id,
        run_seed=int(args.seed),
        ckpt_path=ckpt_path,
        test_micro=float(res["test_at_best_micro"]),
        test_macro=float(res["test_at_best_macro"]),
        best_epoch=int(res["best_epoch"]),
    )


def _make_hgprompt_args(cli_args, ckpt_path: str, split_seed: int, repeat_id: int, save_dir: Path):
    run_seed = _make_run_seed(split_seed, repeat_id, cli_args.run_seed_base)
    return SimpleNamespace(
        root=cli_args.root,
        dataset=cli_args.dataset,
        splits=cli_args.splits,
        shotnum=cli_args.shot,
        seed=split_seed,
        repeat=1,
        tasknum=1,
        pretrain_ckpt=ckpt_path,
        save_dir=str(save_dir),
        device=cli_args.device,
        strict_load=False,
        feats_type=cli_args.hgprompt_feats_type,
        hidden_dim=cli_args.hgprompt_hidden_dim,
        bottle_net_hidden_dim=cli_args.hgprompt_bottle_net_hidden_dim,
        bottle_net_output_dim=cli_args.hgprompt_bottle_net_output_dim,
        edge_feats=cli_args.hgprompt_edge_feats,
        num_heads=cli_args.hgprompt_num_heads,
        epoch=cli_args.epochs,
        patience=cli_args.patience,
        model_type=cli_args.hgprompt_model_type,
        num_layers=cli_args.hgprompt_num_layers,
        lr=cli_args.hgprompt_lr,
        dropout=cli_args.hgprompt_dropout,
        weight_decay=cli_args.hgprompt_weight_decay,
        slope=cli_args.hgprompt_slope,
        tuning=cli_args.hgprompt_tuning,
        subgraph_hop_num=cli_args.hgprompt_subgraph_hop_num,
        pre_loss_weight=cli_args.hgprompt_pre_loss_weight,
        hetero_pretrain=cli_args.hgprompt_hetero_pretrain,
        hetero_pretrain_subgraph=cli_args.hgprompt_hetero_pretrain_subgraph,
        pretrain_semantic=cli_args.hgprompt_pretrain_semantic,
        pretrain_each_loss=cli_args.hgprompt_pretrain_each_loss,
        add_edge_info2prompt=cli_args.hgprompt_add_edge_info2prompt,
        each_type_subgraph=cli_args.hgprompt_each_type_subgraph,
        cat_prompt_dim=cli_args.hgprompt_cat_prompt_dim,
        cat_hprompt_dim=cli_args.hgprompt_cat_hprompt_dim,
        tuple_neg_disconnected_num=cli_args.hgprompt_tuple_neg_disconnected_num,
        tuple_neg_unrelated_num=cli_args.hgprompt_tuple_neg_unrelated_num,
        semantic_prompt=cli_args.hgprompt_semantic_prompt,
        semantic_prompt_weight=cli_args.hgprompt_semantic_prompt_weight,
        freebase_type=cli_args.hgprompt_freebase_type,
        shgn_hidden_dim=cli_args.hgprompt_shgn_hidden_dim,
        downstream_run_seed=run_seed,
    )


def run_hgprompt_once(cli_args, ckpt_path: str, split_seed: int, repeat_id: int, epoch_callback=None) -> RunRecord:
    run_seed = _make_run_seed(split_seed, repeat_id, cli_args.run_seed_base)
    _set_global_seed(run_seed)

    save_dir = (
        Path(cli_args.save_dir)
        / "aligned_protocol"
        / cli_args.dataset
        / "hgprompt"
        / f"{cli_args.shot}-shot"
        / f"pretrainseed{cli_args.pretrain_seed}"
        / f"splitseed{split_seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    args = _make_hgprompt_args(cli_args, ckpt_path, split_seed, repeat_id, save_dir)
    bundle = load_hgprompt_downstream_bundle(
        root=cli_args.root,
        dataset=cli_args.dataset,
        splits=cli_args.splits,
        shot=cli_args.shot,
        seed=split_seed,
    )
    res = hgprompt_run_once(args, bundle, repeat_id, epoch_callback=epoch_callback)
    return RunRecord(
        method="hgprompt",
        split_seed=split_seed,
        repeat_id=repeat_id,
        run_seed=run_seed,
        ckpt_path=ckpt_path,
        test_micro=float(res.test_micro),
        test_macro=float(res.test_macro),
        best_epoch=int(res.best_epoch),
    )


def _summarize_runs(records: list[RunRecord]):
    if len(records) == 0:
        return {}
    micro = np.array([r.test_micro for r in records], dtype=np.float64)
    macro = np.array([r.test_macro for r in records], dtype=np.float64)
    return {
        "count": int(len(records)),
        "micro_mean": float(micro.mean()),
        "micro_std": float(micro.std()),
        "macro_mean": float(macro.mean()),
        "macro_std": float(macro.std()),
    }


def _aggregate_by_seed(records: list[RunRecord]) -> list[SeedAggregate]:
    out = []
    methods = sorted(set(r.method for r in records))
    for method in methods:
        seeds = sorted(set(r.split_seed for r in records if r.method == method))
        for split_seed in seeds:
            subset = [r for r in records if r.method == method and r.split_seed == split_seed]
            stat = _summarize_runs(subset)
            out.append(
                SeedAggregate(
                    method=method,
                    split_seed=split_seed,
                    count=stat["count"],
                    micro_mean=stat["micro_mean"],
                    micro_std=stat["micro_std"],
                    macro_mean=stat["macro_mean"],
                    macro_std=stat["macro_std"],
                )
            )
    return out


def _summarize_seed_means(seed_rows: list[SeedAggregate], method: str):
    subset = [r for r in seed_rows if r.method == method]
    if len(subset) == 0:
        return {}
    micro = np.array([r.micro_mean for r in subset], dtype=np.float64)
    macro = np.array([r.macro_mean for r in subset], dtype=np.float64)
    return {
        "count": int(len(subset)),
        "micro_mean": float(micro.mean()),
        "micro_std": float(micro.std()),
        "macro_mean": float(macro.mean()),
        "macro_std": float(macro.std()),
    }


def _write_csv(path: Path, rows: list[dict]):
    if len(rows) == 0:
        return
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _build_run_config(args, ckpt_by_method: dict[str, str]):
    config = vars(args).copy()
    config.pop("wandb_key", None)
    config["resolved_ckpt_by_method"] = ckpt_by_method
    return _json_ready(config)


def _make_downstream_epoch_logger(wandb_run, args, method: str, split_seed: int, repeat_id: int, run_seed: int):
    if wandb_run is None:
        return None

    def _callback(metrics: dict):
        payload = {f"downstream/{key}": value for key, value in metrics.items()}
        payload.update(
            {
                "downstream/method": method,
                "downstream/dataset": args.dataset,
                "downstream/shot": args.shot,
                "downstream/split_seed": split_seed,
                "downstream/repeat_id": repeat_id,
                "downstream/run_seed": run_seed,
                "downstream/pretrain_seed": args.pretrain_seed,
            }
        )
        log_metrics(wandb_run, payload)

    return _callback


def build_parser():
    ap = argparse.ArgumentParser("Aligned protocol benchmark for hgmp / typepair / hgprompt")

    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--methods", nargs="+", default=["hgmp", "typepair", "hgprompt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--pretrain_seed", type=int, default=0)
    ap.add_argument("--run_seed_base", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=Path, default=ROOT / "artifacts" / "results" / "protocol_benchmark")

    ap.add_argument("--hgmp_ckpt", type=str, default=None)
    ap.add_argument("--typepair_ckpt", type=str, default=None)
    ap.add_argument("--hgprompt_ckpt", type=str, default=None)
    ap.add_argument("--hgmp_ckpt_pattern", type=str, default=None)
    ap.add_argument("--typepair_ckpt_pattern", type=str, default=None)
    ap.add_argument("--hgprompt_ckpt_pattern", type=str, default=None)

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--hgnn_type", type=str, default="HGT")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--num_class", type=int, default=3)
    ap.add_argument("--classification_type", type=str, default="NIG")
    ap.add_argument("--embed_batch_size", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--prompt_lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop_metric", type=str, default="macro", choices=["micro", "macro"])

    ap.add_argument("--relation_prompt_mode", type=str, default="mul", choices=["mul", "add"])
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--relation_prompt_use_ln", action="store_true")

    ap.add_argument("--hgprompt_feats_type", type=int, default=2)
    ap.add_argument("--hgprompt_hidden_dim", type=int, default=64)
    ap.add_argument("--hgprompt_bottle_net_hidden_dim", type=int, default=2)
    ap.add_argument("--hgprompt_bottle_net_output_dim", type=int, default=64)
    ap.add_argument("--hgprompt_edge_feats", type=int, default=64)
    ap.add_argument("--hgprompt_num_heads", type=int, default=8)
    ap.add_argument("--hgprompt_model_type", type=str, default="gcn", choices=["gcn", "gat", "gin", "SHGN"])
    ap.add_argument("--hgprompt_num_layers", type=int, default=2)
    ap.add_argument("--hgprompt_lr", type=float, default=1.0)
    ap.add_argument("--hgprompt_dropout", type=float, default=0.5)
    ap.add_argument("--hgprompt_weight_decay", type=float, default=1e-6)
    ap.add_argument("--hgprompt_slope", type=float, default=0.05)
    ap.add_argument("--hgprompt_tuning", type=str, default="weight-sum-center-fixed")
    ap.add_argument("--hgprompt_subgraph_hop_num", type=int, default=1)
    ap.add_argument("--hgprompt_pre_loss_weight", type=float, default=1.0)
    ap.add_argument("--hgprompt_hetero_pretrain", type=int, default=0)
    ap.add_argument("--hgprompt_hetero_pretrain_subgraph", type=int, default=0)
    ap.add_argument("--hgprompt_pretrain_semantic", type=int, default=0)
    ap.add_argument("--hgprompt_pretrain_each_loss", type=int, default=0)
    ap.add_argument("--hgprompt_add_edge_info2prompt", type=int, default=1)
    ap.add_argument("--hgprompt_each_type_subgraph", type=int, default=1)
    ap.add_argument("--hgprompt_cat_prompt_dim", type=int, default=64)
    ap.add_argument("--hgprompt_cat_hprompt_dim", type=int, default=64)
    ap.add_argument("--hgprompt_tuple_neg_disconnected_num", type=int, default=1)
    ap.add_argument("--hgprompt_tuple_neg_unrelated_num", type=int, default=1)
    ap.add_argument("--hgprompt_semantic_prompt", type=int, default=1)
    ap.add_argument("--hgprompt_semantic_prompt_weight", type=float, default=0.1)
    ap.add_argument("--hgprompt_freebase_type", type=int, default=0)
    ap.add_argument("--hgprompt_shgn_hidden_dim", type=int, default=3)

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="HGEP")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_job_type", type=str, default="protocol_benchmark_v2")
    ap.add_argument("--wandb_tags", nargs="*", default=[])
    ap.add_argument("--wandb_notes", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    ap.add_argument("--wandb_dir", type=Path, default=ROOT / "artifacts" / "wandb")
    ap.add_argument("--wandb_key", type=str, default=None)
    ap.add_argument("--wandb_key_file", type=Path, default=ROOT / ".codex")

    return ap


def main():
    args = build_parser().parse_args()
    wandb_run = None
    try:
        records: list[RunRecord] = []
        ckpt_by_method = {method: _resolve_ckpt(args, method) for method in args.methods}
        config = _build_run_config(args, ckpt_by_method)

        if args.use_wandb:
            maybe_configure_wandb_env(api_key=args.wandb_key, env_file=args.wandb_key_file)
            wandb_run = init_wandb_run(
                enabled=True,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                group=args.wandb_group or f"{args.dataset}-{args.shot}shot",
                job_type=args.wandb_job_type,
                tags=args.wandb_tags or [args.dataset, f"{args.shot}-shot", "protocol_benchmark_v2"],
                notes=args.wandb_notes,
                mode=args.wandb_mode,
                dir_path=args.wandb_dir,
                config=config,
            )

        for method in args.methods:
            ckpt_path = ckpt_by_method[method]
            print(f"================ method={method} | ckpt={ckpt_path} ================")
            for split_seed in args.seeds:
                for repeat_id in range(args.repeats):
                    run_seed = _make_run_seed(split_seed, repeat_id, args.run_seed_base)
                    epoch_logger = _make_downstream_epoch_logger(
                        wandb_run,
                        args,
                        method=method,
                        split_seed=split_seed,
                        repeat_id=repeat_id,
                        run_seed=run_seed,
                    )
                    if method == "hgprompt":
                        record = run_hgprompt_once(args, ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    elif method == "typepair":
                        record = run_legacy_method_once(args, "typepair", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    elif method == "hgmp":
                        record = run_legacy_method_once(args, "hgmp", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    elif method == "hgmp_prompt":
                        record = run_legacy_method_once(args, "hgmp_prompt", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    records.append(record)
                    print(
                        f"[RUN] method={record.method} | split_seed={record.split_seed} | repeat={record.repeat_id} | "
                        f"run_seed={record.run_seed} | ckpt={record.ckpt_path} | "
                        f"micro={record.test_micro:.4f} | macro={record.test_macro:.4f} | best_epoch={record.best_epoch}"
                    )
                    log_run_record(
                        wandb_run,
                        record,
                        extra={
                            "per_run/dataset": args.dataset,
                            "per_run/shot": args.shot,
                            "per_run/pretrain_seed": args.pretrain_seed,
                        },
                    )

        out_dir = _ensure_dir(args.save_dir / args.dataset / f"{args.shot}-shot")
        per_run_rows = [asdict(r) for r in records]
        _write_csv(out_dir / "per_run.csv", per_run_rows)

        seed_rows = _aggregate_by_seed(records)
        seed_row_dicts = [asdict(r) for r in seed_rows]
        _write_csv(out_dir / "per_seed_summary.csv", seed_row_dicts)

        pooled_summary = {}
        seedmean_summary = {}
        for method in sorted(set(r.method for r in records)):
            pooled_summary[method] = _summarize_runs([r for r in records if r.method == method])
            seedmean_summary[method] = _summarize_seed_means(seed_rows, method)

        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        with open(out_dir / "overall_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": config,
                    "pooled_runs": pooled_summary,
                    "seed_mean_then_std": seedmean_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        log_table(wandb_run, "per_run_table", per_run_rows)
        log_table(wandb_run, "per_seed_summary_table", seed_row_dicts)
        log_nested_summary(wandb_run, "pooled_runs", pooled_summary)
        log_nested_summary(wandb_run, "seed_mean_then_std", seedmean_summary)
        upload_dir_artifact(
            wandb_run,
            out_dir,
            name=f"{args.dataset}-{args.shot}shot-protocol-benchmark-v2-{args.pretrain_seed}",
            artifact_type="benchmark-results",
        )

        print("####################################################")
        print(json.dumps({"pooled_runs": pooled_summary, "seed_mean_then_std": seedmean_summary}, indent=2, ensure_ascii=False))
    except Exception:
        finish_wandb_run(wandb_run, exit_code=1)
        raise
    else:
        finish_wandb_run(wandb_run, exit_code=0)


if __name__ == "__main__":
    main()
