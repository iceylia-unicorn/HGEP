#!/usr/bin/env python3
"""Fair graph-classification benchmark on HGMP GIG samples.

The protocol is intentionally strict:
  - all methods consume the same HGMP legacy GIG graph samples;
  - all methods use the same split seed / shot / repeat schedule;
  - PEPrompt adds PE edge features to these GIG graphs instead of resampling
    metapath subgraphs.

HGPrompt's current adapter in this repository is a node-index/full-graph
downstream protocol, not a graph-sample protocol.  It is therefore rejected by
this runner unless a dedicated graph-batch HGPrompt adapter is implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import dgl
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocols.hgmp import data_legacy as legacy_data_module
from protocols.hgmp.utils_legacy import seed_everything
from scripts.peprompt_benchmark import (
    DATASET_NUM_CLASS,
    HOP_NUM,
    PEPROMPT_EDGE_FEATURE_NAME,
    TARGET_NODETYPE,
    _load_raw_heterograph,
    prepare_peprompt_spectral_payload,
)
from scripts.precompute_peprompt_cache import (
    _attach_peprompt_edge_features_from_global_pe,
    _build_global_edge_pe_tables,
)
from gpbench.protocol_bridge import downstream_legacy as legacy_bridge
from gpbench.protocol_bridge.downstream_legacy import (
    build_legacy_fewshot_embeddings,
    train_hgmp_heteroprompt_probe,
    train_mlp_probe,
    train_peprompt_probe,
)


DEFAULT_CKPTS = {
    "ACM": "artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth",
    "DBLP": "artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth",
    "IMDB": "artifacts/checkpoints/hgmp/pretrain/IMDB.GraphCL.GCN.hid512.np500.seed0.pth",
    "Freebase": "artifacts/checkpoints/hgmp/pretrain/Freebase.GraphCL.GCN.hid512.np500.seed0.pth",
}


@dataclass
class RunRecord:
    method: str
    dataset: str
    classification_type: str
    shot: int
    split_seed: int
    repeat_id: int
    run_seed: int
    ckpt_path: str
    test_micro: float
    test_macro: float
    best_epoch: int


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def _summarize(records: list[RunRecord]) -> dict:
    out = {}
    for method in sorted({r.method for r in records}):
        rows = [r for r in records if r.method == method]
        out[method] = {
            "pooled_micro": _mean_std(r.test_micro for r in rows),
            "pooled_macro": _mean_std(r.test_macro for r in rows),
        }

        seed_rows = []
        for split_seed in sorted({r.split_seed for r in rows}):
            seed_records = [r for r in rows if r.split_seed == split_seed]
            seed_rows.append(
                {
                    "split_seed": split_seed,
                    "micro_mean": float(np.mean([r.test_micro for r in seed_records])),
                    "macro_mean": float(np.mean([r.test_macro for r in seed_records])),
                }
            )
        out[method]["per_seed"] = seed_rows
        out[method]["seed_mean_micro"] = _mean_std(row["micro_mean"] for row in seed_rows)
        out[method]["seed_mean_macro"] = _mean_std(row["macro_mean"] for row in seed_rows)
    return out


def _write_csv(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _jsonable_config(args) -> dict:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = value
    return out


def _make_run_seed(split_seed: int, repeat_id: int, run_seed_base: int) -> int:
    return int(run_seed_base) + int(split_seed) * 1000 + int(repeat_id)


def _legacy_gig_file_paths(root: Path, dataset: str, num_class: int, hop_num: int, feats_type: int) -> list[Path]:
    base = root / dataset.lower() / "induced_graphs"
    return [
        base / f"task{task_id}.hop{hop_num}.ft{feats_type}"
        for task_id in range(2 * int(num_class), 3 * int(num_class))
    ]


def _check_gig_files(args) -> None:
    num_class = int(args.num_class or DATASET_NUM_CLASS[args.dataset])
    hop_num = int(args.hop_num if args.hop_num is not None else HOP_NUM[args.dataset])
    paths = _legacy_gig_file_paths(Path(args.hgmp_legacy_data_root), args.dataset, num_class, hop_num, args.feats_type)
    missing = [path for path in paths if not path.exists()]
    if missing and bool(args.prepare_if_missing):
        _prepare_gig_files(args)
        missing = [path for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            "Missing HGMP legacy GIG graph files. Generate them with the HGMP legacy "
            "preprocess pipeline before running this benchmark, or pass --prepare_if_missing.\n"
            f"Expected examples:\n{preview}"
        )


def _prepare_gig_files(args) -> None:
    from protocols.hgmp import preprocess_legacy as legacy_preprocess

    graph, targetnode = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    num_class = int(args.num_class or DATASET_NUM_CLASS[args.dataset])
    hop_num = int(args.hop_num if args.hop_num is not None else HOP_NUM[args.dataset])

    old_data_root = legacy_preprocess.DATA_ROOT
    legacy_preprocess.DATA_ROOT = Path(args.hgmp_legacy_data_root)
    try:
        print(
            f"[prepare-gig] dataset={args.dataset} num_class={num_class} "
            f"targetnode={targetnode} hop={hop_num} feats_type={args.feats_type} "
            f"root={legacy_preprocess.DATA_ROOT}"
        )
        legacy_preprocess.induced_graphs_graphs(
            graph,
            dataname=args.dataset,
            num_classes=num_class,
            smallest_size=50,
            largest_size=300,
            targetnode=targetnode,
            feats_type=args.feats_type,
            hop_num=hop_num,
        )
    finally:
        legacy_preprocess.DATA_ROOT = old_data_root


def _load_gig_splits(args, split_seed: int):
    seed_everything(int(split_seed))
    old_root = legacy_data_module.DATA_ROOT
    legacy_data_module.DATA_ROOT = Path(args.hgmp_legacy_data_root)
    try:
        train_list, valid_list, test_list = legacy_data_module.multi_class_NIG(
            dataname=args.dataset,
            num_class=int(args.num_class),
            shots=int(args.shot),
            classification_type="GIG",
            feats_type=int(args.feats_type),
        )
    finally:
        legacy_data_module.DATA_ROOT = old_root

    if not train_list:
        raise RuntimeError("Loaded empty GIG train split.")
    return train_list, valid_list, test_list, TARGET_NODETYPE[args.dataset]


def _clone_label(label):
    if isinstance(label, torch.Tensor):
        return label.detach().cpu().clone()
    return torch.as_tensor(label, dtype=torch.long)


def _copy_graph_samples(samples):
    return [(graph.clone(), _clone_label(label)) for graph, label in samples]


def _build_peprompt_edge_payload(args):
    graph, _targetnode = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    spectral_args = SimpleNamespace(
        root=args.root,
        dataset=args.dataset,
        feats_type=args.feats_type,
        peprompt_spectral_cache_dir=args.peprompt_spectral_cache_dir,
        peprompt_spectral_dim=args.peprompt_spectral_dim,
        peprompt_spectral_max_nodes=args.peprompt_spectral_max_nodes,
    )
    spectral_payload, cache_path, cache_hit = prepare_peprompt_spectral_payload(spectral_args)
    print(
        f"[peprompt-spectral] dim={spectral_payload['spectral_dim']} "
        f"cache_hit={cache_hit} path={cache_path}"
    )
    edge_pe_tables = _build_global_edge_pe_tables(
        graph=graph,
        spectral_embeddings=spectral_payload["spectral_embeddings"],
        node_offsets=spectral_payload["node_offsets"],
    )
    return spectral_payload, edge_pe_tables


def _attach_peprompt_features_to_samples(samples, spectral_payload: dict, edge_pe_tables: dict, feature_name: str):
    out = _copy_graph_samples(samples)
    for graph, _label in out:
        _attach_peprompt_edge_features_from_global_pe(
            subgraph=graph,
            spectral_embeddings=spectral_payload["spectral_embeddings"],
            node_offsets=spectral_payload["node_offsets"],
            feature_name=feature_name,
            edge_pe_tables=edge_pe_tables,
        )
    return out


@contextmanager
def _patched_split_loader(train_list, valid_list, test_list, targetnode: str):
    orig = legacy_bridge._load_legacy_fewshot_splits
    legacy_bridge._load_legacy_fewshot_splits = lambda _args: (train_list, valid_list, test_list, targetnode)
    try:
        yield
    finally:
        legacy_bridge._load_legacy_fewshot_splits = orig


def _base_args(args, method: str, split_seed: int, repeat_id: int) -> SimpleNamespace:
    run_seed = _make_run_seed(split_seed, repeat_id, args.run_seed_base)
    return SimpleNamespace(
        method=method,
        dataset=args.dataset,
        root=args.root,
        ckpt=args.ckpt,
        device=args.device,
        seed=run_seed,
        shot=args.shot,
        feats_type=args.feats_type,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        hgnn_type=args.hgnn_type,
        num_samples=args.num_samples,
        num_class=args.num_class,
        classification_type="GIG",
        early_stop_metric=args.early_stop_metric,
        prompt_lr=args.prompt_lr,
        # HGMP Prompt options
        hgmp_prompt_recipe=args.hgmp_prompt_recipe,
        hgmp_prompt_prompt_lr=args.hgmp_prompt_prompt_lr,
        hgmp_prompt_head_lr=args.hgmp_prompt_head_lr,
        hgmp_prompt_weight_decay=args.hgmp_prompt_weight_decay,
        hgmp_prompt_batch_size=args.hgmp_prompt_batch_size,
        hgmp_prompt_epochs=args.hgmp_prompt_epochs,
        hgmp_prompt_patience=args.hgmp_prompt_patience,
        hgmp_prompt_early_stop_mode=args.hgmp_prompt_early_stop_mode,
        hgmp_prompt_eval_mode=args.hgmp_prompt_eval_mode,
        # PEPrompt options
        relation_prompt_mode=args.relation_prompt_mode,
        relation_prompt_alpha=args.relation_prompt_alpha,
        relation_prompt_dropout=args.relation_prompt_dropout,
        relation_prompt_aggr=args.relation_prompt_aggr,
        relation_prompt_use_ln=args.relation_prompt_use_ln,
        peprompt_edge_feature_dim=args.peprompt_spectral_dim,
        peprompt_edge_feature_names=["SpectralEmbeddingDiff"],
        peprompt_edge_feature_name=args.peprompt_edge_feature_name,
        peprompt_edge_prompt_hidden=args.peprompt_edge_prompt_hidden,
        peprompt_edge_dropout=args.peprompt_edge_dropout,
        peprompt_head_type="mlp",
        peprompt_early_stop_mode=args.peprompt_early_stop_mode,
        peprompt_eval_mode=args.peprompt_eval_mode,
    )


def _run_once(
    *,
    args,
    method: str,
    split_seed: int,
    repeat_id: int,
    train_list,
    valid_list,
    test_list,
    targetnode: str,
    save_dir: Path,
) -> RunRecord:
    run_args = _base_args(args, method, split_seed, repeat_id)
    best_path = save_dir / method / f"splitseed{split_seed}" / f"repeat{repeat_id}" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    with _patched_split_loader(train_list, valid_list, test_list, targetnode):
        if method == "hgmp":
            emb = build_legacy_fewshot_embeddings(run_args, batch_size=args.batch_size)
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
                save_best_path=str(best_path),
                dataset=args.dataset,
                classification_type="GIG",
            )
        elif method == "hgmp_prompt":
            res = train_hgmp_heteroprompt_probe(
                args=run_args,
                batch_size=args.batch_size,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=str(best_path),
            )
        elif method == "peprompt":
            res = train_peprompt_probe(
                args=run_args,
                batch_size=args.batch_size,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                head_lr=args.lr,
                prompt_lr=args.prompt_lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=str(best_path),
            )
        else:
            raise ValueError(f"Unsupported method for fair GIG graph classification: {method}")

    return RunRecord(
        method=method,
        dataset=args.dataset,
        classification_type="GIG",
        shot=int(args.shot),
        split_seed=int(split_seed),
        repeat_id=int(repeat_id),
        run_seed=_make_run_seed(split_seed, repeat_id, args.run_seed_base),
        ckpt_path=str(args.ckpt),
        test_micro=float(res["test_at_best_micro"]),
        test_macro=float(res["test_at_best_macro"]),
        best_epoch=int(res["best_epoch"]),
    )


def run_benchmark(args) -> dict:
    if "hgprompt" in args.methods:
        raise NotImplementedError(
            "Current HGPrompt adapter is node-index/full-graph downstream, not HGMP GIG graph-sample "
            "classification. Do not include it in the fair GIG protocol until a graph-batch HGPrompt "
            "adapter is implemented."
        )

    args.num_class = int(args.num_class or DATASET_NUM_CLASS[args.dataset])
    args.ckpt = args.ckpt or DEFAULT_CKPTS[args.dataset]
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    _check_gig_files(args)

    out_dir = Path(args.save_dir) / args.dataset / "GIG" / f"{args.shot}-shot"
    out_dir.mkdir(parents=True, exist_ok=True)

    peprompt_payload = None
    if "peprompt" in args.methods:
        peprompt_payload = _build_peprompt_edge_payload(args)

    records: list[RunRecord] = []
    for split_seed in args.seeds:
        base_train, base_valid, base_test, targetnode = _load_gig_splits(args, int(split_seed))
        print(
            f"[split] seed={split_seed} train={len(base_train)} "
            f"val={len(base_valid)} test={len(base_test)} targetnode={targetnode}"
        )

        method_samples = {}
        for method in args.methods:
            if method == "peprompt":
                assert peprompt_payload is not None
                spectral_payload, edge_pe_tables = peprompt_payload
                method_samples[method] = (
                    _attach_peprompt_features_to_samples(
                        base_train,
                        spectral_payload,
                        edge_pe_tables,
                        args.peprompt_edge_feature_name,
                    ),
                    _attach_peprompt_features_to_samples(
                        base_valid,
                        spectral_payload,
                        edge_pe_tables,
                        args.peprompt_edge_feature_name,
                    ),
                    _attach_peprompt_features_to_samples(
                        base_test,
                        spectral_payload,
                        edge_pe_tables,
                        args.peprompt_edge_feature_name,
                    ),
                )
            else:
                method_samples[method] = (
                    _copy_graph_samples(base_train),
                    _copy_graph_samples(base_valid),
                    _copy_graph_samples(base_test),
                )

        for repeat_id in range(int(args.repeats)):
            for method in args.methods:
                train_list, valid_list, test_list = method_samples[method]
                print(f"[run] method={method} split_seed={split_seed} repeat={repeat_id}")
                record = _run_once(
                    args=args,
                    method=method,
                    split_seed=int(split_seed),
                    repeat_id=int(repeat_id),
                    train_list=train_list,
                    valid_list=valid_list,
                    test_list=test_list,
                    targetnode=targetnode,
                    save_dir=out_dir / "checkpoints",
                )
                records.append(record)
                print(
                    f"[result] method={method} split_seed={split_seed} repeat={repeat_id} "
                    f"micro={record.test_micro:.4f} macro={record.test_macro:.4f} "
                    f"best_epoch={record.best_epoch}"
                )

    summary = _summarize(records)
    _write_csv(out_dir / "per_run.csv", records)
    payload = {
        "config": _jsonable_config(args),
        "summary": summary,
        "records": [asdict(r) for r in records],
    }
    with (out_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {out_dir / 'per_run.csv'}")
    print(f"[saved] {out_dir / 'overall_summary.json'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Fair graph classification on HGMP GIG graph samples")
    ap.add_argument("--dataset", choices=sorted(DATASET_NUM_CLASS), default="DBLP")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--hgmp_legacy_data_root", type=Path, default=Path("data"))
    ap.add_argument("--methods", nargs="+", default=["hgmp", "hgmp_prompt", "peprompt"])
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--run_seed_base", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=Path, default=ROOT / "artifacts" / "results" / "gig_graph_classification")
    ap.add_argument("--ckpt", type=str, default=None)

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hop_num", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--hgnn_type", type=str, default="GCN")
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--num_class", type=int, default=None)
    ap.add_argument("--prepare_if_missing", action="store_true")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--prompt_lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--early_stop_metric", choices=["micro", "macro"], default="macro")

    ap.add_argument("--hgmp_prompt_recipe", choices=["legacy", "bridge"], default="legacy")
    ap.add_argument("--hgmp_prompt_prompt_lr", type=float, default=None)
    ap.add_argument("--hgmp_prompt_head_lr", type=float, default=None)
    ap.add_argument("--hgmp_prompt_weight_decay", type=float, default=None)
    ap.add_argument("--hgmp_prompt_batch_size", type=int, default=None)
    ap.add_argument("--hgmp_prompt_epochs", type=int, default=None)
    ap.add_argument("--hgmp_prompt_patience", type=int, default=None)
    ap.add_argument("--hgmp_prompt_early_stop_mode", choices=["auto", "metric", "legacy_loss"], default="legacy_loss")
    ap.add_argument("--hgmp_prompt_eval_mode", choices=["full", "early_stop_only"], default="early_stop_only")

    ap.add_argument("--relation_prompt_mode", choices=["mul", "add"], default="mul")
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", choices=["mean", "sum"], default="mean")
    ap.add_argument("--relation_prompt_use_ln", action="store_true")
    ap.add_argument("--peprompt_edge_feature_name", type=str, default=PEPROMPT_EDGE_FEATURE_NAME)
    ap.add_argument("--peprompt_spectral_cache_dir", type=Path, default=ROOT / "artifacts" / "cache" / "peprompt_spectral_embeddings")
    ap.add_argument("--peprompt_spectral_dim", type=int, default=16)
    ap.add_argument("--peprompt_spectral_max_nodes", type=int, default=50000)
    ap.add_argument("--peprompt_edge_prompt_hidden", type=int, default=128)
    ap.add_argument("--peprompt_edge_dropout", type=float, default=0.0)
    ap.add_argument("--peprompt_early_stop_mode", choices=["metric", "loss"], default="loss")
    ap.add_argument("--peprompt_eval_mode", choices=["full", "early_stop_only"], default="early_stop_only")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.methods = [str(method) for method in args.methods]
    invalid = sorted(set(args.methods) - {"hgmp", "hgmp_prompt", "peprompt", "hgprompt"})
    if invalid:
        raise SystemExit(f"Unsupported methods: {', '.join(invalid)}")
    if args.hop_num is not None and int(args.hop_num) != HOP_NUM[args.dataset]:
        raise SystemExit(
            f"HGMP legacy multi_class_NIG uses fixed hop_num={HOP_NUM[args.dataset]} for {args.dataset}; "
            f"got --hop_num {args.hop_num}."
        )
    run_benchmark(args)


if __name__ == "__main__":
    main()
