from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpbench.downstream.fewshot import load_peprompt_offline_splits  # noqa: E402


def _format_float_for_key(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


RUN_RE = re.compile(
    r"\[RUN\]\s+method=(?P<method>[^|]+)\|\s+split_seed=(?P<split>\d+)\s+\|\s+"
    r"repeat=(?P<repeat>\d+)\s+\|\s+run_seed=(?P<run_seed>\d+).*?\|\s+"
    r"micro=(?P<micro>[0-9.]+)\s+\|\s+macro=(?P<macro>[0-9.]+)\s+\|\s+"
    r"best_epoch=(?P<best_epoch>-?\d+)"
)

FINAL_RE = re.compile(
    r"Final best-state F1\s+\|\s+best_epoch=(?P<best_epoch>-?\d+)\s+\|\s+"
    r"val_f1\(micro/macro\)=(?P<val_micro>[0-9.]+)/(?P<val_macro>[0-9.]+)\s+\|\s+"
    r"test@best_f1\(micro/macro\)=(?P<test_micro>[0-9.]+)/(?P<test_macro>[0-9.]+)"
)


def _cache_subgraph_type(args: argparse.Namespace | SimpleNamespace) -> str:
    subgraph_type = str(getattr(args, "subgraph_type", "khop"))
    if subgraph_type not in {"metapath_topk", "metapath_topk_path", "metapath_topk_adapt", "metapath_topk_path_adapt"}:
        return subgraph_type

    metric = str(getattr(args, "metapath_rank_metric", "count"))
    if subgraph_type in {"metapath_topk_adapt", "metapath_topk_path_adapt"}:
        min_topk = int(getattr(args, "metapath_min_topk", 1))
        max_topk = int(getattr(args, "metapath_max_topk", getattr(args, "metapath_topk", 5)))
        rel_threshold = float(getattr(args, "metapath_rel_threshold", 0.5))
        suffix = (
            f"m{int(getattr(args, 'metapath_max_hop', 3))}_"
            f"k{min_topk}-{max_topk}_"
            f"a{_format_float_for_key(rel_threshold)}_"
            f"{metric}"
        )
    else:
        suffix = (
            f"m{int(getattr(args, 'metapath_max_hop', 3))}_"
            f"k{int(getattr(args, 'metapath_topk', 5))}_"
            f"{metric}"
        )
    if bool(getattr(args, "metapath_keep_self", False)):
        suffix += "_self"

    fusion_mode = str(getattr(args, "peprompt_fusion_mode", "none"))
    ctx_dim = int(getattr(args, "peprompt_ctx_dim", 0) or 0)
    if fusion_mode == "hop_decoupled" and ctx_dim > 0:
        suffix += f"_mpvirt2_d{ctx_dim}"
    elif fusion_mode == "onehop_ctx" and ctx_dim > 0:
        suffix += f"_onehop_d{ctx_dim}"
    elif fusion_mode == "type_ctx" and ctx_dim > 0:
        suffix += f"_typectx_d{ctx_dim}"
    elif fusion_mode in {"graph_summary", "graph_summary_basis"}:
        suffix += "_graphsum"
    elif fusion_mode == "metapath_pos" or str(getattr(args, "peprompt_mp_reg_mode", "none")) != "none":
        suffix += "_mppos"
    return f"{subgraph_type}_{suffix}"


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _min(values: list[float]) -> float:
    return float(min(values)) if values else math.nan


def _max(values: list[float]) -> float:
    return float(max(values)) if values else math.nan


def _sample_graph(sample: Any):
    if isinstance(sample, dict):
        for key in ("graph", "subgraph", "g"):
            if key in sample:
                return sample[key]
    if isinstance(sample, (tuple, list)) and sample:
        return sample[0]
    raise ValueError(f"Unsupported sample format: {type(sample)!r}")


def _num_nodes(graph) -> int:
    return int(sum(int(graph.num_nodes(ntype)) for ntype in graph.ntypes))


def _num_edges(graph) -> int:
    total = 0
    for etype in graph.canonical_etypes:
        total += int(graph.num_edges(etype=etype))
    return total


def _metapath_pos_stats(graph, feature_name: str) -> dict[str, float]:
    total_edges = 0
    edges_with_feat = 0
    edges_with_support = 0
    entries = 0
    positive_entries = 0
    support_mass = 0.0
    active_dims: set[int] = set()

    for etype in graph.canonical_etypes:
        edge_count = int(graph.num_edges(etype=etype))
        total_edges += edge_count
        data = graph.edges[etype].data
        feat = data.get(feature_name)
        if feat is None:
            continue
        feat = feat.detach().cpu()
        if feat.numel() == 0:
            continue
        if feat.dim() == 1:
            feat = feat.view(-1, 1)
        edge_support = feat.abs().sum(dim=-1)
        dim_support = feat.abs().sum(dim=0)

        edges_with_feat += int(feat.size(0))
        edges_with_support += int((edge_support > 0).sum().item())
        entries += int(feat.numel())
        positive_entries += int((feat > 0).sum().item())
        support_mass += float(feat.sum().item())
        active_dims.update(int(i) for i in torch.nonzero(dim_support > 0, as_tuple=False).view(-1).tolist())

    return {
        "mp_pos_edges_with_feat": float(edges_with_feat),
        "mp_pos_edges_with_support": float(edges_with_support),
        "mp_pos_edge_support_ratio": float(edges_with_support / total_edges) if total_edges else math.nan,
        "mp_pos_positive_entries": float(positive_entries),
        "mp_pos_density": float(positive_entries / entries) if entries else math.nan,
        "mp_pos_support_mass": float(support_mass),
        "mp_pos_active_dims": float(len(active_dims)),
    }


def _graph_stats(sample: Any, split_name: str, sample_idx: int, feature_name: str) -> dict[str, Any]:
    graph = _sample_graph(sample)
    present_etypes = sum(1 for etype in graph.canonical_etypes if int(graph.num_edges(etype=etype)) > 0)
    row: dict[str, Any] = {
        "split_name": split_name,
        "sample_idx": int(sample_idx),
        "num_nodes": float(_num_nodes(graph)),
        "num_edges": float(_num_edges(graph)),
        "present_etypes": float(present_etypes),
    }
    row.update(_metapath_pos_stats(graph, feature_name))
    return row


def _label_counts(labels: Any, num_classes: int) -> list[int]:
    y = torch.as_tensor(labels)
    if y.numel() == 0:
        return [0 for _ in range(int(num_classes))]
    if y.dim() == 2:
        counts = (y > 0).long().sum(dim=0)
    else:
        counts = torch.bincount(y.long().view(-1), minlength=int(num_classes))
    return [int(v) for v in counts.tolist()]


def _summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = [
        "num_nodes",
        "num_edges",
        "present_etypes",
        "mp_pos_edge_support_ratio",
        "mp_pos_density",
        "mp_pos_support_mass",
        "mp_pos_active_dims",
    ]
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and not math.isnan(float(r[key]))]
        out[f"{prefix}_{key}_mean"] = _mean(vals)
        out[f"{prefix}_{key}_std"] = _std(vals)
        out[f"{prefix}_{key}_min"] = _min(vals)
        out[f"{prefix}_{key}_max"] = _max(vals)
    return out


def _parse_run_logs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"[warn] log not found: {path}", file=sys.stderr)
            continue
        log_name = path.stem
        pending_final: dict[str, Any] | None = None
        for line in path.read_text(errors="replace").splitlines():
            fm = FINAL_RE.search(line)
            if fm is not None:
                pending_final = {
                    "final_best_epoch": int(fm.group("best_epoch")),
                    "val_micro": float(fm.group("val_micro")),
                    "val_macro": float(fm.group("val_macro")),
                    "test_micro_from_final": float(fm.group("test_micro")),
                    "test_macro_from_final": float(fm.group("test_macro")),
                }
                continue
            m = RUN_RE.search(line)
            if m is None:
                continue
            row = {
                "log_name": log_name,
                "method": m.group("method").strip(),
                "split_seed": int(m.group("split")),
                "repeat": int(m.group("repeat")),
                "run_seed": int(m.group("run_seed")),
                "micro": float(m.group("micro")),
                "macro": float(m.group("macro")),
                "best_epoch": int(m.group("best_epoch")),
            }
            if pending_final is not None:
                row.update(pending_final)
                pending_final = None
            rows.append(row)
    return rows


def _summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["log_name"]), int(row["split_seed"]))].append(row)

    out = []
    for (log_name, split_seed), group in sorted(grouped.items()):
        micros = [float(r["micro"]) for r in group]
        macros = [float(r["macro"]) for r in group]
        epochs = [float(r["best_epoch"]) for r in group]
        val_micros = [float(r["val_micro"]) for r in group if "val_micro" in r]
        val_macros = [float(r["val_macro"]) for r in group if "val_macro" in r]
        out.append(
            {
                "log_name": log_name,
                "split_seed": split_seed,
                "run_count": len(group),
                "micro_mean": _mean(micros),
                "micro_std": _std(micros),
                "macro_mean": _mean(macros),
                "macro_std": _std(macros),
                "val_micro_mean": _mean(val_micros),
                "val_micro_std": _std(val_micros),
                "val_macro_mean": _mean(val_macros),
                "val_macro_std": _std(val_macros),
                "val_test_micro_gap": _mean(val_micros) - _mean(micros),
                "val_test_macro_gap": _mean(val_macros) - _mean(macros),
                "best_epoch_mean": _mean(epochs),
                "best_epoch_std": _std(epochs),
            }
        )
    return out


def _summarize_variance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["log_name"])].append(row)

    out: list[dict[str, Any]] = []
    for log_name, group in sorted(grouped.items()):
        by_split: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_split[int(row["split_seed"])].append(row)

        split_rows = _summarize_runs(group)
        split_micros = [float(r["micro_mean"]) for r in split_rows if str(r["log_name"]) == log_name]
        split_macros = [float(r["macro_mean"]) for r in split_rows if str(r["log_name"]) == log_name]
        within_micro_stds = [float(r["micro_std"]) for r in split_rows if str(r["log_name"]) == log_name]
        within_macro_stds = [float(r["macro_std"]) for r in split_rows if str(r["log_name"]) == log_name]
        all_micros = [float(r["micro"]) for r in group]
        all_macros = [float(r["macro"]) for r in group]

        low_micro_split = min(by_split, key=lambda s: _mean([float(r["micro"]) for r in by_split[s]]))
        high_micro_split = max(by_split, key=lambda s: _mean([float(r["micro"]) for r in by_split[s]]))
        low_macro_split = min(by_split, key=lambda s: _mean([float(r["macro"]) for r in by_split[s]]))
        high_macro_split = max(by_split, key=lambda s: _mean([float(r["macro"]) for r in by_split[s]]))

        out.append(
            {
                "log_name": log_name,
                "run_count": len(group),
                "split_count": len(by_split),
                "pooled_micro_mean": _mean(all_micros),
                "pooled_micro_std": _std(all_micros),
                "pooled_macro_mean": _mean(all_macros),
                "pooled_macro_std": _std(all_macros),
                "between_split_micro_std": _std(split_micros),
                "between_split_macro_std": _std(split_macros),
                "within_split_micro_std_mean": _mean(within_micro_stds),
                "within_split_macro_std_mean": _mean(within_macro_stds),
                "micro_split_range": _max(split_micros) - _min(split_micros),
                "macro_split_range": _max(split_macros) - _min(split_macros),
                "low_micro_split": low_micro_split,
                "low_micro_mean": _mean([float(r["micro"]) for r in by_split[low_micro_split]]),
                "high_micro_split": high_micro_split,
                "high_micro_mean": _mean([float(r["micro"]) for r in by_split[high_micro_split]]),
                "low_macro_split": low_macro_split,
                "low_macro_mean": _mean([float(r["macro"]) for r in by_split[low_macro_split]]),
                "high_macro_split": high_macro_split,
                "high_macro_mean": _mean([float(r["macro"]) for r in by_split[high_macro_split]]),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return math.nan
    xs2, ys2 = zip(*pairs)
    mx = statistics.fmean(xs2)
    my = statistics.fmean(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= 0 or vy <= 0:
        return math.nan
    return float(sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy))


def _print_correlations(joined_rows: list[dict[str, Any]], metric: str = "macro_mean") -> None:
    if not joined_rows:
        return
    candidate_keys = [
        key
        for key in joined_rows[0].keys()
        if key.endswith("_mean") and key not in {"micro_mean", "macro_mean", "best_epoch_mean"}
    ]
    ys = [float(row.get(metric, math.nan)) for row in joined_rows]
    scored = []
    for key in candidate_keys:
        xs = [float(row.get(key, math.nan)) for row in joined_rows]
        corr = _pearson(xs, ys)
        if not math.isnan(corr):
            scored.append((abs(corr), corr, key))
    if not scored:
        return
    print(f"\nTop correlations with {metric}:")
    for _, corr, key in sorted(scored, reverse=True)[:8]:
        print(f"  {key}: r={corr:.4f}")


def analyze(args: argparse.Namespace) -> None:
    cache_key = _cache_subgraph_type(args)
    out_dir = Path(args.out_dir)
    tag = args.tag or f"{args.dataset}_{cache_key}"
    feature_name_default = "peprompt_metapath_pos_feat"

    cache_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for split_seed in args.seeds:
        payload = load_peprompt_offline_splits(
            cache_dir=args.peprompt_offline_cache_dir,
            dataset_name=args.dataset,
            shot=args.shot,
            seed=int(split_seed),
            feats_type=args.feats_type,
            subgraph_type=cache_key,
        )
        feature_name = str(payload.get("peprompt_metapath_pos_feature_name", feature_name_default))
        num_classes = int(payload.get("num_classes", 0) or 0)
        split_row: dict[str, Any] = {
            "dataset": args.dataset,
            "cache_key": cache_key,
            "split_seed": int(split_seed),
            "num_classes": num_classes,
            "metapath_count": int(payload.get("metapath_count", 0) or 0),
            "peprompt_metapath_pos_dim": int(payload.get("peprompt_metapath_pos_dim", 0) or 0),
        }

        for split_name in ("train", "val", "test"):
            samples = list(payload.get(split_name, []))
            if args.max_samples_per_split > 0:
                samples = samples[: int(args.max_samples_per_split)]
            rows = [
                {
                    "dataset": args.dataset,
                    "cache_key": cache_key,
                    "split_seed": int(split_seed),
                    **_graph_stats(sample, split_name, idx, feature_name),
                }
                for idx, sample in enumerate(samples)
            ]
            sample_rows.extend(rows)
            split_row[f"{split_name}_sample_count"] = len(rows)
            split_row.update(_summarize_rows(rows, split_name))

            labels_key = f"{split_name}_labels"
            if labels_key in payload and num_classes > 0:
                counts = _label_counts(payload[labels_key], num_classes)
                split_row[f"{split_name}_label_min"] = min(counts) if counts else 0
                split_row[f"{split_name}_label_max"] = max(counts) if counts else 0
                split_row[f"{split_name}_label_imbalance"] = (
                    float(max(counts) / max(1, min(counts))) if counts else math.nan
                )
                for class_id, count in enumerate(counts):
                    split_row[f"{split_name}_label_c{class_id}"] = int(count)

        cache_rows.append(split_row)

    run_rows = _parse_run_logs([Path(p) for p in args.log_paths])
    run_summary_rows = _summarize_runs(run_rows)
    variance_rows = _summarize_variance(run_rows)

    cache_by_split = {int(row["split_seed"]): row for row in cache_rows}
    joined_rows: list[dict[str, Any]] = []
    for run_row in run_summary_rows:
        cache_row = cache_by_split.get(int(run_row["split_seed"]), {})
        joined_rows.append({**cache_row, **run_row})

    _write_csv(out_dir / f"{tag}.cache_summary.csv", cache_rows)
    _write_csv(out_dir / f"{tag}.run_summary.csv", run_summary_rows)
    _write_csv(out_dir / f"{tag}.variance_summary.csv", variance_rows)
    _write_csv(out_dir / f"{tag}.joined_summary.csv", joined_rows)
    if args.write_sample_stats:
        _write_csv(out_dir / f"{tag}.sample_stats.csv", sample_rows)

    print(f"cache_key={cache_key}")
    print(f"wrote {out_dir / f'{tag}.cache_summary.csv'}")
    print(f"wrote {out_dir / f'{tag}.run_summary.csv'}")
    print(f"wrote {out_dir / f'{tag}.variance_summary.csv'}")
    print(f"wrote {out_dir / f'{tag}.joined_summary.csv'}")
    if args.write_sample_stats:
        print(f"wrote {out_dir / f'{tag}.sample_stats.csv'}")

    if joined_rows:
        print("\nPer-log split results:")
        for row in sorted(joined_rows, key=lambda r: (str(r.get("log_name")), int(r.get("split_seed", -1)))):
            print(
                f"  {row['log_name']} split={row['split_seed']} "
                f"micro={float(row['micro_mean']):.4f} macro={float(row['macro_mean']):.4f} "
                f"val_macro={float(row.get('val_macro_mean', math.nan)):.4f} "
                f"gap={float(row.get('val_test_macro_gap', math.nan)):.4f} "
                f"train_edges={float(row.get('train_num_edges_mean', math.nan)):.1f} "
                f"test_edges={float(row.get('test_num_edges_mean', math.nan)):.1f}"
            )
        if variance_rows:
            print("\nVariance decomposition:")
            for row in variance_rows:
                print(
                    f"  {row['log_name']} "
                    f"pooled_macro={float(row['pooled_macro_mean']):.4f} "
                    f"pooled_std={float(row['pooled_macro_std']):.4f} "
                    f"between_split_std={float(row['between_split_macro_std']):.4f} "
                    f"within_split_std_mean={float(row['within_split_macro_std_mean']):.4f} "
                    f"low_split={row['low_macro_split']}:{float(row['low_macro_mean']):.4f} "
                    f"high_split={row['high_macro_split']}:{float(row['high_macro_mean']):.4f}"
                )
        _print_correlations(joined_rows, metric="macro_mean")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose PEPrompt offline cache and benchmark split stability.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--feats_type", type=int, default=0)
    parser.add_argument("--subgraph_type", default="metapath_topk")
    parser.add_argument("--metapath_max_hop", type=int, default=3)
    parser.add_argument("--metapath_topk", type=int, default=3)
    parser.add_argument("--metapath_min_topk", type=int, default=1)
    parser.add_argument("--metapath_max_topk", type=int, default=5)
    parser.add_argument("--metapath_rel_threshold", type=float, default=0.5)
    parser.add_argument("--metapath_rank_metric", default="count")
    parser.add_argument("--metapath_keep_self", action="store_true")
    parser.add_argument("--peprompt_fusion_mode", default="none")
    parser.add_argument("--peprompt_ctx_dim", type=int, default=0)
    parser.add_argument("--peprompt_mp_reg_mode", choices=["none", "consistency", "predict"], default="none")
    parser.add_argument("--peprompt_offline_cache_dir", default="artifacts/cache/peprompt_offline_splits")
    parser.add_argument("--log_paths", nargs="*", default=[])
    parser.add_argument("--out_dir", default="artifacts/analysis/peprompt_split_diagnostics")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--write_sample_stats", action="store_true")
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
