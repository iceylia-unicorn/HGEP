from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.peprompt_benchmark import TARGET_NODETYPE, _load_raw_heterograph
from scripts.subgraph_sampling_stats import (
    _build_csr_adjs,
    _generate_metapaths,
    _metapath_reachable_scores,
    _target_ids,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _score_stats(scores: np.ndarray, topk: int) -> dict[str, float | int]:
    nonzero = scores[scores > 0]
    if nonzero.size == 0:
        return {
            "candidate_count": 0,
            "top1": 0.0,
            "top2": 0.0,
            "top1_eq_top2": 0,
            "top1_top2_ratio": math.nan,
            "selected_count": 0,
            "selected_unique_count": 0,
            "selected_all_equal": 0,
            "selected_std": math.nan,
            "boundary_tie_size": 0,
            "boundary_tie_fraction": math.nan,
        }

    ordered = np.sort(nonzero)[::-1]
    selected = ordered[: min(int(topk), ordered.size)]
    top1 = float(ordered[0])
    top2 = float(ordered[1]) if ordered.size > 1 else 0.0
    kth = float(selected[-1])
    boundary_tie_size = int(np.sum(nonzero == kth))
    return {
        "candidate_count": int(nonzero.size),
        "top1": top1,
        "top2": top2,
        "top1_eq_top2": int(ordered.size > 1 and top1 == top2),
        "top1_top2_ratio": float(top1 / top2) if top2 > 0 else math.nan,
        "selected_count": int(selected.size),
        "selected_unique_count": int(np.unique(selected).size),
        "selected_all_equal": int(selected.size > 1 and float(selected[0]) == float(selected[-1])),
        "selected_std": float(np.std(selected)) if selected.size > 1 else 0.0,
        "boundary_tie_size": boundary_tie_size,
        "boundary_tie_fraction": float(boundary_tie_size / nonzero.size),
    }


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else math.nan


def _summarize(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {}
    return {
        "records": len(rows),
        "candidate_count_mean": _mean([r["candidate_count"] for r in rows]),
        "candidate_count_median": float(np.median([r["candidate_count"] for r in rows])),
        "top1_mean": _mean([r["top1"] for r in rows]),
        "top2_mean": _mean([r["top2"] for r in rows]),
        "top1_eq_top2_rate": _mean([r["top1_eq_top2"] for r in rows]),
        "top1_top2_ratio_mean": _mean([r["top1_top2_ratio"] for r in rows]),
        "selected_unique_count_mean": _mean([r["selected_unique_count"] for r in rows]),
        "selected_all_equal_rate": _mean([r["selected_all_equal"] for r in rows]),
        "selected_std_mean": _mean([r["selected_std"] for r in rows]),
        "boundary_tie_size_mean": _mean([r["boundary_tie_size"] for r in rows]),
        "boundary_tie_fraction_mean": _mean([r["boundary_tie_fraction"] for r in rows]),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    graph, target_ntype = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    if args.target_ntype:
        target_ntype = args.target_ntype
    if not target_ntype:
        target_ntype = TARGET_NODETYPE[args.dataset]

    ids = _target_ids(graph, target_ntype, args.sample_size, args.seed)
    adjs = _build_csr_adjs(graph)
    metapaths = _generate_metapaths(graph, target_ntype, args.max_hop)
    if args.max_metapaths > 0:
        metapaths = metapaths[: args.max_metapaths]

    rows: list[dict] = []
    by_metapath: dict[str, list[dict]] = defaultdict(list)
    for node_id in ids:
        for metapath_id, metapath in enumerate(metapaths):
            scores = _metapath_reachable_scores(
                adjs=adjs,
                metapath=metapath,
                node_id=int(node_id),
                target_ntype=target_ntype,
                keep_self=bool(args.keep_self),
            )
            stats = _score_stats(scores, args.topk)
            if args.drop_empty and stats["candidate_count"] == 0:
                continue
            key = "->".join([metapath[0][0]] + [etype[2] for etype in metapath])
            row = {
                "dataset": args.dataset,
                "target_ntype": target_ntype,
                "target_id": int(node_id),
                "metapath_id": int(metapath_id),
                "metapath": key,
                "hop": len(metapath),
                "topk": int(args.topk),
                **stats,
            }
            rows.append(row)
            by_metapath[key].append(row)

    summary = {
        "dataset": args.dataset,
        "target_ntype": target_ntype,
        "sample_size": int(len(ids)),
        "max_hop": int(args.max_hop),
        "topk": int(args.topk),
        "num_metapaths": int(len(metapaths)),
        "overall": _summarize(rows),
        "by_metapath": {key: _summarize(value) for key, value in by_metapath.items()},
    }

    out_dir = _ensure_dir(args.out_dir)
    stem = f"{args.dataset}_h{args.max_hop}_k{args.topk}_count_gap"
    _write_csv(out_dir / f"{stem}.rows.csv", rows)
    with open(out_dir / f"{stem}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Analyze whether metapath count scores have enough ranking margin.")
    parser.add_argument("--dataset", default="DBLP", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--feats_type", type=int, default=0)
    parser.add_argument("--target_ntype", default=None)
    parser.add_argument("--max_hop", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_metapaths", type=int, default=0)
    parser.add_argument("--keep_self", action="store_true")
    parser.add_argument("--drop_empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "artifacts" / "analysis" / "metapath_count_gaps")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
