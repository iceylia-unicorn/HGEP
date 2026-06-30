from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse

from gpbench.downstream.fewshot import save_peprompt_split_ids
from scripts.peprompt_benchmark import _load_raw_heterograph
from scripts.precompute_peprompt_cache import _strict_kshot_rest_split, _target_labels


def main():
    ap = argparse.ArgumentParser("Materialize lightweight PEPrompt split-id sidecars without building subgraphs")
    ap.add_argument("--dataset", type=str, required=True, choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--shots", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--split_subgraph_type", type=str, default="khop")
    ap.add_argument("--max_pool_size", type=int, default=0)
    ap.add_argument(
        "--peprompt_offline_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_offline_splits",
    )
    args = ap.parse_args()

    graph, targetnode = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    labels = _target_labels(graph, targetnode)

    for shot in args.shots:
        for seed in args.seeds:
            split = _strict_kshot_rest_split(
                labels=labels,
                shot=int(shot),
                split_seed=int(seed),
                max_pool_size=int(args.max_pool_size),
            )
            payload = {
                **split,
                "dataset": args.dataset,
                "shot": int(shot),
                "split_seed": int(seed),
                "feats_type": int(args.feats_type),
                "subgraph_cache_key": str(args.split_subgraph_type),
            }
            path = save_peprompt_split_ids(
                cache_dir=args.peprompt_offline_cache_dir,
                dataset_name=args.dataset,
                shot=shot,
                seed=seed,
                feats_type=args.feats_type,
                subgraph_type=args.split_subgraph_type,
                split_payload=payload,
            )
            print(
                f"[saved] dataset={args.dataset} shot={shot} seed={seed} "
                f"subgraph_key={args.split_subgraph_type} path={path}"
            )


if __name__ == "__main__":
    main()
