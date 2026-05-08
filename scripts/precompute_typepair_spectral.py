from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts import protocol_benchmark_v2 as benchmark


def build_parser():
    ap = argparse.ArgumentParser("Offline spectral cache builder for typepair edge prompts")
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument(
        "--typepair_spectral_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "typepair_spectral_embeddings",
    )
    ap.add_argument("--typepair_spectral_dim", type=int, default=16)
    ap.add_argument("--typepair_spectral_max_nodes", type=int, default=50000)
    return ap


def main():
    args = build_parser().parse_args()
    payload, cache_path, cache_hit = benchmark.prepare_typepair_spectral_payload(args)
    summary = {
        "dataset": args.dataset,
        "feats_type": args.feats_type,
        "spectral_dim": int(payload["spectral_dim"]),
        "total_nodes": int(payload["total_nodes"]),
        "total_edges": int(payload["total_edges"]),
        "generation_seconds": float(payload["generation_seconds"]),
        "cache_hit": bool(cache_hit),
        "cache_path": str(cache_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
