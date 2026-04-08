from pathlib import Path
import sys
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import shutil

import numpy as np
import torch

from protocols.hgmp.run_legacy import pretrain


DATASET_NUM_CLASS = {
    "ACM": 3,
    "DBLP": 4,
    "IMDB": 5,
    "Freebase": 7,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device(raw_device: Union[str, int]) -> torch.device:
    if isinstance(raw_device, int):
        return torch.device(f"cuda:{raw_device}") if torch.cuda.is_available() else torch.device("cpu")
    raw = str(raw_device).strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if raw.startswith("cuda:"):
        return torch.device(raw) if torch.cuda.is_available() else torch.device("cpu")
    try:
        idx = int(raw)
        return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:
        raise ValueError(f"Unsupported device spec: {raw_device}")


def apply_benchmark_defaults(args):
    args.feats_type = 0
    args.hidden_dim = 128
    args.num_heads = 2
    args.num_layers = 2
    args.dropout = 0.5
    args.pretext = "GraphCL"
    args.hgnn_type = "GCN"
    args.num_samples = 100
    args.pre_lr = 1e-3
    args.aug_ration = 0.2
    args.prompt_lr = 1e-3
    args.head_lr = 1e-3
    args.weight_decay = 5e-4
    args.patience = 7
    args.repeat = 1
    args.prompt_epoch = 1
    args.schedule_step = 300
    args.use_norm = False
    args.edge_feats = args.hidden_dim
    if args.dataset in DATASET_NUM_CLASS:
        args.num_class = DATASET_NUM_CLASS[args.dataset]
    return args


def resolve_upstream_ckpt_path(args) -> Path:
    base = Path("artifacts/checkpoints/hgmp/pretrain")
    return base / f"{args.dataset}.{args.pretext}.{args.hgnn_type}.hid{args.hidden_dim}.np{args.num_samples}.pth"


def build_alias_path(args) -> Path:
    if args.ckpt_alias is not None:
        return Path(args.ckpt_alias)

    save_dir = Path(args.save_dir)
    filename = (
        f"{args.dataset}.{args.pretext}.{args.hgnn_type}.hid{args.hidden_dim}."
        f"np{args.num_samples}.seed{args.seed}.pth"
    )
    return save_dir / filename


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="GCN")
    ap.add_argument("--num_class", type=int, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--classification_type", type=str, default="NIG")

    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--pre_lr", type=float, default=1e-3)
    ap.add_argument("--aug_ration", type=float, default=0.2)

    ap.add_argument("--prompt_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument("--edge_feats", type=int, default=None)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--prompt_epoch", type=int, default=1)
    ap.add_argument("--schedule_step", type=int, default=300)
    ap.add_argument("--use_norm", action="store_true")

    ap.add_argument("--benchmark_defaults", action="store_true")
    ap.add_argument("--save_dir", type=str, default="artifacts/checkpoints/hgmp/pretrain")
    ap.add_argument("--ckpt_alias", type=str, default=None)
    ap.add_argument("--print_ckpt_only", action="store_true")

    args = ap.parse_args()

    if args.benchmark_defaults:
        args = apply_benchmark_defaults(args)

    if args.num_class is None:
        args.num_class = DATASET_NUM_CLASS.get(args.dataset, 3)

    args.pre_epoch = args.epochs
    args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats
    args.device = normalize_device(args.device)

    set_seed(args.seed)

    upstream_ckpt = resolve_upstream_ckpt_path(args)
    alias_ckpt = build_alias_path(args)

    print(
        f"[INFO] HGMP pretrain | dataset={args.dataset} | device={args.device} | seed={args.seed} | "
        f"pretext={args.pretext} | hgnn_type={args.hgnn_type} | hidden_dim={args.hidden_dim} | "
        f"num_samples={args.num_samples}"
    )
    print(f"[INFO] upstream best ckpt path: {upstream_ckpt}")
    print(f"[INFO] benchmark alias ckpt path: {alias_ckpt}")

    if args.print_ckpt_only:
        return

    pretrain(args)

    if not upstream_ckpt.exists():
        raise FileNotFoundError(f"Expected upstream checkpoint not found: {upstream_ckpt}")

    alias_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if upstream_ckpt.resolve() != alias_ckpt.resolve():
        shutil.copy2(upstream_ckpt, alias_ckpt)

    meta = {
        "dataset": args.dataset,
        "device": str(args.device),
        "seed": args.seed,
        "epochs": args.epochs,
        "feats_type": args.feats_type,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pretext": args.pretext,
        "hgnn_type": args.hgnn_type,
        "num_class": args.num_class,
        "num_samples": args.num_samples,
        "pre_lr": args.pre_lr,
        "aug_ration": args.aug_ration,
        "upstream_ckpt": str(upstream_ckpt),
        "alias_ckpt": str(alias_ckpt),
    }
    meta_path = alias_ckpt.with_suffix(alias_ckpt.suffix + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] benchmark-ready ckpt copied to: {alias_ckpt}")
    print(f"[DONE] run metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
