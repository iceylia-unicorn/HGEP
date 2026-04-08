from pathlib import Path
import sys
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import shutil

import torch

from protocols.hgmp.run_legacy import PRETRAIN_DIR, prompt_w_h


DATASET_NUM_CLASS = {
    "ACM": 3,
    "DBLP": 4,
    "IMDB": 5,
    "Freebase": 7,
}


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
    args.prompt_lr = 1e-3
    args.head_lr = 1e-3
    args.weight_decay = 5e-4
    args.slope = 0.05
    args.patience = 30
    args.repeat = 1
    args.prompt_epoch = 200
    args.classification_type = "NIG"
    args.num_samples = 100
    if args.dataset in DATASET_NUM_CLASS:
        args.num_class = DATASET_NUM_CLASS[args.dataset]
    return args


def resolve_expected_legacy_ckpt_path(args) -> Path:
    return PRETRAIN_DIR / f"{args.dataset}.{args.pretext}.{args.hgnn_type}.hid{args.hidden_dim}.np{args.num_samples}.pth"


def stage_ckpt_for_legacy(ckpt_path: str, expected_path: Path) -> tuple[Path, bool]:
    src = Path(ckpt_path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    same_file = False
    try:
        same_file = src.resolve() == expected_path.resolve()
    except Exception:
        same_file = False

    if not same_file:
        shutil.copy2(src, expected_path)
    return expected_path, (not same_file)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--benchmark_defaults", action="store_true")
    ap.add_argument("--save_dir", type=str, default="artifacts/checkpoints/hgmp/downstream")

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="GCN")
    ap.add_argument("--num_class", type=int, default=None)

    ap.add_argument("--prompt_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument("--edge_feats", type=int, default=None)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--prompt_epoch", type=int, default=200)
    ap.add_argument("--classification_type", type=str, default="NIG")
    ap.add_argument("--num_samples", type=int, default=100)

    args = ap.parse_args()

    if args.benchmark_defaults:
        args = apply_benchmark_defaults(args)

    if args.num_class is None:
        args.num_class = DATASET_NUM_CLASS.get(args.dataset, 3)

    args.device = normalize_device(args.device)
    args.shots = args.shot
    args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats

    if args.smoke_test:
        args.prompt_epoch = 1

    expected_ckpt = resolve_expected_legacy_ckpt_path(args)
    staged_ckpt, copied = stage_ckpt_for_legacy(args.ckpt, expected_ckpt)

    print(
        f"[INFO] HGMP downstream | dataset={args.dataset} | shot={args.shot} | seed={args.seed} | "
        f"repeat={args.repeat} | ckpt={args.ckpt}"
    )
    print(f"[INFO] legacy expected ckpt path: {expected_ckpt}")
    if copied:
        print(f"[INFO] copied ckpt to legacy path: {staged_ckpt}")

    acc, maf1 = prompt_w_h(
        dataname=args.dataset,
        hgnn_type=args.hgnn_type,
        num_class=args.num_class,
        task_type="multi_class_classification",
        pre_method=args.pretext,
        args=args,
    )

    save_dir = Path(args.save_dir) / args.dataset / f"{args.shot}-shot" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / f"hgmp.{args.hgnn_type.lower()}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "shot": args.shot,
                "seed": args.seed,
                "repeat": args.repeat,
                "ckpt": args.ckpt,
                "legacy_ckpt": str(staged_ckpt),
                "hgnn_type": args.hgnn_type,
                "num_class": args.num_class,
                "micro_f1": float(acc),
                "macro_f1": float(maf1),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[HGMP faithful] acc/micro_f1={acc:.4f}, macro_f1={maf1:.4f}")
    print(f"[DONE] summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
