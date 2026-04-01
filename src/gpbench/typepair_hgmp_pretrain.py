from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import torch

from gpbench.protocol_bridge.hgmp_typepair import pretrain_typepair_legacy


def main():
    ap = argparse.ArgumentParser()

    # basic
    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=1)

    # faithful HGMP args
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="HGT")
    ap.add_argument("--num_class", type=int, default=3)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--classification_type", type=str, default="NIG")

    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--pre_lr", type=float, default=1e-3)
    ap.add_argument("--aug_ration", type=float, default=0.2)

    ap.add_argument("--prompt_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument("--edge_feats", type=int, default=128)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--prompt_epoch", type=int, default=1)
    ap.add_argument("--schedule_step", type=int, default=300)
    ap.add_argument("--use_norm", action="store_true")

    # type-pair relation prompt args
    ap.add_argument("--relation_prompt_mode", type=str, default="mul", choices=["mul", "add"])
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--relation_prompt_use_ln", action="store_true")

    args = ap.parse_args()

    # keep naming aligned with legacy HGMP runner
    args.pre_epoch = args.epochs
    args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats

    args.device = torch.device(args.device)
    print(f"[INFO] using device: {args.device}")
    print(
        "[INFO] method=typepair_hgmp_bridge | "
        f"dataset={args.dataset} | backbone={args.hgnn_type} | pretext={args.pretext}"
    )

    pretrain_typepair_legacy(args)


if __name__ == "__main__":
    main()