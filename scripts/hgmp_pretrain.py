from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch

from protocols.hgmp.run_legacy import pretrain


def main():
    ap = argparse.ArgumentParser()

    # 基本参数
    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=1)

    # faithful 所需参数：先全部补齐
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="HGT")  # 先固定 HGT，避开 SHGN
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

    args = ap.parse_args()

    # 兼容 HGMP 原始命名
    args.pre_epoch = args.epochs
    args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats

    # 设备处理
    args.device = torch.device(args.device)
    print(f"[INFO] using device: {args.device}")

    pretrain(args)


if __name__ == "__main__":
    main()