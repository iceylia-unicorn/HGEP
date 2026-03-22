from pathlib import Path
import sys
import argparse
import torch

from protocols.hgmp.run_legacy import prompt_w_h


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_test", action="store_true")

    # faithful 所需参数先补齐
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="HGT")
    ap.add_argument("--num_class", type=int, default=3)

    ap.add_argument("--prompt_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument("--edge_feats", type=int, default=128)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--prompt_epoch", type=int, default=2)
    ap.add_argument("--classification_type", type=str, default="NIG")
    ap.add_argument("--num_samples", type=int, default=100)

    args = ap.parse_args()

    args.device = torch.device(args.device)
    args.shots = args.shot
    args.seed = args.seed
    args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats

    # smoke test：尽量缩小训练量
    if args.smoke_test:
        args.prompt_epoch = 1

    acc, maf1 = prompt_w_h(
        dataname=args.dataset,
        hgnn_type=args.hgnn_type,
        num_class=args.num_class,
        task_type="multi_class_classification",
        pre_method=args.pretext,
        args=args,
    )

    print(f"[HGMP faithful] acc/micro_f1={acc:.4f}, macro_f1={maf1:.4f}")


if __name__ == "__main__":
    main()