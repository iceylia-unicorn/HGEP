from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch

from gpbench.protocol_bridge.dispatch import run_protocol_pretrain


def build_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["hgmp", "hgprompt", "typepair"],
    )

    # shared high-level args
    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    # mostly HGMP-aligned args
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="HGT")
    ap.add_argument("--num_class", type=int, default=3)

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

    # HGPrompt-compatible args
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--model_type", type=str, default="gcn")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tuple_neg_disconnected_num", type=int, default=1)
    ap.add_argument("--tuple_neg_unrelated_num", type=int, default=1)
    ap.add_argument("--target_tuple_neg_disconnected_num", type=int, default=1)
    ap.add_argument("--subgraph_hop_num", type=int, default=1)
    ap.add_argument("--subgraph_neighbor_num_bar", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--loss_weight", type=float, default=1.0)
    ap.add_argument("--hetero_pretrain", type=int, default=0)
    ap.add_argument("--target_pretrain", type=int, default=0)
    ap.add_argument("--hetero_subgraph", type=int, default=0)
    ap.add_argument("--semantic_weight", type=int, default=0)
    ap.add_argument("--each_loss", type=int, default=0)
    ap.add_argument("--freebase_type", type=int, default=2)
    ap.add_argument("--run", type=int, default=1)

    # type-pair args
    ap.add_argument("--relation_prompt_mode", type=str, default="mul", choices=["mul", "add"])
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--relation_prompt_use_ln", action="store_true")

    return ap


def normalize_args(args):
    # legacy HGMP runner expects these names
    args.pre_epoch = args.epochs
    args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats

    # HGPrompt runner expects integer device ids or -1
    # We keep the raw string device for hgmp/typepair, and add a helper field for hgprompt.
    if args.device == "cpu":
        args.hgprompt_device = -1
    elif args.device.startswith("cuda:"):
        try:
            args.hgprompt_device = int(args.device.split(":")[1])
        except Exception:
            args.hgprompt_device = 0
    elif args.device == "cuda":
        args.hgprompt_device = 0
    else:
        args.hgprompt_device = -1

    return args


def main():
    ap = build_parser()
    args = ap.parse_args()
    args = normalize_args(args)

    if args.method in {"hgmp", "typepair"}:
        args.device = torch.device(args.device)

    if args.method == "hgprompt":
        args.device = args.hgprompt_device
        args.epoch = args.epochs
        args.model_type = args.model_type.lower()

    print(
        f"[INFO] protocol_pretrain | method={args.method} | "
        f"dataset={args.dataset} | seed={args.seed}"
    )
    run_protocol_pretrain(args.method, args)


if __name__ == "__main__":
    main()