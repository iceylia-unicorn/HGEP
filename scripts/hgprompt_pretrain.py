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

from protocols.hgprompt.source.pretrain import run_model_DBLP


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device(raw_device: Union[str, int]) -> int:
    if isinstance(raw_device, int):
        return raw_device
    raw = str(raw_device).strip().lower()
    if raw == "cpu":
        return -1
    if raw == "cuda":
        return 0
    if raw.startswith("cuda:"):
        return int(raw.split(":", 1)[1])
    return int(raw)


def apply_benchmark_defaults(args):
    args.feats_type = 2
    args.hidden_dim = 64
    args.num_heads = 8
    args.num_layers = 2
    args.model_type = "gcn"
    args.lr = 1e-3
    args.dropout = 0.5
    args.weight_decay = 1e-6
    args.slope = 0.05
    args.repeat = 1
    args.patience = 30
    args.tuple_neg_disconnected_num = 1
    args.tuple_neg_unrelated_num = 1
    args.target_tuple_neg_disconnected_num = 1
    args.subgraph_hop_num = 1
    args.subgraph_neighbor_num_bar = 10
    args.temperature = 1.0
    args.loss_weight = 1.0
    args.hetero_pretrain = 0
    args.target_pretrain = 0
    args.hetero_subgraph = 0
    args.semantic_weight = 0
    args.each_loss = 0
    args.freebase_type = 2
    args.edge_feats = 64
    return args


def resolve_upstream_ckpt_path(args) -> Path:
    base = Path("artifacts/checkpoints/hgprompt/pretrain")

    if args.model_type == "SHGN":
        return base / f"shgn_checkpoint_{args.dataset}_{args.num_layers}.pt"

    if args.dataset == "Freebase":
        if args.hetero_pretrain:
            if args.hetero_subgraph:
                if args.semantic_weight:
                    name = (
                        f"checkpoint_hsubgraph_semantic_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                        f"{args.loss_weight}_{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                    )
                elif args.each_loss:
                    name = (
                        f"checkpoint_hsubgraph_each_loss_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                        f"{args.loss_weight}_{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                    )
                else:
                    name = (
                        f"checkpoint_hsubgraph_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                        f"{args.loss_weight}_{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                    )
            else:
                name = (
                    f"checkpoint_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_{args.loss_weight}_"
                    f"{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                )
        else:
            if args.hetero_subgraph:
                if args.semantic_weight:
                    name = (
                        f"checkpoint_hsubgraph_semantic_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                        f"{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                    )
                elif args.each_loss:
                    name = (
                        f"checkpoint_hsubgraph_each_loss_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                        f"{args.loss_weight}_{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                    )
                else:
                    name = (
                        f"checkpoint_hsubgraph_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                        f"{args.feats_type}_{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                    )
            else:
                name = (
                    f"checkpoint_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_{args.feats_type}_"
                    f"{args.tuple_neg_disconnected_num}_{args.tuple_neg_unrelated_num}.pt"
                )
        return base / name

    if args.hetero_pretrain:
        if args.hetero_subgraph:
            if args.semantic_weight:
                name = (
                    f"checkpoint_hsubgraph_semantic_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                    f"{args.loss_weight}_{args.feats_type}.pt"
                )
            elif args.each_loss:
                name = (
                    f"checkpoint_hsubgraph_each_loss_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                    f"{args.loss_weight}_{args.feats_type}.pt"
                )
            else:
                name = (
                    f"checkpoint_hsubgraph_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                    f"{args.loss_weight}_{args.feats_type}.pt"
                )
        else:
            name = (
                f"checkpoint_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_{args.loss_weight}_{args.feats_type}.pt"
            )
    else:
        if args.hetero_subgraph:
            if args.semantic_weight:
                name = (
                    f"checkpoint_hsubgraph_semantic_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_{args.feats_type}.pt"
                )
            elif args.each_loss:
                name = (
                    f"checkpoint_hsubgraph_each_loss_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_"
                    f"{args.loss_weight}_{args.feats_type}.pt"
                )
            else:
                name = (
                    f"checkpoint_hsubgraph_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_{args.feats_type}.pt"
                )
        else:
            name = f"checkpoint_{args.dataset}_{args.model_type}_{args.subgraph_hop_num}_{args.feats_type}.pt"
    return base / name


def build_alias_path(args) -> Path:
    if args.ckpt_alias is not None:
        return Path(args.ckpt_alias)

    save_dir = Path(args.save_dir)
    filename = (
        f"{args.dataset}.{args.model_type}.ft{args.feats_type}.hop{args.subgraph_hop_num}."
        f"seed{args.seed}.best.pt"
    )
    return save_dir / filename


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epoch", "--epochs", dest="epoch", type=int, default=200)

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--model_type", type=str, default="gcn")

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--patience", type=int, default=30)

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
    ap.add_argument("--edge_feats", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=int, default=1)

    ap.add_argument("--benchmark_defaults", action="store_true")
    ap.add_argument("--save_dir", type=str, default="artifacts/checkpoints/hgprompt/pretrain")
    ap.add_argument("--ckpt_alias", type=str, default=None)
    ap.add_argument("--print_ckpt_only", action="store_true")

    args = ap.parse_args()

    if args.benchmark_defaults:
        args = apply_benchmark_defaults(args)

    args.device = normalize_device(args.device)
    set_seed(args.seed)

    upstream_ckpt = resolve_upstream_ckpt_path(args)
    alias_ckpt = build_alias_path(args)

    print(
        f"[INFO] HGPrompt pretrain | dataset={args.dataset} | device={args.device} | seed={args.seed} | "
        f"model={args.model_type} | feats_type={args.feats_type}"
    )
    print(f"[INFO] upstream best ckpt path: {upstream_ckpt}")
    print(f"[INFO] benchmark alias ckpt path: {alias_ckpt}")

    if args.print_ckpt_only:
        return

    run_model_DBLP(args)

    if not upstream_ckpt.exists():
        raise FileNotFoundError(f"Expected upstream checkpoint not found: {upstream_ckpt}")

    alias_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(upstream_ckpt, alias_ckpt)

    meta = {
        "dataset": args.dataset,
        "device": args.device,
        "seed": args.seed,
        "epoch": args.epoch,
        "feats_type": args.feats_type,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "model_type": args.model_type,
        "lr": args.lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "slope": args.slope,
        "repeat": args.repeat,
        "patience": args.patience,
        "tuple_neg_disconnected_num": args.tuple_neg_disconnected_num,
        "tuple_neg_unrelated_num": args.tuple_neg_unrelated_num,
        "target_tuple_neg_disconnected_num": args.target_tuple_neg_disconnected_num,
        "subgraph_hop_num": args.subgraph_hop_num,
        "temperature": args.temperature,
        "loss_weight": args.loss_weight,
        "hetero_pretrain": args.hetero_pretrain,
        "target_pretrain": args.target_pretrain,
        "hetero_subgraph": args.hetero_subgraph,
        "semantic_weight": args.semantic_weight,
        "each_loss": args.each_loss,
        "freebase_type": args.freebase_type,
        "edge_feats": args.edge_feats,
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
