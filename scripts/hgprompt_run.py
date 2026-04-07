from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from protocols.hgprompt.runner import build_parser as upstream_build_parser
from protocols.hgprompt.runner import run_hgprompt_downstream


def apply_benchmark_defaults(args):
    args.feats_type = 2
    args.hidden_dim = 64
    args.bottle_net_hidden_dim = 2
    args.bottle_net_output_dim = 64
    args.edge_feats = 64
    args.num_heads = 8
    args.model_type = "gcn"
    args.num_layers = 2
    args.lr = 1.0
    args.dropout = 0.5
    args.weight_decay = 1e-6
    args.slope = 0.05
    args.tuning = "weight-sum-center-fixed"
    args.subgraph_hop_num = 1
    args.pre_loss_weight = 1.0
    args.hetero_pretrain = 0
    args.hetero_pretrain_subgraph = 0
    args.pretrain_semantic = 0
    args.pretrain_each_loss = 0
    args.add_edge_info2prompt = 1
    args.each_type_subgraph = 1
    args.cat_prompt_dim = 64
    args.cat_hprompt_dim = 64
    args.tuple_neg_disconnected_num = 1
    args.tuple_neg_unrelated_num = 1
    args.semantic_prompt = 1
    args.semantic_prompt_weight = 0.1
    args.freebase_type = 0
    args.shgn_hidden_dim = 3
    return args


def build_parser():
    upstream = upstream_build_parser()
    ap = argparse.ArgumentParser(parents=[upstream], add_help=False)
    # ap.add_argument("--ckpt", dest="pretrain_ckpt", type=str, default=None)
    # ap.add_argument("--shot", dest="shotnum", type=int, default=None)
    ap.add_argument("--benchmark_defaults", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()

    if args.pretrain_ckpt is None:
        raise ValueError("Please provide --pretrain_ckpt or --ckpt")

    if args.shotnum is None:
        args.shotnum = 10

    if args.benchmark_defaults:
        args = apply_benchmark_defaults(args)

    print(
        f"[INFO] HGPrompt downstream | dataset={args.dataset} | shot={args.shotnum} | seed={args.seed} | "
        f"repeat={args.repeat} | ckpt={args.pretrain_ckpt}"
    )
    run_hgprompt_downstream(args)


if __name__ == "__main__":
    main()
