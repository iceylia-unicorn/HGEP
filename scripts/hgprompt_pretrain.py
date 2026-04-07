from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from protocols.hgprompt.source.pretrain import run_model_DBLP


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--device", type=int, default="cuda")   # -1 => cpu
    ap.add_argument("--epoch", type=int, default=1)

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
    ap.add_argument("--patience", type=int, default=3)

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

    args = ap.parse_args()

    print(f"[INFO] HGPrompt pretrain smoke | dataset={args.dataset} | device={args.device}")
    run_model_DBLP(args)


if __name__ == "__main__":
    main()