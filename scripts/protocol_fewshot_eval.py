from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch

from gpbench.protocol_bridge.downstream_legacy import (
    build_legacy_fewshot_embeddings,
    train_mlp_probe,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--method", type=str, required=True, choices=["hgmp", "typepair"])
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # legacy HGMP-compatible args
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--hgnn_type", type=str, default="HGT")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--num_class", type=int, default=3)
    ap.add_argument("--classification_type", type=str, default="NIG")

    # typepair cfg (only used when method=typepair)
    ap.add_argument("--relation_prompt_mode", type=str, default="mul", choices=["mul", "add"])
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--relation_prompt_use_ln", action="store_true")

    # embedding extraction
    ap.add_argument("--embed_batch_size", type=int, default=32)

    # probe head
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop_metric", type=str, default="macro", choices=["micro", "macro"])

    ap.add_argument("--save_dir", type=str, default="artifacts/checkpoints/protocol_eval")

    args = ap.parse_args()
    args.device = torch.device(args.device)

    if args.dataset == "ACM" and args.num_class == 3:
        pass
    elif args.dataset == "DBLP" and args.num_class == 3:
        print("[WARN] DBLP often uses num_class=4 in this codebase. Check your setting.")

    print(
        f"[INFO] protocol_fewshot_eval | method={args.method} | "
        f"dataset={args.dataset} | shot={args.shot} | seed={args.seed}"
    )

    emb = build_legacy_fewshot_embeddings(
        args=args,
        batch_size=args.embed_batch_size,
    )

    save_dir = (
        Path(args.save_dir)
        / args.dataset
        / args.method
        / f"{args.shot}-shot"
        / f"seed{args.seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "best_head.pt")

    res = train_mlp_probe(
        x_train=emb.x_train,
        y_train=emb.y_train,
        x_val=emb.x_val,
        y_val=emb.y_val,
        x_test=emb.x_test,
        y_test=emb.y_test,
        in_dim=emb.x_train.size(-1),
        num_classes=args.num_class,
        device=args.device,
        hidden_dim=args.head_hidden,
        dropout=args.head_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        save_best_path=best_path,
    )

    print(
        f"[DONE] {args.dataset} | method={args.method} | shot={args.shot} | seed={args.seed} | "
        f"best_val_f1(micro/macro)={res['best_val_micro']:.4f}/{res['best_val_macro']:.4f} | "
        f"test@best_f1(micro/macro)={res['test_at_best_micro']:.4f}/{res['test_at_best_macro']:.4f} | "
        f"best_epoch={res['best_epoch']} | monitor={res['early_stop_metric']}"
    )


if __name__ == "__main__":
    main()