# 曾经的下游脚本
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpbench.data.loaders import load_hgb_node_task
from gpbench.pretrain import build_hetero_neighbor_loader
from gpbench.downstream.model import (
    DownstreamConfig,
    HeteroFeaturePrompt,
    MLPHead,
    PromptedSubgraphClassifier,
    load_frozen_encoder,
)
from gpbench.downstream.fewshot import load_split_file, train_fewshot_subgraph


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB"])
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ckpt", type=str, required=True, help="pretrain checkpoint path (.pt)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # subgraph sampling
    ap.add_argument("--tau_hops", type=int, default=2)
    ap.add_argument("--fanout", type=int, nargs="+", default=[25, 15])
    ap.add_argument("--batch_size", type=int, default=64)

    # encoder cfg (must match pretraining)
    ap.add_argument("--backbone", type=str, default="hgt", choices=["hgt", "to_hetero_gcn", "to_hetero_gat"])
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--enc_dropout", type=float, default=0.2)

    # downstream cfg
    ap.add_argument("--prompt_mode", type=str, default="mul", choices=["mul", "add"])
    ap.add_argument("--prompt_dropout", type=float, default=0.1)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop_metric", type=str, default="macro", choices=["macro", "micro"])

    ap.add_argument("--save_dir", type=str, default="checkpoints/downstream")
    args = ap.parse_args()

    device = torch.device(args.device)

    task = load_hgb_node_task(args.root, args.dataset)
    data = task.data
    target_ntype = task.target_ntype
    dataset_name = task.dataset_name

    split = load_split_file(args.splits, dataset_name, args.shot, args.seed)
    train_idx = split["train_idx"].long()
    val_idx = split.get("val_idx", torch.empty(0, dtype=torch.long)).long()
    test_idx = split["test_idx"].long()

    # build hetero subgraph loaders: one seed node => one induced graph sample
    train_loader = build_hetero_neighbor_loader(
        data=data,
        input_ntype=target_ntype,
        batch_size=args.batch_size,
        num_hops=args.tau_hops,
        fanout=args.fanout,
        seed_nodes=train_idx,
        shuffle=True,
        num_workers=0,
        disjoint=True,
    )

    val_loader = None
    if val_idx.numel() > 0:
        val_loader = build_hetero_neighbor_loader(
            data=data,
            input_ntype=target_ntype,
            batch_size=args.batch_size,
            num_hops=args.tau_hops,
            fanout=args.fanout,
            seed_nodes=val_idx,
            shuffle=False,
            num_workers=0,
            disjoint=True,
        )

    test_loader = build_hetero_neighbor_loader(
        data=data,
        input_ntype=target_ntype,
        batch_size=args.batch_size,
        num_hops=args.tau_hops,
        fanout=args.fanout,
        seed_nodes=test_idx,
        shuffle=False,
        num_workers=0,
        disjoint=True,
    )

    # frozen encoder
    enc, _ = load_frozen_encoder(
        full_data=data,
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.enc_dropout,
        device=device,
    )

    num_classes = int(task.y.max().item()) + 1

    dcfg = DownstreamConfig(
        prompt_mode=args.prompt_mode,
        prompt_dropout=args.prompt_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        use_layernorm=True,
    )

    # hetero prompt over ALL node types
    prompt = HeteroFeaturePrompt(
        node_types=data.node_types,
        dim=args.hidden_dim,
        mode=dcfg.prompt_mode,
        dropout=dcfg.prompt_dropout,
        use_ln=dcfg.use_layernorm,
    ).to(device)

    head = MLPHead(
        in_dim=args.hidden_dim,
        hidden=dcfg.head_hidden,
        num_classes=num_classes,
        dropout=dcfg.head_dropout,
        use_ln=True,
    ).to(device)

    model = PromptedSubgraphClassifier(
        encoder=enc,
        prompt=prompt,
        head=head,
        input_ntype=target_ntype,
    ).to(device)

    save_dir = Path(args.save_dir) / dataset_name / f"{args.shot}-shot" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "best_prompt_head.pt")

    res = train_fewshot_subgraph(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        input_ntype=target_ntype,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        save_best_path=best_path,
        early_stop_metric=args.early_stop_metric,
    )

    print(
        f"[DONE] {dataset_name} shot={args.shot} seed={args.seed} | "
        f"best_val_f1(micro/macro)={res.best_val_micro:.4f}/{res.best_val_macro:.4f} | "
        f"test@best_f1(micro/macro)={res.test_at_best_micro:.4f}/{res.test_at_best_macro:.4f} | "
        f"best_epoch={res.best_epoch} | monitor={res.early_stop_metric}"
    )


if __name__ == "__main__":
    main()