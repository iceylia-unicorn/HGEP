# scripts/pretrain_hgmp.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpbench.data.loaders import load_hgb_node_task
from gpbench.pretrain import (
    build_hetero_neighbor_loader,
    hetero_node_masking,
    hetero_edge_permutation,
    nt_xent,
    HGMPPretrainModel,
    PretrainModelConfig,
)
from gpbench.utils.early_stop import EarlyStopping, EarlyStopConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB"])

    ap.add_argument("--tau_hops", type=int, default=2)
    ap.add_argument("--fanout", type=int, nargs="+", default=[25, 15])  # per hop
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # HGMP-style augmentation
    ap.add_argument("--aug_ratio", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.2)

    # model
    ap.add_argument("--backbone", type=str, default="hgt", choices=["hgt", "to_hetero_gcn", "to_hetero_gat"])
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--min_delta", type=float, default=0.0) # 早停阈值
    ap.add_argument("--es_save", type=str, default="checkpoints/pretrain_hgmp_best.pt")
    ap.add_argument("--save", type=str, default="checkpoints/pretrain_hgmp.pt")
    args = ap.parse_args()

    task = load_hgb_node_task(args.root, args.dataset)
    data = task.data
    input_ntype = task.target_ntype  # 默认用任务目标类型做 seed（ACM=paper）

    device = torch.device(args.device)

    loader = build_hetero_neighbor_loader(
        data=data,
        input_ntype=input_ntype,
        batch_size=args.batch_size,
        num_hops=args.tau_hops,
        fanout=args.fanout,
        shuffle=True,
        num_workers=0,
    )

    cfg = PretrainModelConfig(
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_node_id_emb_if_missing_x=True,
    )

    model = HGMPPretrainModel(data, cfg).to(device)
    es = EarlyStopping(EarlyStopConfig(
        patience=args.patience,
        min_delta=args.min_delta,
        mode="min",
        save_best=True,
        save_path=args.es_save,
    ))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = 0

        for batch in loader:
            # 注意：augment 会复制/改 edge_index/x，建议先在 CPU 上做，再搬到 GPU（更稳）
            g1 = hetero_edge_permutation(hetero_node_masking(batch, args.aug_ratio), args.aug_ratio)
            g2 = hetero_edge_permutation(hetero_node_masking(batch, args.aug_ratio), args.aug_ratio)

            g1 = g1.to(device)
            g2 = g2.to(device)

            z1 = model(g1, input_ntype)  # [B, proj_dim]
            z2 = model(g2, input_ntype)  # [B, proj_dim]
            loss = nt_xent(z1, z2, tau=args.temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"Epoch {epoch:03d} | loss={avg:.4f}")
        should_stop = es.step(
            metric=avg,
            epoch=epoch,
            model=model,
            extra_state={
                "cfg": cfg.__dict__,
                "dataset": task.dataset_name,
                "input_ntype": input_ntype,
            },
        )

        print(f"EarlyStop | best={es.best:.4f} @ epoch={es.best_epoch} | bad_epochs={es.bad_epochs}/{args.patience}")

        if should_stop:
            print("Early stopping triggered.")
            break

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg.__dict__,
            "dataset": task.dataset_name,
            "input_ntype": input_ntype,
        },
        str(save_path),
    )
    print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
