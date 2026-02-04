# scripts/finetune_fewshot.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from gpbench.data.loaders import load_hgb_node_task
from gpbench.downstream.model import (
    DownstreamConfig, TypePrompt, MLPHead,
    load_frozen_encoder, encode_all_nodes,
)
from gpbench.downstream.fewshot import load_split_file, train_fewshot


class PromptClassifier(nn.Module):
    """
    最基础下游：prompt(z) -> head -> logits
    """
    def __init__(self, prompt: TypePrompt, head: MLPHead, target_ntype: str):
        super().__init__()
        self.prompt = prompt
        self.head = head
        self.target_ntype = target_ntype

    def forward(self, z_target: torch.Tensor) -> torch.Tensor:
        z = self.prompt(z_target, self.target_ntype)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB"])
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ckpt", type=str, required=True, help="pretrain checkpoint path (.pt)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # encoder cfg（必须跟预训练一致）
    ap.add_argument("--backbone", type=str, default="hgt", choices=["hgt", "to_hetero_gcn", "to_hetero_gat"])
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--enc_dropout", type=float, default=0.2)

    # downstream cfg
    ap.add_argument("--prompt_mode", type=str, default="add", choices=["add", "mul"])
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
    dataset_name = task.dataset_name  # e.g. "hgb-acm"

    split = load_split_file(args.splits, dataset_name, args.shot, args.seed)
    train_idx = split["train_idx"].long()
    val_idx = split.get("val_idx", torch.empty(0, dtype=torch.long)).long()
    test_idx = split["test_idx"].long()

    # 冻结 encoder + 一次性编码
    enc, ckpt_meta = load_frozen_encoder(
        full_data=data,
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.enc_dropout,
        device=device,
    )

    x_dict = encode_all_nodes(enc, data, device=device)
    z_target = x_dict[target_ntype]          # [N_target, hidden_dim]
    y = task.y.long()                        # [N_target]

    num_classes = int(y.max().item()) + 1

    dcfg = DownstreamConfig(
        prompt_mode=args.prompt_mode,
        prompt_dropout=args.prompt_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        use_layernorm=True,
    )

    prompt = TypePrompt(node_types=[target_ntype], dim=args.hidden_dim, mode=dcfg.prompt_mode,
                        dropout=dcfg.prompt_dropout, use_ln=dcfg.use_layernorm).to(device)
    head = MLPHead(in_dim=args.hidden_dim, hidden=dcfg.head_hidden, num_classes=num_classes,
                   dropout=dcfg.head_dropout, use_ln=True).to(device)

    model = PromptClassifier(prompt, head, target_ntype).to(device)

    save_dir = Path(args.save_dir) / dataset_name / f"{args.shot}-shot" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "best_prompt_head.pt")

    res = train_fewshot(
        z_target=z_target,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model=model,
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
