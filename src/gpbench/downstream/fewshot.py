# src/gpbench/downstream/fewshot.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from gpbench.utils.early_stop import EarlyStopping, EarlyStopConfig


@dataclass
class FewShotResult:
    best_val_micro: float
    best_val_macro: float
    test_at_best_micro: float
    test_at_best_macro: float
    best_epoch: int
    early_stop_metric: str  # "micro" or "macro"


def f1_micro_macro(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> tuple[float, float]:
    pred = logits.argmax(dim=-1)

    cm = torch.zeros((num_classes, num_classes), device=logits.device, dtype=torch.long)
    for t, p in zip(y.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1

    tp = cm.diag().to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp

    eps = 1e-12

    support = cm.sum(dim=1).to(torch.float32)
    f1_c = (2 * tp) / (2 * tp + fp + fn + eps)
    mask = support > 0
    macro = float(f1_c[mask].mean().item()) if mask.any() else 0.0

    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    micro = float((2 * TP / (2 * TP + FP + FN + eps)).item())

    return micro, macro


def train_fewshot(
    z_target: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    model,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
    save_best_path: str,
    early_stop_metric: str = "macro",   # ✅ 新增： "macro" 或 "micro"
) -> FewShotResult:

    assert early_stop_metric in ("micro", "macro"), "early_stop_metric must be 'micro' or 'macro'"

    z_target = z_target.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    es = EarlyStopping(EarlyStopConfig(
        patience=patience,
        min_delta=0.0,
        mode="max",  # F1 越大越好
        save_best=True,
        save_path=save_best_path,
    ))

    # ✅ 同时记录 micro/macro
    best_val_micro = 0.0
    best_val_macro = 0.0
    test_at_best_micro = 0.0
    test_at_best_macro = 0.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(z_target)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(z_target)
            num_classes = int(logits.size(-1))

            if val_idx.numel() > 0:
                val_micro, val_macro = f1_micro_macro(logits[val_idx], y[val_idx], num_classes)
            else:
                val_micro, val_macro = 0.0, 0.0

            test_micro, test_macro = f1_micro_macro(logits[test_idx], y[test_idx], num_classes)

        # ✅ early stop 监控指标（任选一个）
        monitor = val_macro if early_stop_metric == "macro" else val_micro

        # ✅ best 也按 monitor 来决定，但保存当时的 micro+macro
        improved = (
            (monitor > (best_val_macro if early_stop_metric == "macro" else best_val_micro))
        )
        if improved:
            best_val_micro = val_micro
            best_val_macro = val_macro
            test_at_best_micro = test_micro
            test_at_best_macro = test_macro
            best_epoch = epoch

        stop = es.step(
            monitor, epoch, model,
            extra_state={
                "best_val_micro": best_val_micro,
                "best_val_macro": best_val_macro,
                "test_at_best_micro": test_at_best_micro,
                "test_at_best_macro": test_at_best_macro,
                "early_stop_metric": early_stop_metric,
            },
        )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f} | "
                f"test_f1(micro/macro)={test_micro:.4f}/{test_macro:.4f} | "
                f"monitor({early_stop_metric})={monitor:.4f}"
            )

        if stop:
            print(
                f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                f"best_val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
                f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f} | "
                f"monitor={early_stop_metric}"
            )
            break

    return FewShotResult(
        best_val_micro=best_val_micro,
        best_val_macro=best_val_macro,
        test_at_best_micro=test_at_best_micro,
        test_at_best_macro=test_at_best_macro,
        best_epoch=best_epoch,
        early_stop_metric=early_stop_metric,
    )

def load_split_file(splits_dir: str, dataset_name: str, shot: int, seed: int) -> Dict[str, torch.Tensor]:
    base = Path(splits_dir) / dataset_name / f"{shot}-shot"
    if not base.exists():
        raise FileNotFoundError(f"Split dir not found: {base}")

    cands = sorted(base.glob(f"*{seed}*.pt"))
    if len(cands) == 0:
        cands = sorted(base.glob("*.pt"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No split .pt files in: {base}")

    path = cands[0]
    split = torch.load(path, map_location="cpu")
    if isinstance(split, dict):
        return split
    if hasattr(split, "__dict__"):
        return split.__dict__
    raise ValueError(f"Unknown split file format: {path}")