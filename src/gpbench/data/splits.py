# src/gpbench/data/splits.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class FewShotSplit:
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor

    def to(self, device: torch.device) -> "FewShotSplit":
        return FewShotSplit(
            train_idx=self.train_idx.to(device),
            val_idx=self.val_idx.to(device),
            test_idx=self.test_idx.to(device),
        )


def _sample_k_per_class(
    y: torch.Tensor,
    idx_pool: torch.Tensor,
    k: int,
    seed: int,
) -> torch.Tensor:
    """
    y: [num_nodes_target]  (labels for target node type)
    idx_pool: indices on the SAME indexing space as y (target-type local indices)
    """
    g = torch.Generator()
    g.manual_seed(seed)

    classes = torch.unique(y[idx_pool])
    sampled = []

    for c in classes.tolist():
        cand = idx_pool[(y[idx_pool] == c)]
        if cand.numel() == 0:
            continue
        if cand.numel() >= k:
            perm = cand[torch.randperm(cand.numel(), generator=g)[:k]]
        else:
            #如果有类别样本数不足 k，则全部采样
            perm = cand
        sampled.append(perm) #shape为[c_count, <=k]

    if len(sampled) == 0:
        raise RuntimeError("No samples were drawn. Check labels / idx_pool.")
    return torch.cat(sampled, dim=0)


def make_fewshot_split(
    *,
    y: torch.Tensor,
    train_idx_base: torch.Tensor,
    val_idx_base: torch.Tensor,
    test_idx_base: torch.Tensor,
    shot: int,
    seed: int,
    val_shot_per_class: Optional[int] = None,
) -> FewShotSplit:
    """
    官方 split 上做 few-shot：
    - 从 train_idx_base 每类采 shot
    - val 可选：也做 few-shot（更严格）或直接用全量 val（更常见）
    """
    train_idx = _sample_k_per_class(y, train_idx_base, shot, seed)

    # 1) 如果有官方 val，就用官方 val（或对它再做 few-shot）
    if val_idx_base is not None and val_idx_base.numel() > 0:
        if val_shot_per_class is None:
            val_idx = val_idx_base
        else:
            val_idx = _sample_k_per_class(y, val_idx_base, val_shot_per_class, seed + 10_000)
        return FewShotSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx_base)
        # 2) 没有官方 val：从 train 剩余里切 val（主流做法）

    #    默认每类取 max(5, min(20, shot))，你也可以外部传 val_shot_per_class 控制
    if val_shot_per_class is None:
        val_shot_per_class = max(5, min(20, shot))
    n = y.size(0)
    used = torch.zeros(n, dtype=torch.bool) # 初始全为0的布尔掩码
    used[train_idx] = True #将已经分配给 train_idx（训练集）的索引位置设为 True
    remaining_train = train_idx_base[~used[train_idx_base]] #总体逻辑就是挑选出train_idx_base中未被used掩码标记为True的索引,也就是官方训练集中剩余的索引

    val_idx = _sample_k_per_class(y, remaining_train, val_shot_per_class, seed + 20_000)

    return FewShotSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx_base)


class SplitManager:
    """
    负责 split 的落盘/读取，确保每次实验可复现。
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, dataset: str, shot: int, seed: int) -> Path:
        return self.root / dataset / f"{shot}-shot" / f"seed{seed}.pt"

    def save(self, dataset: str, shot: int, seed: int, split: FewShotSplit) -> None:
        p = self.path(dataset, shot, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_idx": split.train_idx.cpu(),
                "val_idx": split.val_idx.cpu(),
                "test_idx": split.test_idx.cpu(),
            },
            p,
        )

    def load(self, dataset: str, shot: int, seed: int) -> FewShotSplit:
        p = self.path(dataset, shot, seed)
        obj = torch.load(p, map_location="cpu")
        return FewShotSplit(
            train_idx=obj["train_idx"].long(),
            val_idx=obj["val_idx"].long(),
            test_idx=obj["test_idx"].long(),
        )
