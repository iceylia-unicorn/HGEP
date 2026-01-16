# src/gpbench/pretrain/loss.py
from __future__ import annotations # 兼容 Python 3.7+

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    z1, z2: [N, D]  (two augmented views) 两种不同增强视图的表示
    Implements the standard NT-Xent / InfoNCE loss used in GraphCL-style methods.
    """
    assert z1.shape == z2.shape
    n = z1.size(0)

    z1 = F.normalize(z1, dim=1) 
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]

    sim = (z @ z.t()) / tau  # [2N, 2N]
    # mask self-similarity
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # positives: i <-> i+N
    pos = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)  # [2N]
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -log_prob[torch.arange(2 * n, device=z.device), pos].mean()
    return loss
