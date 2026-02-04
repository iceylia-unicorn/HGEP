
# src/gpbench/downstream/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from gpbench.pretrain.model import HGMPPretrainModel, PretrainModelConfig


@dataclass
class DownstreamConfig:
    # prompt/head
    prompt_mode: str = "add"     # "add" or "mul"
    prompt_dropout: float = 0.1
    head_hidden: int = 128
    head_dropout: float = 0.3
    use_layernorm: bool = True


class TypePrompt(nn.Module):
    """
    最基础的 type-prompt（embedding 空间）：
    - 每个 node type 一个可训练向量 p_t in R^d
    - add:  z' = LN(z + p_t)
    - mul:  z' = LN(z * (1 + p_t))
    """
    def __init__(self, node_types, dim: int, mode: str = "add", dropout: float = 0.1, use_ln: bool = True):
        super().__init__()
        assert mode in ("add", "mul")
        self.mode = mode
        self.node_types = list(node_types)
        self.prompt = nn.ParameterDict({nt: nn.Parameter(torch.zeros(dim)) for nt in self.node_types})
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()

    def forward(self, z: torch.Tensor, ntype: str) -> torch.Tensor:
        p = self.prompt[ntype].unsqueeze(0)  # [1, d]
        if self.mode == "add":
            out = z + p
        else:
            out = z * (1.0 + p)
        out = self.drop(out)
        out = self.ln(out)
        return out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.3, use_ln: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden) if use_ln else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = F.relu(x)
        x = self.drop(x)
        return self.fc2(x)


def load_frozen_encoder(
    full_data: HeteroData,
    ckpt_path: str,
    backbone: str,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    device: torch.device,
) -> Tuple[HGMPPretrainModel, Dict]:
    """
    复用你预训练的 HGMPPretrainModel（但下游只用 adapter+backbone 输出 node embeddings）。
    """
    cfg = PretrainModelConfig(
        backbone=backbone,
        hidden_dim=hidden_dim,
        proj_dim=hidden_dim,          # 下游不用 proj_dim，这里随便设
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_node_id_emb_if_missing_x=True,
    )
    enc = HGMPPretrainModel(full_data, cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # 允许 key 不完全一致（比如你改过 readout）
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print("[load_state_dict] unexpected keys:", unexpected[:10], "...")
    if len(missing) > 0:
        print("[load_state_dict] missing keys:", missing[:10], "...")

    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, ckpt


@torch.no_grad()
def encode_all_nodes(enc: HGMPPretrainModel, data: HeteroData, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    冻结 encoder 下，整图一次性编码得到每个 node type 的 embedding。
    """
    enc.eval()
    data = data.to(device)
    x_dict = enc.adapter(data)
    x_dict = enc.backbone(x_dict, data.edge_index_dict)
    return x_dict
