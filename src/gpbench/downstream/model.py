
# src/gpbench/downstream/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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

class StructuralPromptMLP(nn.Module):
    """
    与边类型无关的结构 prompt：
    - 输入: 每个目标节点的结构特征 s_i (例如度、2-hop 近似)
    - 输出: 与 z 同维度的 prompt 向量 p_i
    - add: z' = LN(z + p_i)
    - mul: z' = LN(z * (1 + p_i))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        mode: str = "add",
        dropout: float = 0.1,
        use_ln: bool = True,
    ):
        super().__init__()
        assert mode in ("add", "mul")
        self.mode = mode
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()

    def forward(self, z: torch.Tensor, struct_feat: torch.Tensor) -> torch.Tensor:
        p = self.mlp(struct_feat)
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


@torch.no_grad()
def build_target_structural_features(data: HeteroData, target_ntype: str) -> torch.Tensor:
    """
    构建与边类型无关的目标节点结构特征（全图一次性预计算）。

    返回 shape: [N_target, 4]
    特征:
      0) log(1 + in_degree_total)
      1) log(1 + out_degree_total)
      2) log(1 + degree_total)
      3) log(1 + two_hop_back_to_target)
    """
    device = next(iter(data.edge_index_dict.values())).device
    n_target = int(data[target_ntype].num_nodes)

    in_deg = torch.zeros(n_target, device=device)
    out_deg = torch.zeros(n_target, device=device)

    # 与边类型无关：直接跨所有 relation 汇总入/出度
    for (src_t, _, dst_t), edge_index in data.edge_index_dict.items():
        src, dst = edge_index
        if dst_t == target_ntype:
            in_deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        if src_t == target_ntype:
            out_deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))

    deg = in_deg + out_deg

    # 2-hop 回到 target 的近似计数:
    # target --r1--> mid --r2--> target
    two_hop = torch.zeros(n_target, device=device)
    etypes = list(data.edge_types)
    for e1 in etypes:
        s1, _, d1 = e1
        if s1 != target_ntype:
            continue
        src1, dst1 = data[e1].edge_index

        for e2 in etypes:
            s2, _, d2 = e2
            if s2 != d1 or d2 != target_ntype:
                continue

            src2, _ = data[e2].edge_index
            # 统计每个中间节点 m 有多少条到 target 的边
            mid_size = int(data[d1].num_nodes)
            mid_to_target_cnt = torch.zeros(mid_size, device=device)
            mid_to_target_cnt.index_add_(0, src2, torch.ones_like(src2, dtype=torch.float32))

            # 把每条 target->mid 的计数累积回 target 起点
            contrib = mid_to_target_cnt[dst1]
            two_hop.index_add_(0, src1, contrib)

    feat = torch.stack(
        [
            torch.log1p(in_deg),
            torch.log1p(out_deg),
            torch.log1p(deg),
            torch.log1p(two_hop),
        ],
        dim=-1,
    )
    return feat