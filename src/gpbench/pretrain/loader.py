# src/gpbench/pretrain/loader.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union # Union[int,str] 等价于 3.10+ int | str

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


def _fanout_to_list(fanout: Union[int, Sequence[int]], num_hops: int) -> List[int]: #邻居采样策略，比如 fanout=10 表示每跳采样10个邻居节点 fanout=[10,5]表示第一跳采样10个，第二跳采样5个
    if isinstance(fanout, int):
        return [fanout] * num_hops
    fanout = list(fanout)
    if len(fanout) != num_hops:
        raise ValueError(f"fanout length must equal num_hops: got {len(fanout)} vs {num_hops}")
    return fanout


def build_hetero_neighbor_loader(
    data: HeteroData,
    input_ntype: str,
    batch_size: int,
    num_hops: int,
    fanout: Union[int, Sequence[int]] = (25, 15),
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> NeighborLoader:
    """
    Build a NeighborLoader for hetero graphs.
    Each batch corresponds to a set of seed nodes of `input_ntype` and their τ-hop sampled neighborhoods.

    Note:
      - Requires torch-sparse/pyg-lib for best performance.
      - This is *sampled* τ-hop ego subgraph batching (mainstream for large graphs).
    """
    fanout_list = _fanout_to_list(fanout, num_hops)

    num_neighbors: Dict[Tuple[str, str, str], List[int]] = {etype: fanout_list for etype in data.edge_types}
    seed_nodes = torch.arange(data[input_ntype].num_nodes, dtype=torch.long)
    """
    NeighborLoader是PyG中用于图神经网络训练的采样器，适用于大规模图数据。
    如果你有 700 个 paper 节点，设置 batch_size=100，那么每个 batch 会随机选择 100 个 paper 节点作为种子节点，
    并采样它们的邻居节点构建子图进行训练。
    假设采样 2 层，每层 5 个邻居。这 100 个种子会“带出”成百上千个邻居节点（包括 paper、author 等其他类型节点），
    形成一个包含多种类型节点和边的异质子图。这样就
    可以在大图上进行 mini-batch 训练，而不需要一次性加载整个图到内存中。
    所有的节点被合并在一个长张量里，x_dict 里包含每种类型节点的特征张量，可以通过节点类型索引访问。
    用一个不联通的邻接矩阵存储所有子图
    """
    loader = NeighborLoader( #NeighborLoader 内部会自动处理异质图的采样
        data,
        input_nodes=(input_ntype, seed_nodes), # input_nodes 可以指定节点类型和节点索引
        num_neighbors=num_neighbors, # 每种边类型的采样数量
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0, 
    )
    return loader
