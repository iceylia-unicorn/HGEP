# src/gpbench/data/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple #tuple是不可变列表，只读的，与list相对，可以作为字典的键

import torch

from torch_geometric.datasets import HGBDataset

# 辅助函数：将布尔掩码转换为索引张量 [0,1,0,0,1] => [1,4,6,...]
def _mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).view(-1).long()

# 辅助函数：从数据存储对象中获取name对应的掩码索引，防止不同数据集掩码属性名称不同
def _get_mask_idx(store, *names: str) -> torch.Tensor:
    for n in names:
        if hasattr(store, n) and getattr(store, n) is not None:
            return _mask_to_index(getattr(store, n))
    # 没有就返回空
    return torch.empty(0, dtype=torch.long)


@dataclass #工业语法糖， 自动生成 __init__、__repr__ 等方法
class NodeTaskData:
    dataset_name: str
    data: object              # HeteroData
    target_ntype: str
    y: torch.Tensor           # labels on target type local indexing
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor


def load_hgb_node_task(root: str | Path, name: str) -> NodeTaskData: #str | Path 兼容字符串和 Path 对象 python 3.10+
    """
    name in {"ACM", "DBLP", "IMDB"} etc.
    """
    root = Path(root)
    dataset = HGBDataset(root=str(root), name=name) # 加载数据集
    data = dataset[0] #其实HGB大多数都只有一张图，dataset[0]即为图数据


    # HGB 的 target node type 在不同数据集不一样；ACM 通常是 "paper"，因为异质图一般只预测一类节点
    # 为了代码扩展性：用 dataset 名字映射，而不是写死
    target_map = {
        "ACM": "paper",
        "DBLP": "author",
        "IMDB": "movie",
    }
    target_ntype = target_map[name]

    y = data[target_ntype].y 

    node_store = data[target_ntype]
    train_idx = _get_mask_idx(node_store, "train_mask")
    val_idx   = _get_mask_idx(node_store, "val_mask", "valid_mask", "valid", "val")
    test_idx  = _get_mask_idx(node_store, "test_mask", "test")


    return NodeTaskData(
        dataset_name=f"hgb-{name.lower()}",
        data=data,
        target_ntype=target_ntype,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
