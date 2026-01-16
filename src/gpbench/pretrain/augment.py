# src/gpbench/pretrain/augment.py
from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData

# 计算增强时各类型节点/边的权重，权重与数量的平方成正比
def _type_squared_weights(counts: Dict[Tuple, int]) -> Dict[Tuple, float]: #键：类型元组，值：数量
    # weights(k) = count(k)^2 / sum_j count(j)^2
    denom = 0.0  # 数量平方和
    for c in counts.values():
        denom += float(c * c)
    if denom <= 0:
        return {k: 0.0 for k in counts}
    return {k: float(c * c) / denom for k, c in counts.items()} #{"paper": 0.5, "author": 0.5} 示例


def hetero_node_masking(data: HeteroData, r: float, mask_value: float = 0.0) -> HeteroData:
    """
    Node masking augmentation.
    For each node type t:
      num_to_mask(t) = r * a(t) * |V|,  a(t) ∝ count(t)^2
    We implement it on the *current* graph/subgraph (so |V| is subgraph node count).
    """
    out = copy.deepcopy(data) #深拷贝数据对象

    # counts per node type
    node_counts = {ntype: int(out[ntype].num_nodes) for ntype in out.node_types} #构造每一个节点类型的数量字典


    weights = _type_squared_weights(node_counts) #计算每个节点类型的权重

    total_nodes = sum(node_counts.values())
    if total_nodes == 0 or r <= 0:
        return out

    for ntype in out.node_types:
        x = getattr(out[ntype], "x", None) #如果没有节点特征，则返回 None
        if x is None:
            continue  # skip if no features; still OK for now

        num = int(round(r * weights[ntype] * total_nodes)) #round 四舍五入取整
        if num <= 0:
            continue

        num_nodes = out[ntype].num_nodes
        num = min(num, num_nodes) # 避免超过节点总数

        perm = torch.randperm(num_nodes, device=x.device)[:num] # 随机选择要mask的节点索引
        x = x.clone()
        x[perm] = mask_value
        out[ntype].x = x

    return out


def hetero_edge_permutation(data: HeteroData, r: float) -> HeteroData:
    """
    Edge permutation augmentation.
    For each edge type et = (src, rel, dst):
      num_to_permute(et) = r * b(et) * |E|, b(et) ∝ count(et)^2
    We permute by: randomly drop k edges and add k random edges with same (src_type, dst_type).
    """
    out = copy.deepcopy(data)

    edge_counts = {} # 计算每种边类型的数量
    for etype in out.edge_types: # etype是tuple类型的，"author","writes","paper"
        ei = out[etype].edge_index
        edge_counts[etype] = int(ei.size(1))

    weights = _type_squared_weights(edge_counts) #计算每个边类型的权重
    total_edges = sum(edge_counts.values())
    if total_edges == 0 or r <= 0:
        return out

    for etype in out.edge_types:
        src_type, _, dst_type = etype #解包边类型元组
        ei = out[etype].edge_index
        e = ei.size(1) #当前类型边数量
        if e == 0:
            continue

        k = int(round(r * weights[etype] * total_edges))
        if k <= 0:
            continue
        k = min(k, e)

        # drop k edges
        keep = torch.ones(e, dtype=torch.bool, device=ei.device) #生成全为1的mask 掩码和索引必须在同一个设备
        drop_idx = torch.randperm(e, device=ei.device)[:k] #随机挑选k个边索引
        keep[drop_idx] = False #将这些边标记为不保留
        kept = ei[:, keep] # 保留的边索引,按照keep的mask删除掉k条边

        # add k random edges with same endpoint types 添加k条同样类型的边
        num_src = out[src_type].num_nodes
        num_dst = out[dst_type].num_nodes
        new_src = torch.randint(0, num_src, (k,), device=ei.device)
        new_dst = torch.randint(0, num_dst, (k,), device=ei.device)
        added = torch.stack([new_src, new_dst], dim=0)

        out[etype].edge_index = torch.cat([kept, added], dim=1)

    return out
