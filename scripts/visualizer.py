from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv

import dgl
import matplotlib
matplotlib.use("Agg")
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

CanonicalEType = tuple[str, str, str]


@dataclass
class ForwardCapture:
    """
    保存一次子图前向分析时的中间状态。
    """

    layer_embeddings: dict[str, torch.Tensor]
    attention_dict: dict[CanonicalEType, torch.Tensor]
    node_layer_embeddings: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    pe_norm_dict: dict[CanonicalEType, torch.Tensor] = field(default_factory=dict)
    prompt_summary: dict[str, Any] = field(default_factory=dict)
    score_mode: str = "proxy"


def _clone_x_dict(x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {ntype: value.detach().clone() for ntype, value in x_dict.items()}


def _clone_x_dict_cpu(x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {ntype: value.detach().cpu().clone() for ntype, value in x_dict.items()}


def _to_cpu_tensor(value: torch.Tensor | np.ndarray | list[float]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _extract_target_local_id(inverse_indices, targetnode: str) -> int:
    if isinstance(inverse_indices, dict):
        if targetnode not in inverse_indices:
            raise KeyError(f"targetnode={targetnode} not found in inverse_indices keys={list(inverse_indices.keys())}")
        value = inverse_indices[targetnode]
    else:
        value = inverse_indices

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.view(-1)
    if value.numel() == 0:
        raise ValueError("inverse_indices is empty, cannot locate target node.")
    return int(value[0].item())


def _split_h_by_keys(
    h: torch.Tensor,
    keys: list[str],
    sizes: list[int],
) -> dict[str, torch.Tensor]:
    out = {}
    start = 0
    for key, size in zip(keys, sizes):
        out[key] = h[start:start + size]
        start += size
    return out


def _node_offsets(graph: dgl.DGLHeteroGraph) -> dict[str, int]:
    offsets: dict[str, int] = {}
    start = 0
    for ntype in graph.ntypes:
        offsets[ntype] = start
        start += int(graph.num_nodes(ntype))
    return offsets


def _build_edge_index_dict(graph: dgl.DGLHeteroGraph) -> dict[CanonicalEType, torch.Tensor]:
    edge_index_dict: dict[CanonicalEType, torch.Tensor] = {}
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype)
        edge_index_dict[etype] = torch.stack((src, dst), dim=0)
    return edge_index_dict


def _build_edge_feature_dict(graph: dgl.DGLHeteroGraph, feature_name: str) -> dict[CanonicalEType, torch.Tensor] | None:
    edge_feature_dict: dict[CanonicalEType, torch.Tensor] = {}
    for etype in graph.canonical_etypes:
        if feature_name not in graph.edges[etype].data:
            continue
        edge_feature_dict[etype] = graph.edges[etype].data[feature_name].float()
    return edge_feature_dict or None


def _normalize_per_dst(
    scores: torch.Tensor,
    dst: torch.Tensor,
    num_dst_nodes: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    denom = torch.zeros((num_dst_nodes,), device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, dst, scores)
    return scores / denom[dst].clamp_min(eps)


def _mean_attention_dict(layer_scores: list[dict[CanonicalEType, torch.Tensor]]) -> dict[CanonicalEType, torch.Tensor]:
    if not layer_scores:
        return {}

    merged: dict[CanonicalEType, list[torch.Tensor]] = {}
    for payload in layer_scores:
        for etype, score in payload.items():
            merged.setdefault(etype, []).append(score.detach())

    out: dict[CanonicalEType, torch.Tensor] = {}
    for etype, values in merged.items():
        out[etype] = torch.stack(values, dim=0).mean(dim=0)
    return out


def _pe_norm_from_edge_features(
    edge_feature_dict: dict[CanonicalEType, torch.Tensor] | None,
) -> dict[CanonicalEType, torch.Tensor]:
    if edge_feature_dict is None:
        return {}
    return {
        etype: feat.norm(p=2, dim=-1).detach()
        for etype, feat in edge_feature_dict.items()
    }


def _estimate_similarity_attention(
    x_dict: dict[str, torch.Tensor],
    edge_index_dict: dict[CanonicalEType, torch.Tensor],
) -> dict[CanonicalEType, torch.Tensor]:
    """
    HGT/通用路径下的代理边权重。
    这里不伪装成“真实注意力”，而是用源/目标表示相似度构造可解释代理。
    """

    out: dict[CanonicalEType, torch.Tensor] = {}
    for (src_t, rel_t, dst_t), edge_index in edge_index_dict.items():
        src, dst = edge_index
        if src.numel() == 0:
            out[(src_t, rel_t, dst_t)] = torch.zeros_like(src, dtype=torch.float32)
            continue

        src_h = x_dict[src_t][src]
        dst_h = x_dict[dst_t][dst]
        score = F.cosine_similarity(src_h, dst_h, dim=-1)
        score = torch.nan_to_num((score + 1.0) * 0.5, nan=0.0, posinf=1.0, neginf=0.0)
        score = _normalize_per_dst(score, dst, x_dict[dst_t].size(0))
        out[(src_t, rel_t, dst_t)] = score.detach()
    return out


def _estimate_gcn_attention(
    graph: dgl.DGLHeteroGraph,
    edge_index_dict: dict[CanonicalEType, torch.Tensor],
) -> dict[CanonicalEType, torch.Tensor]:
    """
    GCN 路径下可精确写出的结构聚合系数代理：
    w(u,v)=1/sqrt(out_deg(u)*in_deg(v))
    """

    offsets = _node_offsets(graph)
    homo_g = dgl.to_homogeneous(graph)
    out_deg = homo_g.out_degrees().float().clamp_min(1.0).to(homo_g.device)
    in_deg = homo_g.in_degrees().float().clamp_min(1.0).to(homo_g.device)

    out: dict[CanonicalEType, torch.Tensor] = {}
    for etype, edge_index in edge_index_dict.items():
        src_t, _, dst_t = etype
        src, dst = edge_index
        if src.numel() == 0:
            out[etype] = torch.zeros_like(src, dtype=torch.float32)
            continue

        src_global = src + offsets[src_t]
        dst_global = dst + offsets[dst_t]
        score = 1.0 / torch.sqrt(out_deg[src_global] * in_deg[dst_global])
        score = _normalize_per_dst(score, dst, graph.num_nodes(dst_t))
        out[etype] = score.detach()
    return out


def _estimate_relation_prompt_strength(
    relation_prompt,
    edge_index_dict: dict[CanonicalEType, torch.Tensor],
    edge_feature_dict: dict[CanonicalEType, torch.Tensor] | None,
) -> dict[CanonicalEType, torch.Tensor]:
    """
    提取边提示强度。
    对 peprompt 来说，对应 MLP(edge_feat) 生成的边 prompt 向量 L2 范数；
    对 typepair 风格关系提示，则融合类型 prompt 与边 prompt 后取范数。
    """

    if relation_prompt is None:
        return {}

    out: dict[CanonicalEType, torch.Tensor] = {}
    uses_edge_features = bool(getattr(relation_prompt, "uses_edge_features", False))

    for etype, edge_index in edge_index_dict.items():
        src_t, _, dst_t = etype
        src, _ = edge_index
        num_edges = int(src.numel())
        if num_edges == 0:
            out[etype] = torch.zeros((0,), dtype=torch.float32, device=src.device)
            continue

        if hasattr(relation_prompt, "make_pair_key") and hasattr(relation_prompt, "prompt"):
            key = relation_prompt.make_pair_key(src_t, dst_t)
            if key not in relation_prompt.prompt:
                out[etype] = torch.ones((num_edges,), dtype=torch.float32, device=src.device)
                continue

            type_prompt = relation_prompt.prompt[key]
            edge_feat = edge_feature_dict.get(etype) if edge_feature_dict is not None else None
            fused = relation_prompt._fuse_edge_prompt(type_prompt, edge_feat)
            score = fused.norm(p=2, dim=-1)
            out[etype] = score.detach()
            continue

        if uses_edge_features and edge_feature_dict is not None and etype in edge_feature_dict:
            edge_feat = edge_feature_dict[etype]
            score = relation_prompt.edge_prompt_mlp(edge_feat).norm(p=2, dim=-1)
            out[etype] = score.detach()
            continue

        out[etype] = torch.ones((num_edges,), dtype=torch.float32, device=src.device)

    return out


def _compute_prompt_delta_summary(
    before_x_dict: dict[str, torch.Tensor] | None,
    after_x_dict: dict[str, torch.Tensor] | None,
) -> dict[str, dict[str, float]]:
    if before_x_dict is None or after_x_dict is None:
        return {}

    out: dict[str, dict[str, float]] = {}
    for ntype, before in before_x_dict.items():
        if ntype not in after_x_dict:
            continue
        after = after_x_dict[ntype]
        delta = after - before
        delta_norm = delta.norm(p=2, dim=-1)
        base_norm = before.norm(p=2, dim=-1).clamp_min(1e-12)
        out[ntype] = {
            "mean_delta_l2": float(delta_norm.mean().item()),
            "max_delta_l2": float(delta_norm.max().item()),
            "mean_relative_delta": float((delta_norm / base_norm).mean().item()),
        }
    return out


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=1e-12)


def _summarize_node_layer_drift(
    node_layer_embeddings: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, float | int | str]]:
    if not node_layer_embeddings:
        return []

    layer_names = [name for name in ("Layer0", "Layer1", "Layer2") if name in node_layer_embeddings]
    if len(layer_names) < 2:
        return []

    base = node_layer_embeddings[layer_names[0]]
    rows: list[dict[str, float | int | str]] = []
    for ntype, x0 in base.items():
        x1 = node_layer_embeddings.get("Layer1", {}).get(ntype)
        x2 = node_layer_embeddings.get("Layer2", {}).get(ntype)
        num_nodes = x0.size(0)
        for nid in range(num_nodes):
            row: dict[str, float | int | str] = {
                "ntype": ntype,
                "node_id": int(nid),
                "layer0_norm": float(x0[nid].norm(p=2).item()),
            }
            if x1 is not None:
                row["layer1_norm"] = float(x1[nid].norm(p=2).item())
                row["delta01_l2"] = float((x1[nid] - x0[nid]).norm(p=2).item())
                row["cos01"] = float(_safe_cosine(x0[nid].unsqueeze(0), x1[nid].unsqueeze(0))[0].item())
            if x2 is not None:
                row["layer2_norm"] = float(x2[nid].norm(p=2).item())
                row["delta02_l2"] = float((x2[nid] - x0[nid]).norm(p=2).item())
                row["cos02"] = float(_safe_cosine(x0[nid].unsqueeze(0), x2[nid].unsqueeze(0))[0].item())
            if x1 is not None and x2 is not None:
                row["delta12_l2"] = float((x2[nid] - x1[nid]).norm(p=2).item())
                row["cos12"] = float(_safe_cosine(x1[nid].unsqueeze(0), x2[nid].unsqueeze(0))[0].item())
            rows.append(row)
    rows.sort(key=lambda item: float(item.get("delta02_l2", item.get("delta01_l2", 0.0))), reverse=True)
    return rows


def _write_dict_rows_csv(path: str | Path, rows: list[dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LegacyModelVisualizerAdapter:
    """
    适配 HGMP / HGMP Prompt / PEPrompt 三类模型的中间状态提取器。

    这里优先采用“显式适配器”而不是通用 forward hook：
    - 旧代码里 HGT / GCN / GAT 的 forward 签名并不一致；
    - peprompt 和 hgmp_prompt 还分别存在边提示注入和前置 node prompt；
    - 显式重放前向路径，更容易保证层级特征的语义稳定。
    """

    def __init__(
        self,
        method: str,
        hgnn,
        targetnode: str,
        *,
        prompt_module=None,
        edge_feature_name: str = "peprompt_edge_feat",
    ):
        self.method = method
        self.hgnn = hgnn
        self.targetnode = targetnode
        self.prompt_module = prompt_module
        self.edge_feature_name = edge_feature_name

    def capture(
        self,
        graph: dgl.DGLHeteroGraph,
        inverse_indices,
        *,
        prompt_before_x_dict: dict[str, torch.Tensor] | None = None,
    ) -> ForwardCapture:
        model_device = next(self.hgnn.parameters()).device
        graph_cpu = self._ensure_graph_on_cpu(graph)
        graph_for_model = graph if self._same_device(graph, model_device) else graph.to(model_device)
        target_local_id = _extract_target_local_id(inverse_indices, self.targetnode)
        edge_index_dict = {
            etype: edge_index.to(model_device)
            for etype, edge_index in _build_edge_index_dict(graph_cpu).items()
        }
        edge_feature_dict_cpu = _build_edge_feature_dict(graph_cpu, self.edge_feature_name)
        edge_feature_dict = None
        if edge_feature_dict_cpu is not None:
            edge_feature_dict = {
                etype: feat.to(model_device)
                for etype, feat in edge_feature_dict_cpu.items()
            }

        prompt_summary = {}
        if prompt_before_x_dict is not None:
            prompt_summary["node_prompt_delta"] = _compute_prompt_delta_summary(
                prompt_before_x_dict,
                graph_for_model.ndata["x"],
            )

        if self.hgnn.hgnn_type == "HGT":
            capture = self._capture_hgt(
                graph_for_model,
                target_local_id,
                edge_index_dict=edge_index_dict,
                edge_feature_dict=edge_feature_dict,
                prompt_summary=prompt_summary,
            )
        elif self.hgnn.hgnn_type == "GCN":
            capture = self._capture_gcn(
                graph_for_model,
                target_local_id,
                edge_index_dict=edge_index_dict,
                edge_feature_dict=edge_feature_dict,
                prompt_summary=prompt_summary,
            )
        elif self.hgnn.hgnn_type == "GAT":
            capture = self._capture_gat(
                graph_for_model,
                target_local_id,
                edge_index_dict=edge_index_dict,
                prompt_summary=prompt_summary,
            )
        else:
            raise NotImplementedError(f"Unsupported hgnn_type for visualization: {self.hgnn.hgnn_type}")

        if edge_feature_dict_cpu is not None and not capture.pe_norm_dict:
            capture.pe_norm_dict = _pe_norm_from_edge_features(edge_feature_dict_cpu)
        return capture

    def _resolve_graph_conv(self):
        """
        兼容两类 legacy GCN/HGT 对象：
        1. HGNN 包装器：真正卷积模块挂在 `.GraphConv`
        2. 直接实例：如 `GCL_GCN` / `GAT` 本体，本身就是卷积模块
        """
        return getattr(self.hgnn, "GraphConv", self.hgnn)

    @staticmethod
    def _ensure_graph_on_cpu(graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        graph_device = getattr(graph, "device", None)
        if isinstance(graph_device, torch.device) and graph_device.type == "cpu":
            return graph
        return graph.to("cpu")

    @staticmethod
    def _same_device(graph: dgl.DGLHeteroGraph, device: torch.device) -> bool:
        graph_device = getattr(graph, "device", None)
        return isinstance(graph_device, torch.device) and graph_device == device

    def _capture_hgt(
        self,
        graph: dgl.DGLHeteroGraph,
        target_local_id: int,
        *,
        edge_index_dict: dict[CanonicalEType, torch.Tensor],
        edge_feature_dict: dict[CanonicalEType, torch.Tensor] | None,
        prompt_summary: dict[str, Any],
    ) -> ForwardCapture:
        graph_conv = self._resolve_graph_conv()
        relation_prompt = getattr(graph_conv, "relation_prompt", None)

        x_dict = {
            node_type: graph_conv.lin_dict[node_type](x).relu_()
            for node_type, x in graph.ndata["x"].items()
        }
        node_layer_embeddings = {
            "Layer0": _clone_x_dict_cpu(x_dict),
        }
        layer_embeddings = {
            "Layer0": x_dict[self.targetnode][target_local_id].detach().cpu(),
        }

        attn_layers: list[dict[CanonicalEType, torch.Tensor]] = []
        prompt_layers: list[dict[CanonicalEType, torch.Tensor]] = []

        prev_dict = _clone_x_dict(x_dict)
        for layer_idx, conv in enumerate(graph_conv.convs, start=1):
            x_dict = conv(x_dict, edge_index_dict)
            attn_layers.append(_estimate_similarity_attention(prev_dict, edge_index_dict))

            if relation_prompt is not None:
                prompt_layers.append(
                    _estimate_relation_prompt_strength(
                        relation_prompt,
                        edge_index_dict,
                        edge_feature_dict,
                    )
                )
                x_dict = relation_prompt(x_dict, edge_index_dict, edge_feature_dict)

            if layer_idx == 1:
                layer_embeddings["Layer1"] = x_dict[self.targetnode][target_local_id].detach().cpu()
                node_layer_embeddings["Layer1"] = _clone_x_dict_cpu(x_dict)
            prev_dict = _clone_x_dict(x_dict)

        x_dict = {
            node_type: graph_conv.lin(x)
            for node_type, x in x_dict.items()
        }
        layer_embeddings["Layer2"] = x_dict[self.targetnode][target_local_id].detach().cpu()
        node_layer_embeddings["Layer2"] = _clone_x_dict_cpu(x_dict)

        if prompt_layers:
            attention_dict = _mean_attention_dict(prompt_layers)
            score_mode = "prompt_strength"
            prompt_summary["edge_prompt_strength_mean"] = {
                f"{src}-{rel}-{dst}": float(value.mean().item())
                for (src, rel, dst), value in attention_dict.items()
            }
        else:
            attention_dict = _mean_attention_dict(attn_layers)
            score_mode = "proxy_similarity"

        return ForwardCapture(
            layer_embeddings=layer_embeddings,
            attention_dict=attention_dict,
            node_layer_embeddings=node_layer_embeddings,
            pe_norm_dict=_pe_norm_from_edge_features(edge_feature_dict),
            prompt_summary=prompt_summary,
            score_mode=score_mode,
        )

    def _capture_gcn(
        self,
        graph: dgl.DGLHeteroGraph,
        target_local_id: int,
        *,
        edge_index_dict: dict[CanonicalEType, torch.Tensor],
        edge_feature_dict: dict[CanonicalEType, torch.Tensor] | None,
        prompt_summary: dict[str, Any],
    ) -> ForwardCapture:
        graph_conv = self._resolve_graph_conv()
        relation_prompt = getattr(graph_conv, "relation_prompt", None)

        keys = list(graph.ndata["x"].keys())
        sizes = [graph.ndata["x"][key].shape[0] for key in keys]
        feats_emd = [graph.ndata["x"][key] for key in keys]
        h_list = [fc(feature) for fc, feature in zip(graph_conv.fc_list, feats_emd)]
        h = torch.cat(h_list, dim=0)

        x_dict = _split_h_by_keys(h, keys, sizes)
        node_layer_embeddings = {
            "Layer0": _clone_x_dict_cpu(x_dict),
        }
        layer_embeddings = {
            "Layer0": x_dict[self.targetnode][target_local_id].detach().cpu(),
        }

        attn_layers: list[dict[CanonicalEType, torch.Tensor]] = []
        prompt_layers: list[dict[CanonicalEType, torch.Tensor]] = []

        homo_g = dgl.to_homogeneous(graph)
        homo_g = dgl.remove_self_loop(homo_g)
        homo_g = dgl.add_self_loop(homo_g)

        for layer_idx, layer in enumerate(graph_conv.layers, start=1):
            h = graph_conv.dropout(h)
            h = layer(homo_g, h)
            x_dict = _split_h_by_keys(h, keys, sizes)
            attn_layers.append(_estimate_gcn_attention(graph, edge_index_dict))

            if relation_prompt is not None:
                prompt_layers.append(
                    _estimate_relation_prompt_strength(
                        relation_prompt,
                        edge_index_dict,
                        edge_feature_dict,
                    )
                )
                x_dict = relation_prompt(x_dict, edge_index_dict, edge_feature_dict)
                h = torch.cat([x_dict[key] for key in keys], dim=0)

            if layer_idx == 1:
                layer_embeddings["Layer1"] = x_dict[self.targetnode][target_local_id].detach().cpu()
                node_layer_embeddings["Layer1"] = _clone_x_dict_cpu(x_dict)

        layer_embeddings["Layer2"] = x_dict[self.targetnode][target_local_id].detach().cpu()
        node_layer_embeddings["Layer2"] = _clone_x_dict_cpu(x_dict)

        if prompt_layers:
            attention_dict = _mean_attention_dict(prompt_layers)
            score_mode = "prompt_strength"
            prompt_summary["edge_prompt_strength_mean"] = {
                f"{src}-{rel}-{dst}": float(value.mean().item())
                for (src, rel, dst), value in attention_dict.items()
            }
        else:
            attention_dict = _mean_attention_dict(attn_layers)
            score_mode = "gcn_norm_weight"

        return ForwardCapture(
            layer_embeddings=layer_embeddings,
            attention_dict=attention_dict,
            node_layer_embeddings=node_layer_embeddings,
            pe_norm_dict=_pe_norm_from_edge_features(edge_feature_dict),
            prompt_summary=prompt_summary,
            score_mode=score_mode,
        )

    def _capture_gat(
        self,
        graph: dgl.DGLHeteroGraph,
        target_local_id: int,
        *,
        edge_index_dict: dict[CanonicalEType, torch.Tensor],
        prompt_summary: dict[str, Any],
    ) -> ForwardCapture:
        graph_conv = self.hgnn
        keys = list(graph.ndata["x"].keys())
        sizes = [graph.ndata["x"][key].shape[0] for key in keys]
        feats_emd = [graph.ndata["x"][key] for key in keys]
        h_list = [fc(feature) for fc, feature in zip(graph_conv.fc_list, feats_emd)]
        h = torch.cat(h_list, dim=0)
        x_dict = _split_h_by_keys(h, keys, sizes)
        node_layer_embeddings = {
            "Layer0": _clone_x_dict_cpu(x_dict),
        }
        layer_embeddings = {
            "Layer0": x_dict[self.targetnode][target_local_id].detach().cpu(),
        }

        homo_g = dgl.to_homogeneous(graph)
        attn_layers: list[dict[CanonicalEType, torch.Tensor]] = []

        for layer_idx in range(graph_conv.num_layers):
            layer = graph_conv.gat_layers[layer_idx]
            try:
                h_next, attn = layer(homo_g, h, get_attention=True)
            except TypeError:
                h_next = layer(homo_g, h)
                attn = None
            h = h_next.flatten(1)
            x_dict = _split_h_by_keys(h, keys, sizes)
            if layer_idx == 0:
                layer_embeddings["Layer1"] = x_dict[self.targetnode][target_local_id].detach().cpu()
                node_layer_embeddings["Layer1"] = _clone_x_dict_cpu(x_dict)
            if attn is not None:
                attn_layers.append(self._map_homo_attention_to_hetero(graph, attn))

        out_layer = graph_conv.gat_layers[-1]
        try:
            logits, attn = out_layer(homo_g, h, get_attention=True)
        except TypeError:
            logits = out_layer(homo_g, h)
            attn = None
        logits = logits.mean(1)
        x_dict = _split_h_by_keys(logits, keys, sizes)
        layer_embeddings["Layer2"] = x_dict[self.targetnode][target_local_id].detach().cpu()
        node_layer_embeddings["Layer2"] = _clone_x_dict_cpu(x_dict)

        if attn is not None:
            attn_layers.append(self._map_homo_attention_to_hetero(graph, attn))

        attention_dict = _mean_attention_dict(attn_layers) if attn_layers else _estimate_similarity_attention(x_dict, edge_index_dict)
        score_mode = "gat_attention" if attn_layers else "proxy_similarity"
        return ForwardCapture(
            layer_embeddings=layer_embeddings,
            attention_dict=attention_dict,
            node_layer_embeddings=node_layer_embeddings,
            pe_norm_dict={},
            prompt_summary=prompt_summary,
            score_mode=score_mode,
        )

    @staticmethod
    def _map_homo_attention_to_hetero(
        graph: dgl.DGLHeteroGraph,
        attention: torch.Tensor,
    ) -> dict[CanonicalEType, torch.Tensor]:
        """
        将 DGL homogeneous 图上的 attention 重新拆回异构边类型。
        """

        if attention.ndim >= 2:
            score = attention.mean(dim=tuple(range(1, attention.ndim))).view(-1)
        else:
            score = attention.view(-1)

        score = score.detach()
        etype_ids = dgl.to_homogeneous(graph).edata[dgl.ETYPE]
        out: dict[CanonicalEType, torch.Tensor] = {}
        for etype_id, canonical_etype in enumerate(graph.canonical_etypes):
            mask = etype_ids == etype_id
            out[canonical_etype] = score[mask]
        return out


def _flatten_score_dict(score_dict: dict[CanonicalEType, torch.Tensor]) -> list[float]:
    values: list[float] = []
    for score in score_dict.values():
        score = _to_cpu_tensor(score).view(-1)
        values.extend(float(x) for x in score.tolist())
    return values


def _normalize_draw_values(score_dict: dict[CanonicalEType, torch.Tensor]) -> dict[CanonicalEType, np.ndarray]:
    flat = np.asarray(_flatten_score_dict(score_dict), dtype=np.float32)
    if flat.size == 0:
        return {etype: np.zeros((0,), dtype=np.float32) for etype in score_dict}
    v_min = float(flat.min())
    v_max = float(flat.max())
    denom = max(v_max - v_min, 1e-12)
    out: dict[CanonicalEType, np.ndarray] = {}
    for etype, score in score_dict.items():
        arr = _to_cpu_tensor(score).numpy().astype(np.float32)
        out[etype] = (arr - v_min) / denom
    return out


def _etype_to_label(etype: CanonicalEType) -> str:
    return f"{etype[0]}:{etype[1]}:{etype[2]}"


def visualize_subgraph_aggregation(
    subgraph: dgl.DGLHeteroGraph,
    target_nid,
    attention_dict: dict[CanonicalEType, torch.Tensor] | None,
    pe_dict: dict[CanonicalEType, torch.Tensor] | None,
    save_path: str | Path,
    *,
    title: str | None = None,
):
    """
    使用 NetworkX 将异构子图的聚合结构可视化。

    参数说明：
    - subgraph: 单个样本的 DGL 异构子图。
    - target_nid: 目标节点在子图内的局部编号，或由 `inverse_indices` 返回的字典。
    - attention_dict: 每条边的聚合权重/代理权重。
    - pe_dict: peprompt 的边 PE 范数；若为空，则只画边粗细和透明度。
    - save_path: 输出图片路径。
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if attention_dict is None:
        attention_dict = {}
    if pe_dict is None:
        pe_dict = {}

    graph = nx.MultiDiGraph()
    node_colors = {}
    type_to_color = {}
    cmap = plt.get_cmap("tab10")
    for idx, ntype in enumerate(subgraph.ntypes):
        type_to_color[ntype] = cmap(idx % 10)
        for nid in range(subgraph.num_nodes(ntype)):
            node_key = (ntype, int(nid))
            graph.add_node(node_key, ntype=ntype)
            node_colors[node_key] = type_to_color[ntype]

    norm_attn = _normalize_draw_values(attention_dict)
    norm_pe = _normalize_draw_values(pe_dict)

    for etype in subgraph.canonical_etypes:
        src, dst = subgraph.edges(etype=etype)
        attn_values = norm_attn.get(etype)
        pe_values = norm_pe.get(etype)
        for eid, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
            edge_attr = {
                "etype": etype,
                "attention": float(attn_values[eid]) if attn_values is not None and eid < len(attn_values) else 0.5,
                "pe": float(pe_values[eid]) if pe_values is not None and eid < len(pe_values) else None,
            }
            graph.add_edge((etype[0], int(s)), (etype[2], int(d)), key=(etype, eid), **edge_attr)

    target_ntype = subgraph.ntypes[0]
    if isinstance(target_nid, dict):
        if len(target_nid) == 0:
            raise ValueError("target_nid dict is empty.")
        target_ntype = next(iter(target_nid.keys()))
        target_local_id = _extract_target_local_id(target_nid, target_ntype)
    else:
        target_local_id = _extract_target_local_id(target_nid, target_ntype)
    center_node = (target_ntype, int(target_local_id))

    pos = nx.spring_layout(graph, seed=42, k=1.0 / max(np.sqrt(max(graph.number_of_nodes(), 1)), 1.0))

    fig, ax = plt.subplots(figsize=(12, 9), dpi=220)

    draw_nodes = list(graph.nodes())
    node_sizes = [880 if node == center_node else 360 for node in draw_nodes]
    node_edgecolors = ["black" if node == center_node else "white" for node in draw_nodes]
    node_linewidths = [2.4 if node == center_node else 0.8 for node in draw_nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=draw_nodes,
        node_color=[node_colors[node] for node in draw_nodes],
        node_size=node_sizes,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        ax=ax,
    )

    edge_cmap = plt.get_cmap("viridis")
    raw_pe_values: list[float] = []
    if pe_dict:
        raw_pe_values = _flatten_score_dict(pe_dict)
    pe_min = min(raw_pe_values) if raw_pe_values else 0.0
    pe_max = max(raw_pe_values) if raw_pe_values else 1.0
    pe_denom = max(pe_max - pe_min, 1e-12)

    for u, v, _, data in graph.edges(keys=True, data=True):
        alpha = 0.18 + 0.82 * float(data["attention"])
        width = 0.8 + 4.8 * float(data["attention"])
        if data["pe"] is None:
            color = (0.22, 0.22, 0.22, alpha)
        else:
            pe_color = edge_cmap((float(data["pe"]) - pe_min) / pe_denom)
            color = (pe_color[0], pe_color[1], pe_color[2], alpha)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color=[color],
            arrows=True,
            arrowsize=12,
            ax=ax,
            connectionstyle="arc3,rad=0.06",
        )

    label_pos = {
        node: (coord[0], coord[1] + 0.03)
        for node, coord in pos.items()
    }
    labels = {
        node: f"{node[0]}\n{node[1]}"
        for node in graph.nodes()
    }
    nx.draw_networkx_labels(graph, label_pos, labels=labels, font_size=8, ax=ax)

    legend_handles = [
        Patch(facecolor=color, edgecolor="white", label=ntype)
        for ntype, color in type_to_color.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            markersize=12,
            linewidth=0,
            markeredgewidth=2.0,
            label="target node",
        )
    )
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    if pe_dict:
        norm = plt.Normalize(vmin=pe_min, vmax=pe_max)
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("PE L2 Norm", rotation=90)

    ax.set_axis_off()
    ax.set_title(title or "Subgraph Aggregation Visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _prepare_trajectory_array(
    layer_embeddings_list: list[list[torch.Tensor] | tuple[torch.Tensor, ...]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    expected_dim: int | None = None

    for sample_layers in layer_embeddings_list:
        if len(sample_layers) == 0:
            continue
        for emb in sample_layers:
            arr = _to_cpu_tensor(emb).numpy().astype(np.float32).reshape(-1)
            if expected_dim is None:
                expected_dim = int(arr.shape[0])
            if arr.shape[0] != expected_dim:
                raise ValueError(
                    "All layer embeddings must share the same dimension for trajectory visualization. "
                    f"Found {arr.shape[0]} vs expected {expected_dim}."
                )
            rows.append(arr)

    if not rows:
        raise ValueError("layer_embeddings_list is empty, cannot visualize trajectory.")
    return np.stack(rows, axis=0)


def _run_tsne_or_pca(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 4:
        return PCA(n_components=2).fit_transform(x)

    perplexity = min(30, max(2, x.shape[0] // 8))
    perplexity = min(perplexity, x.shape[0] - 1)
    if perplexity < 2:
        return PCA(n_components=2).fit_transform(x)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return tsne.fit_transform(x)


def visualize_feature_trajectory(
    layer_embeddings_list: list[list[torch.Tensor] | tuple[torch.Tensor, ...]],
    labels: list[int] | torch.Tensor | np.ndarray,
    save_path: str | Path,
    *,
    layer_names: list[str] | None = None,
    title: str | None = None,
):
    """
    将多层目标节点特征降维到 2D，并用箭头展示层间漂移轨迹。
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().view(-1).tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.reshape(-1).tolist()
    else:
        labels = list(labels)

    num_samples = len(layer_embeddings_list)
    if num_samples == 0:
        raise ValueError("No samples provided for trajectory visualization.")
    num_layers = len(layer_embeddings_list[0])
    if layer_names is None:
        layer_names = [f"Layer{i}" for i in range(num_layers)]

    x = _prepare_trajectory_array(layer_embeddings_list)
    coords = _run_tsne_or_pca(x)
    coords = coords.reshape(num_samples, num_layers, 2)

    unique_labels = sorted(set(int(x) for x in labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    cmap = plt.get_cmap("tab10")
    layer_markers = ["o", "s", "^", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=220)

    for sample_idx, sample_layers in enumerate(coords):
        label = int(labels[sample_idx])
        color = cmap(label_to_idx[label] % 10)

        for layer_idx, point in enumerate(sample_layers):
            marker = layer_markers[layer_idx % len(layer_markers)]
            ax.scatter(
                point[0],
                point[1],
                color=color,
                marker=marker,
                s=56 if layer_idx < num_layers - 1 else 76,
                alpha=0.88,
                linewidths=0.2,
                edgecolors="black",
            )

        for layer_idx in range(num_layers - 1):
            start = sample_layers[layer_idx]
            end = sample_layers[layer_idx + 1]
            arrow = FancyArrowPatch(
                posA=(start[0], start[1]),
                posB=(end[0], end[1]),
                arrowstyle="->",
                mutation_scale=10,
                linewidth=0.8,
                color=color,
                alpha=0.35,
            )
            ax.add_patch(arrow)

    label_handles = [
        Line2D([0], [0], marker="o", color="w", label=f"class {label}", markerfacecolor=cmap(label_to_idx[label] % 10), markersize=8)
        for label in unique_labels
    ]
    layer_handles = [
        Line2D([0], [0], marker=layer_markers[idx % len(layer_markers)], color="black", label=name, linestyle="None", markersize=8)
        for idx, name in enumerate(layer_names)
    ]

    ax.legend(handles=label_handles + layer_handles, loc="best", frameon=True, ncol=2)
    ax.set_title(title or "Target-Node Feature Trajectory", fontsize=14)
    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def visualize_target_feature_heatmap(
    layer_embeddings: dict[str, torch.Tensor],
    save_path: str | Path,
    *,
    top_k_dims: int = 48,
    title: str | None = None,
):
    """
    对单个目标节点的 Layer0/1/2 特征做热力图。
    只保留变化最明显的 top-k 维度，避免 512 维直接画满难以阅读。
    """
    ordered_names = [name for name in ("Layer0", "Layer1", "Layer2") if name in layer_embeddings]
    if len(ordered_names) < 2:
        raise ValueError("Need at least two layers to visualize target feature heatmap.")

    stacked = torch.stack([_to_cpu_tensor(layer_embeddings[name]).view(-1) for name in ordered_names], dim=0)
    diff_score = (stacked.max(dim=0).values - stacked.min(dim=0).values).abs()
    top_k_dims = min(int(top_k_dims), int(stacked.size(1)))
    top_idx = torch.topk(diff_score, k=top_k_dims, largest=True).indices
    top_idx, _ = torch.sort(top_idx)
    selected = stacked[:, top_idx].numpy()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, top_k_dims * 0.22), 3.8), dpi=220)
    vmax = float(np.abs(selected).max()) if selected.size > 0 else 1.0
    vmax = max(vmax, 1e-12)
    im = ax.imshow(selected, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_yticklabels(ordered_names)
    ax.set_xticks(np.arange(top_k_dims))
    ax.set_xticklabels([str(int(idx)) for idx in top_idx.tolist()], rotation=90, fontsize=7)
    ax.set_xlabel("Feature Dimension")
    ax.set_title(title or "Target Feature Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature Value", rotation=90)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def visualize_target_feature_delta_bars(
    layer_embeddings: dict[str, torch.Tensor],
    save_path: str | Path,
    *,
    top_k_dims: int = 20,
    title: str | None = None,
):
    """
    画出目标节点变化最大的特征维度。
    默认使用 Layer2-Layer0 的绝对变化排序。
    """
    if "Layer0" not in layer_embeddings or "Layer2" not in layer_embeddings:
        raise ValueError("Target delta bar plot requires both Layer0 and Layer2.")

    x0 = _to_cpu_tensor(layer_embeddings["Layer0"]).view(-1)
    x2 = _to_cpu_tensor(layer_embeddings["Layer2"]).view(-1)
    delta = x2 - x0
    top_k_dims = min(int(top_k_dims), int(delta.numel()))
    top_idx = torch.topk(delta.abs(), k=top_k_dims, largest=True).indices
    top_idx_cpu = top_idx.detach().cpu()
    values = delta[top_idx_cpu].numpy()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    colors = ["#c44e52" if value >= 0 else "#4c72b0" for value in values]
    ax.bar(np.arange(top_k_dims), values, color=colors, alpha=0.88)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(top_k_dims))
    ax.set_xticklabels([str(int(idx)) for idx in top_idx_cpu.tolist()], rotation=90)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Layer2 - Layer0")
    ax.set_title(title or "Top Feature Deltas")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def visualize_node_drift_heatmap(
    node_layer_embeddings: dict[str, dict[str, torch.Tensor]],
    save_path: str | Path,
    *,
    top_k_nodes: int = 40,
    title: str | None = None,
) -> list[dict[str, float | int | str]]:
    """
    对样本子图内所有节点的层间漂移做数值热力图。
    返回值会同时被上层写入 CSV。
    """
    rows = _summarize_node_layer_drift(node_layer_embeddings)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return rows

    top_rows = rows[: min(len(rows), int(top_k_nodes))]
    metric_names = [name for name in ("delta01_l2", "delta12_l2", "delta02_l2", "cos01", "cos12", "cos02") if name in top_rows[0]]
    matrix = np.asarray(
        [[float(row.get(metric, 0.0)) for metric in metric_names] for row in top_rows],
        dtype=np.float32,
    )
    labels = [f"{row['ntype']}:{row['node_id']}" for row in top_rows]

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, len(top_rows) * 0.26)), dpi=220)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title or "Node Drift Metrics Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Metric Value", rotation=90)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return rows
