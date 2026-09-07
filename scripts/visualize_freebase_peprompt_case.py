from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dgl  # noqa: E402

from gpbench.downstream.fewshot import load_peprompt_split_ids  # noqa: E402
from scripts.peprompt_benchmark import TARGET_NODETYPE, _load_raw_heterograph  # noqa: E402
from scripts.precompute_peprompt_cache import _adaptive_relative_indices  # noqa: E402
from scripts.subgraph_sampling_stats import (  # noqa: E402
    _build_csr_adjs,
    _endpoint_popularity,
    _generate_metapaths,
    _metapath_reachable_scores,
    _rank_values,
    _topk_indices,
)


EdgeType = tuple[str, str, str]
MetaPath = tuple[EdgeType, ...]


NODE_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]
KEEP_COLOR = "#f28e2b"
DROP_COLOR = "#b8bcc2"
TARGET_COLOR = "#d62728"
EDGE_COLOR = "#8a8f98"
PE_COLOR = "#4e79a7"


@dataclass
class MetaPathRecord:
    metapath_id: int
    metapath: MetaPath
    dst_type: str
    candidate_ids: np.ndarray
    candidate_scores: np.ndarray
    candidate_ranks: np.ndarray
    keep_ids: np.ndarray
    keep_scores: np.ndarray
    keep_ranks: np.ndarray


@dataclass
class CasePayload:
    target_id: int
    target_label: int | None
    selected: dict[str, set[int]]
    candidate: dict[str, set[int]]
    records: list[MetaPathRecord]
    subgraph: Any
    subgraph_stats: dict[str, Any]


def _format_float_for_key(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _cache_subgraph_type(args: argparse.Namespace) -> str:
    if args.subgraph_type in {"metapath_topk_adapt", "metapath_topk_path_adapt"}:
        suffix = (
            f"m{args.metapath_max_hop}_"
            f"k{args.metapath_min_topk}-{args.metapath_max_topk}_"
            f"a{_format_float_for_key(args.metapath_rel_threshold)}_"
            f"{args.metapath_rank_metric}"
        )
    else:
        suffix = f"m{args.metapath_max_hop}_k{args.metapath_topk}_{args.metapath_rank_metric}"
    if args.metapath_keep_self:
        suffix += "_self"
    return f"{args.subgraph_type}_{suffix}"


def _labels_for_target(graph, target_ntype: str) -> np.ndarray | None:
    if "y" not in graph.nodes[target_ntype].data:
        return None
    labels = graph.nodes[target_ntype].data["y"].detach().cpu()
    if labels.ndim == 2:
        labels = labels.argmax(dim=-1)
    return labels.long().view(-1).numpy()


def _choose_target_id(graph, target_ntype: str, args: argparse.Namespace) -> int:
    if args.target_id is not None:
        return int(args.target_id)

    if args.case_source == "labeled":
        labels = _labels_for_target(graph, target_ntype)
        if labels is None:
            raise RuntimeError("Cannot use --case_source labeled because target labels are missing.")
        ids = np.flatnonzero(labels >= 0).astype(np.int64)
    else:
        split_ids = load_peprompt_split_ids(
            cache_dir=args.peprompt_offline_cache_dir,
            dataset_name=args.dataset,
            shot=args.shot,
            seed=args.split_seed,
            feats_type=args.feats_type,
            subgraph_type=_cache_subgraph_type(args),
            write_sidecar_if_missing=False,
        )
        key = f"{args.case_source}_ids"
        ids = np.asarray(split_ids[key], dtype=np.int64)

    if ids.size == 0:
        raise RuntimeError(f"No candidate target ids found for case_source={args.case_source}.")
    return int(ids[min(max(int(args.case_index), 0), ids.size - 1)])


def _metapath_label(metapath: MetaPath) -> str:
    if not metapath:
        return ""
    parts = [metapath[0][0]]
    parts.extend(etype[2] for etype in metapath)
    return " -> ".join(parts)


def _short_rel(rel: str, max_len: int = 16) -> str:
    rel = str(rel)
    if len(rel) <= max_len:
        return rel
    return rel[: max_len - 1] + "."


def _node_key(ntype: str, node_id: int) -> str:
    return f"{ntype}:{int(node_id)}"


def _node_label(ntype: str, node_id: int, target_ntype: str, target_id: int) -> str:
    prefix = ntype[:2] if len(ntype) > 1 else ntype
    if ntype == target_ntype and int(node_id) == int(target_id):
        return f"{prefix}{int(node_id)}\ntarget"
    return f"{prefix}{int(node_id)}"


def _type_colors(ntypes: list[str]) -> dict[str, str]:
    return {ntype: NODE_PALETTE[i % len(NODE_PALETTE)] for i, ntype in enumerate(ntypes)}


def _ranked_candidates(scores: np.ndarray, ranks: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero = np.flatnonzero(scores > 0).astype(np.int64)
    if nonzero.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return nonzero, empty, empty
    order = np.lexsort((nonzero, -np.asarray(ranks[nonzero], dtype=np.float32)))
    ids = nonzero[order][: int(limit)]
    return ids, np.asarray(scores[ids], dtype=np.float32), np.asarray(ranks[ids], dtype=np.float32)


def _replay_selection(graph, target_ntype: str, target_id: int, args: argparse.Namespace) -> CasePayload:
    adjs = _build_csr_adjs(graph)
    metapaths = _generate_metapaths(graph, target_ntype, int(args.metapath_max_hop))
    endpoint_popularity: dict[MetaPath, np.ndarray | np.float32] = {}
    if args.metapath_rank_metric == "degree_norm":
        endpoint_popularity = {metapath: _endpoint_popularity(adjs, metapath) for metapath in metapaths}
    elif args.metapath_rank_metric == "count_idf":
        from scripts.precompute_peprompt_cache import _metapath_source_idf

        endpoint_popularity = {
            metapath: np.float32(_metapath_source_idf(adjs, metapath))
            for metapath in metapaths
        }

    selected: dict[str, set[int]] = {ntype: set() for ntype in graph.ntypes}
    candidate: dict[str, set[int]] = {ntype: set() for ntype in graph.ntypes}
    selected[target_ntype].add(int(target_id))
    candidate[target_ntype].add(int(target_id))
    records: list[MetaPathRecord] = []

    adaptive = args.subgraph_type in {"metapath_topk_adapt", "metapath_topk_path_adapt"}
    for metapath_id, metapath in enumerate(metapaths):
        scores = _metapath_reachable_scores(
            adjs=adjs,
            metapath=metapath,
            node_id=int(target_id),
            target_ntype=target_ntype,
            keep_self=bool(args.metapath_keep_self),
        )
        ranks = _rank_values(scores, endpoint_popularity.get(metapath), args.metapath_rank_metric)
        if adaptive:
            keep = _adaptive_relative_indices(
                scores=scores,
                ranks=ranks,
                min_topk=int(args.metapath_min_topk),
                max_topk=int(args.metapath_max_topk),
                rel_threshold=float(args.metapath_rel_threshold),
            )
        else:
            keep = _topk_indices(scores, ranks, int(args.metapath_topk))

        cand_ids, cand_scores, cand_ranks = _ranked_candidates(
            scores,
            ranks,
            max(int(args.candidates_per_metapath), int(args.metapath_topk), int(args.metapath_max_topk)),
        )
        dst_type = metapath[-1][2]
        selected[dst_type].update(int(v) for v in keep.tolist())
        candidate[dst_type].update(int(v) for v in cand_ids.tolist())
        records.append(
            MetaPathRecord(
                metapath_id=metapath_id,
                metapath=metapath,
                dst_type=dst_type,
                candidate_ids=cand_ids,
                candidate_scores=cand_scores,
                candidate_ranks=cand_ranks,
                keep_ids=keep.astype(np.int64),
                keep_scores=np.asarray(scores[keep], dtype=np.float32),
                keep_ranks=np.asarray(ranks[keep], dtype=np.float32),
            )
        )

    node_dict = {
        ntype: torch.tensor(sorted(node_ids), dtype=torch.int64)
        for ntype, node_ids in selected.items()
        if node_ids
    }
    subgraph = dgl.node_subgraph(graph, node_dict)
    stats = {
        "num_nodes": int(subgraph.num_nodes()),
        "num_edges": int(sum(subgraph.num_edges(etype=etype) for etype in subgraph.canonical_etypes)),
        "nodes_by_type": {ntype: int(subgraph.num_nodes(ntype)) for ntype in subgraph.ntypes},
        "edges_by_type": {
            f"{etype[0]}__{etype[1]}__{etype[2]}": int(subgraph.num_edges(etype=etype))
            for etype in subgraph.canonical_etypes
            if subgraph.num_edges(etype=etype) > 0
        },
    }
    labels = _labels_for_target(graph, target_ntype)
    label = int(labels[int(target_id)]) if labels is not None and 0 <= int(target_id) < len(labels) else None
    return CasePayload(
        target_id=int(target_id),
        target_label=label,
        selected=selected,
        candidate=candidate,
        records=records,
        subgraph=subgraph,
        subgraph_stats=stats,
    )


def _case_score(payload: CasePayload, target_nodes: int = 55, target_edges: int = 180) -> float:
    active = sum(1 for rec in payload.records if rec.keep_ids.size > 0)
    n = int(payload.subgraph_stats["num_nodes"])
    e = int(payload.subgraph_stats["num_edges"])
    type_count = sum(1 for ids in payload.selected.values() if ids)
    return active * 8.0 + type_count * 5.0 - abs(n - target_nodes) * 0.7 - max(e - target_edges, 0) * 0.08


def _auto_choose_case(graph, target_ntype: str, args: argparse.Namespace) -> CasePayload:
    if args.target_id is not None:
        return _replay_selection(graph, target_ntype, int(args.target_id), args)

    split_ids = None
    if args.case_source != "labeled":
        split_ids = load_peprompt_split_ids(
            cache_dir=args.peprompt_offline_cache_dir,
            dataset_name=args.dataset,
            shot=args.shot,
            seed=args.split_seed,
            feats_type=args.feats_type,
            subgraph_type=_cache_subgraph_type(args),
            write_sidecar_if_missing=False,
        )
        ids = np.asarray(split_ids[f"{args.case_source}_ids"], dtype=np.int64)
    else:
        labels = _labels_for_target(graph, target_ntype)
        if labels is None:
            raise RuntimeError("Cannot auto choose from labeled nodes because labels are missing.")
        ids = np.flatnonzero(labels >= 0).astype(np.int64)

    if ids.size == 0:
        raise RuntimeError(f"No candidate target ids found for case_source={args.case_source}.")
    if args.auto_case:
        pool = ids[: int(args.auto_case_pool)] if args.case_source != "labeled" else ids
        if args.case_source == "labeled" and pool.size > int(args.auto_case_pool):
            rng = np.random.default_rng(int(args.seed))
            pool = rng.choice(pool, size=int(args.auto_case_pool), replace=False)
        best_payload = None
        best_score = -float("inf")
        for node_id in pool:
            payload = _replay_selection(graph, target_ntype, int(node_id), args)
            n = int(payload.subgraph_stats["num_nodes"])
            e = int(payload.subgraph_stats["num_edges"])
            if n < args.min_case_nodes or n > args.max_case_nodes or e > args.max_case_edges:
                continue
            score = _case_score(payload)
            if score > best_score:
                best_score = score
                best_payload = payload
        if best_payload is not None:
            return best_payload

    node_id = int(ids[min(max(int(args.case_index), 0), ids.size - 1)])
    return _replay_selection(graph, target_ntype, node_id, args)


def _subgraph_to_nx(
    subgraph,
    target_ntype: str,
    target_id: int,
    selected: dict[str, set[int]] | None = None,
    max_edges: int | None = None,
) -> nx.MultiDiGraph:
    out = nx.MultiDiGraph()
    local_to_key: dict[tuple[str, int], str] = {}
    selected = selected or {}
    for ntype in subgraph.ntypes:
        nids = subgraph.nodes[ntype].data[dgl.NID].detach().cpu().long().tolist()
        for local_id, global_id in enumerate(nids):
            key = _node_key(ntype, int(global_id))
            local_to_key[(ntype, local_id)] = key
            out.add_node(
                key,
                ntype=ntype,
                global_id=int(global_id),
                is_target=ntype == target_ntype and int(global_id) == int(target_id),
                is_selected=int(global_id) in selected.get(ntype, set()),
            )

    edges = []
    for etype in subgraph.canonical_etypes:
        src_t, rel_t, dst_t = etype
        src, dst = subgraph.edges(etype=etype)
        for edge_idx, (s, d) in enumerate(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist())):
            sk = local_to_key[(src_t, int(s))]
            dk = local_to_key[(dst_t, int(d))]
            priority = int(out.nodes[sk]["is_target"] or out.nodes[dk]["is_target"])
            edges.append((priority, sk, dk, rel_t, edge_idx, etype))
    edges.sort(key=lambda item: (-item[0], item[4]))
    if max_edges and max_edges > 0:
        edges = edges[: int(max_edges)]
    for _, sk, dk, rel_t, edge_idx, etype in edges:
        out.add_edge(sk, dk, key=f"{rel_t}:{edge_idx}", rel=str(rel_t), etype=etype, semantic=False)
    return out


def _add_semantic_reachability_edges(
    nx_graph: nx.MultiDiGraph,
    target_key: str,
    records: list[MetaPathRecord],
    *,
    include_candidates: bool,
    max_edges: int,
) -> None:
    """Add visual-only target-to-endpoint links for metapath reachability.

    The fixed Freebase mainline uses endpoint-induced subgraphs, so selected
    endpoints are not guaranteed to remain connected to the target by concrete
    edges. These dashed links make the selection process explicit for PPT.
    """
    by_endpoint: dict[str, dict[str, Any]] = {}
    for rec in records:
        keep_set = {int(v) for v in rec.keep_ids.tolist()}
        if include_candidates:
            endpoint_ids = rec.candidate_ids.tolist()
            scores = rec.candidate_scores.tolist()
        else:
            endpoint_ids = rec.keep_ids.tolist()
            scores = rec.keep_scores.tolist()
        for endpoint_id, score in zip(endpoint_ids, scores):
            endpoint_id = int(endpoint_id)
            if rec.dst_type == target_key.split(":", 1)[0] and endpoint_id == int(target_key.split(":", 1)[1]):
                continue
            key = _node_key(rec.dst_type, endpoint_id)
            if key not in nx_graph:
                continue
            item = by_endpoint.setdefault(
                key,
                {
                    "count": 0,
                    "kept": False,
                    "score": 0.0,
                    "metapaths": [],
                },
            )
            item["count"] += 1
            item["kept"] = bool(item["kept"] or endpoint_id in keep_set)
            item["score"] = max(float(item["score"]), float(score))
            if len(item["metapaths"]) < 3:
                item["metapaths"].append(rec.metapath_id)

    ranked = sorted(
        by_endpoint.items(),
        key=lambda item: (
            not bool(item[1]["kept"]),
            -int(item[1]["count"]),
            -float(item[1]["score"]),
            item[0],
        ),
    )
    for endpoint_key, info in ranked[: int(max_edges)]:
        nx_graph.add_edge(
            target_key,
            endpoint_key,
            key=f"semantic:{endpoint_key}",
            semantic=True,
            kept=bool(info["kept"]),
            count=int(info["count"]),
            score=float(info["score"]),
            rel="metapath reachability",
            etype=("semantic", "metapath", "endpoint"),
        )


def _induced_subgraph_from_node_sets(graph, node_sets: dict[str, set[int]]):
    node_dict = {
        ntype: torch.tensor(sorted(ids), dtype=torch.int64)
        for ntype, ids in node_sets.items()
        if ids
    }
    return dgl.node_subgraph(graph, node_dict)


def _distance_shells(nx_graph: nx.MultiDiGraph, target_key: str) -> dict[str, int]:
    undirected = nx.Graph()
    undirected.add_nodes_from(nx_graph.nodes())
    undirected.add_edges_from((u, v) for u, v in nx_graph.edges())
    dist = {node: 99 for node in nx_graph.nodes()}
    if target_key not in undirected:
        return dist
    queue: deque[str] = deque([target_key])
    dist[target_key] = 0
    while queue:
        node = queue.popleft()
        for nb in undirected.neighbors(node):
            if dist[nb] == 99:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def _layout_target_centered(nx_graph: nx.MultiDiGraph, target_key: str, seed: int) -> dict[str, np.ndarray]:
    if nx_graph.number_of_nodes() == 0:
        return {}
    shells = _distance_shells(nx_graph, target_key)
    grouped: dict[int, list[str]] = defaultdict(list)
    for node in nx_graph.nodes():
        grouped[min(shells.get(node, 99), 4)].append(node)

    init = {}
    rng = np.random.default_rng(int(seed))
    for shell, nodes in grouped.items():
        nodes = sorted(nodes)
        if shell == 0:
            init[nodes[0]] = np.array([0.0, 0.0])
            continue
        radius = 1.2 + 0.62 * shell
        offset = float(rng.uniform(0, 2 * math.pi))
        for idx, node in enumerate(nodes):
            theta = offset + 2 * math.pi * idx / max(len(nodes), 1)
            init[node] = np.array([radius * math.cos(theta), radius * math.sin(theta)])
    try:
        return nx.spring_layout(
            nx.Graph(nx_graph),
            pos=init,
            fixed=[target_key] if target_key in nx_graph else None,
            seed=int(seed),
            k=0.58,
            iterations=120,
        )
    except Exception:
        return init


def _draw_network_panel(
    ax,
    nx_graph: nx.MultiDiGraph,
    target_key: str,
    target_ntype: str,
    target_id: int,
    type_colors: dict[str, str],
    title: str,
    subtitle: str,
    seed: int,
    show_labels: bool,
):
    pos = _layout_target_centered(nx_graph, target_key, seed)
    edge_counter: Counter[tuple[str, str]] = Counter()
    for u, v, data in nx_graph.edges(data=True):
        edge_counter[(u, v)] += 1
        rad = 0.08 * (edge_counter[(u, v)] - 1)
        if nx_graph.has_edge(v, u):
            rad += 0.10
        is_semantic = bool(data.get("semantic", False))
        is_kept = bool(data.get("kept", False))
        if is_semantic:
            color = KEEP_COLOR if is_kept else "#9aa0aa"
            alpha = 0.58 if is_kept else 0.25
            linewidth = 1.0 + 0.28 * min(int(data.get("count", 1)), 6)
            linestyle = (0, (4, 3))
            mutation_scale = 7 if is_kept else 5
            zorder = 1
        else:
            color = EDGE_COLOR
            alpha = 0.18
            linewidth = 0.55
            linestyle = "solid"
            mutation_scale = 5
            zorder = 0
        patch = FancyArrowPatch(
            posA=pos[u],
            posB=pos[v],
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            shrinkA=7,
            shrinkB=7,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
        ax.add_patch(patch)

    nodes = list(nx_graph.nodes())
    node_colors = []
    node_sizes = []
    edgecolors = []
    linewidths = []
    for node in nodes:
        data = nx_graph.nodes[node]
        ntype = str(data.get("ntype"))
        if data.get("is_target"):
            node_colors.append(TARGET_COLOR)
            node_sizes.append(900)
            edgecolors.append("#111111")
            linewidths.append(2.4)
        elif data.get("is_selected"):
            node_colors.append(KEEP_COLOR)
            node_sizes.append(430)
            edgecolors.append("#3b2a12")
            linewidths.append(1.3)
        else:
            node_colors.append(type_colors.get(ntype, "#bab0ac"))
            node_sizes.append(260)
            edgecolors.append("#ffffff")
            linewidths.append(0.6)

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=0.96,
        ax=ax,
    )
    if show_labels:
        labels = {
            node: _node_label(str(data["ntype"]), int(data["global_id"]), target_ntype, target_id)
            for node, data in nx_graph.nodes(data=True)
        }
        nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=6.5, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(
        0.5,
        -0.04,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax.axis("off")


def _top_records(records: list[MetaPathRecord], max_rows: int) -> list[MetaPathRecord]:
    active = [rec for rec in records if rec.keep_ids.size > 0]
    active.sort(key=lambda rec: (-rec.keep_ids.size, -float(rec.keep_ranks.sum() if rec.keep_ranks.size else 0), rec.metapath_id))
    return active[: int(max_rows)]


def _draw_metapath_selection(ax, records: list[MetaPathRecord], max_rows: int, max_endpoints: int):
    rows = _top_records(records, max_rows)
    if not rows:
        ax.text(0.5, 0.5, "No active metapath endpoints", ha="center", va="center")
        ax.axis("off")
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    for row_idx, rec in enumerate(rows):
        y = row_idx
        label = _metapath_label(rec.metapath)
        ax.text(0.02, y, f"MP{rec.metapath_id}", ha="left", va="center", fontsize=9, fontweight="bold")
        ax.text(0.11, y, label, ha="left", va="center", fontsize=8, color="#333333")
        ax.plot([0.46, 0.94], [y, y], color="#d0d3d8", linewidth=1.0, zorder=1)

        keep_set = {int(v) for v in rec.keep_ids.tolist()}
        ids = rec.candidate_ids[: int(max_endpoints)]
        scores = rec.candidate_scores[: int(max_endpoints)]
        if ids.size == 0:
            continue
        xs = np.linspace(0.50, 0.92, ids.size)
        max_score = max(float(scores.max()), 1.0)
        for x, endpoint_id, score in zip(xs, ids.tolist(), scores.tolist()):
            kept = int(endpoint_id) in keep_set
            size = 80 + 170 * float(score) / max_score
            ax.scatter(
                [x],
                [y],
                s=size,
                color=KEEP_COLOR if kept else DROP_COLOR,
                edgecolors="#3b2a12" if kept else "#ffffff",
                linewidths=1.0 if kept else 0.5,
                zorder=3,
            )
            ax.text(
                x,
                y + 0.28,
                str(int(endpoint_id)),
                ha="center",
                va="center",
                fontsize=6.5,
                color="#333333",
            )
        ax.text(
            0.965,
            y,
            f"keep {len(keep_set)}",
            ha="right",
            va="center",
            fontsize=8,
            color="#6b4b13",
        )
    ax.set_title("Metapath endpoint selection", fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_pe_prompt_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Edge structural prompt", fontsize=13, fontweight="bold", pad=8)
    left = (0.18, 0.62)
    right = (0.82, 0.62)
    mid = (0.50, 0.62)
    ax.scatter([left[0], right[0]], [left[1], right[1]], s=[520, 520], color=[KEEP_COLOR, TARGET_COLOR], edgecolors="#111111")
    ax.text(left[0], left[1], "j", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    ax.text(right[0], right[1], "i", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    ax.add_patch(
        FancyArrowPatch(
            left,
            right,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            color="#555555",
            connectionstyle="arc3,rad=0.0",
        )
    )
    ax.text(mid[0], mid[1] + 0.09, "edge (j -> i)", ha="center", fontsize=9, color="#333333")
    ax.text(0.50, 0.38, "PE(j) - PE(i)", ha="center", va="center", fontsize=10, color=PE_COLOR)
    ax.add_patch(
        FancyArrowPatch(
            (0.50, 0.56),
            (0.50, 0.43),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color=PE_COLOR,
        )
    )
    ax.text(0.50, 0.25, "MLP", ha="center", va="center", fontsize=11, fontweight="bold", bbox={"boxstyle": "round,pad=0.25", "fc": "#eef3fb", "ec": PE_COLOR})
    ax.add_patch(
        FancyArrowPatch(
            (0.50, 0.34),
            (0.50, 0.29),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color=PE_COLOR,
        )
    )
    ax.text(0.50, 0.10, "p_ij modulates message  m_j->i = h_j * p_ij", ha="center", va="center", fontsize=9, color="#333333")


def _save_figure(fig, path_stem: Path, formats: list[str], dpi: int):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def _write_records_csv(path: Path, records: list[MetaPathRecord]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metapath_id",
                "metapath",
                "endpoint_type",
                "endpoint_id",
                "score",
                "rank",
                "kept",
            ],
        )
        writer.writeheader()
        for rec in records:
            keep = {int(v) for v in rec.keep_ids.tolist()}
            for endpoint_id, score, rank in zip(rec.candidate_ids.tolist(), rec.candidate_scores.tolist(), rec.candidate_ranks.tolist()):
                writer.writerow(
                    {
                        "metapath_id": rec.metapath_id,
                        "metapath": _metapath_label(rec.metapath),
                        "endpoint_type": rec.dst_type,
                        "endpoint_id": int(endpoint_id),
                        "score": float(score),
                        "rank": float(rank),
                        "kept": int(endpoint_id) in keep,
                    }
                )


def _visualize(args: argparse.Namespace):
    graph, target_ntype = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    if args.target_ntype:
        target_ntype = args.target_ntype
    else:
        target_ntype = TARGET_NODETYPE.get(args.dataset, target_ntype)
    payload = _auto_choose_case(graph, target_ntype, args)

    out_dir = Path(args.out_dir) / (
        f"{args.dataset}_target{payload.target_id}_{_cache_subgraph_type(args)}_"
        f"shot{args.shot}_seed{args.split_seed}_ft{args.feats_type}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    type_colors = _type_colors(list(graph.ntypes))
    target_key = _node_key(target_ntype, payload.target_id)
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]

    candidate_subgraph = _induced_subgraph_from_node_sets(graph, payload.candidate)
    nx_candidate = _subgraph_to_nx(
        candidate_subgraph,
        target_ntype,
        payload.target_id,
        selected=payload.selected,
        max_edges=args.max_draw_edges,
    )
    nx_final = _subgraph_to_nx(
        payload.subgraph,
        target_ntype,
        payload.target_id,
        selected=payload.selected,
        max_edges=args.max_draw_edges,
    )
    _add_semantic_reachability_edges(
        nx_candidate,
        target_key,
        payload.records,
        include_candidates=True,
        max_edges=args.max_semantic_edges,
    )
    _add_semantic_reachability_edges(
        nx_final,
        target_key,
        payload.records,
        include_candidates=False,
        max_edges=args.max_semantic_edges,
    )

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    _draw_network_panel(
        ax,
        nx_candidate,
        target_key,
        target_ntype,
        payload.target_id,
        type_colors,
        "Candidate endpoints around the target",
        "gray/type-colored nodes are reachable candidates; orange nodes are retained endpoints",
        seed=args.seed,
        show_labels=args.show_labels,
    )
    _save_figure(fig, out_dir / "01_candidate_endpoints", formats, args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    _draw_metapath_selection(ax, payload.records, args.max_metapath_rows, args.max_endpoints_per_row)
    _save_figure(fig, out_dir / "02_metapath_endpoint_selection", formats, args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    _draw_network_panel(
        ax,
        nx_final,
        target_key,
        target_ntype,
        payload.target_id,
        type_colors,
        "Final semantic subgraph",
        "induced subgraph from the target and retained metapath endpoints",
        seed=args.seed + 17,
        show_labels=args.show_labels,
    )
    _save_figure(fig, out_dir / "03_semantic_subgraph", formats, args.dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(17.0, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 0.78])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_network_panel(
        ax1,
        nx_candidate,
        target_key,
        target_ntype,
        payload.target_id,
        type_colors,
        "1. Candidate endpoints",
        "top reachable endpoints per metapath",
        seed=args.seed,
        show_labels=False,
    )
    _draw_metapath_selection(ax2, payload.records, args.max_metapath_rows, args.max_endpoints_per_row)
    _draw_network_panel(
        ax3,
        nx_final,
        target_key,
        target_ntype,
        payload.target_id,
        type_colors,
        "3. Semantic subgraph",
        "target + retained endpoints + induced edges",
        seed=args.seed + 17,
        show_labels=False,
    )
    _draw_pe_prompt_panel(ax4)
    handles = [
        Patch(facecolor=TARGET_COLOR, edgecolor="#111111", label=f"target {target_ntype}"),
        Patch(facecolor=KEEP_COLOR, edgecolor="#3b2a12", label="retained endpoint"),
        Patch(facecolor=DROP_COLOR, edgecolor="#ffffff", label="dropped candidate"),
    ]
    for ntype, color in type_colors.items():
        handles.append(Patch(facecolor=color, edgecolor="#ffffff", label=ntype))
    fig.legend(handles=handles[: min(len(handles), 12)], loc="lower center", ncol=6, fontsize=9, frameon=True)
    fig.suptitle(
        f"PEPrompt Freebase case: {target_ntype}:{payload.target_id}"
        + (f" class {payload.target_label}" if payload.target_label is not None else ""),
        fontsize=16,
        fontweight="bold",
    )
    _save_figure(fig, out_dir / "04_peprompt_case_overview", formats, args.dpi)
    plt.close(fig)

    summary = {
        "dataset": args.dataset,
        "target_ntype": target_ntype,
        "target_id": payload.target_id,
        "target_label": payload.target_label,
        "cache_key": _cache_subgraph_type(args),
        "shot": int(args.shot),
        "split_seed": int(args.split_seed),
        "feats_type": int(args.feats_type),
        "subgraph_stats": payload.subgraph_stats,
        "active_metapaths": int(sum(1 for rec in payload.records if rec.keep_ids.size > 0)),
        "selected_nodes_by_type": {ntype: len(ids) for ntype, ids in payload.selected.items() if ids},
        "candidate_nodes_by_type": {ntype: len(ids) for ntype, ids in payload.candidate.items() if ids},
        "outputs": sorted(str(path.relative_to(out_dir)) for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "case_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_records_csv(out_dir / "metapath_endpoint_ranking.csv", payload.records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Visualize one Freebase PEPrompt semantic-subgraph selection case.")
    parser.add_argument("--dataset", default="Freebase", choices=["Freebase", "ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--target_ntype", default=None)
    parser.add_argument("--target_id", type=int, default=None)
    parser.add_argument("--case_source", choices=["train", "val", "test", "labeled"], default="train")
    parser.add_argument("--case_index", type=int, default=0)
    parser.add_argument("--auto_case", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto_case_pool", type=int, default=64)
    parser.add_argument("--min_case_nodes", type=int, default=24)
    parser.add_argument("--max_case_nodes", type=int, default=110)
    parser.add_argument("--max_case_edges", type=int, default=520)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--feats_type", type=int, default=1)
    parser.add_argument("--subgraph_type", default="metapath_topk", choices=["metapath_topk", "metapath_topk_adapt", "metapath_topk_path", "metapath_topk_path_adapt"])
    parser.add_argument("--metapath_max_hop", type=int, default=3)
    parser.add_argument("--metapath_topk", type=int, default=3)
    parser.add_argument("--metapath_min_topk", type=int, default=1)
    parser.add_argument("--metapath_max_topk", type=int, default=5)
    parser.add_argument("--metapath_rel_threshold", type=float, default=0.5)
    parser.add_argument("--metapath_rank_metric", default="count", choices=["count", "degree_norm", "count_idf"])
    parser.add_argument("--metapath_keep_self", action="store_true")
    parser.add_argument("--candidates_per_metapath", type=int, default=6)
    parser.add_argument("--max_metapath_rows", type=int, default=9)
    parser.add_argument("--max_endpoints_per_row", type=int, default=6)
    parser.add_argument("--max_draw_edges", type=int, default=360)
    parser.add_argument("--max_semantic_edges", type=int, default=180)
    parser.add_argument("--peprompt_offline_cache_dir", default="artifacts/cache/peprompt_offline_splits")
    parser.add_argument("--out_dir", default="artifacts/analysis/freebase_peprompt_case")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show_labels", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    _visualize(build_parser().parse_args())


if __name__ == "__main__":
    main()
