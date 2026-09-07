from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpbench.downstream.fewshot import load_peprompt_offline_splits  # noqa: E402
from scripts.peprompt_benchmark import _load_raw_heterograph  # noqa: E402


def _format_float_for_key(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


LABEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
TYPE_COLORS = {
    "author": "#f28e2b",
    "paper": "#4e79a7",
    "term": "#59a14f",
    "venue": "#b07aa1",
}
EDGE_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
]


def _cache_subgraph_type(args: argparse.Namespace) -> str:
    if args.subgraph_type not in {"metapath_topk", "metapath_topk_path", "metapath_topk_adapt", "metapath_topk_path_adapt"}:
        return args.subgraph_type
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
    endpoint_mode = str(getattr(args, "metapath_endpoint_mode", "all"))
    support_mode = _resolve_metapath_support_mode_for_cache(
        subgraph_type=str(args.subgraph_type),
        endpoint_mode=endpoint_mode,
        support_mode=str(getattr(args, "metapath_support_mode", "auto")),
    )
    support_topk = int(getattr(args, "metapath_support_topk", 0) or 0)
    if endpoint_mode != "all":
        suffix += f"_{endpoint_mode}"
        suffix += f"_support{support_mode}"
    elif str(getattr(args, "metapath_support_mode", "auto")) != "auto":
        suffix += f"_support{support_mode}"
    if support_topk > 0:
        suffix += f"_sk{support_topk}"
    return f"{args.subgraph_type}_{suffix}"


def _resolve_metapath_support_mode_for_cache(subgraph_type: str, endpoint_mode: str, support_mode: str) -> str:
    support_mode = str(support_mode)
    if support_mode != "auto":
        return support_mode
    if str(endpoint_mode) == "target_closed":
        return "count"
    if str(subgraph_type) in {"metapath_topk_path", "metapath_topk_path_adapt"}:
        return "one_path"
    return "none"


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _class_id(label: Any) -> int:
    y = torch.as_tensor(label).detach().cpu()
    if y.ndim == 0:
        return int(y.item())
    if y.numel() == 1:
        return int(y.view(-1)[0].item())
    return int(torch.argmax(y.view(-1)).item())


def _sample_parts(sample: Any):
    if isinstance(sample, dict):
        return sample["graph"], sample.get("inverse_indices", sample.get("inverse_index")), sample.get("label")
    if isinstance(sample, (tuple, list)) and len(sample) >= 3:
        return sample[0], sample[1], sample[2]
    raise ValueError(f"Unsupported sample format: {type(sample)!r}")


def _target_local_id(inverse_indices: Any, targetnode: str) -> int:
    if isinstance(inverse_indices, dict):
        value = inverse_indices[targetnode]
    else:
        value = inverse_indices
    value = torch.as_tensor(value).view(-1)
    return int(value[0].item())


def _target_labels(graph, targetnode: str) -> np.ndarray | None:
    if "y" not in graph.nodes[targetnode].data:
        return None
    labels = graph.nodes[targetnode].data["y"].detach().cpu()
    if labels.ndim == 2:
        labels = labels.argmax(dim=-1)
    return labels.long().view(-1).numpy()


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[1] == 1:
        return np.column_stack([x[:, 0], np.zeros((x.shape[0],), dtype=np.float32)])
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_one_sample(sample: Any, split_seed: int, sample_idx: int, targetnode: str, num_classes: int) -> dict[str, Any]:
    graph, inverse_indices, label = _sample_parts(sample)
    center_local = _target_local_id(inverse_indices, targetnode)
    target_ids = graph.nodes[targetnode].data["_ID"].detach().cpu().long().view(-1).numpy()
    target_y = _target_labels(graph, targetnode)

    row: dict[str, Any] = {
        "split_seed": int(split_seed),
        "sample_idx": int(sample_idx),
        "center_global_id": int(target_ids[center_local]),
        "center_label": _class_id(label),
    }
    for ntype in graph.ntypes:
        row[f"nodes_{ntype}"] = int(graph.num_nodes(ntype))
    for etype in graph.canonical_etypes:
        row[f"edges_{etype[0]}__{etype[1]}__{etype[2]}"] = int(graph.num_edges(etype=etype))

    for cid in range(num_classes):
        row[f"neighbor_target_label_c{cid}"] = 0
    if target_y is not None:
        for local_id, cid in enumerate(target_y.tolist()):
            if local_id == center_local:
                continue
            if 0 <= int(cid) < num_classes:
                row[f"neighbor_target_label_c{int(cid)}"] += 1
    return row


def _make_union_graph(samples: list[Any], split_seed: int, targetnode: str, num_classes: int, max_train: int) -> nx.Graph:
    out = nx.Graph()
    for sample_idx, sample in enumerate(samples[:max_train]):
        graph, inverse_indices, label = _sample_parts(sample)
        center_local = _target_local_id(inverse_indices, targetnode)
        center_label = _class_id(label)

        local_to_key: dict[tuple[str, int], str] = {}
        for ntype in graph.ntypes:
            global_ids = graph.nodes[ntype].data["_ID"].detach().cpu().long().view(-1).tolist()
            labels = _target_labels(graph, targetnode) if ntype == targetnode else None
            for local_id, global_id in enumerate(global_ids):
                key = f"{ntype}:{int(global_id)}"
                local_to_key[(ntype, local_id)] = key
                is_center = ntype == targetnode and local_id == center_local
                node_label = int(labels[local_id]) if labels is not None else -1
                if key not in out:
                    out.add_node(
                        key,
                        ntype=ntype,
                        global_id=int(global_id),
                        label=node_label,
                        is_train_center=False,
                        center_count=0,
                        split_seed=int(split_seed),
                    )
                if is_center:
                    out.nodes[key]["is_train_center"] = True
                    out.nodes[key]["center_count"] += 1
                    out.nodes[key]["label"] = center_label

        for etype in graph.canonical_etypes:
            src_t, rel_t, dst_t = etype
            src, dst = graph.edges(etype=etype)
            for s, d in zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()):
                sk = local_to_key[(src_t, int(s))]
                dk = local_to_key[(dst_t, int(d))]
                if out.has_edge(sk, dk):
                    out.edges[sk, dk]["weight"] += 1
                else:
                    out.add_edge(sk, dk, etype=f"{src_t}->{dst_t}", rel=rel_t, weight=1)
    return out


def _sample_to_multidigraph(sample: Any, targetnode: str):
    graph, inverse_indices, label = _sample_parts(sample)
    center_local = _target_local_id(inverse_indices, targetnode)
    center_label = _class_id(label)

    out = nx.MultiDiGraph()
    local_to_key: dict[tuple[str, int], str] = {}
    for ntype in graph.ntypes:
        global_ids = graph.nodes[ntype].data["_ID"].detach().cpu().long().view(-1).tolist()
        labels = _target_labels(graph, targetnode) if ntype == targetnode else None
        for local_id, global_id in enumerate(global_ids):
            key = f"{ntype}:{int(global_id)}"
            local_to_key[(ntype, local_id)] = key
            is_center = ntype == targetnode and local_id == center_local
            node_label = int(labels[local_id]) if labels is not None else -1
            if is_center:
                node_label = center_label
            out.add_node(
                key,
                ntype=ntype,
                global_id=int(global_id),
                label=node_label,
                is_center=is_center,
            )

    for etype in graph.canonical_etypes:
        src_t, rel_t, dst_t = etype
        src, dst = graph.edges(etype=etype)
        etype_key = f"{src_t}->{dst_t}"
        for edge_idx, (s, d) in enumerate(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist())):
            sk = local_to_key[(src_t, int(s))]
            dk = local_to_key[(dst_t, int(d))]
            out.add_edge(sk, dk, key=f"{etype_key}:{edge_idx}", etype=etype_key, rel=rel_t)
    center_key = local_to_key[(targetnode, center_local)]
    return out, center_key, center_label


def _draw_curved_edges(ax, graph: nx.MultiDiGraph, pos: dict[str, np.ndarray]) -> None:
    etypes = sorted({data.get("etype", "") for _, _, data in graph.edges(data=True)})
    etype_to_color = {etype: EDGE_COLORS[i % len(EDGE_COLORS)] for i, etype in enumerate(etypes)}
    pair_seen: Counter[tuple[str, str]] = Counter()
    for src, dst, data in graph.edges(data=True):
        pair = (src, dst)
        pair_seen[pair] += 1
        reverse_count = pair_seen[(dst, src)]
        same_count = pair_seen[pair]
        rad = 0.08 * (same_count - 1)
        if reverse_count > 0:
            rad = 0.14
        if src == dst:
            rad = 0.28
        color = etype_to_color.get(data.get("etype", ""), "#7f7f7f")
        patch = FancyArrowPatch(
            posA=pos[src],
            posB=pos[dst],
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            alpha=0.72,
            shrinkA=9,
            shrinkB=9,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)


def _short_node_label(node: str, data: dict[str, Any]) -> str:
    ntype = str(data.get("ntype", ""))
    prefix = ntype[:1] if ntype else "n"
    label = int(data.get("label", -1))
    suffix = f"/c{label}" if ntype == "author" and label >= 0 else ""
    return f"{prefix}{int(data.get('global_id', 0))}{suffix}"


def _draw_individual_subgraph(
    sample: Any,
    split_seed: int,
    sample_idx: int,
    targetnode: str,
    out_dir: Path,
) -> None:
    graph, center_key, center_label = _sample_to_multidigraph(sample, targetnode)
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes())
    simple.add_edges_from((u, v) for u, v in graph.edges())
    pos = nx.spring_layout(simple, seed=split_seed * 1000 + sample_idx, k=0.8, iterations=120)

    node_colors = []
    node_sizes = []
    edgecolors = []
    linewidths = []
    for node, data in graph.nodes(data=True):
        ntype = str(data.get("ntype", ""))
        label = int(data.get("label", -1))
        is_center = node == center_key
        if ntype == "author" and label >= 0:
            node_colors.append(LABEL_COLORS[label % len(LABEL_COLORS)])
        else:
            node_colors.append(TYPE_COLORS.get(ntype, "#bab0ac"))
        node_sizes.append(920 if is_center else 430)
        edgecolors.append("black" if is_center else "#ffffff")
        linewidths.append(2.2 if is_center else 0.8)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    _draw_curved_edges(ax, graph, pos)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=0.94,
    )
    labels = {node: _short_node_label(node, data) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)

    etypes = sorted({data.get("etype", "") for _, _, data in graph.edges(data=True)})
    handles = [
        FancyArrowPatch((0, 0), (0.25, 0), arrowstyle="-|>", color=EDGE_COLORS[i % len(EDGE_COLORS)], mutation_scale=9)
        for i, _ in enumerate(etypes)
    ]
    if handles:
        ax.legend(handles, etypes, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    ax.set_title(
        f"split {split_seed} train idx {sample_idx} center {center_key} class {center_label}\n"
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} directed edges"
    )
    ax.axis("off")
    fig.tight_layout()
    split_dir = out_dir / "individual_subgraphs" / f"split{split_seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    center_id = center_key.split(":", 1)[1]
    fig.savefig(split_dir / f"idx{sample_idx:03d}_node{center_id}_c{center_label}.png", dpi=220)
    plt.close(fig)


def _plot_global_target_pca(args: argparse.Namespace, payloads: dict[int, dict[str, Any]], out_dir: Path) -> None:
    graph, targetnode = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    x = graph.nodes[targetnode].data["x"].detach().cpu().float().numpy()
    y = graph.nodes[targetnode].data["y"].detach().cpu()
    if y.ndim == 2:
        y = y.argmax(dim=-1)
    y_np = y.long().view(-1).numpy()

    rng = np.random.default_rng(args.seed)
    valid_idx = np.flatnonzero(y_np >= 0)
    if args.max_background_nodes > 0 and len(valid_idx) > args.max_background_nodes:
        valid_idx = rng.choice(valid_idx, size=args.max_background_nodes, replace=False)
    coords = _pca_2d(x)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for cid in sorted(set(y_np[valid_idx].tolist())):
        ids = valid_idx[y_np[valid_idx] == cid]
        ax.scatter(
            coords[ids, 0],
            coords[ids, 1],
            s=8,
            c=LABEL_COLORS[int(cid) % len(LABEL_COLORS)],
            alpha=0.18,
            linewidths=0,
            label=f"class {cid}",
        )

    markers = ["X", "o", "s", "P", "^"]
    for i, split_seed in enumerate(args.splits):
        payload = payloads[split_seed]
        train_ids = _as_numpy(payload["train_ids"]).astype(np.int64)
        train_labels = _as_numpy(payload["train_labels"]).astype(np.int64)
        ax.scatter(
            coords[train_ids, 0],
            coords[train_ids, 1],
            s=90,
            c=[LABEL_COLORS[int(c) % len(LABEL_COLORS)] for c in train_labels],
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.9,
            label=f"split {split_seed} train",
        )

    ax.set_title(f"{args.dataset} {targetnode} PCA with selected train nodes")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "target_pca_train_splits.png", dpi=220)
    plt.close(fig)


def _plot_stacked_counts(summary_rows: list[dict[str, Any]], out_dir: Path, prefix: str, title: str, output_name: str) -> None:
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    keys = sorted({key for row in summary_rows for key in row if key.startswith(prefix)})
    for row in summary_rows:
        split_seed = int(row["split_seed"])
        for key in keys:
            grouped[split_seed][key.removeprefix(prefix)] += int(row.get(key, 0) or 0)

    split_ids = sorted(grouped)
    bottom = np.zeros((len(split_ids),), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, key in enumerate(keys):
        label = key.removeprefix(prefix)
        values = np.array([grouped[s][label] for s in split_ids], dtype=np.float64)
        ax.bar([str(s) for s in split_ids], values, bottom=bottom, label=label)
        bottom += values
    ax.set_title(title)
    ax.set_xlabel("split_seed")
    ax.set_ylabel("count")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / output_name, dpi=220)
    plt.close(fig)


def _draw_union_graph(graph: nx.Graph, split_seed: int, out_dir: Path, max_nodes: int) -> None:
    if graph.number_of_nodes() == 0:
        return

    if graph.number_of_nodes() > max_nodes:
        centers = [n for n, d in graph.nodes(data=True) if d.get("is_train_center")]
        keep = set(centers)
        for center in centers:
            keep.update(graph.neighbors(center))
        if len(keep) < max_nodes:
            for node, degree in sorted(graph.degree, key=lambda item: item[1], reverse=True):
                keep.add(node)
                if len(keep) >= max_nodes:
                    break
        graph = graph.subgraph(list(keep)[:max_nodes]).copy()

    pos = nx.spring_layout(graph, seed=split_seed, k=0.38, iterations=80)
    node_colors = []
    node_sizes = []
    edgecolors = []
    linewidths = []
    for _, data in graph.nodes(data=True):
        ntype = data.get("ntype", "")
        label = int(data.get("label", -1))
        is_center = bool(data.get("is_train_center", False))
        if ntype == "author" and label >= 0:
            node_colors.append(LABEL_COLORS[label % len(LABEL_COLORS)])
        else:
            node_colors.append(TYPE_COLORS.get(ntype, "#bab0ac"))
        node_sizes.append(150 if not is_center else 520)
        edgecolors.append("black" if is_center else "white")
        linewidths.append(1.6 if is_center else 0.4)

    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.18, width=0.6, edge_color="#7f7f7f")
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=0.88,
    )

    center_labels = {
        node: f"c{int(data.get('label', -1))}"
        for node, data in graph.nodes(data=True)
        if data.get("is_train_center")
    }
    nx.draw_networkx_labels(graph, pos, labels=center_labels, font_size=7, ax=ax)
    ax.set_title(
        f"split {split_seed} train-neighborhood union "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"split{split_seed}_train_neighborhood_union.png", dpi=220)
    plt.close(fig)


def visualize(args: argparse.Namespace) -> None:
    cache_key = _cache_subgraph_type(args)
    out_dir = Path(args.out_dir) / f"{args.dataset}_{cache_key}_splits_{'_'.join(str(s) for s in args.splits)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        split_seed: load_peprompt_offline_splits(
            cache_dir=args.peprompt_offline_cache_dir,
            dataset_name=args.dataset,
            shot=args.shot,
            seed=int(split_seed),
            feats_type=args.feats_type,
            subgraph_type=cache_key,
        )
        for split_seed in args.splits
    }
    targetnode = str(next(iter(payloads.values()))["targetnode"])
    num_classes = int(next(iter(payloads.values())).get("num_classes", 0) or 0)

    summary_rows: list[dict[str, Any]] = []
    for split_seed, payload in payloads.items():
        samples = list(payload["train"])
        for sample_idx, sample in enumerate(samples):
            summary_rows.append(_summarize_one_sample(sample, split_seed, sample_idx, targetnode, num_classes))
            if args.draw_individual and sample_idx < args.max_individual_per_split:
                _draw_individual_subgraph(sample, split_seed, sample_idx, targetnode, out_dir)
        if args.draw_union:
            union_graph = _make_union_graph(samples, split_seed, targetnode, num_classes, args.max_train_per_split)
            _draw_union_graph(union_graph, split_seed, out_dir, args.max_union_nodes)

    _write_csv(out_dir / "train_neighborhood_summary.csv", summary_rows)
    if not args.only_individual:
        _plot_global_target_pca(args, payloads, out_dir)
        _plot_stacked_counts(
            summary_rows,
            out_dir,
            prefix="nodes_",
            title=f"{args.dataset} train subgraph node-type distribution",
            output_name="train_node_type_distribution.png",
        )
        _plot_stacked_counts(
            summary_rows,
            out_dir,
            prefix="neighbor_target_label_",
            title=f"{args.dataset} neighbor {targetnode} label distribution excluding centers",
            output_name="train_neighbor_target_label_distribution.png",
        )
        _plot_stacked_counts(
            summary_rows,
            out_dir,
            prefix="edges_",
            title=f"{args.dataset} train subgraph edge-type distribution",
            output_name="train_edge_type_distribution.png",
        )

    print(f"cache_key={cache_key}")
    print(f"targetnode={targetnode} num_classes={num_classes}")
    print(f"wrote {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize training-node neighborhood distributions for few-shot splits.")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--splits", type=int, nargs="+", default=[2, 0])
    parser.add_argument("--feats_type", type=int, default=0)
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--subgraph_type", default="metapath_topk")
    parser.add_argument("--metapath_max_hop", type=int, default=3)
    parser.add_argument("--metapath_topk", type=int, default=2)
    parser.add_argument("--metapath_min_topk", type=int, default=1)
    parser.add_argument("--metapath_max_topk", type=int, default=5)
    parser.add_argument("--metapath_rel_threshold", type=float, default=0.5)
    parser.add_argument("--metapath_rank_metric", default="count")
    parser.add_argument("--metapath_keep_self", action="store_true")
    parser.add_argument("--metapath_endpoint_mode", default="all", choices=["all", "target_closed"])
    parser.add_argument("--metapath_support_mode", default="auto", choices=["auto", "none", "one_path", "count"])
    parser.add_argument("--metapath_support_topk", type=int, default=0)
    parser.add_argument("--peprompt_offline_cache_dir", default="artifacts/cache/peprompt_offline_splits")
    parser.add_argument("--out_dir", default="artifacts/analysis/split_neighborhood_visualization")
    parser.add_argument("--max_train_per_split", type=int, default=40)
    parser.add_argument("--max_union_nodes", type=int, default=260)
    parser.add_argument("--max_individual_per_split", type=int, default=40)
    parser.add_argument("--max_background_nodes", type=int, default=8000)
    parser.add_argument("--draw_individual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw_union", action="store_true")
    parser.add_argument("--only_individual", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    visualize(build_parser().parse_args())


if __name__ == "__main__":
    main()
