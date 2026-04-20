
# scripts/protocol_benchmark_v2.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import argparse
import csv
import json
import pickle as pk
import time
import warnings
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import HGBDataset
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_dgl

from gpbench.protocol_bridge import downstream_legacy as legacy_bridge
from gpbench.protocol_bridge.downstream_legacy import (
    build_legacy_fewshot_embeddings,
    train_hgmp_heteroprompt_probe,
    train_mlp_probe,
    train_typepair_prompt_probe,
)
from gpbench.downstream.fewshot import load_split_file
from gpbench.utils.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_nested_summary,
    log_metrics,
    log_run_record,
    log_table,
    maybe_configure_wandb_env,
    upload_dir_artifact,
    upload_file_artifact,
)
from protocols.hgmp.utils_legacy import create_matrix, seed_everything
from protocols.hgprompt.runner import _run_once as hgprompt_run_once
from protocols.hgprompt.adapter import load_hgprompt_downstream_bundle


HOP_NUM = {
    "ACM": 1,
    "DBLP": 2,
    "IMDB": 2,
    "Freebase": 1,
}

TARGET_NODETYPE = {
    "ACM": "paper",
    "DBLP": "author",
    "IMDB": "movie",
    "Freebase": "book",
}

TYPEPAIR_EDGE_FEATURE_NAME = "typepair_edge_feat"
TYPEPAIR_EDGE_FEATURES = [
    "NodeTypeEncoding",
    "EdgeTypeOneHot",
    "DegreeDiff",
    "NeighborTypeOverlap",
    "NodeAttrCosSim",
    "NeighborAttrVariance",
    "PageRankDiff",
    "CommunityLabelDiff",
    "SpectralEmbeddingDiff",
]


@dataclass
class RunRecord:
    method: str
    split_seed: int
    repeat_id: int
    run_seed: int
    ckpt_path: str
    test_micro: float
    test_macro: float
    best_epoch: int


@dataclass
class SeedAggregate:
    method: str
    split_seed: int
    count: int
    micro_mean: float
    micro_std: float
    macro_mean: float
    macro_std: float


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_global_seed(seed: int):
    seed_everything(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _apply_feats_type(data, feats_type: int):
    features = data.x_dict
    features_list = []
    for value in features.values():
        features_list.append(value)

    if feats_type in (0, 6, 7, -1, 8, 9):
        pass
    elif feats_type in (1, 5):
        save = 0 if feats_type == 1 else 2
        for i in range(len(features_list)):
            if i != save:
                features_list[i] = torch.zeros((features_list[i].shape[0], 10))
    elif feats_type in (2, 4):
        save = feats_type - 2
        for i in range(len(features_list)):
            if i == save:
                continue
            dim = features_list[i].shape[0]
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            idx = torch.LongTensor(idx)
            val = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(idx, val, torch.Size([dim, dim])).to_dense()
    elif feats_type == 3:
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            idx = torch.LongTensor(idx)
            val = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(idx, val, torch.Size([dim, dim])).to_dense()
    else:
        raise ValueError(f"Unsupported feats_type={feats_type}")

    value_dict = {}
    for i, ntype in enumerate(data.node_types):
        value_dict[ntype] = features_list[i]
    data.set_value_dict("x", value_dict)
    return data


def _load_raw_heterograph(root: str, dataset: str, feats_type: int):
    if dataset == "Freebase" and feats_type == -1:
        dataset_obj = HGBDataset(root=root, name=dataset, transform=ToUndirected(merge=False))
    else:
        dataset_obj = HGBDataset(root=root, name=dataset)
    data = dataset_obj[0]

    for node_type, node_store in data.node_items():
        for attr, value in list(node_store.items()):
            if attr == "num_nodes":
                data[node_type]["x"] = create_matrix(value, 0.01)
                del data[node_type][attr]

    if dataset == "IMDB":
        targetnode = TARGET_NODETYPE[dataset]
        oldy = data[targetnode]["y"]
        newy = []
        for arr in oldy:
            one_indices = np.where(arr.cpu().numpy() == 1)[0]
            if one_indices.size > 0:
                newy.append(int(np.random.choice(one_indices)))
            else:
                newy.append(-1)
        data[targetnode]["oldy"] = oldy
        data[targetnode]["y"] = torch.tensor(newy)

    data = _apply_feats_type(data, feats_type)
    graph = to_dgl(data)
    return graph, TARGET_NODETYPE[dataset]


def _typepair_edge_feature_cache_path(args) -> Path:
    cache_dir = Path(getattr(args, "typepair_edge_feature_cache_dir", ROOT / "artifacts" / "cache" / "typepair_edge_features"))
    spectral_dim = int(getattr(args, "typepair_spectral_dim", 8))
    attr_cap = int(getattr(args, "typepair_edge_attr_dim_cap", 256))
    pagerank_damping = str(getattr(args, "typepair_pagerank_damping", 0.85)).replace(".", "p")
    pagerank_iter = int(getattr(args, "typepair_pagerank_max_iter", 50))
    return (
        cache_dir
        / args.dataset
        / f"ft{args.feats_type}"
        / f"attr{attr_cap}.spec{spectral_dim}.pr{pagerank_damping}.pit{pagerank_iter}.pt"
    )


def _edge_type_key(etype) -> str:
    return "__".join(str(part) for part in etype)


def _align_node_attrs(graph, attr_dim_cap: int) -> dict[str, torch.Tensor]:
    raw_attrs = {}
    max_dim = 1
    for ntype in graph.ntypes:
        x = graph.ndata["x"][ntype].detach().cpu()
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        raw_attrs[ntype] = x
        max_dim = max(max_dim, int(x.size(1)))

    width = max_dim if attr_dim_cap <= 0 else min(max_dim, int(attr_dim_cap))
    aligned = {}
    for ntype, x in raw_attrs.items():
        out = torch.zeros((x.size(0), width), dtype=torch.float32)
        take = min(width, x.size(1))
        if take > 0:
            out[:, :take] = x[:, :take]
        aligned[ntype] = out
    return aligned


def _collect_global_edges(graph, offsets: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    src_all = []
    dst_all = []
    for src_t, rel_t, dst_t in graph.canonical_etypes:
        src, dst = graph.edges(etype=(src_t, rel_t, dst_t))
        src_all.append(src.detach().cpu().long() + offsets[src_t])
        dst_all.append(dst.detach().cpu().long() + offsets[dst_t])
    if not src_all:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(src_all), torch.cat(dst_all)


def _pagerank_power_iteration(
    num_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    damping: float,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.float32)
    if src.numel() == 0:
        return torch.full((num_nodes,), 1.0 / max(num_nodes, 1), dtype=torch.float32)

    src = src.long()
    dst = dst.long()
    out_degree = torch.bincount(src, minlength=num_nodes).float()
    rank = torch.full((num_nodes,), 1.0 / num_nodes, dtype=torch.float32)
    teleport = (1.0 - damping) / num_nodes

    for _ in range(max_iter):
        new_rank = torch.full_like(rank, teleport)
        valid = out_degree[src] > 0
        contrib = rank[src[valid]] / out_degree[src[valid]]
        new_rank.index_add_(0, dst[valid], damping * contrib)

        dangling_mass = rank[out_degree == 0].sum()
        if dangling_mass > 0:
            new_rank += damping * dangling_mass / num_nodes

        if torch.norm(new_rank - rank, p=1).item() < tol:
            rank = new_rank
            break
        rank = new_rank
    return rank


def _community_labels(num_nodes: int, src: torch.Tensor, dst: torch.Tensor, seed: int) -> torch.Tensor:
    labels = torch.zeros(num_nodes, dtype=torch.long)
    if num_nodes == 0 or src.numel() == 0:
        return labels

    try:
        import networkx as nx
    except ImportError:
        warnings.warn("networkx is not available; CommunityLabelDiff falls back to a single community.")
        return labels

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(zip(src.tolist(), dst.tolist()))

    try:
        communities = nx.community.louvain_communities(nx_graph, seed=seed)
    except Exception as exc:
        warnings.warn(f"networkx louvain failed ({exc}); falling back to connected components.")
        communities = list(nx.connected_components(nx_graph))

    for label, nodes in enumerate(communities):
        labels[list(nodes)] = int(label)
    return labels


def _spectral_embeddings(
    num_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    dim: int,
    max_nodes: int,
) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    if num_nodes == 0:
        return torch.zeros((0, dim), dtype=torch.float32)
    if num_nodes > max_nodes:
        warnings.warn(
            f"Skip spectral embeddings for {num_nodes} nodes; exceeds max_nodes={max_nodes}."
        )
        return torch.zeros((num_nodes, dim), dtype=torch.float32)

    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import laplacian
        from scipy.sparse.linalg import eigsh
    except ImportError:
        warnings.warn("scipy is not available; SpectralEmbeddingDiff falls back to zeros.")
        return torch.zeros((num_nodes, dim), dtype=torch.float32)

    try:
        row = torch.cat([src, dst]).numpy()
        col = torch.cat([dst, src]).numpy()
        data = np.ones(row.shape[0], dtype=np.float32)
        adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        norm_lap = laplacian(adj, normed=True)

        k = min(dim + 1, max(1, num_nodes - 1))
        if num_nodes <= k + 1:
            values, vectors = np.linalg.eigh(norm_lap.toarray())
            order = np.argsort(values)
            vectors = vectors[:, order]
        else:
            _, vectors = eigsh(norm_lap, k=k, which="SM", tol=1e-3)

        embedding = vectors[:, 1 : dim + 1]
        if embedding.shape[1] < dim:
            pad = np.zeros((num_nodes, dim - embedding.shape[1]), dtype=np.float32)
            embedding = np.concatenate([embedding, pad], axis=1)
        return torch.from_numpy(np.asarray(embedding, dtype=np.float32))
    except Exception as exc:
        warnings.warn(f"Spectral embedding failed ({exc}); falling back to zeros.")
        return torch.zeros((num_nodes, dim), dtype=torch.float32)


def _feature_slices(num_node_types: int, num_edge_types: int, spectral_dim: int) -> dict[str, tuple[int, int]]:
    cursor = 0
    slices = {}

    def add(name: str, width: int):
        nonlocal cursor
        slices[name] = (cursor, cursor + width)
        cursor += width

    add("NodeTypeEncoding", 3 * num_node_types)
    add("EdgeTypeOneHot", num_edge_types)
    add("DegreeDiff", 1)
    add("NeighborTypeOverlap", 1)
    add("NodeAttrCosSim", 1)
    add("NeighborAttrVariance", 1)
    add("PageRankDiff", 1)
    add("CommunityLabelDiff", 1)
    add("SpectralEmbeddingDiff", spectral_dim)
    return slices


def _coerce_feature_name_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_selected_typepair_edge_feature_names(args, feature_slices: dict[str, tuple[int, int]] | None = None) -> list[str]:
    available = list(feature_slices.keys()) if feature_slices is not None else list(TYPEPAIR_EDGE_FEATURES)
    available_set = set(available)

    configured = _coerce_feature_name_list(getattr(args, "typepair_edge_feature_names", None))
    if configured is not None:
        unknown = [name for name in configured if name not in available_set]
        if unknown:
            raise ValueError(f"Unknown typepair edge feature names: {unknown}. Available: {available}")
        return configured

    selected = []
    saw_bool_flag = False
    for name in available:
        flag_value = None
        for attr in (f"use_{name}", f"edge_feat_{name}", f"typepair_use_{name}"):
            if hasattr(args, attr):
                flag_value = getattr(args, attr)
                break
        if flag_value is None:
            continue
        saw_bool_flag = True
        if _coerce_bool(flag_value):
            selected.append(name)

    if saw_bool_flag:
        return selected
    return available


def _summarize_edge_features(edge_feature_table: dict, feature_slices: dict[str, tuple[int, int]]) -> dict:
    matrices = [value.float().cpu() for value in edge_feature_table.values()]
    if not matrices:
        return {}
    matrix = torch.cat(matrices, dim=0)
    if matrix.numel() == 0:
        return {"all": {"mean": 0.0, "var": 0.0, "max": 0.0, "min": 0.0}}

    stats = {}
    for name, (start, end) in feature_slices.items():
        if end <= start:
            continue
        values = matrix[:, start:end].reshape(-1)
        stats[name] = {
            "mean": float(values.mean().item()),
            "var": float(values.var(unbiased=False).item()),
            "max": float(values.max().item()),
            "min": float(values.min().item()),
        }
    stats["all"] = {
        "mean": float(matrix.mean().item()),
        "var": float(matrix.var(unbiased=False).item()),
        "max": float(matrix.max().item()),
        "min": float(matrix.min().item()),
    }
    return stats


def _subset_typepair_edge_feature_payload(payload: dict, args) -> dict:
    selected_names = get_selected_typepair_edge_feature_names(args, payload["feature_slices"])
    full_slices = payload["feature_slices"]

    index_parts = []
    selected_slices = {}
    cursor = 0
    for name in selected_names:
        start, end = full_slices[name]
        width = end - start
        if width <= 0:
            continue
        index_parts.append(torch.arange(start, end, dtype=torch.long))
        selected_slices[name] = (cursor, cursor + width)
        cursor += width

    if index_parts:
        indices = torch.cat(index_parts)
    else:
        indices = torch.empty(0, dtype=torch.long)

    selected_table = {}
    for etype, table in payload["edge_feature_table"].items():
        if indices.numel() == 0:
            selected_table[etype] = table.new_zeros((table.size(0), 0))
        else:
            selected_table[etype] = table[:, indices].contiguous()

    out = dict(payload)
    out["edge_feature_table"] = selected_table
    out["feature_dim"] = int(indices.numel())
    out["feature_slices"] = selected_slices
    out["feature_stats"] = _summarize_edge_features(selected_table, selected_slices)
    out["selected_feature_names"] = selected_names
    out["selected_feature_count"] = int(len(selected_names))
    out["full_feature_dim"] = int(payload.get("feature_dim", 0))
    return out


def _compute_typepair_edge_feature_table(graph, args) -> dict:
    start_time = time.perf_counter()
    ntypes = list(graph.ntypes)
    etypes = list(graph.canonical_etypes)
    node_type_to_idx = {ntype: i for i, ntype in enumerate(ntypes)}
    edge_type_to_idx = {etype: i for i, etype in enumerate(etypes)}

    offsets = {}
    total_nodes = 0
    for ntype in ntypes:
        offsets[ntype] = total_nodes
        total_nodes += graph.num_nodes(ntype)

    aligned_attrs = _align_node_attrs(graph, int(getattr(args, "typepair_edge_attr_dim_cap", 256)))
    attr_width = next(iter(aligned_attrs.values())).size(1) if aligned_attrs else 0

    degree = {ntype: torch.zeros(graph.num_nodes(ntype), dtype=torch.float32) for ntype in ntypes}
    neighbor_type_hist = {
        ntype: torch.zeros((graph.num_nodes(ntype), len(ntypes)), dtype=torch.float32)
        for ntype in ntypes
    }
    neigh_sum = {
        ntype: torch.zeros((graph.num_nodes(ntype), attr_width), dtype=torch.float32)
        for ntype in ntypes
    }
    neigh_sq_sum = {
        ntype: torch.zeros((graph.num_nodes(ntype), attr_width), dtype=torch.float32)
        for ntype in ntypes
    }
    neigh_count = {
        ntype: torch.zeros((graph.num_nodes(ntype), 1), dtype=torch.float32)
        for ntype in ntypes
    }

    for src_t, rel_t, dst_t in etypes:
        src, dst = graph.edges(etype=(src_t, rel_t, dst_t))
        src = src.detach().cpu().long()
        dst = dst.detach().cpu().long()
        if src.numel() == 0:
            continue

        ones = torch.ones(src.numel(), dtype=torch.float32)
        degree[src_t].index_add_(0, src, ones)
        degree[dst_t].index_add_(0, dst, ones)

        dst_type_msg = torch.zeros((src.numel(), len(ntypes)), dtype=torch.float32)
        dst_type_msg[:, node_type_to_idx[dst_t]] = 1.0
        src_type_msg = torch.zeros((dst.numel(), len(ntypes)), dtype=torch.float32)
        src_type_msg[:, node_type_to_idx[src_t]] = 1.0
        neighbor_type_hist[src_t].index_add_(0, src, dst_type_msg)
        neighbor_type_hist[dst_t].index_add_(0, dst, src_type_msg)

        dst_attr = aligned_attrs[dst_t][dst]
        src_attr = aligned_attrs[src_t][src]
        neigh_sum[src_t].index_add_(0, src, dst_attr)
        neigh_sq_sum[src_t].index_add_(0, src, dst_attr * dst_attr)
        neigh_count[src_t].index_add_(0, src, torch.ones((src.numel(), 1), dtype=torch.float32))
        neigh_sum[dst_t].index_add_(0, dst, src_attr)
        neigh_sq_sum[dst_t].index_add_(0, dst, src_attr * src_attr)
        neigh_count[dst_t].index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=torch.float32))

    neighbor_attr_var = {}
    for ntype in ntypes:
        count = neigh_count[ntype].clamp_min(1.0)
        mean = neigh_sum[ntype] / count
        var = (neigh_sq_sum[ntype] / count - mean * mean).clamp_min(0.0)
        neighbor_attr_var[ntype] = var.mean(dim=1)

    global_src, global_dst = _collect_global_edges(graph, offsets)
    pagerank = _pagerank_power_iteration(
        total_nodes,
        global_src,
        global_dst,
        damping=float(getattr(args, "typepair_pagerank_damping", 0.85)),
        max_iter=int(getattr(args, "typepair_pagerank_max_iter", 50)),
        tol=float(getattr(args, "typepair_pagerank_tol", 1e-6)),
    )
    community = _community_labels(
        total_nodes,
        global_src,
        global_dst,
        seed=int(getattr(args, "split_seed", 0)),
    )
    spectral_dim = int(getattr(args, "typepair_spectral_dim", 8))
    spectral = _spectral_embeddings(
        total_nodes,
        global_src,
        global_dst,
        dim=spectral_dim,
        max_nodes=int(getattr(args, "typepair_spectral_max_nodes", 50000)),
    )

    node_eye = torch.eye(len(ntypes), dtype=torch.float32)
    edge_eye = torch.eye(len(etypes), dtype=torch.float32)
    feature_slices = _feature_slices(len(ntypes), len(etypes), spectral_dim)
    edge_feature_table = {}

    for src_t, rel_t, dst_t in etypes:
        etype = (src_t, rel_t, dst_t)
        src, dst = graph.edges(etype=etype)
        src = src.detach().cpu().long()
        dst = dst.detach().cpu().long()
        edge_count = src.numel()

        src_type = node_eye[node_type_to_idx[src_t]].unsqueeze(0).expand(edge_count, -1)
        dst_type = node_eye[node_type_to_idx[dst_t]].unsqueeze(0).expand(edge_count, -1)
        node_type_encoding = torch.cat([src_type, dst_type, src_type - dst_type], dim=1)
        edge_type_onehot = edge_eye[edge_type_to_idx[etype]].unsqueeze(0).expand(edge_count, -1)

        degree_diff = (degree[src_t][src] - degree[dst_t][dst]).unsqueeze(1)

        src_hist = neighbor_type_hist[src_t][src]
        dst_hist = neighbor_type_hist[dst_t][dst]
        overlap_num = torch.minimum(src_hist, dst_hist).sum(dim=1)
        overlap_den = torch.maximum(src_hist, dst_hist).sum(dim=1).clamp_min(1.0)
        neighbor_type_overlap = (overlap_num / overlap_den).unsqueeze(1)

        node_attr_cos = F.cosine_similarity(
            aligned_attrs[src_t][src],
            aligned_attrs[dst_t][dst],
            dim=1,
            eps=1e-8,
        ).unsqueeze(1)

        neighbor_var_diff = (neighbor_attr_var[src_t][src] - neighbor_attr_var[dst_t][dst]).unsqueeze(1)

        src_global = src + offsets[src_t]
        dst_global = dst + offsets[dst_t]
        pagerank_diff = (pagerank[src_global] - pagerank[dst_global]).unsqueeze(1)
        community_diff = (community[src_global] != community[dst_global]).float().unsqueeze(1)
        spectral_diff = spectral[src_global] - spectral[dst_global]

        edge_feature_table[etype] = torch.cat(
            [
                node_type_encoding,
                edge_type_onehot,
                degree_diff,
                neighbor_type_overlap,
                node_attr_cos,
                neighbor_var_diff,
                pagerank_diff,
                community_diff,
                spectral_diff,
            ],
            dim=1,
        ).float()

    total_edges = int(sum(value.size(0) for value in edge_feature_table.values()))
    stats = _summarize_edge_features(edge_feature_table, feature_slices)
    generation_seconds = float(time.perf_counter() - start_time)
    return {
        "edge_feature_table": edge_feature_table,
        "feature_dim": int(next(iter(edge_feature_table.values())).size(1)) if edge_feature_table else 0,
        "feature_slices": feature_slices,
        "feature_stats": stats,
        "node_types": ntypes,
        "edge_types": etypes,
        "total_nodes": int(total_nodes),
        "total_edges": total_edges,
        "generation_seconds": generation_seconds,
    }


def _write_edge_feature_stats_json(cache_path: Path, payload: dict):
    selected_names = payload.get("selected_feature_names")
    if selected_names is None:
        stats_path = cache_path.with_suffix(".stats.json")
    else:
        suffix = "none" if len(selected_names) == 0 else "-".join(selected_names)
        stats_path = cache_path.with_name(f"{cache_path.stem}.selected-{suffix}.stats.json")
    stats_payload = {
        "cache_path": str(cache_path),
        "feature_dim": payload["feature_dim"],
        "feature_slices": {key: list(value) for key, value in payload["feature_slices"].items()},
        "feature_stats": payload["feature_stats"],
        "selected_feature_names": payload.get("selected_feature_names", list(payload["feature_slices"].keys())),
        "selected_feature_count": payload.get("selected_feature_count", len(payload["feature_slices"])),
        "node_types": payload["node_types"],
        "edge_types": [_edge_type_key(etype) for etype in payload["edge_types"]],
        "total_nodes": payload["total_nodes"],
        "total_edges": payload["total_edges"],
        "generation_seconds": payload["generation_seconds"],
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_edge_feature_payload(wandb_run, payload: dict, cache_path: Path, cache_hit: bool):
    if wandb_run is None:
        return

    metrics = {
        "edge_features/feature_dim": int(payload["feature_dim"]),
        "edge_features/total_nodes": int(payload["total_nodes"]),
        "edge_features/total_edges": int(payload["total_edges"]),
        "edge_features/generation_seconds": float(payload["generation_seconds"]),
        "edge_features/cache_hit": bool(cache_hit),
    }
    for name, stats in payload["feature_stats"].items():
        for stat_name, value in stats.items():
            metrics[f"edge_features/{name}/{stat_name}"] = float(value)
    log_metrics(wandb_run, metrics)

    rows = []
    for name, stats in payload["feature_stats"].items():
        row = {"feature": name}
        row.update(stats)
        rows.append(row)
    log_table(wandb_run, "edge_feature_distribution_table", rows)
    upload_file_artifact(
        wandb_run,
        cache_path,
        name=f"{payload['total_edges']}-edge-feature-table-{cache_path.stem}",
        artifact_type="edge-feature-table",
    )


def _load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_typepair_edge_feature_table(args, wandb_run=None) -> dict | None:
    if not getattr(args, "enable_typepair_edge_features", False):
        return None

    cache_path = _typepair_edge_feature_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = cache_path.exists()
    if cache_hit:
        payload = _load_torch_payload(cache_path)
    else:
        graph, _ = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
        payload = _compute_typepair_edge_feature_table(graph, args)
        torch.save(payload, cache_path)

    payload = _subset_typepair_edge_feature_payload(payload, args)
    if getattr(args, "typepair_write_edge_feature_stats", True):
        _write_edge_feature_stats_json(cache_path, payload)
    setattr(args, "typepair_edge_feature_dim", int(payload["feature_dim"]))
    _log_edge_feature_payload(wandb_run, payload, cache_path, cache_hit)
    return payload


def _attach_typepair_edge_features_to_graph(graph, edge_feature_table: dict, feature_name: str):
    for etype in graph.canonical_etypes:
        if etype not in edge_feature_table:
            continue
        table = edge_feature_table[etype]
        if dgl.EID in graph.edges[etype].data:
            edge_ids = graph.edges[etype].data[dgl.EID].detach().cpu().long()
            graph.edges[etype].data[feature_name] = table[edge_ids]
        elif graph.num_edges(etype) == table.size(0):
            graph.edges[etype].data[feature_name] = table
        else:
            raise ValueError(
                f"Cannot attach edge features for etype={etype}: subgraph has no DGL EID "
                f"and edge count {graph.num_edges(etype)} != table rows {table.size(0)}."
            )


def _attach_typepair_edge_features_to_samples(args, sample_lists: list[list]):
    if not getattr(args, "enable_typepair_edge_features", False):
        return

    payload = prepare_typepair_edge_feature_table(args, wandb_run=None)
    if payload is None:
        return
    feature_name = getattr(args, "typepair_edge_feature_name", TYPEPAIR_EDGE_FEATURE_NAME)
    for sample_list in sample_lists:
        for sample in sample_list:
            graph = _first_graph_from_sample_v2(sample, args.classification_type)
            _attach_typepair_edge_features_to_graph(graph, payload["edge_feature_table"], feature_name)


def _first_graph_from_sample_v2(sample, classification_type: str):
    del classification_type
    return sample[0]


def _build_single_sample(graph, targetnode: str, node_id: int, label: int, dataset: str):
    subgraph, inverse_indices = dgl.khop_in_subgraph(
        graph,
        {targetnode: int(node_id)},
        k=HOP_NUM[dataset],
    )
    return (subgraph, inverse_indices, torch.tensor(int(label)))


def _aligned_cache_path(root: str, dataset: str, shot: int, split_seed: int, feats_type: int) -> Path:
    return (
        ROOT
        / "artifacts"
        / "cache"
        / "protocol_aligned_splits"
        / dataset
        / f"{shot}-shot"
        / f"seed{split_seed}"
        / f"ft{feats_type}.pkl"
    )


def load_aligned_legacy_splits(args):
    if args.dataset not in HOP_NUM:
        raise ValueError(f"Unsupported dataset for aligned legacy splits: {args.dataset}")
    if args.dataset == "IMDB":
        raise NotImplementedError("Aligned legacy split builder currently targets ACM/DBLP/Freebase first.")

    cache_path = _aligned_cache_path(args.root, args.dataset, args.shot, args.split_seed, args.feats_type)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            payload = pk.load(f)
        train_list = payload["train"]
        valid_list = payload["val"]
        test_list = payload["test"]
        targetnode = payload["targetnode"]
    else:
        _ = load_split_file(args.splits, args.dataset, args.shot, args.split_seed)
        bundle = load_hgprompt_downstream_bundle(
            root=args.root,
            dataset=args.dataset,
            splits=args.splits,
            shot=args.shot,
            seed=args.split_seed,
        )

        train_idx = np.asarray(bundle.train_val_test_idx["train_idx_local"][0], dtype=np.int64)
        val_idx = np.asarray(bundle.train_val_test_idx["val_idx_local"][0], dtype=np.int64)
        test_idx = np.asarray(bundle.train_val_test_idx["test_idx_local"], dtype=np.int64)

        train_y = np.asarray(bundle.labels["train"][0], dtype=np.int64)
        val_y = np.asarray(bundle.labels["val"][0], dtype=np.int64)
        test_y = np.asarray(bundle.labels["test"], dtype=np.int64)

        graph, targetnode = _load_raw_heterograph(args.root, args.dataset, args.feats_type)

        train_list = [
            _build_single_sample(graph, targetnode, int(node_id), int(label), args.dataset)
            for node_id, label in zip(train_idx, train_y)
        ]
        valid_list = [
            _build_single_sample(graph, targetnode, int(node_id), int(label), args.dataset)
            for node_id, label in zip(val_idx, val_y)
        ]
        test_list = [
            _build_single_sample(graph, targetnode, int(node_id), int(label), args.dataset)
            for node_id, label in zip(test_idx, test_y)
        ]

        _ensure_dir(cache_path.parent)
        with open(cache_path, "wb") as f:
            pk.dump(
                {
                    "train": train_list,
                    "val": valid_list,
                    "test": test_list,
                    "targetnode": targetnode,
                },
                f,
            )

    _attach_typepair_edge_features_to_samples(args, [train_list, valid_list, test_list])
    return train_list, valid_list, test_list, targetnode


def _patched_legacy_split_loader(args):
    return load_aligned_legacy_splits(args)


def _resolve_ckpt(cli_args, method: str) -> str:
    def _maybe_format(pattern: str | None):
        if not pattern:
            return None
        return pattern.format(
            seed=cli_args.pretrain_seed,
            pretrain_seed=cli_args.pretrain_seed,
            dataset=cli_args.dataset,
            shot=cli_args.shot,
            method=method,
        )

    if method == "hgmp":
        candidate = _maybe_format(cli_args.hgmp_ckpt_pattern) or cli_args.hgmp_ckpt
    elif method == "typepair":
        candidate = (
            _maybe_format(cli_args.typepair_ckpt_pattern)
            or cli_args.typepair_ckpt
            or _maybe_format(cli_args.hgmp_ckpt_pattern)
            or cli_args.hgmp_ckpt
        )
    elif method == "hgmp_prompt":
        candidate = _maybe_format(cli_args.hgmp_ckpt_pattern) or cli_args.hgmp_ckpt
    elif method == "hgprompt":
        candidate = _maybe_format(cli_args.hgprompt_ckpt_pattern) or cli_args.hgprompt_ckpt
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not candidate:
        raise ValueError(f"No checkpoint configured for method={method}")
    return candidate


def _make_run_seed(split_seed: int, repeat_id: int, run_seed_base: int) -> int:
    return int(run_seed_base) + int(split_seed) * 1000 + int(repeat_id)


def _make_legacy_args(cli_args, method: str, ckpt_path: str, split_seed: int, repeat_id: int):
    run_seed = _make_run_seed(split_seed, repeat_id, cli_args.run_seed_base)
    return SimpleNamespace(
        method=method,
        ckpt=ckpt_path,
        dataset=cli_args.dataset,
        device=torch.device(cli_args.device),
        seed=run_seed,
        split_seed=split_seed,
        shot=cli_args.shot,
        feats_type=cli_args.feats_type,
        hidden_dim=cli_args.hidden_dim,
        num_heads=cli_args.num_heads,
        num_layers=cli_args.num_layers,
        dropout=cli_args.dropout,
        hgnn_type=cli_args.hgnn_type,
        num_samples=cli_args.num_samples,
        num_class=cli_args.num_class,
        classification_type=cli_args.classification_type,
        relation_prompt_mode=cli_args.relation_prompt_mode,
        relation_prompt_alpha=cli_args.relation_prompt_alpha,
        relation_prompt_dropout=cli_args.relation_prompt_dropout,
        relation_prompt_aggr=cli_args.relation_prompt_aggr,
        relation_prompt_use_ln=cli_args.relation_prompt_use_ln,
        enable_typepair_edge_features=cli_args.enable_typepair_edge_features,
        typepair_edge_feature_dim=cli_args.typepair_edge_feature_dim,
        typepair_edge_feature_names=cli_args.typepair_edge_feature_names,
        typepair_edge_feature_name=cli_args.typepair_edge_feature_name,
        typepair_edge_feature_cache_dir=cli_args.typepair_edge_feature_cache_dir,
        typepair_write_edge_feature_stats=cli_args.typepair_write_edge_feature_stats,
        typepair_edge_attr_dim_cap=cli_args.typepair_edge_attr_dim_cap,
        typepair_pagerank_damping=cli_args.typepair_pagerank_damping,
        typepair_pagerank_max_iter=cli_args.typepair_pagerank_max_iter,
        typepair_pagerank_tol=cli_args.typepair_pagerank_tol,
        typepair_spectral_dim=cli_args.typepair_spectral_dim,
        typepair_spectral_max_nodes=cli_args.typepair_spectral_max_nodes,
        typepair_edge_prompt_hidden=cli_args.typepair_edge_prompt_hidden,
        typepair_edge_prompt_alpha=cli_args.typepair_edge_prompt_alpha,
        typepair_edge_prompt_fusion=cli_args.typepair_edge_prompt_fusion,
        embed_batch_size=cli_args.embed_batch_size,
        head_hidden=cli_args.head_hidden,
        head_dropout=cli_args.head_dropout,
        epochs=cli_args.epochs,
        patience=cli_args.patience,
        lr=cli_args.lr,
        prompt_lr=cli_args.prompt_lr,
        weight_decay=cli_args.weight_decay,
        early_stop_metric=cli_args.early_stop_metric,
        save_dir=str(cli_args.save_dir),
        root=cli_args.root,
        splits=cli_args.splits,
    )


def run_legacy_method_once(cli_args, method: str, ckpt_path: str, split_seed: int, repeat_id: int, epoch_callback=None) -> RunRecord:
    args = _make_legacy_args(cli_args, method, ckpt_path, split_seed, repeat_id)
    _set_global_seed(args.seed)

    orig_loader = legacy_bridge._load_legacy_fewshot_splits
    legacy_bridge._load_legacy_fewshot_splits = _patched_legacy_split_loader
    try:
        save_dir = (
            Path(args.save_dir)
            / "aligned_protocol"
            / args.dataset
            / method
            / f"{args.shot}-shot"
            / f"pretrainseed{cli_args.pretrain_seed}"
            / f"splitseed{split_seed}"
            / f"repeat{repeat_id}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = str(save_dir / "best.pt")

        if method == "typepair":
            res = train_typepair_prompt_probe(
                args=args,
                batch_size=args.embed_batch_size,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                head_lr=args.lr,
                prompt_lr=args.prompt_lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=best_path,
                epoch_callback=epoch_callback,
            )
        elif method == "hgmp_prompt":
            res = train_hgmp_heteroprompt_probe(
                args=args,
                batch_size=args.embed_batch_size,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=best_path,
                epoch_callback=epoch_callback,
            )
        else:
            emb = build_legacy_fewshot_embeddings(args=args, batch_size=args.embed_batch_size)
            res = train_mlp_probe(
                x_train=emb.x_train,
                y_train=emb.y_train,
                x_val=emb.x_val,
                y_val=emb.y_val,
                x_test=emb.x_test,
                y_test=emb.y_test,
                in_dim=emb.x_train.size(-1),
                num_classes=args.num_class,
                device=args.device,
                hidden_dim=args.head_hidden,
                dropout=args.head_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                early_stop_metric=args.early_stop_metric,
                save_best_path=best_path,
                epoch_callback=epoch_callback,
            )
    finally:
        legacy_bridge._load_legacy_fewshot_splits = orig_loader

    return RunRecord(
        method=method,
        split_seed=split_seed,
        repeat_id=repeat_id,
        run_seed=int(args.seed),
        ckpt_path=ckpt_path,
        test_micro=float(res["test_at_best_micro"]),
        test_macro=float(res["test_at_best_macro"]),
        best_epoch=int(res["best_epoch"]),
    )


def _make_hgprompt_args(cli_args, ckpt_path: str, split_seed: int, repeat_id: int, save_dir: Path):
    run_seed = _make_run_seed(split_seed, repeat_id, cli_args.run_seed_base)
    return SimpleNamespace(
        root=cli_args.root,
        dataset=cli_args.dataset,
        splits=cli_args.splits,
        shotnum=cli_args.shot,
        seed=split_seed,
        repeat=1,
        tasknum=1,
        pretrain_ckpt=ckpt_path,
        save_dir=str(save_dir),
        device=cli_args.device,
        strict_load=False,
        feats_type=cli_args.hgprompt_feats_type,
        hidden_dim=cli_args.hgprompt_hidden_dim,
        bottle_net_hidden_dim=cli_args.hgprompt_bottle_net_hidden_dim,
        bottle_net_output_dim=cli_args.hgprompt_bottle_net_output_dim,
        edge_feats=cli_args.hgprompt_edge_feats,
        num_heads=cli_args.hgprompt_num_heads,
        epoch=cli_args.epochs,
        patience=cli_args.patience,
        model_type=cli_args.hgprompt_model_type,
        num_layers=cli_args.hgprompt_num_layers,
        lr=cli_args.hgprompt_lr,
        dropout=cli_args.hgprompt_dropout,
        weight_decay=cli_args.hgprompt_weight_decay,
        slope=cli_args.hgprompt_slope,
        tuning=cli_args.hgprompt_tuning,
        subgraph_hop_num=cli_args.hgprompt_subgraph_hop_num,
        pre_loss_weight=cli_args.hgprompt_pre_loss_weight,
        hetero_pretrain=cli_args.hgprompt_hetero_pretrain,
        hetero_pretrain_subgraph=cli_args.hgprompt_hetero_pretrain_subgraph,
        pretrain_semantic=cli_args.hgprompt_pretrain_semantic,
        pretrain_each_loss=cli_args.hgprompt_pretrain_each_loss,
        add_edge_info2prompt=cli_args.hgprompt_add_edge_info2prompt,
        each_type_subgraph=cli_args.hgprompt_each_type_subgraph,
        cat_prompt_dim=cli_args.hgprompt_cat_prompt_dim,
        cat_hprompt_dim=cli_args.hgprompt_cat_hprompt_dim,
        tuple_neg_disconnected_num=cli_args.hgprompt_tuple_neg_disconnected_num,
        tuple_neg_unrelated_num=cli_args.hgprompt_tuple_neg_unrelated_num,
        semantic_prompt=cli_args.hgprompt_semantic_prompt,
        semantic_prompt_weight=cli_args.hgprompt_semantic_prompt_weight,
        freebase_type=cli_args.hgprompt_freebase_type,
        shgn_hidden_dim=cli_args.hgprompt_shgn_hidden_dim,
        downstream_run_seed=run_seed,
    )


def run_hgprompt_once(cli_args, ckpt_path: str, split_seed: int, repeat_id: int, epoch_callback=None) -> RunRecord:
    run_seed = _make_run_seed(split_seed, repeat_id, cli_args.run_seed_base)
    _set_global_seed(run_seed)

    save_dir = (
        Path(cli_args.save_dir)
        / "aligned_protocol"
        / cli_args.dataset
        / "hgprompt"
        / f"{cli_args.shot}-shot"
        / f"pretrainseed{cli_args.pretrain_seed}"
        / f"splitseed{split_seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    args = _make_hgprompt_args(cli_args, ckpt_path, split_seed, repeat_id, save_dir)
    bundle = load_hgprompt_downstream_bundle(
        root=cli_args.root,
        dataset=cli_args.dataset,
        splits=cli_args.splits,
        shot=cli_args.shot,
        seed=split_seed,
    )
    res = hgprompt_run_once(args, bundle, repeat_id, epoch_callback=epoch_callback)
    return RunRecord(
        method="hgprompt",
        split_seed=split_seed,
        repeat_id=repeat_id,
        run_seed=run_seed,
        ckpt_path=ckpt_path,
        test_micro=float(res.test_micro),
        test_macro=float(res.test_macro),
        best_epoch=int(res.best_epoch),
    )


def _summarize_runs(records: list[RunRecord]):
    if len(records) == 0:
        return {}
    micro = np.array([r.test_micro for r in records], dtype=np.float64)
    macro = np.array([r.test_macro for r in records], dtype=np.float64)
    return {
        "count": int(len(records)),
        "micro_mean": float(micro.mean()),
        "micro_std": float(micro.std()),
        "macro_mean": float(macro.mean()),
        "macro_std": float(macro.std()),
    }


def _aggregate_by_seed(records: list[RunRecord]) -> list[SeedAggregate]:
    out = []
    methods = sorted(set(r.method for r in records))
    for method in methods:
        seeds = sorted(set(r.split_seed for r in records if r.method == method))
        for split_seed in seeds:
            subset = [r for r in records if r.method == method and r.split_seed == split_seed]
            stat = _summarize_runs(subset)
            out.append(
                SeedAggregate(
                    method=method,
                    split_seed=split_seed,
                    count=stat["count"],
                    micro_mean=stat["micro_mean"],
                    micro_std=stat["micro_std"],
                    macro_mean=stat["macro_mean"],
                    macro_std=stat["macro_std"],
                )
            )
    return out


def _summarize_seed_means(seed_rows: list[SeedAggregate], method: str):
    subset = [r for r in seed_rows if r.method == method]
    if len(subset) == 0:
        return {}
    micro = np.array([r.micro_mean for r in subset], dtype=np.float64)
    macro = np.array([r.macro_mean for r in subset], dtype=np.float64)
    return {
        "count": int(len(subset)),
        "micro_mean": float(micro.mean()),
        "micro_std": float(micro.std()),
        "macro_mean": float(macro.mean()),
        "macro_std": float(macro.std()),
    }


def _write_csv(path: Path, rows: list[dict]):
    if len(rows) == 0:
        return
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _build_run_config(args, ckpt_by_method: dict[str, str]):
    config = vars(args).copy()
    config.pop("wandb_key", None)
    config["resolved_ckpt_by_method"] = ckpt_by_method
    return _json_ready(config)


def _make_downstream_epoch_logger(wandb_run, args, method: str, split_seed: int, repeat_id: int, run_seed: int):
    if wandb_run is None:
        return None

    def _callback(metrics: dict):
        payload = {f"downstream/{key}": value for key, value in metrics.items()}
        payload.update(
            {
                "downstream/method": method,
                "downstream/dataset": args.dataset,
                "downstream/shot": args.shot,
                "downstream/split_seed": split_seed,
                "downstream/repeat_id": repeat_id,
                "downstream/run_seed": run_seed,
                "downstream/pretrain_seed": args.pretrain_seed,
            }
        )
        log_metrics(wandb_run, payload)

    return _callback


def build_parser():
    ap = argparse.ArgumentParser("Aligned protocol benchmark for hgmp / typepair / hgprompt")

    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--methods", nargs="+", default=["hgmp", "typepair", "hgprompt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--pretrain_seed", type=int, default=0)
    ap.add_argument("--run_seed_base", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=Path, default=ROOT / "artifacts" / "results" / "protocol_benchmark")

    ap.add_argument("--hgmp_ckpt", type=str, default=None)
    ap.add_argument("--typepair_ckpt", type=str, default=None)
    ap.add_argument("--hgprompt_ckpt", type=str, default=None)
    ap.add_argument("--hgmp_ckpt_pattern", type=str, default=None)
    ap.add_argument("--typepair_ckpt_pattern", type=str, default=None)
    ap.add_argument("--hgprompt_ckpt_pattern", type=str, default=None)

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--hgnn_type", type=str, default="HGT")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--num_class", type=int, default=3)
    ap.add_argument("--classification_type", type=str, default="NIG")
    ap.add_argument("--embed_batch_size", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--prompt_lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop_metric", type=str, default="macro", choices=["micro", "macro"])

    ap.add_argument("--relation_prompt_mode", type=str, default="mul", choices=["mul", "add"])
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    ap.add_argument("--relation_prompt_aggr", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--relation_prompt_use_ln", action="store_true")

    ap.add_argument("--enable_typepair_edge_features", action="store_true")
    ap.add_argument("--typepair_edge_feature_dim", type=int, default=0)
    ap.add_argument("--typepair_edge_feature_names", nargs="*", default=None, choices=TYPEPAIR_EDGE_FEATURES)
    ap.add_argument("--typepair_edge_feature_name", type=str, default=TYPEPAIR_EDGE_FEATURE_NAME)
    ap.add_argument(
        "--typepair_edge_feature_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "typepair_edge_features",
    )
    ap.add_argument("--typepair_write_edge_feature_stats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--typepair_edge_attr_dim_cap", type=int, default=256)
    ap.add_argument("--typepair_pagerank_damping", type=float, default=0.85)
    ap.add_argument("--typepair_pagerank_max_iter", type=int, default=50)
    ap.add_argument("--typepair_pagerank_tol", type=float, default=1e-6)
    ap.add_argument("--typepair_spectral_dim", type=int, default=8)
    ap.add_argument("--typepair_spectral_max_nodes", type=int, default=50000)
    ap.add_argument("--typepair_edge_prompt_hidden", type=int, default=128)
    ap.add_argument("--typepair_edge_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--typepair_edge_prompt_fusion", type=str, default="add", choices=["add", "mul", "gate"])

    ap.add_argument("--hgprompt_feats_type", type=int, default=2)
    ap.add_argument("--hgprompt_hidden_dim", type=int, default=64)
    ap.add_argument("--hgprompt_bottle_net_hidden_dim", type=int, default=2)
    ap.add_argument("--hgprompt_bottle_net_output_dim", type=int, default=64)
    ap.add_argument("--hgprompt_edge_feats", type=int, default=64)
    ap.add_argument("--hgprompt_num_heads", type=int, default=8)
    ap.add_argument("--hgprompt_model_type", type=str, default="gcn", choices=["gcn", "gat", "gin", "SHGN"])
    ap.add_argument("--hgprompt_num_layers", type=int, default=2)
    ap.add_argument("--hgprompt_lr", type=float, default=1.0)
    ap.add_argument("--hgprompt_dropout", type=float, default=0.5)
    ap.add_argument("--hgprompt_weight_decay", type=float, default=1e-6)
    ap.add_argument("--hgprompt_slope", type=float, default=0.05)
    ap.add_argument("--hgprompt_tuning", type=str, default="weight-sum-center-fixed")
    ap.add_argument("--hgprompt_subgraph_hop_num", type=int, default=1)
    ap.add_argument("--hgprompt_pre_loss_weight", type=float, default=1.0)
    ap.add_argument("--hgprompt_hetero_pretrain", type=int, default=0)
    ap.add_argument("--hgprompt_hetero_pretrain_subgraph", type=int, default=0)
    ap.add_argument("--hgprompt_pretrain_semantic", type=int, default=0)
    ap.add_argument("--hgprompt_pretrain_each_loss", type=int, default=0)
    ap.add_argument("--hgprompt_add_edge_info2prompt", type=int, default=1)
    ap.add_argument("--hgprompt_each_type_subgraph", type=int, default=1)
    ap.add_argument("--hgprompt_cat_prompt_dim", type=int, default=64)
    ap.add_argument("--hgprompt_cat_hprompt_dim", type=int, default=64)
    ap.add_argument("--hgprompt_tuple_neg_disconnected_num", type=int, default=1)
    ap.add_argument("--hgprompt_tuple_neg_unrelated_num", type=int, default=1)
    ap.add_argument("--hgprompt_semantic_prompt", type=int, default=1)
    ap.add_argument("--hgprompt_semantic_prompt_weight", type=float, default=0.1)
    ap.add_argument("--hgprompt_freebase_type", type=int, default=0)
    ap.add_argument("--hgprompt_shgn_hidden_dim", type=int, default=3)

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="HGEP")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_job_type", type=str, default="protocol_benchmark_v2")
    ap.add_argument("--wandb_tags", nargs="*", default=[])
    ap.add_argument("--wandb_notes", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    ap.add_argument("--wandb_dir", type=Path, default=ROOT / "artifacts" / "wandb")
    ap.add_argument("--wandb_key", type=str, default=None)
    ap.add_argument("--wandb_key_file", type=Path, default=ROOT / ".codex")

    return ap


def main():
    args = build_parser().parse_args()
    wandb_run = None
    try:
        records: list[RunRecord] = []
        ckpt_by_method = {method: _resolve_ckpt(args, method) for method in args.methods}
        config = _build_run_config(args, ckpt_by_method)

        if args.use_wandb:
            maybe_configure_wandb_env(api_key=args.wandb_key, env_file=args.wandb_key_file)
            wandb_run = init_wandb_run(
                enabled=True,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                group=args.wandb_group or f"{args.dataset}-{args.shot}shot",
                job_type=args.wandb_job_type,
                tags=args.wandb_tags or [args.dataset, f"{args.shot}-shot", "protocol_benchmark_v2"],
                notes=args.wandb_notes,
                mode=args.wandb_mode,
                dir_path=args.wandb_dir,
                config=config,
            )

        if args.enable_typepair_edge_features and "typepair" in args.methods:
            edge_feature_payload = prepare_typepair_edge_feature_table(args, wandb_run=wandb_run)
            if edge_feature_payload is not None and wandb_run is not None:
                wandb_run.config.update(
                    {"typepair_edge_feature_dim": int(edge_feature_payload["feature_dim"])},
                    allow_val_change=True,
                )

        for method in args.methods:
            ckpt_path = ckpt_by_method[method]
            print(f"================ method={method} | ckpt={ckpt_path} ================")
            for split_seed in args.seeds:
                for repeat_id in range(args.repeats):
                    run_seed = _make_run_seed(split_seed, repeat_id, args.run_seed_base)
                    epoch_logger = _make_downstream_epoch_logger(
                        wandb_run,
                        args,
                        method=method,
                        split_seed=split_seed,
                        repeat_id=repeat_id,
                        run_seed=run_seed,
                    )
                    if method == "hgprompt":
                        record = run_hgprompt_once(args, ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    elif method == "typepair":
                        record = run_legacy_method_once(args, "typepair", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    elif method == "hgmp":
                        record = run_legacy_method_once(args, "hgmp", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    elif method == "hgmp_prompt":
                        record = run_legacy_method_once(args, "hgmp_prompt", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    records.append(record)
                    print(
                        f"[RUN] method={record.method} | split_seed={record.split_seed} | repeat={record.repeat_id} | "
                        f"run_seed={record.run_seed} | ckpt={record.ckpt_path} | "
                        f"micro={record.test_micro:.4f} | macro={record.test_macro:.4f} | best_epoch={record.best_epoch}"
                    )
                    log_run_record(
                        wandb_run,
                        record,
                        extra={
                            "per_run/dataset": args.dataset,
                            "per_run/shot": args.shot,
                            "per_run/pretrain_seed": args.pretrain_seed,
                        },
                    )

        out_dir = _ensure_dir(args.save_dir / args.dataset / f"{args.shot}-shot")
        per_run_rows = [asdict(r) for r in records]
        _write_csv(out_dir / "per_run.csv", per_run_rows)

        seed_rows = _aggregate_by_seed(records)
        seed_row_dicts = [asdict(r) for r in seed_rows]
        _write_csv(out_dir / "per_seed_summary.csv", seed_row_dicts)

        pooled_summary = {}
        seedmean_summary = {}
        for method in sorted(set(r.method for r in records)):
            pooled_summary[method] = _summarize_runs([r for r in records if r.method == method])
            seedmean_summary[method] = _summarize_seed_means(seed_rows, method)

        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        with open(out_dir / "overall_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": config,
                    "pooled_runs": pooled_summary,
                    "seed_mean_then_std": seedmean_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        log_table(wandb_run, "per_run_table", per_run_rows)
        log_table(wandb_run, "per_seed_summary_table", seed_row_dicts)
        log_nested_summary(wandb_run, "pooled_runs", pooled_summary)
        log_nested_summary(wandb_run, "seed_mean_then_std", seedmean_summary)
        upload_dir_artifact(
            wandb_run,
            out_dir,
            name=f"{args.dataset}-{args.shot}shot-protocol-benchmark-v2-{args.pretrain_seed}",
            artifact_type="benchmark-results",
        )

        print("####################################################")
        print(json.dumps({"pooled_runs": pooled_summary, "seed_mean_then_std": seedmean_summary}, indent=2, ensure_ascii=False))
    except Exception:
        finish_wandb_run(wandb_run, exit_code=1)
        raise
    else:
        finish_wandb_run(wandb_run, exit_code=0)


if __name__ == "__main__":
    main()
