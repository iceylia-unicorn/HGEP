# scripts/precompute_peprompt_cache.py
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import hashlib
import pickle as pk
import time
from types import SimpleNamespace

import dgl
import numpy as np
import torch

from gpbench.downstream.fewshot import build_peprompt_offline_cache_path, save_peprompt_split_ids
from scripts.peprompt_benchmark import (
    HOP_NUM,
    PEPROMPT_EDGE_FEATURE_NAME,
    PEPROMPT_EDGE_FEATURES,
    PEPROMPT_DEFAULT_COARSE_HOPS,
    PEPROMPT_DEFAULT_TYPE_HOPS,
    _load_raw_heterograph,
    _peprompt_cache_subgraph_type,
    prepare_peprompt_edge_feature_table,
)

from scripts.subgraph_sampling_stats import (
    _build_csr_adjs,
    _generate_metapaths,
    _metapath_reachable_scores,
    _rank_values,
    _topk_indices,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_float_for_key(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _stable_rng(seed: int, *parts) -> np.random.Generator:
    key = "|".join(str(part) for part in (int(seed), *parts))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(value)


def _target_labels(graph, targetnode: str) -> np.ndarray:
    labels = graph.ndata["y"][targetnode].detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return labels


def _target_supervision_labels(graph, targetnode: str, dataset: str) -> np.ndarray:
    """Return labels saved in samples; IMDB follows legacy HGMP multi-hot supervision."""
    ndata_keys = set(graph.ndata.keys())
    label_name = "oldy" if str(dataset) == "IMDB" and "oldy" in ndata_keys else "y"
    labels = graph.ndata[label_name][targetnode].detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim == 1:
        return labels.reshape(-1)
    return labels


def _seed_nodes_dict(target_ntype: str, target_node_id: int) -> dict[str, torch.Tensor]:
    return {target_ntype: torch.tensor([int(target_node_id)], dtype=torch.int64)}


def _node_type_offsets(graph) -> dict[str, int]:
    offsets = {}
    cursor = 0
    for ntype in graph.ntypes:
        offsets[ntype] = int(cursor)
        cursor += int(graph.num_nodes(ntype))
    return offsets


def _find_seed_inverse_indices(subgraph, seed_nodes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    inverse_indices = {}
    for ntype, global_ids in seed_nodes.items():
        subgraph_nids = subgraph.nodes[ntype].data[dgl.NID].detach().cpu().long()
        positions = []
        for global_id in global_ids.detach().cpu().long().view(-1).tolist():
            matched = (subgraph_nids == int(global_id)).nonzero(as_tuple=False).view(-1)
            if matched.numel() == 0:
                raise RuntimeError(
                    f"Seed node {global_id} of type {ntype} is missing from sampled subgraph."
                )
            positions.append(matched[0])
        inverse_indices[ntype] = torch.stack(positions).long()
    return inverse_indices


def extract_khop_subgraph(graph, target_ntype: str, target_node_id: int, hop_num: int):
    return dgl.khop_in_subgraph(
        graph,
        _seed_nodes_dict(target_ntype, target_node_id),
        k=int(hop_num),
    )


def extract_fanout_subgraph(
    graph,
    target_ntype: str,
    target_node_id: int,
    fanout=[15, 10],
    sampler=None,
):
    fanout = [int(v) for v in fanout]
    if sampler is None:
        sampler = dgl.dataloading.NeighborSampler(fanout)

    seed_nodes = _seed_nodes_dict(target_ntype, target_node_id)
    input_nodes, _, _ = sampler.sample_blocks(graph, seed_nodes)

    sampled_nodes = {}
    for ntype, node_ids in input_nodes.items():
        node_ids = node_ids.detach().cpu().long()
        if node_ids.numel() > 0:
            sampled_nodes[ntype] = torch.unique(node_ids, sorted=True)
    if target_ntype not in sampled_nodes:
        sampled_nodes[target_ntype] = seed_nodes[target_ntype]

    subgraph = dgl.node_subgraph(graph, sampled_nodes)
    inverse_indices = _find_seed_inverse_indices(subgraph, seed_nodes)
    return subgraph, inverse_indices


def _metapath_cache_key(args) -> str:
    return _peprompt_cache_subgraph_type(args)


def _metapath_source_idf(adjs: dict, metapath: tuple) -> float:
    reachable = adjs[metapath[0]]
    for etype in metapath[1:]:
        reachable = reachable @ adjs[etype]
    source_count = int(reachable.shape[0])
    reached_sources = int(np.count_nonzero(np.asarray(reachable.getnnz(axis=1)).reshape(-1)))
    return float(np.log((1.0 + source_count) / (1.0 + reached_sources)) + 1.0)


def _build_csc_adjs(adjs: dict) -> dict:
    return {etype: adj.tocsc() for etype, adj in adjs.items()}


def _filter_metapaths_by_endpoint_mode(metapaths: list, target_ntype: str, endpoint_mode: str) -> list:
    endpoint_mode = str(endpoint_mode)
    if endpoint_mode == "all":
        return list(metapaths)
    if endpoint_mode == "target_closed":
        return [
            metapath
            for metapath in metapaths
            if len(metapath) > 0 and metapath[0][0] == target_ntype and metapath[-1][2] == target_ntype
        ]
    raise ValueError(f"Unsupported metapath_endpoint_mode={endpoint_mode}")


def _resolve_metapath_support_mode(subgraph_type: str, endpoint_mode: str, support_mode: str) -> str:
    support_mode = str(support_mode)
    if support_mode != "auto":
        return support_mode
    if str(endpoint_mode) == "target_closed":
        return "count"
    if str(subgraph_type) in {"metapath_topk_path", "metapath_topk_path_adapt"}:
        return "one_path"
    return "none"


def _sparse_row_indices_values(row) -> tuple[np.ndarray, np.ndarray]:
    row = row.tocsr()
    if not row.has_canonical_format:
        row.sum_duplicates()
    if not row.has_sorted_indices:
        row.sort_indices()
    return (
        np.asarray(row.indices, dtype=np.int64),
        np.asarray(row.data, dtype=np.float32),
    )


def _sparse_col_indices_values(col) -> tuple[np.ndarray, np.ndarray]:
    col = col.tocsc()
    if not col.has_canonical_format:
        col.sum_duplicates()
    if not col.has_sorted_indices:
        col.sort_indices()
    return (
        np.asarray(col.indices, dtype=np.int64),
        np.asarray(col.data, dtype=np.float32),
    )


def _top_support_indices_from_sparse_scores(
    prefix,
    suffix,
    support_topk: int,
    rank_mode: str = "score",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    prefix_idx, prefix_val = _sparse_row_indices_values(prefix)
    suffix_idx, suffix_val = _sparse_col_indices_values(suffix)
    if prefix_idx.size == 0 or suffix_idx.size == 0:
        return np.empty(0, dtype=np.int64)

    common, prefix_pos, suffix_pos = np.intersect1d(
        prefix_idx,
        suffix_idx,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        return common.astype(np.int64)

    support_topk = int(support_topk)
    if support_topk <= 0 or common.size <= support_topk:
        if str(rank_mode) == "score":
            values = prefix_val[prefix_pos] * suffix_val[suffix_pos]
            order = np.lexsort((common, -values))
            return common[order].astype(np.int64)
        return common.astype(np.int64)

    if str(rank_mode) == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        keep = rng.choice(common, size=support_topk, replace=False)
        return np.sort(np.asarray(keep, dtype=np.int64))

    values = prefix_val[prefix_pos] * suffix_val[suffix_pos]
    partial = np.argpartition(-values, support_topk - 1)[:support_topk]
    support = common[partial]
    support_values = values[partial]
    order = np.lexsort((support, -support_values))
    return support[order].astype(np.int64)


def _prefix_scores_for_candidates(prefix, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prefix_idx, prefix_val = _sparse_row_indices_values(prefix)
    if prefix_idx.size == 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    common, prefix_pos, _ = np.intersect1d(
        prefix_idx,
        candidates,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        return common.astype(np.int64), np.empty(0, dtype=np.float32)
    return common.astype(np.int64), prefix_val[prefix_pos].astype(np.float32, copy=False)


def _add_count_support_for_endpoints(
    selected: dict[str, set[int]],
    adjs: dict,
    metapath: tuple,
    target_ntype: str,
    target_node_id: int,
    endpoint_ids: list[int],
    support_topk: int = 0,
    support_rank_mode: str = "score",
    random_seed: int = 0,
    metapath_idx: int = 0,
) -> None:
    """Add support nodes that lie on paths from centre to selected endpoints.

    Endpoint selection stays target-typed. For intermediate nodes, contribution
    is prefix_count(centre -> node) * suffix_count(node -> selected endpoints).
    """
    import scipy.sparse as sp

    if not endpoint_ids:
        return

    endpoint_ids = sorted({int(v) for v in endpoint_ids})
    row = sp.csr_matrix(
        ([1.0], ([0], [int(target_node_id)])),
        shape=(1, int(adjs[metapath[0]].shape[0])),
        dtype=np.float32,
    )
    prefixes = [row]
    for etype in metapath:
        row = row @ adjs[etype]
        prefixes.append(row)

    dst_t = metapath[-1][2]
    selected[dst_t].update(endpoint_ids)
    col = sp.csr_matrix(
        (np.ones(len(endpoint_ids), dtype=np.float32), (endpoint_ids, np.zeros(len(endpoint_ids), dtype=np.int64))),
        shape=(int(adjs[metapath[-1]].shape[1]), 1),
        dtype=np.float32,
    )
    suffixes = [None for _ in range(len(metapath) + 1)]
    suffixes[len(metapath)] = col
    for step in range(len(metapath) - 1, -1, -1):
        suffixes[step] = adjs[metapath[step]] @ suffixes[step + 1]

    for pos in range(1, len(metapath)):
        ntype = metapath[pos - 1][2]
        rng = None
        if str(support_rank_mode) == "random":
            rng = _stable_rng(
                int(random_seed),
                "count_support",
                int(target_node_id),
                int(metapath_idx),
                int(pos),
                ",".join(str(v) for v in endpoint_ids),
            )
        keep_support = _top_support_indices_from_sparse_scores(
            prefixes[pos],
            suffixes[pos],
            int(support_topk),
            rank_mode=str(support_rank_mode),
            rng=rng,
        )
        selected[ntype].update(int(v) for v in keep_support.tolist())

    selected[target_ntype].add(int(target_node_id))


def _add_one_path_for_endpoint(
    selected: dict[str, set[int]],
    adjs: dict,
    adjs_csc: dict,
    metapath: tuple,
    target_ntype: str,
    target_node_id: int,
    endpoint_id: int,
    support_rank_mode: str = "score",
    random_seed: int = 0,
    metapath_idx: int = 0,
) -> bool:
    """Add one concrete centre-to-endpoint path for a selected metapath endpoint.

    The top-k rule selects endpoints by metapath reachability. This helper
    backtracks one actual path and adds its intermediate nodes, so the induced
    subgraph preserves a visible route from the centre to the endpoint.
    """
    import scipy.sparse as sp

    row = sp.csr_matrix(
        ([1.0], ([0], [int(target_node_id)])),
        shape=(1, int(adjs[metapath[0]].shape[0])),
        dtype=np.float32,
    )
    prefix_rows = []
    for etype in metapath:
        prefix_rows.append(row)
        row = row @ adjs[etype]

    current = int(endpoint_id)
    selected[metapath[-1][2]].add(current)
    for step in range(len(metapath) - 1, -1, -1):
        etype = metapath[step]
        src_t = etype[0]
        predecessors = np.asarray(adjs_csc[etype].getcol(current).nonzero()[0], dtype=np.int64)
        if predecessors.size == 0:
            return False

        candidates, candidate_scores = _prefix_scores_for_candidates(prefix_rows[step], predecessors)
        if candidates.size == 0:
            return False
        if str(support_rank_mode) == "random":
            rng = _stable_rng(
                int(random_seed),
                "one_path_support",
                int(target_node_id),
                int(endpoint_id),
                int(metapath_idx),
                int(step),
            )
            current = int(rng.choice(candidates))
        else:
            order = np.lexsort((candidates, -candidate_scores))
            current = int(candidates[order[0]])
        selected[src_t].add(current)

    return current == int(target_node_id) and metapath[0][0] == target_ntype


def _adaptive_relative_indices(
    scores: np.ndarray,
    ranks: np.ndarray,
    min_topk: int,
    max_topk: int,
    rel_threshold: float,
) -> np.ndarray:
    candidates = np.flatnonzero(scores > 0).astype(np.int64)
    if candidates.size == 0:
        return candidates

    min_topk = max(0, int(min_topk))
    max_topk = int(max_topk)
    rel_threshold = float(rel_threshold)
    candidate_ranks = np.asarray(ranks[candidates], dtype=np.float32)
    max_rank = float(candidate_ranks.max()) if candidate_ranks.size > 0 else 0.0
    if max_rank <= 0.0:
        keep = np.empty(0, dtype=np.int64)
    else:
        keep = candidates[candidate_ranks >= rel_threshold * max_rank]

    required = min(min_topk, int(candidates.size))
    if keep.size < required:
        keep = np.unique(
            np.concatenate([keep.astype(np.int64), _topk_indices(scores, ranks, required)])
        ).astype(np.int64)

    if keep.size == 0:
        return keep.astype(np.int64)

    keep_ranks = np.asarray(ranks[keep], dtype=np.float32)
    order = np.lexsort((keep, -keep_ranks))
    keep = keep[order]
    if max_topk > 0 and keep.size > max_topk:
        keep = keep[:max_topk]
    return keep.astype(np.int64)


def extract_metapath_topk_subgraph(
    graph,
    target_ntype: str,
    target_node_id: int,
    adjs: dict,
    metapaths: list,
    topk: int,
    rank_metric: str,
    keep_self: bool,
    endpoint_popularity: dict,
    preserve_paths: bool = False,
    adjs_csc: dict | None = None,
    adaptive_mode: str = "none",
    min_topk: int = 1,
    max_topk: int = 5,
    rel_threshold: float = 0.5,
    support_mode: str = "none",
    support_topk: int = 0,
    support_rank_mode: str = "score",
    random_seed: int = 0,
):
    selected: dict[str, set[int]] = {ntype: set() for ntype in graph.ntypes}
    selected[target_ntype].add(int(target_node_id))

    for metapath_idx, metapath in enumerate(metapaths):
        scores = _metapath_reachable_scores(
            adjs=adjs,
            metapath=metapath,
            node_id=int(target_node_id),
            target_ntype=target_ntype,
            keep_self=bool(keep_self),
        )
        if str(rank_metric) == "random":
            ranks = np.zeros_like(np.asarray(scores, dtype=np.float32))
            candidates = np.flatnonzero(scores > 0).astype(np.int64)
            if candidates.size > 0:
                rng = _stable_rng(
                    int(random_seed),
                    "endpoint",
                    int(target_node_id),
                    int(metapath_idx),
                )
                ranks[candidates] = rng.random(candidates.size).astype(np.float32)
        else:
            ranks = _rank_values(scores, endpoint_popularity.get(metapath), rank_metric)
        if str(adaptive_mode) == "relative":
            keep = _adaptive_relative_indices(
                scores=scores,
                ranks=ranks,
                min_topk=int(min_topk),
                max_topk=int(max_topk),
                rel_threshold=float(rel_threshold),
            )
        else:
            keep = _topk_indices(scores, ranks, int(topk))
        if keep.size > 0:
            dst_t = metapath[-1][2]
            keep_ids = [int(v) for v in keep.tolist()]
            if support_mode == "count":
                _add_count_support_for_endpoints(
                    selected=selected,
                    adjs=adjs,
                    metapath=metapath,
                    target_ntype=target_ntype,
                    target_node_id=int(target_node_id),
                    endpoint_ids=keep_ids,
                    support_topk=int(support_topk),
                    support_rank_mode=str(support_rank_mode),
                    random_seed=int(random_seed),
                    metapath_idx=int(metapath_idx),
                )
            elif preserve_paths or support_mode == "one_path":
                if adjs_csc is None:
                    adjs_csc = _build_csc_adjs(adjs)
                for keep_id in keep_ids:
                    ok = _add_one_path_for_endpoint(
                        selected=selected,
                        adjs=adjs,
                        adjs_csc=adjs_csc,
                        metapath=metapath,
                        target_ntype=target_ntype,
                        target_node_id=int(target_node_id),
                        endpoint_id=int(keep_id),
                        support_rank_mode=str(support_rank_mode),
                        random_seed=int(random_seed),
                        metapath_idx=int(metapath_idx),
                    )
                    if not ok:
                        selected[dst_t].add(int(keep_id))
            else:
                selected[dst_t].update(keep_ids)

    node_dict = {
        ntype: torch.tensor(sorted(node_ids), dtype=torch.int64)
        for ntype, node_ids in selected.items()
        if node_ids
    }
    seed_nodes = _seed_nodes_dict(target_ntype, target_node_id)
    subgraph = dgl.node_subgraph(graph, node_dict)
    inverse_indices = _find_seed_inverse_indices(subgraph, seed_nodes)
    return subgraph, inverse_indices


def _strict_kshot_rest_split(
    labels: np.ndarray,
    shot: int,
    split_seed: int,
    max_pool_size: int,
) -> dict:
    if shot <= 0:
        raise ValueError(f"shot must be positive, got {shot}")

    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    class_ids = sorted(np.unique(valid_labels).tolist())
    if len(class_ids) == 0:
        raise RuntimeError("No valid labeled target nodes found for strict k-shot split.")

    rng = np.random.default_rng(int(split_seed))

    train_pairs = []
    val_pairs = []
    test_pairs = []
    class_stats = {}

    for class_id in class_ids:
        node_ids = np.flatnonzero(labels == class_id).astype(np.int64)
        permuted = rng.permutation(node_ids)
        if max_pool_size > 0:
            candidate_pool = permuted[:max_pool_size]
        else:
            candidate_pool = permuted

        min_required = 2 * shot + 1
        if candidate_pool.size < min_required:
            raise RuntimeError(
                f"Class {class_id} only has {candidate_pool.size} candidates after max_pool_size={max_pool_size}; "
                f"need at least {min_required} for strict {shot}-shot train/val plus non-empty rest-test."
            )

        train_ids = candidate_pool[:shot]
        val_ids = candidate_pool[shot : 2 * shot]
        test_ids = candidate_pool[2 * shot :]

        train_pairs.extend((int(node_id), int(class_id)) for node_id in train_ids)
        val_pairs.extend((int(node_id), int(class_id)) for node_id in val_ids)
        test_pairs.extend((int(node_id), int(class_id)) for node_id in test_ids)

        class_stats[int(class_id)] = {
            "available": int(node_ids.size),
            "candidate_pool": int(candidate_pool.size),
            "train": int(train_ids.size),
            "val": int(val_ids.size),
            "test": int(test_ids.size),
        }

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    def _to_arrays(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        if not pairs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        node_ids = np.asarray([node_id for node_id, _ in pairs], dtype=np.int64)
        class_ids = np.asarray([class_id for _, class_id in pairs], dtype=np.int64)
        return node_ids, class_ids

    train_ids, train_labels = _to_arrays(train_pairs)
    val_ids, val_labels = _to_arrays(val_pairs)
    test_ids, test_labels = _to_arrays(test_pairs)

    return {
        "train_ids": train_ids,
        "train_labels": train_labels,
        "val_ids": val_ids,
        "val_labels": val_labels,
        "test_ids": test_ids,
        "test_labels": test_labels,
        "class_stats": class_stats,
        "num_classes": int(len(class_ids)),
    }


def _attach_peprompt_edge_features_from_global_pe(
    subgraph,
    spectral_embeddings: torch.Tensor | None,
    node_offsets: dict[str, int] | None,
    feature_name: str,
    edge_pe_tables: dict | None = None,
):
    for src_t, rel_t, dst_t in subgraph.canonical_etypes:
        etype = (src_t, rel_t, dst_t)
        if edge_pe_tables is not None and etype in edge_pe_tables and dgl.EID in subgraph.edges[etype].data:
            edge_ids = subgraph.edges[etype].data[dgl.EID].detach().cpu().long()
            subgraph.edges[etype].data[feature_name] = edge_pe_tables[etype][edge_ids]
            continue
        if spectral_embeddings is None or node_offsets is None:
            raise ValueError(
                f"Cannot attach PEPrompt edge features for etype={etype}: subgraph has no DGL EID "
                "and no node-level spectral fallback was provided."
            )
        src_local, dst_local = subgraph.edges(etype=etype)
        src_nids = subgraph.nodes[src_t].data[dgl.NID].detach().cpu().long()
        dst_nids = subgraph.nodes[dst_t].data[dgl.NID].detach().cpu().long()
        src_global = src_nids[src_local.detach().cpu().long()] + int(node_offsets[src_t])
        dst_global = dst_nids[dst_local.detach().cpu().long()] + int(node_offsets[dst_t])
        subgraph.edges[etype].data[feature_name] = (
            spectral_embeddings[src_global] - spectral_embeddings[dst_global]
        ).float()


def _build_global_edge_pe_tables(
    graph,
    spectral_embeddings: torch.Tensor,
    node_offsets: dict[str, int],
) -> dict:
    tables = {}
    for src_t, rel_t, dst_t in graph.canonical_etypes:
        etype = (src_t, rel_t, dst_t)
        src, dst = graph.edges(etype=etype)
        src_global = src.detach().cpu().long() + int(node_offsets[src_t])
        dst_global = dst.detach().cpu().long() + int(node_offsets[dst_t])
        tables[etype] = (
            spectral_embeddings[src_global] - spectral_embeddings[dst_global]
        ).float().contiguous()
    return tables


def _make_subgraph_extractor(graph, targetnode: str, args, spectral_payload: dict | None = None):
    subgraph_type = str(args.subgraph_type)
    if subgraph_type == "khop":
        hop_num = int(args.khop_num if args.khop_num is not None else HOP_NUM[args.dataset])

        def _extract(node_id: int):
            return extract_khop_subgraph(
                graph=graph,
                target_ntype=targetnode,
                target_node_id=node_id,
                hop_num=hop_num,
            )

        return _extract, {"subgraph_type": subgraph_type, "hop_num": hop_num}

    if subgraph_type == "fanout":
        fanout = [int(v) for v in args.fanouts]
        sampler = dgl.dataloading.NeighborSampler(fanout)

        def _extract(node_id: int):
            return extract_fanout_subgraph(
                graph=graph,
                target_ntype=targetnode,
                target_node_id=node_id,
                fanout=fanout,
                sampler=sampler,
            )

        return _extract, {"subgraph_type": subgraph_type, "fanout": fanout}

    if subgraph_type in {"metapath_topk", "metapath_topk_path", "metapath_topk_adapt", "metapath_topk_path_adapt"}:
        max_hop = int(args.metapath_max_hop)
        topk = int(args.metapath_topk)
        min_topk = int(getattr(args, "metapath_min_topk", 1))
        max_topk = int(getattr(args, "metapath_max_topk", topk))
        rel_threshold = float(getattr(args, "metapath_rel_threshold", 0.5))
        rank_metric = str(args.metapath_rank_metric)
        keep_self = bool(args.metapath_keep_self)
        endpoint_mode = str(getattr(args, "metapath_endpoint_mode", "all"))
        support_mode = _resolve_metapath_support_mode(
            subgraph_type=subgraph_type,
            endpoint_mode=endpoint_mode,
            support_mode=str(getattr(args, "metapath_support_mode", "auto")),
        )
        support_topk = int(getattr(args, "metapath_support_topk", 0) or 0)
        support_rank_mode = str(getattr(args, "metapath_support_rank_mode", "score"))
        random_seed = int(getattr(args, "metapath_random_seed", 0) or 0)
        preserve_paths = support_mode == "one_path"
        adaptive_mode = "relative" if subgraph_type in {"metapath_topk_adapt", "metapath_topk_path_adapt"} else "none"
        adjs = _build_csr_adjs(graph)
        adjs_csc = _build_csc_adjs(adjs) if preserve_paths else None
        metapaths = _filter_metapaths_by_endpoint_mode(
            _generate_metapaths(graph, targetnode, max_hop),
            target_ntype=targetnode,
            endpoint_mode=endpoint_mode,
        )
        if not metapaths:
            raise ValueError(
                f"No metapaths found for endpoint_mode={endpoint_mode}, "
                f"targetnode={targetnode}, max_hop={max_hop}."
            )
        endpoint_popularity = {}
        if rank_metric == "degree_norm":
            from scripts.subgraph_sampling_stats import _endpoint_popularity

            endpoint_popularity = {
                metapath: _endpoint_popularity(adjs, metapath)
                for metapath in metapaths
            }
        elif rank_metric == "count_idf":
            endpoint_popularity = {
                metapath: np.float32(_metapath_source_idf(adjs, metapath))
                for metapath in metapaths
            }

        def _extract(node_id: int):
            return extract_metapath_topk_subgraph(
                graph=graph,
                target_ntype=targetnode,
                target_node_id=node_id,
                adjs=adjs,
                metapaths=metapaths,
                topk=topk,
                rank_metric=rank_metric,
                keep_self=keep_self,
                endpoint_popularity=endpoint_popularity,
                preserve_paths=preserve_paths,
                adjs_csc=adjs_csc,
                adaptive_mode=adaptive_mode,
                min_topk=min_topk,
                max_topk=max_topk,
                rel_threshold=rel_threshold,
                support_mode=support_mode,
                support_topk=support_topk,
                support_rank_mode=support_rank_mode,
                random_seed=random_seed,
            )

        return _extract, {
            "subgraph_type": subgraph_type,
            "max_hop": max_hop,
            "topk": topk,
            "min_topk": min_topk,
            "max_topk": max_topk,
            "rel_threshold": rel_threshold,
            "adaptive_mode": adaptive_mode,
            "rank_metric": rank_metric,
            "keep_self": keep_self,
            "preserve_paths": preserve_paths,
            "endpoint_mode": endpoint_mode,
            "support_mode": support_mode,
            "support_topk": support_topk,
            "support_rank_mode": support_rank_mode,
            "random_seed": random_seed,
            "num_metapaths": len(metapaths),
        }

    raise ValueError(f"Unsupported subgraph_type: {subgraph_type}")


def _build_single_sample(
    extract_subgraph,
    edge_feature_payload: dict,
    node_id: int,
    label,
    feature_name: str,
    edge_pe_tables: dict | None = None,
):
    subgraph, inverse_indices = extract_subgraph(int(node_id))
    _attach_peprompt_edge_features_from_global_pe(
        subgraph=subgraph,
        spectral_embeddings=edge_feature_payload.get("spectral_embeddings"),
        node_offsets=edge_feature_payload.get("node_offsets"),
        feature_name=feature_name,
        edge_pe_tables=edge_pe_tables,
    )
    return (
        subgraph,
        inverse_indices,
        torch.as_tensor(label, dtype=torch.long).clone(),
    )


def _build_sample_list(
    extract_subgraph,
    edge_feature_payload: dict,
    node_ids: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
    edge_pe_tables: dict | None = None,
    sample_cache: dict[int, tuple] | None = None,
):
    samples = []
    for node_id, label in zip(node_ids, labels):
        node_id = int(node_id)
        if sample_cache is not None and node_id in sample_cache:
            sample = sample_cache[node_id]
        else:
            sample = _build_single_sample(
                extract_subgraph=extract_subgraph,
                edge_feature_payload=edge_feature_payload,
                node_id=node_id,
                label=label,
                feature_name=feature_name,
                edge_pe_tables=edge_pe_tables,
            )
            if sample_cache is not None:
                sample_cache[node_id] = sample
        samples.append(sample)
    return samples


def _build_job_args(cli_args, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        root=cli_args.root,
        dataset=dataset,
        feats_type=cli_args.feats_type,
        peprompt_edge_feature_names=cli_args.peprompt_edge_feature_names,
        peprompt_edge_feature_name=cli_args.peprompt_edge_feature_name,
        peprompt_spectral_cache_dir=cli_args.peprompt_spectral_cache_dir,
        peprompt_spectral_dim=cli_args.peprompt_spectral_dim,
        peprompt_spectral_max_nodes=cli_args.peprompt_spectral_max_nodes,
        peprompt_coarse_cache_dir=cli_args.peprompt_coarse_cache_dir,
        peprompt_coarse_supernodes_per_type=cli_args.peprompt_coarse_supernodes_per_type,
        peprompt_coarse_dim=cli_args.peprompt_coarse_dim,
        peprompt_coarse_hops=cli_args.peprompt_coarse_hops,
        peprompt_coarse_max_nodes=cli_args.peprompt_coarse_max_nodes,
        peprompt_coarse_walk_graph=cli_args.peprompt_coarse_walk_graph,
        peprompt_coarse_propagation=cli_args.peprompt_coarse_propagation,
        peprompt_coarse_seed=cli_args.peprompt_coarse_seed,
        peprompt_type_cache_dir=cli_args.peprompt_type_cache_dir,
        peprompt_type_hops=cli_args.peprompt_type_hops,
        peprompt_type_walk_graph=cli_args.peprompt_type_walk_graph,
        peprompt_type_propagation=cli_args.peprompt_type_propagation,
        peprompt_type_ppr_alpha=cli_args.peprompt_type_ppr_alpha,
        peprompt_type_heat_time=cli_args.peprompt_type_heat_time,
        peprompt_type_edge_onehot=cli_args.peprompt_type_edge_onehot,
        peprompt_cache_include_feature_key=cli_args.peprompt_cache_include_feature_key,
        peprompt_write_edge_feature_stats=False,
    )


def _precompute_dataset(graph, targetnode: str, edge_feature_payload: dict, args):
    labels = _target_labels(graph, targetnode)
    supervision_labels = _target_supervision_labels(graph, targetnode, args.dataset)
    extract_subgraph, subgraph_config = _make_subgraph_extractor(graph, targetnode, args, edge_feature_payload)
    edge_pe_tables = edge_feature_payload.get("edge_feature_table")
    if edge_pe_tables is None:
        edge_pe_tables = _build_global_edge_pe_tables(
            graph=graph,
            spectral_embeddings=edge_feature_payload["spectral_embeddings"],
            node_offsets=edge_feature_payload["node_offsets"],
        )
    sample_cache: dict[int, tuple] = {}

    for shot in args.shots:
        for split_seed in args.seeds:
            start_time = time.perf_counter()
            split = _strict_kshot_rest_split(
                labels=labels,
                shot=int(shot),
                split_seed=int(split_seed),
                max_pool_size=int(args.max_pool_size),
            )
            cache_size_before = len(sample_cache)

            train_list = _build_sample_list(
                extract_subgraph=extract_subgraph,
                edge_feature_payload=edge_feature_payload,
                node_ids=split["train_ids"],
                labels=supervision_labels[split["train_ids"]],
                feature_name=args.peprompt_edge_feature_name,
                edge_pe_tables=edge_pe_tables,
                sample_cache=sample_cache,
            )
            val_list = _build_sample_list(
                extract_subgraph=extract_subgraph,
                edge_feature_payload=edge_feature_payload,
                node_ids=split["val_ids"],
                labels=supervision_labels[split["val_ids"]],
                feature_name=args.peprompt_edge_feature_name,
                edge_pe_tables=edge_pe_tables,
                sample_cache=sample_cache,
            )
            test_list = _build_sample_list(
                extract_subgraph=extract_subgraph,
                edge_feature_payload=edge_feature_payload,
                node_ids=split["test_ids"],
                labels=supervision_labels[split["test_ids"]],
                feature_name=args.peprompt_edge_feature_name,
                edge_pe_tables=edge_pe_tables,
                sample_cache=sample_cache,
            )

            cache_path = build_peprompt_offline_cache_path(
                cache_dir=args.peprompt_offline_cache_dir,
                dataset_name=args.dataset,
                shot=shot,
                seed=split_seed,
                feats_type=args.feats_type,
                subgraph_type=_metapath_cache_key(args),
            )
            _ensure_dir(cache_path.parent)

            payload = {
                "dataset": args.dataset,
                "targetnode": targetnode,
                "subgraph_type": str(args.subgraph_type),
                "subgraph_cache_key": _metapath_cache_key(args),
                "subgraph_config": dict(subgraph_config),
                "shot": int(shot),
                "split_seed": int(split_seed),
                "feats_type": int(args.feats_type),
                "hop_num": subgraph_config.get("hop_num"),
                "fanout": subgraph_config.get("fanout"),
                "metapath_max_hop": subgraph_config.get("max_hop"),
                "metapath_topk": subgraph_config.get("topk"),
                "metapath_min_topk": subgraph_config.get("min_topk"),
                "metapath_max_topk": subgraph_config.get("max_topk"),
                "metapath_rel_threshold": subgraph_config.get("rel_threshold"),
                "metapath_adaptive_mode": subgraph_config.get("adaptive_mode"),
                "metapath_rank_metric": subgraph_config.get("rank_metric"),
                "metapath_keep_self": subgraph_config.get("keep_self"),
                "metapath_endpoint_mode": subgraph_config.get("endpoint_mode"),
                "metapath_support_mode": subgraph_config.get("support_mode"),
                "metapath_support_topk": subgraph_config.get("support_topk"),
                "metapath_support_rank_mode": subgraph_config.get("support_rank_mode"),
                "metapath_random_seed": subgraph_config.get("random_seed"),
                "metapath_count": subgraph_config.get("num_metapaths"),
                "max_pool_size": int(args.max_pool_size),
                "peprompt_edge_feature_name": args.peprompt_edge_feature_name,
                "peprompt_edge_feature_names": list(edge_feature_payload.get("selected_feature_names", PEPROMPT_EDGE_FEATURES)),
                "peprompt_edge_feature_dim": int(edge_feature_payload["feature_dim"]),
                "peprompt_edge_feature_slices": {
                    key: list(value)
                    for key, value in edge_feature_payload.get("feature_slices", {}).items()
                },
                "spectral_dim": int(edge_feature_payload.get("spectral_dim", 0) or 0),
                "spectral_cache_path": str(edge_feature_payload.get("spectral_cache_path", "")),
                "coarse_cache_path": str(edge_feature_payload.get("coarse_cache_path", "")),
                "coarse_supernode_count": edge_feature_payload.get("coarse_supernode_count"),
                "coarse_supernodes_per_type": edge_feature_payload.get("coarse_supernodes_per_type"),
                "coarse_hops": edge_feature_payload.get("coarse_hops"),
                "coarse_dim": edge_feature_payload.get("coarse_dim"),
                "coarse_highorder_dim": edge_feature_payload.get("coarse_highorder_dim"),
                "type_cache_path": str(edge_feature_payload.get("type_cache_path", "")),
                "type_hops": edge_feature_payload.get("type_hops"),
                "type_walk_graph": edge_feature_payload.get("type_walk_graph"),
                "type_neighborhood_dim": edge_feature_payload.get("type_neighborhood_dim"),
                "type_neighborhood_edge_dim": edge_feature_payload.get("type_neighborhood_edge_dim"),
                "train_ids": split["train_ids"],
                "train_labels": split["train_labels"],
                "val_ids": split["val_ids"],
                "val_labels": split["val_labels"],
                "test_ids": split["test_ids"],
                "test_labels": split["test_labels"],
                "class_stats": split["class_stats"],
                "num_classes": int(split["num_classes"]),
                "train": train_list,
                "val": val_list,
                "test": test_list,
            }

            with open(cache_path, "wb") as f:
                pk.dump(payload, f, protocol=pk.HIGHEST_PROTOCOL)
            split_ids_path = save_peprompt_split_ids(
                cache_dir=args.peprompt_offline_cache_dir,
                dataset_name=args.dataset,
                shot=shot,
                seed=split_seed,
                feats_type=args.feats_type,
                subgraph_type=_metapath_cache_key(args),
                split_payload=payload,
            )

            elapsed = time.perf_counter() - start_time
            print(
                f"[saved] dataset={args.dataset} subgraph={_metapath_cache_key(args)} shot={shot} seed={split_seed} "
                f"train={len(train_list)} val={len(val_list)} test={len(test_list)} "
                f"sample_cache={len(sample_cache)} new={len(sample_cache) - cache_size_before} "
                f"path={cache_path} split_ids={split_ids_path} time={elapsed:.2f}s"
            )


def build_parser():
    ap = argparse.ArgumentParser("Offline cache builder for strict k-shot PEPrompt splits")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Freebase"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--max_pool_size", type=int, default=400)
    ap.add_argument(
        "--subgraph_type",
        type=str,
        default="metapath_topk",
        choices=["khop", "fanout", "metapath_topk", "metapath_topk_path", "metapath_topk_adapt", "metapath_topk_path_adapt"],
    )
    ap.add_argument("--khop_num", type=int, default=None)
    ap.add_argument("--fanouts", nargs="+", type=int, default=[15, 10])
    ap.add_argument("--metapath_max_hop", type=int, default=3)
    ap.add_argument("--metapath_topk", type=int, default=5)
    ap.add_argument("--metapath_min_topk", type=int, default=1)
    ap.add_argument("--metapath_max_topk", type=int, default=5)
    ap.add_argument("--metapath_rel_threshold", type=float, default=0.5)
    ap.add_argument("--metapath_rank_metric", type=str, default="count", choices=["count", "degree_norm", "count_idf", "random"])
    ap.add_argument("--metapath_keep_self", action="store_true")
    ap.add_argument("--metapath_endpoint_mode", type=str, default="all", choices=["all", "target_closed"])
    ap.add_argument("--metapath_support_mode", type=str, default="auto", choices=["auto", "none", "one_path", "count"])
    ap.add_argument(
        "--metapath_support_topk",
        type=int,
        default=0,
        help="If >0, cap count-based recovered support nodes per metapath position.",
    )
    ap.add_argument(
        "--metapath_support_rank_mode",
        type=str,
        default="score",
        choices=["score", "random"],
        help="How to choose capped support nodes. score uses path contribution; random is a deterministic random ablation.",
    )
    ap.add_argument(
        "--metapath_random_seed",
        type=int,
        default=0,
        help="Seed for deterministic random endpoint/support-node ablations.",
    )
    ap.add_argument(
        "--peprompt_spectral_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_spectral_embeddings",
    )
    ap.add_argument(
        "--peprompt_offline_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_offline_splits",
    )
    ap.add_argument("--peprompt_edge_feature_names", nargs="*", default=None, choices=PEPROMPT_EDGE_FEATURES)
    ap.add_argument("--peprompt_spectral_dim", type=int, default=16)
    ap.add_argument("--peprompt_spectral_max_nodes", type=int, default=50000)
    ap.add_argument(
        "--peprompt_coarse_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_coarse_highorder",
    )
    ap.add_argument("--peprompt_coarse_supernodes_per_type", type=int, default=16)
    ap.add_argument("--peprompt_coarse_dim", type=int, default=8)
    ap.add_argument("--peprompt_coarse_hops", nargs="*", type=int, default=list(PEPROMPT_DEFAULT_COARSE_HOPS))
    ap.add_argument("--peprompt_coarse_max_nodes", type=int, default=50000)
    ap.add_argument("--peprompt_coarse_walk_graph", type=str, default="undirected", choices=["undirected", "directed"])
    ap.add_argument("--peprompt_coarse_propagation", type=str, default="coarse", choices=["coarse"])
    ap.add_argument("--peprompt_coarse_seed", type=int, default=0)
    ap.add_argument(
        "--peprompt_type_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_type_neighborhood",
    )
    ap.add_argument("--peprompt_type_hops", nargs="*", type=int, default=list(PEPROMPT_DEFAULT_TYPE_HOPS))
    ap.add_argument("--peprompt_type_walk_graph", type=str, default="undirected", choices=["undirected", "directed"])
    ap.add_argument("--peprompt_type_propagation", type=str, default="power", choices=["power", "ppr", "heat"])
    ap.add_argument("--peprompt_type_ppr_alpha", type=float, default=0.15)
    ap.add_argument("--peprompt_type_heat_time", type=float, default=1.0)
    ap.add_argument("--peprompt_type_edge_onehot", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--peprompt_cache_include_feature_key", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--peprompt_edge_feature_name", type=str, default=PEPROMPT_EDGE_FEATURE_NAME)
    return ap


def main():
    args = build_parser().parse_args()

    for dataset in args.datasets:
        if dataset not in HOP_NUM:
            raise ValueError(f"Unsupported dataset: {dataset}")

        args.dataset = dataset
        job_args = _build_job_args(args, dataset)

        print(f"[dataset] {dataset} feats_type={args.feats_type} subgraph={args.subgraph_type}")
        graph, targetnode = _load_raw_heterograph(args.root, dataset, args.feats_type)
        edge_feature_payload = prepare_peprompt_edge_feature_table(job_args, wandb_run=None)
        print(
            f"[edge-features] dataset={dataset} features={edge_feature_payload.get('selected_feature_names')} "
            f"dim={edge_feature_payload['feature_dim']} "
            f"spectral_cache={edge_feature_payload.get('spectral_cache_path')} "
            f"coarse_cache={edge_feature_payload.get('coarse_cache_path')}"
        )
        _precompute_dataset(graph, targetnode, edge_feature_payload, args)


if __name__ == "__main__":
    main()
