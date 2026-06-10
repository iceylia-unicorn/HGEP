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
import pickle as pk
import time
from types import SimpleNamespace

import dgl
import numpy as np
import torch

from gpbench.downstream.fewshot import build_peprompt_offline_cache_path
from scripts.peprompt_benchmark import (
    HOP_NUM,
    PEPROMPT_EDGE_FEATURE_NAME,
    PEPROMPT_EDGE_FEATURES,
    _load_raw_heterograph,
    prepare_peprompt_spectral_payload,
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


def _target_labels(graph, targetnode: str) -> np.ndarray:
    labels = graph.ndata["y"][targetnode].detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
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
    if str(args.subgraph_type) != "metapath_topk":
        return str(args.subgraph_type)
    metric = str(args.metapath_rank_metric)
    suffix = f"m{int(args.metapath_max_hop)}_k{int(args.metapath_topk)}_{metric}"
    if bool(args.metapath_keep_self):
        suffix += "_self"
    fusion_mode = str(getattr(args, "peprompt_fusion_mode", "none"))
    ctx_dim = int(getattr(args, "peprompt_ctx_dim", 0) or 0)
    if fusion_mode == "hop_decoupled" and ctx_dim > 0:
        suffix += f"_mpvirt2_d{ctx_dim}"
    elif fusion_mode == "onehop_ctx" and ctx_dim > 0:
        suffix += f"_onehop_d{ctx_dim}"
    elif fusion_mode == "type_ctx" and ctx_dim > 0:
        suffix += f"_typectx_d{ctx_dim}"
    elif fusion_mode == "graph_summary":
        suffix += "_graphsum"
    elif fusion_mode == "graph_summary_basis":
        suffix += "_graphsum"
    return f"metapath_topk_{suffix}"


def _gather_global_node_feats(graph, ntypes) -> dict[str, np.ndarray] | None:
    """Extract raw node features as numpy arrays, keyed by node type."""
    feats = {}
    for ntype in ntypes:
        x = graph.nodes[ntype].data.get("x")
        if x is None:
            continue
        if isinstance(x, torch.Tensor):
            feats[ntype] = x.detach().cpu().numpy().astype(np.float32)
        else:
            feats[ntype] = np.asarray(x, dtype=np.float32)
    return feats if feats else None


def _coerce_feature_dim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    dim = int(dim)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim]
    out = np.zeros((dim,), dtype=np.float32)
    out[: x.shape[0]] = x
    return out


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
    global_feats: dict | None = None,
    ctx_dim: int = 0,
    context_mode: str = "none",
):
    selected: dict[str, set[int]] = {ntype: set() for ntype in graph.ntypes}
    selected[target_ntype].add(int(target_node_id))

    dropped_metapath_ctx: list[torch.Tensor] = []
    dropped_metapath_stats: list[list[float]] = []
    kept_metapath_assignments: dict[str, set[tuple[int, int]]] = {}
    onehop_scores_by_type: dict[str, np.ndarray] = {}
    onehop_keep_by_type: dict[str, set[int]] = {}
    type_scores_by_type: dict[str, np.ndarray] = {}
    type_keep_by_type: dict[str, set[int]] = {}
    summary_keep_mass_by_type: dict[str, float] = {}
    summary_drop_mass_by_type: dict[str, float] = {}
    summary_total_mass_by_hop = np.zeros(max((len(m) for m in metapaths), default=1), dtype=np.float32)
    summary_drop_mass_by_hop = np.zeros(max((len(m) for m in metapaths), default=1), dtype=np.float32)
    ctx_dim = int(ctx_dim or 0)
    _compute_metapath_ctx = (
        context_mode == "hidden_virtual_metapath"
        and global_feats is not None
        and ctx_dim > 0
    )
    _compute_onehop_ctx = (
        context_mode == "onehop_type_pooled"
        and global_feats is not None
        and ctx_dim > 0
    )
    _compute_type_ctx = (
        context_mode == "type_pooled"
        and global_feats is not None
        and ctx_dim > 0
    )
    _compute_graph_summary = context_mode == "graph_summary"
    max_path_len = max((len(metapath) for metapath in metapaths), default=1)

    for metapath_idx, metapath in enumerate(metapaths):
        scores = _metapath_reachable_scores(
            adjs=adjs,
            metapath=metapath,
            node_id=int(target_node_id),
            target_ntype=target_ntype,
            keep_self=bool(keep_self),
        )
        ranks = _rank_values(scores, endpoint_popularity.get(metapath), rank_metric)
        keep = _topk_indices(scores, ranks, int(topk))
        if keep.size > 0:
            dst_t = metapath[-1][2]
            keep_ids = [int(v) for v in keep.tolist()]
            selected[dst_t].update(keep_ids)
            assignments = kept_metapath_assignments.setdefault(dst_t, set())
            for keep_id in keep_ids:
                assignments.add((keep_id, metapath_idx))

        if _compute_onehop_ctx and len(metapath) == 1:
            dst_t = metapath[-1][2]
            scores_np = np.asarray(scores, dtype=np.float32)
            if dst_t not in onehop_scores_by_type:
                onehop_scores_by_type[dst_t] = scores_np.copy()
            else:
                onehop_scores_by_type[dst_t] += scores_np
            onehop_keep_by_type.setdefault(dst_t, set()).update(int(v) for v in keep.tolist())

        if _compute_type_ctx:
            dst_t = metapath[-1][2]
            scores_np = np.asarray(scores, dtype=np.float32)
            if dst_t not in type_scores_by_type:
                type_scores_by_type[dst_t] = scores_np.copy()
            else:
                type_scores_by_type[dst_t] += scores_np
            type_keep_by_type.setdefault(dst_t, set()).update(int(v) for v in keep.tolist())

        if _compute_graph_summary:
            dst_t = metapath[-1][2]
            nonzero = np.flatnonzero(scores > 0)
            dropped = np.setdiff1d(nonzero, keep)
            if dst_t == target_ntype:
                dropped = dropped[dropped != int(target_node_id)]
            keep_score_sum = float(np.asarray(scores[keep], dtype=np.float32).sum()) if keep.size > 0 else 0.0
            drop_score_sum = float(np.asarray(scores[dropped], dtype=np.float32).sum()) if dropped.size > 0 else 0.0
            summary_keep_mass_by_type[dst_t] = summary_keep_mass_by_type.get(dst_t, 0.0) + keep_score_sum
            summary_drop_mass_by_type[dst_t] = summary_drop_mass_by_type.get(dst_t, 0.0) + drop_score_sum
            hop_idx = max(int(len(metapath)) - 1, 0)
            summary_total_mass_by_hop[hop_idx] += keep_score_sum + drop_score_sum
            summary_drop_mass_by_hop[hop_idx] += drop_score_sum

        if not _compute_metapath_ctx:
            continue

        dst_t = metapath[-1][2]
        zero_ctx = torch.zeros(ctx_dim, dtype=torch.float32)
        zero_stats = [0.0, 0.0, 0.0, float(len(metapath)) / float(max_path_len)]
        if dst_t not in global_feats:
            dropped_metapath_ctx.append(zero_ctx)
            dropped_metapath_stats.append(zero_stats)
            continue

        nonzero = np.flatnonzero(scores > 0)
        dropped = np.setdiff1d(nonzero, keep)
        # Exclude the centre node itself from "dropped" for same-type metapaths
        if dst_t == target_ntype:
            dropped = dropped[dropped != int(target_node_id)]

        X_dst = global_feats[dst_t]  # ndarray [N_dst, D]
        if dropped.size > 0:
            weights = np.asarray(scores[dropped], dtype=np.float32)
            weight_sum = float(weights.sum())
            if weight_sum > 0.0:
                ctx_np = (np.asarray(X_dst[dropped], dtype=np.float32) * weights[:, None]).sum(axis=0)
                ctx_np = ctx_np / weight_sum
                ctx = torch.from_numpy(_coerce_feature_dim(ctx_np, ctx_dim))
            else:
                ctx = zero_ctx
        else:
            weight_sum = 0.0
            ctx = zero_ctx

        keep_score_sum = float(np.asarray(scores[keep], dtype=np.float32).sum()) if keep.size > 0 else 0.0
        total_score_sum = keep_score_sum + weight_sum
        drop_ratio = weight_sum / total_score_sum if total_score_sum > 0.0 else 0.0
        dropped_metapath_ctx.append(ctx)
        dropped_metapath_stats.append(
            [
                float(np.log1p(dropped.size)),
                float(np.log1p(weight_sum)),
                float(drop_ratio),
                float(len(metapath)) / float(max_path_len),
            ]
        )

    node_dict = {
        ntype: torch.tensor(sorted(node_ids), dtype=torch.int64)
        for ntype, node_ids in selected.items()
        if node_ids
    }
    seed_nodes = _seed_nodes_dict(target_ntype, target_node_id)
    subgraph = dgl.node_subgraph(graph, node_dict)
    inverse_indices = _find_seed_inverse_indices(subgraph, seed_nodes)

    if dropped_metapath_ctx:
        payload = {
            "ctx": torch.stack(dropped_metapath_ctx, dim=0),
            "stats": torch.tensor(dropped_metapath_stats, dtype=torch.float32),
        }
        if kept_metapath_assignments:
            payload["kept_assignments"] = {
                ntype: {
                    "node_ids": torch.tensor([node_id for node_id, _ in pairs], dtype=torch.int64),
                    "metapath_ids": torch.tensor([mp_idx for _, mp_idx in pairs], dtype=torch.int64),
                }
                for ntype, pairs in (
                    (ntype, sorted(assignments))
                    for ntype, assignments in kept_metapath_assignments.items()
                    if assignments
                )
            }
        inverse_indices["_dropped_metapath_ctx"] = payload

    if _compute_onehop_ctx:
        zero_stats = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        onehop_payload = {}
        for dst_t, scores in onehop_scores_by_type.items():
            if dst_t not in global_feats:
                onehop_payload[dst_t] = {
                    "ctx": torch.zeros(ctx_dim, dtype=torch.float32),
                    "stats": zero_stats.clone(),
                }
                continue

            keep_ids = np.array(sorted(onehop_keep_by_type.get(dst_t, set())), dtype=np.int64)
            nonzero = np.flatnonzero(scores > 0)
            dropped = np.setdiff1d(nonzero, keep_ids)
            if dst_t == target_ntype:
                dropped = dropped[dropped != int(target_node_id)]

            X_dst = global_feats[dst_t]
            zero_ctx = torch.zeros(ctx_dim, dtype=torch.float32)
            if dropped.size > 0:
                weights = np.asarray(scores[dropped], dtype=np.float32)
                weight_sum = float(weights.sum())
                if weight_sum > 0.0:
                    ctx_np = (np.asarray(X_dst[dropped], dtype=np.float32) * weights[:, None]).sum(axis=0)
                    ctx_np = ctx_np / weight_sum
                    ctx = torch.from_numpy(_coerce_feature_dim(ctx_np, ctx_dim))
                else:
                    ctx = zero_ctx
            else:
                weight_sum = 0.0
                ctx = zero_ctx

            keep_score_sum = float(np.asarray(scores[keep_ids], dtype=np.float32).sum()) if keep_ids.size > 0 else 0.0
            total_score_sum = keep_score_sum + weight_sum
            drop_ratio = weight_sum / total_score_sum if total_score_sum > 0.0 else 0.0
            onehop_payload[dst_t] = {
                "ctx": ctx,
                "stats": torch.tensor(
                    [
                        float(np.log1p(dropped.size)),
                        float(np.log1p(weight_sum)),
                        float(drop_ratio),
                        1.0,
                    ],
                    dtype=torch.float32,
                ),
            }
        if onehop_payload:
            inverse_indices["_dropped_onehop_ctx"] = onehop_payload

    if _compute_type_ctx:
        type_payload = {}
        for dst_t, scores in type_scores_by_type.items():
            if dst_t not in global_feats:
                type_payload[dst_t] = torch.zeros(ctx_dim, dtype=torch.float32)
                continue

            keep_ids = np.array(sorted(type_keep_by_type.get(dst_t, set())), dtype=np.int64)
            nonzero = np.flatnonzero(scores > 0)
            dropped = np.setdiff1d(nonzero, keep_ids)
            if dst_t == target_ntype:
                dropped = dropped[dropped != int(target_node_id)]

            X_dst = global_feats[dst_t]
            zero_ctx = torch.zeros(ctx_dim, dtype=torch.float32)
            if dropped.size > 0:
                weights = np.asarray(scores[dropped], dtype=np.float32)
                weight_sum = float(weights.sum())
                if weight_sum > 0.0:
                    ctx_np = (np.asarray(X_dst[dropped], dtype=np.float32) * weights[:, None]).sum(axis=0)
                    ctx_np = ctx_np / weight_sum
                    ctx = torch.from_numpy(_coerce_feature_dim(ctx_np, ctx_dim))
                else:
                    ctx = zero_ctx
            else:
                ctx = zero_ctx
            type_payload[dst_t] = ctx

        if type_payload:
            inverse_indices["_dropped_ctx"] = type_payload

    if _compute_graph_summary:
        summary_values: list[float] = []
        for dst_t in graph.ntypes:
            keep_mass = float(summary_keep_mass_by_type.get(dst_t, 0.0))
            drop_mass = float(summary_drop_mass_by_type.get(dst_t, 0.0))
            total_mass = keep_mass + drop_mass
            drop_ratio = drop_mass / total_mass if total_mass > 0.0 else 0.0
            summary_values.extend([
                float(np.log1p(keep_mass)),
                float(np.log1p(drop_mass)),
                float(drop_ratio),
            ])
        for hop_idx in range(int(max_path_len)):
            total_mass = float(summary_total_mass_by_hop[hop_idx])
            drop_mass = float(summary_drop_mass_by_hop[hop_idx])
            drop_ratio = drop_mass / total_mass if total_mass > 0.0 else 0.0
            summary_values.extend([
                float(np.log1p(total_mass)),
                float(drop_ratio),
            ])
        inverse_indices["_graph_metapath_summary"] = torch.tensor(summary_values, dtype=torch.float32)

    return subgraph, inverse_indices


def _restore_dropped_ctx_in_sample(
    sample,
    target_ntype: str,
    dropped_ctx_by_target: dict[int, dict] | None = None,
    dropped_onehop_ctx_by_target: dict[int, dict] | None = None,
):
    """Reconstruct padded node-feature context from its compact serialised form."""
    subgraph, inverse_indices, _label = sample
    dropped_metapath_ctx = inverse_indices.pop("_dropped_metapath_ctx", None)
    dropped_onehop_ctx = inverse_indices.pop("_dropped_onehop_ctx", None)
    graph_metapath_summary = inverse_indices.pop("_graph_metapath_summary", None)
    centre_local = int(inverse_indices[target_ntype][0].item())
    centre_global = int(subgraph.nodes[target_ntype].data[dgl.NID][centre_local].item())
    if dropped_metapath_ctx is None and dropped_ctx_by_target:
        dropped_metapath_ctx = dropped_ctx_by_target.get(centre_global)
    if dropped_onehop_ctx is None and dropped_onehop_ctx_by_target:
        dropped_onehop_ctx = dropped_onehop_ctx_by_target.get(centre_global)
    if dropped_metapath_ctx:
        ctx = dropped_metapath_ctx["ctx"].float()
        stats = dropped_metapath_ctx["stats"].float()
        num_metapaths = int(ctx.shape[0])
        target_ctx_mat = torch.zeros(
            subgraph.num_nodes(target_ntype),
            int(ctx.shape[0]),
            int(ctx.shape[1]),
            dtype=torch.float32,
        )
        target_stats_mat = torch.zeros(
            subgraph.num_nodes(target_ntype),
            int(stats.shape[0]),
            int(stats.shape[1]),
            dtype=torch.float32,
        )
        target_ctx_mat[centre_local] = ctx
        target_stats_mat[centre_local] = stats
        subgraph.nodes[target_ntype].data["dropped_metapath_ctx"] = target_ctx_mat
        subgraph.nodes[target_ntype].data["dropped_metapath_stats"] = target_stats_mat

        keep_masks: dict[str, torch.Tensor] = {
            ntype: torch.zeros(subgraph.num_nodes(ntype), num_metapaths, dtype=torch.bool)
            for ntype in subgraph.ntypes
        }
        keep_masks[target_ntype][centre_local] = True
        kept_assignments = dropped_metapath_ctx.get("kept_assignments") or {}
        for ntype, assignment_payload in kept_assignments.items():
            node_ids = assignment_payload.get("node_ids")
            metapath_ids = assignment_payload.get("metapath_ids")
            if node_ids is None or metapath_ids is None:
                continue

            subgraph_nids = subgraph.nodes[ntype].data[dgl.NID].detach().cpu().long().tolist()
            local_by_global = {int(global_id): idx for idx, global_id in enumerate(subgraph_nids)}
            for global_id, metapath_idx in zip(
                node_ids.detach().cpu().long().tolist(),
                metapath_ids.detach().cpu().long().tolist(),
            ):
                local_idx = local_by_global.get(int(global_id))
                if local_idx is None:
                    continue
                keep_masks[ntype][local_idx, metapath_idx] = True

        for ntype, keep_mask in keep_masks.items():
            if keep_mask.any():
                subgraph.nodes[ntype].data["dropped_metapath_keep_mask"] = keep_mask
    if dropped_onehop_ctx:
        for dst_t, payload in dropped_onehop_ctx.items():
            ctx_vec = payload["ctx"].float()
            stats_vec = payload["stats"].float()
            ctx_dim = int(ctx_vec.shape[0])
            ctx_key = f"dropped_onehop_ctx_{dst_t}"
            stats_key = f"dropped_onehop_stats_{dst_t}"
            ctx_mat = torch.zeros(subgraph.num_nodes(target_ntype), ctx_dim, dtype=torch.float32)
            stats_mat = torch.zeros(subgraph.num_nodes(target_ntype), int(stats_vec.shape[0]), dtype=torch.float32)
            ctx_mat[centre_local] = ctx_vec
            stats_mat[centre_local] = stats_vec
            subgraph.nodes[target_ntype].data[ctx_key] = ctx_mat
            subgraph.nodes[target_ntype].data[stats_key] = stats_mat
    if graph_metapath_summary is not None:
        summary_vec = graph_metapath_summary.float()
        summary_mat = torch.zeros(
            subgraph.num_nodes(target_ntype),
            int(summary_vec.shape[0]),
            dtype=torch.float32,
        )
        summary_mat[centre_local] = summary_vec
        subgraph.nodes[target_ntype].data["graph_metapath_summary"] = summary_mat

    # Backward compatibility for older caches that stored one mean vector per
    # destination node type.
    dropped_ctx = inverse_indices.pop("_dropped_ctx", None)
    if not dropped_ctx:
        return
    for dst_t, ctx_vec in dropped_ctx.items():
        ctx_dim = int(ctx_vec.shape[0])
        key = f"dropped_ctx_{dst_t}"
        mat = torch.zeros(subgraph.num_nodes(target_ntype), ctx_dim, dtype=torch.float32)
        mat[centre_local] = ctx_vec
        subgraph.nodes[target_ntype].data[key] = mat


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
    spectral_embeddings: torch.Tensor,
    node_offsets: dict[str, int],
    feature_name: str,
):
    for src_t, rel_t, dst_t in subgraph.canonical_etypes:
        etype = (src_t, rel_t, dst_t)
        src_local, dst_local = subgraph.edges(etype=etype)
        src_nids = subgraph.nodes[src_t].data[dgl.NID].detach().cpu().long()
        dst_nids = subgraph.nodes[dst_t].data[dgl.NID].detach().cpu().long()
        src_global = src_nids[src_local.detach().cpu().long()] + int(node_offsets[src_t])
        dst_global = dst_nids[dst_local.detach().cpu().long()] + int(node_offsets[dst_t])
        subgraph.edges[etype].data[feature_name] = (
            spectral_embeddings[src_global] - spectral_embeddings[dst_global]
        ).float()


def _make_subgraph_extractor(graph, targetnode: str, args):
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

    if subgraph_type == "metapath_topk":
        max_hop = int(args.metapath_max_hop)
        topk = int(args.metapath_topk)
        rank_metric = str(args.metapath_rank_metric)
        keep_self = bool(args.metapath_keep_self)
        ctx_dim = int(getattr(args, "peprompt_ctx_dim", 0) or 0)
        fusion_mode = str(getattr(args, "peprompt_fusion_mode", "none"))
        adjs = _build_csr_adjs(graph)
        metapaths = _generate_metapaths(graph, targetnode, max_hop)
        endpoint_popularity = {}
        if rank_metric == "degree_norm":
            from scripts.subgraph_sampling_stats import _endpoint_popularity

            endpoint_popularity = {
                metapath: _endpoint_popularity(adjs, metapath)
                for metapath in metapaths
            }

        # Pre-extract global node features ONLY when hop_decoupled mode is
        # enabled — otherwise we skip the per-subgraph ctx computation entirely.
        _global_feats = None
        context_mode = "none"
        if fusion_mode == "hop_decoupled" and ctx_dim > 0:
            context_mode = "hidden_virtual_metapath"
            _global_feats = _gather_global_node_feats(graph, graph.ntypes)
        elif fusion_mode == "onehop_ctx" and ctx_dim > 0:
            context_mode = "onehop_type_pooled"
            _global_feats = _gather_global_node_feats(graph, graph.ntypes)
        elif fusion_mode == "type_ctx" and ctx_dim > 0:
            context_mode = "type_pooled"
            _global_feats = _gather_global_node_feats(graph, graph.ntypes)
        elif fusion_mode in {"graph_summary", "graph_summary_basis"}:
            context_mode = "graph_summary"

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
                global_feats=_global_feats,
                ctx_dim=ctx_dim,
                context_mode=context_mode,
            )

        return _extract, {
            "subgraph_type": subgraph_type,
            "max_hop": max_hop,
            "topk": topk,
            "rank_metric": rank_metric,
            "keep_self": keep_self,
            "num_metapaths": len(metapaths),
            "context_mode": context_mode,
            "ctx_dim": ctx_dim,
            "graph_summary_dim": (3 * len(graph.ntypes) + 2 * int(max_hop)) if context_mode == "graph_summary" else 0,
        }

    raise ValueError(f"Unsupported subgraph_type: {subgraph_type}")


def _build_single_sample(
    extract_subgraph,
    spectral_payload: dict,
    node_id: int,
    label: int,
    feature_name: str,
):
    subgraph, inverse_indices = extract_subgraph(int(node_id))
    dropped_ctx_payload = inverse_indices.pop("_dropped_metapath_ctx", None)
    dropped_onehop_ctx_payload = inverse_indices.pop("_dropped_onehop_ctx", None)
    _attach_peprompt_edge_features_from_global_pe(
        subgraph=subgraph,
        spectral_embeddings=spectral_payload["spectral_embeddings"],
        node_offsets=spectral_payload["node_offsets"],
        feature_name=feature_name,
    )
    return (
        subgraph,
        inverse_indices,
        torch.tensor(int(label)),
    ), dropped_ctx_payload, dropped_onehop_ctx_payload


def _build_sample_list(
    extract_subgraph,
    spectral_payload: dict,
    node_ids: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
):
    samples = []
    dropped_ctx_by_target: dict[int, dict] = {}
    dropped_onehop_ctx_by_target: dict[int, dict] = {}
    for node_id, label in zip(node_ids, labels):
        sample, dropped_ctx_payload, dropped_onehop_ctx_payload = _build_single_sample(
            extract_subgraph=extract_subgraph,
            spectral_payload=spectral_payload,
            node_id=int(node_id),
            label=int(label),
            feature_name=feature_name,
        )
        samples.append(sample)
        if dropped_ctx_payload is not None:
            dropped_ctx_by_target[int(node_id)] = dropped_ctx_payload
        if dropped_onehop_ctx_payload is not None:
            dropped_onehop_ctx_by_target[int(node_id)] = dropped_onehop_ctx_payload
    return samples, dropped_ctx_by_target, dropped_onehop_ctx_by_target


def _build_job_args(cli_args, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        root=cli_args.root,
        dataset=dataset,
        feats_type=cli_args.feats_type,
        peprompt_spectral_cache_dir=cli_args.peprompt_spectral_cache_dir,
        peprompt_spectral_dim=cli_args.peprompt_spectral_dim,
        peprompt_spectral_max_nodes=cli_args.peprompt_spectral_max_nodes,
    )


def _precompute_dataset(graph, targetnode: str, spectral_payload: dict, args):
    labels = _target_labels(graph, targetnode)
    extract_subgraph, subgraph_config = _make_subgraph_extractor(graph, targetnode, args)

    for shot in args.shots:
        for split_seed in args.seeds:
            start_time = time.perf_counter()
            split = _strict_kshot_rest_split(
                labels=labels,
                shot=int(shot),
                split_seed=int(split_seed),
                max_pool_size=int(args.max_pool_size),
            )

            train_list, train_dropped_ctx, train_onehop_ctx = _build_sample_list(
                extract_subgraph=extract_subgraph,
                spectral_payload=spectral_payload,
                node_ids=split["train_ids"],
                labels=split["train_labels"],
                feature_name=args.peprompt_edge_feature_name,
            )
            val_list, val_dropped_ctx, val_onehop_ctx = _build_sample_list(
                extract_subgraph=extract_subgraph,
                spectral_payload=spectral_payload,
                node_ids=split["val_ids"],
                labels=split["val_labels"],
                feature_name=args.peprompt_edge_feature_name,
            )
            test_list, test_dropped_ctx, test_onehop_ctx = _build_sample_list(
                extract_subgraph=extract_subgraph,
                spectral_payload=spectral_payload,
                node_ids=split["test_ids"],
                labels=split["test_labels"],
                feature_name=args.peprompt_edge_feature_name,
            )
            dropped_ctx_by_target = {}
            dropped_ctx_by_target.update(train_dropped_ctx)
            dropped_ctx_by_target.update(val_dropped_ctx)
            dropped_ctx_by_target.update(test_dropped_ctx)
            dropped_onehop_ctx_by_target = {}
            dropped_onehop_ctx_by_target.update(train_onehop_ctx)
            dropped_onehop_ctx_by_target.update(val_onehop_ctx)
            dropped_onehop_ctx_by_target.update(test_onehop_ctx)

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
                "metapath_rank_metric": subgraph_config.get("rank_metric"),
                "metapath_keep_self": subgraph_config.get("keep_self"),
                "metapath_count": subgraph_config.get("num_metapaths"),
                "metapath_context_mode": subgraph_config.get("context_mode"),
                "peprompt_ctx_dim": subgraph_config.get("ctx_dim"),
                "peprompt_graph_summary_dim": subgraph_config.get("graph_summary_dim", 0),
                "dropped_metapath_context_by_target": dropped_ctx_by_target,
                "dropped_onehop_context_by_target": dropped_onehop_ctx_by_target,
                "max_pool_size": int(args.max_pool_size),
                "peprompt_edge_feature_name": args.peprompt_edge_feature_name,
                "peprompt_edge_feature_names": list(PEPROMPT_EDGE_FEATURES),
                "peprompt_edge_feature_dim": int(spectral_payload["spectral_dim"]),
                "spectral_dim": int(spectral_payload["spectral_dim"]),
                "spectral_cache_path": str(spectral_payload.get("spectral_cache_path", "")),
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

            elapsed = time.perf_counter() - start_time
            print(
                f"[saved] dataset={args.dataset} subgraph={_metapath_cache_key(args)} shot={shot} seed={split_seed} "
                f"train={len(train_list)} val={len(val_list)} test={len(test_list)} "
                f"path={cache_path} time={elapsed:.2f}s"
            )


def build_parser():
    ap = argparse.ArgumentParser("Offline cache builder for strict k-shot PEPrompt splits")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Freebase"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--max_pool_size", type=int, default=400)
    ap.add_argument("--subgraph_type", type=str, default="metapath_topk", choices=["khop", "fanout", "metapath_topk"])
    ap.add_argument("--khop_num", type=int, default=None)
    ap.add_argument("--fanouts", nargs="+", type=int, default=[15, 10])
    ap.add_argument("--metapath_max_hop", type=int, default=3)
    ap.add_argument("--metapath_topk", type=int, default=5)
    ap.add_argument("--metapath_rank_metric", type=str, default="count", choices=["count", "degree_norm"])
    ap.add_argument("--metapath_keep_self", action="store_true")
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
    ap.add_argument("--peprompt_spectral_dim", type=int, default=16)
    ap.add_argument("--peprompt_spectral_max_nodes", type=int, default=50000)
    ap.add_argument("--peprompt_edge_feature_name", type=str, default=PEPROMPT_EDGE_FEATURE_NAME)
    ap.add_argument("--peprompt_fusion_mode", type=str, default="none", choices=["none", "hop_decoupled", "onehop_ctx", "type_ctx", "graph_summary", "graph_summary_basis"])
    ap.add_argument("--peprompt_ctx_dim", type=int, default=0)
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
        spectral_payload, cache_path, cache_hit = prepare_peprompt_spectral_payload(job_args)
        print(
            f"[spectral] dataset={dataset} dim={spectral_payload['spectral_dim']} "
            f"cache_hit={cache_hit} path={cache_path}"
        )
        _precompute_dataset(graph, targetnode, spectral_payload, args)


if __name__ == "__main__":
    main()
