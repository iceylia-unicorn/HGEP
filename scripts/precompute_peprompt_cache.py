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
    return f"metapath_topk_{suffix}"


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
):
    selected: dict[str, set[int]] = {ntype: set() for ntype in graph.ntypes}
    selected[target_ntype].add(int(target_node_id))

    for metapath in metapaths:
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
            selected[metapath[-1][2]].update(int(v) for v in keep.tolist())

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
        adjs = _build_csr_adjs(graph)
        metapaths = _generate_metapaths(graph, targetnode, max_hop)
        endpoint_popularity = {}
        if rank_metric == "degree_norm":
            from scripts.subgraph_sampling_stats import _endpoint_popularity

            endpoint_popularity = {
                metapath: _endpoint_popularity(adjs, metapath)
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
            )

        return _extract, {
            "subgraph_type": subgraph_type,
            "max_hop": max_hop,
            "topk": topk,
            "rank_metric": rank_metric,
            "keep_self": keep_self,
            "num_metapaths": len(metapaths),
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
    _attach_peprompt_edge_features_from_global_pe(
        subgraph=subgraph,
        spectral_embeddings=spectral_payload["spectral_embeddings"],
        node_offsets=spectral_payload["node_offsets"],
        feature_name=feature_name,
    )
    return (subgraph, inverse_indices, torch.tensor(int(label)))


def _build_sample_list(
    extract_subgraph,
    spectral_payload: dict,
    node_ids: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
):
    return [
        _build_single_sample(
            extract_subgraph=extract_subgraph,
            spectral_payload=spectral_payload,
            node_id=int(node_id),
            label=int(label),
            feature_name=feature_name,
        )
        for node_id, label in zip(node_ids, labels)
    ]


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

            train_list = _build_sample_list(
                extract_subgraph=extract_subgraph,
                spectral_payload=spectral_payload,
                node_ids=split["train_ids"],
                labels=split["train_labels"],
                feature_name=args.peprompt_edge_feature_name,
            )
            val_list = _build_sample_list(
                extract_subgraph=extract_subgraph,
                spectral_payload=spectral_payload,
                node_ids=split["val_ids"],
                labels=split["val_labels"],
                feature_name=args.peprompt_edge_feature_name,
            )
            test_list = _build_sample_list(
                extract_subgraph=extract_subgraph,
                spectral_payload=spectral_payload,
                node_ids=split["test_ids"],
                labels=split["test_labels"],
                feature_name=args.peprompt_edge_feature_name,
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
                "metapath_rank_metric": subgraph_config.get("rank_metric"),
                "metapath_keep_self": subgraph_config.get("keep_self"),
                "metapath_count": subgraph_config.get("num_metapaths"),
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
    ap.add_argument("--subgraph_type", type=str, default="khop", choices=["khop", "fanout", "metapath_topk"])
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
