from __future__ import annotations

from pathlib import Path
import argparse
import pickle as pk
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from gpbench.downstream.fewshot import build_peprompt_offline_cache_path, save_peprompt_split_ids
from gpbench.protocol_bridge.downstream_legacy import build_legacy_hgnn
from scripts.peprompt_benchmark import (
    PEPROMPT_EDGE_FEATURE_NAME,
    _load_raw_heterograph,
    _peprompt_edge_feature_cache_key,
    prepare_peprompt_edge_feature_table,
    prepare_peprompt_spectral_payload,
)
from scripts.precompute_peprompt_cache import (
    _attach_peprompt_edge_features_from_global_pe,
    _strict_kshot_rest_split,
    _target_labels,
    _target_supervision_labels,
)


TARGET_NTYPE = "paper"
REWIRE_RELATIONS = {
    "author": (("paper", "to", "author"), ("author", "to", "paper")),
    "subject": (("paper", "to", "subject"), ("subject", "to", "paper")),
    "term": (("paper", "to", "term"), ("term", "to", "paper")),
}
PAPER_ETYPES = [("paper", "cite", "paper"), ("paper", "ref", "paper")]
CONTEXT_VARIANT_SUFFIX = "_context"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _edge_sets(graph, etype):
    src, dst = graph.edges(etype=etype)
    out = [set() for _ in range(graph.num_nodes(etype[0]))]
    in_ = [set() for _ in range(graph.num_nodes(etype[2]))]
    for s, d in zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()):
        out[int(s)].add(int(d))
        in_[int(d)].add(int(s))
    return out, in_


def _edge_lists(graph, etype):
    src, dst = graph.edges(etype=etype)
    out = [[] for _ in range(graph.num_nodes(etype[0]))]
    in_ = [[] for _ in range(graph.num_nodes(etype[2]))]
    for s, d in zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()):
        out[int(s)].append(int(d))
        in_[int(d)].append(int(s))
    return out, in_


def _build_plain_gcn_h1(graph, args) -> torch.Tensor:
    model_args = SimpleNamespace(
        method="hgmp",
        dataset="ACM",
        root=args.root,
        feats_type=args.feats_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        hgnn_type="GCN",
        num_samples=args.num_samples,
        num_class=3,
        device=torch.device(args.device),
    )
    hgnn = build_legacy_hgnn(model_args, args.ckpt, freeze=True)
    hgnn.eval()

    x_dict = {ntype: graph.nodes[ntype].data["x"].to(model_args.device) for ntype in graph.ntypes}
    keys = list(x_dict.keys())
    sizes = [x_dict[key].shape[0] for key in keys]
    with torch.no_grad():
        projected = [fc(x_dict[key]) for fc, key in zip(hgnn.fc_list, keys)]
        h = torch.cat(projected, dim=0)
        homo = dgl.to_homogeneous(graph).to(model_args.device)
        homo = dgl.remove_self_loop(homo)
        homo = dgl.add_self_loop(homo)
        h1 = hgnn.layers[0](homo, h)

    start = 0
    out = {}
    for key, size in zip(keys, sizes):
        out[key] = h1[start:start + size].detach().cpu()
        start += size
    return out[TARGET_NTYPE].float()


def _top_similar_papers(h1: torch.Tensor, topk: int) -> tuple[np.ndarray, np.ndarray]:
    z = F.normalize(h1, p=2, dim=1)
    sim = (z @ z.t()).cpu()
    sim.fill_diagonal_(-float("inf"))
    values, indices = torch.topk(sim, k=int(topk), dim=1)
    return indices.numpy().astype(np.int64), values.numpy().astype(np.float32)


def _idf_for_relation(out_lists: list[list[int]], num_target: int, num_neighbor: int) -> np.ndarray:
    df = np.zeros(num_neighbor, dtype=np.float32)
    for neighs in out_lists:
        if neighs:
            df[np.fromiter(set(neighs), dtype=np.int64)] += 1.0
    return (np.log((1.0 + float(num_target)) / (1.0 + df)) + 1.0).astype(np.float32)


def _selection_variant(variant: str) -> str:
    variant = str(variant)
    if variant.endswith(CONTEXT_VARIANT_SUFFIX):
        return variant[: -len(CONTEXT_VARIANT_SUFFIX)]
    return variant


def _uses_target_context(variant: str) -> bool:
    return str(variant).endswith(CONTEXT_VARIANT_SUFFIX)


def _candidate_scores(
    paper_id: int,
    relation_name: str,
    similar_ids: np.ndarray,
    similar_scores: np.ndarray,
    relation_out: dict[str, list[list[int]]],
    relation_idf: dict[str, np.ndarray],
) -> dict[int, float]:
    scores: dict[int, float] = {}
    original = set(relation_out[relation_name][paper_id])
    for sim_paper, sim_value in zip(similar_ids, similar_scores):
        weight = max(0.0, float(sim_value))
        if weight <= 0.0:
            continue
        for neigh in relation_out[relation_name][int(sim_paper)]:
            neigh = int(neigh)
            if neigh in original:
                continue
            scores[neigh] = scores.get(neigh, 0.0) + weight * float(relation_idf[relation_name][neigh])
    return scores


def _select_neighbors(
    paper_id: int,
    relation_name: str,
    variant: str,
    budgets: dict[str, int],
    relation_out: dict[str, list[list[int]]],
    relation_idf: dict[str, np.ndarray],
    candidate_scores: dict[int, float],
    original_prior: float,
) -> list[int]:
    variant = _selection_variant(variant)
    original = set(relation_out[relation_name][paper_id])
    budget = int(budgets[relation_name])
    if variant == "baseline":
        return sorted(original)

    ranked_candidates = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    top_candidates = [node_id for node_id, _ in ranked_candidates[:budget]]
    if variant == "add_only":
        return sorted(original.union(top_candidates))
    if variant == "replace_only":
        return sorted(top_candidates)
    if variant != "fixed_budget":
        raise ValueError(f"Unsupported rewire variant: {variant}")

    pool = {node_id: float(score) for node_id, score in candidate_scores.items()}
    for node_id in original:
        pool[int(node_id)] = pool.get(int(node_id), 0.0) + float(original_prior) * float(
            relation_idf[relation_name][int(node_id)]
        )
    ranked_pool = sorted(pool.items(), key=lambda item: (-item[1], item[0]))
    return sorted(node_id for node_id, _ in ranked_pool[:budget])


def _build_rewired_star_graph(
    graph,
    paper_id: int,
    variant: str,
    selected_by_relation: dict[str, list[int]],
    paper_out: dict,
    paper_in: dict,
):
    selected_nodes = {ntype: set() for ntype in graph.ntypes}
    selected_nodes[TARGET_NTYPE].add(int(paper_id))
    for relation_name, nodes in selected_by_relation.items():
        selected_nodes[relation_name].update(int(node_id) for node_id in nodes)

    for etype in PAPER_ETYPES:
        selected_nodes[TARGET_NTYPE].update(paper_out[etype][paper_id])
        selected_nodes[TARGET_NTYPE].update(paper_in[etype][paper_id])

    node_ids = {
        ntype: sorted(node_set)
        for ntype, node_set in selected_nodes.items()
    }
    local = {
        ntype: {global_id: local_id for local_id, global_id in enumerate(ids)}
        for ntype, ids in node_ids.items()
    }

    data_dict = {}
    center_local = local[TARGET_NTYPE][int(paper_id)]
    for etype in graph.canonical_etypes:
        src_t, rel_t, dst_t = etype
        src_edges = []
        dst_edges = []
        if etype in PAPER_ETYPES:
            for neigh in paper_out[etype][paper_id]:
                if neigh in local[TARGET_NTYPE]:
                    src_edges.append(center_local)
                    dst_edges.append(local[TARGET_NTYPE][int(neigh)])
            for neigh in paper_in[etype][paper_id]:
                if neigh in local[TARGET_NTYPE]:
                    src_edges.append(local[TARGET_NTYPE][int(neigh)])
                    dst_edges.append(center_local)
        else:
            for relation_name, (forward, backward) in REWIRE_RELATIONS.items():
                if etype == forward:
                    for neigh in selected_by_relation[relation_name]:
                        src_edges.append(center_local)
                        dst_edges.append(local[relation_name][int(neigh)])
                elif etype == backward:
                    for neigh in selected_by_relation[relation_name]:
                        src_edges.append(local[relation_name][int(neigh)])
                        dst_edges.append(center_local)
        data_dict[etype] = (
            torch.tensor(src_edges, dtype=torch.int64),
            torch.tensor(dst_edges, dtype=torch.int64),
        )

    subgraph = dgl.heterograph(
        data_dict,
        num_nodes_dict={ntype: len(ids) for ntype, ids in node_ids.items()},
    )
    for ntype, ids in node_ids.items():
        gids = torch.tensor(ids, dtype=torch.int64)
        subgraph.nodes[ntype].data[dgl.NID] = gids
        for key, value in graph.nodes[ntype].data.items():
            subgraph.nodes[ntype].data[key] = value.detach().cpu()[gids].clone()

    inverse_indices = {TARGET_NTYPE: torch.tensor([center_local], dtype=torch.int64)}
    return subgraph, inverse_indices


def _limited_sorted(values, limit: int) -> list[int]:
    values = sorted({int(v) for v in values})
    limit = int(limit)
    if limit > 0:
        values = values[:limit]
    return values


def _build_target_rewired_context_graph(
    graph,
    paper_id: int,
    selected_by_relation: dict[str, list[int]],
    relation_out: dict[str, list[list[int]]],
    relation_in: dict[str, list[list[int]]],
    paper_out: dict,
    paper_in: dict,
    second_order_papers_per_neighbor: int = 0,
):
    """Preserve original edges among selected context nodes and only rewire centre edges."""
    selected_nodes = {ntype: set() for ntype in graph.ntypes}
    selected_nodes[TARGET_NTYPE].add(int(paper_id))

    for relation_name, nodes in selected_by_relation.items():
        selected_nodes[relation_name].update(int(node_id) for node_id in nodes)
        for node_id in nodes:
            paper_neighbors = _limited_sorted(
                relation_in[relation_name][int(node_id)],
                int(second_order_papers_per_neighbor),
            )
            selected_nodes[TARGET_NTYPE].update(paper_neighbors)

    for etype in PAPER_ETYPES:
        selected_nodes[TARGET_NTYPE].update(int(v) for v in paper_out[etype][int(paper_id)])
        selected_nodes[TARGET_NTYPE].update(int(v) for v in paper_in[etype][int(paper_id)])

    node_ids = {
        ntype: sorted(node_set)
        for ntype, node_set in selected_nodes.items()
    }
    local = {
        ntype: {global_id: local_id for local_id, global_id in enumerate(ids)}
        for ntype, ids in node_ids.items()
    }
    selected_sets = {ntype: set(ids) for ntype, ids in node_ids.items()}
    center_local = local[TARGET_NTYPE][int(paper_id)]

    data_dict = {}
    for etype in graph.canonical_etypes:
        src_t, _, dst_t = etype
        src_edges = []
        dst_edges = []

        relation_name_for_etype = None
        direction = None
        for relation_name, (forward, backward) in REWIRE_RELATIONS.items():
            if etype == forward:
                relation_name_for_etype = relation_name
                direction = "forward"
                break
            if etype == backward:
                relation_name_for_etype = relation_name
                direction = "backward"
                break

        if relation_name_for_etype is not None:
            relation_name = relation_name_for_etype
            if direction == "forward":
                for src_paper in node_ids[TARGET_NTYPE]:
                    if int(src_paper) == int(paper_id):
                        continue
                    src_local = local[TARGET_NTYPE][int(src_paper)]
                    for dst_node in relation_out[relation_name][int(src_paper)]:
                        if int(dst_node) in selected_sets[relation_name]:
                            src_edges.append(src_local)
                            dst_edges.append(local[relation_name][int(dst_node)])
                for dst_node in selected_by_relation[relation_name]:
                    src_edges.append(center_local)
                    dst_edges.append(local[relation_name][int(dst_node)])
            else:
                for src_node in node_ids[relation_name]:
                    src_local = local[relation_name][int(src_node)]
                    for dst_paper in relation_in[relation_name][int(src_node)]:
                        if int(dst_paper) == int(paper_id):
                            continue
                        if int(dst_paper) in selected_sets[TARGET_NTYPE]:
                            src_edges.append(src_local)
                            dst_edges.append(local[TARGET_NTYPE][int(dst_paper)])
                for src_node in selected_by_relation[relation_name]:
                    src_edges.append(local[relation_name][int(src_node)])
                    dst_edges.append(center_local)
        elif etype in PAPER_ETYPES:
            for src_paper in node_ids[TARGET_NTYPE]:
                src_local = local[TARGET_NTYPE][int(src_paper)]
                for dst_paper in paper_out[etype][int(src_paper)]:
                    if int(dst_paper) in selected_sets[TARGET_NTYPE]:
                        src_edges.append(src_local)
                        dst_edges.append(local[TARGET_NTYPE][int(dst_paper)])

        data_dict[etype] = (
            torch.tensor(src_edges, dtype=torch.int64),
            torch.tensor(dst_edges, dtype=torch.int64),
        )

    subgraph = dgl.heterograph(
        data_dict,
        num_nodes_dict={ntype: len(ids) for ntype, ids in node_ids.items()},
    )
    for ntype, ids in node_ids.items():
        gids = torch.tensor(ids, dtype=torch.int64)
        subgraph.nodes[ntype].data[dgl.NID] = gids
        for key, value in graph.nodes[ntype].data.items():
            subgraph.nodes[ntype].data[key] = value.detach().cpu()[gids].clone()

    inverse_indices = {TARGET_NTYPE: torch.tensor([center_local], dtype=torch.int64)}
    return subgraph, inverse_indices


def _build_sample(
    graph,
    paper_id: int,
    label,
    variant: str,
    relation_out: dict[str, list[list[int]]],
    relation_in: dict[str, list[list[int]]],
    relation_idf: dict[str, np.ndarray],
    similar_ids: np.ndarray,
    similar_scores: np.ndarray,
    budgets: dict[str, int],
    original_prior: float,
    paper_out: dict,
    paper_in: dict,
    edge_feature_payload: dict,
    feature_name: str,
    second_order_papers_per_neighbor: int = 0,
):
    selected_by_relation = {}
    for relation_name in REWIRE_RELATIONS.keys():
        cand = _candidate_scores(
            paper_id,
            relation_name,
            similar_ids[paper_id],
            similar_scores[paper_id],
            relation_out,
            relation_idf,
        )
        selected_by_relation[relation_name] = _select_neighbors(
            paper_id,
            relation_name,
            variant,
            budgets,
            relation_out,
            relation_idf,
            cand,
            original_prior,
        )
    if _uses_target_context(variant):
        subgraph, inverse_indices = _build_target_rewired_context_graph(
            graph=graph,
            paper_id=paper_id,
            selected_by_relation=selected_by_relation,
            relation_out=relation_out,
            relation_in=relation_in,
            paper_out=paper_out,
            paper_in=paper_in,
            second_order_papers_per_neighbor=int(second_order_papers_per_neighbor),
        )
    else:
        subgraph, inverse_indices = _build_rewired_star_graph(
            graph,
            paper_id,
            variant,
            selected_by_relation,
            paper_out,
            paper_in,
        )
    _attach_peprompt_edge_features_from_global_pe(
        subgraph=subgraph,
        spectral_embeddings=edge_feature_payload["spectral_embeddings"],
        node_offsets=edge_feature_payload["node_offsets"],
        feature_name=feature_name,
        edge_pe_tables=None,
    )
    return subgraph, inverse_indices, torch.as_tensor(label, dtype=torch.long).clone()


def _cache_key(args) -> str:
    return f"khop_{_peprompt_edge_feature_cache_key(args)}"


def _precompute_variant(
    graph,
    edge_feature_payload: dict,
    target_labels: np.ndarray,
    supervision_labels: np.ndarray,
    similar_ids: np.ndarray,
    similar_scores: np.ndarray,
    relation_out: dict[str, list[list[int]]],
    relation_in: dict[str, list[list[int]]],
    relation_idf: dict[str, np.ndarray],
    paper_out: dict,
    paper_in: dict,
    variant: str,
    args,
):
    cache_dir = Path(args.output_cache_root) / variant
    subgraph_key = _cache_key(args)
    budgets = {
        "author": int(args.author_budget),
        "subject": int(args.subject_budget),
        "term": int(args.term_budget),
    }
    sample_cache: dict[int, tuple] = {}
    for shot in args.shots:
        for seed in args.seeds:
            start = time.perf_counter()
            split = _strict_kshot_rest_split(target_labels, int(shot), int(seed), int(args.max_pool_size))
            train = []
            val = []
            test = []
            for name, ids in (("train", split["train_ids"]), ("val", split["val_ids"]), ("test", split["test_ids"])):
                rows = []
                for paper_id in ids:
                    paper_id = int(paper_id)
                    if paper_id not in sample_cache:
                        sample_cache[paper_id] = _build_sample(
                            graph=graph,
                            paper_id=paper_id,
                            label=supervision_labels[paper_id],
                            variant=variant,
                            relation_out=relation_out,
                            relation_in=relation_in,
                            relation_idf=relation_idf,
                            similar_ids=similar_ids,
                            similar_scores=similar_scores,
                            budgets=budgets,
                            original_prior=float(args.original_prior),
                            paper_out=paper_out,
                            paper_in=paper_in,
                            edge_feature_payload=edge_feature_payload,
                            feature_name=args.peprompt_edge_feature_name,
                            second_order_papers_per_neighbor=int(args.second_order_papers_per_neighbor),
                        )
                    rows.append(sample_cache[paper_id])
                if name == "train":
                    train = rows
                elif name == "val":
                    val = rows
                else:
                    test = rows

            payload = {
                "dataset": "ACM",
                "targetnode": TARGET_NTYPE,
                "subgraph_type": "fixed_budget_rewire",
                "subgraph_cache_key": subgraph_key,
                "subgraph_config": {
                    "variant": variant,
                    "similarity": "pretrained_gcn_h1_cosine",
                    "top_similar": int(args.top_similar),
                    "budgets": budgets,
                    "original_prior": float(args.original_prior),
                    "base": "target-paper star ego; paper-paper original edges kept; author/subject/term rewired",
                    "selection_variant": _selection_variant(variant),
                    "target_context": _uses_target_context(variant),
                    "second_order_papers_per_neighbor": int(args.second_order_papers_per_neighbor),
                },
                "shot": int(shot),
                "split_seed": int(seed),
                "feats_type": int(args.feats_type),
                "hop_num": 1,
                "max_pool_size": int(args.max_pool_size),
                "peprompt_edge_feature_name": args.peprompt_edge_feature_name,
                "peprompt_edge_feature_names": list(edge_feature_payload.get("selected_feature_names", [])),
                "peprompt_edge_feature_dim": int(edge_feature_payload["feature_dim"]),
                "peprompt_edge_feature_slices": {
                    key: list(value)
                    for key, value in edge_feature_payload.get("feature_slices", {}).items()
                },
                "spectral_dim": int(edge_feature_payload.get("spectral_dim", 0) or 0),
                "spectral_cache_path": str(edge_feature_payload.get("spectral_cache_path", "")),
                "train_ids": split["train_ids"],
                "train_labels": split["train_labels"],
                "val_ids": split["val_ids"],
                "val_labels": split["val_labels"],
                "test_ids": split["test_ids"],
                "test_labels": split["test_labels"],
                "class_stats": split["class_stats"],
                "num_classes": int(split["num_classes"]),
                "train": train,
                "val": val,
                "test": test,
            }
            cache_path = build_peprompt_offline_cache_path(
                cache_dir=cache_dir,
                dataset_name="ACM",
                shot=shot,
                seed=seed,
                feats_type=args.feats_type,
                subgraph_type=subgraph_key,
            )
            _ensure_dir(cache_path.parent)
            with open(cache_path, "wb") as f:
                pk.dump(payload, f, protocol=pk.HIGHEST_PROTOCOL)
            split_ids_path = save_peprompt_split_ids(
                cache_dir=cache_dir,
                dataset_name="ACM",
                shot=shot,
                seed=seed,
                feats_type=args.feats_type,
                subgraph_type=subgraph_key,
                split_payload=payload,
            )
            print(
                f"[saved] variant={variant} shot={shot} seed={seed} "
                f"train={len(train)} val={len(val)} test={len(test)} "
                f"sample_cache={len(sample_cache)} path={cache_path} "
                f"split_ids={split_ids_path} time={time.perf_counter() - start:.2f}s"
            )


def build_parser():
    ap = argparse.ArgumentParser("Build ACM fixed-budget rewiring caches for PEPrompt.")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--output_cache_root", type=Path, default=ROOT / "artifacts" / "cache" / "acm_fixed_budget_rewire")
    ap.add_argument("--ckpt", type=str, default="artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--shots", nargs="+", type=int, default=[1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max_pool_size", type=int, default=400)
    ap.add_argument("--variants", nargs="+", default=["baseline", "add_only", "fixed_budget", "replace_only"])
    ap.add_argument("--top_similar", type=int, default=8)
    ap.add_argument("--author_budget", type=int, default=4)
    ap.add_argument("--subject_budget", type=int, default=2)
    ap.add_argument("--term_budget", type=int, default=16)
    ap.add_argument("--original_prior", type=float, default=1.0)
    ap.add_argument(
        "--second_order_papers_per_neighbor",
        type=int,
        default=0,
        help="For *_context variants, keep up to this many original paper neighbors per selected author/subject/term; 0 keeps all.",
    )
    ap.add_argument("--peprompt_spectral_cache_dir", type=Path, default=ROOT / "artifacts" / "cache" / "peprompt_spectral_embeddings")
    ap.add_argument("--peprompt_spectral_dim", type=int, default=16)
    ap.add_argument("--peprompt_spectral_max_nodes", type=int, default=50000)
    ap.add_argument("--peprompt_edge_feature_names", nargs="*", default=None)
    ap.add_argument("--peprompt_edge_feature_name", type=str, default=PEPROMPT_EDGE_FEATURE_NAME)
    ap.add_argument("--peprompt_cache_include_feature_key", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main():
    args = build_parser().parse_args()
    args.dataset = "ACM"
    graph, target = _load_raw_heterograph(args.root, "ACM", args.feats_type)
    if target != TARGET_NTYPE:
        raise RuntimeError(f"Expected ACM target node type {TARGET_NTYPE}, got {target}")

    edge_feature_payload = prepare_peprompt_edge_feature_table(args, wandb_run=None)
    spectral_payload, _, _ = prepare_peprompt_spectral_payload(args)
    edge_feature_payload = dict(edge_feature_payload)
    edge_feature_payload.update(
        {
            "spectral_embeddings": spectral_payload["spectral_embeddings"],
            "node_offsets": spectral_payload["node_offsets"],
        }
    )
    print(
        f"[edge-features] features={edge_feature_payload.get('selected_feature_names')} "
        f"dim={edge_feature_payload['feature_dim']} spectral={edge_feature_payload.get('spectral_cache_path')}"
    )
    h1 = _build_plain_gcn_h1(graph, args)
    similar_ids, similar_scores = _top_similar_papers(h1, args.top_similar)

    relation_out = {}
    relation_in = {}
    relation_idf = {}
    for relation_name, (forward, _) in REWIRE_RELATIONS.items():
        out, in_ = _edge_lists(graph, forward)
        relation_out[relation_name] = out
        relation_in[relation_name] = in_
        relation_idf[relation_name] = _idf_for_relation(
            out,
            num_target=graph.num_nodes(TARGET_NTYPE),
            num_neighbor=graph.num_nodes(forward[2]),
        )

    paper_out = {}
    paper_in = {}
    for etype in PAPER_ETYPES:
        out, in_ = _edge_sets(graph, etype)
        paper_out[etype] = out
        paper_in[etype] = in_

    labels = _target_labels(graph, TARGET_NTYPE)
    supervision_labels = _target_supervision_labels(graph, TARGET_NTYPE, "ACM")
    for variant in args.variants:
        _precompute_variant(
            graph=graph,
            edge_feature_payload=edge_feature_payload,
            target_labels=labels,
            supervision_labels=supervision_labels,
            similar_ids=similar_ids,
            similar_scores=similar_scores,
            relation_out=relation_out,
            relation_in=relation_in,
            relation_idf=relation_idf,
            paper_out=paper_out,
            paper_in=paper_in,
            variant=str(variant),
            args=args,
        )


if __name__ == "__main__":
    main()
