# scripts/subgraph_sampling_stats.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dgl
import numpy as np
import torch

from scripts.peprompt_benchmark import HOP_NUM, TARGET_NODETYPE, _load_raw_heterograph


EdgeType = tuple[str, str, str]
MetaPath = tuple[EdgeType, ...]


@dataclass
class SubgraphSize:
    nodes: int
    edges: int


@dataclass
class SummaryRow:
    dataset: str
    target_ntype: str
    method: str
    num_targets: int
    avg_nodes: float
    avg_edges: float
    max_nodes: int
    max_edges: int
    min_nodes: int
    min_edges: int
    std_nodes: float
    std_edges: float
    elapsed_sec: float
    khop_num: int | None = None
    max_hop: int | None = None
    topk: int | None = None
    rank_metric: str | None = None
    num_metapaths: int | None = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _total_edges(graph) -> int:
    return int(sum(graph.num_edges(etype=etype) for etype in graph.canonical_etypes))


def _target_ids(graph, target_ntype: str, sample_size: int, seed: int) -> np.ndarray:
    labels = graph.nodes[target_ntype].data.get("y")
    if labels is not None:
        labels = labels.detach().cpu().long().view(-1).numpy()
        ids = np.flatnonzero(labels >= 0).astype(np.int64)
    else:
        ids = np.arange(graph.num_nodes(target_ntype), dtype=np.int64)

    if sample_size and sample_size > 0 and ids.size > sample_size:
        rng = np.random.default_rng(int(seed))
        ids = np.sort(rng.choice(ids, size=int(sample_size), replace=False).astype(np.int64))
    return ids


def _summarize(
    dataset: str,
    target_ntype: str,
    method: str,
    sizes: list[SubgraphSize],
    elapsed_sec: float,
    **kwargs,
) -> SummaryRow:
    if not sizes:
        raise RuntimeError(f"No subgraphs were sampled for dataset={dataset} method={method}.")
    nodes = np.asarray([item.nodes for item in sizes], dtype=np.float64)
    edges = np.asarray([item.edges for item in sizes], dtype=np.float64)
    return SummaryRow(
        dataset=dataset,
        target_ntype=target_ntype,
        method=method,
        num_targets=int(len(sizes)),
        avg_nodes=float(nodes.mean()),
        avg_edges=float(edges.mean()),
        max_nodes=int(nodes.max()),
        max_edges=int(edges.max()),
        min_nodes=int(nodes.min()),
        min_edges=int(edges.min()),
        std_nodes=float(nodes.std()),
        std_edges=float(edges.std()),
        elapsed_sec=float(elapsed_sec),
        **kwargs,
    )


def _khop_sizes(graph, target_ntype: str, target_ids: Iterable[int], hop_num: int) -> list[SubgraphSize]:
    sizes = []
    for node_id in target_ids:
        subgraph, _ = dgl.khop_in_subgraph(
            graph,
            {target_ntype: int(node_id)},
            k=int(hop_num),
        )
        sizes.append(SubgraphSize(nodes=int(subgraph.num_nodes()), edges=_total_edges(subgraph)))
    return sizes


def _build_csr_adjs(graph):
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise RuntimeError("scipy is required for metapath top-k statistics.") from exc

    adjs = {}
    for etype in graph.canonical_etypes:
        src_t, _, dst_t = etype
        src, dst = graph.edges(etype=etype)
        src = src.detach().cpu().numpy()
        dst = dst.detach().cpu().numpy()
        data = np.ones(src.shape[0], dtype=np.float32)
        adjs[etype] = sp.coo_matrix(
            (data, (src, dst)),
            shape=(graph.num_nodes(src_t), graph.num_nodes(dst_t)),
            dtype=np.float32,
        ).tocsr()
    return adjs


def _generate_metapaths(graph, start_ntype: str, max_hop: int) -> list[MetaPath]:
    by_src: dict[str, list[EdgeType]] = defaultdict(list)
    for etype in graph.canonical_etypes:
        by_src[etype[0]].append(etype)

    out: list[MetaPath] = []

    def _dfs(current_ntype: str, depth: int, path: list[EdgeType]) -> None:
        if depth > 0:
            out.append(tuple(path))
        if depth == max_hop:
            return
        for etype in by_src.get(current_ntype, []):
            path.append(etype)
            _dfs(etype[2], depth + 1, path)
            path.pop()

    _dfs(start_ntype, 0, [])
    return out


def _topk_indices(raw_values: np.ndarray, rank_values: np.ndarray, topk: int) -> np.ndarray:
    nonzero = np.flatnonzero(raw_values > 0)
    if nonzero.size == 0:
        return nonzero.astype(np.int64)
    scores = rank_values[nonzero]
    limit = min(int(topk), nonzero.size) if topk > 0 else nonzero.size
    if limit < nonzero.size:
        partial = np.argpartition(-scores, limit - 1)[:limit]
        nonzero = nonzero[partial]
        scores = rank_values[nonzero]
    order = np.lexsort((nonzero, -scores))
    return nonzero[order].astype(np.int64)


def _endpoint_popularity(adjs: dict[EdgeType, object], metapath: MetaPath) -> np.ndarray:
    import scipy.sparse as sp

    first_adj = adjs[metapath[0]]
    row = sp.csr_matrix(np.ones((1, first_adj.shape[0]), dtype=np.float32))
    for etype in metapath:
        row = row @ adjs[etype]
    return np.asarray(row.toarray()).reshape(-1)


def _rank_values(raw_scores: np.ndarray, endpoint_popularity: np.ndarray | None, rank_metric: str) -> np.ndarray:
    if rank_metric == "count":
        return raw_scores
    if rank_metric == "degree_norm":
        denom = np.sqrt(np.maximum(endpoint_popularity, 1.0))
        return raw_scores / denom
    raise ValueError(f"Unsupported rank_metric={rank_metric}")


def _metapath_reachable_scores(
    adjs: dict[EdgeType, object],
    metapath: MetaPath,
    node_id: int,
    target_ntype: str,
    keep_self: bool,
):
    row = None
    for step, etype in enumerate(metapath):
        adj = adjs[etype]
        if step == 0:
            row = adj.getrow(int(node_id))
        else:
            row = row @ adj
    scores = np.asarray(row.toarray()).reshape(-1)
    if not keep_self and metapath[-1][2] == target_ntype and int(node_id) < scores.shape[0]:
        scores[int(node_id)] = 0
    return scores


def _metapath_topk_sizes(
    graph,
    target_ntype: str,
    target_ids: Iterable[int],
    max_hop: int,
    topk: int,
    max_metapaths: int,
    keep_self: bool,
    rank_metric: str,
) -> tuple[list[SubgraphSize], int]:
    adjs = _build_csr_adjs(graph)
    metapaths = _generate_metapaths(graph, target_ntype, int(max_hop))
    if max_metapaths > 0 and len(metapaths) > max_metapaths:
        metapaths = metapaths[:max_metapaths]
    endpoint_popularity = {
        metapath: _endpoint_popularity(adjs, metapath)
        for metapath in metapaths
    } if rank_metric == "degree_norm" else {}

    sizes = []
    for node_id in target_ids:
        selected: dict[str, set[int]] = {ntype: set() for ntype in graph.ntypes}
        selected[target_ntype].add(int(node_id))

        for metapath in metapaths:
            scores = _metapath_reachable_scores(
                adjs=adjs,
                metapath=metapath,
                node_id=int(node_id),
                target_ntype=target_ntype,
                keep_self=bool(keep_self),
            )
            ranks = _rank_values(scores, endpoint_popularity.get(metapath), rank_metric)
            keep = _topk_indices(scores, ranks, int(topk))
            if keep.size == 0:
                continue
            selected[metapath[-1][2]].update(int(v) for v in keep.tolist())

        node_dict = {
            ntype: torch.tensor(sorted(node_ids), dtype=torch.int64)
            for ntype, node_ids in selected.items()
            if node_ids
        }
        subgraph = dgl.node_subgraph(graph, node_dict)
        sizes.append(SubgraphSize(nodes=int(subgraph.num_nodes()), edges=_total_edges(subgraph)))

    return sizes, len(metapaths)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_dataset(args, dataset: str) -> list[SummaryRow]:
    graph, target_ntype = _load_raw_heterograph(args.root, dataset, args.feats_type)
    target_ids = _target_ids(graph, target_ntype, args.target_sample_size, args.seed)
    print(
        f"[dataset] {dataset} target={target_ntype} targets={target_ids.size} "
        f"nodes={graph.num_nodes()} edges={_total_edges(graph)}"
    )

    rows: list[SummaryRow] = []
    if "khop" in args.methods:
        khop_nums = args.khop_nums or [HOP_NUM[dataset]]
        for hop_num in khop_nums:
            start = time.perf_counter()
            sizes = _khop_sizes(graph, target_ntype, target_ids, int(hop_num))
            row = _summarize(
                dataset,
                target_ntype,
                "khop",
                sizes,
                time.perf_counter() - start,
                khop_num=int(hop_num),
            )
            rows.append(row)
            print(
                f"[khop] dataset={dataset} hop={hop_num} "
                f"avg_nodes={row.avg_nodes:.2f} avg_edges={row.avg_edges:.2f} "
                f"max_nodes={row.max_nodes} max_edges={row.max_edges}"
            )

    if "metapath_topk" in args.methods:
        for max_hop in args.max_hops:
            for topk in args.topks:
                start = time.perf_counter()
                sizes, num_metapaths = _metapath_topk_sizes(
                    graph=graph,
                    target_ntype=target_ntype,
                    target_ids=target_ids,
                    max_hop=int(max_hop),
                    topk=int(topk),
                    max_metapaths=int(args.max_metapaths),
                    keep_self=bool(args.keep_self),
                    rank_metric=str(args.rank_metric),
                )
                row = _summarize(
                    dataset,
                    target_ntype,
                    "metapath_topk",
                    sizes,
                    time.perf_counter() - start,
                    max_hop=int(max_hop),
                    topk=int(topk),
                    rank_metric=str(args.rank_metric),
                    num_metapaths=int(num_metapaths),
                )
                rows.append(row)
                print(
                    f"[metapath_topk] dataset={dataset} M={max_hop} topk={topk} "
                    f"metapaths={num_metapaths} avg_nodes={row.avg_nodes:.2f} "
                    f"avg_edges={row.avg_edges:.2f} max_nodes={row.max_nodes} "
                    f"max_edges={row.max_edges}"
                )
    return rows


def build_parser():
    ap = argparse.ArgumentParser("Subgraph size statistics for k-hop and top-k metapath sampling")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB", "Freebase"], choices=list(HOP_NUM.keys()))
    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["khop", "metapath_topk"], choices=["khop", "metapath_topk"])
    ap.add_argument("--khop_nums", nargs="*", type=int, default=None)
    ap.add_argument("--max_hops", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--topks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--rank_metric", type=str, default="count", choices=["count", "degree_norm"])
    ap.add_argument("--target_sample_size", type=int, default=0, help="0 means all labeled target nodes.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_self", action="store_true", help="Keep the target node itself in same-type metapath top-k.")
    ap.add_argument(
        "--max_metapaths",
        type=int,
        default=0,
        help="Debug guard: if >0, truncate generated metapaths to this many per M.",
    )
    ap.add_argument(
        "--save_dir",
        type=Path,
        default=ROOT / "artifacts" / "analysis" / "subgraph_sampling_stats",
    )
    return ap


def main():
    args = build_parser().parse_args()
    rows: list[SummaryRow] = []
    for dataset in args.datasets:
        if dataset not in TARGET_NODETYPE:
            raise ValueError(f"Unsupported dataset: {dataset}")
        rows.extend(_run_dataset(args, dataset))

    out_dir = _ensure_dir(args.save_dir)
    payload = [asdict(row) for row in rows]
    _write_csv(out_dir / "summary.csv", payload)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "rows": payload,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[done] wrote {out_dir / 'summary.csv'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
