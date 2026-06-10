
# scripts/peprompt_benchmark.py
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
import re
import time
import warnings
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import dgl
import numpy as np
import torch
from torch_geometric.datasets import HGBDataset
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_dgl

from gpbench.protocol_bridge import downstream_legacy as legacy_bridge
from gpbench.protocol_bridge.downstream_legacy import (
    build_legacy_fewshot_embeddings,
    train_hgmp_heteroprompt_probe,
    train_mlp_probe,
    train_peprompt_probe,
)
from gpbench.downstream.fewshot import load_peprompt_offline_splits
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

DATASET_NUM_CLASS = {
    "ACM": 3,
    "DBLP": 4,
    "IMDB": 5,
    "Freebase": 7,
}

PEPROMPT_EDGE_FEATURE_NAME = "peprompt_edge_feat"
PEPROMPT_EDGE_FEATURES = ["SpectralEmbeddingDiff"]
LEGACY_HGMP_METHODS = {"hgmp", "peprompt", "hgmp_prompt"}
LEGACY_CKPT_NAME_RE = re.compile(
    r"^(?P<dataset>[^.]+)\.(?P<pretext>[^.]+)\.(?P<hgnn_type>[^.]+)\.hid(?P<hidden_dim>\d+)\.np(?P<num_samples>\d+)"
    r"(?:\.seed(?P<seed>\d+))?\.pth$"
)
LEGACY_CKPT_SYNC_KEYS = ("hgnn_type", "hidden_dim", "num_samples")
LEGACY_CKPT_OPTIONAL_SYNC_KEYS = ("num_class", "feats_type", "num_heads", "num_layers", "dropout")


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


def _collapse_multilabel_to_single_class(label_tensor: torch.Tensor) -> torch.Tensor:
    if label_tensor.ndim == 1:
        return label_tensor.long()
    if label_tensor.ndim != 2:
        raise ValueError(f"Unsupported label tensor shape for downstream collapse: {tuple(label_tensor.shape)}")

    label_tensor = label_tensor.detach().cpu()
    collapsed = torch.full((label_tensor.size(0),), -1, dtype=torch.long)
    for row_idx, row in enumerate(label_tensor):
        one_indices = torch.nonzero(row > 0, as_tuple=False).view(-1).cpu().numpy()
        if one_indices.size > 0:
            collapsed[row_idx] = int(np.random.choice(one_indices))
    return collapsed


def _normalize_dataset_defaults(args):
    if getattr(args, "num_class", None) is None:
        args.num_class = DATASET_NUM_CLASS[args.dataset]
    return args


def _load_legacy_ckpt_metadata(ckpt_path: str) -> dict:
    src = Path(ckpt_path)
    meta = {}

    sidecar = Path(str(src) + ".json")
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Checkpoint metadata is not a JSON object: {sidecar}")
        meta.update(loaded)

    match = LEGACY_CKPT_NAME_RE.match(src.name)
    if match:
        parsed = match.groupdict()
        meta.setdefault("dataset", parsed["dataset"])
        meta.setdefault("pretext", parsed["pretext"])
        meta.setdefault("hgnn_type", parsed["hgnn_type"])
        meta.setdefault("hidden_dim", int(parsed["hidden_dim"]))
        meta.setdefault("num_samples", int(parsed["num_samples"]))
        if parsed.get("seed") is not None:
            meta.setdefault("seed", int(parsed["seed"]))
    return meta


def _sync_args_with_legacy_ckpts(args, ckpt_by_method: dict[str, str]):
    legacy_metas = []
    for method, ckpt_path in ckpt_by_method.items():
        if method not in LEGACY_HGMP_METHODS:
            continue
        meta = _load_legacy_ckpt_metadata(ckpt_path)
        if meta:
            legacy_metas.append((method, ckpt_path, meta))

    if not legacy_metas:
        return args

    dataset_values = {
        str(meta["dataset"])
        for _, _, meta in legacy_metas
        if meta.get("dataset") is not None
    }
    if len(dataset_values) > 1:
        raise RuntimeError(
            f"Resolved legacy checkpoints disagree on dataset: {sorted(dataset_values)}"
        )
    if dataset_values:
        ckpt_dataset = next(iter(dataset_values))
        if ckpt_dataset != args.dataset:
            raise RuntimeError(
                f"CLI dataset={args.dataset} does not match legacy checkpoint dataset={ckpt_dataset}."
            )

    for key in LEGACY_CKPT_SYNC_KEYS + LEGACY_CKPT_OPTIONAL_SYNC_KEYS:
        values = {
            meta[key]
            for _, _, meta in legacy_metas
            if meta.get(key) is not None
        }
        if len(values) > 1:
            detail = ", ".join(
                f"{method}:{meta.get(key)}"
                for method, _, meta in legacy_metas
                if meta.get(key) is not None
            )
            raise RuntimeError(f"Resolved legacy checkpoints disagree on {key}: {detail}")
        if not values:
            continue
        ckpt_value = next(iter(values))
        old_value = getattr(args, key, None)
        if old_value != ckpt_value:
            print(
                f"[ckpt-sync] overriding {key} from {old_value} to {ckpt_value} "
                "based on legacy checkpoint metadata."
            )
            setattr(args, key, ckpt_value)
    return args


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
        newy = _collapse_multilabel_to_single_class(oldy)
        data[targetnode]["oldy"] = oldy
        data[targetnode]["y"] = newy

    data = _apply_feats_type(data, feats_type)
    graph = to_dgl(data)
    return graph, TARGET_NODETYPE[dataset]


def _peprompt_spectral_cache_path(args) -> Path:
    """根据数据集、特征类型和谱维度，生成 PEPrompt 光谱缓存文件路径。"""
    cache_dir = Path(
        getattr(
            args,
            "peprompt_spectral_cache_dir",
            ROOT / "artifacts" / "cache" / "peprompt_spectral_embeddings",
        )
    )
    spectral_dim = int(getattr(args, "peprompt_spectral_dim", 8))
    max_nodes = int(getattr(args, "peprompt_spectral_max_nodes", 50000))
    return (
        cache_dir
        / args.dataset
        / f"ft{args.feats_type}"
        / f"spec{spectral_dim}.mn{max_nodes}.pt"
    )


def _edge_type_key(etype) -> str:
    return "__".join(str(part) for part in etype)


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
    del num_node_types
    del num_edge_types
    return {"SpectralEmbeddingDiff": (0, spectral_dim)}


def _coerce_feature_name_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def get_selected_peprompt_edge_feature_names(args, feature_slices: dict[str, tuple[int, int]] | None = None) -> list[str]:
    available = list(feature_slices.keys()) if feature_slices is not None else list(PEPROMPT_EDGE_FEATURES)
    configured = _coerce_feature_name_list(getattr(args, "peprompt_edge_feature_names", None))
    if configured is None:
        return available
    if not configured:
        return []
    unknown = [name for name in configured if name not in set(available)]
    if unknown:
        raise ValueError(
            f"Unsupported peprompt edge feature names: {unknown}. "
            f"Only spectral edge prompts remain enabled: {available}."
        )
    return configured


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


def _subset_peprompt_edge_feature_payload(payload: dict, args) -> dict:
    selected_names = get_selected_peprompt_edge_feature_names(args, payload["feature_slices"])
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


def _compute_peprompt_spectral_payload(graph, args) -> dict:
    start_time = time.perf_counter()
    ntypes = list(graph.ntypes)
    etypes = list(graph.canonical_etypes)

    offsets = {}
    total_nodes = 0
    for ntype in ntypes:
        offsets[ntype] = total_nodes
        total_nodes += graph.num_nodes(ntype)

    global_src, global_dst = _collect_global_edges(graph, offsets)
    spectral_dim = int(getattr(args, "peprompt_spectral_dim", 8))
    spectral = _spectral_embeddings(
        total_nodes,
        global_src,
        global_dst,
        dim=spectral_dim,
        max_nodes=int(getattr(args, "peprompt_spectral_max_nodes", 50000)),
    )
    generation_seconds = float(time.perf_counter() - start_time)
    return {
        "spectral_embeddings": spectral,
        "spectral_dim": spectral_dim,
        "node_types": ntypes,
        "edge_types": etypes,
        "node_offsets": offsets,
        "total_nodes": int(total_nodes),
        "total_edges": int(global_src.numel()),
        "generation_seconds": generation_seconds,
    }


def _build_spectral_edge_feature_payload(graph, spectral_payload: dict) -> dict:
    ntypes = list(graph.ntypes)
    etypes = list(graph.canonical_etypes)
    spectral = spectral_payload["spectral_embeddings"]
    offsets = spectral_payload["node_offsets"]
    spectral_dim = int(spectral_payload["spectral_dim"])
    feature_slices = _feature_slices(len(ntypes), len(etypes), spectral_dim)
    edge_feature_table = {}

    for src_t, rel_t, dst_t in etypes:
        etype = (src_t, rel_t, dst_t)
        src, dst = graph.edges(etype=etype)
        src = src.detach().cpu().long()
        dst = dst.detach().cpu().long()
        src_global = src + offsets[src_t]
        dst_global = dst + offsets[dst_t]
        spectral_diff = spectral[src_global] - spectral[dst_global]
        edge_feature_table[etype] = spectral_diff.float()

    total_edges = int(sum(value.size(0) for value in edge_feature_table.values()))
    stats = _summarize_edge_features(edge_feature_table, feature_slices)
    return {
        "edge_feature_table": edge_feature_table,
        "feature_dim": int(next(iter(edge_feature_table.values())).size(1)) if edge_feature_table else 0,
        "feature_slices": feature_slices,
        "feature_stats": stats,
        "node_types": ntypes,
        "edge_types": etypes,
        "total_nodes": int(spectral_payload["total_nodes"]),
        "total_edges": total_edges,
        "generation_seconds": float(spectral_payload["generation_seconds"]),
        "spectral_cache_path": spectral_payload.get("spectral_cache_path"),
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
        name=f"{payload['total_nodes']}-peprompt-spectral-cache-{cache_path.stem}",
        artifact_type="peprompt-spectral-cache",
    )


def _load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_peprompt_spectral_payload(args) -> tuple[dict, Path, bool]:
    """加载或生成全图级 Laplacian PE，并返回缓存命中信息。"""
    cache_path = _peprompt_spectral_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = cache_path.exists()
    if cache_hit:
        payload = _load_torch_payload(cache_path)
    else:
        graph, _ = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
        payload = _compute_peprompt_spectral_payload(graph, args)
        torch.save(payload, cache_path)
    payload = dict(payload)
    payload["spectral_cache_path"] = str(cache_path)
    return payload, cache_path, cache_hit


def prepare_peprompt_edge_feature_table(args, wandb_run=None) -> dict:
    """把节点级谱嵌入转换为边级 PE 差值特征表，供 PEPrompt 注入使用。"""
    graph, _ = _load_raw_heterograph(args.root, args.dataset, args.feats_type)
    spectral_payload, cache_path, cache_hit = prepare_peprompt_spectral_payload(args)
    payload = _build_spectral_edge_feature_payload(graph, spectral_payload)
    payload = _subset_peprompt_edge_feature_payload(payload, args)
    if getattr(args, "peprompt_write_edge_feature_stats", True):
        _write_edge_feature_stats_json(cache_path, payload)
    setattr(args, "peprompt_edge_feature_dim", int(payload["feature_dim"]))
    _log_edge_feature_payload(wandb_run, payload, cache_path, cache_hit)
    return payload


def _attach_peprompt_edge_features_to_graph(graph, edge_feature_table: dict, feature_name: str):
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


def _attach_peprompt_edge_features_to_samples(args, sample_lists: list[list]):
    """将预先计算好的 PEPrompt 边特征挂到 few-shot 子图样本上。"""
    payload = prepare_peprompt_edge_feature_table(args, wandb_run=None)
    feature_name = getattr(args, "peprompt_edge_feature_name", PEPROMPT_EDGE_FEATURE_NAME)
    for sample_list in sample_lists:
        for sample in sample_list:
            graph = _first_graph_from_sample_v2(sample, args.classification_type)
            _attach_peprompt_edge_features_to_graph(graph, payload["edge_feature_table"], feature_name)


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


def load_peprompt_offline_legacy_splits(args):
    if args.dataset not in HOP_NUM:
        raise ValueError(f"Unsupported dataset for PEPrompt offline splits: {args.dataset}")

    subgraph_type = _peprompt_cache_subgraph_type(args)
    payload = load_peprompt_offline_splits(
        cache_dir=getattr(
            args,
            "peprompt_offline_cache_dir",
            ROOT / "artifacts" / "cache" / "peprompt_offline_splits",
        ),
        dataset_name=args.dataset,
        shot=args.shot,
        seed=args.split_seed,
        feats_type=args.feats_type,
        subgraph_type=subgraph_type,
    )
    setattr(args, "peprompt_edge_feature_dim", int(payload.get("peprompt_edge_feature_dim", 0)))
    setattr(args, "peprompt_metapath_count", int(payload.get("metapath_count") or 0))
    setattr(args, "peprompt_graph_summary_dim", int(payload.get("peprompt_graph_summary_dim", 0) or 0))
    targetnode = payload["targetnode"]
    dropped_ctx_by_target = payload.get("dropped_metapath_context_by_target")
    dropped_onehop_ctx_by_target = payload.get("dropped_onehop_context_by_target")
    # Lazy import to break circular dependency with precompute_peprompt_cache.
    from scripts.precompute_peprompt_cache import _restore_dropped_ctx_in_sample

    for sample_list in (payload["train"], payload["val"], payload["test"]):
        for sample in sample_list:
            _restore_dropped_ctx_in_sample(
                sample,
                targetnode,
                dropped_ctx_by_target,
                dropped_onehop_ctx_by_target,
            )
    return payload["train"], payload["val"], payload["test"], targetnode


def _peprompt_cache_subgraph_type(args) -> str:
    subgraph_type = str(getattr(args, "subgraph_type", "khop"))
    if subgraph_type != "metapath_topk":
        return subgraph_type
    metric = str(getattr(args, "metapath_rank_metric", "count"))
    suffix = (
        f"m{int(getattr(args, 'metapath_max_hop', 3))}_"
        f"k{int(getattr(args, 'metapath_topk', 5))}_"
        f"{metric}"
    )
    if bool(getattr(args, "metapath_keep_self", False)):
        suffix += "_self"
    fusion_mode = str(getattr(args, "peprompt_fusion_mode", "none"))
    ctx_dim = int(getattr(args, "peprompt_ctx_dim", 0) or 0)
    if fusion_mode == "hop_decoupled" and ctx_dim > 0:
        suffix += f"_mpvirt2_d{ctx_dim}"
    elif fusion_mode == "onehop_ctx" and ctx_dim > 0:
        suffix += f"_onehop_d{ctx_dim}"
    elif fusion_mode == "type_ctx" and ctx_dim > 0:
        suffix += f"_typectx_d{ctx_dim}"
    elif fusion_mode in {"graph_summary", "graph_summary_basis"}:
        suffix += "_graphsum"
    return f"metapath_topk_{suffix}"


def _patched_legacy_split_loader(args):
    return load_peprompt_offline_legacy_splits(args)


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

    hgmp_ckpt_pattern = getattr(cli_args, "hgmp_ckpt_pattern", None)
    hgprompt_ckpt_pattern = getattr(cli_args, "hgprompt_ckpt_pattern", None)

    if method == "hgmp":
        candidate = _maybe_format(hgmp_ckpt_pattern) or cli_args.hgmp_ckpt
    elif method == "peprompt":
        candidate = (
            cli_args.peprompt_ckpt
            or cli_args.hgmp_ckpt
        )
    elif method == "hgmp_prompt":
        candidate = _maybe_format(hgmp_ckpt_pattern) or cli_args.hgmp_ckpt
    elif method == "hgprompt":
        candidate = _maybe_format(hgprompt_ckpt_pattern) or cli_args.hgprompt_ckpt
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
        peprompt_edge_feature_dim=cli_args.peprompt_edge_feature_dim,
        peprompt_edge_feature_names=cli_args.peprompt_edge_feature_names,
        peprompt_edge_feature_name=cli_args.peprompt_edge_feature_name,
        peprompt_spectral_cache_dir=cli_args.peprompt_spectral_cache_dir,
        peprompt_offline_cache_dir=cli_args.peprompt_offline_cache_dir,
        subgraph_type=cli_args.subgraph_type,
        metapath_max_hop=cli_args.metapath_max_hop,
        metapath_topk=cli_args.metapath_topk,
        metapath_rank_metric=cli_args.metapath_rank_metric,
        metapath_keep_self=cli_args.metapath_keep_self,
        peprompt_write_edge_feature_stats=cli_args.peprompt_write_edge_feature_stats,
        peprompt_spectral_dim=cli_args.peprompt_spectral_dim,
        peprompt_spectral_max_nodes=cli_args.peprompt_spectral_max_nodes,
        peprompt_edge_prompt_hidden=cli_args.peprompt_edge_prompt_hidden,
        peprompt_fusion_mode=cli_args.peprompt_fusion_mode,
        peprompt_ctx_dim=cli_args.peprompt_ctx_dim,
        peprompt_generator_hidden=cli_args.peprompt_generator_hidden,
        peprompt_metapath_embed_dim=cli_args.peprompt_metapath_embed_dim,
        peprompt_graph_summary_dim=cli_args.peprompt_graph_summary_dim,
        peprompt_basis_count=cli_args.peprompt_basis_count,
        peprompt_onehop_center_fusion=cli_args.peprompt_onehop_center_fusion,
        embed_batch_size=cli_args.embed_batch_size,
        head_hidden=cli_args.head_hidden,
        head_dropout=cli_args.head_dropout,
        epochs=cli_args.epochs,
        patience=cli_args.patience,
        lr=cli_args.lr,
        prompt_lr=cli_args.prompt_lr,
        weight_decay=cli_args.weight_decay,
        early_stop_metric=cli_args.early_stop_metric,
        peprompt_early_stop_mode=cli_args.peprompt_early_stop_mode,
        hgmp_prompt_recipe=cli_args.hgmp_prompt_recipe,
        hgmp_prompt_prompt_lr=cli_args.hgmp_prompt_prompt_lr,
        hgmp_prompt_head_lr=cli_args.hgmp_prompt_head_lr,
        hgmp_prompt_weight_decay=cli_args.hgmp_prompt_weight_decay,
        hgmp_prompt_batch_size=cli_args.hgmp_prompt_batch_size,
        hgmp_prompt_epochs=cli_args.hgmp_prompt_epochs,
        hgmp_prompt_patience=cli_args.hgmp_prompt_patience,
        hgmp_prompt_early_stop_mode=cli_args.hgmp_prompt_early_stop_mode,
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
        method_dir = method
        if method == "peprompt":
            method_dir = f"{method}.{args.subgraph_type}"
        save_dir = (
            Path(args.save_dir)
            / "aligned_protocol"
            / args.dataset
            / method_dir
            / f"{args.shot}-shot"
            / f"pretrainseed{cli_args.pretrain_seed}"
            / f"splitseed{split_seed}"
            / f"repeat{repeat_id}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = str(save_dir / "best.pt")

        if method == "peprompt":
            res = train_peprompt_probe(
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
                dataset=args.dataset,
                classification_type=args.classification_type,
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
                "downstream/subgraph_type": getattr(args, "subgraph_type", "na"),
            }
        )
        log_metrics(wandb_run, payload)

    return _callback


def build_parser():
    ap = argparse.ArgumentParser("Aligned protocol benchmark for hgmp / peprompt / hgprompt")

    # 基础实验设置：控制数据集、few-shot 协议和重复实验方式。
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])  # 数据集名称
    ap.add_argument("--root", type=str, default="data")  # 原始数据根目录
    ap.add_argument("--splits", type=str, default="splits")  # few-shot 划分文件目录
    ap.add_argument("--shot", type=int, default=1)  # 每类样本数
    ap.add_argument("--methods", nargs="+", default=["hgmp", "peprompt", "hgprompt"])  # 需要对比运行的方法列表
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])  # few-shot 划分随机种子列表
    ap.add_argument("--repeats", type=int, default=50)  # 每个 split seed 下的重复次数
    ap.add_argument("--pretrain_seed", type=int, default=0)  # 预训练 checkpoint 对应的随机种子
    ap.add_argument("--run_seed_base", type=int, default=0)  # 下游运行随机种子基值
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")  # 训练/评测设备
    ap.add_argument("--save_dir", type=Path, default=ROOT / "artifacts" / "results" / "protocol_benchmark")  # 结果输出目录

    # Checkpoint 设置：PEPrompt 只保留显式路径；HGMP/HGPrompt 仍兼容 pattern。
    ap.add_argument("--hgmp_ckpt", type=str, default=None)  # HGMP checkpoint 路径
    ap.add_argument("--peprompt_ckpt", type=str, default=None)  # PEPrompt checkpoint 路径，默认可回退到 hgmp_ckpt
    ap.add_argument("--hgprompt_ckpt", type=str, default=None)  # HGPrompt checkpoint 路径
    # ap.add_argument("--hgmp_ckpt_pattern", type=str, default=None)  # HGMP checkpoint 路径模板
    # ap.add_argument("--hgprompt_ckpt_pattern", type=str, default=None)  # HGPrompt checkpoint 路径模板

    # 共享模型和下游 probe 设置：适用于 hgmp / peprompt / hgmp_prompt 的统一评测。
    ap.add_argument("--feats_type", type=int, default=0)  # 输入特征构造方式
    ap.add_argument("--hidden_dim", type=int, default=512)  # HGNN 隐藏维度
    ap.add_argument("--num_heads", type=int, default=2)  # HGT 注意力头数
    ap.add_argument("--num_layers", type=int, default=2)  # HGNN 层数
    ap.add_argument("--dropout", type=float, default=0.5)  # 编码器 dropout
    ap.add_argument("--hgnn_type", type=str, default="GCN")  # 编码器类型，如 GCN / HGT
    ap.add_argument("--num_samples", type=int, default=500)  # 预训练采样规模或 legacy 接口所需样本数
    ap.add_argument("--num_class", type=int, default=None)  # 下游分类类别数，默认按数据集自动推断
    ap.add_argument("--classification_type", type=str, default="NIG")   # 分类任务类型，默认是节点诱导子图分类
    ap.add_argument("--embed_batch_size", type=int, default=32)  # 图编码/提取 embedding 时的 batch size
    ap.add_argument("--head_hidden", type=int, default=128)  # 下游 MLP probe 的隐藏层维度
    ap.add_argument("--head_dropout", type=float, default=0.3)  # 下游 MLP probe 的 dropout
    ap.add_argument("--epochs", type=int, default=200)  # 下游训练最大 epoch 数
    ap.add_argument("--patience", type=int, default=30)  # early stopping 容忍轮数
    ap.add_argument("--lr", type=float, default=5e-3)  # 分类头学习率
    ap.add_argument("--prompt_lr", type=float, default=None)  # prompt 学习率，未指定时回退到 lr
    ap.add_argument("--weight_decay", type=float, default=5e-4)  # 优化器权重衰减
    ap.add_argument("--early_stop_metric", type=str, default="macro", choices=["micro", "macro"])  # early stopping 监控指标
    ap.add_argument(
        "--hgmp_prompt_recipe",
        type=str,
        default="legacy",
        choices=["bridge", "legacy"],
        help="Use legacy HGMP prompt training by default so the method stays close to the original paper code.",
    )
    ap.add_argument("--hgmp_prompt_prompt_lr", type=float, default=None)  # HGMP prompt 本体学习率
    ap.add_argument("--hgmp_prompt_head_lr", type=float, default=None)  # HGMP prompt 分类头学习率
    ap.add_argument("--hgmp_prompt_weight_decay", type=float, default=None)  # HGMP prompt 权重衰减
    ap.add_argument("--hgmp_prompt_batch_size", type=int, default=None)  # HGMP prompt 训练 batch size
    ap.add_argument("--hgmp_prompt_epochs", type=int, default=None)  # HGMP prompt 最大训练轮数
    ap.add_argument("--hgmp_prompt_patience", type=int, default=None)  # HGMP prompt early stopping 容忍轮数
    ap.add_argument(
        "--hgmp_prompt_early_stop_mode",
        type=str,
        default="metric",
        choices=["auto", "metric", "legacy_loss"],
        help="Use validation metric selection by default so hgmp_prompt and peprompt share the same model-selection protocol.",
    )

    # 关系提示注入设置：控制消息是加法还是乘法，以及聚合和归一化方式。
    ap.add_argument("--relation_prompt_mode", type=str, default="mul", choices=["mul", "add"])  # 提示注入方式：mul 对应乘法，add 对应加法
    ap.add_argument("--relation_prompt_alpha", type=float, default=0.5)  # 聚合后关系提示残差注入强度
    ap.add_argument("--relation_prompt_dropout", type=float, default=0.1)  # 关系提示层输出 dropout
    ap.add_argument("--relation_prompt_aggr", type=str, default="mean", choices=["mean", "sum"])  # 入边消息聚合方式
    ap.add_argument("--relation_prompt_use_ln", action="store_true")  # 是否在关系提示输出后使用 LayerNorm

    # PEPrompt 专属设置：只保留基于 Laplacian PE 的边提示参数。
    ap.add_argument("--peprompt_edge_feature_dim", type=int, default=0)  # 实际挂到边上的特征维度，运行时会被自动更新
    ap.add_argument("--peprompt_edge_feature_names", nargs="*", default=None, choices=PEPROMPT_EDGE_FEATURES)  # 启用的边特征名称，当前仅支持谱差值
    ap.add_argument("--peprompt_edge_feature_name", type=str, default=PEPROMPT_EDGE_FEATURE_NAME)  # DGL 图中保存 PEPrompt 边特征的字段名
    ap.add_argument(
        "--peprompt_spectral_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_spectral_embeddings",
    )  # Laplacian PE 缓存目录
    ap.add_argument(
        "--peprompt_offline_cache_dir",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "peprompt_offline_splits",
    )  # 严格 k-shot + 子图 + 边特征离线缓存目录
    ap.add_argument("--subgraph_type", type=str, default="khop", choices=["khop", "fanout", "metapath_topk"])  # 选择要加载的离线子图缓存类型
    ap.add_argument("--metapath_max_hop", type=int, default=3)  # metapath_topk 最大元路径长度 M
    ap.add_argument("--metapath_topk", type=int, default=5)  # 每条元路径保留的 top-k 终点邻居数
    ap.add_argument("--metapath_rank_metric", type=str, default="count", choices=["count", "degree_norm"])  # top-k 排序指标
    ap.add_argument("--metapath_keep_self", action="store_true")  # 是否允许同类型元路径把目标节点自身纳入 top-k
    ap.add_argument("--peprompt_write_edge_feature_stats", action=argparse.BooleanOptionalAction, default=True)  # 是否写出边特征统计信息
    ap.add_argument("--peprompt_spectral_dim", type=int, default=16)  # Laplacian PE 维度
    ap.add_argument("--peprompt_spectral_max_nodes", type=int, default=50000)  # 计算谱分解允许的最大节点数
    ap.add_argument("--peprompt_edge_prompt_hidden", type=int, default=128)  # 将 PE 边特征映射为提示向量的 MLP 隐层维度
    ap.add_argument(
        "--peprompt_early_stop_mode",
        type=str,
        default="loss",
        choices=["metric", "loss"],
        help="PEPrompt downstream early stopping: validation metric or validation loss.",
    )

    # ---- hop-decomposed compensation (Module 1+2) ----
    ap.add_argument(
        "--peprompt_fusion_mode",
        type=str,
        default="none",
        choices=["none", "hop_decoupled", "onehop_ctx", "type_ctx", "graph_summary", "graph_summary_basis"],
        help="Edge prompt fusion mode: 'none' = spectral PE only; "
             "'hop_decoupled' = PE + dropped-neighbour context + h_src + h_dst; "
             "'onehop_ctx' = 1-hop type-pooled dropped context on 1-hop edges; "
             "'type_ctx' = one pooled dropped-context per dst node type per subgraph; "
             "'graph_summary' = shared graph-level metapath summary for all edges in a subgraph; "
             "'graph_summary_basis' = use graph summary as a selector over prompt bases.",
    )
    ap.add_argument(
        "--peprompt_ctx_dim",
        type=int,
        default=0,
        help="Dimension of the dropped-neighbour context vector (must match node feature dim). "
             "Used when --peprompt_fusion_mode is hop_decoupled, onehop_ctx, or type_ctx.",
    )
    ap.add_argument(
        "--peprompt_graph_summary_dim",
        type=int,
        default=0,
        help="Graph-level metapath summary dimension; populated from offline cache when --peprompt_fusion_mode is graph_summary or graph_summary_basis.",
    )
    ap.add_argument(
        "--peprompt_basis_count",
        type=int,
        default=4,
        help="Number of global prompt bases used when --peprompt_fusion_mode=graph_summary_basis.",
    )
    ap.add_argument(
        "--peprompt_generator_hidden",
        type=int,
        default=128,
        help="Hidden dimension of the fusion MLP when --peprompt_fusion_mode=hop_decoupled.",
    )
    ap.add_argument(
        "--peprompt_metapath_embed_dim",
        type=int,
        default=16,
        help="Metapath-id embedding dimension used by hop_decoupled weighted-metapath context.",
    )
    ap.add_argument(
        "--peprompt_onehop_center_fusion",
        action="store_true",
        help="When --peprompt_fusion_mode=onehop_ctx, also fuse the pooled 1-hop dropped context into the centre node.",
    )

    # HGPrompt 专属设置：仅在 methods 包含 hgprompt 时使用。
    ap.add_argument("--hgprompt_feats_type", type=int, default=2)  # HGPrompt 输入特征类型
    ap.add_argument("--hgprompt_hidden_dim", type=int, default=64)  # HGPrompt 隐藏维度
    ap.add_argument("--hgprompt_bottle_net_hidden_dim", type=int, default=2)  # BottleNet 中间层维度
    ap.add_argument("--hgprompt_bottle_net_output_dim", type=int, default=64)  # BottleNet 输出维度
    ap.add_argument("--hgprompt_edge_feats", type=int, default=64)  # HGPrompt 边特征维度
    ap.add_argument("--hgprompt_num_heads", type=int, default=8)  # HGPrompt 注意力头数
    ap.add_argument("--hgprompt_model_type", type=str, default="gcn", choices=["gcn", "gat", "gin", "SHGN"])  # HGPrompt 主干模型类型
    ap.add_argument("--hgprompt_num_layers", type=int, default=2)  # HGPrompt 层数
    ap.add_argument("--hgprompt_lr", type=float, default=1.0)  # HGPrompt 学习率
    ap.add_argument("--hgprompt_dropout", type=float, default=0.5)  # HGPrompt dropout
    ap.add_argument("--hgprompt_weight_decay", type=float, default=1e-6)  # HGPrompt 权重衰减
    ap.add_argument("--hgprompt_slope", type=float, default=0.05)  # GAT/激活相关斜率参数
    ap.add_argument("--hgprompt_tuning", type=str, default="weight-sum-center-fixed")  # HGPrompt 调参模式
    ap.add_argument("--hgprompt_subgraph_hop_num", type=int, default=1)  # HGPrompt 子图 hop 数
    ap.add_argument("--hgprompt_pre_loss_weight", type=float, default=1.0)  # 预训练损失权重
    ap.add_argument("--hgprompt_hetero_pretrain", type=int, default=0)  # 是否启用异构预训练
    ap.add_argument("--hgprompt_hetero_pretrain_subgraph", type=int, default=0)  # 是否对子图启用异构预训练
    ap.add_argument("--hgprompt_pretrain_semantic", type=int, default=0)  # 是否启用语义预训练信号
    ap.add_argument("--hgprompt_pretrain_each_loss", type=int, default=0)  # 是否分别记录各项预训练损失
    ap.add_argument("--hgprompt_add_edge_info2prompt", type=int, default=1)  # 是否把边信息加入 prompt
    ap.add_argument("--hgprompt_each_type_subgraph", type=int, default=1)  # 是否按节点类型构建子图
    ap.add_argument("--hgprompt_cat_prompt_dim", type=int, default=64)  # 拼接 prompt 的维度
    ap.add_argument("--hgprompt_cat_hprompt_dim", type=int, default=64)  # 拼接 hetero prompt 的维度
    ap.add_argument("--hgprompt_tuple_neg_disconnected_num", type=int, default=1)  # 断连负样本数量
    ap.add_argument("--hgprompt_tuple_neg_unrelated_num", type=int, default=1)  # 无关负样本数量
    ap.add_argument("--hgprompt_semantic_prompt", type=int, default=1)  # 是否启用 semantic prompt
    ap.add_argument("--hgprompt_semantic_prompt_weight", type=float, default=0.1)  # semantic prompt 损失权重
    ap.add_argument("--hgprompt_freebase_type", type=int, default=0)  # Freebase 特定配置
    ap.add_argument("--hgprompt_shgn_hidden_dim", type=int, default=3)  # SHGN 隐藏维度

    # W&B 记录设置：控制在线/离线日志、项目名和 artifact 输出。
    ap.add_argument("--use_wandb", action="store_true")  # 是否启用 W&B
    ap.add_argument("--wandb_project", type=str, default="HGEP")  # W&B project 名称
    ap.add_argument("--wandb_entity", type=str, default=None)  # W&B entity / team 名称
    ap.add_argument("--wandb_name", type=str, default=None)  # 当前 run 名称
    ap.add_argument("--wandb_group", type=str, default=None)  # 当前 run 分组名称
    ap.add_argument("--wandb_job_type", type=str, default="peprompt_benchmark")  # 当前任务类型标记
    ap.add_argument("--wandb_tags", nargs="*", default=[])  # W&B 标签列表
    ap.add_argument("--wandb_notes", type=str, default=None)  # W&B 备注
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])  # W&B 运行模式
    ap.add_argument("--wandb_dir", type=Path, default=ROOT / "artifacts" / "wandb")  # W&B 本地缓存目录
    ap.add_argument("--wandb_key", type=str, default=None)  # W&B API Key
    ap.add_argument("--wandb_key_file", type=Path, default=ROOT / ".codex")  # 保存 API Key 的文件路径

    return ap


def run_benchmark(args):
    wandb_run = None
    try:
        records: list[RunRecord] = []
        ckpt_by_method = {method: _resolve_ckpt(args, method) for method in args.methods}
        args = _sync_args_with_legacy_ckpts(args, ckpt_by_method)
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
                tags=args.wandb_tags or [args.dataset, f"{args.shot}-shot", "peprompt-benchmark"],
                notes=args.wandb_notes,
                mode=args.wandb_mode,
                dir_path=args.wandb_dir,
                config=config,
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
                    elif method == "peprompt":
                        record = run_legacy_method_once(args, "peprompt", ckpt_path, split_seed, repeat_id, epoch_callback=epoch_logger)
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

        run_tag = "mixed-methods"
        if len(args.methods) == 1:
            run_tag = args.methods[0]
            if run_tag == "peprompt":
                run_tag = f"{run_tag}.{_peprompt_cache_subgraph_type(args)}"
        out_dir = _ensure_dir(args.save_dir / args.dataset / f"{args.shot}-shot" / run_tag)
        per_run_rows = []
        for record in records:
            row = asdict(record)
            row["subgraph_type"] = _peprompt_cache_subgraph_type(args) if record.method == "peprompt" else None
            per_run_rows.append(row)
        _write_csv(out_dir / "per_run.csv", per_run_rows)

        seed_rows = _aggregate_by_seed(records)
        seed_row_dicts = []
        for row in seed_rows:
            payload = asdict(row)
            payload["subgraph_type"] = _peprompt_cache_subgraph_type(args) if row.method == "peprompt" else None
            seed_row_dicts.append(payload)
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
            name=f"{args.dataset}-{args.shot}shot-peprompt-benchmark-{args.pretrain_seed}",
            artifact_type="benchmark-results",
        )

        print("####################################################")
        final_summary = {
            "pooled_runs": pooled_summary,
            "seed_mean_then_std": seedmean_summary,
        }
        print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    except Exception:
        finish_wandb_run(wandb_run, exit_code=1)
        raise
    else:
        finish_wandb_run(wandb_run, exit_code=0)
    return {
        "records": records,
        "per_run_rows": per_run_rows,
        "seed_rows": seed_row_dicts,
        "config": config,
        "pooled_runs": pooled_summary,
        "seed_mean_then_std": seedmean_summary,
        "out_dir": out_dir,
        "resolved_ckpt_by_method": ckpt_by_method,
    }


def main():
    args = build_parser().parse_args()
    _normalize_dataset_defaults(args)
    run_benchmark(args)


if __name__ == "__main__":
    main()
