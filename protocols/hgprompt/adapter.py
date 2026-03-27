from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

from gpbench.data.loaders import load_hgb_node_task
from gpbench.downstream.fewshot import load_split_file
from protocols.hgprompt.utils.data_loader import data_loader


RAW_SUBDIR = {
    "ACM": "acm/raw/ACM",
    "DBLP": "dblp/raw/DBLP",
    "IMDB": "imdb/raw/IMDB",
    "Freebase": "freebase/raw/Freebase",
}

# 对 HGB 节点分类数据，旧 HGPrompt/raw loader 中目标节点类型都是 0 号类型
TARGET_TYPE_ID = {
    "ACM": 0,       # paper
    "DBLP": 0,      # author
    "IMDB": 0,      # movie
    "Freebase": 0,  # book
}


@dataclass
class HPromptDownstreamBundle:
    dataset: str
    dataset_name: str
    raw_dir: Path
    target_ntype: str
    target_type_id: int

    features_list: list[Any]
    adjM: sp.csr_matrix
    labels: Dict[str, Any]
    train_val_test_idx: Dict[str, Any]

    dl: Any
    split_obj: Dict[str, Any]


def _ensure_dict(split: Any) -> Dict[str, Any]:
    if isinstance(split, dict):
        return split
    if hasattr(split, "__dict__"):
        return split.__dict__
    raise TypeError(f"Unsupported split object type: {type(split)}")


def _onehot_or_dense_to_class_ids(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(np.int64)

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr.reshape(-1).astype(np.int64)
        return arr.argmax(axis=1).astype(np.int64)

    raise ValueError(f"Unsupported label array shape: {arr.shape}")


def _build_features_and_adj(dl: Any):
    features = []
    for i in range(len(dl.nodes["count"])):
        feat = dl.nodes["attr"][i]
        if feat is None:
            features.append(sp.eye(dl.nodes["count"][i], format="csr"))
        else:
            features.append(feat)
    adjM = sum(dl.links["data"].values())
    return features, adjM


def _map_local_target_idx_to_global(split_idx: np.ndarray, shift: int) -> np.ndarray:
    split_idx = np.asarray(split_idx, dtype=np.int64).reshape(-1)
    return split_idx + int(shift)


def load_hgprompt_downstream_bundle(
    root: str | Path,
    dataset: str,
    splits: str | Path,
    shot: int,
    seed: int,
) -> HPromptDownstreamBundle:
    dataset = str(dataset)
    if dataset not in RAW_SUBDIR:
        raise ValueError(f"Unsupported dataset: {dataset}")

    root = Path(root)
    raw_dir = root / RAW_SUBDIR[dataset]

    # 这一步沿用 HGEP 当前 few-shot / dataset_name 约定
    task = load_hgb_node_task(root, dataset)
    split = _ensure_dict(load_split_file(str(splits), task.dataset_name, shot, seed))

    dl = data_loader(str(raw_dir))
    features_list, adjM = _build_features_and_adj(dl)

    target_type_id = TARGET_TYPE_ID[dataset]
    target_shift = int(dl.nodes["shift"][target_type_id])

    train_idx_local = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx_local = np.asarray(split.get("val_idx", np.array([], dtype=np.int64)), dtype=np.int64)
    test_idx_local = np.asarray(split["test_idx"], dtype=np.int64)

    train_idx = _map_local_target_idx_to_global(train_idx_local, target_shift)
    val_idx = _map_local_target_idx_to_global(val_idx_local, target_shift)
    test_idx = _map_local_target_idx_to_global(test_idx_local, target_shift)

    # 旧 HGPrompt/raw loader 的标签分散在 train/test 文件里，这里合并成全局 one-hot
    all_labels = dl.labels_train["data"] + dl.labels_test["data"]

    train_labels = _onehot_or_dense_to_class_ids(all_labels[train_idx])
    val_labels = _onehot_or_dense_to_class_ids(all_labels[val_idx]) if len(val_idx) > 0 else np.array([], dtype=np.int64)
    test_labels = _onehot_or_dense_to_class_ids(all_labels[test_idx])

    labels = {
        # 保留旧 run.py 喜欢的结构：train/val 是 list，便于后面兼容 tasknum=1
        "train": [train_labels],
        "val": [val_labels],
        "test": test_labels,
    }

    train_val_test_idx = {
        "train_idx": [train_idx],
        "val_idx": [val_idx],
        "test_idx": test_idx,
        "train_idx_local": [train_idx_local],
        "val_idx_local": [val_idx_local],
        "test_idx_local": test_idx_local,
    }

    return HPromptDownstreamBundle(
        dataset=dataset,
        dataset_name=task.dataset_name,
        raw_dir=raw_dir,
        target_ntype=task.target_ntype,
        target_type_id=target_type_id,
        features_list=features_list,
        adjM=adjM,
        labels=labels,
        train_val_test_idx=train_val_test_idx,
        dl=dl,
        split_obj=split,
    )