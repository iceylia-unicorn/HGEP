from __future__ import annotations

from dataclasses import dataclass

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

from protocols.hgmp.data_legacy import multi_class_NIG
from protocols.hgmp.prompt_legacy import GAT, GCL_GCN, HGNN, HeteroPrompt
from protocols.hgmp.utils_legacy import (
    get_graph_metadata_lightweight,
    graph_pool,
    load_data4pretrain,
    seed_everything,
)

from gpbench.downstream.model import MLPHead
from gpbench.protocol_bridge.hgmp_typepair import (
    HGMPTypePairHGNN,
    build_typepair_relation_cfg_from_args,
)
from gpbench.protocol_bridge.hgmp_peprompt import (
    HGMPPEPromptHGNN,
    build_peprompt_relation_cfg_from_args,
)


@dataclass
class LegacyFewShotEmbeddings:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    targetnode: str


LEGACY_HGMP_PROMPT_DATASET_DEFAULTS = {
    "ACM": {
        "prompt_lr": 5e-2,
        "head_lr": 5e-4,
        "weight_decay": 5e-5,
        "batch_size": 10,
        "epochs": 300,
        "patience": 30,
    },
    "IMDB": {
        "prompt_lr": 5e-2,
        "head_lr": 5e-2,
        "weight_decay": 1e-4,
        "batch_size": 10,
        "epochs": 300,
        "patience": 30,
    },
    "oldfreebase": {
        "prompt_lr": 5e-3,
        "head_lr": 5e-3,
        "weight_decay": 1e-5,
        "batch_size": 10,
        "epochs": 300,
        "patience": 30,
    },
}


def _hgmp_prompt_legacy_defaults(dataset: str) -> dict:
    return dict(LEGACY_HGMP_PROMPT_DATASET_DEFAULTS.get(dataset, {}))


def _clone_state_dict(module: nn.Module) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _first_graph_from_sample(sample, classification_type: str):
    if classification_type == "NIG":
        return sample[0]
    return sample[0]


def _infer_targetnode(graph) -> str:
    if "y" not in graph.ndata:
        raise ValueError("Cannot infer target node type: graph.ndata has no 'y'")
    keys = list(graph.ndata["y"].keys())
    if len(keys) == 0:
        raise ValueError("Cannot infer target node type: graph.ndata['y'] has no keys")
    return keys[0]


def _unpack_batch(batch, classification_type: str):
    if classification_type == "NIG":
        if len(batch) == 3:
            batched_graph, _, batched_label = batch
        else:
            raise ValueError(
                "Expected NIG batch to have 3 items: (graph, extra, label). "
                f"Got len={len(batch)}"
            )
    else:
        if len(batch) == 2:
            batched_graph, batched_label = batch
        else:
            raise ValueError(
                f"Expected {classification_type} batch to have 2 items. "
                f"Got len={len(batch)}"
            )
    return batched_graph, batched_label


def _is_legacy_multilabel_task(dataset: str, classification_type: str) -> bool:
    return str(dataset) == "IMDB" and str(classification_type) != "GIG"


def _labels_look_multilabel(labels: torch.Tensor) -> bool:
    return isinstance(labels, torch.Tensor) and labels.ndim == 2 and labels.size(-1) > 1


def _prepare_labels_for_task(
    labels: torch.Tensor,
    device: torch.device,
    dataset: str,
    classification_type: str,
) -> torch.Tensor:
    labels = labels.to(device)
    if _is_legacy_multilabel_task(dataset, classification_type) and _labels_look_multilabel(labels):
        return labels.float()
    return labels.long()


def _legacy_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    dataset: str,
    classification_type: str,
) -> torch.Tensor:
    if _is_legacy_multilabel_task(dataset, classification_type) and _labels_look_multilabel(labels):
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    return nn.functional.cross_entropy(logits, labels.long())


def _to_class_ids(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 1:
        return x.long()
    if x.ndim == 2:
        if x.shape[1] == 1:
            return x.view(-1).long()
        return x.argmax(dim=1).long()
    raise ValueError(f"Unsupported tensor shape for class ids: {tuple(x.shape)}")


def _legacy_multihot_f1_micro_macro(
    pred: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> tuple[float, float]:
    """True multilabel IMDB protocol: sigmoid-threshold predictions vs multi-hot labels."""
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
    if pred.ndim != 2 or pred.size(1) != num_classes:
        raise ValueError(f"Expected logits with shape [N, {num_classes}], got {tuple(pred.shape)}")

    pred_probs = torch.sigmoid(pred.to(torch.float32))
    pred_hot = (pred_probs >= 0.5).to(torch.float32)

    target = y.to(pred_hot.device).to(torch.float32)
    if target.ndim != 2 or target.size(1) != num_classes:
        raise ValueError(f"Expected multi-hot labels with shape [N, {num_classes}], got {tuple(target.shape)}")
    target = (target > 0).to(torch.float32)

    tp = (pred_hot * target).sum(dim=0)
    fp = (pred_hot * (1.0 - target)).sum(dim=0)
    fn = ((1.0 - pred_hot) * target).sum(dim=0)
    eps = 1e-12

    f1_c = (2 * tp) / (2 * tp + fp + fn + eps)
    support = target.sum(dim=0)
    mask = support > 0
    macro = float(f1_c[mask].mean().item()) if mask.any() else 0.0

    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    micro = float((2 * TP / (2 * TP + FP + FN + eps)).item())
    return micro, macro


def _legacy_f1_micro_macro(
    pred: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> tuple[float, float]:
    if _labels_look_multilabel(y):
        return _legacy_multihot_f1_micro_macro(pred, y, num_classes)

    pred = _to_class_ids(pred)
    y = _to_class_ids(y)

    cm = torch.zeros((num_classes, num_classes), device=pred.device, dtype=torch.long)
    for t, p in zip(y.view(-1), pred.view(-1)):
        if int(t) < 0 or int(p) < 0:
            continue
        if int(t) >= num_classes or int(p) >= num_classes:
            continue
        cm[t.long(), p.long()] += 1

    tp = cm.diag().to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp
    eps = 1e-12

    support = cm.sum(dim=1).to(torch.float32)
    f1_c = (2 * tp) / (2 * tp + fp + fn + eps)
    mask = support > 0
    macro = float(f1_c[mask].mean().item()) if mask.any() else 0.0

    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    micro = float((2 * TP / (2 * TP + FP + FN + eps)).item())
    return micro, macro


def _build_edge_index_dict(g):
    edge_index_dict = {}
    for etype in g.canonical_etypes:
        edge_index = g.edges(etype=etype)
        src = edge_index[0].unsqueeze(0)
        dst = edge_index[1].unsqueeze(0)
        edge_index_dict[etype] = torch.cat((src, dst), dim=0)
    return edge_index_dict


def _prepare_edge_indices_for_device(hgnn, g, device: torch.device):
    if hgnn.hgnn_type != "HGT" and not _uses_edge_features(hgnn):
        return None
    edge_index_dict = _build_edge_index_dict(g)
    return {
        etype: edge_index.to(device)
        for etype, edge_index in edge_index_dict.items()
    }


def _build_edge_feature_dict(g, feature_name: str = "typepair_edge_feat"):
    edge_feature_dict = {}
    for etype in g.canonical_etypes:
        edge_data = g.edges[etype].data
        if feature_name not in set(edge_data.keys()):
            continue
        edge_feature_dict[etype] = edge_data[feature_name].float()
    return edge_feature_dict or None


def _prepare_edge_features_for_device(hgnn, g, device: torch.device):
    if not _uses_edge_features(hgnn):
        return None
    edge_feature_name = getattr(hgnn.relation_prompt, "edge_feature_name", "typepair_edge_feat")
    edge_feature_dict = _build_edge_feature_dict(g, edge_feature_name)
    if edge_feature_dict is None:
        return None
    return {
        etype: feat.to(device)
        for etype, feat in edge_feature_dict.items()
    }


def _prepare_homo_graph_for_device(hgnn, g, device: torch.device):
    if hgnn.hgnn_type != "GCN":
        return None
    homo_g = dgl.to_homogeneous(g)
    homo_g = dgl.remove_self_loop(homo_g)
    homo_g = dgl.add_self_loop(homo_g)
    return homo_g.to(device)


def _drop_training_edges(g, drop_prob: float):
    drop_prob = float(drop_prob or 0.0)
    if drop_prob <= 0.0:
        return g
    if drop_prob >= 1.0:
        raise ValueError("peprompt_edge_dropout must be < 1.0")

    out = g
    for etype in g.canonical_etypes:
        num_edges = int(out.num_edges(etype=etype))
        if num_edges <= 0:
            continue
        drop_mask = torch.rand(num_edges) < drop_prob
        if not bool(drop_mask.any()):
            continue
        eids = torch.nonzero(drop_mask, as_tuple=False).view(-1)
        out = dgl.remove_edges(out, eids, etype=etype)
    return out


def _uses_edge_features(hgnn) -> bool:
    relation_prompt = getattr(hgnn, "relation_prompt", None)
    return bool(getattr(relation_prompt, "uses_edge_features", False))


def forward_graph_batch(
    hgnn,
    batched_graph,
    targetnode: str,
    edge_feature_dict=None,
    edge_index_dict=None,
    homo_graph=None,
) -> torch.Tensor:
    x_dict = batched_graph.ndata["x"]
    if edge_index_dict is None and hgnn.hgnn_type == "HGT":
        edge_index_dict = _build_edge_index_dict(batched_graph)
    if edge_feature_dict is None and _uses_edge_features(hgnn):
        edge_feature_name = getattr(hgnn.relation_prompt, "edge_feature_name", "typepair_edge_feat")
        edge_feature_dict = _build_edge_feature_dict(batched_graph, edge_feature_name)

    if hgnn.hgnn_type == "HGT":
        if edge_feature_dict is None:
            node_emb = hgnn(targetnode, x_dict, edge_index_dict, graph=batched_graph)
        else:
            node_emb = hgnn(targetnode, x_dict, edge_index_dict, edge_feature_dict=edge_feature_dict, graph=batched_graph)
    elif hgnn.hgnn_type == "SHGN":
        node_emb = hgnn(targetnode, batched_graph, x_dict)
    elif hgnn.hgnn_type == "GCN":
        # Plain legacy HGMP GCN keeps the original `(graph, x_dict)` signature,
        # while prompt-injected wrappers accept extra keyword arguments.
        if isinstance(hgnn, GCL_GCN):
            node_emb = hgnn(batched_graph, x_dict)
        elif edge_feature_dict is None:
            node_emb = hgnn(
                batched_graph,
                x_dict,
                edge_index=edge_index_dict,
                homo_graph=homo_graph,
            )
        else:
            node_emb = hgnn(
                batched_graph,
                x_dict,
                edge_index=edge_index_dict,
                edge_feature_dict=edge_feature_dict,
                homo_graph=homo_graph,
            )
    elif hgnn.hgnn_type == "GAT":
        node_emb = hgnn(batched_graph, x_dict, False)
    else:
        raise ValueError(f"Unsupported hgnn_type: {hgnn.hgnn_type}")

    graph_emb = graph_pool("mean", node_emb, batched_graph)
    return graph_emb


@torch.no_grad()
def encode_graph_batch(
    hgnn,
    batched_graph,
    targetnode: str,
    edge_feature_dict=None,
    edge_index_dict=None,
    homo_graph=None,
) -> torch.Tensor:
    return forward_graph_batch(
        hgnn,
        batched_graph,
        targetnode,
        edge_feature_dict=edge_feature_dict,
        edge_index_dict=edge_index_dict,
        homo_graph=homo_graph,
    )


def _build_legacy_hgnn(args):
    in_dims, ntypes, edge_types, _ = get_graph_metadata_lightweight(
        feats_type=args.feats_type,
        dataset=args.dataset,
        root_dir=args.root,
    )
    metadata = (ntypes, edge_types)
    num_etypes = len(edge_types) + 1

    if args.method in {"hgmp", "hgmp_prompt"}:
        if args.hgnn_type == "GCN":
            return GCL_GCN(
                None,
                in_dims,
                args.hidden_dim,
                args.hidden_dim,
                args.num_layers,
                F.elu,
                args.dropout,
                args.hgnn_type,
            )
        if args.hgnn_type == "GAT":
            heads = [args.num_heads] * args.num_layers + [1]
            return GAT(
                None,
                in_dims,
                args.hidden_dim,
                args.hidden_dim,
                args.num_layers,
                heads,
                F.elu,
                args.dropout,
                args.dropout,
                0.05,
                False,
                args.hgnn_type,
            )
        return HGNN(
            ntypes=ntypes,
            metadata=metadata,
            hid_dim=args.hidden_dim,
            out_dim=args.hidden_dim,
            hgnn_type=args.hgnn_type,
            num_layer=args.num_layers,
            num_heads=args.num_heads,
            device=args.device,
            dropout=args.dropout,
            num_etypes=num_etypes,
            input_dims=in_dims,
            args=args,
        )

    if args.method == "typepair":
        relation_cfg = build_typepair_relation_cfg_from_args(args)
        return HGMPTypePairHGNN(
            ntypes=ntypes,
            metadata=metadata,
            hid_dim=args.hidden_dim,
            out_dim=args.hidden_dim,
            hgnn_type=args.hgnn_type,
            num_layer=args.num_layers,
            num_heads=args.num_heads,
            device=args.device,
            dropout=args.dropout,
            num_etypes=num_etypes,
            input_dims=in_dims,
            args=args,
            relation_cfg=relation_cfg,
        )

    if args.method == "peprompt":
        relation_cfg = build_peprompt_relation_cfg_from_args(args)
        return HGMPPEPromptHGNN(
            ntypes=ntypes,
            metadata=metadata,
            hid_dim=args.hidden_dim,
            out_dim=args.hidden_dim,
            hgnn_type=args.hgnn_type,
            num_layer=args.num_layers,
            num_heads=args.num_heads,
            device=args.device,
            dropout=args.dropout,
            num_etypes=num_etypes,
            input_dims=in_dims,
            args=args,
            relation_cfg=relation_cfg,
        )

    raise ValueError(f"Unsupported method: {args.method}")


def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        p.requires_grad_(flag)


def _remap_plain_hgmp_state_for_relation_wrapper(args, state):
    if args.method not in {"typepair", "peprompt"}:
        return state, 0
    if not isinstance(state, dict):
        return state, 0
    if any(k.startswith("GraphConv.") for k in state.keys()):
        return state, 0

    remap_prefixes = ()
    if args.hgnn_type == "GCN":
        remap_prefixes = ("fc_list.", "layers.")
    elif args.hgnn_type == "HGT":
        remap_prefixes = ("lin_dict.", "convs.", "lin.")
    else:
        return state, 0

    remapped = {}
    remap_count = 0
    for key, value in state.items():
        if key.startswith(remap_prefixes):
            remapped[f"GraphConv.{key}"] = value
            remap_count += 1
        else:
            remapped[key] = value
    return remapped, remap_count


def _log_state_dict_load(args, missing, unexpected):
    if args.method in {"typepair", "peprompt"}:
        expected_missing_prefixes = ("relation_prompt.", "GraphConv.relation_prompt.")
        expected_missing = [k for k in missing if k.startswith(expected_missing_prefixes)]
        other_missing = [k for k in missing if not k.startswith(expected_missing_prefixes)]

        if len(expected_missing) > 0:
            print(
                f"[load_state_dict] {args.method} prompt keys were missing from ckpt "
                "(expected when loading a plain hgmp checkpoint)."
            )
        if len(other_missing) > 0:
            print("[load_state_dict] missing keys:", other_missing[:10], "...")
    elif len(missing) > 0:
        print("[load_state_dict] missing keys:", missing[:10], "...")

    if len(unexpected) > 0:
        print("[load_state_dict] unexpected keys:", unexpected[:10], "...")


def build_legacy_hgnn(
    args,
    ckpt_path: str,
    *,
    freeze: bool = True,
    train_relation_prompt_only: bool = False,
):
    hgnn = _build_legacy_hgnn(args)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    state, remap_count = _remap_plain_hgmp_state_for_relation_wrapper(args, state)
    if remap_count > 0:
        print(
            f"[load_state_dict] remapped {remap_count} plain HGMP keys to {args.method} wrapper keys."
        )

    missing, unexpected = hgnn.load_state_dict(state, strict=False)
    _log_state_dict_load(args, missing, unexpected)

    hgnn = hgnn.to(args.device)

    if freeze:
        hgnn.eval()
        _set_requires_grad(hgnn, False)

    if train_relation_prompt_only:
        if args.method not in {"typepair", "peprompt"}:
            raise ValueError("train_relation_prompt_only only supports relation-prompt methods.")
        _set_requires_grad(hgnn.relation_prompt, True)
        hgnn.eval()
        hgnn.relation_prompt.train()

    return hgnn


def build_frozen_legacy_hgnn(args, ckpt_path: str):
    return build_legacy_hgnn(args, ckpt_path, freeze=True, train_relation_prompt_only=False)


@torch.no_grad()
def extract_split_embeddings(
    graph_list,
    hgnn,
    targetnode: str,
    classification_type: str,
    batch_size: int,
    device: torch.device,
    dataset: str,
):
    loader = dgl.dataloading.GraphDataLoader(graph_list, batch_size=batch_size, shuffle=False)

    xs = []
    ys = []
    for batch in loader:
        batched_graph, batched_label = _unpack_batch(batch, classification_type)
        edge_index_dict = _prepare_edge_indices_for_device(hgnn, batched_graph, device)
        edge_feature_dict = _prepare_edge_features_for_device(hgnn, batched_graph, device)
        homo_graph = _prepare_homo_graph_for_device(hgnn, batched_graph, device)
        batched_graph = batched_graph.to(device)
        batched_label = _prepare_labels_for_task(
            batched_label,
            device,
            dataset,
            classification_type,
        )

        x = encode_graph_batch(
            hgnn,
            batched_graph,
            targetnode,
            edge_feature_dict=edge_feature_dict,
            edge_index_dict=edge_index_dict,
            homo_graph=homo_graph,
        )
        xs.append(x)
        ys.append(batched_label)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _load_legacy_fewshot_splits(args):
    if args.classification_type != "NIG":
        raise NotImplementedError("v1 only supports classification_type='NIG'")

    seed_everything(args.seed)

    train_list, valid_list, test_list = multi_class_NIG(
        dataname=args.dataset,
        num_class=args.num_class,
        shots=args.shot,
        classification_type=args.classification_type,
        feats_type=args.feats_type,
    )

    sample_graph = _first_graph_from_sample(train_list[0], args.classification_type)
    targetnode = _infer_targetnode(sample_graph)
    return train_list, valid_list, test_list, targetnode


def build_legacy_fewshot_embeddings(args, batch_size: int = 32) -> LegacyFewShotEmbeddings:
    train_list, valid_list, test_list, targetnode = _load_legacy_fewshot_splits(args)

    hgnn = build_frozen_legacy_hgnn(args, args.ckpt)

    x_train, y_train = extract_split_embeddings(
        train_list, hgnn, targetnode, args.classification_type, batch_size, args.device, args.dataset
    )
    x_val, y_val = extract_split_embeddings(
        valid_list, hgnn, targetnode, args.classification_type, batch_size, args.device, args.dataset
    )
    x_test, y_test = extract_split_embeddings(
        test_list, hgnn, targetnode, args.classification_type, batch_size, args.device, args.dataset
    )

    return LegacyFewShotEmbeddings(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        targetnode=targetnode,
    )


def _evaluate_graph_probe(
    graph_list,
    hgnn,
    head,
    targetnode: str,
    dataset: str,
    classification_type: str,
    batch_size: int,
    device: torch.device,
    num_classes: int,
):
    loader = dgl.dataloading.GraphDataLoader(graph_list, batch_size=batch_size, shuffle=False)

    logits_list = []
    labels_list = []

    hgnn.eval()
    head.eval()
    with torch.no_grad():
        for batch in loader:
            batched_graph, batched_label = _unpack_batch(batch, classification_type)
            edge_index_dict = _prepare_edge_indices_for_device(hgnn, batched_graph, device)
            edge_feature_dict = _prepare_edge_features_for_device(hgnn, batched_graph, device)
            homo_graph = _prepare_homo_graph_for_device(hgnn, batched_graph, device)
            batched_graph = batched_graph.to(device)
            batched_label = _prepare_labels_for_task(
                batched_label,
                device,
                dataset,
                classification_type,
            )

            graph_emb = encode_graph_batch(
                hgnn,
                batched_graph,
                targetnode,
                edge_feature_dict=edge_feature_dict,
                edge_index_dict=edge_index_dict,
                homo_graph=homo_graph,
            )
            logits = head(graph_emb)

            logits_list.append(logits)
            labels_list.append(batched_label)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return _legacy_f1_micro_macro(all_logits, all_labels, num_classes)


def _evaluate_graph_probe_loss(
    graph_list,
    hgnn,
    head,
    targetnode: str,
    dataset: str,
    classification_type: str,
    batch_size: int,
    device: torch.device,
):
    loader = dgl.dataloading.GraphDataLoader(graph_list, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_graphs = 0

    hgnn.eval()
    head.eval()
    with torch.no_grad():
        for batch in loader:
            batched_graph, batched_label = _unpack_batch(batch, classification_type)
            edge_index_dict = _prepare_edge_indices_for_device(hgnn, batched_graph, device)
            edge_feature_dict = _prepare_edge_features_for_device(hgnn, batched_graph, device)
            homo_graph = _prepare_homo_graph_for_device(hgnn, batched_graph, device)
            batched_graph = batched_graph.to(device)
            batched_label = _prepare_labels_for_task(
                batched_label,
                device,
                dataset,
                classification_type,
            )

            graph_emb = forward_graph_batch(
                hgnn,
                batched_graph,
                targetnode,
                edge_feature_dict=edge_feature_dict,
                edge_index_dict=edge_index_dict,
                homo_graph=homo_graph,
            )
            logits = head(graph_emb)
            loss = _legacy_task_loss(
                logits,
                batched_label,
                dataset,
                classification_type,
            )

            batch_n = batched_label.size(0)
            total_loss += loss.item() * batch_n
            total_graphs += batch_n

    return total_loss / max(total_graphs, 1)


def _train_relation_prompt_probe(
    args,
    batch_size: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    head_lr: float = 5e-3,
    prompt_lr: float | None = None,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 30,
    early_stop_metric: str = "macro",
    save_best_path: str | None = None,
    epoch_callback=None,
):
    assert early_stop_metric in {"micro", "macro"}

    train_list, valid_list, test_list, targetnode = _load_legacy_fewshot_splits(args)
    early_stop_mode = getattr(args, f"{args.method}_early_stop_mode", "metric")
    if early_stop_mode not in {"metric", "loss"}:
        raise ValueError(f"Unsupported {args.method}_early_stop_mode: {early_stop_mode}")
    eval_mode = getattr(args, f"{args.method}_eval_mode", "full")
    if eval_mode not in {"full", "early_stop_only"}:
        raise ValueError(f"Unsupported {args.method}_eval_mode: {eval_mode}")
    if eval_mode == "early_stop_only" and early_stop_mode != "loss":
        raise ValueError("PEPrompt eval_mode=early_stop_only requires peprompt_early_stop_mode=loss.")
    mp_reg_weight = float(getattr(args, "peprompt_mp_reg_weight", 0.0) or 0.0)
    edge_dropout = float(getattr(args, "peprompt_edge_dropout", 0.0) or 0.0)

    hgnn = build_legacy_hgnn(
        args,
        args.ckpt,
        freeze=True,
        train_relation_prompt_only=True,
    )

    head = MLPHead(
        in_dim=args.hidden_dim,
        hidden=hidden_dim,
        num_classes=args.num_class,
        dropout=dropout,
        use_ln=True,
    ).to(args.device)

    prompt_lr = head_lr if prompt_lr is None else prompt_lr
    opt = torch.optim.AdamW(
        [
            {
                "params": list(hgnn.relation_prompt.parameters()),
                "lr": prompt_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(head.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ]
    )

    train_loader = dgl.dataloading.GraphDataLoader(
        train_list,
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_micro = 0.0
    best_val_macro = 0.0
    test_at_best_micro = 0.0
    test_at_best_macro = 0.0
    best_epoch = -1
    bad_epochs = 0
    best_val_loss = float("inf")
    best_hgnn_state = None
    best_head_state = None

    for epoch in range(1, epochs + 1):
        hgnn.eval()
        hgnn.relation_prompt.train()
        head.train()

        epoch_loss = 0.0
        epoch_graphs = 0

        for batch in train_loader:
            batched_graph, batched_label = _unpack_batch(batch, args.classification_type)
            if edge_dropout > 0.0:
                batched_graph = _drop_training_edges(batched_graph, edge_dropout)
            edge_index_dict = _prepare_edge_indices_for_device(hgnn, batched_graph, args.device)
            edge_feature_dict = _prepare_edge_features_for_device(hgnn, batched_graph, args.device)
            homo_graph = _prepare_homo_graph_for_device(hgnn, batched_graph, args.device)
            batched_graph = batched_graph.to(args.device)
            batched_label = _prepare_labels_for_task(
                batched_label,
                args.device,
                args.dataset,
                args.classification_type,
            )

            if mp_reg_weight > 0.0 and hasattr(hgnn.relation_prompt, "reset_regularization_loss"):
                hgnn.relation_prompt.reset_regularization_loss()
            graph_emb = forward_graph_batch(
                hgnn,
                batched_graph,
                targetnode,
                edge_feature_dict=edge_feature_dict,
                edge_index_dict=edge_index_dict,
                homo_graph=homo_graph,
            )
            logits = head(graph_emb)
            loss = _legacy_task_loss(
                logits,
                batched_label,
                args.dataset,
                args.classification_type,
            )
            if mp_reg_weight > 0.0 and hasattr(hgnn.relation_prompt, "regularization_loss"):
                loss = loss + mp_reg_weight * hgnn.relation_prompt.regularization_loss(
                    device=loss.device,
                    dtype=loss.dtype,
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_n = batched_label.size(0)
            epoch_loss += loss.item() * batch_n
            epoch_graphs += batch_n

        train_loss = epoch_loss / max(epoch_graphs, 1)

        if eval_mode == "full" and epoch_callback is not None:
            train_micro, train_macro = _evaluate_graph_probe(
                train_list,
                hgnn,
                head,
                targetnode,
                args.dataset,
                args.classification_type,
                batch_size,
                args.device,
                args.num_class,
            )
        else:
            train_micro, train_macro = None, None

        if eval_mode == "full":
            val_micro, val_macro = _evaluate_graph_probe(
                valid_list,
                hgnn,
                head,
                targetnode,
                args.dataset,
                args.classification_type,
                batch_size,
                args.device,
                args.num_class,
            )
            test_micro, test_macro = _evaluate_graph_probe(
                test_list,
                hgnn,
                head,
                targetnode,
                args.dataset,
                args.classification_type,
                batch_size,
                args.device,
                args.num_class,
            )
        else:
            val_micro = val_macro = None
            test_micro = test_macro = None

        val_loss = None
        if early_stop_mode == "loss":
            val_loss = _evaluate_graph_probe_loss(
                valid_list,
                hgnn,
                head,
                targetnode,
                args.dataset,
                args.classification_type,
                batch_size,
                args.device,
            )
            improved = val_loss <= best_val_loss
        else:
            monitor = val_macro if early_stop_metric == "macro" else val_micro
            best_monitor = best_val_macro if early_stop_metric == "macro" else best_val_micro
            improved = monitor > best_monitor

        if improved:
            improved = True
            if eval_mode == "full":
                best_val_micro = val_micro
                best_val_macro = val_macro
                test_at_best_micro = test_micro
                test_at_best_macro = test_macro
            best_epoch = epoch
            bad_epochs = 0
            if val_loss is not None:
                best_val_loss = float(val_loss)
            best_hgnn_state = _clone_state_dict(hgnn)
            best_head_state = _clone_state_dict(head)

            if save_best_path is not None:
                payload = {
                    "hgnn_state": hgnn.state_dict(),
                    "head_state": head.state_dict(),
                    "in_dim": args.hidden_dim,
                    "hidden_dim": hidden_dim,
                    "num_classes": args.num_class,
                    "best_val_micro": best_val_micro,
                    "best_val_macro": best_val_macro,
                    "test_at_best_micro": test_at_best_micro,
                    "test_at_best_macro": test_at_best_macro,
                    "best_epoch": best_epoch,
                    "early_stop_metric": early_stop_metric,
                    "early_stop_mode": early_stop_mode,
                    "eval_mode": eval_mode,
                }
                if val_loss is not None:
                    payload["best_val_loss"] = float(val_loss)
                torch.save(payload, save_best_path)
        else:
            improved = False
            bad_epochs += 1

        if epoch_callback is not None:
            payload = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "best_val_micro": float(best_val_micro),
                "best_val_macro": float(best_val_macro),
                "test_at_best_micro": float(test_at_best_micro),
                "test_at_best_macro": float(test_at_best_macro),
                "best_epoch": int(best_epoch),
                "bad_epochs": int(bad_epochs),
                "is_best": bool(improved),
                "early_stop": bool(bad_epochs >= patience),
                "early_stop_mode": early_stop_mode,
                "eval_mode": eval_mode,
            }
            if eval_mode == "full":
                payload.update(
                    {
                        "train_micro": float(train_micro),
                        "train_macro": float(train_macro),
                        "val_micro": float(val_micro),
                        "val_macro": float(val_macro),
                        "test_micro": float(test_micro),
                        "test_macro": float(test_macro),
                    }
                )
            if val_loss is not None:
                payload["val_loss"] = float(val_loss)
                payload["best_val_loss"] = float(best_val_loss)
                payload["monitor"] = float(-val_loss)
            else:
                payload["monitor"] = float(monitor)
            epoch_callback(payload)

        if epoch == 1 or epoch % 10 == 0:
            if eval_mode == "early_stop_only":
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | best_val_loss={best_val_loss:.4f} | "
                    f"bad_epochs={bad_epochs}"
                )
            elif val_loss is not None:
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f} | "
                    f"test_f1(micro/macro)={test_micro:.4f}/{test_macro:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f} | "
                    f"test_f1(micro/macro)={test_micro:.4f}/{test_macro:.4f} | "
                    f"monitor({early_stop_metric})={monitor:.4f}"
                )

        if bad_epochs >= patience:
            if eval_mode == "early_stop_only":
                print(
                    f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                    f"best_val_loss={best_val_loss:.4f} | final F1 will be evaluated once on the best checkpoint"
                )
            elif val_loss is not None:
                print(
                    f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                    f"best_val_loss={best_val_loss:.4f} | "
                    f"best_val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
                    f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f}"
                )
            else:
                print(
                    f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                    f"best_val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
                    f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f} | "
                    f"monitor={early_stop_metric}"
                )
            break

    if best_hgnn_state is not None:
        hgnn.load_state_dict(best_hgnn_state)
    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    if eval_mode == "early_stop_only":
        best_val_micro, best_val_macro = _evaluate_graph_probe(
            valid_list,
            hgnn,
            head,
            targetnode,
            args.dataset,
            args.classification_type,
            batch_size,
            args.device,
            args.num_class,
        )
        test_at_best_micro, test_at_best_macro = _evaluate_graph_probe(
            test_list,
            hgnn,
            head,
            targetnode,
            args.dataset,
            args.classification_type,
            batch_size,
            args.device,
            args.num_class,
        )
        print(
            f"Final best-state F1 | best_epoch={best_epoch} | "
            f"val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
            f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f}"
        )
        if save_best_path is not None and best_hgnn_state is not None and best_head_state is not None:
            torch.save(
                {
                    "hgnn_state": hgnn.state_dict(),
                    "head_state": head.state_dict(),
                    "in_dim": args.hidden_dim,
                    "hidden_dim": hidden_dim,
                    "num_classes": args.num_class,
                    "best_val_micro": best_val_micro,
                    "best_val_macro": best_val_macro,
                    "test_at_best_micro": test_at_best_micro,
                    "test_at_best_macro": test_at_best_macro,
                    "best_epoch": best_epoch,
                    "early_stop_metric": early_stop_metric,
                    "early_stop_mode": early_stop_mode,
                    "eval_mode": eval_mode,
                    "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                },
                save_best_path,
            )

    return {
        "best_val_micro": best_val_micro,
        "best_val_macro": best_val_macro,
        "test_at_best_micro": test_at_best_micro,
        "test_at_best_macro": test_at_best_macro,
        "best_epoch": best_epoch,
        "early_stop_metric": early_stop_metric,
        "early_stop_mode": early_stop_mode,
        "eval_mode": eval_mode,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
    }


def train_typepair_prompt_probe(
    args,
    batch_size: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    head_lr: float = 5e-3,
    prompt_lr: float | None = None,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 30,
    early_stop_metric: str = "macro",
    save_best_path: str | None = None,
    epoch_callback=None,
):
    if args.method != "typepair":
        raise ValueError("train_typepair_prompt_probe expects args.method == 'typepair'")
    return _train_relation_prompt_probe(
        args=args,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        dropout=dropout,
        head_lr=head_lr,
        prompt_lr=prompt_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        early_stop_metric=early_stop_metric,
        save_best_path=save_best_path,
        epoch_callback=epoch_callback,
    )


def train_peprompt_probe(
    args,
    batch_size: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    head_lr: float = 5e-3,
    prompt_lr: float | None = None,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 30,
    early_stop_metric: str = "macro",
    save_best_path: str | None = None,
    epoch_callback=None,
):
    if args.method != "peprompt":
        raise ValueError("train_peprompt_probe expects args.method == 'peprompt'")
    return _train_relation_prompt_probe(
        args=args,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        dropout=dropout,
        head_lr=head_lr,
        prompt_lr=prompt_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        early_stop_metric=early_stop_metric,
        save_best_path=save_best_path,
        epoch_callback=epoch_callback,
    )


def _evaluate_hgmp_prompt_probe(
    graph_list,
    hgnn,
    PG,
    head,
    targetnode: str,
    dataset: str,
    classification_type: str,
    batch_size: int,
    device: torch.device,
    num_classes: int,
):
    loader = dgl.dataloading.GraphDataLoader(
        graph_list,
        batch_size=batch_size,
        shuffle=False,
    )

    logits_all = []
    labels_all = []

    PG.eval()
    head.eval()
    with torch.no_grad():
        for batch in loader:
            batched_graph, batched_label = _unpack_batch(batch, classification_type)
            batched_graph = batched_graph.to(device)
            batched_label = _prepare_labels_for_task(
                batched_label,
                device,
                dataset,
                classification_type,
            )

            prompted_graph = PG(batched_graph)
            graph_emb = forward_graph_batch(hgnn, prompted_graph, targetnode)
            logits = head(graph_emb)

            logits_all.append(logits)
            labels_all.append(batched_label)

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return _legacy_f1_micro_macro(logits_all, labels_all, num_classes)


def _run_hgmp_prompt_epoch(
    train_loader,
    hgnn,
    PG,
    head,
    optimizer,
    targetnode: str,
    dataset: str,
    classification_type: str,
    device: torch.device,
    lossfn=None,
):
    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:
        batched_graph, batched_label = _unpack_batch(batch, classification_type)
        batched_graph = batched_graph.to(device)
        batched_label = _prepare_labels_for_task(
            batched_label,
            device,
            dataset,
            classification_type,
        )

        prompted_graph = PG(batched_graph)
        graph_emb = forward_graph_batch(hgnn, prompted_graph, targetnode)
        logits = head(graph_emb)
        if lossfn is None:
            loss = _legacy_task_loss(logits, batched_label, dataset, classification_type)
        else:
            loss = lossfn(logits, batched_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_n = batched_label.size(0)
        total_loss += loss.item() * batch_n
        total_graphs += batch_n

    return total_loss / max(total_graphs, 1)


def _evaluate_hgmp_prompt_probe_loss(
    graph_list,
    hgnn,
    PG,
    head,
    targetnode: str,
    dataset: str,
    classification_type: str,
    batch_size: int,
    device: torch.device,
    lossfn,
):
    loader = dgl.dataloading.GraphDataLoader(
        graph_list,
        batch_size=batch_size,
        shuffle=False,
    )

    total_loss = 0.0
    total_graphs = 0

    hgnn.eval()
    PG.eval()
    head.eval()
    with torch.no_grad():
        for batch in loader:
            batched_graph, batched_label = _unpack_batch(batch, classification_type)
            batched_graph = batched_graph.to(device)
            batched_label = _prepare_labels_for_task(
                batched_label,
                device,
                dataset,
                classification_type,
            )

            prompted_graph = PG(batched_graph)
            graph_emb = forward_graph_batch(hgnn, prompted_graph, targetnode)
            logits = head(graph_emb)
            loss = lossfn(logits, batched_label)

            batch_n = batched_label.size(0)
            total_loss += loss.item() * batch_n
            total_graphs += batch_n

    return total_loss / max(total_graphs, 1)


def train_hgmp_heteroprompt_probe(
    args,
    batch_size: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 30,
    early_stop_metric: str = "macro",
    save_best_path: str | None = None,
    epoch_callback=None,
):
    assert early_stop_metric in {"micro", "macro"}

    seed_everything(args.seed)

    train_list, valid_list, test_list, targetnode = _load_legacy_fewshot_splits(args)
    sample_graph = _first_graph_from_sample(train_list[0], args.classification_type)
    ntypes = sample_graph.ntypes
    token_dims = [sample_graph.ndata["x"][nt].shape[1] for nt in ntypes]

    hgnn = build_frozen_legacy_hgnn(args, args.ckpt)

    PG = HeteroPrompt(
        token_dims=token_dims,
        ntypes=ntypes,
    ).to(args.device)

    recipe = getattr(args, "hgmp_prompt_recipe", "legacy")
    early_stop_mode = getattr(args, "hgmp_prompt_early_stop_mode", "auto")
    if early_stop_mode == "auto":
        early_stop_mode = "legacy_loss" if recipe == "legacy" else "metric"
    if early_stop_mode not in {"metric", "legacy_loss"}:
        raise ValueError(f"Unsupported hgmp_prompt_early_stop_mode: {early_stop_mode}")
    eval_mode = getattr(args, "hgmp_prompt_eval_mode", "full")
    if eval_mode not in {"full", "early_stop_only"}:
        raise ValueError(f"Unsupported hgmp_prompt_eval_mode: {eval_mode}")
    if eval_mode == "early_stop_only" and early_stop_mode != "legacy_loss":
        raise ValueError("hgmp_prompt_eval_mode=early_stop_only requires hgmp_prompt_early_stop_mode=legacy_loss.")

    legacy_defaults = _hgmp_prompt_legacy_defaults(args.dataset) if recipe == "legacy" else {}
    resolved_batch_size = int(
        getattr(args, "hgmp_prompt_batch_size", None)
        or batch_size
    )
    resolved_epochs = int(
        getattr(args, "hgmp_prompt_epochs", None)
        or epochs
    )
    resolved_patience = int(
        getattr(args, "hgmp_prompt_patience", None)
        or legacy_defaults.get("patience")
        or patience
    )
    resolved_weight_decay = float(
        getattr(args, "hgmp_prompt_weight_decay", None)
        if getattr(args, "hgmp_prompt_weight_decay", None) is not None
        else legacy_defaults.get("weight_decay", weight_decay)
    )

    if recipe == "legacy":
        resolved_prompt_lr = float(
            getattr(args, "hgmp_prompt_prompt_lr", None)
            if getattr(args, "hgmp_prompt_prompt_lr", None) is not None
            else legacy_defaults.get("prompt_lr", getattr(args, "prompt_lr", None) or lr)
        )
        resolved_head_lr = float(
            getattr(args, "hgmp_prompt_head_lr", None)
            if getattr(args, "hgmp_prompt_head_lr", None) is not None
            else legacy_defaults.get("head_lr", lr)
        )
        head = nn.Linear(args.hidden_dim, args.num_class).to(args.device)
        prompt_opt = torch.optim.Adam(
            PG.parameters(),
            lr=resolved_prompt_lr,
            weight_decay=resolved_weight_decay,
        )
        head_opt = torch.optim.Adam(
            head.parameters(),
            lr=resolved_head_lr,
            weight_decay=resolved_weight_decay,
        )
        if _is_legacy_multilabel_task(args.dataset, args.classification_type):
            train_lossfn = nn.BCEWithLogitsLoss()
            valid_lossfn = nn.BCEWithLogitsLoss()
        else:
            train_lossfn = nn.CrossEntropyLoss(reduction="mean")
            valid_lossfn = nn.CrossEntropyLoss(reduction="mean")
    else:
        head = MLPHead(
            in_dim=args.hidden_dim,
            hidden=hidden_dim,
            num_classes=args.num_class,
            dropout=dropout,
            use_ln=True,
        ).to(args.device)
        prompt_opt = torch.optim.AdamW(PG.parameters(), lr=lr, weight_decay=resolved_weight_decay)
        head_opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=resolved_weight_decay)
        train_lossfn = None
        valid_lossfn = None

    train_loader = dgl.dataloading.GraphDataLoader(
        train_list,
        batch_size=resolved_batch_size,
        shuffle=True,
    )

    best_val_micro = 0.0
    best_val_macro = 0.0
    test_at_best_micro = 0.0
    test_at_best_macro = 0.0
    best_epoch = -1
    bad_epochs = 0
    best_val_loss = float("inf")
    best_pg_state = None
    best_head_state = None

    for epoch in range(1, resolved_epochs + 1):
        _set_requires_grad(PG, False)
        _set_requires_grad(head, True)
        PG.eval()
        head.train()

        head_loss = _run_hgmp_prompt_epoch(
            train_loader=train_loader,
            hgnn=hgnn,
            PG=PG,
            head=head,
            optimizer=head_opt,
            targetnode=targetnode,
            dataset=args.dataset,
            classification_type=args.classification_type,
            device=args.device,
            lossfn=train_lossfn,
        )

        _set_requires_grad(PG, True)
        _set_requires_grad(head, False)
        PG.train()
        head.eval()

        prompt_loss = _run_hgmp_prompt_epoch(
            train_loader=train_loader,
            hgnn=hgnn,
            PG=PG,
            head=head,
            optimizer=prompt_opt,
            targetnode=targetnode,
            dataset=args.dataset,
            classification_type=args.classification_type,
            device=args.device,
            lossfn=train_lossfn,
        )

        if eval_mode == "full" and epoch_callback is not None:
            train_micro, train_macro = _evaluate_hgmp_prompt_probe(
                graph_list=train_list,
                hgnn=hgnn,
                PG=PG,
                head=head,
                targetnode=targetnode,
                dataset=args.dataset,
                classification_type=args.classification_type,
                batch_size=resolved_batch_size,
                device=args.device,
                num_classes=args.num_class,
            )
        else:
            train_micro, train_macro = None, None

        if eval_mode == "full":
            val_micro, val_macro = _evaluate_hgmp_prompt_probe(
                graph_list=valid_list,
                hgnn=hgnn,
                PG=PG,
                head=head,
                targetnode=targetnode,
                dataset=args.dataset,
                classification_type=args.classification_type,
                batch_size=resolved_batch_size,
                device=args.device,
                num_classes=args.num_class,
            )
            test_micro, test_macro = _evaluate_hgmp_prompt_probe(
                graph_list=test_list,
                hgnn=hgnn,
                PG=PG,
                head=head,
                targetnode=targetnode,
                dataset=args.dataset,
                classification_type=args.classification_type,
                batch_size=resolved_batch_size,
                device=args.device,
                num_classes=args.num_class,
            )
        else:
            val_micro = val_macro = None
            test_micro = test_macro = None

        val_loss = None
        if early_stop_mode == "legacy_loss":
            val_loss = _evaluate_hgmp_prompt_probe_loss(
                graph_list=valid_list,
                hgnn=hgnn,
                PG=PG,
                head=head,
                targetnode=targetnode,
                dataset=args.dataset,
                classification_type=args.classification_type,
                batch_size=resolved_batch_size,
                device=args.device,
                lossfn=valid_lossfn,
            )
            improved = val_loss <= best_val_loss
        else:
            monitor = val_macro if early_stop_metric == "macro" else val_micro
            best_monitor = best_val_macro if early_stop_metric == "macro" else best_val_micro
            improved = monitor > best_monitor

        if improved:
            improved = True
            if eval_mode == "full":
                best_val_micro = val_micro
                best_val_macro = val_macro
                test_at_best_micro = test_micro
                test_at_best_macro = test_macro
            best_epoch = epoch
            bad_epochs = 0
            if val_loss is not None:
                best_val_loss = float(val_loss)
            best_pg_state = _clone_state_dict(PG)
            best_head_state = _clone_state_dict(head)

            if save_best_path is not None:
                payload = {
                    "pg_state": PG.state_dict(),
                    "head_state": head.state_dict(),
                    "best_val_micro": best_val_micro,
                    "best_val_macro": best_val_macro,
                    "test_at_best_micro": test_at_best_micro,
                    "test_at_best_macro": test_at_best_macro,
                    "best_epoch": best_epoch,
                    "early_stop_metric": early_stop_metric,
                    "hgmp_prompt_recipe": recipe,
                    "hgmp_prompt_early_stop_mode": early_stop_mode,
                    "hgmp_prompt_eval_mode": eval_mode,
                }
                if val_loss is not None:
                    payload["best_val_loss"] = float(val_loss)
                torch.save(payload, save_best_path)
        else:
            improved = False
            bad_epochs += 1

        if epoch_callback is not None:
            payload = {
                "epoch": epoch,
                "head_loss": float(head_loss),
                "prompt_loss": float(prompt_loss),
                "train_loss": float((head_loss + prompt_loss) / 2.0),
                "best_val_micro": float(best_val_micro),
                "best_val_macro": float(best_val_macro),
                "test_at_best_micro": float(test_at_best_micro),
                "test_at_best_macro": float(test_at_best_macro),
                "best_epoch": int(best_epoch),
                "bad_epochs": int(bad_epochs),
                "is_best": bool(improved),
                "early_stop": bool(bad_epochs >= resolved_patience),
                "hgmp_prompt_recipe": recipe,
                "hgmp_prompt_early_stop_mode": early_stop_mode,
                "hgmp_prompt_eval_mode": eval_mode,
            }
            if eval_mode == "full":
                payload.update(
                    {
                        "train_micro": float(train_micro),
                        "train_macro": float(train_macro),
                        "val_micro": float(val_micro),
                        "val_macro": float(val_macro),
                        "test_micro": float(test_micro),
                        "test_macro": float(test_macro),
                    }
                )
            if val_loss is not None:
                payload["val_loss"] = float(val_loss)
                payload["best_val_loss"] = float(best_val_loss)
                payload["monitor"] = float(-val_loss)
            else:
                monitor = val_macro if early_stop_metric == "macro" else val_micro
                payload["monitor"] = float(monitor)
            epoch_callback(payload)

        if epoch == 1 or epoch % 10 == 0:
            if eval_mode == "early_stop_only":
                print(
                    f"Epoch {epoch:03d} | "
                    f"head_loss={head_loss:.4f} | prompt_loss={prompt_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | best_val_loss={best_val_loss:.4f} | "
                    f"bad_epochs={bad_epochs}"
                )
            elif val_loss is not None:
                print(
                    f"Epoch {epoch:03d} | "
                    f"head_loss={head_loss:.4f} | prompt_loss={prompt_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f} | "
                    f"test_f1(micro/macro)={test_micro:.4f}/{test_macro:.4f}"
                )
            else:
                monitor = val_macro if early_stop_metric == "macro" else val_micro
                print(
                    f"Epoch {epoch:03d} | "
                    f"head_loss={head_loss:.4f} | prompt_loss={prompt_loss:.4f} | "
                    f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f} | "
                    f"test_f1(micro/macro)={test_micro:.4f}/{test_macro:.4f} | "
                    f"monitor({early_stop_metric})={monitor:.4f}"
                )

        if bad_epochs >= resolved_patience:
            if eval_mode == "early_stop_only":
                print(
                    f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                    f"best_val_loss={best_val_loss:.4f} | final F1 will be evaluated once on the best checkpoint"
                )
            elif val_loss is not None:
                print(
                    f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                    f"best_val_loss={best_val_loss:.4f} | "
                    f"best_val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
                    f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f}"
                )
            else:
                print(
                    f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                    f"best_val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
                    f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f} | "
                    f"monitor={early_stop_metric}"
                )
            break

    if best_pg_state is not None:
        PG.load_state_dict(best_pg_state)
    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    if eval_mode == "early_stop_only":
        best_val_micro, best_val_macro = _evaluate_hgmp_prompt_probe(
            graph_list=valid_list,
            hgnn=hgnn,
            PG=PG,
            head=head,
            targetnode=targetnode,
            dataset=args.dataset,
            classification_type=args.classification_type,
            batch_size=resolved_batch_size,
            device=args.device,
            num_classes=args.num_class,
        )
        test_at_best_micro, test_at_best_macro = _evaluate_hgmp_prompt_probe(
            graph_list=test_list,
            hgnn=hgnn,
            PG=PG,
            head=head,
            targetnode=targetnode,
            dataset=args.dataset,
            classification_type=args.classification_type,
            batch_size=resolved_batch_size,
            device=args.device,
            num_classes=args.num_class,
        )
        print(
            f"Final best-state F1 | best_epoch={best_epoch} | "
            f"val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
            f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f}"
        )
        if save_best_path is not None and best_pg_state is not None and best_head_state is not None:
            torch.save(
                {
                    "pg_state": PG.state_dict(),
                    "head_state": head.state_dict(),
                    "best_val_micro": best_val_micro,
                    "best_val_macro": best_val_macro,
                    "test_at_best_micro": test_at_best_micro,
                    "test_at_best_macro": test_at_best_macro,
                    "best_epoch": best_epoch,
                    "early_stop_metric": early_stop_metric,
                    "hgmp_prompt_recipe": recipe,
                    "hgmp_prompt_early_stop_mode": early_stop_mode,
                    "hgmp_prompt_eval_mode": eval_mode,
                    "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                },
                save_best_path,
            )

    return {
        "best_val_micro": best_val_micro,
        "best_val_macro": best_val_macro,
        "test_at_best_micro": test_at_best_micro,
        "test_at_best_macro": test_at_best_macro,
        "best_epoch": best_epoch,
        "early_stop_metric": early_stop_metric,
        "hgmp_prompt_recipe": recipe,
        "hgmp_prompt_early_stop_mode": early_stop_mode,
        "hgmp_prompt_eval_mode": eval_mode,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
    }


def train_mlp_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    in_dim: int,
    num_classes: int,
    device: torch.device,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 30,
    early_stop_metric: str = "macro",
    save_best_path: str | None = None,
    epoch_callback=None,
    dataset: str | None = None,
    classification_type: str = "NIG",
):
    assert early_stop_metric in {"micro", "macro"}

    head = MLPHead(
        in_dim=in_dim,
        hidden=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        use_ln=True,
    ).to(device)

    x_train = x_train.to(device)
    y_train = _prepare_labels_for_task(y_train, device, dataset or "", classification_type)
    x_val = x_val.to(device)
    y_val = _prepare_labels_for_task(y_val, device, dataset or "", classification_type)
    x_test = x_test.to(device)
    y_test = _prepare_labels_for_task(y_test, device, dataset or "", classification_type)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_micro = 0.0
    best_val_macro = 0.0
    test_at_best_micro = 0.0
    test_at_best_macro = 0.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        head.train()
        logits = head(x_train)
        loss = _legacy_task_loss(logits, y_train, dataset or "", classification_type)

        opt.zero_grad()
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            train_logits = head(x_train)
            val_logits = head(x_val)
            test_logits = head(x_test)

            train_micro, train_macro = _legacy_f1_micro_macro(train_logits, y_train, num_classes)
            val_micro, val_macro = _legacy_f1_micro_macro(val_logits, y_val, num_classes)
            test_micro, test_macro = _legacy_f1_micro_macro(test_logits, y_test, num_classes)

        monitor = val_macro if early_stop_metric == "macro" else val_micro
        best_monitor = best_val_macro if early_stop_metric == "macro" else best_val_micro

        if monitor > best_monitor:
            improved = True
            best_val_micro = val_micro
            best_val_macro = val_macro
            test_at_best_micro = test_micro
            test_at_best_macro = test_macro
            best_epoch = epoch
            bad_epochs = 0

            if save_best_path is not None:
                torch.save(
                    {
                        "head_state": head.state_dict(),
                        "in_dim": in_dim,
                        "hidden_dim": hidden_dim,
                        "num_classes": num_classes,
                        "best_val_micro": best_val_micro,
                        "best_val_macro": best_val_macro,
                        "test_at_best_micro": test_at_best_micro,
                        "test_at_best_macro": test_at_best_macro,
                        "best_epoch": best_epoch,
                        "early_stop_metric": early_stop_metric,
                    },
                    save_best_path,
                )
        else:
            improved = False
            bad_epochs += 1

        if epoch_callback is not None:
            epoch_callback(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "train_micro": float(train_micro),
                    "train_macro": float(train_macro),
                    "val_micro": float(val_micro),
                    "val_macro": float(val_macro),
                    "test_micro": float(test_micro),
                    "test_macro": float(test_macro),
                    "monitor": float(monitor),
                    "best_val_micro": float(best_val_micro),
                    "best_val_macro": float(best_val_macro),
                    "test_at_best_micro": float(test_at_best_micro),
                    "test_at_best_macro": float(test_at_best_macro),
                    "best_epoch": int(best_epoch),
                    "bad_epochs": int(bad_epochs),
                    "is_best": bool(improved),
                    "early_stop": bool(bad_epochs >= patience),
                }
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f} | "
                f"test_f1(micro/macro)={test_micro:.4f}/{test_macro:.4f} | "
                f"monitor({early_stop_metric})={monitor:.4f}"
            )

        if bad_epochs >= patience:
            print(
                f"Early stop at epoch {epoch}, best_epoch={best_epoch} | "
                f"best_val_f1(micro/macro)={best_val_micro:.4f}/{best_val_macro:.4f} | "
                f"test@best_f1(micro/macro)={test_at_best_micro:.4f}/{test_at_best_macro:.4f} | "
                f"monitor={early_stop_metric}"
            )
            break

    return {
        "best_val_micro": best_val_micro,
        "best_val_macro": best_val_macro,
        "test_at_best_micro": test_at_best_micro,
        "test_at_best_macro": test_at_best_macro,
        "best_epoch": best_epoch,
        "early_stop_metric": early_stop_metric,
    }
