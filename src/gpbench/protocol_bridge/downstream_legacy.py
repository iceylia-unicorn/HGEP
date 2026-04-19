from __future__ import annotations

from dataclasses import dataclass

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

from protocols.hgmp.data_legacy import multi_class_NIG
from protocols.hgmp.prompt_legacy import GAT, GCL_GCN, HGNN, HeteroPrompt
from protocols.hgmp.utils_legacy import graph_pool, load_data4pretrain, seed_everything

from gpbench.downstream.fewshot import f1_micro_macro
from gpbench.downstream.model import MLPHead
from gpbench.protocol_bridge.hgmp_typepair import (
    HGMPTypePairHGNN,
    build_typepair_relation_cfg_from_args,
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


def _build_edge_index_dict(g):
    edge_index_dict = {}
    for etype in g.canonical_etypes:
        edge_index = g.edges(etype=etype)
        src = edge_index[0].unsqueeze(0)
        dst = edge_index[1].unsqueeze(0)
        edge_index_dict[etype] = torch.cat((src, dst), dim=0)
    return edge_index_dict


def _build_edge_feature_dict(g, feature_name: str = "typepair_edge_feat"):
    edge_feature_dict = {}
    for etype in g.canonical_etypes:
        if feature_name not in g.edges[etype].data:
            continue
        edge_feature_dict[etype] = g.edges[etype].data[feature_name].float()
    return edge_feature_dict or None


def _uses_edge_features(hgnn) -> bool:
    relation_prompt = getattr(hgnn, "relation_prompt", None)
    return bool(getattr(relation_prompt, "uses_edge_features", False))


def forward_graph_batch(hgnn, batched_graph, targetnode: str) -> torch.Tensor:
    x_dict = batched_graph.ndata["x"]
    edge_index_dict = _build_edge_index_dict(batched_graph)
    edge_feature_dict = None
    if _uses_edge_features(hgnn):
        edge_feature_name = getattr(hgnn.relation_prompt, "edge_feature_name", "typepair_edge_feat")
        edge_feature_dict = _build_edge_feature_dict(batched_graph, edge_feature_name)

    if hgnn.hgnn_type == "HGT":
        if edge_feature_dict is None:
            node_emb = hgnn(targetnode, x_dict, edge_index_dict)
        else:
            node_emb = hgnn(targetnode, x_dict, edge_index_dict, edge_feature_dict=edge_feature_dict)
    elif hgnn.hgnn_type == "SHGN":
        node_emb = hgnn(targetnode, batched_graph, x_dict)
    elif hgnn.hgnn_type == "GCN":
        if edge_feature_dict is None:
            node_emb = hgnn(batched_graph, x_dict)
        else:
            node_emb = hgnn(batched_graph, x_dict, edge_feature_dict=edge_feature_dict)
    elif hgnn.hgnn_type == "GAT":
        node_emb = hgnn(batched_graph, x_dict, False)
    else:
        raise ValueError(f"Unsupported hgnn_type: {hgnn.hgnn_type}")

    graph_emb = graph_pool("mean", node_emb, batched_graph)
    return graph_emb


@torch.no_grad()
def encode_graph_batch(hgnn, batched_graph, targetnode: str) -> torch.Tensor:
    return forward_graph_batch(hgnn, batched_graph, targetnode)


def _build_legacy_hgnn(args):
    graph_list, in_dims, _ = load_data4pretrain(
        args.feats_type,
        args.device,
        args.dataset,
        batch_size=64,
        num_sample=args.num_samples,
    )
    graph = graph_list[0]
    edge_types = graph.canonical_etypes
    ntypes = graph.ntypes
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

    raise ValueError(f"Unsupported method: {args.method}")


def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        p.requires_grad_(flag)


def _remap_plain_hgmp_state_for_typepair(args, state):
    if args.method != "typepair":
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
    if args.method == "typepair":
        expected_missing_prefixes = ("relation_prompt.", "GraphConv.relation_prompt.")
        expected_missing = [k for k in missing if k.startswith(expected_missing_prefixes)]
        other_missing = [k for k in missing if not k.startswith(expected_missing_prefixes)]

        if len(expected_missing) > 0:
            print(
                "[load_state_dict] typepair prompt keys were missing from ckpt "
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

    state, remap_count = _remap_plain_hgmp_state_for_typepair(args, state)
    if remap_count > 0:
        print(
            f"[load_state_dict] remapped {remap_count} plain HGMP keys to TypePair wrapper keys."
        )

    missing, unexpected = hgnn.load_state_dict(state, strict=False)
    _log_state_dict_load(args, missing, unexpected)

    hgnn = hgnn.to(args.device)

    if freeze:
        hgnn.eval()
        _set_requires_grad(hgnn, False)

    if train_relation_prompt_only:
        if args.method != "typepair":
            raise ValueError("train_relation_prompt_only only supports method='typepair'")
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
):
    loader = dgl.dataloading.GraphDataLoader(graph_list, batch_size=batch_size, shuffle=False)

    xs = []
    ys = []
    for batch in loader:
        batched_graph, batched_label = _unpack_batch(batch, classification_type)
        batched_graph = batched_graph.to(device)
        batched_label = batched_label.to(device).long()

        x = encode_graph_batch(hgnn, batched_graph, targetnode)
        xs.append(x)
        ys.append(batched_label)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _load_legacy_fewshot_splits(args):
    if args.classification_type != "NIG":
        raise NotImplementedError("v1 only supports classification_type='NIG'")

    if args.dataset == "IMDB":
        raise NotImplementedError(
            "v1 intentionally skips IMDB because its legacy batch/label format "
            "differs in this branch. Start with ACM/DBLP first."
        )

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
        train_list, hgnn, targetnode, args.classification_type, batch_size, args.device
    )
    x_val, y_val = extract_split_embeddings(
        valid_list, hgnn, targetnode, args.classification_type, batch_size, args.device
    )
    x_test, y_test = extract_split_embeddings(
        test_list, hgnn, targetnode, args.classification_type, batch_size, args.device
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
            batched_graph = batched_graph.to(device)
            batched_label = batched_label.to(device).long()

            graph_emb = encode_graph_batch(hgnn, batched_graph, targetnode)
            logits = head(graph_emb)

            logits_list.append(logits)
            labels_list.append(batched_label)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return f1_micro_macro(all_logits, all_labels, num_classes)


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
    assert early_stop_metric in {"micro", "macro"}

    train_list, valid_list, test_list, targetnode = _load_legacy_fewshot_splits(args)

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

    for epoch in range(1, epochs + 1):
        hgnn.eval()
        hgnn.relation_prompt.train()
        head.train()

        epoch_loss = 0.0
        epoch_graphs = 0

        for batch in train_loader:
            batched_graph, batched_label = _unpack_batch(batch, args.classification_type)
            batched_graph = batched_graph.to(args.device)
            batched_label = batched_label.to(args.device).long()

            graph_emb = forward_graph_batch(hgnn, batched_graph, targetnode)
            logits = head(graph_emb)
            loss = nn.functional.cross_entropy(logits, batched_label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_n = batched_label.size(0)
            epoch_loss += loss.item() * batch_n
            epoch_graphs += batch_n

        train_loss = epoch_loss / max(epoch_graphs, 1)

        if epoch_callback is not None:
            train_micro, train_macro = _evaluate_graph_probe(
                train_list,
                hgnn,
                head,
                targetnode,
                args.classification_type,
                batch_size,
                args.device,
                args.num_class,
            )
        else:
            train_micro, train_macro = None, None

        val_micro, val_macro = _evaluate_graph_probe(
            valid_list,
            hgnn,
            head,
            targetnode,
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
            args.classification_type,
            batch_size,
            args.device,
            args.num_class,
        )

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
                    "train_loss": float(train_loss),
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
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
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


def _evaluate_hgmp_prompt_probe(
    graph_list,
    hgnn,
    PG,
    head,
    targetnode: str,
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
            batched_label = batched_label.to(device).long()

            prompted_graph = PG(batched_graph)
            graph_emb = forward_graph_batch(hgnn, prompted_graph, targetnode)
            logits = head(graph_emb)

            logits_all.append(logits)
            labels_all.append(batched_label)

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return f1_micro_macro(logits_all, labels_all, num_classes)


def _run_hgmp_prompt_epoch(
    train_loader,
    hgnn,
    PG,
    head,
    optimizer,
    targetnode: str,
    classification_type: str,
    device: torch.device,
):
    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:
        batched_graph, batched_label = _unpack_batch(batch, classification_type)
        batched_graph = batched_graph.to(device)
        batched_label = batched_label.to(device).long()

        prompted_graph = PG(batched_graph)
        graph_emb = forward_graph_batch(hgnn, prompted_graph, targetnode)
        logits = head(graph_emb)
        loss = nn.functional.cross_entropy(logits, batched_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    if args.classification_type != "NIG":
        raise NotImplementedError("v1 only supports classification_type='NIG'")

    if args.dataset == "IMDB":
        raise NotImplementedError(
            "v1 intentionally skips IMDB because its legacy batch/label format "
            "differs in this branch. Start with ACM/DBLP first."
        )

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
    ntypes = sample_graph.ntypes
    token_dims = [sample_graph.ndata["x"][nt].shape[1] for nt in ntypes]

    hgnn = build_frozen_legacy_hgnn(args, args.ckpt)

    PG = HeteroPrompt(
        token_dims=token_dims,
        ntypes=ntypes,
    ).to(args.device)

    head = MLPHead(
        in_dim=args.hidden_dim,
        hidden=hidden_dim,
        num_classes=args.num_class,
        dropout=dropout,
        use_ln=True,
    ).to(args.device)

    train_loader = dgl.dataloading.GraphDataLoader(
        train_list,
        batch_size=batch_size,
        shuffle=True,
    )

    prompt_opt = torch.optim.AdamW(PG.parameters(), lr=lr, weight_decay=weight_decay)
    head_opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_micro = 0.0
    best_val_macro = 0.0
    test_at_best_micro = 0.0
    test_at_best_macro = 0.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
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
            classification_type=args.classification_type,
            device=args.device,
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
            classification_type=args.classification_type,
            device=args.device,
        )

        if epoch_callback is not None:
            train_micro, train_macro = _evaluate_hgmp_prompt_probe(
                graph_list=train_list,
                hgnn=hgnn,
                PG=PG,
                head=head,
                targetnode=targetnode,
                classification_type=args.classification_type,
                batch_size=batch_size,
                device=args.device,
                num_classes=args.num_class,
            )
        else:
            train_micro, train_macro = None, None

        val_micro, val_macro = _evaluate_hgmp_prompt_probe(
            graph_list=valid_list,
            hgnn=hgnn,
            PG=PG,
            head=head,
            targetnode=targetnode,
            classification_type=args.classification_type,
            batch_size=batch_size,
            device=args.device,
            num_classes=args.num_class,
        )
        test_micro, test_macro = _evaluate_hgmp_prompt_probe(
            graph_list=test_list,
            hgnn=hgnn,
            PG=PG,
            head=head,
            targetnode=targetnode,
            classification_type=args.classification_type,
            batch_size=batch_size,
            device=args.device,
            num_classes=args.num_class,
        )

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
                        "pg_state": PG.state_dict(),
                        "head_state": head.state_dict(),
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
                    "head_loss": float(head_loss),
                    "prompt_loss": float(prompt_loss),
                    "train_loss": float((head_loss + prompt_loss) / 2.0),
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
                f"Epoch {epoch:03d} | "
                f"head_loss={head_loss:.4f} | prompt_loss={prompt_loss:.4f} | "
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
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

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
        loss = nn.functional.cross_entropy(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            train_logits = head(x_train)
            val_logits = head(x_val)
            test_logits = head(x_test)

            train_micro, train_macro = f1_micro_macro(train_logits, y_train, num_classes)
            val_micro, val_macro = f1_micro_macro(val_logits, y_val, num_classes)
            test_micro, test_macro = f1_micro_macro(test_logits, y_test, num_classes)

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
