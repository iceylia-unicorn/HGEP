from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import dgl
import torch
import torch.nn as nn

from protocols.hgmp.data_legacy import multi_class_NIG
from protocols.hgmp.prompt_legacy import HGNN, GCL_GCN, GAT
from protocols.hgmp.utils_legacy import graph_pool, load_data4pretrain, seed_everything

from gpbench.downstream.model import MLPHead
from gpbench.downstream.fewshot import f1_micro_macro
from gpbench.protocol_bridge.hgmp_typepair import (
    HGMPTypePairHGNN,
    build_typepair_relation_cfg_from_args,
)
import torch.nn.functional as F
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
        # non-IMDB NIG usually stores (graph, extra, label)
        return sample[0]
    # EIG / GIG fallback
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


@torch.no_grad()
def encode_graph_batch(hgnn, batched_graph, targetnode: str) -> torch.Tensor:
    x_dict = batched_graph.ndata["x"]
    edge_index_dict = _build_edge_index_dict(batched_graph)

    if hgnn.hgnn_type == "HGT":
        node_emb = hgnn(targetnode, x_dict, edge_index_dict)
    elif hgnn.hgnn_type == "SHGN":
        node_emb = hgnn(targetnode, batched_graph, x_dict)
    elif hgnn.hgnn_type == "GCN":
        node_emb = hgnn(batched_graph, x_dict)
    elif hgnn.hgnn_type == "GAT":
        node_emb = hgnn(batched_graph, x_dict, False)
    else:
        raise ValueError(f"Unsupported hgnn_type: {hgnn.hgnn_type}")

    graph_emb = graph_pool("mean", node_emb, batched_graph)
    return graph_emb


def build_frozen_legacy_hgnn(args, ckpt_path: str):
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

    if args.method == "hgmp":
        if args.hgnn_type == "GCN":
            hgnn = GCL_GCN(
                None,
                in_dims,
                args.hidden_dim,
                args.hidden_dim,
                args.num_layers,
                F.elu,
                args.dropout,
                args.hgnn_type,
            )
        elif args.hgnn_type == "GAT":
            heads = [args.num_heads] * args.num_layers + [1]
            hgnn = GAT(
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
        else:
            hgnn = HGNN(
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
    elif args.method == "typepair":
        relation_cfg = build_typepair_relation_cfg_from_args(args)
        hgnn = HGMPTypePairHGNN(
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
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    missing, unexpected = hgnn.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print("[load_state_dict] missing keys:", missing[:10], "...")
    if len(unexpected) > 0:
        print("[load_state_dict] unexpected keys:", unexpected[:10], "...")

    hgnn = hgnn.to(args.device)
    hgnn.eval()
    for p in hgnn.parameters():
        p.requires_grad_(False)
    return hgnn


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


def build_legacy_fewshot_embeddings(args, batch_size: int = 32) -> LegacyFewShotEmbeddings:
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
            val_logits = head(x_val)
            test_logits = head(x_test)

            val_micro, val_macro = f1_micro_macro(val_logits, y_val, num_classes)
            test_micro, test_macro = f1_micro_macro(test_logits, y_test, num_classes)

        monitor = val_macro if early_stop_metric == "macro" else val_micro
        best_monitor = best_val_macro if early_stop_metric == "macro" else best_val_micro

        if monitor > best_monitor:
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
            bad_epochs += 1

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