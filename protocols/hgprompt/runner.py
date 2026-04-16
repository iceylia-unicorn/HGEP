from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from gpbench.utils.early_stop import EarlyStopping, EarlyStopConfig
from sklearn.metrics import f1_score

from protocols.hgprompt.adapter import HPromptDownstreamBundle, load_hgprompt_downstream_bundle
from protocols.hgprompt.source.GNN import GAT, GCN, GIN, semantic_GCN, myGAT
from protocols.hgprompt.source.hgprompt import (
    acm_eachloss_hnode_prompt_layer_feature_weighted_sum,
    acm_hnode_prompt_layer_feature_weighted_sum,
    acm_hnode_semantic_prompt_layer_feature_weighted_sum,
    center_embedding,
    dblp_hnode_prompt_layer_feature_weighted_sum,
    dblp_hnode_semantic_prompt_layer_feature_weighted_sum,
    distance2center,
    freebase_bidirection_hnode_prompt_layer_feature_weighted_sum,
    freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum,
    freebase_des_hnode_prompt_layer_feature_weighted_sum,
    freebase_source_hnode_prompt_layer_feature_weighted_sum,
    hnode_prompt_layer_feature_cat_edge,
    hnode_prompt_layer_feature_sum,
    hnode_prompt_layer_feature_weighted_sum,
    hprompt_gcn,
    node_bottle_net,
    node_prompt_layer_feature_cat,
    node_prompt_layer_feature_cat_edge,
    node_prompt_layer_feature_sum,
    node_prompt_layer_feature_weighted_sum,
    prompt_gcn,
)


@dataclass
class RepeatResult:
    test_micro: float
    test_macro: float
    best_epoch: int


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if isinstance(mat, np.ndarray):
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def _prepare_feature_inputs(features_list, feats_type: int, device: torch.device):
    feats = [mat2tensor(x).to(device) for x in features_list]

    if feats_type == 0:
        in_dims = [f.shape[1] for f in feats]

    elif feats_type in (1, 5):
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(len(feats)):
            if i == save:
                in_dims.append(feats[i].shape[1])
            else:
                in_dims.append(10)
                feats[i] = torch.zeros((feats[i].shape[0], 10), device=device)

    elif feats_type in (2, 4):
        save = feats_type - 2
        in_dims = [f.shape[0] for f in feats]
        for i in range(len(feats)):
            if i == save:
                in_dims[i] = feats[i].shape[1]
                continue
            dim = feats[i].shape[0]
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            idx = torch.LongTensor(idx)
            val = torch.FloatTensor(np.ones(dim))
            feats[i] = torch.sparse.FloatTensor(idx, val, torch.Size([dim, dim])).to(device)

    elif feats_type == 3:
        in_dims = [f.shape[0] for f in feats]
        for i in range(len(feats)):
            dim = feats[i].shape[0]
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            idx = torch.LongTensor(idx)
            val = torch.FloatTensor(np.ones(dim))
            feats[i] = torch.sparse.FloatTensor(idx, val, torch.Size([dim, dim])).to(device)

    else:
        raise ValueError(f"Unsupported feats_type: {feats_type}")

    return feats, in_dims


def _build_edge2type(dl) -> Dict[tuple[int, int], int]:
    edge2type: Dict[tuple[int, int], int] = {}

    for k in dl.links["data"]:
        rows, cols = dl.links["data"][k].nonzero()
        for u, v in zip(rows, cols):
            edge2type[(int(u), int(v))] = int(k)

    for i in range(dl.nodes["total"]):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links["count"])

    for k in dl.links["data"]:
        rows, cols = dl.links["data"][k].nonzero()
        for u, v in zip(rows, cols):
            if (int(v), int(u)) not in edge2type:
                edge2type[(int(v), int(u))] = int(k) + 1 + len(dl.links["count"])

    return edge2type


def _build_graph_and_edge_feat(bundle: HPromptDownstreamBundle, device: torch.device):
    g = dgl.DGLGraph(bundle.adjM + bundle.adjM.T)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    trans_g = dgl.reverse(g)

    edge2type = _build_edge2type(bundle.dl)
    e_feat = []
    src, dst = g.edges()
    for u, v in zip(src, dst):
        uu = int(u.cpu().item())
        vv = int(v.cpu().item())
        e_feat.append(edge2type[(uu, vv)])
    e_feat = torch.tensor(e_feat, dtype=torch.long, device=device)

    return g, trans_g, e_feat


def _build_backbone(args, g, dl, in_dims, num_classes):
    if args.model_type == "gat":
        heads = [args.num_heads] * args.num_layers + [1]
        return GAT(
            g, in_dims, args.hidden_dim, num_classes, args.num_layers,
            heads, F.elu, args.dropout, args.dropout, args.slope, False
        )

    if args.model_type == "gcn":
        if args.pretrain_semantic:
            return semantic_GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
        return GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)

    if args.model_type == "gin":
        return GIN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.relu, args.dropout)

    if args.model_type == "SHGN":
        heads = [args.num_heads] * args.num_layers + [1]
        return myGAT(
            g,
            args.edge_feats,
            len(dl.links["count"]) * 2 + 1,
            in_dims,
            args.hidden_dim,
            num_classes,
            args.num_layers,
            heads,
            F.elu,
            args.dropout,
            args.dropout,
            args.slope,
            True,
            0.05,
        )

    raise ValueError(f"Unsupported model_type: {args.model_type}")


def _extract_state_dict(payload: Any):
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "encoder_state_dict", "encoder_state"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _smart_load_pretrain(model: torch.nn.Module, ckpt_path: str, device: torch.device, strict: bool = False):
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(payload)

    candidates = [
        state_dict,
        _strip_prefix(state_dict, "module."),
        _strip_prefix(state_dict, "encoder."),
        _strip_prefix(state_dict, "backbone."),
        _strip_prefix(state_dict, "gnn."),
    ]

    last_err = None
    for cand in candidates:
        try:
            missing, unexpected = model.load_state_dict(cand, strict=strict)
            print(f"[load] ckpt={ckpt_path} | strict={strict} | missing={len(missing)} | unexpected={len(unexpected)}")
            if len(missing) > 0:
                print(f"[load] missing keys (first 10): {missing[:10]}")
            if len(unexpected) > 0:
                print(f"[load] unexpected keys (first 10): {unexpected[:10]}")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}\nLast error: {last_err}")


def _build_classifier(args, hidden_dim: int, num_classes: int, semantic_weight=None):
    if args.tuning == "linear":
        return torch.nn.Linear(hidden_dim, num_classes)

    if args.tuning == "gcn":
        return GraphConv(hidden_dim, num_classes)

    if args.tuning in ("weight-sum", "weight-sum-center-fixed", "bottle-net"):
        if args.add_edge_info2prompt:
            classify = hnode_prompt_layer_feature_weighted_sum(hidden_dim)

            if args.each_type_subgraph:
                if args.dataset == "ACM":
                    if args.pretrain_semantic:
                        classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim, semantic_weight)
                    elif args.pretrain_each_loss:
                        # 原始上游这条分支本身就没有完整打通，这里明确禁用
                        raise NotImplementedError("pretrain_each_loss downstream path is incomplete in upstream HGPrompt.")
                    elif args.semantic_prompt == 1:
                        classify = acm_hnode_semantic_prompt_layer_feature_weighted_sum(
                            hidden_dim,
                            semantic_prompt_weight=args.semantic_prompt_weight,
                        )
                    else:
                        classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim)

                elif args.dataset == "DBLP":
                    if args.semantic_prompt == 1:
                        classify = dblp_hnode_semantic_prompt_layer_feature_weighted_sum(
                            hidden_dim,
                            semantic_prompt_weight=args.semantic_prompt_weight,
                        )
                    else:
                        classify = dblp_hnode_prompt_layer_feature_weighted_sum(hidden_dim)

                elif args.dataset == "Freebase":
                    if args.semantic_prompt == 1:
                        classify = freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(
                            hidden_dim,
                            semantic_prompt_weight=args.semantic_prompt_weight,
                        )
                    else:
                        if args.freebase_type == 2:
                            classify = freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        elif args.freebase_type == 1:
                            classify = freebase_des_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        else:
                            classify = freebase_source_hnode_prompt_layer_feature_weighted_sum(hidden_dim)

            return classify

        if args.tuning == "bottle-net":
            return node_bottle_net(args.hidden_dim, args.bottle_net_hidden_dim, args.bottle_net_output_dim)

        return node_prompt_layer_feature_weighted_sum(hidden_dim)

    if args.tuning == "cat":
        return node_prompt_layer_feature_cat(args.cat_prompt_dim)

    if args.tuning == "sum":
        if args.add_edge_info2prompt:
            return hnode_prompt_layer_feature_sum()
        return node_prompt_layer_feature_sum()

    if args.tuning == "cat_edge":
        if args.add_edge_info2prompt:
            return hnode_prompt_layer_feature_cat_edge(args.cat_prompt_dim, args.cat_hprompt_dim)
        return node_prompt_layer_feature_cat_edge(args.cat_prompt_dim)

    if args.tuning == "prompt_gcn":
        if args.add_edge_info2prompt:
            return hprompt_gcn(hidden_dim)
        return prompt_gcn(hidden_dim)

    raise ValueError(f"Unsupported tuning: {args.tuning}")


def _forward_classifier(args, classify, g, trans_g, prelogits, e_feat):
    if args.add_edge_info2prompt:
        if args.tuning == "gcn":
            return classify(g, prelogits)
        if args.tuning in ("weight-sum", "weight-sum-center-fixed", "bottle-net", "sum", "cat", "cat_edge", "prompt_gcn"):
            if args.dataset == "Freebase" and args.freebase_type == 2:
                return classify(g, trans_g, prelogits, e_feat)
            return classify(g, prelogits, e_feat)
        if args.tuning == "linear":
            return classify(prelogits)
        raise ValueError(f"Unsupported tuning with edge prompt: {args.tuning}")

    if args.tuning in ("gcn", "weight-sum", "weight-sum-center-fixed", "bottle-net", "cat", "sum", "cat_edge", "prompt_gcn"):
        return classify(g, prelogits)

    return classify(prelogits)


def _f1_from_logp(logp: torch.Tensor, labels: torch.Tensor):
    pred = torch.argmax(logp, dim=1, keepdim=False)
    truth = labels.detach().cpu().numpy()
    output = pred.detach().cpu().numpy()
    micro = f1_score(truth, output, average="micro")
    macro = f1_score(truth, output, average="macro")
    return micro, macro


def _compute_train_or_val(
    args,
    logits: torch.Tensor,
    idx: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    center: torch.Tensor | None = None,
    recompute_center: bool = False,
):
    if args.tuning in ("gcn", "linear"):
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logp[idx], labels)
        micro, macro = _f1_from_logp(logp[idx], labels)
        return loss, micro, macro, None

    emb = logits[idx]

    if center is None or recompute_center:
        center = center_embedding(emb, labels, num_classes)

    distance = distance2center(emb, center)
    logp = F.log_softmax(distance, dim=1)
    loss = F.nll_loss(logp, labels)
    micro, macro = _f1_from_logp(logp, labels)
    return loss, micro, macro, center


@torch.no_grad()
def _evaluate_test(args, logits, test_idx, test_labels, num_classes, center=None):
    if args.tuning in ("gcn", "linear"):
        logp = F.log_softmax(logits, dim=1)
        return _f1_from_logp(logp[test_idx], test_labels)

    emb = logits[test_idx]
    if args.tuning == "weight-sum" or center is None:
        center = center_embedding(emb, test_labels, num_classes)

    distance = distance2center(emb, center)
    logp = F.log_softmax(distance, dim=1)
    return _f1_from_logp(logp, test_labels)


def _encode_once(args, net, features_list, e_feat):
    if args.pretrain_semantic:
        prelogits, semantic_weight = net(features_list)
        return prelogits, semantic_weight

    if args.model_type == "SHGN":
        return net(features_list, e_feat), None

    return net(features_list), None


def _run_once(args, bundle: HPromptDownstreamBundle, repeat_id: int, epoch_callback=None) -> RepeatResult:
    device = torch.device(args.device)

    np.random.seed(args.seed + repeat_id)
    torch.manual_seed(args.seed + repeat_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + repeat_id)

    features_list, in_dims = _prepare_feature_inputs(bundle.features_list, args.feats_type, device)
    g, trans_g, e_feat = _build_graph_and_edge_feat(bundle, device)

    num_classes = int(bundle.dl.labels_train["num_classes"])

    train_idx = torch.tensor(bundle.train_val_test_idx["train_idx"][0], dtype=torch.long, device=device)
    val_idx = torch.tensor(bundle.train_val_test_idx["val_idx"][0], dtype=torch.long, device=device)
    test_idx = torch.tensor(bundle.train_val_test_idx["test_idx"], dtype=torch.long, device=device)

    train_labels = torch.tensor(bundle.labels["train"][0], dtype=torch.long, device=device)
    val_labels = torch.tensor(bundle.labels["val"][0], dtype=torch.long, device=device)
    test_labels = torch.tensor(bundle.labels["test"], dtype=torch.long, device=device)

    net = _build_backbone(args, g, bundle.dl, in_dims, num_classes).to(device)
    _smart_load_pretrain(net, args.pretrain_ckpt, device=device, strict=args.strict_load)

    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    with torch.no_grad():
        prelogits, semantic_weight = _encode_once(args, net, features_list, e_feat)

    hidden_dim = args.shgn_hidden_dim if args.model_type == "SHGN" else args.hidden_dim
    classify = _build_classifier(args, hidden_dim, num_classes, semantic_weight=semantic_weight).to(device)

    optimizer = None
    if args.tuning != "sum":
        optimizer = torch.optim.AdamW(classify.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = (
        Path(args.save_dir)
        / bundle.dataset_name
        / f"{args.shotnum}-shot"
        / f"seed{args.seed}"
        / f"{args.model_type}_{args.tuning}"
        / f"repeat{repeat_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / "best_classifier.pt"

    es = EarlyStopping(EarlyStopConfig(
        patience=args.patience,
        min_delta=0.0,
        mode="min",
        save_best=True,
        save_path=str(best_ckpt),
    ))

    best_epoch = -1

    for epoch in range(1, args.epoch + 1):
        classify.train()

        logits = _forward_classifier(args, classify, g, trans_g, prelogits, e_feat)

        train_loss, train_micro, train_macro, train_center = _compute_train_or_val(
            args=args,
            logits=logits,
            idx=train_idx,
            labels=train_labels,
            num_classes=num_classes,
            center=None,
            recompute_center=True,
        )

        if torch.isnan(train_loss).any():
            raise ValueError("train_loss contains NaN.")

        if optimizer is not None:
            optimizer.zero_grad()
        train_loss.backward(retain_graph=False)
        if optimizer is not None:
            optimizer.step()

        classify.eval()
        with torch.no_grad():
            logits_val = _forward_classifier(args, classify, g, trans_g, prelogits, e_feat)

            if val_idx.numel() > 0:
                # 原始 HGPrompt 行为：
                # - weight-sum: val 上重算 center
                # - 其它中心类变体: 用 train center 做 val
                recompute_center = args.tuning == "weight-sum"
                val_loss, val_micro, val_macro, _ = _compute_train_or_val(
                    args=args,
                    logits=logits_val,
                    idx=val_idx,
                    labels=val_labels,
                    num_classes=num_classes,
                    center=None if recompute_center else train_center,
                    recompute_center=recompute_center,
                )
                monitor = float(val_loss.item())
            else:
                val_loss = train_loss.detach()
                val_micro, val_macro = train_micro, train_macro
                monitor = float(train_loss.item())

            if epoch_callback is not None:
                test_micro, test_macro = _evaluate_test(
                    args=args,
                    logits=logits_val,
                    test_idx=test_idx,
                    test_labels=test_labels,
                    num_classes=num_classes,
                    center=train_center,
                )
            else:
                test_micro, test_macro = None, None

        should_stop = es.step(
            metric=monitor,
            epoch=epoch,
            model=classify,
            extra_state={
                "best_center": None if train_center is None else train_center.detach().cpu(),
                "repeat_id": repeat_id,
                "seed": args.seed,
                "dataset": bundle.dataset_name,
                "model_type": args.model_type,
                "tuning": args.tuning,
            },
        )

        if monitor <= (es.best if es.best is not None else monitor):
            best_epoch = epoch

        if epoch_callback is not None:
            epoch_callback(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss.item()),
                    "train_micro": float(train_micro),
                    "train_macro": float(train_macro),
                    "val_loss": float(val_loss.item()),
                    "val_micro": float(val_micro),
                    "val_macro": float(val_macro),
                    "test_micro": float(test_micro),
                    "test_macro": float(test_macro),
                    "monitor": float(monitor),
                    "best_epoch": int(es.best_epoch),
                    "is_best": bool(epoch == es.best_epoch),
                    "early_stop": bool(should_stop),
                }
            )

        print(
            f"Repeat {repeat_id:02d} | Epoch {epoch:03d} | "
            f"train_loss={train_loss.item():.4f} | "
            f"train_f1(micro/macro)={train_micro:.4f}/{train_macro:.4f} | "
            f"val_loss={float(val_loss.item()):.4f} | "
            f"val_f1(micro/macro)={val_micro:.4f}/{val_macro:.4f}"
        )

        if should_stop:
            print(f"Early stop at epoch {epoch}, best_epoch={es.best_epoch}, best_val_loss={es.best:.6f}")
            break

    payload = torch.load(best_ckpt, map_location=device)
    classify.load_state_dict(payload["model_state"])
    best_center = payload.get("best_center", None)
    if isinstance(best_center, torch.Tensor):
        best_center = best_center.to(device)

    classify.eval()
    with torch.no_grad():
        logits_test = _forward_classifier(args, classify, g, trans_g, prelogits, e_feat)
        test_micro, test_macro = _evaluate_test(
            args=args,
            logits=logits_test,
            test_idx=test_idx,
            test_labels=test_labels,
            num_classes=num_classes,
            center=best_center,
        )

    print(
        f"[TEST] repeat={repeat_id:02d} | "
        f"micro={test_micro:.4f} | macro={test_macro:.4f} | best_epoch={payload.get('best_epoch', es.best_epoch)}"
    )

    return RepeatResult(
        test_micro=float(test_micro),
        test_macro=float(test_macro),
        best_epoch=int(payload.get("best_epoch", es.best_epoch)),
    )


def run_hgprompt_downstream(args):
    if args.tasknum != 1:
        raise ValueError(
            "HGEP split adapter 目前按每次一个 split/task 运行。"
            "请用 --repeat 做重复实验，或多次调用脚本处理多个 seeds。"
        )

    bundle = load_hgprompt_downstream_bundle(
        root=args.root,
        dataset=args.dataset,
        splits=args.splits,
        shot=args.shotnum,
        seed=args.seed,
    )

    all_res: list[RepeatResult] = []
    for repeat_id in range(args.repeat):
        all_res.append(_run_once(args, bundle, repeat_id))

    micro = np.array([x.test_micro for x in all_res], dtype=np.float64)
    macro = np.array([x.test_macro for x in all_res], dtype=np.float64)

    print("####################################################")
    print(f"dataset={bundle.dataset_name} | shot={args.shotnum} | seed={args.seed} | repeat={args.repeat}")
    print(f"micro mean={micro.mean():.4f} std={micro.std():.4f}")
    print(f"macro mean={macro.mean():.4f} std={macro.std():.4f}")


def build_parser():
    ap = argparse.ArgumentParser("HGPrompt downstream adapted to HGEP layout")

    # HGEP-style path / split / checkpoint
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--shot", "--shotnum", dest="shotnum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tasknum", type=int, default=1)
    ap.add_argument("--pretrain_ckpt", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="artifacts/checkpoints/hgprompt/downstream")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--strict_load", action="store_true")

    # faithful HGPrompt args
    ap.add_argument("--feats_type", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--bottle_net_hidden_dim", type=int, default=2)
    ap.add_argument("--bottle_net_output_dim", type=int, default=64)
    ap.add_argument("--edge_feats", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--epoch", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--model_type", type=str, default="gcn", choices=["gcn", "gat", "gin", "SHGN"])
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--slope", type=float, default=0.05)

    ap.add_argument("--tuning", type=str, default="weight-sum-center-fixed",
                    choices=["linear", "gcn", "weight-sum", "weight-sum-center-fixed", "bottle-net", "cat", "sum", "cat_edge", "prompt_gcn"])

    ap.add_argument("--subgraph_hop_num", type=int, default=1)
    ap.add_argument("--pre_loss_weight", type=float, default=1.0)

    ap.add_argument("--hetero_pretrain", type=int, default=0)
    ap.add_argument("--hetero_pretrain_subgraph", type=int, default=0)
    ap.add_argument("--pretrain_semantic", type=int, default=0)
    ap.add_argument("--pretrain_each_loss", type=int, default=0)

    ap.add_argument("--add_edge_info2prompt", type=int, default=1)
    ap.add_argument("--each_type_subgraph", type=int, default=1)

    ap.add_argument("--cat_prompt_dim", type=int, default=64)
    ap.add_argument("--cat_hprompt_dim", type=int, default=64)

    ap.add_argument("--tuple_neg_disconnected_num", type=int, default=1)
    ap.add_argument("--tuple_neg_unrelated_num", type=int, default=1)

    ap.add_argument("--semantic_prompt", type=int, default=1)
    ap.add_argument("--semantic_prompt_weight", type=float, default=0.1)

    ap.add_argument("--freebase_type", type=int, default=0)
    ap.add_argument("--shgn_hidden_dim", type=int, default=3)

    return ap


def main():
    args = build_parser().parse_args()
    run_hgprompt_downstream(args)


if __name__ == "__main__":
    main()
