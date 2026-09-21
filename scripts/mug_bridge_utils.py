from __future__ import annotations

import os
import random
import sys
import time
import types
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MUG_ROOT = Path("/home_A/yuanqilin/repos/MUG")


def import_mug():
    if "GCL" not in sys.modules:
        sys.modules["GCL"] = types.ModuleType("GCL")
    if str(MUG_ROOT) not in sys.path:
        sys.path.insert(0, str(MUG_ROOT))
    old_cwd = Path.cwd()
    os.chdir(MUG_ROOT)
    try:
        from model import MUG, DimensionAwareEncoder, sample_nodes_for_dimensional_basis
        from utils import (
            ContextualStructuralEncoder,
            load_best_configs,
            load_data,
            preprocess_features,
            set_random_seed,
            train_contextual_structural_encoder,
        )
        from utils.params import build_args, datasets_args
    finally:
        os.chdir(old_cwd)
    return {
        "MUG": MUG,
        "DimensionAwareEncoder": DimensionAwareEncoder,
        "sample_nodes_for_dimensional_basis": sample_nodes_for_dimensional_basis,
        "ContextualStructuralEncoder": ContextualStructuralEncoder,
        "load_best_configs": load_best_configs,
        "load_data": load_data,
        "preprocess_features": preprocess_features,
        "set_random_seed": set_random_seed,
        "train_contextual_structural_encoder": train_contextual_structural_encoder,
        "build_args": build_args,
        "datasets_args": datasets_args,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


def make_fewshot_splits(labels: torch.Tensor, shot: int, seeds: list[int]):
    y = labels.detach().cpu().long().numpy()
    classes = sorted(int(c) for c in np.unique(y))
    splits = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_idx = []
        val_idx = []
        used = set()
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            chosen = rng.choice(cls_idx, size=shot * 2, replace=False)
            train_idx.extend(chosen[:shot].tolist())
            val_idx.extend(chosen[shot:].tolist())
            used.update(int(i) for i in chosen)
        test_idx = [i for i in range(y.shape[0]) if i not in used]
        splits.append(
            {
                "seed": int(seed),
                "train": torch.tensor(train_idx, dtype=torch.long),
                "val": torch.tensor(val_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            }
        )
    return splits


def load_mug_eval_split(dataset: str, ratio: int):
    mug = import_mug()
    old_cwd = Path.cwd()
    os.chdir(MUG_ROOT)
    try:
        data, _g, _processed_metapaths = mug["load_data"](
            dataset, [int(ratio)], mug["datasets_args"][dataset]["type_num"]
        )
    finally:
        os.chdir(old_cwd)
    nei_index, _feats, mps, _pos, label, train, val, test = data
    return {
        "labels": torch.argmax(label, dim=-1).long(),
        "train_pool": train[0].long(),
        "val": val[0].long(),
        "test": test[0].long(),
        "mps": mps,
        "nei_index": nei_index,
        "type_num": mug["datasets_args"][dataset]["type_num"],
    }


def sample_fewshot_from_pool(
    labels: torch.Tensor,
    train_pool: torch.Tensor,
    shot: int,
    seed: int,
) -> torch.Tensor:
    y = labels.detach().cpu().long().numpy()
    pool = train_pool.detach().cpu().long().numpy()
    rng = np.random.default_rng(seed)
    selected = []
    for cls in sorted(int(c) for c in np.unique(y)):
        candidates = pool[y[pool] == cls]
        if candidates.shape[0] < shot:
            raise ValueError(f"Class {cls} has only {candidates.shape[0]} train-pool nodes.")
        chosen = rng.choice(candidates, size=shot, replace=False)
        selected.extend(int(i) for i in chosen.tolist())
    return torch.tensor(selected, dtype=torch.long)


def target_type_neighborhood(nei_index, type_num: list[int], hops: list[int]):
    num_types = len(type_num)
    offsets = np.cumsum([0] + type_num[:-1]).tolist()
    total = int(sum(type_num))
    rows = []
    cols = []
    for rel_id, neigh_lists in enumerate(nei_index):
        dst_type = rel_id + 1
        dst_offset = offsets[dst_type]
        for src, neigh in enumerate(neigh_lists):
            if len(neigh) == 0:
                continue
            neigh_np = neigh.detach().cpu().numpy().astype(np.int64, copy=False)
            dst = dst_offset + neigh_np
            rows.extend([src] * len(dst))
            cols.extend(dst.tolist())
            rows.extend(dst.tolist())
            cols.extend([src] * len(dst))
    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.ones(idx.size(1), dtype=torch.float32)
    deg = torch.zeros(total, dtype=torch.float32)
    deg.index_add_(0, idx[0], val)
    val = val / deg[idx[0]].clamp_min(1.0)
    adj = torch.sparse_coo_tensor(idx, val, (total, total)).coalesce()

    x = torch.zeros((total, num_types), dtype=torch.float32)
    for type_id, count in enumerate(type_num):
        start = offsets[type_id]
        x[start : start + count, type_id] = 1.0
    blocks = []
    cur = x
    max_hop = max(hops)
    for hop in range(max_hop + 1):
        if hop in hops:
            blocks.append(cur[: type_num[0]].clone())
        if hop != max_hop:
            cur = torch.sparse.mm(adj, cur)
    return torch.cat(blocks, dim=1)


def sample_metapath_edges(mps, z_type, max_edges_per_metapath: int, seed: int):
    src_parts = []
    dst_parts = []
    weight_parts = []
    feat_parts = []
    rng = torch.Generator().manual_seed(seed)
    for mp_id, mp in enumerate(mps):
        coo = mp.coalesce().cpu()
        idx = coo.indices()
        val = coo.values().float()
        if max_edges_per_metapath > 0 and idx.size(1) > max_edges_per_metapath:
            perm = torch.randperm(idx.size(1), generator=rng)[:max_edges_per_metapath]
            idx = idx[:, perm]
            val = val[perm]
        src = idx[0].long()
        dst = idx[1].long()
        edge_onehot = torch.zeros((src.numel(), len(mps)), dtype=torch.float32)
        edge_onehot[:, mp_id] = 1.0
        zs = z_type[src]
        zd = z_type[dst]
        feat = torch.cat([zs, zd, zs - zd, torch.abs(zs - zd), edge_onehot], dim=1)
        src_parts.append(src)
        dst_parts.append(dst)
        weight_parts.append(val)
        feat_parts.append(feat)
    return (
        torch.cat(src_parts, dim=0),
        torch.cat(dst_parts, dim=0),
        torch.cat(weight_parts, dim=0),
        torch.cat(feat_parts, dim=0),
    )


class LargeTargetMUG(nn.Module):
    """MUG wrapper for target sets too large for dense metapath reconstruction."""

    def __init__(self, base_model: nn.Module, recon_edges_per_metapath: int = 200000):
        super().__init__()
        self.base_model = base_model
        self.recon_edges_per_metapath = int(recon_edges_per_metapath)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def mps_to_gs(self, mps):
        gs = []
        for mp in mps:
            coo = mp.coalesce()
            idx = coo.indices()
            gs.append(dgl.graph((idx[0], idx[1]), num_nodes=mp.size(0), device=idx.device))
        return gs

    def mask_mp_edge_reconstruction(self, feat, mps, epoch):
        del epoch
        gs = self.mps_to_gs(mps)
        enc_rep, _ = self.base_model.encoder(gs, feat, return_hidden=False)
        rep = self.base_model.encoder_to_decoder_edge_recon(enc_rep)
        if self.base_model.decoder_type == "mlp":
            feat_recon = self.base_model.decoder(rep)
            att_mp = torch.full(
                (len(mps),),
                1.0 / max(len(mps), 1),
                device=feat_recon.device,
                dtype=feat_recon.dtype,
            )
        else:
            feat_recon, att_mp = self.base_model.decoder(gs, rep)

        z = F.normalize(feat_recon, p=2, dim=-1)
        losses = []
        gen = torch.Generator(device=z.device)
        gen.manual_seed(0)
        for mp_id, mp in enumerate(mps):
            idx = mp.coalesce().indices()
            if idx.numel() == 0:
                continue
            if self.recon_edges_per_metapath > 0 and idx.size(1) > self.recon_edges_per_metapath:
                perm = torch.randperm(idx.size(1), device=idx.device, generator=gen)[: self.recon_edges_per_metapath]
                idx = idx[:, perm]
            pos_src, pos_dst = idx[0], idx[1]
            neg_src = torch.randint(0, z.size(0), (pos_src.numel(),), device=z.device, generator=gen)
            neg_dst = torch.randint(0, z.size(0), (pos_src.numel(),), device=z.device, generator=gen)
            pos_score = (z[pos_src] * z[pos_dst]).sum(dim=-1)
            neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            losses.append(att_mp[mp_id] * loss)
        if not losses:
            return z.new_tensor(0.0)
        return torch.stack(losses).sum()

    def forward(self, feats, mps, **kwargs):
        dimension_sig = self.base_model.Dimension_encoder(self.base_model.d_sample_matrix)
        if self.base_model.unified_feature:
            unified_feat = self.base_model.feature_sig_propagate(
                feats[0][:, : self.base_model.focused_feature_dim],
                dimension_sig,
            )
        else:
            unified_feat = self.base_model.feature_sig_propagate(feats[0], dimension_sig)

        gs = self.mps_to_gs(mps)
        enc_out, _ = self.base_model.encoder(gs, unified_feat, return_hidden=False)
        edge_recon_loss = self.mask_mp_edge_reconstruction(unified_feat, mps, kwargs.get("epoch", None))
        loss = self.base_model.mp_edge_recon_loss_weight * edge_recon_loss
        loss_scatter = self.base_model.ssl_loss_fn_scatter(enc_out)
        loss = loss + self.base_model.losslam_scatter * loss_scatter
        loss_sig = self.base_model.dim_loss_fn()
        loss = loss + self.base_model.losslam_sig_cross * loss_sig
        return loss, loss.item(), loss_sig.item(), loss_scatter.item()


def _onehot_labels_with_unlabeled(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros((labels.numel(), num_classes), dtype=torch.float32)
    mask = labels >= 0
    if bool(mask.any()):
        out[mask, labels[mask].long()] = 1.0
    return out


def _sample_pairs_from_groups(groups, num_target: int, max_edges: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    cols = []
    for books in groups.values():
        if len(books) <= 1:
            continue
        arr = np.asarray(sorted(books), dtype=np.int64)
        if arr.size > 64:
            arr = rng.choice(arr, size=64, replace=False)
        pair_budget = min(max(2 * arr.size, 8), 256)
        src = rng.choice(arr, size=pair_budget, replace=True)
        dst = rng.choice(arr, size=pair_budget, replace=True)
        keep = src != dst
        rows.extend(src[keep].tolist())
        cols.extend(dst[keep].tolist())
        rows.extend(dst[keep].tolist())
        cols.extend(src[keep].tolist())
        if max_edges > 0 and len(rows) >= max_edges:
            break
    rows.extend(range(num_target))
    cols.extend(range(num_target))
    if max_edges > 0 and len(rows) > max_edges + num_target:
        non_self = len(rows) - num_target
        chosen = rng.choice(non_self, size=max_edges, replace=False)
        rows = [rows[i] for i in chosen] + list(range(num_target))
        cols = [cols[i] for i in chosen] + list(range(num_target))
    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.ones(idx.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (num_target, num_target)).coalesce()


def _build_hgb_freebase_mug_data(args):
    from scripts.peprompt_benchmark import _load_raw_heterograph

    graph, target = _load_raw_heterograph(
        getattr(args, "root", "data"),
        "Freebase",
        int(getattr(args, "feats_type", 1)),
    )
    if target != "book":
        raise RuntimeError(f"Expected HGB Freebase target 'book', got {target!r}.")

    num_target = graph.num_nodes(target)
    labels = graph.ndata["y"][target].detach().cpu().long()
    num_classes = int(labels[labels >= 0].max().item()) + 1
    type_names = [target] + [nt for nt in graph.ntypes if nt != target]
    type_num = [graph.num_nodes(nt) for nt in type_names]

    incidence_by_type = {nt: [set() for _ in range(num_target)] for nt in type_names[1:]}
    groups_by_type = {nt: {} for nt in type_names[1:]}
    for src_t, _rel, dst_t in graph.canonical_etypes:
        if src_t != target and dst_t != target:
            continue
        other_t = dst_t if src_t == target else src_t
        if other_t == target or other_t not in groups_by_type:
            continue
        src, dst = graph.edges(etype=(src_t, _rel, dst_t), order="eid")
        src = src.detach().cpu().numpy()
        dst = dst.detach().cpu().numpy()
        if src_t == target:
            books, others = src, dst
        else:
            books, others = dst, src
        groups = groups_by_type[other_t]
        incidence = incidence_by_type[other_t]
        for book_id, other_id in zip(books.tolist(), others.tolist()):
            incidence[int(book_id)].add(int(other_id))
            groups.setdefault(int(other_id), set()).add(int(book_id))

    selected_types = []
    nei_index = []
    mps = []
    max_edges = int(getattr(args, "hgb_mug_max_edges_per_metapath", 500000))
    for type_id, ntype in enumerate(type_names[1:], start=1):
        groups = groups_by_type[ntype]
        if not groups:
            continue
        selected_types.append(ntype)
        nei_index.append([torch.tensor(sorted(v), dtype=torch.long) for v in incidence_by_type[ntype]])
        mps.append(
            _sample_pairs_from_groups(
                groups,
                num_target=num_target,
                max_edges=max_edges,
                seed=int(getattr(args, "seed", 0)) + type_id,
            )
        )

    feature_dim = int(getattr(args, "hgb_mug_feature_dim", 256))
    gen = torch.Generator().manual_seed(int(getattr(args, "seed", 0)))
    feat = torch.randn((num_target, feature_dim), generator=gen, dtype=torch.float32) * 0.01
    degree_blocks = []
    for ntype in selected_types:
        deg = torch.tensor([len(v) for v in incidence_by_type[ntype]], dtype=torch.float32).view(-1, 1)
        degree_blocks.append(torch.log1p(deg))
    if degree_blocks:
        deg_feat = torch.cat(degree_blocks, dim=1)
        deg_feat = deg_feat / deg_feat.max(dim=0, keepdim=True).values.clamp_min(1.0)
        width = min(feature_dim, deg_feat.size(1))
        feat[:, :width] = deg_feat[:, :width]
    feat = F.normalize(feat, p=2, dim=-1)

    pos = torch.empty((0, 2), dtype=torch.long)
    label_onehot = _onehot_labels_with_unlabeled(labels, num_classes)
    data = (nei_index, [feat], mps, pos, label_onehot, [], [], [])
    meta = {
        "type_num": type_num,
        "type_names": type_names,
        "selected_metapath_types": selected_types,
        "num_classes": num_classes,
        "targetnode": target,
        "source": "hgb_freebase",
        "max_edges_per_metapath": max_edges,
        "feature_dim": feature_dim,
    }
    return data, meta, labels


def train_mug_embeddings(args, device):
    mug = import_mug()
    mug_data_source = str(getattr(args, "mug_data_source", "native"))
    old_argv = sys.argv
    old_cwd = Path.cwd()
    config_dataset = "freebase" if mug_data_source == "hgb" else args.dataset
    sys.argv = [old_argv[0], "--dataset", config_dataset]
    os.chdir(MUG_ROOT)
    try:
        mug_args = mug["build_args"]()
        mug_args = mug["load_best_configs"](mug_args, str(MUG_ROOT / "configs.yml"))
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    mug_args.seed = args.seed
    mug_args.gpu = args.gpu
    mug_args.epochs = args.mug_epochs
    mug_args.device = device
    if mug_data_source == "hgb":
        mug_args.dataset = "hgb_freebase"
        mug_args.unified_feature = False
        mug_args.n_labels = 7
        mug_args.hidden_dim = int(getattr(args, "hgb_mug_hidden_dim", mug_args.hidden_dim))
        mug_args.feature_signal_dim = int(getattr(args, "hgb_mug_feature_signal_dim", mug_args.feature_signal_dim))
        mug_args.sample_size = int(getattr(args, "hgb_mug_sample_size", mug_args.sample_size))
    if getattr(args, "no_unified_feature", False):
        mug_args.unified_feature = False
    mug["set_random_seed"](args.seed)

    hgb_meta = None
    labels_class = None
    if mug_data_source == "hgb":
        (nei_index, feats, mps, _pos, label, _train, _val, _test), hgb_meta, labels_class = _build_hgb_freebase_mug_data(args)
        processed_metapaths = None
        g = None
    else:
        old_cwd = Path.cwd()
        os.chdir(MUG_ROOT)
        try:
            (nei_index, feats, mps, _pos, label, _train, _val, _test), g, processed_metapaths = mug["load_data"](
                args.dataset,
                [int(getattr(args, "mug_ratio", 60))],
                mug["datasets_args"][args.dataset]["type_num"],
            )
        finally:
            os.chdir(old_cwd)

    feats_dim_list = [i.shape[1] for i in feats]
    num_mp = int(len(mps))
    z_dim = feats_dim_list[0]
    feature_dim = feats_dim_list[0]
    if as_bool(getattr(mug_args, "unified_feature", False)):
        cse_model = mug["ContextualStructuralEncoder"](g.edge_index_dict, processed_metapaths, mug_args)
        cse_model = mug["train_contextual_structural_encoder"](mug_args, cse_model, mug_args.mps_epoch, device)
        z_struct = cse_model("target").detach().cpu()
        feats[0] = torch.hstack([feats[0], torch.FloatTensor(mug["preprocess_features"](z_struct))])

    activator = nn.PReLU if mug_args.activator == "PReLU" else nn.ReLU
    dimension_encoder = mug["DimensionAwareEncoder"](
        mug_args.sample_size, mug_args.feature_signal_dim * 2, mug_args.feature_signal_dim, activator
    )
    base_model = mug["MUG"](
        mug_args,
        num_mp,
        feature_dim,
        dimension_encoder,
        mug["sample_nodes_for_dimensional_basis"],
        mug_args.sample_size,
    ).to(device)
    model = base_model
    if mug_data_source == "hgb":
        model = LargeTargetMUG(
            base_model,
            recon_edges_per_metapath=int(getattr(args, "hgb_mug_recon_edges_per_metapath", 200000)),
        ).to(device)
    feats_dev = [feat.to(device) for feat in feats]
    mps_dev = [mp.to(device) for mp in mps]
    opt = torch.optim.Adam(model.parameters(), lr=mug_args.lr, weight_decay=mug_args.l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=mug_args.scheduler_gamma) if mug_args.scheduler else None
    start = time.time()
    for epoch in range(args.mug_epochs):
        model.update_sample(feats_dev[0][:, :z_dim], if_rand=as_bool(mug_args.if_rand))
        model.train()
        opt.zero_grad()
        loss, _loss_item, loss_sig, loss_scatter = model(feats_dev, mps_dev, nei_index=nei_index, epoch=epoch)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        print(
            f"[MUG] epoch={epoch + 1}/{args.mug_epochs} "
            f"loss={loss.item():.6f} sig={loss_sig:.6f} scatter={loss_scatter:.6f}",
            flush=True,
        )
    model.eval()
    with torch.no_grad():
        embeds = model.get_embeds(feats_dev, mps_dev, if_rand=as_bool(mug_args.if_rand)).detach().cpu()
    return {
        "embeds": embeds,
        "labels": labels_class.long() if labels_class is not None else torch.argmax(label, dim=-1).long(),
        "mps": mps,
        "nei_index": nei_index,
        "mug_args": vars(mug_args),
        "hgb_meta": hgb_meta,
        "train_seconds": time.time() - start,
    }


def summarize(rows):
    out = {}
    for method in sorted({r["method"] for r in rows}):
        vals = [r for r in rows if r["method"] == method]
        micro = np.array([v["micro"] for v in vals], dtype=np.float64)
        macro = np.array([v["macro"] for v in vals], dtype=np.float64)
        out[method] = {
            "count": int(len(vals)),
            "micro_mean": float(micro.mean()),
            "micro_std": float(micro.std(ddof=0)),
            "macro_mean": float(macro.mean()),
            "macro_std": float(macro.std(ddof=0)),
        }
    return out
