from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def import_hgmae(hgmae_root: str | os.PathLike | None = None):
    candidates = []
    if hgmae_root is not None:
        candidates.append(Path(hgmae_root))
    if os.environ.get("HGMAE_ROOT"):
        candidates.append(Path(os.environ["HGMAE_ROOT"]))
    candidates.extend([ROOT / "external" / "HGMAE", Path("/home_A/yuanqilin/repos/HGMAE"), Path("/tmp/HGMAE")])

    for candidate in candidates:
        if (candidate / "hgmae" / "models" / "edcoder.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            old_cwd = Path.cwd()
            os.chdir(candidate)
            try:
                from hgmae.models.edcoder import PreModel
                from hgmae.utils import load_best_configs, set_random_seed
            finally:
                os.chdir(old_cwd)
            return {
                "root": candidate,
                "PreModel": PreModel,
                "load_best_configs": load_best_configs,
                "set_random_seed": set_random_seed,
            }

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "HGMAE source tree was not found. Clone https://github.com/meettyj/HGMAE "
        f"and pass --hgmae_root, or set HGMAE_ROOT. Tried: {tried}"
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _row_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def _target_features_from_graph(graph: dgl.DGLHeteroGraph, target: str, args) -> torch.Tensor:
    mode = str(getattr(args, "hgmae_feature_mode", "auto"))
    feature_dim = int(getattr(args, "hgmae_feature_dim", 256))
    use_target_x = mode in {"auto", "target_x", "target_x_degree"}

    target_x = graph.ndata["x"][target].detach().cpu().float()
    has_signal = bool(target_x.numel()) and float(target_x.abs().sum().item()) > 0.0
    if use_target_x and has_signal:
        feat = target_x
        if mode == "target_x":
            return _row_l2_normalize(feat)
    else:
        feat = torch.empty((graph.num_nodes(target), 0), dtype=torch.float32)

    degree_blocks = []
    for etype in graph.canonical_etypes:
        src_t, _, dst_t = etype
        if src_t != target and dst_t != target:
            continue
        src, dst = graph.edges(etype=etype, order="eid")
        deg = torch.zeros(graph.num_nodes(target), dtype=torch.float32)
        ids = src.detach().cpu().long() if src_t == target else dst.detach().cpu().long()
        if ids.numel() > 0:
            deg.index_add_(0, ids, torch.ones(ids.numel(), dtype=torch.float32))
        degree_blocks.append(torch.log1p(deg).view(-1, 1))

    if degree_blocks:
        degree_feat = torch.cat(degree_blocks, dim=1)
        denom = degree_feat.max(dim=0, keepdim=True).values.clamp_min(1.0)
        degree_feat = degree_feat / denom
    else:
        degree_feat = torch.zeros((graph.num_nodes(target), 1), dtype=torch.float32)

    if feat.numel() == 0:
        feat = degree_feat
    elif mode == "target_x_degree":
        feat = torch.cat([feat, degree_feat], dim=1)

    if feat.size(1) > feature_dim:
        gen = torch.Generator().manual_seed(int(getattr(args, "seed", 0)))
        proj = torch.randn((feat.size(1), feature_dim), generator=gen, dtype=torch.float32)
        proj = proj / float(max(1, feat.size(1))) ** 0.5
        feat = torch.tanh(feat @ proj)
    elif feat.size(1) < feature_dim and (mode != "target_x" or not has_signal):
        pad = torch.zeros((feat.size(0), feature_dim - feat.size(1)), dtype=torch.float32)
        feat = torch.cat([feat, pad], dim=1)
    return _row_l2_normalize(feat)


def _sample_pairs_from_groups(groups: dict[int, set[int]], num_target: int, max_edges: int, seed: int):
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    for targets in groups.values():
        if len(targets) <= 1:
            continue
        arr = np.asarray(sorted(targets), dtype=np.int64)
        if arr.size > 128:
            arr = rng.choice(arr, size=128, replace=False)
        pair_budget = min(max(2 * arr.size, 8), 512)
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


def _build_target_metapaths(graph: dgl.DGLHeteroGraph, target: str, args):
    num_target = graph.num_nodes(target)
    max_edges = int(getattr(args, "hgmae_max_edges_per_metapath", 500000))
    max_metapaths = int(getattr(args, "hgmae_max_metapaths", 8))
    mps = []
    metapath_names = []
    nei_index = []
    seen_other_types: set[str] = set()

    for etype_idx, (src_t, rel_t, dst_t) in enumerate(graph.canonical_etypes):
        if src_t != target and dst_t != target:
            continue
        other_t = dst_t if src_t == target else src_t
        if other_t == target:
            continue
        if other_t in seen_other_types:
            continue
        seen_other_types.add(other_t)
        src, dst = graph.edges(etype=(src_t, rel_t, dst_t), order="eid")
        src = src.detach().cpu().numpy()
        dst = dst.detach().cpu().numpy()
        targets = src if src_t == target else dst
        others = dst if src_t == target else src

        groups: dict[int, set[int]] = {}
        incidence = [set() for _ in range(num_target)]
        for target_id, other_id in zip(targets.tolist(), others.tolist()):
            target_id = int(target_id)
            other_id = int(other_id)
            incidence[target_id].add(other_id)
            groups.setdefault(other_id, set()).add(target_id)
        if not groups:
            continue

        mp = _sample_pairs_from_groups(
            groups,
            num_target=num_target,
            max_edges=max_edges,
            seed=int(getattr(args, "seed", 0)) + etype_idx,
        )
        mps.append(mp)
        metapath_names.append(f"{target}-{rel_t}-{other_t}-...-{target}")
        nei_index.append([torch.tensor(sorted(v), dtype=torch.long) for v in incidence])
        if len(mps) >= max_metapaths:
            break

    if not mps:
        idx = torch.arange(num_target, dtype=torch.long).view(1, -1).repeat(2, 1)
        val = torch.ones(num_target, dtype=torch.float32)
        mps.append(torch.sparse_coo_tensor(idx, val, (num_target, num_target)).coalesce())
        metapath_names.append("identity")
        nei_index.append([torch.empty(0, dtype=torch.long) for _ in range(num_target)])
    return nei_index, mps, metapath_names


class SampledEdgeReconPreModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, recon_edges_per_metapath: int):
        super().__init__()
        self.base_model = base_model
        self.recon_edges_per_metapath = int(recon_edges_per_metapath)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def mask_mp_edge_reconstruction(self, feat, mps, epoch):
        del epoch
        gs = self.base_model.mps_to_gs(mps)
        enc_rep, _ = self.base_model.encoder(gs, feat, return_hidden=False)
        rep = self.base_model.encoder_to_decoder_edge_recon(enc_rep)
        if self.base_model.decoder_type == "mlp":
            feat_recon = self.base_model.decoder(rep)
            att_mp = torch.full((len(mps),), 1.0 / max(1, len(mps)), device=feat_recon.device)
        else:
            feat_recon, att_mp = self.base_model.decoder(gs, rep)

        z = F.normalize(feat_recon, p=2, dim=-1)
        losses = []
        for mp_id, mp in enumerate(mps):
            idx = mp.coalesce().indices().to(z.device)
            if idx.numel() == 0:
                continue
            if self.recon_edges_per_metapath > 0 and idx.size(1) > self.recon_edges_per_metapath:
                perm = torch.randperm(idx.size(1), device=z.device)[: self.recon_edges_per_metapath]
                idx = idx[:, perm]
            pos_src, pos_dst = idx[0], idx[1]
            neg_src = torch.randint(0, z.size(0), (pos_src.numel(),), device=z.device)
            neg_dst = torch.randint(0, z.size(0), (pos_src.numel(),), device=z.device)
            pos_score = (z[pos_src] * z[pos_dst]).sum(dim=-1)
            neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
            losses.append(att_mp[mp_id] * F.binary_cross_entropy_with_logits(scores, labels))
        return torch.stack(losses).sum() if losses else z.new_tensor(0.0)


def _make_hgmae_args(args, feature_dim: int):
    cfg = {
        "activation": "prelu",
        "alpha_l": 3,
        "attn_drop": 0.2,
        "decoder": "han",
        "encoder": "han",
        "feat_drop": 0.2,
        "feat_mask_rate": "0.5,0.005,0.8",
        "hidden_dim": int(getattr(args, "hgmae_hidden_dim", 256)),
        "l2_coef": 0.0,
        "leave_unchanged": 0.2,
        "loss_fn": "sce",
        "lr": float(getattr(args, "hgmae_lr", 1e-3)),
        "mae_epochs": int(getattr(args, "hgmae_epochs", 200)),
        "mp_edge_alpha_l": 3,
        "mp_edge_mask_rate": "0.5,0.005,0.8",
        "mp_edge_recon_loss_weight": float(getattr(args, "hgmae_mp_edge_recon_loss_weight", 0.1)),
        "mps_embedding_dim": int(getattr(args, "hgmae_mps_embedding_dim", 64)),
        "mp2vec_feat_alpha_l": 2,
        "mp2vec_feat_drop": 0.2,
        "mp2vec_feat_pred_loss_weight": 0.1,
        "negative_slope": 0.2,
        "norm": "batchnorm",
        "num_heads": int(getattr(args, "hgmae_num_heads", 4)),
        "num_layers": int(getattr(args, "hgmae_num_layers", 2)),
        "num_out_heads": 1,
        "patience": int(getattr(args, "hgmae_patience", 20)),
        "replace_rate": 0.2,
        "residual": False,
        "scheduler": bool(getattr(args, "hgmae_scheduler", False)),
        "scheduler_gamma": 0.99,
        "use_mp2vec_feat_pred": False,
        "use_mp_edge_recon": bool(getattr(args, "hgmae_use_mp_edge_recon", False)),
        "focused_feature_dim": int(feature_dim),
    }
    return SimpleNamespace(**cfg)


def train_hgmae_embeddings(args, device: torch.device):
    hgmae = import_hgmae(getattr(args, "hgmae_root", None))
    from scripts.peprompt_benchmark import _load_raw_heterograph

    set_seed(int(getattr(args, "seed", 0)))
    hgmae["set_random_seed"](int(getattr(args, "seed", 0)))
    graph, target = _load_raw_heterograph(
        getattr(args, "root", "data"),
        str(getattr(args, "dataset")),
        int(getattr(args, "feats_type", 1 if str(getattr(args, "dataset")) == "Freebase" else 0)),
    )
    feats0 = _target_features_from_graph(graph, target, args)
    nei_index, mps, metapath_names = _build_target_metapaths(graph, target, args)
    labels = graph.ndata["y"][target].detach().cpu().long()

    model_args = _make_hgmae_args(args, int(feats0.size(1)))
    PreModel = hgmae["PreModel"]
    base_model = PreModel(model_args, len(mps), int(feats0.size(1)))
    if model_args.use_mp_edge_recon and graph.num_nodes(target) > int(getattr(args, "hgmae_dense_recon_max_nodes", 12000)):
        base_model.mask_mp_edge_reconstruction = SampledEdgeReconPreModel(
            base_model,
            int(getattr(args, "hgmae_recon_edges_per_metapath", 200000)),
        ).mask_mp_edge_reconstruction

    model = base_model.to(device)
    feats = [feats0.to(device)]
    mps_dev = [mp.to(device) for mp in mps]
    opt = torch.optim.Adam(model.parameters(), lr=model_args.lr, weight_decay=model_args.l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=model_args.scheduler_gamma) if model_args.scheduler else None

    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    start = time.time()
    for epoch in range(int(model_args.mae_epochs)):
        model.train()
        opt.zero_grad()
        loss, loss_item = model(feats, mps_dev, nei_index=nei_index, epoch=epoch)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        improved = float(loss_item) < best_loss
        if improved:
            best_loss = float(loss_item)
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch == 0 or (epoch + 1) % max(1, int(getattr(args, "hgmae_log_interval", 10))) == 0:
            print(f"[HGMAE] epoch={epoch + 1}/{model_args.mae_epochs} loss={loss_item:.6f} best={best_loss:.6f}", flush=True)
        if bad_epochs >= int(model_args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeds = model.get_embeds(feats, mps_dev, nei_index).detach().cpu()

    return {
        "embeds": embeds,
        "labels": labels,
        "mps": [mp.cpu() for mp in mps],
        "nei_index": nei_index,
        "hgmae_args": vars(model_args),
        "hgb_meta": {
            "dataset": str(getattr(args, "dataset")),
            "targetnode": target,
            "source": "hgb",
            "feature_dim": int(feats0.size(1)),
            "metapath_names": metapath_names,
            "num_metapaths": int(len(mps)),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
        },
        "train_seconds": time.time() - start,
    }
