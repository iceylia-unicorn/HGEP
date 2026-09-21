from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpbench.protocol_bridge.hgmp_peprompt import PEPromptRelation

from scripts.mug_bridge_utils import (
    load_mug_eval_split,
    make_fewshot_splits,
    sample_metapath_edges,
    sample_fewshot_from_pool,
    set_seed,
    summarize,
    target_type_neighborhood,
    train_mug_embeddings,
)


class LogRegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MUGNoPromptClassifier(nn.Module):
    def __init__(self, embeds: torch.Tensor, num_classes: int):
        super().__init__()
        self.register_buffer("embeds", embeds)
        self.head = LogRegHead(int(embeds.size(1)), num_classes)

    def forward(self):
        return self.head(self.embeds)


class MUGTypeNeighborhoodEdgeClassifier(nn.Module):
    """MUG target embeddings + the existing PEPrompt edge relation module.

    This keeps the prompt mechanism unchanged: TypeNeighborhoodEdge features are
    mapped by PEPromptRelation.edge_prompt_mlp and injected as edge-level messages.
    """

    def __init__(
        self,
        embeds: torch.Tensor,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_feature_dict: dict[tuple[str, str, str], torch.Tensor],
        num_classes: int,
        prompt_mode: str,
        prompt_alpha: float,
        prompt_dropout: float,
        prompt_aggr: str,
        prompt_use_ln: bool,
        prompt_hidden: int,
        prompt_edge_chunk_size: int,
        prompt_constraint: str,
        prompt_constraint_scale: float,
    ):
        super().__init__()
        self.register_buffer("embeds", embeds)
        self.edge_index_dict = edge_index_dict
        self.edge_feature_dict = edge_feature_dict
        edge_dim = int(next(iter(edge_feature_dict.values())).size(1))
        dim = int(embeds.size(1))
        self.hgnn_type = "MUG"
        self.relation_prompt = PEPromptRelation(
            metadata=(["target"], list(edge_index_dict.keys())),
            dim=dim,
            mode=prompt_mode,
            alpha=prompt_alpha,
            dropout=prompt_dropout,
            use_ln=prompt_use_ln,
            aggr=prompt_aggr,
            edge_feature_dim=edge_dim,
            edge_feature_name="peprompt_edge_feat",
            edge_prompt_hidden=prompt_hidden,
            edge_chunk_size=prompt_edge_chunk_size,
            prompt_constraint=prompt_constraint,
            prompt_constraint_scale=prompt_constraint_scale,
        )
        self.head = LogRegHead(dim, num_classes)

    def forward(self):
        x_dict = {"target": self.embeds}
        h_dict = self.relation_prompt(
            x_dict,
            self.edge_index_dict,
            self.edge_feature_dict,
            graph=None,
        )
        return self.head(h_dict["target"])


def _load_embedding_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"embeds", "labels", "mps", "nei_index", "mug_args"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"MUG embedding payload is missing keys: {missing}")
    return payload


def _save_embedding_payload(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeds": payload["embeds"],
            "labels": payload["labels"],
            "mps": payload["mps"],
            "nei_index": payload["nei_index"],
            "mug_args": payload["mug_args"],
            "train_seconds": payload.get("train_seconds"),
        },
        path,
    )


def _make_metapath_edge_dicts(
    mps,
    z_type: torch.Tensor,
    max_edges_per_metapath: int,
    seed: int,
    device: torch.device,
):
    edge_index_dict = {}
    edge_feature_dict = {}
    for mp_id, mp in enumerate(mps):
        src, dst, _weight, feat = sample_metapath_edges(
            [mp],
            z_type,
            max_edges_per_metapath=max_edges_per_metapath,
            seed=seed + mp_id,
        )
        # _sample_metapath_edges sees a single relation at a time, so replace
        # its local one-hot by a global metapath id one-hot.
        global_edge_onehot = torch.zeros((src.numel(), len(mps)), dtype=torch.float32)
        if src.numel() > 0:
            global_edge_onehot[:, mp_id] = 1.0
        base = feat[:, :-1] if feat.size(1) > 0 else feat
        feat = torch.cat([base, global_edge_onehot], dim=1).float().contiguous()
        etype = ("target", f"metapath_{mp_id}", "target")
        edge_index_dict[etype] = torch.stack([src.long(), dst.long()], dim=0)
        edge_feature_dict[etype] = feat
    return edge_index_dict, edge_feature_dict


def _train_once(model, labels, split, args, run_seed: int, device: torch.device):
    set_seed(run_seed)
    model = model.to(device)
    labels = labels.to(device)
    train_idx = split["train"].to(device)
    val_idx = split["val"].to(device)
    test_idx = split["test"].to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_macro = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model()
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        opt.step()

        should_eval = (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(1, int(args.eval_interval)) == 0
        )
        if should_eval:
            model.eval()
            with torch.no_grad():
                logits = model()
                val_loss = F.cross_entropy(logits[val_idx], labels[val_idx]).item()
                val_pred = logits[val_idx].argmax(dim=-1)
                val_macro = _macro_f1(labels[val_idx], val_pred, args.num_classes)

            monitor_improved = (
                val_loss <= best_val_loss
                if args.early_stop_mode == "loss"
                else val_macro > best_val_macro
            )
            if monitor_improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                best_val_loss = float(val_loss)
                best_val_macro = float(val_macro)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model()
        pred = logits[test_idx].argmax(dim=-1)
    return {
        "micro": _micro_f1(labels[test_idx], pred),
        "macro": _macro_f1(labels[test_idx], pred, args.num_classes),
        "best_epoch": int(best_epoch),
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "best_val_macro": float(best_val_macro),
    }


def _train_mug_fixed_split_once(model, labels, split, args, run_seed: int, device: torch.device):
    set_seed(run_seed)
    model = model.to(device)
    labels = labels.to(device)
    train_idx = split["train"].to(device)
    val_idx = split["val"].to(device)
    test_idx = split["test"].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.eva_lr, weight_decay=args.eva_wd)
    val_macro_scores = []
    val_micro_scores = []
    test_macro_scores = []
    test_micro_scores = []
    val_losses = []

    for epoch in range(1, args.eval_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model()
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model()
            val_pred = logits[val_idx].argmax(dim=-1)
            test_pred = logits[test_idx].argmax(dim=-1)
            val_losses.append(float(F.cross_entropy(logits[val_idx], labels[val_idx]).item()))
            val_macro_scores.append(_macro_f1(labels[val_idx], val_pred, args.num_classes))
            val_micro_scores.append(_micro_f1(labels[val_idx], val_pred))
            test_macro_scores.append(_macro_f1(labels[test_idx], test_pred, args.num_classes))
            test_micro_scores.append(_micro_f1(labels[test_idx], test_pred))

    best_macro_epoch = int(np.argmax(val_macro_scores))
    best_micro_epoch = int(np.argmax(val_micro_scores))
    return {
        "micro": float(test_micro_scores[best_micro_epoch]),
        "macro": float(test_macro_scores[best_macro_epoch]),
        "best_epoch": int(best_macro_epoch + 1),
        "best_val_loss": float(val_losses[best_macro_epoch]),
        "best_val_macro": float(val_macro_scores[best_macro_epoch]),
    }


def _train_mug_fewshot_once(model, labels, split, args, run_seed: int, device: torch.device):
    set_seed(run_seed)
    model = model.to(device)
    labels = labels.to(device)
    train_idx = split["train"].to(device)
    test_idx = split["test"].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.eva_lr, weight_decay=args.eva_wd)
    for epoch in range(1, args.fewshot_train_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model()
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model()
        pred = logits[test_idx].argmax(dim=-1)
    return {
        "micro": _micro_f1(labels[test_idx], pred),
        "macro": _macro_f1(labels[test_idx], pred, args.num_classes),
        "best_epoch": int(args.fewshot_train_epochs),
        "best_val_loss": None,
        "best_val_macro": None,
    }


def _run_downstream_once(model, labels, split, args, run_seed: int, device: torch.device):
    if args.downstream_protocol == "mug_fixed_split":
        return _train_mug_fixed_split_once(model, labels, split, args, run_seed, device)
    if args.downstream_protocol == "mug_fewshot_from_train":
        return _train_mug_fewshot_once(model, labels, split, args, run_seed, device)
    return _train_once(model, labels, split, args, run_seed, device)


def _micro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float((y_true.view(-1) == y_pred.view(-1)).to(torch.float32).mean().item())


def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        yt = y_true == cls
        yp = y_pred == cls
        support = yt.sum()
        if int(support.item()) <= 0:
            continue
        tp = (yt & yp).sum().to(torch.float32)
        fp = ((~yt) & yp).sum().to(torch.float32)
        fn = (yt & (~yp)).sum().to(torch.float32)
        values.append(float((2 * tp / (2 * tp + fp + fn + 1e-12)).item()))
    return float(np.mean(values)) if values else 0.0


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser("MUG + unchanged TypeNeighborhoodEdge prompt benchmark")
    parser.add_argument("--dataset", default="acm", choices=["acm", "dblp", "aminer", "freebase"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mug_epochs", type=int, default=50)
    parser.add_argument("--mug_ratio", type=int, default=60)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--save_embedding_path", type=Path, default=None)
    parser.add_argument("--no_unified_feature", action="store_true")
    parser.add_argument(
        "--downstream_protocol",
        choices=["mug_fewshot_from_train", "mug_fixed_split", "hgep_true_fewshot"],
        default="mug_fewshot_from_train",
    )
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--split_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--type_hops", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--max_edges_per_metapath",
        type=int,
        default=0,
        help="0 keeps all MUG metapath edges. Positive values sample that many edges per metapath.",
    )
    parser.add_argument("--methods", nargs="+", default=["mug", "mug_type_neighborhood_edge"], choices=["mug", "mug_type_neighborhood_edge"])
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--eval_epochs", type=int, default=200)
    parser.add_argument("--fewshot_train_epochs", type=int, default=100)
    parser.add_argument("--eva_lr", type=float, default=None)
    parser.add_argument("--eva_wd", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_mode", choices=["loss", "metric"], default="loss")
    parser.add_argument("--relation_prompt_mode", choices=["mul", "add"], default="mul")
    parser.add_argument("--relation_prompt_alpha", type=float, default=0.5)
    parser.add_argument("--relation_prompt_dropout", type=float, default=0.1)
    parser.add_argument("--relation_prompt_aggr", choices=["mean", "sum"], default="mean")
    parser.add_argument("--relation_prompt_use_ln", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--relation_prompt_constraint",
        choices=["none", "identity_tanh", "positive_sigmoid", "identity_l2norm"],
        default="none",
    )
    parser.add_argument("--relation_prompt_constraint_scale", type=float, default=0.5)
    parser.add_argument("--peprompt_edge_prompt_hidden", type=int, default=128)
    parser.add_argument("--peprompt_edge_chunk_size", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "artifacts" / "results" / "mug_peprompt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    print(
        f"[Device] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} selected={device}",
        flush=True,
    )

    if args.embedding_path is not None:
        payload = _load_embedding_payload(args.embedding_path)
    else:
        payload = train_mug_embeddings(args, device)
        if args.save_embedding_path is not None:
            _save_embedding_payload(args.save_embedding_path, payload)

    embeds = payload["embeds"].float()
    labels = payload["labels"].long()
    eval_data = None
    if args.downstream_protocol in {"mug_fewshot_from_train", "mug_fixed_split"}:
        eval_data = load_mug_eval_split(args.dataset, args.mug_ratio)
        labels = eval_data["labels"].long()
    if args.eva_lr is None:
        args.eva_lr = float(payload["mug_args"].get("eva_lr", 0.01))
    if args.eva_wd is None:
        args.eva_wd = float(payload["mug_args"].get("eva_wd", 0.0))
    args.num_classes = int(labels.max().item()) + 1

    z_type = None
    edge_index_dict = {}
    edge_feature_dict = {}
    if "mug_type_neighborhood_edge" in args.methods:
        z_type = target_type_neighborhood(payload["nei_index"], payload["mug_args"]["type_num"], args.type_hops)
        edge_index_dict, edge_feature_dict = _make_metapath_edge_dicts(
            payload["mps"],
            z_type,
            max_edges_per_metapath=int(args.max_edges_per_metapath or 0),
            seed=args.seed,
            device=device,
        )

    if args.downstream_protocol == "hgep_true_fewshot":
        splits = make_fewshot_splits(labels, args.shot, args.split_seeds)
    else:
        assert eval_data is not None
        splits = [
            {
                "seed": int(seed),
                "train_pool": eval_data["train_pool"],
                "train": eval_data["train_pool"],
                "val": eval_data["val"],
                "test": eval_data["test"],
            }
            for seed in args.split_seeds
        ]
    rows = []
    for split in splits:
        for repeat in range(args.repeats):
            run_seed = int(args.seed) * 100000 + int(split["seed"]) * 1000 + repeat
            run_split = split
            if args.downstream_protocol == "mug_fewshot_from_train":
                run_split = dict(split)
                run_split["train"] = sample_fewshot_from_pool(
                    labels, split["train_pool"], args.shot, run_seed
                )
            if "mug" in args.methods:
                print(f"[Downstream] method=MUG+LogReg split={split['seed']} repeat={repeat}", flush=True)
                set_seed(run_seed)
                model = MUGNoPromptClassifier(
                    embeds.to(device),
                    args.num_classes,
                )
                row = _run_downstream_once(model, labels, run_split, args, run_seed, device)
                row.update({"method": "MUG+LogReg", "split_seed": int(split["seed"]), "repeat": repeat})
                rows.append(row)

            if "mug_type_neighborhood_edge" in args.methods:
                print(f"[Downstream] method=MUG+TypeNeighborhoodEdge+LogReg split={split['seed']} repeat={repeat}", flush=True)
                set_seed(run_seed)
                model = MUGTypeNeighborhoodEdgeClassifier(
                    embeds.to(device),
                    edge_index_dict,
                    edge_feature_dict,
                    args.num_classes,
                    args.relation_prompt_mode,
                    args.relation_prompt_alpha,
                    args.relation_prompt_dropout,
                    args.relation_prompt_aggr,
                    args.relation_prompt_use_ln,
                    args.peprompt_edge_prompt_hidden,
                    args.peprompt_edge_chunk_size,
                    args.relation_prompt_constraint,
                    args.relation_prompt_constraint_scale,
                )
                row = _run_downstream_once(model, labels, run_split, args, run_seed, device)
                row.update({"method": "MUG+TypeNeighborhoodEdge+LogReg", "split_seed": int(split["seed"]), "repeat": repeat})
                rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.downstream_protocol == "mug_fixed_split":
        stem = (
            f"{args.dataset}_{args.downstream_protocol}_r{args.mug_ratio}_mug{args.mug_epochs}_"
            f"s{'-'.join(map(str, args.split_seeds))}_r{args.repeats}"
        )
    else:
        stem = (
            f"{args.dataset}_{args.downstream_protocol}_{args.shot}shot_r{args.mug_ratio}_mug{args.mug_epochs}_"
            f"s{'-'.join(map(str, args.split_seeds))}_r{args.repeats}"
        )
    per_run_path = args.out_dir / f"{stem}.per_run.csv"
    with open(per_run_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "split_seed",
                "repeat",
                "micro",
                "macro",
                "best_epoch",
                "best_val_loss",
                "best_val_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    edge_counts = {"/".join(etype): int(edge_index.size(1)) for etype, edge_index in edge_index_dict.items()}
    summary = {
        "config": _jsonable(vars(args)),
        "device": str(device),
        "target_nodes": int(embeds.size(0)),
        "embedding_dim": int(embeds.size(1)),
        "type_hops": list(map(int, args.type_hops)),
        "type_neighborhood_dim": None if z_type is None else int(z_type.size(1)),
        "edge_feature_dim": None if not edge_feature_dict else int(next(iter(edge_feature_dict.values())).size(1)),
        "edge_counts": edge_counts,
        "summary": summarize(rows),
        "per_run": str(per_run_path),
    }
    summary_path = args.out_dir / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["summary"], indent=2), flush=True)
    print(f"[DONE] per_run={per_run_path}", flush=True)
    print(f"[DONE] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
