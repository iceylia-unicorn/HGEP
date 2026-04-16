from pathlib import Path
import sys
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import json
import random
import shutil

import numpy as np
import torch

from gpbench.utils.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_metrics,
    maybe_configure_wandb_env,
    upload_file_artifact,
)
from protocols.hgmp.run_legacy import pretrain


DATASET_NUM_CLASS = {
    "ACM": 3,
    "DBLP": 4,
    "IMDB": 5,
    "Freebase": 7,
}

LEGACY_PRETRAIN_DEFAULTS = {
    "feats_type": 0,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 2,
    "dropout": 0.5,
    "pretext": "GraphCL",
    "hgnn_type": "GCN",
    "num_samples": 100,
    "pre_lr": 1e-3,
    "aug_ration": 1e-3,
    "prompt_lr": 1e-3,
    "head_lr": 1e-3,
    "weight_decay": 5e-4,
    "patience": 7,
    "repeat": 1,
    "prompt_epoch": 300,
    "schedule_step": 300,
    "use_norm": False,
    "edge_feats": 64,
}

LEGACY_DATASET_PRETRAIN_DEFAULTS = {
    "ACM": {
        "hidden_dim": 512,
        "num_samples": 500,
        "prompt_lr": 5e-2,
        "head_lr": 5e-4,
        "weight_decay": 5e-5,
        "pre_lr": 1e-4,
        "aug_ration": 0.1,
    },
    "IMDB": {
        "hidden_dim": 256,
        "num_samples": 200,
        "prompt_lr": 5e-2,
        "head_lr": 5e-2,
        "weight_decay": 1e-4,
        "pre_lr": 1e-3,
        "aug_ration": 0.3,
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device(raw_device: Union[str, int]) -> torch.device:
    if isinstance(raw_device, int):
        return torch.device(f"cuda:{raw_device}") if torch.cuda.is_available() else torch.device("cpu")
    raw = str(raw_device).strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if raw.startswith("cuda:"):
        return torch.device(raw) if torch.cuda.is_available() else torch.device("cpu")
    try:
        idx = int(raw)
        return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:
        raise ValueError(f"Unsupported device spec: {raw_device}")


def collect_provided_options(argv: list[str]) -> set[str]:
    provided = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token[2:].split("=", 1)[0].replace("-", "_")
        provided.add(option)
    return provided


def _set_default_if_missing(args, provided_options: set[str], name: str, value):
    if name not in provided_options:
        setattr(args, name, value)


def apply_benchmark_defaults(args, provided_options: set[str] | None = None):
    provided_options = provided_options or set()
    for key, value in LEGACY_PRETRAIN_DEFAULTS.items():
        _set_default_if_missing(args, provided_options, key, value)

    for key, value in LEGACY_DATASET_PRETRAIN_DEFAULTS.get(args.dataset, {}).items():
        _set_default_if_missing(args, provided_options, key, value)

    if args.dataset in DATASET_NUM_CLASS:
        _set_default_if_missing(args, provided_options, "num_class", DATASET_NUM_CLASS[args.dataset])
    return args


def resolve_upstream_ckpt_path(args) -> Path:
    base = Path("artifacts/checkpoints/hgmp/pretrain")
    return base / f"{args.dataset}.{args.pretext}.{args.hgnn_type}.hid{args.hidden_dim}.np{args.num_samples}.pth"


def build_alias_path(args) -> Path:
    if args.ckpt_alias is not None:
        return Path(args.ckpt_alias)

    save_dir = Path(args.save_dir)
    filename = (
        f"{args.dataset}.{args.pretext}.{args.hgnn_type}.hid{args.hidden_dim}."
        f"np{args.num_samples}.seed{args.seed}.pth"
    )
    return save_dir / filename


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _build_run_config(args, upstream_ckpt: Path, alias_ckpt: Path):
    config = vars(args).copy()
    config.pop("wandb_key", None)
    config["upstream_ckpt"] = str(upstream_ckpt)
    config["alias_ckpt"] = str(alias_ckpt)
    return _json_ready(config)


def _make_pretrain_epoch_logger(wandb_run, args):
    if wandb_run is None:
        return None

    def _callback(metrics: dict):
        payload = {f"pretrain/{key}": value for key, value in metrics.items()}
        payload.update(
            {
                "pretrain/dataset": args.dataset,
                "pretrain/seed": args.seed,
                "pretrain/hidden_dim": args.hidden_dim,
                "pretrain/num_samples": args.num_samples,
            }
        )
        log_metrics(wandb_run, payload)

    return _callback


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)

    ap.add_argument("--feats_type", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--pretext", type=str, default="GraphCL")
    ap.add_argument("--hgnn_type", type=str, default="GCN")
    ap.add_argument("--num_class", type=int, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--classification_type", type=str, default="NIG")

    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--pre_lr", type=float, default=1e-3)
    ap.add_argument("--aug_ration", type=float, default=0.2)

    ap.add_argument("--prompt_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument("--edge_feats", type=int, default=None)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--prompt_epoch", type=int, default=1)
    ap.add_argument("--schedule_step", type=int, default=300)
    ap.add_argument("--use_norm", action="store_true")

    ap.add_argument("--benchmark_defaults", action="store_true")
    ap.add_argument("--save_dir", type=str, default="artifacts/checkpoints/hgmp/pretrain")
    ap.add_argument("--ckpt_alias", type=str, default=None)
    ap.add_argument("--print_ckpt_only", action="store_true")

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="HGEP")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_job_type", type=str, default="hgmp_pretrain")
    ap.add_argument("--wandb_tags", nargs="*", default=[])
    ap.add_argument("--wandb_notes", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    ap.add_argument("--wandb_dir", type=Path, default=ROOT / "artifacts" / "wandb")
    ap.add_argument("--wandb_key", type=str, default=None)
    ap.add_argument("--wandb_key_file", type=Path, default=ROOT / ".codex")

    provided_options = collect_provided_options(sys.argv[1:])
    args = ap.parse_args()
    wandb_run = None
    exit_code = 0

    try:
        if args.benchmark_defaults:
            args = apply_benchmark_defaults(args, provided_options)

        if args.num_class is None:
            args.num_class = DATASET_NUM_CLASS.get(args.dataset, 3)

        args.pre_epoch = args.epochs
        args.edge_feats = args.hidden_dim if args.edge_feats is None else args.edge_feats
        args.device = normalize_device(args.device)

        set_seed(args.seed)

        upstream_ckpt = resolve_upstream_ckpt_path(args)
        alias_ckpt = build_alias_path(args)
        config = _build_run_config(args, upstream_ckpt, alias_ckpt)

        if args.use_wandb:
            maybe_configure_wandb_env(api_key=args.wandb_key, env_file=args.wandb_key_file)
            wandb_run = init_wandb_run(
                enabled=True,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or f"hgmp-pretrain-{args.dataset}-{args.hgnn_type}-seed{args.seed}",
                group=args.wandb_group or f"hgmp-pretrain-{args.dataset}",
                job_type=args.wandb_job_type,
                tags=args.wandb_tags or [args.dataset, "hgmp", "pretrain", args.hgnn_type],
                notes=args.wandb_notes,
                mode=args.wandb_mode,
                dir_path=args.wandb_dir,
                config=config,
            )

        print(
            f"[INFO] HGMP pretrain | dataset={args.dataset} | device={args.device} | seed={args.seed} | "
            f"pretext={args.pretext} | hgnn_type={args.hgnn_type} | hidden_dim={args.hidden_dim} | "
            f"num_samples={args.num_samples}"
        )
        print(f"[INFO] upstream best ckpt path: {upstream_ckpt}")
        print(f"[INFO] benchmark alias ckpt path: {alias_ckpt}")

        if args.print_ckpt_only:
            return

        pretrain(args, epoch_callback=_make_pretrain_epoch_logger(wandb_run, args))

        if not upstream_ckpt.exists():
            raise FileNotFoundError(f"Expected upstream checkpoint not found: {upstream_ckpt}")

        alias_ckpt.parent.mkdir(parents=True, exist_ok=True)
        if upstream_ckpt.resolve() != alias_ckpt.resolve():
            shutil.copy2(upstream_ckpt, alias_ckpt)

        meta = {
            "dataset": args.dataset,
            "device": str(args.device),
            "seed": args.seed,
            "epochs": args.epochs,
            "feats_type": args.feats_type,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "pretext": args.pretext,
            "hgnn_type": args.hgnn_type,
            "num_class": args.num_class,
            "num_samples": args.num_samples,
            "pre_lr": args.pre_lr,
            "aug_ration": args.aug_ration,
            "upstream_ckpt": str(upstream_ckpt),
            "alias_ckpt": str(alias_ckpt),
        }
        meta_path = alias_ckpt.with_suffix(alias_ckpt.suffix + ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        log_metrics(
            wandb_run,
            {
                "pretrain/upstream_ckpt_exists": upstream_ckpt.exists(),
                "pretrain/alias_ckpt_exists": alias_ckpt.exists(),
            },
        )
        upload_file_artifact(wandb_run, alias_ckpt, name=f"hgmp-{args.dataset}-{args.hgnn_type}-seed{args.seed}", artifact_type="checkpoint")
        upload_file_artifact(wandb_run, meta_path, name=f"hgmp-{args.dataset}-{args.hgnn_type}-seed{args.seed}-metadata", artifact_type="metadata")

        print(f"[DONE] benchmark-ready ckpt copied to: {alias_ckpt}")
        print(f"[DONE] run metadata saved to: {meta_path}")
    except Exception:
        exit_code = 1
        raise
    finally:
        finish_wandb_run(wandb_run, exit_code=exit_code)


if __name__ == "__main__":
    main()
