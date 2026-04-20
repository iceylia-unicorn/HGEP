# scripts/typepair_edge_feature_sweep.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from gpbench.protocol_bridge import downstream_legacy as legacy_bridge
from gpbench.protocol_bridge.downstream_legacy import train_typepair_prompt_probe
from gpbench.utils.wandb_utils import maybe_configure_wandb_env
from scripts import protocol_benchmark_v2 as benchmark


FEATURES = benchmark.TYPEPAIR_EDGE_FEATURES


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install it with `pip install wandb`.") from exc
    return wandb


def _bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_parser():
    ap = benchmark.build_parser()
    ap.set_defaults(
        methods=["typepair"],
        repeats=1,
        use_wandb=True,
        enable_typepair_edge_features=True,
        typepair_write_edge_feature_stats=False,
    )

    ap.add_argument("--create_sweep", action="store_true")
    ap.add_argument("--run_agent", action="store_true")
    ap.add_argument("--sweep_id", type=str, default=None)
    ap.add_argument("--sweep_count", type=int, default=80)
    ap.add_argument("--sweep_metric", type=str, default="test_macro", choices=["test_micro", "test_macro"])
    ap.add_argument("--upload_summary_artifact", action="store_true")

    for name in FEATURES:
        ap.add_argument(f"--use_{name}", type=_bool_arg, default=True)
    return ap


def build_sweep_config(args) -> dict:
    return {
        "name": args.wandb_name or f"{args.dataset}-{args.shot}shot-typepair-edge-feature-subsets",
        "method": "random",
        "metric": {"name": args.sweep_metric, "goal": "maximize"},
        "parameters": {
            f"use_{name}": {"values": [True, False]}
            for name in FEATURES
        },
    }


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


def _fixed_run_config(args) -> dict:
    config = vars(args).copy()
    for key in ("wandb_key", "create_sweep", "run_agent", "sweep_id", "sweep_count"):
        config.pop(key, None)
    for name in FEATURES:
        config.pop(f"use_{name}", None)
    return _json_ready(config)


def _make_run_args(base_args, wandb_config) -> tuple[SimpleNamespace, list[str], dict[str, bool]]:
    run_args = SimpleNamespace(**vars(base_args))
    run_args.method = "typepair"
    run_args.methods = ["typepair"]
    run_args.enable_typepair_edge_features = True
    run_args.typepair_write_edge_feature_stats = False
    run_args.device = torch.device(base_args.device)

    split_seed = int(base_args.seeds[0] if base_args.seeds else 0)
    repeat_id = 0
    run_args.split_seed = split_seed
    run_args.seed = benchmark._make_run_seed(split_seed, repeat_id, base_args.run_seed_base)
    run_args.ckpt = benchmark._resolve_ckpt(run_args, "typepair")

    feature_flags = {}
    selected = []
    for name in FEATURES:
        key = f"use_{name}"
        value = _bool_arg(wandb_config.get(key, getattr(base_args, key, True)))
        feature_flags[key] = value
        setattr(run_args, key, value)
        if value:
            selected.append(name)

    run_args.typepair_edge_feature_names = selected
    return run_args, selected, feature_flags


def _write_summary(args, payload: dict) -> Path:
    out_dir = (
        Path(args.save_dir)
        / "typepair_edge_feature_sweep"
        / args.dataset
        / f"{args.shot}-shot"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_one_sweep_trial(base_args):
    wandb = _import_wandb()
    run = wandb.init(
        project=base_args.wandb_project,
        entity=base_args.wandb_entity,
        group=base_args.wandb_group or f"{base_args.dataset}-{base_args.shot}shot-typepair-edge-features",
        job_type="typepair_edge_feature_sweep",
        tags=base_args.wandb_tags or [base_args.dataset, f"{base_args.shot}-shot", "typepair-edge-feature-sweep"],
        notes=base_args.wandb_notes,
        mode=base_args.wandb_mode,
        dir=str(base_args.wandb_dir),
        config=_fixed_run_config(base_args),
    )

    try:
        run_args, selected, feature_flags = _make_run_args(base_args, run.config)
        benchmark._set_global_seed(run_args.seed)
        edge_payload = benchmark.prepare_typepair_edge_feature_table(run_args, wandb_run=None)

        orig_loader = legacy_bridge._load_legacy_fewshot_splits
        legacy_bridge._load_legacy_fewshot_splits = benchmark._patched_legacy_split_loader
        try:
            res = train_typepair_prompt_probe(
                args=run_args,
                batch_size=run_args.embed_batch_size,
                hidden_dim=run_args.head_hidden,
                dropout=run_args.head_dropout,
                head_lr=run_args.lr,
                prompt_lr=run_args.prompt_lr,
                weight_decay=run_args.weight_decay,
                epochs=run_args.epochs,
                patience=run_args.patience,
                early_stop_metric=run_args.early_stop_metric,
                save_best_path=None,
                epoch_callback=None,
            )
        finally:
            legacy_bridge._load_legacy_fewshot_splits = orig_loader

        metrics = {
            "test_micro": float(res["test_at_best_micro"]),
            "test_macro": float(res["test_at_best_macro"]),
            "best_epoch": int(res["best_epoch"]),
            "selected_feature_count": int(len(selected)),
            "selected_features": ",".join(selected),
            "edge_feature_dim": int(edge_payload["feature_dim"] if edge_payload is not None else 0),
            "split_seed": int(run_args.split_seed),
            "run_seed": int(run_args.seed),
        }
        metrics.update(feature_flags)
        run.log(metrics)
        run.summary.update(metrics)

        summary = {
            "dataset": run_args.dataset,
            "shot": run_args.shot,
            "ckpt": run_args.ckpt,
            "metrics": metrics,
        }
        summary_path = _write_summary(base_args, summary)
        if base_args.upload_summary_artifact:
            artifact = wandb.Artifact(
                name=f"typepair-edge-feature-sweep-{run.id}",
                type="sweep-summary",
            )
            artifact.add_file(str(summary_path))
            run.log_artifact(artifact)
    finally:
        run.finish()


def main():
    args = build_parser().parse_args()
    if (args.create_sweep or args.sweep_id) and not 50 <= args.sweep_count <= 100:
        raise ValueError("--sweep_count must be between 50 and 100 for edge feature subset search.")
    maybe_configure_wandb_env(api_key=args.wandb_key, env_file=args.wandb_key_file)
    args.wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb = _import_wandb()
    if args.create_sweep:
        sweep_id = wandb.sweep(
            build_sweep_config(args),
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print(json.dumps({"sweep_id": sweep_id, "count": args.sweep_count}, indent=2))
        if not args.run_agent:
            return
        args.sweep_id = sweep_id

    if args.sweep_id:
        wandb.agent(args.sweep_id, function=lambda: run_one_sweep_trial(args), count=args.sweep_count)
    else:
        run_one_sweep_trial(args)


if __name__ == "__main__":
    main()
