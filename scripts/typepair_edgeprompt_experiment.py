from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "protocol_benchmark_v2.py"
DEFAULT_RESULTS_ROOT = ROOT / "artifacts" / "results" / "edgeprompt2_experiments"
DEFAULT_SWEEP_DIR = ROOT / "artifacts" / "results" / "protocol_benchmark" / "typepair_edge_feature_sweep"
DEFAULT_BASELINE_SUMMARY = ROOT / "artifacts" / "results" / "protocol_benchmark" / "ACM" / "10-shot" / "overall_summary.json"


CONFIG_SPECS = [
    ("config_00", "no_edge", []),
    ("config_01", "node_type_encoding", ["NodeTypeEncoding"]),
    ("config_02", "edge_type_onehot", ["EdgeTypeOneHot"]),
    ("config_03", "neighbor_type_overlap", ["NeighborTypeOverlap"]),
    ("config_04", "degree_diff", ["DegreeDiff"]),
    ("config_05", "spectral_embedding_diff", ["SpectralEmbeddingDiff"]),
    ("config_06", "type_memory_only", ["NodeTypeEncoding", "EdgeTypeOneHot"]),
    ("config_07", "spectral_plus_neighbor_type_overlap", ["SpectralEmbeddingDiff", "NeighborTypeOverlap"]),
    ("config_08", "structural_residual_only", ["SpectralEmbeddingDiff", "NeighborTypeOverlap", "DegreeDiff"]),
    ("config_09", "degree_edge_type_neighbor_overlap", ["DegreeDiff", "EdgeTypeOneHot", "NeighborTypeOverlap"]),
    ("config_10", "edgeprompt2_core", ["NodeTypeEncoding", "EdgeTypeOneHot", "NeighborTypeOverlap", "SpectralEmbeddingDiff"]),
    ("config_11", "community_neighbor_overlap_spectral", ["CommunityLabelDiff", "NeighborTypeOverlap", "SpectralEmbeddingDiff"]),
    (
        "config_12",
        "all_features_control",
        [
            "NodeTypeEncoding",
            "EdgeTypeOneHot",
            "DegreeDiff",
            "NeighborTypeOverlap",
            "NodeAttrCosSim",
            "NeighborAttrVariance",
            "PageRankDiff",
            "CommunityLabelDiff",
            "SpectralEmbeddingDiff",
        ],
    ),
]

CONFIG_MAP = {config_id: {"config_id": config_id, "label": label, "features": features} for config_id, label, features in CONFIG_SPECS}
SINGLE_FEATURE_CONFIG_IDS = ["config_01", "config_02", "config_03", "config_04", "config_05"]


@dataclass
class ResultSummary:
    stage: str
    slot: str
    config_id: str
    label: str
    features: list[str]
    pooled_count: int
    pooled_micro_mean: float
    pooled_micro_std: float
    pooled_macro_mean: float
    pooled_macro_std: float
    seed_count: int
    seed_micro_mean: float
    seed_micro_std: float
    seed_macro_mean: float
    seed_macro_std: float
    output_dir: str
    command: str
    delta_seed_macro_vs_no_edge: float | None = None
    delta_pooled_macro_vs_no_edge: float | None = None
    delta_seed_micro_vs_no_edge: float | None = None
    delta_pooled_micro_vs_no_edge: float | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_base_args(args) -> list[str]:
    base = [
        "--dataset",
        args.dataset,
        "--root",
        args.root,
        "--splits",
        args.splits,
        "--shot",
        str(args.shot),
        "--methods",
        "typepair",
        "--pretrain_seed",
        str(args.pretrain_seed),
        "--device",
        args.device,
        "--hgnn_type",
        args.hgnn_type,
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_heads",
        str(args.num_heads),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--num_samples",
        str(args.num_samples),
        "--num_class",
        str(args.num_class),
        "--classification_type",
        args.classification_type,
        "--embed_batch_size",
        str(args.embed_batch_size),
        "--head_hidden",
        str(args.head_hidden),
        "--head_dropout",
        str(args.head_dropout),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--early_stop_metric",
        args.early_stop_metric,
        "--hgmp_ckpt",
        args.hgmp_ckpt,
        "--typepair_ckpt",
        args.typepair_ckpt,
        "--hgprompt_ckpt",
        args.hgprompt_ckpt,
        "--typepair_spectral_dim",
        str(args.typepair_spectral_dim),
        "--typepair_edge_prompt_hidden",
        str(args.typepair_edge_prompt_hidden),
        "--typepair_edge_prompt_alpha",
        str(args.typepair_edge_prompt_alpha),
        "--typepair_edge_prompt_fusion",
        args.typepair_edge_prompt_fusion,
        "--wandb_mode",
        args.wandb_mode,
    ]
    if args.prompt_lr is not None:
        base.extend(["--prompt_lr", str(args.prompt_lr)])
    if args.use_wandb:
        base.append("--use_wandb")
    if args.wandb_project:
        base.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity:
        base.extend(["--wandb_entity", args.wandb_entity])
    return base


def _stage_dir(exp_root: Path, stage: str, config_id: str) -> Path:
    return exp_root / stage / config_id


def _benchmark_output_dir(save_dir: Path, dataset: str, shot: int) -> Path:
    return save_dir / dataset / f"{shot}-shot"


def _command_string(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _build_run_entry(
    args,
    exp_root: Path,
    stage: str,
    slot: str,
    config_id: str,
    repeats: int,
    seeds: list[int],
    fusion: str | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    spec = CONFIG_MAP[config_id]
    save_dir = _stage_dir(exp_root, stage, slot)
    bench_out = _benchmark_output_dir(save_dir, args.dataset, args.shot)
    cmd = [
        args.python_exec,
        str(BENCHMARK),
        *_build_base_args(args),
        "--save_dir",
        str(save_dir),
        "--repeats",
        str(repeats),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--wandb_group",
        f"{args.experiment_name}-{stage}",
        "--wandb_job_type",
        f"typepair_edgeprompt_{stage}",
        "--wandb_name",
        f"{args.experiment_name}-{slot}",
    ]
    if spec["features"]:
        cmd.append("--enable_typepair_edge_features")
        cmd.extend(["--typepair_edge_feature_names", *spec["features"]])
    if fusion is not None:
        cmd.extend(["--typepair_edge_prompt_fusion", fusion])
    if alpha is not None:
        cmd.extend(["--typepair_edge_prompt_alpha", str(alpha)])
    command = _command_string(cmd)
    if args.cuda_visible_devices:
        command = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} {command}"
    return {
        "stage": stage,
        "slot": slot,
        "config_id": config_id,
        "label": spec["label"],
        "features": list(spec["features"]),
        "repeats": repeats,
        "seeds": list(seeds),
        "fusion": fusion,
        "alpha": alpha,
        "env_overrides": {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices} if args.cuda_visible_devices else {},
        "save_dir": str(save_dir),
        "benchmark_output_dir": str(bench_out),
        "command": command,
        "cmd": cmd,
    }


def _scan_existing_results(args) -> dict[str, Any]:
    history = {
        "sweep_dir": str(DEFAULT_SWEEP_DIR / args.dataset / f"{args.shot}-shot"),
        "baseline_summary": str(DEFAULT_BASELINE_SUMMARY),
        "wandb_csv_found": False,
        "wandb_csv_paths": [],
    }

    csv_paths = sorted(str(path) for path in (ROOT / "artifacts").rglob("*.csv"))
    wandb_csv_paths = [path for path in csv_paths if "wandb" in path.lower()]
    history["wandb_csv_paths"] = wandb_csv_paths
    history["wandb_csv_found"] = bool(wandb_csv_paths)

    if DEFAULT_BASELINE_SUMMARY.exists():
        base = _read_json(DEFAULT_BASELINE_SUMMARY)
        history["current_protocol_baseline"] = base

    sweep_paths = sorted((DEFAULT_SWEEP_DIR / args.dataset / f"{args.shot}-shot").glob("run_*.json"))
    rows = []
    feature_counter = Counter()
    for path in sweep_paths:
        payload = _read_json(path)
        metrics = payload.get("metrics", {})
        feats = [item for item in metrics.get("selected_features", "").split(",") if item]
        feature_counter.update(feats)
        rows.append(
            {
                "path": str(path),
                "macro": float(metrics.get("test_macro", 0.0)),
                "micro": float(metrics.get("test_micro", 0.0)),
                "best_epoch": int(metrics.get("best_epoch", 0)),
                "selected_feature_count": int(metrics.get("selected_feature_count", 0)),
                "selected_features": feats,
            }
        )
    rows.sort(key=lambda item: (item["macro"], item["micro"]), reverse=True)
    history["historical_sweep"] = {
        "count": len(rows),
        "top10": rows[:10],
        "single_feature_top": [item for item in rows if item["selected_feature_count"] == 1][:10],
        "feature_frequency": dict(feature_counter.most_common()),
        "has_no_edge_baseline": any(item["selected_feature_count"] == 0 for item in rows),
    }
    return history


def _planned_manifest(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    stage_a = [
        _build_run_entry(
            args=args,
            exp_root=exp_root,
            stage="stage_a",
            slot=config_id,
            config_id=config_id,
            repeats=1,
            seeds=list(args.stage_a_seeds),
        )
        for config_id, _, _ in CONFIG_SPECS
    ]
    return {
        "experiment_name": args.experiment_name,
        "created_at": _now_iso(),
        "results_root": str(exp_root),
        "python_exec": args.python_exec,
        "benchmark_script": str(BENCHMARK),
        "base_args": {
            "dataset": args.dataset,
            "root": args.root,
            "splits": args.splits,
            "shot": args.shot,
            "pretrain_seed": args.pretrain_seed,
            "device": args.device,
            "cuda_visible_devices": args.cuda_visible_devices,
            "methods": ["typepair"],
            "hgnn_type": args.hgnn_type,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "num_samples": args.num_samples,
            "num_class": args.num_class,
            "classification_type": args.classification_type,
            "embed_batch_size": args.embed_batch_size,
            "head_hidden": args.head_hidden,
            "head_dropout": args.head_dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "prompt_lr": args.prompt_lr,
            "weight_decay": args.weight_decay,
            "early_stop_metric": args.early_stop_metric,
            "hgmp_ckpt": args.hgmp_ckpt,
            "typepair_ckpt": args.typepair_ckpt,
            "hgprompt_ckpt": args.hgprompt_ckpt,
            "typepair_spectral_dim": args.typepair_spectral_dim,
            "typepair_edge_prompt_hidden": args.typepair_edge_prompt_hidden,
            "typepair_edge_prompt_alpha": args.typepair_edge_prompt_alpha,
            "typepair_edge_prompt_fusion": args.typepair_edge_prompt_fusion,
            "use_wandb": args.use_wandb,
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
        },
        "notes": {
            "cli_parameter_truth_source": str(BENCHMARK),
            "wandb_csv_note": "No local wandb export CSV was found during preparation; existing results were reconstructed from sweep JSON summaries and local benchmark outputs.",
            "parameter_mismatch_note": "The code uses --pretrain_seed and --device accepts strings such as cuda:1. If prompt assumptions differ, code truth wins.",
        },
        "existing_results_scan": _scan_existing_results(args),
        "stage_a": {
            "objective": "Controlled multi-seed ablation screening",
            "repeats": 1,
            "seeds": list(args.stage_a_seeds),
            "runs": stage_a,
        },
        "stage_b_selection_rule": {
            "fixed": ["config_00", "config_06", "config_08", "config_10"],
            "dynamic": "best single-feature config from stage_a among config_01..config_05 ranked by seed_macro_mean desc",
            "repeats": 5,
            "seeds": list(args.stage_b_seeds),
        },
        "fusion_selection_rule": {
            "condition": "Run only if stage_b supports combined scheme.",
            "base_config": "config_10",
            "features": CONFIG_MAP["config_10"]["features"],
            "fusions": ["add", "mul", "gate"],
            "alphas": [0.2, 0.5, 0.8],
            "repeats": 3,
            "seeds": list(args.fusion_seeds),
        },
    }


def prepare_manifest(args) -> Path:
    manifest = _planned_manifest(args)
    exp_root = args.results_root / args.experiment_name
    exp_root.mkdir(parents=True, exist_ok=True)
    manifest_path = exp_root / "experiment_manifest.initial.json"
    _json_dump(manifest_path, manifest)

    stage_a_rows = []
    for run in manifest["stage_a"]["runs"]:
        stage_a_rows.append(
            {
                "stage": run["stage"],
                "slot": run["slot"],
                "config_id": run["config_id"],
                "label": run["label"],
                "features": ",".join(run["features"]),
                "repeats": run["repeats"],
                "seeds": ",".join(str(seed) for seed in run["seeds"]),
                "fusion": run["fusion"] or "",
                "alpha": "" if run["alpha"] is None else run["alpha"],
                "save_dir": run["save_dir"],
                "benchmark_output_dir": run["benchmark_output_dir"],
                "command": run["command"],
            }
        )
    _write_csv(exp_root / "experiment_manifest.stage_a.csv", stage_a_rows)
    return manifest_path


def _append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_entry(entry: dict[str, Any], exp_root: Path, force: bool):
    benchmark_out = Path(entry["benchmark_output_dir"])
    summary_path = benchmark_out / "overall_summary.json"
    if summary_path.exists() and not force:
        _append_jsonl(
            exp_root / "command_log.jsonl",
            {"timestamp": _now_iso(), "status": "skip_existing", "slot": entry["slot"], "command": entry["command"]},
        )
        return

    log_path = Path(entry["save_dir"]) / "run.log"
    Path(entry["save_dir"]).mkdir(parents=True, exist_ok=True)
    _append_jsonl(
        exp_root / "command_log.jsonl",
        {"timestamp": _now_iso(), "status": "start", "slot": entry["slot"], "command": entry["command"]},
    )
    started = time.time()
    env = os.environ.copy()
    env.update(entry.get("env_overrides", {}))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            entry["cmd"],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    finished = time.time()
    payload = {
        "timestamp": _now_iso(),
        "status": "ok" if process.returncode == 0 else "failed",
        "slot": entry["slot"],
        "command": entry["command"],
        "returncode": process.returncode,
        "seconds": round(finished - started, 3),
        "log_path": str(log_path),
    }
    _append_jsonl(exp_root / "command_log.jsonl", payload)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed for {entry['slot']}: {entry['command']}")


def run_stage_a(args):
    exp_root = args.results_root / args.experiment_name
    manifest = _read_json(exp_root / "experiment_manifest.initial.json")
    for entry in manifest["stage_a"]["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)


def _collect_summary(entry: dict[str, Any]) -> ResultSummary:
    benchmark_out = Path(entry["benchmark_output_dir"])
    summary = _read_json(benchmark_out / "overall_summary.json")
    pooled = summary["pooled_runs"]["typepair"]
    seed = summary["seed_mean_then_std"]["typepair"]
    return ResultSummary(
        stage=entry["stage"],
        slot=entry["slot"],
        config_id=entry["config_id"],
        label=entry["label"],
        features=list(entry["features"]),
        pooled_count=int(pooled["count"]),
        pooled_micro_mean=float(pooled["micro_mean"]),
        pooled_micro_std=float(pooled["micro_std"]),
        pooled_macro_mean=float(pooled["macro_mean"]),
        pooled_macro_std=float(pooled["macro_std"]),
        seed_count=int(seed["count"]),
        seed_micro_mean=float(seed["micro_mean"]),
        seed_micro_std=float(seed["micro_std"]),
        seed_macro_mean=float(seed["macro_mean"]),
        seed_macro_std=float(seed["macro_std"]),
        output_dir=entry["benchmark_output_dir"],
        command=entry["command"],
    )


def _sort_results(rows: list[ResultSummary]) -> list[ResultSummary]:
    return sorted(rows, key=lambda item: (item.seed_macro_mean, -item.seed_macro_std, item.seed_micro_mean), reverse=True)


def _add_deltas(rows: list[ResultSummary]):
    base = None
    for row in rows:
        if row.config_id == "config_00":
            base = row
            break
    if base is None:
        return
    for row in rows:
        row.delta_seed_macro_vs_no_edge = row.seed_macro_mean - base.seed_macro_mean
        row.delta_pooled_macro_vs_no_edge = row.pooled_macro_mean - base.pooled_macro_mean
        row.delta_seed_micro_vs_no_edge = row.seed_micro_mean - base.seed_micro_mean
        row.delta_pooled_micro_vs_no_edge = row.pooled_micro_mean - base.pooled_micro_mean


def _rows_to_dicts(rows: list[ResultSummary]) -> list[dict[str, Any]]:
    return [asdict(row) | {"features": ",".join(row.features)} for row in rows]


def summarize_stage_a(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    manifest = _read_json(exp_root / "experiment_manifest.initial.json")
    rows = [_collect_summary(entry) for entry in manifest["stage_a"]["runs"]]
    _add_deltas(rows)
    ranked = _sort_results(rows)
    _write_csv(exp_root / "stage_a_summary.csv", _rows_to_dicts(ranked))

    comparisons = {}
    lookup = {row.config_id: row for row in ranked}
    for left, right, key in [
        ("config_06", "config_00", "type_memory_only_vs_no_edge"),
        ("config_08", "config_00", "structural_residual_only_vs_no_edge"),
        ("config_10", "config_06", "edgeprompt2_core_vs_type_memory_only"),
        ("config_10", "config_08", "edgeprompt2_core_vs_structural_residual_only"),
    ]:
        comparisons[key] = {
            "left": left,
            "right": right,
            "seed_macro_delta": lookup[left].seed_macro_mean - lookup[right].seed_macro_mean,
            "seed_micro_delta": lookup[left].seed_micro_mean - lookup[right].seed_micro_mean,
            "pooled_macro_delta": lookup[left].pooled_macro_mean - lookup[right].pooled_macro_mean,
            "pooled_micro_delta": lookup[left].pooled_micro_mean - lookup[right].pooled_micro_mean,
        }

    best_single = max((lookup[config_id] for config_id in SINGLE_FEATURE_CONFIG_IDS), key=lambda item: item.seed_macro_mean)
    payload = {
        "ranked_config_ids": [row.config_id for row in ranked],
        "best_single_feature_config_id": best_single.config_id,
        "best_single_feature_label": best_single.label,
        "comparisons": comparisons,
    }
    _json_dump(exp_root / "stage_a_analysis.json", payload)
    return payload


def resolve_stage_b_manifest(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    stage_a_info = summarize_stage_a(args)
    best_single = stage_a_info["best_single_feature_config_id"]
    slots = [
        ("baseline_no_edge", "config_00"),
        ("best_single_feature", best_single),
        ("type_memory_only", "config_06"),
        ("structural_residual_only", "config_08"),
        ("edgeprompt2_core", "config_10"),
    ]
    runs = [
        _build_run_entry(
            args=args,
            exp_root=exp_root,
            stage="stage_b",
            slot=slot,
            config_id=config_id,
            repeats=5,
            seeds=list(args.stage_b_seeds),
        )
        for slot, config_id in slots
    ]
    payload = {
        "created_at": _now_iso(),
        "best_single_feature_config_id": best_single,
        "runs": runs,
    }
    _json_dump(exp_root / "experiment_manifest.stage_b.json", payload)
    _write_csv(
        exp_root / "experiment_manifest.stage_b.csv",
        [
            {
                "stage": entry["stage"],
                "slot": entry["slot"],
                "config_id": entry["config_id"],
                "label": entry["label"],
                "features": ",".join(entry["features"]),
                "repeats": entry["repeats"],
                "seeds": ",".join(str(seed) for seed in entry["seeds"]),
                "save_dir": entry["save_dir"],
                "benchmark_output_dir": entry["benchmark_output_dir"],
                "command": entry["command"],
            }
            for entry in runs
        ],
    )
    return exp_root / "experiment_manifest.stage_b.json"


def run_stage_b(args):
    exp_root = args.results_root / args.experiment_name
    manifest_path = exp_root / "experiment_manifest.stage_b.json"
    if not manifest_path.exists():
        resolve_stage_b_manifest(args)
    manifest = _read_json(manifest_path)
    for entry in manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)


def summarize_stage_b(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    manifest = _read_json(exp_root / "experiment_manifest.stage_b.json")
    rows = [_collect_summary(entry) for entry in manifest["runs"]]
    _add_deltas(rows)
    ranked = _sort_results(rows)
    _write_csv(exp_root / "stage_b_summary.csv", _rows_to_dicts(ranked))
    lookup = {row.config_id: row for row in ranked}
    slot_lookup = {row.slot: row for row in ranked}

    edge = slot_lookup["edgeprompt2_core"]
    type_memory = slot_lookup["type_memory_only"]
    structural = slot_lookup["structural_residual_only"]
    no_edge = slot_lookup["baseline_no_edge"]

    support_combined = (
        sum(1 for row in ranked if row.seed_macro_mean > edge.seed_macro_mean) <= 1
        and edge.seed_macro_mean > no_edge.seed_macro_mean
        and edge.seed_macro_mean > type_memory.seed_macro_mean
        and edge.seed_macro_mean > structural.seed_macro_mean
    )
    payload = {
        "ranked_slots": [row.slot for row in ranked],
        "ranked_config_ids": [row.config_id for row in ranked],
        "support_combined": support_combined,
        "comparisons": {
            "edgeprompt2_core_vs_no_edge": edge.seed_macro_mean - no_edge.seed_macro_mean,
            "edgeprompt2_core_vs_type_memory_only": edge.seed_macro_mean - type_memory.seed_macro_mean,
            "edgeprompt2_core_vs_structural_residual_only": edge.seed_macro_mean - structural.seed_macro_mean,
        },
        "best_single_feature_config_id": manifest["best_single_feature_config_id"],
        "config_lookup": {row.config_id: asdict(row) for row in ranked},
        "slot_lookup": {row.slot: asdict(row) for row in ranked},
    }
    _json_dump(exp_root / "stage_b_analysis.json", payload)
    return payload


def resolve_fusion_manifest(args) -> Path | None:
    exp_root = args.results_root / args.experiment_name
    analysis_path = exp_root / "stage_b_analysis.json"
    if not analysis_path.exists():
        summarize_stage_b(args)
    analysis = _read_json(analysis_path)
    if not analysis["support_combined"]:
        return None
    runs = []
    for fusion in ("add", "mul", "gate"):
        for alpha in (0.2, 0.5, 0.8):
            slot = f"edgeprompt2_core_{fusion}_alpha{str(alpha).replace('.', 'p')}"
            runs.append(
                _build_run_entry(
                    args=args,
                    exp_root=exp_root,
                    stage="fusion",
                    slot=slot,
                    config_id="config_10",
                    repeats=3,
                    seeds=list(args.fusion_seeds),
                    fusion=fusion,
                    alpha=alpha,
                )
            )
    payload = {"created_at": _now_iso(), "runs": runs}
    manifest_path = exp_root / "experiment_manifest.fusion.json"
    _json_dump(manifest_path, payload)
    _write_csv(
        exp_root / "experiment_manifest.fusion.csv",
        [
            {
                "stage": entry["stage"],
                "slot": entry["slot"],
                "config_id": entry["config_id"],
                "label": entry["label"],
                "features": ",".join(entry["features"]),
                "repeats": entry["repeats"],
                "seeds": ",".join(str(seed) for seed in entry["seeds"]),
                "fusion": entry["fusion"],
                "alpha": entry["alpha"],
                "save_dir": entry["save_dir"],
                "benchmark_output_dir": entry["benchmark_output_dir"],
                "command": entry["command"],
            }
            for entry in runs
        ],
    )
    return manifest_path


def run_fusion(args):
    exp_root = args.results_root / args.experiment_name
    manifest_path = exp_root / "experiment_manifest.fusion.json"
    if not manifest_path.exists():
        manifest_path = resolve_fusion_manifest(args)
    if manifest_path is None:
        return
    manifest = _read_json(manifest_path)
    for entry in manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)


def summarize_fusion(args) -> dict[str, Any] | None:
    exp_root = args.results_root / args.experiment_name
    manifest_path = exp_root / "experiment_manifest.fusion.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    rows = [_collect_summary(entry) for entry in manifest["runs"]]
    ranked = _sort_results(rows)
    _write_csv(exp_root / "fusion_summary.csv", _rows_to_dicts(ranked))
    payload = {"ranked_slots": [row.slot for row in ranked], "best": asdict(ranked[0]) if ranked else None}
    _json_dump(exp_root / "fusion_analysis.json", payload)
    return payload


def finalize_report(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    report = {
        "manifest_initial": str(exp_root / "experiment_manifest.initial.json"),
        "manifest_stage_b": str(exp_root / "experiment_manifest.stage_b.json"),
        "manifest_fusion": str(exp_root / "experiment_manifest.fusion.json"),
        "stage_a_summary": str(exp_root / "stage_a_summary.csv"),
        "stage_b_summary": str(exp_root / "stage_b_summary.csv"),
        "fusion_summary": str(exp_root / "fusion_summary.csv"),
        "stage_a_analysis": str(exp_root / "stage_a_analysis.json"),
        "stage_b_analysis": str(exp_root / "stage_b_analysis.json"),
        "fusion_analysis": str(exp_root / "fusion_analysis.json"),
        "command_log": str(exp_root / "command_log.jsonl"),
    }
    path = exp_root / "report_index.json"
    _json_dump(path, report)
    return path


def build_parser():
    ap = argparse.ArgumentParser("Controlled edge prompt experiment driver")
    ap.add_argument("action", choices=["prepare", "run_stage_a", "run_stage_b", "run_fusion", "summarize", "full"])
    ap.add_argument("--experiment_name", type=str, default="acm10_typepair_edgeprompt2_controlled")
    ap.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--python_exec", type=str, default=sys.executable)

    ap.add_argument("--dataset", type=str, default="ACM")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--shot", type=int, default=10)
    ap.add_argument("--pretrain_seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--cuda_visible_devices", type=str, default=None)

    ap.add_argument("--hgnn_type", type=str, default="GCN")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--num_class", type=int, default=3)
    ap.add_argument("--classification_type", type=str, default="NIG")
    ap.add_argument("--embed_batch_size", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--prompt_lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop_metric", type=str, default="macro")

    ap.add_argument("--hgmp_ckpt", type=str, default="artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth")
    ap.add_argument("--typepair_ckpt", type=str, default="artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth")
    ap.add_argument("--hgprompt_ckpt", type=str, default="artifacts/checkpoints/hgprompt/pretrain/ACM.gcn.ft2.hop1.seed0.best.pt")

    ap.add_argument("--typepair_spectral_dim", type=int, default=8)
    ap.add_argument("--typepair_edge_prompt_hidden", type=int, default=128)
    ap.add_argument("--typepair_edge_prompt_alpha", type=float, default=0.5)
    ap.add_argument("--typepair_edge_prompt_fusion", type=str, default="add")

    ap.add_argument("--stage_a_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--stage_b_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--fusion_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_mode", type=str, default="offline")
    ap.add_argument("--wandb_project", type=str, default="HGEP")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    if args.action == "prepare":
        prepare_manifest(args)
        return
    if not (args.results_root / args.experiment_name / "experiment_manifest.initial.json").exists():
        prepare_manifest(args)
    if args.action == "run_stage_a":
        run_stage_a(args)
        summarize_stage_a(args)
        resolve_stage_b_manifest(args)
        return
    if args.action == "run_stage_b":
        resolve_stage_b_manifest(args)
        run_stage_b(args)
        summarize_stage_b(args)
        resolve_fusion_manifest(args)
        return
    if args.action == "run_fusion":
        resolve_fusion_manifest(args)
        run_fusion(args)
        summarize_fusion(args)
        return
    if args.action == "summarize":
        summarize_stage_a(args)
        if (args.results_root / args.experiment_name / "experiment_manifest.stage_b.json").exists():
            summarize_stage_b(args)
        summarize_fusion(args)
        finalize_report(args)
        return
    if args.action == "full":
        run_stage_a(args)
        summarize_stage_a(args)
        resolve_stage_b_manifest(args)
        run_stage_b(args)
        summarize_stage_b(args)
        resolve_fusion_manifest(args)
        run_fusion(args)
        summarize_fusion(args)
        finalize_report(args)
        return
    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
