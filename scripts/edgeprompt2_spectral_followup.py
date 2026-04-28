from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "protocol_benchmark_v2.py"
DEFAULT_RESULTS_ROOT = ROOT / "artifacts" / "results" / "edgeprompt2_experiments"
PREVIOUS_EXPERIMENT_ROOT = DEFAULT_RESULTS_ROOT / "acm10_typepair_edgeprompt2_controlled"

NO_EDGE_FEATURES: list[str] = []
SPECTRAL_ONLY_FEATURES = ["SpectralEmbeddingDiff"]
SPECTRAL_PLUS_OVERLAP_FEATURES = ["SpectralEmbeddingDiff", "NeighborTypeOverlap"]
SPECTRAL_PLUS_OVERLAP_PLUS_COMMUNITY_FEATURES = [
    "CommunityLabelDiff",
    "NeighborTypeOverlap",
    "SpectralEmbeddingDiff",
]
STRUCTURAL_RESIDUAL_ONLY_FEATURES = ["SpectralEmbeddingDiff", "NeighborTypeOverlap", "DegreeDiff"]

PAIRWISE_AUXILIARY_PRIORITY = [
    "NeighborTypeOverlap",
    "CommunityLabelDiff",
    "DegreeDiff",
    "PageRankDiff",
    "NodeTypeEncoding",
    "EdgeTypeOneHot",
    "NodeAttrCosSim",
    "NeighborAttrVariance",
]

KNOWN_CONFIG_REFS = {
    tuple(NO_EDGE_FEATURES): ("config_00", "no_edge"),
    tuple(["NodeTypeEncoding"]): ("config_01", "node_type_encoding"),
    tuple(["EdgeTypeOneHot"]): ("config_02", "edge_type_onehot"),
    tuple(["NeighborTypeOverlap"]): ("config_03", "neighbor_type_overlap"),
    tuple(["DegreeDiff"]): ("config_04", "degree_diff"),
    tuple(SPECTRAL_ONLY_FEATURES): ("config_05", "spectral_embedding_diff"),
    tuple(["NodeTypeEncoding", "EdgeTypeOneHot"]): ("config_06", "type_memory_only"),
    tuple(SPECTRAL_PLUS_OVERLAP_FEATURES): ("config_07", "spectral_plus_neighbor_type_overlap"),
    tuple(STRUCTURAL_RESIDUAL_ONLY_FEATURES): ("config_08", "structural_residual_only"),
    tuple(["DegreeDiff", "EdgeTypeOneHot", "NeighborTypeOverlap"]): ("config_09", "degree_edge_type_neighbor_overlap"),
    tuple(["NodeTypeEncoding", "EdgeTypeOneHot", "NeighborTypeOverlap", "SpectralEmbeddingDiff"]): ("config_10", "edgeprompt2_core"),
    tuple(SPECTRAL_PLUS_OVERLAP_PLUS_COMMUNITY_FEATURES): ("config_11", "community_neighbor_overlap_spectral"),
}

MANIFEST_FIELDNAMES = [
    "stage",
    "slot",
    "config_ref",
    "label",
    "features",
    "repeats",
    "seeds",
    "spectral_dim",
    "fusion",
    "alpha",
    "hidden",
    "save_dir",
    "benchmark_output_dir",
    "wandb_group",
    "wandb_job_type",
    "wandb_name",
    "wandb_tags",
    "wandb_notes",
    "command",
]

SUMMARY_FIELDNAMES = [
    "stage",
    "slot",
    "config_ref",
    "label",
    "features",
    "repeats",
    "seeds",
    "spectral_dim",
    "fusion",
    "alpha",
    "hidden",
    "pooled_count",
    "pooled_micro_mean",
    "pooled_micro_std",
    "pooled_macro_mean",
    "pooled_macro_std",
    "seed_count",
    "seed_micro_mean",
    "seed_micro_std",
    "seed_macro_mean",
    "seed_macro_std",
    "output_dir",
    "command",
    "delta_seed_macro_vs_baseline_no_edge",
    "delta_seed_micro_vs_baseline_no_edge",
    "delta_seed_macro_vs_spectral_only",
    "delta_seed_micro_vs_spectral_only",
]


@dataclass
class ResultSummary:
    stage: str
    slot: str
    config_ref: str
    label: str
    features: list[str]
    repeats: int
    seeds: list[int]
    spectral_dim: int
    fusion: str
    alpha: float
    hidden: int
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
    delta_seed_macro_vs_baseline_no_edge: float | None = None
    delta_seed_micro_vs_baseline_no_edge: float | None = None
    delta_seed_macro_vs_spectral_only: float | None = None
    delta_seed_micro_vs_spectral_only: float | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0].keys()) if rows else [])
    if not names:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _feature_slug(name: str) -> str:
    out = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and name[idx - 1].islower():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _label_for_features(features: list[str]) -> str:
    if not features:
        return "no_edge"
    config_ref, label = KNOWN_CONFIG_REFS.get(tuple(features), ("custom", None))
    if label is not None:
        return label
    return "plus_".join(_feature_slug(item) for item in features)


def _config_ref_for_features(features: list[str]) -> str:
    config_ref, _ = KNOWN_CONFIG_REFS.get(tuple(features), ("custom", None))
    if config_ref != "custom":
        return config_ref
    return "custom:" + ",".join(features)


def _benchmark_output_dir(save_dir: Path, dataset: str, shot: int) -> Path:
    return save_dir / dataset / f"{shot}-shot"


def _command_string(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _base_args(args) -> list[str]:
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
    if args.wandb_tags:
        base.extend(["--wandb_tags", *args.wandb_tags])
    return base


def _wandb_note(stage: str, hypothesis: str, compare_to: str, decision_rule: str) -> str:
    return (
        f"stage={stage}; hypothesis={hypothesis}; baseline={compare_to}; "
        f"decision_rule={decision_rule}"
    )


def _build_run_entry(
    args,
    exp_root: Path,
    stage: str,
    slot: str,
    features: list[str],
    repeats: int,
    seeds: list[int],
    spectral_dim: int,
    fusion: str,
    alpha: float,
    hidden: int,
    wandb_job_type: str,
    hypothesis: str,
    compare_to: str,
    decision_rule: str,
) -> dict[str, Any]:
    save_dir = exp_root / stage / slot
    benchmark_out = _benchmark_output_dir(save_dir, args.dataset, args.shot)
    wandb_group = f"{args.experiment_name}-{stage}"
    wandb_name = f"{args.experiment_name}-{slot}"
    wandb_notes = _wandb_note(stage, hypothesis, compare_to, decision_rule)
    cmd = [
        args.python_exec,
        str(BENCHMARK),
        *_base_args(args),
        "--save_dir",
        str(save_dir),
        "--repeats",
        str(repeats),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--typepair_spectral_dim",
        str(spectral_dim),
        "--typepair_edge_prompt_hidden",
        str(hidden),
        "--typepair_edge_prompt_alpha",
        str(alpha),
        "--typepair_edge_prompt_fusion",
        fusion,
        "--wandb_group",
        wandb_group,
        "--wandb_job_type",
        wandb_job_type,
        "--wandb_name",
        wandb_name,
        "--wandb_notes",
        wandb_notes,
    ]
    if features:
        cmd.append("--enable_typepair_edge_features")
        cmd.extend(["--typepair_edge_feature_names", *features])
    command = _command_string(cmd)
    if args.cuda_visible_devices:
        command = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} {command}"
    return {
        "stage": stage,
        "slot": slot,
        "config_ref": _config_ref_for_features(features),
        "label": _label_for_features(features),
        "features": list(features),
        "repeats": int(repeats),
        "seeds": list(seeds),
        "spectral_dim": int(spectral_dim),
        "fusion": fusion,
        "alpha": float(alpha),
        "hidden": int(hidden),
        "wandb_group": wandb_group,
        "wandb_job_type": wandb_job_type,
        "wandb_name": wandb_name,
        "wandb_tags": list(args.wandb_tags),
        "wandb_notes": wandb_notes,
        "hypothesis": hypothesis,
        "compare_to": compare_to,
        "decision_rule": decision_rule,
        "save_dir": str(save_dir),
        "benchmark_output_dir": str(benchmark_out),
        "env_overrides": {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices} if args.cuda_visible_devices else {},
        "command": command,
        "cmd": cmd,
    }


def _entry_to_manifest_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": entry["stage"],
        "slot": entry["slot"],
        "config_ref": entry["config_ref"],
        "label": entry["label"],
        "features": ",".join(entry["features"]),
        "repeats": entry["repeats"],
        "seeds": ",".join(str(seed) for seed in entry["seeds"]),
        "spectral_dim": entry["spectral_dim"],
        "fusion": entry["fusion"],
        "alpha": entry["alpha"],
        "hidden": entry["hidden"],
        "save_dir": entry["save_dir"],
        "benchmark_output_dir": entry["benchmark_output_dir"],
        "wandb_group": entry["wandb_group"],
        "wandb_job_type": entry["wandb_job_type"],
        "wandb_name": entry["wandb_name"],
        "wandb_tags": ",".join(entry["wandb_tags"]),
        "wandb_notes": entry["wandb_notes"],
        "command": entry["command"],
    }


def _summary_to_row(summary: ResultSummary) -> dict[str, Any]:
    row = asdict(summary)
    row["features"] = ",".join(summary.features)
    row["seeds"] = ",".join(str(seed) for seed in summary.seeds)
    return row


def _write_manifest(path: Path, payload: dict[str, Any]):
    _json_dump(path, payload)
    rows = [_entry_to_manifest_row(entry) for entry in payload.get("runs", [])]
    _write_csv(path.with_suffix(".csv"), rows, fieldnames=MANIFEST_FIELDNAMES)


def _write_stage_placeholder_files(exp_root: Path, stage: str, reason: str):
    _json_dump(
        exp_root / f"{stage}_analysis.json",
        {
            "stage": stage,
            "status": "pending",
            "reason": reason,
            "updated_at": _now_iso(),
        },
    )
    _write_csv(exp_root / f"{stage}_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)


def _previous_results_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "previous_experiment_root": str(PREVIOUS_EXPERIMENT_ROOT),
        "stage_a_summary_found": False,
        "stage_b_analysis_found": False,
    }
    stage_a_path = PREVIOUS_EXPERIMENT_ROOT / "stage_a_summary.csv"
    stage_b_path = PREVIOUS_EXPERIMENT_ROOT / "stage_b_analysis.json"
    if stage_a_path.exists():
        rows = []
        with stage_a_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
        snapshot["stage_a_summary_found"] = True
        snapshot["stage_a_top4_slots"] = [row["slot"] for row in rows[:4]]
    if stage_b_path.exists():
        analysis = _read_json(stage_b_path)
        snapshot["stage_b_analysis_found"] = True
        snapshot["stage_b_ranked_slots"] = analysis.get("ranked_slots", [])
        snapshot["stage_b_support_combined"] = analysis.get("support_combined")
        snapshot["stage_b_best_single_feature_config_id"] = analysis.get("best_single_feature_config_id")
    return snapshot


def _stage0_manifest(args, exp_root: Path) -> dict[str, Any]:
    decision_rule = (
        "primary=seed_macro_mean desc; tie1=seed_macro_std asc; tie2=seed_micro_mean desc; "
        "winner must come from spectral-centered slots only"
    )
    runs = [
        _build_run_entry(
            args,
            exp_root,
            stage="stage0_confirm",
            slot="baseline_no_edge",
            features=NO_EDGE_FEATURES,
            repeats=5,
            seeds=list(args.stage_seeds),
            spectral_dim=8,
            fusion="add",
            alpha=0.5,
            hidden=128,
            wandb_job_type="typepair_edgeprompt_followup_confirm",
            hypothesis="No-edge reference remains below spectral-centered candidates under strict repeats.",
            compare_to="baseline_no_edge and spectral_only",
            decision_rule=decision_rule,
        ),
        _build_run_entry(
            args,
            exp_root,
            stage="stage0_confirm",
            slot="spectral_only",
            features=SPECTRAL_ONLY_FEATURES,
            repeats=5,
            seeds=list(args.stage_seeds),
            spectral_dim=8,
            fusion="add",
            alpha=0.5,
            hidden=128,
            wandb_job_type="typepair_edgeprompt_followup_confirm",
            hypothesis="SpectralEmbeddingDiff remains the strongest single-feature baseline.",
            compare_to="baseline_no_edge",
            decision_rule=decision_rule,
        ),
        _build_run_entry(
            args,
            exp_root,
            stage="stage0_confirm",
            slot="spectral_plus_overlap",
            features=SPECTRAL_PLUS_OVERLAP_FEATURES,
            repeats=5,
            seeds=list(args.stage_seeds),
            spectral_dim=8,
            fusion="add",
            alpha=0.5,
            hidden=128,
            wandb_job_type="typepair_edgeprompt_followup_confirm",
            hypothesis="config_07 remains competitive under stricter repeats.",
            compare_to="spectral_only",
            decision_rule=decision_rule,
        ),
        _build_run_entry(
            args,
            exp_root,
            stage="stage0_confirm",
            slot="spectral_plus_overlap_plus_community",
            features=SPECTRAL_PLUS_OVERLAP_PLUS_COMMUNITY_FEATURES,
            repeats=5,
            seeds=list(args.stage_seeds),
            spectral_dim=8,
            fusion="add",
            alpha=0.5,
            hidden=128,
            wandb_job_type="typepair_edgeprompt_followup_confirm",
            hypothesis="config_11 may survive Stage B-style confirmation once explicitly rerun.",
            compare_to="spectral_only",
            decision_rule=decision_rule,
        ),
        _build_run_entry(
            args,
            exp_root,
            stage="stage0_confirm",
            slot="structural_residual_only",
            features=STRUCTURAL_RESIDUAL_ONLY_FEATURES,
            repeats=5,
            seeds=list(args.stage_seeds),
            spectral_dim=8,
            fusion="add",
            alpha=0.5,
            hidden=128,
            wandb_job_type="typepair_edgeprompt_followup_confirm",
            hypothesis="config_08 needs confirmation before being discarded as a spectral-centered base.",
            compare_to="spectral_only",
            decision_rule=decision_rule,
        ),
    ]
    return {
        "stage": "stage0_confirm",
        "status": "planned",
        "created_at": _now_iso(),
        "objective": "Confirm top Stage A spectral-centered candidates under strict repeats.",
        "decision_rule": decision_rule,
        "runs": runs,
    }


def _placeholder_manifest(stage: str, objective: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "pending_upstream",
        "created_at": _now_iso(),
        "objective": objective,
        "pending_reason": reason,
        "runs": [],
    }


def prepare(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    exp_root.mkdir(parents=True, exist_ok=True)

    initial_manifest = {
        "experiment_name": args.experiment_name,
        "created_at": _now_iso(),
        "results_root": str(exp_root),
        "benchmark_script": str(BENCHMARK),
        "python_exec": args.python_exec,
        "known_starting_point": {
            "stage_a_top_ranked_configs": ["config_07", "config_05", "config_11", "config_08"],
            "best_single_feature": "config_05 = SpectralEmbeddingDiff",
            "stage_b_ranked_order": [
                "best_single_feature",
                "baseline_no_edge",
                "type_memory_only",
                "structural_residual_only",
                "edgeprompt2_core",
            ],
            "stage_b_support_combined": False,
            "followup_focus": "spectral-centered confirmation and controlled expansion only",
        },
        "previous_results_snapshot": _previous_results_snapshot(),
        "base_args": {
            "dataset": args.dataset,
            "root": args.root,
            "splits": args.splits,
            "shot": args.shot,
            "methods": ["typepair"],
            "pretrain_seed": args.pretrain_seed,
            "device": args.device,
            "cuda_visible_devices": args.cuda_visible_devices,
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
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_mode": args.wandb_mode,
            "wandb_tags": list(args.wandb_tags),
        },
        "stage_plan": {
            "stage0_confirm": {
                "repeats": 5,
                "seeds": list(args.stage_seeds),
                "slots": [
                    "baseline_no_edge",
                    "spectral_only",
                    "spectral_plus_overlap",
                    "spectral_plus_overlap_plus_community",
                    "structural_residual_only",
                ],
            },
            "stage1_scan": {
                "coarse_grid": {
                    "spectral_dim": [4, 8, 16],
                    "fusion": ["add", "gate"],
                    "alpha": [0.2, 0.5, 0.8],
                    "hidden": [128],
                    "repeats": 3,
                    "seeds": list(args.stage_seeds),
                },
                "confirm_top_k": 3,
                "confirm_repeats": 5,
            },
            "stage2_pairwise": {
                "candidate_priority": list(PAIRWISE_AUXILIARY_PRIORITY),
                "repeats": 3,
                "seeds": list(args.stage_seeds),
                "qualify_rule": [
                    "seed_macro_mean >= best_base + 0.002",
                    "or |delta_macro| <= 0.001 and lower seed_macro_std and seed_micro_mean not lower",
                ],
            },
            "stage3_final": {
                "repeats": 5,
                "seeds": list(args.stage_seeds),
                "include": [
                    "best_base_config",
                    "qualified_pairwise_candidates",
                    "baseline_no_edge",
                    "spectral_only_reference_if_needed",
                ],
            },
        },
    }
    _json_dump(exp_root / "experiment_manifest.initial.json", initial_manifest)

    stage0 = _stage0_manifest(args, exp_root)
    _write_manifest(exp_root / "experiment_manifest.stage0_confirm.json", stage0)

    _write_manifest(
        exp_root / "experiment_manifest.stage1_coarse.json",
        _placeholder_manifest(
            "stage1_coarse",
            "Scan spectral_dim/fusion/alpha on the Stage 0 winner.",
            "Waiting for stage0_confirm winner.",
        ),
    )
    _write_manifest(
        exp_root / "experiment_manifest.stage1_confirm.json",
        _placeholder_manifest(
            "stage1_confirm",
            "Re-run top coarse scan configs with stricter repeats.",
            "Waiting for stage1_coarse ranking.",
        ),
    )
    _write_manifest(
        exp_root / "experiment_manifest.stage2_pairwise.json",
        _placeholder_manifest(
            "stage2_pairwise",
            "Add exactly one auxiliary feature on top of the Stage 1 best base config.",
            "Waiting for stage1_confirm winner.",
        ),
    )
    _write_manifest(
        exp_root / "experiment_manifest.stage3_final.json",
        _placeholder_manifest(
            "stage3_final",
            "Final confirmation between best base, qualified pairwise candidates, and references.",
            "Waiting for stage2_pairwise analysis.",
        ),
    )

    for stage, reason in [
        ("stage0_confirm", "Stage 0 not executed yet."),
        ("stage1_coarse", "Stage 1 coarse scan not executed yet."),
        ("stage1_confirm", "Stage 1 confirm reruns not executed yet."),
        ("stage1", "Stage 1 overall analysis pending."),
        ("stage2_pairwise", "Stage 2 not executed yet."),
        ("stage3_final", "Stage 3 not executed yet."),
    ]:
        _write_stage_placeholder_files(exp_root, stage, reason)

    _append_jsonl(
        exp_root / "command_log.jsonl",
        {
            "timestamp": _now_iso(),
            "status": "prepare",
            "experiment_name": args.experiment_name,
            "results_root": str(exp_root),
        },
    )
    finalize_report(args)
    return exp_root / "experiment_manifest.initial.json"


def _run_entry(entry: dict[str, Any], exp_root: Path, force: bool):
    output_dir = Path(entry["benchmark_output_dir"])
    summary_path = output_dir / "overall_summary.json"
    if summary_path.exists() and not force:
        _append_jsonl(
            exp_root / "command_log.jsonl",
            {
                "timestamp": _now_iso(),
                "status": "skip_existing",
                "stage": entry["stage"],
                "slot": entry["slot"],
                "command": entry["command"],
            },
        )
        return

    save_dir = Path(entry["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "run.log"
    _append_jsonl(
        exp_root / "command_log.jsonl",
        {
            "timestamp": _now_iso(),
            "status": "start",
            "stage": entry["stage"],
            "slot": entry["slot"],
            "command": entry["command"],
        },
    )
    started = time.time()
    env = os.environ.copy()
    env.update(entry.get("env_overrides", {}))
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            entry["cmd"],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    payload = {
        "timestamp": _now_iso(),
        "status": "ok" if process.returncode == 0 else "failed",
        "stage": entry["stage"],
        "slot": entry["slot"],
        "command": entry["command"],
        "returncode": process.returncode,
        "seconds": round(time.time() - started, 3),
        "log_path": str(log_path),
    }
    _append_jsonl(exp_root / "command_log.jsonl", payload)
    if process.returncode != 0:
        try:
            log_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
        except Exception as exc:
            log_tail = f"<failed to read log: {exc}>"

        raise RuntimeError(
            f"Command failed for {entry['slot']} with returncode={process.returncode}\n"
            f"Command:\n{entry['command']}\n\n"
            f"Log path:\n{log_path}\n\n"
            f"Last 120 lines of log:\n{log_tail}"
        )


def _missing_runs(manifest: dict[str, Any]) -> list[str]:
    missing = []
    for entry in manifest.get("runs", []):
        summary_path = Path(entry["benchmark_output_dir"]) / "overall_summary.json"
        if not summary_path.exists():
            missing.append(entry["slot"])
    return missing


def _collect_summary(entry: dict[str, Any]) -> ResultSummary:
    summary = _read_json(Path(entry["benchmark_output_dir"]) / "overall_summary.json")
    pooled = summary["pooled_runs"]["typepair"]
    seed = summary["seed_mean_then_std"]["typepair"]
    return ResultSummary(
        stage=entry["stage"],
        slot=entry["slot"],
        config_ref=entry["config_ref"],
        label=entry["label"],
        features=list(entry["features"]),
        repeats=int(entry["repeats"]),
        seeds=list(entry["seeds"]),
        spectral_dim=int(entry["spectral_dim"]),
        fusion=str(entry["fusion"]),
        alpha=float(entry["alpha"]),
        hidden=int(entry["hidden"]),
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
    return sorted(
        rows,
        key=lambda item: (item.seed_macro_mean, -item.seed_macro_std, item.seed_micro_mean),
        reverse=True,
    )


def _add_reference_deltas(rows: list[ResultSummary]):
    baseline = next((row for row in rows if row.slot == "baseline_no_edge" or not row.features), None)
    spectral = next((row for row in rows if row.slot == "spectral_only" or row.features == SPECTRAL_ONLY_FEATURES), None)
    for row in rows:
        if baseline is not None:
            row.delta_seed_macro_vs_baseline_no_edge = row.seed_macro_mean - baseline.seed_macro_mean
            row.delta_seed_micro_vs_baseline_no_edge = row.seed_micro_mean - baseline.seed_micro_mean
        if spectral is not None:
            row.delta_seed_macro_vs_spectral_only = row.seed_macro_mean - spectral.seed_macro_mean
            row.delta_seed_micro_vs_spectral_only = row.seed_micro_mean - spectral.seed_micro_mean


def _stage_payload(stage: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": _now_iso(),
    }
    if reason is not None:
        payload["reason"] = reason
    payload.update(extra)
    return payload


def _write_summary_and_analysis(
    exp_root: Path,
    stage: str,
    rows: list[ResultSummary],
    analysis: dict[str, Any],
    csv_name: str | None = None,
):
    ranked = _sort_results(rows)
    _add_reference_deltas(ranked)
    _write_csv(exp_root / (csv_name or f"{stage}_summary.csv"), [_summary_to_row(row) for row in ranked], fieldnames=SUMMARY_FIELDNAMES)
    _json_dump(exp_root / f"{stage}_analysis.json", analysis)


def summarize_stage0(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    manifest = _read_json(exp_root / "experiment_manifest.stage0_confirm.json")
    missing = _missing_runs(manifest)
    if missing:
        payload = _stage_payload(
            "stage0_confirm",
            "pending",
            reason=f"Missing overall_summary.json for slots: {', '.join(missing)}",
            missing_slots=missing,
        )
        _json_dump(exp_root / "stage0_confirm_analysis.json", payload)
        _write_csv(exp_root / "stage0_confirm_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    rows = [_collect_summary(entry) for entry in manifest["runs"]]
    ranked = _sort_results(rows)
    _add_reference_deltas(ranked)
    _write_csv(exp_root / "stage0_confirm_summary.csv", [_summary_to_row(row) for row in ranked], fieldnames=SUMMARY_FIELDNAMES)

    candidate_slots = {
        "spectral_only",
        "spectral_plus_overlap",
        "spectral_plus_overlap_plus_community",
        "structural_residual_only",
    }
    spectral_candidates = [row for row in ranked if row.slot in candidate_slots]
    winner = spectral_candidates[0]
    payload = _stage_payload(
        "stage0_confirm",
        "complete",
        ranked_slots=[row.slot for row in ranked],
        best_spectral_centered_base_candidate=winner.slot,
        best_spectral_centered_base_config=_summary_to_row(winner),
        candidate_pool=[row.slot for row in spectral_candidates],
        decision_rule=manifest["decision_rule"],
    )
    _json_dump(exp_root / "stage0_confirm_analysis.json", payload)
    return payload


def resolve_stage1_coarse(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    stage0 = summarize_stage0(args)
    if stage0["status"] != "complete":
        reason = stage0.get("reason", "Stage 0 incomplete.")
        _append_jsonl(
            exp_root / "command_log.jsonl",
            {
                "timestamp": _now_iso(),
                "status": "blocked",
                "stage": "stage1_coarse",
                "reason": reason,
            },
        )
        raise RuntimeError(reason)

    winner = stage0["best_spectral_centered_base_config"]
    features = [item for item in winner["features"].split(",") if item]
    decision_rule = "Rank coarse scan by seed_macro_mean, then lower seed_macro_std, then higher seed_micro_mean."
    runs = []
    for spectral_dim in (4, 8, 16):
        for fusion in ("add", "gate"):
            for alpha in (0.2, 0.5, 0.8):
                alpha_tag = str(alpha).replace(".", "p")
                slot = f"{winner['slot']}_dim{spectral_dim}_{fusion}_alpha{alpha_tag}"
                runs.append(
                    _build_run_entry(
                        args,
                        exp_root,
                        stage="stage1_coarse",
                        slot=slot,
                        features=features,
                        repeats=3,
                        seeds=list(args.stage_seeds),
                        spectral_dim=spectral_dim,
                        fusion=fusion,
                        alpha=alpha,
                        hidden=128,
                        wandb_job_type="typepair_edgeprompt_followup_spectral_scan",
                        hypothesis=f"Stage 0 winner improves with tuned spectral_dim/fusion/alpha: {winner['slot']}.",
                        compare_to=f"stage0 winner {winner['slot']}",
                        decision_rule=decision_rule,
                    )
                )
    payload = {
        "stage": "stage1_coarse",
        "status": "planned",
        "created_at": _now_iso(),
        "objective": "Coarse spectral hyperparameter scan on the Stage 0 winner.",
        "base_stage0_slot": winner["slot"],
        "base_features": features,
        "decision_rule": decision_rule,
        "runs": runs,
    }
    path = exp_root / "experiment_manifest.stage1_coarse.json"
    _write_manifest(path, payload)
    return path


def summarize_stage1_coarse(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    manifest = _read_json(exp_root / "experiment_manifest.stage1_coarse.json")
    if not manifest.get("runs"):
        payload = _stage_payload(
            "stage1_coarse",
            "pending",
            reason=manifest.get("pending_reason", "Stage 1 coarse manifest is still waiting on Stage 0."),
        )
        _json_dump(exp_root / "stage1_coarse_analysis.json", payload)
        _write_csv(exp_root / "stage1_coarse_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload
    missing = _missing_runs(manifest)
    if missing:
        payload = _stage_payload(
            "stage1_coarse",
            "pending",
            reason=f"Missing overall_summary.json for slots: {', '.join(missing)}",
            missing_slots=missing,
        )
        _json_dump(exp_root / "stage1_coarse_analysis.json", payload)
        _write_csv(exp_root / "stage1_coarse_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    rows = [_collect_summary(entry) for entry in manifest["runs"]]
    ranked = _sort_results(rows)
    _add_reference_deltas(ranked)
    _write_csv(exp_root / "stage1_coarse_summary.csv", [_summary_to_row(row) for row in ranked], fieldnames=SUMMARY_FIELDNAMES)
    payload = _stage_payload(
        "stage1_coarse",
        "complete",
        ranked_slots=[row.slot for row in ranked],
        top3_slots=[row.slot for row in ranked[:3]],
        base_stage0_slot=manifest["base_stage0_slot"],
    )
    _json_dump(exp_root / "stage1_coarse_analysis.json", payload)
    return payload


def resolve_stage1_confirm(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    coarse = summarize_stage1_coarse(args)
    if coarse["status"] != "complete":
        reason = coarse.get("reason", "Stage 1 coarse incomplete.")
        _append_jsonl(
            exp_root / "command_log.jsonl",
            {
                "timestamp": _now_iso(),
                "status": "blocked",
                "stage": "stage1_confirm",
                "reason": reason,
            },
        )
        raise RuntimeError(reason)

    coarse_manifest = _read_json(exp_root / "experiment_manifest.stage1_coarse.json")
    keep = set(coarse["top3_slots"])
    runs = []
    for entry in coarse_manifest["runs"]:
        if entry["slot"] not in keep:
            continue
        runs.append(
            _build_run_entry(
                args,
                exp_root,
                stage="stage1_confirm",
                slot=entry["slot"],
                features=list(entry["features"]),
                repeats=5,
                seeds=list(args.stage_seeds),
                spectral_dim=int(entry["spectral_dim"]),
                fusion=str(entry["fusion"]),
                alpha=float(entry["alpha"]),
                hidden=int(entry["hidden"]),
                wandb_job_type="typepair_edgeprompt_followup_spectral_scan",
                hypothesis=f"Top coarse config requires confirmation: {entry['slot']}.",
                compare_to="other top-3 coarse scan configs",
                decision_rule="Final Stage 1 winner uses seed_macro_mean, then lower seed_macro_std, then higher seed_micro_mean.",
            )
        )
    payload = {
        "stage": "stage1_confirm",
        "status": "planned",
        "created_at": _now_iso(),
        "objective": "Confirm top-3 coarse scan configs with stricter repeats.",
        "source_stage": "stage1_coarse",
        "runs": runs,
    }
    path = exp_root / "experiment_manifest.stage1_confirm.json"
    _write_manifest(path, payload)
    return path


def summarize_stage1(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    coarse_info = summarize_stage1_coarse(args)
    confirm_manifest = _read_json(exp_root / "experiment_manifest.stage1_confirm.json")
    if not confirm_manifest.get("runs"):
        payload = _stage_payload(
            "stage1",
            "pending",
            reason=confirm_manifest.get("pending_reason", "Stage 1 confirm manifest is still waiting on Stage 1 coarse."),
            stage1_coarse=coarse_info,
        )
        _json_dump(exp_root / "stage1_analysis.json", payload)
        _json_dump(
            exp_root / "stage1_confirm_analysis.json",
            _stage_payload(
                "stage1_confirm",
                "pending",
                reason=confirm_manifest.get("pending_reason", "Stage 1 confirm manifest is still waiting on Stage 1 coarse."),
            ),
        )
        _write_csv(exp_root / "stage1_confirm_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        _write_csv(exp_root / "stage1_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload
    missing = _missing_runs(confirm_manifest)
    if missing:
        payload = _stage_payload(
            "stage1",
            "pending",
            reason=f"Missing overall_summary.json for Stage 1 confirm slots: {', '.join(missing)}",
            stage1_coarse=coarse_info,
            missing_slots=missing,
        )
        _json_dump(exp_root / "stage1_analysis.json", payload)
        _write_csv(exp_root / "stage1_confirm_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        _write_csv(exp_root / "stage1_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    rows = [_collect_summary(entry) for entry in confirm_manifest["runs"]]
    ranked = _sort_results(rows)
    _add_reference_deltas(ranked)
    row_dicts = [_summary_to_row(row) for row in ranked]
    _write_csv(exp_root / "stage1_confirm_summary.csv", row_dicts, fieldnames=SUMMARY_FIELDNAMES)
    _write_csv(exp_root / "stage1_summary.csv", row_dicts, fieldnames=SUMMARY_FIELDNAMES)
    best = ranked[0]
    payload = _stage_payload(
        "stage1",
        "complete",
        stage1_coarse=coarse_info,
        ranked_slots=[row.slot for row in ranked],
        best_base_config=_summary_to_row(best),
        selection_rule="primary=seed_macro_mean desc; tie1=seed_macro_std asc; tie2=seed_micro_mean desc",
    )
    _json_dump(exp_root / "stage1_analysis.json", payload)
    _json_dump(
        exp_root / "stage1_confirm_analysis.json",
        _stage_payload(
            "stage1_confirm",
            "complete",
            ranked_slots=[row.slot for row in ranked],
            best_slot=best.slot,
        ),
    )
    return payload


def resolve_stage2(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    stage1 = summarize_stage1(args)
    if stage1["status"] != "complete":
        reason = stage1.get("reason", "Stage 1 incomplete.")
        _append_jsonl(
            exp_root / "command_log.jsonl",
            {
                "timestamp": _now_iso(),
                "status": "blocked",
                "stage": "stage2_pairwise",
                "reason": reason,
            },
        )
        raise RuntimeError(reason)

    base = stage1["best_base_config"]
    base_features = [item for item in base["features"].split(",") if item]
    base_feature_set = set(base_features)
    runs = []
    for feature in PAIRWISE_AUXILIARY_PRIORITY:
        if feature in base_feature_set:
            continue
        slot = f"{base['slot']}_plus_{_feature_slug(feature)}"
        features = base_features + [feature]
        runs.append(
            _build_run_entry(
                args,
                exp_root,
                stage="stage2_pairwise",
                slot=slot,
                features=features,
                repeats=3,
                seeds=list(args.stage_seeds),
                spectral_dim=int(base["spectral_dim"]),
                fusion=str(base["fusion"]),
                alpha=float(base["alpha"]),
                hidden=int(base["hidden"]),
                wandb_job_type="typepair_edgeprompt_followup_pairwise",
                hypothesis=f"Single auxiliary feature {feature} may improve the Stage 1 best spectral-centered base.",
                compare_to=f"best_base_config {base['slot']}",
                decision_rule="Keep only configs that beat best_base by +0.002 macro, or match within 0.001 with lower std and non-decreasing micro.",
            )
        )
    payload = {
        "stage": "stage2_pairwise",
        "status": "planned",
        "created_at": _now_iso(),
        "objective": "Test one auxiliary feature at a time on top of the Stage 1 winner.",
        "best_base_slot": base["slot"],
        "best_base_features": base_features,
        "runs": runs,
    }
    path = exp_root / "experiment_manifest.stage2_pairwise.json"
    _write_manifest(path, payload)
    return path


def summarize_stage2(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    stage1 = summarize_stage1(args)
    if stage1["status"] != "complete":
        payload = _stage_payload(
            "stage2_pairwise",
            "pending",
            reason="Stage 1 is incomplete, so Stage 2 cannot be evaluated.",
            stage1=stage1,
        )
        _json_dump(exp_root / "stage2_pairwise_analysis.json", payload)
        _write_csv(exp_root / "stage2_pairwise_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    manifest = _read_json(exp_root / "experiment_manifest.stage2_pairwise.json")
    if not manifest.get("runs"):
        payload = _stage_payload(
            "stage2_pairwise",
            "pending",
            reason=manifest.get("pending_reason", "Stage 2 manifest is still waiting on Stage 1."),
        )
        _json_dump(exp_root / "stage2_pairwise_analysis.json", payload)
        _write_csv(exp_root / "stage2_pairwise_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload
    missing = _missing_runs(manifest)
    if missing:
        payload = _stage_payload(
            "stage2_pairwise",
            "pending",
            reason=f"Missing overall_summary.json for slots: {', '.join(missing)}",
            missing_slots=missing,
        )
        _json_dump(exp_root / "stage2_pairwise_analysis.json", payload)
        _write_csv(exp_root / "stage2_pairwise_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    rows = [_collect_summary(entry) for entry in manifest["runs"]]
    ranked = _sort_results(rows)
    _add_reference_deltas(ranked)
    _write_csv(exp_root / "stage2_pairwise_summary.csv", [_summary_to_row(row) for row in ranked], fieldnames=SUMMARY_FIELDNAMES)

    base = stage1["best_base_config"]
    base_macro = float(base["seed_macro_mean"])
    base_std = float(base["seed_macro_std"])
    base_micro = float(base["seed_micro_mean"])
    qualified = []
    reasoning = {}
    for row in ranked:
        macro_gain = row.seed_macro_mean - base_macro
        std_gain = base_std - row.seed_macro_std
        micro_gain = row.seed_micro_mean - base_micro
        pass_rule_a = macro_gain >= 0.002
        pass_rule_b = abs(macro_gain) <= 0.001 and std_gain > 0.0 and micro_gain >= 0.0
        qualifies = pass_rule_a or pass_rule_b
        reasoning[row.slot] = {
            "macro_gain_vs_best_base": macro_gain,
            "micro_gain_vs_best_base": micro_gain,
            "std_reduction_vs_best_base": std_gain,
            "pass_rule_a": pass_rule_a,
            "pass_rule_b": pass_rule_b,
            "qualifies": qualifies,
        }
        if qualifies:
            qualified.append(row.slot)
    payload = _stage_payload(
        "stage2_pairwise",
        "complete",
        ranked_slots=[row.slot for row in ranked],
        best_base_slot=base["slot"],
        qualified_slots=qualified,
        slot_decisions=reasoning,
    )
    _json_dump(exp_root / "stage2_pairwise_analysis.json", payload)
    return payload


def resolve_stage3(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    stage1 = summarize_stage1(args)
    stage2 = summarize_stage2(args)
    if stage1["status"] != "complete":
        reason = stage1.get("reason", "Stage 1 incomplete.")
        _append_jsonl(
            exp_root / "command_log.jsonl",
            {
                "timestamp": _now_iso(),
                "status": "blocked",
                "stage": "stage3_final",
                "reason": reason,
            },
        )
        raise RuntimeError(reason)

    base = stage1["best_base_config"]
    base_features = [item for item in base["features"].split(",") if item]
    candidates: list[tuple[str, list[str]]] = [("best_base_config", base_features)]
    if stage2.get("status") == "complete":
        stage2_manifest = _read_json(exp_root / "experiment_manifest.stage2_pairwise.json")
        lookup = {entry["slot"]: list(entry["features"]) for entry in stage2_manifest["runs"]}
        for slot in stage2.get("qualified_slots", []):
            candidates.append((slot, lookup[slot]))
    candidates.append(("baseline_no_edge", NO_EDGE_FEATURES))
    if base_features != SPECTRAL_ONLY_FEATURES:
        candidates.append(("spectral_only_reference", SPECTRAL_ONLY_FEATURES))

    dedup = []
    seen = set()
    for slot, features in candidates:
        key = (slot, tuple(features))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((slot, features))

    runs = [
        _build_run_entry(
            args,
            exp_root,
            stage="stage3_final",
            slot=slot,
            features=features,
            repeats=5,
            seeds=list(args.stage_seeds),
            spectral_dim=int(base["spectral_dim"]),
            fusion=str(base["fusion"]),
            alpha=float(base["alpha"]),
            hidden=int(base["hidden"]),
            wandb_job_type="typepair_edgeprompt_followup_final",
            hypothesis=f"Final confirmation for {slot} against best_base, baseline, and spectral_only reference.",
            compare_to="best_base_config, baseline_no_edge, and spectral_only_reference",
            decision_rule="Final ranking uses seed_macro_mean, then lower seed_macro_std, then higher seed_micro_mean.",
        )
        for slot, features in dedup
    ]
    payload = {
        "stage": "stage3_final",
        "status": "planned",
        "created_at": _now_iso(),
        "objective": "Final confirmation with strict repeats.",
        "best_base_slot": base["slot"],
        "runs": runs,
    }
    path = exp_root / "experiment_manifest.stage3_final.json"
    _write_manifest(path, payload)
    return path


def summarize_stage3(args) -> dict[str, Any]:
    exp_root = args.results_root / args.experiment_name
    stage1 = summarize_stage1(args)
    if stage1["status"] != "complete":
        payload = _stage_payload(
            "stage3_final",
            "pending",
            reason="Stage 1 is incomplete, so Stage 3 cannot be evaluated.",
            stage1=stage1,
        )
        _json_dump(exp_root / "stage3_final_analysis.json", payload)
        _write_csv(exp_root / "stage3_final_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    manifest = _read_json(exp_root / "experiment_manifest.stage3_final.json")
    if not manifest.get("runs"):
        payload = _stage_payload(
            "stage3_final",
            "pending",
            reason=manifest.get("pending_reason", "Stage 3 manifest is still waiting on Stage 2."),
        )
        _json_dump(exp_root / "stage3_final_analysis.json", payload)
        _write_csv(exp_root / "stage3_final_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload
    missing = _missing_runs(manifest)
    if missing:
        payload = _stage_payload(
            "stage3_final",
            "pending",
            reason=f"Missing overall_summary.json for slots: {', '.join(missing)}",
            missing_slots=missing,
        )
        _json_dump(exp_root / "stage3_final_analysis.json", payload)
        _write_csv(exp_root / "stage3_final_summary.csv", [], fieldnames=SUMMARY_FIELDNAMES)
        return payload

    rows = [_collect_summary(entry) for entry in manifest["runs"]]
    ranked = _sort_results(rows)
    _add_reference_deltas(ranked)
    _write_csv(exp_root / "stage3_final_summary.csv", [_summary_to_row(row) for row in ranked], fieldnames=SUMMARY_FIELDNAMES)

    best = ranked[0]
    base = stage1["best_base_config"]
    base_slot = base["slot"]
    best_vs_base_macro = best.seed_macro_mean - float(base["seed_macro_mean"])
    stable_auxiliary_feature = best.slot not in {"best_base_config", "baseline_no_edge", "spectral_only_reference"} and best_vs_base_macro > 0.0
    spectral_reference = next((row for row in ranked if row.features == SPECTRAL_ONLY_FEATURES), None)
    if spectral_reference is None and base["features"] == "SpectralEmbeddingDiff":
        spectral_delta = 0.0
    else:
        spectral_delta = None if spectral_reference is None else best.seed_macro_mean - spectral_reference.seed_macro_mean
    recommend_learnable_prompt = bool(
        stable_auxiliary_feature and spectral_delta is not None and spectral_delta >= 0.003 and best.seed_macro_std <= float(base["seed_macro_std"])
    )
    payload = _stage_payload(
        "stage3_final",
        "complete",
        ranked_slots=[row.slot for row in ranked],
        best_final_config=_summary_to_row(best),
        best_base_slot=base_slot,
        stable_auxiliary_feature=stable_auxiliary_feature,
        recommend_learnable_prompt=recommend_learnable_prompt,
        rationale={
            "best_vs_best_base_seed_macro": best_vs_base_macro,
            "best_vs_spectral_only_seed_macro": spectral_delta,
            "best_seed_macro_std": best.seed_macro_std,
            "best_base_seed_macro_std": float(base["seed_macro_std"]),
        },
    )
    _json_dump(exp_root / "stage3_final_analysis.json", payload)
    return payload


def _analysis_or_pending(path: Path, stage: str) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return _stage_payload(stage, "pending", reason="Analysis file not created yet.")


def _report_stage_line(stage: str, analysis: dict[str, Any]) -> str:
    status = analysis.get("status", "pending")
    if status != "complete":
        reason = analysis.get("reason", "not executed yet")
        return f"- {stage}: pending ({reason})"
    if stage == "stage0_confirm":
        winner = analysis["best_spectral_centered_base_candidate"]
        return f"- {stage}: best spectral-centered base candidate = `{winner}`"
    if stage == "stage1":
        best = analysis["best_base_config"]
        return (
            f"- {stage}: best base = `{best['slot']}` | macro={best['seed_macro_mean']:.4f} "
            f"| std={best['seed_macro_std']:.4f}"
        )
    if stage == "stage2_pairwise":
        qualified = analysis.get("qualified_slots", [])
        if qualified:
            return f"- {stage}: qualified pairwise configs = {', '.join(f'`{slot}`' for slot in qualified)}"
        return f"- {stage}: no pairwise config passed the qualification gate"
    if stage == "stage3_final":
        best = analysis["best_final_config"]
        return (
            f"- {stage}: final winner = `{best['slot']}` | macro={best['seed_macro_mean']:.4f} "
            f"| std={best['seed_macro_std']:.4f}"
        )
    return f"- {stage}: complete"


def _next_code_location() -> list[str]:
    return [
        f"[src/gpbench/protocol_bridge/hgmp_typepair.py]({ROOT / 'src/gpbench/protocol_bridge/hgmp_typepair.py'}:116) `TypePairRelationPrompt._fuse_edge_prompt`",
        f"[src/gpbench/protocol_bridge/downstream_legacy.py]({ROOT / 'src/gpbench/protocol_bridge/downstream_legacy.py'}:411) `train_typepair_prompt_probe`",
    ]


def write_conclusion_markdown(args):
    exp_root = args.results_root / args.experiment_name
    stage0 = _analysis_or_pending(exp_root / "stage0_confirm_analysis.json", "stage0_confirm")
    stage1 = _analysis_or_pending(exp_root / "stage1_analysis.json", "stage1")
    stage2 = _analysis_or_pending(exp_root / "stage2_pairwise_analysis.json", "stage2_pairwise")
    stage3 = _analysis_or_pending(exp_root / "stage3_final_analysis.json", "stage3_final")

    lines = [
        "# Spectral Follow-up Conclusion",
        "",
        f"- Experiment: `{args.experiment_name}`",
        f"- Updated: `{_now_iso()}`",
        f"- W&B project/entity/mode: `{args.wandb_project}` / `{args.wandb_entity or 'None'}` / `{args.wandb_mode}`",
        "",
        "## Starting Point",
        "",
        "- Stage A top ranks were treated as fixed prior evidence: `config_07`, `config_05`, `config_11`, `config_08`.",
        "- Stage B already showed `best_single_feature > baseline_no_edge > type_memory_only > structural_residual_only > edgeprompt2_core` with `support_combined=false`.",
        "- This follow-up therefore stays spectral-centered and does not reopen all-feature search.",
        "",
        "## Stage Outcomes",
        "",
        _report_stage_line("stage0_confirm", stage0),
        _report_stage_line("stage1", stage1),
        _report_stage_line("stage2_pairwise", stage2),
        _report_stage_line("stage3_final", stage3),
        "",
        "## Required Answers",
        "",
    ]

    if stage3.get("status") == "complete":
        best = stage3["best_final_config"]
        lines.append(
            f"- Best final configuration: `{best['slot']}` with features `{best['features']}` "
            f"(seed_macro_mean={best['seed_macro_mean']:.4f}, seed_macro_std={best['seed_macro_std']:.4f})."
        )
        lines.append(
            f"- Delta vs baseline_no_edge: {best['delta_seed_macro_vs_baseline_no_edge']:+.4f} macro, "
            f"{best['delta_seed_micro_vs_baseline_no_edge']:+.4f} micro."
        )
        if best["delta_seed_macro_vs_spectral_only"] is not None:
            lines.append(
                f"- Delta vs spectral_only: {best['delta_seed_macro_vs_spectral_only']:+.4f} macro, "
                f"{best['delta_seed_micro_vs_spectral_only']:+.4f} micro."
            )
        if stage3.get("stable_auxiliary_feature"):
            lines.append("- Stable auxiliary feature exists: yes.")
        else:
            lines.append("- Stable auxiliary feature exists: no.")
        if stage3.get("recommend_learnable_prompt"):
            lines.append("- Recommendation on learnable prompt generator: worth entering next.")
        else:
            lines.append("- Recommendation on learnable prompt generator: stay with hand-crafted spectral-centered prompts for now.")
    else:
        lines.extend(
            [
                "- Best final configuration: pending Stage 3 execution.",
                "- Delta vs baseline_no_edge: pending.",
                "- Delta vs spectral_only: pending.",
                "- Recommendation on learnable prompt generator: pending final confirmation.",
            ]
        )

    lines.extend(
        [
            "",
            "## Next Code Location",
            "",
            f"- {_next_code_location()[0]}",
            f"- {_next_code_location()[1]}",
        ]
    )

    (exp_root / "followup_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_report(args) -> Path:
    exp_root = args.results_root / args.experiment_name
    report = {
        "manifest_initial": str(exp_root / "experiment_manifest.initial.json"),
        "manifest_stage0_confirm": str(exp_root / "experiment_manifest.stage0_confirm.json"),
        "manifest_stage1_coarse": str(exp_root / "experiment_manifest.stage1_coarse.json"),
        "manifest_stage1_confirm": str(exp_root / "experiment_manifest.stage1_confirm.json"),
        "manifest_stage2_pairwise": str(exp_root / "experiment_manifest.stage2_pairwise.json"),
        "manifest_stage3_final": str(exp_root / "experiment_manifest.stage3_final.json"),
        "stage0_confirm_summary": str(exp_root / "stage0_confirm_summary.csv"),
        "stage1_coarse_summary": str(exp_root / "stage1_coarse_summary.csv"),
        "stage1_confirm_summary": str(exp_root / "stage1_confirm_summary.csv"),
        "stage1_summary": str(exp_root / "stage1_summary.csv"),
        "stage2_pairwise_summary": str(exp_root / "stage2_pairwise_summary.csv"),
        "stage3_final_summary": str(exp_root / "stage3_final_summary.csv"),
        "stage0_confirm_analysis": str(exp_root / "stage0_confirm_analysis.json"),
        "stage1_coarse_analysis": str(exp_root / "stage1_coarse_analysis.json"),
        "stage1_confirm_analysis": str(exp_root / "stage1_confirm_analysis.json"),
        "stage1_analysis": str(exp_root / "stage1_analysis.json"),
        "stage2_pairwise_analysis": str(exp_root / "stage2_pairwise_analysis.json"),
        "stage3_final_analysis": str(exp_root / "stage3_final_analysis.json"),
        "command_log": str(exp_root / "command_log.jsonl"),
        "followup_conclusion": str(exp_root / "followup_conclusion.md"),
    }
    _json_dump(exp_root / "report_index.json", report)
    write_conclusion_markdown(args)
    return exp_root / "report_index.json"


def run_stage0(args):
    exp_root = args.results_root / args.experiment_name
    manifest = _read_json(exp_root / "experiment_manifest.stage0_confirm.json")
    for entry in manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)
    summarize_stage0(args)
    finalize_report(args)


def run_stage1(args):
    exp_root = args.results_root / args.experiment_name
    resolve_stage1_coarse(args)
    coarse_manifest = _read_json(exp_root / "experiment_manifest.stage1_coarse.json")
    for entry in coarse_manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)
    summarize_stage1_coarse(args)
    resolve_stage1_confirm(args)
    confirm_manifest = _read_json(exp_root / "experiment_manifest.stage1_confirm.json")
    for entry in confirm_manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)
    summarize_stage1(args)
    finalize_report(args)


def run_stage2(args):
    exp_root = args.results_root / args.experiment_name
    resolve_stage2(args)
    manifest = _read_json(exp_root / "experiment_manifest.stage2_pairwise.json")
    for entry in manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)
    summarize_stage2(args)
    finalize_report(args)


def run_stage3(args):
    exp_root = args.results_root / args.experiment_name
    resolve_stage3(args)
    manifest = _read_json(exp_root / "experiment_manifest.stage3_final.json")
    for entry in manifest["runs"]:
        _run_entry(entry, exp_root=exp_root, force=args.force)
    summarize_stage3(args)
    finalize_report(args)


def summarize_all(args):
    exp_root = args.results_root / args.experiment_name
    stage0_path = exp_root / "experiment_manifest.stage0_confirm.json"
    if stage0_path.exists():
        summarize_stage0(args)
    stage1_coarse_path = exp_root / "experiment_manifest.stage1_coarse.json"
    if stage1_coarse_path.exists():
        summarize_stage1_coarse(args)
    stage1_confirm_path = exp_root / "experiment_manifest.stage1_confirm.json"
    if stage1_confirm_path.exists():
        summarize_stage1(args)
    stage2_path = exp_root / "experiment_manifest.stage2_pairwise.json"
    if stage2_path.exists():
        summarize_stage2(args)
    stage3_path = exp_root / "experiment_manifest.stage3_final.json"
    if stage3_path.exists():
        summarize_stage3(args)
    finalize_report(args)


def build_parser():
    parser = argparse.ArgumentParser("Spectral-centered edgeprompt2 follow-up driver")
    parser.add_argument("action", choices=["prepare", "run_stage0", "run_stage1", "run_stage2", "run_stage3", "summarize", "full"])
    parser.add_argument("--experiment_name", type=str, default="acm10_typepair_edgeprompt2_spectral_followup")
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--python_exec", type=str, default=sys.executable)

    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--splits", type=str, default="splits")
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--pretrain_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)

    parser.add_argument("--hgnn_type", type=str, default="GCN")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--classification_type", type=str, default="NIG")
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--prompt_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_metric", type=str, default="macro")

    parser.add_argument("--hgmp_ckpt", type=str, default="artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth")
    parser.add_argument("--typepair_ckpt", type=str, default="artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth")
    parser.add_argument("--hgprompt_ckpt", type=str, default="artifacts/checkpoints/hgprompt/pretrain/ACM.gcn.ft2.hop1.seed0.best.pt")

    parser.add_argument("--stage_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_project", type=str, default="HGEP")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        default=["ACM", "10-shot", "typepair", "edgeprompt2", "followup", "spectral"],
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.action == "prepare":
        prepare(args)
        return

    exp_root = args.results_root / args.experiment_name
    if not (exp_root / "experiment_manifest.initial.json").exists():
        prepare(args)

    if args.action == "run_stage0":
        run_stage0(args)
        return
    if args.action == "run_stage1":
        run_stage1(args)
        return
    if args.action == "run_stage2":
        run_stage2(args)
        return
    if args.action == "run_stage3":
        run_stage3(args)
        return
    if args.action == "summarize":
        summarize_all(args)
        return
    if args.action == "full":
        run_stage0(args)
        run_stage1(args)
        run_stage2(args)
        run_stage3(args)
        summarize_all(args)
        return
    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
