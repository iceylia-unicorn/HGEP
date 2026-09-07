#!/usr/bin/env python3
"""Run PEPrompt ablation suites.

This script intentionally stays outside ``peprompt_benchmark.py``.  It builds
reproducible precompute/benchmark commands for the current PE-only PEPrompt
line and keeps each ablation's cache/results/logs isolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = sys.executable


@dataclass(frozen=True)
class DatasetProfile:
    feats_type: int
    hidden_dim: int
    hgnn_type: str
    num_samples: int
    ckpt: str
    cache_args: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    description: str
    cache_args: tuple[str, ...]
    prompt_args: tuple[str, ...] = ()
    needs_precompute: bool = True


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "DBLP": DatasetProfile(
        feats_type=0,
        hidden_dim=512,
        hgnn_type="GCN",
        num_samples=500,
        ckpt="artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth",
        cache_args=(
            "--subgraph_type", "metapath_topk_adapt",
            "--metapath_max_hop", "4",
            "--metapath_min_topk", "1",
            "--metapath_max_topk", "8",
            "--metapath_rel_threshold", "0.5",
            "--metapath_rank_metric", "count",
            "--metapath_endpoint_mode", "all",
            "--metapath_support_mode", "count",
            "--metapath_support_topk", "16",
            "--metapath_support_rank_mode", "score",
        ),
    ),
    "ACM": DatasetProfile(
        feats_type=0,
        hidden_dim=512,
        hgnn_type="GCN",
        num_samples=500,
        ckpt="artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth",
        cache_args=(
            "--subgraph_type", "metapath_topk_adapt",
            "--metapath_max_hop", "3",
            "--metapath_min_topk", "1",
            "--metapath_max_topk", "5",
            "--metapath_rel_threshold", "0.5",
            "--metapath_rank_metric", "count",
            "--metapath_endpoint_mode", "all",
            "--metapath_support_mode", "none",
        ),
    ),
    "IMDB": DatasetProfile(
        feats_type=0,
        hidden_dim=512,
        hgnn_type="GCN",
        num_samples=500,
        ckpt="artifacts/checkpoints/hgmp/pretrain/IMDB.GraphCL.GCN.hid512.np500.seed0.pth",
        cache_args=(
            "--subgraph_type", "metapath_topk_adapt",
            "--metapath_max_hop", "2",
            "--metapath_min_topk", "1",
            "--metapath_max_topk", "5",
            "--metapath_rel_threshold", "0.5",
            "--metapath_rank_metric", "degree_norm",
            "--metapath_endpoint_mode", "all",
            "--metapath_support_mode", "none",
        ),
    ),
    "Freebase": DatasetProfile(
        feats_type=1,
        hidden_dim=512,
        hgnn_type="GCN",
        num_samples=500,
        ckpt="artifacts/checkpoints/hgmp/pretrain/Freebase.GraphCL.GCN.hid512.np500.seed0.pth",
        cache_args=(
            "--subgraph_type", "metapath_topk",
            "--metapath_max_hop", "3",
            "--metapath_topk", "3",
            "--metapath_rank_metric", "count",
            "--metapath_endpoint_mode", "all",
            "--metapath_support_mode", "auto",
        ),
    ),
}


def _arg_map(args: Iterable[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    items = list(args)
    i = 0
    while i < len(items):
        key = items[i]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument token: {key}")
        if i + 1 < len(items) and not items[i + 1].startswith("--"):
            out[key] = items[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _args_from_map(values: dict[str, str | bool]) -> tuple[str, ...]:
    out: list[str] = []
    for key in sorted(values):
        value = values[key]
        out.append(key)
        if value is not True:
            out.append(str(value))
    return tuple(out)


def override_args(base: Iterable[str], **updates: str | int | float | bool | None) -> tuple[str, ...]:
    values = _arg_map(base)
    for key, value in updates.items():
        cli_key = "--" + key
        if value is None:
            values.pop(cli_key, None)
        elif value is False:
            values.pop(cli_key, None)
        elif value is True:
            values[cli_key] = True
        else:
            values[cli_key] = str(value)
    return _args_from_map(values)


def make_variants(profile: DatasetProfile, random_seeds: Iterable[int]) -> list[Variant]:
    base = profile.cache_args
    random_seed_list = [int(v) for v in random_seeds]
    variants: list[Variant] = [
        Variant(
            name="full",
            group="core",
            description="Full PEPrompt setting: PE edge prompt + selected subgraph/support strategy.",
            cache_args=base,
        ),
        Variant(
            name="no_prompt_alpha0",
            group="core",
            description="Disable PE prompt effect by setting relation_prompt_alpha=0.",
            cache_args=base,
            prompt_args=("--relation_prompt_alpha", "0.0"),
            needs_precompute=False,
        ),
        Variant(
            name="support_none",
            group="core",
            description="Remove recovered support nodes; keep endpoint selection unchanged.",
            cache_args=override_args(base, metapath_support_mode="none", metapath_support_topk=None),
        ),
        Variant(
            name="support_random_sk16",
            group="core",
            description="Keep finite support budget but choose support nodes randomly.",
            cache_args=override_args(
                base,
                metapath_support_mode="count",
                metapath_support_topk=16,
                metapath_support_rank_mode="random",
                metapath_random_seed=0,
            ),
        ),
        Variant(
            name="rank_random_r0",
            group="core",
            description="Randomize endpoint ranking with the main support setting.",
            cache_args=override_args(base, metapath_rank_metric="random", metapath_random_seed=0),
        ),
    ]

    variants.extend(
        [
            Variant(
                name="subgraph_khop",
                group="subgraph",
                description="Replace metapath-selected subgraph with default k-hop subgraph.",
                cache_args=("--subgraph_type", "khop"),
            ),
            Variant(
                name="subgraph_fixed_topk",
                group="subgraph",
                description="Use fixed top-k metapath endpoints instead of adaptive top-k.",
                cache_args=override_args(
                    base,
                    subgraph_type="metapath_topk",
                    metapath_topk=_arg_map(base).get("--metapath_max_topk", "5"),
                    metapath_min_topk=None,
                    metapath_max_topk=None,
                    metapath_rel_threshold=None,
                ),
            ),
            Variant(
                name="subgraph_path_adapt",
                group="subgraph",
                description="Use path-preserving adaptive metapath subgraph.",
                cache_args=override_args(base, subgraph_type="metapath_topk_path_adapt"),
            ),
        ]
    )

    for topk in (4, 8, 16, 32):
        variants.append(
            Variant(
                name=f"support_sk{topk}",
                group="support",
                description=f"Count-based support recovery with support_topk={topk}.",
                cache_args=override_args(
                    base,
                    metapath_support_mode="count",
                    metapath_support_topk=topk,
                    metapath_support_rank_mode="score",
                ),
            )
        )

    variants.extend(
        [
            Variant(
                name="support_unlimited",
                group="support",
                description="Count-based support recovery without finite support cap.",
                cache_args=override_args(base, metapath_support_mode="count", metapath_support_topk=0),
            ),
            Variant(
                name="rank_degree_norm",
                group="rank",
                description="Endpoint ranking by degree-normalized score.",
                cache_args=override_args(base, metapath_rank_metric="degree_norm"),
            ),
            Variant(
                name="rank_count_idf",
                group="rank",
                description="Endpoint ranking by count-idf score.",
                cache_args=override_args(base, metapath_rank_metric="count_idf"),
            ),
        ]
    )

    for seed in random_seed_list:
        variants.append(
            Variant(
                name=f"rank_random_r{seed}",
                group="rank",
                description=f"Random endpoint ranking with metapath_random_seed={seed}.",
                cache_args=override_args(base, metapath_rank_metric="random", metapath_random_seed=seed),
            )
        )

    variants.extend(
        [
            Variant(
                name="prompt_add",
                group="prompt",
                description="Use additive prompt injection instead of multiplicative injection.",
                cache_args=base,
                prompt_args=("--relation_prompt_mode", "add"),
                needs_precompute=False,
            ),
            Variant(
                name="prompt_sum_aggr",
                group="prompt",
                description="Use sum aggregation instead of mean aggregation for edge prompts.",
                cache_args=base,
                prompt_args=("--relation_prompt_aggr", "sum"),
                needs_precompute=False,
            ),
            Variant(
                name="prompt_alpha025",
                group="prompt",
                description="Lower prompt residual strength.",
                cache_args=base,
                prompt_args=("--relation_prompt_alpha", "0.25"),
                needs_precompute=False,
            ),
            Variant(
                name="prompt_alpha100",
                group="prompt",
                description="Higher prompt residual strength.",
                cache_args=base,
                prompt_args=("--relation_prompt_alpha", "1.0"),
                needs_precompute=False,
            ),
            Variant(
                name="prompt_hidden64",
                group="prompt",
                description="Smaller edge prompt MLP hidden dimension.",
                cache_args=base,
                prompt_args=("--peprompt_edge_prompt_hidden", "64"),
                needs_precompute=False,
            ),
            Variant(
                name="prompt_hidden256",
                group="prompt",
                description="Larger edge prompt MLP hidden dimension.",
                cache_args=base,
                prompt_args=("--peprompt_edge_prompt_hidden", "256"),
                needs_precompute=False,
            ),
            Variant(
                name="prompt_edge_dropout01",
                group="prompt",
                description="Randomly drop training edges with p=0.1.",
                cache_args=base,
                prompt_args=("--peprompt_edge_dropout", "0.1"),
                needs_precompute=False,
            ),
        ]
    )

    for dim in (8, 16, 32):
        variants.append(
            Variant(
                name=f"pe_dim{dim}",
                group="pe",
                description=f"Use Laplacian PE dimension {dim}.",
                cache_args=base + ("--peprompt_spectral_dim", str(dim)),
            )
        )

    deduped: dict[str, Variant] = {}
    for variant in variants:
        deduped.setdefault(variant.name, variant)
    return list(deduped.values())


def selected_variants(all_variants: list[Variant], groups: list[str], only: list[str]) -> list[Variant]:
    if only:
        wanted = set(only)
        selected = [v for v in all_variants if v.name in wanted]
        missing = sorted(wanted - {v.name for v in selected})
        if missing:
            raise SystemExit(f"Unknown variant(s): {', '.join(missing)}")
        return selected

    expanded = set(groups)
    if "all" in expanded:
        expanded = {"core", "subgraph", "support", "rank", "prompt", "pe"}
    selected = [v for v in all_variants if v.group in expanded]
    if not selected:
        raise SystemExit(f"No variants selected for groups={groups}")
    return selected


def cache_signature(dataset: str, feats_type: int, cache_args: tuple[str, ...]) -> str:
    raw = "\n".join([dataset, f"feats_type={feats_type}", *cache_args])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    name = "cache_" + digest
    values = _arg_map(cache_args)
    subgraph = str(values.get("--subgraph_type", "subgraph"))
    metric = str(values.get("--metapath_rank_metric", "na"))
    return f"{name}_{subgraph}_{metric}"


def printable_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix = ""
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
    return prefix + " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd: list[str], log_path: Path, dry_run: bool, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = printable_command(cmd, env)
    print(f"\n[command] {command_text}")
    print(f"[log] {log_path}")
    if dry_run:
        return 0

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_text}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return int(proc.wait())


def build_precompute_cmd(args, profile: DatasetProfile, variant: Variant, cache_dir: Path) -> list[str]:
    return [
        args.python,
        "-u",
        "scripts/precompute_peprompt_cache.py",
        "--datasets",
        args.dataset,
        "--shots",
        str(args.shot),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--root",
        str(args.root),
        "--feats_type",
        str(args.feats_type if args.feats_type is not None else profile.feats_type),
        "--peprompt_offline_cache_dir",
        str(cache_dir),
        "--peprompt_spectral_cache_dir",
        str(args.spectral_cache_dir),
        "--peprompt_spectral_max_nodes",
        str(args.peprompt_spectral_max_nodes),
        *variant.cache_args,
    ]


def build_benchmark_cmd(args, profile: DatasetProfile, variant: Variant, cache_dir: Path, save_dir: Path) -> list[str]:
    ckpt = args.peprompt_ckpt or profile.ckpt
    return [
        args.python,
        "-u",
        "scripts/peprompt_benchmark.py",
        "--dataset",
        args.dataset,
        "--shot",
        str(args.shot),
        "--methods",
        "peprompt",
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--repeats",
        str(args.repeats),
        "--device",
        args.device,
        "--save_dir",
        str(save_dir),
        "--root",
        str(args.root),
        "--feats_type",
        str(args.feats_type if args.feats_type is not None else profile.feats_type),
        "--peprompt_ckpt",
        ckpt,
        "--hidden_dim",
        str(args.hidden_dim if args.hidden_dim is not None else profile.hidden_dim),
        "--hgnn_type",
        args.hgnn_type or profile.hgnn_type,
        "--num_samples",
        str(args.num_samples if args.num_samples is not None else profile.num_samples),
        "--embed_batch_size",
        str(args.embed_batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--peprompt_early_stop_mode",
        args.peprompt_early_stop_mode,
        "--peprompt_eval_mode",
        args.peprompt_eval_mode,
        "--peprompt_offline_cache_dir",
        str(cache_dir),
        "--peprompt_spectral_cache_dir",
        str(args.spectral_cache_dir),
        "--peprompt_spectral_max_nodes",
        str(args.peprompt_spectral_max_nodes),
        *variant.cache_args,
        *variant.prompt_args,
    ]


def write_manifest(path: Path, variants: list[Variant], cache_dirs: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PEPrompt Ablation Manifest",
        "",
        "| variant | group | cache | description |",
        "|---|---|---|---|",
    ]
    for variant in variants:
        signature = cache_signature("DATASET", 0, variant.cache_args)
        lines.append(
            f"| `{variant.name}` | `{variant.group}` | `{cache_dirs[variant.name].name}` | {variant.description} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flatten_summary(prefix: str, value, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_summary(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, float):
        out[prefix] = f"{value:.6g}"
    else:
        out[prefix] = str(value)


def summarize_results(args, run_name: str, variants: list[Variant]) -> Path:
    rows: list[dict[str, str]] = []
    for variant in variants:
        save_dir = args.result_root / run_name / variant.name
        candidates = sorted(
            save_dir.glob(f"{args.dataset}/{args.shot}-shot/**/overall_summary.json")
        )
        row = {
            "variant": variant.name,
            "group": variant.group,
            "status": "missing",
            "summary_path": "",
            "description": variant.description,
        }
        if candidates:
            summary_path = candidates[-1]
            row["status"] = "ok"
            row["summary_path"] = str(summary_path)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            peprompt_pooled = payload.get("pooled_runs", {}).get("peprompt", {})
            peprompt_seedmean = payload.get("seed_mean_then_std", {}).get("peprompt", {})
            _flatten_summary("pooled", peprompt_pooled, row)
            _flatten_summary("seedmean", peprompt_seedmean, row)
        rows.append(row)

    out_path = args.log_root / run_name / "summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(row.get(col, "") for col in columns) + "\n")
    print(f"[summary] {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run PEPrompt ablation experiments.")
    ap.add_argument("--dataset", choices=sorted(DATASET_PROFILES), default="DBLP")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--groups", nargs="+", default=["core"], choices=["core", "subgraph", "support", "rank", "prompt", "pe", "all"])
    ap.add_argument("--only", nargs="*", default=[], help="Run only the named variants.")
    ap.add_argument("--stage", choices=["all", "precompute", "benchmark", "summarize"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--peprompt_ckpt", default=None)
    ap.add_argument("--feats_type", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--hgnn_type", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--embed_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--peprompt_early_stop_mode", choices=["metric", "loss"], default="loss")
    ap.add_argument("--peprompt_eval_mode", choices=["full", "early_stop_only"], default="early_stop_only")
    ap.add_argument("--peprompt_spectral_max_nodes", type=int, default=50000)
    ap.add_argument("--spectral_cache_dir", type=Path, default=ROOT / "artifacts" / "cache" / "peprompt_spectral_embeddings")
    ap.add_argument("--offline_cache_root", type=Path, default=ROOT / "artifacts" / "cache" / "peprompt_ablation")
    ap.add_argument("--result_root", type=Path, default=ROOT / "artifacts" / "results" / "peprompt_ablation")
    ap.add_argument("--log_root", type=Path, default=ROOT / "artifacts" / "logs" / "peprompt_ablation")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--random_ablation_seeds", nargs="+", type=int, default=[0, 1, 2])
    return ap


def main() -> None:
    args = build_parser().parse_args()
    profile = DATASET_PROFILES[args.dataset]
    run_name = args.run_name or f"{args.dataset}_{args.shot}shot_{'_'.join(args.groups)}"

    variants = selected_variants(
        make_variants(profile, args.random_ablation_seeds),
        groups=args.groups,
        only=args.only,
    )

    cache_dirs: dict[str, Path] = {}
    signature_to_cache_dir: dict[str, Path] = {}
    for variant in variants:
        feats_type = args.feats_type if args.feats_type is not None else profile.feats_type
        signature = cache_signature(args.dataset, feats_type, variant.cache_args)
        cache_dir = args.offline_cache_root / run_name / signature
        signature_to_cache_dir.setdefault(signature, cache_dir)
        cache_dirs[variant.name] = signature_to_cache_dir[signature]

    manifest_path = args.log_root / run_name / "manifest.md"
    write_manifest(manifest_path, variants, cache_dirs)
    print(f"[manifest] {manifest_path}")
    print("[variants] " + ", ".join(v.name for v in variants))

    env = {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices} if args.cuda_visible_devices else None
    completed_precompute: set[str] = set()
    failures: list[tuple[str, int]] = []

    if args.stage in {"all", "precompute"}:
        for variant in variants:
            signature = cache_signature(
                args.dataset,
                args.feats_type if args.feats_type is not None else profile.feats_type,
                variant.cache_args,
            )
            if signature in completed_precompute:
                print(f"[skip-precompute] {variant.name} reuses {cache_dirs[variant.name]}")
                continue
            if not variant.needs_precompute and signature in completed_precompute:
                continue
            completed_precompute.add(signature)
            cmd = build_precompute_cmd(args, profile, variant, cache_dirs[variant.name])
            code = run_command(
                cmd,
                args.log_root / run_name / f"precompute_{variant.name}.log",
                dry_run=args.dry_run,
                env=None,
            )
            if code != 0:
                failures.append((f"precompute:{variant.name}", code))
                if not args.continue_on_error:
                    raise SystemExit(code)

    if args.stage in {"all", "benchmark"}:
        for variant in variants:
            save_dir = args.result_root / run_name / variant.name
            cmd = build_benchmark_cmd(args, profile, variant, cache_dirs[variant.name], save_dir)
            code = run_command(
                cmd,
                args.log_root / run_name / f"benchmark_{variant.name}.log",
                dry_run=args.dry_run,
                env=env,
            )
            if code != 0:
                failures.append((f"benchmark:{variant.name}", code))
                if not args.continue_on_error:
                    raise SystemExit(code)

    if args.stage in {"all", "summarize"} and not args.dry_run:
        summarize_results(args, run_name, variants)

    if failures:
        print("[failures]")
        for name, code in failures:
            print(f"  {name}: exit={code}")
        raise SystemExit(1)

    print("[done] PEPrompt ablation runner finished.")


if __name__ == "__main__":
    main()
