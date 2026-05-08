from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts import protocol_benchmark_v2 as benchmark_v2
from gpbench.utils.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_metrics,
    log_nested_summary,
    log_table,
    maybe_configure_wandb_env,
    upload_file_artifact,
)


DEFAULT_SHOTS = [1, 3, 5, 10, 20]


def _normalize_shots(shots: list[int] | None) -> list[int]:
    values = shots if shots else DEFAULT_SHOTS
    deduped = []
    seen = set()
    for shot in values:
        shot = int(shot)
        if shot <= 0:
            raise ValueError(f"Shot must be positive, got {shot}")
        if shot in seen:
            continue
        seen.add(shot)
        deduped.append(shot)
    return deduped


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _with_shot_suffix(value: str | None, shot: int) -> str | None:
    if not value:
        return value
    return f"{value}-shot{shot}"


def _build_multishot_wandb_config(args, shots: list[int]) -> dict:
    config = vars(args).copy()
    config.pop("wandb_key", None)
    config["shots"] = list(shots)
    config["multishot_summary_dir"] = str(args.multishot_summary_dir) if args.multishot_summary_dir else None
    return _json_ready(config)


def build_parser():
    ap = benchmark_v2.build_parser()
    ap.description = "Run protocol benchmark across multiple shot settings"
    ap.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=None,
        help="List of shot values to run. Defaults to 1 3 5 10 20.",
    )
    ap.add_argument(
        "--multishot_summary_dir",
        type=Path,
        default=None,
        help="Directory for cross-shot summary files. Defaults to <save_dir>/<dataset>/multi-shot.",
    )
    return ap


def main():
    args = build_parser().parse_args()
    shots = _normalize_shots(args.shots)

    multi_out_dir = args.multishot_summary_dir
    if multi_out_dir is None:
        multi_out_dir = Path(args.save_dir) / args.dataset / "multi-shot"
    multi_out_dir.mkdir(parents=True, exist_ok=True)
    args.multishot_summary_dir = multi_out_dir

    shot_summaries = []
    compare_rows = []
    shot_run_refs = []

    for shot in shots:
        shot_args = deepcopy(args)
        shot_args.shot = shot
        shot_args.shots = None
        shot_args.multishot_summary_dir = None
        if shot_args.use_wandb:
            shot_args.wandb_name = _with_shot_suffix(shot_args.wandb_name, shot)
            shot_args.wandb_group = shot_args.wandb_group or f"{shot_args.dataset}-multishot"
            shot_args.wandb_tags = list(dict.fromkeys([*shot_args.wandb_tags, "multishot"]))

        print(f"================ MULTI-SHOT | shot={shot} ================")
        result = benchmark_v2.run_benchmark(shot_args)
        shot_summary = {
            "shot": shot,
            "out_dir": str(result["out_dir"]),
            "resolved_ckpt_by_method": result["resolved_ckpt_by_method"],
            "pooled_runs": result["pooled_runs"],
            "seed_mean_then_std": result["seed_mean_then_std"],
        }
        shot_summaries.append(_json_ready(shot_summary))
        shot_run_refs.append(
            {
                "shot": shot,
                "wandb_group": shot_args.wandb_group,
                "wandb_name": shot_args.wandb_name,
                "out_dir": str(result["out_dir"]),
            }
        )

        methods = sorted(
            set(result["pooled_runs"].keys()) | set(result["seed_mean_then_std"].keys())
        )
        for method in methods:
            pooled = result["pooled_runs"].get(method, {})
            seedmean = result["seed_mean_then_std"].get(method, {})
            compare_rows.append(
                {
                    "shot": shot,
                    "method": method,
                    "pooled_count": pooled.get("count", 0),
                    "pooled_micro_mean": pooled.get("micro_mean"),
                    "pooled_micro_std": pooled.get("micro_std"),
                    "pooled_macro_mean": pooled.get("macro_mean"),
                    "pooled_macro_std": pooled.get("macro_std"),
                    "seedmean_count": seedmean.get("count", 0),
                    "seedmean_micro_mean": seedmean.get("micro_mean"),
                    "seedmean_micro_std": seedmean.get("micro_std"),
                    "seedmean_macro_mean": seedmean.get("macro_mean"),
                    "seedmean_macro_std": seedmean.get("macro_std"),
                    "out_dir": str(result["out_dir"]),
                }
            )

    benchmark_v2._write_csv(multi_out_dir / "multishot_compare.csv", compare_rows)

    summary_payload = {
        "dataset": args.dataset,
        "shots": shots,
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "repeats": int(args.repeats),
        "pretrain_seed": int(args.pretrain_seed),
        "save_dir": str(args.save_dir),
        "multi_shot_results": shot_summaries,
    }
    with open(multi_out_dir / "multishot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    if args.use_wandb:
        summary_run = None
        try:
            maybe_configure_wandb_env(api_key=args.wandb_key, env_file=args.wandb_key_file)
            summary_run = init_wandb_run(
                enabled=True,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=(f"{args.wandb_name}-multishot-summary" if args.wandb_name else f"{args.dataset}-multishot-summary"),
                group=args.wandb_group or f"{args.dataset}-multishot",
                job_type=f"{args.wandb_job_type}_multishot_summary",
                tags=list(dict.fromkeys([*(args.wandb_tags or [args.dataset, "protocol_multishot_eval"]), "multishot"])),
                notes=args.wandb_notes,
                mode=args.wandb_mode,
                dir_path=args.wandb_dir,
                config=_build_multishot_wandb_config(args, shots),
            )
            log_metrics(
                summary_run,
                {
                    "multishot/num_shots": len(shots),
                    "multishot/num_methods": len(args.methods),
                    "multishot/num_seeds": len(args.seeds),
                    "multishot/repeats": int(args.repeats),
                    "multishot/pretrain_seed": int(args.pretrain_seed),
                },
            )
            log_table(summary_run, "multishot_compare_table", compare_rows)
            log_table(summary_run, "multishot_run_refs", shot_run_refs)
            log_nested_summary(summary_run, "multishot_summary", summary_payload)
            upload_file_artifact(
                summary_run,
                multi_out_dir / "multishot_compare.csv",
                name=f"{args.dataset}-multishot-compare",
                artifact_type="evaluation",
            )
            upload_file_artifact(
                summary_run,
                multi_out_dir / "multishot_summary.json",
                name=f"{args.dataset}-multishot-summary",
                artifact_type="evaluation",
            )
        except Exception:
            finish_wandb_run(summary_run, exit_code=1)
            raise
        else:
            finish_wandb_run(summary_run, exit_code=0)

    print("####################################################")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
