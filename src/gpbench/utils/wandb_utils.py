from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install -e .[wandb]` or `pip install wandb`."
        ) from exc
    return wandb


def _coerce_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" in stripped:
        key, value = stripped.split("=", 1)
        return key.strip(), value.strip().strip("'\"")

    if stripped.startswith("wandb_") and " " not in stripped:
        return "WANDB_API_KEY", stripped
    return None


def load_env_file(path: str | Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return env_map

    for line in env_path.read_text(encoding="utf-8").splitlines():
        pair = _coerce_env_line(line)
        if pair is None:
            continue
        key, value = pair
        env_map[key] = value
    return env_map


def maybe_configure_wandb_env(api_key: str | None = None, env_file: str | Path | None = None) -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return "env"

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key.strip()
        return "argument"

    if env_file is None:
        return None

    env_map = load_env_file(env_file)
    key = env_map.get("WANDB_API_KEY")
    if key:
        os.environ["WANDB_API_KEY"] = key
        return str(env_file)
    return None


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    config: dict[str, Any],
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str = "online",
    dir_path: str | Path | None = None,
):
    if not enabled:
        return None

    wandb = _import_wandb()
    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": tags or None,
        "notes": notes,
        "mode": mode,
        "config": config,
    }
    if dir_path is not None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(dir_path)
    return wandb.init(**init_kwargs)


def log_run_record(run, record: Any, step: int | None = None, extra: dict[str, Any] | None = None):
    if run is None:
        return
    payload = {
        "per_run/method": record.method,
        "per_run/split_seed": record.split_seed,
        "per_run/repeat_id": record.repeat_id,
        "per_run/run_seed": record.run_seed,
        "per_run/test_micro": record.test_micro,
        "per_run/test_macro": record.test_macro,
        "per_run/best_epoch": record.best_epoch,
    }
    if extra:
        payload.update(extra)
    log_metrics(run, payload, step=step)


def log_metrics(run, payload: dict[str, Any], step: int | None = None):
    if run is None:
        return
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=step)


def log_table(run, name: str, rows: list[dict[str, Any]]):
    if run is None or not rows:
        return
    wandb = _import_wandb()
    columns = list(rows[0].keys())
    data = [[row.get(col) for col in columns] for row in rows]
    run.log({name: wandb.Table(columns=columns, data=data)})


def _flatten_dict(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten_dict(next_prefix, child))
        return out
    return {prefix: value}


def log_nested_summary(run, prefix: str, payload: dict[str, Any]):
    if run is None:
        return
    flattened = _flatten_dict(prefix, payload)
    for key, value in flattened.items():
        run.summary[key] = value


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return slug or "artifact"


def upload_dir_artifact(run, path: str | Path, name: str, artifact_type: str = "results"):
    if run is None:
        return
    wandb = _import_wandb()
    artifact = wandb.Artifact(name=_slugify(name), type=artifact_type)
    artifact.add_dir(str(path))
    run.log_artifact(artifact)


def upload_file_artifact(run, path: str | Path, name: str, artifact_type: str = "model"):
    if run is None:
        return
    file_path = Path(path)
    if not file_path.exists():
        return
    wandb = _import_wandb()
    artifact = wandb.Artifact(name=_slugify(name), type=artifact_type)
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)


def finish_wandb_run(run, exit_code: int = 0):
    if run is None:
        return
    run.finish(exit_code=exit_code)
