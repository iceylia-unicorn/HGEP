#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASETS=(${DATASETS:-DBLP IMDB})
SHOT="${SHOT:-10}"
SEEDS=(${SEEDS:-0 1 2 3 4})
REPEATS="${REPEATS:-1}"
DEVICE="${DEVICE:-cuda}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
LOG_DIR="${LOG_DIR:-artifacts/logs/peprompt_khop_fanout_suite}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/peprompt_khop_fanout_suite}"
DATASETS_STR="${DATASETS[*]}"

CKPT_DBLP="${CKPT_DBLP:-artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth}"
CKPT_IMDB="${CKPT_IMDB:-artifacts/checkpoints/hgmp/pretrain/IMDB.GraphCL.GCN.hid256.np200.seed0.pth}"
mkdir -p "$LOG_DIR" "$RESULT_ROOT"

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_ARGS=(
    --use_wandb
    --wandb_mode "$WANDB_MODE"
  )
fi

ckpt_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    DBLP) echo "$CKPT_DBLP" ;;
    IMDB) echo "$CKPT_IMDB" ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      return 1
      ;;
  esac
}

khop_num_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    DBLP) echo 2 ;;
    IMDB) echo 2 ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      return 1
      ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  ckpt="$(ckpt_for_dataset "$dataset")"
  khop_num="$(khop_num_for_dataset "$dataset")"

  echo "[dataset=$dataset] precompute khop"
  python scripts/precompute_peprompt_cache.py \
    --datasets "$dataset" \
    --shots "$SHOT" \
    --seeds "${SEEDS[@]}" \
    --max_pool_size 400 \
    --subgraph_type khop \
    --khop_num "$khop_num" \
    | tee "$LOG_DIR/${dataset}.precompute.khop.log"

  echo "[dataset=$dataset] precompute fanout"
  python scripts/precompute_peprompt_cache.py \
    --datasets "$dataset" \
    --shots "$SHOT" \
    --seeds "${SEEDS[@]}" \
    --max_pool_size 400 \
    --subgraph_type fanout \
    --fanouts 15 10 \
    | tee "$LOG_DIR/${dataset}.precompute.fanout.log"

  echo "[dataset=$dataset] benchmark khop"
  python scripts/peprompt_benchmark.py \
    --dataset "$dataset" \
    --methods peprompt \
    --seeds "${SEEDS[@]}" \
    --repeats "$REPEATS" \
    --shot "$SHOT" \
    --device "$DEVICE" \
    --peprompt_ckpt "$ckpt" \
    --subgraph_type khop \
    --save_dir "$RESULT_ROOT" \
    --wandb_name "peprompt_${dataset,,}_khop_suite" \
    --wandb_tags PEPrompt "$dataset" khop "${SHOT}-shot" \
    "${WANDB_ARGS[@]}" \
    | tee "$LOG_DIR/${dataset}.benchmark.khop.log"

  echo "[dataset=$dataset] benchmark fanout"
  python scripts/peprompt_benchmark.py \
    --dataset "$dataset" \
    --methods peprompt \
    --seeds "${SEEDS[@]}" \
    --repeats "$REPEATS" \
    --shot "$SHOT" \
    --device "$DEVICE" \
    --peprompt_ckpt "$ckpt" \
    --subgraph_type fanout \
    --save_dir "$RESULT_ROOT" \
    --wandb_name "peprompt_${dataset,,}_fanout_suite" \
    --wandb_tags PEPrompt "$dataset" fanout "${SHOT}-shot" \
    "${WANDB_ARGS[@]}" \
    | tee "$LOG_DIR/${dataset}.benchmark.fanout.log"
done

export RESULT_ROOT SHOT DATASETS_STR
python - <<'PY'
from __future__ import annotations
from pathlib import Path
import json
import os

result_root = Path(os.environ.get("RESULT_ROOT", "artifacts/results/peprompt_khop_fanout_suite"))
datasets = os.environ.get("DATASETS_STR", "DBLP IMDB").split()
shot = os.environ.get("SHOT", "10")

print("dataset,khop_micro,fanout_micro,delta_micro,khop_macro,fanout_macro,delta_macro")
for dataset in datasets:
    base = result_root / dataset / f"{shot}-shot"
    khop_path = base / "peprompt.khop" / "overall_summary.json"
    fanout_path = base / "peprompt.fanout" / "overall_summary.json"
    if not khop_path.exists() or not fanout_path.exists():
        print(f"{dataset},MISSING,MISSING,MISSING,MISSING,MISSING,MISSING")
        continue

    with open(khop_path, "r", encoding="utf-8") as f:
        khop = json.load(f)["seed_mean_then_std"]["peprompt"]
    with open(fanout_path, "r", encoding="utf-8") as f:
        fanout = json.load(f)["seed_mean_then_std"]["peprompt"]

    khop_micro = float(khop["micro_mean"])
    fanout_micro = float(fanout["micro_mean"])
    khop_macro = float(khop["macro_mean"])
    fanout_macro = float(fanout["macro_mean"])
    print(
        f"{dataset},"
        f"{khop_micro:.6f},{fanout_micro:.6f},{(fanout_micro-khop_micro):+.6f},"
        f"{khop_macro:.6f},{fanout_macro:.6f},{(fanout_macro-khop_macro):+.6f}"
    )
PY
