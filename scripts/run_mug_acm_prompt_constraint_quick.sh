#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/logs/mug_peprompt/prompt_constraints_quick
mkdir -p artifacts/results/mug_peprompt/acm_mug_fixed_split_prompt_constraints_quick

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_one() {
  local constraint="$1"
  local scale="$2"
  local tag="${constraint}_s${scale//./p}"
  local out_dir="artifacts/results/mug_peprompt/acm_mug_fixed_split_prompt_constraints_quick/${tag}"
  local log_path="artifacts/logs/mug_peprompt/prompt_constraints_quick/acm_fixed_${tag}.log"

  mkdir -p "${out_dir}"
  /home_A/yuanqilin/.conda/envs/HGEP/bin/python -u scripts/mug_peprompt_benchmark.py \
    --dataset acm \
    --embedding_path artifacts/cache/mug_embeddings/acm_mug_seed0_epoch50.pt \
    --downstream_protocol mug_fixed_split \
    --mug_ratio 60 \
    --mug_epochs 50 \
    --seed 0 \
    --gpu 0 \
    --shot 1 \
    --split_seeds 0 \
    --repeats 3 \
    --methods mug_type_neighborhood_edge \
    --eval_epochs 50 \
    --type_hops 0 1 2 3 \
    --max_edges_per_metapath 0 \
    --relation_prompt_mode mul \
    --relation_prompt_alpha 0.5 \
    --relation_prompt_dropout 0.1 \
    --relation_prompt_aggr mean \
    --no-relation_prompt_use_ln \
    --relation_prompt_constraint "${constraint}" \
    --relation_prompt_constraint_scale "${scale}" \
    --peprompt_edge_prompt_hidden 128 \
    --peprompt_edge_chunk_size 200000 \
    --out_dir "${out_dir}" \
    2>&1 | tee "${log_path}"
}

run_one none 0.5
run_one identity_tanh 0.25
run_one identity_tanh 0.5
run_one positive_sigmoid 0.5
run_one identity_l2norm 4.0
