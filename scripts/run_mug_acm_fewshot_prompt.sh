#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/logs/mug_peprompt
mkdir -p artifacts/results/mug_peprompt/acm_mug_fewshot_from_train_r60_mug50_full_edges

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

/home_A/yuanqilin/.conda/envs/HGEP/bin/python -u scripts/mug_peprompt_benchmark.py \
  --dataset acm \
  --embedding_path artifacts/cache/mug_embeddings/acm_mug_seed0_epoch50.pt \
  --downstream_protocol mug_fewshot_from_train \
  --mug_ratio 60 \
  --mug_epochs 50 \
  --seed 0 \
  --gpu 0 \
  --shot 1 \
  --split_seeds 0 \
  --repeats 100 \
  --methods mug mug_type_neighborhood_edge \
  --fewshot_train_epochs 100 \
  --type_hops 0 1 2 3 \
  --max_edges_per_metapath 0 \
  --relation_prompt_mode mul \
  --relation_prompt_alpha 0.5 \
  --relation_prompt_dropout 0.1 \
  --relation_prompt_aggr mean \
  --no-relation_prompt_use_ln \
  --peprompt_edge_prompt_hidden 128 \
  --peprompt_edge_chunk_size 50000 \
  --out_dir artifacts/results/mug_peprompt/acm_mug_fewshot_from_train_r60_mug50_full_edges \
  2>&1 | tee artifacts/logs/mug_peprompt/acm_mug_fewshot_from_train_r60_mug50_full_edges.log
