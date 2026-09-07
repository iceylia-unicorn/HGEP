#!/usr/bin/env bash
set -euo pipefail

PY=/home_A/yuanqilin/.conda/envs/HGEP/bin/python
LOGDIR=artifacts/logs/peprompt_1shot_diagnosis
CKPT=artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth

mkdir -p "$LOGDIR"

echo "[$(date)] EXP1 precompute: target_closed count"
"$PY" -u scripts/precompute_peprompt_cache.py \
  --datasets DBLP \
  --shots 1 \
  --seeds 0 1 2 3 4 \
  --feats_type 0 \
  --subgraph_type metapath_topk_adapt \
  --metapath_max_hop 4 \
  --metapath_min_topk 1 \
  --metapath_max_topk 8 \
  --metapath_rel_threshold 0.5 \
  --metapath_rank_metric count \
  --metapath_endpoint_mode target_closed

echo "[$(date)] EXP1 benchmark: target_closed count"
"$PY" -u scripts/peprompt_benchmark.py \
  --dataset DBLP \
  --methods peprompt \
  --shot 1 \
  --splits 5 \
  --seeds 0 1 2 3 4 \
  --repeats 10 \
  --device cuda:1 \
  --feats_type 0 \
  --hidden_dim 512 \
  --hgnn_type GCN \
  --num_samples 500 \
  --peprompt_ckpt "$CKPT" \
  --subgraph_type metapath_topk_adapt \
  --metapath_max_hop 4 \
  --metapath_min_topk 1 \
  --metapath_max_topk 8 \
  --metapath_rel_threshold 0.5 \
  --metapath_rank_metric count \
  --metapath_endpoint_mode target_closed \
  --peprompt_head_type mlp \
  2>&1 | tee "$LOGDIR/DBLP_peprompt_mlp_1shot_m4_k1-8_a0p5_count_target_closed.log"

echo "[$(date)] done"
