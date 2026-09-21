#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/logs/type_neighborhood_edge
mkdir -p artifacts/results/type_neighborhood_edge/ACM_hop_h0-1-2-3_identity_tanh_s0p5

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

/home_A/yuanqilin/.conda/envs/HGEP/bin/python -u scripts/peprompt_benchmark.py \
  --dataset ACM \
  --root data \
  --splits splits \
  --shot 1 \
  --methods peprompt \
  --seeds 0 1 2 \
  --repeats 5 \
  --pretrain_seed 0 \
  --device cuda:0 \
  --save_dir artifacts/results/type_neighborhood_edge/ACM_hop_h0-1-2-3_identity_tanh_s0p5 \
  --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth \
  --feats_type 0 \
  --hidden_dim 512 \
  --num_heads 8 \
  --num_layers 2 \
  --dropout 0.5 \
  --hgnn_type GCN \
  --num_samples 500 \
  --num_class 3 \
  --classification_type NIG \
  --embed_batch_size 32 \
  --head_hidden 128 \
  --head_dropout 0.3 \
  --epochs 100 \
  --patience 30 \
  --lr 0.005 \
  --weight_decay 0.0005 \
  --early_stop_metric macro \
  --relation_prompt_mode mul \
  --relation_prompt_alpha 0.5 \
  --relation_prompt_dropout 0.1 \
  --relation_prompt_aggr mean \
  --relation_prompt_constraint identity_tanh \
  --relation_prompt_constraint_scale 0.5 \
  --peprompt_pretrain_family hgmp \
  --peprompt_edge_feature_names TypeNeighborhoodEdge \
  --peprompt_offline_cache_dir artifacts/cache/type_neighborhood_edge_acm_hop_ablation \
  --subgraph_type metapath_topk_adapt \
  --metapath_max_hop 3 \
  --metapath_min_topk 1 \
  --metapath_max_topk 5 \
  --metapath_rel_threshold 0.5 \
  --metapath_rank_metric count \
  --peprompt_type_hops 0 1 2 3 \
  --peprompt_type_walk_graph undirected \
  --peprompt_type_edge_onehot \
  --peprompt_cache_include_feature_key \
  --peprompt_edge_prompt_hidden 128 \
  --peprompt_early_stop_mode loss \
  --peprompt_eval_mode early_stop_only \
  2>&1 | tee artifacts/logs/type_neighborhood_edge/benchmark_acm_h0-1-2-3_identity_tanh_s0p5.log
