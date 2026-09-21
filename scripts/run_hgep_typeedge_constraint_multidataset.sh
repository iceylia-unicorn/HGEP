#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home_A/yuanqilin/.conda/envs/HGEP/bin/python}"
LOGDIR="artifacts/logs/type_neighborhood_edge/constraint_multidataset"
mkdir -p "${LOGDIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[$(date)] DBLP identity_tanh on existing TypeNeighborhoodEdge cache"
"${PY}" -u scripts/peprompt_benchmark.py \
  --dataset DBLP \
  --root data \
  --splits splits \
  --shot 1 \
  --methods peprompt \
  --seeds 0 1 2 \
  --repeats 5 \
  --pretrain_seed 0 \
  --device cuda:0 \
  --save_dir artifacts/results/type_neighborhood_edge/DBLP_typeonly_identity_tanh_s0p5 \
  --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth \
  --feats_type 0 \
  --hidden_dim 512 \
  --num_heads 8 \
  --num_layers 2 \
  --dropout 0.5 \
  --hgnn_type GCN \
  --num_samples 500 \
  --num_class 4 \
  --classification_type NIG \
  --embed_batch_size 32 \
  --head_hidden 128 \
  --head_dropout 0.3 \
  --epochs 100 \
  --patience 20 \
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
  --peprompt_offline_cache_dir artifacts/cache/type_neighborhood_edge \
  --subgraph_type metapath_topk_adapt \
  --metapath_max_hop 3 \
  --metapath_min_topk 1 \
  --metapath_max_topk 8 \
  --metapath_rel_threshold 0.5 \
  --metapath_rank_metric count \
  --peprompt_type_hops 0 1 2 \
  --peprompt_type_walk_graph undirected \
  --peprompt_type_edge_onehot \
  --peprompt_cache_include_feature_key \
  --peprompt_edge_prompt_hidden 128 \
  --peprompt_early_stop_mode loss \
  --peprompt_eval_mode early_stop_only \
  2>&1 | tee "${LOGDIR}/benchmark_dblp_h0-1-2_identity_tanh_s0p5.log"

echo "[$(date)] Freebase precompute TypeNeighborhoodEdge h0-1-2 cache"
"${PY}" -u scripts/precompute_peprompt_cache.py \
  --root data \
  --datasets Freebase \
  --shots 1 \
  --seeds 0 1 2 \
  --feats_type 1 \
  --peprompt_offline_cache_dir artifacts/cache/type_neighborhood_edge_multidataset \
  --subgraph_type metapath_topk \
  --metapath_max_hop 3 \
  --metapath_topk 3 \
  --metapath_rank_metric count \
  --peprompt_edge_feature_names TypeNeighborhoodEdge \
  --peprompt_type_hops 0 1 2 \
  --peprompt_type_walk_graph undirected \
  --peprompt_type_edge_onehot \
  --peprompt_cache_include_feature_key \
  2>&1 | tee "${LOGDIR}/precompute_freebase_h0-1-2.log"

run_freebase() {
  local constraint="$1"
  local tag="$2"
  echo "[$(date)] Freebase ${tag}"
  "${PY}" -u scripts/peprompt_benchmark.py \
    --dataset Freebase \
    --root data \
    --splits splits \
    --shot 1 \
    --methods peprompt \
    --seeds 0 1 2 \
    --repeats 5 \
    --pretrain_seed 0 \
    --device cuda:0 \
    --save_dir "artifacts/results/type_neighborhood_edge/Freebase_typeonly_${tag}" \
    --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/Freebase.GraphCL.GCN.hid512.np50.seed0.pth \
    --feats_type 1 \
    --hidden_dim 512 \
    --num_heads 8 \
    --num_layers 2 \
    --dropout 0.5 \
    --hgnn_type GCN \
    --num_samples 50 \
    --num_class 7 \
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
    --relation_prompt_constraint "${constraint}" \
    --relation_prompt_constraint_scale 0.5 \
    --peprompt_pretrain_family hgmp \
    --peprompt_edge_feature_names TypeNeighborhoodEdge \
    --peprompt_offline_cache_dir artifacts/cache/type_neighborhood_edge_multidataset \
    --subgraph_type metapath_topk \
    --metapath_max_hop 3 \
    --metapath_topk 3 \
    --metapath_rank_metric count \
    --peprompt_type_hops 0 1 2 \
    --peprompt_type_walk_graph undirected \
    --peprompt_type_edge_onehot \
    --peprompt_cache_include_feature_key \
    --peprompt_edge_prompt_hidden 128 \
    --peprompt_early_stop_mode loss \
    --peprompt_eval_mode early_stop_only \
    2>&1 | tee "${LOGDIR}/benchmark_freebase_h0-1-2_${tag}.log"
}

run_freebase none none_s0p5
run_freebase identity_tanh identity_tanh_s0p5

echo "[$(date)] DONE"
