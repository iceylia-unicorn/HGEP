# HGEP / PEPrompt

本项目用于复现 HGMP/HGPrompt 系列异构图 prompt 方法，并在此基础上研究 **PEPrompt**：一种基于边结构位置编码的异构图边级提示方法。

当前主线不再使用早期的 typepair prompt 作为核心方法。实验表明，单纯的边类型 prompt 收益有限；更稳定的方向是使用离线子图采样和 PE edge feature，为下游消息传递生成 edge prompt。

## 当前方法

PEPrompt 的下游流程：

1. 使用 HGMP GraphCL checkpoint 初始化异构 GNN。
2. 离线生成 few-shot 子图缓存。
3. 对子图边计算 PE edge feature。
4. 用 PE edge feature 通过 MLP 生成 edge prompt。
5. 将 edge prompt 注入 HGMP 的消息传递过程。

当前推荐子图设置：

| Dataset | PEPrompt Subgraph | Notes |
| --- | --- | --- |
| ACM | `metapath_topk`, hop=3, topk=3 | 当前 r10 结果约 `0.8999/0.8997` |
| DBLP | `metapath_topk`, hop=3, topk=3 | 相比 HGMP Prompt 有稳定提升 |
| IMDB | `metapath_topk`, hop=2, topk=3 | 多标签任务，高阶元路径噪声更明显 |
| Freebase | `metapath_topk`, hop=3, topk=3, `feats_type=1` | 避免 dense pseudo-feature OOM |

## 目录

```text
scripts/
  hgmp_pretrain.py              # HGMP GraphCL 预训练入口
  precompute_peprompt_cache.py  # PEPrompt few-shot 子图与边特征离线缓存
  peprompt_benchmark.py         # HGMP / HGMP Prompt / PEPrompt 下游对比

src/gpbench/
  protocol_bridge/              # HGMP legacy 协议桥接和 PEPrompt 实现
  data/                         # 数据加载与预处理
  pretrain/                     # 预训练相关模块

protocols/hgmp/
  *_legacy.py                   # 原 HGMP 逻辑的兼容实现

artifacts/
  checkpoints/                  # 预训练 checkpoint
  cache/                        # 离线子图与 PE 缓存
  logs/                         # 实验日志
  results/                      # benchmark 输出结果

research_log.md                 # 研究日志和关键实验结论
readme_legacy.md                # 旧版命令记录，保留作历史参考
```

## 环境

当前主要环境：

```text
python=3.11
torch=2.4.1
dgl=2.4.0
torch_geometric
scikit-learn
torchmetrics
```

安装参考：

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric scikit-learn torchmetrics
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

## HGMP 预训练

ACM 示例：

```bash
python -u scripts/hgmp_pretrain.py \
  --dataset ACM \
  --device cuda:0 \
  --seed 0 \
  --epochs 200 \
  --benchmark_defaults
```

Freebase 建议使用 `feats_type=1`，避免无属性节点类型被补成巨大 dense 方阵：

```bash
python -u scripts/hgmp_pretrain.py \
  --dataset Freebase \
  --device cuda:0 \
  --seed 0 \
  --epochs 200 \
  --benchmark_defaults \
  --feats_type 1 \
  --hidden_dim 512 \
  --num_samples 50
```

## PEPrompt 离线缓存

ACM h3/k3 示例：

```bash
python -u scripts/precompute_peprompt_cache.py \
  --dataset ACM \
  --shots 10 \
  --seeds 0 1 2 3 4 \
  --subgraph_type metapath_topk \
  --metapath_max_hop 3 \
  --metapath_topk 3 \
  --metapath_rank_metric count \
  --peprompt_fusion_mode none
```

Freebase：

```bash
python -u scripts/precompute_peprompt_cache.py \
  --dataset Freebase \
  --shots 10 \
  --seeds 0 1 2 3 4 \
  --subgraph_type metapath_topk \
  --metapath_max_hop 3 \
  --metapath_topk 3 \
  --metapath_rank_metric count \
  --peprompt_fusion_mode none \
  --feats_type 1
```

## 下游 Benchmark

PEPrompt ACM 示例：

```bash
python -u scripts/peprompt_benchmark.py \
  --dataset ACM \
  --methods peprompt \
  --shot 10 \
  --seeds 0 1 2 3 4 \
  --repeats 10 \
  --device cuda:0 \
  --subgraph_type metapath_topk \
  --metapath_max_hop 3 \
  --metapath_topk 3 \
  --metapath_rank_metric count \
  --peprompt_fusion_mode none \
  --peprompt_early_stop_mode loss \
  --peprompt_eval_mode early_stop_only \
  --lr 1e-3 \
  --prompt_lr 1e-4 \
  --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth
```

HGMP Prompt 对比：

```bash
python -u scripts/peprompt_benchmark.py \
  --dataset ACM \
  --methods hgmp_prompt \
  --shot 10 \
  --seeds 0 1 2 3 4 \
  --repeats 10 \
  --device cuda:0 \
  --subgraph_type khop \
  --hgmp_prompt_recipe legacy \
  --hgmp_prompt_batch_size 10 \
  --hgmp_prompt_epochs 300 \
  --hgmp_prompt_patience 30 \
  --hgmp_prompt_early_stop_mode legacy_loss \
  --hgmp_prompt_eval_mode early_stop_only \
  --hgmp_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth
```

## 最新 r10 结果

| Dataset | Method | Subgraph | Count | Micro-F1 | Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: |
| ACM | HGMP Prompt | khop | 50 | 0.8371 ± 0.0319 | 0.8366 ± 0.0320 |
| ACM | PEPrompt | metapath h3/k3 | 50 | 0.8999 ± 0.0055 | 0.8997 ± 0.0053 |
| DBLP | HGMP Prompt | khop | 50 | 0.5842 ± 0.0360 | 0.5816 ± 0.0370 |
| DBLP | PEPrompt | metapath h3/k3 | 50 | 0.6187 ± 0.0433 | 0.6171 ± 0.0431 |
| IMDB | HGMP Prompt | khop | 50 | 0.6760 ± 0.0121 | 0.5996 ± 0.0150 |
| IMDB | PEPrompt | metapath h2/k3 | 50 | 0.6667 ± 0.0159 | 0.5883 ± 0.0197 |
| Freebase | HGMP Prompt | khop, ft1 | 50 | 0.2098 ± 0.0199 | 0.1318 ± 0.0201 |
| Freebase | PEPrompt | metapath h3/k3, ft1 | 50 | 0.3463 ± 0.0207 | 0.2949 ± 0.0268 |

详细分析见 [research_log.md](research_log.md)。

## W&B

只有传入 `--use_wandb` 时才会启用 W&B。可以在项目根目录创建 `.codex`：

```text
WANDB_API_KEY=xxx
```

## 历史说明

早期版本包含 `typepair`、`typepair_edge_feature_fusion`、`graph_summary_basis`、`dropped-context` 等实验分支。当前记录保留在 [research_log.md](research_log.md) 和 [readme_legacy.md](readme_legacy.md) 中，但主线实验以本文档中的 PEPrompt metapath-topk 配置为准。
