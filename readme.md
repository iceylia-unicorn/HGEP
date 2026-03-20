
## 测试版

### 预训练
```bash
python scripts/pretrain_hgmp.py \ --root data \ --dataset ACM \ --tau_hops 2 \ --fanout 25 15 \ --batch_size 64 \ --epochs 50 \ --lr 1e-3 \ --weight_decay 1e-5 \ --backbone hgt \ --hidden_dim 128 \ --proj_dim 128 \ --num_layers 2 \ --num_heads 2 \ --dropout 0.2 \ --device cuda \ --save checkpoints/pretrain_hgmp_best.pt
```
### 下游任务
python scripts/finetune_fewshot.py \
  --dataset ACM \
  --ckpt checkpoints/pretrain_hgmp_best.pt \
  --shot 1 \
  --seed 1 \
  --tau_hops 2 \
  --fanout 25 15 \
  --batch_size 32 \
  --backbone hgt \
  --hidden_dim 128 \
  --num_layers 2 \
  --num_heads 2 \
  --prompt_mode mul \
  --epochs 200 \
  --patience 30 \
  --early_stop_metric micro \
  --lr 1e-3


## exp/typepair-Prompt
python scripts/finetune_fewshot.py \
  --dataset ACM \
  --root data \
  --splits splits \
  --shot 1 \
  --seed 0 \
  --ckpt checkpoints/pretrain_hgt_acm_last.pt \
  --backbone hgt \
  --hidden_dim 128 \
  --num_layers 2 \
  --num_heads 2 \
  --prompt_scope node \
  --prompt_mode mul \
  --relation_alpha 0.5 \
  --relation_dropout 0.1 \
  --relation_aggr mean \
  --head_hidden 128 \
  --epochs 200 \
  --patience 30 \
  --lr 1e-3