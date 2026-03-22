# 预训练
```bash
python scripts/pretrain_hgmp.py \
  --dataset ACM --root data \
  --tau_hops 2 --fanout 25 15 \
  --backbone to_hetero_gcn \
  --batch_size 64 --epochs 50 \
  --aug_ratio 0.2 --temperature 0.2 \
  --save checkpoints/pretrain_hgmp_acm.pt
```

hgt
```bash
python scripts/pretrain_hgmp.py \
  --dataset ACM --root data \
  --tau_hops 2 --fanout 25 15 \
  --backbone hgt \
  --batch_size 64 --epochs 200 \
  --aug_ratio 0.2 --temperature 0.2 \
  --patience 7 --min_delta 1e-4 \
  --es_save checkpoints/pretrain_hgt_acm_best.pt \
  --save checkpoints/pretrain_hgt_acm_last.pt
```

## 下游结构Prompt（与边类型无关）
```bash
python scripts/finetune_fewshot.py \
  --dataset ACM --root data --splits splits \
  --shot 10 --seed 0  \
  --backbone to_hetero_gcn \
  --ckpt checkpoints/pretrain_hgmp_acm_last.pt \
  --prompt_type type --struct_hidden 64 \
  --prompt_mode mul --head_hidden 128 --epochs 200
```



## gpt修改后

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

# 关于HGMP
代码位于reorg/hgmp-hgprompt-aligned分支下
## 预处理
使用的类ProG induced graphs形式，因此需要先运行scripts/preprocess_legacy.py，运行后位置位于data/{dataname}/induced_graphs下