##数据格式
data/acm/raw/ACM/node.dat


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