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
  --es_save checkpoints/pretrain_hgmp_acm_best.pt \
  --save checkpoints/pretrain_hgmp_acm_last.pt
```

## 下游结构Prompt（与边类型无关）
```bash
python scripts/finetune_fewshot.py \
  --dataset ACM --root data --splits splits \
  --shot 5 --seed 0 \
  --ckpt checkpoints/pretrain_hgmp_acm_best.pt \
  --prompt_type type_struct --struct_hidden 64 \
  --prompt_mode mul --head_hidden 128 --epochs 200
```