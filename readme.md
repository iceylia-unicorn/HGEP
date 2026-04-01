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


# 新版
## 预训练
```bash
python scripts/protocol_pretrain.py \
  --method hgmp \
  --dataset ACM \
  --device cuda \
  --epochs 200 \
  --hgnn_type GCN
```
```bash
python scripts/protocol_pretrain.py \
  --method typepair \
  --dataset ACM \
  --device cuda \
  --epochs 200 \
  --hgnn_type HGT
```
## 下游
```bash
python scripts/protocol_fewshot_eval.py \
  --method hgmp \
  --ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --dataset ACM \
  --device cpu \
  --shot 10 \
  --seed 0 \
  --num_class 3 \
  --classification_type NIG \
  --hidden_dim 128 \
  --num_heads 2 \
  --num_layers 2 \
  --hgnn_type GCN
```


```bash
python scripts/protocol_fewshot_eval.py \
  --method typepair \
  --ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --dataset ACM \
  --device cuda \
  --shot 10 \
  --seed 0 \
  --num_class 3 \
  --classification_type NIG \
  --hidden_dim 128 \
  --num_heads 2 \
  --num_layers 2 \
  --hgnn_type GCN
```



python scripts/protocol_fewshot_eval.py \
  --method hgmp_prompt \
  --ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --dataset ACM \
  --device cuda \
  --shot 10 \
  --seed 0 \
  --num_class 3 \
  --classification_type NIG \
  --hidden_dim 128 \
  --num_heads 2 \
  --num_layers 2 \
  --hgnn_type GCN \
  --epochs 200 \
  --patience 30 \
  --lr 5e-3

# 包
安装dgl 
Cuda 12.1
Python 3.9
torch 12.5
dgl 2.4 
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html