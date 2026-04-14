## 数据存储位置
data/acm/raw/ACM/node.dat


## 分支exp/ v1

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


python scripts/hgmp_pretrain.py \
  --dataset ACM \
  --device cuda:0 \
  --seed 0 \
  --epochs 200 \
  --benchmark_defaults

python scripts/hgmp_run.py \
  --dataset ACM \
  --shot 10 \
  --seed 0 \
  --device cuda:0 \
  --ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --benchmark_defaults

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
dgl 2.4 

python=3.11
torch==2.4.1

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib -f https://data.pyg.org/whl/torch-2.4.1%2Bcu124.html

pip install torch_geometric

pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# hgprompt 
## pretrain
python scripts/hgprompt_pretrain.py \
  --dataset ACM \
  --device cuda:0 \
  --seed 1 \
  --epoch 200 \
  --benchmark_defaults \
  --ckpt_alias artifacts/checkpoints/hgprompt/pretrain/acm_hgprompt_seed1.pt
## downstream

python scripts/hgprompt_run.py \
  --dataset ACM \
  --splits splits \
  --shot 10 \
  --seed 0 \
  --repeat 1 \
  --device cuda \
  --ckpt artifacts/checkpoints/hgprompt/pretrain/ACM.gcn.ft2.hop1.seed0.best.pt \
  --benchmark_defaults


# 有关aligned
需要先用hgprompt生成0-4 seed pretrain.pt，然后就能自动读取不同pt进行

但是这种方式和我以前见过的不一样，应该是同一pretrain，然后不同seed

python scripts/protocol_benchmark_v2.py \
  --dataset ACM \
  --shot 10 \
  --methods hgmp typepair hgprompt \
  --seeds 0 1 2 3 4 \
  --hgnn_type GCN \
  --repeats 2 \
  --hgmp_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --typepair_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --hgprompt_ckpt artifacts/checkpoints/hgprompt/pretrain/ACM.gcn.ft2.hop1.seed0.best.pt

python scripts/protocol_benchmark_v2.py \
  --dataset ACM \
  --shot 10 \
  --methods typepair \
  --seeds 0 1 2 3 4 \
  --hgnn_type GCN \
  --repeats 2 \
  --hgmp_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --typepair_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid128.np100.pth \
  --hgprompt_ckpt artifacts/checkpoints/hgprompt/pretrain/ACM.gcn.ft2.hop1.seed0.best.pt