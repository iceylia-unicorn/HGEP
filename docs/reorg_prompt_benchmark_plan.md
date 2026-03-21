# Prompt benchmark 重组计划

## 当前问题

当前分支 `exp/hgmp-typepair-prompt-v1` 已经把：

- encoder 逻辑
- node prompt
- type-pair relation prompt
- 下游训练脚本

揉在一起，导致很难回答下面三个问题：

1. 预训练 encoder 本身是否有效？
2. type-pair prompt 是否单独有效？
3. 提升来自 encoder、node prompt，还是 relation prompt？

## 重组目标

本次重组优先满足“可归因”和“可持续迭代”，暂时不追求一次性彻底改完。

目标分三层：

### 1) Protocol 层（固定训练/评测协议）
- 固定 dataset / split / backbone / checkpoint / optimizer / metric
- 不在 protocol 层写方法逻辑

### 2) Component 层（可插拔组件）
- encoder
- prompt
- head
- method composition

### 3) Experiment 层（实验配置）
- none
- node_prompt
- type_pair_prompt
- node_plus_type_pair

## 第一阶段（本次提交）

采取“非破坏式重组”：

- 新增 `src/gpbench/prompts/`，把 prompt 组件独立出来
- 新增 `src/gpbench/heads/`，把分类头独立出来
- 新增 `src/gpbench/methods/`，把 encoder + prompt + head 的组合独立出来
- 新增 `configs/method/`，固定方法级配置
- 新增 `scripts/finetune_prompt.py`，作为后续统一入口

注意：
- 暂时不删除旧的 `scripts/finetune_fewshot.py`
- 暂时不删除旧的 `downstream/model.py`
- 暂时不修改已有 checkpoint 路径

## 归因实验矩阵

第一轮只跑下面 6 组：

### A. 检查预训练是否有效
1. random encoder + no prompt
2. pretrained encoder + no prompt
3. pretrained encoder + frozen head protocol

### B. 检查 prompt 是否有效
1. none
2. node_prompt
3. type_pair_prompt
4. node_plus_type_pair

要求：
- 同一 split
- 同一 ckpt
- 同一 backbone
- 同一 head
- 同一 trainer
- 同一 early stop metric

## 后续第二阶段

第二阶段再继续做：

- 把 `load_frozen_encoder` 从 `downstream/model.py` 迁移到 encoder 层
- 把 trainer 完整迁移到 `protocol/`
- 加入统一 registry / runner
- 将旧脚本标记为 legacy

## 开发规则

1. 新模块必须先过 `none / node / new / node+new` 四组对照
2. encoder 目录不直接放 method 组件
3. prompt 组件必须实现统一 forward 接口
4. checkpoint / log / result 不再混入源码目录
