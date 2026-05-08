> # 项目介绍

目前异构图没有针对边的提示，因此可以通过typepair，即边两边不同类型对应一个prompt，通过一个小MLP融合
并且为了影响消息传递，在每一层的后面加上prompt进行消息门控

- typepair是残差连接的，并且并没有比node prompt效果好多少
- 如何添加结构信息对边信息进行去噪，可以采用小结构指标融入prompt
- 实验证明只有Laplacian PE有效，并且效果较好，具体实现是通过看作同质图，计算前K个Laplacian 特征值，去掉0这个平凡解。不同节点对应不同k个特征值，将这k个组成位置向量，两节点的位置向量相减
- PEPrompt与typepair通过add/mul/gate融合，目前发现gate的方式最好，即aA+(1-a)B的方式

## 待解决问题

- [ ] PEPrompt时间较多，需要评估并想办法不用特征分解的方式
- [ ] 对比的基线HGMP没有达到论文84的水平，只有80
- [ ] 有一个alpha没有生效，现在预计可能是代码导致的
- [ ] **消融**，目前没有验证typepair是否也是噪声
- [ ] **离线处理** 可以将生成Laplacian PE以及subgraph的部分预处理
- [ ] **多数据集验证** 当前只验证了ACM以及10shots的情况

---

## 提示融合介绍

在 `_fuse_edge_prompt` 函数中，模型实现了将 **宏观类型提示 $P_{type}$** 与 **微观边提示 $P_{edge}$** 结合的具体逻辑。

设融合后的最终边级提示为 $p_{ij}$，代码提供了三种融合策略（通过 `--typepair_edge_prompt_fusion` 控制）：

#### 1. Add 模式 (残差相加)
```python
if self.edge_prompt_fusion == "add":
    return type_prompt.unsqueeze(0) + self.edge_prompt_alpha * edge_prompt
```
* **数学公式：** $p_{ij} = P_{type} + \alpha_{edge} \cdot P_{edge}$
* **解释：** 以类型提示为主，局部拓扑特征作为加性扰动。

#### 2. Mul 模式 (缩放相乘)
```python
if self.edge_prompt_fusion == "mul":
    scale = 1.0 + self.edge_prompt_alpha * torch.tanh(edge_prompt)
    return type_prompt.unsqueeze(0) * scale
```
* **数学公式：** $p_{ij} = P_{type} \odot (1 + \alpha_{edge} \cdot \tanh(P_{edge}))$
* **解释：** 利用拓扑特征生成一个在 $[1-\alpha, 1+\alpha]$ 之间的放缩因子，对全局类型 Prompt 进行动态缩放（类似于注意力机制的权重调制）。

#### 3. Gate 模式 (门控融合，默认推荐)
```python
gate = torch.sigmoid(edge_prompt)
return gate * type_prompt.unsqueeze(0) + (1.0 - gate) * edge_prompt
```
* **数学公式：** 
  $g_{ij} = \sigma(P_{edge})$
  $p_{ij} = g_{ij} \odot P_{type} + (1 - g_{ij}) \odot P_{edge}$
* **解释：** 经典的门控机制。利用当前边的拓扑特征计算出一个 0~1 的信息流门控 $g_{ij}$，动态决定当前这条边应该依赖“全局类型规律”还是依赖“自身拓扑规律”。

---

注入前向传播：如何影响节点表征？

拿到融合后的边提示 $p_{ij}$ 后，在 `TypePairRelationPrompt.forward` 中执行了**自定义的消息传递（Message Passing）**。

#### 1. 构建消息 (Message)
```python
msg = x_dict[src_t][src]
if self.mode == "mul":
    msg = msg * p
else:
    msg = msg + p
```
* 对于边 $j \rightarrow i$，源节点特征为 $h_j$。
* 如果 `--relation_prompt_mode="mul"`：消息 $m_{j \to i} = h_j \odot p_{ij}$
* 如果 `--relation_prompt_mode="add"`：消息 $m_{j \to i} = h_j + p_{ij}$

#### 2. 聚合消息并进行残差注入 (Aggregation & Residual Injection)
```python
agg_dict[dst_t].index_add_(0, dst, msg) 
# ... 计算完 mean 后 ...
h = x + self.alpha * agg
h = self.drop(h)
h = self.ln[ntype](h)
```
* **聚合：** 将所有流向节点 $i$ 的消息通过 `index_add_` 累加起来，然后取平均（`mean`）或求和（`sum`），得到聚合信息 $\Delta h_i$。
* **残差更新：** $h_i^{(new)} = \text{LayerNorm}\Big(\text{Dropout}(h_i + \alpha \cdot \Delta h_i)\Big)$

---
## 子图提取
K-Hop Ego-Network 采用的dgl的`khop_in_subgraph`函数，直接采样目标节点的K阶邻居以内
#### 预训练阶段
预训练阶段会进行子图分割，在子图上进行预训练，根据METIS算法将大图分割为500个社区子图


> # 26.5.5 HGMP复现水平提升
初始typepair的水平为80%
**当前水平**： 0.8264 std=0.0246 | macro_f1 mean=0.8254 std=0.0241
与论文相比可能存在的问题：
1. **划分不同** HGMP采用的划分方式不同
2. **早停机制** HGMP采用loss早停，而typepair采用macro/micro
3. **文件缺失** github代码缺少两个文件，我的补齐代码可能与原论文有一定差距

- [x] 添加environment.yml用于生成代码

> # 26.5.8 

