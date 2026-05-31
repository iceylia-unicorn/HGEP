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
- [x] **消融**，目前没有验证typepair是否也是噪声
- [ ] **离线处理** 可以将生成Laplacian PE以及subgraph的部分预处理
- [ ] **多数据集验证** 当前只验证了ACM以及10shots的情况
- [ ] **预训练与子图采样方法** 原hgmp的预训练方法以及子图构建方式可能并不适用边提示
- [ ] **hgprompt的比较** 无论是在hgmp还是hgprompt的论文，hgprompt方法在1至5shot下有着较大优势。
- [ ] **子图构建** 原HGMP采用的子图构建方式大多是粗暴地用1阶邻居，部分用的2阶邻居
- [ ] **大图** 当前设定了特征分解只能在50000个节点以内，假如数据集的节点数特别多，很可能造成特征分解的占用很大。


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
#### 下游阶段
khop_in_subgraph,每个数据集拥有不同的跳数。

## 划分方式
`k-shot pretrain`/ `官方val`/ `官方test`

hgmp原始划分: 每类最多先随机选400个节点，再从这400个节点里面进行划分。`k-shot pretrain`/ `k-shot val`/ `rest test`




> # 26.5.5 HGMP复现水平提升
初始hgmp的水平为80%
**当前水平**： 0.8264 std=0.0246 | macro_f1 mean=0.8254 std=0.0241
与论文相比可能存在的问题：
1. **划分不同** HGMP采用的划分方式不同，hgmp通过k-shot pretrain/k-shot 
2. **早停机制** HGMP采用loss早停，而typepair采用macro/micro
3. **文件缺失** github代码缺少两个文件，我的补齐代码可能与原论文有一定差距

- [x] 添加environment.yml用于追踪环境
- [x] 添加research_log.md 用于写实验记录

> # 26.5.8 hgmp与hgmp_prompt历史命名问题
发现有两个历史遗留问题
1. 之前methods分为hgmp与hgmp_prompt但我忘记了这一点。
2. 下游中也会调用METIS，但按理来说这是上游预训练出现的东西

第二点已经解决了，实际上METIS是为了在构建HGNN的时候获取输入，由于之前是预处理好的，现在不行，因此进行了修改，单独拿一个函数get_graph_metadata_lightweight获取

第一点出现了问题，通过实验表明hgmp_prompt的性能还在hgmp之下，按理来说脚本中的代码是hgmp拥有node prompt接近原文设计的。
- 当只运行一次时，hgmp是82，hgmp_prompt是72

**hgmp_prompt训练逻辑** 双轮训练：先训练一轮head,再一轮prompt，因此有三个loss，head_loss、prompt_loss以及val_loss。

待解决问题：
- [ ] 下游中我设计的early_stop采用的val_macro_f1,hgmp采用的loss；hgmp采用双轮训练，单次epoch中先训练一轮head，再训练一轮prompt。哪种方式更优呢

hgmp若不使用离线处理子图速度极慢，并且采用双轮loss机制导致一直难以收敛

> # 26.5.9 hgmp继续优化

hgmp对于不同的数据集参数不同，因此为了达到最优效果，hgmp使用默认参数。

现在看typepair是否需要使用代码进行实验

> # 26.5.11 测试typepair是否为噪声
实验结果表明，这个typepair确实是噪声：
```json
{
  "pooled_runs": {
    "typepair::edge_only": {
      "count": 5,
      "micro_mean": 0.87790367603302,
      "micro_std": 0.01592069231989516,
      "macro_mean": 0.8782463073730469,
      "macro_std": 0.016758606129274448
    },
    "typepair::full": {
      "count": 5,
      "micro_mean": 0.8689329743385314,
      "micro_std": 0.015263020657882632,
      "macro_mean": 0.8683551907539367,
      "macro_std": 0.015857310858026625
    },
    "typepair::none": {
      "count": 5,
      "micro_mean": 0.8052879929542541,
      "micro_std": 0.020337499889572643,
      "macro_mean": 0.807218337059021,
      "macro_std": 0.019532940423152168
    },
    "typepair::type_only": {
      "count": 5,
      "micro_mean": 0.806987726688385,
      "micro_std": 0.016247219963764537,
      "macro_mean": 0.8077319741249085,
      "macro_std": 0.017660173171727078
    }
  },
  "seed_mean_then_std": {
    "typepair::edge_only": {
      "count": 5,
      "micro_mean": 0.87790367603302,
      "micro_std": 0.01592069231989516,
      "macro_mean": 0.8782463073730469,
      "macro_std": 0.016758606129274448
    },
    "typepair::full": {
      "count": 5,
      "micro_mean": 0.8689329743385314,
      "micro_std": 0.015263020657882632,
      "macro_mean": 0.8683551907539367,
      "macro_std": 0.015857310858026625
    },
    "typepair::none": {
      "count": 5,
      "micro_mean": 0.8052879929542541,
      "micro_std": 0.020337499889572643,
      "macro_mean": 0.807218337059021,
      "macro_std": 0.019532940423152168
    },
    "typepair::type_only": {
      "count": 5,
      "micro_mean": 0.806987726688385,
      "micro_std": 0.016247219963764537,
      "macro_mean": 0.8077319741249085,
      "macro_std": 0.017660173171727078
    }
  }
}
```
实验证明typepair确实是噪声

在1shot的情况下比full版本提升了两个点，因此其实也是削弱

> # 26.5.12 

#### 关于子图采样方式和PE处理

AI给出的Laplacian PE的优化：
1. **RWPE** 随机游走的幂次作为节点的微观结构特征。 RWPE去替代
2. **anchor-based** 基于部分锚点，计算每个节点到这些锚点的最短路径

子图采样的优化建议：
1. **PPR subgraph** PPR密集子图采样去获取top-k 去替代
2. **带数量限制的邻居采样** 采用主流的采样逻辑，并每一挑限定采样数量
3. **基于元路径的子图采样** 其中提到了保留特定元路径的邻居来构建子图，手动删除一些边。

其中我觉得第三点可玩性最高，因此可以先验证前两点，最后再来验证第三点。甚至可以将元路径的子图采样作为一种创新手段。

#### 离线处理

1. 修改划分逻辑为hgmp的逻辑，降低了val的数量，设置为了严格的k-shot，导致了性能下降严重

**性能下降** 在观察训练过程时，发现训练集的那几个shot很快就能达到F1=1的效果。相比起之前降低了1到2个点

**训练loss下降快** 训练态中loss下降速度很快，且最优epoch基本出现在前30轮，这说明loss有问题，需要设置weight-decay权重衰退大一点，并且应该还是要设置双轮的训练机制

#### 子图采样
对比三种子图采样方式，原本的k-hop以及fanout以及ppr

实验发现fanout有极其微弱的优势，ppr则下降很多。

#### 思考
为了围绕graph PE做一个prompt，但为了避免这个设计与graph PE本身重合，因此需要额外的一些围绕graph PE和prompt的融合。

graph PE实际上算是一种额外的信息，如何将这个额外的信息很好地利用起来呢。回到原来的题目，如何去控制目标节点的邻域走向呢，这个是否可以和采样meta-path sample以及

> # 26.5.13

#### 多数据集测试
为了测试fanout是否在其余的数据集上也有这样的优势，准备先将多数据集跑通

其中Freebase的预训练存在问题，跑不通的。

#### 数据集规模分析

为了判断多数据集实验的难度来源，需要将“图结构规模”和“运行时特征规模”分开看。

先看原始图的结构统计：

| 数据集 | 目标类型 | 类别数 | 总节点 | 总边 | 节点类型数 | 边类型数 | 目标节点 | 已标注目标节点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | `paper` | 3 | 10,942 | 547,872 | 4 | 8 | 3,025 | 3,025 |
| DBLP | `author` | 4 | 26,128 | 239,566 | 4 | 6 | 4,057 | 4,057 |
| IMDB | `movie` | 5 | 21,420 | 86,642 | 4 | 6 | 4,932 | 4,573 |
| Freebase | `book` | 7 | 180,098 | 2,115,376 | 8 | 72 | 40,402 | 7,954 |

从图结构上看，Freebase 明显是最大图；而 ACM、DBLP、IMDB 仍处于同一数量级。

但真正决定是否容易 OOM 的并不只是图结构，而是运行时的特征构造。当前 HGMP / PEPrompt 流程会对没有原始特征的节点类型补一个 dense 的 `N x N` 伪特征矩阵，因此各数据集的运行时特征体量大致为：

| 数据集 | 运行时总特征量 |
| --- | ---: |
| ACM | 约 83 MB |
| DBLP | 约 249 MB |
| IMDB | 约 437 MB |
| Freebase | 约 34.5 GiB |

其中最关键的几项是：

- ACM 只有 `term` 是伪特征，约 14.5 MB，因此整体较轻。
- DBLP 的主要负担来自 `paper` 原始特征，约 242 MB，但整体仍可控。
- IMDB 的主要负担来自 `keyword` 的 dense 伪特征，单这一项约 254 MB，因此虽然图边数不多，但运行时并不轻。
- Freebase 最致命，因为几乎所有节点类型都没有原始特征，会被补成 dense 方阵。仅 `music` 一类就约 25.3 GiB，`book` 约 6.1 GiB，因此预训练阶段极易直接被系统杀死。

这说明 Freebase 跑不通并不是因为 `processed/data.pt` 很大，而是因为加载后会动态补出超大的 dense 伪特征矩阵。

再看本地磁盘占用：

| 数据集 | `processed/data.pt` | 数据目录总大小 |
| --- | ---: | ---: |
| ACM | 74M | 8.8G |
| DBLP | 242M | 368M |
| IMDB | 176M | 355M |
| Freebase | 17M | 41M |

这里需要注意，ACM 的 `8.8G` 不是原始数据本体，而是历史 `data/acm/induced_graphs` 缓存占了约 `8.7G`。如果只看原始处理后的图文件，ACM 并不大。

> # 26.5.14
#### 子图采样对比出问题
PPR采样已经被证明性能下降严重，因此

#### 思考当前方法的设计初衷
最初的edgeprompt想法是异构图上并非每条边都是有效的

什么样的节点很可能是噪声：

1. 比如同一个节点，Paper1只是因为各种原因进来了，但不是Author1的文章。
2. 如何去区分这种不同。
3. paper-author-paper，同一语义情况下的paper特征更有意义，但如何跳过这个author节点去获取paper的提示。

> # 26.5.17
IMDB数据集出现问题，性能没有原论文那么好，并且相较于原始论文性能差距很大，只有30多。
因为IMDB是多标签的，但



1. 验证集数量也改为k-shot，10-shot下性能下降1个点左右
2. ACM的子图从1阶邻居改为2阶OOM，测试随机采样邻居子图fanout，0.001左右的提升，测试PPR采样子图，性能下降10个点
3. typepair与laplacian位置编码融合反而性能下降1个点左右
4. 将子图采样以及图结合

**计划**
1. 思考基于元路径的子图采样方法，作为边提示，之前的子图采样都是基于同质图，是否可以有一个基于异构图的子图采样与边提示结合。
2. 测试更多数据集 IMDB和DBLP， freebase太大得做特殊处理
3. 阅读图位置编码和子图采样相关论文

> # 26.5.19

正在做可视化

> # 26.5.31

上一个实验出现了一点问题，实际上本身可能存在一些问题