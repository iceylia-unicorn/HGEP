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

上一个实验出现了一点问题，实际上本身可能存在一些问题，现在开始考虑异构图子图的构建问题。

所以采用元路径采样子图，m跳内所有元路径，然后做topk的截断，先对比了这么做后子图的大小差异，结果如下，其中截断采用的是同样元路径
[dataset] ACM target=paper targets=3025 nodes=10942 edges=547872
[khop] dataset=ACM hop=1 avg_nodes=93.30 avg_edges=376.89 max_nodes=206 max_edges=3328
[khop] dataset=ACM hop=2 avg_nodes=3281.55 avg_edges=217934.26 max_nodes=4605 max_edges=478076
[metapath_topk] dataset=ACM M=1 topk=5 metapaths=5 avg_nodes=12.94 avg_edges=51.05 max_nodes=22 max_edges=180
[metapath_topk] dataset=ACM M=1 topk=10 metapaths=5 avg_nodes=18.53 avg_edges=85.13 max_nodes=35 max_edges=426
[metapath_topk] dataset=ACM M=1 topk=20 metapaths=5 avg_nodes=28.71 avg_edges=145.99 max_nodes=53 max_edges=920
[metapath_topk] dataset=ACM M=2 topk=5 metapaths=18 avg_nodes=38.98 avg_edges=378.25 max_nodes=72 max_edges=1166
[metapath_topk] dataset=ACM M=2 topk=10 metapaths=18 avg_nodes=64.39 avg_edges=1014.06 max_nodes=123 max_edges=2862
[metapath_topk] dataset=ACM M=2 topk=20 metapaths=18 avg_nodes=105.56 avg_edges=2558.96 max_nodes=197 max_edges=6522
[metapath_topk] dataset=ACM M=3 topk=5 metapaths=59 avg_nodes=94.53 avg_edges=1390.76 max_nodes=156 max_edges=3262
[metapath_topk] dataset=ACM M=3 topk=10 metapaths=59 avg_nodes=171.67 avg_edges=3899.24 max_nodes=276 max_edges=7854
[metapath_topk] dataset=ACM M=3 topk=20 metapaths=59 avg_nodes=305.69 avg_edges=9546.19 max_nodes=490 max_edges=18374
[dataset] DBLP target=author targets=4057 nodes=26128 edges=239566
[khop] dataset=DBLP hop=1 avg_nodes=5.84 avg_edges=9.68 max_nodes=169 max_edges=336
[khop] dataset=DBLP hop=2 avg_nodes=32.20 avg_edges=84.12 max_nodes=635 max_edges=3100
[metapath_topk] dataset=DBLP M=1 topk=5 metapaths=1 avg_nodes=3.86 avg_edges=5.72 max_nodes=6 max_edges=10
[metapath_topk] dataset=DBLP M=1 topk=10 metapaths=1 avg_nodes=4.67 avg_edges=7.34 max_nodes=11 max_edges=20
[metapath_topk] dataset=DBLP M=1 topk=20 metapaths=1 avg_nodes=5.27 avg_edges=8.54 max_nodes=21 max_edges=40
[metapath_topk] dataset=DBLP M=2 topk=5 metapaths=4 avg_nodes=12.33 avg_edges=27.75 max_nodes=21 max_edges=82
[metapath_topk] dataset=DBLP M=2 topk=10 metapaths=4 avg_nodes=17.21 avg_edges=43.01 max_nodes=41 max_edges=154
[metapath_topk] dataset=DBLP M=2 topk=20 metapaths=4 avg_nodes=22.59 avg_edges=58.73 max_nodes=76 max_edges=302
[metapath_topk] dataset=DBLP M=3 topk=5 metapaths=7 avg_nodes=21.47 avg_edges=64.42 max_nodes=36 max_edges=170
[metapath_topk] dataset=DBLP M=3 topk=10 metapaths=7 avg_nodes=37.24 avg_edges=130.34 max_nodes=71 max_edges=378
[metapath_topk] dataset=DBLP M=3 topk=20 metapaths=7 avg_nodes=64.79 avg_edges=249.73 max_nodes=131 max_edges=806
[dataset] IMDB target=movie targets=4573 nodes=21420 edges=86642
[khop] dataset=IMDB hop=1 avg_nodes=9.82 avg_edges=17.64 max_nodes=10 max_edges=18
[khop] dataset=IMDB hop=2 avg_nodes=115.08 avg_edges=235.69 max_nodes=642 max_edges=1370
[metapath_topk] dataset=IMDB M=1 topk=5 metapaths=3 avg_nodes=9.82 avg_edges=17.64 max_nodes=10 max_edges=18
[metapath_topk] dataset=IMDB M=1 topk=10 metapaths=3 avg_nodes=9.82 avg_edges=17.64 max_nodes=10 max_edges=18
[metapath_topk] dataset=IMDB M=1 topk=20 metapaths=3 avg_nodes=9.82 avg_edges=17.64 max_nodes=10 max_edges=18
[metapath_topk] dataset=IMDB M=2 topk=5 metapaths=6 avg_nodes=20.92 avg_edges=45.04 max_nodes=25 max_edges=88
[metapath_topk] dataset=IMDB M=2 topk=10 metapaths=6 avg_nodes=29.51 avg_edges=63.38 max_nodes=40 max_edges=126
[metapath_topk] dataset=IMDB M=2 topk=20 metapaths=6 avg_nodes=43.29 avg_edges=91.74 max_nodes=70 max_edges=182
[metapath_topk] dataset=IMDB M=3 topk=5 metapaths=15 avg_nodes=38.29 avg_edges=69.48 max_nodes=54 max_edges=152
[metapath_topk] dataset=IMDB M=3 topk=10 metapaths=15 avg_nodes=76.13 avg_edges=140.93 max_nodes=105 max_edges=244
[metapath_topk] dataset=IMDB M=3 topk=20 metapaths=15 avg_nodes=142.36 avg_edges=281.83 max_nodes=208 max_edges=502
[dataset] Freebase target=book targets=7954 nodes=180098 edges=1057688
[khop] dataset=Freebase hop=1 avg_nodes=10.89 avg_edges=29.63 max_nodes=2446 max_edges=5589
[khop] dataset=Freebase hop=2 avg_nodes=795.49 avg_edges=3871.76 max_nodes=18242 max_edges=86061
[metapath_topk] dataset=Freebase M=1 topk=5 metapaths=5 avg_nodes=4.33 avg_edges=7.13 max_nodes=26 max_edges=58
[metapath_topk] dataset=Freebase M=1 topk=10 metapaths=5 avg_nodes=5.13 avg_edges=9.31 max_nodes=51 max_edges=107
[metapath_topk] dataset=Freebase M=1 topk=20 metapaths=5 avg_nodes=5.95 avg_edges=11.85 max_nodes=101 max_edges=221
[metapath_topk] dataset=Freebase M=2 topk=5 metapaths=22 avg_nodes=19.70 avg_edges=44.81 max_nodes=98 max_edges=449
[metapath_topk] dataset=Freebase M=2 topk=10 metapaths=22 avg_nodes=31.68 avg_edges=81.26 max_nodes=184 max_edges=808
[metapath_topk] dataset=Freebase M=2 topk=20 metapaths=22 avg_nodes=51.34 avg_edges=140.51 max_nodes=346 max_edges=1695
[metapath_topk] dataset=Freebase M=3 topk=5 metapaths=72 avg_nodes=65.19 avg_edges=268.10 max_nodes=256 max_edges=1485
[metapath_topk] dataset=Freebase M=3 topk=10 metapaths=72 avg_nodes=118.11 avg_edges=559.37 max_nodes=510 max_edges=3530
[metapath_topk] dataset=Freebase M=3 topk=20 metapaths=72 avg_nodes=212.79 avg_edges=1086.07 max_nodes=990 max_edges=6983

发现ACM的大小与

我想到一种可能，元路径实际上是语义树，如果纯粹


conda run -n HGEP python scripts/peprompt_benchmark.py \
  --dataset ACM \
  --methods peprompt \
  --shot 10 \
  --seeds 0 1 2 3 4 \
  --repeats 10 \
  --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth \
  --subgraph_type metapath_topk \
  --metapath_max_hop 3 \
  --metapath_topk 5 \
  --save_dir artifacts/results/peprompt_metapath_topk_suite



conda run -n HGEP python scripts/peprompt_benchmark.py \
  --dataset ACM \
  --methods peprompt \
  --shot 1 \
  --seeds 0 1 2 3 4 \
  --repeats 1 \
  --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth \
  --subgraph_type metapath_topk \
  --metapath_max_hop 3 \
  --metapath_topk 5 \
  --save_dir artifacts/results/peprompt_metapath_topk_suite


python scripts/peprompt_benchmark.py \
  --dataset ACM \
  --shot 10 \
  --seeds 0 \
  --repeats 10 \
  --methods peprompt \
  --subgraph_type metapath_topk \
  --metapath_rank_metric degree_norm \
  --peprompt_ckpt artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth
# 26.6.4
#### 元路径采样有效
相较于直接的khop，metapath的结构是有效的，能够达到ACM 10-shot 88.6 此时的子图采样策略与prompt提示没有比较
```json
"pooled_runs": {
  "peprompt": {
    "count": 50,
    "micro_mean": 0.8864736843109131,
    "micro_std": 0.017332137665788987,
    "macro_mean": 0.8858819735050202,
    "macro_std": 0.0178124920473951
  }
},
"seed_mean_then_std": {
  "peprompt": {
    "count": 5,
    "micro_mean": 0.8864736843109131,
    "micro_std": 0.011757842815277595,
    "macro_mean": 0.8858819735050203,
    "macro_std": 0.012037870901193072
  }
}
```
#### 元路径提示
由于是一种截断的提示，那么如果我将截断的信息融入到提示中，那么会怎么样呢
这种融合方式

我用了第一种融合方式，是将不同节点对，这种方式有损失


第二种方式，元路径可达分数加权融合，这种方式也能达到88.3 ，但不使用情况下结果应该是88.6，然后我改为使用loss早停能达到89
{
  "pooled_runs": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8912280678749085,
      "micro_std": 0.00929990487588275,
      "macro_mean": 0.8902406096458435,
      "macro_std": 0.009421494158723006
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8912280678749085,
      "micro_std": 0.00929990487588275,
      "macro_mean": 0.8902406096458435,
      "macro_std": 0.009421494158723006
    }
  }
}

  --lr 5e-3 \
  --prompt_lr 1e-3 \
  "pooled_runs": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8915789365768433,
      "micro_std": 0.009024427702802477,
      "macro_mean": 0.8912229776382447,
      "macro_std": 0.00877641540383437
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8915789365768433,
      "micro_std": 0.009024427702802477,
      "macro_mean": 0.8912229776382447,
      "macro_std": 0.00877641540383437
    }
  }

  --lr 5e-3 \
  --prompt_lr 5e-4 \

  {
  "pooled_runs": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8929824590682983,
      "micro_std": 0.007169419664572823,
      "macro_mean": 0.8928701519966126,
      "macro_std": 0.007008618695628356
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8929824590682983,
      "micro_std": 0.007169419664572823,
      "macro_mean": 0.8928701519966126,
      "macro_std": 0.007008618695628356
    }
  }
}


### dropped-context
#### 1. top-k
对每条metapath 都会算从目标节点出发的reachable scores，然后取top-k 保留终点节点

#### 2. 找出被丢弃的metapath终点
nonzero = scores > 0
dropped = nonzero - keep

但是实际上如何将metapath的dropped context信息保存，并且和下游的边进行较好的融合是个问题

没有一个高效的方案。

有多种融合方式

1. 只用一阶的ctx，但是这种方式目前存在很多的问题
2. 边类型的ctx 融合edgeprompt 生成 0.5的提升
3. 虚拟节点的方式，提升也差不多0.5提升，但无论是离线还是下游时间很长
4. metapath的方式，爆显存，每个sample要存


# 26.6.10
自从发现dropped ctx在效率上不高外，我开始着手处理prompt本身的处理

如果将ctx作为一个子图级别的提示，作为一个子图级别的统计量，而不是一个图级别的统计量，那么是否会有效果呢

z_graph = summary(subgraph metapath structure)
b_r = Linear_r(z_graph)
p_e = MLP([h_src, h_dst, pe_e]) + b_r

当前的计算公式

这个子图的结构包含：
各 hop 的 keep mass
各终点类型的 reachable mass
各 metapath 长度的比例
命中节点类型分布 等

这个方式的变化还行：{
  "pooled_runs": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8929824590682983,
      "micro_std": 0.013544030239639418,
      "macro_mean": 0.8928787112236023,
      "macro_std": 0.013357733794538291
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8929824590682983,
      "micro_std": 0.013544030239639418,
      "macro_mean": 0.8928787112236023,
      "macro_std": 0.013357733794538291
    }
  }
}
但是依旧是很小的收益


#### 基向量
如果使用基向量，而
alpha_e = softmax(Selector([h_src, h_dst, pe_e, z_graph]))
p_e = alpha_e @ B

这种方式并不可行，实验结果证明效果并不好，效果并未有多大的变化，但训练的epoch 从60 增到100，但稳定性有所提升

之前的PE prompt并未使用到prompt的信息
{
  "pooled_runs": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8903508901596069,
      "micro_std": 0.00835868257925349,
      "macro_mean": 0.8903262257575989,
      "macro_std": 0.0078447418263402
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8903508901596069,
      "micro_std": 0.00835868257925349,
      "macro_mean": 0.8903262257575989,
      "macro_std": 0.0078447418263402
    }
  }
}

#### metapath生成prompt
mp_code_e = Σ w_m(e) * emb(m)
alpha_e = softmax(MLP([pe_e, etype_emb, mp_code_e]))
p_e = alpha_e @ B

# 26.6.11

#### topk参数实验

在ACM上采用hop3 top3的形式是最好的，能达到90.5两个都是，0.6的标准差。


#### 接下来的问题

1. 子图的选取方式的进一步优化


#### Freebase
Freebase在进行预训练的时候会爆CPU内存

原本是
```python
if attr == "num_nodes":
    data[node_type]["x"] = create_matrix(value, 0.01)
```
其中`value`是`node_type`对应的节点数，也就是说当节点数很多的时候，这一node_type会创建一个极大的dense(密集)节点特征矩阵，这是为了某些无属性节点类型设计的

但由于原本就会压缩到10维，因此也没有必要
```python
if dataname == "Freebase" and feats_type in (1, 5):
    data[node_type]["x"] = torch.zeros((int(value), 10))
else:
    data[node_type]["x"] = create_matrix(value, 0.01)
```
其中涉及到了feats_type这一超参

- 0/-1: 保留原始特征，无属性则补充为NxN的dense 矩阵
- 1 只保留第零个节点特征，其余改为Nx10的零特征
- 2 保留第0个，其余改为identity-like
- 3 所有改为identity-like
- 4 save 2，其余改为identity-like
- 5 save 2, 其余改为Nx10 零特征

# 2026.6.14
1. ACM数据集 在10shot阶段稳定能超越HGMP
2. IMDB数据集 10shot阶段由于一些原因，效果很差

# 2026.6.16
修改了下游的分类标准：什么先linear然后sofatmax,主要是如何将multihot 改为onehot，以及如何比对的问题，在imdb上

并且刚需torchmetrics这个库进行下游的分类。

# 2026.6.17

## HGMP Prompt vs PEPrompt r10 对比记录

统一设置：

- shot=10
- seeds=0 1 2 3 4
- repeats=10
- 下游早停使用 loss，并开启 early_stop_only，只在 best checkpoint 上最终评估 F1
- HGMP Prompt 使用 khop 子图，`hgmp_prompt_recipe=legacy`
- PEPrompt 使用 metapath_topk，`rank_metric=count`，`fusion_mode=none`
- Freebase 使用 `feats_type=1`

日志路径：

- HGMP Prompt: `artifacts/logs/hgmp_prompt_r10_compare/`
- PEPrompt: `artifacts/logs/peprompt_r10_compare/`
- PEPrompt IMDB: `artifacts/logs/imdb_compare/peprompt_metapath_h2_k3_lr1e3_plr1e4_r10_fast.log`

### 结果汇总

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

### 初步结论

1. ACM 上 PEPrompt 明显优于 HGMP Prompt，且方差更小：micro 提升约 6.28 个点，macro 提升约 6.30 个点。
2. DBLP 上 PEPrompt 也优于 HGMP Prompt：micro 提升约 3.45 个点，macro 提升约 3.55 个点，但方差略大。
3. IMDB 上 HGMP Prompt 略优于 PEPrompt：micro 高约 0.92 个点，macro 高约 1.14 个点。IMDB 当前更适合 h2/k3，h3/k3 和 h2/k2 都不如 h2/k3。
4. Freebase 上 PEPrompt 相比 HGMP Prompt 提升明显：micro 提升约 13.66 个点，macro 提升约 16.32 个点。但 HGMP Prompt 在 Freebase 上 best_epoch 经常为 1，说明当前 khop + ft1 组合可能训练不稳定或特征信息不足。
5. 当前结果支持：metapath_topk 子图在 ACM、DBLP、Freebase 上能带来比 khop HGMP Prompt 更好的下游表现；IMDB 受多标签、few-shot split 和高阶噪声影响更大，需要单独使用较小的 h2/k3 配置。
