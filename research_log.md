# PEPrompt 研究日志

本文档用于记录 HGEP/PEPrompt 的方法设计、实现变化和实验结果。早期内容中包含一些已经被否定的尝试，例如 typepair prompt、dropped-context 复杂融合和基向量选择器；这些内容保留在历史记录中，当前推荐结论以前面的“当前状态”为准。

## 当前项目介绍

PEPrompt 的核心目标是在 HGMP 的异构图 prompt 框架上，引入**边级结构提示**，使下游子图分类时的消息传递不仅依赖节点特征和节点 prompt，也能感知边在局部结构中的位置。

当前稳定版本采用以下流程：

1. 使用 HGMP 的 GraphCL 预训练 checkpoint 初始化异构 GNN。
2. 离线生成 few-shot 下游子图缓存，支持 `khop`、`fanout`、`metapath_topk`、`metapath_topk_path`、`metapath_topk_adapt` 和 `metapath_topk_path_adapt`。
3. PEPrompt 当前主线使用基于元路径的动态子图。每个目标节点按元路径可达计数筛选语义邻域；DBLP 上固定 top-k 不够稳定，当前最优方向是 adaptive top-k。最新 ablation 表明核心收益主要来自 adaptive endpoint selection，path-preserving 不是必要条件。
4. 对子图边计算 PE edge feature。当前有效主线是把异构子图视作同质图，计算 Laplacian PE，并用边两端 PE 差作为边结构编码。
5. 用 PE edge feature 经过 MLP 生成 edge prompt，再注入 HGMP 的消息传递模块。

当前推荐实验配置：

| Dataset | PEPrompt Subgraph | Notes |
| --- | --- | --- |
| ACM | metapath h3/k3 | 当前 r10 结果最稳定，约 `0.8999/0.8997` |
| DBLP | metapath adapt h3/k1-8/a0.5 | 当前最强配置，`macro=0.8234 ± 0.0342`，显著优于 k1-5 和固定 top-k |
| IMDB | metapath h2/k3 | 多标签任务，高阶 metapath 噪声更明显 |
| Freebase | metapath h3/k3, `feats_type=1` | 避免 dense pseudo-feature 导致 CPU OOM |

## 当前结论

- Typepair prompt 单独作为类型级边 prompt 效果有限，后续不再作为主线。
- Laplacian PE edge feature 是当前最有效的结构提示来源。
- Metapath-topk 子图比 khop 更适合 PEPrompt，尤其在 ACM、DBLP、Freebase 上优势明显。
- DBLP 上固定 top-k 的主要问题是 split 间差异很大；adaptive top-k 通过相对阈值动态选择每条元路径的终点数量，显著缓解了 split 低分问题。最新确认结果显示 no-path adaptive `h3/k1-8/a0.5` 是当前最强 DBLP 配置，path-preserving 不是主要收益来源。
- Dropped-context 虽然有小幅提升迹象，但存储、显存和运行时间代价过高，当前不作为主线。
- IMDB 是特殊情况：任务是多标签，few-shot split 波动较大，高阶 metapath 容易引入噪声，因此当前使用 h2/k3。
- Freebase 的主要问题不是原始图文件大小，而是无属性节点类型补 dense pseudo-feature 会导致巨大 CPU 内存占用；当前通过 `feats_type=1` 规避。

## 待解决问题

- [ ] PEPrompt 仍然比纯节点 prompt 慢，需要进一步减少 PE 与 edge prompt 的下游开销。
- [x] 验证 typepair 是否是噪声。结论：typepair 不是当前主线。
- [x] 离线处理子图与 PE edge feature。
- [x] 多数据集验证：ACM、DBLP、IMDB、Freebase 已跑通 r10 对比。
- [ ] 进一步检查 Freebase 上 HGMP Prompt 异常偏低的原因。
- [ ] 评估是否需要 hybrid early stopping，减少 IMDB 上 val loss 与 test F1 不一致的问题。
- [ ] 整理最终论文表格所需的统一协议、日志路径和 checkpoint 说明。
- [x] 将 DBLP `metapath_topk_path_adapt h3/k1-5/a0.5` 从 seeds 0/2 扩展到 seeds 0-4。结论：保持高均值，`macro=0.7861 ± 0.0317`，seed-mean macro std `0.0237`。
- [x] 在 ACM、Freebase、IMDB 上验证 adaptive top-k 是否普适。结论：adaptive 不是普适提升，当前主要在 DBLP 上显著有效；IMDB 和 Freebase 没有稳定超过已有基线。
- [ ] 系统比较 `metapath_topk_adapt` 与 `metapath_topk_path_adapt`，确认 path-preserving 是否应从主线中移除。

## 方法细节

### Edge Prompt 注入

对边 $j \rightarrow i$，PEPrompt 先由边结构特征生成边提示 $p_{ij}$，再将其注入消息传递：

```python
msg = x_dict[src_t][src]
if self.mode == "mul":
    msg = msg * p
else:
    msg = msg + p
```

- `mul`：$m_{j \to i} = h_j \odot p_{ij}$
- `add`：$m_{j \to i} = h_j + p_{ij}$

随后对目标节点入边消息聚合，并做残差更新：

```python
agg_dict[dst_t].index_add_(0, dst, msg)
h = x + self.alpha * agg
h = self.drop(h)
h = self.ln[ntype](h)
```

### 子图构建

- `khop`：使用 DGL 的 `khop_in_subgraph`，直接取目标节点 K 阶邻域。
- `metapath_topk`：枚举 M 跳内元路径，按可达 count 或 degree-normalized score 保留每条元路径固定 top-k 终点，再诱导得到子图。
- `metapath_topk_path`：在固定 top-k 终点基础上，为每个被选终点回溯并保留一条实际路径。
- `metapath_topk_adapt`：每条元路径按 `rank >= alpha * max_rank` 动态决定保留终点数，但不回溯中间路径，直接对选中终点诱导子图。
- `metapath_topk_path_adapt`：在 path-preserving 基础上，每条元路径按 `rank >= alpha * max_rank` 动态决定保留终点数，并用 `min_topk/max_topk` 控制范围。
- 当前主线：ACM/Freebase 仍使用固定 `metapath_topk` 结果作为已验证基线；DBLP 当前最优方向是 adaptive top-k，`metapath_topk_adapt` 因效率更高、效果持平或更好，已成为新的优先候选；HGMP Prompt 对比使用 `khop`。

### Few-shot 划分

当前对齐 HGMP 的 few-shot 设定：

- 每个 split seed 生成 train/val/test；
- 10-shot 下 train 和 val 都是 k-shot；
- test 为剩余目标节点；
- IMDB 多标签使用原 HGMP 风格的 F1 计算逻辑。

## 历史实验记录




## 2026-05-05 HGMP 复现水平提升
初始hgmp的水平为80%
**当前水平**： 0.8264 std=0.0246 | macro_f1 mean=0.8254 std=0.0241
与论文相比可能存在的问题：
1. **划分不同** HGMP采用的划分方式不同，hgmp通过k-shot pretrain/k-shot 
2. **早停机制** HGMP采用loss早停，而typepair采用macro/micro
3. **文件缺失** github代码缺少两个文件，我的补齐代码可能与原论文有一定差距

- [x] 添加environment.yml用于追踪环境
- [x] 添加research_log.md 用于写实验记录

## 2026-05-08 HGMP 与 HGMP Prompt 历史命名问题
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

## 2026-05-09 HGMP 继续优化

hgmp对于不同的数据集参数不同，因此为了达到最优效果，hgmp使用默认参数。

现在看typepair是否需要使用代码进行实验

## 2026-05-11 测试 typepair 是否为噪声
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

## 2026-05-12 子图采样与 PE 处理
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

## 2026-05-13 多数据集测试

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

## 2026-05-14 子图采样问题与方法动机
#### 子图采样对比出问题
PPR采样已经被证明性能下降严重，因此

#### 思考当前方法的设计初衷
最初的edgeprompt想法是异构图上并非每条边都是有效的

什么样的节点很可能是噪声：

1. 比如同一个节点，Paper1只是因为各种原因进来了，但不是Author1的文章。
2. 如何去区分这种不同。
3. paper-author-paper，同一语义情况下的paper特征更有意义，但如何跳过这个author节点去获取paper的提示。

## 2026-05-17 IMDB 问题排查
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

## 2026-05-19 可视化

正在做可视化

## 2026-05-31 Metapath-topk 子图构建

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
## 2026-06-04 元路径采样与早停
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


## 2026-06-10 Dropped-context 与 Prompt 结构
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

## 2026-06-11 Top-k 参数与 Freebase

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

## 2026-06-14 多数据集阶段结论
1. ACM数据集 在10shot阶段稳定能超越HGMP
2. IMDB数据集 10shot阶段由于一些原因，效果很差

## 2026-06-16 IMDB F1 修正
修改了下游的分类标准：什么先linear然后sofatmax,主要是如何将multihot 改为onehot，以及如何比对的问题，在imdb上

并且刚需torchmetrics这个库进行下游的分类。

## 2026-06-17

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

## 2026-06-18 Metapath-aware Prompt 结构尝试

### 背景

现有 PEPrompt 使用 metapath_topk 构建子图，但下游 edge prompt 主要依赖 PE edge feature：

```text
p_e = MLP(PE_e)
message_e = p_e * h_src
```

因此尝试让下游 prompt 显式利用 metapath_topk 子图中的元路径结构信息，而不是只把 metapath 用在子图采样阶段。

### 尝试过的结构

1. `metapath_id / metapath_pos`：将边所属 metapath 或 metapath-position 信息拼接进 prompt 生成器。
2. `metapath_anchor`：学习一组 metapath anchor basis，根据边的 metapath-position 选择 anchor，并注入消息。
3. `metapath_anchor residual`：将 anchor 改为独立残差项：

```text
message_e = p_e * h_src + beta_e * anchor_e
```

4. `metapath_weighted`：不再添加 anchor，而是用 metapath-position 控制 weighted mean 聚合：

```text
g_e = 1 + scale * tanh(MLP(metapath_pos_e))
agg_v = sum_e g_e * (p_e * h_src) / sum_e g_e
```

### ACM 10-shot 结果

统一设置：

- dataset=ACM
- shot=10
- seeds=0 1 2 3 4
- repeats=10
- subgraph=metapath_topk h3/k3
- rank_metric=count
- early_stop=loss
- eval_mode=early_stop_only
- lr=1e-3
- prompt_lr=1e-4

| Variant | Count | Micro-F1 | Macro-F1 | 结论 |
| --- | ---: | ---: | ---: | --- |
| PE-only baseline | 50 | 0.8999 ± 0.0055 | 0.8997 ± 0.0053 | 当前最稳 |
| metapath_id / pos | 50 | 0.8986 ± 0.0054 | 0.8984 ± 0.0054 | 轻微下降 |
| anchor residual | 50 | 0.8991 ± 0.0055 | 0.8989 ± 0.0054 | 基本无变化 |
| weighted aggregation, scale=0.3 | 50 | 0.8990 ± 0.0064 | 0.8988 ± 0.0064 | 基本无变化且方差略大 |

`metapath_weighted, scale=0.3` 详细结果：

```json
{
  "pooled_runs": {
    "peprompt": {
      "count": 50,
      "micro_mean": 0.8989824569225311,
      "micro_std": 0.006437329577886571,
      "macro_mean": 0.8987558817863465,
      "macro_std": 0.006366986148410977
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8989824569225311,
      "micro_std": 0.005417217560814098,
      "macro_mean": 0.8987558817863464,
      "macro_std": 0.005341396818890045
    }
  }
}
```

### 阶段结论

这些结构都没有稳定超过 PE-only baseline。说明当前瓶颈大概率不在“把 metapath 信息再塞进 edge prompt 生成器”或“给消息增加一个 metapath 残差项”，而在更上游的子图选择和保留边质量上。

因此暂时移除 `metapath_anchor` 和 `metapath_weighted` 这些失败分支，回到最初 PEPrompt 版本作为主线：

```text
PE edge feature -> edge prompt p_e -> message_e = p_e * h_src
```

后续优化更应优先考虑：

1. metapath_topk 的 `max_hop/topk/rank_metric` 参数；
2. 子图规模、噪声和不同数据集的最优采样深度；
3. 是否需要在子图构建阶段引入更强的结构筛选，而不是在下游补充复杂 prompt 模块。


## 2026-06-20 DBLP split 不稳定性与 adaptive top-k

### 背景

DBLP 上固定 `metapath_topk` 的一个核心问题是 split 间差异很大：部分 split 可以达到较高性能，但 split 2 等低分 split 明显拖低均值。此前主要观察到：

- 固定 `metapath_topk h3/k2 count` 全 5 个 split 的均值约为 `micro=0.6317`、`macro=0.6306`，seed-mean std 约 `0.046`。
- `degree_norm` 和 `count_idf` 没有解决 split 2 低分问题。
- `edge_dropout=0.1/0.2` 能小幅提升均值，但 pooled std 反而变大，说明它更像正则化训练噪声，并没有从根本上修复子图选择不稳定。
- 可视化 split 0 和 split 2 的训练节点子图后发现，固定 top-k 的元路径终点选择不一定保留实际连接路径；只保留终点再诱导子图，可能导致部分可达语义在子图中不可解释。

### path-preserving top-k

先实现了 `metapath_topk_path`：仍然按固定 top-k 选择每条元路径终点，但对每个被选终点回溯并保留一条从中心节点到终点的实际路径。

DBLP seeds 0/2, repeats 10, h3/k2/count：

```json
{
  "pooled_runs": {
    "peprompt": {
      "count": 20,
      "micro_mean": 0.6449342042207717,
      "micro_std": 0.07528573036144878,
      "macro_mean": 0.6422702252864838,
      "macro_std": 0.07480264465980478
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 2,
      "micro_mean": 0.6449342042207717,
      "micro_std": 0.0701315850019455,
      "macro_mean": 0.6422702252864838,
      "macro_std": 0.06921493411064145
    }
  }
}
```

结论：path-preserving 能明显改善 split 0，但 split 2 仍然偏低，split 方差没有解决。

### dynamic top-k / adaptive top-k

固定 top-k 的问题在于：不同目标节点、不同元路径的 reachable score 分布差异很大。强行每条元路径都取同样的 top-k，会出现两种问题：

1. score 集中时，固定 k 可能引入很多弱相关终点；
2. score 分散但多个候选接近最高分时，固定较小 k 又会丢掉有效邻域。

因此实现了新的 `subgraph_type=metapath_topk_path_adapt`：

- 仍然枚举 `metapath_max_hop=M` 内的所有元路径；
- 对每个中心节点、每条元路径分别计算 reachable score；
- 选择满足 `rank >= alpha * max_rank` 的候选终点；
- 用 `min_topk` 和 `max_topk` 做下限和上限；
- 对每个被选终点保留一条实际路径；
- 当前测试配置为 `M=3, min_k=1, max_k=5, alpha=0.5, rank_metric=count`。

缓存 key：

```text
metapath_topk_path_adapt_m3_k1-5_a0p5_count
```

相关日志：

- 预计算：`artifacts/logs/peprompt_r10_compare/precompute_DBLP_h3_k1-5_a0p5_path_adapt_s0_s2.log`
- benchmark：`artifacts/logs/peprompt_r10_compare/DBLP_peprompt_h3_k1-5_a0p5_path_adapt_s0_s2_r10_fast.log`

DBLP seeds 0/2, repeats 10 结果：

```json
{
  "pooled_runs": {
    "peprompt": {
      "count": 20,
      "micro_mean": 0.7897368371486664,
      "micro_std": 0.033750320570326106,
      "macro_mean": 0.7877550244331359,
      "macro_std": 0.033386411548086616
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 2,
      "micro_mean": 0.7897368371486664,
      "micro_std": 0.029342108964920066,
      "macro_mean": 0.7877550244331359,
      "macro_std": 0.02869718074798583
    }
  }
}
```

阶段结论：

1. adaptive top-k 是目前 DBLP 上最重要的改进，远高于固定 top-k 和 path-only。
2. split 0/2 的 seed-mean std 从 path-only 的约 `0.0692` macro 降到约 `0.0287` macro，说明它不只是提高均值，也明显降低了 split 间不稳定性。
3. 这个结果支持一个更强的方法动机：PEPrompt 的关键不只是边 prompt 生成器，而是“如何根据异构语义路径动态选择局部结构”。固定 top-k 是过于粗糙的子图构建规则。

### 需要继续验证

当前状态：

1. DBLP `metapath_topk_path_adapt` 已完成 seeds `0 1 2 3 4`，保持高均值和较低 split 方差。
2. DBLP `metapath_topk_adapt` no-path 当前只完成 seeds `0/2`，结果与 path-adapt 基本持平；后续可补完整 seeds `0 1 2 3 4`。
3. 对 adaptive 参数做小范围网格：
   - `alpha`: `0.4, 0.5, 0.6`
   - `max_k`: `3, 5, 8`
   - `min_k`: 先保持 `1`
4. 统计 adaptive 子图大小，确认性能提升不是因为子图无约束膨胀。
5. 在 ACM、Freebase 和 IMDB 上测试是否仍然有提升；IMDB 需要谨慎，因为高阶 metapath 可能引入多标签噪声。

### 当前实现状态

已经修改：

- `scripts/precompute_peprompt_cache.py`
  - 新增 `metapath_topk_path_adapt`
  - 新增参数 `--metapath_min_topk`, `--metapath_max_topk`, `--metapath_rel_threshold`
  - 新增相对阈值选择器，并保留 path-preserving 逻辑
- `scripts/peprompt_benchmark.py`
  - 支持加载 adaptive cache key
  - benchmark 参数同步 adaptive top-k 配置
- `scripts/analyze_peprompt_splits.py`
  - 支持 adaptive cache key，便于后续 split 诊断
- `scripts/visualize_split_neighborhoods.py`
  - 支持 adaptive cache key，便于后续可视化训练节点子图

### 追加实验：adaptive no-path ablation

为了验证 `path-preserving` 是否真的是 adaptive top-k 的关键收益来源，新增了一个 ablation 子图类型：

```text
subgraph_type=metapath_topk_adapt
```

它与 `metapath_topk_path_adapt` 使用完全相同的 adaptive endpoint selection：

```text
M=3, min_k=1, max_k=5, alpha=0.5, rank_metric=count
```

区别只有一个：`metapath_topk_adapt` 不回溯和保留中间路径，只保留 adaptive 选出的终点并诱导子图。因此它可以直接衡量 path-preserving 模块的贡献。

#### DBLP no-path adaptive


与 DBLP `metapath_topk_path_adapt` seeds 0/2 对比：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 | Seed-mean Macro Std |
| --- | --- | ---: | ---: | ---: | ---: |
| path-adapt | 0/2 | 20 | 0.7897 ± 0.0338 | 0.7878 ± 0.0334 | 0.0287 |
| adapt no-path | 0/2 | 20 | 0.7889 ± 0.0327 | 0.7870 ± 0.0323 | 0.0270 |
| path-adapt full | 0/1/2/3/4 | 50 | 0.7883 ± 0.0318 | 0.7861 ± 0.0317 | 0.0237 |

结论：DBLP 上 no-path 与 path-adapt 基本持平，差距约 `0.08` 个百分点，远小于 run std。path-preserving 不是 DBLP adaptive top-k 提升的主要来源。

#### ACM adaptive no-path

ACM 10-shot, seed 0, repeats 10：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 | 预计算时间 |
| --- | --- | ---: | ---: | ---: | ---: |
| path-adapt | 0/2 | 20 | 0.8870 ± 0.0200 | 0.8855 ± 0.0210 | seed0 `183.58s`, seed2 `171.06s` |
| adapt no-path | 0 | 10 | 0.8881 ± 0.0100 | 0.8866 ± 0.0101 | seed0 `27.56s` |

注意：这里 path-adapt 是 seeds 0/2，no-path 当前只读到 seed 0，因此不是严格同 seeds 对比。但 no-path seed 0 已经达到或略高于 path-adapt pooled 均值，并且预计算时间从约 3 分钟降到约 28 秒，效率优势非常明显。

ACM 1-shot, seed 0, repeats 10：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 | 预计算时间 |
| --- | --- | ---: | ---: | ---: | ---: |
| path-adapt | 0 | 10 | 0.7711 ± 0.0336 | 0.7561 ± 0.0384 | 未单独记录到同名 seed0 日志 |
| adapt no-path | 0 | 10 | 0.7846 ± 0.0365 | 0.7756 ± 0.0390 | `29.28s` |

ACM 1-shot 上 no-path 明显更好，macro 约提升 `+0.0195`。


## 2026-06-21 多数据集 adaptive 参数确认

### 背景

在 DBLP 上发现 `metapath_topk_adapt` 的 `max_k` 可能比是否 path-preserving 更关键。于是做了一轮 stage-1 小网格：

- DBLP：`h3`，调 `max_k=3/5/8`、`alpha=0.4/0.5/0.6`、`rank_metric=count/degree_norm`
- IMDB：以降噪为目标，调 `h1/h2`、`max_k=3/5`、`alpha=0.5/0.6`、`rank_metric=count/degree_norm`
- Freebase：以固定 `h3/k3` 为基线，调 `h2/h3`、`max_k=3/5/8`、`alpha=0.4/0.5/0.6`

stage-1 使用 seeds `0/2`、repeats `5`。随后对每个数据集最有希望的配置做 full confirm：seeds `0/1/2/3/4`、repeats `10`。

### DBLP：h3/k1-8/a0.5 确认有效

stage-1 中 `h3/k1-8/a0.5/count` 明显最好：

```text
DBLP h3 k1-8 a0.5 count, seeds 0/2, repeats 5:
micro = 0.8343 ± 0.0188
macro = 0.8327 ± 0.0188
seed-mean macro std = 0.0137
```

full confirm 日志：

- `artifacts/logs/peprompt_r10_compare/DBLP_h3_k1-8_a0p5_count_adapt_nopath_s0_s4_r10_confirm.log`

full confirm 结果：

```json
{
  "pooled_runs": {
    "peprompt": {
      "count": 50,
      "micro_mean": 0.8254868447780609,
      "micro_std": 0.03385488631425971,
      "macro_mean": 0.82337895154953,
      "macro_std": 0.03416613137294566
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.8254868447780609,
      "micro_std": 0.028675414817826636,
      "macro_mean": 0.82337895154953,
      "macro_std": 0.02869757898229863
    }
  }
}
```

对比此前 DBLP 关键结果：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 | Seed-mean Macro Std |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed `metapath_topk h3/k2` | 0-4 | 50 | 0.6317 | 0.6306 | ~0.046 |
| path-adapt `h3/k1-5/a0.5` | 0-4 | 50 | 0.7883 ± 0.0318 | 0.7861 ± 0.0317 | 0.0237 |
| adapt no-path `h3/k1-5/a0.5` | 0/2 | 20 | 0.7889 ± 0.0327 | 0.7870 ± 0.0323 | 0.0270 |
| adapt no-path `h3/k1-8/a0.5` | 0-4 | 50 | 0.8255 ± 0.0339 | 0.8234 ± 0.0342 | 0.0287 |

结论：

1. DBLP 上 adaptive top-k 的核心不只是“动态阈值”，还包括给每条元路径足够的上限容量。`max_k=5` 仍然偏保守，`max_k=8` 明显更好。
2. `h3/k1-8/a0.5/count` 已经成为当前 DBLP 最强配置。
3. seed-mean 方差相较 path-adapt `k1-5` 略高，但均值提升约 `+3.7` 个 macro 点，收益远大于方差代价。

### IMDB：degree_norm 小网格有信号，但 full confirm 回落

stage-1 最好的是 `h2/k1-5/a0.5/degree_norm`：

```text
IMDB h2 k1-5 a0.5 degree_norm, seeds 0/2, repeats 5:
micro = 0.6781 ± 0.0067
macro = 0.6029 ± 0.0087
seed-mean macro std = 0.0079
```

full confirm 日志：

- `artifacts/logs/peprompt_r10_compare/IMDB_h2_k1-5_a0p5_degree_norm_adapt_nopath_s0_s4_r10_confirm.log`

full confirm 结果：

```json
{
  "pooled_runs": {
    "peprompt": {
      "count": 50,
      "micro_mean": 0.6671663200855256,
      "micro_std": 0.015704459163242403,
      "macro_mean": 0.5894608116149902,
      "macro_std": 0.01941842145287495
    }
  },
  "seed_mean_then_std": {
    "peprompt": {
      "count": 5,
      "micro_mean": 0.6671663200855256,
      "micro_std": 0.014730146186656724,
      "macro_mean": 0.5894608116149902,
      "macro_std": 0.018220297291866795
    }
  }
}
```

对比历史 IMDB：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| HGMP Prompt khop | 0-4 | 50 | 0.6760 ± 0.0121 | 0.5996 ± 0.0150 |
| PEPrompt fixed `h2/k3` | 0-4 | 50 | 0.6667 ± 0.0159 | 0.5883 ± 0.0197 |
| adapt no-path `h2/k1-5/a0.5/count` | 0-4 | 50 | 0.6464 ± 0.0046 | 0.5641 ± 0.0064 |
| adapt no-path `h2/k1-5/a0.5/degree_norm` | 0-4 | 50 | 0.6672 ± 0.0157 | 0.5895 ± 0.0194 |

结论：

1. IMDB 上 `degree_norm` 比 adaptive count 明显更合理，说明多标签场景中直接按可达 count 扩邻域会引入噪声。
2. 但 full confirm 后它只与 fixed `h2/k3` 基本持平，仍未超过 HGMP Prompt khop。
3. IMDB 的瓶颈可能不在 adaptive 子图上，而在多标签 few-shot 的监督噪声、label collapse/评估协议和 early stopping 的不稳定性。

### Freebase：adaptive 未超过固定 h3/k3

stage-1 使用 hid256 checkpoint 时，较好的两个候选为：

```text
h3/k1-3/a0.5/count:
micro = 0.3401 ± 0.0189
macro = 0.2877 ± 0.0273

h3/k1-8/a0.5/count:
micro = 0.3428 ± 0.0268
macro = 0.2847 ± 0.0309
```

由于固定 `h3/k3` baseline 使用 hid512 checkpoint，为公平比较，使用 `Freebase.GraphCL.GCN.hid512.np50.seed0.pth` 做 full confirm。

full confirm 日志：

- `artifacts/logs/peprompt_r10_compare/Freebase_ft1_h3_k1-3_a0p5_count_adapt_nopath_hid512_s0_s4_r10_confirm.log`
- `artifacts/logs/peprompt_r10_compare/Freebase_ft1_h3_k1-8_a0p5_count_adapt_nopath_hid512_s0_s4_r10_confirm.log`

full confirm 结果：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 | Seed-mean Macro Std |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed `h3/k3`, hid512 | 0-4 | 50 | 0.3463 ± 0.0207 | 0.2949 ± 0.0268 | 0.0177 |
| adapt no-path `h3/k1-3/a0.5`, hid512 | 0-4 | 50 | 0.3331 ± 0.0235 | 0.2791 ± 0.0256 | 0.0122 |
| adapt no-path `h3/k1-8/a0.5`, hid512 | 0-4 | 50 | 0.3376 ± 0.0303 | 0.2856 ± 0.0362 | 0.0084 |

结论：

1. Freebase 上 adaptive no-path 没有超过固定 `metapath_topk h3/k3`。
2. `k1-8` 比 `k1-3` 均值略高、seed-mean 方差更低，但 pooled 方差更大，仍然不如固定 top-k。
3. Freebase 可能更需要固定大小的稳定语义邻域，adaptive threshold 反而会在类别稀疏、节点类型不均衡时引入不稳定选择。

### 2026-06-21 阶段结论

1. `metapath_topk_adapt` 不是普适提升，目前主要在 DBLP 上显著有效。
2. DBLP 的关键配置是 `h3/k1-8/a0.5/count`，说明 DBLP 需要更宽的高阶语义邻域；`k1-5` 上限不足。
3. IMDB 上 `degree_norm` 优于 count，但 full confirm 只和 fixed `h2/k3` 持平，仍不如 HGMP Prompt。IMDB 后续应优先研究多标签协议、early stopping 或标签噪声，而不是继续扩大 metapath 网格。
4. Freebase 上固定 `h3/k3` 仍是当前最好配置，adaptive threshold 没有带来收益。
5. 后续调参应数据集分治：
   - DBLP：围绕 `max_k=8` 继续微调 `alpha=0.4/0.5/0.6`，并统计子图大小；
   - IMDB：保留 `h2/k3` 或 `degree_norm` 作为对照，优先处理多标签评估/早停；
   - Freebase：回到固定 top-k 或尝试很小范围的固定 k，而不是继续 adaptive。

#### IMDB数据集的F1计算
由于当前的F1会被argmax缩成onehot，因此改回来试一下，发现效果还不如之前的效果。
