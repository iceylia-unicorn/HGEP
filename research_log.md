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
| DBLP | metapath adapt h3/k1-8/a0.5；1-shot 可用 target-closed count full-support | 常规主线为 h3 adaptive；当前 protocol benchmark 中 target-closed count full-support 达到约 `0.8869/0.8851`，但 cache 很大 |
| IMDB | metapath h2/k3 | 多标签任务，高阶 metapath 噪声更明显 |
| Freebase | metapath h3/k3, `feats_type=1` | 避免 dense pseudo-feature 导致 CPU OOM |

## 当前结论

- Typepair prompt 单独作为类型级边 prompt 效果有限，后续不再作为主线。
- Laplacian PE edge feature 是当前最有效的结构提示来源。
- Metapath-topk 子图比 khop 更适合 PEPrompt，尤其在 ACM、DBLP、Freebase 上优势明显。
- DBLP 上固定 top-k 的主要问题是 split 间差异很大；adaptive top-k 通过相对阈值动态选择每条元路径的终点数量，显著缓解了 split 低分问题。常规 h3 no-path adaptive 仍是较轻量主线。
- DBLP 1-shot 诊断中，`target_closed + count endpoint ranking + full support recovery` 是目前最强分支，当前 protocol benchmark 达到约 `micro=0.8869 / macro=0.8851`。其收益来自 target-to-target semantic endpoint 和完整 support 结构，而不是 PE sim rerank；代价是离线 cache 显著膨胀。
- Target-closed 并非普适策略。Freebase 的有向 schema 使许多重要类型无法进入 `BOOK -> ... -> BOOK` closed metapath，因此 target-closed 在 Freebase 1-shot 上效果较差；Freebase 更适合 open endpoint selection 加 support recovery。
- Dropped-context 虽然有小幅提升迹象，但存储、显存和运行时间代价过高，当前不作为主线。
- IMDB 是特殊情况：任务是多标签，few-shot split 波动较大，高阶 metapath 容易引入噪声，因此当前使用 h2/k3。
- Freebase 的主要问题不是原始图文件大小，而是无属性节点类型补 dense pseudo-feature 会导致巨大 CPU 内存占用；当前通过 `feats_type=1` 规避。

## 待解决问题

- [ ] PEPrompt 仍然比纯节点 prompt 慢，需要进一步减少 PE 与 edge prompt 的下游开销；DBLP full-support cache 已暴露明显存储/IO 压力。
- [x] 验证 typepair 是否是噪声。结论：typepair 不是当前主线。
- [x] 离线处理子图与 PE edge feature。
- [x] 多数据集验证：ACM、DBLP、IMDB、Freebase 已跑通 r10 对比。
- [ ] 进一步检查 Freebase 上 HGMP Prompt 异常偏低的原因。
- [ ] 评估是否需要 hybrid early stopping，减少 IMDB 上 val loss 与 test F1 不一致的问题。
- [ ] 整理最终论文表格所需的统一协议、日志路径和 checkpoint 说明，并单独报告离线 cache size / precompute time。
- [x] 将 DBLP `metapath_topk_path_adapt h3/k1-5/a0.5` 从 seeds 0/2 扩展到 seeds 0-4。结论：保持高均值，`macro=0.7861 ± 0.0317`，seed-mean macro std `0.0237`。
- [x] 在 ACM、Freebase、IMDB 上验证 adaptive top-k 是否普适。结论：adaptive 不是普适提升，当前主要在 DBLP 上显著有效；IMDB 和 Freebase 没有稳定超过已有基线。
- [ ] 系统比较 `metapath_topk_adapt` 与 `metapath_topk_path_adapt`，确认 path-preserving 是否应从主线中移除。
- [ ] 复查 HGPrompt native baseline 的协议差异。当前 ACM/DBLP 结果偏强，split overlap 已检查为 0，但还需要修正/验证 HGPrompt center 与 best checkpoint 的同步逻辑。
- [x] 增加 PEPrompt 的 HGPrompt-style prototype/class-center 下游开关。结论：ACM 上表现较强，DBLP 1-shot 明显不稳，说明 HGPrompt 的 prototype 分类逻辑不能直接作为 PEPrompt 的通用下游替代。

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
- `metapath_endpoint_mode=target_closed`：只保留从 target type 出发并回到 target type 的 closed metapath。配合默认 `metapath_support_mode=auto` 时，会解析为 count-based support recovery：endpoint 仍然是 target-type 节点，中间非 target 节点按 `prefix_count(center -> support) * suffix_count(support -> selected endpoints)` 的路径贡献保留。DBLP 1-shot 当前最强是 `rank_metric=count`，即按 target-target typed path count 选择 endpoint；`pe_sim` 只作为对照，结果表明它在该设置下会削弱 DBLP 的强语义 count 信号。
- `metapath_endpoint_mode=all` + `metapath_support_mode=count`：保留原始任意类型 endpoint 选择，但为多跳 endpoint 恢复中间 support/bridge nodes。该分支更适合 Freebase 这种有向 schema 下难以形成完整 target-closed path 的数据集。
- 当前主线：ACM/Freebase 仍使用固定 `metapath_topk` 结果作为已验证基线；DBLP 当前最优方向是 adaptive top-k，`metapath_topk_adapt` 因效率更高、效果持平或更好，已成为新的优先候选；HGMP Prompt 对比使用 `khop`。

### Few-shot 划分

当前对齐 HGMP 的 few-shot 设定：

- 每个 split seed 生成 train/val/test；
- 10-shot 下 train 和 val 都是 k-shot；
- test 为剩余目标节点；
- IMDB 多标签使用原 HGMP 风格的 F1 计算逻辑。

## 历史实验记录




### 2026-05-05 HGMP 复现水平提升
初始hgmp的水平为80%
**当前水平**： 0.8264 std=0.0246 | macro_f1 mean=0.8254 std=0.0241
与论文相比可能存在的问题：
1. **划分不同** HGMP采用的划分方式不同，hgmp通过k-shot pretrain/k-shot 
2. **早停机制** HGMP采用loss早停，而typepair采用macro/micro
3. **文件缺失** github代码缺少两个文件，我的补齐代码可能与原论文有一定差距

- [x] 添加environment.yml用于追踪环境
- [x] 添加research_log.md 用于写实验记录

### 2026-05-08 HGMP 与 HGMP Prompt 历史命名问题
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

### 2026-05-09 HGMP 继续优化

hgmp对于不同的数据集参数不同，因此为了达到最优效果，hgmp使用默认参数。

现在看typepair是否需要使用代码进行实验

### 2026-05-11 测试 typepair 是否为噪声
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

### 2026-05-12 子图采样与 PE 处理
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

### 2026-05-13 多数据集测试

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

### 2026-05-14 子图采样问题与方法动机
#### 子图采样对比出问题
PPR采样已经被证明性能下降严重，因此

#### 思考当前方法的设计初衷
最初的edgeprompt想法是异构图上并非每条边都是有效的

什么样的节点很可能是噪声：

1. 比如同一个节点，Paper1只是因为各种原因进来了，但不是Author1的文章。
2. 如何去区分这种不同。
3. paper-author-paper，同一语义情况下的paper特征更有意义，但如何跳过这个author节点去获取paper的提示。

### 2026-05-17 IMDB 问题排查
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

### 2026-05-19 可视化

正在做可视化

### 2026-05-31 Metapath-topk 子图构建

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

### 2026-06-04 元路径采样与早停
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


#### dropped-context
##### 1. top-k
对每条metapath 都会算从目标节点出发的reachable scores，然后取top-k 保留终点节点

##### 2. 找出被丢弃的metapath终点
nonzero = scores > 0
dropped = nonzero - keep

但是实际上如何将metapath的dropped context信息保存，并且和下游的边进行较好的融合是个问题

没有一个高效的方案。

有多种融合方式

1. 只用一阶的ctx，但是这种方式目前存在很多的问题
2. 边类型的ctx 融合edgeprompt 生成 0.5的提升
3. 虚拟节点的方式，提升也差不多0.5提升，但无论是离线还是下游时间很长
4. metapath的方式，爆显存，每个sample要存


### 2026-06-10 Dropped-context 与 Prompt 结构
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

### 2026-06-11 Top-k 参数与 Freebase

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

### 2026-06-14 多数据集阶段结论
1. ACM数据集 在10shot阶段稳定能超越HGMP
2. IMDB数据集 10shot阶段由于一些原因，效果很差

### 2026-06-16 IMDB F1 修正
修改了下游的分类标准：什么先linear然后sofatmax,主要是如何将multihot 改为onehot，以及如何比对的问题，在imdb上

并且刚需torchmetrics这个库进行下游的分类。

### 2026-06-17 | HGMP Prompt vs PEPrompt r10 对比记录

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

#### 结果汇总

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

#### 初步结论

1. ACM 上 PEPrompt 明显优于 HGMP Prompt，且方差更小：micro 提升约 6.28 个点，macro 提升约 6.30 个点。
2. DBLP 上 PEPrompt 也优于 HGMP Prompt：micro 提升约 3.45 个点，macro 提升约 3.55 个点，但方差略大。
3. IMDB 上 HGMP Prompt 略优于 PEPrompt：micro 高约 0.92 个点，macro 高约 1.14 个点。IMDB 当前更适合 h2/k3，h3/k3 和 h2/k2 都不如 h2/k3。
4. Freebase 上 PEPrompt 相比 HGMP Prompt 提升明显：micro 提升约 13.66 个点，macro 提升约 16.32 个点。但 HGMP Prompt 在 Freebase 上 best_epoch 经常为 1，说明当前 khop + ft1 组合可能训练不稳定或特征信息不足。
5. 当前结果支持：metapath_topk 子图在 ACM、DBLP、Freebase 上能带来比 khop HGMP Prompt 更好的下游表现；IMDB 受多标签、few-shot split 和高阶噪声影响更大，需要单独使用较小的 h2/k3 配置。

### 2026-06-18 Metapath-aware Prompt 结构尝试

#### 背景

现有 PEPrompt 使用 metapath_topk 构建子图，但下游 edge prompt 主要依赖 PE edge feature：

```text
p_e = MLP(PE_e)
message_e = p_e * h_src
```

因此尝试让下游 prompt 显式利用 metapath_topk 子图中的元路径结构信息，而不是只把 metapath 用在子图采样阶段。

#### 尝试过的结构

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

#### ACM 10-shot 结果

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

#### 阶段结论

这些结构都没有稳定超过 PE-only baseline。说明当前瓶颈大概率不在“把 metapath 信息再塞进 edge prompt 生成器”或“给消息增加一个 metapath 残差项”，而在更上游的子图选择和保留边质量上。

因此暂时移除 `metapath_anchor` 和 `metapath_weighted` 这些失败分支，回到最初 PEPrompt 版本作为主线：

```text
PE edge feature -> edge prompt p_e -> message_e = p_e * h_src
```

后续优化更应优先考虑：

1. metapath_topk 的 `max_hop/topk/rank_metric` 参数；
2. 子图规模、噪声和不同数据集的最优采样深度；
3. 是否需要在子图构建阶段引入更强的结构筛选，而不是在下游补充复杂 prompt 模块。


### 2026-06-20 DBLP split 不稳定性与 adaptive top-k

#### 背景

DBLP 上固定 `metapath_topk` 的一个核心问题是 split 间差异很大：部分 split 可以达到较高性能，但 split 2 等低分 split 明显拖低均值。此前主要观察到：

- 固定 `metapath_topk h3/k2 count` 全 5 个 split 的均值约为 `micro=0.6317`、`macro=0.6306`，seed-mean std 约 `0.046`。
- `degree_norm` 和 `count_idf` 没有解决 split 2 低分问题。
- `edge_dropout=0.1/0.2` 能小幅提升均值，但 pooled std 反而变大，说明它更像正则化训练噪声，并没有从根本上修复子图选择不稳定。
- 可视化 split 0 和 split 2 的训练节点子图后发现，固定 top-k 的元路径终点选择不一定保留实际连接路径；只保留终点再诱导子图，可能导致部分可达语义在子图中不可解释。

#### path-preserving top-k

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

#### dynamic top-k / adaptive top-k

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

#### 需要继续验证

当前状态：

1. DBLP `metapath_topk_path_adapt` 已完成 seeds `0 1 2 3 4`，保持高均值和较低 split 方差。
2. DBLP `metapath_topk_adapt` no-path 当前只完成 seeds `0/2`，结果与 path-adapt 基本持平；后续可补完整 seeds `0 1 2 3 4`。
3. 对 adaptive 参数做小范围网格：
   - `alpha`: `0.4, 0.5, 0.6`
   - `max_k`: `3, 5, 8`
   - `min_k`: 先保持 `1`
4. 统计 adaptive 子图大小，确认性能提升不是因为子图无约束膨胀。
5. 在 ACM、Freebase 和 IMDB 上测试是否仍然有提升；IMDB 需要谨慎，因为高阶 metapath 可能引入多标签噪声。

#### 当前实现状态

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

#### 追加实验：adaptive no-path ablation

为了验证 `path-preserving` 是否真的是 adaptive top-k 的关键收益来源，新增了一个 ablation 子图类型：

```text
subgraph_type=metapath_topk_adapt
```

它与 `metapath_topk_path_adapt` 使用完全相同的 adaptive endpoint selection：

```text
M=3, min_k=1, max_k=5, alpha=0.5, rank_metric=count
```

区别只有一个：`metapath_topk_adapt` 不回溯和保留中间路径，只保留 adaptive 选出的终点并诱导子图。因此它可以直接衡量 path-preserving 模块的贡献。

##### DBLP no-path adaptive


与 DBLP `metapath_topk_path_adapt` seeds 0/2 对比：

| Variant | Seeds | Count | Micro-F1 | Macro-F1 | Seed-mean Macro Std |
| --- | --- | ---: | ---: | ---: | ---: |
| path-adapt | 0/2 | 20 | 0.7897 ± 0.0338 | 0.7878 ± 0.0334 | 0.0287 |
| adapt no-path | 0/2 | 20 | 0.7889 ± 0.0327 | 0.7870 ± 0.0323 | 0.0270 |
| path-adapt full | 0/1/2/3/4 | 50 | 0.7883 ± 0.0318 | 0.7861 ± 0.0317 | 0.0237 |

结论：DBLP 上 no-path 与 path-adapt 基本持平，差距约 `0.08` 个百分点，远小于 run std。path-preserving 不是 DBLP adaptive top-k 提升的主要来源。

##### ACM adaptive no-path

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


### 2026-06-21 多数据集 adaptive 参数确认

#### 背景

在 DBLP 上发现 `metapath_topk_adapt` 的 `max_k` 可能比是否 path-preserving 更关键。于是做了一轮 stage-1 小网格：

- DBLP：`h3`，调 `max_k=3/5/8`、`alpha=0.4/0.5/0.6`、`rank_metric=count/degree_norm`
- IMDB：以降噪为目标，调 `h1/h2`、`max_k=3/5`、`alpha=0.5/0.6`、`rank_metric=count/degree_norm`
- Freebase：以固定 `h3/k3` 为基线，调 `h2/h3`、`max_k=3/5/8`、`alpha=0.4/0.5/0.6`

stage-1 使用 seeds `0/2`、repeats `5`。随后对每个数据集最有希望的配置做 full confirm：seeds `0/1/2/3/4`、repeats `10`。

#### DBLP：h3/k1-8/a0.5 确认有效

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

#### IMDB：degree_norm 小网格有信号，但 full confirm 回落

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

#### Freebase：adaptive 未超过固定 h3/k3

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

#### 2026-06-21 阶段结论

1. `metapath_topk_adapt` 不是普适提升，目前主要在 DBLP 上显著有效。
2. DBLP 的关键配置是 `h3/k1-8/a0.5/count`，说明 DBLP 需要更宽的高阶语义邻域；`k1-5` 上限不足。
3. IMDB 上 `degree_norm` 优于 count，但 full confirm 只和 fixed `h2/k3` 持平，仍不如 HGMP Prompt。IMDB 后续应优先研究多标签协议、early stopping 或标签噪声，而不是继续扩大 metapath 网格。
4. Freebase 上固定 `h3/k3` 仍是当前最好配置，adaptive threshold 没有带来收益。
5. 后续调参应数据集分治：
   - DBLP：围绕 `max_k=8` 继续微调 `alpha=0.4/0.5/0.6`，并统计子图大小；
   - IMDB：保留 `h2/k3` 或 `degree_norm` 作为对照，优先处理多标签评估/早停；
   - Freebase：回到固定 top-k 或尝试很小范围的固定 k，而不是继续 adaptive。

##### IMDB数据集的F1计算
由于当前的F1会被argmax缩成onehot，因此改回来试一下，发现效果还不如之前的效果。



### 2026-06-29 hgprompt
hgprompt对每条边会生成多个版本，通过多个prompt * h， 然后得到不同的消息版本，其中有一个全局版本和不同边对应的版本，当前边是哪个版本，消息传递就会用哪个版本

hgprompt不是采用子图的方式，而是一种全局传递的方式。

并且在下游中，不是采用的一种直接的分类方式，而是一种接近原型的方式，few-shot得到的样本将作为原型，其它预测的节点将不断通过cosin相似度靠近这个原型节点，这种方式得到的结果是79，接近80，但是不稳定，有接近8的标准差


### 2026-06-30 HGPrompt native baseline 与 PEPrompt prototype 下游

#### 背景

为了更公平地比较 PEPrompt 与 HGPrompt，近期补充了两类实验：

1. **HGPrompt native baseline**：HGPrompt 使用自己的原生预训练 checkpoint，并在当前 PEPrompt/HGMP 对齐 split 上评估。
2. **PEPrompt prototype 下游**：PEPrompt 仍使用 HGMP/GraphCL 预训练和 PE edge prompt，但将下游 MLP head 替换成 HGPrompt 风格的 class-center/prototype 分类。

新增命令开关：

```bash
--peprompt_head_type prototype
```

prototype 下游逻辑：

- 对训练子图提取 graph embedding；
- 用训练集 embedding 按类别求 class center；
- 对 val/test embedding 计算到各 class center 的 cosine similarity；
- 用 validation loss early stopping；
- 当前仅用于单标签任务，未用于 IMDB 多标签。

#### HGPrompt native baseline

使用 HGPrompt 原生 ckpt：

- `artifacts/checkpoints/hgprompt/pretrain/ACM.gcn.ft2.hop1.seed0.best.pt`
- `artifacts/checkpoints/hgprompt/pretrain/DBLP.gcn.ft2.hop1.seed0.best.pt`

日志：

- `artifacts/logs/hgprompt_native/ACM_hgprompt_native_1shot.log`
- `artifacts/logs/hgprompt_native/ACM_hgprompt_native_10shot.log`
- `artifacts/logs/hgprompt_native/DBLP_hgprompt_native_1shot.log`
- `artifacts/logs/hgprompt_native/DBLP_hgprompt_native_10shot.log`

结果：

| Dataset | Shot | Method | Count | Micro | Macro |
| --- | ---: | --- | ---: | ---: | ---: |
| ACM | 1 | HGPrompt native | 50 | `0.7735 ± 0.1079` | `0.7519 ± 0.1216` |
| ACM | 10 | HGPrompt native | 50 | `0.8660 ± 0.0230` | `0.8635 ± 0.0244` |
| DBLP | 1 | HGPrompt native | 50 | `0.8400 ± 0.0452` | `0.8252 ± 0.0532` |
| DBLP | 10 | HGPrompt native | 50 | `0.9266 ± 0.0098` | `0.9197 ± 0.0132` |

观察：

- HGPrompt native 在 DBLP 上非常强，尤其 10-shot 达到 `macro=0.9197`。
- DBLP 1-shot pooled mean 是 `0.8252`
- 
#### HGMP64 统一预训练 ablation

为了让 HGPrompt semantic prompt 在完整双 prompt 设置下跑通，尝试过将统一上游降到 HGMP-64。

日志：

- `artifacts/logs/hgmp_hid64_unified/ACM_hgmp64_to_hgprompt_semantic_1shot.log`
- `artifacts/logs/hgmp_hid64_unified/ACM_hgmp64_to_peprompt_1shot_h3_k1-5_a0p5_adapt_nopath.log`
- `artifacts/logs/hgmp_hid64_unified/ACM_hgprompt_ckpt_to_hgprompt_semantic_1shot.log`

结果：

| Dataset | Shot | Pretrain -> Downstream | Count | Micro | Macro |
| --- | ---: | --- | ---: | ---: | ---: |
| ACM | 1 | HGPrompt native -> HGPrompt | 50 | `0.7735 ± 0.1079` | `0.7519 ± 0.1216` |
| ACM | 1 | HGMP64 -> HGPrompt | 50 | `0.6080 ± 0.0983` | `0.5868 ± 0.1147` |
| ACM | 1 | HGMP64 -> PEPrompt | 50 | `0.7351 ± 0.0850` | `0.7191 ± 0.1016` |

结论：

- HGMP-64 不是合适的统一上游，既削弱 HGPrompt，也明显削弱 PEPrompt。
- HGMP-64 可以作为“统一低维上游 ablation”，但不适合作为主实验协议。
- HGMP-512 接 HGPrompt semantic prompt 会 OOM，原因是 HGPrompt 在全图边上做 semantic prompt message passing，hidden dim 从 64 到 512 后中间边消息张量显存约放大 8 倍。

#### PEPrompt prototype 下游

为了测试“PEPrompt 表征 + HGPrompt class-center 分类”的效果，新增 PEPrompt prototype 下游实验。

使用 HGMP-512 ckpt，数据集不包含 IMDB：

- ACM: `metapath_topk_adapt h3/k1-5/a0.5/count`
- DBLP: `metapath_topk_adapt h3/k1-8/a0.5/count`
- Freebase: `metapath_topk h3/k3/count`, `feats_type=1`

日志：

- `artifacts/logs/peprompt_prototype/ACM_hgmp512_peprompt_prototype_1shot_h3_k1-5_a0p5_adapt_nopath.log`
- `artifacts/logs/peprompt_prototype/ACM_hgmp512_peprompt_prototype_10shot_h3_k1-5_a0p5_adapt_nopath.log`
- `artifacts/logs/peprompt_prototype/DBLP_hgmp512_peprompt_prototype_1shot_h3_k1-8_a0p5_adapt_nopath.log`
- `artifacts/logs/peprompt_prototype/DBLP_hgmp512_peprompt_prototype_10shot_h3_k1-8_a0p5_adapt_nopath.log`
- `artifacts/logs/peprompt_prototype/Freebase_hgmp512_peprompt_prototype_1shot_ft1_h3_k3_count.log`
- `artifacts/logs/peprompt_prototype/Freebase_hgmp512_peprompt_prototype_10shot_ft1_h3_k3_count.log`

结果：

| Dataset | Shot | PEPrompt Subgraph | Count | Micro | Macro |
| --- | ---: | --- | ---: | ---: | ---: |
| ACM | 1 | adapt h3/k1-5/a0.5 | 50 | `0.7997 ± 0.0750` | `0.7889 ± 0.0836` |
| ACM | 10 | adapt h3/k1-5/a0.5 | 50 | `0.9032 ± 0.0114` | `0.9026 ± 0.0118` |
| DBLP | 1 | adapt h3/k1-8/a0.5 | 50 | `0.5220 ± 0.0753` | `0.5147 ± 0.0769` |
| DBLP | 10 | adapt h3/k1-8/a0.5 | 50 | `0.8255 ± 0.0339` | `0.8234 ± 0.0342` |
| Freebase | 1 | fixed h3/k3, ft1 | 50 | `0.2401 ± 0.0517` | `0.1924 ± 0.0493` |
| Freebase | 10 | fixed h3/k3, ft1 | 50 | `0.3322 ± 0.0257` | `0.2760 ± 0.0260` |

观察：

- ACM 上 prototype 下游表现较强，10-shot `macro=0.9026`，高于 HGPrompt native 的 `0.8635`。
- DBLP 10-shot 与当前最佳 PEPrompt 结果一致量级，但 DBLP 1-shot 明显崩塌，仅 `macro=0.5147`，说明 prototype 下游对 DBLP 极少样本非常不稳定。
- Freebase 上 prototype 下游仍偏弱，10-shot `macro=0.2760`，与此前 Freebase 困难现象一致。
- 结论上，prototype/class-center 分类可以作为 ablation，但目前不能替代 PEPrompt 的默认 MLP 下游。


### 2026-07-02

当前整个项目的故事性比较弱，特别是在子图提取的方面，并且在dblp上有些弱，因此我的想法是修改子图的划分方式

当前的子图提取依赖hop topk 进行遍历metapath取endpoints，

1. 采用closed path，也就是target -> x -> x -> target 的metapath 来选取endpoints，然后再恢复中间节点，通过两边同时拓展的方式给中间节点一定分数。这种方式的效果很显著，如果设置endpoints 1-8, support nodes 16 1shot 能达到80，如果不限制support nodes的数量，直接为full,那么就能达到恐怖的88。 但这种方式不能用于Freebase，因为Freebase的边是单向的，如果采用closed metapath，不一定能涵盖所有的节点类型。


2. Freebase 当边类型比较多的时候，metapath的数量也会很多，也就导致按照每条metapath来计算久比较耗时

#### DBLP 
确实，如何去编造故事是比较重要的，当前的故事模式就是太简单了

### 2026-07-21
当前的工作模式太简单了，现在先要找到一个问题：问题是当前的方法并没有找到一个核心的出发点，我的出发点是什么：对于大图来说，使用全图进行任务的代价是比较大的，有些人采用了子图的构造方式，但子图难免会缺失高阶的信息，如何使用将这种高阶的信息用到这种子图模式中，并且，传统的子图方式：比如split


为了实验老师所说的高阶子图的方式，我进行了一版代码，设置了几个


原图 G
  ↓
只取目标节点类型，比如 DBLP 的 author
  ↓
通过 P^hop 得到 target-target 语义邻接
  ↓
选 K 个 target semantic anchors
  ↓
得到 assignment S
  ↓
构造高阶图 A_H = S^T A_target S
  ↓
在高阶图上传播
  ↓
回传到原始目标节点
  ↓
再通过原图随机游走扩散给其它类型节点
  ↓
构造边特征 [z_u - z_v, |z_u - z_v|, z_u * z_v]
  ↓
作为 peprompt_edge_feat 注入下游 PEPrompt


这种方式太烂了，不行，不仅时空复杂度比较高，而且性能不强。

### 2026-08-27 | DBLP | PEPrompt/HGMP-GCN | 长距离元路径节点有效性与物理距离拉近

设置：

```text
dataset=DBLP
gnn=HGMP-GCN
layers=2
shot=1
splits=0,1,2
epochs=100
prompt/mtg_layers=PEPrompt edge prompt, SpectralEmbeddingDiff dim=16
extra=比较 metapath_topk_adapt 的最大元路径长度与 support recovery；checkpoint=artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth；repeats=5；topk=1-8；alpha=0.5；rank=count；endpoint_mode=all；early_stop=val_loss；eval_mode=early_stop_only
```

结果文件：

```text
artifacts/results/long_metapath_utility/DBLP/m2_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m2_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/per_run.csv
artifacts/results/long_metapath_utility/DBLP/m2_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m2_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/long_metapath_utility/DBLP/m2_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m2_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/overall_summary.json
artifacts/results/long_metapath_utility/DBLP/m3_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m3_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/per_run.csv
artifacts/results/long_metapath_utility/DBLP/m3_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m3_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/long_metapath_utility/DBLP/m3_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m3_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/overall_summary.json
artifacts/results/long_metapath_utility/DBLP/m4_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/per_run.csv
artifacts/results/long_metapath_utility/DBLP/m4_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/long_metapath_utility/DBLP/m4_supportnone/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_supportnone_pespectralembdiff_s16/overall_summary.json
artifacts/results/long_metapath_utility/DBLP/m4_supportcount_sk16/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_supportcount_sk16_pespectralembdiff_s16/per_run.csv
artifacts/results/long_metapath_utility/DBLP/m4_supportcount_sk16/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_supportcount_sk16_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/long_metapath_utility/DBLP/m4_supportcount_sk16/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_supportcount_sk16_pespectralembdiff_s16/overall_summary.json
artifacts/logs/long_metapath_utility/DBLP/precompute_m2_supportnone.log
artifacts/logs/long_metapath_utility/DBLP/precompute_m3_supportnone.log
artifacts/logs/long_metapath_utility/DBLP/precompute_m4_supportnone.log
artifacts/logs/long_metapath_utility/DBLP/precompute_m4_supportcount_sk16.log
artifacts/logs/long_metapath_utility/DBLP/benchmark_m2_supportnone.log
artifacts/logs/long_metapath_utility/DBLP/benchmark_m3_supportnone.log
artifacts/logs/long_metapath_utility/DBLP/benchmark_m4_supportnone.log
artifacts/logs/long_metapath_utility/DBLP/benchmark_m4_supportcount_sk16.log
```

结果：

```text
m2 supportnone: count=15, micro=0.4167 ± 0.0390, macro=0.4132 ± 0.0345; seed-mean macro std=0.0330; precompute_time=21.74s
m3 supportnone: count=15, micro=0.5393 ± 0.0923, macro=0.5356 ± 0.0917; seed-mean macro std=0.0890; precompute_time=28.86s
m4 supportnone: count=15, micro=0.6523 ± 0.0233, macro=0.6439 ± 0.0262; seed-mean macro std=0.0208; precompute_time=39.16s
m4 supportcount_sk16: count=15, micro=0.8049 ± 0.0217, macro=0.8025 ± 0.0228; seed-mean macro std=0.0115; precompute_time=156.88s
```

关键观察：

```text
在 DBLP 1-shot 快速诊断中，长距离元路径节点对目标 author 明显有用：只把 endpoint 拉近时，m2 -> m3 -> m4 的 macro 从 0.4132 提升到 0.5356 再到 0.6439。
进一步把 m4 长路径上的 support/bridge 节点按路径贡献恢复到子图后，macro 提升到 0.8025，说明长距 endpoint 的收益不是孤立 endpoint 本身，而是 endpoint 与中间 typed evidence 共同构成的语义通路。
代价也很明显：m4 supportcount_sk16 的预计算时间为 156.88s，约为 m4 supportnone 的 4.0 倍、m2 supportnone 的 7.2 倍；因此“拉近物理距离”应做 selective support recovery，而不是无条件 full-support。
当前结果支持下一步方向：对 DBLP 这类 schema 明确的数据集，应该保留长距离 target-related metapath，但需要用路径贡献、类型约束或可解释 anchor 对 support 节点进行筛选；简单扩大 hop 有收益，但完整收益来自恢复长路径中的关键中间节点。
```

### 2026-08-31 | ACM | PEPrompt/HGMP-GCN | 一层预训练表示相似性的 fixed-budget 邻域重构

设置：

```text
dataset=ACM
gnn=HGMP-GCN
layers=2
shot=1
splits=0,1,2
epochs=100
prompt/mtg_layers=PEPrompt edge prompt, SpectralEmbeddingDiff dim=16
extra=新增 scripts/precompute_acm_fixed_budget_rewire.py；目标节点 paper；用 HGMP-GCN 预训练 checkpoint 的第一层 hidden state 计算 paper-paper cosine 相似性；top_similar=8；对 author/subject/term 直接邻居做重构；新增边复用原始 relation type，不使用 pseudo relation；author_budget=4，subject_budget=2，term_budget=16；paper-paper citation/ref 原始边保留；repeats=5；early_stop=val_loss；eval_mode=early_stop_only
```

结果文件：

```text
scripts/precompute_acm_fixed_budget_rewire.py
artifacts/cache/acm_fixed_budget_rewire/
artifacts/logs/acm_fixed_budget_rewire/precompute.log
artifacts/logs/acm_fixed_budget_rewire/benchmark_baseline.log
artifacts/logs/acm_fixed_budget_rewire/benchmark_add_only.log
artifacts/logs/acm_fixed_budget_rewire/benchmark_fixed_budget.log
artifacts/logs/acm_fixed_budget_rewire/benchmark_replace_only.log
artifacts/results/acm_fixed_budget_rewire/baseline/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_run.csv
artifacts/results/acm_fixed_budget_rewire/baseline/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/acm_fixed_budget_rewire/baseline/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/overall_summary.json
artifacts/results/acm_fixed_budget_rewire/add_only/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_run.csv
artifacts/results/acm_fixed_budget_rewire/add_only/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/acm_fixed_budget_rewire/add_only/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/overall_summary.json
artifacts/results/acm_fixed_budget_rewire/fixed_budget/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_run.csv
artifacts/results/acm_fixed_budget_rewire/fixed_budget/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/acm_fixed_budget_rewire/fixed_budget/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/overall_summary.json
artifacts/results/acm_fixed_budget_rewire/replace_only/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_run.csv
artifacts/results/acm_fixed_budget_rewire/replace_only/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/acm_fixed_budget_rewire/replace_only/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/overall_summary.json
```

结果：

```text
baseline star-ego: count=15, micro=0.6323 ± 0.1576, macro=0.6034 ± 0.1945; seed-mean macro std=0.1806; avg_nodes=93.04, avg_edges=184.11
add_only: count=15, micro=0.6368 ± 0.1536, macro=0.6088 ± 0.1892; seed-mean macro std=0.1798; avg_nodes=114.43, avg_edges=226.89
fixed_budget: count=15, micro=0.5788 ± 0.1390, macro=0.5536 ± 0.1634; seed-mean macro std=0.1575; avg_nodes=26.38, avg_edges=50.80
replace_only: count=15, micro=0.5457 ± 0.1305, macro=0.5143 ± 0.1603; seed-mean macro std=0.1319; avg_nodes=25.93, avg_edges=49.89
precompute_time: baseline=21.34s, add_only=22.24s, fixed_budget=18.29s, replace_only=18.03s
```

关键观察：

```text
add_only 相比 baseline 只有极小提升：macro 0.6034 -> 0.6088，说明用预训练一层表示寻找相似 paper 后，从相似 paper 邻域补充 author/subject/term 有一定信号，但第一版收益很弱。
fixed_budget 和 replace_only 明显低于 baseline，说明“直接删除原始 author/subject/term 边”在 ACM 1-shot 上不稳定；当前预算把平均子图规模从 93.04 nodes / 184.11 edges 压到约 26 nodes / 51 edges，删除过强，很可能丢掉了关键原始 term/author 证据。
replace_only 低于 fixed_budget，支持保留原始边先验是必要的；完全用相似节点投票邻域替代原始邻域不适合作为主策略。
当前第一版更像负结果：重构边复用真实 relation type 可以跑通，但 fixed-budget 不能直接用小固定预算硬删边。下一步如果继续该方向，应改为 soft pruning 或 relation-wise adaptive budget，例如只删低 IDF hub term、保留 subject 原边、对 author/term 使用不同 original_prior，而不是统一 top-B 替换。
```

### 2026-08-31 | ACM | PEPrompt/HGMP-GCN | 只重构目标节点一阶边并保留其它原始上下文

设置：

```text
dataset=ACM
gnn=HGMP-GCN
layers=2
shot=1
splits=0,1,2
epochs=100
prompt/mtg_layers=PEPrompt edge prompt, SpectralEmbeddingDiff dim=16
extra=在 scripts/precompute_acm_fixed_budget_rewire.py 中新增 *_context variant；只替换目标 paper 与 author/subject/term 的一阶出入边，其余被选中上下文节点之间的原始边保持不变；二阶 paper 按新邻居自身在原图中存在的连边恢复；paper-paper citation/ref 原始边保留；top_similar=8；author_budget=4，subject_budget=2，term_budget=16；repeats=5；early_stop=val_loss；eval_mode=early_stop_only；cache 因体积较大写入 /tmp/acm_fixed_budget_rewire_context_full
```

结果文件：

```text
scripts/precompute_acm_fixed_budget_rewire.py
artifacts/logs/acm_fixed_budget_rewire_context/precompute_context.log
artifacts/logs/acm_fixed_budget_rewire_context/size_estimate.log
artifacts/logs/acm_fixed_budget_rewire_context/benchmark_fixed_budget_context.log
artifacts/logs/acm_fixed_budget_rewire_context/benchmark_replace_only_context.log
artifacts/results/acm_fixed_budget_rewire_context/fixed_budget_context/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_run.csv
artifacts/results/acm_fixed_budget_rewire_context/fixed_budget_context/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/acm_fixed_budget_rewire_context/fixed_budget_context/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/overall_summary.json
artifacts/results/acm_fixed_budget_rewire_context/replace_only_context/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_run.csv
artifacts/results/acm_fixed_budget_rewire_context/replace_only_context/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/per_seed_summary.csv
artifacts/results/acm_fixed_budget_rewire_context/replace_only_context/ACM/1-shot/peprompt.khop_pespectralembdiff_s16/overall_summary.json
/tmp/acm_fixed_budget_rewire_context_full/fixed_budget_context/
/tmp/acm_fixed_budget_rewire_context_full/replace_only_context/
```

规模：

```text
fixed_budget_context: unique_samples=2388, avg_nodes=2354.32, avg_edges=17268.10, total_edges=41,236,214, cache_size=65G, precompute_time=308.64s
replace_only_context: unique_samples=2388, avg_nodes=2410.86, avg_edges=17763.59, total_edges=42,419,444, cache_size=67G, precompute_time=252.33s
add_only_context: only estimated, not benchmarked; avg_nodes=3134.92, avg_edges=143682.47, total_edges=343,113,750
```

结果：

```text
fixed_budget_context: count=15, micro=0.5937 ± 0.0550, macro=0.5318 ± 0.0807; seed-mean macro std=0.0475
replace_only_context: count=15, micro=0.4372 ± 0.0858, macro=0.3720 ± 0.1023; seed-mean macro std=0.0603
```

关键观察：

```text
只改目标节点一阶 author/subject/term 边、并保留二阶上下文原始边后，fixed_budget_context 没有恢复旧 baseline 表现，macro=0.5318 仍低于 baseline star-ego 的 0.6034，也略低于旧 fixed_budget 的 0.5536。说明 fixed-budget 的主要问题不只是“把其它节点自身边删掉”，更可能是目标 paper 原始一阶 author/term/subject 证据被预算重排后发生了语义偏移。
replace_only_context 进一步下降到 macro=0.3720，说明完全替换目标 paper 的原始一阶邻域即使保留新邻居的二阶原始上下文，也会引入强噪声；目标节点自身真实一阶边仍是必要先验。
context 版本显著降低 fixed_budget 的方差，但代价极高：两个 context cache 共约 132G，单个 split pkl 约 23G，benchmark I/O 和内存压力明显增加。add_only_context 的规模估计达到 avg_edges=143682.47，接近 ACM 2-hop khop 级别，因此本轮未跑 benchmark。
当前结论偏负：单纯“只改目标节点邻接，其余全部保留”不足以让相似 paper 邻域重构有效。下一步若继续该方向，应先限制二阶上下文，如对 term/subject 的 paper fanout 或 IDF hub 进行截断，并把重构从 hard replace 改成 original-prior 更强的 soft augmentation。
```

### 2026-08-31 | ACM/DBLP | PEPrompt/HGMP-GCN | TypeNeighborhoodEdge 类型邻域边编码

设置：

```text
idea=不重新训练 GNN；用固定类型传播替代 Laplacian PE 作为边 prompt 输入
node_init=节点类型 one-hot
type_propagation=undirected normalized propagation, hops=[0,1,2]
edge_feature=TypeNeighborhoodEdge = [z_src, z_dst, z_src - z_dst, |z_src - z_dst|, one_hot(canonical_etype)]
prompt=保持原 PEPrompt edge_prompt_mlp，不改下游训练逻辑
ACM subgraph=metapath_topk_adapt h3/k1-5/a0.5/count
DBLP subgraph=metapath_topk_adapt h3/k1-8/a0.5/count
shot=1
splits=0,1,2
repeats=5
epochs=100
early_stop=val_loss
eval_mode=early_stop_only
```

结果文件：

```text
artifacts/logs/type_neighborhood_edge/precompute_acm_type_neighborhood_edge.log
artifacts/logs/type_neighborhood_edge/precompute_dblp_type_neighborhood_edge.log
artifacts/logs/type_neighborhood_edge/benchmark_acm_typeonly.log
artifacts/logs/type_neighborhood_edge/benchmark_dblp_typeonly.log
artifacts/results/type_neighborhood_edge/ACM_typeonly/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/per_run.csv
artifacts/results/type_neighborhood_edge/ACM_typeonly/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/per_seed_summary.csv
artifacts/results/type_neighborhood_edge/ACM_typeonly/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/DBLP_typeonly/DBLP/1-shot/peprompt.metapath_topk_adapt_m3_k1-8_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/per_run.csv
artifacts/results/type_neighborhood_edge/DBLP_typeonly/DBLP/1-shot/peprompt.metapath_topk_adapt_m3_k1-8_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/per_seed_summary.csv
artifacts/results/type_neighborhood_edge/DBLP_typeonly/DBLP/1-shot/peprompt.metapath_topk_adapt_m3_k1-8_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/overall_summary.json
```

#### 快速协议结果

| Dataset | Method | Protocol | Count | Micro-F1 | Macro-F1 | Seed-mean macro std |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ACM | TypeNeighborhoodEdge | h3/k1-5/a0.5/count | 15 | 0.8009 +/- 0.0363 | 0.7896 +/- 0.0418 | 0.0245 |
| ACM | SpectralEmbeddingDiff | Same-protocol matched subset | 15 | 0.7877 +/- 0.0797 | 0.7758 +/- 0.0894 | 0.0649 |
| DBLP | TypeNeighborhoodEdge | h3/k1-8/a0.5/count | 15 | 0.5247 +/- 0.0682 | 0.5151 +/- 0.0693 | 0.0674 |
| DBLP | SpectralEmbeddingDiff | Same-protocol matched subset | 15 | 0.5369 +/- 0.0942 | 0.5347 +/- 0.0931 | 0.1082 |

#### DBLP 强协议补充：TypeNeighborhoodEdge 与无边级 Prompt 消融

```text
subgraph=metapath_topk_adapt h4/k1-8/a0.5/count + target_closed + full count support recovery
shot=1, splits=0-4, repeats=10, total=50
epochs=200, patience=30, early_stop=val_loss, eval_mode=early_stop_only
NoEdgePrompt=relation_prompt_alpha=0.0；relation_prompt_use_ln=false，因此 relation layer 为恒等映射
```

| Method | Edge-prompt input | Count | Micro-F1 | Macro-F1 | Seed-mean macro std | Delta macro vs PE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SpectralEmbeddingDiff | Laplacian spectral PE | 50 | 0.8869 +/- 0.0339 | 0.8851 +/- 0.0369 | 0.0206 | 0.0000 |
| TypeNeighborhoodEdge | Fixed type propagation + edge type | 50 | 0.8452 +/- 0.0592 | 0.8397 +/- 0.0691 | 0.0376 | -0.0454 |
| NoEdgePrompt | `relation_prompt_alpha=0` | 50 | 0.7916 +/- 0.0607 | 0.7876 +/- 0.0636 | 0.0611 | -0.0975 |

| Comparison | Macro-F1 delta |
| --- | ---: |
| TypeNeighborhoodEdge - NoEdgePrompt | +0.0521 |
| TypeNeighborhoodEdge - SpectralEmbeddingDiff | -0.0454 |

```text
TypeNeighborhoodEdge strong result:
artifacts/logs/type_neighborhood_edge/precompute_dblp_strong.log
artifacts/logs/type_neighborhood_edge/benchmark_dblp_strong_type_prompt.log
artifacts/results/type_neighborhood_edge/DBLP_strong_type_prompt/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_target_closed_supportcount_petypeneighborhoodedge_th0-1-2_und_eto/overall_summary.json

NoEdgePrompt strong result:
artifacts/logs/type_neighborhood_edge/benchmark_dblp_strong_no_edge_prompt.log (split 0-1 completed before interruption)
artifacts/logs/type_neighborhood_edge/benchmark_dblp_strong_no_edge_prompt_resume_s2-4.log (split 2-4 rerun)
artifacts/results/type_neighborhood_edge/DBLP_strong_no_edge_prompt_resume_s2-4/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_target_closed_supportcount/per_run.csv
full 50-run NoEdgePrompt summary=split 0-1 from the first log + split 2-4 from the resumed per_run.csv; interrupted split-2 records are excluded
```

#### DBLP 强协议补充：TypeNeighborhoodEdge 内部消融

```text
subgraph=metapath_topk_adapt h4/k1-8/a0.5/count + target_closed + full count support recovery
shot=1, splits=0-4, repeats=10, total=50
epochs=200, patience=30, early_stop=val_loss, eval_mode=early_stop_only
checkpoint=artifacts/checkpoints/hgmp/pretrain/DBLP.GraphCL.GCN.hid512.np500.seed0.pth
```

| Method | Type hops | Edge type one-hot | Feature dim | Count | Micro-F1 | Macro-F1 | Seed-mean macro std | Delta macro vs h0-1-2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TypeNeighborhoodEdge h0 | 0 | yes | 22 | 50 | 0.8351 +/- 0.0640 | 0.8287 +/- 0.0750 | 0.0416 | -0.0110 |
| TypeNeighborhoodEdge h0-1 | 0,1 | yes | 38 | 50 | 0.8384 +/- 0.0709 | 0.8298 +/- 0.0863 | 0.0435 | -0.0099 |
| TypeNeighborhoodEdge h0-1-2 | 0,1,2 | yes | 54 | 50 | 0.8452 +/- 0.0592 | 0.8397 +/- 0.0691 | 0.0376 | 0.0000 |
| TypeNeighborhoodEdge h0-1-2 no edge type | 0,1,2 | no | 48 | 50 | 0.8418 +/- 0.0710 | 0.8364 +/- 0.0810 | 0.0660 | -0.0033 |

| Comparison | Macro-F1 delta |
| --- | ---: |
| h0-1 - h0 | +0.0011 |
| h0-1-2 - h0-1 | +0.0099 |
| h0-1-2 - h0 | +0.0110 |
| edge type one-hot on - off | +0.0033 |

```text
Precompute logs:
artifacts/logs/type_neighborhood_edge/precompute_dblp_strong_h0.log
artifacts/logs/type_neighborhood_edge/precompute_dblp_strong_h0-1.log
artifacts/logs/type_neighborhood_edge/precompute_dblp_strong_h0-1-2_no_edge_type.log

Benchmark logs:
artifacts/logs/type_neighborhood_edge/benchmark_dblp_strong_h0.log
artifacts/logs/type_neighborhood_edge/benchmark_dblp_strong_h0-1.log
artifacts/logs/type_neighborhood_edge/benchmark_dblp_strong_h0-1-2_no_edge_type.log

Result summaries:
artifacts/results/type_neighborhood_edge/DBLP_strong_h0/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_target_closed_supportcount_petypeneighborhoodedge_th0_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/DBLP_strong_h0-1/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_target_closed_supportcount_petypeneighborhoodedge_th0-1_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/DBLP_strong_type_prompt/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_target_closed_supportcount_petypeneighborhoodedge_th0-1-2_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/DBLP_strong_h0-1-2_no_edge_type/DBLP/1-shot/peprompt.metapath_topk_adapt_m4_k1-8_a0p5_count_target_closed_supportcount_petypeneighborhoodedge_th0-1-2_und_etx/overall_summary.json
```

#### ACM 1-shot 补充：TypeNeighborhoodEdge 高阶类型聚合消融

```text
subgraph=metapath_topk_adapt h3/k1-5/a0.5/count
shot=1, splits=0-2, repeats=5, total=15
epochs=100, patience=30, early_stop=val_loss, eval_mode=early_stop_only
checkpoint=artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth
edge type one-hot=yes
```

| Method | Type hops | Feature dim | Count | Micro-F1 | Macro-F1 | Seed-mean macro std | Delta macro vs h0-1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TypeNeighborhoodEdge h0-1-2 | 0,1,2 | 56 | 15 | 0.8009 +/- 0.0363 | 0.7896 +/- 0.0418 | 0.0245 | 0.0000 |
| TypeNeighborhoodEdge h0-1-2-3 | 0,1,2,3 | 72 | 15 | 0.8236 +/- 0.0464 | 0.8168 +/- 0.0528 | 0.0163 | +0.0273 |
| TypeNeighborhoodEdge h0-1-2-3-4 | 0,1,2,3,4 | 88 | 15 | 0.8224 +/- 0.0493 | 0.8172 +/- 0.0523 | 0.0428 | +0.0277 |

| Comparison | Macro-F1 delta |
| --- | ---: |
| h0-1-2-3 - h0-1-2 | +0.0273 |
| h0-1-2-3-4 - h0-1-2-3 | +0.0004 |
| h0-1-2-3-4 - h0-1-2 | +0.0277 |

```text
Precompute logs:
artifacts/logs/type_neighborhood_edge/precompute_acm_h0-1-2-3.log
artifacts/logs/type_neighborhood_edge/precompute_acm_h0-1-2-3-4.log

Benchmark logs:
artifacts/logs/type_neighborhood_edge/benchmark_acm_h0-1-2-3.log
artifacts/logs/type_neighborhood_edge/benchmark_acm_h0-1-2-3-4.log

Result summaries:
artifacts/results/type_neighborhood_edge/ACM_typeonly/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/ACM_hop_h0-1-2-3/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2-3_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/ACM_hop_h0-1-2-3-4/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2-3-4_und_eto/overall_summary.json
```

#### ACM 1-shot 补充：TypeNeighborhoodEdge 无参数扩散算子消融

```text
subgraph=metapath_topk_adapt h3/k1-5/a0.5/count
shot=1, splits=0-2, repeats=5, total=15
type_hops=0,1,2,3
edge type one-hot=yes
epochs=100, patience=30, early_stop=val_loss, eval_mode=early_stop_only
checkpoint=artifacts/checkpoints/hgmp/pretrain/ACM.GraphCL.GCN.hid512.np500.seed0.pth
```

| Propagation | Definition | Feature dim | Count | Micro-F1 | Macro-F1 | Seed-mean macro std | Delta macro vs power |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| power | raw `A^k X` blocks | 72 | 15 | 0.8236 +/- 0.0464 | 0.8168 +/- 0.0528 | 0.0163 | 0.0000 |
| PPR alpha=0.15 | normalized cumulative PPR blocks | 72 | 15 | 0.8238 +/- 0.0402 | 0.8179 +/- 0.0442 | 0.0201 | +0.0010 |
| heat t=1.0 | normalized truncated heat-kernel blocks | 72 | 15 | 0.8257 +/- 0.0427 | 0.8206 +/- 0.0466 | 0.0240 | +0.0038 |

```text
Implemented args:
--peprompt_type_propagation {power,ppr,heat}
--peprompt_type_ppr_alpha 0.15
--peprompt_type_heat_time 1.0

Precompute logs:
artifacts/logs/type_neighborhood_edge/precompute_acm_h0-1-2-3_ppr0p15.log
artifacts/logs/type_neighborhood_edge/precompute_acm_h0-1-2-3_heat1.log

Benchmark logs:
artifacts/logs/type_neighborhood_edge/benchmark_acm_h0-1-2-3_ppr0p15.log
artifacts/logs/type_neighborhood_edge/benchmark_acm_h0-1-2-3_heat1.log

Result summaries:
artifacts/results/type_neighborhood_edge/ACM_hop_h0-1-2-3/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2-3_und_eto/overall_summary.json
artifacts/results/type_neighborhood_edge/ACM_diffusion_ppr0p15_h0-1-2-3/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2-3_und_eto_ppr0p15/overall_summary.json
artifacts/results/type_neighborhood_edge/ACM_diffusion_heat1_h0-1-2-3/ACM/1-shot/peprompt.metapath_topk_adapt_m3_k1-5_a0p5_count_petypeneighborhoodedge_th0-1-2-3_und_eto_heat1/overall_summary.json
```

关键观察：

```text
TypeNeighborhoodEdge 不是一阶邻域重构实验，而是替代 PE 的 prompt 输入实验；它保持子图结构不变，只改变 peprompt_edge_feat 的来源。
matched subset 指同一子图协议、同一 split/repeat 子集下的局部对照：ACM 使用 h3/k1-5/a0.5/count 的 seeds 0-2、repeats 0-4；DBLP 使用 h3/k1-8/a0.5/count 的 seeds 0-2、repeats 0-4。它不是 DBLP 当前最强 PE 配置。
ACM 1-shot 快速结果略高于同协议 SpectralEmbeddingDiff 子集，且方差更小，说明“节点类型 one-hot 固定传播 + 边类型 one-hot”确实能提供有效的类型邻域结构信号。
ACM 1-shot hop 消融显示二阶不是当前 TypeNeighborhoodEdge 的最优上限；加入三阶后 macro 从 0.7896 提升到 0.8168，说明更远的类型上下文在 ACM 上有明显价值。继续加四阶仅提升约 +0.0004 macro，且 seed-mean macro std 从 0.0163 增到 0.0428，因此 h0-1-2-3 更像当前更稳的候选。
ACM 1-shot 无参数扩散算子消融显示，裸 `A^k X` 不一定是最优；PPR alpha=0.15 相比 power 只提升约 +0.001 macro，heat t=1.0 提升约 +0.0038 macro。当前证据只能说明 heat 是一个略好的候选，不能说明扩散算子已经解决 TypeNeighborhoodEdge 的主要瓶颈。
DBLP h3 普通 adaptive 下，TypeNeighborhoodEdge 与同协议 SpectralEmbeddingDiff 都只有约 0.52-0.53 macro；这不是代码异常，而是该协议本身弱。DBLP 真正高分来自 target-closed/support recovery 这类语义端点与支撑节点恢复，而不是只靠 prompt 边特征替换。
DBLP 强协议下，TypeNeighborhoodEdge 相比 NoEdgePrompt 有稳定的正向增益，说明固定类型传播得到的边级邻域编码确实在生成 prompt 时提供了有效信号；它不是无效替代。
DBLP 强协议内部消融显示：h0 到 h0-1 只提升约 +0.001 macro，说明一阶类型邻域传播本身贡献很弱；h0-1 到 h0-1-2 提升约 +0.010 macro，二阶类型上下文是当前 TypeNeighborhoodEdge 里更主要的增益来源。
去掉 edge type one-hot 后，h0-1-2 从 0.8397 macro 降到 0.8364 macro，边类型特征有正向贡献但幅度较小；它更像稳定补充项，而不是主要性能来源。
但 TypeNeighborhoodEdge 仍落后于 SpectralEmbeddingDiff 约 4.5 macro points，且跨 split 方差更大，因此目前不能声称它已完全替代 PE。当前最准确结论是：它可减少对谱 PE 的依赖并显著优于无结构 prompt，但仍需与谱 PE 组合，或改进类型/关系传播设计后才可能达到原 PE 水平。
```
