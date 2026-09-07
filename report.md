# PEPrompt 阶段性实验汇报

## 汇报主线

本阶段主要汇报三件事：

1. HGPrompt native baseline 已经跑通。DBLP 上 HGPrompt 明显强于当前 PEPrompt，尤其 1-shot 差距最大。
2. DBLP 上 PEPrompt 的主要瓶颈在子图构建。adaptive endpoint selection、扩大 `max_k`、扩大 hop 都能提升结果。
3. HGPrompt-style prototype/class-center 下游在 ACM 上有效，但不能直接解决 DBLP 1-shot；HGMP64 统一上游也不成立。

IMDB 的 multihot F1 已经修正。修正后 PEPrompt 仍没有超过 HGMP Prompt，但这部分暂时不放表，后续单独分析多标签协议和 early stopping。

## 1. HGPrompt 与 PEPrompt 对比

| Dataset | Shot | Method | Micro-F1 | Macro-F1 |
| --- | ---: | --- | ---: | ---: |
| ACM | 1 | HGPrompt paper | 0.7404 ± 0.1249 | 0.7447 ± 0.1428 |
| ACM | 1 | HGPrompt  | 0.7735 ± 0.1079 | 0.7519 ± 0.1216 |
| ACM | 1 | PEPrompt | 0.7997 ± 0.0750 | 0.7889 ± 0.0836 |
| ACM | 10 | HGPrompt  | 0.8660 ± 0.0230 | 0.8635 ± 0.0244 |
| ACM | 10 | PEPrompt | 0.9032 ± 0.0114 | 0.9026 ± 0.0118 |
| DBLP | 1 | HGPrompt paper | 0.8231 ± 0.1067 | 0.7744 ± 0.1205 |
| DBLP | 1 | HGPrompt  | 0.8400 ± 0.0452 | 0.8252 ± 0.0532 |
| DBLP | 1 | PEPrompt h4 | 0.6287 ± 0.0634 | 0.6061 ± 0.0922 |
| DBLP | 10 | HGPrompt  | 0.9266 ± 0.0098 | 0.9197 ± 0.0132 |
| DBLP | 10 | PEPrompt h4 | 0.8588 ± 0.0121 | 0.8573 ± 0.0121 |
| Freebase | 1 | HGPrompt paper | 0.2264 ± 0.0794 | 0.2108 ± 0.0572 |
| Freebase | 1 | PEPrompt | 0.2401 ± 0.0517 | 0.1924 ± 0.0493 |
| Freebase | 10 | PEPrompt | 0.3322 ± 0.0257 | 0.2760 ± 0.0260 |

HGPrompt paper 行由论文百分制结果换算为小数。当前已跑通的 HGPrompt native 暂时只有 ACM 和 DBLP；Freebase 先用论文结果作为参照。

## 2. DBLP 10-shot 子图构建实验

| Setting | Micro-F1 | Macro-F1 |
| --- | ---: | ---: |
| fixed h3/k2 | 0.6317 | 0.6306 |
| adaptive h3/k1-5/a0.5 | 0.7883 ± 0.0318 | 0.7861 ± 0.0317 |
| adaptive h3/k1-8/a0.5 | 0.8255 ± 0.0339 | 0.8234 ± 0.0342 |
| adaptive h4/k1-8/a0.5 | 0.8588 ± 0.0121 | 0.8573 ± 0.0121 |

10-shot 结果说明，DBLP 需要更宽、更高阶的语义邻域。`h3 -> h4` 继续提升，说明当前 PEPrompt 的短板更像是采样覆盖不足，而不是单纯的分类头问题。

## 3. DBLP 1-shot 结果

| Setting | Micro-F1 | Macro-F1 |
| --- | ---: | ---: |
| PEPrompt old h3/k1-8/a0.5 | 0.5220 ± 0.0753 | 0.5147 ± 0.0769 |
| PEPrompt h4/k1-8/a0.5 | 0.6287 ± 0.0634 | 0.6061 ± 0.0922 |
| HGPrompt native | 0.8400 ± 0.0452 | 0.8252 ± 0.0532 |

原来的 DBLP 1-shot 是 `micro=0.5220`、`macro=0.5147`。h4 后提升到 `micro=0.6287`、`macro=0.6061`，但距离 HGPrompt native 仍然很远。

## 4. Prototype/Class-Center 下游

| Dataset | Shot | Method | Micro-F1 | Macro-F1 |
| --- | ---: | --- | ---: | ---: |
| ACM | 1 | PEPrompt prototype | 0.7997 ± 0.0750 | 0.7889 ± 0.0836 |
| ACM | 10 | PEPrompt prototype | 0.9032 ± 0.0114 | 0.9026 ± 0.0118 |
| DBLP | 1 | PEPrompt old h3 | 0.5220 ± 0.0753 | 0.5147 ± 0.0769 |
| DBLP | 1 | PEPrompt h4 | 0.6287 ± 0.0634 | 0.6061 ± 0.0922 |

prototype 思路在 ACM 上表现很好，但 DBLP 1-shot 没有被解决。可能原因是每类只有一个训练样本时，class center 基本退化成单个训练点，类别中心不稳定。

## 5. HGMP64 统一上游实验

| Setting | Micro-F1 | Macro-F1 |
| --- | ---: | ---: |
| HGPrompt native -> HGPrompt | 0.7735 ± 0.1079 | 0.7519 ± 0.1216 |
| HGMP64 -> HGPrompt | 0.6080 ± 0.0983 | 0.5868 ± 0.1147 |
| HGMP64 -> PEPrompt | 0.7351 ± 0.0850 | 0.7191 ± 0.1016 |

HGMP64 不能作为统一上游。它接 HGPrompt 下游明显失败，接 PEPrompt 也比 HGMP512 弱；而 HGMP512 接 HGPrompt semantic prompt 容易 OOM，所以当前不适合强行统一预训练 checkpoint。

## 下一步

DBLP 后续优先继续改子图构建，而不是只调 head：

1. 做 h4 的 path-adapt 与 no-path 对照。
2. 尝试 `h4/k1-12`、`a0.4/a0.6`。
3. 设计 coverage-aware target-biased metapath walk：从不同类型节点出发，约束覆盖更多节点类型，并提高后段走回目标节点类型的概率。
