---
title: "Research Ideas：面向非平稳多变量时间序列的动态预测"
description: "围绕条件参数调制、动态变量分组与测试时自适应，探索能够随当前工况改变预测行为的多变量时间序列模型。"
date: 2026-08-13
authors:
  - maokaihe
tags:
  - Time Series
  - Research Ideas
  - Non-stationarity
  - Multivariate Forecasting
---

## 总体动机

现有多变量时间序列预测主要面临两个问题：

1. **时间非平稳性**：训练阶段学到的固定模型，难以持续适应测试阶段不断变化的工况和数据分布。
2. **变量依赖的双刃剑问题**：Channel-Independent（CI）建模更加鲁棒，但会忽略变量间的有用信息；Channel-Dependent（CD）建模能够利用跨变量信息，却容易引入无关变量和噪声。

这个系列的核心目标是：

> **让预测模型根据当前输入窗口的 condition，动态决定“模型应该如何预测”以及“哪些变量之间应该交互”；如果仍然无法适应当前分布，再在 test time 对少量参数进行在线调整。**

## 三个研究方向

| Idea                                      | 核心问题                    | 主要机制                                      |
| ----------------------------------------- | ----------------------- | ----------------------------------------- |
| Condition-aware Parameter Modulation      | 当前状态下，模型应该怎么预测？         | 根据输入窗口生成少量动态参数，调制固定的 forecasting backbone |
| Condition-aware Dynamic Variable Grouping | 当前状态下，模型应该看哪些变量？        | 预测未来变量关系，动态决定变量分组或 interaction mask       |
| Test-Time Adaptation / Training           | 训练时学到的动态规则仍然不够时，如何继续适应？ | 使用自监督信号或已经揭示的标签，只更新少量适应参数                 |

三个方向并不是互斥的。它们可以共享同一个 Condition / Relation Encoder：

```text
                  Multivariate Context X
                           │
                           ▼
              Condition / Relation Encoder
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        Condition State z       Relation Matrix R
                │                     │
                ▼                     ▼
       γ / β / Adapter           Dynamic Groups
                │                     │
                └──────────┬──────────┘
                           ▼
                     CI Backbone
                           │
                           ▼
                          Ŷ
```

它们分别解决三个层面的问题：

$$
\begin{aligned}
\text{Idea 1: }& X \rightarrow z \rightarrow \text{动态调制参数}, \\
\text{Idea 2: }& X \rightarrow R \rightarrow \text{动态变量交互}, \\
\text{Idea 3: }& X \rightarrow \mathcal L_{\mathrm{TTA}}
\rightarrow \nabla \rightarrow \theta_{\mathrm{adapt}}'.
\end{aligned}
$$

## 可能的统一主线

一个更完整的方案是把条件建模和测试时自适应结合起来：

$$
X
\xrightarrow{\text{Condition Encoder}}
\theta_X^{(0)}
\xrightarrow{\text{Test-Time Optimization}}
\theta_X^*.
$$

Condition Network 首先根据当前工况直接产生一组较好的 adaptation parameters，然后只对这组少量参数进行一步或数步 test-time optimization。

因此，整体研究主线可以概括为：

$$
\boxed{
\text{Static CI Forecasting}
\rightarrow
\text{Condition-aware Forecasting}
\rightarrow
\text{Selective Channel Interaction}
\rightarrow
\text{Test-Time Adaptive Forecasting}
}.
$$

从现有工作密度看，Idea 2 的 channel grouping 已经有 CCM、DGCformer、DUET 等直接相关工作；Idea 3 的 TTA 也已经形成 TAFAS、PETSA、COSA、FAC 等路线。相对而言，更值得继续挖掘的是 **Idea 1 与 Idea 2 / Idea 3 的结合**：让统一的 condition representation 同时决定模型参数、变量关系，以及 test-time adaptation 的初始化或强度。

这是基于当前文献脉络得到的研究方向判断，而不是已有论文已经给出的结论。后续的三篇子博客会分别展开这三个 Idea。

---

如果用一个 Encoder 去提取标准差，这个标准差不能是负数，所以可以用 Softplus 激活：

$$
\operatorname{Softplus}(x) = \log\left(1 + e^x\right).
$$

$$
Z_n
= \operatorname{Encoder}_{\mu}(X)
+ \epsilon \odot
\operatorname{Softplus}\left(\operatorname{Encoder}_{\sigma}(X)\right).
$$

---


如果用来写毕设的话，目前倾向于的方向：

设计一个基于long context的一个模型，然后利用latent space空间的形状，再来一个test time adapt；

这个是不涉及到具体的backbone选择什么样子的模型.
如果需要额外设计模型的话，可以从APN（自适应patch），DUET（soft group），还有VLM/ViT的方法来做辅助信息提取。

考虑一下是不是可以利用vision的方案来做long context identify去给出一个新的中间量/或者生成一个参数调制模型

对于TTA来说，可以使用sana那个论文做参考，在predictor/encoder/decoder上面分别做消融实验来验证效果；然后可以学习latent TSF那样做可视化来验证 latent space的loss得到的结果更好，是在空间中可以动态连续的发展。

之前还想到过把时间序列的本身做patch去提取pattern 作为一个类似于 vocabulary的东西，预测的时候是从词表中拿出东西去做加权softmax聚合

Recast是离散的不重叠的embedding而不是重叠的，所以我们可以在这个角度去重新考量面对存在面对存在重叠的情况如何 根据预测出来的embedding来恢复真正的预测值


---

可能可以用到的话术


这个利用codebook的思想和我的vocabulary思想很像
**ReCast**（Reliability-aware Codebook-Assisted Time series forecasting，**可靠性感知的码本辅助时间序列预测**），
- **传统方法的局限**：传统时间序列模型通常依赖“全局分解”（把数据强行拆成趋势、季节性和残差）。但在真实世界中，数据往往由**局部的、复杂的、高度动态的模式**主导，全局分解效果很差。
- ReCast 框架主要通过以下三个机制来实现“轻量”且“鲁棒”的预测：
- **基于码本的局部模式量化（Patch-wise Quantization）**： 将时间序列切分成小块（patch），并使用一个**可学习的码本（learnable codebook）** 将这些局部模式编码为离散的嵌入向量。这相当于用一本“字典”来紧凑地表示数据中稳定的、重复出现的规则结构，大大降低了计算复杂度。
- **双路径架构（Dual-path Architecture）**： 因为量化过程会丢失一些细节，ReCast 设计了两条并行的路径：
    1. **量化路径**：高效建模和预测那些规则的、可被码本捕获的结构。
    2. **残差路径**：专门负责捕捉和重构量化过程中丢失的“不规则波动”（残差变化）。两者结合，既保证了效率，又挽回了精度。
- **可靠性感知的码本更新策略（核心贡献）**： 这是论文的最大亮点。为了让模型适应非平稳数据（分布偏移），它提出了一种增量更新码本的方法。通过**分布鲁棒优化（DRO）** 方案，从多个互补的视角融合出“可靠性因子”，并以此作为权重来指导码本的修正。这意味着模型能自动判断哪些数据模式是“可靠”的，从而在面对数据分布突变时保持稳健，不会被噪声带偏。

recast里面如何做cluster以及几个score打分的东西可以仔细看看，设计的比较复杂但是适合水论文。可以再结合APT那个时间序列论文里面 prototype learn的东西，如何让每个prototype间隔远一点，能够更有区分度一点

三个权重本质上分别看三个问题：**这个 codeword 代表得好不好、相比过去变了多少、是不是稀有/新模式。**

- **$w_{\mathrm{rep}}$：Representational Quality**

  看当前 cluster center $\hat{s}_k^t$ 能不能很好地代表分到它这一类的 patches。核心就是看簇内重构误差：

  $$
  \operatorname{error}_k
  = \left\|
  B_k\left(
  \operatorname{Rec}\left(\hat{S}^t(\tilde{P}^t) \mid \hat{S}^t\right)
  - \tilde{P}^t
  \right)
  \right\|_2^2.
  $$

  误差越小，说明这个中心越能代表自己的成员，所以 $w_{\mathrm{rep},k}$ 越大。简单理解：**这个 codeword 自己“代表得准不准”。**

- **$w_{\Delta}$：Historical Consistency**

  看当前 pseudo codeword 和上一 epoch 的正式 codeword 差多少：

  $$
  \left\|\hat{s}_k^t - s_k^{t-1}\right\|_2^2.
  $$

  论文里设计成差得越大，$w_{\Delta,k}$ 越大。它的含义其实不是“越一致越高”，而是：

  > 如果当前 cluster 本身又是可靠的，但它与历史 codeword 差很多，说明数据分布可能真的变了，需要更积极更新 codebook。

  简单理解：**“新模式和旧模式相比变了多少”。**

- **$w_{\mathrm{je}}$：OOD Sensitivity**

  看这个 codeword 是不是一个比较少见、比较新颖的模式。论文用当前所有 patch 到这个中心的距离构造一个 joint-energy 风格的分数：

  $$
  w_{\mathrm{je},k}
  = 1 -
  \frac{
  \exp\left(\sum_i \left|\tilde{p}_i^t - \hat{s}_k^t\right|\right)
  }{
  \exp\left(\sum_j \sum_i \left|\tilde{p}_i^t - \hat{s}_j^t\right|\right)
  + \epsilon
  }.
  $$

  论文的解释是：某个 cluster 被选择得越少、越像 rare/OOD pattern，就给它更高权重，防止 codebook 只集中在少数常见模式上。

  简单理解：**“这个 codeword 是不是稀有的新模式”。**

最后三个分数组成：

$$
z_k^t
= \left[
w_{\mathrm{rep},k}^t,
w_{\Delta,k}^t,
w_{\mathrm{je},k}^t
\right].
$$

再通过 DRO 做一个偏保守的 soft-min 融合，得到最终：

$$
\hat{w}_k^t.
$$

所以一句话记忆就是：

$$
\boxed{
\begin{aligned}
w_{\mathrm{rep}} &: \text{代表得好不好}, \\
w_{\Delta} &: \text{和历史差多少}, \\
w_{\mathrm{je}} &: \text{是不是稀有或新模式}.
\end{aligned}
}
$$

这三个分数的定义和用途就在论文 3.4 Reliability-aware Scoring 部分。


不再以原始时间分辨率处理序列，而是将时间序列划分为不重叠或部分重叠的分块，使模型能够在紧凑表示上运行并缩短序列长度。尽管这些方法在长时程预测中有效，但它们通常依赖连续嵌入，缺乏显式机制来利用现实世界时间序列中普遍存在的重复性局部形态、

其他的可用的论文，到时候在看

timestack；patchtst；APT，APN，itransformer，autoformer

the forecast after the forecast
