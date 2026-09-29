---
title: "BERT 预训练与 CLIP 文本表示"
description: "整理 BERT 的 MLM、NSP 预训练任务，以及 CLIP 使用 EOS token 表示文本的方式。"
date: 2026-09-10
order: 6
authors:
  - maokaihe
tags:
  - LLM
  - NLP
  - Pretraining
  - BERT
  - CLIP
draft: true
---

# BERT 的预训练任务

BERT 原始论文提出了两个预训练任务：Masked Language Model（MLM）和 Next Sentence Prediction（NSP）。

## Masked Language Model

GPT-1 等模型采用从左到右的单向语言模型，而 BERT 随机选择 token 作为预测目标，因此可以利用目标词左右两侧的上下文。

1. 在输入中随机挑选 15% 的 token 作为预测目标。
2. 对这些目标 token 做如下处理：
   - 80% 替换为特殊 token `[MASK]`；
   - 10% 替换为词汇表中的随机 token；
   - 10% 保持不变，使模型在没有 `[MASK]` 标记时也能完成预测，减小预训练与下游使用之间的差异。

模型只在这些目标位置上输出概率分布并计算交叉熵损失。由于每次只预测约 15% 的 token，计算量较小；同时，模型能够利用双向上下文建立更完整的语义表示。

## Next Sentence Prediction

从训练语料中构造句子对 `(A, B)`：50% 的情况下，B 是 A 在原文中的下一句；另外 50% 的情况下，从语料中随机抽取 B。模型通过 `[CLS]` 位置的输出判断 B 是否为 A 的后续句子。

## CLIP 的 EOS 表示

CLIP 的文本输入可以写成：

```text
[sos] xxxxx [eos]
```

`sos` 表示 start of sequence，`eos` 表示 end of sequence。对于词表大小为 `v`、编码器维度为 `d` 的模型，token embedding 是一个 `v × d` 的矩阵，每一行对应一个 token 的向量。`[eos]` 对应的向量初始化时通常是随机的，语义会在训练过程中逐渐学到。

CLIP 的文本 Transformer 使用因果 mask，每个位置只能看到自己和前面的 token。由于 `[eos]` 位于序列末尾，它的输出可以汇总前面整段文本的信息，因此 CLIP 取最后一层 `[eos]` 位置的向量作为全局文本表示，再通过线性投影得到 text feature。

相比于 Word2Vec 的静态词向量，BERT 生成的是上下文相关的表示，因此能够在一定程度上区分一词多义。

- `[CLS]` 通常被视为整个句子的全局表示，用于分类等下游任务。
- `[SEP]` 是分隔标记，用于区分两个句子或标记输入的结束。

目前主流的大语言模型多采用 decoder-only 架构和因果 attention；BERT 这类 encoder-only、双向 attention 模型仍常用于理解类任务。

## BERT 的一些改进版本

### RoBERTa

RoBERTa 移除了 NSP 任务，并将 BERT 的静态 masking 改为动态 masking：每次构造 batch 时重新随机选择要遮挡的 token。它还把 BERT 原来的 WordPiece 改为 GPT-2 风格的 BPE 分词，以获得更灵活的子词切分。

### ALBERT

BERT 的 token embedding 维度与 Transformer 隐藏层维度相同。ALBERT 认为二者承担的功能不同，不必强制相等，因此先把 token 映射到较低维度 `E`，再映射到 Transformer 的隐藏维度 `H`。当 `E ≪ H` 时，性能基本不受影响，同时可以显著减少 embedding 参数量。此外，多个 Transformer 层共享参数，进一步降低了模型参数量。

ALBERT 把 NSP 改为了 **Sentence Order Prediction（SOP）**：正样本是原文中连续且顺序正确的两个句子；负样本来自同一篇文档中的句子，但将它们交换顺序。相比 NSP，SOP 更关注句子顺序和连贯性，减少了模型利用“主题不同”这一简单线索进行判断的机会。

### SpanBERT

SpanBERT 将 token-level masking 改为 span-level masking：先采样一个 span 长度，再从句子中选择起点，连续遮挡一段 token，更贴近模型理解连续短语和实体的需求。

它还新增了 **Span Boundary Objective（SBO）**：使用被遮挡 span 两端边界 token 的表示，预测 span 内的 token，显式训练模型从边界上下文恢复整段内容。

### XLNet

XLNet 的核心创新是 permutation language modeling 和 two-stream attention。它打乱预测顺序，并预测排列中的后续 token，既不需要 `[MASK]`，又能利用双向上下文。但 XLNet 的训练目标和实现较为复杂，工程上已经较少采用这种设计。

### DeBERTa（v1–v3）

以往模型通常将内容表示和位置表示相加后再计算注意力。DeBERTa 认为这样可能让位置信息干扰内容信息，因此设计了解耦的注意力机制：

$$
Attention = Attention(Q^c,K^c,V^c) + Attention(Q^p,K^p,V^p)
$$

token $i$ 和 $j$ 之间的注意力可以分为四个部分（DeBERTa 主要使用前三项）：

$$
\begin{aligned}
A_{i,j} &= \left\{ H_i,\, P_{i|j} \right\} \times \left\{ H_j,\, P_{j|i} \right\}^{\top} \\
&= H_i H_j^{\top} + H_i P_{j|i}^{\top} + P_{i|j} H_j^{\top} + P_{i|j} P_{j|i}^{\top}
\end{aligned}
$$

DeBERTa 还改进了 MLM。预测 `[MASK]` 时，模型使用上下文 token 的内容和相对位置信息；同时额外注入绝对位置信息，因为绝对位置在部分任务中也有助于判断被遮挡词。

![DeBERTa 的 MLM 解码过程](./assets/Pasted%20image%2020260915223519.png)

图中的 `I₁` 表示绝对位置表示 `p_abs`。由于该交互过程会重复多次，绝对位置信息会逐步融入隐藏状态。最终预测 `[MASK]` 时，再把绝对位置表示作为 Query，从 Encoder 的上下文表示 `H` 中重新读取信息。论文将这部分称为 decoder，但它更接近一个增强版的 MLM prediction head。
