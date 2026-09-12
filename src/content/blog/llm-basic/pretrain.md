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

## Bert的预训练任务

包含两个：
Masked LM，mask部分token，让模型根据上下文推理被mask的词
next sentence prediction，随机抽取句子对，让模型判断第二句是不是第一句的后续

Bert的MLM任务和早期的训练方法不一样，gpt1时期都是从左到右单向训练的，而bert是随机的把一些tokens替换为mask token。

具体而言：

1. 每个输入的句子中，随机挑选15%的token作为预测目标
2. 对于这15%的token，做以下处理：
	 - 80% 被替换为特殊的`[MASK]`
	 - 10%被替换为词汇表中的一个随机的词
	 - 10%保持不变，让模型即使在没有mask标记的时候依然可以正确输出，弥合训练和实际推理时候的差异

模型在训练的时候 会在被mask或者被替换的位置输出一个概率分布，计算交叉熵loss，由于只有15%的token是预测目标，所以大大节省了计算量。并且这样的训练方式可以让模型去实现双向理解，使得它对于语言的理解能力比当时单向训练的模型强很多


NSP 任务：

1. 从训练预料中随机抽取句子，50%的情况B是A的下一个句子，50%的情况下不是



CLIP   的EOS token embedding：

`[sos] xxxxx [eos]`

sos 表示 start of sequence
eos 表示 end of sequence

对于词表大小v，编码器维度d 有vxd这个嵌入矩阵
里面每一个元素表示某一个词在某一个维度上的对应的值。

所以eos对应的id在其中的一行，它初始化的时候是随机初始化。

clip 使用因果mask，每个位置只能看到自己和以前的token，所以位于最后的eos的位置可以看到前面所有token的信息，因此clip把最后一层的eos位置的输出向量拿出来作为整段本文的全局表示，也就是拿出来之后通过一个linear映射，作为text的feature
