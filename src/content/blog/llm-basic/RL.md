---
title: "RL 与 SFT 的对比笔记"
description: "记录 RL 与 SFT 的关系、相对优势加权和采样策略等学习要点。"
date: 2026-10-24
order: 8
authors:
  - maokaihe
tags:
  - LLM
  - Reinforcement Learning
draft: true
---

## RL 退化为 SFT 的条件

1. 每个 prompt 只采样一条输出。
2. 奖励使用 0/1 奖励。
3. 去掉 KL、clip 项。

## RL 相比 SFT 的优势

1. 如果使用 $A_i = (r_i - \operatorname{mean}(r)) / \operatorname{std}(r)$，则可以表示负奖励，也就是把错误的概率也拉低。SFT 只能把正确的拉高。
2. SFT 每个样本是等权，RL 可实现组内相对优势加权。RL 也可以做到细粒度的分配。
3. RLVR 来自 on-policy 自身的采样，可以学习到对应数据集中不存在的轨迹，比如 DeepSeek 的 R1 经过 RL 可以实现涌现。
4. RL 还减少了 SFT 的灾难性遗忘现象。

## Off-policy

Off-policy 是指可以使用以前的数据、别人的数据，或其他策略 $\pi_\theta$ 下的数据。
