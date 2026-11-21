---
title: "搜索结果中的流式 Top-K"
description: "记录使用 nucleo-matcher 为搜索候选项打分，以及流式 Top-K 的待整理思路。"
date: 2026-10-24
order: 3
authors:
  - maokaihe
tags:
  - Rust
  - Search
  - Top-K
draft: true
---

## 候选项打分

每次用户搜索时，需要基于用户的 query 对所有候选项打分。

我们使用 Helix 开源的 `nucleo-matcher` 库进行模式匹配。对于每个 query，通过模糊匹配为各个 candidate 的匹配程度打分。

## 待补充

获取到每个 candidate 对应的分数之后，如何进行流式 Top-K 筛选。
