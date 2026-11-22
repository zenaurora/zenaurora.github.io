---
title: "SAG 阅读笔记"
description: "记录传统 RAG 在全局宏观问答与多跳推理上的不足，以及 SAG 这条思路的初步想法。"
date: 2026-11-21
authors:
  - maokaihe
tags:
  - RAG
  - Graph RAG
  - LLM
---

![SAG2 的索引与检索流程](./assets/sag2-indexing-and-retrieval-workflow.png)

传统的 RAG 存在全局宏观问答无力，多跳推理的问题