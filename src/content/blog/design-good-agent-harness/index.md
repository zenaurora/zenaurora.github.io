---
title: "Agent Harness 中的长时记忆"
description: "思考长周期 Agent 任务中的上下文压缩、检索式记忆与文件系统存储。"
date: 2026-09-12
authors:
  - maokaihe
tags:
  - Agent
  - Agent Harness
  - Memory
draft: true
---

agent harness里面，在long horizon agent场景下存在memory的问题。
之前的主流做法都是做两种：压缩上下文，对之前的流程进行摘要； 或者使用rag技术进行存储，在需要的时候进行查询；

因此一个更好的方式就是把一些大的产物放到文件系统中，然后模型自身
