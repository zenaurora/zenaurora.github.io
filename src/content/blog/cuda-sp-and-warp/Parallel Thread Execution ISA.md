---
title: "Parallel Thread Execution ISA（PTX）"
description: "介绍 NVIDIA PTX 虚拟 GPU 指令集，以及它在 CUDA 编译流程中的位置。"
date: 2026-09-08
order: 1
authors:
  - maokaihe
tags:
  - CUDA
  - GPU
draft: true
---

Parallel Thread Execution ISA ,简称PTX，是英伟达定义的虚拟GPU指令集

长得很像汇编那种东西，但是并不是GPU真正执行的机器码，是一种类似于IR的中间状态的东西，后续会被继续翻译为更底层的形式

- NVIDIA GPU 真正执行的是 **SASS**，也就是某一代 GPU 架构的物理机器指令。

- PTX 是更稳定、更抽象的中间层。老 PTX 可以由新显卡驱动 **JIT 编译**成对应新架构的 SASS，因此有一定跨代兼容性。
