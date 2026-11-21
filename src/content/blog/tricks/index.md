---
title: "技术随记"
description: "记录学习过程中遇到的技术技巧与实现思路。"
date: 2026-10-24
authors:
  - maokaihe
tags:
  - Notes
draft: true
---

## Heaviside 阶跃函数的近似梯度

前向传播仍然使用真正的 Heaviside 阶跃函数，反向传播时则使用 sigmoid 导数作为近似梯度。
