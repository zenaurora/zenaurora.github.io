---
title: "My learn about minilsm"
description: "I try a minilsm"
date: 2026-06-14
authors:
  - maokaihe
image: ./assets/banner.png
tags:
  - lsm
---

This is a index blog about minilsm.

访问 [minilsm](https://skyzh.github.io/mini-lsm/00-preface.html) 获取官方的教程

自从开始接触rust之后，一直想做一个偏向底层的一个项目，但是能力有限，只能在网络上找一些教程

偶然间在推特上面刷到了这个，于是在闲暇之余开始做这个项目，大概从25年10月开始做的，做到11月底，把week1的东西做差不多了，
当时的ai还不是很厉害，在测试时候经常遇到一些小问题，debug半天，这时候我也锻炼了一点在vscode里面调试rust的能力，当时也意识到ai模型其实还是很有局限性的(不过现在最强的模型比那个时候强多了)

这个项目我还是很推荐的，跟着教程可以一点点了解lsm数据库内核的基本的设计，以及在上手的时候逐渐熟练rust的语法和标准库，特别是和
底层io打交道的地方，比如file，buffer的读写操作。此外还有关于如何使用`mutex`的思考，如果最小化lock的范围，以及使用rwlock来实现一些功能的时候read和write的时机。

我会在下面的sub blog里面写我从这个过程中可以学习到的东西。


