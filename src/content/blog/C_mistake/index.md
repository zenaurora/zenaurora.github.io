---
title: "pragraming language learn - C's mistake"
description: ""
date: 2026-06-15
authors:
  - maokaihe
image: ./assets/banner.png
tags:
  - C
---

根据 https://digitalmars.com/articles/C-biggest-mistake.html 这个博客简单总结的

首先来说 C语言的问题其实主要源于当时的时代限制，当时面对的都是资源有限的机器，所以评价它的设计缺陷的时候应该考虑当时的环境，而不能用现在的新语言的特性去苛责。

## 指针和数组的混淆

C语言中，函数参数看起来是数组的一些参数，其实并不是数组，而是指针。
比如：
```c
void foo(char a[]) 
// 其实完全就是
void foo(char *a)
```
这导致C语言编译器无法知道数组的长度信息，a也仅仅是一个指向第一个元素的指针

数组本身的概念应该是 起始地址+长度，但是C语言没有在这个方面做设计

为了弥补这个缺陷，使用了其他方式：

### 1.使用 `\0` 作为字符串结尾

```c
char c = "qwert";
// 在内存中： 
// 'q','w','e','r',t','\0'
```
但是这样的设计不太安全，最后的结尾可能会被缓冲区覆盖，以及函数写入的时候可能会超过真实的容量的限制

### 2.给函数加上一个长度字段

```c
void fun(int a[],size_t len); 
```

### 3.通过某些注释，文档等人工约束

---

所以C语言在使用strcpy之类的函数的时候，很容易出现越界问题

后来标准库提供了一些更安全的版本，比如要求传入缓冲区的大小：
`snprintf(dst, sizeof dst,"%s",src)`

此外，C++ 也继承了这些问题，以至于c++ 标准库不得不引入其他工具
```cpp
std::string
std::vector
std::array
std::span
...
```
这些标准库的容器可以知道自己的长度。

### 其他可能的解决方案

可以给C语言加上一个新的语法，比如
```c
void foo(char a[..])
```
这里面a表示一个fat pointer,包含了数组的起点指针和数组长度

