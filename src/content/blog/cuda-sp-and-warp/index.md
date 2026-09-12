---
title: "CUDA 基础：SM、SP、Warp 和 Block Size"
description: "从一次 kernel 启动出发，理清线程、block、warp 与 GPU 硬件的对应关系，以及 block size 为什么会影响性能。"
date: 2026-09-06
authors:
  - maokaihe
tags:
  - CUDA
  - GPU
draft: true
---

看 CUDA 时，很容易把 SM、SP、thread、warp、block 当成同一套层级里的东西。先把它们分开：**SM、SP 描述硬件；thread、block、grid 描述程序如何组织工作；warp 连接了线程与硬件执行。**

## 先认清名字

| 名称         | 全称 / 含义                         | 可以怎么理解                                                                                                                                         |
| ---------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| SM         | Streaming Multiprocessor，流式多处理器 | GPU 上容纳线程并调度执行的硬件单元                                                                                                                            |
| SP         | Streaming Processor，流处理器        | 旧资料中对标量运算核心的称呼，常与 CUDA Core 对应                                                                                                                 |
| kernel     | 核函数                             | 在 GPU 上由许多线程执行的函数                                                                                                                              |
| thread     | 线程                              | kernel 的一个执行实例，有自己的索引和局部状态                                                                                                                     |
| block      | 线程块                             | 一组可以共享数据、进行块内同步的线程。一个grid由许多block组成，block由许多线程组成，同样可以有一维、二维或者三维。block内部的多个线程可以同步（synchronize），可访问共享内存（share memory）。                           |
| grid       | 网格                              | 一次 kernel 启动中的全部 block，block可以是一维二维三维，然后内部的所有thread共享一个global memory                                                                           |
| warp       | 线程束                             | 同一 block 内按顺序划分的一组 32 个线程。**warp是调度和运行的基本单元**。warp中所有threads并行的执行相同的指令。warp由SM的硬件warp scheduler负责调度，一个SM同一个时刻可以执行多个warp，这取决于warp scheduler的数量。 |
| block size | 每个 block 的线程总数                  | 启动 kernel 时指定的参数                                                                                                                               |

SM 的名字里有 **Multiprocessor**，SP 没有。一个 SM 内含多个运算单元，还包括 warp 调度器、寄存器文件、shared memory 等资源。SP 这一叫法常见于早期架构资料；现代 GPU 还区分不同运算管线，不能用一个 SP 概念概括所有执行单元。

SP 的历史含义可参考 NVIDIA 架构论文 [NVIDIA Tesla: A Unified Graphics and Computing Architecture](https://www.techpowerup.com/gpu-specs/docs/nvidia-tesla-architecture.pdf)；现代 CUDA 的概念划分见 [Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。

## 从一次 kernel 启动看 thread、block 和 grid

假设要把两个数组逐元素相加，每个线程负责一个元素：

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// host 端代码片段：假设 d_a、d_b、d_c 已分配好设备内存，输入已就绪。
int n = 1000;
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

`__global__` 标记核函数；`<<<gridSize, blockSize>>>` 指定启动多少个 block，以及每个 block 放多少个线程。这里的 `blockSize` 是普通变量名，kernel 内通过内建变量 `blockDim` 获取块的尺寸。

| 表达式 | 含义 | 本例的值 |
| --- | --- | --- |
| `gridDim.x` | grid 在 x 方向的 block 数 | 4 |
| `blockIdx.x` | 当前 block 的编号 | 0～3 |
| `blockDim.x` | block 在 x 方向的线程数 | 256 |
| `threadIdx.x` | 当前线程在 block 内的编号 | 0～255 |

例如，block 2 中的 thread 7 处理 `2 × 256 + 7 = 519` 号元素。

这次启动有 `4 × 256 = 1024` 个逻辑线程。最后 24 个线程的 `i >= 1000`，不会读写数组。末尾的边界判断不能省掉。

block 和 grid 也可以是二维或三维。例如 `dim3 block(16, 16)` 表示每个 block 有 256 个线程，而不是 16 个。一般情况下：

```text
block size = blockDim.x × blockDim.y × blockDim.z
```

维度主要用于方便地对应图像、矩阵等数据。更多启动语法和索引示例见 [Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)。

## Block 分配给 SM，线程划分成 Warp

对于这里讨论的普通 CUDA kernel，一个 block 的全部线程在同一个 SM 上执行；一个 SM 可以同时驻留多个 block，具体数量取决于资源是否够用。block 数量可以远大于 SM 数量，硬件会陆续安排它们执行。

因此，`blockIdx.x = 2` 不表示“在第 2 个 SM 上执行”。程序也不能依赖普通 block 按编号顺序执行。这些约束见 [Thread Blocks and Grids](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids)。

每个 block 内部再按线程的线性编号划分 warp：

```text
一个 256-thread block
  warp 0：thread   0～31
  warp 1：thread  32～63
  ...
  warp 7：thread 224～255
```

block中thread数目是32的倍数。这是因为同一个block必须在一个SM内，而SM的Warp调度是32个线程一组进行的。
例如 33 个线程需要两个 warp，第二个只有一个有效线程，而不是与另一个 block 的线程拼成完整 warp。

二维、三维 block 先按 x 最快变化的顺序展开：

```cpp
int tid = threadIdx.x
        + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
int warpId = tid / 32;  // block 内的 warp 编号
int laneId = tid % 32;  // warp 内的位置
```

所以 `dim3 block(16, 16)` 的第一个 warp 覆盖前两行，每行 16 个线程。它并不是“一行一个 warp”。划分规则见 [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)。

## 线程数为什么可以比 SP 数多？

**线程不会各自永久占有一个 SP。** SM 保存驻留线程的执行状态，warp 调度器从已经准备好的 warp 中选择指令，交给相应执行单元处理。某个 warp 等待数据或指令结果时，其他就绪 warp 可以继续推进。

因此要区分两件事：驻留了多少线程，以及一个周期能执行多少运算。一个 SM 即使驻留了上千个线程，也不意味着每个周期都能让这些线程各完成一条指令。

同样，warp 有 32 个线程，不代表它必须独占 32 个 SP，也不代表一条 warp 指令必定在一个周期内完成。吞吐取决于 GPU 架构、指令类型和对应执行管线，不能用“SP 数除以 32”推算 SM 能驻留多少个 warp。执行和调度机制见 [SIMT Architecture](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)。

## Block Size 为什么影响性能？

先区分两个上限：**每个 block 最多多少线程**，以及**每个 SM 最多驻留多少线程**。它们是不同参数。当前常见 CUDA GPU 的前者为 1024，但实际启动还要满足设备各维度限制、kernel 限制和资源要求，不能看到 1024 就直接填满。

可通过 `cudaGetDeviceProperties` 查看 `maxThreadsPerBlock`、`maxThreadsDim`、`maxThreadsPerMultiProcessor` 和 `multiProcessorCount` 等字段；kernel 自身的限制可通过 `cudaFuncGetAttributes` 查询。各架构的硬件上限见 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)。

影响 block size 选择的资源主要有：

- 每个 SM 的驻留线程数、warp 数和 block 数上限。
- 每个线程占用的寄存器，以及 SM 的寄存器总量。
- 每个 block 使用的 shared memory，以及 SM 可提供的容量。

用一个**假设的配置**算一下：某 SM 最多驻留 2048 个线程、64 个 warp，最多 16 个 block。先忽略寄存器和 shared memory 的限制。

| Block size | 每块 warp 数 | 最多驻留 block 数 | 驻留 warp 数 | 理论 occupancy |
| --- | --- | --- | --- | --- |
| 32 | 1 | 16 | 16 | 25% |
| 128 | 4 | 16 | 64 | 100% |
| 256 | 8 | 8 | 64 | 100% |
| 1024 | 32 | 2 | 64 | 100% |

这里 32-thread block 先撞到了“最多 16 个 block”的限制，线程容量还有剩余。反过来，如果每个 block 使用 48 KiB shared memory，而 SM 可供这些 block 使用的容量为 96 KiB，那么最多只能驻留两个 block；256-thread block 此时只能提供 16 个驻留 warp。

occupancy 是“SM 上 active（驻留）warp 数 / 支持的最大驻留 warp 数”。驻留 warp 也可能正在等待，因此它不是算力利用率。上表中相同的 occupancy，也不意味着相同的性能。

实际可以先试 128 或 256，并优先选 32 的倍数。然后在目标 GPU 上测量 kernel 时间，结合寄存器、shared memory 和访存行为调整。更大的 block 不保证更高 occupancy，更高 occupancy 也不保证更快。上述取舍见 [CUDA Best Practices：Execution Configuration Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration-optimizations)。

## Warp 内分支与同步

warp 采用 SIMT（Single Instruction, Multiple Threads）执行模型。同一 warp 内线程走不同分支时，会出现 warp divergence：不同路径上的指令只对相应线程生效，其他 lane 暂时被屏蔽。不同 warp 走不同分支，本身不构成 warp 内分歧。见 [Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)。一条相同的指令，对多个 thread 自己的数据分别执行。

线程之间交换数据时还需要正确同步：

- `__syncthreads()` 用于 block 内同步，不能同步整个 grid；放在条件分支中时，需要保证整个 block 对该条件的判断一致。
- warp 内协作需要按操作要求使用 `__syncwarp(mask)` 或带同步语义的 warp 原语，不能仅凭“在同一 warp”假设隐式同步成立。

Volta 及后续架构支持 independent thread scheduling，更不能依赖线程永远逐条锁步推进来写通信代码。同步语义见 [Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)。

## 回到开头的例子

`add<<<4, 256>>>(...)` 启动 4 个 block，每个 block 有 256 个线程、8 个 warp，整个 grid 共 1024 个线程、32 个 warp。

这 4 个 block 由硬件分配到 SM，每个 block 内的 warp 再由 SM 调度执行。即使 GPU 有 80 个 SM，这次启动最多也只能让 4 个 SM 承担这个 kernel 的工作；但数组只有 1000 个元素，单纯为铺满 SM 增加工作并不划算。

读一段 CUDA 启动代码时，可以依次算清楚：一共多少 block、每块多少线程、每块多少 warp，再结合每块的资源需求判断一个 SM 能容纳多少工作。SP 数量描述硬件运算能力，不能代替这些计算。


----

在Cuda编程角度来说主要是关注grid，block，thread。

```
Thread 0 ─┐
Thread 1  │
...       ├──→ Warp 0
Thread31 ─┘
             │
             │ 下一条是什么指令？
             │
      ┌──────┼────────┬─────────┐
      ▼      ▼        ▼         ▼
    FP32    INT     Load/Store  Tensor
    Core    Core       Unit      Core
```
scheduler 调度的是 Warp 的下一条 instruction

block size是线程块的大小，如果是256 就是有256个线程，每32个线程形成一个warp。
block数量是一个grid中的block数量，通常写作 `gridDim` 或 `gridSize`

写一个kernel的时候：
`kernel<<<gridDim, blockDim>>>(...);``gridDim` 指定这次调用要创建多少个 thread blocks，`blockDim` 指定每个 block 里有多少 threads。CPU/Host 每调用一次 CUDA kernel，就通常产生一个新的 grid。

一次 kernel launch 会创建一个 grid；grid 里有很多 block；每个 block 里有很多 thread；硬件再把每个 block 里的 thread 按 32 个一组划成 warp。所有这些 thread 都执行“同一个 kernel 函数的代码”，只是每个 thread 的索引和数据不同。

GPU 会把 blocks 分配到 SM，然后每个 SM 的 warp scheduler 再从当前 resident 的 warp 里挑 ready warp 发射下一条指令。

---

### warp divergence

同一个 warp 内，不同 thread 对分支条件得到不同结果。
```
if (threadIdx.x < 16) {
    foo();
} else {
    bar();
}
```

导致
```
Thread 0~15  → foo()
Thread 16~31 → bar()
```
导致需要先不能所有thread并行一起跑，所以在编程的时候需要避免让同一个 warp 内的 thread 经常走不同的控制流。