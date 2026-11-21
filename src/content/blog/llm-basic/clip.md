---
title: "CLIP 的对比学习流程与数值稳定性"
description: "通过 PyTorch 示例和公式，整理 CLIP 的编码投影、L2 归一化、对称对比损失与 Softmax 数值稳定性。"
date: 2026-10-24
order: 7
authors:
  - maokaihe
tags:
  - LLM
  - CLIP
  - PyTorch
draft: true
---

## 编码与归一化

**L2 归一化**就是：把一个向量除以它自己的 **L2 范数**（也就是欧几里得长度），让它的长度变成 1。结果向量叫**单位向量**，方向不变，只改变长度。

1. 图像经过 image encoder 得到 $I_f$。
2. 文本经过 text encoder 得到 $T_f$。
3. 二者都需要再经过一个 linear 映射和 L2 归一化，得到 $I_e$、$T_e$。

下面是一个最小 PyTorch 版本的关键流程。代码里写了注释，并在后面附上对应公式。

## CLIP 关键流程代码（PyTorch）

### 1. 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder  # 图像编码器 f_v
        self.text_encoder = text_encoder    # 文本编码器 f_t

        # 两个线性投影层，把不同模态映射到统一维度
        # v_i = W_v f_v(x_i)
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim, bias=False)
        # t_i = W_t f_t(y_i)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim, bias=False)

        # 可学习温度参数：logit_scale = log(1 / tau)
        # 初始温度 tau = 0.07，所以 logit_scale = log(1 / 0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.tensor(torch.log(torch.tensor(1.0 / 0.07))))

    def forward(self, images, texts):
        # 1. 编码
        img_feat = self.image_encoder(images)      # [B, D_v]，对应 f_v(x_i)
        txt_feat = self.text_encoder(texts)        # [B, D_t]，对应 f_t(y_i)

        # 2. 投影到共享空间
        img_emb = self.image_proj(img_feat)        # [B, E]，对应 v_i = W_v f_v(x_i)
        txt_emb = self.text_proj(txt_feat)         # [B, E]，对应 t_i = W_t f_t(y_i)

        # 3. L2 归一化
        # \hat v_i = v_i / ||v_i||_2
        img_emb = F.normalize(img_emb, dim=-1)
        # \hat t_i = t_i / ||t_i||_2
        txt_emb = F.normalize(txt_emb, dim=-1)

        # 4. 相似度矩阵，带温度
        # logit_scale = exp(logit_scale_param) = 1 / tau
        logit_scale = self.logit_scale.exp().clamp(max=100)  # 防止数值爆炸
        # s_ij = \hat v_i^T \hat t_j * logit_scale
        # 等价于 s_ij = \hat v_i^T \hat t_j / tau
        logits = logit_scale * img_emb @ txt_emb.t()  # [B, B]

        return logits, img_emb, txt_emb
```

### 2. 对称对比损失

```python
def clip_loss(logits):
    """
    logits: [B, B]，logits[i, j] = s_ij
    i 表示图像索引，j 表示文本索引。
    对角线是正样本对 (i, i)。
    """
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)  # 正样本标签 [0, 1, ..., B-1]

    # 图像 -> 文本：对每一行做 softmax
    # L_i2t = -1/B * sum_i log( exp(s_ii) / sum_j exp(s_ij) )
    loss_i2t = F.cross_entropy(logits, labels)

    # 文本 -> 图像：对每一列做 softmax，等价于对 logits.t() 的每一行做 softmax
    # L_t2i = -1/B * sum_i log( exp(s_ii) / sum_j exp(s_ji) )
    loss_t2i = F.cross_entropy(logits.t(), labels)

    # 对称平均
    # L = (L_i2t + L_t2i) / 2
    return (loss_i2t + loss_t2i) / 2
```

### 3. 训练循环

```python
# 假设 image_encoder 和 text_encoder 已经定义好
model = CLIP(image_encoder, text_encoder, embed_dim=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for images, texts in dataloader:
    # 前向：得到相似度矩阵 logits
    logits, img_emb, txt_emb = model(images, texts)

    # 计算对称对比损失
    loss = clip_loss(logits)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

> 注意：训练时梯度会更新：
>
> 1. 图像编码器 `image_encoder`
> 2. 文本编码器 `text_encoder`
> 3. 图像投影层 `image_proj`
> 4. 文本投影层 `text_proj`
> 5. 温度参数 `logit_scale`
>
> 两个 linear 只是投影头，负责把不同模态映射到同一个嵌入空间；训练时也会更新编码器和温度参数。

## 对应公式汇总

### 1. 编码与投影

图像：

$$
f_v(x_i) \rightarrow v_i = W_v f_v(x_i)
$$

文本：

$$
f_t(y_i) \rightarrow t_i = W_t f_t(y_i)
$$

### 2. L2 归一化

$$
\hat v_i = \frac{v_i}{\|v_i\|_2}, \quad
\hat t_i = \frac{t_i}{\|t_i\|_2}
$$

### 3. 相似度矩阵

$$
s_{ij} = \hat v_i^\top \hat t_j \cdot \exp(\text{logit\_scale})
$$

或者写成温度形式：

$$
s_{ij} = \frac{\hat v_i^\top \hat t_j}{\tau}, \quad
\tau = \frac{1}{\exp(\text{logit\_scale})}
$$

其中 $\tau$ 是温度，CLIP 初始 $\tau = 0.07$。

### 4. 对称对比损失

图像到文本：

$$
L_{i2t} = -\frac{1}{B}\sum_{i=1}^{B}
\log \frac{\exp(s_{ii})}{\sum_{j=1}^{B} \exp(s_{ij})}
$$

文本到图像：

$$
L_{t2i} = -\frac{1}{B}\sum_{i=1}^{B}
\log \frac{\exp(s_{ii})}{\sum_{j=1}^{B} \exp(s_{ji})}
$$

最终损失：

$$
L = \frac{1}{2}\left(L_{i2t} + L_{t2i}\right)
$$

## 关键点

- 负样本来自同一个 batch 内的其他样本，不需要人工标注。
- 对角线 $s_{ii}$ 是正样本对，非对角线 $s_{ij}, i \ne j$ 是负样本对。
- batch 越大，负样本越多，对比学习效果通常越好。CLIP 原始训练 batch 达到 32768。
- 推理时通常只用 L2 归一化后的 embedding 计算余弦相似度：

  $$
  \text{sim}(v, t) = \hat v^\top \hat t
  $$

## Softmax 的数值稳定性

“对每一个值减去当前 batch 最大值”是 CLIP 训练中会用到的做法，但它属于一个更通用的数值计算技巧，叫 **“减去最大值稳定化”（max-shift stabilization）**，通常和 Softmax 一起出现。

它的核心目的是**防止指数运算 `exp()` 时发生数值溢出（overflow）**。

Softmax 函数里有个关键步骤是计算 `e^x`。如果 `x` 是一个很大的数（比如 1000），`e^1000` 会变成一个巨大的数字，超出计算机浮点数能表示的范围，结果就是 `inf`（无穷大），训练会直接崩溃。

这个技巧利用了一个数学性质：**Softmax 的概率结果，在给所有输入同时减去一个常数后，是完全不变的**。

公式上就是：

$$
\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
$$

所以，我们可以安全地让所有 `z` 都减去一个常数 `c`。而在数值计算中，**最安全的选择就是让 `c` 等于当前这批数据里的最大值** `max(z)`。

### 减去最大值后发生了什么？

执行 `z - max(z)` 之后，情况就变得安全了：

1. **最大的那个值变成了 0**：因为 `max(z) - max(z) = 0`。
2. **所有其他的值都变成了负数**：因为任何数减去最大值都小于或等于 0。
3. **指数运算的结果被限制在 (0, 1] 之间**：`e^0 = 1`，`e^负数` 是一个小于 1 的正数。这样就**绝对不会发生溢出**了。
4. **分母不会变成 0**：因为至少有一个值是 1（来自最大值），分母的和至少为 1，避免了除以零的风险。

### 和之前讨论的 L2 归一化有什么区别？

这两个操作的目标完全不同，但都是为了让 CLIP 训练更稳定：

| 操作 | L2 归一化 | 减去最大值（Max-shift） |
| --- | --- | --- |
| 解决的问题 | **模型“作弊”**：防止模型通过放大向量长度来降低 loss，而非学习语义方向。 | **数值“溢出”**：防止 `exp()` 计算产生无穷大，导致训练崩溃。 |
| 作用对象 | 图像和文本的 embedding 向量 | Softmax 的输入 logits |
| 目的 | 让点积变成余弦相似度，迫使模型学习方向。 | 保证 Softmax 计算在计算机浮点数范围内安全进行。 |

### 计算顺序

在 CLIP 的实现中，**通常会先对 embedding 做 L2 归一化，然后计算相似度 logits，最后在计算交叉熵损失（内部包含 Softmax）时，对 logits 使用“减去最大值”的技巧来确保数值稳定**。这两个操作各司其职，共同保障了 CLIP 训练的稳定和有效。
