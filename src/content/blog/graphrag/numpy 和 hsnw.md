---
title: "向量检索：numpy 暴力检索与 HNSW"
description: "对比 numpy 全量矩阵乘法的暴力检索与 HNSW 分层小世界图的近似最近邻，并给出两者的查询代码。"
date: 2026-11-21
order: 3
authors:
  - maokaihe
tags:
  - Vector Search
  - HNSW
  - numpy
  - ANN
---

使用 numpy 的时候是暴力检索：
```python
def _cosine_query(self, query, top_k, better_than_threshold, filter_lambda=None):
    query = normalize(query)                      # ① 查询向量归一化
    if filter_lambda is None:
        use_matrix = self.__storage["matrix"]     # ② 全库矩阵，形状 (N, D)
        filter_index = np.arange(len(self.__storage["data"]))
    else:
        ...                                       # 带过滤条件时只取子集
        use_matrix = self.__storage["matrix"][filter_index]

    scores = np.dot(use_matrix, query)            # ③ 矩阵 × 向量 = N 个相似度
    sort_index = np.argsort(scores)[-top_k:]      # ④ 全排序，取最大的 k 个
    sort_index = sort_index[::-1]                 #    反转成从大到小
    sort_abs_index = filter_index[sort_index]

    results = []
    for abs_i, rel_i in zip(sort_abs_index, sort_index):
        if better_than_threshold is not None and scores[rel_i] < better_than_threshold:
            break                                 # ⑤ 低于阈值就停
        results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i]})
    return results
```

每次查询都要把 `query` 和全库 $N$ 条向量做一次完整的矩阵乘法（时间复杂度 $O(N \cdot D)$），再做全量排序（$O(N \log N)$）


HNSW（Hierarchical Navigable Small World，分层可导航小世界图） 是目前 RAG 向量数据库（如 Milvus、Qdrant、pgvector、Chroma）最主流的索引算法。它放弃了“100% 精确”，通过预先构建多层图索引，换取 近似最近邻（ANN）的毫秒级检索（查询复杂度降至约 $O(\log N)$）。

- 多层结构（借鉴 Skip List 跳表思想）
    - 高层（Layer 2 / Layer 1，稀疏层）： 只有极少数节点被随机抽中进入高层，节点之间的连线跨度极大，充当“高速公路”。
    - 最底层（Layer 0，稠密层）： 包含全部 $N$ 个向量节点，节点只与周围距离最近的邻居相连，充当“城市街道”。
    - 建索引时，每个节点的最高层数按指数衰减概率随机决定，参数 `M` 限制了每个节点最多能连几条边（如图中 `M = 5`）。
        
- 贪心路由检索（借鉴 NSW 小世界图思想）
    - 粗定位（上层下探）： 查询从最顶层的固定入口 `Entry point` 出发，计算当前节点及其邻居与 `query` 的距离，每次都跳向离 `query` 更近的邻居（图中蓝色箭头），直到找不到更近的节点（到达该层的局部最近点 `Local minimum`），然后垂直落入下一层继续搜索。
    - 精搜索（Layer 0 扩散）： 落到最底层 `Layer 0` 时，搜索起点已经非常接近目标区域。此时算法会维护一个大小为 `ef`（图中绿色圆圈代表的搜索广度）的动态候选队列，在局部邻域内充分探索，最后从队列中截取距离最近的 `top_k` 个结果。

```python
import numpy as np
import hnswlib

class HNSWStorage:
    def __init__(self, dim, max_elements=100000):
        # 建索引阶段（离线完成，比 numpy 多出建图开销）
        self.index = hnswlib.Index(space="cosine", dim=dim)
        # M 控制每个节点的最大连接数，ef_construction 控制建图时的搜索精度
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.data = []

    def _hnsw_query(self, query, top_k, better_than_threshold, filter_lambda=None):
        query = normalize(query)    # ① 查询向量归一化（与 numpy 一致）

        # ② 设定检索时的搜索广度 ef（必须大于等于 top_k，越大越准但越慢）
        self.index.set_ef(max(top_k * 2, 50))

        # ③ 图上贪心游走，只计算路径附近少量节点的距离，无需全库矩阵乘法
        #    注意：过滤条件需在图遍历过程中动态判断（见下文说明）
        labels, distances = self.index.knn_query(
            query,
            k=top_k,
            filter=filter_lambda  # hnswlib 支持传入基于 id 的过滤回调
        )

        # ④ knn_query 返回的已经是按距离从小到大排好序的 top_k，无需全量 argsort
        #    注意：hnswlib 的 cosine 空间返回的是“余弦距离 (1 - 相似度)”，需转回相似度
        scores = 1.0 - distances[0]
        sort_abs_index = labels[0]

        results = []
        for abs_i, score in zip(sort_abs_index, scores):
            if better_than_threshold is not None and score < better_than_threshold:
                break                                 # ⑤ 低于阈值就停（与 numpy 一致）
            results.append({**self.data[abs_i], f_METRICS: float(score)})
        return results
```