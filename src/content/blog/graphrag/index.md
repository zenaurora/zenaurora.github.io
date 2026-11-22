---
title: "Graph RAG 核心概念笔记"
description: "整理 Graph RAG 的实体与关系抽取、图与社区报告的存储结构，以及 local 与 global 两种查询模式。"
date: 2026-11-21
authors:
  - maokaihe
tags:
  - Graph RAG
  - RAG
  - Knowledge Graph
  - LLM
---

Graph RAG 核心概念：

## 实体和关系

文档切分为chunk 之后，每一个 chunk 都会给大模型，输出得到实体关系。

prompt 大致是：
```
-Goal-      任务目标：从文本里识别指定类型的实体及其关系
-Steps-     三个步骤 + 输出格式要求：
              1. 抽实体  → ("entity"<|>名字<|>类型<|>描述)
              2. 抽关系  → ("relationship"<|>源<|>目标<|>描述<|>强度)
              3. 用 ## 做列表分隔符
              4. 结束输出 <|COMPLETE|>
-Examples-  3 个few-shot 示例（Alex/Taylor 那个故事等）
-Real Data- 最后才是真正的数据：
            Entity_types: {entity_types}
            Text: {input_text}
            Output:
```

返回的响应是：

```
("entity"<|>"Alex"<|>"person"<|>"Alex is a character who experiences frustration.")##
("entity"<|>"Taylor"<|>"person"<|>"Taylor is portrayed with authoritarian certainty.")##
("relationship"<|>"Alex"<|>"Taylor"<|>"Alex is affected by Taylor's certainty."<|>"7")##
("relationship"<|>"Taylor"<|>"The Device"<|>"Taylor shows reverence towards the device."<|>"9")<|COMPLETE|>
```

然后会 gleaning 追问把可能遗漏的东西补全。把「原文 + 第一次输出」作为对话历史，让 LLM 再看一遍补漏；每轮问一句「还有遗漏吗 yes/no」，不是 yes 就停。最大轮数由配置控制。

这里面涉及到几个符号：

`tuple delimiter is <|>, record delimiter is ##, completion delimiter is <|COMPLETE|>`
这些符号在程序里面需要预先定义好，然后我们可以接下来解析它

实体记录是：
```
("entity"<|>"Alex"<|>"person"<|>"Alex is a character...")
 ①标签     ②名字    ③类型       ④描述
```

关系记录是：
```
("relationship"<|>"Alex"<|>"Taylor"<|>"...description..."<|>"7")
 ①标签          ②源实体    ③目标实体   ④关系描述           ⑤强度(数字)
```

第一个字段是程序用来区分实体还是关系的。

解析流程：

首先把字符串按照`["##", "<|COMPLETE|>"]` 进行切分，得出一条一条的记录。

然后按照正则`\(.*\)`去掉两侧的括号

然后每一个记录用`<|>`进行拆分，得到一个`['"entity"', '"Alex"', '"person"', '"...描述..."']`这样的数组，接下来就好办了，基于第一个字段来判断是实体抽取还是关系抽取，得出一个结构化的字典。
```
• 实体: {'entity_name': 'ALEX', 'entity_type': 'PERSON', 'description': ..., 'source_id': chunk_key}
• 关系: {'src_id': 'ALEX', 'tgt_id': 'TAYLOR', 'weight': 7.0, 'description': ..., 'source_id': chunk_key}
```

由于多个 chunk 会得到多个同名实体和边，因此需要进行聚合操作：
1. 合并同名实体：如果 entity type 有好几种，则取众数；
2. 所有的实体描述需要进行去重和排序，然后用`<SEP>` 拼接起来。（排序本身并没有啥用，而是为了让结果可复现）
3. 如果描述拼起来太长了，就需要调用一次 LLM 进行压缩
4. 把 entity type，描述，和 source id 拼起来作为 node data 插入到图中

对于边来说也是需要进行聚合：
1. 边的 weight 进行求和，如果一个关系在多个 chunk 里面提到说明证据更多，因此 weight 需要求和，这是 graph rag 的重要设计
2. 描述太长也是用 LLM 压缩一下

> 实体抽取在 nano graph rag 里面除了使用这种 prompt 的方式之外，还可使用 DSPy 的方式来进行。DSPy 的思想是把 prompt 变成编程的方式，不再手写一大堆提示词。



### 数据库存储的东西

chunks_vdb 文本块向量（naive 模式才有），entities_vdb 实体向量（local 模式） 放在向量数据库

full_docs,text_chunks,community_reports 这三个存储在 KV 数据库中

nano graph 里面，图使用的是 python 的 networkx 库，节点是实体，边就是关系，然后在磁盘上就是一个 GraphXML 格式的文件。

`entities_vdb`中 存储的主键是`"ent-" + md5(实体名)`,向量是实体名+描述拼接之后embedding 的向量，最终 `save()` 写到 `vdb_entities.json` 的真实结构是长这样：
```json
{
  "embedding_dim": 1024,
  "data": [
    {"__id__": "ent-3f7a...", "entity_name": "ALEX"}
  ],
  "matrix": "base64 编码的向量矩阵, data[i] ↔ matrix[i] 按下标对齐"
}
```

社区报告community_reports：

| 字段                | 来源                   | 说明                                          |
| ----------------- | -------------------- | ------------------------------------------- |
| `level`           | 聚类                   | 层级，0 最粗                                     |
| `title`           | `community_schema()` | **固定为 `"Cluster {id}"`**，没有实际语义             |
| `nodes`           | 节点聚合                 | 社区内实体名列表                                    |
| `edges`           | 节点聚合                 | `[[src, tgt], ...]`，无向、端点已排序                |
| `chunk_ids`       | 节点 `source_id` 并集    | 社区覆盖的原文块                                    |
| `occurrence`      | 计算                   | `本社区 chunk 数 / 所有社区中最大的 chunk 数`，重要度        |
| `sub_communities` | 节点集合包含关系推导           | 下一层中节点集合是本社区子集的社区 id                        |
| `report_string`   | LLM 输出转换             | Markdown 全文，**喂给 LLM 用的**                   |
| `report_json`     | LLM 原始输出             | **给程序用的**结构化报告，包括 title，rating，findings 等字段 |

社区内的节点：
每个 节点的存储的数据包括：
- entity type 是实体类型，比如 PERSON 和 ORGANIZATION
- description 是多重描述去重后用`<SEP>` 拼接的，太长了需要压缩
- source id 表示 这个实体的来源的 chunk id 列表，也使用`<SEP>`拼接
- clusters 是一个 json 字段 形如
`[{"level":0,"cluster":3},{"level":1,"cluster":17}]`


社区里的边：edge data
- weight 
- description
- source id 来源 chunk id
- order 

>  向量存储后端还可以使用 hnsw，HNSW = Hierarchical Navigable Small World（分层可导航小世界），是一种近似最近邻 (ANN) 算法。向量组织成一个多层图，查询时候从稀疏的高层开始，向下查找，快速找到最近的向量。
>  
## 社区

在图拓扑结构上内部联系紧密、与外部联系相对稀疏的实体集合。GraphRAG 会为每个社区调用 LLM 生成一份社区摘要报告

社区中的 level 是 使用 leiden 算法进行层级社区发现的时候，聚类层级的深度。
```
level 0：对整张图跑一次普通 Leiden → 得到初始划分（最粗）
         找出成员数 >= max_graph_cluster_size 的社区
level 1：把这些"超标"社区单独抽成子图再跑 Leiden → 拆成更小的社区
level 2：继续直到所有社区都小于阈值，或拆不动为止
```

为啥还需要找出成员数 >= max_graph_cluster_size 的社区：是因为 leiden 算法本身并没有一个社区内成员数量的约束，所以在经过这个 leiden 算法还需要手动判断再来继续拆分。Leiden 算法本身确实是包含了 local moving，refinement，和 aggregation 三个阶段的，但是业务逻辑层面是无需关心这个的

只有所在社区被拆过的节点才会有 level >=1 的条目。 如果一个社区没有超标就不会出现这个问题。

一个实体不会属于多个社区，但是它 node data 的 clusters 字段可能存在这样的：
`[{"level": 0, "cluster": 0}, {"level": 1, "cluster": 5}]`

这里表示 level0 中属于社区 0，level 1 中属于社区 5，这里的社区 5 其实是社区 0 的一个子集。如果一个社区没有被拆分过，那么这个社区里面所有的实体只有 level 0 一条数据。

社区之间要么是包含关系，要么就是没有关系，不存在那种相交的关系。

注意，社区本身也是有 level的，表示社区本身的一个层级高度。
## 查询

### local模式（比 naive 好，没有 global 那么重）

把query embedding 之后，从实体向量数据库检索 top-k 个最相似的实体(默认 20)。

检索的是索引阶段写入的实体名+描述 向量，而非直接的实体名称向量

然后用这些实体名字从图中批量的获取信息，包括 node data 和 node degrees。其中度数表示实体的 rank，rank 越高，就越重要。后续会作为重要性给 LLM 发送

接下来，从每个实体的 clusters 字段取出它所处的社区, 并统计这些社区被命中的次数，如果一个社区被频繁命中说明更重要。然后把社区报告整合并截断

同时，需要从每个实体的 source id 中看一下来自哪些 chunk，再取得实体的一跳的邻居节点，看看邻居都来自哪些 chunk。 如果某一个chunk 同时命中该实体 和它的邻居，说明这个 chunk 更重要，会relation counts 加分。然后按顺序把 chunk 拼起来截断

还需要取出命中实体的每一个边，进行排序（边有度数，也有一开始存进来的 relation 强度），使用`(rank,weight)`降序。然后把边的信息合起来，截断。


最后变成一个结构化文本，包含社区报告，实体，关系，和源文本块。
```
-----Reports-----     ```csv ...```   （社区报告）
-----Entities-----    ```csv ...```   （实体表：id/name/type/description/rank）
-----Relationships----- ```csv ...``` （关系表：id/source/target/desc/weight/rank）
-----Sources-----     ```csv ...```   （源文本块）
```

把这个和回答格式要求拼起来作为 system prompt，用户的原始问题作为 user prompt 送给 LLM 得到一个 response

> 不支持多跳，向量检索命中的是 topk 个实体的直接向量的边，取出边也只是排序+截断，不会走多跳。但是 LLM 会拿到对应实体的社区报告和边表写的摘要

```
假设图结构是：A — B — C（A、B、C 在同一个社区）
种子实体只命中了 A
  → 图遍历只拿到边 A-B（1-hop），拿不到 B-C（2-hop）
  → 但 A 所在社区的报告是 LLM 对着 {A,B,C} 和一整张社区边表写的
  → 报告里已经用自然语言描述了 B 和 C 的关系
  → 这段描述被塞进 local 上下文 → LLM 间接"看到"了 2-hop 信息
```

### Global 模式

在 nano-graphrag 里，global 模式不使用 `entities_vdb`，也不需要给 query 做 embedding。它直接从索引阶段生成的社区报告出发，适合回答「这些项目复盘里，哪些问题在不同团队反复出现？」这类需要综合多份材料的问题。

**先挑社区报告。** 从图存储的 `community_schema()` 里筛出 `level <= QueryParam.level` 的社区。默认 `level=2`，所以第 0、1、2 层都可能入选，父社区和子社区也可以同时被选中。按 `occurrence`（社区覆盖的 chunk 数相对于最大社区的比例）降序取前 `global_max_consider_community` 个，默认 512 个。

用这些社区 ID 去 `community_reports` 批量读报告。跳过缺失的报告，以及 `report_json.rating` 低于 `global_min_community_rating`（默认 0）的报告，再按 `(occurrence, rating)` 降序排列。到这里为止，选哪些报告还和问题内容无关。

global 的主排序键是 occurrence 而不是 rating。也就是说"覆盖文本多的大社区"比"LLM 觉得重要的社区"优先级更高。

**Map：从每组报告提取能回答问题的要点。** 按上面的顺序把报告逐份装进各组，用 `report_string` 的 token 数估算每组大小，预算由 `global_max_token_for_community_report` 控制（默认 16,384）。每组拼成 `id, content, rating, importance` 的 CSV，其中 `content` 是报告正文，`importance` 是社区的 `occurrence`。然后并发请求 LLM，每组 CSV 放进各自的 system prompt，原始问题作为 user prompt。LLM 返回一组 `points`，并给每条要点打 `0–100` 分，表示它对回答当前问题有多大帮助；回答不了就给 0 分。这个 `score` 是在查询时打的，报告本身的 `rating` 在索引阶段就已经生成了。[Map 源码](https://github.com/gusye1234/nano-graphrag/blob/acb35c065614eb5a2f5f1be9a56b235f5a2e0a7a/nano_graphrag/_op.py#L970-L1014)

```json
{
  "points": [
    {"description": "几份复盘都提到，跨团队交接时找不到最新的接口文档。", "score": 85}
  ]
}
```

**Reduce：合并各组的要点，再回答一次。** 把各组的 `points` 放到一起，去掉缺少 `description` 或 `score <= 0` 的条目，按 `score` 降序排列。然后用同一个 token 预算截断要点列表，高分的先保留。最后拼出来的上下文像这样，`Analyst 0` 表示这个要点来自第 0 组报告：

```text
----Analyst 0----
Importance Score: 85
几份复盘都提到，跨团队交接时找不到最新的接口文档。
```

把这份要点列表和 `response_type`（回答格式）放进 Reduce 的 system prompt，再把原始问题交给 LLM，得到最终回答。如果开了 `only_need_context`，就直接返回上面的要点上下文；如果没有有效要点，就返回 `fail_response`。[Reduce 源码](https://github.com/gusye1234/nano-graphrag/blob/acb35c065614eb5a2f5f1be9a56b235f5a2e0a7a/nano_graphrag/_op.py#L1062-L1104)

local 从命中实体出发，能把附近的关系和原文 chunk 一起交给 LLM。global 可以把分散在不同社区的信息汇总起来，但模型能看到的细节取决于社区报告有没有写进去。报告分成几组，就要做几次 Map；汇成最终回答还要再做一次 Reduce。

