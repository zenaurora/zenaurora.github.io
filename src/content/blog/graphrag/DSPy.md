---
title: "DSPy 核心概念与实战流程"
description: "用编程替代手写 Prompt：Signature、Module、Metric、Optimizer 四大基石，七步开发流程，以及一个 RAG 自动优化示例。"
date: 2026-11-21
order: 2
authors:
  - maokaihe
tags:
  - DSPy
  - LLM
  - Prompt Engineering
  - RAG
---

# DSPy 核心概念与实战流程全解

> **核心哲学**：**Programming—not prompting—LMs**（用代码与模块化编程替代手工调试 Prompt）。  
> 将大模型应用开发抽象为类似 **PyTorch** 的深度学习范式：你只负责定义“数据流结构”和“评价指标”，框架负责自动“编译与优化”出最佳的 Prompt 和示例。

---

## 一、 核心概念解析

要真正理解 DSPy，需要掌握其类比于深度学习的四大基石：

```
                ┌──────────────────────────────────────────────┐
                │              1. Signature (签名)             │
                │        定义“任务接口” (输入字段 -> 输出字段)   │
                └──────────────────────┬───────────────────────┘
                                       │ 嵌入到
                                       ▼
                ┌──────────────────────────────────────────────┐
                │               2. Module (模块)               │
                │      定义“推理流程” (Predict, CoT, ReAct等)   │
                └──────────────────────┬───────────────────────┘
                                       │ 输入到
                                       ▼
┌───────────────────────┐      ┌───────────────────────────────┐      ┌───────────────────────┐
│     训练/验证数据      │ ───► │      3. Optimizer (优化器)     │ ◄─── │    4. Metric (指标)   │
│ (Few-shot 候选样本池)  │      │   自动搜索/合成最佳 Prompt与示例 │      │   量化评估输出质量     │
└───────────────────────┘      └──────────────┬────────────────┘      └───────────────────────┘
                                              │ 编译输出 (Compile)
                                              ▼
                               ┌───────────────────────────────┐
                               │       Compiled Program        │
                               │  拥有最优 Prompt 和示例的预测程序 │
                               └───────────────────────────────┘
```

### 1. Signature（签名）：任务接口定义
- **传统方式**：写一大段 Prompt（“你是一个严谨的助手，请阅读以下文章并回答问题...”）。
- **DSPy 方式**：声明**输入变量**和**输出变量**，完全解耦“任务意图”与“具体措辞”。
- 既支持极简的字符串语法（`"question -> answer"`、`"context, question -> rationale, answer"`），也支持强类型的 Class 语法。

### 2. Module（模块）：推理行为抽象
将常见的 Prompt 技巧抽象为可复用的类（类似于 PyTorch 的 `nn.Module`）：
- `dspy.Predict`：最基础的单次 LLM 调用。
- `dspy.ChainOfThought`：自动在最终输出前加入思维链推理步骤（Rationale / Thought）。
- `dspy.ReAct`：具备工具调用与多轮观察能力的 Agent 循环。
- `dspy.ProgramOfThought`：让 LLM 生成代码并执行，用于精确计算或严谨逻辑。

### 3. Metric（评估指标）：优化的指挥棒
- 衡量模型输出好坏的量化函数。
- 可以是简单的规则比对（完全匹配 Exact Match、包含关键词、F1 Score），也可以是进阶的 **LLM-as-a-judge**（让大模型对回答的相关性、幻觉、忠实度进行打分）。

### 4. Optimizer / Teleprompter（优化器 / 自动编译器）：DSPy 的灵魂
优化器优化的不是模型的权重矩阵（浮点数），而是**提示词文本、指令指导语和 Few-shot 样本集**：
- **BootstrapFewShot**：自动筛选并验证有效的 Few-shot 输入输出样本（自举生成高质量示范）。
- **COPRO (Chain of Thought / Prompt Optimizer)**：利用大模型自动重写并搜索最高分的 Prompt 指令。
- **MIPROv2 (Multi-prompt Instruction Proposal Optimizer)**：同时联合搜索最优的“系统指令（Instructions）”与“Few-shot 示范样本集”。

---

## 二、 DSPy 的标准研发流程（7 步走）

开发一个 DSPy 程序遵循非常清晰的标准流水线：

```
[步骤 1: 初始化配置 LM] ──► [步骤 2: 准备微型数据集] ──► [步骤 3: 声明 Signature]
                                                                  │
                                                                  ▼
[步骤 6: 优化器编译 Pipeline] ◄── [步骤 5: 定义 Metric 指标] ◄── [步骤 4: 组装 Module Pipeline]
           │
           ▼
[步骤 7: 评估、持久化与部署]
```

---

## 三、 实战代码：构建一个带思维链的 RAG 自动优化系统

下面给出一套结构完整、涵盖 7 步生命周期的标准代码示例。

### 步骤 1 & 2：配置模型与准备数据

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# 1. 配置默认大语言模型 (以 OpenAI 为例，也可配置本地 Ollama、vLLM、Anthropic 等)
turbo = dspy.LM('openai/gpt-4o-mini', api_key="sk-xxxx")
dspy.configure(lm=turbo)

# 2. 准备少量的带标注样本 (通常 10~50 条就足够启动优化)
raw_trainset = [
    {
        "context": "GraphRAG 采用层次化聚类算法（如 Leiden）对知识图谱进行社区发现，并生成社区摘要报告。",
        "question": "GraphRAG 中社区是如何划分的？",
        "ground_truth": "利用层次化聚类算法（如 Leiden）在知识图谱上做社区发现。"
    },
    {
        "context": "DSPy 的核心理念是用编程代替手写 Prompt，核心模块包含 Signature、Module 和 Optimizer。",
        "question": "DSPy 用什么替代了传统手写 Prompt？",
        "ground_truth": "用模块化编程、参数化优化器与代码定义来替代手写 Prompt。"
    },
    {
        "context": "Transformer 架构的核心自注意力机制计算公式为 Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V。",
        "question": "注意力机制中 Q 和 K 点积后需要除以什么？",
        "ground_truth": "除以键向量维度的平方根 sqrt(d_k)。"
    }
]

# 转换为 DSPy 的 Example 对象，并使用 .with_inputs() 指定哪些字段是输入
trainset = [
    dspy.Example(
        context=item["context"],
        question=item["question"],
        ground_truth=item["ground_truth"]
    ).with_inputs("context", "question")
    for item in raw_trainset
]
```

### 步骤 3 & 4：定义 Signature 并组装 Module

```python
# 3. 声明任务签名：定义输入与期望输出
class GenerateAnswerWithCitation(dspy.Signature):
    """根据给定的检索上下文准确回答用户问题，回答必须有理有据且避免幻觉。"""
    
    context = dspy.InputField(desc="检索到的背景知识片段")
    question = dspy.InputField(desc="用户提出的具体问题")
    answer = dspy.OutputField(desc="根据上下文提炼的事实回答，不超过两句话")


# 4. 组装 Pipeline 模块
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 ChainOfThought 模块，在生成 answer 前自动引导大模型先输出 rationale 推理逻辑
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerWithCitation)

    def forward(self, context, question):
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(
            rationale=prediction.rationale,
            answer=prediction.answer
        )
```

### 步骤 5：定义量化评估指标（Metric）

```python
# 5. 定义评估指标
def rag_eval_metric(example, pred, trace=None):
    """
    example: 包含真实标签 (ground_truth) 的真实样本
    pred: Module 输出的预测对象
    trace: 追踪链路信息 (优化器在回溯自举时使用)
    """
    gold = example.ground_truth.strip().lower()
    answer = pred.answer.strip().lower()

    # 简单规则：真实关键词是否被覆盖 (也可用包含度、语义相似度、或 LLM Judge)
    hit = gold in answer or answer in gold
    
    # 在优化阶段，如果 trace 存在，DSPy 可以感知该步骤是否执行成功
    return hit
```

### 步骤 6：利用优化器编译（Compile）最佳 Prompt

这是最关键的一步：原本你可能需要反复调整 “请仔细思考...”、“严格遵循以下示例...” 等 Prompt 词句，现在全部交给优化器。

```python
# 6. 选择优化器并编译
# BootstrapFewShot 会在 trainset 上试跑未编译的程序，
# 只有在 metric 评分通过的执行轨迹（Trace），才会被提炼成高价值的 Few-shot Demonstrations
optimizer = BootstrapFewShot(
    metric=rag_eval_metric,
    max_bootstrapped_demos=2,  # 每个模块最多塞入的优秀自举示例数
    max_labeled_demos=1        # 原始标注示例数
)

print("正在自动编译优化 Pipeline...")
compiled_rag = optimizer.compile(student=RAGModule(), trainset=trainset)
print("编译完成！")
```

### 步骤 7：执行推理、查看底层生成的 Prompt 与持久化

```python
# 7. 测试与推理
test_context = "Leiden 算法是 Louvain 算法的升级版，解决了社区中可能出现分离孤岛的缺陷。"
test_query = "Leiden 算法相比 Louvain 解决了什么问题？"

pred = compiled_rag(context=test_context, question=test_query)

print("\n--- 推理结果 ---")
print("【思考逻辑 (Rationale)】:", pred.rationale)
print("【最终回答 (Answer)】:", pred.answer)

# 查看 DSPy 内部实际替我们合成并发给大模型的完整 Prompt
print("\n--- 实际发送给大模型的完整 Prompt 历史 ---")
turbo.inspect_history(n=1)

# 保存编译好的模块（保存的是最优指令和示例配置，并非模型权重）
compiled_rag.save("compiled_rag_config.json")

# 加载恢复
loaded_rag = RAGModule()
loaded_rag.load("compiled_rag_config.json")
```

---

## 四、 底层揭秘：DSPy 到底背地里生成了什么？

当你在代码里写了：
```python
self.generate_answer = dspy.ChainOfThought(GenerateAnswerWithCitation)
```
并且经过 `BootstrapFewShot` 编译后，调用 `inspect_history()`，你会发现 DSPy 自动把下面的结构组装成一段严密的 Prompt 发送给 LLM：

```
[System Prompt 结构]
Given the fields `context`, `question`, produce the fields `rationale`, `answer`.

[任务说明]
根据给定的检索上下文准确回答用户问题，回答必须有理有据且避免幻觉。

[自动挖掘的高分 Few-shot 示范 1]
---
Context: GraphRAG 采用层次化聚类算法（如 Leiden）对知识图谱进行社区发现...
Question: GraphRAG 中社区是如何划分的？
Reasoning: Let's think step by step in order to ... 上下文提到使用了 Leiden 层次化聚类...
Answer: 利用层次化聚类算法（如 Leiden）在知识图谱上做社区发现。

[本次实际输入]
---
Context: Leiden 算法是 Louvain 算法的升级版，解决了社区中可能出现分离孤岛的缺陷。
Question: Leiden 算法相比 Louvain 解决了什么问题？
Reasoning: Let's think step by step in order to
```

> **结论**：  
> 开发者无需再为每一个提示词句式、Few-shot 模板标点纠结。**你编写结构，优化器在数据和评测标准的驱动下自动寻找全局最优提示策略。**
