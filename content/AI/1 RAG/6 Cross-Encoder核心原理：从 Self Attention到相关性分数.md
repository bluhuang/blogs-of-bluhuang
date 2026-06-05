---
title: "Cross-Encoder核心原理：从 Self Attention到相关性分数"
categories: ["AI"]
author: "BluHuang"
date: 2026-06-04T19:01:05+0800
lastmod: 2026-06-02
---

## 1. Cross‑Encoder 的核心定位

### 1.1 一句话定义

Cross‑Encoder 是一个基于 Transformer 的神经网络，它将 Query 和 Document **拼接成一个序列**，通过多头自注意力（Multi‑Head Self‑Attention）让两个文本的 **每一个 token 都与对方的所有 token 进行交互**，最终输出一个 [0, 1] 范围内的相关性分数。

### 1.2 RAG 中的定位

- **第一阶段（检索）** ：Bi‑Encoder 或 BM25 快速从全量知识库中召回 **Top‑50 / Top‑100** 候选文档（高召回）
- **第二阶段（重排序）** ：Cross‑Encoder 对这些候选文档逐一精细打分并重新排序，输出 **Top‑3 / Top‑5** 最相关的文档（高精度）

> Cross‑Encoder 的目的不是“替换”检索，而是“补救”检索的粗糙排序。其计算复杂度为 O(|Q|×|D|)，无法在百万级文档上实时计算；必须用 Bi‑Encoder 先做粗筛。

## 2. 与 Bi‑Encoder 的根本区别

| 维度 | Bi‑Encoder | Cross‑Encoder |
|------|-----------|---------------|
| **输入方式** | Query 和 Document **分开编码** | Query 和 Document **拼接后一起编码** |
| **特征交互** | 仅在最后一步通过点积/余弦相似度 **浅层交互** | 每一层 Attention 都进行 **token‑to‑token 深度交互** |
| **向量生成** | 生成 query 向量 + doc 向量，可预计算 doc 向量 | 不生成可复用的向量表示，只能打分 |
| **速度** | 极快（毫秒级，可扩展至百万级文档） | 慢（每个 (q, d) 对都需要一次完整的前向传播） |
| **适用阶段** | 第一阶段：大规模召回 | 第二阶段：候选集精细重排序 |
| **典型指标** | 高召回率（Recall） | 高精确率（Precision / nDCG） |

> Bi‑Encoder 与 Cross‑Encoder 选择的关键是**速度与精度的权衡**。能用 Bi‑Encoder 解决的场景不要上 Cross‑Encoder，能用规则解决的不要上模型。

## 3. Self‑Attention：为什么 Cross‑Encoder 更准？

### 3.1 Bi‑Encoder 的致命弱点

Bi‑Encoder 的流程是：`query → Transformer → 向量 q`，`doc → Transformer → 向量 d`，然后计算 `cos(q, d)`。**Query 和 Document 在编码时完全不知道对方的存在**，所有交互都压缩进最后一个点积运算中。

- 语序差异难以捕捉：例如“猫追狗”和“狗追猫”的向量可能非常接近，但语义完全相反
- 否定词容易被忽略：`"not connect"` 和 `"connect"` 的向量距离可能很近，但相关性完全相反
- 长上下文依赖失效：超过向量维度表达上限的细粒度关系无法建模

### 3.2 Cross‑Encoder 的深度交互机制

Cross‑Encoder 的流程是：`[CLS] query tokens [SEP] doc tokens [SEP] → Transformer → [CLS] output → Linear → relevance score`

**关键机制**：Transformer 的多头自注意力让 Query 中的每个 token **直接“看到”** Document 的所有 token，反之亦然。这意味着：

- `"not"` 可以直接“找到” `"connect"`，建立否定关系
- `"北京"` 可以直接“匹配” `"北京"`，同时无视 `"上海"`
- `"2500 元一晚"` 可以直接“对比” `"便宜"`，判断矛盾

### 3.3 输出层：从交互到分数

经过 N 层 Transformer 后，`[CLS]` token 的隐藏状态聚合了整个 (Query, Document) 对的全量交互信息。通过一个线性分类头（+ Sigmoid）映射为 [0,1] 区间的相关性分数。对于某些 T5 类模型，也可以让模型生成 `"true"` 或 `"false"` token，用 `"true"` 的对数概率作为相关性分数。

**数学形式**：

$$s(q, d) = \sigma( \mathbf{w}^\top \mathbf{h}_{[CLS]}(\text{[CLS]} q \text{[SEP]} d \text{[SEP]}) + b )$$

## 4. 训练策略

### 4.1 点对式（Pointwise）

把重排序当作 **二分类**（相关 / 不相关）或 **回归**（0-1 分数）任务。每条训练数据是一个 (query, document, label) 三元组。损失函数通常为二元交叉熵（BCE）或均方误差（MSE）。

### 4.2 列表式（Listwise）

一次性输入同一个 query 对应的多个文档，训练模型学习这些文档的**相对顺序**。常用 RankNet/ListNet 损失函数，通过两两比较学习偏序关系。Listwise 更贴近实际推理场景，精度通常更高。

### 4.3 蒸馏

用大而精的 Cross‑Encoder 作为教师模型，将知识蒸馏到轻量模型，在精度和速度之间取得平衡。

## 5. 主流模型与选型

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| **BGE‑Reranker‑v2‑m3** | ~560M | 中文社区主流，Apache 2.0，量化后 <200MB | 中文 + 自托管，平衡精度与资源 |
| **商业 Reranker API** | 闭源 API | 精度领先，多语言，按量付费 | 最快落地，不折腾运维 |
| **ms‑marco‑MiniLM‑L‑6‑v2** | 22M | 超轻量，CPU 可跑，MIT 协议 | 英文通用，边缘部署，学习入门 |
| **4B 参数多语言 Reranker** | 4B | 100+ 语言，Apache 2.0，32K 上下文 | 多语言 + 长文档 + 开源 |
| **ColBERT v2** | 110M | Token‑级后期交互，高吞吐 | 大规模英文知识库精排 |

> **选型建议**：从**精度、延迟、成本、隐私**四个维度权衡。中文技术文档首选 BGE，追求精度但不想自托管选商业 API，学习入门用 MiniLM。

## 6. 工程落地架构

### 6.1 典型两阶段流水线

```
用户 Query
    ↓
[第一阶段] Bi‑Encoder 或 BM25 快速检索
    ↓ 召回 Top‑100（可缓存、可并行）
[第二阶段] Cross‑Encoder 对 100 个 (q, d) 对逐一打分
    ↓ 重排序后取 Top‑5
[第三阶段] 将 Top‑5 作为上下文喂给 LLM 生成答案
```

### 6.2 常见生产配置

- 检索阶段召回 **50–200** 个候选
- Cross‑Encoder 精排后取 **3–10** 个送入 LLM
- 可配合 GPU 批处理降低延迟（同时处理 32–64 个 (q, d) 对）

## 7. 常见误区澄清

1. **Cross‑Encoder 可以替代检索** ❌  
   Cross‑Encoder 无法在百万级文档上实时计算，必须先用 Bi‑Encoder 或 BM25 缩小候选范围。

2. **Rerank 一定能提升效果** ❌  
   如果第一阶段召回质量极差（相关文档根本不在 Top‑100 内），Rerank 无法起死回生。Rerank 的前提是检索召回率足够高。

3. **Cross‑Encoder 只能输出 0/1** ❌  
   实际可输出 0–1 连续分数，也可配置为多级分类（不相关 / 部分相关 / 高度相关）。

4. **Rerank 和微调是替代关系** ❌  
   Rerank 优化排序，微调优化生成，两者正交且可同时使用。Rerank 属于检索侧优化，不改变生成模型。

## 8. 常见问题解答

### Q1：Cross‑Encoder 比 Bi‑Encoder 准在哪里？

Bi‑Encoder 将 Query 和 Document 分开编码，在最后算一次点积，交互非常浅；Cross‑Encoder 将两者拼接，Transformer 的每一层 Self‑Attention 都让 Query 的每个 token 与 Document 的每个 token 交互，能捕捉到否定词、语序、矛盾信息等深层语义关系。实验表明，Cross‑Encoder 重排序能使 nDCG@10 比纯向量检索提升 15–30%，显著减少 LLM 幻觉。

### Q2：Cross‑Encoder 的训练数据怎么构造？

训练数据是 (query, document, label) 三元组。label 可来自点击日志（相关/不相关）、公开数据集（MS MARCO、Natural Questions）的人工标注，或由 LLM 自动标注并经人工校验。典型损失函数是二元交叉熵（BCE）。

### Q3：Cross‑Encoder 太慢，怎么解决？

①限制候选集大小，只对检索 Top‑50/100 重排序；②批量推理，一次前向传播处理多个 (q, d) 对；③模型量化（INT8）或蒸馏到轻量模型；④用 GPU 加速（每对约 20–80ms）。轻量 MiniLM 模型在 CPU 上可跑至百毫秒级。

### Q4：Rerank 怎么评估效果？

用 nDCG@K、MRR、context_precision（RAGAS）等指标。同一测试集上对比“不加 Rerank”和“加 Rerank”，重点看排序质量提升（nDCG@10），而不只是召回率。

## 9. 参考链接

- 《高级 RAG 检索实战：用 Cross‑Encoders（交叉编码器）彻底解决检索不准的顽疾》
- 《RAG检索模型选型：Bi‑Encoder、Cross‑Encoder、SPLADE与ColBERT的技术对比》
- 《3.2 重排序 (Reranker) 原创》
- 《Cross‑Encoders: Neural Re‑Rankers》
- 《The Power of Cross‑Encoders in Re‑Ranking for NLP and RAG Systems》
- 论文：*Cross‑encoders allow queries and documents to exchange information via symmetric attention*
- 论文：*Trans‑Encoder：通过自蒸馏和相互蒸馏的无监督句对建模*（ICLR 2022）