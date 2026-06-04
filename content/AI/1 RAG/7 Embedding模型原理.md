---
title: "Embedding模型原理"
categories: ["AI"]
author: "BluHuang"
date: 2026-06-02
lastmod: 2026-06-02
---

## 一、核心概念

### 1.1 什么是 Embedding？
**定义**：将离散的文本符号（词、句子、文档）映射为连续的低维实数向量（通常 128~4096 维）。向量空间中的位置和方向编码了语义信息。

**核心性质**：
- 语义相似的文本 → 向量夹角小（余弦相似度高）
- 可进行向量运算：`国王 - 男人 + 女人 ≈ 王后`

### 1.2 静态 Embedding vs 动态 Embedding（重点）

| 特性 | 静态 Embedding | 动态 Embedding |
|------|----------------|----------------|
| **代表模型** | Word2Vec, GloVe, FastText | BERT, SBERT, BGE, M3E, OpenAI embeddings |
| **向量生成方式** | 每个词对应一个固定向量，查表得到 | 根据上下文实时计算，同一个词在不同句子中向量不同 |
| **上下文感知** | ❌ 无 | ✅ 有（通过自注意力机制） |
| **一词多义处理** | 无法区分（“bank” 河流/银行共享同一向量） | 能区分（根据周围词生成不同向量） |
| **输入粒度** | 通常为词或子词 | 句子、段落、文档（可变长） |
| **训练方式** | 共现统计（Skip-gram, CBOW） | 预训练 + 微调（MLM, 对比学习） |

#### 动态 Embedding 的详细原理

动态 Embedding 基于 **Transformer 编码器**（如 BERT）。核心机制：

1. **输入表示**：将文本切分为 token（如 WordPiece），每个 token 初始化为一个随机向量 + 位置编码。
2. **自注意力层**：每个 token 的向量会“看到”句子中所有其他 token，通过计算注意力权重，聚合全局信息。公式：
   $$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
   其中 Q、K、V 均由输入向量线性变换得到。
3. **多层堆叠**：经过多层自注意力 + 前馈网络，最终每个 token 的向量融合了上下文信息。例如：
   - “bank” 出现在 “river bank” → 向量接近“河岸”
   - “bank” 出现在 “savings bank” → 向量接近“银行”
4. **池化策略**：将多个 token 向量聚合成一个固定长度的句子/文档向量。常用方法：
   - **CLS 池化**：取 `[CLS]` 位置的输出向量（BERT 风格）
   - **均值池化**：对所有 token 向量取平均（SBERT、BGE 常用）
   - **最大池化**：取每个维度的最大值

**为什么动态 Embedding 更适合 RAG？**
- 能理解查询中的歧义（如“苹果”指水果还是公司）
- 能编码长文本的全局语义，而非孤立词
- 可通过微调适配特定领域

---

## 二、训练范式：对比学习

当前顶尖动态 Embedding 模型均采用**对比学习**。

**训练数据格式**：三元组 `(anchor, positive, negative)`
- anchor：查询文本
- positive：与 anchor 语义相似的文本（相关文档）
- negative：与 anchor 语义不相似的文本（不相关文档）

**损失函数：InfoNCE**
$$ L = -\log \frac{\exp(\text{sim}(q, p) / \tau)}{\exp(\text{sim}(q, p) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(q, n_i) / \tau)} $$
- `sim`：余弦相似度
- `τ`：温度系数（典型值 0.01~0.1），控制模型对负样本的严厉程度
  - τ 越小 → 对负样本区分越严厉，适合困难负采样
  - τ 越大 → 分布越平滑，训练更稳定

**数据来源**：
- 自然语言推理（NLI）：蕴含对为正例，矛盾为负例
- 检索数据集（MS MARCO, Natural Questions）：用户点击为正，随机采样为负
- 自监督：标题-正文、相邻段落、回译等

---

## 三、评价维度与 MTEB 基准

### 3.1 关键评价维度

| 维度 | 含义 | 对系统影响 |
|------|------|------------|
| **向量维度** | 输出向量长度（如 384, 768, 1024） | 高维表达能力强，但存储和计算成本高 |
| **最大输入长度** | 模型能处理的 token 数上限 | 决定能否编码长文档（技术手册需 2048+） |
| **推理速度** | 每秒 token 数或单条延迟 | 影响索引耗时和在线查询延迟 |
| **精度** | 在基准上的 Hit Rate / MRR / NDCG | 直接决定 RAG 召回上限 |
| **语言支持** | 单语 vs 多语 | 混合语言场景需多语模型 |

### 3.2 MTEB 基准
- **全称**：Massive Text Embedding Benchmark（HuggingFace 维护）
- **覆盖**：58 个数据集，8 类任务（检索、重排序、分类、聚类等）
- **中文子集**：C-MTEB, MLDR, T2Retrieval
- **重要指标**：
  - **NDCG@10**：考虑排序位置的归一化折损累计增益（检索主要指标）
  - **MRR**：平均倒数排名（只关心第一个正确答案）
  - **Hit Rate@K**：前 K 个结果包含正确答案的比例

⚠️ **注意**：MTEB 分数仅作参考。你的领域数据可能存在巨大偏差，务必用小规模领域数据集实测。

---

## 四、主流模型对比

### 代表模型一览

| 模型 | 类型 | 最大长度 | 语言 | 特点与推荐场景 |
|------|------|----------|------|----------------|
| **BGE-M3** | 开源 | 8192 | 多语言 | **首选推荐**：长文档、中英混合、需混合检索（稠密+稀疏） |
| **M3E-base** | 开源 | 512 | 中英 | **无 GPU 备选**：CPU 可跑，速度快，中文优化 |
| **text-embedding-3-small** | 商业 API | 8192 | 多语言 | **快速验证**：不想管基础设施，精度中等，成本低 |
| **paraphrase-multilingual-MiniLM**（SBERT） | 开源 | 512 | 多语言 | **速度优先**：极轻量，实时性要求极高场景 |

> 其他模型（BGE-large-zh、text-embedding-3-large、all-mpnet 等）可作为备选，但上述四款已覆盖 90% 需求。

### 选型建议（决策逻辑）
1. **数据隐私敏感？** → 用开源模型（BGE-M3 或 M3E）
2. **文档很长（>512 tokens）？** → BGE-M3（唯一支持 8192 且中文优秀）
3. **有 GPU？** → BGE-M3 或 M3E-base（GPU 加速）
4. **无 GPU，纯 CPU？** → M3E-base（速度尚可）
5. **不想部署，快速验证？** → text-embedding-3-small（商业 API）

---

## 五、常见误区与陷阱

| 误区 | 真相 |
|------|------|
| 维度越高越好 | 高维收益递减，存储和计算成本高 |
| MTEB 高分 = 我的数据高分 | 领域偏移可导致精度下降 20%+ |
| 模型能完美处理任意长度 | 超过最大长度会被截断，丢失信息 |
| 纯中文模型比多语言好 | 技术文档含英文术语，多语言模型反而更好 |
| Embedding 可替代重排序 | 精排必须用 Cross-Encoder，Embedding 只能做召回 |
| 微调总是提升性能 | 数据不足或质量差时，微调会损害通用能力 |

---

## 六、参考资源

1. **BGE M3 论文**："BGE M3-Embedding" (2024)  
   https://arxiv.org/abs/2402.03216
2. **MTEB 基准论文**："MTEB: Massive Text Embedding Benchmark" (2022)  
   https://arxiv.org/abs/2210.07316
3. **对比学习综述**："A Survey on Contrastive Self-Supervised Learning" (2021)  
   https://arxiv.org/abs/2011.00362
4. **Sentence-Transformers 官方文档**：https://www.sbert.net/
5. **FlagEmbedding (BGE) 代码库**：https://github.com/FlagOpen/FlagEmbedding