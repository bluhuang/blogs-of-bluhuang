---
title: "GraphRAG 与 LightRAG"
categories: ["RAG"]
author: "BluHuang"
date: 2026-06-09T20:07:40+0800
lastmod: 2026-06-09T20:07:40+0800
---

# GraphRAG与LightRAG

## 1. 传统 RAG 的三大天花板（痛点）

通常讨论：“RAG 系统有没有遇到检索到了但答不对的情况？什么类型的问题答不好？”或者“RAG 检索到了正确信息，但生成的回答还是拼凑感很强，你怎么理解这个问题？”

### 1.1 一个例子说清楚 RAG 撞墙在哪

假设你有一个内部知识库，里面全是项目文档、技术方案、会议纪要。有人问了一个问题：

**“我们所有项目的技术栈趋势是什么？”**

传统 RAG 把问题转成向量，去向量库里找最相似的文本块。找到的是一堆零散的片段：“项目 A 用了 Spring Boot”“项目 B 迁移到了 Go”“项目 C 在试 Rust”……然后把这些片段丢给 LLM 拼一个回答。拼出来的是一堆事实的堆砌，不是“趋势”。因为没有全局视角能把所有项目的全貌看清楚，LLM 拿到的就是碎片。这不是 RAG 的 bug，是**向量检索的本质限制**。

### 1.2 传统 RAG 的三个天花板

通常讨论：“RAG 检索到了正确信息，但为什么答不对？什么类型的问题答不好？”

**核心结论**：传统 RAG 擅长找相似的文本块，但不擅长**跨文档关联推理**、**全局归纳**和**关系理解**。以下三个问题是其本质限制，不是调参能解决的。

**① 碎片化检索——查到的是文本块，不是知识**

传统 RAG 把文档切成 chunk，每个 chunk 独立变成向量。检索时拿到的是“和问题最相似的文本块”。但很多问题的答案不是一个文本块能覆盖的，需要把多个文本块里的信息关联起来。比如“某人和某人在哪个项目上有合作”，信息可能分散在三个文档里——文档1提到某人负责项目X，文档2提到另一人参与了项目X，文档3提到项目X的具体内容。传统 RAG 最多能捞到其中一个，很难同时把三个都捞出来并关联上。

**② 全局问题瞎答——问“整体”只能拼局部**

像“核心技术主题有哪些”“整体技术路线怎么演变的”这种全局性问题，需要的是**对整个知识库的理解**，而不是几个相似的文本块。Rerank 和混合检索能提升检索精度，但它们优化的是“找更相似的文本块”，不是“把碎片拼成全貌”。把 Top-5 变成 Top-20，拿到的还是碎片，只是更多了。

**③ 跨文档关系断裂——A 和 B 的联系全丢了**

“公司 A 收购了公司 B”在文档1里，“公司 B 和公司 C 有合作”在文档2里。那么公司 A 和公司 C 之间有没有间接关系？传统 RAG 答不了——因为每个 chunk 是独立 embedding 的，chunk 之间没有“关系”这个概念。

### 1.3 这不是调参能解决的

混合检索、Rerank、Query 改写，都是在“找更好的文本块”。但有些问题需要的不是更好的文本块，是**实体之间的关系**和**全局的结构性理解**。这才是 GraphRAG 要解决的问题。

![RAG VS GraphRAG](https://mmbiz.qpic.cn/mmbiz_jpg/UVianz7ybbib7cTsu7ricMXaSpmg6NHRDCmlseHmcUG621ShvMKegIPD9SZ4lDJTqukuFGh1M6taJpv2AaUX0dH8wUT7m8O1Ax2mrP53oYiarCY/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

---

## 2. RAG 的演进：从“找文本”到“找关系”

通常讨论：“GraphRAG 和传统 RAG 本质区别是什么？RAG 这条线是怎么演进过来的？”

### 2.1 三代 RAG 的演进路线

要理解 GraphRAG 为什么出现，得先看 RAG 这条线是怎么一步步走过来的。

- **Naive RAG**：最原始的 RAG：文档切块 → 向量化 → 检索 → 塞给 LLM 生成。问题很多：检索不准、幻觉严重、没有 Rerank。
- **Advanced RAG**：混合检索补上关键词匹配的短板、Rerank 做精排提准、Query 改写对付模糊问题、Parent-Child 检索兼顾精度和上下文。这些优化确实把“找文本块”这件事做到了极致。
- **GraphRAG**：换了一条路：不再只找文本块，而是先建一个知识图谱，把实体和关系都结构化地存下来，检索时走图谱找关系。**从“找文本”变成了“找关系”。**

演进逻辑特别清晰：**Naive RAG 的问题是“找不准”→ Advanced RAG 把检索策略调到最好 → 但有些问题不是找文本块能解决的 → GraphRAG 换了检索范式。**

![RAG三代演进：从找文本到找关系](https://mmbiz.qpic.cn/mmbiz_png/UVianz7ybbib7KHLzBXXKnqrQJ93CibtUEsL05M1SCotHqbibYgvuCsJWWcwYKEdCKoJLQVtG0AeAt172pVIPiaY01TI9S4VxYBwYGqRVnpLzVJ0/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

### 2.2 一句话定位 GraphRAG

**给 RAG 装上知识图谱，让检索从“找文本块”变成“找实体和关系”。**

传统 RAG 检索的是“和问题相似的文本”，GraphRAG 检索的是“和问题相关的实体、关系、社区”。前者是局部匹配，后者是结构化理解。

---

## 3. GraphRAG（某团队，2024.04）

### 3.1 什么是 GraphRAG？为什么会有 GraphRAG？

**GraphRAG** 这个词，其实有两层含义。

狭义上，它指的是**某研究团队在 2024 年 4 月发布的那篇论文和配套的开源项目**。论文名字叫 **《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》**（从局部到全局：一种面向查询聚焦摘要的图 RAG 方法），作者是某团队等人。

广义上，它指的是**所有基于知识图谱做 RAG 的方法**，业界还有很多变种（比如某些公司的 GraphRAG 方案等）。

咱们这篇文章**主要讲的是这一套 GraphRAG**，因为它目前影响力最大，开源社区也最活跃，几乎是所有 GraphRAG 方案的参照对象。后面如果没特别说明，我说的 GraphRAG 都指这一套。

好，定义说清楚了，那 GraphRAG 到底在干嘛？

**一句话概括：GraphRAG 就是用 LLM 把文档「读成一张知识图谱」，然后基于这张图谱来做检索和回答。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/ysyAxM1rgX31Xiaxtr1UGWN8fMiaRqkRZWMwpBibUwVBQRibUwzVbYwDmAOOoBuQXxmT9iaXKdq9e0eKMLbHLBibo6gkjyDKPDua5yUqGI4qiaJG8U/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

#### 3.1.1 传统 RAG 和 GraphRAG 的核心差异

你如果把传统 RAG 和 GraphRAG 放一起对比，会发现它们的**关键分歧就在于：知识是以什么形式存储的**。

- 传统 RAG：知识以**离散的文本块**形式存储。每个文本块都是独立的，块和块之间没有任何联系。
- GraphRAG：知识以**图（Graph）** 的形式存储。文档里的人、地、物、概念（这些是「实体/节点」），以及它们之间的关系（这些是「边」），都被显式地提取出来，形成一张结构化的关系网络。

我举个更具体的例子你就懂了。

假如你有一段文本：「某人在 2025 年创立了 A 公司，某人是 A 公司的 CTO，A 公司主要做自动驾驶业务。」

传统 RAG 会把这段话整体变成一个文本块，存到向量数据库里。你下次问「A 公司的 CTO 是谁？」，它靠语义相似度检索回这段话，然后让大模型去阅读理解，答出「某人」。

GraphRAG 则会从这段话里**抽取出实体和关系**，变成这样的结构化数据：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ysyAxM1rgX3eRqdj3xM6kBGEgQ9PBoBic6zFiaJbC53EFX4dPCaibe4TBvzVyEsabnMee6cnV3rdoCc2a9OD3JB3PPAV6Ljmazame7ickO8U07k/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

你看出差别了吗？**GraphRAG 里的知识不是「一段话」，而是一张「关系网」**。这张关系网里，每个节点是谁、每条边是什么关系，都是明确的、可追溯的。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/ysyAxM1rgX2SMP7QeIQsqU2rKsZfx3chlbL3ZgQ1zIXmRl38iayh33jOToLT3wuXxnrlGPkB6c1YtCDzAmUORI3dMRHRv4Cao19DLjiafic6rA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=11)

#### 3.1.2 为什么要搞成图？这么做到底好在哪？

你可能会问：费这么大劲把文档变成图，这么做到底有什么好处？

其实就是**为了解决咱们在第 1 章讲的那三个痛点**。咱们一个一个来看。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/ysyAxM1rgX16F7cNYBPTfxEmLU71tAVS4ykaBiaqKwiaeLYCWmGycLotZrsAmiatImblG0lSSI6D7FyjbKFic6dJGXp7Dw9EoZCZJxkQvjYbhII/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=12)

> **第一，多跳推理变得天然可行**。

上一节咱们提到那个「欧洲供应商 + 审计没过 + 处理 PII 数据」的查询，传统 RAG 干不了，是因为它没法做「交集」。但在图上，这件事就变成了图遍历（Graph Traversal）：从「欧洲」这个节点出发，找它连着的所有「供应商」节点，再筛选出那些连着「审计未通过」的，再筛选出那些连着「PII 数据处理」的。

**图天然就擅长做这种「顺藤摸瓜」的事**。这也是 GraphRAG 在多跳问答任务上（比如 HotpotQA 数据集）比传统 RAG 的 F1 分数高 5% 以上的根本原因。

> **第二，全局性问题能答了**。

这是 GraphRAG 最核心的创新点，也是那篇论文标题里「From Local to Global」（从局部到全局）的来源。

该团队的做法很巧妙：**先把整张知识图谱通过社区检测算法（比如 Leiden 算法）划分成若干个「社区」，再用 LLM 为每个社区生成一份摘要**。

什么叫社区？你可以把它理解成**图里那些互相抱团、关系特别密切的一群节点**。比如在《某古典小说》这本书里，某集团内部人物关系非常紧密，但社区之间的连接就比较稀疏。

一旦每个社区都有了摘要，你问「整本书主要讲了几路势力的斗争？」这种全局性问题时，GraphRAG 就可以**把各个社区摘要拿出来归纳一下**，给你一个完整的答案。传统 RAG 没有这种「全局视角」，自然就答不出来。

> **第三，实体和关系显式化，切块语义断裂的问题也缓解了**。

因为 GraphRAG 已经提前把文档里的实体和关系抽出来了，哪怕原文里「某药物属于某类」和「某情况下禁用」分散在两个地方，它们在图上也能通过「该药物」这个节点连起来。检索时不会丢掉因果关系，大模型也就不容易混淆了。

#### 3.1.3 一个更形象的类比：从「图书馆找书」到「带导游游书店」

讲到这儿，你可能已经有感觉了，但咱们再用一个生活化的类比来强化一下理解。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/ysyAxM1rgX2YY08R4oAaARibrdfdiaFLbKTZt7GQymIQSUuY3bhO5v0tGxgkN0rqeDblpxEcQz56HuyEFGAicE1jtRq2aQ3yw