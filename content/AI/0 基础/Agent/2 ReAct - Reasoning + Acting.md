---
title: "ReAct - Reasoning + Acting"
image: "/images/AI/0%20%E5%9F%BA%E7%A1%80/Agent/Pasted%20image%2020260413144128.png"
categories: ["AI"]
author: "BluHuang"
date: 2026-06-09T20:07:40+0800
lastmod: 2026-04-13
---

# 1 什么是 ReAct

ReAct（Reasoning + Acting）是由谷歌研究院和普林斯顿大学于 2022 年提出的智能体框架，核心创新是让 LLM 交替生成**推理轨迹**与**任务操作**，模拟人类解决问题时的"思考-行动-观察"循环。

### 🧠 前言：从 CoT 到 ReAct
	COT：Chain-of-Thought, 思维链
在只有基础对话能力的阶段，大模型更多像一个“一次性回答机”：
`User: 问题 → LLM: 一次性生成答案`
即便加上了 Memory、RAG，智能体也只是多了“能记”和“会查”：
- Memory：记住过去发生了什么（多轮对话、历史任务状态）
- RAG：在回答前去查一查知识库或互联网
但这仍然是“问一答一”的模式，缺少真正的多步决策与行动能力。
ReAct（Reasoning + Acting） 正是为了解决这个问题提出的：
在推理过程中，显式地交替输出“思考内容（Thought）”和“行动指令（Action）”，再利用环境反馈（Observation）更新后续推理。
> 一句话概括：**ReAct 让 LLM 一边自言自语地推理，一边调用工具，是一种更加高级的 prompting 技术**

# 2 为什么需要 ReAct
在 ReAct 出现之前，利用 LLM 解决复杂任务主要有两种独立方法，各有局限：
- **思维链（CoT）**：模型生成逐步的逻辑推理，但完全依赖内部知识，常导致事实错误和"幻觉"。
- **仅行动规划（Act-Only）**：模型生成行动与外部环境互动，但缺乏高层规划，处理错误时表现不佳。
ReAct 将两者的优点结合：模型能够推理，并将其推理"植根于"外部世界的事实，根据反馈调整行动。

# 3 ReAct 的工作机制："思考—行动—观察"循环
## 3.1 工作机制

![](/images/AI/0%20%E5%9F%BA%E7%A1%80/Agent/Pasted%20image%2020260413144128.png)
模型在一个迭代循环中生成由**思考（Thought）**、**行动（Action）** 和 **观察（Observation）** 步骤组成的轨迹。

**典型轨迹示例**：
```
Question: 以交流电闻名的发明家出生于哪个首都城市？
Thought 1: 我需要查明交流电的发明者，然后找到他的出生地，并核实该城市是否为首都。
Action 1: Search[交流电发明者]
Observation 1: 交流电（AC）由尼古拉·特斯拉开发和推广。
Thought 2: 现在我需要找出尼古拉·特斯拉的出生地。
Action 2: Search[尼古拉·特斯拉出生地]
Observation 2: 尼古拉·特斯拉出生于奥地利帝国的斯米连村（现属克罗地亚）。
Thought 3: 斯米连不是首都。他的出生地不是首都。
Final Answer: 尼古拉·特斯拉出生于斯米连（克罗地亚），该地不是首都。
```

## 3.2 三个核心元素的区别

| 元素 | 是否改变环境 | 主要作用 | 示例 |
|---|---|---|---|
| **Thought（思考）** | ❌ 不改变环境 | 作为"内心独白"，规划下一步、整理信息、解释为什么调用某个工具 | "我需要先查一下2024年奥运会的举办城市。" |
| **Action（行动）** | ✅ 通过调用工具间接改变环境 | 发出结构化"命令"，触发具体操作 | search["query"]、lookup["entity"] |
| **Observation（观察）** | ✅ 环境产生的结果 | 记录环境或工具对 Action 的反馈，为下一步 Thought 提供依据 | 搜索结果文本、API 返回的 JSON |
| **Final Answer** | ❌ 自身不再行动 | 标志推理/行动序列结束，给出最终对用户的回答 | "2024年奥运会将在巴黎举办。" |

## 3.3 ReAct 的优势
- **信息获取**：与外部环境交互，获取实时信息
- **动态调整**：根据环境反馈更新操作计划，处理异常情况
- **可解释性**：生成的推理轨迹让人类能理解决策过程
- **减少幻觉**：基于外部事实而非内部知识
ReAct 让 LLM 一边"自言自语"地推理，一边调用工具，是一种更高级的 prompting 技术。

# 4 ReAct 实现
手撕简化版 ReAct 循环（伪代码，逻辑关键）

## 4.1 工具定义
```python
import re
from typing import Dict, Callable 

# 假设这是一个 Chat LLM 接口
def call_llm(prompt: str) -> str: 
	pass

TOOLS: Dict[str, Callable[[str], str]] = {} 

def register_tool(name: str): 
	def decorator(fn): 
		TOOLS[name] = fn 
		return fn 
	return decorator 
	
@register_tool("calculator") 
def calculator(expr: str) -> str: 
	"""计算简单数学表达式""" 
	try: 
		return str(eval(expr)) 
	except Exception as e: 
		return f"计算错误: {e}"
```

## 4.2 在 Prompt 里定协议
```python
REACT_SYSTEM_PROMPT = """
你是一个可以一边思考一边使用工具的助手。
交互格式如下：

Thought: 先解释你在想什么
Action: 工具名["参数"]
Observation: 我会用工具返回的结果填在这里
...
最后当你可以回答用户问题时，请输出：
Final Answer: 给出最终回答

当前可用工具：
- calculator: 计算数学表达式，格式如 calculator["1+2*3"]
"""
```

## 4.3 循环控制
```python
def react_loop(question: str, max_steps: int = 5):
    scratchpad = ""
    for step in range(1, max_steps + 1):
        prompt = (
            REACT_SYSTEM_PROMPT
            + f"\nQuestion: {question}\n"
            + scratchpad
            + "\n请给出下一步 Thought / Action 或 Final Answer："
        )
        llm_output = call_llm(prompt)

        # 1. 先看是否已经给出 Final Answer
        if "Final Answer:" in llm_output:
            answer = llm_output.split("Final Answer:")[1].strip()
            print("✅ Final Answer:", answer)
            return answer

        # 2. 解析 Action
        action_match = re.search(r"Action:\s*(\w+)\[\"(.*)\"\]", llm_output)
        if action_match:
            tool_name, tool_input = action_match.groups()
            if tool_name not in TOOLS:
                observation = f"工具 {tool_name} 不存在。"
            else:
                observation = TOOLS[tool_name](tool_input)

            # 把这一步的 Thought / Action / Observation 追加到 scratchpad
            scratchpad += (
                f"\n{llm_output.strip()}\n"
                f"Observation: {observation}\n"
            )
        else:
            # 如果没解析出 Action，只把 Thought 拼进去，继续下一轮
            scratchpad += f"\n{llm_output.strip()}\n"

    print("⚠️ 达到最大步数仍未得到 Final Answer")
    return None
```

1. **拼 prompt**：系统提示 + 问题 + 之前的 Thought/Action/Observation 轨迹；
2. **让 LLM 输出下一步**：
    - 要么继续 Thought + Action；
    - 要么直接给出 Final Answer；
3. **解析 Action**，调用对应工具，补上 Observation；
4. **循环直到终止**。
LangChain 做的事情，本质上就是把这个“循环 + 解析 + 工具调用”**封装成一个可复用的 LCEL 运行图**，并提供不少内置工具和 Prompt 模板。