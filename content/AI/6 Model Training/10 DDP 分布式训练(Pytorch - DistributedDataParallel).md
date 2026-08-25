---
title: "DDP 分布式训练(Pytorch - DistributedDataParallel)"
image: "/images/AI/6%20Model%20Training/attachments/ddp_01_data_parallel.png"
categories: ["AI"]
author: "BluHuang"
date: 2026-08-25T20:24:00+0800
lastmod: 2026-08-25T20:24:00+0800
---

> **DDP（DistributedDataParallel，分布式数据并行）**解决的是：模型本身能放进一张 GPU，但希望让多张 GPU 同时处理不同数据，从而加速训练。
> 
> 它最核心的思想只有一句话：
> 
> **每张 GPU 上有一份完整模型，各自计算不同数据的梯度；在更新参数之前，把各张 GPU 的梯度同步成相同结果。**
> 
> 当前 PyTorch 仍推荐多 GPU 数据并行优先使用 DDP，而不是旧的 `DataParallel`；如果模型本身已经无法放入单张 GPU，则问题转向 FSDP、Tensor Parallel 等模型分片方案。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 1 从单卡训练到 DDP：多张 GPU 到底并行了什么
先从最熟悉的单卡训练开始。

假设一次训练：
```text
Batch
  ↓
Model
  ↓
Forward
  ↓
Loss
  ↓
Backward
  ↓
Gradient
  ↓
Optimizer.step()
```

现在有 4 张 GPU。

一个直观想法是：
> 能不能让 4 张 GPU 同时计算？

可以，但首先要回答：
> 到底把什么拆到 4 张 GPU 上？

DDP 的选择是：
> **不拆模型，拆数据。**

这叫 **Data Parallelism（数据并行：每个计算设备保存完整模型，但处理不同的数据子集）**。

![](/images/AI/6%20Model%20Training/attachments/ddp_01_data_parallel.png)

---

## 1.1 每张 GPU 都有完整模型
假设模型为：
```text
Model:
Conv1
Conv2
Conv3
...
```

使用 4 张 GPU 后：
```text
GPU0：完整 Model
GPU1：完整 Model
GPU2：完整 Model
GPU3：完整 Model
```

每一份叫 **Model Replica（模型副本：同一个模型在不同进程/GPU 上的一份完整复制）**。

所以 DDP 并没有：
```text
GPU0：Conv1
GPU1：Conv2
GPU2：Conv3
```

那属于另一类 **Model Parallelism（模型并行：把一个模型自身拆到多个设备）**。

DDP 是：
```text
GPU0：完整 Model + Data0
GPU1：完整 Model + Data1
GPU2：完整 Model + Data2
GPU3：完整 Model + Data3
```

## 1.2 为什么要复制模型，而不是拆模型

假设一个 Batch 有 32 张图。

单卡：
```text
GPU0
↓
一次处理 32 张
```

DDP 4 卡：
```text
GPU0 → 8 张
GPU1 → 8 张
GPU2 → 8 张
GPU3 → 8 张
```

每张卡只承担一部分数据的 Forward 和 Backward。

这样计算可以并行。

但新的问题马上出现：
```text
GPU0 看 Data0
→ 得到 Gradient0

GPU1 看 Data1
→ 得到 Gradient1

GPU2 看 Data2
→ 得到 Gradient2

GPU3 看 Data3
→ 得到 Gradient3
```

由于输入不同：
$$
g_0\neq g_1\neq g_2\neq g_3
$$

如果每张卡直接：
```python
optimizer.step()
```

那么四份模型很快就会变成四个不同模型。

所以 DDP 真正困难的地方不是：
> 怎么让四张 GPU 同时 Forward。

而是：
> **怎么让四个独立计算出来的 Gradient 在更新参数前重新统一。**

这个问题会在第 4 章解决。

## 1.3 DDP 与 DataParallel 的区别

PyTorch 还有一个旧接口：
```python
torch.nn.DataParallel
```

它也是 Data Parallel，但采用：
```text
一个 Python Process
↓
多个 Thread
↓
多张 GPU
```

而 DDP 通常采用：
```text
多个 Python Process
↓
一个 Process 对应一张 GPU
```

PyTorch 官方目前明确建议多 GPU 训练优先使用 DDP；DDP 使用多进程，并支持多机扩展，而 `DataParallel` 是单进程、多线程方式。([PyTorch Documentation](https://docs.pytorch.org/docs/main/notes/cuda.html?utm_source=chatgpt.com "CUDA semantics — PyTorch main documentation"))


## 1.4 DDP 与 FSDP 的区别

还有一个很容易混淆的：
**FSDP（Fully Sharded Data Parallel，完全分片数据并行：把参数、梯度和优化器状态等模型状态拆到多个 Rank 上保存）**。

核心区别：

|场景|方案|
|---|---|
|模型能放入单卡，希望多卡加速|DDP|
|模型本身单卡放不下|FSDP / Model Parallel|

DDP 中每张 GPU 都有完整模型，因此：

```text
单卡放不下 Model
```

时，DDP 本身不能解决这个内存问题。PyTorch 当前的分布式选择指南也是这一划分。([PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/dist_overview.html?utm_source=chatgpt.com "PyTorch Distributed Overview — PyTorch Tutorials 2.13.0+cu130 documentation"))

### 本章总结

DDP 的“Distributed”并不是把一个模型拆开，而是**把数据分给多个完整 Model Replica 并行处理**。真正需要同步的是各个 Replica 根据不同数据计算出的 Gradient。

# 2 Process、Rank、World Size：DDP 的运行世界是什么

现在知道：
```text
GPU0 → Model0
GPU1 → Model1
GPU2 → Model2
GPU3 → Model3
```

接下来要理解一个非常重要的事实：
> DDP 通常不是一个 Python 程序控制 4 张 GPU，而是启动 4 个 Python 进程。

![](/images/AI/6%20Model%20Training/attachments/ddp_02_process_rank.png)

---

## 2.1 Process 是什么

**Process（进程）**

假设运行：
```bash
torchrun --standalone --nproc-per-node=4 train.py
```

不是：
```text
运行一个 train.py
```

而是类似：
```text
Process 0：python train.py
Process 1：python train.py
Process 2：python train.py
Process 3：python train.py
```

每个进程都会：
```text
创建 Dataset
创建 Model
创建 Optimizer
执行 Training Loop
```

只是每个进程负责不同 GPU 和不同数据。

# 2.2 Rank：每个进程的身份证

多个进程必须能够互相区分。

于是每个 Process 有一个：

**Rank（进程在分布式通信组中的全局编号）**。

例如：
```text
Process A → rank 0
Process B → rank 1
Process C → rank 2
Process D → rank 3
```

因此代码经常看到：
```python
rank = dist.get_rank()
```

例如：
```python
if rank == 0:
    save_checkpoint()
```

意思不是：
> GPU0 保存。

更准确是：
> **全局编号为 0 的 Process 保存。**

# 2.3 World Size：总共有多少个参与者

**World Size（全局进程数：当前分布式训练中共同参与通信的 Rank 总数）**。

如果单机 4 卡：
```text
world_size = 4
```

如果两台机器，每台 4 卡：
```text
Node 0：4 processes
Node 1：4 processes
```
那么：
$$
world_size=8
$$

注意：
> World Size 统计的是 Process 数，不是一个抽象的“GPU 数”。

只是最常见的 DDP 配置恰好：
```text
1 Process ↔ 1 GPU
```

所以两者数值相同。

# 2.4 Local Rank：我在当前机器里排第几

多机时，仅有 Rank 不够方便。

假设：
```text
Node 0:
rank 0
rank 1
rank 2
rank 3

Node 1:
rank 4
rank 5
rank 6
rank 7
```

在 Node 1 上：
```text
rank 4 → local_rank 0
rank 5 → local_rank 1
rank 6 → local_rank 2
rank 7 → local_rank 3
```

所以：
**Local Rank（进程在当前机器内部的编号）**主要用于决定使用本机哪张 GPU。

代码：
```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

于是：
```text
local_rank 0 → cuda:0
local_rank 1 → cuda:1
...
```

`torchrun` 会设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/elastic/run?utm_source=chatgpt.com "torchrun (Elastic Launch) — PyTorch 2.13 documentation"))

一个容易踩的坑是：
> **不要把 global rank 永久等同于某个 GPU ID。**

在弹性任务重启等场景，Rank 可能重新分配；PyTorch 文档也明确提示不要依赖 Rank 的稳定性。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/elastic/run?utm_source=chatgpt.com "torchrun (Elastic Launch) — PyTorch 2.13 documentation"))

# 2.5 Process Group：谁和谁允许通信

DDP 还需要：
**Process Group（进程组：参与一组 Collective Communication 的 Process 集合）**。

最常见：
```text
rank 0
rank 1
rank 2
rank 3
   │
   └── WORLD Process Group
```

初始化：
```python
dist.init_process_group(backend="nccl")
```

之后大家才知道：
```text
我是谁
一共有几个人
通信伙伴有哪些
如何进行 AllReduce
```

这里的 **NCCL（NVIDIA Collective Communications Library，NVIDIA 为多 GPU 集体通信优化的通信后端）**是 GPU DDP 中常用的 backend；PyTorch 当前也推荐 GPU 场景使用 NCCL。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 2.6 单机和多机没有改变 DDP 的核心

单机：
```text
Process
↕
PCIe / NVLink
↕
Process
```

多机：
```text
Node A Process
↕
Network
↕
Node B Process
```

从 DDP 逻辑来看仍然只是：
```text
Rank 0
Rank 1
Rank 2
...
```
只是底层通信成本发生变化。

### 本章总结
DDP 的基本运行单位不是 GPU，而是 **Process**。`rank` 标识全局进程，`local_rank` 决定当前机器上的 GPU，`world_size` 表示总进程数，而 `process group` 定义哪些进程共同参与梯度同步。

# 3 数据怎么分：DistributedSampler、Local Batch 与 Global Batch

现在有 4 个 Rank。

如果每个 Rank 都这样：
```python
loader = DataLoader(dataset, shuffle=True)
```

会发生一个严重问题：
```text
Rank0 → 整个 Dataset
Rank1 → 整个 Dataset
Rank2 → 整个 Dataset
Rank3 → 整个 Dataset
```

甚至可能同时读到相同样本。

这样虽然用了 4 张 GPU，但没有正确实现：
```text
不同 GPU 处理不同数据
```

所以 DDP 还需要数据划分。
![](/images/AI/6%20Model%20Training/attachments/ddp_03_sampler_batch.png)

# 3.1 DDP 本身不负责切数据

这是非常重要的一点：
> **DDP 负责同步 Model Gradient，但不会自动把 Dataset 切给不同 Rank。**

通常使用：
**DistributedSampler（分布式采样器：根据 `rank` 和 `world_size` 给每个进程分配 Dataset 中的一部分 index）**。

```python
sampler = DistributedSampler(dataset)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
)
```

PyTorch 文档明确说明 DDP 不负责 input sharding，而 `DistributedSampler` 用于让不同进程读取原始 Dataset 的不同子集。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 3.2 DistributedSampler 实际上只是分 index

假设 Dataset：
```text
index:

0 1 2 3 4 5 6 7
```

两张 GPU：

```text
Rank0
0 2 4 6

Rank1
1 3 5 7
```

真实实现会先 Shuffle，再根据 Rank 分配，但概念上就是：
> 每个 Rank 只拿 Dataset 的一个子集。

所以每个进程虽然：
```python
Dataset(...)
```
对象本身可能都表示完整 Dataset，

但：
```python
DistributedSampler
```
决定当前 Rank 实际读取哪些 index。

# 3.3 Local Batch 是什么
假设：
```python
batch_size = 8
```

在 DDP 代码中，每个 Process 都创建自己的 DataLoader。
因此这个 `8` 通常表示：
**Local Batch Size（单个 Rank 每个 iteration 实际处理的样本数）**。

4 个 Rank：
```text
Rank0 → 8
Rank1 → 8
Rank2 → 8
Rank3 → 8
```
一次 iteration 实际处理：
$$
8+8+8+8=32
$$
这个 32 叫：
**Global Batch Size（一次同步参数更新所共同使用的总样本数）**。

这里要特别注意：

```python
DataLoader(batch_size=8)
```

在 DDP 中通常不是：
```text
全局 Batch = 8
```

而是：
```text
每张卡 8
```

# 3.4 为什么 `sampler.set_epoch(epoch)` 很重要

DistributedSampler 支持 Shuffle。

但是多个 Rank 必须满足：
```text
大家使用同一个 Shuffle 规则
+
各自切走不同部分
```

并且不同 epoch 应该重新 Shuffle。

所以：
```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
```

`set_epoch()` 会让 epoch 参与随机序列生成。

如果不调用，PyTorch 官方明确指出多个 epoch 可能一直使用相同的样本顺序。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/data.html?utm_source=chatgpt.com "torch.utils.data — PyTorch 2.13 documentation"))

正确逻辑：
```text
Epoch 0
↓
生成 Shuffle A
↓
按 Rank 切分

Epoch 1
↓
生成 Shuffle B
↓
按 Rank 切分
```

# 3.5 Dataset 数量不能整除 World Size 怎么办

假设：
```text
Dataset size = 10
world_size = 4
```

无法平均分：
$$
10/4=2.5
$$

`DistributedSampler` 要保证每个 Rank iteration 数量一致，否则某些 Rank 会提前结束。

默认：
```python
drop_last=False
```

会补充额外 index，让总样本数能够被 `world_size` 整除。

例如概念上：
```text
10 samples
↓
补到 12
↓
每个 Rank 3 samples
```

因此极少量样本可能重复。

如果：
```python
drop_last=True
```

则会丢弃尾部，使总数能够整除。PyTorch 当前文档对这两个行为有明确说明。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/data.html?utm_source=chatgpt.com "torch.utils.data — PyTorch 2.13 documentation"))

为什么 DDP 特别关心“每个 Rank 有多少 iteration”？

因为后面每一次 `backward()` 都要进行同步 Collective。

如果：

```text
Rank0 已经没有数据
Rank1 还在 backward
```

Rank1 发起 AllReduce 后没人和它配对，就可能等待。

第 6 章会详细讲这个问题。

### 本章总结

DDP 不切 Dataset，`DistributedSampler` 才负责让各 Rank 获取不同样本。`batch_size` 是 Local Batch，而一次同步更新实际看到的数据量是 `Local Batch × World Size`。不同 Rank 还必须保持可协调的 iteration 节奏。

# 4 DDP 最核心原理：Backward 时梯度怎么同步

现在终于到 DDP 最重要的地方。

假设两张 GPU：
```text
Rank0
↓
Batch0
↓
Loss0
↓
Gradient g0


Rank1
↓
Batch1
↓
Loss1
↓
Gradient g1
```

因为 Batch 不同：

$$
g_0\neq g_1
$$

如何让两个 Model Replica 最终保持一致？

答案是：

**AllReduce（集体通信操作：把所有 Rank 的 Tensor 汇总，并把汇总后的结果返回给所有 Rank）**。

![](/images/AI/6%20Model%20Training/attachments/ddp_04_allreduce.png)

# 4.1 先从一个参数理解

假设模型只有一个参数：

$$
w
$$

Rank0 算出：

$$
g_0=2
$$

Rank1：

$$
g_1=4
$$

经过梯度同步：

第一步，求和：

$$
g_{sum} =  g_0+g_1 =2+4 = 6
$$

第二步，除以 World Size：
$$
\bar g =  \frac{g_{sum}}{2} = 3
$$

最终：
```text
Rank0 param.grad = 3
Rank1 param.grad = 3
```

然后两边分别：
```python
optimizer.step()
```

模型继续保持一致。

DDP 的设计不是每次：
```text
Rank0 更新模型
↓
再把整个 Model 发给 Rank1
```

而是：
> **同步 Gradient，然后各个 Rank 自己执行完全相同的 Optimizer Update。**

PyTorch 官方 DDP 文档也明确说明：参数不会在每个 iteration 后重新广播；DDP 依赖各 Rank 拥有相同的平均 Gradient，并执行相同 Optimizer Step 来保持参数一致。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 4.2 为什么这等价于一个更大的 Batch

这是 DDP 最值得真正推导清楚的部分。

假设每张卡 Local Batch：

$$
B
$$

World Size：

$$
W
$$

Rank $r$ 的 Local Loss 使用常见的 Mean Reduction：

$$
L_r \frac{1}{B} \sum_{i=1}^{B} \ell_{r,i}
$$

对参数 $\theta$ 求梯度：

$$
g_r \nabla_\theta L_r
$$

代入：

$$
g_r \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta\ell_{r,i}
$$

DDP 对 $W$ 个 Rank 的 Gradient 求平均：
$$
g_{DDP} = \frac{1}{W} \sum_{r=1}^{W}g_r
$$

把 $g_r$ 展开：
$$
g_{DDP} = \frac{1}{W} \sum_{r=1}^{W} \left( \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta\ell_{r,i} \right)
$$

把常数合并：
$$
g_{DDP} = \frac{1}{WB} \sum_{r=1}^{W} \sum_{i=1}^{B} \nabla_\theta\ell_{r,i}
$$
而：
$$
WB=B_{global}
$$

所以：
$$
g_{DDP} = \frac{1}{B_{global}} \sum_{j=1}^{B_{global}} \nabla_\theta\ell_j
$$
这正是：
> **把所有 Rank 的数据拼成一个 Global Batch，然后对 Global Batch Mean Loss 求 Gradient。**

所以：
```text
4 GPUs
×
Local Batch 8
```

在 Gradient 意义上对应：
```text
Global Batch 32
```

前提是常见的：
```text
Loss reduction = mean
```

如果 Loss 本身用 `sum`，缩放关系就不同，不能机械认为完全等价；PyTorch 官方 DDP 文档也特别指出这一点。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 4.3 DDP 为什么不等整个 Backward 结束后再同步所有梯度

假设模型：
```text
Layer1
Layer2
Layer3
Layer4
```

Forward：
```text
1 → 2 → 3 → 4
```

Backward 反向：
```text
4 → 3 → 2 → 1
```

Layer4 的 Gradient 最先算出来。

如果 DDP 一定等：
```text
Layer4 grad
Layer3 grad
Layer2 grad
Layer1 grad
全部完成
```

再开始通信，那么时间会是：
```text
Backward Compute
-------------------->

然后 Communication
                     ----------->
```

GPU 算力和通信链路不能很好重叠。

DDP 的设计则是：
```text
Layer4 grad ready
↓
开始同步一部分 Gradient

同时：
GPU 继续计算 Layer3 / Layer2 的 Backward
```

于是：
```text
Backward Compute
------------------------>

Communication
       ------
             ------
                   ------
```

部分通信被隐藏在 Backward Compute 后面。

# 4.4 Autograd Hook 是怎么触发同步的

**Autograd Hook（自动求导回调：当某个参数 Gradient 计算完成时自动触发的一段函数）**。

DDP 初始化时会给参数注册 Hook。

Backward：
```text
某个 Parameter Gradient ready
            ↓
Autograd Hook 被触发
            ↓
DDP 知道：
“这个梯度已经可以同步了”
```

PyTorch DDP 内部负责梯度同步的核心组件叫：

**Reducer（DDP 中负责收集 Gradient、管理 Bucket 并启动 Reduction 的组件）**。

DDP 官方内部设计文档明确描述：Reducer 为参数注册 Autograd Hook；Backward 中梯度 Ready 后 Hook 被触发，Reducer 再推进对应 Bucket 的 AllReduce。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/ddp?utm_source=chatgpt.com "Distributed Data Parallel — PyTorch 2.13 documentation"))

# 4.5 为什么还有 Gradient Bucket

如果模型有：
```text
10,000 个 Parameter Tensor
```

每计算出一个小 Gradient 就立即做一次网络通信：

```text
10,000 次 AllReduce
```

通信启动开销会很大。

所以 DDP 会把多个 Gradient 打包：
**Gradient Bucket（梯度桶：把多个参数的梯度拼到一个较大的 Buffer 中，一次进行 Collective Communication）**。

例如：
```text
Parameter grad 1 ─┐
Parameter grad 2 ─┤
Parameter grad 3 ─┼→ Bucket 0 → AllReduce
Parameter grad 4 ─┘

Parameter grad 5 ─┐
Parameter grad 6 ─┼→ Bucket 1 → AllReduce
Parameter grad 7 ─┘
```

当：
```text
Bucket 0 中所有 Gradient Ready
```

Reducer 就可以立即启动：

```text
AllReduce Bucket0
```

而不是等待所有 Parameter Gradient。

这就是 DDP 能做到：

**Compute / Communication Overlap（反向计算与梯度通信时间重叠）**的基础。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/ddp?utm_source=chatgpt.com "Distributed Data Parallel — PyTorch 2.13 documentation"))

### 本章总结

DDP 的核心发生在 `backward()`：每个 Rank 先根据自己的 Local Batch 计算 Gradient，再通过 AllReduce 得到相同的平均 Gradient。Autograd Hook 和 Gradient Bucket 让同步能够在 Backward 尚未完全结束时提前启动，从而同时保证数学一致性和通信效率。


# 5 一次完整 DDP Iteration 到底发生了什么

现在把前四章完整串起来。
![](/images/AI/6%20Model%20Training/attachments/ddp_05_iteration_timeline.png)

---

## 5.1 启动阶段

首先：

```bash
torchrun --standalone --nproc-per-node=4 train.py
```

`torchrun` 启动：

```text
Process 0
Process 1
Process 2
Process 3
```

并提供：

```text
RANK
LOCAL_RANK
WORLD_SIZE
MASTER_ADDR
MASTER_PORT
```

然后每个 Process 执行同一个 `train.py`。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/elastic/run?utm_source=chatgpt.com "torchrun (Elastic Launch) — PyTorch 2.13 documentation"))

---

## 5.2 初始化通信

每个 Process：

```python
dist.init_process_group(backend="nccl")
```

之后各个 Rank 才能参与：

```text
AllReduce
Broadcast
Barrier
...
```

这些多个 Rank 必须共同参与的操作统称：

**Collective Communication（集体通信：一组进程共同参与的数据交换操作）**。

---

## 5.3 绑定 GPU

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

然后：

```text
Process local_rank=0 → cuda:0
Process local_rank=1 → cuda:1
...
```

---

## 5.4 创建 Model

每个 Rank 都执行：

```python
model = Model().to(device)
```

所以现在是：

```text
Rank0 → Model0
Rank1 → Model1
Rank2 → Model2
Rank3 → Model3
```

然后：

```python
model = DDP(
    model,
    device_ids=[local_rank]
)
```

DDP Wrapper 在初始化阶段会检查/同步模型状态，使训练开始时 Replica 保持一致；之后主要靠梯度同步维持一致。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 5.5 创建 DistributedSampler

每个 Rank：
```python
sampler = DistributedSampler(dataset)
```

由于每个进程拥有不同 Rank：
```text
Rank0 sampler
Rank1 sampler
Rank2 sampler
Rank3 sampler
```

自然获得不同 Dataset Index。

# 5.6 一次 Iteration

Rank0：
```text
Batch0
↓
Forward
↓
Loss0
↓
Backward
↓
Gradient0
```

Rank1 同时：
```text
Batch1
↓
Forward
↓
Loss1
↓
Backward
↓
Gradient1
```

以此类推。

Backward 内部随着 Bucket Ready：
```text
AllReduce Bucket0
AllReduce Bucket1
...
```

全部结束以后：
```text
Rank0 gradients
=
Rank1 gradients
=
Rank2 gradients
=
Rank3 gradients
```

然后每个 Rank：
```python
optimizer.step()
```

因为：
```text
起点参数相同
+
Gradient 相同
+
Optimizer State 相同
```

所以：
```text
更新后的参数仍然相同
```

# 5.7 最小 DDP 代码

下面这份代码只保留真正理解 DDP 所需结构：
```python
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = build_dataset()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model().to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    for epoch in range(10):
        sampler.set_epoch(epoch)

        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = compute_loss(outputs, targets)

            loss.backward()
            optimizer.step()

        if rank == 0:
            torch.save(
                model.module.state_dict(),
                "model.pth",
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

真正新增的 DDP 核心只有：

```text
init_process_group
LOCAL_RANK
DistributedSampler
DDP(model)
sampler.set_epoch
destroy_process_group
```

而：

```text
Forward
Loss
Backward
Optimizer
```

和单卡训练几乎没有改变。

这也是 DDP API 设计得比较好的地方：

> **分布式同步隐藏在 DDP Wrapper 和 Autograd 中，而不是要求你手写每个梯度的 AllReduce。**

### 本章总结

一次 DDP Iteration 可以看成多个独立的单卡训练同时进行；唯一关键区别是：`backward()` 期间 DDP 自动插入 Gradient AllReduce，使所有 Replica 在 `optimizer.step()` 前重新拥有相同 Gradient。

# 6 DDP 最容易出错的地方：同步、Checkpoint 与 Debug

理解 DDP 后，很多“玄学卡死”其实都能从一个原则解释：

> **DDP 是同步训练，多个 Rank 必须执行兼容的 Collective Communication 序列。**

![](/images/AI/6%20Model%20Training/attachments/ddp_06_collective_hang.png)

# 6.1 为什么一个 Rank 少跑一次就可能 Hang

假设：
```text
Rank0:
Iteration 1 → AllReduce
Iteration 2 → AllReduce

Rank1:
Iteration 1 → AllReduce
结束
```

Iteration 2：
```text
Rank0:
“我要和 Rank1 AllReduce”

Rank1:
已经退出
```

Rank0 会等待。

所以 DDP 不是：
```text
每个 Rank 想跑多少就跑多少
```

而是多个 Rank 共同完成一个同步训练协议。

PyTorch DDP 内部也要求各 Rank 的 Reducer 按一致顺序执行 AllReduce，否则可能出现错误或 backward hang。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/ddp?utm_source=chatgpt.com "Distributed Data Parallel — PyTorch 2.13 documentation"))

# 6.2 Uneven Inputs

**Uneven Inputs（不均匀输入：不同 Rank 的 Training Loop iteration 数不同）**是典型问题。

PyTorch 提供：

```python
with model.join():
    ...
```

来处理一些不均匀输入场景。

已经耗尽数据的 Rank 会继续“影子执行”必要 Collective，使仍在训练的 Rank 不会因为缺少通信伙伴而直接 Hang；最终模型还会重新同步。([PyTorch Documentation](https://docs.pytorch.org/tutorials/advanced/generic_join.html?highlight=distributeddataparallel&utm_source=chatgpt.com "Distributed Training with Uneven Inputs Using the Join Context Manager — PyTorch Tutorials 2.13.0+cu130 documentation"))

但最简单、最稳定的场景仍然是：
```text
所有 Rank
执行相同数量 Iteration
```

这也是 DistributedSampler 默认会尽量让各 Rank 样本数对齐的原因之一。

# 6.3 `find_unused_parameters` 是什么

假设模型有条件分支：
```python
if condition:
    x = branch_a(x)
else:
    x = branch_b(x)
```

某一次 Forward：
```text
branch_a 参数参与 Loss
branch_b 参数完全没用
```

那么 branch_b 的 Parameter：
```text
不会产生 Gradient
```

DDP 默认 Reducer 可能还在等待：
```text
“这个参数的 Gradient 什么时候 Ready？”
```

`find_unused_parameters=True` 会从 Forward Output 反向遍历 Autograd Graph，提前识别没有参与当前 Backward 的 Parameter，并把它们标记为无需继续等待。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/ddp?utm_source=chatgpt.com "Distributed Data Parallel — PyTorch 2.13 documentation"))

但这会增加 Graph Traversal 开销。

所以不是：
```python
find_unused_parameters=True
```

永远更安全。

而是：
> **只有模型确实存在动态未使用 Parameter 时才开启。**

# 6.4 为什么 Rank 0 经常负责保存

所有 Rank Model 参数本来就应该一致。
如果：
```text
Rank0 保存 model.pth
Rank1 保存 model.pth
Rank2 保存 model.pth
Rank3 保存 model.pth
```

得到的是几份重复文件，而且还可能同时写同一路径。
所以通常：
```python
if rank == 0:
    torch.save(...)
```

这里只是减少重复 I/O。

但不要把这个经验错误扩展为：
> “只有 Rank0 才需要参与训练。”

Training Collective 仍然必须让所有 Rank 正常参与。

# 6.5 为什么保存 `model.module`
DDP 后：
```python
model = DDP(original_model)
```

此时：
```text
model
=
DDP Wrapper
```

真正的原模型位于：
```python
model.module
```

因此常见：
```python
torch.save(
    model.module.state_dict(),
    path
)
```

这样得到的 state dict 不依赖 DDP Wrapper，更方便之后：
```text
单卡推理
DDP Resume
CPU Load
```

# 6.6 Resume 时真正要保持什么一致

Resume 不只是模型 Weight。

完整训练状态通常还有：
```text
Model State
Optimizer State
Scheduler State
Scaler State
Iteration / Epoch
Random State
```

至少 Model 和 Optimizer 必须正确恢复，否则：
```text
Weight 回到了 checkpoint
但 Adam momentum 没恢复
```

训练轨迹已经不是原来的 Resume。

在 DDP 中，各 Rank 还应该从同一逻辑 checkpoint 恢复，使：

```text
Model Replica
Optimizer State
Current iteration
```

重新一致。

# 6.7 Barrier 是不是越多越安全

**Barrier（屏障：所有 Rank 必须都到达该位置后才能继续）**：

```python
dist.barrier()
```

很容易让人产生：
> 多加几个 Barrier 会更安全。

实际上它只是强制同步。

不必要的 Barrier 会让：
```text
快 Rank 等慢 Rank
```

增加空闲时间。
因此 Barrier 应该用于确实需要全局阶段同步的位置，而不是拿来“治疗一切分布式问题”。

# 6.8 Debug DDP 要先问什么

遇到 Hang 时，最重要的问题不是：
> NCCL 是不是坏了？

而是先检查：
```text
每个 Rank iteration 数一样吗？

每个 Rank 是否走了相同模型分支？

是不是某个 Rank 抛异常退出？

Collective 顺序是否一致？

是否有 unused parameter？

是否有额外 distributed operation？
```

如果某个 Rank 提前异常退出，其他 Rank 很可能只表现为：
```text
卡在某个通信操作
```

根因却在另一张卡。

### 本章总结

DDP 的绝大多数同步问题都可以追溯到：**不同 Rank 没有执行匹配的分布式通信流程**。Rank0-only 应主要用于 I/O；模型分支、iteration 数和 Collective 顺序则必须从全局角度设计。

# 7 DDP 为什么能加速，又为什么不会无限线性加速

理想状态：

```text
1 GPU → 1x speed
2 GPU → 2x speed
4 GPU → 4x speed
8 GPU → 8x speed
```

现实通常达不到。
![](/images/AI/6%20Model%20Training/attachments/ddp_07_scaling_efficiency.png)


# 7.1 Speedup 怎么定义

假设单卡训练一个 iteration：

$$
T_1
$$

$N$ 卡：

$$
T_N
$$

Speedup：

$$
S_N = \frac{T_1}{T_N}
$$

理想：

$$
S_N=N
$$

例如 4 卡：

$$
S_4=4
$$

# 7.2 Scaling Efficiency

定义：

$$
E_N = \frac{S_N}{N}
$$

理想：

$$
E_N=1
$$

也就是：

$$
100%
$$

假设 8 卡只达到：

$$
S_8=6
$$

那么：

$$
E_8 = \frac{6}{8} = 0.75
$$

即：75%


# 7.3 为什么卡越多，效率越难保持

单卡主要是：
```text
Forward
Backward
Optimizer
```

DDP：
```text
Forward
Backward
+
Gradient Communication
+
Synchronization
+
Distributed Data Loading
```

可以粗略写：

$$
T_{DDP} \approx T_{compute} + T_{communication} + T_{wait}
$$

GPU 增加后：
$$
T_{compute}
$$

通常下降。

但：

$$
T_{communication}
$$

不会按相同比例下降。

所以最终：
```text
GPU 越多
↓
每卡计算越来越少
↓
通信占比越来越高
```

# 7.4 为什么大模型有时反而更适合 DDP

假设一个非常小的网络：
```text
Forward + Backward
= 很快
```

但每一步仍要：
```text
AllReduce
```

于是：
```text
Compute  1 ms
Comm     2 ms
```

通信已经比计算贵。

增加 GPU 很难加速。

而大网络可能：
```text
Compute  100 ms
Comm      20 ms
```

并且其中部分 Comm 还能和 Backward Overlap。

那么 DDP 更容易获得收益。

所以判断 DDP 是否值得，不应该只看：

```text
有几张 GPU
```

还要看：
> **单卡计算量与通信量的比例。**

# 7.5 Bucket 为什么会影响性能

Bucket 太大：
```text
等很多 Gradient 都 Ready
↓
才能启动 AllReduce
```

通信开始晚。

Bucket 太小：
```text
大量小 AllReduce
↓
通信启动开销增加
```

所以 Bucket Size 实际是在权衡：
```text
早点开始通信
vs
减少通信次数
```

PyTorch 的 `bucket_cap_mb` 就用于控制 Bucket 大小，使梯度 Reduction 有机会和 Backward Compute 重叠。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

通常不需要一开始就手动调。

先确认：
```text
数据加载不是瓶颈
GPU 利用率正常
DDP 逻辑正确
```

再做 Bucket 优化。

# 7.6 最慢 Rank 决定整体速度

假设：
```text
Rank0 backward：100 ms
Rank1 backward：101 ms
Rank2 backward：100 ms
Rank3 backward：160 ms
```

AllReduce 需要大家参与。

所以其他 Rank 会等待 Rank3。

最终 iteration 不可能是：
```text
约 100 ms
```

而会被：
```text
Rank3
```

拖慢。

这种 Rank 叫：
**Straggler（拖尾进程：执行速度明显慢于其他 Rank、迫使同步系统整体等待的进程）**。

原因可能包括：
```text
DataLoader 偶发慢
GPU 被其他程序占用
CPU 资源不均衡
网络波动
数据样本处理成本差异
```

所以分布式性能分析不能只看：
```text
平均 GPU 利用率
```

还要观察不同 Rank 是否均衡。

# 7.7 Gradient Accumulation 为什么会浪费通信

假设要累积 4 个 Micro Batch：
```text
Backward 1
Backward 2
Backward 3
Backward 4
↓
Optimizer.step()
```

默认 DDP：
```text
Backward 1 → AllReduce
Backward 2 → AllReduce
Backward 3 → AllReduce
Backward 4 → AllReduce
```

但前三次没有更新 Parameter。

它们只是在累积 Gradient。

于是前三次通信可以省掉。

DDP 提供：
```python
with model.no_sync():
    ...
```
**`no_sync()`（暂时关闭 DDP Gradient Synchronization 的 Context Manager）**。

例如：
```python
optimizer.zero_grad()

with model.no_sync():
    for micro_batch in micro_batches[:-1]:
        output = model(micro_batch)
        loss = compute_loss(output)
        loss.backward()

output = model(micro_batches[-1])
loss = compute_loss(output)
loss.backward()

optimizer.step()
```

前三次：
```text
Local Gradient Accumulation
```

最后一次：
```text
AllReduce
+
最终累计 Gradient
```

PyTorch 官方性能指南也明确建议 Gradient Accumulation 时避免每个 backward 都执行不必要的 AllReduce。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

注意：
> `no_sync()` 要包住 Forward 和 Backward；只包 Backward 可能不能得到预期行为。([PyTorch Documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch main documentation"))

# 7.8 单机多卡为什么通常比跨机器更容易扩展

单机 GPU 之间可能通过：
```text
NVLink
PCIe
```
通信。

多机：
```text
GPU
↓
NIC
↓
Network
↓
NIC
↓
GPU
```
网络延迟和 Bandwidth 通常成为新的限制。

因此相同 4 GPU：
```text
1 node × 4 GPUs
```

往往比：
```text
4 nodes × 1 GPU
```

更容易获得较高扩展效率。PyTorch 多机 DDP 教程也明确指出 inter-node communication latency 会成为多机训练瓶颈。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode?utm_source=chatgpt.com "Multinode Training — PyTorch Tutorials 2.13.0+cu130 documentation"))

### 本章总结

DDP 加速来自多个 Rank 并行承担计算，但必须额外支付 Gradient Communication 和同步等待成本。性能优化的核心不是“增加更多 GPU”，而是让 **Compute 足够大、通信尽量重叠、各 Rank 足够均衡，并避免无意义的同步**。

# 整体知识链

整个 DDP 可以最终压缩成：
```text
torchrun
↓
启动多个 Process
↓
每个 Process 绑定一张 GPU
↓
每张 GPU 创建完整 Model Replica
↓
DistributedSampler 给每个 Rank 不同数据
↓
各 Rank 独立 Forward
↓
各 Rank 独立计算 Loss
↓
各 Rank 独立开始 Backward
↓
Gradient Ready
↓
Autograd Hook
↓
Gradient Bucket
↓
AllReduce
↓
所有 Rank 得到相同平均 Gradient
↓
每个 Rank 独立 optimizer.step()
↓
所有 Model Replica 继续保持一致
```

真正需要理解的不是 DDP API，而是四句话：
1. **DDP 并行的是 Data，不是 Model。**
2. **每个 Rank 独立 Forward / Backward，但处理不同 Local Batch。**
3. **Backward 过程中 Gradient 通过 AllReduce 被同步成一致结果。**
4. **因为更新前 Gradient 相同，所以所有 Model Replica 始终保持同步。**

# 一句话总结

**DDP 的本质是：多个独立 Rank 各自在完整模型上处理不同数据，再在 Backward 过程中同步 Gradient，使它们逻辑上像一个使用更大 Global Batch 的统一训练过程。**