---
title: "时域中的两种IIR问题解决方案调研"
categories: ["AI"]
author: "BluHuang"
date: 2026-08-25T20:24:00+0800
lastmod: 2026-08-25T20:24:00+0800
---

## 3 解决方案调研：如何保留时域去噪，同时避免错误递归

要解决该问题，需先梳理视频降噪中几种典型的时域处理思路，并结合当前两个具体问题判断这些方法是否对症：

- **静态倒计时**：历史输出中的局部轮廓错误会继续传播到下一帧；
- **旋转场景**：桌子、窗帘等运动边界附近出现黑块；
- 但与此同时，**开启历史参考时平坦区域去噪效果明显更好**。

因此我们真正希望实现的目标不是“少用历史帧”，而是：
> **可靠的历史信息继续使用，不可靠的历史信息不再传播。**

### 3.1 当前已有 Motion Mask

在设计新的网络结构之前，先梳理当前代码中已有的内容。

当前 `IIR_Train` 中已经定义了：
```python
def _motion_mask(self, image, ref_image):
    image_blur = self._gaussian_blur(image)
    ref_image_blur = self._gaussian_blur(ref_image)
    motion_mask = torch.abs(image_blur - ref_image_blur)
    return motion_mask
```

其中 Gaussian Blur 默认：
```python
kernel_size = 5
sigma = 2.0
```

因此其实际逻辑非常简单：
$$
M_t = |G(I_t)-G(I_{t-1})|
$$

其中：
- (I_t)：当前帧；
- (I_{t-1})：相邻参考帧；
- (G)：Gaussian Blur；
- (M_t)：两帧的差异图。

先进行 Blur 再做差分，是为了降低像素级随机噪声对差分的影响，使差异更聚焦于实际内容变化。当前代码还会将 `diff` clamp 到 `[0,1]`。

直观理解：

```text
两帧内容相似
    ↓
diff ≈ 0
    ↓
认为这里比较静态


两帧内容明显不同
    ↓
diff ↑
    ↓
认为这里发生运动 / 内容变化
```

例如相机旋转时：

```text
上一帧 (x,y)：窗帘
当前帧 (x,y)：桌子
```

该位置经过 Blur 后依然会存在较大差异，因此 `diff` 会比较大。

#### 当前 Mask 实际用在哪里？

它目前主要作用于**训练监督构造**：
```python
noise_sum = (
    noise_sum * (1.0 - diff)
    + noise_motion_weight * diff * noise_map_decomp
)

update_noise = (
    noise_sum
    * (1 - mask_all)
    * noise_ctr_map
    * noise_update_scale
)
```

也就是说，设计思想已经是：
```text
静态区域
diff 小
→ 可以继续累积历史 noise

运动区域
diff 大
→ 减少历史 noise 的累积
→ 更多使用当前帧 noise
```

此外 `diff_max` 还被用于 `motion_loss`。
**这里已体现出一个正确思想：运动区域不能像静态区域那样无限信任历史。**

但存在一个关键问题：

#### 当前 Mask 没有作用到真正的 IIR ref

真正进入网络的仍然是：
```python
if n == 0:
    ref_img_decomp = zeros
else:
    ref_img_decomp = net_outputs.detach()

net_inputs = cat(
    ref_img_decomp,
    input_img_decomp,
    ...
)
```

也就是说，即使：

```text
diff = 1
```

表示某个位置已经发生大幅运动，真正喂给网络的：

```text
prev_out
```

**仍然完全没有被 mask。**

所以目前逻辑实际是：

```text
                   ┌── Motion Mask ──→ 调整训练 Target / Noise
GT current/previous│
                   │
                   │
prev_out ─────────────────────────────→ SNRNet
                   ↑
              没有 Mask
```

这可能正是值得利用的地方：
> 我们已经有“哪些区域可能不应信任历史”的概念，但目前它只参与监督，没有参与真正的历史信息传播。

#### 当前 Mask 的一个重要限制

它是通过：
```python
target_img_decomp
pre_target_img_decomp
```

计算的。

也就是说，使用的是**训练时的 clean target**。

真实推理中不存在 clean target，因此不能直接把这个 Mask 原封不动搬到 inference。

推理时真正可获取的是：
```text
current noisy RAW
previous model output
k / b / sfs
```

所以如果将来要用 Mask 控制 ref，需要构造一个**推理时可获得的 confidence/motion mask**。

另外，`time_loss` 使用的并不是上面这个 `diff`，而是另一个：
```python
pt_motionmask_norm_to_01(...)
```

#### 当前 Mask 在计算 diff 时使用 clean GT

假设真实数字连续两帧完全没有变化：

```
GT frame5：
数字边缘正确

GT frame6：
数字边缘仍然正确
```

那么 clean GT 计算得到：
```
diff ≈ 0
```

但是：
```
model output frame5：
数字边缘多了一个错误白点
```

当前的 Motion Mask 根本不知道这个白点存在。

因为它比较的是：

```
GT(t)

vs

GT(t-1)
```

而不是：
```
model_prev_out

vs

current GT/current input
```

所以会出现这种情况：
```
真实场景：

完全静止
      ↓
GT diff ≈ 0

但是：
prev_out 已经错了
      ↓
当前 Motion Mask 仍然认为：

“这里很静态，history 理论上应该可靠”
```

这正是**倒计时异常轮廓最难处理的一点**。

### 3.2 方案一：Time Loss —— 让前后帧输出更一致

这是当前工程已有但尚未开启的方案。

基本思想是：

```text
静态区域：

Output(t)
   ↓ 应该接近
Output(t-1)
```

当前实现大致为：
```python
time_loss_l1 = mean(
    abs(
        (net_out - ref)
        * (1 - motion_mask)
    )
)
```

即：
- 运动区域：不强制一致；
- 静态区域：要求当前输出接近历史输出。

该方法主要解决：
> **Temporal consistency：前后帧不要闪、不要抖。**

但当前问题在于：
> **Temporal correctness：上一帧的 state 本身可能已经错误。**

以倒计时为例：
```text
frame5 的 prev_out：
正确数字 + 一个错误白点
```

如果开启 Time Loss，它识别到这是静态区域后，会要求：
```text
frame6 output ≈ frame5 output
```

那么错误白点反而可能被继续维持。

所以：
> **Time Loss 可以防止“正确结果乱跳”，却不能保证“历史结果本身正确”。**

因此单独开启 Time Loss 对当前问题并不对症，甚至可能让静态错误更加稳定。

### 3.3 方案二：降低 Ref 权重 / Periodic Reset

最直接的方法是将：
```text
当前：
ref = prev_out
```

改为：
```text
ref = α × prev_out
```

例如：
```text
α = 1.0    原始 IIR
α = 0.75
α = 0.5
α = 0.25
α = 0      zero-ref
```

或者每隔若干帧：
```text
output0 → output1 → output2 → reset
                              ↓
                              ref=0
```

该方案容易理解，也确实能减轻长期的 error accumulation。

但它有明显缺陷：
```text
好的历史信息      被削弱
坏的历史信息      也被削弱
```

它没有能力判断历史信息的好坏。

而当前 zero-ref 消融已经证明：
```text
历史信息全部删除
→ 异常减少
→ 但整图噪声明显增加
```

因此这类方法更适合作为**问题诊断或安全兜底**，但不是理想的最终方案。

### 3.4 方案三：有限邻帧，不做无限 IIR

FastDVDnet 代表了另一种思路：

> 不维护永久递归的历史 state，而是在一个固定短时间窗口内联合处理若干帧。

FastDVDnet 使用邻近帧进行多阶段视频去噪，不依赖 optical flow，也不存在 `output_t → input_{t+1} → output_{t+1}` 这种无限递归 state。([CVF开放存取](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html?utm_source=chatgpt.com "CVPR 2020 Open Access Repository"))

它的最大优势是：
> **历史错误不会无限向后传播。**

但代价也很明显：
- 需要缓存多帧；
- 网络结构变化较大；
- 无法像当前 IIR 那样自然地无限利用历史信息；
- 对当前已经存在且平坦区表现不错的 recurrent pipeline 改动较大。

因此它更像是一种**重新设计时域架构**的方案，而不是修复当前 IIR 的第一选择。

### 3.5 方案四：Alignment / Motion Compensation

这条路线主要针对旋转黑块。

当前网络是：

```text
prev_out(x,y)
+
current(x,y)
```

但相机旋转后：
```text
prev_out(x,y) = 窗帘
current(x,y)  = 桌子
```

二者已经不是同一个真实物体。

Alignment 的思想是：
```text
previous
   ↓
估计 motion / optical flow
   ↓
warp
   ↓
aligned previous
   ↓
和 current 融合
```

WACV 2023 的 _Video Joint Denoising and Demosaicing With Recurrent CNNs_ 专门比较了 recurrent / non-recurrent、传播信息类型以及 motion compensation 等设计，实验表明 recurrent network 配合 motion compensation 表现最好。([CVF开放存取](https://openaccess.thecvf.com/content/WACV2023/html/Dewil_Video_Joint_Denoising_and_Demosaicing_With_Recurrent_CNNs_WACV_2023_paper.html?utm_source=chatgpt.com "WACV 2023 Open Access Repository"))

BasicVSR++ 则使用 **flow-guided deformable alignment**，目的是让 recurrent propagation 在帧间存在明显错位时仍能正确利用历史信息。([CVF开放存取](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.html?utm_source=chatgpt.com "CVPR 2022 Open Access Repository"))

因此对于：
```text
相机旋转
物体移动
遮挡 / 反遮挡
```

Alignment 是非常关键的技术。

但它有一个局限。

#### Alignment 不能完全解决倒计时异常

倒计时基本没有明显移动。

假如：
```text
上一帧数字的位置已经完全正确
但是模型自己多预测了一个白点
```

Alignment 之后只是：
```text
把这个错误白点非常准确地对齐回来
```

错误依然存在。

所以 Alignment 解决的是：
> **历史位置不对。**

而倒计时还存在：
> **历史内容本身不对。**

因此只做 Alignment 不能同时覆盖当前两个问题。

### 3.6 方案五：Gate / Confidence —— 判断历史信息是否可信

这是 GRU 类方法的核心思想，也是目前最值得深入理解的一类。

普通 IIR：
```text
prev_out
    ↓
无条件使用
    ↓
network
```

Gate 方式：

```text
             current
                +
             prev_out
                +
          noise information
                ↓
       估计 history confidence
                ↓
       0 ←──── gate ────→ 1
       │                  │
   不信历史            完全信历史
                ↓
        gate × prev_out
                ↓
             network
```

GRU-VD 将 GRU 的思想用于 video denoising：它用 reset gate 判断上一帧输出中哪些内容仍然与当前帧相关，再利用这些相关历史内容进行降噪；随后 update gate 控制当前结果与历史结果的递归融合，并把 noise standard deviation 也作为 gate 判断的输入。([arXiv](https://arxiv.org/abs/2210.09135?utm_source=chatgpt.com "Gated Recurrent Unit for Video Denoising"))

该机制与当前问题非常契合。

#### 对平坦区域

```text
current ≈ prev_out
而差异主要来自随机 noise
```

那么：
```text
confidence 高
→ 大量使用 prev_out
→ 保留现在 IIR 很好的平坦区去噪能力
```

#### 对倒计时错误轮廓

假设：
```text
current：
真实边缘附近没有白色凸起

prev_out：
出现了错误白点
```

这里 current 和 prev_out 出现局部不一致：
```text
confidence ↓
→ 减少错误 ref 的传播
```

#### 对旋转场景
```text
previous = 窗帘
current  = 桌子
```

差异很大：
```text
confidence ↓
→ 不再强行使用这块 history
```

所以 Gate 的本质不是：
> “少用时域。”

而是：
> **动态决定每个位置应该使用多少时域。**

这一点与 zero-ref 消融得到的需求完全一致。

### 3.7 当前 Motion Mask 与 Gate 的关系

这里很容易混淆。

两者核心思想其实非常接近：
```text
Motion Mask：
这里和上一帧差多少？

Gate：
根据这种差异判断上一帧还能信多少？
```

可以认为：
```text
Motion Mask
     ↓
一种最简单的 confidence signal
```

例如在概念上：
```text
motion_mask ≈ 0
→ 历史可靠
→ gate ≈ 1

motion_mask ≈ 1
→ 历史不可靠
→ gate ≈ 0
```

即：

$$
gate \approx 1-motion_mask
$$

因此一个自然的方向是：

> **能否直接利用当前 mask？**

这个方向在逻辑上成立。

而且对我们而言有一个明显优势：
> **不一定需要一开始就增加一个复杂的新网络。**

但现有 Mask 不能直接使用，因为它依赖 clean target。

真正用于 inference 的 Mask 至少要从：

```text
current noisy RAW
prev_out
k / b / sfs
```

这些实际可获得的信息中估计。

这里还必须考虑噪声：

```text
ISO4000：

current 和 prev_out 本来就会因为 noise 差很多
```

所以不能简单地：

```python
abs(current - prev_out) > 固定阈值
```

否则大量正常噪声也会被误判为运动。

更合理的思想应该是：
> **当前差异是否已经大到超过 k/b 所能解释的正常噪声范围？**

这也与 GRU-VD 将 noise level 一并送入 gate 的设计逻辑一致。([arXiv](https://arxiv.org/abs/2210.09135?utm_source=chatgpt.com "Gated Recurrent Unit for Video Denoising"))

### 3.8 方案六：EMVD —— 将“时域融合”和“空间去噪”拆开

EMVD 是另一项值得了解的工作，因为它面向实际视频降噪和低计算复杂度。

它不是简单：
```text
prev_output + current
→ 一个网络全部解决
```

而是明确拆分为：

```text
历史帧
   +
当前帧
   ↓
Temporal Fusion
   ↓
Denoising
   ↓
Refinement
```

即首先递归融合过去帧以降低噪声，然后单独执行 denoise，最后通过 refinement 恢复被损失的高频细节。论文特别强调低计算量，并在真实 RAW 数据及移动 SoC 上进行了实验。([CVF开放存取](https://openaccess.thecvf.com/content/CVPR2021/html/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.html?utm_source=chatgpt.com "CVPR 2021 Open Access Repository"))

它给出的一个重要启发是：
> **“如何使用历史帧”和“如何做单帧空间恢复”不一定都要让 CNN 隐式完成。**

当前 SNRNet 则是：
```text
prev_out + current
         ↓
     一个 SNRNet
```

历史融合规则完全隐藏在网络内部。

EMVD 的思路更可控，但如果完全迁移其结构，工程改动会比单独加 Mask/Gate 大得多。

### 3.9 方案七：Alignment + Spatial Re-weighting

将前面的思想组合起来，就得到目前 video restoration 中更完整的一类方案：

```text
history
   ↓
Alignment
解决“位置是否对应”
   ↓
Aligned history
   ↓
Confidence / Weight
解决“内容是否可信”
   ↓
可靠 history
   ↓
与 current 融合
```

_Revisiting Temporal Alignment for Video Restoration_ 特别指出，长距离 propagation 本身可能产生 error accumulation；除了改进 alignment，该工作还提出 spatial-wise adaptive re-weighting，使不同位置的邻帧贡献可动态调整。([CVF开放存取](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Revisiting_Temporal_Alignment_for_Video_Restoration_CVPR_2022_paper.html?utm_source=chatgpt.com "CVPR 2022 Open Access Repository"))

因此从完整理论上讲：

> **Alignment + spatial confidence / re-weighting**

要比单纯 Alignment 或单纯 Gate 更完善。

因为它同时回答两个问题：
```text
1. 历史像素现在应该在哪里？
   → Alignment

2. 对齐以后，这个历史内容还能不能信？
   → Confidence / Gate
```

### 3.10 不同方案对比

结合当前两个真实问题，可以得到以下比较。

| 方法                         | 倒计时错误累积        | 旋转黑块     | 保留平坦区时域降噪 | 对当前网络改动 | 判断              |
| -------------------------- | -------------- | -------- | --------- | ------- | --------------- |
| Time Loss                  | 较弱，甚至可能固化错误    | 较弱       | 好         | 很小      | 不对症             |
| Ref 整体降权                   | 有效             | 有效       | **会损失**   | 很小      | 适合消融            |
| Periodic Reset             | 有效             | 有效       | 会损失一部分    | 很小      | 可做兜底            |
| Zero-ref                   | 已验证有效          | 可能有效     | **明显损失**  | 极小      | 不是最终方案          |
| FastDVDnet 类有限帧            | 能避免长期 state 污染 | 有一定能力    | 可以        | **很大**  | 相当于换架构          |
| Alignment                  | 有限             | **很强**   | 好         | 较大      | 主要解决运动          |
| EMVD 类显式时域融合               | 有潜力            | 有潜力      | **很好**    | 较大      | 产品化思路很好          |
| Gate / Confidence          | **很强**         | **较强**   | **很好**    | 中等      | **很匹配当前问题**     |
| Alignment + Gate           | **很强**         | **很强**   | **很好**    | 最大      | 理论最完整           |
| **利用现有 Mask 做 confidence** | **很有潜力**       | **很有潜力** | **很好**    | **较小**  | **最适合当前架构继续探索** |