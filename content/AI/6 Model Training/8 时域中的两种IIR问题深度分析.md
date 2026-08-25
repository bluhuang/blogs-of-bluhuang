---
title: "时域中的两种IIR问题深度分析"
categories: ["AI"]
author: "BluHuang"
date: 2026-08-25T20:24:00+0800
lastmod: 2026-08-25T20:24:00+0800
---

## 1. 问题背景

当前模型采用 IIR 方式利用历史帧：

```text
当前 noisy RAW + 上一帧模型输出
              ↓
            SNRNet
              ↓
          当前帧输出
              ↓
      作为下一帧参考帧
```

从消融实验来看，IIR 本身有明显收益：
- 开启历史参考后，平坦区域去噪效果明显更好；
- zero-ref 后，倒计时异常轮廓明显减少，但整图噪声也明显增大；
- 因此不能简单通过“降低时域强度”或“去掉参考帧”解决问题。

当前主要存在两类 IIR 异常：
1. **手机倒计时异常轮廓**：真实场景基本静止，但模型输出中的局部错误可能进入后续递归。
2. **旋转场景黑块 / 蒙砂**：真实场景发生运动后，上一帧 history 与当前帧同坐标内容已经不匹配。

两类问题的共同点都是：

> **history 在部分位置已经不可靠，但仍然参与下一帧推理。**

# 2. 当前 IIR 训练机制

```text
【模型输入链路】

上一帧 output ─┐
当前 noisy RAW ├→ SNRNet → 当前 output
k / b / sfs ──┘


【训练监督链路】

相邻 clean GT → Motion Mask(diff)
                    ↓
当前 noise + 历史 noise_sum
                    ↓
             target_mix_noise
                    ↓
              各种 Loss
                    ↓
                 output
```

## 2.1 IIR 模型输入：上一帧输出如何递归

多帧模式的核心代码为：

```python
if n == 0:
    ref_img_decomp = torch.zeros_like(input_img_decomp)
else:
    ref_img_decomp = net_outputs.detach()

net_inputs = torch.cat([
    ref_img_decomp,
    input_img_decomp,
    noise_level_map,
    noise_ctr_map,
], dim=1)

net_inputs = torch.clamp(net_inputs, 0.0, 1.0)
net_outputs = self.model_fn(net_inputs)
```

因此时序关系为：

```text
frame0:
ref = 0
current0 → model → output0

frame1:
ref = output0
current1 → model → output1

frame2:
ref = output1
current2 → model → output2
```

网络输入共 11 个 channel：

```text
ref / prev_out     4
current noisy RAW  4
k + b              2
noise_ctr / sfs    1
--------------------
total             11
```

此处没有 `0.6 × history + 0.4 × current` 这种显式融合。`prev_out` 和 `current` 都完整送入网络，具体更信任谁更多，由 SNRNet 自己学习。

### `detach()` 的作用

```python
ref_img_decomp = net_outputs.detach()
```

它**不影响 forward 递归**，上一帧结果依然会进入下一帧。

它切断的是跨帧梯度：

```text
forward:
output0 → output1 → output2       ✓

backward:
loss2 → output2 → output1 → output0
                      ↑
                 被 detach 截断
```

因此当前属于：

> **有 recurrent forward，但没有沿整个 IIR state 做完整 BPTT。**

## 2.2 Motion Mask：判断真实场景中哪里发生了变化

网络 forward 完成后，代码计算：

```python
diff = self._motion_mask(
    self._sim_gamma_noclip(target_img_decomp),
    self._sim_gamma_noclip(pre_target_img_decomp),
)

diff = torch.clamp(diff, 0.0, 1.0)
```

`_motion_mask()` 实现为：

```python
def _motion_mask(self, image, ref_image):
    image_blur = self._gaussian_blur(image)
    ref_image_blur = self._gaussian_blur(ref_image)

    motion_mask = torch.abs(
        image_blur - ref_image_blur
    )

    return motion_mask
```

Gaussian Blur 使用 `kernel_size=5, sigma=2.0`。

因此可记为：

$$
diff=|Blur(GT_t)-Blur(GT_{t-1})|
$$

## 2.3 `noise_sum`：如何构造真正用于监督的 Target

这是当前代码中最容易混淆的一部分。

训练输入中的实际 noise 在 `_prepare_batch()` 中已经生成：

```python
data = torch.clamp(
    degraded_clean_12bit + sampled_noise,
    0.0,
    4095.0,
)

noise = data - degraded_clean_12bit

batch_data["input"] = data / 4095.0
batch_data["noise"] = noise / 4095.0
```

因此可将 `noise_map_decomp` 理解为：

> **当前 noisy input 中实际存在的噪声残差。**

然后 IIR loss 中维护一个 `noise_sum`。

### 第一步：累积历史 Noise

首帧：

```python
noise_sum = (
    self.noise_first_frame_weight
    * noise_map_decomp
)
```

后续帧：

```python
noise_sum = (
    self.noise_prev_weight * noise_sum
    +
    self.noise_curr_weight
    * noise_level
    * noise_map_decomp
)
```

例如当前配置：

```text
noise_prev_weight = 0.6
noise_curr_weight = 0.4
noise_level       = 0.25
```

则大致为：

```text
noise_sum
=
0.6 × 历史 noise_sum
+
0.1 × 当前 noise
```

这里需要特别记住：
> **`noise_sum` 是为了构造监督 Target 的 noise state，不是模型的 `prev_out`。**

> **`noise_level` 也不是 history 权重。**

这些配置的默认值分别为 `noise_level=0.25`、`noise_prev_weight=0.6`、`noise_curr_weight=0.4`。

### 第二步：根据 Motion 调整 `noise_sum`

接下来：

```python
noise_sum = (
    noise_sum * (1.0 - diff)
    +
    self.noise_motion_weight
    * diff
    * noise_map_decomp
)
```

直观理解：

```text
diff 小：
→ 保留更多之前累积的 noise_sum

diff 大：
→ 减少历史 noise_sum
→ 更多转向当前帧 noise
```

但这里还不是最终 target。

### 第三步：生成最终 Residual Noise

```python
mask_all = diff

update_noise = (
    noise_sum
    * (1 - mask_all)
    * noise_ctr_map
    * noise_update_scale
)

target_mix_noise = torch.clamp(
    target_img_decomp + update_noise,
    0.0,
    1.0,
)
```

这一步最关键。

如果是静态区域：

```text
diff ≈ 0
→ 1-diff ≈ 1
→ target 可以带一定 residual noise
```

如果是强运动区域：

```text
diff ≈ 1
→ 1-diff ≈ 0
→ update_noise ≈ 0
→ target_mix_noise ≈ clean GT
```

因此整个 target 构造可以总结为：

> **静态区域允许监督 Target 保留一定残余噪声；运动越明显，Target 越接近当前 clean GT。**

这也是理解 `noise_level` 的关键：它主要调整的是**监督 Target 中允许保留多少 residual noise**，而不是直接调整 IIR 时域权重。

## 2.4 Loss：模型最终被要求学什么

网络得到：

```text
output = net_out
```

训练侧得到：

```text
supervision = target_mix_noise
```

普通 RAW loss 即为：

```python
raw_loss_l1 = self.l1_loss(
    net_out,
    target_mix_noise,
)

raw_loss_ssim = (
    1
    - self.ssim_loss(
        target_mix_noise,
        net_out,
    )
)
```

此外还有 Bright、Edge、Laplacian 等 loss。

### Motion Loss

如果开启：

```python
motion_area_lossl1 = torch.mean(
    torch.abs(
        net_out - target_mix_noise
    )
    * diff_max
)
```

其中 `diff_max` 是四个 Bayer channel 的 `diff` 取最大值，再复制回四通道。

因此它的作用很直接：

```text
静态区域：
diff 小
→ 几乎不额外加权

运动区域：
diff 大
→ reconstruction error 被额外放大
```

所以 Motion Loss 的含义是：
> **真实发生变化的位置，要额外强调“当前帧恢复正确”。**

它和 Time Loss 不同：

```text
Motion Loss
→ 运动区域更接近当前 Target

Time Loss
→ 静态区域当前输出更接近历史输出
```

目前配置中 `motion_loss_enable` 默认为 False，当前实验配置也未开启。

## 2.5 把整个训练过程串起来

至此，一帧的训练过程可以完整理解成：

```text
① 取当前 noisy RAW
        +
   上一帧模型 output
        ↓
② 拼成 11 channel
        ↓
③ SNRNet
        ↓
④ 得到当前 output
        │
        │
        ├───────────────────────────────┐
        │                               │
        ↓                               ↓
   下一帧作为 ref              当前 clean GT
                                上一帧 clean GT
                                      ↓
                               Motion Mask(diff)
                                      ↓
                         当前 noise + 历史 noise_sum
                                      ↓
                               target_mix_noise
                                      ↓
                         L1 / SSIM / Edge / ...
                                      ↓
                               监督当前 output
```

这里最值得记住三个概念：

|概念|是什么|作用在哪里|
|---|---|---|
|`prev_out`|上一帧模型输出|**模型输入 / IIR state**|
|`diff`|相邻 clean GT 的局部变化程度|**监督构造 / Motion Loss**|
|`noise_sum`|用于构造 Target 的 residual-noise state|**训练监督**|

三者不是同一个“时域变量”。

最后还有一个独立的代码行为需要记录：多帧循环中存在

```python
if n == (len(frame_list) - 5):
    ...
    break
```

这个 `break` 位于 GAN 条件之外，因此当前 15 帧数据会在 `n=10` 结束这一轮递归；它是否为历史设计遗留，需要另行确认。

# 3. 两种 IIR 问题分析

## 3.1 手机倒计时：局部 History-Current 失配导致异常轮廓

逐帧分析后，当前不再倾向于将倒计时问题简单解释为“模型自身错误逐帧累积”。

实际观察：

```text
frame0: 00:42.08
frame1: 00:42.11
frame2: 00:42.14
```

倒计时数字**每一帧都在变化**，同时 frame0→frame1 还存在轻微相机移动。

推理结果表现为：

```text
frame0:
无历史 ref
→ 数字基本正常

frame1:
ref = frame0 output
current = 新的数字 + 轻微位置变化
→ 已经出现异常轮廓

frame2:
数字继续变化
→ 异常轮廓仍主要围绕当前数字 14 出现
```

同时：
- recursive 下 `00:42` 和当前数字周围都有明显异常轮廓；
- zero-ref 下 `00:42` 和 `14` 周围都明显更正常；
- 没有观察到明确的“08 残留在 11 后面”或“11 残留到 14”这种旧数字形状；
- 因此目前不支持简单的“上一数字直接残影”解释。

当前更符合的机制是：

```text
上一帧 output
        +
当前帧数字发生变化
        +
相机存在轻微移动
        ↓
history 与 current 局部结构不完全对应
        ↓
但完整 history 仍然进入 SNRNet
        ↓
模型对 history 使用过强 / current 优先级不足
        ↓
当前高亮数字边缘被错误融合
        ↓
产生异常轮廓
```

关键证据：

- 原始 RAW 不存在异常轮廓；
- frame0 没有 history 时基本正常；
- 第一次引入 recursive ref 的 frame1 就已经出现异常；
- zero-ref 下当前数字轮廓明显恢复正常；
- clip 无效；
- synthetic glyph 增强没有明显解决真实场景。

因此当前更准确的判断是：

> **倒计时异常属于局部 History-Current mismatch 下的 temporal fusion artifact。**

它不一定是长期 error accumulation，也不一定是旧数字 ghosting，而更可能是：

> **history 在局部快速结构变化/轻微位移区域已经不够可靠，但网络仍然过度依赖 history，从而污染当前帧边缘恢复。**

## 3.2 旋转黑块：真实运动导致 History 失配

旋转场景的链路不同：

```text
相机发生旋转
→ 当前 (x,y) 和上一帧 (x,y)
  不再对应同一个真实内容

例如：
prev_out(x,y) = 窗帘
current(x,y)  = 桌子

→ clean GT diff 较大
→ Motion Mask 能识别这个真实变化

但：
完整 prev_out 仍然进入 SNRNet

→ 网络需要自己学会：
  history 与 current 冲突时应该更相信 current

→ 如果学习不足
→ 运动边缘出现黑块 / 蒙砂
```

当前 Motion Mask 已经通过 supervision 告诉模型：
> motion 越大，target 越接近当前 clean GT。

但：
- Motion Loss 当前没有开启；
- ref 仍然完整进入模型；
- 没有显式 alignment / confidence mechanism。

因此当前更倾向于：

> **Motion-induced reference mismatch**

即 history 本身可能并没有预测错，只是由于真实运动，它已经不再适用于当前坐标。

## 3.3 两种问题的区别

|                  | 手机倒计时               | 旋转黑块 / 蒙砂          |
| ---------------- | ------------------- | ------------------ |
| 真实场景             | 基本静态                | 明显运动               |
| clean GT diff    | 通常较小                | 通常较大               |
| History 为什么失效    | 模型自身产生错误            | 当前内容位置发生变化         |
| Motion Mask 能否发现 | 不一定                 | 可以                 |
| 问题本质             | State contamination | Reference mismatch |

共同点都是：
> **history 已经不可靠，但仍继续参与 IIR。**

区别则在于 **history 为什么不可靠**。

---

# 4. 问题分析记录

## 4.1 手机倒计时异常轮廓

### 8/14
- 构造数字、字母、高对比字符 synthetic glyph 数据。
- Exp33 加入 glyph hard sample 后，synthetic validation MAE 有一定改善，但肉眼差异不明显。
- 真实倒计时异常轮廓无明显改善。
- 判断：问题不太像单纯字符训练数据不足。

### 8/18
- model output clip：无明显改善。
- zero-ref：异常轮廓明显减少。
- zero-ref + clip：相比 zero-ref 无额外明显收益。
- 判断：异常与 IIR recursive history 强相关。
- 同时发现 zero-ref 会导致平坦区域噪声增加，因此不能简单删除 history。

### 8/20
进一步逐帧检查：

```text
frame0: 00:42.08
frame1: 00:42.11
frame2: 00:42.14
```

发现：
- 倒计时数字实际上每帧都发生变化；
- frame0→frame1 同时存在轻微相机位移；
- frame0 无 history 时基本正常；
- frame1 第一次引入 history 后即出现异常轮廓；
- frame2 的异常仍主要围绕当前数字 `14`，未观察到明确的上一数字 `11` 残影；
- zero-ref 下 `00:42` 和当前数字边缘均明显更正常。

更新判断：
> 倒计时异常目前更像 **局部 History-Current mismatch 导致的 temporal fusion artifact**，而不是单纯的“上一数字残影”或长期 error accumulation。

## 4.2 旋转场景黑块与蒙砂

### 8/19
- 相机旋转时，在桌子、窗帘等运动边缘观察到黑块 / 蒙砂。
- 初步怀疑与 IIR history 有关。

### 8/20
- 当前 Motion Mask 是局部 diff map，因此能够识别相机旋转产生的真实局部变化。
- Motion Mask 已经通过 noise-target 构造让运动区域 supervision 更接近当前 clean GT。
- 但完整 `prev_out` 仍然进入模型。
- 当前 `motion_loss_enable=False`，运动区域没有额外 reconstruction 权重。
- 当前倾向：  
    **Motion-induced reference mismatch**。