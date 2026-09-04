# Pi0

## 引言

Pi0要解决的问题是什么？

现在机器人学习有三大痛点：

1. **数据稀缺**：给每个任务单独采集机器人演示数据成本极高。
2. **泛化差**：换物体、换环境、换机器人硬件就失效。
3. **灵巧操作弱**：叠衣服、组装纸箱这类精细、长时序任务做不好。

现有 VLA 问题：

1. RT‑2/OpenVLA 这类自回归离散 VLA：动作拆成 token，不支持高频精细操作。
2. Diffusion Policy、ACT：擅长灵巧任务，但**没有互联网预训练 VLM 语义知识，泛化差，不懂语言指令，很难迁移新任务**



### 为什么OpenVLA不支持高频精细操作

- [ ] 动作表示：离散 Token 量化 

把每一个机器人动作维度做**分箱离散化**：例如每个轴把连续浮点数切为 256 个 bin，把浮点动作变成整数 token，把动作当成和单词一模一样的文本 token 输出。

> 8 维动作 → 输出 8 个离散 token； 若想输出一整条 H 步 chunk，就要输出 `H × 8` 个 token。

**固有缺陷：量化误差** 256 个档位精度有限，精细的微小位移、夹爪微调会被离散桶 “卡住”，容易抖动、跳变，不适合折叠布料这类高频灵巧接触操作。

> 就算把 bin 加大，词表会爆炸，训练、显存成本急剧上涨。

- [ ] 训练损失：交叉熵（分类）

动作 token 和文本 token 完全一视同仁，统一用**next‑token‑prediction 交叉熵分类损失**。

- 把每一个 bin 当成独立类别；
- 交叉熵不感知数值距离：`bin=127`和`bin=128`物理上几乎一样，但在 loss 眼里是两个完全无关类别。

> 对连续控制非常不友好，模型很难学到 “动作大小渐变、平滑” 的先验。

- [ ] 推理生成范式：自回归逐 token 生成 

生成 token 是串行链式：**生成第 N 个 token，必须等待前面 N‑1 个全部生成完毕**。

- 输出单时间步动作，就要串行解码 8 个 token，需要 8 次 Transformer 前向；
- 如果想像 π₀一样输出 H=50 步 chunk，那就要输出 `50 ×8 =400`个 token，400 次前向传播！延迟爆炸，完全不可用arXiv。

> 所以原始 OpenVLA 只能**一步一观测，一步推理输出单步动作**，推理延迟高，只能跑到 3‑5Hz，达不到灵巧任务需要 20‑50Hz。 同时自回归存在**误差累积**：前面 token 一点错，后面全部跟着错。

- [ ] 架构

只有一套完整 VLM Transformer，**没有路由，没有两套 FFN/Norm**。

1. 图像 + 文本输入；
2. 输出端直接复用 VLM 原生词表头；
3. 动作被编码为文本 token，和语言 token 混在一起，全部走同样 FFN、同样输出头，全部交叉熵损失。

> 它的设计初衷：尽量不改 VLM，最小改动把机器人数据塞进去，代价就是必须适配 LLM 离散 token 的范式，牺牲连续控制能力。



## 基本架构

![image-20260904152600705](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260904152600761.png)

更详细一点

![ChatGPT Image Sep 4, 2026, 12_45_39 AM](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260904154546585.png)

### Pre-Training

!!! note "为什么要有预训练"
    基于多样化多任务数据进行预训练的通用基础模型往往优于那些针对单一任务精细定制和专门化模型的解决方案
    例如，如果目标是在照片中识别鸟类，更高效的方法可能是先在大量不同的图像-语言关联数据上进行预训练，然后再针对鸟类识别任务进行微调或prompt，而不是仅仅在鸟类识别数据上训练



因此，这里预训练的数据集采用了Cross‑embodiment 跨本体训练

- 自有 π 数据集：7 种机器人硬件，68 种任务，上万小时演示数据；
- 开源 OXE (Open X‑Embodiment) 数据集，来自 22 种不同机器人。 两者混合，构成预训练混合数据集。





### Action Expert

为了解决上面提到的目前VLA的困境，Pi参考**TransFusion**的思路，在Gemma这里加入一个Action Expert，对输入的不同类的Token采取不同的超参数。使得`action expert`成为对机器人的 `state`， `noise`更加专业的一个网络


### π₀ 中 Self-Attention

#### π₀ 的 Token 序列

π₀ 在时刻 $t$ 的输入包括：

$$
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t]
$$

其中：

* $I_t^{1:n}$：多视角 RGB 图像；
* $\ell_t$：语言指令；
* $q_t$：机器人 proprioceptive state；
* $A_t^\tau$：Flow Matching 中的 noisy action chunk。

整个 Transformer 输入按顺序划分为三个 Block：

$$
\boxed{
[\underbrace{I_t^{1:n},\ell_t}_{Block~1}]
[\underbrace{q_t}_{Block~2}]
[\underbrace{A_t^\tau}_{Block~3}]
}
$$

其中

$$
A_t^\tau=
[a_t^\tau,\ldots,a_{t+H-1}^\tau],\qquad H=50.
$$


#### Self-Attention 的作用

Self-Attention 的核心作用可以理解为：

$$
\boxed{
\text{让一个 token 根据其他相关 token 的信息更新自身表示}
}
$$

基本计算为：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V.
$$

因此，每一个 token 都可以根据 Attention 权重，决定：

> “为了理解自己，我应该从哪些其他 token 中获取多少信息？”

在 π₀ 中，Self-Attention 是 **VLM Expert 和 Action Expert 之间最重要的信息交互通道**。

#### Self-Attention 在 π₀ 中具体传递什么信息

对于某一个 noisy action token：

$$
a_{t+i}^{\tau},
$$

它通过 Self-Attention 可以读取：

$$
\boxed{
\text{Image}
+
\text{Language}
+
q_t
+
\text{其他 Action Tokens}
}
$$

因此一个未来动作能够知道：

* 环境中有什么；
* 用户要求机器人完成什么任务；
* 机器人当前姿态如何；
* 整个未来动作序列的其他时间步准备怎样运动。

例如语言指令为：

$$
\text{“Pick up the red cup”}
$$

某个 Action Token 可以同时利用：

$$
\text{图像：红杯子的位置}
$$

$$
\text{语言：需要拿起红杯子}
$$

$$
q_t:\text{当前机械臂关节状态}
$$

$$
A_t^\tau:\text{整个未来动作序列}
$$

最后形成带有完整上下文信息的 action hidden representation。

因此：

$$
\boxed{
\text{Self-Attention 是视觉、语言、机器人状态和动作之间的信息桥梁}
}
$$

### Attention Mask

如果没有 Mask，所有 token 都可以双向读取：

$$
[I,\ell,q,A^\tau]
$$

即：

$$
Image/Text
\leftrightarrow
State
\leftrightarrow
Action.
$$

π₀ 并没有这样做，而是使用 **Blockwise Causal Attention Mask**。

其规则为：

| Query        | Image + Text | $q_t$ | $A_t^\tau$ |
| ------------ | -----------: | ------: | -----------: |
| Image + Text |            ✓ |       × |            × |
| $q_t$      |            ✓ |       ✓ |            × |
| $A_t^\tau$ |            ✓ |       ✓ |            ✓ |

注意：

$$
\boxed{\text{每个 Block 内部都是 Bidirectional Attention}}
$$

所谓 causal，是指：

> 前面的 Block 不能读取后面的 Block。

#### Block 1：为什么 Image + Text 不能看 State 和 Action？

规则为：

$$
[I,\ell]
\rightarrow
[I,\ell]
$$

但：

$$
[I,\ell]\not\rightarrow q_t
$$

$$
[I,\ell]\not\rightarrow A_t^\tau.
$$

主要原因是：

$$
\boxed{
\text{尽量保持 PaliGemma 原有的预训练数据分布}
}
$$

PaliGemma 原本是在 Internet-scale image-text 数据上训练的，它只熟悉：

$$
Image+Text.
$$

它原来并没有：

$$
q_t,\qquad A_t^\tau.
$$

如果让 Image/Text Token 强烈依赖机器人状态和 noisy action，会改变预训练 VLM 原来的计算方式，增加：

$$
\text{distribution shift}.
$$

因此 π₀ 希望：

$$
Image+Text
\rightarrow
\text{保持原来的视觉语言知识}
$$

再让：

$$
Action
\leftarrow Image+Text.
$$

而不是让 noisy action 反过来不断修改 VLM 的视觉语言 representation。


#### Mask 还有一个重要作用：提高 Flow Matching 推理效率

π₀ 推理时需要进行 10 次 Flow Matching：

$$
A_t^0
\rightarrow
A_t^{0.1}
\rightarrow
A_t^{0.2}
\rightarrow
\cdots
\rightarrow
A_t^{1.0}.
$$

在这 10 次迭代过程中：

$$
I_t,\ell_t,q_t
$$

是不变的，而：

$$
A_t^\tau
$$

一直变化。

如果 Image/Text/State 都能读取 Action，那么：

$$
A_t^\tau\text{改变}
$$

就会导致：

$$
I,\ell,q_t
$$

的 hidden states 也发生变化。

那么每一次 Flow Matching 都需要重新计算整个网络。

现在通过 Mask：

$$
Image/Text\nleftarrow Action
$$

$$
q_t\nleftarrow Action
$$

因此：

$$
\boxed{
Image+Text+State 的 K/V 可以提前计算并缓存
}
$$

后面的 10 次 Flow Matching 主要重新计算 Action Token 部分即可。

所以 Mask 同时服务于：

$$
\boxed{
\text{模型结构稳定性}
+
\text{推理效率}
}
$$



#### 为什么 50 个 Action Token 可以互相看？

这里和 GPT 的自回归生成有很大区别。

GPT 是：

$$
x_1\rightarrow x_2\rightarrow x_3
$$

所以前面的 token 不能看未来 token。

但 π₀ 不是逐个预测 Action。

它一次建模的是整个 Action Chunk：

$$
A_t
=
[a_t,a_{t+1},...,a_{t+49}].
$$

Flow Matching 每次处理的也是整个：

$$
A_t^\tau.
$$

所以 Action Block 内部使用：

$$
\boxed{
\text{Full Bidirectional Self-Attention}
}
$$

即：

$$
a_t^\tau
\leftrightarrow
a_{t+1}^\tau
\leftrightarrow
\cdots
\leftrightarrow
a_{t+49}^\tau.
$$

这样可以让整个动作序列联合协调。

例如：

$$
a_t:\text{机械臂靠近物体}
$$

$$
a_{t+1}:\text{继续靠近}
$$

$$
a_{t+2}:\text{闭合夹爪}
$$

$$
a_{t+3}:\text{抬起物体}.
$$

这些动作显然不是相互独立的。

Action Self-Attention 可以学习：

$$
\boxed{
\text{动作连续性}
}
$$

$$
\boxed{
\text{时间协调关系}
}
$$

$$
\boxed{
\text{双臂/多关节协调}
}
$$

$$
\boxed{
\text{完整动作 Chunk 的整体结构}
}
$$




