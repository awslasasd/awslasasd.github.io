# Pi0-fast

## 简介

> 对于自回归形的VLA（如OpenVLA），输出的是离散的Token，再转为动作，在低频时效果优秀，但在高频时则表现不行，因此Fast提出了一种从Token到连续动作序列的一个映射网络

![image-20260904205749290](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260904205749436.png)

## 训练流程

### 获取未来真实动作块

从训练数据中取未来 \(H\) 步、每步 \(D\) 维的真实连续动作作为监督目标。

$$
\boxed{
A^*=[a_1^*,a_2^*,\dots,a_H^*]
\in\mathbb R^{H\times D}
}
$$

### Action Normalization

对每个动作维度使用训练集的 1% 和 99% quantile 归一化到约 \([-1,1]\)，减小不同动作尺度和异常值的影响。

$$
\boxed{
\tilde a_{t,d}
=
2\frac{a_{t,d}-q_d^{1\%}}
{q_d^{99\%}-q_d^{1\%}}
-1
}
$$

因此：

$$
\tilde A^*\in\mathbb R^{H\times D}
$$



###  DCT：时间域 → 频率域

对每个动作维度沿时间轴分别执行离散余弦变换，将平滑动作序列转换为低频集中的频率系数。

$$
\boxed{
C^{*(d)}
=
\operatorname{DCT}
\left(
\tilde a_{1:H,d}^*
\right)
}
$$

整体记为：

$$
\boxed{
C^*
=
\operatorname{DCT}_{time}(\tilde A^*)
\in\mathbb R^{D\times H}
}
$$


###  DCT Coefficient Quantization

将 DCT 系数乘以尺度系数 \(\gamma\) 后取整，使大量很小的高频系数变成 0。

$$
\boxed{
\bar C^*
=
\operatorname{round}
\left(
\gamma C^*
\right)
}
$$

其中：

$$
\bar C^*\in\mathbb Z^{D\times H}
$$

论文单数据集实验采用：

$$
\gamma=10
$$

。



###  Low-frequency-first Flatten

将二维频率系数矩阵按照“所有动作维度的低频系数优先”的顺序展开成一维整数序列。

$$
\boxed{
u^*
=
\operatorname{Flatten}_{LF}
(\bar C^*)
}
$$

其中顺序类似：

$$
[
\bar C_1^{(1)},
\bar C_1^{(2)},
\dots,
\bar C_1^{(D)},
\bar C_2^{(1)},
\dots
]
$$

且：

$$
u^*\in\mathbb Z^{HD}
$$

。


### BPE Compression

利用 BPE 将大量重复的整数模式和零系数组合压缩为较短的离散 FAST Action Token 序列。

$$
\boxed{
T_{1:N}^*
=
\operatorname{BPE}_{\Phi}(u^*)
}
$$

通常：

$$
\boxed{
N\ll HD
}
$$

。



###  图像编码

相机图像通过预训练视觉编码器转换成视觉 Token，作为 VLM 的条件输入。

$$
\boxed{
X_I=f_{\mathrm{vision}}(I)
}
$$

。


###  语言编码

自然语言任务指令通过语言 tokenizer 转换为 Text Tokens。

$$
\boxed{
X_L
=
\operatorname{Tokenizer}_{text}(L)
}
$$



### Robot State 编码

机器人 proprioceptive state 使用简单的 256-bin discretization，并作为文本输入序列的一部分输入 VLM。

$$
\boxed{
X_s
=
\operatorname{Tokenizer}
\left(
Q_{256}(s)
\right)
}
$$



###  构造 VLM 条件输入

视觉、语言和机器人状态共同构成 π0-FAST 自回归 Transformer 的条件。

$$
\boxed{
o=(X_I,X_L,X_s)
}
$$


###  Autoregressive Action Token Prediction

VLM 根据当前观测和之前的真实 Action Tokens，逐个预测下一个 FAST Action Token。

$$
\boxed{
p_\theta
(T_i^*\mid o,T_{<i}^*)
}
$$

因此整个 Token 序列的概率为：

$$
\boxed{
p_\theta(T_{1:N}^*|o)
=
\prod_{i=1}^{N}
p_\theta(T_i^*|o,T_{<i}^*)
}
$$


###  Next-token Cross-Entropy Loss

利用真实 FAST Action Token 对自回归 Transformer 进行 next-token prediction 训练。

$$
\boxed{
\mathcal L_{\mathrm{AR}}
=
-\sum_{i=1}^{N}
\log
p_\theta
\left(
T_i^*
\mid
o,T_{<i}^*
\right)
}
$$


## 推理流程

推理时最大的区别是：

$$
\boxed{
\text{不存在 Ground Truth Future Action}
}
$$

所以不会先做：

$$
A^*\rightarrow FAST
$$

而是让 VLM **自己生成 FAST Tokens**。

###  获取当前条件输入

当前图像、语言指令和机器人状态构成策略的观测条件。

$$
\boxed{
o=(I,L,s)
}
$$


###  Image Encoding

当前相机图像通过视觉编码器转换为 Image Tokens。

$$
\boxed{
X_I=f_{\mathrm{vision}}(I)
}
$$


###  Language Encoding

语言指令被 tokenizer 转换成 Text Tokens。

$$
\boxed{
X_L
=
\operatorname{Tokenizer}_{text}(L)
}
$$



###  State Encoding

当前 proprioceptive state 经过离散化和 tokenization 得到 State Tokens。

$$
\boxed{
X_s
=
\operatorname{Tokenizer}
(Q_{256}(s))
}
$$



###  Autoregressive FAST Token Generation

π0-FAST 根据当前条件逐个生成未来 Action Tokens。

$$
\boxed{
\hat T_i
\sim
p_\theta
(
T_i
\mid
o,\hat T_{<i}
)
}
$$

若使用 greedy decoding，则：

$$
\boxed{
\hat T_i
=
\arg\max_T
p_\theta
(
T\mid o,\hat T_{<i}
)
}
$$


### 得到完整 FAST Token 序列

重复自回归生成直到得到完整的一段 FAST Action Tokens。

$$
\boxed{
\hat T_{1:N}
=
[
\hat T_1,\hat T_2,\dots,\hat T_N
]
}
$$


###  BPE Decode

通过 BPE 逆解码，将少量 FAST Tokens 恢复成原始量化频率系数的一维整数序列。

$$
\boxed{
\hat u
=
\operatorname{BPE}_{\Phi}^{-1}
(
\hat T_{1:N}
)
}
$$

其中：

$$
\hat u\in\mathbb Z^{HD}
$$

。



###  Unflatten

将一维整数序列按照训练时相反的顺序重新组成 \(D\times H\) 的频率系数矩阵。

$$
\boxed{
\hat{\bar C}
=
\operatorname{Unflatten}_{LF}
(\hat u)
}
$$

其中：

$$
\hat{\bar C}
\in
\mathbb Z^{D\times H}
$$

。


###  Dequantization

将整数频率系数除以量化尺度 \(\gamma\)，近似恢复连续 DCT 系数。

$$
\boxed{
\hat C
=
\frac{\hat{\bar C}}{\gamma}
}
$$


###  Inverse DCT

对每个动作维度执行 IDCT，将频率域表示重新恢复为时间域的连续动作序列。

$$
\boxed{
\hat{\tilde A}
=
\operatorname{IDCT}_{time}
(
\hat C
)
}
$$

因此：

$$
\hat{\tilde A}
\in
\mathbb R^{H\times D}
$$

。



###  Action Denormalization

将归一化动作恢复到真实机器人的控制量范围。

$$
\boxed{
\hat a_{t,d}
=
\frac{\hat{\tilde a}_{t,d}+1}{2}
\left(
q_d^{99\%}-q_d^{1\%}
\right)
+
q_d^{1\%}
}
$$



### 得到最终 Continuous Action Chunk

最终得到未来 \(H\) 步的连续机器人动作，供机器人控制器执行。

$$
\boxed{
\hat A
=
[
\hat a_1,\hat a_2,\dots,\hat a_H
]
\in
\mathbb R^{H\times D}
}
$$



## 最后压缩成两条总公式

### 训练

首先把真实动作转换成 FAST Token：

$$
\boxed{
T^*
=
\operatorname{BPE}_{\Phi}
\left[
\operatorname{Flatten}
\left(
\operatorname{round}
\left[
\gamma
\operatorname{DCT}
(
\operatorname{Norm}(A^*)
)
\right]
\right)
\right]
}
$$

然后训练 VLM：

$$
\boxed{
\mathcal L
=
-\sum_i
\log
p_\theta
(
T_i^*
\mid
I,L,s,T_{<i}^*
)
}
$$



### 推理

VLM 首先生成：

$$
\boxed{
\hat T_{1:N}
\sim
p_\theta
(
T_{1:N}\mid I,L,s
)
}
$$

然后 FAST 逆变换：

$$
\boxed{
\hat A
=
\operatorname{Denorm}
\left[
\operatorname{IDCT}
\left(
\frac{
\operatorname{Unflatten}
[
\operatorname{BPE}_{\Phi}^{-1}
(
\hat T_{1:N}
)
]
}{\gamma}
\right)
\right]
}
$$

所以最适合记在笔记最上面的就是：

$$
\boxed{
\textbf{训练：}
A^*
\xrightarrow{FAST}
T^*
\xrightarrow{\text{监督}}
VLM
}
$$

$$
\boxed{
\textbf{推理：}
(I,L,s)
\xrightarrow{VLM}
\hat T
\xrightarrow{FAST^{-1}}
\hat A
}
$$

这两条就是 π0-FAST 从数据变化角度最核心的训练/推理逻辑。
