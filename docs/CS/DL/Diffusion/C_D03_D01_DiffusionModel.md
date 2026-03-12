# Diffusion Model

## 总览

DDPM（Denoising Diffusion Probabilistic Models）的核心是两件事：

- **前向扩散**：把真实数据逐步加噪，最后变成高斯噪声。
- **反向扩散**：训练模型逐步去噪，从噪声还原出数据。

主线可以概括为：`forward -> reverse -> optimization(ELBO) -> training -> sampling`。

前向扩散是为了反向扩散训练Noise Predictor作为训练数据

## 前向扩散（Forward Process）

![image-20260311165250366](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111652487.png)

设数据为 $x_0$，时间步 $t=1,\dots,T$，噪声调度为 $\beta_t \in (0,1)$，并定义：
$$
\alpha_t = 1-\beta_t,\quad \bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

前向一步马尔可夫过程：

$$
q(x_t|x_{t-1}) = \mathcal{N}\left(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I\right)
$$

前向闭式：

$$
q(x_t|x_0)=\mathcal{N}\left(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right)
$$

等价重参数化：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

![image-20260311184413574](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111844710.png)



!!! note "直观理解"
    当 $t$ 增大时，$\bar{\alpha}_t$ 变小，图像信号衰减、噪声占比上升，最终接近纯高斯噪声。



## 反向扩散（Reverse Process）

这里面Denoise的模块和参数是完全一样的，但是噪声的模糊程度是不一样的，因此引入了一个新的参数step(Time Embedding)。

![image-20260311185003513](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111850660.png)

上面是一个图简介，但实际上并不是简单的加噪声，详细的数学推导如下

![image-20260311194855092](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111948217.png)



真实后验 $q(x_{t-1}|x_t)$ 直接求很难，因此用神经网络近似：
$$
p_\theta(x_{t-1}|x_t)=\mathcal{N}\left(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I\right)
$$

DDPM常用“预测噪声”参数化：网络输出 $\epsilon_\theta(x_t,t)$，再由它构造均值：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)
$$

采样时（从 $x_T\sim\mathcal{N}(0,I)$ 开始）：

$$
x_{t-1}=\mu_\theta(x_t,t)+\sigma_t z,\quad z\sim\mathcal{N}(0,I)
$$

当 $t=1$ 时通常不再加随机噪声（令 $z=0$）。


## 训练目标与 ELBO

最大似然目标：

$$
\max_\theta \log p_\theta(x_0)
$$

通过变分推导可得到 ELBO（课件中的 optimization view）：

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_q\left[
D_{\mathrm{KL}}\!\left(q(x_T|x_0)\|p(x_T)\right)
+\sum_{t=2}^{T}D_{\mathrm{KL}}\!\left(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t)\right)
-\log p_\theta(x_0|x_1)
\right]
$$

实践中常用简化损失（MSE 预测噪声）：

$$
\mathcal{L}_{\text{simple}}=
\mathbb{E}_{t,x_0,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\ t\right)\right\|_2^2\right]
$$

!!! question "为什么通常预测噪声而不是直接预测 $x_0$？"
    噪声在不同时间步的统计形式更一致，目标尺度更稳定，训练通常更容易收敛。



## 方差设置

反向过程的 $\sigma_t^2$ 常见策略：

- **Fixed-small**：固定较小方差
- **Fixed-large（DDPM）**：固定较大方差
- **Hybrid（IDDPM）**：部分学习、部分固定
- **Analytic/Optimal**：解析或近似最优设定

方差设置会影响生成质量、采样稳定性和速度。



## 训练与采样流程

**训练**

1. 采样真实样本 $x_0$、时间步 $t$、高斯噪声 $\epsilon$  
2. 用闭式公式构造 $x_t$  
3. 输入 $(x_t,t)$ 到 UNet，输出 $\epsilon_\theta$  
4. 用 $\mathcal{L}_{\text{simple}}$ 回传更新参数

**采样**

1. 从 $x_T\sim\mathcal{N}(0,I)$ 开始  
2. 对 $t=T,T-1,\dots,1$ 迭代去噪得到 $x_{t-1}$  
3. 最终得到生成样本 $x_0$


## 总结

DDPM 的本质是：**先学会“加噪数据的统计规律”，再用神经网络学习其逆过程，实现从纯噪声到数据分布的逐步生成**。

