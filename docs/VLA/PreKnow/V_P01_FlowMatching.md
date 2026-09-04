# FlowMatching算法介绍

## 基本定义

- 轨迹

![image-20260903214718335](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903214718385.png)

- 向量场

![image-20260903214822973](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903214823023.png)

>  向量场的含义——在某一时刻内，多维空间下该点上的速度

- 流

![image-20260903214928853](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903214928910.png)

!!! note "流和向量场之间的关系"
    是可以互相推到出来的



## FlowMatching的基本思想

![image-20260903214535119](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903214535293.png)

**训练的思路**

1. 用神经网络来学习一个向量场$u_t^\theta$。 
2.  从已知分布$p_{init}$里采样一个$x_0$。 
3.  在NN学到的向量场$u_t^\theta$指导下运动到$x_1$。 
4. 这个流$\psi$初始点$x_0$满足$p_{init}$，终点$x_1$满足$p_{data}$。

而神经网络需要通过`loss`来进行反向传播，因此接下来就涉及到了`loss`是怎么计算的

$$
L(\theta)=\left\|u_t^\theta(x_t)-u_t^{target}(x_t)\right\|^2 
$$

那么，我们如何获得 $u_t^{target}(x_t)?$

## 理论推导

### 概率路径

#### 条件概率路径

概率密度的分布随时间变化公式

![image-20260903215610918](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903215610999.png)

#### 边缘概率路径

![image-20260903215725427](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903215725505.png)

### 求解目标向量场

#### 目标流构建

构造一个flow:

$$
\psi_t^{target}(x_0|z) = \alpha_t z + \beta_t x_0
$$

其中$\alpha_t = t \quad \beta_t = 1-t$

当 $t=0$ 时：

$$
\alpha_t = 0 \quad \beta_t = 1
$$

$$
\psi_t^{target}(x_0|z) = x_0
$$

当 $t=1$ 时：

$$
\alpha_t = 1 \quad \beta_t = 0
$$

$$
\psi_t^{target}(x_0|z) = z
$$

因此我们构造的这个目标流满足我们的构造要求

#### 求解目标向量场

在构建符合要求的流后，根据流与向量场之间的关系计算条件向量场：

$$
\frac{d\psi_t(x_0|z)}{dt}=u_t\big(\psi_t(x_0|z)\big|z\big)
$$

$$
\frac{d\alpha_t}{dt}z+\frac{d\beta_t}{dt}x_0=u_t(x_t|z)
$$

$$
\frac{d\alpha_t}{dt}=1
$$

$$
\frac{d\beta_t}{dt}=-1
$$

$$
z-x_0=u_t(x_t|z)
$$


其中$x_0$是满足正态分布的，因此可以得到下面的结果


$$
u_t(x_t|z)=z-\varepsilon,\quad \varepsilon\sim \mathcal{N}(0,I)
$$

$$
\boxed{u_t^{target}(x_t|z)=z-\varepsilon}
$$



![image-20260903220507411](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903220507482.png)

从上面的推到中不难看出，当目标图片`z`和采样噪声$\varepsilon$ 确定，那么他的速度向量就是确定的


!!! question "能否用条件向量场作为Label"
    ![image-20260903220832153](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260903220832197.png)
    是可以的，直观上去理解，通过给定大量的不同的照片z，可以让这个条件向量场逐渐的变为无条件向量场，因此是可以的。推导见下文。


!!! question "FlowMatching 和DDPM的区别"   
    1.FlowMatching的流是线性的，$x_t=(1-t)\,x_0 + t\,z,\quad t\in[0,1]$,路径是**确定性直线插值**，没有逐步加噪、没有累积高斯噪声方差、没有 $\bar\alpha$ 累乘项。
    
    2.DDPM他是马尔可夫决策过程，$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，虽然$\beta$是从0.0001线性增加到0.02，但是由于开根号和累乘项，他本质是来学习去噪的噪声的(本来是学习去噪的高斯分布，但是最终化简发现只有噪声一个变量)

