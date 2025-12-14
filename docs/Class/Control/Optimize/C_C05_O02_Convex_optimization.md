# 凸优化

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=530498371&bvid=BV16u411b7Es&cid=1183243887&p=1&autoplay=0" width="640" height="480" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

速通版本，宝藏视频

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=582732000&bvid=BV1S64y1u7Ne&cid=174623577&p=1&autoplay=0" width="640" height="480" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

## 基本概念

### 直线与线段

$\theta x_{1} + (1-\theta) x_{2}$

- 当 $\theta \in \mathbb{R}$时，为直线
- 当 $\theta \in [0,1]$时，为线段

### 放射集


- **仿射集**：$\forall x_1, x_2 \in A, \forall \theta \in \mathbb{R}$，满足$\theta x_1+(1-\theta)x_2 \in A$，那么A是放射集
>  注：对仿射组合运算封闭的集合

- **仿射(Affine)组合**：$\theta_1 x_1+…+\theta_k x_k$，满足$\theta_1+…+\theta_k=1$，$\forall x_1,.., x_k \in A$

>  注：如果A是放射集，那么放射组会也是放射集($\theta_1 x_1+…+\theta_k x_k \in A$)

!!! note "$\theta_1 x_1+ \theta_2 x_2$，$\forall x_1, x_2 \in A, \forall \theta_i \in \mathbb{R}$"
    如果$x_1$和$x_2$不共线，那$\theta_1 x_1+ \theta_2 x_2$扩展为一个平面

- **仿射包(Affine hull)**：集合$S$中点$x_i$的所有仿射组合构成的集合
  
  $$
  \text{aff } S = \left\{ x = \theta_1 x_1 + \dots + \theta_m x_m \mid \sum_{i=1}^m \theta_i = 1, x_i \in S, i=1,\dots,m \right\}
  $$

- **性质**：仿射包为包含$S$的最小仿射集

- 如果S是放射集，那aff S是放射集S本身

###  线性子空间

取$\forall x_0 \in \text{Aff}$，构造$V = \text{Aff} - x_0 = \left\{ x - x_0 \mid x \in \text{Aff} \right\}$（$V$含零元素）

对**线性组合运算封闭**的集合（$V$对加法和数乘运算封闭），不需要约束$\theta_1+…+\theta_k=1$

- 含有零元素
- 齐次线性方程组$Ax=0$的解集$\Leftrightarrow$线性子空间
- 几何上是过原点的超平面，依然是放射集

!!! note "线性子空间与放射集的关系"
    - 仿射集是子空间的平移，也称**仿射子空间**
        - 线性方程组$Ax=b$的解集$\Leftrightarrow$仿射集
        - 几何上是经平移的超平面，称为**仿射超平面** 

### 凸集

**凸集**：$\theta x_1 + (1-\theta) x_2 \in C$，其中$0 \leq \theta \leq 1$，$\forall x_1, x_2 \in C$

> 注：对凸组合运算封闭的集合,即$x_1$,$x_2$两点的线段还$\in C$

**凸(convex)组合**：$\theta_1 x_1 + \dots + \theta_k x_k$，满足$\theta_1 + \dots + \theta_k = 1$且$\theta_i \geq 0$

**凸包（Convex hull）**：把一个非凸集合，通过凸组合变为凸集 
$$
\text{conv } S = \left\{ x = \theta_1 x_1 + \dots + \theta_m x_m \mid \sum_{i=1}^m \theta_i = 1, \theta_i \geq 0, x_i \in S, i=1,\dots,m \right\}
$$

  - 示例：离散点凸包、扇形凸包
  - **性质**：凸包为包含$S$的最小凸集


**$C$严格凸**：$\forall x_1, x_2 \in C$，$\theta \in (0, 1) \implies \theta x_1 + (1-\theta) x_2 \in \text{relint}C$


要不要我帮你整理一份**凸集相关概念的核心条件总结表**？

!!! note "放射集和凸集"
    如果C是放射集，那他一定是凸集
    区别：放射集$\theta$无非负要求

### 凸锥

**锥(cone)**：$\forall x \in S$和$\theta \geq 0$，满足$\theta x \in S$
![image-20251214133051258](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512141330383.png)



!!! note "对于固定的$x_1$,$\theta_1 x_1$是一个射线，又 $\forall x \in S$，因此是无数条射线的集合，形成了锥"

**锥组合**：$\theta_1 x_1 + \theta_2 x_2$，其中$\theta_1, \theta_2 \geq 0$

>  非负线性组合，因为AX=0的解X就是一个锥


**凸锥**：$\theta_1, \theta_2 \geq 0$，$\forall x_1, x_2 \in S$, 满足$\theta_1 x_1 + \theta_2 x_2 \in S$

> 注：对锥组合运算封闭的集合

!!! note "$\theta_1 x_1$和$\theta_2 x_2$分别为两个射线上的矢量，矢量叠加一定在这两个射线中间"



性质：
  - 凸锥是包含零元素的凸集
  - 不是所有的锥都是凸锥！

**锥包（Conic hull）**：

$$
  \text{conic } S = \left\{ x = \theta_1 x_1 + \dots + \theta_m x_m \mid \theta_i \geq 0, x_i \in S, i=1,\dots,m \right\}
$$
  注：锥包是包含$S$的最小凸锥



