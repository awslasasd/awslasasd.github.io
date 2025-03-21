---
comments: true
---


# 机器人建模

???+note "课程资源"


    === "作业资源"
       [98往年答案](https://www.cc98.org/topic/6115926)<br>
    
    === "实验资料"
       主要内容：基于Coppeliasim仿真平台进行的实验。  


##  绪论

### 常见机械臂几何结构

![](https://philfan-pic.oss-cn-beijing.aliyuncs.com/img/20250217105608273.png)


## 空间描述与变换

### 坐标系与向量

**笛卡尔直角坐标系**

- 交于原点的三条不共面的数轴（常称x轴、y轴和z轴）构成空间的仿射坐标系
-  三条数轴（主轴）上度量单位相等的仿射坐标系称为空间笛卡尔坐标系
- 下面所有坐标系均**采用直角右手坐标系**



**向量的表示**

向量 $r_{O_AD}$，将它分别向 $\hat{X}_A, \hat{Y}_A, \hat{Z}_A$ 作投影，得到3个向量 $d_x \hat{X}_A, d_y \hat{Y}_A, d_z \hat{Z}_A$

$r_{O,D} = d_x \hat{X}_A + d_y \hat{Y}_A + d_z \hat{Z}_A = \left( \hat{X}_A \quad \hat{Y}_A \quad \hat{Z}_A \right) \begin{pmatrix} d_x \\ d_y \\ d_z \end{pmatrix}$

简洁表达 $^A D = \begin{pmatrix} d_x \\ d_y \\ d_z \end{pmatrix}$


=== "向量点乘"
    内积结果是标量<br>
    两个向量 $r_{OP}$ 和 $r_{OQ}$ 的点乘(内积)可按下式计算:

    $$
    r_{OP} \cdot r_{OQ} 
    = ^AP \cdot ^AQ = ^AP^T \cdot ^AQ\\ 
    = (p_x \quad p_y \quad p_z)\begin{pmatrix}q_x\\q_y\\q_z\end{pmatrix} \\ 
    = p_xq_x + p_yq_y + p_zq_z
    $$

=== "向量叉乘"
    两个向量 $\vec{a}$ 和 $\vec{b}$ 的叉乘结果是一个新向量 $\vec{c}$:

    $$
    \vec{c} = \vec{a} \times \vec{b}= |a||b|\sin\theta
    $$
    
    方向遵循右手定则，垂直于这两个向量所在的平面。
    
    简单计算方法:
    
    - 把 $\vec{a}$ 和 $\vec{b}$ 写成下面的矩阵形式
    
    $$
    \begin{pmatrix}
    a_x & a_y & a_z & a_x & a_y & a_z \\
    b_x & b_y & b_z & b_x & b_y & b_z
    \end{pmatrix}
    $$
    
    - 去掉第一列和最后一列，剩下的3个2x2的矩阵（每次滑动1格子），计算行列式即可

​            

### 点和刚体的描述

**点**


$$
r_{O_AP} = \begin{pmatrix} \hat{X}_A & \hat{Y}_A & \hat{Z}_A \end{pmatrix}^A P
$$


**刚体**



在 $\{A\}$ 中表示出 $\{B\}$ 的姿态：


$$
\begin{pmatrix} \hat{X}_B & \hat{Y}_B & \hat{Z}_B \end{pmatrix} = \begin{pmatrix} \hat{X}_A & \hat{Y}_A & \hat{Z}_A \end{pmatrix}_B^A R
$$







| 表示方法 | 核心思想 | 公式 | 缺点 |
| --- | --- | --- | --- |
| **旋转矩阵** | 使用3x3矩阵表示三维旋转 | $\mathbf{R} = \begin{pmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{pmatrix}$ | 1. 参数多（9个），冗余<br> 2. 难以直观理解旋转过程<br> 3. 插值复杂 |
| **欧拉角** | 将旋转分解为绕三个正交轴的旋转 | $(\alpha, \beta, \gamma)$，常用ZYX顺序：$\mathbf{R} = R_z(\alpha) R_y(\beta) R_x(\gamma)$ | 易于理解和可视化<br> 但是<br> 1. 万向锁问题（奇异性）<br> 2. 不同顺序定义不唯一<br> 3. 插值不平滑 |
| **等效轴角** | 用一个单位轴和一个旋转角表示旋转 | $(\mathbf{k}, \theta)$，其中$\mathbf{k} = (k_x, k_y, k_z)$为单位向量，$\theta$为旋转角。旋转矩阵为：<br>$\mathbf{R} = \mathbf{I} + \sin\theta \mathbf{K} + (1 - \cos\theta) \mathbf{K}^2$，<br>其中$\mathbf{K} = \begin{pmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{pmatrix}$ | 1. 无法直接表示0°旋转（需特殊处理）<br>2. 插值时需注意旋转角的周期性 |
| **四元数** | 使用四维超复数表示旋转 | $q = \eta + i\varepsilon_1 + j\varepsilon_2 + k\varepsilon_3$，其中$\eta^2 + \varepsilon_1^2 + \varepsilon_2^2 + \varepsilon_3^2 = 1$。 | 参数最少（4个）避免了奇异性问题<br>1. 较难直观理解<br>2. 计算稍复杂（但比旋转矩阵简单） |



## 机器人运动学

- 使两个刚体直接接触而又能产生一定相对运动的联接称为运动副 ，机器人的运动副也称关节，连杆即指由关节所联的刚体
- **本课程中的关节仅限转动副和移动副**
- 串联机构：多个连杆通过关节以串联形式连接成首尾不封闭的机械结构


!!!note 
  为了确定末端执行器在3维空间的位置和姿态，串联机器人至少需要6个关节<br>



### 改进D-H参数

**确定坐标系的方法**

- **第1步**：确定 $Z_i$ 轴。基本原则是：$Z_i$ 轴沿关节 $i$ 的轴向。
- **第2步**：确定原点 $O_i$。基本原则是：$O_i$ 在过 $Z_i$ 和 $Z_{i+1}$ 轴的公法线上。
- **第3步**：确定 $X_i$ 轴。基本原则是：$X$ 轴沿过 $Z_i$ 和 $Z_{i+1}$ 轴的公法线方向，从 $Z_i$ 指向 $Z_{i+1}$。
- **第4步**：确定 $Y_i$ 轴。基本原则是：$Y_i = Z_i \times X_i$，使坐标系为右手坐标系。

**DH参数的定义**

1. **杆件长度** $a_i$，定义为从 $Z_{i-1}$ 到 $Z_i$ 的距离，沿 $X_{i-1}$ 轴指向为正。
2. **杆件扭角** $\alpha_i$，定义为从 $Z_{i-1}$ 到 $Z_i$ 的转角。绕 $X_{i-1}$ 轴正向转动为正。
3. **关节距离** $d_i$，定义为从 $X_{i-1}$ 到 $X_i$ 的距离，沿 $Z_i$ 轴指向为正。
4. **关节转角** $\theta_i$ 定义为从 $X_{i-1}$ 到 $X_i$ 的转角，绕 $Z_i$ 轴正向转动为正。



![image-20250303144706365](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503031447461.png)





### 正运动学

**相邻连杆的坐标系变换**

{*i*-1}经四步变换成为{*i*}：

- 沿联体x轴平移$a_{i-1}$
- 沿联体x轴旋转$\alpha_{i-1}$
- 沿联体z轴平移$d_i$
- 沿联体z轴旋转$\theta_i$

$$
T = \begin{pmatrix}
1 & 0 & 0 & a_{i-1} \\
0 & \cos\alpha_{i-1} & -\sin\alpha_{i-1} & 0 \\
0 & \sin\alpha_{i-1} & \cos\alpha_{i-1} & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
\cos\theta_i & -\sin\theta_i & 0 & 0 \\
\sin\theta_i & \cos\theta_i & 0 & 0 \\
0 & 0 & 1 & d_i \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

经过矩阵乘法后，得到的结果为：

$$
= \begin{pmatrix}
\cos\theta_i & -\sin\theta_i & 0 & a_{i-1} \\
\sin\theta_i\cos\alpha_{i-1} & \cos\theta_i\cos\alpha_{i-1} & -\sin\alpha_{i-1} & -\sin\alpha_{i-1}d_i \\
\sin\theta_i\sin\alpha_{i-1} & \cos\theta_i\sin\alpha_{i-1} & \cos\alpha_{i-1} & \cos\alpha_{i-1}d_i \\
0 & 0 & 0 & 1
\end{pmatrix}
$$


## 逆运动学


### 数值解

**牛顿-拉夫逊迭代**

给定N自由度机器人期望的齐次变换矩阵 $T^d$，求关节变量 $\Phi = [\phi_1 \quad \phi_2 \quad \cdots \quad \phi_N]^T$ 使得：

$$
{}^0 T = {}^0 T_1(\phi_1) \cdot {}^1 T_2(\phi_2) \cdots \cdots {}^N-1 T(\phi_N) = T^d
$$

记末端执行器的位姿为 $X(\Phi)$，期望位姿为 $X^d$，则问题转化为求关节向量 $\Phi$ 满足 $X(\Phi) = X^d$。利用牛顿-拉夫逊（Newton-Raphson）法可以迭代求解上述方程。

记期望的关节变量为 $\Phi^d$，即 $X(\Phi^d) = X^d$，牛顿-拉夫逊法是从一个猜测的初始关节变量 $\Phi^0$ 开始，迭代计算 $\Phi^k$，最终逼近 $\Phi^d$。过程中需要利用末端位姿关于关节变量的微分，这正是分析雅可比矩阵。

记 $\delta \Phi^k = \Phi^d - \Phi^k$，$\delta X(\Phi^k) = X(\Phi^d) - X(\Phi^k)$，则由一阶泰勒展开近似得到：

$$
X(\Phi^d) = X(\Phi^k) + \frac{\partial X}{\partial \Phi}(\Phi^k) \delta \Phi^k + O((\delta \Phi^k)^2)
$$

$$
\delta X(\Phi^k) = \frac{\partial X}{\partial \Phi}(\Phi^k) \delta \Phi^k = J_a(\Phi^k) \delta \Phi^k
$$

由此得到迭代计算式：

$$
\Phi^{k+1} = \Phi^k + J_a^{-1}(\Phi^k) \delta X(\Phi^k)
$$


## 微分运动学与静力学

### 微分运动学

#### 定义

##### 符号表示

|       符号       |            含义             |              注意               |
| :--------------: | :-------------------------: | :-----------------------------: |
|     $^AV_Q$      |      A坐标系下Q的速度       |                                 |
|   $^B(^AV_Q)$    |   在$B'$坐标系下的Q的速度   |      $^B(^AV_Q) \ne ^BV_Q$      |
|  $\upsilon _c$   | 世界坐标系下C坐标原点的速度 |   $\upsilon _c = ^UV_{CORG}$    |
|   $^A\Omega_Q$   |      A坐标系下Q的速度       |                                 |
| $^B(^A\Omega_Q)$ |   在$B'$坐标系下的Q的速度   | $^B(^A\Omega_Q) \ne ^B\Omega_Q$ |
|   $\omega _c$    | 世界坐标系下C坐标原点的速度 |  $\omega _c = ^U\Omega_{CORG}$  |

!!! attention 注意符号
    $^A\upsilon _c = {}^A_U\mathbf{R} \upsilon _c = {}^A_U\mathbf{R} ^UV_{CORG} \ne ^AV_{CORG}$<br>
    $^A\omega _c = {}^A_U\mathbf{R} \omega _c = {}^A_U\mathbf{R} ^U\Omega _{CORG} \ne ^A \Omega _{CORG}$<br>
    $B'$坐标系其实是与A坐标系原点相同，姿态与B坐标系相同的一个新的坐标系<br>

##### 矩阵定义

$$
S = \dot{R}R^\top = \dot{R}R^{-1}
$$

其中S为反对称矩阵，具有以下性质：

$$
S + S^\top = 0_n
$$

**三维向量与三维反对称矩阵的关系**

针对三维向量 $P = \begin{pmatrix} p_x \\ p_y \\ p_z \end{pmatrix}$，记由 $P$ 生成的三维反对称矩阵为 $P^\wedge$，则有：

$$
P^\wedge = \begin{pmatrix} p_x \\ p_y \\ p_z \end{pmatrix}^\wedge = \begin{pmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{pmatrix}
$$

三维反对称矩阵与三维向量是一一对应的，记 $P^\wedge$ 对应的三维向量为 $\left( P^\wedge \right)^\vee = P$，则有：

$$
P = \begin{pmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{pmatrix}^\vee = \begin{pmatrix} p_x \\ p_y \\ p_z \end{pmatrix}
$$



##### 线速度向量

若 \( {}^B\mathbf{Q} \) 是描述某个点的位置矢量，该点关于{B}的速度是 \( {}^B\mathbf{V}_Q \)。

$$
{}^B\mathbf{V}_Q = \frac{d}{dt} {}^B\mathbf{Q} = \lim_{\Delta t \to 0} \frac{{}^B\mathbf{Q}(t + \Delta t) - {}^B\mathbf{Q}(t)}{\Delta t}
$$

速度矢量 \( ^A({}^B\mathbf{V}_Q ) \) 

$$
{}^A({}^B\mathbf{V}_Q) = {}^A_B\mathbf{R} {}^B\mathbf{V}_Q = \frac{d}{dt} {}^A_B\mathbf{Q} = \lim_{\Delta t \to 0} {}^A_B\mathbf{R}(t) \left( \frac{{}^B\mathbf{Q}(t + \Delta t) - {}^B\mathbf{Q}(t)}{\Delta t} \right)
$$

当两个上标相同时，无需给出外层上标，即：

$$
^B(^BV_Q) = ^BV_Q
$$

!!! note 需要注意，\( {}^A({}^B\mathbf{V}_Q) \) 不同于 \( {}^A\mathbf{V}_Q \)

    $$
    \begin{align*}
    {}^A\mathbf{V}_Q &= \lim_{\Delta t \to 0} \frac{{}^A\mathbf{Q}(t + \Delta t) - {}^A\mathbf{Q}(t)}{\Delta t}\\
    &= \lim_{\Delta t \to 0} \frac{{}^A\mathbf{P}_{BORG}(t + \Delta t) + {}^A_B\mathbf{R}(t + \Delta t) {}^B\mathbf{Q}(t + \Delta t) - {}^A\mathbf{P}_{BORG}(t) - {}^A_B\mathbf{R}(t) {}^B\mathbf{Q}(t)}{\Delta t}\\
    &= ^AV_{BORG} + {}^A_B\mathbf{\dot{R}}{}^B\mathbf{Q}+^A_B\mathbf{R}  {}^B\mathbf{V}_Q
    \end{align*}
    $$

#### 角速度向量

刚体的定点转动：刚体绕体内或其外延部分的一固定点旋转

!!! note "定点转动不同于定轴转动"
    | 区别点 | 定点转动 | 定轴转动 |
    | --- | --- | --- |
    | **转轴性质** | 转轴通过一个固定点，但转轴的方向在空间中会随时间改变 | 转轴在空间中的位置和方向始终保持不变 |
    | **角速度特性** | 角速度的大小和方向都是时间的函数，角速度矢量在空间中不断变化 | 角速度的方向始终沿着固定轴，只有大小可以随时间变化 |
    | **运动复杂程度** | 运动复杂，需要考虑多个方向的转动，通常用欧拉角描述 | 运动简单，仅在一个固定平面内进行 |
    | **自由度** | 3个自由度 | 1个自由度 |
    | **运动描述** | 可以看作绕瞬时轴的定轴转动，但瞬时轴会不断变化 | 始终绕一个固定的轴转动 |
    | **典型实例** | 陀螺、回转罗盘 | 门的转动、风车的旋转 |

由理论力学知：刚体（其联体坐标系为 $\{B\}$）在参考坐标系 $\{A\}$ 中的任何运动都可以分解为：

- 点 ${}^A O_B$ 的运动
- 刚体绕 ${}^A O_B$ 的定点转动

由理论力学知：

- 在任一瞬间，$\{B\}$ 在 $\{A\}$ 中的定点转动可以看作是绕瞬时转动轴（简称瞬轴，瞬轴上的每个点在该瞬时相对于 $\{A\}$ 的速度为零）的转动。
- 瞬轴的位置可随时间变化，但原点始终在瞬轴上。

在 $\{A\}$ 中描述 $\{B\}$ 的定点转动可用角速度向量 ${}^A\Omega_B$ 表示：

- ${}^A\Omega_B$ 的方向是瞬轴在 $\{A\}$ 中的方向；
- ${}^A\Omega_B$ 的大小表示在 $\{A\}$ 中 $\{B\}$ 绕瞬轴的旋转速度。




#### 线速度变化

**纯平移的线速度变化**

$$
^A\mathbf{V}_Q = {^A\mathbf{V}_{BORG}} + {^A_B}\mathbf{R}{^B\mathbf{V}_Q} 
$$

**一般运动的线速度变化**

!!! note "公式推导"
    ${}^A_B \dot{R} = {}^A_B S \, {}^A_B R = {}^A \mathbf{\Omega}_B \times {}^A_B R $<br>
    ${}^A_B S = {}^A_B \dot{R} \, {}^A_B R^{-1} = {}^A_B \dot{R} \, {}^A_B R^\top$


$$
^A\mathbf{V}_Q = {^A\mathbf{V}_{BORG}} + {}^A_B\mathbf{\dot{R}}{}^B\mathbf{Q} +^A_B\mathbf{R}  {}^B\mathbf{V}_Q\\
= {^A\mathbf{V}_{BORG}} + {^A_B}\mathbf{R}{^B\mathbf{V}_Q} + {^A\mathbf{\Omega}_B} \times {^A_B}\mathbf{R}{^B\mathbf{Q}}
$$

$Q$ 点对 ${A}$ 的线速度为坐标系 ${B}$ 原点的线速度、$Q$ 点在坐标系 ${B}$ 中的线速度和坐标系 ${B}$ 针对坐标系 ${A}$ 旋转形成的 $Q$ 点切向线速度三者的向量合成

#### 角速度变化

在参考坐标系 $\{A\}$ 中，坐标系 $\{C\}$ 的角速度 ${}^A\Omega_C$ 可以表示为：

$$
{}^A\Omega_C = {}^A\Omega_B + {}^A_B R \, {}^B\Omega_C
$$

在同一坐标系中，角速度可以相加

#### 速度传递

!!! note "规律"
    - 前一个关节的线速度和角速度都要转换到后一个关节上面
    - 转动型关节会增加角速度的项，平动型关节会增加线速度的项。

**转动型关节**

角速度：连杆 i+1 针对世界坐标系角速度在{i+1}坐标系的表示

$$
^{i+1}\!\omega_{i+1} = ^{i+1}_i\!R {^i\omega_i} + \dot{\theta}_{i+1} {}^{i+1}\!\hat{Z}_{i+1}\\
$$

> 其中，$\hat{Z}_{i+1}$ 是轴 $i+1$在 ${i+1}$ 中的表示；${\theta}_{i+1}$ 是转动型关节 $i+1$ 的关节转速。

线速度：连杆i+1 针对世界坐标系线速度在{i+1}坐标系的表示

$$
{}^{i+1}\!v_{i+1} = {^{i+1}_iR} (^iv_i + ^i\omega_i \times {^iP_{i+1}})
$$

> 连杆的长度隐含在了 $^iP_{i+1}$ 叉乘项当中

**平动型**

角速度：没有关节转动对下一个关节角速度的影响

$$
^{i+1}\omega_{i+1} = _i^{i+1}R {^i\omega_i}
$$


线速度：要加一个在轴线上的速度

$$
^{i+1}v_{i+1} = _i^{i+1}R (^iv_i + ^i\omega_i \times ^iP_{i+1}) + \dot{d_{i+1}} \hat{Z}_{i+1}
$$

**向外迭代法**

知道了这样的变换方法，就可以从连杆0，变换到连杆N，一个个地计算速度和角速度

### 雅可比矩阵

雅可比矩阵用于描述函数的输入变量和输出变量之间的线性关系。对于函数 $F(X)$，其雅可比矩阵 $J(X)$ 定义为：

$$
\delta Y = \frac{\partial F}{\partial X} \delta X = J(X) \delta X
$$

在动态系统中，输出变量 $Y$ 的时间导数 $\dot{Y}$ 可以表示为：

$$
\dot{Y} = J(X) \dot{X}
$$

!!! tip "注意"
    雅可比矩阵可看成是X中的速度向Y中速度的映射。<br>
    $J(X)$是一个时变的线性变换。<br>

##### 几何雅可比矩阵

几何雅可比矩阵描述了操作臂的关节速度 $\dot{\theta}$ 与末端速度（包括线速度和角速度）$v = \begin{pmatrix} v \\ \omega \end{pmatrix}$ 之间的映射关系矩阵 $J(\theta)$。

$$
v = \begin{pmatrix} v \\ \omega \end{pmatrix} = J(\theta) \dot{\theta}
$$

**前述向外迭代法计算机械臂末端速度的算法本质上是计算操作臂几何雅
可比矩阵的方法之一。**

!!! note "机械臂末端相对于基坐标系的角速度向量 \(\omega = (\omega_x, \omega_y, \omega_z)^T\) 并不是直接通过对基坐标系下的末端姿态（例如欧拉角）进行求导得到的。"
    当我们使用欧拉角来表示机械臂末端的姿态时，欧拉角是一组描述物体在空间中方向的三个角度。然而，直接对欧拉角进行求导并不能得到正确的角速度向量，因为欧拉角之间存在耦合效应，即一个角度的变化会影响到其他角度的变化。这种耦合效应会导致直接求导得到的角速度向量不准确。

采用向量积法直接求出末端线速度和角速度，可以构造几何雅可比矩阵

假设其他关节固定不动，只有第 $i$ 个关节运动，则由此运动产生的连杆 $N$ 的线速度和角速度如下：

若第 $i$ 个关节为移动关节
- 线速度 $v_N^{(i)} = \dot{d}_i \hat{Z}_i$
- 角速度 $\omega_N^{(i)} = 0$

若第 $i$ 个关节为转动关节
- 线速度 $v_N^{(i)} = \dot{\theta}_i \hat{Z}_i \times (O_N - O_i)$
- 角速度 $\omega_N^{(i)} = \dot{\theta}_i \hat{Z}_i$


!!! note "下图是以$Z_2$为例的图示"
    注意角速度对末端水平速度的影响是角速度的矢量与关节位置与末端位置的矢量叉乘<br>
    ![image-20250313151031280](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503131510456.png)

末端实际线速度和角速度就是各关节造成的线速度和角速度的总和：

$$
v_N = \sum_{i=1}^{N} v_N^{(i)}, \quad \omega_N = \sum_{i=1}^{N} \omega_N^{(i)}
$$


以机械臂**每个关节均为旋转关节**为例构造雅可比矩阵，定义笛卡尔速度向量 $v_N = \begin{pmatrix} v_N \\ \omega_N \end{pmatrix} \in \mathbb{R}^6$ 和关节空间角速度向量 $\dot{\theta} = \begin{pmatrix} \dot{\theta}_1 \\ \dot{\theta}_2 \\ \vdots \\ \dot{\theta}_N \end{pmatrix} \in \mathbb{R}^N$，则有：

$$
v_N = \begin{pmatrix}
\dot{Z}_1 \times (O_N - O_1) & \dot{Z}_2 \times (O_N - O_2) & \cdots & \dot{Z}_{N-1} \times (O_N - O_{N-1}) & 0 \\
\dot{Z}_1 & \dot{Z}_2 & \cdots & \dot{Z}_{N-1} & \dot{Z}_N
\end{pmatrix} \begin{pmatrix} \dot{\theta}_1 \\ \dot{\theta}_2 \\ \vdots \\ \dot{\theta}_N \end{pmatrix}\\
= J(\theta) \dot{\theta}
$$

其中，$J(\theta) \in \mathbb{R}^{6 \times N}$ 即为雅可比矩阵。


向量积构造法是计算几何雅可比矩阵的方法之一。对于任意已知的操作臂位形，关节速度和操作臂末端速度的关系是线性的，然而这种线性关系仅仅是瞬时的，因为在下一刻，雅可比矩阵就会有微小的变化。雅可比矩阵是时变的。

**参考坐标系变换下的雅可比矩阵**

若关心 $\{i\}$ 中的笛卡尔速度向量，则有：

$$
\begin{pmatrix} \dot{v}_N \\ \dot{\omega}_N \end{pmatrix} = \begin{pmatrix} {}^i_0 R & 0 \\ 0 & {}^i_0 R \end{pmatrix} \begin{pmatrix} v_N \\ \omega_N \end{pmatrix} = \begin{pmatrix} {}^i_0 R & 0 \\ 0 & {}^i_0 R \end{pmatrix} J(\theta) \dot{\theta}
$$

可记变换后的雅可比为：

$$
{}^i J(\theta) = \begin{pmatrix} {}^i_0 R & 0 \\ 0 & {}^i_0 R \end{pmatrix} J(\theta)
$$

即：

$$
\begin{pmatrix} \dot{v}_N \\ \dot{\omega}_N \end{pmatrix} = {}^i J(\theta) \dot{\theta}
$$

!!! attention "一般来说，雅可比矩阵都会计算到末端，即最后的$\theta$值为0"
    ![image-20250314182625966](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503141826094.png)<br>
    如图中所示，计算从Frame{1}——{3},$\dot{\theta_3}=0$，${}^3v_3$和${}^3 \omega_3$均按照之前的推导计算，如下图。<br>
    ![image-20250314183019508](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503141830595.png)


### 逆微分运动学

#### 一些定义

若已知末端执行器笛卡尔空间速度 $v_N$（包括线速度和角速度），产生 $v_N$ 的各关节角速度如下计算：

$$
\dot{\theta} = J^{-1}(\theta) v_N
$$

由于对于冗余机械臂和欠驱动机械臂，雅可比矩阵不是方阵，需要考虑雅可比矩阵的伪逆（广义逆）。

若矩阵 $A$ 的维度为 $m \times n (m \neq n)$，且 $A$ 为满秩，则 $A$ 的伪逆（广义逆） $A^+$ 为：

1. 当 $m > n$ 时，$A^+$ 为左逆矩阵，$A^+ = A_{\text{left}}^{-1} = (A^T A)^{-1} A^T$，且 $A^T A$ 满秩。
2. 当 $m < n$ 时，$A^+$ 为右逆矩阵，$A^+ = A_{\text{right}}^{-1} = A^T (A A^T)^{-1}$，且 $A A^T$ 满秩。

!!! note "冗余机械臂和欠驱动机械臂"
    | 特性 | 冗余机械臂 | 欠驱动机械臂 |
    |------|-------------|--------------|
    | **定义** | 自由度（DOF）大于完成任务所需自由度的机械臂 | 控制输入少于自由度的机械臂，即驱动器数量少于关节数量 |
    | **自由度与驱动器关系** | 自由度多于任务需求，驱动器数量等于自由度 | 自由度多于驱动器数量 |
    | **应用场景** | 复杂任务，如避障、优化路径、提高灵活性 | 简化结构、降低成本，适用于抓取不规则物体等 |
    | **优点** | 提高灵活性、避障能力、优化关节力矩 | 简化结构、降低成本、减少能耗 |
    | **缺点** | 结构复杂、控制算法复杂、成本较高 | 控制难度大、运动精度受限、灵活性较差 |
    | **典型应用** | 工业机器人、空间机器人、医疗机器人 | 机器人手爪、水下机器人、服务机器人 |
    | **控制难度** | 高（需要解决冗余自由度的优化问题） | 高（需要解决欠驱动系统的运动规划和控制问题） |

#### 逆微分运动

若 $A$ 为 $m \times n$ 维矩阵，且 $A$ 为满秩，则线性方程组 $Ax = b$ 的解：

1. 当 $m > n$ 时

方程组是过定的，通常方程组无解。此时，使得 $\|Ax - b\|^2$ 最小的 $x$ 为方程的最小二乘解，由左伪逆计算：

$$
x^* = A^+ b = A_{\text{left}}^{-1} b = (A^T A)^{-1} A^T b
$$

例如：

$$
\begin{bmatrix} 1 \\ 1 \end{bmatrix} x = \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \quad x^* = \left( \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \right)^{-1} \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \end{bmatrix} = (2)^{-1} \begin{bmatrix} 2 \end{bmatrix} = \begin{bmatrix} 1 \end{bmatrix}
$$

2. 当 $m < n$ 时

方程组是欠定的，通常方程组可能存在无数个解。此时，所有解中使得 $\|x\|$ 范数最小的 $x$ 为方程的最小范数解，由右伪逆计算：

$$
x^* = A^+ b = A_{\text{right}}^{-1} b = A^T (A A^T)^{-1} b
$$

例如：

$$
\begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = 2, \quad \begin{bmatrix} x_1^* \\ x_2^* \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \left( \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right)^{-1} 2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} (2)^{-1} (2) = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$



**零空间（Null Space）**

若 $A$ 为 $m \times n$ 维矩阵，则 $A$ 的零空间为线性方程组 $Ax = 0$ 的所有解集合，记为 $N(A) = \{x \in \mathbb{R}^n : Ax = 0\}$

1. 当 $m \geq n$ 时，若 $A$ 为列满秩，$A$ 的零空间只有零向量

$$
A = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad x = 0
$$

2. 当 $m < n$ 时，若 $A$ 为行满秩，$A$ 的零空间中的向量为

$$
\tilde{x} = (I - A^+ A)x = (I - A_{\text{right}}^{-1} A)x = (I - A^T (A A^T)^{-1} A)x
$$

其中，$x$ 为任意 $n$ 维向量

例如：

$$
A = \begin{bmatrix} 1 & 1 \end{bmatrix}, \quad \tilde{x} = \left( \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix} - \begin{bmatrix}1/2 \\ 1/2 \end{bmatrix} \begin{bmatrix}1 & 1 \end{bmatrix} \right) \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 1/2 & -1/2 \\ -1/2 & 1/2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

#### 逆运动学解的情况

在机械臂关节角处于 $\theta$ 时，雅可比矩阵为 $J(\theta)$，由末端执行器的笛卡尔空间速度 $v_N$ 求关节角速度的公式如下：

1. 无冗余
   

机械臂操作空间维度等于机械臂关节数，若雅可比矩阵满秩，则

$$
\dot{\theta} = J^{-1}(\theta) v_N
$$

2. 冗余
   

机械臂操作空间维度小于机械臂关节数，对于末端执行器的某一笛卡尔空间速度，通常会有无穷组对应的关节速度，若雅可比矩阵是行满秩的，其中满足关节速度范数最小的一个特解（最小范数解），用右伪逆计算

$$
\dot{\theta}_r = J^T (J J)^{-1} v_N
$$

通解为

$$
\dot{\theta} = \dot{\theta}_r + \dot{\tilde{\phi}}_f = J^T (J^T)^{-1} v_N + (I - J^T (J^T)^{-1} J) \dot{\phi}_f
$$

其中 $\phi_f$ 遍历所有的关节速度向量

3. 欠驱动
   

机械臂操作空间维度大于机械臂关节数，对于末端执行器的某一笛卡尔空间速度，可能没有对应的关节速度，这时，若雅可比矩阵是列满秩的，只能得到误差范数最小的关节速度（最小二乘解），用左伪逆计算

$$
\dot{\theta} = (J^T J)^{-1} J^T v_N
$$

#### 奇异性

大多数 $6 \times 6$ 的雅可比矩阵 $J$ 都有使得其不可逆的 $\theta$ 值，这些 $\theta$ 值所对应的位姿称为机构的奇异位形或简称奇异状态。

所有的操作臂在工作空间的边界都存在奇异位形，并且大多数操作臂在它们的工作空间也有奇异位形。

>对于空间机械臂，总有 $\text{rank}(J) \leq \min(6, n)$
>对于平面机械臂，总有 $\text{rank}(J) \leq \min(2, n)$
>其中n表示机械臂关节数

=== "空间机械臂"
    其末端执行器在三维空间中具有6个自由度（3个平移自由度和3个旋转自由度），因此雅可比矩阵 $J$ 的行数为6。
    
    雅可比矩阵的秩 $\text{rank}(J)$ 表示机械臂能够独立控制的自由度数量，由于机械臂的关节数 $n$ 限制了其能够独立控制的自由度数量，所以 $\text{rank}(J) \leq \min(6, n)$。

=== "平面机械臂"
    末端执行器在二维平面中具有3个自由度（2个平移自由度和1个旋转自由度），因此雅可比矩阵 $J$ 的行数为3。
    
    但是，由于平面机械臂的旋转轴一直垂直于平面，转角大小为所有转动关节的转角之和，所以实际上平面机械臂的末端执行器在平面内只有2个独立的自由度（2个平移自由度）。
    
    因此，平面机械臂的雅可比矩阵 $J$ 的秩 $\text{rank}(J)$ 表示机械臂能够独立控制的自由度数量，由于机械臂的关节数 $n$ 限制了其能够独立控制的自由度数量，所以 $\text{rank}(J) \leq \min(2, n)$。



**奇异点的判断条件**

对于一般机械臂，奇异位形为令雅可比矩阵 $J$ 不满秩的 $\theta$ 值所构成的位形，此时 $\text{rank}(J(\theta)) < \min(m, n)$。

1. **无冗余（$m=n$）**：在此 $\theta$ 时 $J$ 不可逆，即 $\text{det}(J(\theta))=0$
2. **冗余（$m<n$）**：在此 $\theta$ 时 $J$ 不行满秩，即 $\text{rank}(J(\theta))<m$
3. **欠驱动（$m>n$）**：在此 $\theta$ 时 $J$ 不列满秩，即 $\text{rank}(J(\theta))<n$

!!! attention 

    对于平面机械臂，由于其末端姿态只有一个旋转自由度，且旋转轴一直垂直于平面，转角大小为所有转动关节的转角之和，所以判断奇异性时，平面机械臂只需关心平面二维线速度部分的雅可比矩阵，即
    
    $$
    \begin{pmatrix} \dot{v}_x \\ \dot{v}_y \end{pmatrix} = J_0 \dot{q}
    $$
    
    因此，对于平面机械臂，上述奇异位形的判断条件需利用雅可比矩阵 $J_0$。



奇异位形大致分为两类：

 1. 边界奇异性
    工作空间边界的奇异位形。出现在操作臂完全展开或者收回使得末端执行器处于或非常接近空间边界的情况。

 1. 内点奇异性
    工作空间内部的奇异位形。出现在远离工作空间的边界，通常是由于两个或两个以上的关节轴线共线引起的。

> 当操作臂处于奇异位形时，操作臂的末端在笛卡尔空间中会失去一个或多个自由度，即此时无论选择多大的关节速度，操作臂的末端在笛卡尔空间的某个方向上（或某个子空间中）都不能运动。

#### 可操作度

可操作度是衡量机器人位形与奇异位形距离的一种度量方式

!!! note "由于欠驱动机器人的逆微分运动只有最小二乘解，一般只讨论无冗余和冗余机器人的可操作性问题"


### 分析雅可比矩阵

分析雅可比矩阵：通过操作臂末端的最小表示的运动学方程对关节变量的微分计算得到的雅可比矩阵。

令 $X = \begin{pmatrix} P(\theta) \\ \psi(\theta) \end{pmatrix}$ 表示末端执行器的位姿，其中 $P(\theta)$ 为基座坐标系原点到末端执行器坐标系原点的一般向量，$\psi(\theta)$ 为末端执行器坐标系相对于基座坐标系姿态的最小表示（例如固定角表示或欧拉角表示）。

分析雅可比满足以下形式：

$$
\dot{X} = \begin{pmatrix} \dot{P} \\ \dot{\psi} \end{pmatrix} = J_a(\theta) \dot{\theta}
$$

#### 刚体角速度与欧拉角速率的关系

由 $\dot{R} R R^T = S$ 得到：

$$
\begin{pmatrix}
\dot{r}_{11} & \dot{r}_{12} & \dot{r}_{13} \\
\dot{r}_{21} & \dot{r}_{22} & \dot{r}_{23} \\
\dot{r}_{31} & \dot{r}_{32} & \dot{r}_{333}
\end{pmatrix}
\begin{pmatrix}
r_{11} & r_{21} & r_{31} \\
r_{12} & r_{222} & r_{32} \\
r_{13} & r_{23} & r_{33}
\end{pmatrix}
=
\begin{pmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{pmatrix}
$$

得到：

$$
\omega_x = \dot{r}_{31} r_{21} + \dot{r}_{32} r_{22} + \dot{r}_{33} r_{23}
$$

$$
\omega_y = \dot{r}_{11} r_{31} + \dot{r}_{12} r_{32} + \dot{r}_{13} r_{33}
$$

$$
\omega_z = \dot{r}_{21} r_{11} + \dot{r}_{22} r_{12} + \dot{r}_{23} r_{13}
$$

以 Z-Y-Z 欧拉角表示刚体在基坐标系中的姿态为例，记 $\Psi = (\alpha \beta \gamma)^T$，则旋转矩阵 $R$ 可以表示为：


$$
R = R_{Z Y Z}(\alpha, \beta, \gamma) = R_{Z}(\alpha) R_{Y}(\beta) R_{Z}(\gamma) 
$$

即：

$$
R = 
\begin{pmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{pmatrix}
= 
\begin{pmatrix}
cac\beta c\gamma - s\alpha s\gamma & -cac\beta s\gamma - sac\gamma & cas\beta \\
sac\beta c\gamma + c\alpha s\gamma & -sac\beta s\gamma + c\alpha c\gamma & sac\beta \\
-s\beta c\gamma & s\beta s\gamma & c\beta
\end{pmatrix}
$$

欧拉角速率为 $\dot{\Psi} = (\dot{\alpha} \dot{\beta} \dot{\gamma})^T$

则：

$$
\omega_x = i_{31} r_{21} + i_{32} r_{22} + i_{33} r_{23} = \left( \frac{\partial r_{31}}{\partial \alpha} r_{21} + \frac{\partial r_{32}}{\partial \alpha} r_{2} + \frac{\partial r_{33}}{\partial \alpha} r_{23} \right) \dot{\alpha} + \left( \frac{\partial r_{31}}{\partial \beta} r_{21} + \frac{\partial r_{32}}{\partial \beta} r_{2} + \frac{\partial r_{33}}{\partial \beta} r_{23} \right) \dot{\beta} + \left( \frac{\partial r_{31}}{\partial \gamma} r_{21} + \frac{\partial r_{32}}{\partial \gamma} r_{2} + \frac{\partial r_{33}}{\partial \gamma} r_{23} \right) \dot{\gamma}
$$

$$
= -s\alpha \dot{\beta} + c\alpha s\beta \dot{\gamma} = \left( 0 -s\alpha \quad c\alpha s\beta \right) \dot{\Psi}
$$

同样有：

$$
\omega_y = i_{1} r_{31} + i_{12} r_{32} + i_{13} r_{3} = (0 \quad c\alpha \quad s\alpha s\beta) \dot{\Psi}
$$

$$
\omega_z = i_{21} r_{11} + i_{2} r_{12} + i_{23} r_{13} = (1 \quad 0 \quad c\beta) \dot{\Psi}
$$

因此：

$$
\omega = \begin{pmatrix} \omega_x \\ \omega_y \\ \omega_z \end{pmatrix} = 
\begin{pmatrix} 
0 & -s\alpha & c\alpha s\beta \\
0 & c\alpha & s\alpha s\beta \\
1 & 0 & c\beta 
\end{pmatrix} \dot{\Psi}
$$

其中 $\Psi = (\alpha ~~ \beta ~~ \gamma)^T$
上述角速度与欧拉角速率的关系公式也称为“欧拉运动学方程”


!!! note "分析雅可比与几何雅可比的关系"

    $$
    J(\theta) \dot{\theta} = \begin{pmatrix} \dot{v} \\ \dot{\omega} \end{pmatrix} = \begin{pmatrix} \dot{P} \\ B_a(\psi) \dot{\psi} \end{pmatrix} = \begin{pmatrix} I & 0 \\ 0 & B_a(\psi) \end{pmatrix} \begin{pmatrix} \dot{P} \\ \dot{\psi} \end{pmatrix} = \begin{pmatrix} I & 0 \\ 0 & B_a(\psi) \end{pmatrix} J_a(\theta) \dot{\theta}
    $$
    
    $$
    J_a(\theta) = \begin{pmatrix} I & 0 \\ 0 & B_a^{-1}(\psi) \end{pmatrix} J(\theta)
    $$
    
    要求 $B_a$ 矩阵可逆。
    
    记 $T_a = \begin{pmatrix} I & 0 \\ 0 & B_a^{-1}(\psi) \end{pmatrix}$ 则 $J_a(\theta) = T_a J(\theta)$


### 静力

!!! quote "本章假设"
    - 由于操作臂底座静止，因此静态平衡只考虑静止。操作臂的自由末端在工作空间推某个物体，该物体未动
    - 本章静力学不考虑作用在连杆上的重力
    - 所有的关节和操作臂位形不变化


操作臂在静态平衡（静止或匀速直线运动）状态下，考虑力和力矩如何从一个连杆向下一个连杆传递

!!! info "定义"
      - **力**：3维向量 $f_i = $ 连杆 $i-1$ 施加在连杆 $i$ 上的力。包含大小、方向、作用点三个属性。需要用两个向量来表示一个力
      - **力矩**：3维向量 $n_i = $ 连杆 $i-1$ 施加在连杆 $i$ 上的力矩。只有大小和方向两个属性。

#### 作用在操作臂上的静力

力 $f$ 对原点 $O$ 的矩在 $i$ 中可表示为 $^iP \times ^if \in \mathbb{R}^3$

矩的大小 $|^iP||^if| \sin \theta = h||^if|$，$\theta$ 是 $^iP$ 与 $^if$ 的夹角，$h$ 是力臂

垂直于 $^iP$ 和 $^if$ 所在平面的矩方向意味着**“矩使刚体产生绕 $^iP \times ^if$ 旋转的趋势”**

![image-20250315193055616](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503151931757.png)

**力偶**

- 两个大小相等、方向相反且不共线的平行力组成的力系
- 力偶的作用只改变刚体的转动状态，其转动效应可用力偶矩来度量

力偶 $(\vec{f}, -\vec{f})$ 对点 $O$ 的矩：

$$
\vec{OA} \times \vec{f} + \vec{OB} \times (-\vec{f}) = \vec{BA} \times \vec{f}
$$

![image-20250315193055616](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503151934828.png)

对刚体上的任何点，力偶矩 $\vec{BA} \times \vec{f}$ 不变

力偶矩向量 $\vec{BA} \times \vec{f}$ 可在刚体上任意转移

!!! example "力的平移"
    在刚体上作用于A点的力 $\vec{f}$，在刚体上任取一点B，在点B加上一对平衡力 $\vec{f}$ 和 $-\vec{f}$ 构成力偶。作用于点A的 $\vec{f}$ 和作用于点B的 $-\vec{f}$ 构成力偶，其力偶矩 $\vec{BA} \times \vec{f} = $ 作用于点A的 $\vec{f}$ 对点B的矩。

    刚体上作用于A点的力 $\vec{f}$ 可以转换为刚体上作用于B点的力，附加 $\vec{BA} \times \vec{f}$。
    
    ![image-20250315194859935](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503151949037.png)

对于静止的机械臂来说，存在力与力矩平衡

对于连杆i收到的力只有两个，将力相加并令其等于零：

$$
^i f_i - ^i f_{i+1} = 0 \Longrightarrow ^i f_i = ^i f_{i+1}
$$

对于连杆i收到的力矩有三个，分别是$^i n_i$、$^i n_{i+1}$以及$^i f_{i+1}$作用产生的力矩(由于力${^i f_{i}}$经过原点，因此对连杆i的力矩为0，因此只有三个极距)，将绕坐标系 $i$ 原点的力矩相加：


$$
^i n_i - ^i n_{i+1} - ^i P_{i+1} \times ^i f_{i+1} = 0
$$

由此得到：

$$
^i n_i = ^i n_{i+1} + ^i P_{i+1} \times ^i f_{i+1}
$$

![image-20250315201525429](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503152015539.png)

#### 静力传递

=== "静力传递表达式"
    $$
    ^i f_i = ^i_{i+1} R^{(i+1)} f_{i+1}
    $$

    $$
    ^i n_i = ^i_{i+1} R^{(i+1)} n_{i+1} + ^i P_{i+1} \times ^i f_i
    $$

**为了平衡施加在连杆上的力和力矩，需要在关节提供多大的力矩（旋转关节）或力（移动关节）**

旋转关节：

- $f_i$ 不是约束力，它阻止连杆 $i$ 作直线运动，$n_i$ 阻止连杆 $i$ 作旋转运动。在 $\{i\}$ 中对 $n_i$ 进行正交分解，可得到 1 个沿 $\hat{Z}_i$ 的力矩向量和 1 个垂直于 $\hat{Z}_i$ 的力矩向量。
- 垂直于 $\hat{Z}_i$ 的力矩向量是约束力矩；沿 $\hat{Z}_i$ 的力矩向量是主动力矩，主动力矩需由关节 $i$ 的旋转驱动器提供。
- 主动力矩可表示为 $\tau_i \hat{Z}_i$，其中 $\tau_i = |n_i| \cos \theta = |n_i| \hat{Z}_i| \cos \theta = n_i^T \hat{Z}_i$。

移动关节：

- $n_i$ 是约束力矩。在 $\{i\}$ 中对 $f_i$ 进行正交分解，得到 1 个主动力和 1 个约束力，需由关节 $i$ 的直线驱动器提供的主动力表示为 $\tau_i \hat{Z}_i$，其中 $\tau_i = f_i^T \hat{Z}_i$。

![image-20250315205755134](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503152057222.png)

!!! example "两连杆操作臂，在末端执行器施加作用力向量 $\vec{F}$，求出所需的关节力矩。"
    1. 写出各齐次变换矩阵、外力和外力矩
    - 齐次变换矩阵 

    $$
    {}^0 T_1 = \begin{pmatrix} c_1 & -s_1 & 0 & 0 \\ s_1 & c_1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}{}^1 ,T_2 = \begin{pmatrix} c_2 & -s_2 & 0 & l_1 \\ s_2 & c_2 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}  {}^2 ,T_3 = \begin{pmatrix} 1 & 0 & 0 & l_2 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
    $$
    
    - 外力和外力矩：
      
    $$
    {}^3 f_3 = \begin{pmatrix} f_x \\ f_y \\ 0 \end{pmatrix}，
    {}^3 n_3 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
    $$
    
    2. 从 $\{3\}$ 开始向内迭代 



## 轨迹规划

!!! note "路径和轨迹的区别"
    路径：机器人位形的一个特定序列，而不考虑机器人位形的时间因素<br>
    轨迹：与何时到达路径中的每个部分有关，强调了时间性和连续性<br>

### 关节空间轨迹规划

以关节角的函数来描述轨迹的轨迹生成方法

**三项式规划**

为获得一条确定的光滑运动曲线，至少需要对 $\phi(t)$ 施加四个约束条件：

1. $\phi(0) = \phi_0$，初始角度为 $\phi_0$。
2. $\phi(t_f) = \phi_f$，最终角度为 $\phi_f$。
3. $\dot{\phi}(0) = \dot{\phi}_0$，初始角速度为 $\dot{\phi}_0$。
4. $\dot{\phi}(t_f) = \dot{\phi}_f$，最终角速度为 $\dot{\phi}_f$。

这些约束条件唯一确定了一个三次多项式 $\phi(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$。

解为:

$$
a_0 = \phi_0 \\
a_1 = \dot{\phi}_0\\
a_2 = -\frac{3\phi_0 - 3\phi_f + 2\dot{\phi}_0 t_f + \dot{\phi}_f t_f}{t_f^2}\\
a_3 = \frac{2\phi_0 - 2\phi_f + \dot{\phi}_0 t_f + \dot{\phi}_f t_f}{t_f^3}
$$


**五次多项式**

五次多项式可以指定$t_0 (= 0)$和$t_f$时刻关节角度和速度以及加速度来规划轨迹：

$$
\phi(t) = a_0 + a_1t + a_2t^2 + a_3t^3 + a_4t^4 + a_5t^5
$$

$$
\dot{\phi}(t) = a_1 + 2a_2t + 3a_3t^2 + 4a_4t^3 + 5a_5t^4
$$

$$
\ddot{\phi}(t) = 2a_2t + 6a_3t^2 + 12a_4t^3 + 20a_5t^4
$$

解为

$$
a_0 = \phi_0 \\
a_1 = \dot{\phi}_0 \\
a_2 = \frac{\ddot{\phi}_0}{2} \\
a_3 = \frac{20\phi_f - 20\phi_0 - (8\dot{\phi}_f + 12\dot{\phi}_0)t_f - (3\dot{\phi}_0 - \dot{\phi}_f)t_f^2}{2t_f^3} \\
a_4 = \frac{30\phi_0 - 30\phi_f + (14\dot{\phi}_f + 16\dot{\phi}_0)t_f + (3\dot{\phi}_0 - 2\dot{\phi}_f)t_f^2}{2t_f^4} \\
a_5 = \frac{12\phi_f - 12\phi_0 - (6\dot{\phi}_f + 6\dot{\phi}_0)t_f - (\dot{\phi}_0 - \dot{\phi}_f)t_f^2}{2t_f^5}
$$

#### 确定中间点期望关节速度

##### 法一

- 将相邻的关节中间点用直线相连，则该直线的斜率就是两个相邻关节中间点的平均速度
- 如果某一关节中间点前后两段直线的斜率符号相反，则可将该点的速度取为0，如$\phi_1$和$\phi_3$处
- 如果某一关节中间点前后两段直线的斜率符号相同，则可将该点的速度取为两者的平均值，如$\phi_2$处



![image-20250318201033589](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503182010744.png)


##### 法二

- 不直接指定关节中间点处的速度，而是以**保证相邻两段三次多项式加速度连续**为原则选取三次多项式系数
- 考虑三个相邻的关节中间点，依次为$\phi_i$，$\phi_j$和$\phi_k$
- 连接$\phi_i$和$\phi_j$的三次多项式为
  
$$
\phi_{ij}(t) = a_0 + a_1t + a_2t^2 + a_3t^3, t \in [0, t_{f1}]
$$

- 连接$\phi_j$和$\phi_k$的三次多项式为
  
$$
\phi_{jk}(t) = b_0 + b_1t + b_2t^2 + b_3t^3, t \in [0, t_{f2}]
$$

 !!! attention "这里将第二段三次多项式的起始时间定为0，目的是简化系数计算"

满足条件的等式（求解参数）


$$
\phi_{ij}(0) = \phi_i \\
\phi_{ij}(t_{f1}) = \phi_j \\
\phi_{jk}(0) = \phi_j \\
\phi_{jk}(t_{f2}) = \phi_k \\
\dot{\phi}_{ij}(0) = \dot{\phi}_i \\
\dot{\phi}_{jk}(t_{f2}) = \dot{\phi}_k \\
\dot{\phi}_{ij}(t_{f1}) = \dot{\phi}_{jk}(0) \\
\ddot{\phi}_{ij}(t_{f1}) = \ddot{\phi}_{jk}(0)
$$

#### 考虑关节中间点的带抛物线过渡的直线段

!!! attention "需检验加速度值是否超过限定"

**一些定义**

>用$j$，$k$和$l$表示三个相邻的路径点
位于路径点$k$处的拟合区段的时间间隔为$t_k$
位于点$j$和$k$之间的直线段的时间间隔为$t_{jk}$
点$j$和$k$之间总的时间间隔为$t_{ajk}$
直线段的速度为$\dot{\phi}_{jk}$
点$j$处拟合区段的加速度为$\ddot{\phi}_j$

**内部路径点计算公式**

$$
\dot{\phi}_{jk} = \frac{\phi_k - \phi_j}{t_{ajk}} \\
\ddot{\phi}_k = \text{SGN}(\dot{\phi}_{kl} - \dot{\phi}_{jk}) |\ddot{\phi}_k| \\
t_k = \frac{\dot{\phi}_{kl} - \dot{\phi}_{jk}}{\ddot{\phi}_k} \\
t_{jk} = t_{ajk} - \frac{1}{2}t_j - \frac{1}{2}t_k
$$

![image-20250318203554106](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503182035204.png)

**第一个路径点**

- 对于第一个路径段，由$\frac{\phi_2 - \phi_1}{t_{d12} - \frac{1}{2}t_1} = \ddot{\phi}_1 t_1$，取$\ddot{\phi}_1 = \text{SGN}(\phi_2 - \phi_1)|\dot{\phi}_1|$
  
- 可计算得到
  
$$
t_1 = t_{d12} - \sqrt{t_{d12}^2 - \frac{2(\phi_2 - \phi_1)}{\ddot{\phi}_1}}
$$

- 进而
  
$$
\dot{\phi}_{12} = \frac{\phi_2 - \phi_1}{t_{d12} - \frac{1}{2}t_1} \\
t_{12} = t_{d12} - t_1 - \frac{1}{2}t_2
$$

![image-20250318203945107](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503182039241.png)

**最后一个路径点**

- 对于最后一个路径段，同样由$\ddot{\phi}_n = \text{SGN}(\phi_{n-1} - \phi_n)|\ddot{\phi}_n|$和$\frac{\phi_{n-1} - \phi_n}{t_{d(n-1)n} - \frac{1}{2}t_n} = \ddot{\phi}_n t_n$
  
- 可计算得到

$$
t_n = t_{d(n-1)n} - \sqrt{t_{d(n-1)n}^2 + \frac{2(\phi_n - \phi_{n-1})}{\ddot{\phi}_n}}
$$

- 进而
  
$$
\dot{\phi}_{(n-1)n} = \frac{\phi_n - \phi_{n-1}}{t_{d(n-1)n} - \frac{1}{2}t_n} \\
t_{(n-1)n} = t_{d(n-1)n} - t_n - \frac{1}{2}t_{n-1}
$$

![image-20250318204106045](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503182041180.png)
















## 小测

###  Q1-旋转矩阵性质

证明，$R(a \times b) = (Ra) \times (Rb)$，其中 $R$ 是旋转矩阵，$a, b \in \mathbb{R}^3$。

**定义法证明补全**  

- **首先**，我们知道向量叉积 $a \times b$ 的性质是：


$$
\det(x, a, b) = \langle x, a \times b \rangle \quad \text{对于任意 } x \in \mathbb{R}^3
$$


$$
\det(x, a, b) =
\begin{vmatrix}
x_1 & x_2 & x_3 \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}
$$

$$
\langle x, a \times b \rangle =  \begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix}
\begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix} =   \begin{vmatrix}
x_1 & x_2 & x_3 \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}
$$

- **现在**，考虑 $R \in SO(3)$：
  
  - 由于 $R$ 是旋转矩阵，满足：
  
$$
R^T = R^{-1}。
$$

- **因此**，有：
  
$$
\langle x, R(a \times b) \rangle = \langle R^T x, a \times b \rangle = \det(R^{-1}x, a, b)。
$$

- **又因为**：
  
$$
\det(R) = 1，
$$

  所以：

$$
\det(R)\det(R^{-1}x, a, b) = \det(x, Ra, Rb) = \langle x, Ra \times Rb \rangle。
$$

- **由于对于任意 $x \in \mathbb{R}^3$ 都成立**：
  
$$
\langle x, R(a \times b) \rangle = \langle x, Ra \times Rb \rangle，
$$

  根据内积的性质，可得：

$$
R(a \times b) = (Ra) \times (Rb)。
$$

