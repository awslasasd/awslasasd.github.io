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

## 微分运动学与静力学

### 定义

#### 符号表示

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

#### 矩阵定义

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



#### 线速度向量

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

### 角速度向量

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




### 线速度变化

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

### 角速度变化

在参考坐标系 $\{A\}$ 中，坐标系 $\{C\}$ 的角速度 ${}^A\Omega_C$ 可以表示为：

$$
{}^A\Omega_C = {}^A\Omega_B + {}^A_B R \, {}^B\Omega_C
$$

在同一坐标系中，角速度可以相加

### 速度传递

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

#### 几何雅可比矩阵

几何雅可比矩阵描述了操作臂的关节速度 $\dot{\theta}$ 与末端速度（包括线速度和角速度）$v = \begin{pmatrix} v \\ \omega \end{pmatrix}$ 之间的映射关系矩阵 $J(\theta)$。

$$
v = \begin{pmatrix} v \\ \omega \end{pmatrix} = J(\theta) \dot{\theta}
$$

**前述向外迭代法计算机械臂末端速度的算法本质上是计算操作臂几何雅
可比矩阵的方法之一。**

!!! note "机械臂末端相对于基坐标系的角速度向量 \(\omega = (\omega_x, \omega_y, \omega_z)^T\) 并不是直接通过对基坐标系下的末端姿态（例如欧拉角）进行求导得到的。"
    当我们使用欧拉角来表示机械臂末端的姿态时，欧拉角是一组描述物体在空间中方向的三个角度。然而，直接对欧拉角进行求导并不能得到正确的角速度向量，因为欧拉角之间存在耦合效应，即一个角度的变化会影响到其他角度的变化。这种耦合效应会导致直接求导得到的角速度向量不准确。




## 小测

###  证明，$R(a \times b) = (Ra) \times (Rb)$，其中 $R$ 是旋转矩阵，$a, b \in \mathbb{R}^3$。

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

