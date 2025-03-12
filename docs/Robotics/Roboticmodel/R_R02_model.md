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










## 小测

####  1、证明，$R(a \times b) = (Ra) \times (Rb)$，其中 $R$ 是旋转矩阵，$a, b \in \mathbb{R}^3$。

**定义法证明补全**  

- **首先**，我们知道向量叉积 $a \times b$ 的性质是：

  $$
  \det(x, a, b) = \langle x, a \times b \rangle \quad \text{对于任意 } x \in \mathbb{R}^3。
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

