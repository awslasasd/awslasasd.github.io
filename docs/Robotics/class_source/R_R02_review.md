---
comments: true
---

# 复习资料

!!! note "复习资料"
    机器人建模回忆卷:
    [23-24](https://www.cc98.org/topic/5871439)<br>
    [22-23](https://www.cc98.org/topic/5595634)<br>
    [21-22](https://www.cc98.org/topic/5353760)<br>
    机器人学回忆卷:
    [23-24](https://www.cc98.org/topic/5920696)<br>
    [23-24](https://www.cc98.org/topic/5639736)<br>
    [21-22](https://www.cc98.org/topic/5352203)<br>
    [21](https://www.cc98.org/topic/5071145)<br>

## 判断

!!! example "具有6个旋转关节的操作臂存在封闭解的充分条件是**相邻的**三个关节轴线相交于一点"
    正确

!!! example "存在几个正交关节轴或者有多个$\alpha_i$为0或$\pm$90°时，对于6自由度操作臂有解析解"
    正确

!!! example "分析雅可比是姿态的最小表示"
    基于对末端执行器姿态的最小表示

!!! example "牛顿欧拉法是基于力学的方法，拉格朗日法是基于能量的方法"
    正确

!!! example "四元素与机器人姿态一一对应"
    错误

!!! example "关节空间规划不需要考虑奇异位型"
    不确定，中间点需要注意奇异位型
    

!!! example "刚体的不同姿态与SO(3)中的不同旋转矩阵是一一对应的"
    正确

!!! example "若$A \in \mathbb{R}^{n \times n}$是反对称矩阵，则$\forall x \in \mathbb{R}^n$，有$x^T A x = 0$"
    正确

!!! example "机器人力位混合控制通过动力学方程求出自然约束和人工约束"

!!! example "旋转矩阵的转置仍然是旋转矩阵"
    正确

!!! example "在一个串联关节机器人场景下，若其笛卡尔空间是6维，则其关节空间也是6维"
    错误，关节空间的维数与关节数量有关
    

!!! example "等效轴角表示是姿态的最小表示"
    错误

!!! example "6关节机器人中，速度域的奇异位形不一定是力域的奇异位形"
    错误 两者的雅可比矩阵是转置关系
    
!!! example "单位四元数ab=ba"
    错误

!!! example "鲁棒控制和自适应控制参数变不变"
    鲁棒控制参数不变，自适应控制参数可变

!!! example "奇异位形"
    奇异位形是指机器人在某些特定位置时，其雅可比矩阵的秩下降，导致运动或力的自由度受限


## 选择

!!! example "下面说法错误的是：(A)"
    A.旋转矩阵特征值为1<br>
    B.旋转矩阵行列式为1<br>
    C.旋转矩阵列向量互相正交<br>
    D.旋转矩阵列向量为单位向量

!!! example "一个旋转矩阵对应()个单位四元数：(B)"
    A.1<br>
    B.2

!!! example "关节空间动力学模型的G和笛卡尔空间动力学模型的Gx是否一样：(B)"
    A.是<br>
    B.否

!!! example "下面哪个不是机器人姿态的最小表示：(A)"
    A.四元素<br>
    B.欧拉角<br>
    C.固定角<br>

!!! example "下面关于齐次变换矩阵说法错误的是"
    A.最后一行一定是0 0 01<br>

!!! example "下面关于工作空间说法正确的是：(A)"
    A.灵巧空间是可达空间的子集

!!! example "关于$_B^A \Omega$的说法正确的是"
    

!!! example "对于机器人动力学方程$\tau = M(\dot{\Phi}) + V(\dot{\Phi}, \Phi) + G(\Phi)$说法正确的是：(A)"
    A. M是正定<br>
    B. V是正定<br>
    C. G是正定<br>
    D. 均不是

!!! example "下面说法错误的是：(B)"
    A. 单位四元数的积仍是单位四元数<br>
    B. 单位四元数的和仍是单位四元数<br>
    C. 单位四元数的共轭仍是单位四元数

!!! example "姿态等效轴角表示有几组：()"
    A. 1<br>
    B. 2<br>
    C. 无穷多

!!! example "DH表达法中，滑动关节的参数哪个是变量：(D)"
    A. $\alpha_i$<br>
    B. $a_i$<br>
    C. $\theta_i$<br>
    D. $d_i$

!!! example "角速度矩阵S一定是：(B)"
    A. 对称矩阵<br>
    B. 反对称矩阵<br>
    C. 旋转矩阵<br>
    D. 可逆矩阵


## 填空

!!! example "线加速度公式"
    如果${}^B Q$静止

    $$
    ^A\dot{V}_Q = ^A\dot{V}_{BORG} +  ^A\dot{\Omega}_B \times _B^AR^BQ + ^A\Omega_B \times (^A\Omega_B \times _B^AR^BQ)
    $$

!!! example "齐次变换矩阵求逆"
    $$
    T^{-1} = 
    \begin{pmatrix}
    R^\top & -R^\top O \\
    0  & 1
    \end{pmatrix}
    $$

!!! example "旋转矩阵转四元数"
    ![image-20250417145901516](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202504171459618.png)


!!! example "ZYX欧拉角，旋转矩阵R的左边6项已知，求右边3项以及α、β、γ。"

!!! example "给了一些条件，需要几次多项式规划；初末用多项式连接，中间用直线规划，有多少解。"

!!! example "XY平面是质量对称平面，哪几个惯量积为0。"
    $I_{xz}$和$I_{yz}$

!!! example "均质圆柱，原点位于质心，Z重合于转轴，求$I_{xy}, I_{zz}$"
    $I_{xy} = 0、I_{zz} = \frac{1}{2}MR^2$

!!! example "对于旋转矩阵,求五个未知参数"
    $$
    \begin{bmatrix}
    \frac{\sqrt{3}}{2} & -\frac{1}{2} & r_{13} \\
    \frac{\sqrt{3}}{4} & r_{22} & r_{23} \\
    r_{31} & \frac{\sqrt{3}}{4} & r_{33}
    \end{bmatrix}
    $$

!!! example "已知$_B^A \Omega = \begin{bmatrix} 10 \\ 5 \\ 0 \end{bmatrix}, _C^B \Omega = \begin{bmatrix} 0 \\ 7 \\ 3 \end{bmatrix}$,求：$_C^A \Omega$"

!!! example "已知$_B^A R$和$_B^A \dot{R}$，求$\Omega$"
    $$S = _B^A \dot{R} \cdot (_B^A R)^\top $$

!!! example "已知$_B^A T$和${}^B P$，求${}^A P$"
    $$
    \begin{bmatrix}
    {}^A P \\
    1
    \end{bmatrix}= 
    _B^A T
    \cdot
    \begin{bmatrix}
    {}^B P \\
    1
    \end{bmatrix}
    $$

!!! example "已知0处函数值为5，导数值为10，1处函数值为23，导数值为28，求插值三次多项式"
    设三次多项式为$f(x) = ax^3 + bx^2 + cx + d$，根据已知条件列出方程组求解$a, b, c, d$。


!!! example "请描述四元数插值与欧拉角插值进行笛卡尔空间轨迹规划的优势与劣势"
    四元数插值：避免万向节锁现象、平滑插值。但理解复杂，且旋转角速度是定制。<br>
    欧拉角插值：理解简单。但存在万向节锁现象、且插值出来的结果可能不一定是旋转矩阵。

