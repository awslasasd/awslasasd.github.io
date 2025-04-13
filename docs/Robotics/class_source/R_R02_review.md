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
    错误

!!! example "牛顿欧拉法是基于力学的方法，拉格朗日法是基于能量的方法"
    正确

!!! example "四元素与机器人姿态一一对应"
    正确

!!! example "关节空间规划不需要考虑奇异位型"
    

!!! example "刚体的不同姿态与SO(3)中的不同旋转矩阵是一一对应的"
    正确

!!! example ""
    

## 选择

!!! example "下面说法错误的是：(A)"
    A.旋转矩阵特征值为1
    B.旋转矩阵行列式为1
    C.旋转矩阵列向量互相正交
    D.旋转矩阵列向量为单位向量

!!! example "一个旋转矩阵对应()个单位四元数：(B)"
    A.1
    B.2

!!! example "关节空间动力学模型的G和笛卡尔空间动力学模型的Gx是否一样：(B)"
    A.是
    B.否

!!! example "下面哪个不是机器人姿态的最小表示：(A)"
    A.四元素

!!! example "下面关于齐次变换矩阵说法错误的是"
    

!!! example "下面关于工作空间说法正确的是：(A)"
    A.灵巧空间是可达空间的子集

!!! example "关于$_B^A \Omega$的说法正确的是"
    

!!! example "对于机器人动力学方程$ \tau = M(\dot{\Phi}) + V(\dot{\Phi}, \Phi) + G(\Phi) $：(A)"
    A. M是正定
    B. V是正定
    C. G是正定
    D. 均不是
## 填空

!!! example "线加速度公式"

!!! example "齐次变换矩阵求逆"

!!! example "旋转矩阵转四元数"

!!! example "ZYX欧拉角，旋转矩阵R的左边6项已知，求右边3项以及α、β、γ。"

!!! example "给了一些条件，需要几次多项式规划；初末用多项式连接，中间用直线规划，有多少解。"

!!! example "XY平面是质量对称平面，哪几个惯量积为0。"
    $I_{xz}$和$I_{yz}$

!!! example "均质圆柱，原点位于质心，Z重合于转轴，求$ I_{xy}, I_{zz} $"
    $I_{xy} = 0、I_{zz} = \frac{1}{2}MR^2$

!!! example "对于旋转矩阵,求五个未知参数"
    $$
    \begin{bmatrix}
    \frac{\sqrt{3}}{2} & -\frac{1}{2} & r_{13} \\
    \frac{\sqrt{3}}{4} & r_{22} & r_{23} \\
    r_{31} & \frac{\sqrt{3}}{4} & r_{33}
    \end{bmatrix}
    $$

!!! example "已知$_B^A \Omega = \begin{bmatrix} 10 \\ 5 \\ 0 \end{bmatrix}, _C^B \Omega = \begin{bmatrix} 0 \\ 7 \\ 3 \end{bmatrix}$,求：$ _C^A \Omega $"