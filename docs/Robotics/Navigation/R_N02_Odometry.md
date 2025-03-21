---
comments: true
---

# 里程估计

## 里程估计

### 基本概念

根据传感器感知信息推导机器人位姿（位置和角度）变化

主要用途：
  - 航位推算 (Dead-reckoning)，基于已知位置，利用里程估计，推算现在位置，可作为定位估计初值



**里程估计与航位推算的关系**

![image-20250319100140639](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503191001809.png)

用若干帧的数据进行里程估计，

!!! note "为什么不用发送给机器人的运动控制指令进行航位推算？"

    $$
    \begin{pmatrix}
    x' \\
    y' \\
    \theta'
    \end{pmatrix}
    =
    \begin{pmatrix}
    x - \frac{v}{\omega} \sin(\theta) + \frac{v}{\omega} \sin(\theta + \omega \Delta t) \\
    y + \frac{v}{\omega} \cos(\theta) - \frac{v}{\omega} \cos(\theta + \omega \Delta t) \\
    \theta + \omega \Delta t
    \end{pmatrix}
    $$
    
    由于控制的滞后性和可能存在超调，
    机器人实际执行控制指令与发送控制指令存在偏差

**里程估计方法**

基于机器人运动感知信息，结合运动学模型
  - 电机码盘（轮式里程计）
  - IMU（惯性单元，加速度计+陀螺仪）（惯性里程计）

基于环境感知传感器信息，通过最佳匹配估计
  - 激光里程计（LO）
  - 视觉里程计（VO）

**位姿变化的数学描述**

二维平面运动 （Δx, Δy, Δθ）

三维空间运动 （Δx, Δy, Δz, Δα, Δβ, Δγ）

一般统一表示为旋转矩阵和平移向量形式 R, t

$$
R = \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{bmatrix}, t = [\Delta x, \Delta y, \Delta z]^T
$$

$$
x' = Rx + t \quad RR^T = I
$$

由于存在万向锁的问题，一般用四元素代替欧拉角

里程估计问题：即根据感知信息求旋转矩阵和平移向量 R, t 或者齐次矩阵 T

### 基于运动感知

#### 电机码盘的轮式移动机器人里程估计

**(1)根据电机码盘获得轮子转速**

$$
\dot{\varphi} = \frac{2\pi n}{\eta}
$$

- n:码盘测量得到的电机转速（转/分）
- η:齿轮减速比

![image-20250319102045880](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503191020011.png)

**(2) 结合运动学模型计算参考点速度**

**(3) 假设短时间片内为匀速运动，计算位姿变化**

差分驱动的轮式机器人

![image-20250319102252797](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503191022882.png)


$$
v = \frac{r \dot{\varphi}_l}{2} + \frac{r \dot{\varphi}_r}{2}
$$

$$
w = \frac{r \dot{\varphi}_l}{2l} - \frac{r \dot{\varphi}_r}{2l}
$$

$$
\Delta d = v \Delta t, \Delta \theta = w \Delta t
$$

**航位推算**


![image-20250319102431877](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503191024924.png)


$$
\begin{cases}
x_t = x_{t-1} + \Delta d \cos(\theta_{t-1} + \Delta \theta) \\
y_t = y_{t-1} + \Delta d \sin(\theta_{t-1} + \Delta \theta) \\
\theta_t = \theta_{t-1} + \Delta \theta
\end{cases}
$$

**轮式里程估计误差**

系统误差
  - 轮半径误差
  - 轮子安装精度误差（不平行，两边距离不相等）
  - 编码器精度误差
  - 采样精度误差
  - 齿轮减速比精度

$$
v = \frac{r \dot{\varphi}_l}{2} + \frac{r \dot{\varphi}_r}{2}\\
w = \frac{r \dot{\varphi}_l}{2l} - \frac{r \dot{\varphi}_r}{2l}\\
\Delta d = v \Delta t, \Delta \theta = w \Delta t\\
\dot{\varphi} = \frac{2\pi n}{\eta}
$$

偶然误差(导致了里程估计偏差)
  - 地面不平
  - 轮子打滑



#### 惯性单元的里程估计

- 通过积分运算可得载体在导航坐标系中的姿态、速度和位置等信息

优点：
- 全天候
- 采样频率高
- 短时精度较好

缺点：
- 随着时间的增长累积误差较大，无法满足移动机器人长距离精确定位的要求，需要融合其它传感器进行组合导航

### 激光里程计




#### ICP算法

[CSDN解释](https://blog.csdn.net/qq_41685265/article/details/107140349)

Step：
  - 估计P'集合点与P集合点的初始位姿关系
  - 根据最近邻域规则建立P'集合点与P集合点的关联
  - 利用线性代数/非线性优化的方式估计旋转平移量
  - 对点集合P'的点进行旋转平移
  - 如果旋转平移后重新关联的均方差小于阈值，则结束
  - 否则迭代重复上述步骤


**POINT-POINT ICP**

输入：

- 点集合A, A = {a_1, ..., a_n}
- 点集合B, B = {b_1, ..., b_n}

目标：计算两组数据之间的旋转平移量R, t，使得两组数据形成最佳匹配，即两组数据的距离误差最小

第i个匹配对点的误差为
$$
e_i = a_i - (Rb_i + t)
$$

构建成最小二乘问题，求使得误差平方和达到最小的R, t

$$
\min \frac{1}{2} \sum_{i=1}^{n} \|a_i - (Rb_i + t)\|^2
$$


算法伪代码

![img](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503191111605.png)

**线性代数求解方法**

1. 定义两组点集合的质心位置a, b
   $$
   a = \frac{1}{n} \sum_{i=1}^{n} a_i \quad b = \frac{1}{n} \sum_{i=1}^{n} b_i
   $$

2. 计算每个点的去质心坐标
   $$
   q_i = a_i - a \quad q'_i = b_i - b
   $$

3. 根据以下优化问题计算旋转矩阵
   $$
   R^* = \arg\min_R \frac{1}{2} \sum_{i=1}^{n} \|q_i - Rq'_i\|^2
   $$

4. 根据R计算t
   $$
   t^* = a - R^*b
   $$

R, t 分解计算说明

$$
\frac{1}{2} \sum_{i=1}^{n} \|a_i - (Rb_i + t)\|^2 = \frac{1}{2} \sum_{i=1}^{n} \|a_i - Rb_i - t - a + Rb + a - Rb\|^2\\
= \frac{1}{2} \sum_{i=1}^{n} \|(a_i - a - R(b_i - b)) + (a - Rb - t)\|^2\\
= \frac{1}{2} \sum_{i=1}^{n} \left\| (a_i - a - R(b_i - b)) \right\|^2 + \|a - Rb - t\|^2 + 2(a_i - a - R(b_i - b))^T(a - Rb - t)\\
= \frac{1}{n} \sum_{i=1}^{n} a_i \quad b = \frac{1}{n} \sum_{i=1}^{n} b_i\\
\sum_{i=1}^{n} = 0
$$







































