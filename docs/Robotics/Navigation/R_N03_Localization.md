---
comments: true
---

# 定位

!!! note "一些资料"
    [B站视频](https://www.bilibili.com/video/BV1ez4y1X7eR?spm_id_from=333.788.videopod.sections&vd_source=ace17a48ec1787387c4c8d582e6808cb)

## KF

### 基础知识
!!! example "例子引入"
    ![image-20250330132621609](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503301326717.png)
    随着K的增加，测量结果不再重要
    当前的估计值 = 上一次的估计值+系数X(当前测量值-上一次的估计值)
    在卡尔曼滤波中，系数就是卡尔曼增益

估计误差$e_{EST}$:当前估计值与上一次估计值的差

预测误差$e_{MEA}$:当前测量值与当前估计值的差

系数$K_k = \frac{e_{EST_{k-1}}}{e_{EST_{k-1}}+e_{MEA_k}}$

**卡尔曼滤波三个步骤**

Step 1: 计算 Kalman Gain \( K_k = \frac{e_{EST_{k-1}}}{e_{EST_{k-1}} + e_{MEAS_k}} \)

Step 2: 计算 \( \hat{x}_k = \hat{x}_{k-1} + K_k (z_k - \hat{x}_{k-1}) \)

Step 3: 更新 \( e_{EST_k} = (1 - K_k) e_{EST_{k-1}} \)


!!! example "例子"
    ![image-20250330134114475](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503301341522.png)


### 算法推导

#### 数据融合

参考下面的例子进行解释，两次观测值与方差的数据，进行数据融合，得到最终的估计值

本质上用的还是卡尔曼滤波的思想

![image-20250330135251150](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503301352214.png)

#### 协方差矩阵

协方差矩阵：方差与协方差在一个矩阵中表现出来(变量之间的联动关系)

$$
P = \begin{bmatrix}
\sigma_x^2 & \sigma_{xy} & \sigma_{x\sigma_z} \\
\sigma_{yx} & \sigma_y^2 & \sigma_{y\sigma_z} \\
\sigma_{z\sigma_x} & \sigma_{z\sigma_y} & \sigma_z^2
\end{bmatrix}
$$

参考下面的例子进行理解

![image-20250330135721886](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202503301357936.png)

#### 状态空间表达

$$
\dot{X}(t) = AX(t) + BU(t)\\
Z(t) = HX(t)
$$

$Z(t)$代表测量量

再进行离散化处理

$$
X(k) = A' X(k-1) + B' U(k-1) + W(k-1)\\
Z(k) = H' X(k) + V(k)
$$

其中$w(k-1)$代表过程噪声，符合高斯分布，$w$ ~$p(0,Q)$，$v(k)$代表测量噪声，符合高斯分布，$v$ ~$p(0,R)$。代表了不确定性。

下面的B站视频有很详细的公式推导

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=796490974&bvid=BV1hC4y1b7K7&cid=213756096&p=1&autoplay=0" width="640" height="480" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

最终结论如下

先验估计

\[ \hat{x}_k^- = A \hat{x}_{k-1} + B u_{k-1} \]


后验估计

\[ \hat{x}_k = \hat{x}_k^- + K_k (z_k - H \hat{x}_k^-) \]

卡尔曼增益

$$
K_k = \frac{P_k^{-} H^T}{P_k^{-} H^T+R}\\
其中,P_k^{-}表示先验协方差矩阵 R代表测量噪声的协方差矩阵 
$$

从上面三个公式可以看出，先验估计值可以计算出来，要计算后验估计值，需要得到先验协方差矩阵，下面的视频则去求解先验协方差矩阵$P_k^{-}$

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=413926941&bvid=BV1yV411B7DM&cid=214104384&p=1&autoplay=0" width="640" height="480" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

结论如下：

$$
P_k^{-} = A P_{k-1} A^T + Q
$$

### 结论

卡尔曼滤波分为两步，预测与校正

!!! attention "注意"
    在使用卡尔曼滤波时，要给出后验协方差的初值$P_0$和后验估计的初值$x_0$

#### 预测

先验：

\[ \hat{x}_k^- = A \hat{x}_{k-1} + B u_{k-1} \]

先验协方差：

\[ P_k^- = A P_{k-1} A^T + Q \]

#### 校正

卡尔曼增益：

\[ K_k = \frac{P_k^- H^T}{H P_k^- H^T + R} \]

后验估计：

\[ \hat{x}_k = \hat{x}_k^- + K_k (z_k - H \hat{x}_k^-) \]

更新误差协方差:

\[ P_k = (I - K_k H) P_k^- \]

## EKF算法

!!! note "一些资料"
    [CSDN博客](https://blog.csdn.net/O_MMMM_O/article/details/106078679)<br>
