# 误差反向传播

> 高效计算权重参数的梯度的方法——误差反向传播

## 链式法则

传递这个局部导数的原理，是基于链式法则（chain rule）的

![image-20260309145201500](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603091452556.png)

局部导数$\frac{\partial y}{\partial x}$乘以上游传来的值$E$，然后传递给前面的节点。这就是反向传播的计算顺序。实现的原理是链式法则

!!! example "$z=(x+y)^2$"
    可以写成两个式子，反向传播图如下所示
    $$
    z=t^2\\
    t=x+y
    $$

    ![image-20260309145310270](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603091453312.png)


## 反向传播

### 加法节点的反向传播









































