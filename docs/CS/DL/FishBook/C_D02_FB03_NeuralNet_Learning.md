# 神经网络的学习

## 简介

传统机器学习提取特征量，再用机器学习技术学习这些特征量的模式。

神经网络直接学习图像本身。

> 深 度 学 习 有 时 也 称 为 端 到 端 机 器 学 习（end-to-end machine learning）。

## 损失函数

损失函数是表示神经网络性能的“恶劣程度”的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致。

### 均方误差（mean squared error）

均方误差是神经网络中常用的损失函数，定义如下：

$$
E=\frac{1}{2}\displaystyle\sum_k(y_k-t_k)^2
$$

$y_k$是表示神经网络的输出，$t_k$表示监督数据，$k$表示数据的维数。

```python
import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

mse = mean_squared_error(np.array(y), np.array(t))
print(mse)
# 0.09750000000000003
```

!!! note "one-hot表示"
    将正确解标签表示为1，其他标签表示为0的表示方法称为one-hot表示，即是独热编码。

### 交叉熵误差 （cross entropy error）

交叉熵误差是神经网络中常用的损失函数，定义如下：

$$
E=-\displaystyle\sum_kt_k\log{y_k}
$$

$y_k$是表示神经网络的输出，$t_k$表示监督数据，$k$表示数据的维数，log表示以e为底的自然对数ln。

因为$t_k$采用one-hot表示方式，其中只有正确解标签的索引为1，所以实际上只计算对应正确解标签的输出的自然对数，那么此时交叉熵误差的值是由正确解标签所对应的输出结果决定的。

!!! note "正确解标签对应的输出越大，误差的值越接近0；当输出为1时，交叉熵误差为0。"

```python
def cross_entropy_error(y, t):
    delta = 1e-7  # 防止log(0)情况
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

cross_entropy_error(np.array(y), np.array(t))
#0.510825457099338
```

为了防止出现`log(0)=-Inf`的情况导致无法运算，所以加了一个微小值`delta`。

## mini-batch学习

前面介绍的损失函数的例子中考虑的都是针对单个数据的损失函数，如果要求所有训练数据的损失函数的总和，以交叉熵误差为例：

$$
E=-\frac{1}{N}\displaystyle{\sum_n\sum_kt_{nk}\log{y_{nk}}}
$$

$N$表示训练数据的总数，$n$表示训练数据的索引，$k$表示数据的维数，$y_{nk}$表示神经网络对第n个训练数据的第k个输出，$t_{nk}$表示第n个训练数据的监督数据。

!!! note "除以n的目的"
    除以N进行正规化。
    通过除以N，可以求单个数据的“平均损失函数”。
    通过这样的平均化，可以获得和训练数据的数量无关的统一指标。

### 定义

> 定义:我们从全部数据中选出一部分，作为全部数据的“近似”。神经网络的学习也是从训练数据中选出一批数据（称为mini-batch,小批量），然后对每个mini-batch进行学习。这种学习方式称为mini-batch学习。


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)# (60000, 784)
print(t_train.shape)# (60000, 10)
```

读入MNIST的数据之后，选取小批量

```python
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

### 交叉熵误差

我们来实现一个可以同时处理单个数据和批量数据（数据作为batch集中输入）两种情况的函数。

```python
def cross_entropy_error(y,t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

如果不用one-hot表示，则不能忽略t

```python
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	
	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

