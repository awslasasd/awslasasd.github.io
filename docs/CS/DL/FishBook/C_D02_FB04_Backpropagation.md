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

!!! note "加法节点的反向传播将上游的值原封不动地输出到下游"

![image-20260309150043717](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603091500768.png)

### 乘法节点的反向传播

乘法的反向传播会将上游的值乘以正向传播时的输入信号的“翻转值”后传递给下游。翻转值表示一种翻转关系，如图5-12所示，正向传播时信号是x的话，反向传播时则是y；正向传播时信号是y的话，反向传播时则是x。

![image-20260309150357474](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603091503519.png)


!!! attention "加法的反向传播只是将上游的值传给下游，并不需要正向传播的输入信号。但是，乘法的反向传播需要正向传播时的输入信号值"

## 简单层的实现

###　乘法层的实现


```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y 
        dy = dout * self.x
        return dx, dy
```

用前面苹果的例子

![image-20260309151008007](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603091510046.png)


```python
# 买苹果
apple = 100
apple_num = 2
tax = 1.1
#layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
#forward
apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price,tax)

print(price)# 220
```


### 加法层的实现

```python
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```

## 激活函数层的实现

### ReLU层

激活函数ReLU（Rectified Linear Unit）

$$
y=
\begin{cases}
x&(x\gt0)\\
0&(x\leq0)
\end{cases}
$$

可求得导数

$$
\frac{\partial y}{\partial x}=
\begin{cases}
1&(x\gt0)\\
0&(x\leq0)
\end{cases}
$$

也就是说，如果正向传播输入x大于0，那么反向传播时原封不动传递给上游。如果x小于0，那反向传播到此为止。

```python
# ReLU层
class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
```

Relu类有实例变量mask。这个变量mask是由True/False构成的NumPy数组，它会把正向传播时的输入x的元素中小于等于0的地方保存为True，其他地方（大于0的元素）保存为False。

### Sigmoid层

$$
y=\frac{1}{1+\exp{(-x)}}
$$

最终反向传播的输出是$\frac{\partial L}{\partial y}y^2\exp{(-x)}$，这个反向传播的值与正向传播的输入x和输出y相关。简化这个sigmoid层表示为

![image-20260309160548814](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603091605864.png)

```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
```

## Affine/Softmax层的实现

### Affine层

> 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”A。因此，这里将进行仿射变换的处理实现为“Affine层”。
>
> 几何中，仿射变换包括一次线性变换和一次平移，分别对应神经网络的加权和运算与加偏置运算。

回顾一下神经网络的正向传播，是一个加权信号的总和，用到了矩阵的乘法

```python
X = np.random.rand(2) # 输入
W = np.random.rand(2,3) # 权重
B = np.random.rand(3) # 偏置

X.shape # (2,)
W.shape # (2, 3)
B.shape # (3,)

Y = np.dot(X, W) + B
```

> (2,)是一个1×2的哦，但是(2,3)是2×3的矩阵

![image-20260309200214454](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603092002573.png)

> 注意，矩阵翻转的同时还要注意转置

$$
\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}\cdot{W^T}\\
\frac{\partial L}{\partial Y}=X^T\cdot{}\frac{\partial L}{\partial Y}
$$

**批版本的Affine**

![image-20260309203027254](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603092030312.png)

这里还有一个细节，正向传播的时候，虽然偏置看上去是一行的，但是实际加到加权和还是根据输入的行数来的，所以偏置实际上是N行的。所以反向传播的时候，传回去还得重新压缩回一行。那就得按列求总和。

```python
#正向传播
>>> X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
>>> B = np.array([1, 2, 3])
>>>
>>> X_dot_W
array([[ 0, 0, 0],
[ 10, 10, 10]])
>>> X_dot_W + B
array([[ 1, 2, 3],
[11, 12, 13]])
```

```python
#反向传播
>>> dY = np.array([[1, 2, 3,], [4, 5, 6]])
>>> dY
array([[1, 2, 3],
[4, 5, 6]])
>>>
>>> dB = np.sum(dY, axis=0)
>>> dB
array([5, 7, 9])
```

### Softmax-with-Loss 层

> 最后介绍一下输出层的softmax函数。前面我们提到过，softmax函数会将输入值正规化之后再输出

!!! example "手写数字识别"
    ![image-20260309203647881](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603092036944.png)

考虑到这里也包含作为损失函数的交叉熵误
差（cross entropy error），所以称为“Softmax-with-Loss层”，内部结构如下

![image-20260309203820345](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603092038417.png)

上面的结构有点复杂，下面是简化后的结果

![image-20260309204256652](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603092042714.png)

Softmax层的反向传播得到了（y1 − t1, y2 − t2, y3 − t3）这样“漂亮”的结果。由于（y1, y2, y3）是Softmax层的输出，（t1, t2, t3）是监督数据，所以（y1 − t1, y2 − t2, y3 − t3）是Softmax层的输出和教师标签的差分。神经网络的反向传播会把这个差分表示的误差传递给前面的层

!!! note "回归问题中输出层使用恒等函数，损失函数使用平方和误差,反向传播才是上面这样漂亮的结果"

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx   
```
请注意反向传播时，将要传播的值除以批的大小（batch_size）后，传递给前面的层的是单个数据的误差。






















