# 感知机



## 基本知识

定义：感知机接收多个输入信号，输出一个信号

输入信号被送往神经元时，会被分别乘以固定的权重（$w_1x_1,w_2x_2$）。神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为“神经元被激活”。这里将这个界限值称为阈值，用符号*θ*表示。

![image-20251209140431344](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091404415.png)

$$
y = 
\begin{equation}
\begin{cases}
0\quad(w_1x_1+w_2x_2)\leq\theta\\
1\quad(w_1x_1+w_2x_2)\gt\theta
\end{cases}
\end{equation}
$$

> 权重越大，对应该权重的信号的重要性就越高。

## 简单逻辑电路

**与门（AND gate）**

只有两输入都是1输出才是1

| $x_1$ | $x_2$ | $y$  |
| ----- | ----- | ---- |
| 0     | 0     | 0    |
| 1     | 0     | 0    |
| 0     | 1     | 0    |
| 1     | 1     | 1    |

比如$(w_1,w_2,\theta)=(0.5,0.5,0.7)$

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

**与非门（NAND gate）**

两输入同时为1输出为0，其余都是1(**与门取非**)

| $x_1$ | $x_2$ | $y$  |
| ----- | ----- | ---- |
| 0     | 0     | 1    |
| 1     | 0     | 1    |
| 0     | 1     | 1    |
| 1     | 1     | 0    |

比如$(w_1,w_2,\theta)=(-0.5,-0.5,-0.7)$

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```


**或门（OR gate）**

只要有1个输入是1，输出就是1

| $x_1$ | $x_2$ | $y$  |
| ----- | ----- | ---- |
| 0     | 0     | 0    |
| 1     | 0     | 1    |
| 0     | 1     | 1    |
| 1     | 1     | 1    |

比如$(w_1,w_2,\theta)=(0.5,0.5,0.2)$

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

## 感知机的局限性

**异或门（XOR gate）**

只有输入一方为1时才会输出1

| $x_1$ | $x_2$ | $y$  |
| ----- | ----- | ---- |
| 0     | 0     | 0    |
| 1     | 0     | 1    |
| 0     | 1     | 1    |
| 1     | 1     | 0    |

感知机的局限性就在于它只能表示由一条直线分割的空间，对于与或们来说，是一个非线性的

![image-20251209142427697](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091424747.png)

## 多层感知机

事实上其实只要把前面其中的两个结合起来一起使用就可以实现异或门的功能

![image-20251209142907819](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091429865.png)

$x_1$和$x_2$与非门和或门的输入，而与非门和或门的输出则是与门的输入。

| $x_1$ | $x_2$ | $s_1$ | $s_2$ | y    |
| ----- | ----- | ----- | ----- | ---- |
| 0     | 0     | 1     | 0     | 0    |
| 1     | 0     | 1     | 1     | 1    |
| 0     | 1     | 1     | 1     | 1    |
| 1     | 1     | 0     | 1     | 0    |


```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

![image-20251209143119140](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091431189.png)