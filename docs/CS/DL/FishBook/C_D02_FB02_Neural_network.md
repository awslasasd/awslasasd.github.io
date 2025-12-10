# 神经网络

>神经网络的一个重要性质是它可以自动地从数据中学习到合适的权重参数

## 简介

![image-20251209143555432](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091435474.png)

### 从感知机到神经网络

先对感知机的表达式改进一下


$$
y = 
\begin{equation}
\begin{cases}
0\quad(b+w_1x_1+w_2x_2)\leq0\\
1\quad(b+w_1x_1+w_2x_2)\gt0
\end{cases}
\end{equation}
$$

$$
y=h(b+w_1x_1+w_2x_2) \\
h(x)=
\begin{equation}
\begin{cases}
0\quad{(x\leq0)}\\
1\quad{(x\gt0)}
\end{cases}
\end{equation}
$$

$h(x)$函数会将输入信号的总和转换为输出信号，这种函数一般称为激活函数



!!! note "激活函数的作用在于决定如何来激活输入信号的总和"



取

$$
a=b+w_1x_1+w_2x_2\\
y=h(a)
$$
于是便有了(偏置的输入就一直是1的，**画成灰色**以便与其他输入区分)

![image-20251209144414950](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091444985.png)

如上图所示，神经元中明确显示了激活函数的计算过程

## 激活函数

### 阶跃函数

> 数以阈值为界，一旦输入超过阈值，就切换输出

 ```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
 ```


### sigmoid函数

$$
h(x)=\frac{1}{1+\exp{(-x)}}=\frac{1}{1+e^{-x}}
$$

输出的是一个0-1的一个值

```python
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))


x = np.arange(-5, 5, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()
```

**比较sigmoid函数和阶跃函数**

![image-20251209151054452](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091510500.png)


=== "相同点"
    输入越大输出越大（靠近1）

    都是**非线性函数**。神经网络的输出必须得是非线性函数，如果是线性函数就神经网络的层数就没有意义了。

=== "不同点"
    平滑性的不同：sigmoid函数是一条平滑的曲线，输出随着输入发生连续性的变化。而阶跃函数以0为界，输出发生急剧性的变化。

    返回值不同：相对于阶跃函数只能返回0或1，sigmoid函数可以返回连续值


### ReLU函数

输入大于0时，直接输出该值；小于等于0，输出是0。

$$
h(x)=
\begin{equation}
\begin{cases}
x\quad{(x\gt0)}\\
0\quad{(x\leq0)}
\end{cases}
\end{equation}
$$

python实现

```python
import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1, 5.5)  # 指定y轴的范围
plt.show()
```

## 3层神经网络的实现

![image-20251209153508474](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091535530.png)

### 符号表示

![image-20251209153607646](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091536692.png)

权重和隐藏层的神经元的右上角有一个(1)，它表示权重和神经元的层号（即第1层的权重、第1层的神经元）。此外，权重的右下角有两个数字，它们是后一层的神经元和前一层的神经元的索引号。

> $w_{12}^{(1)}$表示前一层的第2个神经元$x_2$到后一层的第1个神经元$a_1^{(1)}$的权重

!!! attention "权重右下角按照后一层的索引号、前一层的索引号的顺序排列"

### 传递过程

![image-20251209153902536](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091539596.png)

第1层第1个神经元

$$
a_1^{(1)}=w_{11}^{(1)}x_1+w_{12}^{(1)}x_2+b_1^{(1)}
$$

那第1层用矩阵就可以写成

$$
A^{(1)}=XW^{(1)}+B
$$

其中

$$
A^{(1)}=(a_1^{(1)}\quad{}a_2^{(1)}\quad{}a_3^{(1)})\\
\quad \\
X=(x_1\quad{}x_2)\\
\quad \\
B^{(1)}=(b_1^{(1)}\quad{}b_2^{(1)}\quad{}b_3^{(1)})\\
\quad \\
W^{(1)}=
\left(
\begin{array}{l}
w_{11}^{(1)} & w_{21}^{(1)} & w_{31}^{(1)}\\
w_{12}^{(1)} & w_{22}^{(1)} & w_{32}^{(1)}
\end{array}
\right)
$$


**过程展示**

- 输入层到第一层

![image-20251209154725652](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091547720.png)

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.array([1, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)  # (2,3)
print(W1.shape) # (2,)
print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1) # [0.3, 0.7, 1.1]
print(Z1) # [0.57444252, 0.66818777, 0.75026011]
```

- 第一层到第二层

![image-20251209154848108](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091548179.png)

```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2,)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2) # [0.51615984, 1.21402696]
print(Z2) # [0.62624937, 0.7710107 ]
```

- 第二层到输出层

![image-20251209155744689](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091557756.png)

```python
#恒等函数
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)# 或者Y = A3
```

!!! note "输出层的激活函数"
    输出层所用的激活函数，要根据求解问题的性质决定。一般地，回归问题可以使用恒等函数，二元分类问题可以使用 sigmoid函数，多元分类问题可以使用 softmax函数。关于输出层的激活函数，我们将在下一节详细介绍。


```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [ 0.31682708 0.69627909]
```

## 输出层的设计

一般而言，回归问题用恒等函数，分类问题用softmax函数。

### 恒等函数

恒等函数会将输入按原样输出，对于输入的信息，不加以任何改动地直接输出。因此，在输出层使用恒等函数时，输入信号会原封不动地被输出。

输出层的激活函数用*σ*()表示

```python
def identity_function(x):
 return x
```

### softmax函数


$$
y_k=\frac{\exp{(a_k)}}{\displaystyle\sum^n_{i=1}(a_i)}
$$

输出层有$n$有个神经元，计算第k个神经元的输出 $y_k$。

分子是输入信号$a_k$的指数函数，分子是所有输入信号的指数函数的和。

![image-20251209162608139](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091626214.png)


```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

!!! note "softmax函数在计算机运算中会出现溢出问题"
    softmax函数的实现中要进行指数函数的运算，但是此时指数函数的值很容易变得非常大。如果在这些超大值之间进行除法运算，结果会出现“不确定”的情况。

改进措施：相分子分母都乘$\text{C}$这个常数，然后移动到指数函数里面，最后改为$\text{C}^\prime$

$$
\begin{aligned}
y_k=\frac{\exp{(a_k)}}{\displaystyle\sum^n_{i=1}(a_i)}&=\frac{\text{C}\exp{(a_k)}}{\text{C}\displaystyle\sum^n_{i=1}(a_i)}\\
&=\frac{\exp{(a_k+\log\text{C})))}}{\displaystyle\sum^n_{i=1}(a_i+\log\text{C})}\\
&=\frac{\exp{(a_k+\text{C}^{\prime})}}{\displaystyle\sum^n_{i=1}(a_i+\text{C}^{\prime})}
\end{aligned}
$$

$\text{C}^\prime$其实可以取任何值，为了防止溢出，一般都会使用输入信号中的最大值。

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

!!! note "性质"
    输出总和为1是softmax函数的一个重要性质。正因为有了这个性质，我们才可以把softmax函数的输出解释为“概率”。

    一般而言，神经网络只把输出值最大的神经元所对应的类别作为识别结果。并且，即便使用softmax函数，输出值最大的神经元的位置也不会变。因此，神经网络在进行分类时，输出层的softmax函数可以省略。
    
    求解机器学习问题的步骤可以分为“学习”A 和“推理”两个阶段。推理过程中不需要softmax函数，学习阶段需要





## 手写数字识别

- 数据集: MNIST

数据集获取并进行预处理

```python
# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()

```

获取数据并处理后，可以用下面的代码查看其中图片，进行预处理验证

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
```

之后，就可以进行推理(预测)，得到预测的精度

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

这里有一个load_mnist函数的参数normalize设置成了True，它会将将图像的各个像素值除以255，使得数据的值在0.0～1.0的范围内。这就是经常用的**正则化(Normalization)**

### 批处理

上面的代码是输入一张图片下的情况，查看每一环节数组对应维度

![image-20251209182157906](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091821985.png)

```python
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
x.shape
#(10000, 784)
x[0].shape
#(784,)
W1.shape
#(784, 50)
W2.shape
#(50, 100)
W3.shape
#(100, 10)
```

现在我们改成一次输入100张图片，来看一下各个环节的数组维度变化

![image-20251209182143140](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202512091821229.png)

这是100个图形都进行运算。这种打包式的输入数据称为**批**（batch）。

可以修改为下

```python
x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
```

逻辑: 批量处理完后，统一取第一个维度下的最大值np.argmax(y_batch, **axis=1**)