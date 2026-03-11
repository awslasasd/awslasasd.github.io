# 自注意力机制

输入和输出向量数量相同，输出的每一个向量都考虑了输入的所有向量，简单框架如下

![image-20260310204656951](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102046146.png)

## 框架

### 输入到输出推导

我们希望考虑所有的输入向量，但是又不希望所有的信息全部输入，因此引入关联参数$\alpha$，这个向量决定了$a^1$分别与另外几个输入的关联程度



![image-20260310205312678](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102053883.png)



$\alpha$的计算则引出了最为关键的三个向量$q$、 $k$、$v$ ，下面是一个主流的计算方法，引入矩阵$W^q$、$W^k$、$W^v$

- **查询（Query，Q）**：当前需要处理的信息，是模型“想要找什么”的核心依据。
- **键（Key，K）**：输入序列的特征表示，用于和 Query 计算相关性（判断“哪些信息和当前需求相关”）。
- **值（Value，V）**：输入序列的特征表示，是最终要提取的信息，会根据相关性权重加权求和。


![image-20260310205459886](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102054935.png)



下面是求第一个词与四个词之间的关联度，在进行softmax规则化处理，得到attention的分数

![image-20260310210641000](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102106055.png)

得到分数后，再求$b^1$

![image-20260310210716592](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102107649.png)

然后依次可以得到所有的$b^i$

#### 矩阵形式

其中$I$代表输入，$A$的每一列对应每一个$[\alpha_{i,1},\alpha_{i,2},\alpha_{i,3},\alpha_{i,4}]$

![image-20260310211331485](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102113576.png)

## Multi-head Self-attention

在上面的Self-attention中，我们是用$q$去找相关的$k$，但是相关有很多不同的定义，因此这里提出Multi-head Self-attention，引入多组$q,k,v$

![image-20260310212200729](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102122822.png)

!!! note "对Self-attention的缺陷"
    缺少了每一个Input在Sequence的位置信息


!!! note "Self-attention和attention的区别"
    attention只规定了后面对于$QKV$三个矩阵运算规则，没有规定这三个矩阵是怎么得来的
    而Self-attention则规定了$QKV$三个矩阵是同源的，都是由X乘不同矩阵得到的








