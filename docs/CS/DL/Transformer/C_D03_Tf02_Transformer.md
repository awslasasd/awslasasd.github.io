# Transformer

!!! note "提出于论文《Attention is all you nedd》"

Seq2Seq问题——Input a sequence,output a sequence

## 框架解读

基于翻译的背景下提出架构，主要是由编码器(Emcoders)和解码器(Decoders)组成，其中编码器和解码器分别由六个结构完全相同(参数不相同，分开训练)的小编码器(解码器组成)

![image-20260310200052906](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102000996.png)

详细的结构如下所示

![image-20260310200456101](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102004175.png)

### Encoder


![image-20260310200800949](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102008001.png)

#### 输入部分

- 词嵌入(Word Embedding)：将输入的单词转换为向量表示。
- 位置编码(Positional Encoding)：由于Transformer没有循环结构，需要添加位置信息来捕捉单词的顺序关系。

这里文章中给出这样的位置编码公式,其中$2i$和$2i+1$代表这句话中的奇数和偶数位置：

$$
\begin{aligned}
PE_{(pos,2i)} &= sin\big(pos / 10000^{2i/d_{\text{model}}}\big) \\
PE_{(pos,2i+1)} &= cos\big(pos / 10000^{2i/d_{\text{model}}}\big)
\end{aligned}
$$

获得Pos信息后，与原向量相加，引入位置信息。

![image-20260310201710792](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102017837.png)

!!! note "为什么引入位置嵌入是有用的"
    ![image-20260310202148155](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603102021242.png)
    后面的字可以用前面的字表达出来。
    但是这种相对位置信息会在注意力机制那里消失。

 #### 注意力机制

 在进行Self-attention之后，Transfomer还引入了残差(把输入向量与self-attention加起来作为最后的输出)

!!! note "Layernorm和batchnorm的区别"
    batchnorm是对一个batch中的不同样本，在特征维度进行求残差
    layernorm是以样本维度进行求残差
    ![image-20260311103114174](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111031295.png)

Encoder的总体的计算逻辑如下
​	![image-20260311103332214](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111033283.png)


### Decoder

#### Masker Multi-Head Attention

> Decoder过去的输出会作为下一时刻的输入

Decoder最开始的输入是一个Special Token

![image-20260311104717278](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111047344.png)

#### Encoder与Decoder的通讯

![image-20260311104817949](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111048994.png)

Encoder传来$K$和 $V$两个参数，Decoder提供$Q$

详细的步骤如下

![image-20260311105140563](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111051642.png)

下面以此类推得到剩下的结果

## 模型训练过程

> Teacher Forcing : using the ground truth as input

在训练时，Decoder的输入就是我们的目标结果，比如下图中的，Decoder的输入不再是上一阶段Decoder的输出和输入，而是直接把正确答案当作输入，把输出的结果与真实值求误差

![image-20260311121836857](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111218956.png)





























