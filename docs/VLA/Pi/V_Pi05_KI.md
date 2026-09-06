# Knowledge Insulating

## 简介

### Motivation

对之前的Pi0以及Pi0.5来说，都在原来的VLM的后面加入了一个完全没有经过训练的Action Expert模块，他们希望利用这个模块来输出连续的动作序列。

但是在计算loss的时候，梯度更新也会去更新已经训练好的VLM的参数，这就是所谓的$\text{Gradient Interference}$(梯度干扰)，会影响VLM本身的表现效果。

![image-20260905192249630](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260905192249667.png)

## 主要方法



**联合训练（Joint Training）**：模型同时使用离散动作标记和连续动作输出进行训练。离散动作标记通过自回归语言建模损失进行训练，而连续动作输出则通过流匹配损失进行训练。这种联合训练方式使得模型在训练时能够快速收敛，并且在推理时能够快速生成连续动作。

**共训练（Co-training）**：模型不仅在机器人动作数据上进行训练，还同时在非动作数据（如通用视觉-语言数据）上进行训练。这有助于模型在适应机器人控制任务的同时，保留更多的预训练知识。

**梯度阻断（Gradient Blocking）**：通过修改注意力层的计算方式，阻止动作专家的梯度回传到VLM骨干网络。具体来说，通过在注意力计算中引入一个停止梯度操作（stop-gradient operator），确保信息只能单向从VLM骨干网络流向动作专家，而不能反向传播。



和Pi0.5不同，这里不再采用两个阶段，直接使用一个阶段，同时训练VLM和ActionExpert的参数，只是ActionExpert的参数做SelfAttention的时候，可以接收VLM的影响，但是梯度下降的时候是不去影响VLM的，这就是Knowledge Insulating



![ChatGPT Image Sep 5, 2026, 04_12_48 AM](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260905191302805.png)
