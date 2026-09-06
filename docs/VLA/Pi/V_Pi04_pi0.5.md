# Pi0.5

## 简介

> 近期研究表明，将机器人操作策略的训练数据分布从狭窄的单任务数据集拓展到涵盖众多场景和任务的多样化数据集，不仅可以使生成的策略开箱即用地解决更广泛的任务，还能提升其泛化到新场景和任务的能力。

## 架构

![image-20260905145344069](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260905145344129.png)

分为Pre-Train和Post-Train两部分

解决的问题：

- VLA的泛化问题——到一个没见过的环境，VLA的表现不好
- 对于简单的语义任务，比如拿枕头表现优秀，但是对于High-Level的任务，比如清理房间，效果就表现不好



## 训练流程

![image-20260905160208918](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260905160209074.png)

!!! attention "Pre-train和Post-train用到的数据不同"



![ChatGPT Image Sep 5, 2026, 01_39_57 AM](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260905164016187.png)



这里需要注意一下，这里用的`Autoregressive`是`standard autoregressive next-token prediction`，也就是说，真实的FastToken是要作为Transformer的输入的，因为预测的时候，比如真实为$[Z_1,Z_2,Z_3,Z_4]$，预测$[\hat{Z}_1,\hat{Z}_2,\hat{Z}_3,\hat{Z}_4]$。常规的Autoregressive是拿$\hat{Z}_1$去当作预测 $\hat{Z}_2$的条件，但是这里是用$Z_1$去当预测$\hat{Z}_2$的条件。



### Pre-train

!!! note "Pre-train在做什么"
> 和pi0一样，起始都是采用PaliGemma，也就是说，一开始已经有一个会：看图片；理解语言；图像描述；VQA；的 VLM。
> 但它此时还不能算真正的机器人 VLA，因为它还没有系统学会：$ \text{Observation}\rightarrow\text{Robot Action} $
> π0.5 的 Pre-training 就是在把这个 VLM“改造成 VLA”。

每次抽到什么数据，就做对应的任务。可以这样看：

| Pre-training 数据类型   | 输入                                      | 训练目标                          |
| ------------------- | --------------------------------------- | ----------------------------- |
| 机器人动作数据             | 图像 + Robot State + `pick up the pillow` | FAST 离散动作 tokens              |
| High-Level 数据       | 图像 + Robot State + `clean the kitchen`  | `"put the plate in the sink"` |
| Web Caption 数据      | Web 图片 + `caption the image`            | `"a dog catches a frisbee"`   |
| Object Localization | 图片 + `localize the gripper`             | Bounding-box tokens          


统一通过：

$\boxed{\text{Autoregressive Cross-Entropy}}$

训练原来的 `PaliGemma VLM`，使其成为 `VLA`。

它表示的是：

$$ \boxed{\text{很多不同种类的训练样本混在一起 Co-training}} $$




> pre-training 阶段把这些机器人数据、High-Level 预测和 Web 多模态任务放在一起训练，并统一做成 standard autoregressive next-token prediction。


#### CE-Loss计算

这里输入的种类、训练样本很多，但无论是一个tets(经过tokenize)、一个动作(利用FAST)都最终转化为Token来进行loss计算Cross-Entory(交叉熵)。​

loss计算公式是：

$$ \boxed{ L_{\rm CE}^{(b)} = -\frac{1}{M_b} \sum_{i=1}^{M_b} \log p_\theta \left( y_{b,i} \mid c_b,y_{b,<i} \right) } $$

这里：

- $c_b$：这一条样本的条件输入，比如 Image、Robot State、Prompt；
- $y_{b,1:M_b}$：这一条样本真正要模型预测的 token


!!! attention "Fast tokens里的causal attention mask"
    $$ [z_1,z_2,z_3,z_4] $$

    训练预测 $z_2$ 时只能看到：
    
    $$ [c,z_1] $$
    
    不能看到：
    
    $$ z_3,z_4 $$
    
    否则就相当于提前看到未来答案。




​	

### Post-train

!!! question "为什么还要Post-Train"
    Pre-train已经把PaliGemma训练成了能够自回归生成动作的VLA，但是我们想要得到既会做 high-level 语义决策，又能用 Flow Matching 快速产生连续动作的最终模型

和pi0一样，这里引入了一个Action expert，专门负责FlowMatching的连续动作生成

Pre-training 你已经知道，大概是：

$$ \boxed{ MM+ME+CE+HL+WD } $$

而 Post-training 变成：

$$ \boxed{MM+ME+HL+WD+VI} $$

!!! note "post-training 阶段加入 verbal instructions，并去掉 laboratory cross-embodiment CE 数据，以便更专注于 mobile manipulation 和 diverse environments"


**MM和ME的数据还进行了一次筛选**

> post-training action dataset consists of the MM and ME robot data, filtered down to successful episodes that are below a fixed length threshold.	​

也就是：

$$
 D_{\rm action}^{post} = \operatorname{Filter} (D_{\rm MM}\cup D_{\rm ME}) 
$$

筛选条件至少有两个：

$ \boxed{\text{episode 成功}} $ 以及：episode 长度低于某个固定阈值
	​

!!! attention "时间步注入方式与pi0不同"
    pi0的时间是与noise进行拼接然后输入到Transformer里

    这里的注入方式是将时间步，直接注入到Transformer里的Ada RMSNorm里

