# WM综述

从机器人的视角看WM，可能会有下面两个问题

- 从VLA是怎么发展到WM的
- WM到底有哪些流派

看来下面这篇综述或许会有很好的理解

<div style="position: relative; width: 100%; height: 0; padding-bottom: 75%; overflow: hidden;">
  <iframe
    src="https://ntumars.github.io/wm-robot-survey/"
    title="WM Robot Survey"
    loading="lazy"
    style="position: absolute; inset: 0; width: 100%; height: 100%; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px;"
    allowfullscreen>
  </iframe>
</div>

如果页面无法在此处加载，也可以[在新窗口打开 WM Robot Survey](https://ntumars.github.io/wm-robot-survey/){target="_blank" rel="noopener"}。

## 综述概要

### 3 World Model for Policy：从架构角度：世界模型如何和机器人策略耦合在一起

核心问题：怎么把 “世界模型的预测能力” 嵌入到机器人策略网络内部，辅助策略生成动作。

#### 3.2 IDM‑style 逆动力学解耦策略

✅**底座：独立预训练视频生成模型（外挂世界模型） + 另外一套独立策略网络**

- 架构：**两套完全分开、架构解耦的模块**
模块 A：世界模型（预训练视频扩散模型，独立）；模块 B：逆动力学策略头（独立）
- 流程：世界模型先生成未来表征（像素 /latent 隐特征 / 几何轨迹），**输出给另一个策略网络，由策略反解动作**。先生成，后执行。
- 代表：UniPi、VidMan、VPP、MimicVideo、TC‑IDM
- 推理：可以输出像素，也可以只输出 latent 特征，但**世界模型永远是外挂独立零件**
- 核心弱点：如果世界模型预测错，错误直接传导到策略；两个模块之间存在表征 gap。

> 
> 一句话：**两个分开的模型，一个脑补未来，另一个看脑补结果输出动作。**

#### 3.3 Single‑backbone 单主干统一策略

✅**底座：视频生成主干（Video‑DiT 视频扩散 Transformer），只有**一套完整主干网络**。**

- 架构：**不再拆成两个模块**；视觉 token、action token 全部塞进同一个视频生成主干，**世界建模和动作生成是同一个生成过程**。
- 没有 “世界模型输出结果喂给策略” 这一步；预测未来画面、生成动作在同一个 diffusion 去噪流程里同时完成。
- 推理时可以选择：要不要输出完整视频画面；可以边缘化视觉分支，只输出动作，加速推理。
- 代表：UVA、UWA、VideoVLA、Cosmos Policy、DreamZero

> 
> 一句话：**拆掉两个模块的边界；用视频预训练的同一个大网络，同时脑补未来画面 + 输出机器人动作。**

#### 3.4 MoE / MoT‑Style Policies（专家世界模型主干）

✅**底座依然是视频生成主干（Video‑DiT），但是网络内部拆成多个专家分支**

> 
> 和 3.3 区别：3.3 全部参数共享；3.4 保留多个专门专家分支（视频专家、动作专家），**专家参数不完全共享，靠共享 / 交叉注意力互相通信**。
> 3.4 内部又分 3 种子模式：

1. 并行专家耦合（GE‑Act）：完整视频专家（可生成像素）+ 轻量动作分支；交叉注意力交互
2. 深度交互 MoT（Motus、LingBot‑VA）：MoT 架构；视频 token、action token 交错自回归；原生支持完整视频 rollout 推演未来
3. latent‑space expertization 潜空间专家化（LDA‑1B、FRAPPE）：专家分支还保留，但是全部工作在隐空间，**不再生成像素图片**。

> 
> 关键：**视频专家是原生完整的世界模型，继承大规模视频预训练权重，具备完整推演世界变化的能力**，只是网络内部做了专家分工。
> 和 3.2 区分：不是两个外部独立模型；专家都属于同一个大网络内部组件，不是外挂。
> 和 3.3 区分：不做全部参数共享；保留模态专属专家分支。

> 
> 一句话：**还是以视频生成模型作为本体，但是网络内部分工，视频专家专门脑补世界，动作专家专门输出动作，互相通信。**

#### 3.5 Unified Vision‑Language‑Action Models 统一 VLA 模型

✅**底座：MLLM 多模态大语言模型（VLA，比如 RT‑2、OpenVLA 这类）**

> 
> ⚠️**最重要分水岭：底座不再是视频扩散生成模型！本体是 VLA 大语言模型。**
> 3.5 分为 3 个子类：

1. 子类 1：显式预测未来图像（GR‑1，UP‑VLA）：VLA 内部增加头，预测未来图像作为辅助训练 loss；推理不一定输出图像。
2. 子类 2：隐式世界知识预测（DreamVLA、UniVLA、CoWVLA）：不预测像素，在 VLA 内部学结构化世界知识、运动隐表征。
3. 子类 3：多专家 / 多系统统一 VLA（F1、InternVLA‑A1、HALO、TriVLA）：**虽然用 MoT/MoE 工具，但是专家是附加在 VLA 之上；预测分支只做视觉预见、子目标生成；不是原生完整视频世界模型，不能做长序列完整视频 rollout。**

> 
> 论文原句：`rather than as a native video‑backbone world model`
> 翻译：它不是原生的视频主干世界模型。

> 
> 一句话：**本体是 VLA 语言大模型，把世界建模能力作为附加训练目标 / 附加小模块塞进去；不是以视频生成本身为本体。**


#### 3.6 Policies with Latent‑Space World Modeling 隐空间世界建模策略

✅**底座：MLLM/VLA 多模态大语言模型**

- 核心本质：**把世界建模完全内化在 VLA 主干网络的表征空间，不存在独立的视频生成主干，也不存在专门的视频 / 动作专家分支**。
- 不做像素、视频解码。不维护独立外挂世界模型，也不设置 MoE/MoT 专家分支。
- 训练目标参考 JEPA 思想：只预测**对控制有用的未来隐嵌入**，直接在表征空间学习环境状态转移；未来隐表征直接作为动作生成的条件。
- 代表论文：FLARE、VLA‑JEPA、JEPA‑VLA、WoG、DIAL；还包含符号世界模型作为补充路线。
- 关键点：没有独立模块、没有专家分支，世界建模能力是**VLA 网络自身隐层的内在能力**。

### 4 World Model as Simulator：世界模型充当仿真器

**核心问题：把世界模型当做一个完整虚拟环境，机器人可以在里面交互、跑完整轨迹。**