# FAST-WAM

> 并非构建了一个新的WAM的架构，是在探究，对于WAM来说，预测下一帧的视频对于action到底有没有用，如果有用的话，是在训练阶段还是在推理阶段

![image-20260914185900123](https://zyysite.oss-cn-hangzhou.aliyuncs.com/20260914185900309.png)

基于一个MoT系统，通过joint预测VAE和Action的field大小，然后在推理的时候，不再进行Video DiT的输出，只进行一次foward得到一个latent frame，然后action输出完整的chunk
