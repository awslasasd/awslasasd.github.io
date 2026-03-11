# ViT

> 核心思想：图像化整为零，切分为patch,每一个patch作为一个token

## 架构

![image-20260311153052702](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111530805.png)

分为以下5个步骤
- 图片切分为16x16的patch
- patch转化为embedding
- 位置embedding与tokensembedding相加
- 输入到TRM模型
- CLS输出做分类任务
  
!!! note "为什么第三步要引入CLS符号"
    带疑问
    但是不引入CLS也不会影响效果，可以通过修改学习率来提高效果，但是在学习率相同的情况下，引入CLS的效果会更好




