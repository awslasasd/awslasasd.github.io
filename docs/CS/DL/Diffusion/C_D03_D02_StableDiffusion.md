# Stable Diffusion

## 框架

![image-20260311190429051](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111904132.png)



### Text Encoder

> 文字的Encoder对后面的结果影响很大



![image-20260311191739585](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111917670.png)



### Generation Model

这里的方法也是一样，首先拿出在训练Decoder中的Encoder，通过照片输出一个中间产物，然后加入噪声得到一个新的中间产物，不断如此。

![image-20260311192854838](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111928936.png)
 
然后设计一个Noise Predicter 通过文字Embedding和中间产物得到噪声，从而实现Generation Model的设计 

### Decoder

Decoder 的训练不需要文字数据

- 如果latent representation(潜在表示)是一个小图，那Decoder只需要训练把小图变为大图的Decoder。

![image-20260311192256685](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111922763.png)

- 如果不是小图，那就需要训练一个Auto-encoder.
  
![image-20260311192412902](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202603111924965.png)










