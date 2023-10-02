# 笔记合成


> **Facebook AI**最新出品的“文字风格刷”（TextStyleBrush），它只需要一张笔迹的照片，就能完美还原出一整套文本字迹来。为了同时实现`图像分割`和`文字风格转换`，TextStyleBrush模型基于**StyleGAN2**进行了设计，后者能生成非常逼真的图像照片。
>
> 然而，StyleGAN2存在两个问题：
>
> - 首先，它生成图像的方式是“随便乱打”的，也就是没办法控制输出图像特征。但TextStyleBrush必须要生成**指定文本**的图像。
> - 其次，StyleGAN2的整体风格**不受控制**，但TextStyleBrush中的风格涉及大量信息组合，包括颜色、尺度和风格转换等特征，甚至是带有个人特色的笔迹细节差异。
>
> 为此，TextStyleBrush首先通过将文本信息和风格作为两个“附加条件”控制模型输出，来解决模型随机生成图像的问题。[TextStyleBrush数据集](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) [论文地址](https://scontent-fml2-1.xx.fbcdn.net/v/t39.8562-6/10000000_944085403038430_3779849959048683283_n.pd)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210831152053710.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210831152108869.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E7%AC%94%E8%AE%B0%E5%90%88%E6%88%90/  

