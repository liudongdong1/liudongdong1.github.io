# OpticalFlow


> 光流（optical flow）是空间运动物体在观察成像平面上的像素运动的瞬时速度。光流法是利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性来找到上一帧跟当前帧之间存在的对应关系，从而计算出相邻帧之间物体的运动信息的一种方法。

### 1. 介绍

#### 1.1. 稠密光流

> 一种针对`图像`或指定的`某一片区域`进行`逐点匹配`的图像配准方法，它计算图像上`所有的点的偏移量`，从而形成一个稠密的光流场。通过这个稠密的光流场，可以进行像素级别的图像配准。
>
> Horn-Schunck算法以及基于区域匹配的大多数光流法都属于稠密光流的范畴。

#### 1.2. 稀疏光流

> 稠密光流相反，稀疏光流并不对图像的每个像素点进行逐点计算。它通常需要`指定一组点进行跟踪`，这组点最好具有某种明显的特性，例如Harris角点等，那么跟踪就会相对稳定和可靠。稀疏跟踪的计算开销比稠密跟踪小得多。
>

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128125200642.png)

稠密光流描述图像每个像素向下一帧运动的光流，为了方便表示，使用不同的颜色和亮度表示光流的大小和方向，如图2-2右图的不同颜色。图2-3展示了一种光流和颜色的映射关系，使用颜色表示光流的方向，亮度表示光流的大小。

> 视觉算法库OpenCV中，提供光流估计[算法接口](https://link.zhihu.com/?target=https%3A//docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html)，包括稀疏光流估计算法cv2.calcOpticalFlowPyrLK()，和稠密光流估计cv2.calcOpticalFlowFarneback()

### 2. 光流估计算法介绍

#### 2.1. 基于 梯度

> 利用时变图像灰度（或其滤波形式）的时空微分（即时空梯度函数）来计算像素的速度矢量。

- Horn-Schunck算法在`光流基本约束方程`的基础上附加了`全局平滑假设`，假设在整个图像上光流的变化是光滑的，即物体运动矢量是平滑的或只是缓慢变化的。

#### 2.2. 基于匹配

- 基于`特征`的方法不断地对目标`主要特征进行定位和跟踪`，对`目标大的运动和亮度变化具有鲁棒性`。存在的问题是`光流通常很稀疏`，而且`特征提取和精确匹配`也十分困难。
- 基于`区域`的方法先对类似的区域进行定位，然后`通过相似区域的位移计算光流`。这种方法在视频编码中得到了广泛的应用。然而，它计算的光流仍不稠密。另外，这两种方法估计亚像素精度的光流也有困难，计算量很大。

#### 2.3. 基于能量

> 基于能量的方法又称为基于`频率`的方法，在使用该类方法的过程中，要`获得均匀流场的准确的速度估计`，就必须对输入的图像进行时空滤波处理，即对时间和空间的整合，但是这样会降低光流的时间和空间分辨率。基于频率的方法往往会涉及大量的计算，另外，要进行可靠性评价也比较困难。

#### 2.4. 基于相位

> Fleet和Jepson最先提出将`相位信息用于光流计算的思想`。当我们计算光流的时候，相比亮度信息，图像的相位信息更加可靠，所以`利用相位信息获得的光流场`具有更好的鲁棒性。基于相位的光流算法的优点是：对图像序列的适用范围较宽，而且速度估计比较精确，但也存在着一些问题：第一，基于相位的模型有一定的合理性，但是有较高的时间复杂性；第二，基于相位的方法通过两帧图像就可以计算出光流，但如果要提高估计精度，就需要花费一定的时间；第三，基于相位的光流计算法对图像序列的`时间混叠是比较敏感`的。

> 任何信号都可以表示成（或者无限逼近）一系列正弦信号的叠加。在一维领域，信号是一维正弦波的叠加，那么想象一下，在二维领域，实际上是无数二维平面波的叠加，$(x，y)$对应的是一维领域的 $t$，而灰度（Brightness Variation）就是其变量对应一维领域的振幅$F(t)$。

对于一幅M×N的图像，其傅里叶变换公式如下：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128154610987.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128154743715.png)

- 实部就可以看成：原图和余弦图的卷积，得到的值我们假设为 **R(u, v)**
- 虚部可以看成：原图和正弦图的卷积，得到的值我们假设为 **I(u, v)**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128155117753.png)

> `低频分量（低频信号）代表着图像中亮度或者灰度值变化缓慢的区域`，也就是图像中大片平坦的区域，描述了图像的主要部分。`高频分量（高频信号）对应着图像变化剧烈的部分`，也就是图像的`边缘（轮廓）或者噪声`以及细节部分。将图像从灰度分布转化到频率分布（频谱图）上去观察图像的特征,图像进行二维傅立叶变换之后得到的`频谱图`，就是`图像梯度的分布图`。具体的，傅立叶频谱图上我们能看到`明暗不一的亮点`，实际是图像上`某一点与邻域点差异的强弱，即梯度的大小`。` 如果一幅图像的各个位置的强度大小相等，则图像只存在低频分量`。从图像的频谱图上看，只有`一个主峰,且位于频率为零`的位置。 也就是说白色代表高频。

```python
import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('../1.jpg')
plt.subplot(231),plt.imshow(img),plt.title('picture')
#根据公式转成灰度图
img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
#显示灰度图
plt.subplot(232),plt.imshow(img,'gray'),plt.title('original')
#进行傅立叶变换，并显示结果
fft2 = np.fft.fft2(img)
plt.subplot(233),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')
#将图像变换的原点移动到频域矩形的中心，并显示效果
shift2center = np.fft.fftshift(fft2)
plt.subplot(234),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')
#对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')
#对中心化后的结果进行对数变换，并显示结果
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')
plt.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128155802727.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128160050443.png)

#### 2.5. 神经动力学

> 神经动力学方法是利用`神经网络建立的视觉运动感知的神经动力学模型`，它是对生物视觉系统功能与结构比较直接的模拟。尽管光流计算的神经动力学方法还很不成熟，然而对它的研究却具有极其深远的意义。随着生物视觉研究的不断深入，神经方法无疑会不断完善，也许光流计算乃至计算机视觉的根本出路就在于神经机制的引入。神经网络方法是光流技术的一个发展方向。

### 3. 算法

#### 3.1. Lucas-Kanade

- **亮度不变假设**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128130513933.png)

- **邻域光流相似假设**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128130911572.png)

#### 3.2. 深度学习

- **FlowNet**

> 该模型的输入为待估计光流的两张图像，输出即为图像每个像素点的光流。我们从Loss的设计，训练数据集和网络设计来分析FlowNet。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128131038313.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128131122682.png)

### 4. 学习链接

- https://lear.inrialpes.fr/~verbeek/mlor.slides.16.17/optical_flow.pdf
- FFT: https://blog.csdn.net/qq_41997920/article/details/100122021

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/opticalflow/  

