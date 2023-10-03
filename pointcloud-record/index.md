# PointCloud Record


> 3D数据通常可以用不同的格式表示，包括深度图像，点云，网格和体积网格。 作为一种常用格式，点云表示将原始几何信息保留在3D空间中，而不会进行任何离散化。在自动驾驶、AR&VR、机器人、遥感图像、3D人脸&医学、3D游戏动画的形状设计具有很大应用场景。在3D点云上进行深度学习仍然面临数个挑战，例如数据集规模小，维数高和3D点云的非结构化性质。

## 0. 三维数据

> 三维数据本身有一定的复杂性，2D图像可以轻易的表示成矩阵，3D表达形式由应用驱动的：其中非常接近原始传感器的数据集，激光雷达扫描之后的直接就是点云，深度传感器（深度图像）只不过是一个局部的点云，原始的数据可以做端到端的深度学习，挖掘原始数据中的模式。
> 　　point cloud ，深度传感器扫描得到的深度数据，点云。
> 　　Mesh，三角面片在计算机图形学中渲染和建模话会很有用。
> 　　Volumetric，将空间划分成三维网格，栅格化。
> 　　Multi-View，用多个角度的图片表示物体。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606100309507.png)

> 点云数据是一种不规则的数据，在空间上和数量上可以任意分布，之前的研究者在点云上会先把它转化成一个规则的数据，比如栅格让其均匀分布，然后再用3D-cnn 来处理栅格数据。
> **挑战和难点：**
> 不规则，无序。输入顺序不同，卷积结果会发生变化。需要建立置换排列不变性
> 刚性变换的鲁棒性(robustness to rigid transformations)：naive的toy点云
> 对于点云的corruption, outlier noise的鲁棒性; partial data局部数据；large-scale data保持高效
## 1. 研究领域

![](https://gitee.com/github-25970295/blogImage/raw/master/img/3Dcloud.png)

### 1.1. 三维重建

> 将点云的多视匹配放在这里，比如人体的三维重建，点云的多视重建不仅强调逐帧的匹配，还需要考虑不同角度观测产生误差累积，因此也存在一个优化或者平差的过程在里面。通常是通过观测形成闭环进行整体平差实现，多视图重建强调整体优化。可以只使用图像，或者点云，也可以两者结合（深度图像）实现。重建的结果通常是Mesh网格。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606100958028.png)

### 1.2. 3D Slam

> 点云匹配（最近点迭代算法 ICP、正态分布变换方法 NDT）+位姿图优化（[g2o](https://link.jianshu.com?t=http://www.cnblogs.com/yhlx125/p/5417246.html)、LUM、ELCH、Toro、SPA）；实时3D SLAM算法 （LOAM）；Kalman滤波方法。3D SLAM通常产生3D点云，或者Octree Map。基于视觉（单目、双目、鱼眼相机、深度相机）方法的SLAM，比如orbSLAM，lsdSLAM.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606100923607.png)

### 1.3. 目标识别

> 无人驾驶汽车中基于激光数据检测场景中的行人、汽车、自行车、以及道路和道路附属设施（行道树、路灯、斑马线等）。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606095913925.png)

### 1.4.  形状检测分类

> 点云技术在逆向工程中有很普遍的应用。构建大量的几何模型之后，如何有效的管理，检索是一个很困难的问题。需要对点云（Mesh）模型进行特征描述，分类。根据模型的特征信息进行模型的检索。同时包括如何从场景中检索某类特定的物体，这类方法关注的重点是模型。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200102151219418.png)

### 1.5. 语义分类

> 获取场景点云之后，如何有效的利用点云信息，如何理解点云场景的内容，进行点云的分类很有必要，需要为每个点云进行Labeling。可以分为基于点的方法，基于分割的分类方法。从方法上可以分为基于监督分类的技术或者非监督分类技术，深度学习也是一个很有希望应用的技术。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606100747400.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606100014333.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606094256838.png)



## 2. Paper



## 3. Project

- https://github.com/nicolas-chaulet/torch-points3d

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202132017461.png)

## 4. 应用领域

### 4.1. 文物重建

> 文物重建、 AR旅游。目前大家去很多博物馆或旅游景点其实都已经有了类似的产品。比如AR游西湖之类的。

### 4.2. 虚拟试衣

> 对人体重建后才能根据不同人体的胖瘦高矮自动适配不同尺码的衣服。

### 4.3. 智能家居

> 用来放置虚拟家具看看和自己家里的调性是否匹配，还有实际尺寸，可以看看能不能放得下。

课程大纲

![图片](https://mmbiz.qpic.cn/mmbiz_png/rqpicxXx8cNnxbbrFmg2U8cnmibUJq8UdwgrasJAZMxOoZK7GvOWXicAlfWEjaEibobNn1ozQqf44D9V5ibsUiajCgWw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pointcloud-record/  

