# 3DPoseRelative


### 1. Mediapipe 3D detection

> 使用移动增强现实(AR)会话数据(session data)，开发了新的数据pipeline。大部分智能手机现在都具备了增强现实的功能，在这个过程中捕捉额外的信息，包括相机姿态、稀疏的3D点云、估计的光照和平面。
>
> - 利用`相机的姿势`、`检测到的平面`、`估计的照明`，来生成物理上可能的位置以及具有与场景匹配的照明位置 。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/fe7664fd36314781844b67377e46a3b3)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/c5841dc3fa794b4fa86f9dafd0cba1c2)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/dc2a5b20865b4369a90cc088a79236eb)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726141438375.png)

> 对于`形状任务`，根据可用的ground truth注释(如分割)来预测对象的形状信号；对于`检测任务`，使用`带注释的边界框，并将高斯分布拟合到框中，以框形质心为中心，并与框的大小成比例的标准差。`
>
> `回归任务`估计边界框`8个顶点的2D投影`。为了获得边界框的最终3D坐标，还利用了一个成熟的`姿态估计算法(EPnP)`，可以`在不知道物体尺寸的前提下恢复物体的3D边界框。`

### 2. [FrankMocap](https://github.com/facebookresearch/frankmocap)

> FrankMocap 是港中文联合 Facebook AI 研究院提出的**3D [人体姿态](https://cuijiahua.com/blog/tag/人体姿态/)和形状估计**算法**。**不仅仅是估计人体的运动姿态，甚至连**身体的形状**，**手部的动作**都可以一起计算出来。使用 SMPL-X 人体模型,
>
> - 给定一张彩色图片，通过两个网络模块分别预测手部姿态和人体姿态。
> - 然后再通过整合模块将手和身体组合在一起，得到最终的3D全身模型

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/ai-1-7.png)

### 3. [PIFuHD](https://github.com/facebookresearch/pifuhd)

- 论文：https://arxiv.org/pdf/2004.00452.pdf
- GitHub地址：https://github.com/facebookresearch/pifuhd
- 项目地址：https://shunsukesaito.github.io/PIFuHD/
- Demo地址：https://colab.research.google.com/drive/11z58bl3meSzo6kFqkahMa35G5jmh2Wgt?usp=sharing#scrollTo=afwL_-ROCmDf

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726142508042.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726142540661.png)

### 4. [Deep_Object_Pose](https://github.com/NVlabs/Deep_Object_Pose)

> DOPE ROS package for detection and 6-DoF pose estimation of **known objects** from an `RGB camera`. The network has been trained on the following YCB objects: cracker box, sugar box, tomato soup can, mustard bottle, potted meat can, and gelatin box. For more details, see our [CoRL 2018 paper](https://arxiv.org/abs/1809.10790) and [video](https://youtu.be/yVGViBqWtBI).

![DOPE Objects](https://gitee.com/github-25970295/blogpictureV2/raw/master/dope_objects.png)

<iframe width="1280" height="720" src="https://www.youtube.com/embed/yVGViBqWtBI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### 5. keypose

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726144514747.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726144535552.png)

<iframe width="1154" height="783" src="https://www.youtube.com/embed/DBY4gycGzXM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### 6. [DenseFusion](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1901.04780.pdf)

>  "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion"([arXiv](https://arxiv.org/abs/1901.04780), [Project](https://sites.google.com/view/densefusion), [Video](https://www.youtube.com/watch?v=SsE5-FuK5jo)) by Wang et al. at [Stanford Vision and Learning Lab](http://svl.stanford.edu/) and [Stanford People, AI & Robots Group](http://pair.stanford.edu/). The model takes an RGB-D image as input and predicts the 6D pose of the each object in the frame. 

<iframe width="1280" height="720" src="https://www.youtube.com/embed/SsE5-FuK5jo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/3dposerelative/  

