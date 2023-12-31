# Multi-Camera Basic


> 利用计算机视觉技术对场景中的人进行智能分析是当前最重要的研究问题之一，但挑战仍未被完全解决，比如<font  color=red>遮挡、运动模糊、尺度变化、光照变化、适应新场景的问题等。</font>具体任务包括<font color=red>人体检测、部件分割、姿态估计与跟踪、行为识别等</font>。

## 1. 目标跟踪

FairMOT: A Simple Baseline for Multi-Object Tracking, arXiv 2020（代码已开源）![Object Tracking](C:/Users/dell/Pictures/%E5%9B%BE%E5%BA%8A/arduino-4916880__340.webp)

​        给定一个视频作为输入，Multi-Object Tracking（MOT）需要完成两个任务：<font color=red>1）首先在每一帧进行人体检测，（2）然后将不同帧中相同的人进行关联从而得到每个人在视频中的轨迹。</font>当前大部分方法都会采用两个独立的模型分别完成检测和关联任务。这类方法在多个公开数据集上都取得了不错的结果，但在实际部署时，它们通常会面临可扩展能力较弱的问题，尤其是当视频中人数很多时（比如超市等人群密集的场景）为每个人单独经过一个网络提取 Re-ID 特征的成本就会线性增加，从而导致较大的延时。

​        “特征共享”是提升模型扩展能力的常用思路。<font color=red>Track R-CNN [1] 对 Mask R-CNN 进行扩展，使用 roi-pool 从共享的特征图中获取候选框所对应的图像特征，并通过一个轻量的网络针对每一个候选框同时进行（1）检测框回归与分类，（2）前景 mask回归，（3）Re-ID 特征回归。</font>特征共享在一定程度上降低了计算量，跟踪的精度却比基于两阶段的方法明显变差，在实验中 Track R-CNN 的检测结果要好于我们的 FairMOT，但是发生 id switch (跟踪错误)的次数却是 FairMOT 的3倍还多。这个观察其实和更早期一个多任务学习的工作 UberNet 是一致的，也就是多个任务同时学习经常会得到比每个任务单独学习更差的精度。我们猜测造成 Track R-CNN 结果不佳的原因是 proposal 通常不会和物体的中心恰好对齐，参考下图中的左子图，因此在进行 roi-pool 操作时容易导致跟踪特征具有歧义。另外，该方法的速度仍然比较慢，大概是 FairMOT 的1/15。

> FairMOT 会对每一个像素进行预测，预测其是否是物体的中心、物体的大小和以其为中心的图像区域的 Re-ID 特征。检测和跟踪两个任务都是以“当前像素”为中心，所以不存在对齐的问题，也不存在严重的顾此失彼的不公平问题。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200630080422531.png)

## 2. 姿态估计

### 2.1. 单摄像机

Optimizing Network Structure for 3D Human Pose Estimation, ICCV 2019

![Pose Estimation](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200630080540.gif)

​		单目三维人体姿态估计的目标是从单张图像恢复人体关节点的三维坐标。在成像过程中深度信息的丢失导致该任务存在很强的歧义，比较前期的工作经常通过引入人体测量学约束、低维流行表达或时间平滑约束等先验信息来降低歧义。

> 采用和依赖深度神经网络来隐式地解决歧义，考虑到常见的三维人体姿态数量有限，因此采用神经网络去“记住” 2D 到 3D 的映射关系也是一种可行的方案。SimpleBaseline [4] 过度依赖关节点二维坐标的准确率。

### 2.2. 多摄像机

- Cross View Fusion for 3D Human Pose Estimation, ICCV 2019（代码已开源）
- MetaFuse: A Pre-trained Fusion Model for Human Pose Estimation, CVPR 2020
- Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: A Geometric Approach, CVPR 2020（代码已开源）

​		单摄像机三维姿态估计只能恢复出“以 root 节点为中心”的相对姿态，而无法推测其在世界坐标系下的绝对坐标。

​		 绝对姿态估计通常包括两个步骤：首先在每个摄像机下分别进行二维姿态估计，然后通过几何方法，比如<font color=red>三角测量（triangulation）或者图模型（pictorial structure model）等</font>，求得关节点的三维坐标。如果二维姿态是完全准确的，就可以无误差地恢复三维姿态。现实应用中，二维姿态检测并不可靠，尤其是存在遮挡的情况下，会给三维姿态估计带来较大的误差。

在发表在 CVPR'20 的一个工作中 [9]，我们提出一种新的几何方法将多视角下的摄像机和可穿戴式的惯性传感器（IMU）进行融合。惯性传感器的使用能够正确地估计那些在所有视角下都被遮挡的节点。该方法的一个优势是当摄像机的位姿发生变化时，不需要对模型进行调整，只需知道相机参数即可。该方法在 Total Capture 数据集上取得了最好的结果。

## 3. 基础知识

> 大幅面 多相机 视觉系统的需求越来越多，主要应用方向为大幅面高精度的定位与测量和场景拼接等。多相机视觉系统的难点在于多相机坐标系的统一. 可以分为两类，一是相机视野间无重叠部分，二是相机视野间有重叠部分。相机间无重叠部分的情况主要用于大幅面多相机高精度的定位和测量，相机间有重叠部分的情况主要用于场景的拼接等。

### 3.1. 相机无重叠部分

**【使用大标定板统一坐标】**

> 此方法采用一块大标定板来统一各个相机的坐标，每个大标定板中有若干小标定板，各个小标定板间的位置关系都是已知的，各个相机都能拍摄到一个小标定板。通过各个小标定板可以标定每个相机的内部参数和外部参数，每个相机的坐标都可以转换到各个小标定板的坐标系上，从而统一各个相机的坐标。 

```html
<figure class="third">
    <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200630103514026.png" width="200" alt="相机在各个位置拍摄Mark图像，通过图像处理方法得到Mark坐标"/><img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200630103433760.png" width="200" alt="单个标定板"/>
</figure>
```

- 检测目标分析
- 图像获取

**【使用相对运动统一坐标】**

> 此方法采用相机和被测物之间的相对运动来统一相机的坐标，相机和被测物只要一方运动即可，记录各个位置的坐标，然后通过数学运算来统一坐标。通常情况下是相机位置固定，被测物通过机械手等运动装置进行移动，然后把相机坐标系统一到机械手等运动装置的原点。 

- 通过相机拍摄的图像对Mark点进行定位，从而计算出被测物相对于标准位置的偏差，包含角度偏差和位移偏差，最终确定机械装置需要旋转的角度和平移的距离。

### 3.2. 相机间有重叠部分

**【标顶方法拼接】**

> 对于有些大幅面物体 ，可以通过拍摄多幅图像，每幅图像覆盖物体的不同的部分。如果摄像机经过标定并且它们与一个共有的世界坐标系之间的相对关系已知，就可以通过不同的图像进行精确测量。 甚至可以将多幅图像拼接为一副覆盖整个物体的大图，这个可以通过将每幅图像都校正到同一个测量平面上实现。在结果图像上，可以直接在世界坐标系中进行测量。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200630110059490.png)

## 4. 图像拼接

> 视频拼接在体育直播、全景显示、数字娱乐、视频处理中都被广泛应用，同时视频/图像拼接涉及到矫正图像、对其与匹配图像、融合、统一光照、无缝连接、多尺度重建等各个图像算法模型与细节处理，可以说是图像处理技术的综合运用。常见就是基于SIFT/SURF/OBR/AKAZE等方法实现特征提取，基于RANSAC等方法实现对齐，基于图像融合或者无缝克隆算法实现对齐图像的拼接。
>
> 针对不同的拼接方式可以分为图像拼接、视频拼接、全景拼接。针对图像拼接可以分为像素相似与特征相似；视频拼接又分为固定相机、移动相机；全景拼接分为单相机、相机列阵、鱼眼相机列阵。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210204080156421.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210204080428186.png)



- https://cloud.tencent.com/developer/article/1730811
- 开源代码：https://github.com/kushalvyas/Python-Multiple-Image-Stitching
- https://github.com/zhaobenx/Image-stitcher
- https://github.com/samggggflynn/image-stitching-opencv



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/multi-camera-basic/  

