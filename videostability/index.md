# VideoStability


### 1. 3D Video Stabilization with Depth Estimation by CNN-based Optimization （CVPR 2021）

<iframe width="1182" height="665" src="https://www.youtube.com/embed/pMluFVA7NDQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

> [[论文]](https://drive.google.com/file/d/1vTalKtMz2VEowUg0Cb7nW3pzQhUWDCLA/view?usp=sharing)[[项目](https://yaochih.github.io/deep3d-stabilizer.io/)] 
>
> 基于CNN优化的深度估计三维视频稳定我们提出了一种新的基于深度的三维视频稳定学习方法Deep3D稳定器。我们的方法不需要预训练数据，而是直接通过三维重建来稳定输入视频。校正阶段结合三维场景深度和摄像机运动，平滑摄像机轨迹，合成稳定的视频。与大多数基于学习的方法不同，我们的平滑算法允许用户有效地操纵视频的稳定性。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210528095408041.png)

推荐方法的pipeline：pipeline由两个阶段组成。首先，三维几何优化阶段通过测试时训练，分别用PoseNet和DepthNet估计输入RGB序列的三维摄像机轨迹和稠密场景深度。优化阶段以输入序列和相应的光流作为学习3D场景的引导信号。其次，视频帧校正阶段以估计的摄像机轨迹和场景深度作为输入，在平滑后的轨迹上进行视点合成。平滑过程使用户可以通过操纵平滑滤波器的参数来获得不同程度的稳定度，然后对得到的视频进行包装和裁剪，得到稳定的视频。

## 2. Deep Online Fused Video Stabilization

> [论文](https://arxiv.org/pdf/2102.01279.pdf) [项目](https://zhmeishi.github.io/dvs/)  提出了一种利用传感器数据（陀螺仪）和图像内容（光流）通过无监督学习来稳定视频的深度神经网络（DNN）。该网络将光流与真实/虚拟摄像机姿态历史融合成关节运动表示。接下来，LSTM块推断出新的虚拟相机姿势，并使用该虚拟姿势生成一个扭曲网格，以稳定帧。提出了一种新的相对运动表示方法和多阶段的训练过程来优化模型。据我们所知，这是第一个DNN解决方案，采用传感器数据和图像稳定。我们通过烧蚀研究验证了所提出的框架，并通过定量评估和用户研究证明了所提出的方法优于现有的替代解决方案。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210528095802371.png)

<iframe width="1182" height="665" src="https://www.youtube.com/embed/LF_JVdUFIw8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

> deep-FVS概述。在给定输入视频的情况下，我们首先去除了OIS转换，提取原始光流。我们还从陀螺仪获得真实的相机姿态，并将其转换为相对四元数。一个二维卷积编码器将光流嵌入到一个潜在的表示，然后将其与真实和虚拟摄像机的姿态连接起来。该关节运动表示被馈送到LSTM单元和FC层，以预测新的虚拟相机姿态为四元数。最后，基于OIS和虚拟摄像机姿态对输入帧进行扭曲，生成稳定帧



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/videostability/  

