# HumanPoseProject


#### 1. **[ residual_pose](https://github.com/idiap/residual_pose)**

- Hourglass model for multi-person `2D pose estimation from depth images`.
- Our regressor NN architecture for `3D human pose estimation`.
- 3D pose prior for `recovering from 2D missed detections`.
- Tranined models for `2D and 3D pose estimation`.
- Code for obtaining `2D and 3D pose from a depth image`.
- 11 months ago.    star12

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210819141418882.png)

#### 2. [depth_human_synthesis](https://github.com/idiap/depth_human_synthesis)

> - We have created a collection of 24 human characters, 12 men and 12 women, with Makehuman. The characters exhibit different body features such as height and weight. 
> - By` synthesizing images with people` we have the benefit that 2D and 3D body landmark annotations are extracted automatically during the rendering process. 

![](https://github.com/idiap/depth_human_synthesis/raw/main/imgs/depth_generation.gif)

#### 3. [A2J](https://github.com/zhangboshen/A2J)

> Xiong F, Zhang B, Xiao Y, et al. A2j: Anchor-to-joint regression network for 3d articulated pose estimation from a single depth image[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 793-802. [[pdf](https://arxiv.org/abs/1908.09999)]
>
> - propose a simple and effective approach termed A2J, for` 3D hand and human pose estimation from a single depth image`. Wide-range evaluations on 5 datasets demonstrate A2J's superiority.
> -  anchor points able to capture global-local spatial context information are densely set on depth image as local regressors for the joints. They contribute to predict the positions of the joints in ensemble way to enhance generalization ability. 
> - star 208, 12 month ago;

![pipeline](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/A2Jpipeline.png)

##### .1.  [NYU](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) hand pose dataset:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/NYU_1.png)

##### .2. [ITOP](https://www.alberthaque.com/projects/viewpoint_3d_pose/) body pose dataset:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/ITOP_1.png)

#### 4. [depth-pose-estimation](https://github.com/Mostro-Complexity/depth-pose-estimation)

> - Train models to predict body parts or joints. Using depth images to recognise the human pose;
>
> - 3 years ago; star 7;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/header.png)

#### 5. [b_depth_estimator](https://github.com/echen4628/b_depth_estimator)

> - baseline depth estimator using single image and bounding box
> - 通过相机数学相似模型，在已知物体实际大小的时候，估计深度信息。

#### 6. [monodepth2](https://github.com/nianticlabs/monodepth2)

> Godard C, Mac Aodha O, Firman M, et al. Digging into self-supervised monocular depth estimation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 3828-3838.
>
> -  (i) a minimum reprojection loss, designed to robustly handle occlusions, (ii) a full-resolution multi-scale sampling method that reduces visual artifacts, and (iii) an auto-masking loss to ignore training pixels that violate camera motion assumptions.
> - Per-pixel ground-truth depth data is challenging to acquire at scale
> - stars 2.5k, Fork: 638;

<p align="center">
  <img src="https://github.com/nianticlabs/monodepth2/raw/master/assets/teaser.gif" alt="example input output gif"/>
</p>

#### 7. [S2R-DepthNet](https://github.com/microsoft/S2R-DepthNet)

> Chen X, Wang Y, Chen X, et al. S2R-DepthNet: Learning a Generalizable Depth-specific Structural Representation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 3034-3043.
>
> - the first to explore the learning of a `depth-specific structural representation`, which captures the essential feature for depth estimation and ignores irrelevant style information. 
> - Our S2R-DepthNet (Synthetic to Real DepthNet) can be well generalized to unseen real-world data directly even though it is only trained on synthetic data. S2R-DepthNet consists of: a) a` Structure Extraction (STE) module which extracts a domaininvariant structural representation` from an image by disentangling the image into domain-invariant structure and domain-specific style components, b) `a Depth-specific Attention (DSA) module, which learns task-specific knowledge to suppress depth-irrelevant structures for better depth estimation and generalization`, and c) a depth prediction module (DP) to` predict depth from the depth-specific representation.`
> - star 79, 3 months ago;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/overview.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/intro.PNG)

#### 8. [PENet_ICRA2021](https://github.com/JUGGHM/PENet_ICRA2021)

> Hu, M., Wang, S., Li, B., Ning, S., Fan, L., & Gong, X. (2021). Towards Precise and Efficient Image Guided Depth Completion. *arXiv e-prints*, arXiv-2103.  ICRA 2021.
>
> - 6 months ago; Star 93;
> - `Image guided depth completion` is the task of generating a` dense depth map from a sparse depth map and a high quality image`
> -  This paper proposes a two-branch backbone that consists of a color-dominant branch and a depth-dominant branch to exploit and fuse two modalities thoroughly. More specifically, one branch` inputs a color image and a sparse depth map to predict a dense depth map`. The other branch takes as `inputs the sparse depth map and the previously predicted depth map, and outputs a dense depth map as well`. The depth maps predicted from two branches are complimentary to each other and therefore they are adaptively fused.
> -  a simple geometric convolutional layer to encode 3D geometric cues. The geometric encoded backbone conducts the fusion of different modalities at multiple stages, leading to good depth completion results.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210819142532697.png)

#### 9. [depthai_movenet](https://github.com/geaxgx/depthai_movenet)

> - A convolutional neural network model that runs on` RGB images` and predicts [human joint locations](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#coco-keypoints-used-in-movenet-and-posenet) of a single person. Two variant:` Lightning and Thunder,` the latter being slower but more accurate.
> - The cropping algorithm determines from the body detected `in frame N, on which region of frame N+1 the inference will run.` The mode (Host or Edge) describes where this algorithm runs :
>   - in Host mode, the cropping algorithm runs on the host cpu. Only this mode allows images or video files as input. The flow of information between the host and the device is bi-directional: in particular, the host sends frames or cropping instructions to the device;
>   - in Edge mode, the cropping algorithm runs on the MyriadX. So, in this mode, all the functional bricks of MoveNet (inference, determination of the cropping region for next frame, cropping) are executed on the device. The only information exchanged are the body keypoints and optionally the camera video frame.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/movenet_nodes.png)

- demo: codehttps://github.com/geaxgx/openvino_movenet

<div>
    <img src="https://github.com/geaxgx/depthai_movenet/raw/main/img/dance.gif" />
</div>

##### .1. Aphabet classification

- https://github.com/geaxgx/depthai_movenet/tree/main/examples/semaphore_alphabet

<div>
    <img src="https://github.com/geaxgx/depthai_movenet/raw/main/examples/semaphore_alphabet/medias/semaphore.gif" />
</div>

##### .2. Yoga  classification

- https://github.com/geaxgx/depthai_movenet/raw/main/examples/yoga_pose_recognition/medias/yoga_pose.gif
-  **[ Realtime-Action-Recognition](https://github.com/felixchenfy/Realtime-Action-Recognition)** 该项目也是通过利用人体谷歌点进行粗粒度动作分类的。

<div>
    <img src="https://github.com/geaxgx/depthai_movenet/raw/main/examples/yoga_pose_recognition/medias/yoga_pose.gif" />
</div>

#### 10. [arcore-depth-lab](https://github.com/googlesamples/arcore-depth-lab)

> **Depth Lab** is a set of ARCore Depth API samples that `provides assets using depth for advanced geometry-aware features in AR interaction and rendering.  `[Depth API overview](https://www.youtube.com/watch?v=VOVhCTb-1io) [**ARCore Depth API**](https://developers.google.com/ar/develop/unity/depth/overview) is enabled on a subset of ARCore-certified Android devices.

<div>
    <img src="https://github.com/googlesamples/arcore-depth-lab/raw/master/depthlab.gif" />
</div>
#### 11. **[ RGBD-Face](https://github.com/liggest/RGBD-Face)**  **[hifi3dface](https://github.com/tencent-ailab/hifi3dface)**

> Lin X, Chen Y, Bao L, et al. High-fidelity 3d digital human creation from rgb-d selfies[J]. arXiv preprint arXiv:2010.05562, 2020.
>
> - 使用RGBD数据，得到粗糙的静态面部模型

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210822163814188.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210822163832544.png)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/humanposeproject/  

