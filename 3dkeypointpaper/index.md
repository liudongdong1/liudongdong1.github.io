# 3DkeyPointPaper


**level**: 2020, CCF_A  CVPR 
**author**: Yang You, Shanghai Jiao Tong University
**date**:  2020
**keyword**:

- 3D keypoint

> You, Y., Lou, Y., Li, C., Cheng, Z., Li, L., Ma, L., ... & Wang, W. (2020). KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 13647-13656).

------

# Paper: KeypointNet

<div align=center>
<br/>
<b>KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations</b>
</div>


#### Summary

1. present keypointnet, the first large-scale and diverse 3D keypoint dataset that contains 103450 keypoints and 8234 3D models from 16 object categrories;
2. propose a novel methods to aggregate these keypoints automatically through minimization of a fidelity loss;
3. propose two large-scale keypoint prediction tasks: keypoint saliency estimation, and keypoint correspondence estimation; and experiment including point cloud, graph, voxel and local geometry based keypoint detection.
4.  In order to generate ground-truth keypoints from raw human annotations where identification of their modes are non-trivial

#### Research Objective

  - **Application Area**: object matching, object tracking, shape retrieval, registration, pose estimation, matching, segmentation; which is invariant to rotations, scales and other transformations;
- **Purpose**:  

#### Proble Statement

- few 3D datasets focusing on the keypoint representation of an object;
- different people may annotate different keypoints  which need to identify the consensus and patterns;
- predefined distance threshold fail to identify closely spaced keypoints;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021091359516.png)

previous work:

- **Detection of keypoints:** 
  - traditional methods: 3D Harris, HKS, Salient Points, Mesh Saliency etc extract geometric features as local descriptors,  but only consider the local geometric information without semantic knowledge.
  - DNN methods: SyncSpecCNN don't handle rotations well.
- **Keypoint Datasets:** 
  - Keypoints for human skeletons:  MPII human pose dataset, MSCOCO keypoint challenge, PoseTrack;
  - Animals: PUB provides 15 part locations on 11788 images from 200 bird categories
  - 3D objects: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021093151042.png)

#### Methods

- **Problem Formulation**:

given  a valid annotation from c-th person, the keypoint set is:![image-20201021093552067](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021093552067.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021093404099.png)

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021095034963.png)

【Dataset Visualization】

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021093300132.png)

【**Keypoint Saliency**】 assume that the each annotation is allowed to be erroneous within small region. $\Phi$ is the Gaussian kernel, and $Z$ is the normalization function;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021094235010.png)

【**Ground Truth Keypoint Generation】**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021094726837.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021094746679.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021094938673.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021095438254.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021095502755.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021095556538.png)

#### Notes <font color=orange>去加强了解</font>

  - [ ] https://github.com/qq456cvb/KeypointNet.
  - [ ] PointNet, RSCNN, PointConv, SpiderCNN, DGCNN, GrapCNN 3D 点检测相关模型



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/3dkeypointpaper/  

