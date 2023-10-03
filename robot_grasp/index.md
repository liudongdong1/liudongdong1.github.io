# robot_grasp


> **2D planar grasp** means that the target object lies on a plane workspace and the grasp is constrained from one direction. The essential information is simplified from 6D into 3D, which are the` 2D in-plane positions and 1D rotation angle.` There exist methods of **evaluating grasp contact points** and methods of **evaluating grasp oriented rectangles**.

> **6DoF grasp** means that the gripper can grasp the object from various angles in the 3D domain, and the essential 6D gripper pose could not be simplified. Based on whether the grasp is conducted on the complete shape or on the single-view point cloud, methods are categorized into **methods based on the partial point cloud** and **methods based on the complete shape**. Methods based on the partial point cloud contains **methods of estimating candidate grasps** and **methods of transferring grasps** from existing grasps database. Methods based on the complete shape contains **methods of estimating 6D object pose** and **methods of shape completion**. Most of current 6DoF grasp methods aim at known objects where the grasps could be precomputed manually or by simulation, and the problem is thus transformed into a **6D object pose estimation** problem.

> most of the robotic grasping approaches require **the target object’s location** in the input data first. This involves three different stages: **object localization without classification**, **object detection** and **object instance segmentation**. 

- https://github.com/GeorgeDu/vision-based-robotic-grasping

### 1. 机器人Baxter

- https://www.youtube.com/watch?v=JWBqXLHlqjE   37万元
- how it work: https://www.youtube.com/watch?v=gXOkWuSCkRI
- move it: https://www.youtube.com/watch?v=0og1SaZYtRc
- move it 各种抓取机器人： https://www.youtube.com/watch?v=7KvF7Dj7bz0
- 说明文档： chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.ohio.edu%2Fmechanical-faculty%2Fwilliams%2Fhtml%2FPDF%2FBaxterKinematics.pdf

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/007Ys3FFgy1gpucfvtvrtj30fq0l80yp.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102104626410.png)

- 安全因素： 机械臂安全，但是机器人的工具不太安全；
- 碰撞检测原理大多基于动力学模型与外力观测器，速度越快，误差越大
- 需要非常合适的，人与机器人一同工作的环境，但是大部分工厂对机器人的需求是调试好之后，就一直在不停的跑。

> Ren R, Rajesh M G, Sanchez-Riera J, et al. Grasp-Oriented Fine-grained Cloth Segmentation without Real Supervision[J]. arXiv preprint arXiv:2110.02903, 2021. 

------

# Paper: 

<div align=center>
<br/>
<b>Grasp-Oriented Fine-grained Cloth Segmentation without Real Supervision</b>
</div>


#### Summary

1. tackle the problem of fine-grained region detection in deformed clothes using only a depth image, introduce a U-net based network to segment and label these parts.
2. defy the limitations of the synthetic data, and propose a multilayered domain adaptation strategy that does not use real annotations at all.

#### Research Objective

- **Purpose**:   manipulating highly deformable objects such as cloth

#### previous work:

- finding suitable grasping points for towels, or t-shirts, pants, sweaters, according to the geometric cues.
- classify cloth deformation to indirectly infer the grasping points.

#### Methods

- **system overview**:

> intruduce a pipline for fine-grained semantic segmentation of depth maps of cloths.
>
> propose a multi-layered domain adaptation strategy to train the proposed network with only synthetic GT labels, which can then be applied to real depth maps.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211106160816108.png)

> the model consist of two main branches, a `U-Net that segments the cloth parts` and `a multi-layered domain adaption classifier that helps to reduce the domain gap between real and synthetically generated depth maps`.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211106162315246.png)

![Loss Function](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211106163024432.png)

#### Evaluation

  - **Environment**:   

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211106163204508.png)

> visualization of the results, where the background, cloth body, edges are denoted in black, green, blue respectively.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211106163303339.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/robot_grasp/  

