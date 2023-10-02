# HumanPosePaper


> Zhang S, Zhang Y, Bogo F, et al. Learning motion priors for 4d human body capture in 3d scenes[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 11343-11353.  [[pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2108.10399.pdf)] [[code](https://github.com/sanweiliti/LEMO)]

------

# Paper: LEMO

<div align=center>
<br/>
<b>Learning Motion Priors for 4D Human Body Capture in 3D Scenes</b>
</div>


#### Summary

1. a `marker-based motion smoothness prior` and a `contact-aware motion infillter` wihcih is fine-tuned per-instance in a self-supervised fashion.
2. a novel `marker-based moiton smoothness prior` that `encodes the whole-body motion` in a learned latent space, which can be easily plugged into an optimization pipeline.
3. a novel` contact-aware moiton infiller` that can be adapted to per-test-instance via self-supervised learning
4. a new` optimization pipeline` that explores both learned motin priors and the physics-inspired contact friction term for scene-aware human motion capture.

previous work:

- human motion recovery from RGB(D) sequences:
  -  adopting `skeleton/joint-based representations` for the body. not adequately model the 3D shape of the body and body-scene interactions.
  - use parametric 3D human models to abtain complete `3D body meshes` from multi-view or monocular RGB(D) sequences.
- Person-scene interaction: 
  - obtain scene constraints for body pose estimation by reconstructing the scene in 3D with multiple unsynchronized moving cameras.
- Human motion priors: priors for smooth and natural motion
  - body joint velocity or acceleration
  - regress body joints and foot-ground contact from images to conduct physics-based trajectory ooptimization.

#### Methods

- **system overview**:

> provided a scene mesh and RGBD sequence with body occlusion, recovers a realistic global motion, with natural person-scene interactions.  The markers trajectories(left) and accelerations(right) of each stage are shown at the bottom, as well as a walking sequence from AMASS.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211103102647565.png)

#### Evaluation

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102185206231.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102185224112.png)

> Wang, Yue, et al. "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries." *arXiv preprint arXiv:2110.06922* (2021). [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fopenreview.net%2Fpdf%3Fid%3DxHnJS2GYFDz) [code](https://github.com/WangYueFt/detr3d)
>

------

# Paper: DETR3D

<div align=center>
<br/>
<b>DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries.</b>
</div>

#### Summary

1. present a streamlined 3D object detection model from RGB images, our methods fuses information from all the camera views in each layer of computation.
2. manipulates predictions directly in 3D space, the architecture extracts 2D features from multiple camera images and then uses a sparse set of 3D object queries to index into these 2D features, linking 3D positions to multi-view images using camera transformation matrices.
3. model makes a bounding box prediction per object query, using a set-to-set loss to measure the discrepancy between the ground-truth and the prediction.
4. DETR3D starts link 2D feature extraction and 3D object prediction via geometric back-projection with camera transformation matrices. The methods start from a sparse set of object priors, shared across the dataset and learned end-to-end.
   1. use back-project a set of reference points decoded from these object priors to each camera and fetch the corresponding image features extreacted by a ResNet backbonne, to gather scene-specific information.
   2. the features collected from the image features of the reference points then interact with each other through a multi-head self-attention layer.

#### Methods

- **system overview**:

> the inputs to the model are a set of multi-view images, which are encoded by a ResNet and FPN.
>
> the model operates on a set of sparse object queries in which each query is deocded to a 3D reference point.
>
> 2D features are transformed to refine the object queries by projecting the 3D reference point into the image space.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211106154832864.png)

> Jiang W, Xue H, Miao C, et al. Towards 3D human pose construction using WiFi[C]//Proceedings of the 26th Annual International Conference on Mobile Computing and Networking. 2020: 1-14.  [[video](https://www.youtube.com/watch?v=puU4EvBTPxA)]

------

# Paper: WiPose

<div align=center>
<br/>
<b>Towards 3D Human Pose Construction Using WiFi</b>
</div>


#### Summary

1. present WiPose, the first 3D human pose construction framework using commercial WiFi devices, which can reconstruct 3D skeletons composed of the joints on both limbs and torso of human body with an average error of 2.83cm.
2. WiPose can encode the prior knowledge of human skeleton into the posture construction process to ensure the estimated joints satisfy the skeletal structure of the human body.
3. WiPose takes as input a 3D velocity profile which can capture the movements of the whole 3D space, and thus separate posture-specific features from the static objects in the ambient environment.
4. WiPose employs a recurrent neural network and a smooth loss to enforce smooth movements of the generated skeletons.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211129152221849.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211129152839987.png)

- transformed th eraw CSI data extracted from M distributed antennas into a sequence of input data.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211129152903257.png)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/humanposepaper/  

