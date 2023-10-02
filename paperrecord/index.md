# PaperRecord


clothes classification, attribute prediction, clothing item retrieval.

- clothes have large variations in style, texture, and cutting.
- clothing items are frequently subject to `deformation and occlusion`.
- clothes images often exhibit serous variations when they are taken under different scenarios.

> Liu, Ziwei, et al. "Deepfashion: Powering robust clothes recognition and retrieval with rich annotations." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

------

# Paper: Deepfashion

<div align=center>
<br/>
<b>Deepfashion: Powering robust clothes recognition and retrieval with rich annotations
</b>
</div>

#### Summary

- introduce DeepFashion, a large-scale clothes dataset with comprehensive annotations.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223160104102.png)

- propose FashionNet,which learns `clothing features` by jointly `predicting clothing attributes and landmarks`. the landmarks are human labeled:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223160228492.png)

-  FashionNet work pipeline:   can be used to predict the landmark of clothes.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223160438816.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223160510415.png)

##### Relative

- (a) Additional landmark locations improve clothes recognition. (b) Massive attributes lead to better partition of the clothing feature space.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223155654297.png)

- Relative datasets

![image-20210223155820943](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223155820943.png)

- clothing recognition and retrieval: hand-craft features like sift, hog, color histogram.

> Yamaguchi, Kota, et al. "Parsing clothing in fashion photographs." *2012 IEEE Conference on Computer vision and pattern recognition*. IEEE, 2012.

------

# Paper: ParsingClothing

<div align=center>
<br/>
<b>Parsing clothing in fashion photographs
</b>
</div>

#### Summary

- a novel dataset for studying clothing parsing, consisting of 158.235 fashion photos with associated text annotations, and web-based tools for labeling.

- A methods to parse pictures of people into their constituent garments. And the clothes pipelines are as follows:

  - (a) Parsing the image into Superpixels [1],

  -  (b) Original pose estimation using state of the art flexible mixtures of parts model [27]. 

  - (c) Precise clothing parse output by our proposed clothing estimation model (note the accurate labeling of items as small as the wearer’s necklace, or as intricate as her open toed shoes).

  -  (d) Optional reestimate of pose using clothing estimates (note the improvement in her left arm prediction, compared to the original incorrect estimate down along the side of her body).

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223161459743.png)

- Prototype garment search application results. Query photo (left column) retrieves similar clothing items (right columns) independent of pose and with high visual similarity.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223161201232.png)

- dataset create funciton:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223162134723.png)

##### Scenarios suitable

- pose estimation:  incorporated mixtures of parts to obtain state of the art results, and extending the approach to incorporate clothing estimations in models for pose identification.

> Yu, Tao, et al. "Simulcap: Single-view human performance capture with cloth simulation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2019.

------

# Paper: Simulcap

<div align=center>
<br/>
<b>SimulCap: Single-View human performance capture with cloth simulation
</b>
</div>

#### Summary

- proposes a new method for live free viewpoint human performance capture with dynamic details(eg. cloth wrinkles) using single RGBD, simulate plausible cloth dynamics and cloth-body interactions even in the occluded regions.
  - can predict the occluded cloth part more accurately than the commonly used surface skinning, and non-rigid warping.
  - the observed cloth details, suh as wrinkles, can be reconstructed by formulating data fitting as a physical process.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223162822865.png)

![pipelines](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223163215023.png)

- Cloth simulation: 这部分没有看，里面有

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223163503575.png)

> Patel, Chaitanya, Zhouyingcheng Liao, and Gerard Pons-Moll. "Tailornet: Predicting clothing in 3d as a function of human pose, shape and garment style." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

------

# Paper: Tailornet

<div align=center>
<br/>
<b>Tailornet: Predicting clothing in 3d as a function of human pose, shape and garment style
</b>
</div>

#### Summary

- present TailorNet, a neural model which `predicts clothing deformation in 3D as a function of three factors: pose, shape and style (garment geometry)`, while retaining wrinkle detail.
- The first joint model of `clothing style, pose and shape variation`, which is simple, easy to deploy and fully differentiable for easy integration with deep learning.
-  decomposition of mesh deformations into low and high-frequency components, which coupled with a mixture model, allows to retain high-frequency wrinkles.
- Pipelines: Overview of our model to `predict the draped garment with style γ on the body with pose θ and shape β`. 
  - Low frequency of the deformations are predicted using a single model. 
  - High frequency of pose dependent deformations for K prototype shape-style pairs are separately computed and mixed using a RBF kernel to get the final high frequency of the deformations.
  -  The low and high frequency predictions are added to get the unposed garment output, which is posed to using standard skinning to get the garment

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223164238842.png)

##### Relative

- animation of clothing
  - Physics Based Simulation (PBS): 
  - Data-driven cloth models:
  - Pixel based models:

![Relative work](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210223164357336.png)

> Chance, Greg, et al. "“elbows out”—predictive tracking of partially occluded pose for robot-assisted dressing." *IEEE Robotics and Automation Letters* 3.4 (2018): 3598-3605.

------

# Paper: elbows out

<div align=center>
<br/>
<b>“elbows out”—predictive tracking of partially occluded pose for robot-assisted dressing
</b>
</div>

#### Summary

> Hsiao, Wei-Lin, and Kristen Grauman. "ViBE: Dressing for diverse body shapes." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

------

# Paper: ViBE

<div align=center>
<br/>
<b>ViBE: Dressing for Diverse Body Shapes
</b>
</div>


#### Summary

- introduce ViBE, a Visual Body-aware Embedding that captures clothing's affinity with different body shapes,`given an image of a person, the proposed embedding identifies garments that will flatter her specific body  shape`.

- Example categories of body shapes, with styling tips and recommended dresses for each, according to fashion blogs

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210224122356267.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210224122312681.png)

#### Relative 

- Trained largely from images of slender fashionistas and celebrities (bottom row), existing methods ignore body shape’s effect on clothing recommendation and exclude much of the spectrum of real body shapes.
- Our proposed embedding considers diverse body shapes (top row) and learns which garments flatter which across the spectrum of the real population. `address the influence of body shape on garment compatibility or style`.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210224104055992.png)

- **Fashion Styles and compatibility:** 
  - recognition problems, like `matching items seen on the street to a catalog searching for products`, or `parsing an outfit into garments`.
  - style-meta-patterns in what people wear with visual attri.
- **Virtual try on clothing retargeting:** 
  - estimate garment draping on a 3D images scan;
  - retarget styles for people in 2D images or video, or render a virtual try-on with sophisticated image generation.
- **Body and garment shape estimation:** 
  - estimating people and clothing's 3D geometry from 2D RGB images;
- **Sizing clothing:**  given a product and the purchase history of a user, these methods predict whether a given size will be too large, small or just right.



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/paperrecord/  

