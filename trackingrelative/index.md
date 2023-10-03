# TrackingRelative


**level**: CVPR
**author**: Boqiang Xu (University of Chinese Academy of Sciences), Lingxiao He(AI Research of JD)
**date**: 2020
**keyword**:

- Person re-identification

------

# Paper: Black Re-ID

<div align=center>
<br/>
<b>Black Re-ID: A Head-shoulder Desccriptor for the problem of Person Re-Identification</b>
</div>

#### Summary

1. propose the study of the Black Re-ID problem and establish the first Black-reID dataset;
2. propose the head-shoulder adaptive attention network(HAA), which make use of the head-shoulder information to support person re-identification through the adaptive attention module, and can be integrated with the most current Re-ID framework and is end-to-end trainable;
3. both effective for Black Re-ID problem but also valid in similar clothing;

#### Research Objective

  - **Application Area**:
- **Purpose**:  retrieving the same person from overlapping cameras;

#### Proble Statement

previous work:

- most of pervious work extract features in terms of the attributes of clothing(color, texture), but it's common for people to `wear black clothes or be captured by surveillance systems in low light`;
- **Person Re-ID:** treated as classification task, aiming at dividing person with same identity. 
  - pose-based Re-ID, which uses an off-the-shelf pose estimator to extract the pose information for aligning body parts or generating person images; (network bigger and slower)
  - part-based Re-ID, which slices image or global feature into several horizontal grids, training individually and assembling for a discriminative person representation; (sensitive to the pose variations and occlusions)
  - leverage local information with attention maps, pay more attention to the regions of IOTs, and robust to the background clutter.
- **Head-shoulder information:** these features such as haircut, complexion or appearance offer abundant discriminative information;

#### Methods

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902100811.png)

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902100620.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902100712.png)

**【Adaptive Attention】** to determinate the global and local feature weights by distinguishing input types;

#### Evaluation

**【Test One】 Accuracy comparison**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902101215.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902101319.png)

**【Test two】The Impact of Adaptive model**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902101537.png)

**【Test three】Ablation Study of the GeM Pooling**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902101600.png)

**【Test four】Performance comparison of global and head-shoulder features**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902101630.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902101706.png)

#### Notes <font color=orange>去加强了解</font>

  - Code avaible: https://github.com/xbq1994/.

**level**: 
**author**: Jinlong Peng( Tencent), Fudan University, Nara Institute of Science and Technology
**date**:  2020
**keyword**:

- MOT

> Peng, J., Wang, C., Wan, F., Wu, Y., Wang, Y., Tai, Y., ... & Fu, Y. (2020). Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking. *arXiv preprint arXiv:2007.14557*.

------

# Paper: Chained-Tracker

<div align=center>
<br/>
<b>Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-object Detection and  Tracking</b>
</div>

#### Summary

1. propose an end-to-end online MOT model, to optimize object detection, feature extraction and data association simultaneously.
2. CTracker is the first method that converts the challenging data association problem to a pair-wise object detection problem;
3. design a joint attention module to highlight informative regions for box pair regression and the performance of CTracker is further improved; 

#### Proble Statement

- **existence of occlusions; object trajectory overlap; possibly challenging background**
- propose a novel  on-line model CTracker, which unifies object detection, feature extraction and data association into a single end-to-end model;

previous work:

- based on tracking-by-detection paradigm, contains three sequential subtasks: Object detection, feature extraction and data association;  <font color=red>lead to local optima and more computation cost, discards the temporal relationships of adjacent frames</font>
- Re-identification and attention: the former extracts more robust features for data association; the latter hep model focused; <font color=red>greatly increase the model complexity and computational cost</font>
- **Detection-based MOT Methods: ** detection model and tracking model are completely independent, which is complex and time-consuming;
- **Partially End-to-end MOT Methods:** 
- **Attention-assistant MOT Methods:**
  - Chu et. al[12] introduced a Spatial-Temporal Attention Mechanism to handle the tracking drift caused by the occlusion and interaction among targets.
  - [14] utilized an attention-based appearance model to solve the inter-object occlusion;

#### Methods

- **system overview**:

> takes adjacent frame pairs as input to perform joint detection and tracking in a single regression model that simultaneously regress the paired bounding boxes for targets that appear in both of the two adjacent frames;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200902104022.png)

> - end-to-end model using adjacent frame pair as input and generating the box pair representing the same target;
> - convert the challenging corss-frame association problem into pair-wise object detection problem;

#### Notes <font color=orange>去加强了解</font>

  - code:  github.com/pjl1995/CTracker

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/trackingrelative/  

