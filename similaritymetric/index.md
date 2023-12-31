# SimilarityMetric


**level**: CVPR
**author**:  Paul-Edouard Sarlin
**date**:   2020
**keyword**:

- features matching; data association;

> Sarlin, Paul-Edouard, et al. "Superglue: Learning feature matching with graph neural networks." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

------

# Paper: SuperGlue

<div align=center>
<br/>
<b>SuperGlue: Learning Feature Matching with Graph Neural Networks</b>
</div>


#### Summary

1. demonstrates the power of attention-based graph neural networks for local feature matching:
   1. self-attention, boosts the receptive field of local descriptors;
   2. cross-attention, enables cross-image communication and is inspired by the way humans look back-and-forth when matching images;
2. SuperGlue learn the priors over geometric transformations and regularities of the 3D world through end-to-end training from image pairs;

#### Proble Statement

- large viewpoint and lighting changes, occlusion, blur, and lack of texture are factors that make 2D-to-2D data association particularly challenging;

previous work:

- **Local Feature Matching: **
  - Traditional： detecting interest points;computing visual descriptors, like sift, surf; matching these with NN search; filtering incorrect matches; estimating a geometric transformation;
  - Recent works: learning better sparse detectors and local descriptors using CNNs, look at a wider context using regional features or log-polar patches; <font color=red> ignore the assignment structure and discard visual information</font>
  - dense matching[46] or 3D point clouds [65]
- **Graph matching:** 
  - [9] learn the cost of the optimization for a simpler linear assignment;
  - optimal transport [63] a generalized linear assignment with an efficient yet simple approximate solution, the Sinkhorn algorithm;
- **Deep learning for sets:**  self-attention can be seen as instance of a Message Passing Graph Neural Network on a complete graph, apply attention to graphs with multiple types of edges;

#### Methods

- **Problem Formulation**: 

  - A, B : two images;
  - $p$:  a set of keypoint position;   $P_i:=(x,y,c)_i$:  x,y coordinate and detection confidence c;
  - $d_i\epsilon R^D$:  associated visual descriptors, extracted by a CNN like SuperPoint, or traditional descriptors like SIFT;
  - $f_i\epsilon R^D$: matching descriptors;
  - $\epsilon_{self}$: the intra-image edges;  $\epsilon_{cross}$: the inter-image edges;
  - $P\epsilon [0,1]^{M*N}$: the partial soft assignment matrix;

  $$
  P1_N\leq1_M\\
  P^T1_M\leq1_N
  $$

  

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917093314039.png)

- **Attentional Graph Neural Network:** sift through tentative matching keypoints, examine each, and look for contextual cues that help disambiguate the true match from other self-similarities;

- **Keypoint Encoder:** $x_i^{(0)}=d_i+MLP_{enc}(p_i)$;

- **Multiplex Graph Neural Network:** 

  - $\varepsilon \epsilon \{\varepsilon_{self},\varepsilon_{cross}\}$: the connect edge set;

  - $m_{\varepsilon->i}$:  the result of the aggregation from all keypoints $\{j:(i,j)\epsilon \varepsilon\}$

  - $x_i^{A(l)}$: the intermediate representation for element i in image A at layer l ;

  - updation:
    $$
    x_i^{A(l+1)}=x_i^{A(l)}+MLP([x_i^A(l)||m_{\varepsilon->i}])
    $$
  
- **Attentional Aggregation:** a representation of i, the query $q_i$, retrieves the values $v_j$ of some elements based on their attributes, the keys $k_j$; $\alpha=Softmax_j(a_i^Tk_j)$: the attention weight;

  - $$
    m_{\varepsilon->i}=\sum_{j:(i,j)\epsilon \varepsilon}\alpha_{ij}V_j \\
    f_i^A=W*x_i^{A(L)}+b, \forall i\epsilon A;
    $$

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917101808583.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917105811099.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917090251965.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917090538654.png)

#### Evaluation

> The precision (P) is the average ratio of the number of correct matches over the total number of estimated matches. The matching score (MS) is the average ratio of the number of correct matches over the total number of detected keypoints
>
> Poses are computed by first estimating the essential matrix with OpenCV’s findEssentialMat and RANSAC with an inlier threshold of 1 pixel divided by the focal length, followed by recoverPose

**【 Homography estimation】**

- Dataset: 1M distractor images in Oxford and Paris dataset[42];   DLT: directed linear transformation

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917102406299.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917105710761.png)

**【Indoor Position】**

- lack of texture, the abundance of self-similarities, the complex 3D geometry of scenes, and large view changes;
- Dataset: ScanNet[13],  indoor dataset composed of monocular sequences with ground truth poses and depth images and well-define training,  validation, and test splits corresponding to different scenes.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917103336668.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917105550177.png)

**【Outdoor position estimation】**

- dataset: PhotoTourism dataset: YFCC100M dataset[56] , with ground truth poses and sparse 3D models obtained from SfM tool;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917103743941.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917104006795.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917105632954.png)

**【Ablation of SuperGlue】**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917104118772.png)

#### Notes <font color=orange>去加强了解</font>

  - https://github.com/magicleap/SuperGluePretrainedNetwork
  - superpoint18 
  - 46/65 匹配方法
  - 图网络中: message passing formulation [23,  4]
  - 网络内部细节待细致阅读，以及实验测试结果
  - 不清楚这个pose evaluate 是怎么评估的

**level**: 
**author**: University of California, Berkeley
**date**: 2003
**keyword**:

- Similarity

> Xing, E. P., Jordan, M. I., Russell, S. J., & Ng, A. Y. (2003). Distance metric learning with application to clustering with side-information. In *Advances in neural information processing systems* (pp. 521-528).

------

# Paper:  Distance Metric Learning

<div align=center>
<br/>
<b>Distance Metric Learning with Application to Clustering with side-information</b>
</div>


#### Summary

1. 

#### Research Objective

  - **Application Area**:
- **Purpose**:  given examples of similar pairs of points in $R^n$ , learns a distance metric over $R^n$ that respects these relationships.

#### Methods

- **Problem Formulation**: suppose a user indicates that certain points in an input space $R^n$ are considered by them to be similar, can we automatically learn a distance metric over $R^n$ that respects these relationships ?

- **system overview**:

**【Module one】Learning Distance Metrics**

- $\{x_i\}_{i=1}^m\epsilon R^n$  :datasets;
- $D$: a set of pairs of points known to be dissimilar

$$
d(x,y)=d_A(x,y)=||x-y||_A=sqrt[(x-y)^TA(x-y)]\\
min_A \sum_{(x_i,x_j)\epsilon S}||x_i-x_j||^2_A\\
s.t. \sum_{(x_i,x_j)\epsilon D}||x_i-x_j||_A \geq 1\\
A\geq0.
$$



- **the case of diagonal A**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200913102622569.png)

- **The case of Full A**

  - gradient step: $A: =A+\alpha \nabla _ag(A)$
  - project A into the sets :

  $$
  C_1=\{A: \sum_{(x_i,x_j)\epsilon S}||x_i-x_j||_A^2 \leq 1\} \\
  C_2=\{A: A\geq 0\}
  $$

  

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200913102806610.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200913103321326.png)

#### Notes <font color=orange>去加强了解</font>

- 后买你的公式理解部分不太明白
- 后期那本书可以好好看看



### Resource

- https://mp.weixin.qq.com/s/2h_EfNTgLknY0fE30rY29w


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/similaritymetric/  

