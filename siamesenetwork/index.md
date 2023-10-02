# SiameseNetwork


> Siamese Network 是一种神经网络的框架，用于评估两个输入样本的相似度，而不是具体的某种网络，就像seq2seq一样，具体实现上可以使用RNN也可以使用CNN。

# 1. Siamese Network

## 1.Paper 

**level**: 
**author**: Sumit Chopra(Courant Institute of Mathematical Sciences), Raia Hadsell(New York University), Yann LeCun
**date**: 
**keyword**:

- similarity metric

> Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to face verification." *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*. Vol. 1. IEEE, 2005.  Cited by 2402

------

### Paper: Similarity Metric

<div align=center>
<br/>
<b>Learning a Similarity Metric Discriminatively, with Application to Face
Verification
</b>
</div>



#### Summary

1. present a method for training a similarity metric from data, which is used for recognition or verification applications where the number of categories is very large and not known during training, and where the number of training samples for a single category is very small.
2. learn a function that maps input patterns into a target space such that the $L_1$ norm in the target space approximates the semantic distance in the input space;

#### Proble Statement

- traditional approaches to classification using discriminative methods require that all categories be known in advance, also require that all the categories available for all categories.
  - computing a similarity metric between the pattern to be classified or verified and a library of stored prototypes;
  - use non-discriminative probablilistic methods in a reduce-dimension space,

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200912230425.png)
$$
E_w(X_1,X_2)=||G_w(X_1)-G_w(X_2)||
$$

> **Condition 1:**
>
> $\exists m>0$, such that $E_w(x_1,x_2)+m<E_w(X_1,X_2')$;

**【Contrastive Loss Function】 **

> a contrastive term to ensure not only that the energy for a pair of inputs from the same category is low, but also that the energy for a pair from different categories is large;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200912231025.png)

- $(X_1,X_2)i$Y，X_1,X_2)i$ : the i-th sample with a pair of images and a label;
- $L_G$: the partial loss function for a genuine pair;
- $L_I$: the partial loss function for an imposter pair;
- $P$: the number of trainning samples;

$$
H(E_w^G,H_w^I)=L_G(E_w^G)+L_I(E_w^I)
$$

> **Condition 2:**  the minima of $H(E_w^G,H_w^I)$ should be inside the half plane $E_w^G+m<E_w^I$;

> **Condition 3:** the nagative of the gradient of $H(E_W^G,E_w^I)$ on the margin line $E_w^G+m=E_w^I$ has a positive dot product with the direction [-1,1];

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200912232118.png)

#### Notes <font color=orange>去加强了解</font>

  - condition 证明部分没有看懂；

## 2. 应用

### 2.1. Signature Verification

> Bromley, Jane, et al. "Signature verification using a" siamese" time delay neural network." *Advances in neural information processing systems*. 1994. cited by 1942

- base on Siamese neural network, design a system for verification of signatures written on a pen-input tablet;
- contain two sub-networks to extract features from two signatures, while the joining neuron measures the distance between the two feature vectors;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200912232456.png)

### 2.2. Image patches comparation

> Zagoruyko, Sergey, and Nikos Komodakis. "Learning to compare image patches via convolutional neural networks." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015. cited by 978

**level**: CVPR
**author**: Sergey Zagoruyko
**date**: 2015
**keyword**:

- image patches

------

### Paper: Compare Image Patches

<div align=center>
<br/>
<b>Learning to Compare Image Patches via Concolutional Neural Networks</b>
</div>



#### Summary

1. directly learn from image data( i.e., without any manually-designed features) a general similarity function for patches that can implicitly take into account various types of transformations and effects;
2. explore and propose a variety of different neural network models adapted for representing such a function, highlighting at the same time network architectures that offer improved performance;

#### Research Objective

  - **Application Area**: structure from motion; wide baseline matching; building panoramas; image super-resolution; object recognition; image retrieval; classification of object categories;
- **Purpose**:  make proper use of such datasets to automatically learn a similarity function for image pathes?

#### Proble Statement

- many factors that affect the final appearance:
  - changes in viewpoint;
  - variations in the overall illumination;
  - occlusions;
  - shading;
  - differences in camera settings;

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913070606042.png)

**【Module one】Base Models**

- **Siamese network**

> two branches in the network that share exactly the same architecture and same set of weights;
>
> each branch takes as input one of the two patches and then applies a series of convolutional, ReLU and max-pooling layers.
>
> branch outputs are concatenated and given to a top network that consists of linear fully conncted and ReLU layers for decision;

- **Pseudo-siamese network**

> has the structure of the siamese net except that the weights of the two branches are uncoupled, not shared.

- **2-channel network**

> <font color=red>consider the two patches of an input pair as a 2-channel image</font>, which is directly fed to the first convolutional layer of the network;
>
> the output is then given as input to a top module that consists of a fully connected linear decision layer;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913072610.png)

**【Module two】Central-surround two-stream network**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913072749.png)

- central high-resolution stream: two 32*32 patches that are generated by cropping the central 32\*32 part of each input 64\*64 patch;
- the surround low-resolution stream: 32\*32 patches that are generated by downsampling at half the original pair of input patches;
- **Spatial pyramid pooling network for comparing pathes:** essentially amounts to inserting a spatial pyramid pooling layer between the conv and NN layers fo the network, to aggregates the features of the last convolutional layer through spatial coding, where the size of the pooling regions is dependent on the size of the input;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913073942.png)

#### Evaluation

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913074532.png)

#### Notes <font color=orange>去加强了解</font>

  - SPP network 有时间去了解一下

> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial pyramid pooling in deep convolutional networks for visual recognition. *IEEE transactions on pattern analysis and machine intelligence*, *37*(9), 1904-1916. cited by 4764;

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/siamesenetwork/  

