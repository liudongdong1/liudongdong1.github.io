# Object Tracking


## Paper: SiamRPN++

<div align=center>
<br/>
<b>SiamRPN++: Evolution of Siamese Visual Tracking with Very DeepNetworks
</b>
</div>
#### Summary

1. propose a new model architecture to perform layer-wise and depth-wise aggregations, which not only further improves the accuracy but also reduces the model size.
2. provide a deep analysis of Siamese trackers and prove that when using deep networks the decrease in accuracy comes from the destroy of the strict translation invariance.
3. present a simple yet effective sampling strategy to break the spatial invariance restriction which successfully trains Siamese tracker driven by ResNet architecture.
4. propose  a layer wise feature aggregation structure for the cross-correlation operation, which helps the tracker to predict the similarity map from features learned at multiple levels.
5. propose a depth-wise separable correlation structure to enhance the cross-correlation to produce multiple similarity maps associated with different semantic meanings.

#### Research Objective

  - **Application Area**:  object tracking, velocity measurement, multi-object analyse

#### Proble Statement

previous work:

- Siamese trackers formulate the visual object tracking problem as learning a general similarity map by cross-correlation between the feature representations learned for the target template and the search region.

#### Methods

- **Problem Formulation**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200422120358518.png)

【Qustion 1】for strict translation 

- **the spatial aware sampling strategy** effectively alleviate the break of the strict tranlation invariance property caused by the networks with padding.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200422120557975.png)

【Question 2】 how to  transfer a deep network into our tracking algorithms

- propose a SiamRPN++ network.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200422120847811.png)

- lay-wise aggregation: compounding and aggregating these representations improve inference of recognition and localization.
  - features from earlier layers mainly foces on low level information such as color, shape, are essential for localization, the latter layers have rich semantic information like motion blur, huge deformation.
  - the output sizes of the three RPN modules have same spatial resolution, weighted sum is adopted directly on the RPN output.$S_all=\sum_{l=5}^5a_i*S_l, B_all=\sum_{l=3}^5b_i*B_l$

- Depthwise cross correlation: the object in the same category have high resppnse on same channels.![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200422121835492.png)

#### Notes <font color=orange>去加强了解</font>

  -  https://lb1100.github.io/SiamRPN++.   开源代码pysot

## Paper: SiamRPN++

<div align=center>
<br/>
<b>SiamRPN++: Evolution of Siamese Visual Tracking with Very DeepNetworks
</b>
</div>

#### Summary

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/object-tracking/  

