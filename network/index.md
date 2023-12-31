# Network


**level**: CVPR   CCF A
**author**: Eddy IIg
**date**:  2017
**keyword**:

- Optical Flow

------

## Paper: FlowNET2.0<font color=red>不理解</font>

<div align=center>
<br/>
<b>Evolution of Optical Flow Estimation with Deep Networks</b>
</div>



#### Summary

1. 文章以实验主导,数据的顺序,  模型的叠加,各种算法结果

#### Note  去加强

- https://lmb.informatik.uni-freiburg.de/  

#### Proble Statement

previous work:

- End-to-End optical flow estimation with CNNs was first proposed by [10]. Featuring a 3D CNN[30],unsupervised learning [1,33],carefullly designed rotationally invariant architectures, pyramidal approach based on the coarse-to-fine idea of variational methods[20]. No significantly outperform.
- An alternative approach to learning-based optical flow estimation using CNN to match image patches. reach good accuracy ,but require lots computing.
- CNN trained for per-pixel prediction tasks often produce noisy or blurry results.   refinement can be obtained by stacking several CNNs on top of each other,leding improve results in human pose estimation,semantic instace segmentation.

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191116180045883.png)

【Qustion 1】Dataset Schedules   ?

both the kind of data and the order in which it is presented during training.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191116181737160.png)

#### Conclusion

- foces on training data and show that the schedule of presenting data during training is very import
- develop a stacked architecture that includes warping of the second image with intermediate optical flow
- elaborate on small displacements by introducing a subnetwork specializing on small motions.

------

## Paper: Focal Loss

<div align=center>
<br/>
<b>Focal Loss for Dense Object Detection</b>
</div>



#### Summary

1. 

#### Research Objective

- **Application Area**:
- **Purpose**:  One stage detectors applied over a regular dense sampling of object locations ,scales and aspectratios.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191116183901028.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191116184807649.png)

#### Proble Statement

- class imbalance is addressed in R-CNN like detectors by a two-stage cascade and sampling heuristics .The proposal stage(selective Search[35],EdgeBoxes[39],DeepMask[24,25],RPN[28]).The second classification stage (sampling heuristics,fixed foreground-to-background ratio,online hard  example mining[31])
- training procedure is still dominated by easily classified background examples. some method( bootstrapping[33,29], hard example mining[37,8,31])

previous work:

- Classic Object Detectors:
  - sliding-window paradigm .applied convolutional neural networks to handwritten digit recognition.HOG[4] and integral channel features[5] for pedestrian detection.
- Two-stage Detectors: R-CNN , RPN , Faster R-CNN
- One-stage Detectors: OverFeat[30] the first, SDD[22,9] ,YOLO .
- Class Imbalance:   
  - Training is inefficient as most location are easy negatives that contribute no useful learning signal
  - the easy negatives can overwhelm training and lead to degenerate models.
- Robust Estimation: focal loss designed to address class imbalance by down-weighting inliers,their contribution to total loss is small.

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191116190856305.png)

#### Conclusion

- discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause,we reshaping the standard cross entropy loss such that is down-weights the loss assigned to well-classified examples.
- training on a sparse set of hard examples and prevents the vast number os easy negatives from easy negatives from  overwhelming the detector during training.
- design a simple-one-stage object detector called RetinaNet.

#### Notes <font color=orange>去加强了解</font>

- https://github.com/facebookresearch/Detectron
- 4,5,8  传统方法
- 20 FPN method.



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/network/  

