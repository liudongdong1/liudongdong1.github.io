# Multi-Sense


**level**: IEEE Robotics and automation letters
**date**: '2019,10'
**keyword**:

- Deep learning in robotics and automation,action segmentation,ergonomic safety.

# Paper: Ergonomic Risk predition

<div align=center>
<br/>
<b>Toward Ergonomic Risk Prediction via Segmentation
of Indoor Object Manipulation Actions Using
Spatiotemporal Convolutional Networks</b>
</div>
#### Research Objective

 we present a first of its kind *end-to-end* deep learning  system for ergonomic risk assessment during indoor object manipulation using camera videos. Our learning system is based on action segmentation*, where an action class (with a corresponding risk label) is predicted for every video frame.

The REBA model assigns scores to the human poses, within a range of 1–15, on a frame-by-frame basis by accounting for the joints motions and angles, load conditions, and activity repetitions. An action with an overall score of less than 3 is labeled as ergonomically safe, a score between 3–7 is deemed to be medium risk that requires monitoring, and every other action is considered high risk that needs attention. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191101153604076.png)

------

**level**: International Conference on Open Source System and Technology(ICOSST)
**author**: 
**date**:
**keyword**:

- .Smart Home,Android, RaspiberryPi ,OpenCV

------

# Paper: 

<div align=center>
<br/>
<b>Facilitating Gesture-based Actions for a Smart Home Concept</b>
</div>
#### Research Objective

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191101155544019.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191101155623730.png)



# Paper： Hauar 

**home automation using action recognition**： using action recognition to  fully automate the home appliances. We recognize the three  actions of a person (sitting, standing and lying) along with the  recognition of an empty room.  使用了PIR Motion 传感器

# Paper： Hybrid user action prediction

Hybrid user action prediction system for automated home using association rules and ontology：  based on the frequent pattern (FP)-growth and ontology graphs for home automation systems. Their proposed system simulates the human prediction actions by adding common sense data by utilizing the advantages of the ontology graph and the FP-growth to find a better solution in predicting home user actions for automated systems .使用了室内开关数据预测，关联分析，马尔可夫状态转换，聚类

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191101161301257.png)

------

**level**:IEEE access Special section on mobile multimedia for healthcare
**author**: M.Shamim Hossain
**date**:2017
**keyword**:

- .wireless sensors,inhome activities

------

# Paper: SmartHomeMonitor

<div align=center>
<br/>
<b>An annotation Technique for in-Home Smart Monitoring Environments</b>
</div>
#### Research Objective

 no previous research has considered automatically segmenting data during the process of data acquisition.       humans may perform two or more actionsconcurently

​    The  proposed technique defifines the annotation process as an optimization problem in which each incoming action is modeled to increase the probability of assigning a given set of actions  to a specifific activity. Hidden Markov Model (HMM) and Conditional Random Field (CRF) are applied to model the joint probability and features of activities in terms of actions.

 (1) modeling activity actions as a set of states and transitions using HMM, (2) modeling a transition feature function that embeds temporal and spatial relations among consecutive actions, and (3) defifining the segmentation problem as an optimization problem to minimize 

 This paper focuses only on data segmentation, in which an agent must decide the 

size of the block of actions that represents an activity. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191101163430182.png)

# Paper:

**Multi-Task Deep Learning for Pedestrian detection ,action recognition and Time to cross prediction:** on pedestrian detection and pedestrian action recognition but also on estimating if the pedestrian’s action presents a risky situation according to time to cross the street. We propose 1) a pedestrian detection and action recognition component based, on RetinaNet;   2) an estimation of the time to cross the street for multiple pedestrians using a recurrent neural network. For each pedestrian, the recurrent network estimates the pedestrian’s action intention in order to predict the time to cross the street. We based our experiments on the JAAD dataset, and show that integrating multiple pedestrian action tags for the detection part when merge with a recurrent neural network (LSTM) allows a signifificant performance improvement. 

# Paper: Skeleton-basedOnlineActPre

**Skeleton-Based Online Action Prediction Using Scale Selection Network**:

Action Prediction, Scale Selection, Sliding Window, Dilated Convolution, Skeleton Data.  online action prediction in streaming 3D skeleton sequences. A dilated convolutional network is introduced to model the motion dynamics in temporal dimension via a sliding window over the temporal axis. Since there are signifificant temporal scale variations in the observed part of the ongoing action at different time steps, a novel window scale selection method is proposed to make our network focus on the performed part of the ongoing action and try to suppress the possible incoming interference from the previous actions at each step.

# Paper:Task-Oriented Grasping

**Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191104134717938.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191104134753308.png)



**level**: CCF_A    CVPR
**author**: MahdiAbavisani 
**date**: 2019 
**keyword**:

- hand gesture recognitioln

------

## Paper: Unimodal Dynamic HandGesture

<div align=center>
<br/>
<b>Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition with Multimodal Training</b>
</div>



#### Summary

1. present an efficient approach for leveraging the knowledge from multiple modalities in training unimodal 3D convolutional neural networks for the task of dynamic hand gesture recognition.
2. dedicate separate networks per available modality and enforce them to collaborate and learn to develop networks with common semantics and better representations

#### Research Objective

  - **Application Area**: human-computer interaction, sign language recognition, gaming and virtual reality control
- **Purpose**:  multimodal learning and unimodal testing.

#### Proble Statement

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312135338379.png)

previous work:

- most hand gesture recognition methods exploit multiple sensors such as visible RGB cameras, depth camera or compute an extra modality like optical flow.
- Dynamic hand gesture recognition: same to video analysis approaches, derive properties such as appearance, motion cues, or body skeleton to perform classification
  - 3D-CNN-base hand gesture recognition methods, Multi-sensor system: fuses streams of data from multiple sensors including short-range radar, color and depth sensors for recongition. 
  - ResC3D combines multimodal data and expoits an attention model
- Transfer Learning:  an agent is independently trained on a source task, then another agent uses the knowledge of the source agent by repurposing the learned features or transferring them to improve its learning on a target task.
- 

#### Methods

- **Problem Formulation**:

the stream of data is available in M modalities， and there are M classifier networks with similar architectures that classify based on their corresponding input, we aim to improve the learning process by transferring the knowledge of different modalities.

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312140717210.png)

【Qustion 1】how to alignment the spatiotemporal semantic multi-modal data?

- assume that different modalities of the input videos are aligned over the time and spatial positions, the networks are expected to have the same understanding and share semantics for spatial positions and frame of the input videos across the different modalities.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312141904565.png)

【Qustion 2】 how to avoid Negative Transfer throw multi-modal data?

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312142405938.png)

![](https://cdn.pixabay.com/photo/2015/06/24/16/36/office-820390__340.jpgimage-20200312142343590.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312142431917.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312142521221.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312142536425.png)

#### Evaluation

  - **Environment**:   
    - Dataset:  
      - VIVA hand gesture dataset: for studying natural human activities in real-world driving settings, 19 hand gesture classes collected from 8 subjects.
      - EgoGesture Dataset: for the task of egocentric gesture recognition, contains 24161 hand gesture clips of 83 classes of gestures performed by 50 subjects, including both static and dynamic gesture.   <font color=red>重点了解下这个数据集</font>
      - NVGestures datasets: multiple sensors and from multiple viewpoints for studying human-computer interfaces, contains 1532 dynamic hand gestures inside a car simulator with artificial lighting conditions.<font color=red>重点了解下这个数据集</font>
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312143418786.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200312143443999.png)

#### Conclusion

- propose a new framework for single modality networks in dynamic hand gesture recognition task to learn from multiple modalities
- introduce the SSA loss to share the knowledge of single modality networks
- develop the focal regularization parameter for avoiding negative transfer.

#### Notes <font color=orange>去加强了解</font>

  -  Multi-sensorsystemfordriver’shand-gesturerecognition. InAutomaticFaceandGestureRecognition(FG)
  -  Deep multimodal learning: A survey on recent advances and trends
  -  Online detection and classiﬁcation of dynamic hand gestures with recurrent 3d convolutional neural network
  -  Multimodal gesture recognition based on the resc3d network
  -  ImageNet+Kinectics pre-trained networks
  -  Quo vadis, action recognition? a new model and the kinetics dataset
  -  该论文没有代码，介绍了模型实现但是现在无法复现。
  -  VGG16+LSTM [13]                          这些网络模型都是可以学习使用的
  -  C3D+LSTM+RLSTM [8]
  -  I3D
  -  C3D    
  -  VGG16
  -  HOG+HOG2 [29]

**level**: *IEEE international conference and workshops on automatic face and gesture recognition*
**author**: Pavlo Molchanov (NVIDIA Research,)
**date**:  2015
**keyword**:

- hand gesture understand

------

# Paper: Multi-sensor System

<div align=center>
<br/>
<b>Multi-sensor System for Driver's Hand-Gesture Recogniton</b>
</div>



#### Summary

1. using short-range radar, a color camera, and a depth camera which together make the system robust against variable lighting conditions.
2. present a jointly calibrate the radar and depth sensors.
3. employ concolutional deep neural networks to fuse data from multiple sensors and to classify the gestures.(10 different gestured indoors and outdoors in a car)

#### Research Objective

- **Purpose**:  fuse multi-modal data to recognise the hand gesture.

#### Proble Statement

- color sensors are ineffective under low-light conditions at night
- commodity depth cameras that typically use projected IR signals are ineffective under direct bright sunlight.
- suffer from from the presence of harsh shadows and hand self-occlusion
- micro-Doppler signatures of acoustic signals have also been developed ,while acoustical sensors for gesture recognition are not directly applicable inside vehicles because of the presence of significant ambient acoustical noise.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313165655037.png)

- build a prototype radar system, with an operational range of <1m, the system measures the range(z) and angular velocity(v) of moving objects in the scene, and estimates their azimuth(x) 方位角and elevation海拔(y) angles.  (FMCW)[27, 28] 重点了解雷达特性

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313170714490.png)

【Radar Relative】

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313170714490.png)

【Calibration】

- assume a rigid transfermation exists between the optical imaging centers of the radar and depth sensors. 
- experiment: concurrently observe 3D coordinates of the center of a moving spherical ball of radius 3cm with both sensor. Using linear-squares optimization.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313171052341.png)

【Gesture Detection and classifier】

- assume that a true gesture occurs only when the radar detects signiﬁcant motion, i.e., with velocity above a conﬁgurable threshold (0.05m/s), roughly in the center of the FOV of the UI。
- The duration of a true gesture is assumed to be between 0.3 and 3 seconds. The gesture ends when no motion is observed by the radar continuously for 0.5 seconds. 
- normalize the depth values of the detected hand region to the range of [0,1], and generate a mask for hand region. And conert RGB image of the hand to a single grayscale image with values in the range[0,1]

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313171445019.png)

- temporally normalize the gestures to 60 frames by re-sampling them via nearest neibor interpolation.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313171637448.png)

#### Evaluation

  - **Environment**:   
    - Dataset: 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313171706749.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313171715796.png)

#### Conclusion

- a novel multi-sensor gesture recognition system that effectively combines imaging and radar sensors
- use of the radar sensor for dynamic gesture segmentation, recognition,and reduced power consumption
- demonstration of a real-time illumination robust geture interface for the challenging use case of vehicles.

#### Notes <font color=orange>去加强了解</font>

  - Voronoi diagram ：根据点集划分的区域到点的距离最近的特点，其在地理学、气象学、结晶学、航天、核物理学、机器人等领域具有广泛的应用。如在障碍物点集中，规避障碍寻找最佳路径。![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200313181217743.png)
  - FMCW ：即调频连续波。FMCW技术和脉冲雷达技术是两种在高精度雷达测距中使用的技术。其基本原理为发射波为高频连续波，其频率随时间按照三角波规律变化。接收的回波频率与发射的频率变化规律相同，都是三角波规律，只是有一个时间差，利用这个微小的时间差可计算出目标距离。
  - “Monopulse range-doppler FMCW radar signal processing for spatial localization of moving targets
  - https://zhuanlan.zhihu.com/p/77474295  
  - https://training.eeworld.com.cn/TI/show/course/4132  FMCW 学习链接



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/multi-sense/  

