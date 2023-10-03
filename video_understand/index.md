# Video_Undertand


> 安防监控领域，包括人脸识别、行为识别、运动跟踪、人群分析等等，利用卡口精准位置布控视频监测，实现了监控区域内异常的自动识别，例如动态视频中的人脸与黑名单库实时比对检测，多视点视频协同分析运行轨迹，视频数据结构化后对关键目标的检索等等；
>
> 互联网娱乐场景，包括拍照优化、视频优化、实时人像美颜、AR特效、自定义背景等等，丰富了直播、短视频等互联网娱乐应用；
>
> 金融身份认证场景，包括各种刷脸的金融应用，如远程开户、支付取款等等；
>
> 无人商场与广告营销，包括线下零售、商品识别、广告AR赋能等等；
>
> 工业机器的视觉系统，包括物品分拣、缺陷检验等等，通常是自动图像分析与光学成像等其他方法技术相结合；
>
> 无人机无人车控制，包括视觉导航、行人分析、障碍物检测等等，通常作为一种传感器和激光雷达、毫米波雷达、红外探头与惯性测量单元融合生成供自主决策的信息；

# 0. 视频理解方向

> - Task1：未修剪视频分类(Untrimmed Video Classification)。这个有点类似于图像的分类，未修剪的视频中通常含有多个动作，而且视频很长。有许多动作或许都不是我们所关注的。所以这里提出的Task就是希望通过对输入的长视频进行全局分析，然后软分类到多个类别。
> - Task2：修剪视频识别(Trimmed Action Recognition)。这个在计算机视觉领域已经研究多年，给出一段只包含一个动作的修剪视频，要求给视频分类。
> - Task3：时序行为提名(Temporal Action Proposal)。这个同样类似于图像目标检测任务中的候选框提取。在一段长视频中通常含有很多动作，这个任务就是从视频中找出可能含有动作的视频段。
> - Task4：时序行为定位(Temporal Action Localization)。相比于上面的时序行为提名而言，时序行为定位于我们常说的目标检测一致。要求从视频中找到可能存在行为的视频段，并且给视频段分类。
> - Task5：密集行为描述(Dense-Captioning Events)。之所以称为密集行为描述，主要是因为该任务要求在时序行为定位(检测)的基础上进行视频行为描述。也就是说，该任务需要将一段未修剪的视频进行时序行为定位得到许多包含行为的视频段后，对该视频段进行行为描述。比如：man playing a piano

# 1. 手语论文

> ### 工业界：
>
> 腾讯优图实验室AI手语识别 https://www.jiqizhixin.com/articles/2019-05-16-16
>
> 中科大和微软推出了基于Kinect的手语翻译系统，加州大学曾经推出过的手语识别手套
>
> #####  **潜在需求分析**：
>
> ​	1. **听障人士数量数量多** 世界卫生组织最新数据显示[1]，目前全球约有4.66亿人患有残疾性听力损失，超过全世界人口的5%，估计到2050年将有9亿多人（约十分之一）出现残疾性听力损失。据北京听力协会2017年公开数据，估计中国残疾性听力障碍人士已达7200万[2]，
>
> 2. **无障碍普及率有待提升，听障人群需求被忽视**
>         	
> 3.  提供一套兼容全球手语的双向翻译器/或是简单的识别器
>    - 立即可以为上千万聋哑人获得更多的电脑控制权
>    - 结合 IFTTT 以及 Home 类似智能家庭控制器
>    - 完全可以形成一个嵌入专用硬件的产业了
>
> ##### 问题
>
> ​	1.  自动区分手语表达中的各类手势、动作以及这些手势和动作之间的切换，最后将表达的手语翻译成文字。传统的方法通常会针对特定的数据集设计合理的特征，再利用这些特征进行动作和手势的分类。受限于人工的特征设计和数据量大小，这些方法在适应性、泛化性和鲁棒性上都非常有限。
>
> 使用Kinect摄像机的多种传感器来提前获取手语表达者的肢体关节点信息： 传感器手套、或配备EMG、IMU传感器的手环来获取手臂和手掌的活动信息

**level**: CVPR  CCF_A
**author**:Junfu Pu    CAS Key Laboratory of GIPAS, University of Science and Technology of China
**date**: 2019
**keyword**:

- ASL , CTC

------

## Paper: Iterative Alignment Network

<div align=center>
<br/>
<b>Iterative Alignment Network for Continuous Sign Language Recognition</b>
</div>


#### Summary

1. 

#### Research Objective

- **Application Area**:
  - sign language (SL) is used by millions of people with hearing or spoken damage in their daily life
  - lack of systematic study for sign language, it becomes very difficult for many people to communicate with the deaf-mute
- **Purpose**:  propose an alignment network with iterative optimization for weakly supervised continuous signlanguage recognition

#### Proble Statement

previous work:

- isolated SLR  recognition [16, 22, 42, 43]
- video representation： 3D-CNN  ResNet  P3D 
- sequence modeling:
  - attention-based encoder-decoder network
    - Bahdanau et al. [1] introduce attention mechanism into encoder-decoder network to learn the correspondence between source sequence and target sequence
  - connectionist temporal classification(CTC) based network
    - CTC is able to deal withunsegmented input data, and learn the correspondence between the input sequence and output sequence.
- continuous SLR 
  - hand-crafted feature based
    - Hidden Markov Model (HMM) or Hidden Conditional Random Fields (HCRF)
    - [35] two real-time HMM-based systems for recognizing
      sentence-level continuous American Sign Language (ASL).
    - [40]a discriminative sequence model with Hidden Conditional Random Field (HCRF) for gesture recognition
  - deep learning based  [9, 23, 25] datasets 了解一下
    - video represntations by redidual network ResNet[18], 3D-CNN [33, 37]
    - [23] with hierarchical attention in latent space

#### Methods

- **Problem Formulation**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223092845392.png)

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223092906125.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093357368.png)

**CTC_Loss**: 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093649877.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093518165.png)

**LSTM_Loss**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093740963.png)

![image-20191223093756890](../../../../MEGA/MEGAsync/actionPrediction/ActionPrediction.assets/image-20191223093756890.png)

**The Whole NetworkLoss**:

![image-20191223093851361](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093851361.png)

![image-20191223093859610](../../../../MEGA/MEGAsync/actionPrediction/ActionPrediction.assets/image-20191223093859610.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093025875.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223093947114.png)

#### Evaluation

- **Environment**:   
  - Dataset: 
    - RWTH-PHOENIX-Weather multi-signer [25] for German SLR
    - CSL [23] for Chinese SLR
- **Evaluate Methods**: ![image-20191223094117184](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223094117184.png)
- The window size is set to be 8 with a stride of 4,the 3D-ResNet is pre-trained on an isolated sign language recognition dataset released in [43]
- **Performance**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223094423096.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223094431353.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223094443353.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223094507331.png)

#### Conclusion

- A unified deep learning architecture integrating encoderdecoder network and connectionist temporal classification (CTC) for continuous sign language recognition.
- A soft dynamic time warping (soft-DTW) alignment constraint between the LSTM and CTC decoders, which indicates the temporal segmentation in sign videos
- Iterative optimization strategy to train feature extractor and encoder-decoder network alternately with alignment proposals by warping path

#### Notes <font color=orange>去加强了解</font>

- [ ] 论文23: Video-based sign language recognition without
  temporal segmentation   China
- [ ] paper25 数据集 German Continuous
  sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers
- [ ] SubUNets: End-to-end hand shape and continuous sign language recognition
- [ ] Online early-late fusion based on adaptive HMM for sign language recognition
- [ ] Can spatiotemporal 3D CNNs retrace the history of 2D CNNs and imagenet
- [ ] Attention based 3D-CNNs for large-vocabulary sign language recognition
- [ ] Video-based sign language recognition without temporal segmentation
- [ ] Dilated convolutional network with iterative optimization for continuous
  sign language recognition
- [ ] Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers
- [ ] Online early-late fusion based on adaptive HMM for sign
  language recognition
- [ ] Joint CTC/attention decoding for end-to-end speech recognition
- [ ] Attention based 3D-CNNs for large-vocabulary sign language recognition
- [ ] Video-based sign language recognition without
  temporal segmentation
- [ ] Deep sign: hybrid CNN-HMM for continuous sign language recognition
- [ ] Re-sign: Re-aligned end-to-end sequence modelling with deep recurrent CNN-HMMs.  
- [ ] Online detection and classification of dynamic hand gestures with recurrent 3D
  convolutional neural networks
- [ ] Dilated convolutional network with iterative optimization for continuous sign language recognition

**level**: Sensys     CCF_B
**author**:Biyi Fang  Michigan State University
**date**: 2017
**keyword**:

- ASL, Leep Motion(an infrared light-based sensing device)

------

## Paper: DeepASL

<div align=center>
<br/>
<b>DeepASL: Enabling Ubiquitous and Non-Intrusive Word and
Sentence-Level Sign Language Translation</b>
</div>
#### Summary

1. performance at both word level and sentence level (unseen ASL sentences ,unseen users)
2. robustness under various real-world settings (various ambient lighting conditions, body postures,and interference sources )
3. system performance test in terms of runtime , memory usage and energy consumption.

#### Research Objective

- **Application Area**:seeking help from a sign language interpreter, writing on paper, or typing on a mobile phone,each of these methods has its own key limitations in terms of cost,
  availability, or convenience
- **Purpose**:  

#### Proble Statement

- ASL : hand shape, hand movement, relative location of two hands, body movement, face emotions
- Electromyography (EMG) sensors, RGB cameras, Kinect sensors intrusive where
  sensors have to be attached to !ngers and palms of users, lack of resolutions to capture the key characteristics of signs, or significantly constrained by ambient lighting conditions or backgrounds
  in real-world settings
- existing sign language translation systems can only translate a single sign at a time, thus
  requiring users to pause between adjacent signs.

previous work:

- wearable sensor-based :motion sensors(accelerometers, gyroscopes), EMG sensors, bending of fingers to infer the performed fingers. <font color=red>intrusive and impractical for daily usage</font>
- Radio Frequency-based: <font color=red>wire-less signals have very limited resolutions to see the hands</font>
- RGB camera-based: <font color=red> poor lighting conditions or generally uncontrolled backgrounds, privacy </font>
- Kinect-based: hard to capture the hand shape information

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224090129594.png)

- Leap Motion is able to extract skeleton joints of the fingers, palms and forearms from the raw infrared images.![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224090111350.png)

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224084552045.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224090211496.png)

1.  a temporal sequence of 3D coordinates of the skeleton joints of !ngers, palms and forearms
2.  the key characteristics of ASL signs including hand shape, hand movement and relative location of two hands    spatio-temporal trajectories of ASL characteristics
3.  models the spatial structure and temporal dynamics of the spatio-temporal trajectories of ASL characteristics for word-level ASL translation
4.  CTC-based framework that leverages the captured probabilistic dependencies between words in one complete sentence and translates the whole sentence end-to-end without requiring users to pause between adjacent signs.

**【ASL Characteristics Extraction】**

- Savitzky-Golay flter [37] to improve the signal to noise ratio of the raw skeleton joints data

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224090953165.png)

- extract hand shape: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224092151069.png)
- hand movement information:![image-20191224092217948](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224092217948.png)

**【Word-Level ASL Translation】**： translation errors when different signs share very similar characteristics at the beginning of the signs

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224092513888.png)

- Hierarchical Bidirectional RNN for Single-Sign Modeling:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191224092932396.png)

**【Sentence level Translation】** using CTC network





# 2. 视频理解

**level**:  CVPR_CCFA
**author**:Romero Morais
**date**: 
**keyword**:

- video analyse, anomaly detection

------

## Paper: Anomaly Detection

<div align=center>
<br/>
<b>Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos
</b>
</div>




#### Summary

1. model the normal patterns of human movement in surveillance video for `anomaly detection using dynamic skeleton features.`
2. decompose the skeletal movements into two sub-components: global body movement and local body posture. The global body movement tracks the dynamics of the whole body in the scene, while the body posture describe the skeleton configuration in the canonical coordinate frame of the body's bounding box.
3. model the dynamic and interaction of the coupled features in our novel Message-Passing Encoder-Decoder Recurrent Network.
4. skeleton features are compact, strongly structured, semantically rich, and highly descriptive about human aciton and movement, which are keys to anomaly detection. 

#### Proble Statement

- The human behavioral irregularity can be factorized into few factors regarding body motion and posture: location, velocity, direction, pose, and action.

#### Methods

【Qustion 1】 the scales of human skelons vary largely depending on their location and actions

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422104952789.png)
$$
f_t^i=f_t^g+f_t^{l,i};f^g=(x^g,y^g,w,h),f^{l,i}=(x^{l,i},y^{l,i})
$$
![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422105234027.png)

【qustion 2】how to fuse local and global features

- propose MPED-RNN models, consisting two recurrent encoder-decoder network branches, each of them dedicated to one of the components, each branch of them has the single-encoder-dual-decoder architecture with three RNNS: Encoder,Reconstructing Decoder and Predicting Decoder.
- use Gated Recurrnet Units in every segment of MPED_RNN for its simplicity and similar performance to LSTM

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422105321588.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422105656331.png)

【qurestion 3】 how to detect video anomalies?

1. **Extract segments**: select the overlapping skeleton segments by using sliding window of size T and stride s on the trajectory
2. **Estimate segment losses**: decompose the segment to two sub-component, feed all segment features to the traind MPED-RNN, output the normality loss
3. **Gather skeleton anomaly score**:  the measure the conformity of a sequence to the model given both the past and future context, using voting scheme to gather the losses of related segments into an anomaly score of each skeleton instance:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422111136464.png)
4. Calculate frame anomaly score: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422111221056.png)

#### Evaluation

  - **Environment**:   
    - Dataset: ShanghaiTech Campus Dataset for video anomaly detection currently available, combines footage of 13 different cameras .

#### Notes <font color=orange>去加强了解</font>

  - video anomaly detection
  - human trajectory modeling
  - sequence 一致性

**level**: 
**author**: waqas sultani, UCF
**date**: 
**keyword**:

- anomaly detection, video analyse

------

# Paper: Real-world Detection

<div align=center>
<br/>
<b>Real-world Anomaly Detection in Surveillance Videos</b>
</div>
#### Summary

1. propose to learn anomaly through the deep multiple instance ranking framework by leveraging weakly labeled training video, the training labels(anomalous or normal) are at video-level instead of clip-level.
2. introduce a new large-scale dataset of 128 hours of videos with 13 realistic anomalies such as fighting, road accident, burglary robbery.
3. propose a MIL solution to anomaly detection by leveraging only weakly labeled training videos, propose MIL ranking loss with sparsity and smoothness constraints for a deep learning network to learn anomaly scores for video segments.

#### Research Objective

  - **Application Area**:  traffic accidents, crimes  or illegal activities.

#### Proble Statement

- **Anomaly detection**:
  - considering all anomalies in one group and all normal activities in another group
  - recognise specific activities.
  - impossible to define a normal event which takes all possible normal patterns/behaviors into account.
  - detect human violence by exploiting motion and limbs orientation of people 
  - employed video and audio data to detect aggressive actions in surveillance videos.
  - violent flow descriptors to detect violence in crowd videos.
  - using deep learning based autoencoders to learn the model of normal behaviors and employed reconstruction loss to detect anomalies.
- **Ranking**: focus on improving relative scores of the items instead of individual scores.
  - deep rankinng networking: used for feature learning, highlight detection, graphics interchange format generation, face detection and verification, person re-identification, place recognition, metric learning and image retrieval.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422150129902.png)

【Qustion 1】less annotation learning

- only video-level labels indicating the presence of an anomaly in the whole video is needed. A video containing anomalies is labeled as positive and a video without any anomaly is labeled as negative.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422150330960.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422150529485.png)

【qustion 2】 how to detect anomaly activities without much precise annotation?

- Deep MIL Ranking model:  the scores of instances in the anomalous bag should be sparse,    the anomaly score should vary smoothly between video segments.
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200422150817733.png)

#### Notes <font color=orange>去加强了解</font>

  - sparse-coding based approaches.
  - deep rank
  - 项目代码： https://github.com/hangxu124/MyRes3D_AnoDect
    - https://github.com/dexXxed/abnormal-event-detection
    - https://github.com/nevinbaiju/anomaly-detection

**level**:  AAAI   CCF_A
**author**: Yijun Cai , Haoxin Li, Jian-Fang Hu , Wei-Shi Zheng
**date**: 2019
**keyword**:

- 

------

## Paper: Action Knowledge Transfer

<div align=center>
<br/>
<b>Action Knowledge Transfer for Action Prediction with Partial Videos</b>
</div>


#### Summary

1. 通过完整的视频动作序列来指导部分视频序列的预测？

#### Research Objective

- **Application Area**: in reducing computational resource, traffic system. 
- **Purpose**:  Propose to transfer action knowledge learned from fully observed videos for improving the prediction of partially observed videos

#### Proble Statement

- action prediction mainly lies in the lack of discriminative action information for the partially observed videos. partially observed videos often contain incomplete action executions thus have less action information than the fully observed ones.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202141323929.png)

- the existing action recognition systems can be directly used for action prediction by treating partial videos as full videos.

previous work:

- focus on improving the discriminative power of partial videos by developing max margin learning(Kong and Fu 2015) or soft regression  framework(Hu 2016)
- Action prediction: 
  - Ryoo et al.(Ryoo2012) proposed to use integral and dynamic bag-fo-words for action prediction
  - Kong and Fu 2015 a max margin learning framework was presented to learn discriminative features for prediction
  - Vondrick,Pirsiavash and Torralba2016 propose to predict the feature of future frames to learn better representations for action prediction
  - Lan,Chen developed hierarchical representations at multiple granularities to predict human action 
  - <font color=red>they dont seek to make use of the action knowledge learned from full sequences for prediction, we propose to mine rich action knowledge from full videos</font>
- Knowledge distillation  :(Hinto ,Vinyals and Dean 2015; Huang and Wang 2017; Yim et al.2017) the knowledge contained in a large network was distilled and transferred to a small network,by enforcing the outputs or intermediate activations of the small network to match those the large network . <font color=red>our goal is to improve the discriminative power of partially observed videos </font>

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143008477.png)

[**Question one**] how to Learn Action Knowledge from Full Videos

Given a set of full videos {x i } with corresponding features {f i } and labels {y i }, we intend to learn an embedding function G to project the original feature onto an embed-ding space, and a discriminative classifier D to project the embedding to the label space:
$$
e i = G(f i ),\\
p i = D(e i ).
$$


To encourage large distances between embeddings from different classes:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143440633.png)

[**Qustion Two**] Transferring Action Knowledge to Partial Videos

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143637577.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143706194.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143144128.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143845522.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202143753309.png)

#### Conclusion

- propose a novel knowledge transfer framework to boost the performance of action prediction with partial videos ,by transferring knowledge from feature embeddings and discriminative classifier of full videos.
- the method shows remarkable improvement for action prediction 

#### Notes <font color=orange>去加强了解</font>

- [ ] Kong Tao and Fu 2017    Qin et al.2017
- [ ] paper 18
- [ ] Additive Margin (AM) Softmax (Wang et al. 2018)
- [ ] Max-margin action predictionmachine
- [ ] spartiotemporal multiplier networks for video action recognition
- [ ] Distilling the knowlege in a neural network   2015 Hinton
- [ ] early action prediction by soft regression  
- [ ] Like what you like: Knowledge distill via neuron selectivity transfer
- [ ] Deep sequential context networks for action prediction
- [ ] learning activity progression in lstm for activity detection and early detection
- [ ] learning spatialtemporal features with 3d convolutional networks
- [ ] action recognition with improved trajectories
- [ ] action recognition by dense trajectories
- [ ] a gift from knowledge distillation:Fast optimization network minimization and transfer learning

**level**:  CCF_A 
**author**:  Tian Lan , Tsung-Chuan , Silvio Savarese  Standford University
**date**: 2014 ECCV
**keyword**:

- action prediction

------

## Paper: Hierarchical Representation

<div align=center>
<br/>
<b></b>
</div>


#### Summary

1.   adop an hierarchical structure to predict action from different granularity.

#### Research Objective

- **Application Area**: autonomous robots, surveillance and health care , robotic applications[24], [29]
- **Purpose**:  predict future action

#### Problem Statement

- capture the subtle details inherent in human movements that may imply a future action
- humans are highly articulated objects
- actions can be described at different levels of semantic granularities.
- prediction should carried out as quickly as possible
- from recognizing simple human actions such as walking and standing in constrained settings[19] to understanding complex actions in realistic video and still images collected from movies,TV show , sport games , Internet (background clutter, occlusions, viewpoint, changes)
  - in video: bag-of-features representations of local space-time features [22]
  - in image : contextural information such as attributes ,objects ,poses are jointly modeled with actions.

previous work:

- Human ability of the visual system to predict future actions based on previous observations of interactions among humans
- recent early event detection: expand spectrum of human action recognition to actions in future
  -  [18] addresses the problem of early recognition of unfinished activities
  -  [6] SVM framework for early event detection
  -  predicting motion from still images[29]
  -  prediction the future trajectories of pedestrians[15, 7]
- <font color=red>different from previous work</font>
  - <font color=red>predict future actions from any timestamp in a video , don’t constrain the input to the “early stage of an action”</font>
  - <font color=red>predict from a short video clip or even a static image</font>
  - <font color=red>expand the scope of action prediction from controlled lab settings to unconstrained “in-the-wild” footage</font>
  - <font color=red> predicting future actions from still images  or short video clips in unconstrained data</font>

#### Methods

- **Problem Formulation**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202132446815.png)

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202134658266.png)

【Qustion 1】how to construct hierarchy construction?

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202132617020.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202134937117.png)

【Qustion 2】Model formulation

define X : person example     $Y={y_i}_{i=1}^L$   ,L the total number of levels of the hierarchy and $y_i$ is the index of the corresponding moveme at level i.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202135449997.png)

![image-20191202135543989](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202135543989.png)

【Qustion 3】 optimization problem

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202135652819.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202135725669.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191202135751580.png)

#### Conclusion

- predict future actions from a single frame in the challenging real-word scenarios
- a hierarchical movemes to capture multiple levels of granularities in human movements
- develop a max-margin learning framework that jointly learns the appearance models of different movemes as well as their relations

#### Notes 

- [ ] 论文【1】 moveme concept   Learning and recognizing human dynamics in video sequences
- [ ] paper [22] Action recognition with improved trajectories
- [ ] paper[18]  Early recognition of ongoing activities from streaming videos
- [ ] paper[24]  Probabilistic modeling of human movements for intention inference
- [ ] paper[29] A data-driven approach for event prediction
- [ ] paper[9] anticipating future activities from RGB-D data by considering
  human-object interactions   Anticipating human activities using object a↵ordances
  for reactive robotic response
- [ ] 有没有实现代码 运行看看预测效果，通过代码进一步加深理解，在优化问题定义那块需要加强理解

**level**: ECCV  CCF_A
**author**: George Papandreou ,Tyler Zhu  Google Research

------

## Paper: PersonLab

<div align=center>
<br/>
<b>Person Pose Estimation and Instance Segmentation with a Bottom-Up,Part-based,Geometric Embedding Model</b>
</div>


#### Summary

1. present a box-free bottom-up approach for the tasks of pose estimation and instance segmentation of people in multi-person images using an efficient single-shot model
2. tackles both semantic-level reasoning and object-part associations using part-based modeling. Empoys a convolutional network to learns to detect individual keypoints and predict their relative displacements,then group key-points into person pose instances
3. propose a part-induced geometric embedding descriptor which allows us to associate semantic person pixels with their corresponding person instance,dilevering instance-level person segmentations

#### Research Objective

- **Application Area**:
- **Purpose**:  

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191130152232740.png)

- **Keypoint detection** : detect all visible key-points belonging to any person in the image .
  - 具体计算heatmap 到时再细看 ， short-range offset vector is to improve the keypoint localization accuracy.  <font color=red>aggregate the heatmap and short-range offsets via Hough voting into 2-D Hough score maps
- **Grouping keypoints into person detection instances**    Fast greedy decoding algorithm
- **Instance-level person segmentation** : Given the set of keypoint-level person instance detections, the task of our method’s egmentation stage is to identify pixels that belong to people (recognition) and associate them with the detected person instances (grouping)![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191130154003763.png)
- **Semantic person segmentation  && Associating segments with instances via geometric embeddings**

#### Notes <font color=orange>去加强了解</font>

- not read carefully

# 3. 视频行为预测(不同粒度)

**level**: CVPR  
**author**: Agrim Gupta , Li Fei-Fie
**date**: 2018 
**keyword**:

- trajectory prediction

------

## Paper: Social GAN

<div align=center>
<br/>
<b>Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</b>
</div>


#### Summary

1. 使用LSTM 来编码用户的行为,使用SocialLSTM 池化层来表示较远距离的行为关系,使用生成模型来产生多种路径,利用判别模型从中选择最佳路径

#### Research Objective

- **Purpose**:  predict the future trajectory

#### Proble Statement

- InterPersonal: human have innate ability to read the behavior of others when navigating crowds
- Socially Acceptable: social norms
- Multimodal: multiple trajectories

previous work:

- theymodel a local neighborhood around each person 
- they tend to learn average behavior <font color=red>we aim in learning multiple socially acceptable trajectories</font>

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118093649383.png)

【Qustion 1】Using LSTM to encode the location of each person.And model human-human interaction via a Pooling Module (PM). After tobs we pool hidden states of all the people present inthe scene to get a pooled tensor Pi for each person.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118094457342.png)

condition the generationof output trajectories by initializing the hidden state of the decoder as to produce future scenarios which are consistent
with the past![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118094655624.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118094908418.png)

Discriminator. The discriminator consists of a separate encoder. Specifically, it takes as input Treal = [Xi, Yi] or fake = [Xi, Yˆi] and classifies them as real/fake

[Question 2] Pooling Module Challenge: 1. Variable and large number of people in a scene,we need a compact representation whichcombines information from all the people. 2. Scattered Human-Human Interaction,the network needs to model global configuration.

- passing the input coordinates through a MLP followed by symmetric function(Max-Pooling).use relative coordinates for translation invariance  we augment the input to the pooling module with relative position of each person with respect to person i.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118095339973.png)

[question3 ] diverse sample generation . propose a variety lossfunction that encourages the network to produce diverse sample .generate k possible out put samples and choose the best prediction in L2 sense ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118100138050.png)

#### Conclusion

- introduce variety loss which encourages the generative network of GAN to spread its distribution and cover the space of possible paths while being consistent with the observed inputs.
- a new pooling mechanism that learns a global pooling vector which encodes the subtle cues for all people involved in a scene.

**level**: CVPR
**author**:Chih-Yao Ma ,Min-Hung Chen
**date**: 30 Mar 2017 
**keyword**:

- LSTM, action prediction

------

## Paper: TS-LSTM and Temporal-Inception

<div align=center>
<br/>
<b>Exploiting Spatiotemporal Dynamics for Activity Recognition</b>
</div>


#### Proble Statement

- methods extending the basic-stream ConvNet have not systematically explored possible network architectures to further exploit spatiotemporal dynamics within video sequences.The network often use different baseline two-stream networks.
- traditional two-stream ConvNets unable to expoit the most critical component in action recognition<font color=red> visual appearance across both spatial and temporal streams and their correlations are not considered </font>,
- previous work mainly try individual methods with little analysis of whether and how they can successfully use temporal information.
- each individual work uses different networks for the baseline two-stream approach with varied performance depending on training and testing procedure as well as the optical flow method used.

previous work:

- <font color=red> hand-craft or learned features for training</font>   3D ConvNets:  [9] stacked consecutive video frames and extended the first convolutional layer to learn the spatiotemporal features while exploring different fusion approaches including early fusion and slow fusion. C3D[20] replacing all the 2D convolutional kernels with 3D kernels at the expense of GPU memory. [16] factorize the original 3D kernels into 2D spatial and 1D temporal kernels and achieve comparable performance.  <font color=red> multiple layers can extract temporal correlations at different time scales and provide better capability to distinguish different types of actions</font>
- ConvNets with RNNs: directly take variable length inputs and learn long-term dependencies.
- Two-stream ConvNets:spatial features and temporal features from optical flow images.  <font color=red>we only use the feature vector representations instead of features maps</font>

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191124134305103.png)

【Qustion 1】Spatial stream  $ Temporal stream

spatial stream: the ResNet-101 spatial-stream ConvNet is pre-trained on ImageNet and fine-tured on RGB images extracted from UCF101 datasets with classification loss for predicting activities.

Temporal stream: stacking 10 optical flow images for temporal stream has been considered as a standard for two-stream ConvNets[13,6,28,25,27] <font color=red>follow the same pre-train procedure shown by [25]</font>

[model 1] **Temporal Segment LSTM**:  using 25 to divide the sampled video frames into several segments, a temporal pooling layer is applied to extrct distinguishing features from each of the segments.and LSTM is used ot extract the embedded features from all segments.

 ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191124140942805.png)

[model 2] Temporal-ConvNet :leveraging the temporal relation across diferent frames.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191124141556429.png)

![image-20191124141419689](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191124141419689.png)

different types o faction have different temporal characteristics and different kernels in different layers essentially search for different actions by expoiting different receptive fields to encode the temporal characteristics.

#### Evaluation

- **Environment**:   
  - Dataset:  experiment on spatial-stream ,temporal-stream and two-stream on three different splits in the UCF101, and HMDB51 datasets.
- comparison evaluation 这部分没有看,如果以后用到再来细看

#### Conclusion

- first demostrate a strong baseline two-stream ConvNet using ResNet-101.
- propose and investigate two different networks to further integrate spatiotemporal information: temporal segment RNN  and Inception-style Temporal-ConvNet.  but all need propercare.

#### Notes <font color=orange>去加强了解</font>

- [ ] [13] incorporate spatial and temporal information extracted from RGB and optical flow images.Two-stream convolutional networks for action recognition in videos
- [ ] 了解学习 14,18 8 模型 [25][28 ] 7 
- [ ] code available: 
- [ ] [6] fusion stage   Convolutional two-stream network fusion for video action recognition
- [ ] Temporal segment networks: Towards good practices for deep action recognition
- [ ] optical methods Brox[2] or TV-L1[29],and the results ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191124141955787.png)

**level**:   2019 winter conference on application of computer vision
**author**: Erwin Wu   Tokyo Institute of Technology
**date**: 2019
**keyword**:

- action prediction

------

## Paper: FutruePose

<div align=center>
<br/>
<b>Mixed Reality Martial Arts Training using Real-time 3D Human Pose Forecasting with a RGB Camera</b>
</div>


#### Summary

1. 这篇文章将人体2D坐标和光流信息结合起来通过LSTM网络预测未来0.5s 姿态2D坐标,并使用VNect网络建立3D模型,去3D模型上的若干点通过数值分析模型判断是否碰撞.
2. shortcoming: 
   1. only experiment on boxing,there are still other activities
   2. foces on inference and accuracies on different algorithms ,if using hyper-parameter like d for lattice point flow and threshhold for noise filter
   3. the forcasting information is limited, if building an orientation-based 3D pose estimation by dividing the human body into different parts and learning the bone rotation ,not only to related to their mother joints but relative to the entire body part
   4. the frame rate of normal high speed movement like keck form a professional martial athlete
   5. this paper focus on single person ,if there are multiperson.

#### Research Objective

- **Application Area**:analyse a player's habit ,determinate strengths and predict next movement 
- **Purpose**:  a novel mixed reality martial arts training system using deep learning based real time human pose forecasting.

#### Proble Statement

- Recent 3D motion capture systems are based on<font color=red> fabric technology</font> ,requiring to wear specific suits or sensors.
- special cameral <font color=red>RGB-Depth,IR cameras</font>
- <font color=red>normal dense optical flow</font> requires many computations and leads to a heavy inference time in LSTM

previous work:

- Martial Sports in AR/VR: wear VR HMD and take a pair of controllers
- Real-time 3D pose estimation:
  - VNect :provide a better accuracy for the 3D skeleton recognition with less computation and good real-time ability, can't be used in multi-person detection.
  - OpenPose detect multiple people in a single image,but the inference time is greater.
  - [20] 3D pose Recovery using a simple and deep neural network with only two linear layers and two residual blocks ,demostrate 3D pose could be created by 2D joint positions.
- Pose forecasting:
  - 3D-PFNet :the first to forecasting human dynamics from single RGB images. forcasting 2D skeletal poses and converting them into 3D space   87.6mm error.  <font color=red>offline network require large computation</font>
  - [12] forecast human body motion 0.5s advance using five layered neural network   ,7.9cm  <font color=red>IR sensor is not suitable to use in an outdoor environment or a large area.  not for more complicated athletic movemnet such as boxing</font>

#### Methods

- **Problem Formulation**:forecasting of 3D pose from a single image and the model fitting and collision detection.
- **system overview**:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116143629118.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116152552658.png)

【Qustion 1】<font color=red>how to estmate 2D pose?</font> cropped using bounding box tracker.

use ResNet50[10] to allow the convolutional layer to regress the 2D joint data 

【Qustion 2】 <font color=red>2D pose forecasting?</font>

using optical flow and joint positions data to do a regression on the LSTMs.  developed a sparse optical flow called Keypoint Lattice-Optical Flow ,creates several lattice points and only calculates the optical flows of the lattice points which close to keypoint.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116153025285.png)

【Qustion 3】 <font color=red>3D pose recovery?</font>  [20] an effective 3D pose recovery  using VNect network

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116153558928.png)

【Qustion 4】<font color=red>How to understand person's position and detect the collision in virtual environment ?</font>

using 3D model to represent user and have a surface to collide with one another.

using makehumanAPI to gernerate 3D model![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116154051111.png)

divided model into more than 200 segments,called 'hulls',each of these hulls contains a convex(凸) collider .detect a collision between two hulls using basic convex polytopes.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116154239879.png)

#### Evaluation

- **Environment**:   

  - Hardware: using TensorFlow on the TSUBAME3.0( Xeon E5-2680  v4 CPU*2,Nvidia SXM2 P100 GPU*4),tensorflow1.4.1,cuda 8.0 cudnn 5.1lib    HTC VIve(VR HMD),Sony DSCQX10 camera,LogitechC270 webcamera.
  - Dataset: MPI-INF-3D  and Human36M datasets for pre-training and validation. ratio of 6:2:2 for
    training, testing, and validation

- **result**

  **RealTimePerformance**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116155736081.png)

**Pose forecasting accuracy:**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116155725964.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116155800489.png)

**User case study:**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116161933715.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191116162001653.png)

#### Contribution

- the first to realize real-time 3D human pose forecasting based on normal video frames and apply it to mixed reality martial arts use
- a customized residual network[10]to obtain 2d human joints ,uses recurrent networks to learn the temporal features of the human motion.
- use a lattice optical flow algorithm to calculate the joint movement with less computation

#### Notes <font color=orange>去加强了解</font>

- [ ] paper[20],[12],[22],[6],19],[18]
- [x] 3D-PFNet  12
- [x] PCKh@0.05 evaluation [1] measure which calculates the percentage of correct key point that uses a matching threshold of 50% of the head segment length.
- [x] RMSE: The root-mean-squared error (RMSE) was also calculated to show the deviation of the predicted data



------

**level**:  ACM  
**author**: Yuuki Horiuchi , Yasutoshi Makino
**date**: 2017 .10
**keyword**:

- Machine learning , Motion estimation,Human-centered computing ,computing methodologies

------

## Paper: Computational Foresight

<div align=center>
<br/>
<b>Forecasting Human Body Motion in Real-time for Reducing Delays in Interactive System</b>
</div>

#### Research Objective

- **Application Area**:<font color=red>instruct sports actions ,prevent elderly form falling to the ground, prevent accident in advance. reducing delay in remote interactive system</font>
- **Purpose**:  forecast human body 0.5s before the actual motion in real-time  with accuracy of 7.9cm

#### Proble Statement

- diverse communication with remote areas has become possible, <font color=red>information transmission delay</font>

previous work:

- Holoportation system[1] communicate with remote people using HMD.
- TELESAR V system[2] feel object through remotely connected robot with haptic sensing and feedback.
- pattern categorized or estimation using DNN,  Predicting trajectory of movie in realtime.  <font color=red>there is no research that forecast body motions which is not repetitive and personalized but universal in realtime and visualize it to user.</font>

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122164724372.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122164824683.png)

【Qustion 1】how to extract 25 body point and COG?    论文[14] 需要学一下

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122165225765.png)

[解决问题2]Neural Network design ?

combine past 10 frames of 26 data as a one learning dataset .(joints+COG position)*3 demensions(x,y,z)* * 10 frames=780

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122165657463.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122165302517.png)

[解决问题3] 损失函数 和 优化器选择

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122165849130.png)

![image-20191122170046854](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191122170046854.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122171352648.png)

#### Evaluation

- **Environment**:   
  - Dataset:   Kinect V2  eleven subject to jump as many time as they could ,one duration for one minute. they allow to jump either ways in random order and the distance less than 2.5m. 
  - laptop(CPU: intel core-i7-7820HK 2.9-3.9Ghz,GPU: Nvidia Geforece GTX 1080)  3.32ms for COG NN  matrix operation, 5.75ms for redering bone image less than 33ms for measuring depth map and 3D position of 25 body joints and COG
  - ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122171323598.png)
  - ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122170458248.png)
- 

#### Conclusion

- 均方误差MSE   注:RMSE（即MSE的平方根)![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122171538058.png)
- 平均绝对误差（MAE）![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191122171610156.png)

#### Notes <font color=orange>去加强了解</font>

- human gesture recognition by using depth map [10]  Neural network for 
  dynamic human motion prediction
- 论文14 计算重心



------

**level**: CVPR   CCF A
**author**: Junwei Liang ,Li Fei-Fei
**date**: '2019-05-31'
**keyword**:

- LSTM, activity prediction

------

## Paper: Peeking into the future

<div align=center>
<br/>
<b>Predicting future person activities and location in videos</b>
</div>


#### Summary

这篇文章通过 分析 人的位置，行为，与周围事务的距离，周围的环境信息来预测未来轨迹何未来动作，并通过位置预测算法来减少 人位置的累计误差。

1. 在编码用户与周围的事务互动时，能不能编码用户的之间的动作联系对应起来，而不是简单的距离

#### Research Objective

- **Application Area**:Future person path/trajectory activity prediction (accident avoidance , smart personal assistance , self-driving car , socially-aware robots , anticipating pedestrian movement at traffic intersections or a road)
- **Purpose**:  deciphering human behaviors to predict pedestrian's future path jointly with future activities.

#### Proble Statement

- Humans navigate through public spaces often with specific purposes in mind.

previous work:

- Person-person models for trajectory prediction.
  - [32,34] predict person path by considering human social interactions and behaviors in crowded scene.
  - [36]learned human hehavior in crowds by imitating a decision-making process
  - Social-LSTM[1] added social pooling to model nearby pedestrian trajectory patterns.
  - Social-GAN[7] added advertisarial training on social LSTM to improve perfomance.
  - <font color=red>they simply consider a person as points ,we use geometric ralation to explicitly model the person-scene interaction and the person-object relatoinsj</font>
- Person-scene models for trajectory prediction :learning the effect of the physical scene
  - [13] using Inverse Reinforcement learning to forecast human trajectory
  - Scene-LSTM divided the static scene into Manhattan Grid and predict pedestrian's location using LSTM
  - CAR-Net proposed an attention network on top of scene semantic CNN to predict person trajectory
  - SoPhie vombined deep neural network features form scene semantic segmentation model and generatice adbersarial network using attention to model person trajectory
  - <font color=red>we explicitly pool scene semantic features around each person at each time instant ,the model directly learn from such interactions</font>
- Person visual features for trajectory prediction  :using individual's visual features instead of considering them as points in the scene.
  - [14] looked at pedestrian 's faces to model their awareness to  predict whether they wil corss the road using Dynamic Bayesian Network .
  - [33] person keypoint features with a convolutional neural network to predict future path .
  - <font color=red>we consider both person behavior and their interactions with soundings</font>
- Activity prediction /early recognition 
  - [29] utilized unsupervised learning with LSTM to reconstruct and predict video representations.
- Multiple cues for tracking/group activity recognition:  
  - previous works take into account multiple cues in video for tracking ,group activity recognition
  - <font color=red>rich vision features focal attention,  location prediction to bridge the two taks</font>
- most existing work [31,1,7,26,21,31] which oversimplifies a person as a point in space,we encode a person through <font color=red>rich semantic features about visual appearance ,body movement ,and interaction with the surroundings ,</font>

#### Methods

- **Problem Formulation**:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109103802813.png)

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109103550662.png)

【Qustion 1】<font color=red>how to model the person's appearance and body movement about every individual in a scene</font>?

- utilize a pre-trained object detection model with "**RoIAlign[8**]" to extrace fixed size CNN features for each person bounding box. for every person ,average the feature along the spatial dimentions and feed them into LSTM encoder   -> obtain T*d ,where d is the hidden size of the LSTM.
- utilize a person key-point detection model trained on MSCOCO dataset[6] to extract preson keypoint information.we apply <font color=red>the linear transformation to embed the keypoint coordinates ,这里的线性处理不理解</font>before feed into LSTM.->obtain T*d ,where d is the hidden size of the LSTM![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109111014991.png)

【Qustion 2】<font color=red>how to model the interaction between a person and their surroundings,person-scene and person-object</font>

- Person-scene: <font color=red>whether the person is near the sidewolk or grass</font> use a **pre-trained scene segmentation model[4]** to extract pixel-level scene semantic classes(10 class eg.roads , sidewalks...) for each frame. the scene   the semantic features are integeres of the size T * h * w ,Given a person's xy coordinate ,we pool the scene features at the person;s current location from the convolution feature map.  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109111844802.png)

- Person-object: <font color=red>how far away the person is to the other person or object</font>models the **geometric relation** and **the object type** of all objects/persons in the scene.在论文[9]中证明了这个方法的高效性![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109112732633.png)

  - geometric relation:                   

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109112036904.png)

  - object type: 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109112647171.png)

【Qustion 3】<font color=red>how to predict the trajectory ?</font> using effective focal attention[17]  ,原始模型见[7]

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109113310088.png)

![image-20191109113735888](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191109113735888.png)

【Qustion 4】<font color=red>how to predict activity ?</font> introduce an auxiliary task :activity location prediction in addition to predicting the future activity label of the person .![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109114434310.png)

- activity location prediction with Manhattan Grid  (location classification(to predict correct grid block in which the  final location coordinates reside),location regression(to predict the deviation of the grid block center to final location coordinate))   <font color=red>how to accurate localization using multi-scale features in a cost-effective way</font>
- activity label prediction: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191109142619975.png)

#### Evaluation

- **Environment**:   
  - Dataset: ActEV/ViRAT 
- model the intention in terms of a predefined set of 29 activities provided by NIST .

#### Conclusion

- propose an end-to-end multi-task learnig system<font color=blue> utilizing rich visual features about human behavioral information and interaction with their surroundings</font> .
- the first empirical evidence that joint medeling of paths and activities benefits future path prediction.
  - learning activity together with the path may benefit the future path prediction
  - joint model advances the capability of understanding not only the future path but also the future activity by taking into account the rich semantic context in videos.
  - introduce an auxiliary task for future activity prediction,activity location.
- propose multi-task learning framework with new techniques to tackle the challenge of joint future path and activity prediction.
- validate the model on two benchmarks: ETH&UCY , and ActEV/VIRAT.

#### Notes <font color=orange>去加强了解</font>

- [ ] Effective focal attention was originally proposed to carry out multimode inference over a sequence of images for visual question answering. key idea is project multiple features into a space of correlation where discriminative features can be easier to capture by attention mechanism.
- [ ] Attention mechanism   ???
- [ ] 论文37 decision-making process 方法是什么？
- [ ] [13] using Inverse Reinforcement  方法是什么
- [ ] [33] person keypoint features to predict trajectory ?
- [ ] **RoIAlign[8**]   学习使用这个网络
- [ ] **pre-trained scene segmentation model[4]**  学习了解下场景分割技术
- [ ] Code 学习使用： https://github.com/google/next-prediction 



------

**level**: CVPR ccf A
**author**:  alexandre Alahi , Kratarth Goel     stanford.edu
**date**:
**keyword**:

- 

------

## Paper: Social LSTM

<div align=center>
<br/>
<b>Human Trajectory Prediction in Crowded Space</b>
</div>


#### Research Objective

- **Application Area**: social aware roots[41], intelligent tracking system[43]
- **Purpose**:  predict the motion dynamics in crowded scenes.

#### Proble Statement

previous work:

- they use hand-craft functions(人工特征) to model interactions for specific settings rather than inferring them in data driven fashion.
- they focus on modeling interactions among people in close proximity to each other(to avoid immediate collisions), don‘t anticipate interactions that could occur in the more distant future.
- RNN model for sequence prediction (speech recognition , caption generation , machine translation , image/vedio classification, human dynamic)
  - Model  and <font color=red>Gated Recurrent Units[12]</font>  most common methods.
  - [20] predict isolated handwriting sequence 

#### Methods

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112092925006.png)

【定义问题1】every person has different motion pattern,they move with different velocities ,acceleration and have different gaits ,how to model person-specific motion properties from a limited set of initial observation corrosponding to the person

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112092949875.png)

![image-20191112093015584](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191112093015584.png)

【定义问题2】 every person has a different number of neighbors and in very dese crowds,the number could prohibitively high?

 a compact representaion "Social " pooling layers ,and preserve the spatial information through grid based pooling .

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112094414250.png)

【定义问题3】 how to estimate the Position?

![image-20191112100947577](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191112100947577.png)



![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112101003868.png)

【定义问题4】 how to deal with occupancy map pooling?

the Social LSTM model can be used to pool any set of features from neighboring trajectory ,and learn to reposition a trajectory to avoid immediate collision with neighbors.<font color=red>这一部分不太明白</font>

#### Evaluation

- **Environment**: ETH  ,UCY
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112101541086.png)

#### Conclusion

- introduce the Social pooling layer which allows the LSTMs of partially proximal sequences to share their hidden-states with each other.
- analyze the trajectory patterns generated by our model to understant the social constrains learned from the trajectory datasets.
- predicting the trajectories of pedestrians much  more accurately than state-of-the-art models on ETH,UCY

#### Notes  <font color=yellow>去加强了解</font>

- [ ] Generating sequences with recurrent neural networks

- [ ] LSTM speech generation【21】demo  去github上找代码 

- [ ] [32] 学习和了解 Inverse Reinforcement Learning to predict human paths in static scenes.

- [ ] Theano: A cpu and gpu math compiler in python

- [x] bivariate Gaussian distribution多元正态分布![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191121101725026.png)



------

**level**: IEEE Access   CCF B 类
**author**: 10.25.2019
**date**: '2019-10-25'
**keyword**:

- Action recognition,deep learning ,pedestrian detection ,time-to-cross estimation

------

## Paper: Multi-Task Pedestrian 

<div align=center>
<br/>
<b>Multi-Task Deep learning for Pedestrian Detection ,Action Recognition and Time to Cross Prediction</b>
</div>


#### Summary

这篇文章解决了如何去检测行人，识别行人的动作（利用JAAD数据库和现有的方法），并且预测了在行人过马路的状态下穿过时间预测。使用了RetinaNet 网络检测，LSTM网络去预测。

问题：

1. 如何去检测多个人的，以及多个人的相应的动作  引用文章的没有细说？需要看下
2. 在预测的时候LSTM网络仅仅是BB坐标，每个人的步速步伐大小可能不一样这里应该怎么解决？
3. 现有RF实现骨架检测技术，以及手势识别技术，人体骨架各部分运动检测，能否用LSTM预测下一个动作，这是预测行人过马路，如果在室内可能需要检测或预测哪些动作

#### Research Objective

- **Application Area**:understand the intention of road users involved to ensure their safety and secure the traffic flow.
- **Purpose**:  estimate TTC.
- **System_Design**: 
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191106112647294.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191106111040275.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191106111058015.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191106111116174.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191106111520160.png)

#### Proble Statement

- **pedestrian detection problem**: progress in pedestrian detection is hindered by the difficulty of detecting all(partially)occluded pedestrians and the problem of operating efficiently in severe weather conditions.
- **ADAS need to solve three problems**: 1. a detection model for localizing and recognizing the pedestrians among other road users 2. a prediction model to estimate the pedestrian actions over next frames(short,medium,long-time prediction)
- **Datashortcoming:** there are no public databases annotated with pedestrian time to cross while there are several interesting huge pedestrian detection databases(Kitti,caltech,among others),and some databases don't provide any pedestrian action labels
- Estimation of the pedestrian intention and especcially of the pedestrian  actions is even more challenging because of the <font color=red>ambiguities</font> in pedestrian motions.

previous work:

- pedestrian movement and pedestrian behaviors[13],[14],interacctions between pedestrians[15] [16] ,pedestrian tracking paths[9] ,[10],  a review of the predicting pedestrian behavior[12],<font color="red">pedestrian intention requires to use pedestrian specific dynamic information and contextual road environment</font> ,in [17]  present a pedestrian action recognition based on <font color='red'>AlexNet handling JAAD dataset and use temporal and spatial-temporal contextual information to increase the prediction perfomance</font>
- [9] A pedestrian position estimation based on the<font color=red> Extended Kalman Filter and Interacting Multiple Model algorithm using Constant Velocity</font>
- [18] combination of the Gaussian Precess Dynamic Models ,Probablistic Hierarchical Trajectory Machine with Kalman Filter and interacting Multiple Model-based on the Daimler Data.
- [10] A short-term prediction of pedestian behaviors <font color=blue>using Daimler datasets,to predict the pedestrian trajectory and its final destination using CNN base on LSTM and path planning</font>.
- [13] mixture of CNN based pedestrian detection tracking and pose estimation to predict the pedestrian crossing actions based on the JAAD dataset 
- **Summary**:previous only discriminates between the pedetrian from the non-pedestrian among other road users and estimates the pedestrian action or its final destination for the next frames（short medium and long term) <font color="red">the Time to cross estimation of pedestrians is more challengin than predicting the pedestrian action since it requires contextual spatial-termprary:a fine analysis  of the pedestrian motion and the whole scene</font>

#### Methods         [2],[19]

【定义问题0】no public databases anntated with pedestrian time to cross,the databases don't provide any pedestrian action labels?

we select some cues from the JAAD [1] public data set in order to solve this issue and then we made our pedestrian TTC annotation for all videos.  <font color=red>这个cue是指什么</font>    JAAD 数据已经包括了 pedestrian bounding boxes for pedestrian detection and pedestrian attributes.

【定义问题1】how to detect pedestrian ?

Applying a generic object detector based on the public RetineNet[2], the author handled the Resnet50[19]CNN architecture for the classification task with the Keras public open-source implementation described in [2],all the training process is based on the JAAD dataset,which provides an annotation of pedestrians with behavioral tags and pedestrians without behaciors tags.

![image-20191106112333433](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191106112333433.png)

【定义问题2】how to split the pedestrian Joint Attention for Autonomous Driving into four class?   previous work

【定义问题3】how to estimate Time to cross?

![image-20191106112423122](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191106112423122.png)

#### Evaluation

- **Environment**: dataset: JAAD dataset[17] provides pedestrian bounding boxes for pedestrian detection,pedestrian attributes for estimating the pedestrian behavior and traffic scene elements.

#### Conclusion

- Train all pedestrian Bounding Boxes samples with the <font color='red'>RetinaNet</font> for pedestrian detection purpose
- Split the pedestrian Joint Attention for Autonomous Driving(<font color="red">JAAD</font>) data set into four classes for pedestrian action functionality :pedestrian is preparing to cross the street ,pedestrian is crossing the street ,pedestrian is about to cross the street and pedestrian intention is ambiguous
- Train <font color="red">LSTM</font> model using only BB coordinates in order to estimate the time to cross of each pedestrian.

#### Notes  <font color=orange>去加强了解</font>

- [ ] 了解 JAAD 数据集什么格式
- [ ] 学习和运行RetinaNet网络
- [ ] 学习何使用AlexNet 网洛
- [x] 了解LSTM网络
- [ ] LSTM 网络运行使用
- [ ] 论文 9，18，10中的方法
- [ ] SSD 网络学习使用
- [ ] Faster-RCNN网络学习使用
- [ ] Yolo3网络学习使用

# 4.感知系统

**level**: Pro.ACM Interact.Mob   Wearable Ubiquitous Technol
**author**:  	karan ahuja Carnegie Mello University  
**date**: 2019 9
**keyword**:

- Human-centered computing ,Interactive systems and tools, Classroom sensing,Compute Vision,Speech

------

## Paper: EduSense

<div align=center>
<br/>
<b>Practical Classroom Sensing at Scale</b>
</div>


#### Proble Statement

- need an expert to instructs   expensive
- lack of sufficient feedback opportunities on pedagogical skill

previous work:

- Instrumented Classrooms
  - instrumented with <font color=red> pressure sensors</font> to characterize varying levels of interest and engagement,such as slumped back sitting upright.
  - Affectiva's wrist-worn Q sensor[62] which senses the wearer's skin conducance ,temperature and motion to infer engegement level.
  - EngageMeter[32] used electroencephalography headsets to detect shifts in student engagement,alertness and workload
- Non-Invasive Class Sensing
  - [19] an omnidirectional room microphone and head-mounted teacher microphone to automatically segment teacher and student speech events, intervals of silence.
  - Oral presentation practice systems AwareMe[11],Presentation Sensei[46],PoboCOP[75] compute speeck quality metrics (pitch variety,pauses fillers speaking rate)
  - Equally versatile using cameras ,detect hand rises,skin tone ,edge detection
  - Robust Face detection find and count student,estimate their  head orientation,coarsely signaling their area of focus,facial landmarks to analyse engagement,frustration,off-task behavior.

#### Methods

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112104005467.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112114633868.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112114906902.png)

【定义问题1】Featurization Modules

- Sit & Stand detection: using body keypoints hips , knees , feet  and ratio of distances between chest and foot, the chest and knee both legs.

- Hand Raise detection: neck ,chest, shoulder elbow wrist and compute the direction unit vectors btween all pairs of these points,and compute the distance between all pairs of points ,normalized by the distance between all pairs of these points.

- Upper Body detection: utilize the same eight upper body keypoints  to predict arms at rest, arms at closed and hands on face.![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112115847153.png)

- Smile Detection: ten mouth landmarks on the outer lip and ten landmarks on the inner lip. compute direction unit vectors from the left lip corner to all other points  SVM to binary classifaction.

- Mouse Open Detection: estimate if a mouse is open,to produce the talking confidence.use a Binary SVM and two highly descriptive features adapted from [71 predict eys open & closed],the height of the mouth to the left and right of center ,divided by the width of the mouth.

- Head Oriention & Class Gaze : Using a perspective-n-point algorithm[50] in combination with anthropometric face data[53],produces a coarse 3D orientation of the head for each body.

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112121139650.png)

- Body Position & Classroom Topology: perspective-n-point produces the orientation for each body founded in a scene ,and estimate 3D position in real world coordinates. to reveal the classroom topologies and help illuminate spatial patterns in the class.

- Synthetic Accelerometer: uuse the 3D head position produced during scene parsing and claculate a delta X/Y/Z .

- Student & Instructor Speech: the RMS of the student-facing camera's microphone,the RMS of the instructor facing camera's microphone ,the ratio between the latter two values. uisng random forest classifier to predict the current speech is coming from the instructor or students.

- Speech Act delimiting: 

#### Evaluation

- **Environment**: 
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191112121150907.png)

#### Conclusion

- a comprehensive sensing system that produces a plethora of theoretically-motivated visual and audio features correlated with effective instruction.
- the first to unify them into a cohesive real-time,in-the-wild evaluated and practically-deployable system.

#### Notes

- [ ] Classroom Discourse Analyzer [15] 了解这个系统
- [ ] [19]了解下这篇文章
- [ ] 了解下  Equally versatile 系统
- [ ] 了解 CERT 技术
- [ ] 了解下FFMPEG
- [ ] NVIDIA Visual Profiler 技术是什么
- [ ] 学习使用OpenPose   dlib64-point face landmarks【44】
- [ ] adaptive background noise filter  to remove background noises how??
- [ ] open source system  http://www.EduSense.io

# 5. 骨骼提取

## 5.1. [40个骨骼提取开源项目](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247507854&idx=2&sn=e02294c31e6867bf2e4270178c2a75e8&chksm=ec1c3277db6bbb612de6d5b3edaa2a4fb8da19e7b6e62b10feb6c21d8fddc748e570ef05c050&scene=126&sessionid=1600264884&key=3542bed875d644de951ff14ae71a83001ab1e1812ce7aa66a998b68fd92197d07eb6ed465593d3aac71554ab7ccfb47f11533fe51eec433dba65046ba6244a25e8050051445366bb86635b10cd4bcf9f1c9220c9515042c7795e056f147b995b4688cde692c65c611ca97ef9d78c191310ee8ebb1c30e3cd4a12b77aa5d24fe7&ascene=1&uin=MzE0ODMxOTQzMQ%3D%3D&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=Ax1vPAqMPtdpFIw823EzgRY%3D&pass_ticket=TfC86Xzy4b6ESRk%2FasnYpQs4p0qrNXFR4RzKdh4co%2FPl3pb2EHboMmNDJmdTviPd&wx_header=0)

**level**: CVPR  CCF A 
**author**: Zhe Cao 
**date**: '2019-5-30'
**keyword**:

- 2D human pose estimate ，2D foot keypoint estimate，real time，multiple person，part affinity fields

------

## Paper: OpenPose   

<div align=center>
<br/>
<b>Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields</b>
</div>


#### Summary

- prove PAF refinement is critical and sufficient for high accuracy ,removing the body part confidence map refinement while increasing the network depth.
- using body and foot key-point detector.
- Open-Pose library

#### Research Objective

- **Application Area**: key-point detect eg: body skeleton ,hand skeleton ,face skeleton,hand skeleton ,the body part location for futher explored like prediction
- **Purpose**:  using part affinity fields to real-time calculate multiperson 2D pose.

#### Problem Statement

- each image may contain an <font color=red>unknown number</font> of people that can appear at any <font color=red>position or scale</font>
- <font color=red>interactions</font> between people induce <font color=red> complex spatial interference due to contact occlusion or limb articulations,making association of parts difficult</font>
- runtime complexity tends to grow with the number of people 

previous work:

- **Single Person Pose Estimation**: perform inference over a combination of local observations on body parts and the spatial dependencies . the spatial model for articulated pose is either based on <font color=red>tree-structured graphical models</font>, which parametrically encode the spatial relationship between adjacent parts following a kinematic chain or not tree models that <font color=red>augment the tree structure with additional edges </font>to capture occlusion symmetry and long range relationship.

  - [34] used a multi-stage architecture based on a sequential prediction framework, incorporating global context to refine part confidence maps and preserving multi-modal uncertainty form previous iterations.
  - <font color=red>all this methods assume a single person ,where the location and scale of the person of interest is given.</font>

- **Multi-person Pose estimation**: <font color=red>Top-down</font> strategy that first detects people and then have estimated the pose of each person independently on  detected region.<font color=blue>suffers form early commitment  on person detection,fail to capture the spatial dependencies across different people</font> .<font color=red>bottom up</font> approach that jointly labels part detection candidates and associated them to individual people,with pairwise scores regressed from spatial offsets of detections

  - [47] further simplified their body-part relationship graph for faster inference in single-frame model and formulated articulated human tracking as spatial-temporal grouping of part proposals.
  - [49] detect individual key-points and predict their relative displacements,using greedy decoding method.

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191111163031878.png)

#### Methods

**system overview**: given a color image of size w*h ,produces the 2D positions of anatomical key-points for each person in the image.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191111163106168.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191111182334970.png)

【Question1】how to detect limbs and bodypart?

define <font color=red> body part location S</font>:  
$$
S=(S_1,S_2,...,S_j)  ,S_j\varepsilon R^{w*h*2}
$$
define <font color=red>vetor field of PAFS L</font>:
$$
L=(L_1,L_2,...,L_j) , L_c represent a limb,L_C\varepsilon R^{w*h*2}
$$

$$
Image->\frac{VGG-19}{CNN}->asetofeature mapsF\frac{stage\quad t,(t<T_p)}{limbs}>\begin{cases} L^1=\phi^1(F),t=1\\ L^t=\phi ^t(F,L^{t-1}),\forall 2\leq t\leq T_p \end{cases}\quad \frac{}{bodyPartLocation}>\begin{cases} S^T_p=\rho(F,L^T_p),\forall t\varepsilon T_p\\ S^t=\rho(F,L^{T_p},S^{t-1}),\forall T_p\le t\leq T_p+c \end{cases}
$$

<font color=red>define Loss functions</font>:  w is a binary mask with  W(p)=0 when annotation is missing.
$$
L_{stage}: f_L^{t_i}=\sum_{c=1}^{C}\sum_Pw(p)*||L_c^{ti}(p)-L_c^*(p)||^2 \\
S_{stage}: f_S^{t_i}=\sum_{j=1}^{J}\sum_Pw(p)*||S_j^{ti}(p)-S_j^*(p)||^2
$$
<font color=red>For vanishing gradient </font>:
$$
f=\sum_{i=1}^{T_p}f_L^t+\sum_{t=T_p+1}^{T_p+T_c}f_S^t
$$
confidence map: maximum   ,belief a particularly body part can be located at any given pixel

obtain body part conditions: Non-maximum suppression

<font color=red>confidence mpas:</font>  $S_{j,k}^* $   :  individual maps for each person k ,  $X_{j,k}\epsilon R^2$ ground truth,body part j for person k

for calculate the confidence body part: the value at location $p\epsilon R^2 \quad S_{j,k}^*(p)=exp(-\frac{||p-x_{j,k}||^2}{\rho^2})$  ,$S_j^*(p)=max_kS_{j,k}^*(p)$

【Question2】Given a set of detected body parts ,how do we assemble them to form the full-body poses o fan unknown number of people？<font color=red> ecodes both the location and orientation, don’t reduce the region of support of a limbs to a single point</font>

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191129113237565.png)

$X_{j_1,k} ,X_{j_2,k}$  : the groundtruth positions of body parts j1,j2 from the limb c.
$$
L_{c,k}^*(p)=\begin{cases} v \quad if \quad p \quad on \quad limb c,k \\0 \quad otherwise\end{cases}, \frac {v=(x_{j2,k}-x_{j1,k})/||x_{j_2,k}-x_{j_1,k}||^2}{0\leq v*(p-x_{j_1,k})\leq l_{c,k} and |v\bot *(p-x_{j_1,k)})|\leq \rho_l}
$$
ground-truth part affinity field : $L_c^*(p)=\frac{1}{n_c(p)}\sum_kL_{c,k}^*(p)\quad n_c(p)$:is the number of non-zero vectors at point p across all k people.

eg: for two candidate part locations $d_{j_1}\quad d_{j2}$,we sample the predicted part affinity field, Lc alone the line segment to measure the confidence in their association:$E=\int_{u=0}^{u=1}{L_c(p(u))*\frac {d_{j2}-d{j1}}{||d_j2-d_j1||^2}du}dx$     where p(u) interpolates the position of two body parts dj1,dj2 ,$p(u)=(1-u)d_{j1}+ud_{j2}$

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191125091725286-1577068664572.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191129115349521.png)

a set of body part detection candidates Dj for multiple people,where $D_j={d_j^m :for\quad j\epsilon {1...J},m\epsilon {1,...,N_j}} $ where Nj is the number of candidates of part j, and $d_j^m\epsilon R^2$ is the location of m-th detection candidate of body part 

define $z_{j_1,j_2}^{mn}\epsilon (0,1)$ to indicate wether two detection candidates $d_{j_1}^m ,d_{j_2}^n $ are connected ,the goal is to find optimal assignment for the rest set of all possible connections ,
$$
Z={z_{j_1j_2}^{mn}:for \quad j_1,j_2\epsilon (1,...,J),m\epsilon(1,...,N_{j_1}),n\epsilon(1,...,N_{j_2})}​
$$
![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191129121106421.png)

1. choose the minimal number of edges to obtain a spanning tree skeleton of human pose rather than using the complete graph
2. decompose the matching problem into a set of bipartite matching subproblems and determine the matching in adjacent tree nodes independently.

#### Evaluation

- **Environment**: 
  - datasets : MPII human multi-person dataset[66] consisting of 3844 training and 1758 testing groups of multiple interacting individuals in highly articulated pose with 14 body parts, COCO keypoint challenge dataset  requires simultaneously detecting people and localizing 17 key-points

#### Conclusion

- present an explicit nonparametric representation of the key-point association that encodes <font color=red>both position and orientation of human limbs</font>
- design an architecture that jointly <font color=red>learns part detection and association </font>
- demonstrate that <font color=red> a greedy parsing algorithm is sufficient to produce high-quality parses of body poses and preserves efficiency regardless of the number of people </font>
- prove that PAF and body part location refinement is far more important that combined PAF and body part location refinement 
- combining body and foot estimation into a single model boosts thee accuracy of each component individually and reduces the inference time of running them sequentially
- open-sourced OpenPose system and included in OpenCV library.

#### Notes   <font color=orange>去加强了解</font>  记录下以下论文

- [ ] 学习何使用OpenPose system  ,进行到一般,模型文件需要下载
- [x] [20] Convolutional pose machines,   DenseNet[52]Densely connected convolutional networks, [3] Realtime multi-person 2d pose estimation using part affinity fields,  <font color=orange> 大致了解了网络,需要几个代码了解下运行效果</font>
- [x] 学习和使用 Mask R-CNN[5]  <font color=orange>need to practice with code</font>
- [x] 学习和使用Alpha-Pose[6]
- [x] ResNet[46] 学习使用    <font color=orange>need to practice with code</font>
- [ ] [2] pairwise representations   了解下是什么
- [ ] 论文34 ,47,49 ,[50]需要学习了解
- [ ] [49] detect individual key-points and predict their relative displacement allowing a greedy decoding process to group keypoints.
- [ ] 学习和了解VGG-19【53】模型并学会使用fine-tuned方法
- [x] Hungrian ALgorithm 用于解决二分图匹配问题

level**: CCF_A   CVPR

**author**: ChaoLi,QiaoyongZhong,DiXie,ShiliangPu HikvisionResearchInstitute 

**date**: 2018 4.17

**keyword**:

- skeleton based action recognition

------

## Paper: Co-occurence Feature 

<div align=center>
<br/>
<b>Co-occurrence Feature Learning fromS keletonData for ActionRecognition and  Detection with Hierarchical Aggregation
</b>
</div>
#### Summary

1. point-level information of each joint is encoded independently,and then assembled into semantic representation  in both spatial and temporal domains
2. independent point-level features learning  and cross joint co-occurrence feature learning

#### Research Objective

- **Application Area**:  intelligent surveillance system, human-computer interaction, game-control and robotics.
  - skeleton provides good representation for describing human actions
  - skeleton data are inherently robust against background noise and provide abstract information and high-level features of human action
  - compared with RGB data, skeleton data are extremely small in size
- **Purpose**:  

#### Proble Statement

previous work:

- design and extract co-occurrence features form skeleton sequences
  - pairwise relative position of each joint
  - spatial orientation of pair wise joints
  - statistics-based feature like Conv3Dj   HOJ3D
- RNN with LSTM to model the time series of skeleton prevalently
- CNN models to learn spatial-temporal features from skeletons 
  - cast the frame, joint, and coordinate dimensions of skeleton sequence into width, height, and channel of an image respectively  [Du et al., 2016]
  - 3D coordinates are separated into three gray-scale images [Ke et al.,2017]
  - a new skeleton transformer module to incorporate skeleton motion features [Li et al., 2017b]
  - Shortcoming:   the co-occurrence feature are aggregated locally, which may not be able to capture the long-range joint interactions involved in actions like wearing ..

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229112040352.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229115749562.png)

**[Input Define]**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229114736296.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229114716246.png)

**[Multiple Persons]**

- early fusion: all joints from multiple persons are stacked as input of the network, zero padding if less than pre-defined maximal number
- Late fusion:  input of multiple persons go through the same subnetwork and their conv6 features maps are merged with either concatenation along channels or element-wise maximum/mean operation

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229114756753.png)

**[Loss Function Define]**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229115411034.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229115422855.png)

#### Evaluation

- **Environment**:   
  - Dataset: 
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229115851023.png)
- ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200229115907932.png)

#### Conclusion

- CNN model for learning global co-occurrences from skeleton data
- end-to-end hierarchical feature learning network, where features are aggregated gradually from point level features to global co-occurrence features
- exploit multi-person feature fusion strategies 

#### Notes <font color=orange>去加强了解</font>

- [ ] recognition and detection benchmarks <font color=red>NTU RGB+D,SBU kinect Interaction and PKU-MMD</font>
- [ ] Learning actionlet ensemble for 3d human actionrecognition  2014
- [ ] Essential body-joint and atomic action detection for human activity recognition using longest common subsequence algorithm   2012
- [ ] A NewRepresentationofSkeletonSequencesfor3DAction Recognition   2017
- [ ] Co-occurrence feature learning for skeleton based action recognition using regularized deep lstm networks   2016
- [ ] PKU-MMD: A large scale benchmark for continuous multi-modal human action understanding.  2017
- [ ] Two-stream convolutional networks for action recognition in videos   2014
- [ ] window regression   Girshick et al., 2014 
- [ ] Cascade region proposal and global context for deep object detection  2016 
- [ ] An end-to-end spatiotemporal attention model for human action recognition fromskeletondata   2017 AAAi

**level**: 
**author**:
**date**:  2016
**keyword**:

- point detection; heatmap

------

# Paper: Stacked Hourglass

<div align=center>
<br/>
<b>Stacked Hourglass Networks for
Human Pose Estimation</b>
</div>


#### Summary

1. On MPII there is over a 2% average accuracy improvement across all joints, with as much as a 4-5% improvement on more difficult joints like the knees and ankles.
2. propose  a stacked hourglass networks for human pose estimation;

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201019210140895.png)

【**Single Hourglass module **】

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201019210700629.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201019210520022.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201020134658441.png)

```python
#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.autograd import Variable
#https://sourcegraph.com/github.com/raymon-tian/hourglass-facekeypoints-detection/-/blob/models.py
class HourGlass(nn.Module):
    """不改变特征图的高宽"""
    def __init__(self,n=4,f=128):
        """
        :param n: hourglass模块的层级数目
        :param f: hourglass模块中的特征图数量
        :return:
        """
        super(HourGlass,self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        # 上分支
        setattr(self,'res'+str(n)+'_1',Residual(f,f))
        # 下分支
        setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
        setattr(self,'res'+str(n)+'_2',Residual(f,f))
        if n > 1:
            self._init_layers(n-1,f)
        else:
            self.res_center = Residual(f,f)
        setattr(self,'res'+str(n)+'_3',Residual(f,f))
        setattr(self,'unsample'+str(n),Upsample(scale_factor=2))


    def _forward(self,x,n,f):
        # 上分支
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        # 下分支
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1,n-1,f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)
        print(up1.shape,up2.shape)
        return up1+up2

    def forward(self,x):
        return self._forward(x,self._n,self._f)

class Residual(nn.Module):
    """
    残差模块，并不改变特征图的宽高
    """
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        # 卷积模块
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs//2,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs//2,3,1,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs,1)
        )
        # 跳层
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs
    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

class Lin(nn.Module):
    def __init__(self,numIn=128,numout=4):
        super(Lin,self).__init__()
        self.conv = nn.Conv2d(numIn,numout,1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class KFSGNet(nn.Module):

    def __init__(self):
        super(KFSGNet,self).__init__()
        self.__conv1 = nn.Conv2d(1,64,1)
        self.__relu1 = nn.ReLU(inplace=True)
        self.__conv2 = nn.Conv2d(64,128,1)
        self.__relu2 = nn.ReLU(inplace=True)
        self.__hg = HourGlass()
        self.__lin = Lin()
    def forward(self,x):
        x = self.__relu1(self.__conv1(x))
        x = self.__relu2(self.__conv2(x))
        x = self.__hg(x)
        x = self.__lin(x)
        return x


from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.optim as optim

class tempDataset(Dataset):
    def __init__(self):
        self.X = np.random.randn(16,1,512, 512)
        self.Y = np.random.randn(16,4,512, 512)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        # 这里返回的时候不要设置batch_size
        return self.X[item],self.Y[item]

if __name__ == '__main__':
    from torch.nn import MSELoss
    critical = MSELoss()

    dataset = tempDataset()
    dataLoader = DataLoader(dataset=dataset,batch_size=64)
    shg = KFSGNet().cuda()
    optimizer = optim.SGD(shg.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)

    for e in range(200):
        for i,(x,y) in enumerate(dataLoader):
            x = Variable(x,requires_grad=True).float().cuda()
            y = Variable(y).float().cuda()
            y_pred = shg.forward(x)
            #print(y_pred.shape,y.shape)
            loss = critical(y_pred[0],y[0])
            #print('loss : {}'.format(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i>2:
                break
        break
```


**level**: CVPR
**author**: Shih-En Wei
**date**: 
**keyword**:

- skeleton extract

## Paper: CPM

<div align=center>
<br/>
<b>Convolutional Pose Machine</b>
</div>


#### Summary

1. show a systematic design for how convolutional networks can be incorporated into the pose machine frame-work for learning image features and image-dependent spatial models for the task of pose estimation.
2. <font color=red>CPM: consists of a sequence of convolutional networks that repeatly produce 2D belief maps for the location of each part, at each stage in a CPM, image features and the belief maps produced by the previous stage are used as input</font>
   1. learn feature representations for both image and spatial context directly from data
   2. a different architecture that allows for globally joint trainning with back propagation
   3. efficiently handle large training datasets
3. <font color=red>large receptive fields on the belief maps are crucial for learning long range spatial relationships and result in improved accuracy.</font>

#### Research Object

previous work:

- classic approach:

  - <font color=red>pictorial structures:</font>  spatial correlations between parts of the body are expressed as a tree-structured graphical model with kinectmatic priors that couple connected limbs.
  - <font color=red>Hierarchical models:</font> represent relationships between parts at different scales and size in a hierarchical tree structure.
  - <font color=red>Non-tree models:</font> incorporate interactions that introduce loops to augment the tree structure with additional edges that capture symmetry, occlusion and long-range relationship.<font color=red>rely on approximate inference</font>
  - <font color=red>sequential prediction:</font> learn an implicit model with potentially complex interactions between variables by directly training an inference procedure.

  

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309105620041.png)

【**Pose Machines**】

![image-20200309105711936](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309105711936.png)

![image-20200309105732538](../../../../MEGA/MEGAsync/actionPrediction/skeleton.assets/image-20200309105732538.png)

![image-20200309105745261](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309105745261.png)

**【Convolutional Pose Machines】**

- Keypoint Localization Using Local Image Evidence:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309105837745.png)

- Sequential Prediction with Learned Spatial Context Features:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309105933784.png)

-  Learning in Convolutional Pose Machines

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309110012428.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200309110025606.png)



#### Conclusion

- learning implicit spatial models via a sequential composition of convolutional architectures
- a systematic approach to designing and training such an architecture to learn both image features and image-dependent spatial models for structured presiction tasks, without the need for any graphical model style inference.

#### Notes <font color=orange>去加强了解</font>

  - code available: https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release

**level**:  CVPR  CCF_A
**author**: Charles R.Qi  Stanford University
**date**:  
**keyword**:

- 3D object detection, Point Cloud

------

## Paper: Frustum PointNets

<div align=center>
<br/>
<b>Frustum PointNets for 3D Object Detection from RGB-D Data
</b>
</div>



#### Summary

1. 3D sensor data is often in the form of point clouds.

#### Research Objective

- **Application Area**: autonomous driving, augmented reality
- **Purpose**:  how to effective localize objects in point clouds of large-scale scenes.

#### Proble Statement

- study 3D object detection from RGB-D data in both indoor and outdoor scenes.
- previous work focus on images or 3D voxels, often obscuring natural 3D patterns and invariances of 3D voxels.

previous work:

- object detection and instance segmentation based on 2D image.
- most existing works convert 3D point clouds to images by projection, or to volumetric grids by quantization and then apply convolutional networks.
- 3D object detection from RGB-D data
  - front view image based methods:
    - take monocular RGB images and shape priors or occlusion patterns to infer 3D bounding boxes
    - represent depth data as 2D maps.
  - Bird's eye view based methods:
    - projects Li-DAR point cloud to bird's eye view and trains RPN
  - 3D based methods:
    - train 3D object classifiers by SVMs on hand-designed geometry features extracted from point cloud and localize objects using window search,
- Deep Learning on Point Clouds:
  - convert point clouds to images or volumetric forms before feature learning
  - <font color=red>require quantitization of point clouds with certain voxel resolution</font>

#### Methods

- **Problem Formulation**:

input RGB-D data, classify and localize objects in 3D space.

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200307115907354.png)

**【Model 1】Frustum Proposal**

- with known camera projection matrix, a 2D bounding box can be lifted to a frustum(with near and far planes specified by depth sensor range) that defines a 3D search space for the object. 	
- using FPN ,pre-train the model weights on ImageNet classification and COCO object detection datasets and further fine-tune it on KITTI 2D object detection to classify and predict amodal 2D boxes.

**【Model 2】3D Instance Segmentation**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200307120512402.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200307120608061.png)

**【Model 3】3D Instance Segmentation PointNet**

- Similar to the case in 2D instance segmentation, depending on the position of the frustum, object points in on frustum may become cluttered or occlude points in another.
- transform the point cloud into a local coordinate by subtracting XYZ values by its centroid. considering the bounding sphere size of a partial point cloud can be greatly affected by viewpoints and the real size of the point clouds helps the box size estimation.

**【Model 4】Amodal 3D Box Estimation**

- learning-based 3D alignment by T-Net
- amodal 3D box estimation pointnet

#### Evaluation

- **Environment**:   
  - Dataset: KITTI  , 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200307121349449.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200307121605990.png)

#### Conclusion

- propose a novel framework Frustum PointNets for RGB-D data based 3D object detection 
- provide extensive quantitative evaluations to validate the design.

#### Notes <font color=orange>去加强了解</font>

- KITTI 3D object detection:  http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d   contain datasets, many methods and compare.<font color=red>如果后面学习和使用到3D object detection 可以从这里学习</font>
- bird's eye view detection benchmarks

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200307122059591.png)

**level**: ICCV   CCF_B
**author**:         National Institute of Advanced Industrial Science and Technology
**date**: 2017
**keyword**:

- Spatio-Temporal ,action recognision

------

## Paper: 3D-Resnet

<div align=center>
<br/>
<b>Learning Spatio-Temporal Features with 3D Residual Networks
for Action Recognition</b>
</div>


#### Summary

1.  exploring the effectiveness of ResNets with 3D convolutional kernels 
2.  学会使用作为基本的视频特征提取方法

#### Research Objective

- **Application Area**: surveillance systems, video indexing, and human computer  interaction
- **Purpose**:  propose a 3D CNNs based on ResNets toward a better action representation

#### Proble Statement

- **Action Recognition Database**:
  - HMDB51 [13]
  - UCF101 [16]
  - Kinetics human action video dataset [12]

- Residual block

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223101841019.png)

#### Methods

- **network design**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223101025170.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223101347554.png)

#### Notes <font color=orange>去加强了解</font>

- [ ] code available: https://github.com/kenshohara/3D-ResNets.   需要做实验

**level**:  CVPR    CCF_A
**author**: DUSHYANT
**date**:  2017
**keyword**:

- 3D human pose estimation

------

## Paper: VNect

<div align=center>
<br/>
<b>Real-time 3D Human Pose Estimation with a Single RGB Camera</b>
</div>


#### Research Objective

- **Application Area**:realtime motion-driven 3D game character control,self-immersion in 3D virtual and augmented reality and human-computer interaction.

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118101846038.png)

- **Purpose**:  <font color=red>stable 3D skeletal motion capture from a single camera in real-time</font>

#### Proble Statement

previous work:

- Multi-view: using multi-view setups markerless motion-capture solutions attain high accuracy. <font color=red>we combine discriminative pose estimation with kinematic fitting to succeed in our underconstrained setting</font>
- Monocular Depth-based: RGB-D sensors overcomes forward backward ambiguities in monocular pose estimation.
- Monocular RGB:structure-from-motion techniques exploit motion cues in a batch of frames and also been applied to human motion estimation.
- Given 2D joint locations ,existing approaches  use bone length and depth ordering constraints ,sparsity asumptions,joint limits,inter-penetration constraints,temporal dependencies and regression to create 3D  pose.<font color=red>sparse set of 2D locations losesimage evidence-> discriminative methods</font>
- previous work only obtain temporally unstable coarse pose not directly usable in applications.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118103828661.png)

【Qustion 1】how to use CNN to regress Pose by single RGB image?

extending the 2D heatmap formulation to 3D using three additional location-maps(x,yz,) per joint j , captureing the root-relative locations (x,y,z) respectively.![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191118110049931.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191121100234719.png)

#### Contribution

- the first real-time method to capture the full global 3D skeletal pose of a human in a stable ,temporally consistent manner using a single RGB camera.
- novel fully convolutional pose formulation regresses 2D and 3D joint positions jointly in real time and doesn't require tightly cropped input frames, and forgoes the need to perform expensive bounding box computations.
- model-based kinematic skeleton fitting against the 2D/3D pose predictions to produce temporally stable joint angles of a metric global 3D skeleton in rea time.
- more applicable for outdoor scenes ,community video and low quality commodity RGB cameras.

#### Notes 

- [x] heatmap based bodyjoint detection formulation  ,heatmap is the distribution of confidence probability of body part. 

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191121094023973.png)

# 



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/video_understand/  

