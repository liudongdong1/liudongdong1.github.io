# IMU Trajectory


> Advantages of IMU : (1) energy-efficient, capable of running 24h a day without draining a battery; (2) works any where even inside a bag or a pocket(get device acc); Disadvantage: small sensor errors or biases explode quickly in the double integration process.
>
> In Augmented Reality applications(eg., apple ARKit, Google ARCore, Microsoft HoloLens), IMU augments Slam by resolving scale ambiguities and providing motion cues in the absence of visual features. UAVs, automous cars, humanoid robots, and smart vacuum cleaners are other emerging domains, utilizing IMUs for enhanced navigation, control, and beyond.

# 1. Pose Estimation Relative

**author**: Zhe Zhang( Southeast University), Chunyu Wang(Microsoft Research Asia)
**date**: 2020, 4.10
**keyword**:

- 3D pose estimation

------

## Paper: Fusing Wearable IMUs 

<div align=center>
<br/>
<b>Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: Geometric Approach</b>
</div>

#### Summary

1. present a geometric approach to reinforce the visual features of each pair of joints based on the IMU,  improving the 2D pose estimation accuracy especially when one joint is occluded.
2. fit the multi-view 2D poses to the 3D space by an orientation Regularized pictorial Structure Model which jointly minimizes the projection error between the 3D and 2D poses along with the discrepancy between the 3D pose and IMU orientations.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612163514924.png)

#### Research Objective

- **Purpose**:  solve the occlusion problem in estimating 3D poses from images.

#### Proble Statement

- instead of estimating 3D poses or pose embeddings from images and IMUs separately and then fusing them in a late stage, we propose to fuse IMUs and image features in a very early stage with the aid of 3D geometry.
- in 3D pose estimation step, we levarage IMUs in the pictorial structure model.

previous work:

- previous pose estimation use the limb length prior to prevent from generating abnormal 3D poses, and the prior is fixed for the same person and doesn't change over time.
- **Image-based:** 
  - model/optimization based: defines the 3D parametric human body model and optimizes its parameters to minimize the discrepancy between model projections and extracted image features.
  - Supervised learning: learning a mapping from images to 3D pose , lack of abundant ground truth 3D poses, not aware of their absolute locations in the world coordinate system.
  - two-step mothods: first estimate 2D poses in each camera view, and recovers the 3D pose in a world coordinate system with camera paremeters.
- **IMUs-based:**  suffer from drifting over time
  - Slyper et al. Tautges et al. propose to reconstruct human pose from 5 accelerometers by retrieving pre-recorded poses with similar accelerations from a databases.
  - Roetenberg et al. use 17 IMUs equipped with 3D accelerometers, gyroscopes and magnetometers and all the measurements are fused using Kalman Filter, to get the pose of the subject.
- **Images+IMUs-based**:
  - estimate 3D human pose by minimizing an energy function which is related to both IMUs and image feature.
  - estimate 3D poses separately from the images and IMUs and then combine them to get the final estimation.

#### Methods

- **Procedure**:

> introduce 2D Orientation Regularized Network to jointly estimate 2D poses for multi-view images. Using IMU orientations as a structural prior to mutually fuse the image features of each pair of joints linked by IMUs.
>
> estimate 3D pose from multi-view 2D poses(heatmaps) by a Pictorial Structure Model, it jointly minimizes the projection error between the 3D and 2D poses

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612171943527.png)

【Qustion 1】<font color=red> 关键部分不理解,具体细节后面遇到再说</font>

#### Evaluation

  - **Environment**:   
    - Dataset: 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612172211707.png)

#### Notes <font color=orange>去加强了解</font>

  - https://github.com/CHUNYUWANG/imu-human-pose-pytorch
  - Daniel Roetenberg, Henk Luinge, and Per Slycke. Xsens mvn: full 6dof human motion tracking using miniature inertial sensors. Xsens Motion Technologies BV, Tech. Rep, 1, 2009. 

**level**: on arXiv, don;t know the meeting.
**author**: Hyeokhyen Kwon(Georgia Tech), Catherine Tong(University of Oxford)
**date**: 2018,6
**keyword**:

- Ubiquitous and Mobile computing, Data Collection, Activity Recognition.

------

## Paper: IMUTube

<div align=center>
<br/>
<b>IMUTube: Automatic extraction of virtual on-body accelerometry from video for human activity recognition</b>
</div>



#### Summary

1. introduce IMUTube, an automated processing pipeline that integrated existing computer vision and signal processing techniques to convert videos of human activity into virtual streams of IMU data.
2. processing pipline:
   - applies standard pose tracking and 3D scene understanding techniques to estimate full 3D human motion from video segment that captures a target activity.
   - translates the visual tracking information into IMU that are placed on dedicated body position.
   - adapts the virtual IMU data towards the target domain through distribution matching.
   - derives activity recognizers from the generated virtual sensor data, potentially enriched with small amounts of real sensor data.

#### Research Objective

  - **Application Area**: HAR( behavioral analysis like user authentication, healthcare, and tracking everyday activities)
- **Purpose**:   aim at harvesting existing video data from large-scale repositories, such as YouTube and automatically generate data for virtual, IMUs that will then used for deriving sensor-based human activity recognition system.

#### Proble Statement

- the lack of large-scale, labeled data sets impedes progress in developing robust and generalized predictive models for  on-body sensor-based human activity recognition.
- sensor data collection is expensive and the annotation is time-consuming and error-prone.

#### Challenges:

1. the datasets needs to be curated and filtered towards the actual activities of interest;
2. even though video data capture the same information about activities in principle, sophisticated preprocessing is required to match the source and target sensing domains;
3. the opportunistic use of activity videos requires adaptations to account for contextual factors such as multiple scene changes, rapid camera orientation changes, scale of the performer in the far sight, or multiple background people not involved in the activity;<font color
4. new forms of features and activity recognition models will need to be designed to overcome the short-comings of learning from video-sourced motion information for eventual IMU-based inference.

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625110657529.png)

【Module 1】Motion Estimatioln for 3D joints

> 1. estimate 2D pose skeletons for potentially multiple people in a scene using OpenPose.
> 2. lift each 2D pose to 3D pose by estimating the depth information using <font color=red>VideoPose3D model [56].</font>
> 3. apply SORT tracking algorithm[7] to track each person across the vedio sequence.<font color=red> 去了解</font>
> 4. interpolate and smooth missing or noise keypoints using KF algorithm.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625134748743.png)

【Module 2】Global Body Tracking in 3D

> To extract global 3D scene information from the 2D video to track a person's movement in the whole scene.
>
> 1. 3D localization in each 2D frame.
> 2. the camera viewpoint changes(ego-motion) between subsequence 3D scenes

- 3D Pose Calibration:  
  - using Pnp algorthm to calculate perspective projection between corresponding 3D and 2D keypoints.[33]
  - estimate the camera intrinsic parameters from video using DeepCalib model[8].
- Estimate camera egomotion： potential viewpoint changes across frames.
  - Camera ego-motion estimation from one viewpoint to another requies 3D point clouds of both scenes. To create 3D point of a scene requires two information: 1. the depth map(using DepthWild model[22]); 2. camera intrinsic parameters

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625135916080.png)

【Module 3】Generating Virtual Sensor Data

- extract 3D motion information for each person in a video.
- To tack the orientation change of person joint from the perspective of the body coordinates.

# 2. Navigation Relative

**level**: Institufe of physic publishing
**author**: J Gao(University of Nottingham)
**date**: 2002,12,5

**keyword**:

- IMU, Navigation, Trajectory

------

## Paper: Error reduction for IMU

<div align=center>
<br/>
<b>Error reduction for an inertial-sensor based dynamic parallel kinematic machine positioning system</b>
</div>

#### Summary

**【Knowledge one】Category of the errors for IMU system**

- Inertial sensor errors, including sensor bias error, cross-axis coupling and scale factor error;
- misalignment error, due to the misalignment angle along the sensitive axis an gravity acceleration component will be sensed as a part of acceleration
- computational process errors, such as numerical integration error.

![image-20200620085340958](C:/Users/dell/AppData/Roaming/Typora/typora-user-images/image-20200620085340958.png)

> waveform numerical integration methods, such as cumulative rectangle, trapezoidal and Simpson integration methods.

**【Knowledge two】Velocity waveform reconstruction**

> a high sampling rate should be used when data acquisition is performed to reduce the sampling interval and thus reduce the integrationerror. However,this will result in more noise in the measured data and increase the calculation burden.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200620090253185.png)

**【Knowledge three】Waveform distortion**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200620090545825.png)

**【Knowledge four】 Velocity waveform correction**

> Fit the distortion of the velocity by a polynomial.     Trujillo and Carter
>
> the physical properties and boundary conditions of machine motion the velocity waveform can be corrected by second-order or first-order polynomials.

$$
v_c(t)=v(t)+b_1t^2+b_2t+b_3 \\
x_c(t)=x(t)+b_1/3t^3+b_2/2t^2+b_3t+b_4
$$

$$
v_c(t)=v(t)+b_1t+b_2 \\
x_c(t)=x(t)+b_1/2t^2+b_2t+b_3
$$

assuming $v(0)=0,v(T)=0,x(0)=0,x(T)!=0$
$$
v_c(t)=v(t)+1/T[v(0)-v(T)]-v(0)  \\
x_c(t)=x(t)+1/(2T)[v(0)-v(T)]t^2-v(0)t-x(0)
$$
![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200620092722253.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200620093339022.png)



**keyword**:

- IMU, Navigation, Trajectory

------

## Paper: RoNIN

<div align=center>
<br/>
<b>RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluation, and New Methods</b>
</div>



#### Summary

1. a new benchmark containing more than 40h of IMU sensor data from 100 human subjects with ground-truth 3D trajectory under natural human motions
2. novel neural inertial navigation architectures, making significant improvements for challenging motion cases
3. qualitative and quantitative evaluations of the competing methods over three inertial navigation benchmarks.

#### Proble Statement

- **Physics-based:** IMU double integration. (Sensor biases explode quickly in the double integration process)
  - foot mounted IMU with zero speed update, the sensor bias can be corrected subject to a constraint that the velocity must become zero whenever foot touches the ground
- **Heuristic:** human motion are highly repetitive
  - step counting: (1) the IMU is rigidly attached to body. (2) the motion direction is fixed with respect to IMU. (3) the distance of travel is proportional to the number of foot-steps.
  -  PCA[10] or frequency domain analysis[13] to infer motion direction.
- **Data-driven priors:** 
  - RIDI focuses on regressing velocity vectors in a device coordinate frame, while rely on traditional sensor fusion methods to estimate device orientation.

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612111925446.png)

【Qustion 1】how to get body heading information?

> the body orientation differs from the device orientation arbitrarily depending on how one carries a phone.   Assuming the headings of the tracking phone with an constant offset introduced by the misalignment of the harness, and ask the subject to walk straight for the first five seconds, then estimate this offset angle as the difference between the average motion heading and the tracking phone's heading.

【Qustion 2】RoNIN network to regress the heading direction? <font color=red> don't understand</font>

> RoNIN seeks to regress a velocity vector given an IMU sensor histroy with two key design principles,(1) Coordinate frame normalization defining the input and output feature space;(2) robust velocity losses improving the signal-to-noise-ratio even with noisy regression targets.

- **Coordinate frame normalization:** 

  > uses a heading-agnostic coordinate frame, any coordinate frame whose Z axis is aligned with gravity.
  >
  > IMU data is transformed into the same HACF by the device orientation and the same horizontal rotation.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612160431127.png)

#### Evaluation

  - **Environment**:   

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612155100851.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612162530065.png)
**date**: 2017
**keyword**:

- Smartphones, Trajectory, IMU

------

## Paper: RIDI

<div align=center>
<br/>
<b>RIDI: Robust IMU Double Integration</b>
</div>



#### Summary

1. proposes a novel data-driven approach for inertial navigation, which learns to estimate trajectories of natural human motions.
2. based obervation: human motions are repetitive and consist of a few major modes(standing, walking, turning).
3. regresses a velocity vector from the history of linear accelerations and angular velocities, then corrects low-frequency bias in the linear accelerations, which are integrated twice to estimate position.

previous work:

- **V-slame**: (1) a camera must have a clear light-of-sight under well-lit environments all the time.(2) the recording and processing of the video data quickly drain a battery.

#### Methods

- **system overview**: use the IMU to estimate trajectories of natural human motions

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612101206414.png)

【Algorithm】

> First, regresses a velocity vector from angular velocities and linear accelerations( accelerometer readings minus gravity).
>
> Second, RIDI estimates low-frequency corrections in the linear accelerations so that their integrated velocities math the regressed values.

【Question 1】how to learning to regress volocities?

> transform the device poses( world Coor) and angular velocities and linear acc( device Coor) into s, and apply Gaussian smoothing to supress high-frequency.
>
> concatenate smoothed angular velocities and linear acc from the past 200 frames to construct a 1200 dimensional feature vector

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612103657344.png)

【Question 2】how to correcting acceleration errors?

> Error Sources: IMU readings, system gravities, system rotations, intract in a complex way.
>
> Predicted velocities to provide effective cues in removing sensor noises and biases.

#### Evaluation

  - **Environment**:   
    - Dataset:  six human subjects with four popular smartphone placements, and relative ground-truth.

#### Conclusion

- the first to integrate sophisticated machine learning techniques with inertial navigation.
- database of IMU sensor measurements and 3D motion trajectories across multiple human subjects and multiple device placements.( six human, four kinds placements, various type motions including walking forward/backward, side motion, acceleration/deceleration)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200612105303609.png)
**author**:Martin Brossard（MINES Paris Tech)
**date**: 2019
**keyword**:

- AI, Trajectory, IMU

------

## Paper: AI-IMU

<div align=center>
<br/>
<b>AI-IMU Dead-Reckoning</b>
</div>

#### Summary

- propose a novel accurate method for dead-reckoning of wheeled vehicles based only on an Inertial Measurement Unit.
- for intelligent vehicles, robust and accurate dead-reckoning based on IMU may prove useful to correlate feeds from imaging sensors, to safely navigate through obstructions, or for safe emergency stops in the extreme case of exteroceptive sensors failure.
- <font color=red> using Kalman filter and use of deep neural networks to dynamically adapt the noise parameters of the filter.</font>
- tested on KITTI odometry dataset, this article estimates 3D position, velocity, orientation of the vehicle and self-calibrates the IMU biases, and achieve on average a 1.10% translational error and the algorithm competes with top-ranked methods.
- the code:https://github.com/mbrossar/ai-imu-dr

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624220627710.png)

**level**: UbiComp/ISWC'19
**author**:Yasuo Katsuhara(Frontier Research Center,Toyata Motor Corp)
**date**: 2019
**keyword**:

- AI, Trajectory, IMU

------

## Paper: Multi-Person Motion Forecasting

<div align=center>
<br/>
<b>Poster: Towards Multi-Person Motion Forecasting: IMU based Motion Capture Approach</b>
</div>



#### Summary

1. propose a multi-person motion forecasting system by using IMU motion captures to overcome these difficulties simultaneously.

#### Research Objective

  - **Application Area**: Human-centered computing, body motion forecasting; IMU based motion capture; sports and entertainment; anomaly motion prediction for factory workers and drivers;
- **Purpose**:  

#### Proble Statement

- camera and optical based methods have to take into account the environmental settings and occlusion problems
- previous studies don't consider plural persons.

previous work:

- employed cameras and optical motion captures to measure the joint positions of person, and predicted them about 0.5s before by using DNN.

#### Methods

- **Problem Formulation**:

- **system overview**:

【Module 1】Data Collection

- using Perception Neuron2.0 to collect three dimensianal position(x,y,z) of 21 body joints. sampleing interval was 50 ms

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624223907049.png)

【Module 2】Data Processing

- employed Horiuchi's manner[2], in each frame capture 21 3d joints position and center of gravity.
- to forecast the positions after 0.5s, the predicted position vector $z_t$ is define by the 3D position of t+10 frames.
- using 3-layer neural networks to forcaset the position.
- <font color=red> to predict standing up and walking acitvities</font>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624225012923.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624225645127.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624225813310.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624225901410.png)

#### Notes <font color=orange>去加强了解</font>

  - paper 7: a mixed reality martial arts training system with RGB camera.

**level**: Proceedings of the 2020 IEEE/SICE International Symposium on System Integration, Hawaii,USA
**author**: Wataru Takano
**date**: 2020, 1-12

------

Paper: Sentence Generation 

<div align=center>
<br/>
<b>Sentence Generation from IMU-based Human Whole Body MOtions in Daily Life Behaviors</b>
</div>



#### Summary

1. presents a probabilistic approach toward integrating human whole-body motions with natural language.
2. <font color=red> human whole-body motions in daily life are recorded by IMU and subsequently encoded into motion symbols. Sentences are manually attached to the human motion primitives for their annotation by probabilistic graphical models.</font>
3. <font color=red> One probabilistic model trains the lining of motion symbols to words, and the other represents sentence structure as word sequences</font>
4. translating human whole-body motions into descriptions, where multiple words are associated from the human motions by the first model and the second model searches for syntactically consistent sentences consisting of associated words.
5. seventeen IMU sensors are attached to a performer, the sesulting sensor data are transformed to positions of 34 virtual markers attached to the performer, and human whole body posture is expressed by a feature vector whose elements are positions of the virtual makers in a trunk coordinate system.

#### ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624232127699.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624232815685.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624232753572.png)

#### Contribution

- recorded many human whole-body daily life motions by using IMU sensors, by annotated these human motions, we create large datesets.
- present a framework for combining human whole body motions with sentences, Mappings between human motions and relevant words, and transitioning among words in sentences are represented by probabilistic graphical models.
- the methods computes three probabilities: the probability of observed human motion being generated by motion symbols, probability of words being associated with those motion symbols, and probability of word transitions.

**level**: IEEE Transactions on Emerging Topic in Computational Intellegence.
**author**: Tse-Uu Pan( Student Member,IEEE), National Cheng Kung University ,Tainan
**date**: 2019.6
**keyword**:

- IMU, Trajectory

------

## Paper: Handwriting Trajectory

<div align=center>
<br/>
<b>Handwriting Trajectory Reconstruction Using Low-Cost IMU</b>
</div>

#### Summary

1. propose a trajectory reconstruction method based on low-cost IMU in smartphones.
2. intrinsic bias and random noise usually cause unreliable IMU signals, filtering methods are utilized to reduce high- or low-frequency noises of signal.
3. extract multiple features from IMU signals and train a movement detection model based on LDA.
4. recognize the handwritten letter by constructing the trajectory.

#### Research Objective

  - **Application Area**: positioning, various HCI applications(human activity recognition, gait detection, patient resuscitation, sports sciences)
- **Purpose**:  

#### Proble Statement

- the writing space is often too small for many handwriting applications.
- previous IMU based focus on large motion gesture/activity, can't deal with subtle interactions in VR.

previous work:

- infrared sensors, ultrasonic sensors, tiny cameras to record the trajectory. <font color=red> high power-consumed</font>
- Noitom Ltd. develop a product, Perception Neuron, a set of IMU sensors worn on a human body to record human movement and then control animation of virtual human in virtual space based on the recorded data.
- Thalmic Labs introduced a wearable device named Myo, equipped with eight sEMG sensors and one IMU sensor to measure the trajectory of hand movements and hand gestures.
- Wacon Inkling used an external sensor to sense ultrasound and infrared to reconstruct entire trajectory.
- Livescrible Echo introduced a digital pen and a notebook to reconstruct the trajectory of handwritting by using invisible points on the notebook with which the pen could locate itself by sensing the invisible points.
- Sperbel et al.[11] proposed a vision-based digital pen to reconstruct trajectory of handwriting.
- Xie et al.[20] presented an accelerometer-based smart ring and used a similarity matching-based smart ring and used a similarity matching-based extensible algorithm to recognize basic and complex gestures.
- Agrawal et al. [25] proposed a method to reconstruct the trajectory of the handwriting by using the build-in acc in a phone.
- Yang et al.[26] proposed a mechanism ZVC to reduce the accumulative error of IMUs when reconstructing the trajectory.
- Wang et al.[27] proposed an attitude error compensation method and a mechanism Multi-Axis Dynamic switch to discard some signals of noise, drift, and the users' trembles by setting an appropriate threshold.

#### Methods

- **system overview**:

【Module 1】Signal Preprocessing

- Sensor  Calibration and Digital-Analog Conversion: ZGO: the offset between zero and the output of a stationary IMU. The sensitivity is related to the full-scare range and the resolution of the IMUs.

$$
a=(Raw_{acc}-ZGO)/sensitivity_acc
$$

$$
w=(Raw_{gyr}-ZRO)/sensitivity_{gyr}
$$

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625091630619.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625091647837.png)

- Smoothing Filter: using exponential weighted moving average(EWMA) filter.

$$
EWMA(t)=\frac{Data_i+(1-\epsilon)Data_{t-1}+(1-\epsilon)^2Data_{t-2}+...}{1+(1-\epsilon)+(1-\epsilon)^2}+...
$$

【Module 2】Trajectory Reconstruction

- Attitude Estimation&Coordinate Transformation:![Coordinate Systems](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625092320299.png)

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625092412445.png)

- Gravity Erasion: to get the linear-acc
- Movement Detection:  

> segment the signals using sliding window and extract features for each segment, and using LDA to classify a segment into moving or stationary signals, and morphological operations are then applied to smooth the classification results obtained by LDA.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625092926820.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625092950136.png)

- Displacement Calculation: $x(t)=x(t-1)+v(t)*dt$
  $$
  offset=\frac{v(t_{end}-v{t_{end}-1})}{t_{end}-t_{begin}}\\
  v_c(t)=v_m(t)+(t-t_{begin}+1)*offset
  $$

【**Module 3】Trajectory Recognize**

- **Sequence-Based Recognition**: using orientation of short-term trajectory displacement, for a trajectory composed of k sampled points {(x0,y0),(x1,y1),...,(xk−1,yk−1)}, segment the trajectory into 20 short-term trajectories with equal sampled points,the nth segment can be presented by:
  $$
  C(n)head=(x_{\frac{k}{20}*n},y_{\frac{k}{20}*n})\\
  C(n)tail=(x_{\frac{k}{20}*(n+1)-1},y_{\frac{k}{20}*(n+1)-1})
  $$

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625094307301.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625093913283.png)

> using above orientation to denote each segment to get the sequence and using DTW or HMM methods to classify.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200625093123952.png)

- **Image-Based Reconstruction** 
- **LSTM-Based Time Sequence Methods**

#### Conclusion

- design a framework which can not only reconstruct handwritting trajectory but also recognize the constructed letter based on low-cost IMU.
- to overcome the intrinsic drift and the high-degree noise, a reset switch method is designed to reduce the accumulated error caused by the IMU sensor.



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/imu-trajectory/  

