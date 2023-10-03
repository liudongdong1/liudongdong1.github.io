# MediaPipe


> [MediaPipe](https://google.github.io/mediapipe/) is the simplest way for researchers and developers to build world-class ML solutions and applications for mobile, desktop/cloud, web and IoT devices.

# 1. Introduce

1. **End-to-End acceleration**: *built-in fast ML inference and processing accelerated even on common hardware*
2. **Build one, deploy anywhere**: *Unified solution works across Android, iOS, desktop/cloud, web and IoT*
3. **Ready-to-use solutions:** cutting-edge ML solutions demonstrating full power of the framework
4. **Free and Open Source**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200824115410.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200824115501.png)

# 2.PaperReading

**level**: CCF_A CVPR
**author**:  Mediapipe@google.com
**date**: 2019.6.14
**keyword**:

- Perception, Framework

------

## Paper: MediaPipe

<div align=center>
<br/>
<b>MediaPipe: A Framework for Building Perception Pipelines
</b>
</div>



#### Summary

1. propose a framework consists of three main parts:
   - inference from sensory data;
   - a set of tools for performance evaluation;
   - a collection of re-usable inference and processing components called calculators.
2. MediaPipe is targeted towards applications in the audio/video processing domain and not limited to the scope of modeling the performance of concurrent systems.

#### Research Objective

#### Proble Statement

- select and develop corresponding machine learning algorithms and models;
- build a series of prototypes and demos;
- balance resource consumption against the quality of the solutions;
- identify and mitigate problematic cases;

> Modifying a perception application to incorporate additional processing steps or inference models can be difficult due to excessive coupling between steps;
>
> different platforms consuming time and involves optimizing inference and processing steps to run correctly and efficiently on a target device.

#### System overview

> MediaPipe allows to prototype a pipeline incrementally as a directed graph of components where each component is a calculator; The graph is specified using **GraphConfig** protocol buffer and then run using a Graph object; the calculators are connected by data Stream, each Stream represents a time-series of data Packets;

【**Module one**】 **Component**

- **Packets：** consists of a numeric timestamp and a shared pointer to an immutable payload.
- **Streams**: carries a sequence of packets whose timestamps must be monotonically increasing. Each input stream receives a separate copy of the packets from  an output stream, and maintains its own queue to allow the receiving node to consume the packets.
- **Side packets:** a side-packets connection between nodes carries a single packet with an unspecified timestamp.
- **Calculators:** a calculator may receive zero or more output streams or its packets, comprise of four essential methods: **GetContract(),Open(),Process(),Close()**;
- **Graph:** the context of all processing, contains a collection of nodes joined by directed connections along which packets can flow, some constraints are as follows:
  - each stream and side packet must be produced by one source;
  - the type of an input stream/side packet must be compatible with the type of the output stream/side packet to which it is connected;
  - each node's connections are compatible with its contract.
- **GraphConfig:** a specification that describes the topology and functionality of the graph

#### Implementation

> scheduling logic and powerful synchronization primitives to process time-series in a customizable fashion.

【Scheduling】

- each graph has at least one scheduler queue, each scheduler has exactly one executor, nodes are statically assigned to a queue.
- each node has a scheduling state, **not ready, ready, running**; 
- when a node becomes ready for execution, a task is added to the corresponding scheduler queue, the nodes are topologically sorted and assigned a priority based on the graph's layout;

【**Synchronization**】

> mediapipe graph execution is decentralized: there is no global clock, and different nodes can process data from different timestamps at the same time;

- the packets pushed into a given stream must have monotonically increasing timestamps;
- each stream has a timestamp bound, which is the lowest possible timestamp allowed for a new packet on the stream.

【**Input policies**】

> Synchronization is handled locally on each node, using input policy specified by the node.

- if packets with the same timestamp are provided on multiple input streams, they will always be processed together regardless of their arrival order in real time;
- input set are processed in strictly ascending timestamp order;
- no packets are dropped, and the processing is fully deterministic;
- the node becomes ready to process data as soon as possible given the guarantees above.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831085500.png)

【**Flow Control**】

> packets may be generated faster than they can be process, flow control is necessary to keep resource usage under control;

- a simple back-pressure system: throttles the execution of upstream nodes when the packets buffered on a stream reach limit; by maintaining deterministic behavior and includes a deadlock avoidance system that relaxes configured limits;
- a richer node-based system: consists of inserting special nodes which can drop packets according to real-time constraints;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831090000.png)

【GPU support】

【Opaque buffer type】

【OpenGL support】

#### Tools

- **Tracker**：follow individual packets across a graph and records timing events along the way, recording a **TraceEvent** structure with several data fields event_time, packet_timestamp, packet_data_id, node_id, and stream_id;
- **Visualizer**: help to understand the topology and overall behavior of their pipelines:
  - Timeline View
  - Graph view

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831090454.png)



#### Experiment

- Object Detection

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831090533.png)

> -  In the detection branch, a frame-selection node ﬁrst selects frames to go through detection based on limiting frequency or scene-change analysis, and passes them to the detector while dropping the irrelevant frames. 
> - The objectdetection node consumes an ML model and the associated label map as input side packets, performs ML inference on the incoming selected frames using an inference engine (e.g., [12] or [2]) and outputs detection results.
> - the tracking branch updates earlier detections and advances their locations to the current camera frame.  
> - the detection-merging node compares results and merges them with detections from earlier frames removing duplicate results based on their location in the frame and/or class proximity. 

- FaceLandmark
  - demultiplexing node splits the packets in the input stream into interleaving subsets of packets, with subset going into a separate output stream;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831091200.png)

#### Notes <font color=orange>去加强了解</font>

  - Beam[1]; Apache beam: An advanced uniﬁed programming model. 
  -  Dataflow[5];  Thedataﬂowmodel: Apracticalapproach to balancing correctness, latency, and cost in massive-scale, unbounded,out-of-orderdataprocessing
  - Gstream[8];  https: //gstreamer.freedesktop.org/, 
  - CV4.0(graph api[9])；OpenCV Graph API. Intel Corporation, 2018. 

**level**: 
**author**: Valentin Bazarevsky (google research)
**date**: 2020
**keyword**:

- Pose estimation

> Bazarevsky, Valentin, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. "BlazePose: On-device Real-time Body Pose tracking." *arXiv preprint arXiv:2006.10204* (2020).

------

## Paper: BlazePose

<div align=center>
<br/>
<b>BlazePose: On-device Real-time Body Pose tracking
</b>
</div>
#### Summary

1. blazepose, a lightweight convolutional neural network architecture for human pose estimation that is tailored for real-time inference on mobile devices.
2. produces 33 body keypoints for a single person and runs at over 30    frames per second on a Pixel 2 phone.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821200224.png)

#### Research Objective

  - **Application Area**: fitness tracking ; sign language recognition; Yoga;
- **Purpose**:  estimate human pose from images or video with edge devices.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821195628.png)

> a lightweight body pose detector followed by a pose tracker network, the tracker predicts keypoint coordinates, the presence of the person on the current frame, and the refined region of interest for the current frame, when the tracker indicates that there is no human present, we re-run the detector network on the next frame.

【Person Detector】use a fast on-device face-detector as a proxy for a person detector. the middle point between the person's hips, the size of the circle circumscribing the whole person, and incline(the angle between the lines connecting the two mid-shoulder and mid-hip points).

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821212805.png)

> - use the heatmap and offset loss only in the training stage and remove the corresponding output layers from the model before running the inference. use the heatmap to supervise the lightweight embedding, 
>
> - stack a tiny encoder-decoder heatmap-based network and subsequent regression encoder network.
>
> - utilize skip-connections between all the stages of the network to achieve a balance between high and low-level features.
> - for invisible points, simulate occlusions during training and introduce a per-point visibility classifier that indicates whether a particular point is occluded and if the position prediction is deemed inaccurate.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821213938.png)

#### Notes <font color=orange>去加强了解</font>

  - [ ] 运行代码，学习代码中模型结构，以及数据格式

**level**: 
**author**: Fan Zhang(google research)
**date**: 2020
**keyword**:

- hand pose estimation;

> Zhang, Fan, et al. "MediaPipe Hands: On-device Real-time Hand Tracking." *arXiv preprint arXiv:2006.10214* (2020).

------

## Paper: MediaPipe Hands

<div align=center>
<br/>
<b>MediaPipe Hands: On-device Real-time Hand Tracking
</b>
</div>
#### Summary

1. the pipeline consists of two models:
   -  a palm detector, that is providing a bounding box of a hand to
   -  a hand landmark model, that is predicting the hand skeleton.
2. an efficient two-stage hand tracking pipeline that can track multiple hands in real-time on mobile devices.
3. a hand pose estimation model that is capable of predicting 2.5D hand pose with only RGB input.
4. an open source hand tracking pipelines as a ready-to-go solution on variety of platforms, including android, ios, web, and desktop PCs.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821214746.png)

**【BlazePalm Detector】**

> - work across a variety of hand sizes with a large scale span 
> - be able to detect occluded and self-occluded hands
> - the hands is dynamic, lack of contrast patterns.

1. train a palm detector instead of a hand detector;
2. use an encoder-decoder feature extractro similar to FPN for larger scene-context awareness even for small objects.
3. minimize the focal loss during training to support a large amount of anchors resulting from the high scale variance.

**【Hand Landmark Model】**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821220035.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821220256.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200821220320.png)

#### Notes <font color=orange>去加强了解</font>

> 2.5D游戏仅仅是在2D游戏基础上把视角横向旋转了45度。2.5D视角带来的最核心的问题是每个图片和其他图片之间的遮挡关系如何处理，才能更符合人类对3D世界的常识性认知呢，也就是用2D的方式来模拟3D。2D游戏的做法很简单粗暴。2D游戏世界中每一个物件都会用一个2维坐标来表示其位置，x表示其横向位置，y表示其纵向位置。**当一个物件的y值越小，也就是其越靠近画面底部，则渲染顺序越靠后**。就像一个画家在Photoshop上作画一样，离相机越近的图层要越后面画，才能盖住离相机远的图层，所以画家要从远到近地画。3D游戏的渲染，简单来说可以理解为将三维数据在二维平面上做投影的过程。所以所谓的3D游戏，呈现在玩家面前依然是一个二维的画面，三维空间中的物件移动表现在二维画面上，也就是二维坐标位置的移动而已。θ就是相机的俯仰角,投影线段 = 3D线段 * sin(俯仰角)。CD长度=3D世界中正方形CD长度 * sin(俯仰角)。遍历三角函数查找表，只有sin(30°)的分子分母都为整数，也就是说只有30°这个角度有可能让长宽都为整数，具体可参看尼文定理：。因此人们通常说的斜45度视角游戏只是人们通过臆测而给2.5D游戏取得俗名，准确来说我们应该称这类游戏叫做斜30度视角游戏。或者可以采用另一种对斜45度视角游戏的解释，斜45度指的是相机水平方向上（围绕世界空间Y轴）的旋转角度。
>
> http://matov.me/isometric-toolset/  能不能将2.5D坐标转化为3D坐标。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201101104102167.png)

  - [x] multi-handDetection pipeline
- **multi_hand_detection_gpu**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200916212815466.png)

- **multi_hand_landmark_gpu**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200916214124663.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200916214222259.png)



- **multi_hand_renderer_gpu**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200916221843111.png)



**level**:  CVPR
**author**: Artsiom Ablavatski
**date**: 2020
**keyword**:

- iris tracking

------

## Paper: PupilTracking

<div align=center>
<br/>
<b>Real-time Pupil Tracking from Monocular Video for Digital Puppetry</b>
</div>

#### Summary

1. present a simple, real-time approach for pupil tracking from live video on mobile devices.
2. consists of two new component: a tiny neural network that predicts positions of the pupils in 2D, and a displacement-based estimation of pupil blend shape coefficients.
3. this methods can be used to accurately control the pupil movements of a virtual puppet, and lends liveliness and energy to it, run 50FPs on model phones.
4. detects 5 points of the pupil, outer iris circle and eye contour for each eye;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831094224.png)

#### System Overview

【**Neural network based eye landmarks**】

> using a tiny neural network combined face mesh model to produces additional higher quality landmarks.

- combine the corresponding landmarks (16 points of eye contour)from the face estimation pipeline with those from the eye reﬁnement network by <font color=red>replacing the x,y coordinates of the former while leaving z untouched</font>.
- extend the face mesh with 5 pupil landmarks(pupil center and  4 points of outer iris circle ), with z coordinate set to the average of the z coordinate of the eye corners.

【**Displacement-based pupil blend shape estimation**】基于位移的瞳孔混合变形估计

- refine the mesh to predict 4 blend shapes for the pupils:<font color=red> pupils pointing outwards, inwards, upwards and downwards respectively.</font>
- by combine 3 displacement to obtain scalar value in the range of [0,1] for each pupil blend shape.  <font color=red> 不明白这一步达到的效果是什么？</font>

> for the pupil pointing inwards, using the vertex of the pupil and vertex of eye corner, and measure the displacement $D_{current}$ between these two vertices and compare it to two empirically derived displacements $D_{neutral}$ , the displacement with the minimum activation of the blend shape and $D_{activated}$ the displacement measured using maximum activation of the blend shape. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831100400.png)

【**Real-time heuristics calibratin**】

> the initial displacements are empirically estimated based on the representative face mesh dataset, but unable to model all person-specific variations.
>
> employ the standard score calculation algorithm with a few modificaitons, the main idea of the filter is to check the displacement on every iteration and add it to a circular buffer of the trusted displacement if it falls within the specified confidence interval, and the calibrated displacement is calculated as an average of the trusted displacement, the standard deviation of these trusted displacements is used as the confidence interval in the next iteration.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831102522.png)

#### Evaluation

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831102637.png)

#### Notes <font color=orange>去加强了解</font>

  - human face model predicts a 468 vertex mesh [4]  Real-time facial surface geometry from monocular video on mobile gpus. 
  - [1] predicts 5 locations in 2D(pupil center , 4 points of outer iris circle, and 16 points of eye contour)  Blazeface: Sub-milli second neural face detection on mobile gpus. arXiv preprint arXiv:1907.05047, 2019. 2 
  - [8]Mnasnet: Platform-aware neural architecture search for mobile

## 3. 案例

### 3.1. Video Reframing

[AutoFlip: An Open Source Framework for Intelligent Video Reframing](http://ai.googleblog.com/2020/02/autoflip-open-source-framework-for.html)

> AutoFlip provides a fully automatic solution to smart video reframing, making use of state-of-the-art ML-enabled object detection and tracking technologies to intelligently understand video content. AutoFlip detects changes in the composition that signify scene changes in order to isolate scenes for processing. Within each shot, video analysis is used to identify salient content before the scene is reframed by selecting a camera mode and path optimized for the contents.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825153825.png)

- **Shot (Scene) Detection**: 

> A scene or shot is a continuous sequence of video without cuts (or jumps). To detect the occurrence of a shot change, AutoFlip computes the color histogram of each frame and compares this with prior frames. If the distribution of frame colors changes at a different rate than a sliding historical window, a shot change is signaled. AutoFlip buffers the video until the scene is complete before making reframing decisions, in order to optimize the reframing for  entire scene.

- **Video Content Analysis:** utilize deep  learning-based object detection models to find interesting, salient content in the frame.
- **Reframing:** 

> AutoFlip automatically chooses an optimal refremingn strategy, stationary, paining or tracking, depending on the way obects behave during the scene.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/jitter_combined.gif)

> **Top:** Camera paths resulting from following the bounding boxes from frame-to-frame. **Bottom:** Final smoothed camera paths generated using [Euclidean-norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) path formation. **Left:** Scene in which objects are moving around, requiring a tracking camera path. **Right:** Scene where objects stay close to the same position; a stationary camera covers the content for the full duration of the scene.

### **3.2.** Real-Time 3D Object Detection

> robotics, self-driving vehicles, image retrieval, and augmented reality. [built a single-stage model](https://arxiv.org/abs/2003.03522) to predict the pose and physical size of an object from a single RGB image

- **Real-World 3D Training Data**: With the arrival of [ARCore](https://developers.google.com/ar) and [ARKit](https://developer.apple.com/augmented-reality/), [hundreds of millions](https://arinsider.co/2019/05/13/arcore-reaches-400-million-devices/) of smartphones now have AR capabilities and the ability to capture additional information during an AR session, including the camera pose, sparse [3D point clouds](https://en.wikipedia.org/wiki/Point_cloud), estimated lighting, and planar surfaces.
- **AR Synthetic Data Generation:** AR Synthetic Data Generation, places virtual objects into scenes that have AR session data, which allows us to leverage camera poses, detected planar surfaces, and estimated lighting to generate placements that are physically probable and with lighting that matches the scene.

![Network architecture and post-processing for 3D object detection.](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825170636.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825170655.png)

> Sample results of our network — [**left**] original 2D image with estimated bounding boxes, [**middle**] object detection by Gaussian distribution, [**right**] predicted segmentation mask.

### 3.3. Afred Camera

>  users are able to turn their spare phones into security cameras and monitors directly, which allows them to watch their homes, shops, pets anytime. 

- **Moving Object Detection:**  continuously uses the device's camera to monitor a target scene, once detected recording the video and send notifications to the device owner.
- **Low-light Detection and Low-light Filter:** calculate the average luminance of the scene, and conditionally process the incoming frames to intensify the brightness of the pixel.
- **Motion Detection: ** by calculating the difference between two frames with some additional tricks that take the movements detected in a few frames
- **Area of Interest:** manually mask out the area where they don't want the camera to see.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825173215.png)

### 3.4. Iris Tracking and Depth Estimation

> A wide range of real-world applications, including computational photography (e.g., [portrait mode](https://ai.googleblog.com/2019/12/improvements-to-portrait-mode-on-google.html) and glint reflections) and [augmented reality effects](https://ai.googleblog.com/2019/03/real-time-ar-self-expression-with.html) (e.g., virtual avatars) rely on estimating eye position by tracking the iris

> Iris tracking is a challenging task to solve on mobile devices, due to limited computing resources, variable light conditions and the presence of occlusions, such as hair or people squinting. Often, sophisticated specialized hardware is employed, limiting the range of devices on which the solution could be applied.

- Depth-from-Iris from a single Image: <font color=red>by relying on the fact that the horizontal iris diameter of the human eye remains roughly constant at 11.7+-0.5 mm across a wide population, along with some simple geometric arguments.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825174114.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mediapipe/  

