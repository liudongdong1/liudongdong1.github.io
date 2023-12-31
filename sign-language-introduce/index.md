# Sign_Language_Introduce


> 手语是聋哑人士的主要沟通工具，它是利用手部和身体的动作来传达意义。虽然手语帮助它的使用者之间互相沟通，但聋哑人士与一般人的沟通却十分困难，这个沟通障碍是源于大部分人不懂得手语。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210222232624578.png)

## 1. 手势&&手语

- **手势**：手的姿势 ，通常称作手势。它指的是人在运用手臂时，所出现的具体动作与体位。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200713095739826.png)

- **手语：**手语是用手势比量动作，根据手势的变化模拟形象或者音节以构成的一定意思或词语。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200713095832667.png)

## 2. 手语分类

- 单手手语； 双手手语
- 静态手语；动态手语
- 动态手语：
  - 孤立词手语，包括 **准备动作（Prestoke）、有效动作（Stocke）和结束动作（Poststoke）** 三个部分。
  - 连续手语识别。

## 3. 数据集

- 著名的连续手语数据集是RWTH- PHOENIX-Weather
  包含由 9 个人提供的45 760 个视频样本，其中包含 5 356 个与天气预报相关的句子、1200个德国手语词汇，大约占 52 GB的存储空间。

- SIGNUM数据集：它包含由 25 个人提供的 33 210 个视频样本，其中包含 780 个句子、450 个德国手语词汇，每个句子包含 2 个至 11 个手语词汇不等。数据集大约占920 GB的存储空间。


- 波士顿大学的 ASLLVD：它包含由6 名手语者根据超过 3 300 个美国手语词汇提供的 9 800 个视频样本。


- **[RWTH-BOSTON-104](http://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-104.php)**:http://ww1.chalearn.org/resou/databases
- [中国手语 **DEVISIGN** 数据集](http://vipl.ict.ac.cn/homepage/ksl/data_ch.html)： 在微软亚洲研究院的赞助下建立的，旨在为世界范围内的研究者提供一个大型的词汇级的中国手语数据集，用于训练和评估他们的识别算法。目前，该数据集包含 4 414 个中国手语词汇，共包含 331 050 个 RGB-D 视频及对应的骨骼信息，由13名男性和17名女性提供
- 自己采集数据要求：
  - 图像采集设备要求：单目摄像机、双目摄像机、红外摄像机、深度摄像机（kinetic等）
  - 图像的丰富性：同一种手势，不同背景、不同光照、不同角度、不同人多方面全面考虑；不同手势尽可能全的手语特征手型类种。
- ChaLearn LAP IsoGD Dataset：
- RWTH-PHOENIXWeather 2014T dataset ：both gloss level annotations and spoken language translations for sign language videos that is of sufficient size and challenge for deep learning.f a parallel corpus of German sign language videos from 9 different signers, gloss-level annotations with a vocabulary of 1,066 different signs and translations into German spoken language with a vocabulary of 2,887 different words.   PHOENIX14T

  - https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/
- **word-level ASL datasets** ：2019年开源 , i.e. Purdue RVL-SLLL ASL Database [69], Boston ASLLVD [6] and RWTH-BOSTON-50
- https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/
- https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

------

# Paper: SignCol

<div align=center>
<br/>
<b>SignCol: Open-Source Software for Collecting Sign Language Gestures</b>
</div>



#### Summary

1. Signs recognition software should be capable of handling eight different types of sign combinations,e.g. numbers, letters, words, and sentences, including hand gestures, head actions, facial expressions,etc.
2. SignCol can capture and store colored frames, depth frames, infrared frames, body index frames, coordinate mapped color-body frames, skeleton information of each frame and camera parameters simultaneously.
3. **Visual data capturing**, connect to kinect2, capture and simultaneously save different information and data modalities of the gestures.
4. **Database managing& statistics reporting**;
5. SignLanguage parameters:
   - **Palm orientation** Palm could face up, down, left, right, out and in. For example, consider baby vs. table. 
   -  **Hand Shape** Shape of hands and ﬁngers are so useful for alphabetical letters and numbers. For example, consider I am Rita vs. My Rita. •
   - **Facial Expressions** Head nodes, head shakes, eyebrows, nose, eyes, lips, and emotions can be attached to the sign and bring a new meaning. For example, consider you vs. is it you? whose difference is in the surprising form of the eyes and eyebrows. Similarly, the hand shape and movement for I’m late and I haven’t are same and just the face shape makes them different. 
   -  **Location** Begin and end the sign at the correct position. Usually, signs originate from the body and terminate away or originate away from the body and terminate close to the body. For example, consider I’ll see you tomorrow. •
   - **Movement** Different kinds of movements are usually arc, straight line, circle, alternating in and out, the twist of the wrist and ﬁnger ﬂick. In addition, in movement duration, the location, direction and also shape of the hands could change. For example, consider happy or enjoy. 
6. Language Category Divided:
   -  cat1 – Number < 10 – such as ’4’, ’8’ 
   -  cat2 – Number > 10 – such as ’16’, ’222’
   - cat3 – Alphabet Letter – such as ’A’, ’F’ •
   - cat4 – Word by a Sign – such as ’I’, ’My’, ’Mom’ •
   - cat5 – Word by Letters – such as ’Entropy’, ’Fourier’ •
   - cat6 –SentencebyWords(by concatenatedletters) –such as ’Entropy learning’, ’Fourier Transform’ 
   - cat7 – Sentence by Signs – such as ’I love you’ 
   -  cat8 – Arbitrary sentence – such as ’Entropy of Mike’s image is high’
7. https://github.com/mohaEs/SignCol  star 5

**level**:   2020 **The IEEE Winter Conference on Applications of Computer Vision**
**author**: Dongxu Li (The Australian National University)
**date**: 2020
**keyword**:

- Sign Language

------

# Paper: Word-level Sign Language

<div align=center>
<br/>
<b>Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison</b>
</div>

#### Summary

1. introduce a new large-scale word-level ASL(WLASL) video dataset, containing more than 2000 words performed by over 100 signers.
2. implement and compare two different models,(i) holistic visual appearance based approach,(ii) 2D human pose based approach.
3. propose a novel pose-based temporal graph convolution networks(Pose-TGCN) that model spatial and temporal dependencies in human pose trajectories simultaneously, which has further boosted the performance of the pose-based method.
4. Pose-based and appearance-based models achieve comparable performance-based models achieve  comparable performances up to 62.63% at top-10 accuracy on 2000 words/glosses.
5. word-level sign language recognition (or “isolated sign language recognition”) and sentence-level sign language recognition (or “continuous sign language recognition”).

#### Proble Statement

- The meaning of signs mainly depends on the <font color=red> combination of body motions, manual movements and head poses, and subtle differences</font> may translate into different meanings
- the vocabulary of signs in daily use is large and usually in the magnitude of thousands,( gesture recognition and action recognition only contains at most a few hundred categories).
- a word in sign language may have multiple counterparts in natural languages depending on the context.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715103232337.png)

#### Methods

> For appearance-based methods, providing a base-line be re-training VGG backbone, and GRU as a representative for CNN, and provide a 3D CNN baseline using fine-tuned I3D, which perform better than VGG-GRU baseline. For pose-based methods,extract human poses from orignal videos and use them as input features,propose a novel pose-based model temporal graph convolutional network.reaching up to 62.63%.

- **system overview**:

【DataSet Construction】

- **Collection:**  construct a large-scale signer-independent ASL dataset from websites,like ASLU and ASL_LEX. ASL as well as tutorial on YouTube where a signer performs only one sign.
- **Annotations: ** using some meta information to provide a gloss label for each video, including temporal boundary, body bounding box, signer annotation and sign dialect/variation annotaions.
  - Temporal boundary: indicate the start and the end frames of a sign.
  - Body Bounding-box: using YOLOv3 as  a person detection tool to identify body bounding-boxes of signers in videos, and use the largest bounding-box size to crop the person from video.
  - Signer Diversity: employ the face detector and the face dataset, and then compare the Euclidean distances among the face embeddings to identify the person,
  - Dialect Variation: Annotation: count the number of dialects and assign labels for different dialects automatically.
- Data Arrangement: 

> select top-K glosses with K = {100, 300, 1000, 2000}, and organize them to four subsets, named WLASL100, WLASL300, WLASL1000 and WLASL2000

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715103334772.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715104553269.png)

**【Method Comparison】Image-appearance**

- **Image-appearance based Baselines:**  using VGG16 pretrained on ImageNet to extract spatial features and then feed the extracted features to a stacked GRU.

> to avoid overfiting the training set, the hidden sizes of GRU for the four subsets are set to 64,96,128 and 256, the number of the stacked RNN layers in GRU is 2, and randomly select at most 50 consecutive frames from each video, using cross-entropy losses.

- **3D Convolutional Networks:**  employ the network of I3D as image-appearance base baseline, <font color=red>focusing on the hand shapes and orientations as well as arm movements.</font>,to better capture the spatio-temporal information of signs.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715105724235.png)

**【Method Comparison】Pose-based Baselines**

- two mainstream approaches of pose estimation: regressing the key-points and estimating key-point heatmaps followed by a non-maximal suppression techique.

- **RNN-Methods:**  employs RNN to model the temporal sequential information of the pose movements, and representation output by RNN is used for sign recognition.

  > using Openpose to extract the keypoints of person, 55 body and hand 2D keypoints, and concatenate all the 2D coordinates of each joint as the input feature and feed it to a stacked GRU of 2 layers. 

- **Temporal Graph Neural Networks:**  models the spatial and temporal dependencies of the pose sequence.

  - $$
    X_{1:N}=[x_1,x_2,x_3,...x_N] \\
    x_i\epsilon R^k \\
    H_{n+1}=G_b(H_n)=\theta(A_nH_nW_n)
    $$

> view human body as a fully-connected graph with K vertices and represent the edges in the graph as a weighted adjacency matric$A \epsilon R^{K*K}$ 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200829170303.png)

#### Evaluation

- **Data Pre-processing and Augmentation:**  
  - resize the frame to 256 pixels;
  - randomly crop a 224*224 patch from an input frame and apply a horizontal flipping with probability of 0.5
  - the video of 50 frames are randomly selected and the models are asked to predict labels based on only partial observations of the input video.
  - 4:1:1 to split data

- **Evaluation Metric:** using the mean scores of top-K classification accuracy with K={1,5,10}

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715111050347.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715111127599.png)

#### Notes

  - [x] GRU && LSTM

**level**: ECCV
**author**: Gyeongsik Moon(Seoul National University, Korea)  Shoou-I Yu [(FaceBook Reality Labs)](blob:https://www.facebook.com/b18d4b2f-c5f4-4a45-8110-8eef7a151a5b)
**date**: 2020
**keyword**:     

- handinteraction

------

# Paper: InterHand2.6M

<div align=center>
<br/>
<b>InterHand2.6M: A Dataset and Baseline for
3D Interacting Hand Pose Estimation
from a Single RGB Image</b>
</div>

#### Summary

1. propose a large-scale high-resolution multi-view single and interacting hand sequences dataset, InterHand2.6M;
2. a baseline network, InterNet, for 3D interacting hand pose estimation from a single RGB image, estimate handedness, 2.5D hand pose, and right hand-relative left hand depth from a single RGB image;
3. show that single hand data is not enough, and interacting hand data is essential for accurate 3D interacting hand pose estimation;

#### Proble Statement

- single hand scenarios have limitations in covering all realistic human hand postures for people often interact with each other to interact with each other people or objects;
- the 2.5D hand pose consists of 2D pose in x- and y- axis and root joint-relative depth in z-axis, widely used in 3D human body and hand pose estimation from a single RGB image;
- RootNet[16] obtain an absolute depth of the root joint to lift 2.5D right and left hand pose to 3D space;
  - the interNet predict right hand-relative left hand depth by leveraging the appearance of the interacting hand from the interacting hand from input image;

previous work:

- **Depth-based 3D single hand pose estimation:** 
  - fit a pre-defined hand model to the input depth image by minimizing hand-crafted cost functions, particle swarm optimization, iterative closest point...
  - directly localizes hand joints from an input depth map,
    - by estimating 2D heatmaps for each hand joint;
    - by estimating multi-view 2D heatmaps;
- **RGB-based 3D single hand pose estimation:** 
  - [17] used an image-to-image translation model to generate a realistic hand pose dataset from a synthetic dataset.
  - [41] proposed deep generative models to learn latent space for hand;
- **3D interacting hand pose estimation:** 
  - [1] present a framework that outputs 3D hand pose and mesh from multi-view RGB sequences;
  - [37] by incorporating a physical model;

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915204915815.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915211644483.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915211931856.png)

**【Model one】Dataset Prepare**

- **Dataset release**

> downsized 512×334 image resolution at 5 fps, and downsized 512×334 resolution at 30 fps
>
> The annotation file includes camera type, subject index, camera index, bounding box, handedness, camera parameters, and 3D joint coordinates

- **Single Hand Sequences:**
  1. neutral relaxed: the neutral hand pose. Hands in front of the chest, fingers do not touch, and palms face the side. 
  2. neutral rigid: the neutral hand pose with maximally extended fingers, muscles tense. 
  3. good luck: hand sign language with crossed index and middle fingers.
  4. fake gun: hand gesture mimicking the gun. 
  5. star trek: hand gesture popularized by the television series Star Trek. 
  6. star trek extended thumb: “star trek” with extended thumb.
  7. thumb up relaxed: hand sign language that means “good”, hand muscles relaxed. 
  8. thumb up normal: “thumb up”, hand muscles average tenseness. 
  9. thumb up rigid: “thumb up”, hand muscles very tense. 
  10. thumb tuck normal: similar to fist, but the thumb is hidden by other fingers. 
  11. thumb tuck rigid: “thumb tuck”, hand muscles very tense. 
  12. aokay: hand sign language that means “okay”, where palm faces the side. 
  13. aokay upright: “aokay” where palm faces the front. 
  14. surfer: the SHAKA sign. 
  15. rocker: hand gesture that represents rock and roll, where palm faces the side. 
  16. rocker front: the “rocker” where palm faces the front. 
  17. rocker back: the “rocker” where palm faces the back.
  18. fist: fist hand pose. 
  19. fist rigid: fist with very tense hand muscles. 
  20. alligator closed: hand gesture mimicking the alligator with a closed mouth. 
  21. one count: hand sign language that represents “one.” 
  22. two count: hand sign language that represents “two.” 
  23. three count: hand sign language that represents “three.”
  24. four count: hand sign language that represents “four.” 
  25. five count: hand sign language that represents “five.” 
  26. indextip: thumb and index fingertip are touching.
  27. middletip: thumb and middle fingertip are touching.
  28. ringtip: thumb and ring fingertip are touching. 
  29. pinkytip: thumb and pinky fingertip are touching. 
  30. palm up: has palm facing up. 
  31. finger spread relaxed: spread all fingers, hand muscles relaxed. 
  32. finger spread normal: spread all fingers, hand muscles average tenseness. 
  33. finger spread rigid: spread all fingers, hand muscles very tense. 
  34. capisce: hand sign language that represents “got it” in Italian. 
  35. claws: hand pose mimicking claws of animals. 
  36. peacock: hand pose mimicking peacock. 
  37. cup: hand pose mimicking a cup. 
  38. shakespeareyorick: hand pose from Yorick from Shakespeare’s play Hamlet. 
  39. dinosaur: hand pose mimicking a dinosaur. 
  40. middle finger: hand sign language that has an offensive meaning

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916185336884.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916185414522.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916185437395.png)

- **ROM of single hand sequence**  ROM represents conversational gestures with minimal instructions.
  - five count: count from one to five. 
  -  five countdown: count from five to one. 
  -  fingertip touch: thumb touch each fingertip. 
  -  relaxed wave: wrist relaxed, fingertips facing down and relaxed, wave.
  -  fist wave: rotate wrist while hand in a fist shape. 
  - prom wave: wave with fingers together. 
  -  palm down wave: wave hand with the palm facing down. 
  -  index finger wave: hand gesture that represents “no” sign. 
  - palmer wave: palm down, scoop towards you, like petting an animal. 
  -  snap: snap middle finger and thumb. 
  -  finger wave: palm down, move fingers like playing the piano. 
  -  finger walk: mimicking a walking person by index and middle finger. 
  -  cash money: rub thumb on the index and middle fingertips. 
  - snap all: snap each finger on the thumb.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916185530746.png)

- **Interacting hand sequences**
  - right clasp left: two hands clasp each other, right hand is on top of the left hand.
  -  left clasp right: two hands clasp each other, left hand is on top of the right hand.
  -  fire gun: mimicking a gun with two hands together. 
  -  right fist cover left: right fist completely covers the left hand. 
  -  left fist cover right: left fist completely covers the right hand. 
  -  interlocked fingers: fingers of the right and left hands are interlocked. 
  - pray: hand sign language that represents praying.
  - right fist over left: right fist is on top of the left fist. 
  - left fist over right: left fist is on top of the right fist. 
  -  right babybird: mimicking caring a babybird with two hands, the right hand is placed at the bottom. 
  -  left babybird: mimicking caring a babybird with two hands, the left hand is placed at the bottom. 
  - interlocked finger spread: fingers of the right and left hands are interlocked yet spread. 
  - finger squeeze: squeeze all five fingers with the other hand.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916185555201.png)

- **ROM of Interacting hand sequence**
  - palmerrub: rub palm of one hand with opposite hand’s thumb. 
  -  knuckle crack: crack each finger by having the opposite hand compress a bent finger. 
  -  golf claplor: light clap, left over right. 
  -  itsy bitsy spider: finger motion used when singing the children song “itsy bitsy spider”, like this (link). 
  -  finger noodle: fingers interlocked, palms facing opposite directions, wiggle middle fingers.
  -  nontouch: two hands random motion, hands do not touch.
  -  sarcastic clap: exaggerated, slow clap. 
  -  golf claprol: light clap, right over left. 
  - evil thinker: wrist together, tap fingers together one at a time. 
  -  rock paper scissors: hold rock, then paper, then scissors. 
  -  hand scratch: using the opposite hand, lightly scratch palm then top of hand; switch and do the same with the other hand. 
  - touch: two hands interacting randomly in a frame, touching. 
  -  pointing towards features: using the opposite index finger, point out features on the palm and back of the hand (trace lifelines/wrinkles, etc.). 
  -  interlocked thumb tiddle: interlock fingers, rotate thumbs around each other. 
  -  right finger count index point: using the right pointer finger, count up to five on the left hand, starting with the pinky. 
  - left finger count index point: using left pointer finger, count up to five on the right hand, starting with the pinky. 
  -  single relaxed finger: this consists of a series of actions: (1) touch each fingertip to the center of the palm for the same hand, do this for both hands, (2) interlock fingers and press palms out, (3) with the opposite hand, hold wrist, (4) with the opposite hand, bend wrist down and back, (5) point at watch on both wrists, (6) circle wrists, (7) look at nails, and (8) point at yourself with thumbs then with index fingers

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916185619762.png)

**【Module Two】Exact Methods**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915212634843.png)

- **Handedness estimation:** takes the image feature F , using two fully-connected layers with hidden activation size 512 followed by the ReLU activation function, and the last is sigmoid activation function;
- **2.5D right and left hand pose estimation:**     用到时再看网络结构

- **Right hand-relative left hand depth estimation:** 用到时再看网络结构

#### Notes <font color=orange>去加强了解</font>

  - https://github.com/facebookresearch/InterHand2.6M/releases  下载地址。
  - facebookAI lab:   https://v.qq.com/x/page/j0956p2juqp.html

![Oculus Connect](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916193851551.png)

**level**: 
**author**: Nicolas Pugeault & Richard Bowden
**date**: 2011
**keyword**:

- ASL

> Pugeault, Nicolas, and Richard Bowden. "Spelling it out: Real-time ASL fingerspelling recognition." *2011 IEEE International conference on computer vision workshops (ICCV workshops)*. IEEE, 2011.

------

# Paper: Spelling It Out

<div align=center>
<br/>
<b>Spelling It Out: Real-Time ASL Fingerspelling Recognition</b>
</div>
#### Summary

1. presents an interactive hand shape recognition user interface for ASL finger-spelling;
2. Hand-shapes corresponding to letters of the alphabet are characterized using appearance and depth images and classified using random forests;
3. extract feature from images and depth images;

#### Methods

- **Problem Formulation**:
  - finger-spelling is single-handed, removes the difficulties of hands occluding one another;
  - some of the signs in the alphabet are visually very similar;
  - different persons perform signs in different way;
  - the differences between people's hands and natural dexterity leads to differences in the execution of signs between different signers;
  - need to run in real-time;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915084116179.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915084158379.png)

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915083100309.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915083755524.png)

![features](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915084527995.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915084833977.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915084945109.png)

#### Notes <font color=orange>去加强了解</font>

  - http://www. youtube.com/watch?v=0tCGMhbTDmw

**level**: 
**author**: Hamid Reza Vaezi Joze(Microsoft)
**date**:  2018
**keyword**:

- ASL

> Joze, Hamid Reza Vaezi, and Oscar Koller. "Ms-asl: A large-scale data set and benchmark for understanding american sign language." *arXiv preprint arXiv:1812.01053* (2018).

------

# Paper: MS-ASL

<div align=center>
<br/>
<b>MS-ASL: A Large-Scale Data Set and
Benchmark for Understanding
American Sign Language</b>
</div>

#### Summary

1. propose the first real-life large-scale sign language data set comprising over 25000 annotated videos, covering of 1000 signs in challenging and unconstrained real-life recording conditions;
2. evaluate current state-of-the-art approaches: 2D-CNN-LSTM, body key point, CNN-LSTM-HMM and 3D-CNN as baseline;
3. propose I3D as a powerful and suitable architecture for sign language recognition;

#### Research Objective

  - **Application Area**:
- **Purpose**:  

#### Proble Statement

- **Sign Language Data sets:** 
  - **Purdue RVL-SLLL ASL dataset:** contains 10 short stories with a vocabulary of 104 signs and total count of 1834 produced by 14 native signers in a lab environment under controlled lighting;
  - **RWTH-BOSTON-50\100\400:** contain isolated language with a vocabulary of 50-104 signs. the 400 contains a vocabulary of 483 signs and also constitutes of continuous signing by 5 signers;
  - **Devisign:** a chinese sign language data set featuring isolated single signs perform by 8 non-natives in a laboratory environment, covering a vocabulary of up to 2000 isolated signs and provides RGB with depth information in 24000 recordings.
  - **Finish S-pot:** base on lexicon, covers 1211 isolated citation form signs that need to be spotted in 4328 continuous sign language videos.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915095247256.png)

#### Methods

- **Dataset:**
  - one sample video may include repetitive act of a distinct signs;
  - one word can sign differently in different dialects based on geographical regions;
  - includes large number of signers and is a signer independent data set;
  - they are larege visual variabilities in the videos such as background, lighting, clothing and camera view point;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915100200887.png)

【**Evaluation Scheme** 】using average per class accuracy; using average per class top-five accuracy;

> 2D-CNN: used VGG16 network followed by an average pooling and LSTM layer of size of 256 with batch normalization, the final layer are a 512 hidden units followed by a fully connected layer for classification; 
>
> use googlenets as 2D-CNN with 2 bi-directional LSTM layers and 3-state HMM on Phoenix2014;

> body key-points: extracted all the key-frame point for all sample , using 64 frames for time window;
>
> - the input: 64\*137\*3 representing x,y coordinates and confidence values for the 137 key-points;
> - using co-occurrence network(HCN), to learn form 137 keypoints as well as per frame difference of them;

> 3D-CNN: using C3D, and I3D network;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915102058478.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915102149811.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915102239850.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200915102321944.png)

#### Notes <font color=orange>去加强了解</font>

  - 如果使用跨源数据转化，则第二种方法可以使用；

**level**: CVPR
**author**:  Necati Cihan Camg¨oz(camgoz), Microsoft Munich (German)
**date**: 2020
**keyword**:

- sign languages

------

# Paper: Sign Language Transformers

<div align=center>
<br/>
<b>Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation
</b>
</div>

#### Summary

1. introduce a novel transformer based architecture that jointly learns continuous sign language recognition and translation while being trainable in an end-to-end manner.
2. a novel multi-task formalization of CSLR and SLT which exploits the supervision power of glosses, without limiting the translation to spoken language.
3. the first successful application of transformers for CSLR and SLT which achieves state-of-art results in both recognition and translation accuracy, vastly outperforming all comparable previous approaches.
4. a broad range of new baseline results to guide future research in this field.

#### Proble Statement

- such translation system requires several sub-tasks, **Sign Segmentation, Sign Language Recognition and Understanding, Sign Language Translation**

previous work:

- previous sequence-to-sequence base literature on SLT can be categorized into two groups:
  - utilized a state-of-art CSLR method to obtain sign glosses, and then used an attention-based text-to-text NMT model to learn the sign gloss to spoken language sentence translation.
  - focus on translation from sign video representation to spoken language with no intermediate representations, to learn $p(S|V)$ directly. With enough data and a sufficiently sophisticated network architecture, such model can theoretically realize end-to-end SLT with no need for human-interpretable information that act as a bottle-neck.

#### Methods

- **Problem Formulation**:

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200827104715.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200827105016.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200827104944.png)

**【Module One】Spatial and Word Embeddings**
$$
m_u=WordEmdedding(w_u)\\
f_t=SpatialEmbedding(I_t)
$$

> $m_u$  is the embedded represetation of the spoken language word $w_u$ and $f_t$ corresponds to the non-linear frame level spatial representation obtained from a CNN.

$$
f_t'=f_t+PositionalEncoding(t)\\
m_u'=m_u+PositionEncoding(u)
$$

> PositionalEncoding is a predefined function which produces a unique vector in the form of a phase shifted sine wave for each time step.

**【Module Two】 Sign Language Recognition Transformers**
$$
z_t=SLRT(f_t'|f_{1:T}')
$$

> where $z_t$ denotes the  spatio-temporal representation of the frame $I_t$ which is generated by SLRT at time step t, given the spatial representations of all of the video frames,$f_{1:T}'$'

**【Module Three】 Sign language Translation Transformers**

#### Evaluation

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200827111159.png)

#### Notes <font color=orange>去加强了解</font>

  - https://github.com/neccam/slt

**level**: arxiv 
**author**: Kayo Yin
**date**: 2020
**keyword**:

- SLR, CSLR, SLT

------

# Paper: SLT Transformers

<div align=center>
<br/>
<b>Sign Language Translation with Transformers</b>
</div>

#### Summary

1. Using SLR system to extract sign language glosses from videos, and using translation system generates spoken language translations from the sign language glosses.
2. <font color=red>Report a wide range of experimental results for various Transformer setups and introduce the use of Spatial-Temporal Multi-Cue networks in an ent-to-end SLT system with Transformer.</font>
3. Perform experiment on RWTH-PHOENIX-Weather 2014T, and ASLG-PC12, and improves on the current state-of-art by over 5-7 points.

#### Proble Statement

- **Tokenization Problem:** analyze videos of sign language to generate sign language glosses that capture the meaning of the sequence of different signs.

【Previous work】

- CSLR: divide the task into three sub-tasks: alignmnet, single-gloss SLR, and sequence construction while others perform the task in an end-to-end fashion using DNN. <font color=red>Sequence to Sequence model that directly translate ASL glosses from ASLG_PC12 dataset without taking sign language data itself.</font>

> approach to the problem as visual recognition task and ignores the underlying grammatical and linguistic structures unique to SL.

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715162316328.png)

【Transformer Model】

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715163955352.png)

#### Evaluation

- Dataset: 
  - **RWTH-PHOENIX-Weather 2014T**: the only publicly available dataset with both gloss level annotations and spoken language translations for sign language videos that is of sufficient size and challenge for deep learning.
  - **ASLG-PC12**: SLT on this dataset has only been performed using RNN-based sequence-to-sequence attention networks.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715164236920.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715212229107.png)

#### Conclusion

- Perform the first thorough study of using the Transformer network for SLT and demonstrate how it outperforms previous NMT architectures for this task.
- Make the first use of weight tying, transfer learning with spoken language data and ensemble learning SLT and report baseline results of Transformers in various setups.
- Improve on the state-of-art results in German SLT on the RWTH-PHOENIX-Weather 2014T dataset for both sign language gloss to spoken language text translation and end-to-end sign language video to spoken language text translation, and in American SLT on the ASLG-PC12 dataset.
- demonstrate how a Spatial-Temporal Multi-Cue network provides better end-to-end performance when use for CSLR in STL than previous approaches and even surpass translation using ground truth glosses.

#### Notes <font color=orange>去加强了解</font>

  - [x] Sign Language Glossing:

> Corresponds to transcribing sign language word-for-word by means of another written language, differing from translation as they merely indicate what each part in a sign language sentence mean, and does not form an appropriate sentence in the writtn language that signifies the same thing.

- [ ] https://github.com/kayoyin/transformer-slt
- [ ] Open-NMT library

**level**: CCF_A   CVPR
**author**: Necati Cihan Camgoz1 (University of Surrey)
**date**: 2018
**keyword**:

- Sign Language

------

# Paper: Neural Sign Language Translation

<div align=center>
<br/>
<b>Neural Sign Language Translation</b>
</div>

#### Summary

1. Object at generate spoken language translations from sign language videos, taking into account the different word orders and grammar.
2. Formalize SLT in the framework of Neural Machine Translation(NMT) for both end-to-end and pretrained settings(using expert knowledge), to jointly learn the spatial representations, the underlying language model, and the mapping between sign and spoken language.
3. Evaluate on the PHOENIX-WEATHER 2014T datasets.

#### Proble Statement

previous work:

- computer vision researchers adopted CTC and applied it to weakly labeled visual  problems, such as lip reading, action recognition, hand shape recognition and CSLR.
- seq2seq along with Encoder-Decoder network architectures and the emergence of the NMT field.

#### Methods

- **Question Define:** learn the conditional probability $p(y|x)$ of generating a spoken language sentence $y=(y_1,y_2,...,y_U)$with U number of words given a sign video $x=(x_1,x_2,...,x_T)$ with $T$ number of frames.
  - the number of video frames is much more than the number of words in its spoken language tranlation.
  - the alignment between sing and spoken language sequences are usually unknown and non-monotonic.

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716161409107.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716230416733.png)

- Spatial and Word Embedding: to transform the sparse one-hot vector representations. Given a sing video x, using 2D CNN learns  to extract non-linear frame level spatial representation as:

$$
f_t=SpatialEmbedding(x_t)
$$

$x_t$: video frame. for word embedding, using a fully connected layer that learns a linear projection from one-hot vectors of spoken language words to a denser space as:   and $g_u$ is the embedded version of the spoken word $y_u$.
$$
g_u=WordEmbedding(y_u)
$$


- Tokenization Laryer: in NMT the input and output sequences can be tokenized at many different levels of complexity: characters, words, N-grams or phases.

  - Low level tokenization schemes, like character level, allow smaller vocabularies to be used, but require long term relationships to be maintained.
  - High level tokenization makes the recognition problem far more difficult due to vastly increased vocabularies.
  - explore both "frame level" and "gloss level" input tokenization, with RNN-HMM to force alignment, the output tokenization is at the word level.

$$
z_{1,N}=Tokenization(f_{1:T})
$$

- **Attention-based Encoder-Decoder Networks: ** learn a mapping function $B(z_{1:N})->y$, which will maximize the probability $p(y|x)$ based on tokenized embeddings $z_{1:N}$ of a sign video x. 
  - $o_n:$ the hidden state produced by recurrent unit n;
  - $o_{n+1}:$ a zero vector;
  - $z_n:$ reverse representations of a sequence.

$$
O_n=Encoder(z_n,o_{n+1})\\
y_u,h_u=Decoder(g_{u-1},h_{u-1})
$$

​            By generating sentences word by word, the Decoder decomposes the conditional probability $p(y|x)$ into ordered conditional probabilities:
$$
p(y|x)=\prod_{u=1}^Up(y_u|y_{1:u-1},h_{sign})
$$

- **Attention Mechanisms:** deal with information bottleneck caused by representing a whole sign language video with a fixed size vector.

  - suffer from long term dependencies and vanishing gradients.
  - utilize attention mechanisms to provide additional information to decoding phase, to learn where to focus while generating each word, providing alignment of sign videos and spoken languages  sentences.
  - $c_u$: context vector
  - $u$: decoding step
  - $y_n^u:$ the attention weights, regarded as interpreted as the relevance of an encoder input $z_n$ to generating the word $y_u$.
  - $tanh(W[h_u;o_n])$: the attention vector.

  $$
  c_u=\sum_{n=1}^Ny_n^uo_n\\
  For Alignment: y_n^u=exp(score(h_u,o_n))/\sum_{n'=1}exp(score(h_u,o_{n'}))\\
  score(h_u,o_n)=\begin{cases}h_u^TWo_n [Multiplication]\\
  V^Ttanh(W[h_u;o_n]) [Concatenation]
  \end{cases}\\
  y_u,h_u=Decoder(g_{u-1},h_{u-1},a_{u-1})
  $$

#### Evaluation

  - **Environment**:   
    - Dataset: RWTH-PHOENIX-Weather 2014

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717083519814.png)

- **Quantitative Experiments:**
  - Gloss2Text: simulate having perfect SLR system as an intermediate tokenization.
  - sign2Text: covers the end-to-end pipeline translating directly from frame level sign language video into spoken language.
  - Sign2Gloss2Text: uses SLR system as tokenization layer to add intermediate supervision.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717085452987.png)

#### Conclusion

- use state-of-the-art seq2seq based deep learning methods to learn: the spatio-temporal representation of the signs, the relation between these signs and how these sign map to the spoken or written language.
- the first exploration of the video to text SLT problem.
- the first publicly available video segments, gloss annotations and spoken language translations.
- A broad range of baseline results on the new corpus including a range of different tokenization and attention schemes in addition to parameter recommendations.

#### Notes <font color=orange>去加强了解</font>

  - [ ] https://github.com/neccam/nslt
  - [x] BLEU 介绍

> BLEU(Bilingual Evaluation understudy)方法由IBM提出，这种方法认为如果熟译系统魏译文越接近人工翻翻译结果，那么它的翻译质量越高。所以，<font color=red>评测关键就在于如何定义系统译文与参考译文之间的相似度。</font>BLEU 采用的方式是比较并统计共同出现的n元词的个数，即<font color=red>统计同时出现在系统译文和参考译文中的n元词的个数，最后把匹配到的n元词的数目除以系统译文的单词数目，得到评测结果。</font>
>
> - $Count(n-gram)$是某个n元词在候选译文中的出现次数，而$MaxRefCount(n-gram)$是该n元词在参考译文中出现的最大次数。
> - ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717085120019.png)
>
> $$
> Count_{clip}(n-gram)=min{Count(n-gram),MaxRefCount(n-gram)}\\
> 
> BLEU=BP*exp(\sum_{n=1}^Nw_nlogP_n)\\
> BP=\begin{cases}1 if C>r\\
> e^{1-r/c} if c\le r\end{cases}
> $$

- [x] NIST评测方法介绍

> NIST(National Institute of standards and Technology)方法是在BLEU方法上的一种改进。它并不是简单的将匹配的n—gram片段数目累加起来，而是求出每个n-gram的信息量(information)，然后累加起来再除以整个译文的n-gram片段数目。
>
> ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717085318749.png)

**level**: CCF_A
**author**: Simon Alexanderson, Gustav Eje Henter, Taras Kucherenc
**date**: 2020,  
**keyword**:

- Motion Cpature; Animation; Neural Networks

------

# Paper: Style-Controllable

<div align=center>
<br/>
<b>Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows
</b>
</div>

#### Summary

1. Given a high-level input, the learned network translates these instructions into an appropriate sequence of body poses.
2. By adapting a deep learning-based motion synthesis method called MoGlow, we propose a new generative model for generating state-of-the-art realistic speech-driven gesticulation.
3. Produce a battery of different, yet plausible, gestures given the same input speech signal, and demonstrate the ability to exert directorial control over the output style, such as gesture level, speed, symmetry and spacial extent.

#### Research Objective

  - **Application Area**: Animation, crowd simulation, virtual agents, social robots.
- **Purpose**:  

#### Proble Statement

- Lack of coherence in gesture production-the same speech utterance is usually accompanied by different gestures from speaker to speaker and time to time.

previous work:

- **Data-driven human body-motion generation:**  locomotion, lip movements, and head motion.
  - large variation in the output given the same control.
- **Deterministic and Probabilistic gesture generation:**  this article are capable of generating unseen gestures.
  - Hasegawa et al.[HKS 18] designed  a speech-driven neural network producing 3D motion sequences.
  - Kucherenko et al.[KHH19] representation learning for the motion, achieving smoother gestures.
  - Yoon et al.[YKJ 19] <font color=red>using seq2seq on TED-talk data to map text transcriptions to 2D gestures.</font>
  - Ahuja et al. [AMMS 19] conditioned pose prediction not only on the audio of the agent, but also on the audio and pose of the interlocutor.
  - Sadoughi& Busso [SB19] <font color=red> used a probabilistic graphical model for mapping speech to gestures, but only experimented on three hand gesture and two head motions.
- **Style Control:** different levels of animated motion control.
  - Normoyle et al. [NLK*13] used motion editing to identify links between motion statistics and the emotion intensities recognised by human obervers.
- **Probabilistic generative sequence models:**  

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717173653914.png)

【Function 1】Normalising flows and glow

> learn the multi-dimensional next-step distribution of poses $X$ in a stationary autoregressive model of pose sequences $x=[x_1,x_2,x_3,...,x_t]$ using normalising flows.

$$
x=f(z)=f_1(f_2(...f_N(z)))\\
z_n(x)=f_n^{-1}(...f_1^{-1}(x))\\
z_0(x)=x;\\
z_n(x)=z;
$$

【Function 2】MoGlow for gesture generation

> By using Glow to describe the next-step distribution in an autoregressive model, it also adds control over the output and uses recurrent neural networks for the long-term memory across time.
>
> - $x_{t-i:t-1}$: the previous poses.
> - $c_t$: a current control signal.
>
> ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717205409444.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200717174844348.png)

#### Evaluation

  - **Dataset**  :
      - [Trinty Gesture Dataset:]( trinityspeechgesture.scss.tcd.ie) joint speech and gestures, consists of 244 minutes of motion capture and audio of one male actor speaking spontaneously on different topics. the actor's movements were captured with a 20-camera Vicon system and solved to a skeleton with 69 joints.

#### Conclusion

- Adapting MoGlow to speech-driven gesture systhesis.
- Adding a framework for high-level control over gesturing style.
- Evaluating the use of these methods for probabilistic gesture systhesis.

#### Notes <font color=orange>去加强了解</font>

  - [ ] github.com/simonalexanderson/StyleGestures
  - [x] Normalizing Flows: https://gebob19.github.io/normalizing-flows/

> Normalizing flows learn an invertible mapping $f:x->z$, where $X$ is our data distribution and $Z$ is a chosen laten-distribution.
>
> Normalizing flows are part of the generative model family, which includes Variational Autoencoders(VAEs), and generative adversarial networks. Once get the mapping $f$, generate data by sampling $z~p_z$ and then applying the inverse transformation, $f^{-1}(z)=x_{gen}$
>
> - Advantage:
>   - NFs optimize the exact log-likelihood of the data
>     - VAEs optimize the lower bound
>     - GANs learn to fool a discriminator network.
>   - NFs infer exact latent-variable values $z$, which are useful for downstream tasks 
>     - VAE infers a distribution over latent-variable values.
>     - GANs don't have a latent-distribution
>   - Potential for memory savings, with NFs gradient computations scaling constant to their depth.
>     - VAE's and GAN's gradient  computations scale linearly to their depth 
>   - NFs require only an encoder to be learned.
>     - VAEs require encoder and decoder networks
>     - GANs require generative and discriminative networks.



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/sign-language-introduce/  

