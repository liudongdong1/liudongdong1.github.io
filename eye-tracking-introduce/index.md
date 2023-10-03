# Eye-Tracking Introduce


### Paper: **Weakly-Supervised Physically Unconstrained Gaze Estimation**

> 本次工作所探讨的问题是从人类互动的视频中进行弱监督的视线估计，基本原理是利用人们在进行 "相互注视"（LAEO）活动时存在的与视线相关的强烈的几何约束这一发现。通过提出一种训练算法，以及为该任务特别设计的几个新的损失函数，可以从 LAEO 标签中获得可行的三维视线监督信息。在两个大规模的 CMU-Panoptic 和 AVA-LAEO 活动数据集的弱监督下，证明了半监督视线估计的准确性和对最先进物理无约束的自然 Gaze360 视线估计基准的跨域泛化的显著改善。
>
> - 论文链接：https://arxiv.org/abs/2105.09803
> - 项目链接：https://github.com/NVlabs/weakly-supervised-gaze

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210528104346661.png)

**level**:    visapp cited by 334
**author**:Fabian Timm
**date**: 2011
**keyword**:

- Eye Center localisation, pupil and iris localisation, image gradients, feature extraction, shape analysis    <font color=red>user attention , gaze estimation</font>

------

# Paper: Eye Center Localisation

<div align=center>
<br/>
<b>ACCURATE EYE CENTRE LOCALISATION BY MEANS OF GRADIENTS
</b>
</div>

#### Summary

1. propose an approach for accurate and robust <font color=red>eye center localization by using image gradients.</font>
2. The <font color=red>maximum of this function corresponds to the location where most gradient vectors intersect and thus to the eye's center.</font>
3. evaluate the method on the very challenging BioID database for eye center and iris localisation.
4. <font color=red>using  Gaussian ﬁlter, make it invariant to changes in scale, pose, contrast and variations in illumination.</font>

#### Proble Statement

- low resolution,  low contrast or occlusions

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200407162422047.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200407162455332.png)

#### Conclusion

- a novel approach for eye center localization, which defines the center of the circular pattern as the location where most of image gradients intersect, and derive a mathematical function that reaches its maximum at the center of the circular pattern.
- incorporate prior knowledge about the eye appearance and increase the robustness
- apply simple post precessing techniques to reduce problems that arise in the presence of glasses, reflections inside glasses, or prominent eyebrows.

#### Notes 

  - c++:  drishti-master
  - python: eye-tracker![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429155547489.png)
  - 基于google vision                       基于opencv

<figure class="second">
	<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410121540479.png" width=35% height=250>
    <img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410121622569.png" width=35% height=250>
</figure>
> Abdelkareem Bedri, Diana Li, Rushil Khurana, Kunal Bhuwalka, and Mayank Goel. 2020. FitByte: Automatic Diet Monitoring in Unconstrained Situations Using Multimodal Sensing on Eyeglasses. In *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems* (*CHI '20*). Association for Computing Machinery, New York, NY, USA, 1–12. DOI:https://doi.org/10.1145/3313831.3376869

------

# Paper: FitByte

<div align=center>
<br/>
<b>FitByte: Automatic Diet Monitoring in Unconstrained Situations Using Multimodal Sensing on Eyeglasses
</b>
</div>

#### Summary

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203084558388.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203085202343.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203090958185.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203091509352.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203091819812.png)

- Introduce FitByte, a multi-modal sensing approach on a pair of eyeglasses that tracks all phases of food intake.
  - chewing by monitoring jaw motion using four gyroscopes around the wearer's ears;
  - swallowing by listening to vibrations in the throat using a proximity sensor;
  - visuals of the consumed food using a downward-pointing camera.
- a data processing pipeline to identify food consumption moments and automatically record food visuals to aid in identifying the food type.
- a real-time implementation of the algorithm that allows an untethered wearable to monitor diet and capture food visual using the build-in battery.
- a preliminary investigation of FitByte's social acceptability and privacy concerns.

#### Relative

- **Automatic diet monitor(ADM):**  
  - prior work focused on detecting atomic actions that a suer makes to eat or drink, such as detecting hand to mouth movement, chewing and swallowing by monitoring activities of the wrist, jaw and throat, detecting chewing and swallowing sounds using different sensing modalities.
  - **Jaw Motions: ** GlassSense monitors jaw activity from the template using two load cells embedded in the hinge of custom eyeglasses to detect eating episodes. Chun et al. used an infrared proximity sensor placed on a necklace and positioned it pointing upward to detect jaw motion.  EarBit uses inertial sensors to detect jaw motion due to chewing,

- **Monitroing diet:** 
  - whether the user is eating,   what is the user eating,  how much is the user eating
  - usable performance in the user's environment has been elusive,  many food types are hard to detect(such as liquids and soft fodds); the wearable sensors and machine learning models don't generalize across users.

> c*Rushil Khurana and Mayank Goel. 2020. Eyes on the Road: Detecting Phone Usage by Drivers Using On-Device Cameras. In* *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems* *(**CHI '20**). Association for Computing Machinery, New York, NY, USA, 1–11. DOI:https://doi.org/10.1145/3313831.3376822*

------

# Paper: Eyes on the Road

<div align=center>
<br/>
<b>Eyes on the Road: Detecting Phone Usage by Drivers Using On-Device Cameras
</b>
</div>

#### Summary

- present a lightweight, soft-ware-only solution that uses the phone 's camera to observe the car's interior geometry to distinguish phone position and orientation, to distinguish between driver and passenger phone use.

- given many users use their phone's camera to unlock the phone, the camera is the perfect sensor to sense the usage context too.  based on that the interiors of the cars are very similar, the exact placement, color, texture, etc. of the objects such as handlebar, sunroof, visor, windows might change but the basic geometry remains consistant.

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203083446419.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203083605728.png)

> Christoph Schröder, Sahar Mahdie Klim Al Zaidawi, Martin H.U. Prinzler, Sebastian Maneth, and Gabriel Zachmann. 2020. Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length. In *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems* (*CHI '20*). Association for Computing Machinery, New York, NY, USA, 1–7. DOI:https://doi.org/10.1145/3313831.3376534

------

# Paper: Robustness of Eye Movement Biometrics

<div align=center>
<br/>
<b>Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length
</b>
</div>

#### Summary

- biometric identification based on human's eye movement characteristics.
- investigate some of the factors that affect the robustness of the recognition rate different classifiers on gate trajectories, the type of stimulus and the tracking trajectory length.
- present two extensions of the methods by George and Routray, using more features of different classifier.
- first to compare the stimulus-agnostic performance of gaze biometrics methods.
- analyze the effect of different tracking lengths on classification performance.
- [opensource](https://cgvr.cs.uni-bremen.de/)

#### Relative 

- **Datasets**:

  - Tex dataset, participants read a complex poem presented on a monitor,

  - RAN dataset, participants observed a randomly moving dot on the screen,

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203075256974.png)

> Chun Yu, Ke Sun, Mingyuan Zhong, Xincheng Li, Peijun Zhao, and Yuanchun Shi. 2016. One-Dimensional Handwriting: Inputting Letters and Words on Smart Glasses. *Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems*. Association for Computing Machinery, New York, NY, USA, 71–82. DOI:https://doi.org/10.1145/2858036.2858542

# Paper: One-Dimensional

<div align=center>
<br/>
<b>One-Dimensional Handwriting: Inputting Letters and Words on Smart Glasses
</b>
</div>

#### Summary

- map two-dimensional handwriting to a reduced one-dimensional space, while achieving a balance between memorability and performance efficiency,

- derive a set of  ambiguous two-length unistroke gestures, each mapping to 1-4 letters, and the guidelines is as follows:

  - mimic traditional handwriting: 1D handwriting stroke gesture should be easy to learn.
  - minimize levels of stroke lengths: to perform stroke gestures accurately and efficiently.
  - Single stroke input: to perform stroke gestures efficiently.

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202225458759.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210203074604154.png)

  > Gesture desing for individual charatersl, the red arrows starting with a cirvle are the fist strokes in traditional handwriting, letters e,s, z are rotated 90 degrees counterclockwise, black arrwos repersent the proposed design for one-dimensional gesture.

#### Relative work

- H4-writer uses four buttons to input letters and leverages huffman coding to minimize key sequences b  considering letter frequency.

- **Multi-step interation**

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202224418678.png)

  - GesText allows user to input gesture with two-step directional gestures.

  - Lurd-writer uses horizontal and vertical movements of cursor to recursively reduce the letter range to the target one.

  - Swipeboard divides the QWERTY keyboard on very small touchscreen into nine region.

  - **Unistroke letters**: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202224519171.png)

    - Graffiti : resemble handwritten Roman;

    - EdgeWrite:  the alphabet was designed to mimic its handwritten counterparts for quick learning. the gestures recognition was performed based on the sequence of corners that were hit.

      ![image-20210202224734676](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202224734676.png)

> Jun Gong, Yang Zhang, Xia Zhou, and Xing-Dong Yang. 2017. Pyro: Thumb-Tip Gesture Recognition Using Pyroelectric Infrared Sensing. In *Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology* (*UIST '17*). Association for Computing Machinery, New York, NY, USA, 553–563. DOI:https://doi.org/10.1145/3126594.3126615

------

# Paper: Pyro

<div align=center>
<br/>
<b>Pyro: Thumb-Tip Gesture Recognition Using
Pyroelectric Infrared Sensing
</b>
</div>

#### Summary

- Pyro, a micro thumb-tip gesture recognition technique based on thermal ingrared signals radiating from the fingers.
- developed a self-contained prototype consisting of the infrared pyoelectric sensor (PIR sensor), a custom sensing circuit and software for signal processing and machine learning.
- PIR sensor is energy-efficient and generates very little heat.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202222512388.png)

#### Relative

- **Pyroelectric Infrared Sensing:**  sensitive to thermal radiation emitted by the human boty(8-14um), to detect the presence of humans or trigger alarms.
  - human localization, motion direction detection, thermal imaging, radiometry, thermometers, and biometry.
- `pressure sensors, electrical impedance tomography sensors, magnetic senors, acoustic sensing, capacitive sensors.`

> Zheer Xu, Weihao Chen, Dongyang Zhao, Jiehui Luo, Te-Yen Wu, Jun Gong, Sicheng Yin, Jialun Zhai, and Xing-Dong Yang. 2020. BiTipText: Bimanual Eyes-Free Text Entry on a Fingertip Keyboard. In *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems* (*CHI '20*). Association for Computing Machinery, New York, NY, USA, 1–13. DOI:https://doi.org/10.1145/3313831.3376306

------

# Paper: BiTipText

<div align=center>
<br/>
<b>BiTipText: Bimanual Eyes-Free Text Entry on a Fingertip Keyboard
</b>
</div>

#### Summary

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202214710223.png)

- first conducted a study to understand the natural expectation of the handedness of the keys in a QWERTY layout for users. using two index finger, the size of the input space doubles, thus the keys are larger and less crowded, which is helpful for reducing tapping errors.
- the text input can be carried out unobtrusively and even with the user looking at the keyboard, leading a better performance when compared with eyes-on input and save screen real estate for devices with limited screen space.
- a printed 3×3 capacitive touch sensor matrix with diamond-shaped electrodes of 5 mm diameter and 6.5mm center-to-center spacing. Our prototype was developed using a flexible printed circuit (FPC). The sensor is 0.025 – 0.125 mm thick and 21.5mm × 27mm wide. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210202221241703.png)

#### Relative

- Micro-Gesture input: navigation, triggering commands, text input with thumb-tip;
  - FingerPad: detects 2D  touch input on the tip of the index finger using a figner-worn device with electromagnetic sensing.
  - Pyro detects thumb-tip gestures drawn on the tip of index of finger based on the thermal radiation emitting from the user's finger.
  - **Interactive sking technologies:** 
    - **iSkin**  is a thin, flexible, stretchable skin overlay, made of biocompatible materials capable of sensing touch input on the skin.
    - **DuoSkin** detect touch input on the skin using an interactive tatoos that can detect touch, squeeze and bend on the skin.
- Text Entry on small Devices: due to the lack of the input space;
  - Yu et al's work allows users to type on a one-dimensional touch sensor using unistroke gestures.
  - WrisText, perform text input by whirling the wrist.
  - ThumbText allows a user to perform text input using a ring-sized touchpad worn on the index finger.

> Fuhl, Wolfgang, Gjergji Kasneci, and Enkelejda Kasneci. "TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types." *arXiv preprint arXiv:2102.02115* (2021).

# Paper: TEyeD

<div align=center>
<br/>
<b>TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types
</b>
</div>


#### Summary

- present the world's largest unified public data set of eye images taken with head-mounted devices, the images are from various tasks, including car rides, simulator rides, outdoor sports activities and daily indoor activities.
- the dataset inlcudes 2D&3D landmarks, semantic segmentation, 3D eyeball annotation and the gaze vector and eye movement types for all images.
- data&code available: t https://unitc-my.sharepoint.com/:f:/g/personal/iitfu01_cloud_unituebingen_de/EvrNPdtigFVHtCMeFKSyLlUBepOcbX0nEkamweeZa0s9SQ?e=fWEvPp
- potential areas: human-machine interaction, automatic focusing in surgical microscopes operation,  predict the expertise of a subject, diagnose a variety of diseases, VR/AR gaming.
  - frequence of eyelid closure--> person's fatigue
  - pupil size--> estimate the cognitive load of  a person in a given task.
  - eye-related information.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210221224646821.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210221224708600.png)

## 1. 开源项目

- https://github.com/SpEm1822/Eye-Trackin-Gaze-Prediction

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210301211659983.png)

## 一、眼动发展现状

**1.眼动追踪融资情况**

2014年末，三星投资了FOVE，主打眼球追踪的VR头盔；

2016年10月，Google收购专注于眼球追踪的初创公司Eyefluence，布局AR交互；

2017年初Facebook旗下Oculus确认收购TheEye Tribe，将眼球追踪技术改善产品；

2017年6月，苹果公司收购了德国计算机视觉公司SensoMotoric Instruments（SMI），打造苹果AR眼镜；

高通和英伟达则选择与七鑫易维合作，专注VR底层的优化；另一方面巨头们也一直源源不断地提供在智能眼镜上眼动追踪技术的实施方案的国际专利。

**2.眼动追踪眼镜硬件情况**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/eye.jpg)

**3.中美眼动追踪技术相关专利情况**

2019年专利情况：在映维网上每隔几天就会公开最新的AR/VR方面给的专利技术，其中包括AMD、Apple、Facebook、Google、Intel、Magic Leap、Microsoft、Nvidia、Oculus、Qualcomm、Sony、Valve这些公司。我们整理出从1月-9月这段时间每一个月与眼动追踪技术相关的技术专利占比的示意图。显而易见，国外的AR/VR的领头羊公司在眼动追踪技术的布局上显得越来越重视，尤其是微软、Magic leap以及Facebook。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200514095527200.png)

映维网公开的国际VR/AR眼动追踪相关专利占比统计，图中眼动相关专利来自微软、谷歌、Facebook等知名公司。

由此可见国内的AR/VR行业的绝大部分相关科技公司并没有在眼球追踪技术方面进行技术的储备，未来中国AR智能眼镜的发展可能受制于人。

![](https://pic4.zhimg.com/80/v2-3522c17e0e61e20e6d6f6ba3cae2f9a3_720w.jpg)

以上来此佰腾专利搜索的统计，2019年以来在美国每个月公开和授权的眼动追踪相关专利都是两位数字，中国都是个位数字，差距近10倍。

## 二、眼动追踪在智能眼镜上的应用 

眼动追踪可以在<font color=red>AR智能眼镜成像显示、交互控制、目标识别、身份验证、健康监测、社交和多人协作</font>多个方面

#### **1.用于光学显示**

​        利用眼动追踪技术使用户更清晰、更流畅的观看AR/VR眼镜显示的影像。包括<font color=red> 注视点渲染；像差校正；影像深度信息；视网膜成像；屈光度检测；亮度调节。</font>

##### **1.1 注视点渲染**

**原因：**为了使人们在使用近眼显示设备时体验到高清的、逼真的、有景深的虚拟画面，对图像计算渲染能力要求是极高的，但是AR/VR智能眼镜的体积、重量、续航能力限制了计算能力。

**解决方案和效果：**利用眼动追踪技术获取眼球的注视中心，<font color=red>将注视点映射到头显的屏幕上或者真实的空间环境中</font>。最终实现人眼视觉中心看哪里，就重点渲染注视点所在的区域，而其他外围区域都以较少分辨率处理（较低的图像质量）。大大降低了处理器的计算能力。注视点渲染也是AR/VR行业内广泛已知的功能，这个技术概念最早是德国SMI提出，也是最早将VR眼镜oculus与眼动追踪技术相结合的。

![](https://pic4.zhimg.com/80/v2-1310823a0ae52711bc08f5ee5d5d5913_720w.jpg)

**注视点渲染原理：**人在看东西时，视觉中心的影像最为高清，这是中央凹视锥细胞处理的光影像。视觉中心外围影像逐渐模糊，这是由于视锥细胞的数目逐渐减少，视杆细胞的数目逐渐增多。所以在近眼显示设备渲染的图像没必要全部高清，视觉中心以外的区域人本身就看不清。

![](https://pic2.zhimg.com/80/v2-d80a0e2ff584cde1cbf1accdcba4db8d_720w.jpg)

眼睛感知影像是通过视细胞接收视网膜上所成的像，视细胞分为<font color=red>视锥细胞和视杆细胞</font>。在人的视网膜内约含有600万～800万个视锥细胞，12000万个视杆细胞，但是相同面积内**视锥细胞**的密度远远大于**视杆细胞**，分布于视网膜的不同部位**，视锥细胞存在于视网膜的黄斑中央凹处**，仅占视网膜很小的一块区域。<font color=red>视锥细胞是感受强光和颜色的细胞，具有高度的分辨能力，光线可直接到达视锥细胞，故此处感光和辨色最敏锐，人的视觉中心能够呈现高清影像就是中央凹视锥细胞的功能</font>。再往外，视杆细胞的数目逐渐增多，视锥细胞的数目逐渐减少。而以视杆细胞为主的视网膜周缘部，则光的分辨率低，色觉不完善，但对暗光敏感，例如家鸡等动物视网膜中视锥细胞较多，故黄昏以后视觉减弱。

##### **1.2 像差校正**

**原因：**目前AR智能眼镜主流采用光波导（光学元件）作为虚拟全息影像的成像媒介，用这种瞳孔扩展的成像方案在显示的过程中会遇到图像畸变的问题，或者该智能眼镜具有针对于近视/远视的屈光度自动调节功能变焦镜片，因此镜片度数的变化也会引起图像的光学畸变，其中其他质量较差的显示光学器件也可能会产生像差（几何和/或色差），从而导致用户观看的图像质量下降。产生这些问题的具体原因如下。**波导镜片导致图像失真：**在堆叠波导显示组件中，存在一系列潜在的现象，这些现象可能导致图像质量产生伪像。这些可能包括重影（多个图像），失真，未对准（颜色或深度之间）以及整个视场的颜色强度变化。另外，在其他类型的条件下可能发生的某些类型的伪像。由于光场显示器的光学器件中的缺陷，当通过光学器件显示时，渲染引擎中的完美三维网格可能变得失真。为了识别和校正预期图像与实际显示图像之间的失真，可以使用显示系统投影校准图案，例如棋盘图案。<font color=red>目前当眼睛直视波导显示器时（眼睛处于波导正前方时），计算机能够有效的校准图像。但对于其他的眼睛姿势、注视方向或位置则校准不太准确。因此，显示器的校准可能取决于眼睛与显示器的相对位置或眼睛方向。如果仅使用单个（例如，基准）位置的校准，则可能存在当佩戴者朝向不同位置（例如，远离基准位置）时未校正的错误。</font>

**解决方案和效果：**利用**眼动追踪技术**实时获取眼睛的注视方向，根据眼睛的注视方向或位置动态的校准的智能眼镜显示的图像。可根据眼睛位置（或在某些情况下的眼睛方向）动态校准虚拟影像的空间位置或颜色。动态校准可以补偿（或校正）显示器的视场中的空间误差和/或彩色（颜色）误差。例如，空间误差可以包括平面内平移，旋转，缩放或扭曲误差以及平面外（例如，焦深）误差。色度误差可以包括可以显示的每种颜色的亮度平坦度或色度均匀性误差（例如，R，G和B）.

##### **1.3 调整图像帧（优化波导的彩虹现象）**

**原因：**当用户的注视方向快速变化时，智能眼镜所呈现的影像可能模糊或者出现不良的颜色（伪影）。通常智能眼镜光波导镜片可能是由3层或者更多层镜片堆叠而成，三种单元色分别被不同的三个波导镜片传导，最终三种单元色在眼睛上合成为有色彩的图像。如果全息虚拟影像的帧速率足够高，并且用户的眼睛没有移动或者相对于显示器上的图像移动相对较慢（例如，视网膜速度相对较慢），则用户在观察智能眼镜的虚拟影像时无法察觉不良的体验。另一方面，如果用户的眼睛相对于显示器上的图像（例如，整个图像的对象或特定部分）相当快速地移动（例如，由眼睛的旋转运动引起的相对较快的视网膜速度），例如<font color=red>眼跳运动、扫视运动、平移头部运动等，用户会察觉到不良的伪影现象（彩虹现象），例如模糊和/或彩虹效果。这种伪影现象是由于所显示的图像的组件原色（例如红色，绿色和蓝色）在不同时间到达观察者的视网膜。如果眼睛没有跟踪图像，则可能发生这种情况</font>。即使在每秒60帧的高帧速率下，来自显示器的红色，绿色和蓝色信息也可以相隔5.5毫秒到达视网膜。眼睛移动得越快，彩虹效果或“图像分裂”就越严重。

因此，由于眼睛视网膜和智能眼镜的影像之间的相对运动过快造成了波导显示的彩虹现象。

**解决方案和效果：**基于<font color=red>眼球移动的速度、加速度来修改用户观看的图像帧的显示时间。通过眼动追踪设备检测到眼球正在快速的平滑的移动时或者扫视某物体时，则智能眼镜显示器提高虚拟影像的帧率。</font>

##### **1.4 影像深度信息**

**原因：**我们已知用户通过AR智能眼镜能够看到叠加在真实世界虚拟的影像，那么如何能够使看到的虚拟物体具有空间的景深感，可以给用户更好视觉体验感。如果此时光学显示器展示的虚拟影像与虚拟深度信息不对应时（例如AR/VR头显的显示屏始终都与我们的眼睛保持固定的距离），这导致了一个名为“视觉辐辏调节冲突”的问题，及人眼可能经历调节冲突，导致不稳定的成像，有害的眼睛疲劳，头痛，同时观察者可能无法在一个瞳孔尺寸处清楚地感知两个不同深度平面的细节。那么如何确定用户此时的视觉深度信息呢？

**解决方案和效果：**在多平面聚焦系统或可变平面聚焦系统中，智能眼镜可以采用眼睛跟踪来确定<font color=red>用户眼睛的聚散度和瞳孔大小，以此确定用户的当前焦点，并将虚拟图像投影到所确定的焦点。深度平面或景深的数量和分布可以基于观察者的眼睛的或注视方向动态地改变</font>。这里所用到的智能眼镜显示器指的为**可变焦光学元件**或者是由多个深度的波导镜片堆叠而成的光学显示器。视觉调节是指弯曲眼睛晶状体以聚焦不同距离下的物体.在现实世界中，为了聚焦近处物体，眼睛的晶状体会弯曲，令物体反射而来的光线到达视网膜上的合适位置，从而让你清晰地看到物体。对于距离较远的物体，光线则以不同的角度进入眼睛，而晶状体必须再次弯曲以确保光线聚焦在视网膜上。所以，如果你将一只手指举到离面部数厘米远，然后闭上一只眼睛并将注意力集中在这只手指，这时手指后面的世界将会变得模糊不清。相反，如果你把注意力集中在手指后面的世界，你的手指将变得模糊不清。这就是视觉调节。

![](https://pic2.zhimg.com/80/v2-7a3b4379e55ef27f2486fab7e53a2df1_720w.jpg)

通常利用双眼视线的汇聚点夹角计算视觉位置深度原理示意图

视觉辐辏（辐辏是指两只眼睛向内旋转以将每只眼睛的视图重叠成一个对齐的图像），这是指两只眼睛向内旋转以将每只眼睛的单独视图“聚合”成一个重叠的图像。对于非常遥远的物体，两只眼睛几乎是平行并列，因为它们之间的距离相较于离物体的距离非常小（这意味着每只眼睛几乎都能看到物体的相同部分）。对于非常靠近的物体，两只眼睛必须向内旋转才能令每只眼睛的视角对齐。对此，你也可以借鉴上面的手指技巧：这一次，用双眼看着面前的手指。请留心，你会注意到手指后面的物体出现了重影。当你将注意力集中在手指后面的物体时，你则会看到手指出现了重影。

##### **1.5 屈光度校正**

**原因：**世界卫生组织报告称，目前全球约有14亿人罹患近视，高达全球人口的18.4%，在中国、美国、日本以及新加坡等地更是平均每两人便有一位近视患者，目前市面上的智能眼镜对这部分人群并不友好，通常情况下需要同时佩戴屈光度校正眼镜和智能眼镜。如果不能提供使近/远视人群一种简易舒适智能眼镜佩戴的方案，则这将会成为智能眼镜面向消费者发展的另外一大阻力。

![](https://pic2.zhimg.com/v2-50d67e086e682d1a3b4a6db4701f42c1_b.jpg)

其中，斯坦福大学的研究人员创造了可以追踪眼球并自动聚焦在您所看事物上的眼镜。在《科学进展》杂志上发表的一篇论文中详细介绍了所谓的自动对焦，它可以证明比过渡镜片或渐进镜片更好。作者指出，随着时间的流逝，随着眼睛中的晶状体变硬，我们在近距离处重新聚焦的能力会变差。这种被称为老花眼的疾病通常会在45岁左右发作，并影响超过10亿人。这是为什么我们许多人需要在中年开始戴老花镜，渐进镜片或单视眼镜的关键因素。

**解决方案和效果：**智能眼镜系统通过光波导镜片动态的射出几种不同视觉深度的图像到眼镜的视网膜上进行成像，眼动追踪系统捕获视网膜上所成的几种不同视觉深度图像的反射光影像，计算机系统可以使用各种图像处理算法来确定患者何时适当地聚焦在图像上，并且随后确定用户的光学屈光度处方。例如图像处理算法包括可以在分析中使用图像质量分析/对比度峰值检测技术。同样，Scheiner双针孔对齐，也可以使用Shack-Hartmann网格对准和/或视网膜反射中和。当测得眼睛的屈光度时，系统可以控制变焦显示透镜投射适应用户眼球屈光度的影像。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200514095843345.png)



##### **1.6 屏幕亮度控制**

**原因：**手机屏幕的显示亮度通过环境光的亮度来调节屏幕的亮度，AR眼镜也会遇到调节屏幕亮度的情况。大多数AR眼镜的光波导镜片投射出来的光与现实周围的光的亮度会直接影响到AR眼镜的体验效果，如果仅仅只是根据周围环境光的强弱来调节智能眼镜成像的光来让用户能够看清楚虚拟的影像，环境光一个维度的标准来调节AR眼镜亮度是远远不够的，利用眼动追踪技术了解眼睛觉得屏幕亮度合适不合适。

**解决方案和效果：**我们需要智能眼镜设备充分了解每一位用户眼睛的差异性、注视需求以及眼睛此时的工作状态，通过<font color=red>眼动追踪技术实时检测眼睛的**瞳孔的位置和大小**、晶状体的状态等，计算和分析这些眼部数据，根据用户眼睛的差异性调节到默认舒适的显示亮度</font>；根据注视需求，<font color=red>判断用户的注意目标，比如当用户注意力在AR眼镜的虚拟影像上时可以适当增加成像亮度，当用户注意力在现实环境中的物体时可以适当降低成像亮度</font>；根据眼睛的工作状态适当调节成像亮度，比如当检测到用户用眼疲劳时，设备能够适当降低成像亮度，降低用眼负担。

![](https://pic1.zhimg.com/80/v2-4677012419514a8749d414561208ef50_720w.jpg)瞳孔对亮度的响应示意图：强光和暗光下瞳孔的大小情况

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200514100017231.png)

​                                                                 

##### **1.7 图像数据无线传输**

**原因：**我们都希望未来的头戴式显示设备（HMD）是轻便、美观、舒适的，例如苹果和Facebook公开的专利一种分体式HMD，为是一个头戴显示设备（AR眼镜）和一个计算终端（例如手机），头戴显示设备不具备运算能力，数据计算过程由计算终端完成，AR眼镜和计算终端通过无线传输数据。然而要让HMD实时显示高保真度的内容，这需要较大的无线传输带宽、功耗以及渲染消耗，这违背了我们的初衷。

**解决方案和效果：**通过结合上述眼动追踪技术，实现注视点图像压缩，极大地降低了传输带宽以及功耗，及<font color=red>看哪里传输哪里的高清图像，类似于注视点渲染</font>。注视点压缩需要实时采集用户注视点数据，将用户视场中心范围图像保持清晰，外围视场的图像进行压缩，将每一帧压缩过后的图像传输至HMD，这几乎将原来所需的带宽降低了三分之二。

高通的《深度数据的异步时间扭曲》专利是通过用户眼睛的姿势数据来生成渲染的帧；谷歌的《集中压缩显示流》专利是压缩用户注视点外围图像；苹果的《基于注视点方向的自适应视频数据预过滤》专利是以不同性能处理用户的注视区域和其他区域。

![](https://pic1.zhimg.com/80/v2-44fd517ebcd1f7a38f3b2a80def9b0a4_720w.jpg)

#### **2.用于交互控制**

**背景和原因：**

目前AR/VR智能眼镜的交互技术主要有<font color=red>手势识别、语音识别、眼动追踪、脑机接口、控制手柄</font>这几种交互技术，其中眼动追踪是其使用过程中最重要的交互方式之一。对比这几种交互方式，眼动追踪交互方式具有<font color=red>移动选择快、隐私性强、消耗体力少、方便快捷、上手快</font>等无可取代的优势。

**手势识别交互技术：**在艺术创作（绘画、制作3D模型）、办公应用、游戏等应用场景中，手势识别具有不错的体验。但是如果让我们在平常的生活场景中，在公众场合张牙舞爪的操作着全息虚拟界面，这不太现实。我们知道绝大多数人不希望在公共场合引起太多人注意，这不仅违反人类的习惯，而且还会造成个人信息的泄露。

**语音识别交互技术：**很难想象在等地铁的时候通过语音识别来对计算机输入信息，这不仅可能会泄露个人隐私，还会对他人造成干扰。对于很多性格内向的人，并不喜欢在公众场合大声说话，语音识别的交互方式对这类人群的用户体验并不好；

监听：语音接口总是监听对话，当不需要的时候，只会在稍后被一个特定的触发词明显激活(例如“hi，Siri”激活苹果助手，但应用程序仍然打开)；

非个人设备：这些设备不是个人设备，任何其他用户都可以有意或无意地向这些设备发送有效的语音输入。目前的语音交互设备作为一种设备的可用性较低，用户不能随时随地使用语音界面，这在固定的专用语音设备(例如-亚马逊Echo)。此外，最佳语音识别(电信设备)需要用户接近设备。

**脑机接口交互技术：**脑机接口是未来最有效的人机交互方式，但是目前的人们对大脑的研究是有限且缓慢的，现在非侵入式脑机接口只能检测大脑皮质层的脑电波和人体神经网络的生物电，因此现在脑机接口交互技术不是很成熟，并不能准确地读取人的思维和想法。

<font color=red>通过注视目标一定时长、双目眨眼、单目眨眼、眼球运动方向（眼球往上看、双面往中心看等）等一系列的眼睛行为与虚拟目标进行交互，交互包括选择确认、点击、翻页滑动、属性弹出</font>。眼睛眨眼、注视一定时长的眼动交互方式让人很不舒服，就好比平时我们用眼睛看东西都是下意识控制的，但是在与电脑系统交互过程中却要将控制眼睛由我的主观意识来控制，这给用户带来了巨大的认知负荷和精力去协调操作界面的交互，时间不久眼睛就很累了。并且眼动交互无法准确地选择过于细小的对象，原因在于眼睛在选择对象的最后关头存在眼颤行为，导致最后眼睛很难选中你注视的目标，因此不能指望通过眼睛来选择细小的对象，也不能驱使眼睛进行高频率的点击行为。最好有交互按钮有磁性功能和眼球的增稳。

**解决方案和效果：**眼珠进行搜索，点击确认

目前智能眼镜的所有的交互方式有手势识别、语音识别、眼动追踪、头动追踪、脑机接口、控制手柄（6DoF）等。每一种交互在特定的应用场景下都有它的优势，那么在注重隐私和简单操作的应用场景下就是眼动交互的优势，但是我相信如果智能眼镜在未来能够取代手机成为下一代计算终端，成为普通消费者都能够简单上手的产品，眼动交互一定是其中较为重要交互方式。

**参考文献：**

论文：激发点击的不同注视时间参数对眼控交互操作绩效的影响_李宏汀

微软专利：三维空间中的对象选择 [http://NO.US](https://link.zhihu.com/?target=http%3A//NO.US) 10268266

Apple：带有眼动追踪的电子设备NO.US10379612

天蝎科技：一种近眼显示设备的眼动追踪交互的方法及系统

天蝎科技：MR智能眼镜内容交互信息输入应用推荐技术的方法



#### **3.用于目标识别**

**背景和原因：**眼睛是心灵的窗户：人所接收到的外界信息有80%来自于眼睛所建立的视觉通道，同时人在进行思维或心理活动时会将其活动过程反映在眼动行为上。可以说，眼动追踪技术是当前科技允许的条件下，“透视”人类思维的最为直观有效的途径。

<font color=red>传统的眼动追踪技术是将眼睛的注视点映射在传统的平面显示上，能做的应用大多是与心理学相关的实验、广告分析、用户体验评估等</font>。但是未来将眼动追踪技术应用<font color=red>在近眼显示设备上（AR智能眼镜），其特点是通过光学元件既可以看到虚拟的全息影像也可以看到真实的世界，用户看到是一个被叠加了虚拟影像的真实世界</font>。如果我们通过眼动追踪技术将眼睛的注视点映射在真实世界，**那么将达到一个很具有想象空间的效果，智能眼镜计算机能够以用户的第一人称视角感知用户的所闻所见。**

但是有以下几个相关因素还需要我们考虑：

• 有时候注视不一定会转化为有意识的认知过程（“视而不见”现象）。例如，盯着屏幕发呆，眼动仪依然会判断你在注视某部分的内容，但实际上你此时并没有相关的心理活动，并且计算机也无法仅通过眼动数据一个维度判断你对正在注视的内容是感兴趣还是疑惑。

• 注视转化的方式可能有所不同，这取决于研究的内容和目的。例如，若是让被试随意浏览某个网站，在网页某个区域注视的次数较多，就可能表明这个人对该区域感兴趣（如某张照片或某个标题），也可能是因为目标区域比较复杂，理解起来比较困难。因此清楚地理解研究目的以及认真仔细地制定测试方案对于眼动追踪结果的阐释很重要。

• 眼动追踪只是提供了我们“透视”人类思维的方法，但和人的真实想法肯定是有差距的，不可唯眼动数据论，结合其他方法，如“有声思维”，访谈等也是十分重要的。

![](https://pic1.zhimg.com/80/v2-a99c00c894d662ae11fa6e1df032372c_720w.jpg)

​                                               计算机将通过眼动追踪知道用户喜欢看什么，对什么感兴趣

**解决方案和效果：**我们可以通过智能眼镜上的眼动追踪系统获取<font color=red>用户眼睛的行为和注视点，智能眼镜的前置摄像头捕获用户视觉前方画面，经过摄像头画面，用户视野的校准匹配使得计算机系统能够以用户的第一人称视角感知用户的所闻所见。当计算机根据心理学理论（瞳孔放大、注视时长等）判断用户对注视物体感兴趣时，眼睛的注视点引导计算机对用户所注视区域的对象进行图像识别，这就实现了通过眼动追踪所引导的图像识别技术</font>，**眼动引导的图像识别**可以使计算机对用户的需求分析的更为精准。当然眼睛所注视的对象可以包括人脸、条形码、宣传海报、户外广告等，通过图像对兴趣识别后可以为用户提供其他相关信息，这些信息可以是文字、图片、三维模型、甚至是应用的推荐。

**例1.电子商务：**当一名女性用户在大街上看到一个人身上穿的衣服非常感兴趣，智能眼镜通过眼动追踪获取眼动行为判断用户正在对目前注视的目标图像感兴趣，对这个人的衣服进行人工智能的图像识别，数据库检索出和感兴趣衣服一模一样的商品或者相似的商品。当然商品不局限于衣服，还可以是数据库记录的任意商品。在一定程度上也可以理解成淘宝的拍立淘功能与眼动追踪结合在智能眼睛上呈现。

视频效果描述：视频中使用的Magic leap one作为应用的载体开发而成，使用手机拍摄的应用演示效果。当把眼睛对准桌子上的大疆无人机时，计算机了解我看了什么，并且把一模一样的无人机产品展示在我面前，用户可以很方便的获取或购买感兴趣的商品，最后通过虹膜识别验证身份，支付成功。（PS：由于是手机对着成像屏幕拍摄的，所以眼动追踪模块并没追踪的眼睛，所以视频效果看着像头动追踪）

**例2.应用推荐和应用的启动:**通过用户此时注视的图像和前后的情景为用户提供智能的推荐。比如，你正在超市挑选蔬菜，在选购过程中你的视觉搜索着各式各样的蔬菜。当看到感兴趣的食材时，你可能会有好多的需求，比如你想知道这个食材可以做那些菜好吃，这时可以推荐食谱的app，比如你想知道这个食材的新鲜度，这时可以推荐你检测食材新鲜度的app，比如你想知道这个食材的营养价值和热量，这时可以食材信息呈现给用户；再比如当你在商业街寻找餐厅，眼睛会看一看餐厅的招牌，这时智能眼镜可以推荐类似于美团或者大众点评的应用推荐。

通过智能眼镜的眼动追踪技术真正的实现“既见即所得”，或许未来有某种可能，眼动追踪可以成为AR智能眼镜移动终端的**流量的入口，**眼动追踪不仅可以第一时间获取用户的需求，而且还可能开创一种全新的用户数据类型-眼动数据，眼动数据可以记录大量用户平时间的感兴趣的物体图像。

**参考文献：**

微软：基于情景的应用程序启动 NO.US10249095

微软：凝视目标应用程序启动 NO.10248192

高通：现实世界中的视觉搜索使用具有增强现实和用户交互跟踪的光学透视头戴式显示器NO.US10372751

高通：用户控制AR智能眼镜的设备和方法 NO.US10359841

天蝎科技：AR智能眼镜的应用基于眼动追踪技术的广告推送方法

NO.201810911203.4



#### **4.用于身份验证：虹膜识别**

**背景和原因：**当我们使用手机时，有大量身份验证和验活的环节。例如，需要通过指纹识别或人脸识别来确认是否是手机用户本人后才可以进入手机的操作界面；当购买商品（银行app）后进行支付验证过程中会通过指纹识别或人脸识别进行生物特征的识别；还有系统中需要用户确认的操作都需要身份的验证操作。

那么智能眼镜想要发展到面向普通消费人群，其系统中的应用也必然需要“身份验证”的操作。作为近眼现实设备-AR智能眼镜，眼睛是最好获取生物特征信息的来源。**虹膜识别**是智能眼镜上目前已知的最佳身份验证方式。

**虹膜识别原理：**虹膜识别技术是基于眼睛中的虹膜进行身份识别，应用于安防设备（如门禁等），以及有高度保密需求的场所。<font color=red>人的眼睛结构由巩膜、虹膜、瞳孔晶状体、视网膜等部分组成</font>。<font color=red>虹膜是位于黑色瞳孔和白色巩膜之间的圆环状部分，其包含有很多相互交错的斑点、细丝、冠状、条纹、隐窝等的细节特征。而且虹膜在胎儿发育阶段形成后，在整个生命历程中将是保持不变的。这些特征决定了虹膜特征的唯一性，同时也决定了身份识别的唯一性</font>。因此，可以将眼睛的虹膜特征作为每个人的身份识别对象。

**解决方案和效果：**全球的眼动追踪技术绝大多数都是通过微型摄像头拍摄被红外光照射的眼球图形进行眼睛运动计算的，因此利用眼动追踪模块摄像头获取虹膜的图像是很顺其自然的事情。除了虹膜识别，还可以从眼睛的其他生理特征进行身份的验证，例如视网膜上毛细血管布局。

![虹膜模板的用户眼睛的示意图](https://pic2.zhimg.com/80/v2-ab74defcec919048596e3a9c8c0353a9_720w.jpg)

#### **5.用于健康检测**

**背景和原因：**智能眼镜还可以应用在健康领域。<font color=red>眼睛的眼底包括眼睛的视网膜，视盘，黄斑，中央凹和后极，眼底的一些病变和眼睛的健康情况可以反映出脑异常、心脏异常、眼癌、高血压等身体疾病。</font>

**解决方案和效果：**将眼科常用检测设备的眼底镜的原理与眼动追踪技术相结合。红外光源被波导镜片或者扫描光纤传导至眼睛生理结构上，进而照亮了眼部的特征，眼睛追踪相机捕获眼睛图像，然后使用模式匹配算法或颜色匹配算法将捕获的图像与指示各种眼睛异常的若干已知图像进行匹配计算。例如，可以分析图像的边缘是否看起来模糊来确定视神经盘是否肿胀。例如1，可以分析图像以测量视神经盘和和视杯的尺寸。视盘和杯子的测量尺寸可用于获得杯盘比，其被计算为视盘的杯部分的直径与视盘的总直径之间的比率。杯与盘比率的较高值可指示青光眼。例2，可以分析图像以确定眼底的颜色。深色底色可指示色素性视网膜炎。相反，在患有动脉闭塞的使用者中可以看到浅色的眼底。例3，可以分析由检眼镜获得的图像以检测其他异常，例如出血或渗出物。绿色滤光器（基本上使红光衰减）可以有利地使得更容易检测出血或渗出物。患有高血压性视网膜病的使用者可以表现出硬性渗出物，出血（很少有乳头水肿）和/或视网膜水肿。例4，一些糖尿病性视网膜病变的使用者可以表现出斑点和印迹出血和/或硬性渗出物，患有糖尿病性视网膜病的一些使用者也可以表现出棉毛斑或软性渗出物。

**参考文献：**

Magic leap专利：诊断眼睛的方法和系统 NO,US10365488

#### **6.用于社交和多人协作：虚拟人像**

**背景和原因：**社交和远程协助、多人游戏也是AR/VR眼镜的重要应用。例如远程会议中，我们通过一个代表自己的虚拟人物来与代表别人的虚拟人物交流。回忆下平时间我们在现实与人的沟通中，大部分时候会看着对方的面部和眼睛来相互沟通，正所谓眼睛是心灵的窗户、眼神交流说的也是这个行为过程。那么在AR/VR世界中，利用虚拟人物进行远程交流时，如果虚拟人物的眼睛是死气沉沉的，不会动，也无法表达惊讶、愤怒、厌恶、微笑等情绪，这让用户体验很不好。如果AR/VR中的面部表情存在非常多的应用，相信这将可以显著改善用户之间交流和交互。我们的目标是令虚拟交互与面对面交互一样自然。我们将其称之为‘社交临场感’。这是一种基于3D的感觉，即使彼此之间相距遥远，但双方依然感觉大家共存于同一个空间，并且能够无缝，轻松地交流各自的想法和情感。”为了在VR中实现这一目标，我们需要可以忠实地再现面部表情，手势和声音的逼真虚拟化身。

![](https://pic4.zhimg.com/80/v2-a6ec02fb4d7f176813ceeda771a2fafb_720w.jpg)

**解决方案和效果：**在面对面的交流的应用程序中，<font color=red> 利用眼动追踪将真人眼睛与虚拟人物的眼睛映射对应，实现眼球的同步运动。同时利用眼动追踪的微型摄像头拍摄眼部的特征，例如巩膜、眼角、眉毛，皱纹，眼睛上的闪烁，雀斑，上角、眼皮的开合程度、眼袋的鼓起程度、，也可以将这些特征绑定虚拟人物上。我们主要是从眼睛区域的特征来推断惊讶，愤怒，厌恶和真正的微笑，哭泣等等，最终使虚拟人物与真人的遍布表情保持一致。</font>

Emteq的解决方案名为FaceTeq，其主要是通过新型传感器模式来检测面部肌肉收缩时产生的微小电气变化。每个面部表情都会在皮肤上产生一定的电活动，而这可以进行非侵入性地检测，无需任何摄像头。

目前Magicleap one、HoloLens2、oculus都相继出台了利用眼动追踪进行眼神和表情捕捉的技术。

**参考文献：**

[MagicLeap独占，Weta多人游戏《Gordbattle》正式发布：]()

[MagicLeap最新LuminOS增加了手掌追踪、多人共享功能 ：]()

[MagicLeap专利欲用眼动追踪摄像头进行面部表情捕捉 ：]()

[Facebook展示『未来虚拟社交』最新研究成果，令AR未来更接近现实]()

[Facebook50年征途：CodecAvatars，创造逼真虚拟角色]()

Magicleap专利：使用眼部注册，头部扫描对准 NO.US20190265783

Magicleap专利 眼动相机的面部表情 [http://NO.US](https://link.zhihu.com/?target=http%3A//NO.US) 20190285881

## 三、眼球生理结构和特征

所有的眼动追踪技术都是根据眼球的生理结构实现的，不同的眼动追踪技术会用到不同的眼球生理特征，因此在了解眼动追踪技术之前有必要先了解一下眼球的生理结构。

人类眼睛能够感知周围环境光线的明暗，主要包括眼球及人眼附属器官。眼球所接收的外界光线通过视神经传送给大脑，大脑对接收信号进行分析，支配人眼附属器官完成眼球的转动，使视线聚焦在目标区域。人眼生理结构如图 1、图2 所示。

![图1 人眼正视图](https://pic1.zhimg.com/80/v2-6c46e5ab8068015d19d274d0083cd1d4_720w.jpg)

![人眼生理结构图](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200514100129261.png)

(1)**巩膜：**是人类眼球白色的外围部分，是保护眼球内部结构的最外层纤维膜。

(2)**角膜：**是人眼最外层的一层透明薄膜，具有十分敏感的光线感知神经末梢，不但可以保护眼球内部，并且具有很强折光能力。角膜具有自己的曲率且不可调节，不同人的角膜曲率各不相同并且**角膜中心是最高点**。图 1 中O2为角膜曲率中心。

(3)**虹膜：**虹膜是位于黑色瞳孔和白色巩膜之间的圆环状部分，其包含有很多相互交错的斑点、细丝、冠状、条纹、隐窝等的细节特征。而且虹膜在胎儿发育阶段形成后，在整个生命历程中将是保持不变的。这些特征决定了虹膜特征的唯一性，同时也决定了身份识别的唯一性。因此，可以将眼睛的虹膜特征作为每个人的身份识别对象。

(4)**瞳孔：**眼睛中心颜色更深的环形部分就是瞳孔，人类可以根据外界不同的光线强度条件反射地通过虹膜肌的运动调节瞳孔大小及形状，进而将外界光线进入瞳孔的数量控制在一定范围内。图 1 中O1 为瞳孔中心。

(5)**晶状体：**晶状体和照相机里的镜头非常相似，是眼球中比较重要的屈光间质之一。晶状体对通过瞳孔进入的光线具有折射作用，并且其形状和曲率可变，在不同距离观测目标物体时使眼球聚光的焦点汇聚到视网膜上。

(6)**视网膜：**视网膜是一个专门负责感光成像的薄膜。由瞳孔进入的光线经过晶状体的折射作用汇聚到视网膜上，视网膜将汇聚的光信号转化为电信号传递给大脑。视网膜的分辨能力是不均匀的，黄斑区域是对光感觉最敏感的部位，其中间有一个小凹为黄斑中心凹，它包含大量的感光细胞。图 1 中P2 为中心凹。

视线追踪系统中，黄斑中心凹 P2 与瞳孔中心O1 的连线称为视轴，即人眼实际注视方向，而实际估计的为角膜中心 O2 与瞳孔中心O1 的连线，称之为光轴。所以，大多数的视线追踪系统需要有一个定标过程，消除人眼视轴与光轴固有的生理偏差，得到真正视线的方向或注视点的位置。

------

## 四、 相关技术

![](https://pic1.zhimg.com/v2-9987e95461f79c78f756dfe2a34ca194_b.jpg)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410121203551.png)

- 基于眼睛视频分析（VOG，Video oculographic）的“非侵入式”技术，其基本原理是：将一束光线（近红外光）和一台摄像机对准被试者的眼睛，通过光线和后端分析来推断被试者注视的方向，摄像机则记录交互的过程。

- 基于视频的眼睛跟踪器，除了监视注视，还可以显示其它有用的测量指标，包括瞳孔大小和眨眼率等。

- 眼睛旋转时，相机传感器上瞳孔中心的位置会改变。但是，（当头部稳定时），角膜反射（CR）的位置相对固定在摄像头传感器上（因为反射源不会相对于摄像头移动）。下图说明了当眼睛向前看，然后旋转到一侧然后再旋转到另一侧时相机所看到的眼图像。如您所见，CR的中心保持在大致相同的位置（就相机像素坐标而言），而瞳孔的中心在移动。

  ![](https://pic4.zhimg.com/80/v2-9f61cc31f3db51be3f6b91f4e944fbbb_720w.jpg)

  

  如果眼睛完全固定在空间中并简单地绕其自身的中心旋转，则仅在摄像机传感器上跟踪瞳孔中心的变化就可以确定注视/凝视的位置。实际上，仅瞳孔跟踪仍可以在某些头戴式或基于“眼镜”的眼动仪中使用，无论头部如何移动，相机和眼睛之间的关系都保持相对固定。

##### 项目一：[通过追踪眼球运动实现 输入交互](https://github.com/despoisj/DeepEyeControl)![](https://miro.medium.com/max/1325/1*d24jbD_j3iTOys3lpgrcnA.png)

![](https://miro.medium.com/max/1775/1*NC08lUA9yc2l6gfThiAEsA.png)

![](https://pic1.zhimg.com/v2-ffc81b16a390f34a962e21ff55590fb4_b.webp)

##### 项目二：基于眼动的显示界面

- 解决眼动交互中的米达斯接触问题，最重要的就是对眼动指标的分析，把用户的真正意图（intention)和无意活动（意的眼跳或是眨眼）分离开来。
- 空间精确性， 系统响应有延迟， 有无反馈![](https://pic1.zhimg.com/80/v2-994f9fb23bb28c736093515f224617ca_720w.jpg)

##### 项目三： 眼动仪

- 眼动仪是一种能够跟踪测量眼球位置及眼球运动信息的一种设备，在视觉系统、心理学、认知语言学的研究中有广泛的应用。![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410120045735.png)

- 指标：

  1、注视：超过100毫秒，认知加工
  2、眼跳：注视点或注视方向发生改变，获取时空信息，无认知加工
  3、追随运动：眼球追随物体移动，有认知加工

- 基础数据：

  - 眼动轨迹：测试出用户的视线在网页上移动的轨迹和关注的重点部位，可以帮助研究者对页面设计进行改进。研究者基于以上眼动仪记录的信息对网页的信息进行了调整，将重要信息放在用户关注点集中的位置。

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410120253799.png)

  - 热点图：热点图主要用来<font color=red>反映用户浏览和注视的情况</font>。红色代表浏览和注视最集中的区域，黄色和绿色代表目光注视较少的区域，可帮助研究者了解界面或产品的哪些特点是最受关注或容易被人忽视的，此外还可以为汇总数据提供视觉参考。热点图可以应用于AB测试，设计师可以分析AB各版的优点与不足，从而进行选定方案及改进。

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410120346062.png)

  - 机场可穿戴式眼动追踪与导向标识系统研究

  - 网页可用性、移动端可用性、软件可用性、视线交互、游戏可用性研究

  - 市场研究与消费者调研（包装设计、购物行为、广告研究）

  - 眼动追踪技术用于自闭症儿童研究，眼病以及大脑和神经障碍的诊断，例如自闭症和帕金森病等。

  - 心理学与神经科学（认知心理学、神经心理学、社会心理学、视觉感知、灵长类动物研究）
    眼动追踪可用于心理学和神经科学的各个不同研究领域，研究眼动行为发生的原因和机制以及我们用眼睛采集信息的方式。

    - 德国柏林自由大学使用眼动追踪验证不同文化群体间的情绪性倾向观点

  - 人的效能研究（体育运动、新手-专家范式、操作员效率评估） 

  - 教育研究（眼动实验室/教室、教学环境研究）:课堂注意力



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: liudongdong1.github.io/eye-tracking-introduce/  
