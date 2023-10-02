# Data Glove Record


> 动作捕捉(Motion capture)，简称动捕(Mocap)，又称运动捕捉。是指记录并处理人或其他物体动作的技术。它广泛应用于军事，娱乐，体育，医疗应用，计算机视觉以及机器人技术等诸多领域。

## 1. 介绍

> ​       在电影制作和电子游戏开发领域，它通常是记录人类演员的动作，并将其转换为数字模型的动作，并生成二维或三维的计算机动画。捕捉面部或手指的细微动作通常被称为表演捕捉(performance capture)。在许多领域，动作捕捉有时也被称为运动跟踪(motion tracking)，但在电影制作和游戏开发领域，运动跟踪通常是指运动匹配(match moving)。

> ​        17个物理惯性传感器每个都包括陀螺仪、加速计和磁力计。它可以感应绕空间3轴的旋转，通过复杂的算法来计算横滚俯仰和航向。通信设备包括传感器输出的数据，并计算四肢相对“主心骨"的位置。同时运用特别的算法来帮助计算出主心骨相对地面的位置。
>
> ​        所有数据将通过无线蓝牙传送到计算机。软件处理并传输数据到3D动画软件如MotionBuilder。所有步骤都在动态中用最小时间间隔完成，真正做到实时的动作捕捉。

## 2. 技术对比

### 2.1. 机械式运动捕捉

>   机械式运动捕捉依靠机械装置来跟踪和测量运动轨迹。
>
>   优点：成本低，精度也较高，可以做到实时测量，还可容许多个角色同时表演。
>
>   缺点：使用起来非常不方便，机械结构对表演者的动作阻碍和限制很大。

### 2.2. 声学式运动捕捉

>   常用的声学式运动捕捉装置由发送器、接收器和处理单元组成。
>
>   优点：装置成本较低。
>
>   缺点：对运动的捕捉有较大延迟和滞后，实时性较差，精度一般不很高，声源和接收器间不能有大的遮挡物体，受噪声和多次反射等干扰较大。由于空气中声波的速度与气压、湿度、温度有关，所以还必须在算法中做出相应的补偿。

###  2.3. 电磁式运动捕捉

>   电磁式运动捕捉系统是比较常用的运动捕捉设备。
>
>   优点：它记录的是六维信息，同时得到空间位置，方向信息。速度快，实时性好，便于排演、调整和修改。装置的定标比较简单，技术较成熟，鲁棒性好，成本相对低廉。
>
>   缺点：对环境要求严格，表演场地附近不能有金属物品，否则会造成电磁场畸变，影响精度。系统的允许表演范围比光学式要小，特别是电缆对表演者的活动限制比较大，对于比较剧烈的运动和表演则不适用。

### 2.4. 光学式运动捕捉

基于摄像机的捕捉系统，是人体动作捕捉曾经的行业标准。标准系统由一组摄像机与数据处理服务器（通常称为HUB）组成，相机发出红外光，在场景中特殊材质做成Maker球上发生反射，从而捕获到场景中Maker点的绝对位置信息。而现在，为了避免场景中其他反光体对捕捉造成的影响，许多公司也会采用主动式的发光体。也就是将Maker点换成主动发出红外光的设备，这样的话：

（1）不会受到场地中反光体影响

（2）光球在不同频率下发出红外光可以被相机捕捉到，从而区分每个主动发光体的编号，增加识别度和精度（理论上，目前还大幅受限于硬件等因素）

> 在人体运动的关键部位粘贴反光标志点，可以对相机发出的红外线光线进行反射，由三维动捕系统捕捉标志点位置，病人走动时，系统可以实时的进行数据采集，并以三维动画的形式进行展示，通过三维坐标数据进行计算分析，可用于三维临床步态分析，临床医生可根据步态报告记录病人步长、步频、步态周期等基本数据以及结合时间参数得出位移、速度、加速度等运动学参数，与正常步态数据进行比对，对病人进行诊断、安排康复训练等。

- `NOKOV动作捕捉分析系统`是一套用于精确捕捉和采集运动物体在三维空间运动数据的高性能设备。采用领先的`被动式光学动作捕捉技术`，基于多目立体视觉测量原理，结合计算机图像处理和先进算法，把机体标志点的动作转化为数据。即将标记点安置在机体的各个部位，由多部近红外高感度摄像机同步实时捕捉运动信息，然后将信息进行数字化处理，得到三维坐标信息，再结合时间参数，从而计算出相应的位移、速度、加速度、角度、角速度、角加速度等运动学参数，实现对机体及其运动状态的定量分析和全方位六自由度的精确测量。完成步态分析周期划分的自动化，可评估生物力学分析数据关节屈伸、内收外展、旋内旋外，上下肢倾斜，左右脚步态数据，以及与正常步态数据的参考对比，为步态分析提供精准数据，以及便捷的操作体验。

### 2.5. 惯性导航式动作捕捉

>   通过惯性导航传感器AHRS(航姿参考系统)、IMU(惯性测量单元)测量表演者运动加速度、方位、倾斜角等特性。
>
>   优点：不受环境干扰影响，不怕遮挡。捕捉精确度高，采样速度高，达到每秒1000次或更高。由于采用高集成芯片、模块，体积小、尺寸小，重量轻，性价比高。惯导传感器佩戴在表演者头上，或通过17个传感器组成数据服穿戴，通过USB线、蓝牙、2.4Gzh DSSS无线等与主机相联，分别可以跟踪头部、全身动作，实时显示完整的动作。

## 3. 应用领域

> **动画制作**：表演动画技术将会得到越来越广泛的应用，而运动捕捉技术作为表演动画系统不可缺少的、最关键的部分，必然显示出更加重要的地位。

> **提供新的人机交互手段**： 表情和动作是人类情绪、愿望的重要表达形式，运动捕捉技术完成了将表情和动作数字化的工作，提供了新的人机交互手段。 比传统的键盘、鼠标更直接方便，不仅可以实现“三维鼠标”和“手势识别”，还使操作者能以自然的动作和表情直接控制计算机，并为最终实现可以理解人类表情、动作的计算机系统和机器人提供了技术基础。

>   **虚拟现实系统**： 为实现人与虚拟环境及系统的交互，必须确定参与者的头部、手、身体等的位置与方向，准确地跟踪测量参与者的动作，将这些动作实时检测出来，以便将这些数据反馈给显示和控制系统。这些工作对虚拟现实系统是必不可少的，这也正是运动捕捉技术的研究内容。

>   **机器人遥控**： 机器人将危险环境的信息传送给控制者，控制者根据信息做出各种动作，运动捕捉系统将动作捕捉下来，实时传送给机器人并控制其完成同样的动作。与传统相比，这种系统可以实现更为直观、细致、复杂、灵活而快速的动作控制，大大提高机器人应付复杂情况的能力。在当前机器人全自主控制尚未成熟的情况下，这一技术有着特别重要的意义。

>   **互动式游戏**： 可利用运动捕捉技术捕捉游戏者的各种动作，用以驱动游戏环境中角色的动作，给游戏者以一种全新的参与感受，加强游戏的真实感和互动性。

>   **体育训练**： 运动捕捉技术可以捕捉运动员的动作，便于进行量化分析，结合人体生理学、物理学原理，研究改进的方法，使体育训练摆脱纯粹的依靠经验的状态，进入理论化、数字化的时代。还可以把成绩差的运动员的动作捕捉下来，将其与优秀运动员的动作进行对比分析，从而帮助其训练。另外，在人体工程学研究、模拟训练、生物力学研究等领域，运动捕捉技术同样大有可为。

## 4. 公司

### 4.1. 北京诺亦腾

- **入门级产品 Perception Neuron 2.0**   2w
- **高级无线动捕产品 Perception Neuron PRO** 4w

> 是一款基于惯性传感器的高级无线动作捕捉系统，具有全身无线数据传输、低延迟高精度、高电磁耐受性等多种特性。凭借其专有的嵌入式数据融合系统、人体动力学系统和物理引擎算法，可以捕捉大型动态运动，为使用者提供平滑而准确的动作捕捉数据。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907084835.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907084912.png)

- [**专业级高精度光惯混合动作捕捉系统 Perception Neuron Studio**](https://shopcdn.noitom.com.cn/html/190.html#top)

> 效易用。在纯惯性模式下，Perception NeuronTM

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907084445.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907084515.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907084128.png)

其中**PNS手套**： 每一只手套内置6枚高精度惯性传感器，分别置于手部关节点文职，可精确捕捉手部完整动作。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907084424.png)

- [Noitom Hi5 VR手套](https://shop.noitom.com.cn/product/list/85.jhtml)
  - 9-DOF IMU for 5 fingers and the back of hand 
  - Vibration feedback for each Glove 
  - Supply voltage range of 1.0-1.5VDC with one AA battery for each Glove 
  -  Supply voltage range of 5±0.25VDC for Dongle 
  - 7 hours of working time with 2100mAh Alkaline battery 
  - Latency less than 5 ms (From motion to SDK, under clean RF condition) 
  -  Output date rate up to 180Hz 
  - RF working area: 5m×5m (open area without interference) 
  - automatic channel-switching to avoid RF interference
  - [vivi tracker](https://shopcdn.noitom.com.cn/html/137.html)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907091312.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907093639.png)

- position, rotation;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907094023.png)

### 4.2. [VRTRIX](http://vrtrix.com.cn/product/data-gloves/)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200907090150.png)

## 5. 手语识别发展历程

**2013年7月，**微软团队和中国科学院计算技术研究所进行合作，通过Kinect For Windows创建手语识别软件，可根据手语动作的追踪识别转换成正常人能够读懂的内容；

**2018年2月，**中科大发布了一篇手语识别的论文被人工智能顶级学术会议AAAI 2018收录；该论文提出一种新型连续手语识别框架 LS-HAN，无需时间分割；

**2018年3月，**Magic Leap的头戴式设备识别手语和文本“感官眼镜”，据3月新专利申请，相关信息概述了使用头戴式设备检测和翻译手语的方法，并介绍了如何识别标牌和店面上的文字；

**2018年7月，**软件开发者 Abhishek Singh演示了一款能够理解手语手势的 MOD，通过摄像头的捕捉和深度学习，让亚马逊 Alexa 对手语手势作出反馈；

**2018年12月，**爱奇艺研发的AI手语主播在中国网络视听大会上首次亮相，可识别用户语音并转换为文字，还能对健听人自然语言进行理解，并智能翻译为手语表达。

**2019年5月**，腾讯优图实验室联合深圳市信息无障碍研究会发布“优图AI手语翻译机”。据官方资料显示，用户通过面对翻译机摄像头进行手语表达，翻译机屏幕界面便能快速把手语转换为文字。

> 利用图像或者是深度信息，双目摄像头或者是深度摄像头。利用手套等可穿戴设备，测量由于关节运动导致的物理变化量。还有基于生理信号，测量肌肉收缩时，肌肉两端产生的电压信号，再用深度学习识别意图。还有一些人利用IMU做的数据手套，或则是机械外骨骼，测量连杆的运动角度，解算出手部的姿态。

## 6.相关案例

### 6.1. 微软亚洲研究院手腕捕捉设备

> 包括了红外感应器、近红外线激光器和惯性测量装置，从而可以跟踪 5 个手指的动作，包括弯曲的幅度，关节的细节变化等。
>
> 成品：https://www.microsoft.com/en-us/research/video/haptic-pivot-on-demand-handhelds-in-vr-2/

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606195020773.png)

### 6.2. ShapeHandPlus

> ​	由ShapeHand捕捉系统与手臂跟踪系统ShapeTape组合而成，可跟踪整个手和手臂的动作和姿态，包括旋转和偏移等。而ShapeHand本身也是一款无线便携式轻型手部动作捕捉系统，用于捕获手和手指的动作。虚拟现实，动画人物手捕捉，MRI研究，动作识别，机器人设计，动作分析，3D输入，手语。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606195111690.png)

### 6.3. 动作捕捉全身

[Perception Neuron Pro惯性无线全身动作捕捉系统传感器](https://detail.tmall.com/item.htm?spm=a1z10.5-b-s.w4011-21214616185.233.7ec436d3pBMVYJ&id=586316188617&rn=5242faf4e90ade70dfadaf80da5f95f6&abbucket=15)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200624213929633.png)

## 7. 相关Sensor

### 7.0. [Leap Motion](https://detail.tmall.com/item.htm?spm=a230r.1.14.1.745131e9fwwaXv&id=586420620094&ns=1&abbucket=3&skuId=4107812332163)  （[MV](https://www.bilibili.com/video/BV1zx411E7sS?p=2))

> 手部正反判定比较困难，容易进行误判；受光的影响比较大，包括室外可见光、激光相机自己发出的激光、捕捉相机自身识别干扰等；识别范围有限，受光路限制，对障碍的容忍度较低，双手叠交的识别判定有误。因此基于计算机视觉的手势识别设备的识别效率相比于数据手套的低。

> [Leap Motion](https://developer.leapmotion.com/setup/desktop)是一种检测和跟踪hands, fingers and finger-like tools的设备。该设备在一个较近的环境中操作，精度高，跟踪帧速率高。Leap Motion 视野是集中在设备上方的一个倒置的金字塔。Leap Motion检测的有效范围是约25毫米至600毫米（1英寸到2英尺）。可以识别出四种特定的动作: Circle，Swipe，Key Taps，Screen Taps; 通过持续跟踪动作流，Leap Motion还可以将一个区域内的动作理解为三种基本元素：scaling, translation, and rotation。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200624210009611.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200624214110152.png)

#### 7.0.1.  **动作跟踪数据**

> leap motion可以跟踪手，手指，和一些小工具，并以帧的形式更新跟踪数据。每一帧包括跟踪对象的列表，和描述对象动作的特征。每检测到一个对象leap motion就自动给它分配一个唯一的ID，直到对象移动出检测区域，重新进入检测区域的对象会重新分配ID。
>
> leap motion靠形状识别手状物体，工具指比手更长、更细或者更直的物体（图5）。在leap motion模型中，手指和工具被抽象为pointable对象。其物理属性包括：length长度。可见部分长度 ;  width宽度。可见部分平均宽度
> direction方向。物体的单位向量，例如从指根到指尖，图6;tip position指尖的位置。指尖相对leap motion原点的位置，单位mm;tip velocity指尖的速度。单位mm/s手的动作包括：平移，旋转，缩放等

> <font color=red>手模型可以提供位置、特征、动作，以及和手关联的手指、工具等信息。对手的模型leap motion API提供了尽可能多的信息，但并不是每一帧都能完全检测到这些属性。例如握拳时，手指不可见</font>，所以手指的列表就可能为空，编码时要注意到这些情况。leap motion并不区分左右手，hand列表也可以包含超过2只手，但是超出两只手时会影响跟踪效果。
>
> 手的属性包括：
>
> - palm position手掌位置，手掌中心位置距leap motion原点的距离，单位毫米
> - palm velocity手掌速度，单位mm/s
> - palm normal手掌法向量，由掌心向下指向外部
> - direction方向，掌心指向手指的向量
> - sphere center球心，根据手的曲线拟合出的球的球心
> - sphere radius球半径，拟合球的半径
> - API 提供动作有： SreenTapGesture; KeyTapGesture;SwipeGesture;CircleGesture.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200624212951397.png)

> leap motion首次识别出一个手势后将其加入帧，如果这是一个持续性动作，leap motion将一个更新的手势对象加入后续帧。画圆和挥扫是持续性动作，leap motion在每一帧中更新这些手势，tap轻击是不连续的动作，所以每次敲击只需一个手势对象。
>
> 每一个帧的实例都包括跟踪数据，手势和动作因子（factor）等。leap motion通过分析当前帧动作与之前帧动作的变化，将动作翻译成平移、旋转、缩放等动作因子。

### 7.1.  大鸟智能虚拟手套

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108091204926.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108091223036.png)

### 7.2. 基于Flex Sensor IMU手套

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108091446439.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108091525951.png)

### 7.3. Flex Sensor

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108092208496.png)

### 7.4. 2D-BF系列薄膜弯曲传感器

> ​		BF系列薄膜弯曲传感器基本功能是可以对物体弯曲度进行测试，可用于多种应用环境和场景；传感器为电阻式，当感应的弯曲度发生变化，传感器的电阻也随之发生变化。主要用于弯曲和弯曲身体运动装置诸如机器人手指弯曲、竞技游戏虚拟动作、医疗设备、计算机外设、乐器等领域。原理简单、使用方便，可直接采集弯曲传感器电阻值或经过模块转换成标准信号电压数据进行采集。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108095332518.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108092306247.png)



### 7.5. BendLabs[柔性弯曲角度传感器](http://www.cnbytec.com/productshow.asp?id=211)

> BEND LABS的柔性传感器由医用级有机硅构成的，可以满足精确，多轴，柔软，灵活的弯曲角度传感需求，并且具有高精度、低功耗、无漂移等优点。柔性传感器采用差分电容原理，是一种角度输出与弯曲路径无关的创新传感技术。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108100434645.png)

### 7.6. Bebop[数据手套传感器](http://www.cnbytec.com/productshow.asp?id=237)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108100117268.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108100237698.png)

### 7.7. Pickit

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108095819117.png)

### 7.8. 光纤传感器

> 应用领域： [高电压环境物理量测量；有电磁干扰环境物理量测量；宇航，核电项目研究，参量测量；有害有毒环境物理量测量；医疗设备；大型建筑结构；风电，机车等。](http://www.cnbytec.com/productshow.asp?id=232)

### 7.9.[9Axis传感器](https://wiki.wit-motion.com/lib/exe/fetch.php?media=module:wt901:docs:jy901%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E%E4%B9%A6v4.0.pdf)

> 模块集成高精度的陀螺仪、加速度计、地磁场传感器，采用高性能的微处理器和先进 的动力学解算与卡尔曼动态滤波算法，能够快速求解出模块当前的实时运动姿态。

大小：体积：15.24mm X 15.24mm X 2mm

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108184307876.png)



### 7.10. 智能穿戴数据手套

![image-20200108093636846](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108093636846.png)

- [UNITY3D骨骼动画数据手套](http://v.youku.com/v_show/id_XMzgwNDU5ODcy.html?from=y1.7-2 )
- [数据手套演示代码[wiseglove.com]](http://v.youku.com/v_show/id_XMzgwMzc1MzQw.html?from=y1.7-2 )
- [WiseGlove数据手套的UNITY3D演示和程序](http://v.youku.com/v_show/id_XMTM4NjIxOTQyMA==.html?from=y1.7-1.2 )
- [5传感器WISEGLOVE数据手套的传感器心电图波](http://v.youku.com/v_show/id_XMTM3MDgzNTcyOA==.html?from=y1.7-1.2 )
- [5传感器WISEGLOVE数据手套的骨骼动画虚拟手演示](http://v.youku.com/v_show/id_XMTM3MDgzMDQyNA==.html?from=y1.7-1.2 ) 
- [视频: 虚拟装配数据手套实现手臂跟踪，OPENGL 实时显示](http://v.youku.com/v_show/id_XNzEyNjU0ODg4.html?from=y1.7-1.2 )

## 10. 其他新型传感器

### 10.1. MIT 手套压力手套

> 装了 548 个传感器，戴上它就能测到物体、物体的重量等，而且仅需 10 美元成本

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108165030828.png" alt="image-20200108165030828" style="zoom:50%;" />

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108165158012.png)

### 10.2. Google Soli

> 运用微型雷达监测空中手势动作的新型传感技术。这种特殊设计的雷达传感器可以追踪亚毫米精准度的高速运动，然后将雷达信号进行各种处理之后，识别成一系列通用的交互手势，方便控制各种可穿戴和微型设备。

- 微软Kinect为代表的深度感应技术（结构光和飞行时间两种）
- LeapMotion为代表的红外线投影与成像
- uSense为代表的光学立体成像技术

相比这几种常见的解决方案，采用毫米波雷达的Soli技术有以下几种优点：

- 依赖<font color=red>红外线的深度感应和投影技术在室外红外线干扰多的环境可靠性很差</font>，毫米波雷达则无这方面问题。
- 基于光学立体成像的技术需要相当的计算量获取深度数据，高的分辨率较难实现，功耗不低。同时由于依赖可见光，在低光亮环境无法使用，毫米雷达波也无这方面问题。
- 同时因为毫米波雷达的频率远低于红外线和可见光，相比基于红外线的时间飞行技术，毫米波雷达可以计算相移（Phase shift）和多普勒效应（Doppler Effect），从而以很低的计算量获取物体的运动与方向。
- 毫米波雷达对于一些材料还有很好的穿透性，不受光路遮挡的影响。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108135045567.png)

**目前应用：**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606200433588.png)

毫米波雷达对部分材质有一定的穿透作用，反射信号也有一些差别，有开发者据此设计出材质探测器，不仅可以塑料和多种金属，还能识别水跟牛奶。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606200449526.png)

车内手势是一项有意思的应用，相比可见光和红外光的技术，Soli可以完美解决不同光照环境下传统技术不稳定的问题，而且雷达对捕捉细微运动有很强的优势。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606200509593.png)

Hover在屏幕上的手指现在也可以精确的捕捉和预测了。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606201450732.png)

Soli可以扫描和计算出深度图，为何不拿来做成3D扫描与成像的应用

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606201620067.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200606201634228.png)

1. 捕捉原始反射信号
2. 将接收的时序信号处理和转换到Range Doppler Map
3. 特征提取，识别，定位与追踪
4. 从提取的特征实现手势识别

### 10.3.光纤传感器

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132310530.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132412661.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132448860.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132504727.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132533125.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132558023.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132641409.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132658145.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132716952.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132234631.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200108132212222.png)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/data-glove-record/  

