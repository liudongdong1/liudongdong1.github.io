# LeapMotionRelative


## 1. [Leap Motion](https://detail.tmall.com/item.htm?spm=a230r.1.14.1.745131e9fwwaXv&id=586420620094&ns=1&abbucket=3&skuId=4107812332163) Introduce

> [Leap Motion](https://developer.leapmotion.com/setup/desktop)是一种检测和跟踪hands, fingers and finger-like tools的设备。该设备在一个较近的环境中操作，精度高，跟踪帧速率高。Leap Motion 视野是集中在设备上方的一个倒置的金字塔。Leap Motion检测的有效范围是约25毫米至600毫米（1英寸到2英尺）。可以识别出四种特定的动作: Circle，Swipe，Key Taps，Screen Taps; 通过持续跟踪动作流，Leap Motion还可以将一个区域内的动作理解为三种基本元素：scaling, translation, and rotation。

> Three Infrared Light emmitters and two cameras which received the IR lights.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803183730104.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624210009611.png)

![image-20200624214110152](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624214110152.png)

### 1.2.  **动作跟踪数据**

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

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803183941605.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200624212951397.png)

> leap motion首次识别出一个手势后将其加入帧，如果这是一个持续性动作，leap motion将一个更新的手势对象加入后续帧。画圆和挥扫是持续性动作，leap motion在每一帧中更新这些手势，tap轻击是不连续的动作，所以每次敲击只需一个手势对象。
>
> 每一个帧的实例都包括跟踪数据，手势和动作因子（factor）等。leap motion通过分析当前帧动作与之前帧动作的变化，将动作翻译成平移、旋转、缩放等动作因子。

## 2. Relative Paper

**level**: 
**author**: Lin Shao*(Stanford EE 267)
**date**: 2016
**keyword**:

- Leap Motion

Shao, Lin. "Hand movement and gesture recognition using Leap Motion Controller." *Virtual Reality, Course Report* (2016).

------

### Paper: Hand_Mov&Gestreu_Rec

<div align=center>
<br/>
<b>Hand movement and gesture recognition using Leap Motion Controller
</b>
</div>

#### Summary

1. Recognise hand movement and gestures accuracy when no occlusion happens.

#### Research Objective

  - **Application Area**: stoke rehabilitation, hand gesture recognition

#### Methods

- **system overview**:

【**Features**】 

- **Static Gestured Features:**   relative distance between palm and fingers.
  - $D_i$: distances between fingertips $F_{pos}^i$ and palm center $P_pos$;
  - distance between two fingers which are adjacent.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803200943810.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803201534310.png)

- **Dynamic Gesture:** using the velocity of fingertips and palm to detect the movement patterns.

  - global movement: to detect hand translation movement, hand rotation movement, hand circle movement
  - details of the fingers' movement: focus on the movement of index finger.

  > calculate the total value of velocity magnitude among fingers and palm. If the total movement value is greater than a user-defined threshold, we believe the hand is moving.

**Hand Translation Feature:** fingers and palm are moving together straightly without rotation. <font color=red> calculate the cross correlation of velocity vectors between fingers $F_v^i$ and palm $P_v$ for all figners.</font>

**Hand Rotation Feature:** 1). difference of current palm normal $P_N^t$ and previous palm normal $P_N^{t-1}$ defined by $DP_N$. 2).the angle between difference of current palm $DP_N$ and hand direction $P_D$, and calculate the cross correlation of $DP_N$ and hand direction $P_D$.

**Hand Circle Features:** indicates the palm is drawing a great circle. <font color=red>calculate the first order difference between palm normals</font>

**Index Key Tapping and Index Swipe:**  <font color=red> calculate the cross correlation between direction of index finger velocity $F_v^1$ and the palm normal $P_N$. By considering the absolute cross correlation  with threshold.</font>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803204623569.png)

**Index Circling Direction Features:** <font color=red>predict the circle direction whether it is clockwise or counter clockwise when the index is moving along a circle</font>. calculate the first order difference of the index finger velocity between $F_{vt}^1$ and $F_{v(t-1)}^1$denoted by  $DF_{vt}^1$  denoted by CP. 通过三角形内积。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803205125973.png)

> - When important fingers of regions are self-occluded by other hand parts, tracking data quality will be greatly reduced.
> - **Detection Region:**  tracking data becomes unstable when hands are near the region boundaries.
> - **Parameters:** if the hand sizes and corresponding parameters are not matching, failures cases happen
> - **Error accumulation:**  for hand movement gestures, using the first order differences cause errors.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803205202860.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803205358787.png)

#### Evaluation

  - **Environment**:   

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803210006896.png)

#### Notes <font color=orange>去加强了解</font>

  - <font color=red> for **Self occlusion**:  using two leap motion controllers to be put far from each others with nearly orthogonal angle.</font>

**level**: 2017 IEEE World Haptics Conference (WHC) 
**author**: Inwook Hwang
**date**: 2017
**keyword**:

- Leap motion, HCI

------

### Paper: AirPiano

<div align=center>
<br/>
<b>AirPiano: Enhancing Music Playing Experience in Virtual Reality with Mid-Air Haptic Feedback
</b>
</div>

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803210601000.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803210629531.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803210655529.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200803210715223.png)

**level**: IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS
**author**: Hua Li, Member, IEEE
**date**: 2020
**keyword**:

- leap motion, hand gesture.

------

### Paper: Hand Gesture Recognition

<div align=center>
<br/>
<b>Hand Gesture Recognition Enhancement Based on Spatial Fuzzy Matching in Leap Motion</b>
</div>

#### Summary

1. presented a spatial fuzzy matching algorithm by matching and fusing spatial information to construct a fused gesture datasets.
2. For dynamic hand recognition, an initial frame correction strategy based on SFM is proposed to fast initialize the trajectory of test gesture with respect to the gesture dataset.
3. Experiment results show the system recognizes static hand gesture at recognition rates of 94%-100%, over 90% of gynamic gesture.


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/leapmotionrelative/  

