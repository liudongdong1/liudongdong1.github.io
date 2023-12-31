# StructLight


> 结构光（structured light）是计算机立体视觉技术中的一个重要的分支。它的主要原理是`通过一个投影器投射出一个特定的图案，然后在摄像机所拍摄的图像中识别出这个图案（见图2），通过计算这个图案变形的程度和位置，推断出物体的三维形状（见图1）`。投影器投射出的图案称为结构光系统的编码（codification） 或模式编码（pattern codification）。
>
> 一类是基于光速来进行测量的距离传感器（range-sensors），例如 ToF 技术；另一类是利用三角测量法 （triangulation），称为计算机立体视觉（computer stereo vision）。利用这些技术，可以在不接触物体的情况 下，对物体进行测量和建模 [2]。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325110856.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325164939.png)

### 1. 编码原理

> 结构光系统的关键在于使用恰当的图案 C，使得可以将 π1 上图案 中的点 P1 和 π2 上图像上的点 P2 对应，即让 P1 和 P2 对应的是物体表面上的同一个点 P0，这样的话，就可以通过计算直线 P1F1 和 P2F2 的交点来唯一地确定出 P0 在三维空间中的位置。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325111220.png)

#### 1.1. 直接编码

> 让像平面上的`每个点的颜色信息`可以`直接推断出它所对应的投影平面上的位置`。这种编码称为直接编码。直接编码根据所采用的编码方式，可以分为灰度编码和色彩编码。

`灰度（gray scale）指的是只包含一种颜色的强度信息的图像`，它一般用来`衡量光照的强度`。投影器投射 出一个灰度随着横轴变化的图像，然后摄像机通过测量像平面上每个点的灰度来推测它在投影平面上的横坐标。由于绝对灰度会受到物体离投影器的距离的影响，因此一般基于灰阶编码的结构光系统都采用相对的灰 阶来进行测量。例如，Experiments with the Intensity Ratio Depth Sensor 提出的`基于光照强度比例的深度传感器`（intensity ratio depth sensor），摄像机进 行拍摄时需要在投影器前部放置滤波片：先使用定场滤波片（constant field filter），也就是说，它透过的光 的强度时恒定的；然后使用线性楔形滤波片（linear wedge filter），它透过的光线强度随着 x 轴线性变化。通 过对比这两张图片中的光照强度来推断图形中的点的深度信息。色彩编码和灰度编码的原理基本相同，不同 的是它采用的是不同的色相（hue）而不是灰阶来进行编码 [6]。这样编码的好处在于`色相受距离的影响较小， 因此只需要拍摄一张图片即可完成测量。`

- 缺点：易受环境光纤和噪点影响；

#### 1.2. 时序编码

> 投影器在`不同的时间里投射出一系列不同的编码图案`，摄像机通过观察`每个点在不同时间下被光线照射的情况确定这个点相对于光源的位置`，从而获得这个点的空间坐标。

- 线扫描结构光：比起每次投影一个点，我们`每次投影一条直线`。这种编码方 式称为线扫描结构光。例如，在每个时刻，选取一个特定的 x˜，图案 C 由直线 x1 = ˜x 构成。这样的话，摄 像机就能拍摄到物体表面的一条曲线并且得知这条曲线上每个点对应于 π1 平面上的横坐标。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325111929.png)

- 基于二进制的时序编码方式：投影器在几个不同 的时刻投射出一系列黑白条纹图案，摄像机在每个时刻拍摄一张图像。在每一张图像中，如果一个点被光线 照射到了，那么把这个点标记为 1，否则标记为 0，将不同图像上这个点的标记连接起来，就得到了一个由 0 和 1 组成的序列，这个序列称为这个点的码字（codeword）。二进制编码的原理就是根据码字来推断出这 个点在投影平面上对应的位置。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325111912.png)

#### 1.3. 空间编码

> 空间编码的优势在于它只需要一次成像便可以计算出三维信息，可以满足一些实时性要求较高的场合。 缺点在于它比较容易受到环境光线的干扰；同时，由于它依赖于一个点的邻域，当物体表面起伏很大时，一 个点的邻域可能部分被遮挡住了，在这种情况下测量的精度会受到影响。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325112134.png)

### 2. 三维测量技术

#### 2.1. 不可见结构光

> `红外结构光（InfraRed Structured Light）, 无感知结构光（Imperceptible Structured Light, ISL）和滤波结构光（Filtered Structured Light, FSL）`

- ToF(Time of Light技术)：是测量光线从光源发出到摄像机接收所需要的时间，从而乘以光速计算出物体的距离。等辐波强度调制（continuous wave intensity modulation, CWIM）是一种常见的间接测量方法。，只需要测量混合后产生电信号的强度，就可测量出两个信号之间的相位差，进而测量出距离。g ill 和 g ref 是两个同步的高频电路，g ill 驱动光 源发出光线，光照的强度随着时间而呈现正弦状的周期性变化，变化的频率称为调制频率

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325112531.png)

- RGB双目：非常**依赖纯图像特征匹配**，所以在光照较暗或者过度曝光的情况下效果都非常差，另外如果**被测场景**本身**缺乏纹理**，也**很难**进行**特征提取和匹配**。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325164731.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210821101219712.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325164844.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325112357.png)

#### 2.2. 应用前景

##### 2.2.1. 消费电子产品

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325112711.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210325112811.png)

##### 2.2.2. 医学

> 以利用结构光来扫描或者监测患者的体征，从而诊断患者是否患有某些疾病。

- 种利用结构光技术来扫描人的脸部来定量地诊断面部瘫痪的技术。
- 监测器官的三维运动 情况，因此可以得到更加全面的信息，法研究心脏的运动。

##### 2.2.3. 工业

> 测量`零件的尺寸`，检测`零件的缺陷`，对生 产过程进行质量保障。

- 将结构光和三坐标测量仪（Coordinate Measuring Machine, CMM, 利用探针来获取物体表面高精度三维坐标的装置）相结合的逆向工程流程：先利用结构光获取大致的 表面信息，然后利用这些信息来更好地规划 CMM 的使用，尽可能减少使用 CMM 的次数。
- 即时定位和地图构建（Simultaneous localization and mapping, SLAM）

### 3. TOF 相机

> 在DMS领域，2D平面相机也面临诸多挑战，一是`强光或快速光线变化`，如阳光照射摄像头、林荫大道等场景，2D平面相机会致盲或者反应不过来。二是`算法，深度学习模型越来越大，消耗的算法资源越来越多`，意味着硬件处理器的成本越来越高。三是`准确度，没有深度数据`，或者用深度学习推测的深度数据不仅精度很低且耗费大量运算资源。`2D平面相机通常只能计算眨眼次数，眼帘开合程度这种平面信息，对欧美人种，大眼还能凑合，亚洲人眼小，眯成一条缝很常见，准确度非常低，可能频繁误报`，驾驶员一上车就会关掉DMS。`眼球追踪，头部姿态算法是未来DMS的主流`，用2D平面相机来做，准确度低且消耗运算资源多，成本高。
>
> ToF相机具备一切优势，`包括阳光干扰、光线变化干扰、隐私、有效距离、深度精度、体积方面`。`dToF相机就是Flash激光雷达`，`iToF就是FMCW激光雷达`。其物理重建3D过程包括点云数据生成和点云配准，点云数据生成主要是坐标变换，点云匹配最常见的是ICP算法（ICP算法由Besl and McKay 1992, Method for registration of 3-D shapes文章提出，该算法已经在PCL库中实现）。ICP算法本质上是基于最小二乘法的最优配准方法。通过选择对应两个点云的关系点，然后重复计算最优变换，直到满足正确配准条件。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210821101748184.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210821101820834.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210821101943916.png)

### 4.学习链接

- https://www.huaweicloud.com/articles/b61eb54e90457e932582713f2ceaae20.html

- https://sharzy.in/assets/doc/structured-light.pdf

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/structlight/  

