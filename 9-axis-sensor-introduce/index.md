# 9_Axis Sensor Introduce


## 0. 前言

> 3轴传感器指的是3轴的加速度，根据这个加速度我们解算出X Y两轴的角度；6轴传感器指的是3轴的加速度和3轴角速度，根据这两个数据我们解算出X Y Z三轴的角度(Z轴是角速度积分解算，所以存在累计误差)；9轴传感器指的是3轴的加速度、3轴角速度和3轴磁场，根据这三个数据我们解算出X Y Z三轴的角度(Z轴是磁场解算，相当于电子罗盘，但受磁场干扰的影响)；10轴传感器指的是3轴的加速度、3轴角速度、和3轴磁场气压，功能比9轴传感器多了气压和高度

## 1. 九轴传感器

9轴传感器包括3轴加速度计、3轴陀螺仪、3轴磁力计，在实际应用中，需要把这些数据需要经过融合算法后，才能够被应用程序使用，下面对每种传感器功能、原理以及融合算法进行介绍。

### 1.1 加速度计

人们常说的G-sensor，用来检测物理在X、Y、Z轴上的重力加速度，单位:m/s^2.

以手机为例，X、Y、Z轴如下图所示（右手坐标系）：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200428120734264.png)

当手机平放在桌面时，Z轴指向天空，这时候X、Y轴的数值接近为0，Z轴的重力加速度约为9.81，将手机翻转后，即屏幕面朝向桌面，此时的Z轴重力加速度约为-9.81。

X、Y轴指向天空时，与上面Z轴同理，有兴趣的可以在手机上安装一个”sensor_list.apk”来抓取这些数据。

> 原理：[英文url](http://www.starlino.com/imu_guide.html)

当我们在想象一个加速度计的时候我们可以把它想作一个圆球在一个方盒子中。

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200428121529543.png" alt="image-20200428121529543" style="zoom:50%;" />



<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200428121627605.png" alt="image-20200428121627605" style="zoom:50%;" />

- $R^2=R_x^2+R_y^2+R_z^2$


- 数字加速度计可通过I2C，SPI或USART方式获取信息，
- 模拟加速度计的输出是一个在预定范围内的电压值，你需要用ADC（模拟量转数字量）模块将其转换为数字值。

现在我们得到了惯性力矢量的三个分量，如果设备除了重力外不受任何外力影响，那我们就可以认为这个方向就是重力矢量的方向。如果你想计算设备相对于地面的倾角，可以计算这个矢量和Z轴之间的夹角。如果你对每个轴的倾角都感兴趣，你可以把这个结果分为两个分量：X轴、Y轴倾角，这可以通过计算重力矢量和X、Y轴的夹角得到

<img src="https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20200428122029289.png" alt="image-20200428122029289" style="zoom:50%;" />

- $cos(Axr) = Rx / R $
- $cos(Ayr) = Ry / R$
- $cos(Azr) = Rz / R$

从公式1我们可以推导出$ R = SQRT( Rx^2 + Ry^2 + Rz^2)$

> 常用加速度传感器：加速度计种类繁多，MMA、LSM、MPU、BMA等系列，如：MMA7460、MMA8452、MPU6050（A+G）、MPU6800(A+G)、LSM6DSL(A+G)、IMC20603(A+G)、MPU9150（A+G+M）

> 使用场景：加速度计通过一定的算法，就可以做成我们常用的功能，如：计步器、拍照防抖、GPS补偿、跌落保护、图像旋转、游戏控制器等。

### 1.2 陀螺仪

通常称为Gyro-sensor，用来测量在X、Y、Z轴上的旋转速率，单位:rad/s。
以手机为例，将手机平放桌面，屏幕朝上，以逆时针方向旋转手机，获得到的是Z轴的加速度值。

> 原理：陀螺仪，是一种用来感测与维持方向的装置，基于角动量不灭的理论设计出来的。陀螺仪一旦开始旋转，由于轮子的角动量，陀螺仪有抗拒方向改变的趋向。一个旋转物体的旋转轴所指的方向在不受外力影响时，是不会改变的。大家如果玩过陀螺就会知道，旋转的陀螺遇到外力时，它的轴的方向是不会随着外力的方向发生改变的。我们骑自行车其实也是利用了这个原理。轮子转得越快越不容易倒，因为车轴有一股保持水平的力量

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200522165644900.png)

、1、陀螺转子（常采用同步电机、磁滞电机、三相交流电机等拖动方法来使陀螺转子绕自转轴高速旋转，并见其转速近似为常值）。

2、内、外框架（或称内、外环，它是使陀螺自转轴获得所需角转动自由度的结构）。

3、附件（是指力矩马达、信号传感器等）。

陀螺仪的两个重要特性

- 一为定轴性
- 另一是进动性，这两种特性都是建立在角动量守恒的原则下。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200522165805551.png)

定轴性： 当陀螺转子以高速旋转时，在没有任何外力矩作用在陀螺仪上时，陀螺仪的自转轴在惯性空间中的指向保持稳定不变，即指向一个固定的方向；同时反抗任何改变转子轴向的力量。这种物理现象称为陀螺仪的定轴性或稳定性。

其稳定性随以下的物理量而改变：

​		1、转子的转动惯量愈大，稳定性愈好；

​		2、转子角速度愈大，稳定性愈好。

> 所谓的“转动惯量”，是描述刚体在转动中的惯性大小的物理量。当以相同的力矩分别作用于两个绕定轴转动的不同刚体时，它们所获得的角速度一般是不一样的，转动惯量大的刚体所获得的角速度小，也就是保持原有转动状态的惯性大；反之，转动惯量小的刚体所获得的角速度大，也就是保持原有转动状态的惯性小。

进动性： 当转子高速旋转时，若外力矩作用于外环轴，陀螺仪将绕内环轴转动；若外力矩作用于内环轴，陀螺仪将绕外环轴转动。其转动角速度方向与外力矩作用方向互相垂直。这种特性，叫做陀螺仪的进动性。

进动性的大小有三个影响的因素：

​		1、外界作用力愈大，其进动角速度也愈大；

​		2、转子的转动惯量愈大，进动角速度愈小；

​		3、转子的角速度愈大，进动角速度愈小。

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200428122341115.png" alt="image-20200428122341115" style="zoom:50%;" />

陀螺仪的每个通道检测一个轴的旋转。例如，一个2轴陀螺仪检测绕X和Y轴的旋转。为了用数字来表达这些旋转，我们先引进一些符号。首先我们定义：

- $Rxz$ – 惯性力矢量R在XZ平面上的投影
- $Ryz$ – 惯性力矢量R在YZ平面的上投影

在由Rxz和Rz组成的直角三角形中，运用勾股定理可得：

- $Rxz^2 = Rx^2 + Rz^2$ ，同样：
- $Ryz^2 = Ry^2 + Rz^2$

相反，我们按如下方法定义Z轴和Rxz、Ryz向量所成的夹角：

- $AXZ - Rxz$（矢量R在XZ平面的投影）和Z轴所成的夹角
- $AYZ - Ryz$（矢量R在YZ平面的投影）和Z轴所成夹角

陀螺仪测量上面定义的角度的变化率。假设在$t0$时刻，我们已测得绕Y轴旋转的角度（也就是$Axz$），定义为$Axz0$，之后在t1时刻我们再次测量这个角度，得到$Axz1$。角度变化率按下面方法计算：

- $RateAxz = (Axz1 – Axz0) / (t1 – t0)$.

换句话说设备绕Y轴（也可以说在XZ平面内）以306°/s速度和绕X轴（或者说YZ平面内）以-94°/s的速度旋转。请注意，负号表示该设备朝着反方向旋转。按照惯例，一个方向的旋转是正值。一份好的陀螺仪说明书会告诉你哪个方向是正的，否则你就要自己测试出哪个旋转方向会使得输出脚电压增加。最好使用示波器进行测试，因为一旦你停止了旋转，电压就会掉回零速率水平。如果你使用的是万用表，你得保持一定的旋转速度几秒钟并同时比较电压值和零速率电压值。如果值大于零速率电压值那说明这个旋转方向是正向。

> 常用陀螺仪传感器：目前市面上较多的都是二合一模块（加速度+陀螺仪），如：MPU6050（A+G）、MPU6800(A+G)、LSM6DSL(A+G)、IMC20603(A+G)、MPU9150（A+G+M）。

> 使用场景：航海、航空、游戏、拍照防抖、控制等。

### 1.3 磁力计

> **原理**:	如下图所示，地球的磁场象一个条形磁体一样由磁南极指向磁北极。在磁极点处磁场和当地的水平面垂直，在赤道磁场和当地的水平面平行，所以在北半球磁场方向倾斜指向地面。用来衡量磁感应强度大小的单位是Tesla或者Gauss（1Tesla=10000Gauss）。随着地理位置的不同，通常地磁场的强度是0.4-0.6 Gauss。需要注意的是，磁北极和地理上的北极并不重合，通常他们之间有11度左右的夹角。

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200428124113951.png" alt="image-20200428124113951" style="zoom: 67%;" />

地磁场是一个矢量，对于一个固定的地点来说，这个矢量可以被分解为两个与当地水平面平行的分量和一个与当地水平面垂直的分量。如果保持电子罗盘和当地的水平面平行，那么罗盘中磁力计的三个轴就和这三个分量对应起来，如下图所示。

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200428124403184.png" alt="image-20200428124403184" style="zoom:67%;" />

> 常用磁力计传感器：AKM8963（很经典的一颗，目前停产）、AKM09911、AKM09915、LIS3MDL，磁传感器目前还是AKM一家独大，其他家的性能差距还是比较明显的。

> 使用场景：主要是指南针，在应用中对6轴数据进行偏航校正。

## 2. 融合算法

想想我们为什么需要9轴的数据来确认物体的姿态呢？有了加速度计数据可以确定物体摆放的状态，例如有加速度计的手机，可以根据手机的横竖屏状态来触发屏幕相应的旋转，但对于物体的翻转、旋转的快慢无从得知，检测不到物体的瞬时状态，这时候就需要加入陀螺仪，通过加速度和陀螺仪的积分运算（这部分计算可以看下面Oculus的融合算法说明），可以获得到物体的运动状态，积分运算与真实状态存在微小差值，短时间内影响很小，但这个误差会一直累积，随着使用时间增加，就会有明显的偏离，6轴的设备，在转动360度后，图像并不能回到原点因此引入磁力计，来找到正确的方向进行校正。融合算法是通过这9轴的数据来计算出物体正确的姿态。目前9轴融合算法包括卡尔曼滤波、粒子滤波、互补滤波算法，对于开发者而言，所有的融合算法本基本都是丢入9轴传感器的数据和时间戳，然后获取到融合算法输出的四元素，应用所需的就是这组四元素，目前我这里接触到的算法包括：

> Oculus融合算法：[Oculus:”sensor fusion:Keeping It Simple”](http://blog.csdn.net/dabenxiong666/article/details/52957370)
>
> **注**：代码实现在openHMD中ofusion_update接口，不包含航向偏转。

> [MIT互补滤波算法](https://www.codeproject.com/articles/729759/android-sensor-fusion-tutorial)： MIT上发表的互补滤波算法的原理和基于Android平台的算法实现，很完整的算法

> AHRS：  在四轴飞行器论坛上，比较多人使用AHRS开源融合算法,[这里获取源码](https://github.com/TobiasSimon/MadgwickTests/blob/master/MadgwickAHRS.c)

## 3. 传感器调试

这里不对特定平台（MCU、Android、Linux等），传感器通讯接口（I2C、SPI等）、数据传递子系统(input、IIO等)详细说明，这部分代码由各sensor厂家直接提供，这里主要说明一下调试基本流程和方法：

> **通讯接口**：传感器IC的通讯接口I2C或SPI，通讯接口能够读写正常即可。
>
> **寄存器**： 寄存器参数配置，一般原厂会提供，根据自己需求设置full scale(量程)、ODR（采样速率）、中断、休眠模式 即可。
>
> **坐标系转换**：坐标系的匹配，一般通过驱动的旋转矩阵，来调整。这里需要注意，融合算法一般直接适配的是右手坐标系，而VR应用多数是基于unity引擎开发的，即采用左手坐标系，这里不能将IC的坐标系直接与左手坐标系做匹配，否则会有漂移！这个转换应用会有对应的API去做转换，将驱动坐标系与世界坐标系匹配。
>
> **硬件环境**：比如IC摆放不能靠近边缘，下方走线规范，附近几毫米内不允许有大电流，马达，软磁、硬磁干扰等等，这方面最好是把PCB给原厂审核，磁方面用他们专门的设备扫描磁力计周围的磁场环境是否正常。[Android平台调试sensor的文档](http://download.csdn.net/detail/dabenxiong666/9718883?locationNum=4&fps=1)

**链接**： https://blog.csdn.net/dabenxiong666/article/details/53836503

## 3. 传感器误差及处理方式

### 3.1. Category of the errors for IMU system

- Inertial sensor errors, including sensor bias error, cross-axis coupling and scale factor error;
- misalignment error, due to the misalignment angle along the sensitive axis an gravity acceleration component will be sensed as a part of acceleration
- computational process errors, such as numerical integration error.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20200620085340958.png)

> waveform numerical integration methods, such as cumulative rectangle, trapezoidal and Simpson integration methods.

### 3.2. Velocity waveform reconstruction

> a high sampling rate should be used when data acquisition is performed to reduce the sampling interval and thus reduce the integrationerror. However,thiswillresultinmorenoiseinthe measured data and increase the calculation burden.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200620090253185.png)

### 3.3. Waveform distortion

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200620090545825.png)

### 3.4. Velocity waveform correction

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
![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200620092722253.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200620093339022.png)



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/9-axis-sensor-introduce/  

