# SlamRelative


> Common functional pieces of autonomous vehicles often fall into sensing, computing, and actuation.The sensing devices or sensor include cameras, laser scanners (LiDAR), milliwave radars, and GNSS/IMU.  Using sensor data , autonomous vehicles perform localization, detection, prediction, planning, and control.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703082515701.png)

## 1. 常用传感器

- 激光雷达或深度摄像头
- 摄像头：单目、双目、多目。
- 惯性传感器

![传感器](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703082557398.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703084255829.png)

## 2. 发展历程

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703084657334.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703084724687.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211126214043470.png)

## 3. slam在AR上挑战

- 场景复杂多变，运动类型复杂难测，计算维度高，边缘设备计算能力有限
- 动态物体，遮挡，弱纹理，重复纹理。

### 3.1. 提升稳定性

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703085121310.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703085246477.png)

### 3.2. 提高计算效率

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703085435509.png)

### 3.3. 云-端协同

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703085737787.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703090321487.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703090446927.png)

![VICON系统](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703090620678.png)

- ENFT-SFM or LS_ACTS: http://www.zjucvg.net/ls-acts/la-acts.html
- RKSLAM: http://www.zjucvg.net/rkslam/rkslam.html
- RDSLAM: http://www.zjucvg.net/rdslam/rdslam.html
- ACTS:  http://www.zjucvg.net/acts/acts.html
- SenseSLAM: http://www.zjucvg.net/senseslam
- SenseAR: http://openar.sensetime.com

## 4.未来趋势

![image-20200703091448777](C:/Users/dell/AppData/Roaming/Typora/typora-user-images/image-20200703091448777.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703092235403.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200703092305819.png)

## 5.基于GPS+IMU+MM车载多传感器融合

> GPS(GlobalPositioning System)：指美国国防部研制的全球定位系统。用户设备通过接收GPS信号，得到用户设备和卫星的距离观测值，经过特定算法处理得到用户设备的三维坐标、航向等信息。使用不同类型的观测值和算法，定位精度为厘米级到10米级不等。GPS的优点是精度高、误差不随时间发散，缺点是要求通视，定位范围无法覆盖到室内。
>
> IMU(Inertial measurementunit)：指惯性测量单元。包括陀螺仪和加速度计。陀螺仪测量物体三轴的角速率，用于计算载体姿态；加速度计测量物体三轴的线加速度，可用于计算载体速度和位置。IMU的优点是不要求通视，定位范围为全场景；缺点是定位精度不高，且误差随时间发散。GPS和IMU是两个互补的定位技术。
>
> MM(Map matching)：指地图匹配。该技术结合用户位置信息和地图数据，推算用户位于地图数据中的哪条道路及道路上的位置。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712110553648.png)

- 偏航重算：是指在高架或城市峡谷，信号遮挡引起位置点漂移；<font color=red>GPS定位精度差和DR航位推算精度差。</font>
- 无法定位：是指在无信号区域（停车场、隧道）推算的精度低，导致出口误差大；
- 抓路错误：是指主辅路、高架上下抓路错误。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712110406899.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712110531503.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712110642270.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712110914514.png)

> GPS质量评估模块的功能是计算GPS位置、速度、航向角和全局可靠性指标。根据可靠性指标的大小将其投影到状态空间（GOOD、DOUBT、BAD、ABNORMAL）中，状态空间的值表征GPS数据质量的好坏。第一，决定是否使用GPS数据进行器件误差标定或某些状态的判断（如转弯行为、动静状态等）；第二，在数据融合模块，为设定GPS观测量的方差—协方差阵提供参考。

> 器件补偿:无GPS信号环境时，定位只能依靠DR算法。DR算法精度主要取决于IMU（陀螺仪和加速度计）和测速仪的误差，陀螺仪误差将引起位置误差随时间的二次方增长，测速仪误差将引起位置误差随时间线性增长。补偿模块的主要功能是利用GPS数据来补偿速度敏感器误差参数（比例因子）和IMU的误差参数（陀螺仪天向比例因子和陀螺仪三轴零偏）。补偿的目的是在无GPS信号或弱GPS信号的场景，仅靠DR算法也能得到较为可靠的导航信息。

> DR(DeadReckoning，航位推算)算法是指已知上一时刻导航状态（状态、速度和位置），根据传感器观测值推算到下一时刻的导航状态。DR算法包括姿态编排和位置编排两个部分。姿态编排使用的是AHRS（Attitude andheading reference system ）融合算法，处理后输出车机姿态信息。位置编排是指结合姿态编排结果，对测速仪观测值进行积分后得到车机位置。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712111338207.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200712111400756.png)

- 高架桥识别；停车场识别；主辅路识别。

视频地址：https://www.bilibili.com/video/BV1eK411n7aa



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/slamrelative/  

