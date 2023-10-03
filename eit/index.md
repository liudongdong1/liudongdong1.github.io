# EIT


> `生物电阻抗断层成像`(Electrical Impedance Tomography，EIT)技术是一种新型医学功能成像技术，它的原理是`在人体表面电极上施加一微弱的电流，并测得其他电极上的电压值，根据电压与电流之间的关系重构出人体内部电阻抗值或者电阻抗的变化值`。由于该方法未使用核素或射线，对人体无害，因此可以多次测量重复使用，且成像速度快，具有功能成像等特点，加之其成本较低，不要求特殊的工作环境，因此EIT是一种理想的、具有诱人应用前景的无损伤医学成像技术，在20世纪末迅速成为研究热点，目前，一些商用的电阻抗断层成像设备已在临床开展应用。 [EIT-Kit_opensource](https://github.com/HCIELab/EIT-kit_open-source)

### 1. 基本原理

> 在人体表明粘贴一圈电极，由一对电极施加一微弱的电流激励，然后测量其余电极对上对应的电压值，再切换一对电极进行激励，并测量其余电极对上对应的电压值，重复下去，可以测的一组电流激励下的电压值，根据`电流和电压之间的关系，根据一定的重构算法即可重构出内部的电导率分布或者电导率变化的分布`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/format,f_auto)

### 2. 设备组成

- 一台EIT设备通常包括电极、EIT主机以及一台计算机。电极常采用ECG或者EEG电极，部分商用设备设计了针对特定用途的电极带

EIT主机的主要有`激励模块和测量模块`。其常用的`阻抗测量`方法有两种：

- 一种是`向目标物体施加幅度和相位已知的电压并测量其边界电流`；多用在频率较高，输出电流无法精确控制的场合，其测量结果易受电极-皮肤接触阻抗的影响
- 另一种是`向目标内注入已知大小的电流并测量其边界电压分布`。目前的EIT研究中常采用的方法，该方法基于四电极法生物电阻抗测量原理，通过一对驱动电极注入电流，测量其它电极上的电压差。采用这种方式测量时，由电极-皮肤接触阻抗导致的分压对测量电极上的电压信号影响很小，故测量结果不易受接触阻抗的影响，测量的精度较高。其缺点是当频率较高时，由于杂散电容的分流作用而降低测量精度。EIT系统的结构如图4所示，由电流源、多路开关、电压放大器、AD转换器、单片机等共同组成。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211119201143353.png)

### 3. 成像方式

> `静态电阻抗断层成像`（Static EIT）又称为EIT绝对成像（Absolute EIT），它`以人体内部电导率的绝对分布`为成像目标，早期的EIT研究大多采用这种成像方式。进行静态EIT成像时，常假设一个初始的电导率分布，然后根据测量数据不断的进行重构迭代，以此来求出一个最优解反映电导率的绝对分布。由于EIT的不适定性和病态性，边界形状、电极位置、系统噪声等微小的测量误差都有可能产生很大的重构误差，并最终导致迭代发散无法重构出目标。因此，静态电阻抗断层成像技术仍处于研究阶段。

> `动态电阻抗断层成像`（Dynamic EIT）又称为EIT相对成像（Relative EIT），它`以人体内部电导率的分布变化为成像目标`，目前大多数商业系统和研究采用这种成像方式。进行动态EIT成像时，常`选取某一时刻的数据作为参考帧，然后将当前时刻的数据作为测量帧与参考帧数据相减进行差分成像，以此来求出一个最优解反映测量帧时刻相对参考帧时刻的电导率的分布变化`。通过差分的形式，可有效降低边界形状、电极位置、接触阻抗、系统噪声等测量误差对图像重构的影响。由于动态EIT图像重构时，不需要进行迭代计算，因此成像速度快，可以及时反映组织阻抗的变化，这也是为什么动态EIT相对静态EIT成果更多、应用更早的原因。

> `多频电阻抗断层成像`（Multi-frequency EIT）不同于静态和动态电阻抗断层成像，它是`以人体内部不同频率点的电导率分布变化为成像目标`。进行多频EIT成像时，常`基于不同组织具有不同阻抗频谱特性的特点，通过同一时刻不同频率的数据重建出被测体内部的阻抗分布情况`。利用多频电阻抗断层成像技术可以显示人体组织的阻抗随频率变化的图像，在研究`人体生理功能和疾病诊断`方面具有特定的临床价值，但目前还未有临床应用的报道。

### 4. 优缺点

> EIT图像以不同的伪彩色带表示组织或器官的相对阻抗变化。不同于CT、MRI等解剖成像技术，EIT图像属于功能图像，如图5所示，相对于CT、MRI图像而言，图像分辨率低，无法观察到具体的脏器结构，但是却具有灵敏度高、安全性好、成本低廉、连续成像等优势。

> 电阻抗断层成像技术，特别是动态电阻抗断层成像技术，已经展现出了良好的应用前景，主要集中在：乳腺癌检测成像、腹部脏器功能成像、肺部呼吸功能成像、脑部功能成像等。

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/eit/  
