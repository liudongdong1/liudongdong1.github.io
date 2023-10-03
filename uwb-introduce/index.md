# UWB Introduce


> UWB超宽带定位技术属于无线定位技术的一种。无线定位技术是指用来判定移动用户位置的测量方法和计算方法，即定位算法。目前最常用的定位技术主要有：时差定位技术、信号到达角度测量(AOA)技术、到达时间定位(TOA)和到达时间差定位(TDOA)等。其中，TDOA技术是目前最为流行的一种方案，UWB超宽带定位采用的也是这种技术。下面以国内UWB定位解决方案供应商恒高科技为例，为大家详细介绍UWB超宽带定位。

### 0. 背景

####  0.1 室内定位系统（IPS）

##### 0.1.1 室内定位系统介绍

&emsp;&emsp;GPS主导了室外场景下米级精度的定位市场，然而却无力进一步扩展到对位置感知需求愈加强烈的室内场景。作为未来工业4.0基础的智慧工厂、智能制造、智能物流、人机协作等，需要通过对原料、货物、资产、设备、人员等进行实时高精度的定位与追踪，从而实现对生产流程、客户需求、市场反馈、商品成本等进行快速调整和优化。

&emsp;&emsp;IPS(Indoor Positioning System)指在室内环境中利用<font color=red>无线、光学、声学、地磁场、惯性导航</font>等多种技术方式提供对人员、物体等进行定位能力的系统。尽管IPS在商业、零售、工业等领域有大量的应用，然而由于室内环境的复杂和多变性，使得当前没有单一或标准化的方案能主导室内定位需求，在不同的应用场景和业务需求以及成本预算下大多使用一种传感器数据或组合多种传感数据的定制化的解决方案。

##### 0.1.2 基于传感器IPS分类

- **无线定位技术：** WiFi,  Bluetooth,	RFID,	Zigbee,	UWB
- **光学定位技术：** Infra-red Sensor，LiFi，	Camera，	Lidar
- **声学定位技术：**：Ultrasound sensor
- **地磁定位技术：** Magnetometer
- **惯性定位技术：** Accelerometer， Gyroscope， IMU

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429170356767.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429170413736.png)

#### 0.2  基于位置服务（LBS）

LBS(location-based service)是一种根据用户的位置信息为用户提供相关信息/业务/需求的服务方式，国内最常见的LBS应用包括：

- 地图类APP基于用户实时位置所提供的导航服务，天气预警信息，交通拥塞通知等；
- 本地生活类APP等基于用于用户位置投放广告或促销信息等，如推荐的住宿服务，餐饮服务，娱乐场所等；
- 匿名/熟人社交APP所提供的附近人交友的功能；

LBS系统主要基于定位技术和网络通信技术两大支撑，其中定位能力更是LBS的前提条件，精确的定位能力可进一步扩展LBS的应用场景同时优化其使用体验。C端场景当前主要利用手机自带的WiFi，Bluetooth实现~10m精度的定位能力，随着Iphone11增加U1芯片提供对UWB技术的支持，未来C端用户可以借助手机内置的UWB芯片获得<10cm的定位精度而无需携带额外的UWB定位标签设备。

### 1. UWB 介绍

&emsp;&emsp;超宽带无线通信技术（UWB）是一种无载波通信技术，UWB不使用载波，而是使用短的能量脉冲序列，并通过<font color=red>正交频分调制或直接排序将脉冲</font>扩展到一个频率范围内。UWB的主要特点是传输速率高、空间容量大、成本低、功耗低等。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429162149692.png)

&emsp;&emsp;超宽带室内定位系统则包括UWB接收器、UWB参考标签和主动UWB标签。定位过程中由UWB接收器接收标签发射的UWB信号，通过过滤电磁波传输过程中夹杂的各种噪声干扰，得到含有效信息的信号，再通过中央处理单元进行测距定位计算分析。

&emsp;&emsp;未来无线定位技术的趋势是室内定位与室外定位相结合，实现无缝的、精确的定位。现有的网络技术还不能完全满足这个要求，而<font color=red>UWB技术由于功耗低、抗多径效果好、安全性高、系统复杂度低、定位精度极高</font>等优点，在众多无线定位技术中脱颖而出。

### 2. 定位原理

&emsp;&emsp;超宽带（Ultra Wide-Band，UWB）是一种新型的无线通信技术，根据美国联邦通信委员会的规范，UWB的<font color=red>工作频带为3.1~10.6GHz，系统-10dB带宽与系统中心频率之比大于20%或系统带宽至少为500MHz</font>。UWB信号的发生可通过发射时间极短（如2ns）的窄脉冲（如二次高斯脉冲）通过微分或混频等上变频方式调制到UWB工作频段实现。
&emsp;&emsp;超宽带的主要优势有，低功耗、对信道衰落（如多径、非视距等信道）不敏感、抗干扰能力强、不会对同一环境下的其他设备产生干扰、穿透性较强（能在穿透一堵砖墙的环境进行定位），具有很高的定位准确度和定位精度。

#### 2.1 UWB-TDOA 到达时间差

&emsp;&emsp;该技术采用TDOA（到达时间差原理），利用UWB技术测得定位标签相对于两个不同定位基站之间无线电信号传播的时间差，从而得出定位标签相对于四组定位基站（假设1#、2#为第一组，2#、3#为第二组，3#、4#为第三组，4#、1#为第四组）的距离差：$d_{i,12} $为标签到基站1，2的距离差
$$
\begin{cases} d_{i,12}=r_{i,1}-r_{i,2}\\
d_{i,23}=r_{i,2}-r_{i,3}\\
d_{i,34}=r_{i,3}-r_{i,4}\\
d_{i,14}=r_{i,1}-r_{i,4}
\end{cases}
$$

$$
d_{i,12}=\sqrt{(x_1-x_i)^2+(y_1-y_i)^2+(z_1-z_i)^2}-\sqrt{(x_2-x_i)^2+(y_2-y_i)^2+(z_2-z_i)^2}
$$

同理得到： $d_{i,23},d_{i,34},d_{i,14}$  示意图如下：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429164030335.png)

#### 2.2  信号强度

&emsp;&emsp;RSSI(Receive Signal Strength Indicator)通过测量无线信号在接收端的功率大小并根据无线信号的Friis传输模型计算出收发端之间的距离。
 ![](https://math.jianshu.com/math?formula=%5Cbegin%7Bgathered%7D%20P_r%5BdBm%5D%20%3D%20P_t%5BdBm%5D%20%2B%20G_t%5BdB%5D%20%2B%20G_r%5BdB%5D%20-%20L%5BdB%5D%20-%2020%5Clog_%7B10%7D(4%5Cpi%20d%2F%5Clambda)%20%5C%5C%20%5CDownarrow%20%5C%5C%20d%20%3D%20%5Cfrac%7B%5Clambda%7D%7B4%5Cpi%7D%2010%5E%7B(P_t%20-%20P_r%20%2B%20G_t%20%2B%20G_r%20-%20L)%2F20%7D%20%5C%5C%20%5Cend%7Bgathered%7D)
 [其中](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429171842598.png)，$P_r/P_t$分别表示接收/发送信号功率级，$G_R/G_T$分别表示接收/发送天线增益，$L$表示PCB、连接线、连接器等带来的损耗，$d$表示设备间距离，$\lambda$表示无线信号的中心波长。

从Friis传输模型中可以看出，RSSI的测距结果受收发天线设计，多径传播，非视距传播，直接路径损耗等环境因数影响较大，实际应用中测距精度~10m量级，远低于基于时间戳测距的方法，因而基于RSSI的方法很少直接用于UWB定位。

#### 2.3 [飞行时间（TOF）/到达时间（TOA)](https://www.jianshu.com/p/c7d1bdcd126c)

&emsp;&emsp;TOF(Time of Flight)/TOA(Time of Arrival)通过记录测距消息的收发时间戳来计算无线信号从发送设备到接收设备的传播时间，乘以光速然后得到设备间的距离。根据测距消息的传输方式不同可分为单向测距和双向测距，其中<font color=red>单向测距中测距消息仅单向传播，为获得设备间的飞行时间需要双方设备保持精确的时钟同步，系统实现复杂度和成本较高，而双向测距对双方设备的时钟同步没有要求</font>，系统实现复杂度和成本很低，因而我们主要关注双向测距这种方案。

- 单边双向测距(SS-TWR)

SS-TWR(Single-Sided Two-Way Ranging)算法中测距请求设备发起测距请求，而测距响应设备监听并响应测距请求，然后测距请求设备利用所有时间戳信息计算出设备间的飞行时间。

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429172413251.png" alt="image-20200429172413251" style="zoom:50%;" />

#### 2.4 到达角（AOA）到达相位差（PDOA）

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200429172125457.png)

### 3. UWB 定位优势

> UWB是一种高速、低成本和低功耗新兴无线通信技术。UWB信号是带宽大于500MHz或基带带宽和载波频率的比值大于0.2的脉冲信号(UWBWG，2001)，具有很宽的频带范围，FCC规定UWB的频带从3.1GHz～10.6GHz，并限制信号的发射功率在-41dBm以下。

由此可见，UWB聚焦在两个领域的应用上，一是符合IEEE802.15.3a标准的短距离高速数据通信，即无线无延迟地传播大量多媒体数据，速率要达到1OOMbit/s-500Mbit/s；另一个是符合IEEE802.15.4a的低速低功率传输，用于室内精确定位，例如工地施工人员的位置发现、化工厂危险源检测、监狱服刑犯人危险行为预测、机器人运动跟踪等。

UWB信号的特点说明它在定位上具有低成本、抗多径干扰、穿透能力强的优势，所以可以应用于静止或者移动物体以及人的定位跟踪，能提供十分精确的定位精度。

### 4. 使用需求

#### 4.1 实时导航

> 需要对自己所处位置信息有实时感知的人员或智能机器人等，通过及时地获取自身所处的位置从而进行路线规划与导航、自主巡检、定时定点操作、自动驾驶等任务。主要用于大型公共场所如医院导航、商场/购物中心导购、停车场反向寻车、展厅/博物馆自助导游、飞机场/火车站/地铁站导引、AGV/AMR智能机器人等。

#### 4.2 定位/监控/追踪:

> 需要对人员或设备资产等进行监控和追踪的场景，通过在网络服务器端对定位信息进行聚合和追踪，可有效的解决潜在的业务安全问题，优化生产流程瓶颈问题，防止资产丢失/人员走散问题，以及提供室内LBS相关应用等。
>
> 主要应用于：
>  \- 消费者服务：家人防走散、物品防丢失、LBS产品服务推送、LBS交友等。
>  \- 企业服务：人流监控和分析、访客分布热图、智慧仓储和物流、智能制造、紧急救援、人员资产管理和服务机器人等。

### 5. 应用场景

#### 5.1 隧道管廊

> 在隧道施工现场，通过部署UWB超宽带定位系统，将定位标签集成至员工胸卡、安全帽等穿戴设备内，可以提供的集风险管控、人员管理、实时显示、应急救援等功能能够准确定位工人位置，保障工人施工安全、施工质量、施工进度。

#### 5.2 工业制造

> 在工厂中，UWB超宽带定位系统可以帮助传统工厂实现数字化管理，可实时查看员工位置、在岗时间、离岗时间、移动轨迹，提高岗位巡查效率。通过后台对仓储货物位置的监管，可查看物品位置、所属仓库等数据，防止物资设备的丢失。

#### 5.3 司法监狱

> 在监狱，通过UWB超宽带定位系统，将定位标签集成至犯人定位腕带中，能够对服刑犯人进行实时监控。
>
> 包括：实时掌握人员的实时位置、人数清点、犯人腕带防拆报警、电子围栏、聚众分析、行动轨迹跟踪、 回放、摄像联动警报等，能够很大程度的降低监管执法的风险，防止意外事故的发生。

#### 5.4 养老院：

> 在养老院，通过给老人佩戴智能手环或胸牌，不仅能够实时查看老人位置，还能够通过设置电子围栏来圈定安全活动范围，一旦老人走出安全区域，系统就会及时预警，通知管理人员前往查看，避免老人走失。

#### 5.5 展厅

> 在会展、展厅内，UWB超宽带定位系统可实现智能化导览服务，一方面可以实时导引观众前往想去的展位，另外也可以对观众的位置数据进行精准统计，查看参展客户及工作人员在展厅内的观览轨迹、停留时间等，实现办展效果精准分析。让展方了解参展人员感兴趣产品，为后期提升展会质量工作提供明确方向。

#### 5.6 细粒度动作检测

> 利用毫米波雷达来检测人的呼吸频率，心跳频率，身体行为来判断睡眠质量

### 6. Apple UWB

#### 6.1 房间/汽车门禁

通过高精度的测距测向能力，当识别到用户靠近门禁时自动打开，相比Bluetooth等其他使用RSSI(接收信号强度)技术，UWB使用的TOF测距方法可提供更高的精度以及有效地防止中间人攻击；

#### 6.2 物品防丢失

给手机、钱包等易丢失的物品配备UWB定位芯片后，用户可进行高精度(<10cm)的实时监控和追踪；

#### 6.3 智能家居系统

通过感知用户在家里的位置信息，可有效地对环境的温度、湿度、光强等进行控制调整，从而带来更智能化的体验和更绿色的能效；

#### 6.4 智能设备控制器

通过确定Iphone11和被控制设备(如电视、空调等)的相对位置和姿态信息，可无需手动切换而智能化地实现利用手机对周围的电器设备进行控制；

#### 6.5 数据传输

#### 6.6. 数字钥匙

> 2021 年 7 月，CCC 联盟发布了汽车数字秘钥 3.0 版规范，**明确了第三代数字钥匙是基于 UWB/BLE (蓝牙)+NFC 的互联方案。**

- 第一代数字钥匙基于 NFC 近场通讯技术，实现了车辆进入与启动功能，但基本没有位置感知的能力， 需要近距离接触
- 第二代数字钥匙是采用 BLE 蓝牙技术，通过蓝牙信号的强弱粗略感知车与钥匙的位置关系，但其感知精度与准确性都有所欠缺。
- 第三代数字钥匙是 UWB、BLE、NFC 三种无线通信技术相结合的产品，位置感知精度大大提高。
  - BLE 蓝牙作为低功耗的通信技术，在方案中是默认常开的，用于钥匙与车端的远距离感知、鉴权与交互等通信功能。在蓝牙完成通信鉴权后将开启 UWB。
  - UWB 在方案中的主要作用是完成车端和钥匙端的测距。
  - NFC 主要负责出厂时的秘钥注入，以及在 BLE、UWB 同时失效时（例如，手机和钥匙同时没电）作为备用预案，确保可以用 NFC 刷开车门并启动车辆。

![]https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220118150408259.png)

> 车端架构中，与 UWB 相关的设备为 `UWB 锚点`与`解算单元（中控）`。其中 `UWB 锚点`包含了 UWB 射频和天线模组，`实现钥匙与车身的测距功能`。`解算单元里内置了 BLE 蓝牙模组，实现与钥匙端的通信`；解算单元同时负责接收 UWB 锚点信息，解算出钥匙与车身的相对位置关系，然后将位置信息再传输到车内其它的控制器（例如：BCM 与 DCU），从而进一步实现基于位置的应用层服务。

### 7. Paper Record

#### UWB(Itra-wideband) communication system

- refer to a signal or system that either has a large relative bandwidth that exceeds 20% or a large absolute bandwidth of more than 500MHz.    3.1-10.6Ghz

- transmitting large amounts of digital data over a wide frequency spectrum using short pulse, low powered radio signals

- traditional transmissions transmit information by varying the power /frequency/phase of a sinusoidal wave <font color=red>uwb transmissions can transmit information by generating radio energy at specific time instants and occupying large bandwidth thus enabling a pulse-position or time-modulation. information can be imparted by encoding the polarity of the pulse ,the amplitude of the pulse and also by using orthogonal pulses</font>

- <font color=red>uwb is carrier-less ,the data is not modulated on a continuous waveform with a specific carrier frequency as in narrowband and wideband technologies</font>

- in multiband uwb spectrum of 7.5GHz as a single band ,it can be divided into 15 subbands of 500 MHz or more ,OFMD techniques to transmit the information using orthogonal carriers in each subband

  ###### Advantage:

  1. <font color=red>future generations of communication systems require high mobility , flexibility, high data-rate</font>
  2. high data rate and very low power ,the UWB system can achieve high data rate even for low SNR in noisy enviroments. simple transreceiver architecture.
  3. very low power secure communications,  the low emission and impulsive nature of UWB radio leads to enhanced security in communications ,using very low energy per frequency and precisely timed patterns <font color=red> suitable for high security applications such as military communications</font>
  4. very low power operation, low cost , minimal hardware , transmits short impulses constantly instead of transmitting modulated waves continuously as do most narrowband systems
  5. multiple access communications: Timing hopping , Direct sequence
  6. Resolvable multiple components of UWB signals: pulses are very short in space(less than 60cm for a 500MHz wide pulse ,less than 23 cm for 1.3GHz bandwidth pulse)  ?? 了解下品绿于发射距离关系

  ###### Disadvantage:

  1. Interference ,UWB occupy a large frequency spectrum ,interference mitigation or avoidance with coexisting users(IEEE 802.11a 5.150-5.825)
  2. complex signal processing: carrier-less system has to rely on relatively complex and sophisticated signal processing techniques to recover data from the noisy environment
  3. Bit synchronization time: the pulses with picoseconds, the time for a transmiiter and receiver to achieve bit synchronization can be as high as few milliseconds, the channel acquisition time is very high.

  ###### Application:

  1. UWB sensor networks, UWB RFID ,and UWB positioning system (FC只是限于13.56MHz的频段！而RFID的频段有低频（125KHz到135KHz），高频（13.56MHz）和超高频（860MHz到960MHz之间) factors that influence the distance the ware transfer: 天气影响 ，天线增益 高度， 供电能量， 频率（频率越高自由空间损耗越大，通讯距离越近） 阻碍物![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191204185429509.png)
  
  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191204185449376.png)
  
  3. transferring large amounts of data in short-range for home or office networking
  2. short range voice ,data , video applications
  3. military communications on board helicopters and aircrafts that would otherwise have too many interfering multipath components
  4. anticollision vehicular radars
  5. through wall imaging used ro rescue,security and medical applications 
  6. reducing inventory time and increasing efficiency in several ways
  7. accurately locating a person or object within one inch of its location through any structure
  8. localization in search and rescue efforts ,tracking of livestock and pets
  9. detecting land mines
  10. assessing enemy locations and tracking troops

<center class="half">
    <img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191204181401744.png" width=40%/><img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191204181422217.png" width=40%/>
</center>
### 参考学习

- **EHIGH恒高：**https://zhuanlan.zhihu.com/p/47190294，http://www.everhigh.com.cn/
- **DrivexTech：** https://www.jianshu.com/p/52f659ce8304
- **联睿电子：**http://www.locaris-tech.com/sndw
- **全迹科技：**http://www.ubitraq.com/html/index.html
- **上海有为科技：**http://www.uwitec.cn/ServiceIntelligent/mobile?typeid=104
- **青岛硕盈科技：**http://www.mr-designr.com/



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/uwb-introduce/  

