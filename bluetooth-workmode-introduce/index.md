# BlueTooth WorkMode Introduce


> `经典蓝牙模块（BT）`：泛指支持蓝牙协议在4.0以下的模块，一般用于`数据量比较大的传输，如：语音、音乐等较高数据量传输`。经典蓝牙模块可再细分为：`传统蓝牙模块和高速蓝牙模块`。传统蓝牙模块在2004年推出，主要代表是支持蓝牙2.1协议的模块，在智能手机爆发的时期得到广泛支持。高速蓝牙模块在2009年推出，速率提高到约24Mbps，是传统蓝牙模块的八倍，可以轻松用于录像机至高清电视、PC至PMP、UMPC至打印机之间的资料传输。功耗高，传输数据量大，有效距离10米；
>
> `低功耗蓝牙模块（BLE）`：是指`支持蓝牙协议4.0或更高的模块`，也称为BLE模块，最大的特点是成本和功耗的降低，应用于`实时性要求比较高的产品`中，比如：智能家居类（蓝牙锁、蓝牙灯）、`传感设备`的数据发送（血压计、温度传感器）、消费类电子（电子烟、遥控玩具）等。相比传统蓝牙技术，低功耗蓝牙技术所增加的一项新功能就是“广播”功能。通过这项功能，从设备可以告知其需要向主设备发送数据。低功耗，数据量小，有效距离40米；
>
> 应用区别：BLE低功耗蓝牙一般多用在蓝牙数据模块，拥有极低的运行和待机功耗，使用一粒纽扣电池可连续工作数年之久；BT经典蓝牙模块多用在蓝牙音频模块，音频需要大码流的数据传输更适合使用。

## 1. 蓝牙介绍

### 1.1. 低功耗

> 低功耗蓝牙（BluetoothLow Energy），简称BLE。蓝牙低能耗无线技术利用许多智能手段最大限度地降低功耗。蓝牙低能耗架构共有两种芯片构成：`单模芯片和双模芯片`。蓝牙单模器件是蓝牙规范中新出现的一种只支持蓝牙低能耗技术的芯片——是专门针对ULP操作优化的技术的一部分。蓝牙单模芯片可以和其它单模芯片及双模芯片通信，此时后者需要使用自身架构中的蓝牙低能耗技术部分进行收发数据。双模芯片也能与标准蓝牙技术及使用传统蓝牙架构的其它双模芯片通信。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521171951850.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211129100658165.png)

注：按应用可分为数据`蓝牙模块`和`语音蓝牙模块`，前者完成无线数据传输，后者完成语音和立体声音频的无线数据传输。

### 1.2. 蓝牙协议组成

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527182619005.png)

蓝牙协议体系中的协议按SIG的关注程度分为四层：

（1）核心协议：BaseBand、LMP、L2CAP、SDP；

（2）电缆替代协议：RFCOMM；

（3）电话传送控制协议：TCS-Binary、AT命令集；

（4）选用协议：PPP、UDP/TCP/IP、OBEX、WAP、vCard、vCal、IrMC、WAE。

除上述协议层外，规范还定义了主机控制器接口（HCI），它为基带控制器、连接管理器、硬件状态和控制寄存器提供命令接口。在上图中可见，HCI位于L2CAP的下层，但HCI也可位于L2CAP上层。

蓝牙核心协议由SIG制定的蓝牙专用协议组成。绝大部分蓝牙设备都需要核心协议（加上无线部分），而其他协议则根据应用的需要而定。总之，电缆替代协议、电话控制协议和被采用的协议在核心协议基础上构成了面向应用的协议。

### 1.3. 蓝牙基本架构

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527182726206.png)

#### 1.3.1 底层硬件模块

（1）**无线射频模块（Radio）**：蓝牙最底层，带微带天线，负责数据接收和发送。

（2）<font color=red>**基带模块（BaseBand）：**无线介质访问约定。提供同步面向连接的物理链路（SCO）和异步无连接物理链路（ACL），负责跳频和蓝牙数据及信息帧传输，并提供不同层次的纠错功能（FEC和CTC）。</font>

（3）**链路控制模块（LC）：**蓝牙数据包的编码和解码。

（4）**链路管理模块（LM）**：负责创建、修改和发布逻辑链接，更新设备间物理链接参数，进行链路的安全和控制。

（5）**主机控制器接口（HCI）**：是软硬件接口部分，由基带控制器、连接管理器、控制和事件寄存器等组成；软件接口提供了下层硬件的统一命令，解释上下层消息和数据的传递。硬件接口包含UART、SPI和USB等。

#### 1.3.2. 中间协议层

（1）**逻辑链路控制与适配协议（L2CAP）**：蓝牙协议栈的基础，也是其他协议实现的基础。向上层提供面向连接和无连接的数据封装服务；采用了多路技术、分割和重组技术、组提取技术来进行协议复用、分段和重组、认证服务质量、组管理等行为。

（2）**音视频发布传输协议（AVDTP）和音视频控制传输协议（AVCTP）**：二者主要用于Audio/Video在蓝牙设备中传输的协议，前者用于描述传输，后者用于控制信号交换的格式和机制。

（3）**服务发现协议（SDP）**：蓝牙技术框架至关重要一层，所有应用模型基础。动态的查询设备信息和服务类型，建立一条对应的服务通信通道，为上层提供发现可用的服务类型和属性协议信息。

（4）**串口仿真协议（RFCOMM）**：实现了仿真9针RS232串口功能，实现设备间的串行通信。

（5）**二进制电话控制协议（TCS）**：基于 ITU-T Q.931 建议的采用面向比特的协议，它定义了用于蓝牙设备之间建立语音和数据呼叫的控制信令（Call Control Signalling），并负责处理蓝牙设备组的移动管理过程。

#### 1.3.3. 蓝牙Profile

Bluetooth Profile是蓝牙设备间数据通信的无线接口规范。目前有四大类、十三种协议规则，厂商可以自定义规格。几种最常见的Profile文件：

（1）**通用访问配置文件（GAP）**：其他所有配置文件的基础，定义了在蓝牙设备间建立基带链路的通用方法，并允许开发人员根据GAP定义新的配置文件。包含所有蓝牙设备实施的功能，发现和连接设备的通用步骤，基本用户界面等通用操作。

（2）**服务发现应用配置文件（SDAP）**：描述应用程序如何用SDP发现远程设备服务，可与向/从其他蓝牙设备发送/接收服务查询的SDP连接。

（3）**串行端口配置文件（SPP）**：基于ETSI TS 07.10规格定义如何设置虚拟串行端口及如何连接两个蓝牙设备。速度可达128kb/s。

（4）**通用对象交换配置文件（GOEP）**：可以将任意对象（如图片、文档等）从一个设备传输到另一个设备。

#### 1.3.3. 蓝牙协议栈层次

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527182816190.png)

- **物理层（PHY）**：射频传输。
- **链路层（LL）**：控制射频状态，包括等待、广告、扫描、初始化、连接。
- **主机控制接口层（HCI）**：主机和控制器通信接口。
- **逻辑链路控制及适配协议层（L2CAP）**：提供数据封装服务，允许逻辑上点对点通信。
- **安全管理层（SM）**：加解密，为安全连接和数据交换提供服务。
- **属性协议层（ATT）**：允许设备（服务器）向另一个设备（客户端）展示特定的数据（属性）。
- **通用属性配置文件层（GATT）**：定义了使用ATT的服务框架，两个建立连接的设备之间的所有数据通信都是通过GATT子程序处理。GATT全称Generic Attribute Profile，中文名叫通用属性协议，它定义了services和characteristic两种东西来完成低功耗蓝牙设备之间的数据传输。它是建立在通用数据协议Attribute Protocol (ATT),之上的，ATT把services和characteristic以及相关的数据保存在一张简单的查找表中，该表使用16-bit的id作为索引。一旦两个设备建立了连接，GATT就开始发挥作用，同时意味着GAP协议管理的广播过程结束了。`GATT连接是独占的`，也就意味着一个BLE周边设备同时只能与一个中心设备连接。一旦周边设备与中心设备连接成功，直至连接断开，它不再对外广播自己的存在，其他的设备就无法发现该周边设备的存在了。
  - Profile: 蓝牙组织规定了一些标准的profile，例如 `HID OVER GATT ，防丢器 ，心率计`等。`每个profile中会包含多个service，每个service代表从机的一种能力`。
  - Service: service可以理解为一个服务，在ble从机中，通过有多个服务，例如电量信息服务、系统信息服务等，每个service中又包含多个characteristic特征值。`每个具体的characteristic特征值才是ble通信的主题`。比如当前的电量是80%，所以会通过电量的characteristic特征值存在从机的profile里.
  - Characteristic: `ble主从机的通信均是通过characteristic来实现`，可以理解为一个标签，通过这个标签可以获取或者写入想要的内容，例如加速度计的 X/Y/Z 三轴值。
  - UUID： 统一识别码，我们刚才提到的service和characteristic，都需要一个唯一的uuid来标识

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211117112619047.png)

- **通用访问配置文件层（GAP）**：对所有蓝牙设备提供共同的功能，如传输模式和访问程序、协议和应用描述。GAP服务包含设备发现、连接模式、安全、认证、联合模型和服务发现。

### 1.4. 通信过程

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521173001501.png)

## 2. 蓝牙工作模式

### 2.1. 主设备模式

> 在主机模式下的蓝牙模块可以对周围设备进行搜索并选择需要连接的从机进行连接。可以发送和接收数据，也可以设置默认连接从机的MAC地址，这样模块一上电就可以查找此从机模块并进行连接。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521171002588.png)

> 典型案例：`beacon处于广播模式时，一般会被设置成了不可连接的状态`，Beacon 会每隔一定的时间（SKYLAB的beacon为100毫秒）广播一个数据包到周围，作为独立的蓝牙主机在执行扫描动作时，会间隔地接收到 Beacon 广播出来的数据包。该数据包内容最多可以包含 31 个字节的内容。同时，在主机接收到广播包时，其中会指示该广播包来自于哪一个蓝牙从机 MAC 地址(每个 Beacon 拥有唯一的 MAC 地址)的从机设备和当前的接收发送信号强度指示值(RSSI)为多少。
>

### 2.2. 从设备模式

> 工作在从机模式的低功耗蓝牙模块也处于广播状态，等待被扫描。和广播模式不同的是，`从机模式的蓝牙模块是可以被连接的，在数据传输过程中作从机`； SKYLAB的蓝牙智能手环VG08，可以向蓝牙主机发送一些心率数据、记步数据、消耗的卡路里等等数据。
>
> 典型案例：蓝牙心率带，蓝牙智能手环等
>

### 2.3.广播模式

> 在这种模式下蓝牙模块可以进行`一对多的广播`。用户可以`通过AT指令设置模块广播的数据`，模块可以在低功耗的模式下持续的进行广播，应用于极低功耗，小数据量，单向传输的应用场合，比如信标、广告牌、室内定位、物料跟踪等。这类模块有蓝牙4.1模块：HY-264018W、HY-264027P；蓝牙4.2模块：HY-40R201P、HY-40R204P。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521171144385.png)

### 2.4. Mesh 组网模式

> 此模式下，可以简单的将多个模块加入到网络中来，利用星型网络和中继技术，每个网络可以连接超过65000个节点，网络和网络还可以互连，最终可将无数蓝牙模块通过手机或平板进行互联或直接操控。并且不需要网关，即使某一个设备出现故障也会跳过并选择最近的设备进行传输。整个联网过程只需要设备上电并设置通讯密码就可以自动组网，真正实现简单互联。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521171235972.png)

注：这种模式下会受到一定限制，首先是因为模块传输过程中需要不断切换模式，导致传输数据的量每次限制到20字节，并且传输速度会有几秒的延迟，这种场景类似于UDP的方式，并不能保证数据一定会被送达目的模块。

## 3. 应用场景

### 3.1 蓝牙灯控方案

> 蓝牙灯控方案说明：手机蓝牙和彩灯上的蓝牙模块进行配对，实现APP命令控制彩灯蓝牙，实现不同的功能，比如可以通过色板、声音调节喜欢的颜色、亮度等。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521173355565.png)

### 3.2 智能锁方案

> 蓝牙智能锁方案说明：智能锁中内置BLE蓝牙模块HY-40R204，手机通过APP读取智能锁蓝牙信息，尝试配对，并发送开锁请求到服务器端，服务器端向手机发送开锁指令，手机接受到指令，通过蓝牙再把指令发送给智能锁进行解锁。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521173423277.png)

### 3.3. 蓝牙MAC地址扫描打印

> 包含蓝牙MAC地址读取设备、MAC地址读取软件、MAC地址管理软件、二维码生成软件、二维码打印驱动的一整套解决方案。蓝牙MAC地址扫描打印解决方案说明：把低功耗蓝牙模块（比如昇润科技的BLE蓝牙4.2模块HY-40R204）充当主机角色，扫描周边设备，根据广播名称过滤，筛选出周边信号最强的设备，获取MAC地址；获取MAC地址后，通过串口将数据发送给标签打印机，标签打印机打印出符合要求的二维码。以二维码的形式将蓝牙MAC地址打印出来，方便蓝牙产品对蓝牙MAC地址进行读取，能够有效提高工作效率。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521173535475.png)

### 3.4. Mesh 智能家居组网

> 蓝牙Mesh网络是用于建立多对多（many：many）设备通信的低能耗蓝牙（Bluetooth Low Energy，也称为Bluetooth LE）新的网络拓扑。它允许您创建基于多个设备的大型网络，网络可以包含数十台，数百甚至数千台蓝牙Mesh设备，这些设备之间可以相互进行信息的传递，无疑这样一种应用形态为楼宇自动化，无线传感器网络，资产跟踪和其他解决方案提供了理想的选择。有了蓝牙Mesh，智能家居便涌现出很多新的应用可能性。

### 3.5. Beacon 室内定位

> Beacon是建立在低功耗蓝牙协议基础上的一种广播协议，同时它也是拥有这个协议的一款低功耗蓝牙从机设备。<font color=red>Beacon设备，通常放在室内的某个固定位置，每隔一定时间广播一个数据包到周围，作为独立的蓝牙主机在扫描时，会间隔地接收到 Beacon广播出来的数据包，可以用在超市商品促销</font>，用来向走进它的顾客推送促销信息或者优惠券等，或者通过当前接收发送信号强度指示值(RSSI)、和MAC地址解析等来进行复杂的数据运算，进而对顾客进行室内定位。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521173905860.png)

## 4. HC-08 4.0模块

> ​		HC-08蓝牙串口通信模块是新一代的基于Bluetooth Specification V4.0 BLE蓝牙协议 的数传模块。无线工作频段为 2.4GHzISM，调制方式是 GFSK。模块最大发射功率为4dBm， 接收灵敏度-93dBm，空旷环境下和 iphone4s 可以实现 80米超远距离通信。 模块大小26.9mm×13mm×2.2mm，集成了邮票封装孔和排针焊接孔，既可以贴片封 装，也又可以焊接排针，很方便嵌入应用系统之内。自带LED 状态指示灯，可直观判断蓝牙 的连接状态。 模块采用 TI的 CC2540F256 芯片，配置256K 字节空间，支持 AT 指令，用户可根据 需要更改角色（主、从模式）以及串口波特率、设备名称等参数，使用灵活。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200519222634002.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521174717901.png)

- <font color=red>工作电压   2.2v-3.6v</font>

> 按住按键或EN脚拉高，此时灯是慢闪， SPP-05进入AT命令模式，默认波特率是38400；此模式我们叫原始模式。原始模式下一直处于AT命令模式状态。
>
> HC-05上电开机，红灯快闪，按住按键或EN拉高， HC-05进入AT命令模式，默认波特率是9600；此模式我们叫正常模式。正常模式下只有按住按键或拉高EN才处于AT命令模式状态。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200519223233710.png)

| 指令          | 响应                                                         | 说明                                                         |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AT            | OK                                                           | 测试                                                         |
| AT+RX         | Name:HC-08-- >>>>蓝牙名是用户设定的名字 Role:Slave-->>>>模块角色（主/从） Baud:9600,NONE-->>>>串口波特率，校验位 Addr:xx,xx,xx,xx,xx,xx-->>>>蓝牙地址 PIN :000000-->>>>蓝牙密码（密码无效） www.hc01.com-->>>>汇承官网网址，欢迎登录！ www.hc01.com www.hc01.com | 查询模块基础参数                                             |
| AT+DEFAULT    | OK                                                           | 注：不会清除主机已记录的从机地址！若要清除，请在 未连线状态下使用AT+CLEAR 指令进行清除。 模块会自动重启，重启 200ms 后可进行新的操作 |
| AT+RESET      | OK                                                           | 模块会自动重启，重启200ms后 可进行新的操作！                 |
| AT+VERSION    | HC-08V3.1, 2017-07-07                                        | 获取软件版本和发布日期                                       |
| AT+ROLE=X     | Master/Slave                                                 | AT+ROLE=M 返回：Master(并重启) AT+ROLE=？ 返回：Master(不会重启),设置主从后自动配对 |
| AT+ADDR=XXX   | OKsetADDR                                                    | 地址必须为12 位的 0~F 数字或大写 字符，即 16 进制字符。查询填“? |
| AT+RFPM=X     | 4dBm（0 dBm / -6 dBm / -23dBm ）                             | 参数x 如下表所示，设置 和查询都是用代号表示                  |
| AT+BAUD=XX    | OK9600                                                       | 设置波特率和校验位                                           |
| CONT=X        | OK/Connectable/Non-Connectable                               | 设置可连接性，不可连接时 主要用于广播数据                    |
| AT+AVDA=XXX   | OK                                                           | 1、参数“xxx”可以是 1~12字节的任意用户数据。如果此 时主机状态 AT+CONT=1， 那么主机串口就会输出 xxx 的 数据。此广播数据不会永久保存，模块重启后会失效。 2、由于主机是固定 2s 扫描一次，所以，2s内最多只输出一 次从同一个从机接收到的广播数据。并且，此模式的特点是 “从机不断的广播、主机不断的扫描”，所以主机是会不断 的输出数据。 3、从机广播密度越高，数据越容易被主机接收到；广播密 度越高，从机的功耗也越高。 |
| AT+MODE=X     | OK                                                           | 功耗模式设置。 注意：仅限从机                                |
| AT+LED=X      | OK+LED=x                                                     | ?：查询 0：关闭 1：打开                                      |
| AT+LUUID=xxxx | OK+LUUID=xxxx                                                | 由于蓝牙设备繁多，所以一般蓝牙主机（因 为没有显示屏，很难人工选择）都设置了搜 索UUID 过滤。这样的话， 只有 UUID 相 同的从机才能被搜索到。 默认：FFF0（意为 0xFFF0）； 参数必须要在 0~F范围内 |
| AT+SUUID=xxxx | OK+SUUID=xxxx                                                | 此服务UUID 是主机找到服务的依据，找到服 务才能找到具体的特征值。 默认：FFE0（意为 0xFFE0）； 参数必须要在 0~F范围内 |
| AT+TUUID=xxxx | OK+TUUID=xxxx                                                | 此透传 UUID 必须正确才能正常透传， 收发数据。 默认：FFE1（意为 0xFFE1） ； 参数必须要在 0~F 范围内 |

使用USB-TTL 工具，通过串口测试工具进行测试，之前AT指令一直发送失败，但是数据透传没有问题。重启下电脑可以了

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521165637752.png)

![ ](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211115164408938.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200521185424254.png)

## 5. Android 开发记录

- 经典蓝牙：经典蓝牙设备发现其它经典蓝牙设备的方式是调用BluetoothAdapter的startDiscovery()方法，该方法可以同时发现经典蓝牙和ble的。

#### .1. android 官方API和蓝牙版本支持

> Android 4.2 版本系统之前，Google 一直使用的是 Linux 官方 Bluetooth 协议栈，即知名老牌开源项目` BlueZ`。BlueZ 实际上是由高通公司在2001年5月基于 GPL 协议发布的一个开源项目，该项目仅发布一个月后就被 Linux 之父 Linux Torvalds 纳入了 Linux 内核，并做为 Linux 2.4.6 内核的官方 Bluetooth 协议栈。随着 Android 设备的流行，BlueZ 也得到了极大的完善和扩展。例如 Android 4.1 版本系统中 BlueZ 的版本升级为4.93，它支持 Bluetooth 核心规范 4.0，并实现了绝大部分的 Profile。从Android 4.2 即 Jelly Bean 开始，Google 便在 Android 源码中推出了它和博通公司一起开发的 `BlueDroid `以替代 BlueZ。

![android-4.2-bluedroid-structure](https://gitee.com/github-25970295/blogimgv2022/raw/master/android-4.2-bluedroid-structure.jpg)

**Android支持的蓝牙协议栈：Bluz，BlueDroid，BLE**；

- Bluz是Linux推出的，目前使用最广泛；
- BlueDroid是Android4.0之后推出来的，简化了Bluz的操作；
- BLE是最新的低功耗协议，传输效率和传输速率都是很高的；

#### .2. 关键代码

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211206085800758.png)

##### 1. BluetoothAdapter

> 表示`本地蓝牙适配器`（蓝牙无线装置）。BluetoothAdapter 是所有蓝牙交互的入口点。借助该类，您可以`发现`其他蓝牙设备、`查询`已绑定（已配对）设备的列表、`使用已知的 MAC 地址实例 BluetoothDevice`，以及通过`创建 BluetoothServerSocket `侦听来自其他设备的通信。

```java
<!--蓝牙连接权限-->
<uses-permission android:name="android.permission.BLUETOOTH" />
<!--蓝牙通讯权限-->
<uses-permission android:name="android.permission.BLUETOOTH_ADMIN" />
// 启动蓝牙
public void turnOnBlueTooth(Activity activity, int requestCode) {
	Intent intent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
	activity.startActivityForResult(intent, requestCode);
	// mAdapter.enable(); // 谷歌不推荐这种方式
}
 
// 关闭蓝牙
public void turnOffBluetooth() {
	mAdapter.disable();
}
 
// 打开蓝牙可见性
public void enableVisibily(Context context) {
	Intent intent = new Intent(BluetoothAdapter.ACTION_REQUEST_DISCOVERABLE);
	intent.putExtra(BluetoothAdapter.EXTRA_DISCOVERABLE_DURATION, 300);
	context.startActivity(intent);
}
```

##### 2. BluetoothDevice

> 表示`远程蓝牙设备`。借助该类，您可以通过 BluetoothSocket 请求与某个远程设备建立连接，或查询有关该设备的信息，例如设备的`名称、地址、类和绑定状态`等。

```java
// 查找设备
public void findDevice() {
	assert (mAdapter != null);
	mAdapter.startDiscovery();
}
 
// 绑定设备
public boolean createBond(BluetoothDevice device) {
	boolean result = device.createBond();
	return result;
}
 
// 绑定状态
BluetoothDevice.BOND_BONDED
BluetoothDevice.BOND_BONDING
BluetoothDevice.BOND_NONE
 
// 获取已绑定的蓝牙设备
public List<BluetoothDevice> getBondedDeviceList() {
	return new ArrayList<>(mAdapter.getBondedDevices());
}
 
// 解除绑定
public boolean removeBond(Class btClass, BluetoothDevice btDevice)
throws Exception {
	Method removeBondMethod = btClass.getMethod("removeBond");
	Boolean returnValue = (Boolean) removeBondMethod.invoke(btDevice);
	return returnValue.booleanValue();
}
 
 
// 蓝牙操作中发出的广播
private void registerBluetoothReceiver() {
	IntentFilter filter = new IntentFilter();
	//开始查找
	filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED);
	//结束查找
	filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED);
	//查找设备
	filter.addAction(BluetoothDevice.ACTION_FOUND);
	//设备扫描可见改变 当我可以被看见时就会发送一个广播过来
	filter.addAction(BluetoothAdapter.ACTION_SCAN_MODE_CHANGED);
	//绑定状态
	filter.addAction(BluetoothDevice.ACTION_BOND_STATE_CHANGED);
 
	registerReceiver(receiver, filter);
}
```

##### 3. **BluetoothSocket**

> 表示蓝牙套接字接口（类似于 TCP Socket）。这是允许应用使用 InputStream 和 OutputStream 与其他蓝牙设备交换数据的连接点。

```
close(),关闭
connect()连接
getInptuStream()获取输入流
getOutputStream()获取输出流
getRemoteDevice()获取远程设备，这里指的是获取bluetoothSocket指定连接的那个远程蓝牙设备
```

##### 4. **BluetoothServerSocket**

> 表示用于侦听传入请求的开放服务器套接字（类似于 TCP ServerSocket）。如要连接两台 Android 设备，其中一台设备必须使用此类开放一个服务器套接字。当远程蓝牙设备向此设备发出连接请求时，该设备接受连接，然后返回已连接的 BluetoothSocket。

> 服务器端 : 使用BluetoothServerSocket对象可以创建一个BluetoothSocket对象, 调用BluetoothServerSocket的accept()方法就可以获取该对象;
> 客户端 : 调用BluetoothDevice的createRfcommSocketToServiceRecord()可以获取该对象; 
>
> 在服务器端BluetoothServerSocket进行accept()阻塞, 在客户端BluetoothSocket调用connect()连接服务器, 如果连接成功, 
> 服务器端的accept()方法就会返回BluetoothSocket对象, 同时客户端的BluetoothSocket也成功连接服务器, 
> 此时服务器端和客户端的BluetoothSocket对象就可以获取输入输出流, 对数据进行操作;

### 学习连接

- https://zhuanlan.zhihu.com/p/42552026
- https://zhuanlan.zhihu.com/p/30408637

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/bluetooth-workmode-introduce/  

