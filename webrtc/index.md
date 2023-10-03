# WebRTC


> WebRTC，名称源自网页实时通信（Web Real-Time Communication）的缩写，是一项`实时通讯技术`，它允许`网络应用或者站点`，在不借助中间媒介的情况下，建立`浏览器之间点对点（Peer-to-Peer）的连接`，实现`视频流和（或）音频流或者其他任意数据的传输`，支持网页浏览器进行实时语音对话或视频对话。并不是像WebSocket一样，打通一个浏览器与WebSocket服务器之间的通信，而是通过一系列的信令，建立一个浏览器与浏览器之间（peer-to-peer）的信道，这个信道可以发送任何数据，而不需要经过服务器。并且WebRTC通过实现MediaStream，通过浏览器调用设备的摄像头、话筒，使得浏览器之间可以传递音频和视频。
>
> 点对点并不意味着不涉及服务器，这只是意味着正常的数据没有经过它们。至少，两台客户机仍然需要一台服务器来交换一些基本信息（我在网络上的哪些位置，我支持哪些编解码器），以便他们可以建立对等的连接。用于`建立对等连接的信息被称为信令，而服务器被称为信令服务器`。
>
> WebRTC（Web Real-Time Communication）项目的最终目的主要是让`Web开发者能够基于浏览器（Chrome\FireFox...）轻易快捷开发出丰富的实时多媒体应用，而无需下载安装任何插件`，Web开发者`也无需关注多媒体的数字信号处理过程`，只需编写简单的Javascript程序即可实现。
>
>  方案二： [WVP+ZLMediaKit+MediaServerUI实现摄像头GB28181推流播放录制](https://notemi.cn/wvp---zlmedia-kit---mediaserverui-to-realize-streaming-playback-and-recording-of-camera-gb28181.html)

#### 1. 应用场景

- 社交平台，如视频聊天室应用
- 远程实时监控
- 远程学习，如在线教育、在线培训
- 远程医疗，如在线医疗
- 人力资源和招聘，如在线面试
- 会议和联系中心之间的协作，如客户服务、呼叫中心
- Web IM，如web qq
- 游戏娱乐，如双人对战游戏（如象棋这种双人对战游戏，每一步的数据服务器时不关心的，所以完全可以点对点发送）
- 屏幕共享
- 人脸检测识别
- 虚拟现实
- 市场调研
- 金融服务
- 其它即时通信业务

#### 2. 信令服务器

> 1. 用来`控制通信开启或者关闭的连接控制消息`
> 2. `发生错误时用来彼此告知`的消息
> 3. `媒体适配：媒体流元数据，比如像解码器、解码器的配置、带宽、媒体类型`等等
> 4. 用来`建立安全连接的关键数据`
> 5. `网络配置：外界所看到的的网络上的数据，比如IP地址、端口`等

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217210209019.png)

##### 2.1.   NAT 技术

> NAT（Network Address Translation，网络地址转换）属接入广域网(WAN)技术，是一种将私有（保留）地址转化为合法IP地址的转换技术，主要用于实现私有网络访问公共网络的功能，它被广泛应用于各种类型Internet接入方式和各种类型的网络中。原因很简单，NAT不仅完美地解决了lP地址不足的问题，而且还能够有效地避免来自网络外部的攻击，隐藏并保护网络内部的计算机。

###### 2.1.1.  NAT 分类

- **Full Cone** ：NAT内部的机器A连接过外网机器C后，NAT会打开一个端口。然后外网的任何发到这个打开的端口的UDP数据报都可以到达A，不管是不是C发过来的。例如 A:192.168.8.100 NAT:202.100.100.100 C:292.88.88.88 A(192.168.8.100:5000) -> NAT(202.100.100.100 : 8000) -> C(292.88.88.88:2000) 任何发送到 NAT(202.100.100.100:8000)的数据都可以到达A(192.168.8.100:5000)

- **Restricted Cone**： NAT内部的机器A连接过外网的机器C后，NAT打开一个端口，然后C可以用任何端口和A通信，其他的外网机器不行。

  例如 A:192.168.8.100 NAT:202.100.100.100 C:292.88.88.88 A(192.168.8.100:5000) -> NAT(202.100.100.100 : 8000) -> C(292.88.88.88:2000) 任何从C发送到 NAT(202.100.100.100:8000)的数据都可以到达A(192.168.8.100:5000)

-  **Port Restricted Cone**  ： 这种NAT内部的机器A连接过外网的机器C后，NAT打开一个端口，然后C可以用原来的端口和A通信，其他的外网机器不行。

  例如 A:192.168.8.100 NAT:202.100.100.100 C:292.88.88.88 A(192.168.8.100:5000) -> NAT(202.100.100.100 : 8000) -> C(292.88.88.88:2000) C(202.88.88.88:2000)发送到 NAT(202.100.100.100:8000)的数据都可以到达A(192.168.8.100:5000)

###### 2.1.2. UDP hole Punching

> 需要一个公网机器C来充当”介绍人”，内网的A、B先分别和C通信，打开各自的NAT端口，C这个时候知道A、B的公网IP:Port，现在A和B想直接连接，比如A给B发，除非B是Full Cone，否则不能通信。反之亦然，但是我们可以这样：
>
> A要连接B，A给B发一个UDP包，同时，A让那个介绍人给B发一个命令，让B同时给A发一个UDP包，这样双方的NAT都会记录对方的IP，然后就会允许互相通信。

###### 2.1.3. NAT穿越

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217213147425.png)

- **STUN**： 
  - `探测和发现通讯对方是否躲在防火墙或者NAT路由器后面`。
  - 确定`内网客户端所暴露在外的广域网的IP`和`端口以及NAT类型`等信息;STUN服务器利用这些信息协助不同内网的计算机之间建立点对点的UDP通讯.

- **TURN**: 

> 对直播系统，难的不是服务器，而是客户端。客户端难的地方则主要体现在两个方面，一是网络传输有关，像侦听事件，同步主线程和读线程，穿透；二是流数据有关，像编码、解码、回声消除。而这些正是WebRTC帮你解决了。
>
> 现在的直播服务大部分的情况下是`一对多的通信`，一个主播可能会有成千上万个接收端，这种方式用传统的P2P来实现是不可能的，所以目前`直播的方案基本上都是会有直播服务器来做中央管理`，`主播的数据首先发送给直播服务器`，`直播服务器为了能够支持非常多用户的同事观看`，还要通过边缘节点CDN的方式来做地域加速，所有的接收端都不会直接连接主播，而是从服务器上接收数据。

#### 3. 多方会话

##### 3.1. 全网状模型

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217210448227.png)

> 多个浏览器通过Web服务器访问网站，浏览器之间的通话并不通过任何流媒体服务器，而是直接通过对等连接，通过UDP来实现浏览器之间的通信。这个叫做全网状模型。
>
> 全网状：不需要架设媒体服务器，媒体延迟低质量高。但是如果人数很多的话就会导致浏览器的本地宽带增加，不适合多人会议。

##### 3.2.  集中模型

> 服务端除了Web服务器之外还需要架构一个台媒体服务器，媒体服务器和各个浏览器之间实现对点连接。架设媒体服务器的目的在于接收各个浏览器的媒体流，之后通过媒体服务器把媒体流发给各个浏览器。
>
> 比较适合多人会话，节省本地宽带，但是只有少量浏览器查询的时候，这种体系的效率非常低（因为要走媒体服务器）。

#### 4. 直播协议

| 协议         | 优点                                                         | 缺点                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **HTTP FLV** | 实时性和RTMP相等； 相比于RTMP省去了一些协议交互时间，首屏时间更短，可拓展的功能更多； 将RTMP封装在HTTP协议之上的，可以更好的穿透防火墙等 | 不支持双向互动；目前在网页上只能用flash或者插件的方式解码播放，而且flash在cpu和内存上都是占用很高。 |
| **RTMP**     | CDN 支持良好，主流的 CDN 厂商都支持；协议简单，在各平台上实现容易，PC flash原生支持；支持双向互动；实时性很好；防HTTP下载。 | 基于TCP，传输成本高，在弱网环境丢包率高的情况下问题显著；不支持浏览器推送；Adobe 私有协议，Adobe已经不再更新； 需要访问1935端口，国内网络情况的恶劣程度，并不是每个网络防火墙都允许1935包通过；目前在网页上只能用flash或者插件的方式解码播放，而且flash在cpu和内存上都是占用很高。 |
| **HLS**      | 跨平台，支持度高，H5浏览器支持比较好，可以直接打开播放；IOS、安卓原生支持；技术实现简单。 | 延迟性比较大。                                               |
| **WebRTC**   | W3C 标准，主流浏览器支持程度高；Google 在背后支撑，并在各平台有参考实现；底层基于 SRTP 和 UDP，弱网情况优化空间大；可以实现点对点通信，通信双方延时低。 | 传统CDN没有ICE、STUN、TURN及类似的服务提供                   |

#### 5. WebRTC 组件框架

- 音视频引擎：OPUS、VP8 / VP9、H264
- 传输层协议：底层传输协议为 UDP
- 媒体协议：SRTP / SRTCP
- 数据协议：DTLS / SCTP
- P2P 内网穿透：STUN / TURN / ICE / Trickle ICE
- 信令与 SDP 协商：HTTP / WebSocket / SIP、 Offer Answer 模型

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217210926980.png)

- **[MediaStream](https://www.html5rocks.com/en/tutorials/webrtc/basics/#toc-mediastream)：**通过MediaStream的API能够通过`设备的摄像头及话筒获得视频、音频的同步流`。![**WebRTC内部结构**](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217211113147.png)
- **[RTCPeerConnection](https://www.html5rocks.com/en/tutorials/webrtc/basics/#toc-rtcpeerconnection)：**RTCPeerConnection是WebRTC用于构建`点对点之间稳定、高效的流传输的组件`。![**WebRTC 协议栈**](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217211132306.png)
- **[RTCDataChannel](https://www.html5rocks.com/en/tutorials/webrtc/basics/#toc-rtcdatachannel)：**RTCDataChannel使得浏览器之间（点对点）建立一个`高吞吐量、低延时的信道`，用于传输任意数据。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217212024597.png)

> WebRTC使用RTCPeerConnection来在浏览器之间传递流数据，这个流数据通道是点对点的，不需要经过服务器进行中转。但是这并不意味着我们能抛弃服务器，我们仍然需要它来为我们传递信令（signaling）来建立这个信道。WebRTC没有定义用于建立信道的信令的协议：信令并不是RTCPeerConnection API的一部分。
>
> 既然没有定义具体的信令的协议，我们就可以选择任意方式（AJAX、WebSocket），采用任意的协议（SIP、XMPP）来传递信令，建立信道。比如可以使用node的ws模块，在WebSocket上传递信令。

> RTCDataChannel API就是用来干这个的，基于它我们可以在浏览器之间传输任意数据。DataChannel是建立在PeerConnection上的，不能单独使用。建立RTCPeerConnection连接之后，两端可以打开一或多个信道交换文本或二进制数据。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201217212226979.png)

#### 6. 学习资源

- [WebRTC介绍](https://github.com/bovinphang/WebRTC)
- WebRTC 1.0: Real-time Communication Between Browsers：https://www.w3.org/TR/webrtc/
- Media Capture and Streams：https://w3c.github.io/mediacapture-main/
- Media Capture from DOM Elements：https://w3c.github.io/mediacapture-fromelement/

- WebRTC官方网站：https://webrtc.org/start/
- A Study of WebRTC Security：http://webrtc-security.github.io/
- [WebRTC API](https://developer.mozilla.org/zh-CN/docs/Web/API/WebRTC_API)
- demo
  - https://121.15.167.230:8090/demos/
  - https://webrtc.github.io/samples/
  - https://www.webrtc-experiment.com/
  - https://webcamtoy.com/zh/app/
  - https://idevelop.ro/ascii-camera/







---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/webrtc/  

