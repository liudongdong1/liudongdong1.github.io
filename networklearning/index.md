# NetworkLearning


### 1.  Concept

- open system; closed system; computer network; osi stands; 
- protocals: 
  - host name; ip address; mac address(`ipconfig/all`);  port; socket(combined with ip and port);
  - DNS;   ARP;   RARP;
- internet and web programming: 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208183139193.png)

- **Goals:**
  - **Performance –** It is measured in terms of transit time and response time.
    - The number of users
    - Type of transmission medium
    - Capability of connected network
    - Efficiency of software
  - **Reliability –** It is measured in terms of
    - Frequency of failure
    - Recovery from failures
    - Robustness during catastrophe
  - **Security –** It means protecting data from unauthorized access.
- **Media:**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208183510240.png)

### 2. Data Link

#### 2.1. Ethernet  LAN

> Ethernet` CSMA/CD`, Token Ring and Wireless LAN using IEEE 802.11 are examples of standard LAN technologies.

- **Data Terminal Equipment (DTE):** he end devices that convert the user information into signals or reconvert the received signals;
- **Data Communication Equipment (DCE):**   the intermediate network devices that receive and forward frames across the network. 
- **ALOHA**
  - **Pure Aloha**: the stations simply transmit frames whenever they want data to send.
  - **Slotted Aloha**: the time of the shared channel is divided into discrete intervals called *Slots*. The stations are eligible to send a frame only at the beginning of the slot and only one frame per slot is sent.

### 3. Usage

#### 3.1. [端口映射](https://service.tp-link.com.cn/detail_article_2441.html)

> 使用路由器后，Internet用户无法访问到局域网内的主机，因此不能访问内网搭建的Web、FTP、Mail等服务器。虚拟服务器功能可以实现将内网的服务器映射到Internet，从而实现服务器对外开放。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208190110382.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208210148525.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208210330567.png)

#### 3.2. VPN 路由器

> `VPN路由器的最终目标与桌面或移动VPN应用程序的目标完全相同`。

- **提供无限连接：**过路由器连接到VPN服务器时，你可以使用任意数量的设备。此外，你还可以与朋友和访客共享加密连接，而无需担心共享帐户（这在VPN服务的服务条款中，通常是被禁止的）。
- **DD-WRT是更多功能的选项**，为**80多个路由器品牌**提供支持 – 包括TP-Link、Tenda和D-Link等供应商的入门级型号。
- Tomato可兼容的路由器选择较少，**但通常更适合与OpenVPN合作**，并具有独特的功能（例如允许两个VPN服务器同时运行），且被广泛认为**具有比DD-WRT更简洁的界面**。

##### 3.2.1.传统VPN架设方案

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208191953609.png)

- 路由器A能上网就行了,假设路由器A的网关是192.168.1.1;

- 登陆路由器B管理页面进行VPN设置:

  1、选择`PPTP或L2TP`连接方式设置VPN连接（多数L2TP VPN线路连接需要输入IPSEC共享密钥，有些路由器可能没有这个选项，建议优先使用PPTP，兼容性较好，成功几率也比较大）。

  2、`VPN用户名、密码和服务器地址，这个由VPN服务商提供`。

  3、IP地址：192.168.1.X（最后一个数字为2-254之间任意数字）。

  4、子网掩码：一般为255.255.255.0 （路由器A的子网掩码，可根据实际情况修改）

  5、网关地址：一般为192.168.1.1 （路由器A的网关，可根据实际情况修改）

  6、[DNS](https://hsk.oray.com/)服务器和路由器MAC地址这些选项都保持为默认的。设置后点击应用确定生效，稍等一会就可以连接上了。

##### 3.2.2.  蒲公英Cloud VPN组网方案

- 准备：两台或两台以上蒲公英路由器
- 登录智能组网管理平台，点击【添加路由器】;填写SN码和设备名称就可以了。通过【智能组网】—— 【立即创建网络】开始组网。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208192235866.png)

#### 3.3. 静态路由

在一个公司网络中，不仅可以通过无线路由器B连接外网，还可以通过无线路由器A来连接公司内网服务器。在不修改本地连接的IP地址及网关情况下，公司电脑需要能够同时访问外网和内网服务器。配置实例如下图：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208192815202.png)

PC默认将不与自己在同一网段的数据发送给网关192.168.1.1，即无线路由器B。路由器B接收到数据后，检查数据包的目的地址。如果发现目的IP为10.70.1.0的数据包，则路由器会发送一个ICMP重定向数据包给PC，告知PC后续发往10.70.1.0网段的数据包，都发送给192.168.1.2，即路由器A即可。这样PC就可以直接访问公司内网服务器了。

- 使用路由器管理地址登陆路由器B管理界面，点击“路由功能”菜单，选择“静态路由表”，点击“添加新条目”按钮，在静态路由表中填写相应的参数。

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208193049723.png)

- `route print` : 打印路由表信息

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208193726116.png)

> 指明了到达目的网段的下一跳地址，可以将其描述为，192.168.2.91 或者192.168.2.115前往本地路由表中未知的网段时，会选择默认路由进行传递，（全0代表全网），此时他的下一跳地址就是192.168.2.1。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208194048155.png)

- `display ip routing-talbe` : 查看路由表：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208194318243.png)

- **Destination/mask**：目的/掩码，指明了目的网段或者主机，使用掩码来区分究竟是网段还是主机，当你配置完IP地址后，路由表中会自动生成直连与本地的路由，其中127.0.0.0/8和127.0.0.1/32代表本地的环回口地址。
- **Proto（protocol）**：协议，用来指明这条路由是通过什么方式来获取到的，其中Direct代表直连获取，除去直连还有static代表静态，ospf代表从OSPF中获取，RIP代表从rip协议中获取的，诸如此类，还有许多。
- **Pre（preference）**：路由优先级。为了对于不同的路由协议进行区分，某些优秀的、值得相信的路由协议，就会将优先级的阈值设低，而优先级越低的路由协议，越可靠。
- **next-hop：下一跳地址。**用来指明到达目的网络应该将路由传递给谁。
- **Interface：**出接口。用于指明到达目的网络的本地出接口，与下一跳同理，一般运用于出口设备没有固定ip地址的环境，此时必须使用出接口。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208195116683.png)

```shell
# traceroute 172.22.102.250
traceroute to 172.22.102.250 (172.22.102.250), 30 hops max, 38 byte packets
1  192.168.5.64 (192.168.5.64)  10.000 ms  10.000 ms  10.000 ms
2  172.22.102.251 (172.22.102.251)  10.000 ms *  10.000 ms
```

#### 3.4.  远程管理

> 在网络任何地方都可以管理到路由器，进行实时、安全的管控配置。远程WEB管理功能，可以实现在接入互联网的地方即可远程管理路由器。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201208195525879.png)

#### 3.5. UPnP

> 路由器UPnP功能用于实现局域网计算机和智能移动设备，通过网络自动彼此对等连接，而且连接过程无需用户的参与。`路由器UPnP功能用于局域网络计算机和智能移动设备，流畅使用网络`，加快P2P软件访问网络的速度，如观看在线视频和多点下载等方面的软件，使网络更加稳定。为了保证像BT这样的P2P软件正常工作.	

- 设备可以动态地进入网络中，随后获得IP地址，“学习” 或查找自己应当进行的操作和服务的信息；“感知”别的设备是否存在以及它们的作用和当前的状态 。所有这些，都应当是可自动完成的。

#### 3.6. 公网ip判断

- 方法一：
  - 点击链接 http://www.net.cn/static/customercare/yourIP.asp 抓取自己的IP地址
  - tracert <刚才获取的IP>
  - ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210124191303322.png)
- 路由器ip查看： 输入cmd ——然后输入**ipcongfig**——然后按回车键，就可以看到下面有一个`Default Gateway`就可以看见IP了

### 4. 学习资源

- https://www.tutorialspoint.com/communication_technologies/communication_technologies_quick_guide.htm
- https://tutorialspoint.dev/computer-science/computer-network-tutorials/computer-network-introduction-mac-address
- https://www.juniper.net/documentation/zh_Hans/junos/topics/topic-map/layer-2-understanding.html
- 捣鼓一下[OpenWrt Project](https://openwrt.org/)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/networklearning/  

