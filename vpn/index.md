# vpn


> VPN技术通过`密钥交换、封装、认证、加密手段`在公共网络上建立起`私密的隧道`，保障传输数据的完整性、私密性和有效性。OpenVPN是近年来新出现的开放源码项目，实现了SSL VPN的一种解决方案。 传统SSL VPN通过端口代理的方法实现，代理服务器根据应用协议的类型（如http，telnet等）做相应的端口代理，客户端与代理服务器之间建立SSL安全连接，客户端与应用服务器之间的所有数据传输通过代理服务器转发。这种实现方式烦琐，应用范围也比较窄：`仅适用于用TCP固定端口进行通信的应用系统，且对每个需要代理的端口进行单独配置`；对于每个需要用到动态端口的协议都必须重新开发，且在代理中解析应用协议才能实现代理，如FTP协议；不能对TCP以外的其它网络通信协议进行代理；代理服务器前端的防火墙也要根据代理端口的配置变化进行相应调整。 `OpenVPN以一种全新的方式实现了SSL VPN的功能`，克服了传统SSL VPN的一些缺陷，`扩展了应用领域，并且防火墙上只需开放TCP或UDP协议的一个端口`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201206224201009.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201206224430365.png)

### 0. VPN

> VPN，英文全称是Virtual Private Network，字面解释可理解为：虚拟私有网络；VPN可以运行为Point-to-Point模式，也可以运行为Server模式，在Server模式下，VPN服务器可作为网关设备使用，用以给处于外部的不安全网络环境中的Client提供安全访问内网服务的通道.

### 1. 工作原理

#### 1.1. 虚拟网卡

> 在Linux2.4版本以上，操作系统支持一个名为tun的设备，tun设备的驱动程序中包含两个部分，一部分是`字符设备驱动`，一部分是`网卡驱动`。网卡的驱动把从TCP/IP协议栈收到的`数据包结构skb`放于`tun设备的读取队列`，用户进程通过调用字符设备接口read获得完整的IP数据包，字符驱动read函数的功能是从设备的读取队列读取数据，将核心态的skb传递给用户；反过来字符驱动write函数给用户提供了把用户态的数据写入核心态的接口，write函数把用户数据写入核心空间并穿入TCP/IP协议栈。该设备既能以字符设备的方式被读写，作为系统的虚拟网卡，也具有和物理网卡相同的特点：能够`配置IP地址和路由`。对虚拟网卡的使用是OpenVPN实现其SSL VPN功能的关键。

#### 1.2. 地址池以及路由

> OpenVPN服务器一般需要配置一个`虚拟IP地址池`和`一个自用的静态虚拟IP地址`（静态地址和地址池必须在同一个子网中），然后`为每一个成功建立SSL连接的客户端动态分配一个虚拟IP地址池中未分配的地址`。这样，物理网络中的客户端和OpenVPN服务器就连接成一个虚拟网络上的星型结构局域网，`OpenVPN服务器成为每个客户端在虚拟网络上的网关`。OpenVPN服务器同时提供对客户端虚拟网卡的路由管理。当客户端对OpenVPN服务器后端的应用服务器的任何访问时，`数据包都会经过路由流经虚拟网卡`，OpenVPN程序在虚拟网卡上截获数据IP报文，然后`使用SSL协议将这些IP报文封装起来`，再经过物理网卡发送出去。OpenVPN的服务器和客户端在虚拟网卡之上建立起一个虚拟的局域网络，这个虚拟的局域网对系统的用户来说是透明的。

#### 1.3. 客户服务器端

> OpenVPN的服务器和客户端`支持tcp和udp`两种连接方式，只需在服务端和客户端`预先定义好使用的连接方式（tcp或udp）和端口号`，`客户端和服务端在这个连接的基础上进行SSL握手`。连接过程包括SSL的握手以及虚拟网络上的管理信息，`OpenVPN将虚拟网上的网段、地址、路由发送给客户端。连接成功后，客户端和服务端建立起SSL安全连接`，客户端和服务端的数据都`流入虚拟网卡做SSL的处理`，再在`tcp或udp的连接上从物理网卡`发送出去。

#### 1.4. 数据包的处理过程

- **发送数据流程**

> 应用层的外出数据，经过系统调用接口`传入核心TCP/IP层做处理`，在`TCP/IP经过路由到虚拟网卡`，虚拟网卡的`网卡驱动发送处理程序hard_start_xmit()将数据包加入skb表`并完成数据包`从核心区到用户区的复制`，`OpenVPN调用虚拟网卡的字符处理程序tun_read()`，读取到设备上的数据包，对读取的数据包`使用SSL协议做封装`处理后，通过`socket系统`调用发送出去。

- **接受数据流程**

>  物理网卡接收数据包，经过核心TCP/IP上传到OpenVPN，OpenVPN通过`link_socket_read()`接收数据包，使用`SSL协议进行解包处理`，经过处理的数据包OpenVPN调用虚拟网卡的字符处理程序`tun_write()写入虚拟网卡的字符设备`，设备驱动程序完成数据`从用户区到核心区的复制`，并将`数据写入skb链表`，然后`调用网卡netif_rx()接收程序`，`数据包再次进入系统TCP/IP协议栈`，传到上层应用程序。如图1所示。 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201206223626554.png)

#### 1.5. 数据封装

>  OpenVPN提供`tun`和`tap`两种工作模式。在tun模式下，从`虚拟网卡上收到的是不含物理帧头IP数据包`，SSL处理模块对IP包进行SSL封装；在tap模式下，`从虚拟网卡上收到的是包含物理帧头的数据包`，SSL处理模块对整个物理帧进行SSL封装。Tap模式称为网桥模式，整个虚拟的网络就像网桥方式连接的物理网络。这种模式可以传输以太网帧、IPX、NETBIOS等数据包，应用范围更广。

#### 1.6. OpenVPN与Openssl

> OpenVPN软件包需要和openssl软件一起安装，因为OpenVPN调用了Openssl函数库，`OpenVPN的客户端和服务端建立SSL链接的过程是通过调用Openssl来实现的`。通过bio_write()/函数把数据写入Openssl的状态机通道，bio_read()从Openssl读取结果。OpenVPN还调用Openssl的加解密函数处理转发的数据包。

### 2. 安装

#### 2.1. Linux

- **server**

```shell
sudo apt-get install openvpn
# OpenVPN服务器需要四个文件ca.crt、server.crt、server.key和dh1024.pem
sudo cp /usr/share/doc/openvpn/examples/sample-keys/ca.crt /etc/openvpn
sudo gunzip -c /usr/share/doc/openvpn/examples/sample-keys/server.crt.gz > /tmp/server.crt
sudo mv /tmp/server.crt /etc/openvpn/
sudo cp /usr/share/doc/openvpn/examples/sample-keys/server.key /etc/openvpn/
sudo cp /usr/share/doc/openvpn/examples/sample-keys/dh*.pem /etc/openvpn
#编写服务器端配置文件/etc/openvpn/server.conf  
#/usr/share/doc/openvpn/examples/sample-config-files/server.conf.gz
port 1194
proto udp
dev tun
ca ca.crt
cert server.crt
key server.key
dh dh1024.pem
server 10.8.0.0 255.255.255.0
keepalive 10 120
user nobody
group nogroup
persist-key
persist-tun
verb 3
#  运行
cd /etc/openvpn
sudo openvpn --config server.conf
ifconfig
```

- **client**

```shell
sudo apt-get install openvpn
sudo cp /usr/share/doc/openvpn/examples/sample-keys/ca.crt ~
sudo gunzip -c /usr/share/doc/openvpn/examples/sample-keys/client.crt.gz > ~/client.crt
sudo cp /usr/share/doc/openvpn/examples/sample-keys/client.key ~

#在客户端的/etc/openvpn下创建文件client.conf
#/usr/share/doc/openvpn/examples/sample-config-files/client.conf
client
dev tun
proto udp
remote 192.168.1.100 1194   #修改为服务器ip 
ca ca.crt
cert client.crt
key client.key
user nobody
group nogroup
persist-key
persist-tun
verb 3

#测试
cd /etc/openvpn
sudo openvpn --config client.conf
ifconfig
ping 10.8.0.1
```

#### 2.2. [windows](https://swupdate.openvpn.org/community/releases/openvpn-install-2.4.8-I602-Win10.exe)

- OpenVPN的客户端配置文件为`*.ovpn`  ，将*.conf 后缀改为 *.ovpn 文件就行。
- 在使用证书认证的情况下，在`ovpn`文件同一个目录下面会有
  - `*.crt`
  - `*.key`
  - `ca.crt`
  - 再开启了`tls-auth`时还会有`ta.key`文件
- 证书文件可以内嵌到`ovpn`文件中，因此有时候会只有一个`ovpn`文件

- 开机自启动：
  - 把 **ovpn** 配置文件放在 **C:\Program Files\OpenVPN\config**
  - 运行 **services.msc**
  - 找到 **OpenVPNService** ，点击 **右键** ，选择 **属性**
  - 把启动类型改为 **自动** ，点击 **启动** ，点击 **确定**

### 3. 问题

```
TLS Error: TLS key negotiation failed to occur within 60 seconds (check your network connectivity)
```

- **NAT/PAT** - openVPN is not aware of any NAT in use at the firewall, and will use it's configured management interface address when creating the OpenVPN client configuration files. this can be changed on the client after install by editing the file /etc/openvpn/
- **Firewall/router blocking port** - If the firewall is not allowing the configured TCP port for the connection. the defalt port for the VPN server is TCP port 33800)
- **Multi-Route (link/load balancer) issues** - Load balancers may attempt to fail the connection to a secondary connection, this may result in routing issues which prevent the connection from establishing.
- **Incorrect Client configuration** - On manual uploads, please confirm that the client configuration file being uploaded to the VPN client is the correct file.

### 4. 资源

- https://openvpn.net/community-resources/how-to/

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/vpn/  

