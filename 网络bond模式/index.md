# 网络bond模式


## 双网卡绑定单个IP

> 双网卡绑定单个IP 地址 为了提供网络的高可用性，我们可能需要将多块网卡绑定成一块虚拟网卡对外提供服务，这样即使其中的一块物理网卡出现故障，也不会导致连接中断。
>
> 共支持bond[0-6]共七种模式，常用的就三种，如下:
>
> - mode=0：默认，平衡负载模式，有自动备援，但需要配置交换机。 
> - mode=1：主备模式，其中一条线若断线，其他线路将会自动备援，不需要配置交换机。 
> - mode=2：选择网卡的序号=(源MAC地址 XOR 目标MAC地址) % Slave网卡（从网卡）的数量，其他的传输策略可以通过xmit_hash_policy配置项指定 
> - mode=3：使用广播策略，数据包会被广播至所有Slave网卡进行传送 
> - mode=4：使用动态链接聚合策略，启动时会创建一个聚合组，所有Slave网卡共享同样的速率和双工设定 但是，mode4有两个必要条件
> - mode=6：平衡负载模式，有自动备援，不需要配置交换机。 

### 配置bond

| 网卡         | bond1 IP        | bond 模式 |
| :----------- | :-------------- | :-------- |
| ens33、ens36 | 192.168.171.111 | mode 1    |

**注: ip地址配置在bond1 上，物理网卡无需配置IP地址**

```shell
#加载bonding模块，并确认已经加载
[root@web01 ~]# modprobe --first-time bonding
[root@web01 ~]# lsmod | grep bonding
bonding               141566  0 
#编辑bond1配置文件
[root@web01 ~]# cat > /etc/sysconfig/network-scripts/ifcfg-bond1 << EOF
> DEVICE=bond1
> TYPE=Bond
> IPADDR=192.168.171.111
> NETMASK=255.255.255.0
> GATEWAY=192.168.171.2
> DNS1=114.114.114.114
> DNS2=8.8.8.8
> USERCTL=no
> BOOTPROTO=none
> ONBOOT=yes
> EOF
#修改ens33配置文件
[root@web01 ~]# cat > /etc/sysconfig/network-scripts/ifcfg-ens33 << EOF
> DEVICE=ens33
> TYPE=Ethernet
> ONBOOT=yes
> BOOTPROTO=none
> DEFROUTE=yes
> IPV4_FAILURE_FATAL=no
> NMAE=ens33
> MASTER=bond1               # 需要和上面的ifcfg-bond0配置文件中的DEVICE的值一致
> SLAVE=yes
> EOF
#修改ens36配置文件
[root@web01 ~]# cat > /etc/sysconfig/network-scripts/ifcfg-ens36 << EOF
> DEVICE=ens36
> TYPE=Ethernet
> ONBOOT=yes
> BOOTPROTO=none
> DEFROUTE=yes
> IPV4_FAILURE_FATAL=no
> NMAE=ens36
> MASTER=bond1
> SLAVE=yes
> EOF

# 配置bonding
[root@web01 ~]# cat >> /etc/modules-load.d/bonding.conf << EOF
> alias bond1 bonding
> options bonding mode=1 miimon=200           # 加载bonding模块，对外虚拟网络接口设备为 bond1
> EOF

#重启网卡使配置生效
[root@web01 ~]# systemctl restart network                  # 如果重启失败，则说明bond没配置成功
```

**注：如果配置完毕后重启网卡服务一直启动失败，而且日志里面也检查不出错误来，可以关闭NetworkManager后再次重启网卡试试** **重启网络后查看各个网卡的信息**

```shell
[root@web01 ~]# ip a show ens33
2: ens33: <BROADCAST,MULTICAST,SLAVE,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast master bond1 state UP group default qlen 1000
    link/ether 00:0c:29:9f:33:9f brd ff:ff:ff:ff:ff:ff
[root@web01 ~]# ip a show ens36
3: ens36: <BROADCAST,MULTICAST,SLAVE,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast master bond1 state UP group default qlen 1000
    link/ether 00:0c:29:9f:33:9f brd ff:ff:ff:ff:ff:ff
[root@web01 ~]# ip a show bond1
7: bond1: <BROADCAST,MULTICAST,MASTER,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether 00:0c:29:9f:33:9f brd ff:ff:ff:ff:ff:ff
    inet 192.168.171.111/24 brd 192.168.171.255 scope global noprefixroute bond1
       valid_lft forever preferred_lft forever
    inet6 fe80::20c:29ff:fe9f:339f/64 scope link 
       valid_lft forever preferred_lft forever
```

**查看bond1相关信息**

```shell
#查看bond1的接口状态
[root@web01 ~]# cat /proc/net/bonding/bond1               
Ethernet Channel Bonding Driver: v3.7.1 (April 27, 2011)

Bonding Mode: load balancing (round-robin)            # 绑定模式
MII Status: up           # 接口状态
MII Polling Interval (ms): 100
Up Delay (ms): 0
Down Delay (ms): 0

Slave Interface: ens33              # 备用接口: ens33
MII Status: up               # 接口状态
Speed: 1000 Mbps                 # 端口速率
Duplex: full
Link Failure Count: 0
Permanent HW addr: 00:0c:29:9f:33:9f              # 接口永久MAC地址
Slave queue ID: 0

Slave Interface: ens36           # 备用接口: ens36
MII Status: up
Speed: 1000 Mbps
Duplex: full
Link Failure Count: 0
Permanent HW addr: 00:0c:29:9f:33:a9
Slave queue ID: 0
```

**当做到这一步的时候，ens33或ens36中任意一块网卡down掉，都不会影响通信**

### bond 解绑

```shell
rm -rf /etc/sysconfig/network-scripts/ifcfg-bond0

rm -rf /etc/modprob.d/bonding.conf

rm -rf /etc/sysconfig/network-scripts/ifcfg-ens33

rm -rf /etc/sysconfig/network-scripts/ifcfg-ens37

rmmod bonding   #这里操作，系统无法链接，可以不卸载bonding驱动

systemctl restart network
```

## 静态网络配置

- ifconfig：IP地址、子网掩码、网关信息

![](../../../blogimgv2022/image-20220810195918097.png)

- route -n：路由表和网关信息

![](../../../blogimgv2022/image-20220810195936933.png)

- 修改配置文件

```shell
vim /etc/sysconfig/network-scripts/ifcfg-eth0
DEVICE=eth0   #配置的网卡名称，通过ifconfig查看
BOOTPROTO=static  #如设置为none则禁用网卡，static则启用静态IP地址，设置为dhcp则为开启DHCP服务
ONBOOT=yes   #开机自启动网卡
IPADDR=192.168.X.68
NETMASK=255.255.255.0
GATEWAY=192.168.X.253
DNS1=[$DNS1]
DNS2=[$DNS2]
```

- DNS解析服务

```shell
#如需DNS解析服务，则可以在配置网卡文件时加入DNS1、DNS2等等，或修改 “/etc/resolv.conf”文件
IPADDR=192.168.1.200
GATEWAY=192.168.1.1
PREFIX=24
DNS1=114.114.114.114
#或
# vi /etc/resolv.conf
nameserver 114.114.114.114
```

- 重启网络：systemctl restart network

## 动态IP

```shell
DEVICE=ens33                         # 网卡的设备名称
NAME=ens33                           # 网卡设备的别名
TYPE=Ethernet                        #网络类型：Ethernet以太网
BOOTPROTO=none                       #引导协议：static静态、dhcp动态获取、none不指定（可能出现问题
DEFROUTE=yes                         #启动默认路由
IPV4_FAILURE_FATAL=no                #不启用IPV4错误检测功能
IPV6INIT=yes                         #启用IPV6协议
IPV6_AUTOCONF=yes                    #自动配置IPV6地址
IPV6_DEFROUTE=yes                    #启用IPV6默认路由
IPV6_FAILURE_FATAL=no                #不启用IPV6错误检测功能
UUID=sjdfga-asfd-asdf-asdf-f82b      #网卡设备的UUID唯一标识号
ONBOOT=yes                           #开机自动启动网卡
DNS=114.114.114.114                  #DNS域名解析服务器的IP地址 可以多设置一个DNS1
IPADDR=192.168.1.22                  #网卡的IP地址
PREFIX=24                            #子网前缀长度
GATEWAY=192.168.1.1                  #默认网关IP地址
IPV6_PEERDNS=yes
IPV6_PEERROUTES=yes
IPADDR=192.168.1.22                  #你想要设置的固定IP，理论上192.168.2.2-255之间都可以，请自行验证；如果是dhcp可以不填写
NETMASK=255.255.255.0                #子网掩码，不需要修改；
```

- 一般需要修改的几个

```shell
BOOTPROTO=static        #static静态、dhcp动态获取、none不指定（可能出现问题）
ONBOOT=yes              #特别注意 这个是开机启动,需要设置成yes
DNS1=8.8.8.8            #DNS域名解析服务器的IP地址
IPADDR=192.168.1.2      #网卡的IP地址
GATEWAY=192.168.1.1     #网关地址
NETMASK=255.255.255.0   #子网掩码
```

## 网络配置文件和参数介绍

![](../../../blogimgv2022/image-20220810200831456.png)

![](../../../blogimgv2022/image-20220810201026351.png)

## Resource

- https://cloud.tencent.com/developer/article/1688618


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E7%BD%91%E7%BB%9Cbond%E6%A8%A1%E5%BC%8F/  

