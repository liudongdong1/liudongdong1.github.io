# 存储架构


> 高性能，高可用，可伸缩，安全性，扩展性（功能的扩展）
>
> 分层设计（应用，服务，数据，管理，分析）
>
> 集：一个应用/模块/功能部署在多台物理机器上，通过负载均衡对外提供服务
>
> 异步：通过通知和轮询的发那个是告知请求方
>
> 缓存：将数据放在距离用户最近的位置
>
> 冗余：副本，纠删码
>
> 架构前端优化，应用层优化，代码层优化，存储层优化。
>
> - 前端优化：网站业务逻辑之前的部分；
> - 浏览器优化：减少Http请求数，使用浏览器缓存，启用压缩，Css Js位置，Js异步，减少Cookie传输；
> - CDN加速，反向代理；
> - 应用层优化：处理网站业务的服务器。使用缓存，异步，集
> - 代码优化：合理的架构，多线程，资源复用（对象池，线程池等），良好的数据结构，JVM调优，单例，Cache等；
> - 存储优化：缓存，固态硬盘，光纤传输，优化读写，磁盘冗余，分布式存储（HDFS），NOSQL等；

### 1. 存储使用方式分类

- **块存储**：体现形式是`卷或者硬盘`，主要操作对象是磁盘，将`裸磁盘空间整个映射给主机使用`。在此种方式下操作系统需要对挂载的裸硬盘进`行分区、格式化后，才能使用`。块存储无法进行文件共享。
- **文件存储**：`目录和文件`，数据以文件的方式存储和访问，按照目录结构进行组织。此种方式也需要`挂载，挂载后为一个目录，可直接存取其中的文件`；不需要格式化。
- **对象存储**：主要操作对象是对象 Object，本质上是键值对存储系统，不需要挂载，直接通过应用接口访问。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307160753118.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307160832539.png)

### 2. 存储协议

- **NFS(Network File System, 网络文件系统)**：主要应用于 Unix 环境下。通过使用 NFS，用户和程序可以象访问本地文件一样访问远端系统上的文件，使得每个计算机的节点能够像使用本地资源一样方便地使用网上资源。换言之，NFS 可用于`不同类型计算机、操作系统、网络架构和传输协议运行环境中的网络文件远程访问和共享`。**「针对共享文件存储。」**
- **CIFS（Common Internet File System，公共互联网文件系统)**: 主要应用在 NT/Windows 环境下，其工作原理是让 CIFS 协议运行于 TCP/IP 通信协议之上，让 Unix 计算机可以在网络邻居上被 Windows 计算机看到。**「针对共享文件存储。」**
- **ISCSI （Internet SCSI／SCSI over IP） ：**主要应用在 Windows 环境下，适用于 TCP／IP 通讯协议，是通过 TCP/IP 网络传输文件时的文件组织格式和数据传输方式。**「针对数据块存储。」**

### 3. 存储链接方式

#### .1. DAS(直接附加存储)

>`直接附加存储方式`与我们普通的 PC 存储架构一样，`外部存储设备都是直接挂接在服务器内部总线上`，数据存储设备是整个服务器结构的一部分，任何客户端想要访问存储设备上的资源就必须要通过服务器。
>
>**内部 DAS：**在内部 DAS 架构中，`存储设备通过服务器机箱内部的并行或串行总线连接到服务器上`。但是，物理的总线有距离限制，只能支持短距离的高速数据传输。此外，很多内部总线能连接的设备数目也有限，并且将存储设备放在服务器机箱内部，也会占用大量的空间 ，对服务器其它部件的维护造成困难。
>**外部 DAS：**在外部 DAS 结构中，`服务器与外部的存储设备直接相连`。在大多数情况下，他们之间`通过 FC 协议或者 SCSI 协议进行通信`。与内部 DAS 相比，外部 DAS 克服了内部 DAS 对连接设备的距离和数量的限制。另外，外部 DAS 还可以提供存储设备集中化管理，更加方便。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307145707889.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220409142916300.png)

#### .2. NAS（网络附加存储）

>在 NAS 存储结构中，存储系统`不再通过 I/O 总线附属于某个特定的服务期或客户机`，而是`直接通过网络接口直接与网络相连`，用户`通过网络访问`。NAS 实际上是带有一个 “瘦服务器” 的存储设备，作用类似于一个专用的文件服务器，而不是传统通用服务器，去掉了大多数功能，仅仅提供文件系统功能，用于存储服务。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307145904751.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220409143222083.png)

#### .3. SAN（存储区域网络）

>SAN (storage area network) 是一种`以网络为中心的存储结构`，不同于普通以太网，SAN 是位于服务器的后端，为连接服务器、磁盘阵列、带库等存储设备而建立的高性能**「专用网络（光纤通道）」**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307145944181.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220409143021150.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220409143124505.png)

#### .4. 存储模型比较

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220409135009600.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220409135439593.png)

### 4. 分布式存储架构

#### .1. 是否对称

##### .1. 对称式

> 在对称式架构中每个节点的角色均等，共同管理和维护元数据，节点间通过高速网络进行信息同步和互斥锁等操作。

![Swift典型架构](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307150515167.png)

##### .2. 非对称式

>有`专门的一个或者多个节点负责管理元数据`，其他节点需要频繁与元数据节点通信以获取`最新的元数据比如目录列表、文件属性`等等。（元数据节点与存储节点分离）

>FastDFS 采用的是非对称架构，分为 Tracker server 和 Storage server。
>
>- Tracker server 作为中心结点，管理集群拓扑结构，其主要作用是[负载均衡](https://cloud.tencent.com/product/clb?from=10680)和调度。
>
>- Storage server 以`卷为单位组织`，一个卷内包含多台 storage 机器，每个卷中的服务器是镜像关系，数据互为备份，存储空间以卷内容量最小的 storage 为准，所以建议 group 内的多个 storage 尽量配置相同，以免造成存储空间的浪费。

![FastDFS 架构](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220307150344740.png)

#### .2. P2P 网络存储

> P2P 网络存储技术的应用使得内容不是存在几台主要的服务器中，而是存在所有用户的个人电脑中。这就为网络存储提供了可能性，可以将`网络中的剩余存储空间利用起来，实现网络存储`。人们对存储容量的需求是无止境的，提高存储能力的方法有更换能力更强的存储器。另外就是把多个存储器用某种方式连接在一起，实现网络并行存储。相对于现有的网络储存系统而言，应用 P2P 技术将会有更大的优势。P2P 技术的主体就是网络中 Peer，也就是各台客户机，数量很大的。这些客户机的空闲存储空间很多，把这些空间利用起来实现网络存储。

#### 分布式存储种类

##### 中间控制节点架构 -hdfs

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723231819018.png)

##### 完全无中心架构-ceph

- 客户端是通过设备映射关系计算出要写入数据的位置，客户端与中心节点直接通信

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723232019679.png)

##### 完全无中心架构-swift

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723232248302.png)

### 5. 分布式理论

#### 一致性与可用性

- 时间一致性：所有的数据组件的数据在任意时刻都是完全一致的
- 事务一致性：只能存在事务开始前，或者事务开始后
- 线性一致性

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723234131842.png)

- 顺序一致性：不同的处理器对变量的写操作必须在所有的处理器上以相同的顺序看到

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723234438247.png)

- 因果一致性

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723234504009.png)

- 最终一致性： 

- 可用性： 如负载均衡，Web服务器，数据服务器等

##### CAP 理论

- 一致性，可用性，分区容错性
- 从服务器角度：如何尽快的将更新的数据分布到整个系统，降低达到最终一致性时间窗口
- 从客户端角度：多进程并发访问时，非分布数据库要求更新过的数据能被后续的访问都能看到。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220723235357495.png)

#### 数据分布

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly92aWdvdXJ0eXktemhnLmJsb2cuY3Nkbi5uZXQ=,size_16,color_FFFFFF,t_70.png)

##### 哈希分布

- 传统hash：访问数据时，先计算hash值，然后查询元数据服务器，获得该hash值对于的服务器
- 一致性hash：ring hash, jump hash
- 缺点：破坏数据的有序性，只支持随即读取操作

##### 顺序分布

- 将大表顺序划分为连续的范围，每个范围成为一个子表， 类似btree

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220724000425516.png)

##### crush 分布

- 通过crush算法将PG映射到一组OSD中，最后把x存放到对应的OSD中。
- crush  算法当增加和删除节点时，怎么进行处理

#### 数据复制

##### 强同步复制

- 同步日志操作：主部分将操作日至同步到备副本，备副本回放操作日志，完成后通知主副本，接着主副本修改本机，知道所有操作都完成后通知客户端写入成功。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220724001021109.png)

##### 异步复制

- 主副本不需要等待被副本回应，只需要本地修改成功就可以告知客户端写操作成功，另外通知异步机制，比如单独的复制线程将修改操作推送到其他副本。

##### NWR复制

- 基于写多个存储设备协议 （Replicated-write protocal）。N： 副本数量， w：写操作副本数量，R为读操作副本数量。
- 该协议中不再区分主从设备

#### 分布式协议

#### 跨机房部署

### 6. 分布式文件系统

### Google 文件系统

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220724001813510.png)

#### Taobao 文件系统

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220724002153112.png)

#### CDN 内容分发网络

### 7. 分布式键值系统

#### Amazon Dynamo

#### Taobao Tiar

#### ETCD

### 8. 架构设计角度

#### 可伸缩架构

- 通过添加/减少硬件（服务器）的方式，提高/降低系统的处理能力。
- 应用层：对应用进行`垂直或水平切分`。然后`针对单一功能进行负载均衡`（DNS,HTTP[反向代理],IP,链路层）。

- 服务层：与应用层类似；
- 数据层：`分库，分表，NOSQL等`；常用算法Hash，一致性Hash。

#### 可扩展架构

- 模块化，组件化：高内聚，内耦合，提高复用性，扩展性。
- 稳定接口：定义稳定的接口，在接口不变的情况下，内部结构可以“随意”变化。
- 设计模式：应用面向对象思想，原则，使用设计模式，进行代码层面的设计。
- 消息队列：模块化的系统，通过消息队列进行交互，使模块之间的依赖解耦。
- 分布式服务：公用模块服务化，提供其他系统使用，提高可重用性，扩展性。

#### 高可用架构

- 不同层级使用的策略不同，一般采用`冗余备份和失效转移`解决高可用问题。
- 应用层：一般设计为`无状态的`，对于每次请求，使用哪一台服务器处理是没有影响的。一般使用`负载均衡`技术（需要解决Session同步问题），实现高可用。
- 服务层：`负载均衡`，`分级管理`，`快速失败（超时设置）`，`异步调用`，`服务降级`，`幂等设计`等。
- 数据层：`冗余备份`（冷，热备[同步，异步]，温备），`失效转移`（确认，转移，恢复）。数据高可用方面著名的理论基础是CAP理论（持久性，可用性，数据一致性[强一致，用户一致，最终一致]）

#### 大型架构设计

- 客户层：支持PC浏览器和手机APP。差别是手机APP可以直接访问通过IP访问，反向代理服务器。
- **前端层**：使用`DNS负载均衡`，`CDN本地加速`以及`反向代理服务`；
- **应用层**：网站应用集；按照`业务进行垂直拆分`，比如商品应用，会员中心等；
- **服务层**：提供公用服务，比如用户服务，订单服务，支付服务等；
- **数据层**：支持关系型数据库集（支持读写分离），NOSQL集，分布式文件系统集；以及分布式Cache；
- **大数据存储层**：支持应用层和服务层的日志数据收集，关系数据库和NOSQL数据库的结构化和半结构化数据收集；
- **大数据处理层**：通过Mapreduce进行离线数据分析或Storm实时数据分析，并将处理后的数据存入关系型数据库。（实际使用中，离线数据和实时数据会按照业务要求进行分类处理，并存入不同的数据库中，供应用层或服务层使用）。

## Resource

- https://cloud.tencent.com/developer/article/1703986
- https://cuichongxin.blog.csdn.net/article/details/113616740
- http://www.360doc.com/content/21/0602/08/57557960_980065610.shtml todo 进一步完善

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%AD%98%E5%82%A8%E6%9E%B6%E6%9E%84/  

