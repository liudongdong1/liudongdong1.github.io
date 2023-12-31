# 云计算课程


> 计算设备也称为计算资源，计算资源包括 CPU、内存、硬盘和网络。而在机房中，磁盘只是存储大类中的一种，存储还包括磁带库、阵列、SAN、NAS 等，这些统称为存储资源。另外，CPU、内存只是服务器的部件，我们统一用服务器资源来代替 CPU 和内存资源的说法。云计算引入了一种全新的方便人们使用计算资源的模式，即云计算能让人们方便、快捷地自助使用远程计算资源。

#### 1. IT 系统组成

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403122608142.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403122811184.png)

#### 2. 3种服务模式

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403123003378.png)

#### 3. 云计算架构参考模型

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403123439881.png)

#### 4. 一套完整云计算产品需要解决哪些问题

- 虚拟化平台（硬件、虚拟软件）—— 解决如何运行虚拟机的问题。
- 管理工具 —— 解决如何管理大量虚拟机的问题，包括`创建、启动、停止、备份、迁移虚拟机，以及计算资源的管理和分配`。
- 交付部分 —— 解决如何让远端的用户使用虚拟机的问题。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403123654345.png)

#### 5. 关键技术- 存储技术

##### .1. 外部存储

> 存储和 CPU 不在同一台计算机上，如` SAN 和 NAS` 存储是单独的存储设备，它们通过以太网线或者光纤与计算机连接。专门的存储网络设备很贵，随着以太网速度越来越快，基于以太网的存储技术逐渐流行起来，如 iSCSI，10Gbit/s 的网卡能提供 1GB/s 的理论速度。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403124738109.png)

##### .2. 直接存储

> 存储直接接插到主板上，通过` PATA、SATA、mSATA、SAS、SCSI 或者 PCI-E 接口总线通信`。传统的机械硬盘一般采用 PATA、SATA、SAS、SCSI 接口，相对于外部存储，直接接插主板的机械硬盘的速度优势越来越不明显，但是固态硬盘（如 mSATA、PCI-E）的速度优势还是比较明显的，尤其是 PCI-E 的固态硬盘，代表着业界顶尖的存储技术。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403124836282.png)

##### .3. 分布式存储

> 通过分布式文件系统把各台计算机上的直接存储整合成一个大的存储，对参与存储的每台计算机来说，既有直接存储部分，也有外部存储部分，所以说分布式存储融合了前面两种存储方案。由于需要采用分布式文件系统来整合分散于各台计算机上的直接存储，使之成为单一的名字空间，所以所涉及的技术、概念和架构非常复杂，而且还要消耗额外的计算资源。
>
> 服务器存储局域网（Server SAN）逐渐被数据中心采用，而且发展很快，Ceph 分布式存储系统就属于 Server SAN，被很多云中心采用。目前的软件定义存储（SDS）概念就是分布式存储。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220403124947825.png)

---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/%E4%BA%91%E8%AE%A1%E7%AE%97%E8%AF%BE%E7%A8%8B/  

