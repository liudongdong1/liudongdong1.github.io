# 虚拟化技术介绍


> `虚拟化技术（virtualization technology，VT）`是云计算的基础。简单的说，虚拟化使得在一台物理的服务器上可以跑多台虚拟机，虚拟机共享物理机的 `CPU`、`内存`、`IO`、`网络` 等硬件资源，但逻辑上虚拟机之间是相互隔离的。
>
> - 系统虚拟化，存储虚拟化，网络虚拟化，GPU 虚拟化，硬件支持虚拟化
> - 通过 virt-what 查看具体的虚拟化技术

> 在 ESXi 中，所有虚拟化功能都在内核实现。Xen 内核仅实现 CPU 与内存虚拟化， IO 虚拟化与调度管理由 Domain0（主机上启动的第一个管理 VM）实现。KVM 内核实现 CPU 与内存虚拟化，QEMU 实现 IO 虚拟化，通过 Linux 进程调度器实现 VM 管理。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503203132014.png)

### 1. 虚拟化类型

> `物理机` 将自己的硬件资源虚拟化，并提供给 `Guest` 一般通过 `Hypervisor` 的程序实现。`Hypervisor` 是为客户端操作系统提供虚拟机环境的软件，根据实现方式和所处的位置，分为两种

#### .1.1 型虚拟化

`Hypervisor` 直接安装在物理机上，多个虚拟机在 `Hypervisor` 上运行。`Hypervisor` 实现方式一般是一个特殊定制的 `Linux` 系统。`Xen` 和 `VMWare` 的 `ESXi` 都属于这个类型。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503155756205.png)

#### .2. 2 型虚拟化

物理机上首先安装常规的操作系统，比如 `Redhat`、`Ubuntu` 和 `Windows`。`Hypervisor` 作为 OS 上的一个`程序`模块运行，并对管理虚拟机进行管理。`KVM`、`VirtualBox` 和 `VMWare Workstation` 都属于这个类型。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/2.png)

> - aws : Amazon Web Services
> - docker : This is a Docker container.
> - hyperv : This is Microsoft Hyper-V hypervisor.
> - lxc : This process is running in a Linux LXC container.
> - kvm : This guest is running on the KVM hypervisor using hardware acceleration.
> - openvz : The guest appears to be running inside an OpenVZ or Virtuozzo container.
> - parallels : The guest is running inside Parallels Virtual Platform (Parallels Desktop, Parallels Server).
> - qemu : This is QEMU hypervisor using software emulation.
> - virt : Some sort of virtualization appears to be present, but we are not sure what it is.
> - virtualbox : This is a VirtualBox guest.
> - vmware : The guest appears to be running on VMware hypervisor.
> - xen The guest appears to be running on Xen hypervisor.
> - xen-hvm : This is a Xen guest fully virtualized (HVM).

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503210104119.png)

### 2. 硬件虚拟化

#### .1. CPU 虚拟化

物理服务器上通常配置 2 个物理 pCPU（Socket），每个 CPU 有多个核（core）；开启超线程 Hyper-Threading 技术后，每个 core 有 2 个线程（Thread）；在虚拟化环境中一个 Thread 对应一个 vCPU。在 KVM 中每一个 VM 就是一个用户空间的 QEMU 进程，分配给 Guest 的 vCPU 就是该进程派生的一个线程 Thread，由 Linux 内核动态调度到基于时分复用的物理 pCPU 上运行。KVM 支持设置 CPU 亲和性，将 vCPU 绑定到特定物理 pCPU，如通过 libvirt 驱动指定从 NUMA 节点为 Guest 分配 vCPU 与内存。KVM 支持 vCPU 超分（over-commit）使得分配给 Guest 的 vCPU 数量超过物理 CPU 线程总量。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/v2-8d1b327ffe85af1d10dd8b4b42960345_720w.jpg)

> KVM 是依赖于硬件辅助的全虚拟化（如 Inter-VT、AMD-V），目前也通过 virtio 驱动实现半虚拟化以提升性能。Inter-VT 引入新的执行模式：VMM 运行在 VMX Root 模式， GuestOS 运行在 VMX Non-root 模式，执行特权指令时两种模式可以切换。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503203426483.png)



#### .2. 内存虚拟化

内存虚拟化的目的是给虚拟客户机操作系统提供一个从 0 地址开始的连续物理内存空间，同时在多个客户机之间实现隔离和调度。在虚拟化环境中，内存地址的访问会涉及如下四个概念：

1）`客户机虚拟地址，GVA`（Guest Virtual Address）

2）客户机物理地址，GPA（Guest Physical Address）

3）宿主机虚拟地址，HVA（Host Virtual Address）

4）`宿主机物理地址，HPA`（Host Physical Address）

##### 1. EPT&VPID

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503203614913.png)

内存虚拟化就是要将 GVA 转化为最终能够访问的 HPA。在没有硬件提供的内存虚拟化之前，系统通过**影子页表（**Shadow Page Table) 完成这一转化。内存的访问和更新通常是非常频繁的，要维护影子页表中对应关系会非常复杂，开销也较大。同时需要为每一个客户机都维护一份影子页表，当客户机数量较多时，其影子页表占用的内存较大也会是一个问题。

- Intel CPU 在硬件设计上就引入了 **EPT**（Extended Page Tables，扩展页表），从而`将 GVA 到 HPA 的转换通过硬件来实现`。这个转换分为两步。首先，`通过客户机 CR3 寄存器将 GVA 转化为 GPA，然后通过查询 EPT 来实现 GPA 到 HPA 的转化`。

- CPU 使用 TLB（Translation Lookaside Buffer）缓存线性虚拟地址到物理地址的映射，地址转换时 CPU 先根据 GPA 先查找 TLB，如果未找到映射的 HPA，将根据页表中的映射填充 TLB，再进行地址转换。不同 Guest 的 vCPU 切换执行时需要刷新 TLB，影响了内存访问效率。Intel 引入了 VPID（Virtual-Processor Identifier）技术`在硬件上为 TLB 增加一个标志，每个 TLB 表项与一个 VPID 关联`，`唯一对应一个 vCPU，当 vCPU 切换时可根据 VPID 找到并保留已有的 TLB 表项`，减少 TLB 刷新。

##### .2. THP

x86 CPU 默认使用 4KB 的内存页面，目前已经支持 2MB，1GB 的内存大页（Huge Page）。`使用大页内存可减少内存页数与页表项数，节省了页表所占用的 CPU 缓存空间，同时也减少内存地址转换次数，以及 TLB 失效和刷新的次数，从而提升内存使用效率与性能`。但使用内存大页也有一些弊端：如大页必须在使用前准备好；应用程序必须显式地使用大页（一般是调用 mmap、shmget 或使用 libhugetlbfs 库进行封装）；需要超级用户权限来挂载 hugetlbfs 文件系统；如果`大页内存没有实际使用会造成内存浪费等`。2009 年实现的透明大页 THP（Transparent Hugepage）技术创建了一个抽象层，能够自动创建、管理和使用传统大页，实现发挥大页优势同时也规避以上弊端。当前主流的 Linux 版本都默认支持。KVM 中可以在 Host 和 Guest 中同时使用 THB 技术。

##### .3.  内存超分

> 由于 Guest 使用内存时采用瘦分配按需增加的模式，KVM 支持内存超分（Over-Commit）使得`分配给 Guest 的内存总量大于实际物理内存总量`

1） **内存交换（Swapping）**：当系统内存不够用时，把部分长时间未操作的内存交换到磁盘上配置的 Swap 分区，等相关程序需要运行时再恢复到内存中。

2） **气球（Ballooning）**：通过 virtio_balloon 驱动实现动态调整 Guest 与 Host 的可用内存空间。气球中的内存是 Host 可使用，Guest 不能使用。当 Host 内存不足时，可以使气球膨胀，从而回收部分已分配给 Guest 的内存。当 Guest 内存不足时可请求压缩气球，从 Host 申请更多内存使用。

3） **页共享（Page Sharing）**：通过 KSM（Kernel Samepage Merging）让内核扫描正在运行进程的内存。如果发现完全相同的内存页就会合并为单一内存页，并标志位写时复制 COW（Copy On Write）。如果有进程尝试修改该内存页，将复制一个新的内存页供其使用。KVM 中 QEMU 通过 madvise 系统调用告知内核那些内存可以合并，通过配置开关控制是否企业 KSM 功能。Guest 就是 QEMU 进程，如果多个 Guest 运行相同 OS 或应用，且不常更新，使用 KSM 能大幅提升内存使用效率与性能。当然扫描和对比内存需要消耗 CPU 资源对性能会有一定影响。

#### .3. 设备虚拟化

（1）**设备模拟**：在虚拟机监控器中模拟一个传统的 I/O 设备的特性，比如在` QEMU` 中模拟一个 Intel 的千兆网卡或者一个 IDE 硬盘驱动器，在客户机中就暴露为对应的硬件设备。客户机中的 I/O 请求都由虚拟机监控器捕获并模拟执行后返回给客户机。

（2）**前后端驱动接口**：在虚拟机监控器与客户机之间定义一种全新的适合于虚拟化环境的交互接口，比如常见的 `virtio 协议`就是在客户机中暴露为 virtio-net、virtio-blk 等网络和磁盘设备，在 QEMU 中实现相应的 virtio 后端驱动。

（3）**设备直接分配**：将一个物理设备，如一个网卡或硬盘驱动器直接分配给客户机使用，这种情况下 I/O 请求的链路中很少需要或基本不需要虚拟机监控器的参与，所以性能很好。

（4）**设备共享分配**：其实是设备直接分配方式的一个扩展。在这种模式下，一个（具有特定特性的）物理设备可以支持多个虚拟机功能接口，可以将虚拟功能接口独立地分配给不同的客户机使用。如 SR-IOV 就是这种方式的一个标准协议。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503204219179.png)

### 3. KVM 概述

> KVM 就是在硬件辅助虚拟化技术之上构建起来的**虚拟机监控器**。当然，并非要所有这些硬件虚拟化都支持才能运行 KVM 虚拟化，KVM 对硬件**最低的依赖是** CPU 的硬件虚拟化支持（也就是说，KVM 可以运行在所有支持 CPU 虚拟化的机器上）

（1）**KVM 内核模块**，它属于标准 Linux 内核的一部分，是一个专门提供虚拟化功能的模块，主要负责` CPU 和内存的虚拟化`，包括：`客户机的创建、虚拟内存的分配、CPU 执行模式的切换、vCPU 寄存器的访问、vCPU 的执行`。

（2）**QEMU 用户态工具**，它是一个普通的 Linux 进程，为客户机提供`设备模拟的功能`，包括模拟 BIOS、PCI/PCIE 总线、磁盘、网卡、显卡、声卡、键盘、鼠标等。同时它通过 ioctl 系统调用与内核态的 KVM 模块进行交互。

KVM 是在硬件虚拟化支持下的**完全虚拟化技术**，所以它能支持在**相应硬件**上能运行的几乎所有的操作系统，x86 下如：Linux、Windows、FreeBSD、MacOS 等。KVM 的基础架构如图所示。在 KVM 虚拟化架构下，**每个客户机就是一个 QEMU 进程**，在一个宿主机上有多少个虚拟机就会有多少 QEMU 进程；客户机中的每一个虚拟 CPU 对应 QEMU 进程中的一个执行线程；一个宿主机中只有一个 KVM 内核模块，所有客户机都与这个内核模块进行交互。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503202834999.png)

### Resource

- https://www.xiexianbin.cn/virtualization-technology/index.html?to_index=1
- https://zhuanlan.zhihu.com/p/105499858
- https://github.com/yifengyou/learn-kvm  to learning

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E8%99%9A%E6%8B%9F%E5%8C%96%E6%8A%80%E6%9C%AF%E4%BB%8B%E7%BB%8D/  

