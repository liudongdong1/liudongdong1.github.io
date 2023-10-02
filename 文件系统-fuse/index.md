# 文件系统Fuse


> - filesystem types: 文件系统类型
> - superblock: 整个文件系统的元信息
> - inode: 单个文件的元信息
> - dentry: 目录项，一个文件目录对应一个dentry
> - file: 进程打开的一个文件

![](https://img-blog.csdnimg.cn/c53c328d30884e10843dbed72890bc64.png)

> - `基于块设备的文件系统(Block-based FS)` ：ext2-4, btrfs, ifs, xfs, iso9660, gfs, ocfs, …基于物理存储设备的文件系统，用来管理设备的存储空间
> - `网络文件系统(Network FS) `：NFS, coda, smbfs, ceph, …用于访问网络中其他设备上的文件。网络文件系统的目标是网络设备，所以它不会调用系统的Block层
> - `伪文件系统(Pseudo FS)` ：proc, sysfs, pipefs, futexfs, usbfs, …因为并不管理真正的存储空间，所以被称为伪文件系统。它组织了一些虚拟的目录和文件，通过这些文件可以访问系统或硬件的数据。它不是用来存储数据的，而是`把数据包装成文件用来访问`，所以不能把伪文件系统当做存储空间来操作。
> - `特殊文件系统(Special Purpose FS) `：tmpfs, ramfs, devtmpfs，特殊文件系统也是一种伪文件系统，它使用起来更像是一个磁盘文件系统，但`读写通常是内存而不是磁盘设备`。
> - `堆栈式文件系统(Stackable FS) `：ecryptfs(加密文件系统), overlayfs(不直接参与磁盘空间结构的划分，仅将原来文件系统中不同目录和文件进行“合并”), unionfs(联合文件系统）, wrapfs，叠加在其他文件系统之上的一种文件系统，本身不存储数据，而是对下层文件的扩展。
> - `用户空间文件系统(FUSE)`： 它提供一种方式可以让开发者在用户空间实现文件系统，而不需要修改内核。这种方式更加灵活，但效率会更低。FUSE 直接面向的是用户文件系统，也不会调用Block层。

# 1    概述

Fuse是filesystem in user space，`一个用户空间的文件系统框架`，允许`非特权用户建立功能完备的文件系统，而不需要重新编译内核`。`fuse模块仅仅提供内核模块的入口，而本身的主要实现代码位于用户空间中`。对于读写虚拟文件系统来讲，fuse是个很好的选择。fuse包含包含一个内核模块和一个用户空间守护进程，将大部分的VFS调用都委托一个专用的守护进程来处理。

# 2    工作原理

Fuse用户空间文件系统与真实的文件系统不同，`它的supper block, indoe, dentry等都是由内存虚拟而来`，具体在物理磁盘上存储的真实文件结构是什么，它不关心，且`对真实数据的请求通过驱动和接口一层层传递到用户空间的用户编写的具体实现程序里来`，这样就为用户开发自己的文件系统提供了便利，这也就是所谓的“用户空间文件系统”的基本工作理念。

## 2.1  模块架构

FUSE分为三大模块：

- FUSE内核模块（内核态）：FUSE内核模块实现VFS接口（实现fuse文件驱动模块的注册、fuse 的(虚拟)设备驱动、提供supper block、dentry、inode的维护），接收来至后者的请求,传递给LibFUSE，LibFUSE再传递给我们用户程序的接口进行实现操作。
- LibFUSE模块（用户态）：用户程序在用户空间实现LibFUSE库封装的文件系统操作；LibFUSE实现文件系统主要框架、对“用户实现的文件系统操作代码“的封装、mount管理、通过字符设备/dev/fuse与内核模块通信；
- 用户程序模块（用户态）

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/ebabcdb3fe854af5bf19e011c53d3b1e.png)

- 一个内核模块： 加载时被注册成 Linux 虚拟文件系统的一个 fuse 文件系统驱动。实现和VFS的对接，实现一个能被用户空间打开的设备。该块设备作为fuse daemon与内核通信的桥梁，fuse daemon通过/dev/fuse读取fuse request，处理后将reply写入/dev/fuse。

- 基于用户空间库libfuse一个用户空间守护进程（下文称fuse daemon）：负责和内核空间通信，接收来自/dev/fuse的请求，并将其转化为一系列的函数调用，将结果写回到/dev/fuse
- 挂载工具：实现对用户态文件系统的挂载

> - IO先进内核，经过VFS传递给内核的FUSE文件系统模块
> - 内核FUSE模块把请求发送给用户态，由hello程序接受并处理，处理完成后，响应原路返回

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/79fd78fab286405d89b8e79e91401a6a.png)

内核 FUSE 模块在内核态中间做协议封装和协议解析的工作，它接收从VFS下来的请求并按照 FUSE 协议转发到用户态，然后接收用户态的响应，并随后回复给用户。FUSE在这条IO路径是做了一个透明中转站的作用，用户完全不感知这套框架。内核 fuse.ko用于接收VFS下来的IO请求，然后封装成 FUSE 数据包，转发给用户态，其内核也是一个文件系统，其满足文件系统的几个数据结构

- fs/fuse/inode.c —> 主要完成fuse文件驱动模块的注册，提供对supper block的维护函数以及其它(驱动的组织开始文件)
- fs/fuse/dev.c —> fuse 的(虚拟)设备驱动
- fs/fuse/control.c —> 提供对于dentry的维护及其它
- fs/fuse/dir.c —> 主要提供对于目录inode索引节点的维护
- fs/fuse/file.c —> 主要提供对于文件inode索引节点的维护

# Resource

- http://scm.zoomquiet.top/data/20110504180609/index.html
- https://blog.csdn.net/u012489236/article/details/125116724

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F-fuse/  

