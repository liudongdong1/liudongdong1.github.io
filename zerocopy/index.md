# VideoRelative


> 对于RocketMQ来说这两个步骤使用的是`mmap+write`，而Kafka则是使用`mmap+write`持久化数据，发送数据使用`sendfile`。

### 1. 传统IO

> 在开始谈零拷贝之前，首先要对传统的IO方式有一个概念。
>
> 基于传统的IO方式，底层实际上通过调用`read()`和`write()`来实现。
>
> 通过`read()`把数据从硬盘读取到内核缓冲区，再复制到用户缓冲区；然后再通过`write()`写入到`socket缓冲区`，最后写入网卡设备。

1. 用户进程通过`read()`方法向操作系统发起调用，此时上下文从用户态转向内核态
2. DMA控制器把数据从硬盘中拷贝到读缓冲区
3. CPU把读缓冲区数据拷贝到应用缓冲区，上下文从内核态转为用户态，`read()`返回
4. 用户进程通过`write()`方法发起调用，上下文从用户态转为内核态
5. CPU将应用缓冲区中数据拷贝到socket缓冲区
6. DMA控制器把数据从socket缓冲区拷贝到网卡，上下文从内核态切换回用户态，`write()`返回

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210320223909633.png)

### 2. 零拷贝

> 零拷贝技术是指计算机执行操作时，CPU不需要先将数据从某处内存复制到另一个特定区域，这种技术通常用于通过网络传输文件时节省CPU周期和内存带宽。

#### 2.1. mmap+write

> `mmap`主要实现方式是将读缓冲区的地址和用户缓冲区的地址进行映射，内核缓冲区和应用缓冲区共享，从而减少了从读缓冲区到用户缓冲区的一次CPU拷贝。

整个过程发生了**4次用户态和内核态的上下文切换**和**3次拷贝**，具体流程如下：

1. 用户进程通过`mmap()`方法向操作系统发起调用，上下文从用户态转向内核态
2. DMA控制器把数据从硬盘中拷贝到读缓冲区
3. **上下文从内核态转为用户态，mmap调用返回**
4. 用户进程通过`write()`方法发起调用，上下文从用户态转为内核态
5. **CPU将读缓冲区中数据拷贝到socket缓冲区**
6. DMA控制器把数据从socket缓冲区拷贝到网卡，上下文从内核态切换回用户态，`write()`返回

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210320224021132.png)

#### 2.2. sendfile

> `sendfile`是Linux2.1内核版本后引入的一个系统调用函数，通过使用`sendfile`数据可以直接在内核空间进行传输，因此避免了用户空间和内核空间的拷贝，同时由于使用`sendfile`替代了`read+write`从而节省了一次系统调用，也就是2次上下文切换。

整个过程发生了**2次用户态和内核态的上下文切换**和**3次拷贝**，具体流程如下：

1. 用户进程通过`sendfile()`方法向操作系统发起调用，上下文从用户态转向内核态
2. DMA控制器把数据从硬盘中拷贝到读缓冲区
3. CPU将读缓冲区中数据拷贝到socket缓冲区
4. DMA控制器把数据从socket缓冲区拷贝到网卡，上下文从内核态切换回用户态，`sendfile`调用返回

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210320224108385.png)

#### 2.3. sendfile+DMA Scatter/Gather

> 它将读缓冲区中的数据描述信息--内存地址和偏移量记录到socket缓冲区，由 DMA 根据这些将数据从读缓冲区拷贝到网卡，相比之前版本减少了一次CPU拷贝的过程

整个过程发生了**2次用户态和内核态的上下文切换**和**2次拷贝**，其中更重要的是完全没有CPU拷贝，具体流程如下：

1. 用户进程通过`sendfile()`方法向操作系统发起调用，上下文从用户态转向内核态
2. DMA控制器利用scatter把数据从硬盘中拷贝到读缓冲区离散存储
3. CPU把读缓冲区中的文件描述符和数据长度发送到socket缓冲区
4. DMA控制器根据文件描述符和数据长度，使用scatter/gather把数据从内核缓冲区拷贝到网卡
5. `sendfile()`调用返回，上下文从内核态切换回用户态

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210320224219006.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/zerocopy/  

