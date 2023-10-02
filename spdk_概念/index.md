# SPDK_概念


### 1. 相关知识

#### .1. Linux driver UIO

> A small kernel module to` set up the device`, `map device memory to user-space` and `register interrupts`. In many cases, the standard uio_pci_generic module included in the Linux kernel can provide the uio capability. For some devices which lack support for legacy interrupts, e.g. virtual function (VF) devices, the igb_uio module may be needed in place of uio_pci_generic.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503135539378.png)

**用户驱动工作流程**

01 - The kernel space UIO device driver(s) must be loaded before the user space driver is started (if using modules)

02 - The user space application is started and the UIO device file is opened (/dev/uioX where X is 0, 1, 2 ...)

- \- From user space, the UIO device is a device node in the file system just like any other device

03 - The device memory address information is found from the relevant sysfs directory, only the size is needed

04 - The device memory is mapped into the process address space by calling the mmap() function of the UIO driver

05 - The application accesses the device hardware to control the device

06 - The device memory is unmapped by calling munmap()

07 - The UIO device file is closed

```c
#define UIO_SIZE "/sys/class/uio/uio0/maps/map0/size"

int main(int argc, char **argv)
{
        int             uio_fd;
        unsigned int    uio_size;
        FILE            *size_fp;
        void            *base_address;

        /*
         * 1. Open the UIO device so that it is ready to use
         */
        uio_fd = open("/dev/uio0", O_RDWR);

        /*
         * 2. Get the size of the memory region from the size sysfs file
         *    attribute
         */
        size_fp = fopen(UIO_SIZE, O_RDONLY);
        fscanf(size_fp, "0x%08X", &uio_size);

        /*
         * 3. Map the device registers into the process address space so they
         *    are directly accessible
         */
        base_address = mmap(NULL, uio_size,
                           PROT_READ|PROT_WRITE,
                           MAP_SHARED, uio_fd, 0);

        // Access to the hardware can now occur ...

        /*
         * 4. Unmap the device registers to finish
         */
        munmap(base_address, uio_size);

        ...
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503135701273.png)

#### .2. VFIO

> 向用户态开放了 IOMMU 接口，通过 IOCTL 配置 IOMMU 将 DMA 地址空间映射并将其限制在进程虚拟地址空间。`IOMMU 提供了 IO 设备访问实际物理内存的一套机制`。在虚拟化领域，`内部实现了 guest 虚机内存地址和 host 内存地址的转换`

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503140332056.png)

#### .3. PCI BAR

> PCI 配置机制，包括寄存器配置帧头，设备编号（B/D/F）及对应的软硬件实现，最终实现 PCI 设备的寻址。

与其他I/O架构相比，PCI局部总线的主要改进之一是其配置机制。`除了正常的内存映射和I/O端口空间外`，`总线上的每个设备功能都有一个配置空间`，该空间长256字节，可通过了解设备的8位PCI总线、5位设备和3位功能编号（通常称为BDF或B/D/F，是总线/设备/功能的缩写）进行寻址。这允许多达256个总线，每个总线有多达32个设备，每个设备支持8种功能。单个PCI扩展卡可以作为一个设备响应，并且必须至少实现0号功能。配置空间的前64个字节是标准化的；其余部分可用于供应商定义的目的。

#### .4. MMIO

> MMIO 和 PMIO（port-mapped I/O）作为互补的解决方案实现了 CPU 和外围设备的 IO 互通。`IO 和内存使用相同的地址空间`，即 CPU 指令中的地址既可以指向内存，也可以指向特定的 IO 设备。`每个 IO 设备监控 CPU 的地址总线并对 CPU 对该地址的访问进行回应，同时连接数据总线至指定设备的硬件寄存器`，使得 CPU 指令可以像访问内存一样访问 IO 设备，类比于 DMA 的 memory-to-device，MMIO 是一种 cpu-to-device 的技术。

### 2. SPDK

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/webp-16515581918284.webp)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/13192585-50cc2d2e5782db65.png)

### Resource

- https://www.cnblogs.com/vlhn/p/7761869.html


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/spdk_%E6%A6%82%E5%BF%B5/  

