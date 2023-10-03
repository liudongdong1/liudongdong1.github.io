# SPDK_paper


> [Ziye Yang](https://dblp.uni-trier.de/pid/126/0621.html), [James R. Harris](https://dblp.uni-trier.de/pid/78/3263.html), [Benjamin Walker](https://dblp.uni-trier.de/pid/56/11088.html), [Daniel Verkamp](https://dblp.uni-trier.de/pid/211/9050.html), [Changpeng Liu](https://dblp.uni-trier.de/pid/211/9167.html), [Cunyin Chang](https://dblp.uni-trier.de/pid/211/9116.html), [Gang Cao](https://dblp.uni-trier.de/pid/49/982.html), [Jonathan Stern](https://dblp.uni-trier.de/pid/07/9304.html), [Vishal Verma](https://dblp.uni-trier.de/pid/94/5953.html), [Luse E. Paul](https://dblp.uni-trier.de/pid/211/9057.html):**SPDK: A Development Kit to Build High Performance Storage Applications.** [CloudCom 2017](https://dblp.uni-trier.de/db/conf/cloudcom/cloudcom2017.html#YangHWVLCCSVP17): 154-161

------

# Paper: SPDK

<div align=center>
<br/>
<b>SPDK: A Development Kit to Build High Performance Storage Applications
</b>
</div>
### Summary

1. provide a set of tools and libraries for writing high performance, scalable, user-mode strong applications.
2. achieves high performance by moving the necessary drivers into user space and operating them in a polled mode instead of interrupt mode and lockless resource access.

### Proble Statement

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220427141726878.png)

-  strong demand on building high performance storage service upon emerging fast storage devices.
- most storage software stack become the bottle neck for developing high performance storage applications.
  - kernel I/O stacks, due to context switch, data copy, interrupt, resource synchronization.


### System Design and implementaion

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220427152603792.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220428153019098.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220428155416570.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220428155558990.png)

- **App scheduling:**  event framework for writing asynchronous, polled-mode, shared-nothing server applications
- **Drivers:** user space pooled mode NVMe driver, providing zero copy, highly parallel and direct access to NVMe SSDs.
- **Storages devices**: `abstracts the device exported by drivers` and `providers the user space block I/O interface` to storage applications above.
- **Storage Protocals:** contains the accelerated applications upon SPDK framework to  support various different storage protocols.

#### App scheduling

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220427143747174.png)

##### 1. Events

- **Target:** accomplish cross-thread communication while minimizing synchronization overhead
- **Traditional:** thread-per-connection server design, depends on OS to schedule many threads issuing blocking I/O onto limited number of cores.
- runs on events loop` thread(reactor)` per CPU core, to process incoming events from a queue, each event consists of a bundled function pointer and its arguments, destined for a particular CPU core.

##### 2. Reactor

- loop running on each core checks for incoming events and executes them in first-in, first-out order.

##### 3. Pollers

- functions with arguments that can bundled ad sent to a specific core to be executed.
- pollers are executed repeatedly until unregistered.

#### User space polled drivers

- moves the device driver implementation in user space instead of kernel space.
- use pooling instead of interrupt, allows users to determine how much CPU time for each task instead of letting the kernel scheduler decide.

##### .1. Asynchronous I/O mode

- call asynchronous read/write interface to send I/O requests, and use the corresponding I/O completion check functions to pull the completed I/Os.

##### .2. Lockless architecture

- adopts a lockless architecture which requires each thread to access its own resource, e.g., memory, I/O submission queue, I/O completion queue.

#### Storage service

##### .1. Blobstore

- a persistent, power-fail safe block allocator designed to be used as local storage system backing a higher level storage service.
- designed to allow asynchronous, uncacked, parallel reads and writes to groups of blocks on a block device called 'blob'.

##### .2. Blobfs

- Filenames are currently stored as xattrs in each blob, the filename lookup is an O(n) operation.  `SPDK btree`

##### .3. BDEV

- abstract the device identified by user space drivers and other third part libraries to export the block service interface to applications.
- ` a driver module API` for implementing bdev drivers which enumerate and claim SPDK block devices and performance operations (read, write, `unmap`, etc.) on those devices
- bdev drivers for NVMe, Linux AIO, Ceph RBD, blobdev

> Kim, H. J., Lee, Y. S., & Kim, J. S. (2016). {NVMeDirect}: A User-space {I/O} Framework for Application-specific Optimization on {NVMe}{SSDs}. In *8th USENIX Workshop on Hot Topics in Storage and File Systems (HotStorage 16)*. CCF-A [[link](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.usenix.org%2Fsystem%2Ffiles%2Fconference%2Fhotstorage16%2Fhotstorage16_kim.pdf#=&zoom=180)]   [[slide](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.usenix.org%2Fsites%2Fdefault%2Ffiles%2Fconference%2Fprotected-files%2Fhotstorage16_slides_kim.pdf#=&zoom=180)]

------

# Paper: NVMeDirect

<div align=center>
<br/>
<b>NVMeDirect: A User-space I/O Framework
for Application-specific Optimization on NVMe SSDs
</b>
</div>


### Summary

1. propose a novel user-level I/O framework called NVMeDirect, which improves the performance by allowing the user applicatins to access the storage device directly.
2. NVMeDirect can co-exist with legacy with legacy of I/O stack of the kernel, existing kernel based application can use the same NVMe SSD with NVMeDirect-enable applications simultanesously on different disk partitions.
3. provides `flexibility in queue management, I/O completion method, caching, and I/O scheduling` where each user application can select its own I/O policies according to its I/O characteristics and requirements.

### Proble Statement

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503151514674.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503151737965.png)

- as the storage devices are getting faster, the overhead of the legacy kernel I/O stack becomes noticeable since it has been optimized for slow HDDs.
  - the kernel should be general, so as to provides an abstraction layer for applications, managing all the hardware resources.
  - the kernel cann't implement any policy that favors a certain application because it should provide fairness among applications.
  - the frequent update of the kernel requires a constant effort to port such application-specific optimization.

### System Design

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503150031783.png)

- Admin tool: controls the kernel driver with the root privilege to manage the access permission of I/O queues.
- the kernal checks the premisions, and then creates the required submission queue and completion queue, and maps their memory regions and the associated doorbell registers to the user-space memory region of the application.
- a thread can create one or more I/O handles to access the queues and each handle can be bound to a dedicated queue or a shared queue. Each handle can be configured to use different features such as caching, I/O scheduling, and I/O completion.  (todo when a handle bound to a shared queue, how to solve data )

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503152204762.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220503152303769.png)

> Yang Z, Liu C, Zhou Y, et al. Spdk vhost-nvme: Accelerating i/os in virtual machines on nvme ssds via user space vhost target[C]//2018 IEEE 8th International Symposium on Cloud and Service Computing (SC2). IEEE, 2018: 67-76. [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8567374&tag=1)]

------

# Paper: SPDK vhost-NVMe

<div align=center>
<br/>
<b>SPDK vhost-NVMe: Accelerating I/Os in virtual machines on NVMe SSDs via uer space vhost target
</b>
</div>

### Summary

1. 

### Proble Statement

- 


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/spdk_paper/  

