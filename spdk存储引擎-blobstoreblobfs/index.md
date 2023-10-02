# SPDK存储引擎-Blobstore&BlobFs


>Blobstore是位于SPDK bdev之上的Blob管理层，用于与用户态文件系统Blobstore Filesystem （BlobFS）集成，从而代替传统的文件系统，支持更上层的服务，如数据库MySQL、K-V存储引擎Rocksdb以及分布式存储系统Ceph、Cassandra等。**以Rocksdb为例，通过BlobFS作为Rocksdb的存储后端的优势在于，I/O经由BlobFS与Blobstore下发到bdev，随后由SPDK用户态driver写入磁盘。整个I/O流从发起到落盘均在用户态操作，完全bypass内核。**
>
>BlobFS与Blobstore的关系可以理解为Blobstore实现了对Blob的管理，包括Blob的分配、删除、读取、写入、元数据的管理等，而BlobFS是在Blobstore的基础上进行封装的一个轻量级文件系统，用于提供部分对于文件操作的接口，并将对文件的操作转换为对Blob的操作，BlobFS中的文件与Blobstore中的Blob一一对应。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811215740086.png)

## Blobstore

### 块组织结构

 在blobstore中，将SSD中的块划分为多个抽象层，主要由Logical Block、Page、Cluster、Blob组成.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811213455418.png)

- Logical Block：与块设备中所提供的逻辑块相对应，通常为512B或4KiB。
- Page：由多个连续的Logical Block构成，通常一个page的大小为4KiB，因此一个Page由八个或一个Logical Block构成，取决于Logical Block的大小。在Blobstore中，Page是连续的，即从SSD的LBA 0开始，多个或一个块构成Page 0,接下来是Page 1，依次类推。
- Cluster：由多个连续的Page构成，通常一个Cluster的大小默认为1MiB，因此一个Cluster由256个Page构成。Cluster与Page一样，是连续的，即从SSD的LBA 0开始的位置依次为Cluster 0到Cluster N。
- Blob：Blobstore中主要的操作对象为Blob，与BlobFS中的文件相对应，提供read、write、create、delete等操作。一个Blob由多个Cluster构成，但构成Blob中的Cluster并不一定是连续的。

### 块管理分配

- cluster0用于存放Blobtore的`所有信息以及元数据`，对每个`blob数据块的查找`、`分配`都是依赖cluster 0中所记录的元数据所进行的。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811214105065.png)

- super block： `cluster的大小`、`已使用page的起始位置`、`已使用page的个数`、`已使用cluster的起始位置`、`已使用cluster的个数`、`Blobstore的大小`等信息。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811214232773.png)

- `Metadata Page Allocation`：用于记录所有`元数据页的分配`情况。在分配或释放元数据页后，将会对metadata page allocation中的数据做相应的修改。
- `Cluster Allocation`：用于记录所有cluster的分配情况。在分配新的cluster或释放cluster后会对cluster allocation中的数据做相应的修改。
- `Blob Id Allocation`：用于记录blob id的分配情况。对于blobstore中的所有blob，都是通过唯一的标识符blob id将其对应起来。在元数据域中，将会在blob allocation中记录所有的blob id分配情况。
- `Metadata Pages Region`：元数据页区域中存放着`每个blob的元数据页`。`每个blob中所分配的cluster都会记录在该blob的元数据页中`，在读写blob时，首先会通过blob id定位到该blob的元数据页，其次根据元数据页中所记录的信息，检索到对应的cluster。对于每个blob的元数据页，并不是连续的。

> 当对blob进行写入时，首先会为其分配cluster，其次更新该blob的metadata page，最后将数据写入，并持久化到磁盘中。
>
> 对于每个blob，通过相应的结构维护当前使用的cluster以及metadata page的信息：clusters与pages。Cluster中记录了当前该blob所有cluster的LBA起始地址，pages中记录了当前该blob所有metadata page的LBA起始地址。

## BlobFs

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811220058695.png)

> Blob FS在工作状态下会注册两个spdk_io_device，分别是blobfs_md以及blobfs_sync，前者带有md后缀，在Blob FS框架下，这被设计为与元数据(metadata)的操作有关，例如create，后者则是与I/O(read & write)操作有关。`对元数据的操作只能经由reactor 0实现`，其他用户线程或者绑定在其他线程中的reactor对元数据的操作均需要通过SPDK中的消息机制来实现，交由reactor 0来进行处理

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811220140246.png)

### Cache 机制

> DPDK提供的大页管理是SPDK实现零拷贝机制的基础，实际上这也是Blob FS中cache机制的基础。借助DPDK内存池对大页的管理，一次性申请到了一块较大的缓冲区mempool，该区域除了头部以外主要由固定大小的内存单元组成，并构成了ring队列，可方便的用于存取数据。

| rte_mempool_creat(count  ,element…) | spdk_mempool_create(count  ,element…) | 利用大页，创建一个内存池，内存池中存放有一定数量的固定大小内存单元 |
| ----------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| rte_mempool_get                     | spdk_mempool_get                      | 获取内存池中的一个内存单元                                   |
| rte_mempool_put                     | spdk_mempool_put                      | 将不再使用的内存单元放回内存池中                             |

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811220425171.png)

> g_cache_pool_thread。它主要维护了`g_caches`和`g_cache_pool`两个数据结构，前者维护了`所有拥有cache的文件列表`，后者则指向了前文提到的借助`DPDK大页管理所申请到的内存池`。

### 文件写流程

![image-20220811215351863](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811215351863.png)

### 文件读流程

- 内存中cache buffer：在文件读写时，首先会进行read ahead操作，将一部分数据从磁盘预先读取到内存的buffer中。
- buffer node存储真实的数据，其他层用于构建树的索引，值为offset

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811215104528.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220811214829192.png)



## Resource

- https://mp.weixin.qq.com/s?__biz=MzI3NDA4ODY4MA==&mid=2653335159&idx=1&sn=014beb5a0b3e5c74359a1f1910ceef9f&chksm=f0cb59f0c7bcd0e6d535ee32d5f9547f0fba36d5f42f8bc48958c75cb55d6f8ba291724b49e6&scene=21#wechat_redirect
- https://mp.weixin.qq.com/s/TvoAs4DqX1xiqmoPrKQnCg

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/spdk%E5%AD%98%E5%82%A8%E5%BC%95%E6%93%8E-blobstoreblobfs/  

