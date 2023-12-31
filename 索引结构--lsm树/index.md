# 存储结构--LSM树


> LSM树并不像B+树、红黑树一样是一颗严格的树状数据结构，它其实是一种存储结构，目前HBase,LevelDB,RocksDB这些NoSQL存储都是采用的LSM树。
>
> LSM树的核心特点是利用顺序写来提高写性能，但因为分层(此处分层是指的分为内存和文件两部分)的设计会稍微降低读性能，但是通过牺牲小部分读性能换来高性能写，使得LSM树成为非常流行的存储结构。
>
> LSM 树通过尽可能减少写磁盘次数，实际落地存储的数据按 key 划分，形成有序的不同文件；结合其 “ 先内存更新后合并落盘 ” 的机制，尽量达到顺序写磁盘，尽可能减少随机写；对于读则需合并磁盘已有历史数据和当前未落盘的驻于内存的更新。LSM 树存储支持有序增删改查，写速度大幅提高，但随机读取数据时效率低。

## **LSM树的核心思想**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-37576525d52091fd713bb13556c92861_720w-16619597014363.jpg)

### MemTable

> MemTable是在***内存\***中的数据结构，用于保存最近更新的数据，会按照Key有序地组织这些数据，LSM树对于具体如何组织有序地组织数据并没有明确的数据结构定义, 比如红黑树、 map 之类，甚至可以是跳表，例如Hbase使跳跃表来保证内存中key的有序。
>
> 因为数据暂时保存在内存中，内存并不是可靠存储，如果断电会丢失数据，因此通常会`通过WAL(Write-ahead logging，预写式日志)的方式来保证数据的可靠性`。

### Immutable MemTable

> 当 MemTable达到一定大小后，会转化成Immutable MemTable。Immutable MemTable是将转MemTable变为SSTable的一种中间状态。`写操作由新的MemTable处理，在转存过程中不阻塞数据更新操作`。

### SSTable(Sorted String Table)

> 有序键值对集合，是LSM树组在***磁盘\***中的数据结构。为了加快SSTable的读取，可以通过建立key的索引以及布隆过滤器来加快key的查找。

![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-9eeda5082f56b1df20fa555d36b0e0ae_720w.png)

这里需要关注一个重点，LSM树(Log-Structured-Merge-Tree)正如它的名字一样，LSM树会`将所有的数据插入、修改、删除等操作记录(注意是操作记录)保存在内存之中`，`当此类操作达到一定的数据量后，再批量地顺序写入到磁盘当中`。这与B+树不同，`B+树数据的更新会直接在原数据所在处修改对应的值`，但是LSM数的数据更新是日志式的，当一条数据更新是直接append一条更新记录完成的。这样设计的目的就是为了顺序写，不断地将Immutable MemTable flush到持久化存储即可，而不用去修改之前的SSTable中的key，保证了顺序写。

因此当MemTable达到一定大小flush到持久化存储变成SSTable后，在不同的SSTable中，可能存在相同Key的记录，当然最新的那条记录才是准确的。这样设计的虽然大大提高了写性能，但同时也会带来一些问题：

> 1）`冗余存储`，对于某个key，实际上除了最新的那条记录外，其他的记录都是冗余无用的，但是仍然占用了存储空间。因此需要进行Compact操作(合并多个SSTable)来清除冗余的记录。
> 2）`读取时需要从最新的倒着查询，直到找到某个key的记录`。最坏情况需要查询完所有的SSTable，这里可以通过前面提到的索引/布隆过滤器来优化查找速度。

## LSM树的Compact策略

> 1）`读放大`:读取数据时实际读取的数据量大于真正的数据量。例如`在LSM树中需要先在MemTable查看当前key是否存在，不存在继续从SSTable中寻找`。
> 2）`写放大`:写入数据时实际写入的数据量大于真正的数据量。例如在LSM树中写入时可能`触发Compact操作`，导致实际写入的数据量远大于该key的数据量。
> 3）`空间放大:`数据实际占用的磁盘空间比数据的真正大小更多。上面提到的冗余存储，`对于一个key来说，只有最新的那条记录是有效的`，而之前的记录都是可以被清理回收的。

### size-tiered 策略

![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-bedb057fde7a4ce4d5be2ea34fe86f59_720w.jpg)

> size-tiered策略保证`每层SSTable的大小相近`，`同时限制每一层SSTable的数量`。如上图，每层限制SSTable为N，当每层SSTable达到N后，则触发Compact操作合并这些SSTable，并将合并后的结果写入到下一层成为一个更大的sstable。
>
> 由此可以看出，*当层数达到一定数量时，最底层的单个SSTable的大小会变得非常大*。并且*size-tiered策略会导致空间放大比较严重*。即使对于同一层的SSTable，每个key的记录是可能存在多份的，只有当该层的SSTable执行compact操作才会消除这些key的冗余记录。

### leveled策略

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-5f8de2e435e979936693631617a60d16_720w.jpg)

> - `每一层的总大小固定，从上到下逐渐变大`
> - leveled策略也是采用分层的思想，`每一层限制总文件的大小`。
> - 但是跟size-tiered策略不同的是，`leveled会将每一层切分成多个大小相近的SSTable`。这些SSTable是这一层是**全局有序**的，意味着`一个key在每一层至多只有1条记录，不存在冗余记录`。之所以可以保证全局有序，是因为合并策略和size-tiered不同，接下来会详细提到。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-8274669affe5b9602aff45ddff29e628_720w.jpg)

每一层的SSTable是全局有序的,假设存在以下这样的场景:

1) L1的总大小超过L1本身大小限制：

![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-2546095c6b6e02af02de10cd236302f8_720w.jpg)

2) 此时会从L1中选择至少一个文件，然后把它跟L2***有交集的部分(非常关键)***进行合并。生成的文件会放在L2:此时L1第二SSTable的key的范围覆盖了L2中前三个SSTable，那么就需要将L1中第二个SSTable与L2中前三个SSTable执行Compact操作。

![img](https://pic2.zhimg.com/80/v2-663d136cefaaf6f8301833bf29c833e9_720w.jpg)

3) 如果L2合并后的结果仍旧超出L5的阈值大小，需要重复之前的操作 —— 选至少一个文件然后把它合并到下一层:

![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-715d76154b33abbe51e158b0cfcdc2bc_720w.jpg)

需要注意的是，***多个不相干的合并是可以并发进行的\***：

![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-2065db94c8837edd583b6ec639eaae6e_720w.jpg)

leveled策略相较于size-tiered策略来说，每层内key是不会重复的，即使是最坏的情况，除开最底层外，其余层都是重复key，按照相邻层大小比例为10来算，冗余占比也很小。因此空间放大问题得到缓解。但是写放大问题会更加突出。举一个最坏场景，如果LevelN层某个SSTable的key的范围跨度非常大，覆盖了LevelN+1层所有key的范围，那么进行Compact时将涉及LevelN+1层的全部数据。

## LSM-Tree 存储引擎读写算法

### LSM-Tree 数据写入过程

> 分为`实时的写入`以及`滞后的刷盘`以及`合并`两部分过程。
>
> - 实时的写入主要涉及到 WAL 日志以及 MemTable ，我们前面说过 WAL 是一个顺序日志文件，而 MemTable 是内存当中的 Skip-List 的数据结构，实时写入的时候：
> - 首先，将数据顺序写入 WAL Log 文件，由于是顺序追加，没有检索的过程，会非常快。然后，将数据写入 MemTable ，也涉及到在 Skip-List 当中追加链的操作，内存处理也非常快。
>
> - 当内存 Memtable 当中的数据达到一定的规模触及阀值条件的时候，后台会启动排序及合并操作，将 MemTable 当中的数据排序并刷入磁盘形成 SSTable ，同时周期性的将 SSTable 文件再进行排序分组，形成更大的 SSTable 文件，并转入下一个 Level 存储。

### LSM-Tree 数据修改过程

> LSM-Tree 存储引擎的更新过程其实并不存在，它不会像 B 树存储引擎那样，先经过检索过程，然后再进行修改。它的`更新操作是通过追加数据来间接实现`，也就是说更新最终转换为追加一个新的数据。
>
> - 只是在读取的时候，会从 Level0 层的 SSTable 文件开始查找数据，数据在低层的 SSTable 文件中必然比高层的文件中要新，所以总能读取到最新的那条数据。也就是说`此时在整个 LSM Tree 中可能会同时存在多个 key 值相同的数据，只有在之后合并 SSTable 文件的时候，才会将旧的值删除`。

### LSM-Tree 数据删除过程

> LSM-Tree 存储引擎的对数据的删除过程与追加数据的过程基本一样，区别在于`追加数据的时候，是有具体的数据值的，而删除的时候，追加的数据值是删除标记`。同样在读取的时候，会从 Level0 层的 SSTable 文件开始查找数据，数据在低层的 SSTable 文件中必然比高层的文件中要新，所以如果有删除操作，那么一定会最先读到带删除标记的那条数据。后期合并 SSTable 文件的时候，才会把数据删除。

### LSM-Tree 数据读取过程

> 对于 LSM-Tree 的读取操作，按照 Memtable 、 SSTable 的架构来讲，那么肯定是`先扫描 MemTable 当中的元素，然后扫描 SSTable 文件当中的数据`，然后找到相应数据。虽然我们讲过 SSTable 本身是有序的数据元素片段，而且对于读取概率较大的数据基本会在 Memtable 当中，但是这依然会造成巨大的扫描，读取效率会非常低下。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220810235421939.png)

>
> 虽然 LSM-Tree 的设计思想并不是为了随机读的性能而设计，但是为了改善随机读的效率，那么在其数据结构的设计当中，其实是有考虑索引设计的。正如图 2.3.3 所示，在每一个 SSTable 文件的尾部是有记录数据位置的索引信息的。这些信息是可以被流失读入内存当中，然后形成一个稀疏索引，那么检索的时候如果从 MemTable 当中没有检索到数据，接下来需要访问 SSTable 的时候，是可以先通过这个稀疏索引定位到 SSTable 的具体位置，然后在准确读取数据，这样的话就会大大提高随机读取的效率。
>

## Resource

- https://blog.51cto.com/u_15127582/2749141
- https://blog.51cto.com/u_15127582/2749141

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E7%B4%A2%E5%BC%95%E7%BB%93%E6%9E%84--lsm%E6%A0%91/  

