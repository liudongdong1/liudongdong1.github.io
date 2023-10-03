# mysql 索引结构


## Innodb 架构

> 1. Buffer Pool 用于加速读
> 2. Change Buffer 用于`没有非唯一索引的加速写`
> 3. Log Buffer 用于`加速redo log写`
> 4. 自适应Hash索引主要用于加快查询页。在查询时，Innodb通过监视索引搜索的机制来判断当前查询是否能走Hash索引。比如LIKE运算符和% 通配符就不能走。

![](https://upload-images.jianshu.io/upload_images/24630328-701b5738c2cb5ce3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 当更新一个没有 unique index 的数据时，直接将修改的数据放到 change buffer，然后通过 merge 操作完成更新，从而减少了 IO 操作。
- merge 操作时机：
  - 有其他访问，访问到了当前页的数据，就会合并到磁盘
  - 后台线程定时
  - 系统正常shut down之前
  - redo log写满的时候（redo log 默认存储在两个文件中 ib_logfile0 ib_logfile1，这两个文件都是固定大小的。）
- Log Buffer，其就是用来写 redo log 之前存在的缓冲区
- redo log具体的执行策略有三种：
  1. 不用写Log Buffer，只需要每秒写redo log 磁盘数据一次，性能高，但会造成数据 1s 内的一致性问题。适用于强实时性，弱一致性，比如评论区评论
  2. 写Log Buffer，同时写入磁盘，性能最差，一致性最高。 适用于弱实时性，强一致性,比如支付场景
  3. 写Log Buffer，同时写到os buffer（其会每秒调用 fsync 将数据刷入磁盘），性能好，安全性也高。这个是实时性适中 一致性适中的，比如订单类。我们通过innodb_flush_log_at_trx_commit就可以设置执行策略。默认为 1

## 硬盘结构

### System Tablespace

存储在一个叫ibdata1的文件中，其中包含：

1. InnoDB Data Dictionary，存储了`元数据，比如表结构信息、索引`等
2. Doublewrite Buffer 当Buffer Pool写入数据页时，不是直接写入到文件，而是先写入到这个区域。这样做的好处的是，一但操作系统，文件系统或者mysql挂掉，可以直接从这个Buffer中获取数据。
3. Change Buffer 当Mysql shut down的时候，修改就会被存储在磁盘这里
4. Undo Logs `记录事务修改操作`

### File-Per-Table Tablespaces

每一张表都有一张 .ibd 的文件，`存储数据和索引`。

1. 有了每表文件表空间可以使得 ALTER TABLE与 TRUNCATE TABLE 性能得到很好的提升。比如 ALTER TABLE，相较于对驻留在共享表空间中的表，在修改表时，会进行表复制操作，这可能会增加表空间占用的磁盘空间量。此类操作可能需要与表中的数据以及索引一样多的额外空间。该空间不会像每表文件表空间那样释放回操作系统。
2. 可以在单独的存储设备上创建每表文件表空间数据文件，以进行I / O优化，空间管理或备份。这就意味着表数据与结构容易在不同数据库中迁移。
3. 当发生数据损坏，备份或二进制日志不可用或无法重新启动MySQL服务器实例时，存储在单个表空间数据文件中的表可以节省时间并提高成功恢复的机会。

当然有优点就有缺陷：

1. 存储空间的利用率低，会存在碎片，在Drop table的时候会影响性能（除非你自己管理了碎片）
2. 因为每个表分成各自的表文件，操作系统不能同时进行fsync一次性刷入数据到文件中
3. mysqld会持续保持每个表文件的 文件句柄， 以提供维持对文件的持续访问

## update sql 流程

![img](https://gsmtoday.github.io/2019/02/08/how-update-executes-in-mysql/update%20process.png)

1. 查询到我们要修改的那条数据，我们这里称做 origin，返给执行器
2. 在执行器中修改数据，称为 modification
3. 将modification刷入内存，Buffer Pool的 Change Buffer
4. 引擎层：记录undo log （实现事务原子性）
5. 引擎层：记录redo log （崩溃恢复使用）
6. 服务层：记录bin log（记录DDL）
7. 返回更新成功结果
8. 数据等待被工作线程刷入磁盘

![](https://upload-images.jianshu.io/upload_images/24630328-c56f279b458b41d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 主从复制方案

1. 全同步复制，事务方式执行，主节点先写入，然后让所有slave写，必须要所有 从节点 把数据写完，才返回写成功，这样的话会大大影响写入的性能
2. 半同步复制，只要有一个salve写入数据，就算成功。（如果需要半同步复制，主从节点都需要安装semisync_mater.so和 semisync_slave.so插件）
3. GTID（global transaction identities）复制，主库并行复制的时候，从库也并行复制，解决主从同步复制延迟，实现自动的failover动作，即主节点挂掉，选举从节点后，能快速自动避免数据丢失。

## 俩阶段提交

### 两阶段提交2PC

2PC即Innodb对于事务的两阶段提交机制。当MySQL开启binlog的时候，会存在一个内部XA的问题：事务在存储引擎层（redo）commit的顺序和在binlog中提交的顺序不一致的问题。如果不使用两阶段提交，那么数据库的状态有可能用它的日志恢复出来的库的状态不一致。

事务的commit分为prepare和commit两个阶段：
1、`prepare阶段：redo持久化到磁盘（redo group commit），并将回滚段置为prepared状态，此时binlog不做操作。`
[![](https://gitee.com/github-25970295/blogimgv2022/raw/master/prepare.png)]
2、`commit阶段：innodb释放锁，释放回滚段，设置提交状态，binlog持久化到磁盘，然后存储引擎层提交。`

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220730235130369.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E6%95%B0%E6%8D%AE%E5%BA%93%E7%B4%A2%E5%BC%95%E7%BB%93%E6%9E%84/  

