# RedisRelative


> Redis是一种支持key-value等多种数据结构的存储系统。可用于`缓存`，`事件发布或订阅`，`高速队列`等场景。支持`网络`，提供字符串，哈希，列表，队列，集合结构直接存取，基于内存，可持久化。

> 学习转载于： https://www.pdai.tech/md/db/nosql-redis/db-redis-data-types.html
>
> - [ ] 过一遍,了解redis知识体系, 有哪些部分,能在什么场景使用
> - [ ] 深入学习中间件知识,Stream机制,然后使用在AIOT场景中
> - [ ] 学习一些数据结构,使用
> - [ ] 事件处理机制功能代码,能不能自己分离出来,使用在自己项目中,或者在redis基础上.

## 1. 基本概念

### .1. NoSql简介

> NoSql泛指非关系型数据库，这些类型的数据存储不需要固定的模式，无需多余操作就可以横向扩展
>
> - `K-V键值对(Redis)  `
> - `文档型数据库(MongDB) `
> - `列存储数据库 `
> - `图关系数据库`

### .2. 分布式数据库CAP原理+BASE

- **CAP是什么**
  - C(Consistency)  强一致性 
  - A(Availability) 高可用性 
  - P(Partition tolerance)分区容错性

- **分布式数据库CAP特性不能三个都满足，只能满足其中的两条，其中P即分区容错性必须要实现。**

- 传统的Oracle Mysql等满足CA, 传统的关系型数据库: ACID即原子性 一致性 隔离性 持久性


- **BASE是什么**
  - BASE是为了解决关系数据库强一致性引起的问题而引发的可用性降低而提出的解决方案。
  - BA(Basically Available)基本可用
  - S(Soft state)软状态
  - E(Eventually consistent)最终一致性

## 2. [Redis](https://redis.io/commands)

- 单线程 | K-V键值对 | 默认端口6379 | 共有16个数据库，且索引从零开始
- 命令参考大全:http://redisdoc.com

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210707224001891.png)

### .0. 应用场景

####  1. 热点数据的缓存

作为缓存使用时，一般有两种方式保存数据：

- 读取前，先去读Redis，如果没有数据，读取数据库，将数据拉入Redis。
- 插入数据时，同时写入Redis。

方案一：实施起来简单，但是有两个需要注意的地方：

- 避免缓存击穿。（数据库没有就需要命中的数据，导致Redis一直没有数据，而一直命中数据库。）
- 数据的实时性相对会差一点。

方案二：数据实时性强，但是开发时不便于统一处理。

当然，两种方式根据实际情况来适用。如：`方案一`适用于`对于数据实时性要求不是特别高`的场景。`方案二`适用于`字典表、数据量不大`的数据存储。

#### 2. 限时业务的运用

> redis中可以使用`expire命令设置一个键的生存时间`，到时间后redis会删除它。利用这一特性可以运用在限时的优惠活动信息、手机验证码等业务场景。

#### 3. 计数器相关问题

> redis由于`incrby命令可以实现原子性的递增`，所以可以运用于`高并发的秒杀活动`、`分布式序列号的生成`、具体业务还体现在比如限制一个手机号发多少条短信、一个接口一分钟限制多少请求、一个接口一天限制调用多少次等等。

#### 4. 分布式锁

> 这个主要利用redis的setnx命令进行，`setnx："set if not exists"就是如果不存在则成功设置缓存同时返回1`，否则返回0 ，这个特性在俞你奔远方的后台中有所运用，因为我们服务器是集群的，`定时任务可能在两台机器上都会运行`，所以在定时任务中首先 通过setnx设置一个lock， 如果成功设置则执行，如果没有成功设置，则表明该定时任务已执行。 当然结合具体业务，我们可以给这个`lock加一个过期时间`，比如说30分钟执行一次的定时任务，那么这个过期时间设置为小于30分钟的一个时间就可以，这个与定时任务的周期以及定时任务执行消耗时间相关。在`分布式锁`的场景中，主要用在比如秒杀系统等。

#### 5. 延时操作

> 比如在订单生产后我们占用了库存，10分钟后去检验用户是够真正购买，如果没有购买将该单据设置无效，同时还原库存。 由于redis自2.8.0之后版本`提供Keyspace Notifications功能，允许客户订阅Pub/Sub频道`，以便以某种方式接收影响Redis数据集的事件。 所以我们对于上面的需求就可以用以下解决方案，我们在订单生产时，设置一个key，同时设置10分钟后过期， 我们在后台实现一个监听器，监听key的实效，监听到key失效时将后续逻辑加上。当然我们也可以利用`rabbitmq`、`activemq`等`消息中间件`的延迟队列服务实现该需求。

#### 6. 排行榜相关问题

> 关系型数据库在排行榜方面查询速度普遍偏慢，所以可以借助`redis的SortedSet进行热点数据的排序`。比如点赞排行榜，做一个SortedSet, 然后以用户的openid作为上面的username, 以用户的点赞数作为上面的score, 然后针对每个用户做一个hash, 通过zrangebyscore就可以按照点赞数获取排行榜，然后再根据username获取用户的hash信息，这个当时在实际运用中性能体验也蛮不错的。

#### 7. 点赞、好友等相互关系的存储

> Redis 利用集合的一些命令，比如求`交集、并集、差集`等。在微博应用中，每个用户关注的人存在一个集合中，就很容易实现求两个人的共同好友功能。

#### 8. 简单队列

> 由于Redis有`list push和list pop`这样的命令，所以能够很方便的执行队列操作。

### .1. 安装

- D:\database\Redis\

##### 1. 软件安装

- 二进制方式安装

```python
#下载地址： https://github.com/tporadowski/redis/releases
# 现在.msi文件
# 启动 redis-server.exe
# 查看redis默认配置文件： redis-server.exe redis.windows.conf 
# 客户端连接： redis-cli.exe -h 192.168.10.61 -p 6379
# 启动 redis-sli.exe
# 输入ping ； 如果返回 PONG 则说明启动成功。
```

- 源码编译安装

```shell
#下载
wget http://download.redis.io/releases/redis-5.0.5.tar.gz
#解压
tar -zxvf redis-5.0.5.tar.gz
#make编译
#不存在gcc时的解决方案
yum install gcc-c++      make distclean
make
#安装
make install (默认安装目录在/opt/redis-5.0.5/src)
#运行redis,需要修改redis.conf,使得redis可以在后台运行
cd /usr/local/bin/          redis-server /redis/redis.conf
#命令行操作
cd /usr/local/bin/             redis-cli -p 6379
```

- Dockr安装(推荐)


```shell
#搜索redis镜像
docker search redis
#拉取redis镜像
docker pull redis
#启动redis
docker run -p 6379:6379  -d redis  redis-server --appendonly yes
#连接到redis
docker exec -it [reids容器id] redis-cli
```

- 安装后的杂项配置


```shell
#该命令用于测试redis的性能如何
redis-benchmark
#Redis总共有16个数据库，默认从0库开始使用
select 0-15选择要使用的库
#查看某一个库中键的数量
DBSIZE
#FLUSHALL清除所有库中的键值对，FLUSHDB清除当前库中的键值对
FLUSHALL FLUSHDB
```

##### 2. 配置文件

- 配置文件放在/usr/redis-5.0.5/中，想要修改该配置文件时先复制出一份到某个路径下，避免造成文件损坏

- Units单位

  **Redis中1k与1kb的区别，另外Redis对大小写不敏感**

  ```shell
  # 1k => 1000 bytes
  # 1kb => 1024 bytes
  # 1m => 1000000 bytes
  # 1mb => 1024*1024 bytes
  # 1g => 1000000000 bytes
  # 1gb => 1024*1024*1024 bytes
  # units are case insensitive so 1GB 1Gb 1gB are all the same.
  ```

- INCLUDES用于包含Redis的其他的配置文件

  ```shell
  # Include one or more other config files here.  This is useful if you
  # have a standard template that goes to all Redis servers but also need
  # to customize a few per-server settings.  Include files can include
  # other files, so use this wisely.
  #
  # Notice option "include" won't be rewritten by command "CONFIG REWRITE"
  # from admin or Redis Sentinel. Since Redis always uses the last processed
  # line as value of a configuration directive, you'd better put includes
  # at the beginning of this file to avoid overwriting config change at runtime.
  #
  # If instead you are interested in using includes to override configuration
  # options, it is better to use include as the last line.
  #
  # include /path/to/local.conf
  # include /path/to/other.conf
  ```

- GENERAL指通用的标准配置

  ```shell
  #是否允许Redis在后台运行，默认为no，如果开启后台运行会在/var/run/redis.pid写入一个pid文件
  daemonize yes
  
  # Redis的日志级别，默认为notice
  # debug (a lot of information, useful for development/testing)
  # verbose (many rarely useful info, but not a mess like the debug level)
  # notice (moderately verbose, what you want in production probably)
  # warning (only very important / critical messages are logged)
  loglevel notice
  
  # 指定日志文件的名称，默认为"",日志文件将保存在/dev/null目录下
  logfile ""
  
  #默认系统日志关闭
  # syslog-enabled no
  
  # 默认系统日志以redis开头
  # syslog-ident redis
  
  # 系统日志设备使用LOCAL0-LOCAL7，默认为local0
  # syslog-facility local0
  
  #总数据库个数为16个，默认使用第0个，数据库下标0-15，可以使用select 0-15进行数据库的切换
  databases 16
  
  ```

- NETWORK

  ```shell
  #Redis绑定可以访问的ip地址，默认为本机
  bind 127.0.0.1
  
  #端口号，如果端口号为0，Redis不会监听TCP连
  port 6379
  
  # TCP listen() backlog.
  # In high requests-per-second environments you need an high backlog in order
  # to avoid slow clients connections issues. Note that the Linux kernel
  # will silently truncate it to the value of /proc/sys/net/core/somaxconn so
  # make sure to raise both the value of somaxconn and tcp_max_syn_backlog
  # in order to get the desired effect.
  tcp-backlog 511
  
  # 空闲多少秒以后关闭这个连接，如果为0则不会自动关闭(0 to disable)
  timeout 0
  
  #单位为秒，如果设置为0则不会进行keepalive检测，建议设置为60，每隔多少秒进行一次检测，检测该redis连接是否可用
  tcp-keepalive 300
  ```

- SECURITY

  连接安装在Linux上的Redis默认不需要输入密码

  ```shell
  #连接Redis以后输入下面命令可以获得默认的密码
  #设定Redis的密码，在设定密码后每次进行Redis的访问都需要输入密码
  #连接Redis进行访问，存在密码时输入密码进行验证密码的命令
  config get requirepass
  config set requirepass
  auth password
  
  #得到启动Redis的路径(一般日志会保存在Redis启动的目录下面)
  config get dirCLIENTS
  ```


- CLIENTS

  ```shell
  # 设置同一时间内最大连接数，默认为10000
  maxclients 10000
  ```

- MEMORY MANAGEMENT

  ```shell
  # 最大的内存用量
  maxmemory <bytes>
  
  #当达到Redis的最大可用内存时，Redis中缓存的过期策略，有一下8种选择:
  # volatile-lru -> Evict using approximated LRU among the keys with an expire set.(使用LRU算法移除key，只对设置了过期时间的key有用)
  # allkeys-lru -> Evict any key using approximated LRU.(使用LRU算法移除key)
  # volatile-lfu -> Evict using approximated LFU among the keys with an expire set.(使用LFU算法移除key，只对设置了过期时间的key有用)
  # allkeys-lfu -> Evict any key using approximated LFU.(使用LFU算法移除key)
  # volatile-random -> Remove a random key among the ones with an expire set.(在过期的集合中随机移除key，只对设置了过期时间的key有用)
  # allkeys-random -> Remove a random key, any key.(随机移除key)
  # volatile-ttl -> Remove the key with the nearest expire time (移除ttl最小的key即最近要过期的key)
  # noeviction -> Don't evict anything, just return an error on write operations.(永不过期，达到最大缓存时报错)
  
  #默认的缓存过期策略
  # maxmemory-policy noeviction
  
  
  #LRU，LFU和最小TTL算法不是精确的算法，而是近似算法（为了节省内存），因此您可以调整它以获得速度或精度。 默认情况下，Redis将检查五个键并选择最近使用的键，您可以使用以下配置指令更改样本大小。
  #默认值为5会产生足够好的结果。 10近似非常接近真实的LRU但成本更高的CPU。 3更快但不是很准确
  # maxmemory-samples 5
  ```


### 2. 数据结构

#### .1. 基本数据类型

> 首先对redis来说，`所有的key（键）都是字符串`。我们在谈基础数据结构时，讨论的是存储值的数据类型，主要包括常见的5种数据类型，分别是：String、List、Set、Zset、Hash。

##### 1. String 字符串

> String类型是二进制安全的，意思是 redis 的 string 可以包含任何数据。如数字，字符串，jpg图片或者序列化的对象。
>
> - **缓存**： 经典使用场景，把常用信息，字符串，图片或者视频等信息放到redis中，redis作为缓存层，mysql做持久化层，降低mysql的读写压力。
> - **计数器**：redis是单线程模型，一个命令执行完才会执行下一个，同时数据可以一步落地到其他的数据源。
> - **session**：常见方案spring session + redis实现session共享

##### 2. List 列表

> Redis中的List其实就是链表（Redis用双端链表实现List）。
>
> - lpush+lpop=Stack(栈)
> - lpush+rpop=Queue（队列）
> - lpush+ltrim=Capped Collection（有限集合）
> - lpush+brpop=Message Queue（消息队列）
> - **微博TimeLine**: 有人发布微博，用lpush加入时间轴，展示新的列表信息。
> - **消息队列**

##### 3. [Set](https://www.runoob.com/redis/redis-sets.html)

> Redis 中集合是通过哈希表实现的，所以添加，删除，查找的复杂度都是 O(1)。
>
> - **标签**（tag）,给用户添加标签，或者用户给消息添加标签，这样`有同一标签或者类似标签`的可以给`推荐`关注的事或者关注的人。
> - **点赞，或点踩，收藏等**，可以放到set中实现

##### 4. Hash 散列

> Redis hash 是一个 string 类型的 field（字段） 和 value（值） 的映射表，hash 特别适合用于存储对象。
>
> - **缓存**： 能直观，相比string更节省空间，的维护缓存信息，如用户信息，视频信息等



##### 5. [Zset 有序集合](https://www.runoob.com/redis/redis-sorted-sets.html)

> Redis 有序集合和集合一样也是 string 类型元素的集合,且不允许重复的成员。不同的是每个元素都会关联一个 double 类型的分数。redis 正是`通过分数来为集合中的成员进行从小到大的排序`。
>
> - **排行榜**：有序集合经典使用场景。例如小说视频等网站需要对用户上传的小说视频做排行榜，榜单可以按照用户关注数，更新时间，字数等打分，做排行。

#### .2. 特殊数据类型

##### 1. HyperLogLogs（基数统计）

>举个例子，A = {1, 2, 3, 4, 5}， B = {3, 5, 6, 7, 9}；那么基数（不重复的元素）= 1, 2, 4, 6, 7, 9； （允许容错，即可以接受一定误差）
>
>- 非常省内存的去统计各种计数，比如注册 IP 数、每日访问 IP 数、页面实时UV、在线用户数，共同好友数等

##### 2. Bitmap （位存储）

>Bitmap 即位图数据结构，都是操作二进制位来进行记录，只有0 和 1 两个状态。
>
>- 统计用户信息，活跃，不活跃！ 登录，未登录！ 打卡，不打卡！ **两个状态的，都可以使用 Bitmaps**！

##### 3. geospatial ([地理位置](http://www.jsons.cn/lngcode))

>推算地理位置的信息: 两地之间的距离, 方圆几里的人
>
>- 有效的经度从-180度到180度。
>- 有效的纬度从-85.05112878度到85.05112878度。

```shell
#添加地理位置
geoadd china:city 118.76 32.04 manjing 112.55 37.86 taiyuan 123.43 41.80 shenyang
#获取指定的成员的经度和纬度
geopos china:city taiyuan manjing
#获取俩地方距离
geodist china:city taiyuan shenyang m
#附近的人 ==> 获得所有附近的人的地址, 定位, 通过半径来查询
georadius china:city 110 30 1000 km			#以 100,30 这个坐标为中心, 寻找半径为1000km的城市
# 显示与指定成员一定半径范围内的其他成员
 georadiusbymember china:city taiyuan 1000 km withcoord withdist count 2
```

#### .3. Stream 数据类型

>Redis Stream也是一种`超轻量MQ`并没有完全实现消息队列所有设计要点
>
>- 
>
>**PUB/SUB，订阅/发布模式**
>
>- 但是`发布订阅模式是无法持久化的`，如果出现网络断开、Redis 宕机等，消息就会被丢弃；
>
>基于**List LPUSH+BRPOP** 或者 **基于Sorted-Set**的实现
>
>- 支持了持久化，但是`不支持多播，分组消费等`

##### 1. 结构

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210708075818540.png

- `Consumer Group` ：消费组，使用 XGROUP CREATE 命令创建，一个消费组有多个消费者(Consumer), 这些消费者之间是竞争关系。
- `last_delivered_id` ：游标，每个消费组会有个游标 last_delivered_id，`任意一个消费者读取了消息都会使游标 last_delivered_id 往前移动。`
- `pending_ids` ：`消费者(Consumer)的状态变量`，作用是`维护消费者的未确认的 id`。 pending_ids 记录了`当前已经被客户端读取的消息`，但是还没有 `ack` (Acknowledge character：确认字符）。如果客户端没有ack，这个变量里面的消息ID会越来越多，一旦某个消息被ack，它就开始减少。这个pending_ids变量在Redis官方被称之为PEL，也就是Pending Entries List，这是一个很核心的数据结构，它用来确保客户端`至少消费了消息一次`，而不会在网络传输的中途丢失了没处理。

- `消息ID`: 消息ID的形式是timestampInMillis-sequence，例如1527846880572-5，它表示当前的消息在毫米时间戳1527846880572时产生，并且是该毫秒内产生的第5条消息。消息ID可以由服务器自动生成，也可以由客户端自己指定，但是形式必须是整数-整数，而且必须是后面加入的消息的ID要大于前面的消息ID。
- `消息内容`: 消息内容就是键值对，形如hash结构的键值对，这没什么特别之处。

##### 2. 增删查改

```shell
#XADD - 添加消息到末尾
# *号表示服务器自动生成ID，后面顺序跟着一堆key/value
xadd codehole * name laoqian age 30  #  名字叫laoqian，年龄30岁
#XTRIM - 对流进行修剪，限制长度
#XDEL - 删除消息
del codehole  # 删除整个Stream
#XLEN - 获取流包含的元素数量，即消息长度
xlen codehole
#XRANGE - 获取消息列表，会自动过滤已经删除的消息
xrange codehole - +  # -表示最小值, +表示最大值
xrange codehole 1527849629172-0 +  # 指定最小消息ID的列表
#XREVRANGE - 反向获取消息列表，ID 从大到小
#XREAD - 以阻塞或非阻塞方式获取消息列表
```

##### 3. 独立消费

>当Stream没有新消息时，甚至可以`阻塞等待`。Redis设计了一个单独的消费指令xread，可以将Stream当成普通的消息队列(list)来使用。使用xread时，我们可以完全忽略消费组(Consumer Group)的存在，就好比Stream就是一个普通的列表(list)。
>
>- 客户端如果想要使用xread进行顺序消费，一定要记住当前消费到哪里了，也就是返回的消息ID。下次继续调用xread时，将上次返回的最后一个消息ID作为参数传递进去，就可以继续消费后续的消息。

```shell
#block 0表示永远阻塞，直到消息到来，block 1000表示阻塞1s，如果1s内没有任何消息到来，就返回nil
xread block 1000 count 1 streams codehole $
```

##### 4. 消费组消费

#### .4.  对象机制详解

>- Redis的每种对象其实都由**对象结构(redisObject)** 与 **对应编码的数据结构**组合而成，而每种对象类型对应若干编码方式，不同的编码方式所对应的底层数据结构是不同的。
>- **Redis 必须让每个键都带有类型信息, 使得程序可以检查键的类型, 并为它选择合适的处理方式**.
>- **操作数据类型的命令除了要对键的类型进行检查之外, 还需要根据数据类型的不同编码进行多态处理**.
>- redisObject 对象:  基于redisObject对象类型检查; 显示多态函数; 分配,共享和销毁机制;

##### 1. redisObject数据结构

```c
/*
 * Redis 对象
 */
typedef struct redisObject {

    // 类型
    unsigned type:4;

    // 编码方式
    unsigned encoding:4;

    // LRU - 24位, 记录最末一次访问时间（相对于lru_clock）; 或者 LFU（最少使用的数据：8位频率，16位访问时间）
    unsigned lru:LRU_BITS; // LRU_BITS: 24

    // 引用计数
    int refcount;

    // 指向底层数据结构实例
    void *ptr;

} robj;
```

```c
/*
* 对象类型
*/
#define OBJ_STRING 0 // 字符串
#define OBJ_LIST 1 // 列表
#define OBJ_SET 2 // 集合
#define OBJ_ZSET 3 // 有序集
#define OBJ_HASH 4 // 哈希表

/*
* 对象编码
*/
#define OBJ_ENCODING_RAW 0     /* Raw representation */
#define OBJ_ENCODING_INT 1     /* Encoded as integer */
#define OBJ_ENCODING_HT 2      /* Encoded as hash table */
#define OBJ_ENCODING_ZIPMAP 3  /* 注意：版本2.6后不再使用. */
#define OBJ_ENCODING_LINKEDLIST 4 /* 注意：不再使用了，旧版本2.x中String的底层之一. */
#define OBJ_ENCODING_ZIPLIST 5 /* Encoded as ziplist */
#define OBJ_ENCODING_INTSET 6  /* Encoded as intset */
#define OBJ_ENCODING_SKIPLIST 7  /* Encoded as skiplist */
#define OBJ_ENCODING_EMBSTR 8  /* Embedded sds string encoding */
#define OBJ_ENCODING_QUICKLIST 9 /* Encoded as linked list of ziplists */
#define OBJ_ENCODING_STREAM 10 /* Encoded as a radix tree of listpacks */
```

##### 2. 处理过程

- 根据给定的key，在数据库字典中`查找和他相对应的redisObject`，如果没找到，就返回NULL；
- 检查redisObject的`type属性`和`执行命令所需的类型是否相符`，如果不相符，返回类型错误；
- 根据redisObject的`encoding属性所指定的编码`，选择合适的操作函数来处理`底层的数据结构`；
- 返回数据结构的`操作结果作为命令的返回值`。

##### 3. 对象共享

>redis一般会把一些常见的值放到一个共享对象中，这样可使程序避免了重复分配的麻烦，也节约了一些CPU时间。
>
>- 共享对象只能`被字典和双向链表这类能带有指针的数据结构使用`。像整数集合和压缩列表这些只能保存字符串、整数等自勉之的内存数据结构
>
>- **为什么redis不共享列表对象、哈希对象、集合对象、有序集合对象，只共享字符串对象**？
>
>  列表对象、哈希对象、集合对象、有序集合对象，本身可以包含字符串对象，复杂度较高。
>
>  如果共享对象是保存字符串对象，那么验证操作的复杂度为O(1)
>
>  如果共享对象是保存字符串值的字符串对象，那么验证操作的复杂度为O(N)
>
>  如果共享对象是包含多个值的对象，其中值本身又是字符串对象，即其它对象中嵌套了字符串对象，比如列表对象、哈希对象，那么验证操作的复杂度将会是O(N的平方)

**redis`预分配的值对象`如下**：

- 各种`命令的返回值`，比如成功时返回的OK，错误时返回的ERROR，命令入队事务时返回的QUEUE，等等
- 包括0 在内，小于REDIS_SHARED_INTEGERS的所有整数（REDIS_SHARED_INTEGERS的默认值是10000）

##### 4. 引用计数

> redisObject中有refcount属性，是对象的引用计数，显然计数0那么就是可以回收。
>
> - 当新创建一个对象时，它的refcount属性被设置为1；
> - 当对一个对象进行共享时，redis将这个对象的refcount加一；
> - 当使用完一个对象后，或者消除对一个对象的引用之后，程序将对象的refcount减一；
> - 当对象的refcount降至0 时，这个RedisObject结构，以及它引用的数据结构的内存都会被释放

#### .5. [底层数据结构](https://www.pdai.tech/md/db/nosql-redis/db-redis-x-redis-ds.html)

##### 1. 简单动态字符串 - sds

>用于存储二进制数据的一种结构, 具有动态扩容的特点. 其实现位于src/sds.h与src/sds.c中。

##### 2. 压缩列表 - ZipList

##### 3. 快表 - QuickList

>以ziplist为结点的双端链表结构. 宏观上, quicklist是一个链表, 微观上, 链表中的每个结点都是一个ziplist.

- `quicklistNode`, 宏观上, quicklist是一个链表, 这个结构描述的就是链表中的结点. 它通过zl字段持有底层的ziplist. 简单来讲, 它描述了一个ziplist实例
- `quicklistLZF`, ziplist是一段连续的内存, 用LZ4算法压缩后, 就可以包装成一个quicklistLZF结构. 是否压缩quicklist中的每个ziplist实例是一个可配置项. 若这个配置项是开启的, 那么quicklistNode.zl字段指向的就不是一个ziplist实例, 而是一个压缩后的quicklistLZF实例
- `quicklistBookmark`, 在quicklist尾部增加的一个书签，它只有在大量节点的多余内存使用量可以忽略不计的情况且确实需要分批迭代它们，才会被使用。当不使用它们时，它们不会增加任何内存开销。
- `quicklist`. 这就是一个双链表的定义. head, tail分别指向头尾指针. len代表链表中的结点. count指的是整个quicklist中的所有ziplist中的entry的数目. fill字段影响着每个链表结点中ziplist的最大占用空间, compress影响着是否要对每个ziplist以LZ4算法进行进一步压缩以更节省内存空间.
- `quicklistIter`是一个迭代器
- `quicklistEntry`是对ziplist中的entry概念的封装. quicklist作为一个封装良好的数据结构, 不希望使用者感知到其内部的实现, 所以需要把ziplist.entry的概念重新包装一下

##### 4. 字典/哈希表 - Dict

```c
typedef struct dictht{
    //哈希表数组
    dictEntry **table;
    //哈希表大小
    unsigned long size;
    //哈希表大小掩码，用于计算索引值
    //总是等于 size-1
    unsigned long sizemask;
    //该哈希表已有节点的数量
    unsigned long used;
}dictht

typedef struct dictEntry{
     //键
     void *key;
     //值
     union{
          void *val;
          uint64_tu64;
          int64_ts64;
     }v;
 
     //指向下一个哈希表节点，形成链表
     struct dictEntry *next;
}dictEntry
```

##### 5. 整数集 - IntSet

>整数集合（intset）是集合类型的底层实现之一，当一个集合只包含整数值元素，并且这个集合的元素数量不多时，Redis 就会使用整数集合作为集合键的底层实现。

```c
typedef struct intset {
    uint32_t encoding;
    uint32_t length;
    int8_t contents[];
} intset;
```

- `encoding` 表示编码方式，的取值有三个：INTSET_ENC_INT16, INTSET_ENC_INT32, INTSET_ENC_INT64
- `length` 代表其中存储的整数的个数
- `contents` 指向实际存储数值的连续内存区域, 就是一个数组；整数集合的每个元素都是 contents 数组的一个数组项（item），各个项在数组中按值得大小**从小到大有序排序**，且数组中不包含任何重复项。（虽然 intset 结构将 contents 属性声明为 int8_t 类型的数组，但实际上 contents 数组并不保存任何 int8_t 类型的值，contents 数组的真正类型取决于 encoding 属性的值）

##### 6. 跳表 - ZSkipList

>跳跃表结构在 Redis 中的运用场景只有一个，那就是作为`有序列表 (Zset) 的使用`。跳跃表的性能可以保证在查找，删除，添加等操作的时候在对数期望时间内完成，这个性能是可以和平衡树来相比较的，而且在实现方面比平衡树要优雅，这就是跳跃表的长处。跳跃表的缺点就是需要的存储空间比较大，属于利用空间来换取时间的数据结构。



### 3. 持久化

>Redis是个基于内存的数据库。那服务一旦宕机，内存中的数据将全部丢失。通常的解决方案是从后端数据库恢复这些数据，但后端数据库有性能瓶颈，如果是大数据量的恢复，1、会对数据库带来巨大的压力，2、数据库的性能不如Redis。Redis服务提供四种持久化存储方案：`RDB`、`AOF`、`虚拟内存（VM）`和　`DISKSTORE`。

##### 1. RDB 持久化

> RDB持久化是把当前进程数据生成快照保存到磁盘上的过程，由于是某一时刻的快照，那么快照中的值要早于或者等于内存中的值。
>
> - RDB方式`实时性不够`，无法做到秒级的持久化；
> - 每次调用bgsave都需要fork子进程，fork子进程属于重量级操作，`频繁执行成本较高`；
> - R`DB文件是二进制的，没有可读性`，AOF文件在了解其结构的情况下可以手动修改或者补全；
> - `版本兼容RDB文件问题`；

- **手动触发**
  - **save命令**：`阻塞`当前Redis服务器，`直到RDB过程完成为止`，对于内存 比较大的实例会造成长时间**阻塞**，线上环境不建议使用
  - **bgsave命令**：Redis进程执行`fork操作创建子进程`，RDB持久化过程由子 进程负责，完成后自动结束。阻塞只发生在fork阶段，一般时间很短
    1. redis客户端执行bgsave命令或者自动触发bgsave命令;
    2. `主进程判断当前是否已经存在正在执行的子进程`，如果存在，那么主进程直接返回；
    3. 如果不存在正在执行的子进程，那么就`fork一个新的子进程进行持久化数据`，fork过程是阻塞的，`fork操作完成后主进程即可执行其他操作`；
    4. 子进程`先将数据写入到临时的rdb文件中`，`待快照数据写入完成后再原子替换旧的rdb文件`；
    5. `同时发送信号给主进程，通知主进程rdb持久化完成`，主进程更新相关的统计信息（info Persitence下的rdb_*相关选项）。
- **自动触发**
  - redis.conf中配置`save m n`，即在m秒内有n次修改时，自动触发bgsave生成rdb文件；
  - `主从复制时`，从节点要从主节点进行全量复制时也会触发bgsave操作，生成当时的快照发送到从节点；
  - 执行`debug reload命令重新加载redis时也会触发bgsave操作`；
  - 默认情况下执行shutdown命令时，如果没有开启aof持久化，那么也会触发bgsave操作；

>如果主线程对这些数据也都是读操作（例如图中的键值对 A），那么，主线程和 bgsave 子进程相互不影响。但是，如果主线程要修改一块数据（例如图中的键值对 C），那么，这块数据就会被复制一份，生成该数据的副本。然后，bgsave 子进程会把这个副本数据写入 RDB 文件，而在这个过程中，主线程仍然可以直接修改原来的数据。

##### 2. AOF 持久化

>Redis是“写后”日志，R`edis先执行命令，把数据写入内存，然后才记录日志`。日志里记录的是Redis收到的每一条命令，这些命令是以文本形式保存。PS: 大多数的数据库采用的是写前日志（WAL），例如MySQL，通过写前日志和两阶段提交，实现数据和逻辑的一致性。
>
>- AOF日志`记录Redis的每个写命令`，步骤分为：`命令追加（append）、文件写入（write）和文件同步（sync）`。

### 4.  消息传递

>Redis 发布订阅(pub/sub)是一种消息通信模式：发送者(pub)发送消息，订阅者(sub)接收消息。

##### 1. 基于频道的发布/订阅

>当客户端调用 SUBSCRIBE 命令时， 程序就将客户端和要订阅的频道在 pubsub_channels 字典中关联起来。
>
>举个例子，如果客户端 client10086 执行命令 `SUBSCRIBE channel1 channel2 channel3`

```shell
#subscribe。表示订阅成功的反馈信息。第二个值是订阅成功的频道名称，第三个是当前客户端订阅的频道数量。
#message。表示接收到的消息，第二个值表示产生消息的频道名称，第三个值是消息的内容。
#unsubscribe。表示成功取消订阅某个频道。第二个值是对应的频道名称，第三个值是当前客户端订阅的频道数量，当此值为0时客户端会退出订阅状态，之后就可以执行其他非"发布/订阅"模式的命令了。
publish channel:1 hi
subscribe channel:1
```

##### 2. 基于模式的发布/订阅

>通配符中?表示1个占位符，*表示任意个占位符(包括0)，?*表示1个以上占位符。

```c
//数据结构 redisServer.pubsub_patterns 属性是一个链表，链表中保存着所有和模式相关的信息：
struct redisServer {
    // ...
    list *pubsub_patterns;
    // ...
};
//链表中的每个节点都包含一个 redis.h/pubsubPattern 结构：
typedef struct pubsubPattern {
    redisClient *client;
    robj *pattern;
} pubsubPattern;
```

### 5. 事件机制

>Redis中的事件驱动库只关注网络IO，以及定时器。
>
>该事件库处理下面两类事件：
>
>- **文件事件**(file  event)：用于处理 Redis 服务器和客户端之间的网络IO。
>- **时间事件**(time  eveat)：Redis 服务器中的一些操作（比如serverCron函数）需要在给定的时间点执行，而时间事件就是处理这类定时操作的。
>
>事件驱动库的代码主要是在`src/ae.c`中实现的,`aeEventLoop`是整个事件驱动的核心，它管理着文件事件表和时间事件列表，不断地循环处理着就绪的文件事件和到期的时间事件。

##### 1. 文件事件

>Redis基于**Reactor模式**开发了自己的网络事件处理器，也就是文件事件处理器。文件事件处理器使用**IO多路复用技术**（ [Java IO多路复用详解]() ），同时监听多个套接字，并为套接字关联不同的事件处理函数。当套接字的可读或者可写事件触发时，就会调用相应的事件处理函数。
>
>- 文件事件处理器有四个组成部分，它们分别是`套接字`、`I/O多路复用程序`、`文件事件分派器`以及`事件处理器`。

1. 客户端向服务端发起**建立 socket 连接的请求**，那么监听套接字将产生 AE_READABLE 事件，触发连接应答处理器执行。处理器会对客户端的连接请求
2. 进行**应答**，然后创建客户端套接字，以及客户端状态，并将客户端套接字的 AE_READABLE 事件与命令请求处理器关联。
3. 客户端建立连接后，向服务器**发送命令**，那么客户端套接字将产生 AE_READABLE 事件，触发命令请求处理器执行，处理器读取客户端命令，然后传递给相关程序去执行。
4. **执行命令获得相应的命令回复**，为了将命令回复传递给客户端，服务器将客户端套接字的 AE_WRITEABLE 事件与命令回复处理器关联。当客户端试图读取命令回复时，客户端套接字产生 AE_WRITEABLE 事件，触发命令回复处理器将命令回复全部写入到套接字中。

>图中的多个 FD 就是刚才所说的多个套接字。Redis 网络框架调用 epoll 机制，让内核监听这些套接字。此时，Redis 线程不会阻塞在某一个特定的监听或已连接套接字上，也就是说，不会阻塞在某一个特定的客户端请求处理上。

##### 2. 事件事件

>- **定时事件**：让一段程序在指定的时间之后执行一次。
>- **周期性事件**：让一段程序每隔指定时间就执行一次。
>
>服务器所有的时间事件都放在一个`无序链表中`，每当时间事件执行器运行时，它就遍历整个链表，`查找所有已到达的时间事件`，并调用相应的事件处理器。正常模式下的`Redis服务器只使用serverCron一个时间事件`，而在benchmark模式下，服务器也只使用两个时间事件，所以不影响事件执行的性能

```c
typedef struct aeTimeEvent {
    /* 全局唯一ID */
    long long id; /* time event identifier. */
    /* 秒精确的UNIX时间戳，记录时间事件到达的时间*/
    long when_sec; /* seconds */
    /* 毫秒精确的UNIX时间戳，记录时间事件到达的时间*/
    long when_ms; /* milliseconds */
    /* 时间处理器 */
    aeTimeProc *timeProc;
    /* 事件结束回调函数，析构一些资源*/
    aeEventFinalizerProc *finalizerProc;
    /* 私有数据 */
    void *clientData;
    /* 前驱节点 */
    struct aeTimeEvent *prev;
    /* 后继节点 */
    struct aeTimeEvent *next;
} aeTimeEvent;
```

##### 3. [aeEventoop 具体实现](https://www.pdai.tech/md/db/nosql-redis/db-redis-x-event.html)

1. 首先`创建aeEventLoop对象`。
2. 初始化`未就绪文件事件表`、`就绪文件事件表`。events指针指向未就绪文件事件表、fired指针指向就绪文件事件表。表的内容在后面添加具体事件时进行初变更。
3. 初始化`时间事件列表`，设置`timeEventHead和timeEventNextId属性`。
4. `调用aeApiCreate 函数创建epoll实例，并初始化 apidata`。

```c
aeEventLoop *aeCreateEventLoop(int setsize) {
    aeEventLoop *eventLoop;
    int i;
    /* 创建事件状态结构 */
    if ((eventLoop = zmalloc(sizeof(*eventLoop))) == NULL) goto err;
    /* 创建未就绪事件表、就绪事件表 */
    eventLoop->events = zmalloc(sizeof(aeFileEvent)*setsize);
    eventLoop->fired = zmalloc(sizeof(aeFiredEvent)*setsize);
    if (eventLoop->events == NULL || eventLoop->fired == NULL) goto err;
    /* 设置数组大小 */
    eventLoop->setsize = setsize;
    /* 初始化执行最近一次执行时间 */
    eventLoop->lastTime = time(NULL);
    /* 初始化时间事件结构 */
    eventLoop->timeEventHead = NULL;
    eventLoop->timeEventNextId = 0;
    eventLoop->stop = 0;
    eventLoop->maxfd = -1;
    eventLoop->beforesleep = NULL;
    eventLoop->aftersleep = NULL;
    /* 将多路复用io与事件管理器关联起来 */
    if (aeApiCreate(eventLoop) == -1) goto err;
    /* 初始化监听事件 */
    for (i = 0; i < setsize; i++)
        eventLoop->events[i].mask = AE_NONE;
    return eventLoop;
err:
   .....
}


static int aeApiCreate(aeEventLoop *eventLoop) {
    aeApiState *state = zmalloc(sizeof(aeApiState));

    if (!state) return -1;
    /* 初始化epoll就绪事件表 */
    state->events = zmalloc(sizeof(struct epoll_event)*eventLoop->setsize);
    if (!state->events) {
        zfree(state);
        return -1;
    }
    /* 创建 epoll 实例 */
    state->epfd = epoll_create(1024); /* 1024 is just a hint for the kernel */
    if (state->epfd == -1) {
        zfree(state->events);
        zfree(state);
        return -1;
    }
    /* 事件管理器与epoll关联 */
    eventLoop->apidata = state;
    return 0;
}
typedef struct aeApiState {
    /* epoll_event 实例描述符*/
    int epfd;
    /* 存储epoll就绪事件表 */
    struct epoll_event *events;
} aeApiState;


typedef struct aeFileEvent {
    /* 监听事件类型掩码,值可以是 AE_READABLE 或 AE_WRITABLE */
    int mask;
    /* 读事件处理器 */
    aeFileProc *rfileProc;
    /* 写事件处理器 */
    aeFileProc *wfileProc;
    /* 多路复用库的私有数据 */
    void *clientData;
} aeFileEvent;
/* 使用typedef定义的处理器函数的函数类型 */
typedef void aeFileProc(struct aeEventLoop *eventLoop, 
int fd, void *clientData, int mask);


aeCreateFileEvent(server.el,fd,AE_READABLE|AE_WRITABLE,syncWithMaster,NULL);
//以fd为索引，在events未就绪事件表中找到对应事件。
//调用aeApiAddEvent函数，该事件注册到具体的底层 I/O 多路复用中，本例为epoll。
//填充事件的回调、参数、事件类型等参数。
/* 符合aeFileProc的函数定义 */
void syncWithMaster(aeEventLoop *el, int fd, void *privdata, int mask) {....}

```

### 6. 事务机制

>Redis 事务的本质是一组命令的集合。事务支持一次执行多个命令，一个事务中所有命令都会被序列化。在事务执行过程，会按照顺序串行化执行队列中的命令，其他客户端提交的命令请求不会插入到事务执行命令序列中。
>
>- MULTI ：`开启事务`，redis会`将后续的命令逐个放入队列中`，然后`使用EXEC命令来原子化执行这个命令系列。`
>- EXEC：`执行`事务中的所有操作命令。
>- DISCARD：取消事务，`放弃执行事务块中的所有命令`。
>- WATCH：`监视一个或多个key`,如果事务在执行前，这个key(或多个key)被其他命令修改，则事务被中断，不会执行事务中的任何命令。
>- UNWATCH：`取消WATCH对所有key的监视`。

##### 1. CAS 操作乐观锁

>被 WATCH 的键会被监视，并会发觉这些键是否被改动过了。 如果有至少一个被监视的键在 EXEC 执行之前被修改了， 那么整个事务都会被取消， EXEC 返回nil-reply来表示事务已经失败。

```shell
#当多个客户端同时对同一个键进行这样的操作时， 就会产生竞争条件。举个例子， 如果客户端 A 和 B 都读取了键原来的值， 比如 10 ， 那么两个客户端都会将键的值设为 11 ， 但正确的结果应该是 12 才对。
WATCH mykey
val = GET mykey
val = val + 1
MULTI
SET mykey $val
EXEC
```

##### 2. 执行过程

- Redis使用`WATCH命令来决定事务是继续执行还是回滚`，那就需要`在MULTI之前使用WATCH来监控某些键值对`，然后使`用MULTI命令来开启事务`，执行对数据结构操作的各种命令，此时这些命令入队列。
- 当`使用EXEC执行事务时`，首先会`比对WATCH所监控的键值对`，如果`没发生改变`，它会执行事务队列中的命令，`提交事务`；如果`发生变化`，将`不会执行事务中的任何命令，同时事务回滚`。当然无论是否回滚，Redis都会`取消执行事务前的WATCH命令`。

### 7. 主从复制

>**数据冗余**：主从复制实现了数据的热备份，是持久化之外的一种数据冗余方式。
>
>**故障恢复**：当主节点出现问题时，可以由从节点提供服务，实现快速的故障恢复；实际上是一种服务的冗余。
>
>**负载均衡**：在主从复制的基础上，配合读写分离，可以由主节点提供写服务，由从节点提供读服务（即写Redis数据时应用连接主节点，读Redis数据时应用连接从节点），分担服务器负载；尤其是在写少读多的场景下，通过多个从节点分担读负载，可以大大提高Redis服务器的并发量。
>
>**高可用基石**：除了上述作用以外，主从复制还是哨兵和集群能够实施的基础，因此说主从复制是Redis高可用的基础。



##### 1. 全量复制

>当我们启动多个 Redis 实例的时候，它们相互之间就可以通过 replicaof（Redis 5.0 之前使用 slaveof）命令形成主库和从库的关系，之后会按照三个阶段完成数据的第一次同步。

##### 2. 增量复制

- `repl_backlog_buffer`：它是为了从库断开之后，`如何找到主从差异数据而设计的环形缓冲区`，从而避免全量复制带来的性能开销。如果从库`断开时间太久`，`repl_backlog_buffer环形缓冲区被主库的写命令覆盖了`，那么从库连上主库后只能乖乖地进行一次`全量复制`，所以**repl_backlog_buffer配置尽量大一些，可以降低主从断开后全量复制的概率**。而在repl_backlog_buffer中找主从差异的数据后，如何发给从库呢？这就用到了replication buffer。
- `replication buffer`：Redis和客户端通信也好，和从库通信也好，Redis都需要给分配一个 内存buffer进行数据交互，客户端是一个client，从库也是一个client，我们每个client连上Redis后，Redis都会分配一个client buffer，所有数据交互都是通过这个buffer进行的：`Redis先把数据写到这个buffer中`，`然后再把buffer中的数据发到client socket中再通过网络发送出去`，这样就完成了数据交互。所以主从在增量同步时，`从库作为一个client，也会分配一个buffer`，只不过这`个buffer专门用来传播用户的写命令到从库`，保证主从数据一致，我们通常把它叫做replication buffer。

### 8. 哨兵机制

>解决在主从复制的时候, 如果注节点出现故障该怎么办呢？ 在 Redis 主从集群中，哨兵机制是实现主从库自动切换的关键机制，它有效地解决了主从复制模式下故障转移的问题.
>
>- **监控（Monitoring）**：哨兵会不断地检查主节点和从节点是否运作正常。
>- **自动故障转移（Automatic failover）**：当主节点不能正常工作时，`哨兵会开始自动故障转移操作，它会将失效主节点的其中一个从节点升级为新的主节点`，并`让其他从节点改为复制新的主节点`。
>- **配置提供者（Configuration provider）**：客户端在初始化时，通过`连接哨兵来获得当前Redis服务的主节点地址`。
>- **通知（Notification）**：哨兵可以`将故障转移的结果发送给客户端`。

##### .1. 哨兵集群的组建

>在主从集群中，主库上有一个名为`__sentinel__:hello`的频道，不同哨兵就是通过它来相互发现，实现互相通信的。在下图中，哨兵 1 把自己的 IP（172.16.19.3）和端口（26579）发布到`__sentinel__:hello`频道上，哨兵 2 和 3 订阅了该频道。那么此时，哨兵 2 和 3 就可以从这个频道直接获取哨兵 1 的 IP 地址和端口号。然后，哨兵 2、3 可以和哨兵 1 建立网络连接。

##### .2. 哨兵监控Redis库

>由哨兵向主库发送 INFO 命令来完成的。就像下图所示，哨兵 2 给主库发送 INFO 命令，主库接受到这个命令后，就会把从库列表返回给哨兵。接着，哨兵就可以根据从库列表中的连接信息，和每个从库建立连接，并在这个连接上持续地对从库进行监控。

##### .3. 主库下线的判定

- **主观下线**：任何一个哨兵都是可以监控探测，并作出Redis节点下线的判断；
- **客观下线**：有哨兵集群共同决定Redis节点是否下线；

- **哨兵集群的选举**： Raft选举算法： **选举的票数大于等于num(sentinels)/2+1时，将成为领导者，如果没有超过，继续选举**

##### .4. 新主库的选出

- 过滤掉不健康的（下线或断线），没有回复过哨兵ping响应的从节点
- 选择`salve-priority`从节点优先级最高（redis.conf）的
- 选择复制偏移量最大，指复制最完整的从节点

### 9. 分片计数

### 10. 一致性缓存

### 11. 运维监控

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/redisrelative/  

