# Zookeeper


> ZooKeeper是一个`分布式的、开放源码的分布式协调服务`，是Google的Chubby一个开源的实现，`是Hadoop和Hbase的重要组件`。它是一个为分布式应用提供一致性服务的软件，提供的功能包括：`配置维护、域名服务、分布式同步、组服务等`。由于Hadoop生态系统中很多项目都依赖于zookeeper，如Pig，Hive等， 似乎很像一个动物园管理员，于是取名为Zookeeper。 [Zookeeper](https://github.com/apache/zookeeper)官网地址为[http://zookeeper.apache.org/](https://link.zhihu.com/?target=http%3A//zookeeper.apache.org/)。
>
> - **顺序一致性** 从`同一个客户端`发起的事务请求，将会`严格按照其发起顺序`被应用到zookeeper中
> - **原子性** 所有事物请求的处理结果在整个集群中所有机器上的应用情况是一致的，要么`整个集群中所有机器都成功应用了某一事务`，`要么都没有应用某一事务`，不会出现集群中部分机器应用了事务，另一部分没有应用的情况。
> - **单一视图** 无论客户端连接的是哪个zookeeper服务端，`其获取的服务端数据模型都是一致的`。
> - **可靠性** 一旦服务端成功的应用了一个事务，并完成对客户端的响应，那么`该事务所引起的服务端状态变更将会一直保留下来`，直到有另一个事务又对其进行了改变。
> - **实时性** 一旦服务端成功的应用了一个事物，那客户端立刻能看到变更后的状态

### 1.  基本概念

#### .1. 角色

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/2020060621.png)

#### .2. 网路结构

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/2020060622.png)

- Leader和各个follower是互相通信的，对于zk系统的数据都是保存在内存里面的，同样也会备份一份在磁盘上。
- 对于每个zk节点而言，可以看做每个zk节点的命名空间是一样的，也就是有同样的数据。（可查看下面的树结构）
- 如果Leader挂了，zk集群会重新选举，在毫秒级别就会重新选举出一个Leaer。
- 集群中除非有一半以上的zk节点挂了，zk service才不可用。

#### .3. 命名控件结构

> 与linux文件系统不同的是，linux文件系统有目录和文件的区别，而zk统一叫做znode，一个znode节点可以包含子znode，同时也可以包含数据。比如/Nginx/conf，/是一个znode，/Nginx是/的子znode，/Nginx还可以包含数据，数据内容就是所有安装Nginx的机器IP，/Nginx/conf是/Nginx子znode，它也可以包含内容，数据就是Nginx的配置文件内容。在应用中，我们可以通过这样一个路径就可以获得所有安装Nginx的机器IP列表，还可以获得这些机器上Nginx的配置文件。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/2020060623.png)

#### .4. 读写数据

> - 写数据，当一个客户端进行写数据请求时，会指定zk集群中节点，如果是`follower接收到写请求`，就会`把请求转发给Leader`，`Leader通过内部的Zab协议进行原子广播，直到所有zk节点都成功写了数据后（内存同步以及磁盘更新）`，这次写请求算是完成，然后`zk service就会给client发回响应`
>
> - 读数据，因为集群中所有的zk节点都呈现一个同样的命名空间视图（就是结构数据），上面的写请求已经保证了写一次数据必须保证集群所有的zk节点都是同步命名空间的，所以`读的时候可以在任意一台zk节点上`
>
> ps:其实写数据的时候不是要保证所有zk节点都写完才响应，而是`保证一半以上的节点写完了`就把这次变更更新到内存，`并且当做最新命名空间的应用`。所以在读数据的时候可能会读到不是最新的zk节点，这时候只能通过sync()解决。

#### .5. znode类型

- **PERSISTENT**：`持久化znode节点`，一旦创建这个znode点存储的数据不会主动消失，除非是客户端主动的delete。
- **SEQUENCE**：`顺序增加编号znode节点`，比如ClientA去zk service上建立一个znode名字叫做/Nginx/conf，指定了这种类型的节点后zk会创建/Nginx/conf0000000000，ClientB再去创建就是创建/Nginx/conf0000000001，ClientC是创建/Nginx/conf0000000002，以后任意Client来创建这个znode都会得到一个比当前zk命名空间最大znode编号+1的znode，也就说任意一个Client去创建znode都是保证得到的znode是递增的，而且是唯一的。
- **EPHEMERAL**：`临时znode节点`，Client连接到zk service的时候会建立一个session，之后用这个zk连接实例创建该类型的znode，一旦Client关闭了zk的连接，服务器就会清除session，然后这个session建立的znode节点都会从命名空间消失。总结就是，这个`类型的znode的生命周期是和Client建立的连接一样的`。比如ClientA创建了一个EPHEMERAL的/Nginx/conf0000000011的znode节点，一旦ClientA的zk连接关闭，这个znode节点就会消失。整个zk service命名空间里就会删除这个znode节点。
- **PERSISTENT|SEQUENTIAL**：`顺序自动编号的znode节点`，这种znoe节点会根据当前已近存在的znode节点编号自动加 1，而且不会随session断开而消失。
- **EPHEMERAL|SEQUENTIAL**：`临时自动编号节点`，znode节点编号会自动增加，但是会随session消失而消失

### 2. 工作原理

> `Zookeeper 的核心是广播`，这个机制保证了各个Server之间的同步。实现这个机制的协议叫做`Zab协议`。
>
> Zab协议有两种模式，它们分别是`恢复模式（选主`）和`广播 模式（同步）`。当服务启动或者在领导者崩溃后，Zab就进入了恢复模式，当领导者被选举出来，且大多数Server完成了和leader的状态同步以后， 恢复模式就结束了。状态同步保证了leader和Server具有相同的系统状态。`为了保证事务的顺序一致性，zookeeper采用了递增的事务id号 （zxid）来标识事务`。所有的提议（proposal）都在被提出的时候加上了zxid。实现中zxid是一个64位的数字，它高32位是epoch用 来标识leader关系是否改变，每次一个leader被选出来，它都会有一个新的epoch，标识当前属于那个leader的统治时期。低32位用于递增计数。
>
> 每个Server在工作过程中有三种状态：
>
> - `LOOKING`：当前`Server不知道leader是谁`，正在搜寻。
> - `LEADING`：当前`Server即为选举出来的leader`。
> - `FOLLOWING`：leader已经选举出来，`当前Server与之同步`。

#### .1. Sessions（会话）

> 会话对于ZooKeeper的操作非常重要。`会话中的请求按FIFO顺序执行`。一旦客户端连接到服务器，将建立会话并向客户端分配会话ID 。客户端`以特定的时间间隔发送心跳以保持会话有效`。如果ZooKeeper集合在超过服务器开启时指定的期间（会话超时）都没有从客户端接收到心跳，则它会判定客户端死机。会话超时通常`以毫秒为单位`。当会话由于任何原因结束时，在该会话期间创建的临时节点也会被删除。

#### .2. Watches（监视）

> 监视是一种简单的机制，`使客户端收到关于ZooKeeper集合中的更改的通知`。客户端可以在读取特定znode时设置Watches。`Watches会向注册的客户端发送任何znode（客户端注册表）更改的通知`。Znode更改是与znode相关的数据的修改或znode的子项中的更改。只触发一次watches。如果客户端想要再次通知，则必须通过另一个读取操作来完成。当连接会话过期时，客户端将与服务器断开连接，相关的watches也将被删除。

### 3. 安装

#### .1. 单机安装

```shell
# 查看jdk版本
java -version

# 下载 zookeeper
# 下载对应版本安装包
wget http://mirrors.bfsu.edu.cn/apache/zookeeper/zookeeper-3.6.2/apache-zookeeper-3.6.2-bin.tar.gz
# 解压到指定安装目录
tar -zxvf apache-zookeeper-3.6.2-bin.tar.gz -C /usr/local/zookeeper
# 创建数据和日志目录
cd /usr/local/zookeeper && mv apache-zookeeper-3.6.2-bin apache-zookeeper-3.6.2
mkdir -pv data logs
# 设置自己的配置文件
vi /usr/local/zookeeper/apache-zookeeper-3.6.2/conf/zoo.cfg
```

##### .1. zookeeper配置文件

```shell
# 心跳间隔 毫秒
tickTime=1000

# 启动时leader连接follower，超过多少次心跳间隔，follower连接超时
initLimit=10
# leader与follower 通信超时 最长心跳间隔次数
syncLimit=5

# 客户端连接的端口
clientPort=2181
# 客户端连接的最大数量
#maxClientCnxns=60

# 数据文件目录
dataDir=/usr/local/zookeeper/apache-zookeeper-3.6.2/data
# 日志目录
dataLogDir=/usr/local/zookeeper/apache-zookeeper-3.6.2/logs

# 最小会话超时时间
# minSessionTimeout=2000
# 最大会话超时时间
# maxSessionTimeout=20000

# 清除任务间隔（以小时为单位）
# 设置为0以禁用自动清除功能
autopurge.purgeInterval=5
# 最多保存20个文件 日志文件、快照
autopurge.snapRetainCount=20 

#开启四字命令
#4lw.commands.whitelist=*

# 集群信息
# server.A=B:C:D
# A 代表记号服务器，在dataDir目录下myid文件下记录
# B 服务器ip地址
# C 服务器与集群中的leader服务器交换信息的端口
# D 选举leader所用的端口
# server.1=127.0.0.1:3181:4181
```

##### .2. shell 命令

```shell
启动zk : bin/zkServer.sh start
查看ZK服务状态: bin/zkServer.sh status
停止ZK服务: bin/zkServer.sh stop
重启ZK服务: bin/zkServer.sh restart
连接服务器 : bin/zkCli.sh -server 127.0.0.1:2181
```

#### .2. 单机伪集群安装

- #### 指定服务机器名称

```shell
# 创建每个服务对应的目录 这里使用2181，2182，2183三个端口，data是数据目录，conf是配置文件目录，logs是日志目录。
mkdir -pv cluster/{2181/{conf,data,logs},2182/{conf,data,logs},2183/{conf,data,logs}}
```

- **指定服务机器名称**

```shell
# 指定服务机器名称， zookeeper集群中的每台机器都知道其他的机器的信息，便于在集群出现问题后，重新选举leader等。
echo 1 > cluster/2181/data/myid
echo 2 > cluster/2182/data/myid
echo 3 > cluster/2183/data/myid
```

-  **创建服务配置文件**

```shell
#为每个服务配置指定的配置文件，主要修改端口号，数据、日志目录。
# 进入2181服务目录，修改其配置文件
vi cluster/2181/conf/zoo.cfg

# 更改端口号，数据、日志目录，并配置集群机器信息
clientPort=2181
dataDir=/usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2181/data
dataLogDir=/usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2181/logs

server.1=127.0.0.1:3181:4181
server.2=127.0.0.1:3182:4182
server.3=127.0.0.1:3183:4183
```

```shell
# 复制配置文件到其他服务目录
cp cluster/2181/conf/zoo.cfg cluster/2182/conf/zoo.cfg
cp cluster/2181/conf/zoo.cfg cluster/2183/conf/zoo.cfg

# 修改其他服务配置文件 
vi cluster/2182/conf/zoo.cfg
vi cluster/2182/conf/zoo.cfg

# vim 快速替换 %s/2181/2182/g
```

- **启动集群服务**

```shell
./bin/zkServer.sh start /usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2181/conf/zoo.cfg
./bin/zkServer.sh start /usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2182/conf/zoo.cfg
./bin/zkServer.sh start /usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2183/conf/zoo.cfg
```

- **查看服务状态**

```shell
./bin/zkServer.sh status /usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2181/conf/zoo.cfg
./bin/zkServer.sh status /usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2182/conf/zoo.cfg
./bin/zkServer.sh status /usr/local/zookeeper/apache-zookeeper-3.6.2/cluster/2183/conf/zoo.cfg
```

### 4. zookeeper CLI

> ZooKeeper命令行界面（CLI）用于与ZooKeeper集合进行交互以进行开发。它有助于调试和解决不同的选项。
> 要执行ZooKeeper CLI操作，首先打开ZooKeeper服务器（“bin/zkServer.sh start”），然后打开ZooKeeper客户端（“bin/zkCli.sh”）。

#### .1. 节点属性

| 属性名         | 属性说明                                                     |
| -------------- | ------------------------------------------------------------ |
| cZxid          | 数据节点创建的事务id。                                       |
| ctime          | 数据节点创建的时间。                                         |
| mZxid          | 数据节点最后一次更新时的事务id                               |
| mtime          | 数据节点最后一次更新的时间。                                 |
| pZxid          | 数据节点的子节点最后一次被修改时的事务id。                   |
| cversion       | 子节点的更改次数。                                           |
| dataVersion    | 节点数据的更改次数，也就是数据的版本。类似于关系型数据库的乐观锁。 |
| aclVersion     | 节点ACL权限的更改次数。                                      |
| ephemeralOwner | 如果节点是临时节点，则该属性表示创建该节点的会话SessionId。如果该节点是持久节点，该属性为0，可以使用该属性来判断节点是否为临时节点。 |
| dataLength     | 数据的内容长度，单位字节。                                   |
| numChildren    | 子节点的个数                                                 |

#### .2. shell 命令

```shell
# 创建一个节点 -e为临时节点 -s为顺序节点 -c为容器节点 -t为ttl节点
create /file "file data"

# 获取当前节点数据
get /file

# 设置节点数据
set /file "data"

# 创建子节点
create /file/test "data"

# 获取当前节点状态
stat /file

# 删除一个节点
delete /file
# 删除指定版本的节点
delete -v 1 /file

# 删除节点以及其包含的子节点
deleteall /file

# 设置节点配置
# -n 子节点个数 -b 节点数据长度
# 超出配额并不会报错，而是会在日志（*.out）中出现警告
setquota /data -n 3
setquota /data -b 100

# 查看配额
listquota /data

# 删除配额
delquota /data

# 列出节点下面的所有子节点
ls -R /

# 强制同步所有的更新操作
sync /data
```

### 5. Java API

#### .1. 连接

> `每个构造器创建连接都是异步的`，构造方法启动与服务器的连接,然后立马返回,此时会话处于CONNECTING状态，通过watcher通知。此通知可以在构造方法调用返回之前或之后的任何时候到来。会话创建成功之后,状态会改为CONNECTED。

| 参数名                     | 描述                                                         |
| -------------------------- | ------------------------------------------------------------ |
| connectString              | 要创建ZooKeeper客户端对象，应用程序需要传递一个连接字符串，其中包含逗号分隔的host:port列表，每个对应一个ZooKeeper服务器。例如：127.0.0.1:2181,127.0.0.1:2182,127.0.0.1:2183实例化的ZooKeeper客户端对象将从connectString中选择一个任意服务器并尝试连接到它。如果建立连接失败，将尝试连接字符串中的另一个服务器（顺序是非确定性的，因为是随机），直到建立连接。客户端将继续尝试，直到会话显式关闭。在3.2.0版本之后,也可以在connectString后面添加后缀字符串，如：127.0.0.1:2181,127.0.0.1:2182,127.0.0.1:2183/app/a,客户端连接上ZooKeeper服务器之后，所有对ZooKeeper的操作，都会基于这个根目录。例如，客户端对/foo/bar的操作，都会指向节点/app/a/foo/bar——这个目录也叫Chroot，即客户端隔离命名空间。 |
| sessionTimeout             | 会话超时（以毫秒为单位）客户端和服务端连接创建成功之后,ZooKeeper中会建立一个会话，在一个会话周期内，ZooKeeper客户端和服务端之间会通过心跳检测机制来维持会话的有效性，一旦在sessionTimeout时间内没有进行有效的心跳检测，会话就会失效。 |
| watcher                    | 创建ZooKeeper客户端对象时,ZooKeeper允许客户端在构造方法中传入一个接口Watcher（org.apache.zookeeper.Watcher）的实现类对象来作为默认的Watcher事件通知处理器。当然，该参数可以设置为null以表明不需要设置默认的Watcher处理器。如果设置为null，日志中会有空指针异常，但是并不影响使用。 |
| canBeReadOnly              | 3.4之后添加的boolean类型的参数，用于标识当前会话是否支持“read-only”模式。默认情况下，在ZooKeeper集群中，一个机器如果和集群中过半以上机器失去了网络连接，那么这个机器将不再处理客户端请求（包括读写请求）。但是在某些使用场景下，当ZooKeeper服务器发生此类故障的时候，我们还是希望ZooKeeper服务器能够提供读服务（当然写服务肯定无法提供）——这就是ZooKeeper的“read-only”模式。 |
| sessionId 和 sessionPasswd | 会话id和 会话密码，这两个参数能够唯一确定一个会话，同时客户端使用这两个参数实现客户端会话复用，从而达到恢复会话的效果，使用方法：第一次连接上ZooKeeper服务器后，客户端使用getSessionId()和getSessionPasswd()获取这两个值，如果需要会话复用,在重新创建ZooKeeper客户端对象的时候可以传过去，如果不需要会话复用，请使用不需要这些参数的其他构造函数。 |
| HostProvider               | 客户端地址列表管理器                                         |

```java
package cn.sivan.test;

import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooKeeper;

import java.util.concurrent.CountDownLatch;

/**
 * zookeeper的连接和数据库的连接不同，数据库通过DriverManager的getConnect方法就可以直接获取到连接。
 * 但zookeeper在获取连接的过程中，使用了Future。也就意味着，new之后所拿到的仅仅是一个zookeeper对象，而这个对象可能还没有连接到zookeeper服务。
 */
public class Connect implements Watcher {

    private CountDownLatch countDownLatch = new CountDownLatch(1);

    @Override
    public void process(WatchedEvent watchedEvent) {

        if (watchedEvent.getState() == Event.KeeperState.SyncConnected) {
            //连接创建成功，唤醒等待线程。
            countDownLatch.countDown();
        }
    }

    /**
     * 连接zookeeper
     * @param connStr address
     * @param timeout 超时时间
     */
    public ZooKeeper connect(String connStr, int timeout) {

        try {
            //创建zookeeper
            ZooKeeper zooKeeper = new ZooKeeper(connStr, timeout, this);

            //当前线程 等待连接连接成功的通知
            countDownLatch.await();
            return zooKeeper;
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }
}
```

| state             | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| Disconnected      | 客户端处于断开状态 - 未连接                                  |
| SyncConnected     | 客户端处于连接状态 - 已连接                                  |
| AuthFailed        | 验证失败状态                                                 |
| ConnectedReadOnly | 客户端连接到只读服务器                                       |
| SaslAuthenticated | 客户端已通过 SASL 认证，可以使用其 SASL 授权的权限执行 Zookeeper 操作 |
| Expired           | 会话已失效                                                   |
| Closed            | 客户端已关闭，客户端调用时在本地生成                         |

#### .2. API 使用

```java
package cn.sivan.test;

import org.apache.zookeeper.AddWatchMode;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.ZooDefs;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.ACL;
import org.apache.zookeeper.data.Id;
import org.apache.zookeeper.data.Stat;
import org.apache.zookeeper.server.auth.DigestAuthenticationProvider;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ZookeeperSimple {

    public static void main(String[] args) throws Exception {
        String path = "/chodble";
        String address = "127.0.0.1:2181";
        int timeout = 3000;

        Connect zkConnect = new Connect();
        ZooKeeper zk = zkConnect.connect(address, timeout);

        //判断节点是否存在
        if (null == zk.exists(path, true)) {

            //设置认证方式
            Id ADMIN_IDS = new Id("digest", DigestAuthenticationProvider.generateDigest("root:root"));
            Id USER_IDS = new Id("digest", DigestAuthenticationProvider.generateDigest("user:user"));
            Id ANYONE_ID_UNSAFE = ZooDefs.Ids.ANYONE_ID_UNSAFE;

            /**
             *设置ACL权限
             *Create  允许对子节点Create 操作
             *Read    允许对本节点GetChildren 和GetData 操作
             *Write   允许对本节点SetData 操作
             *Delete  允许对子节点Delete 操作(本节点也可以删除)
             *Admin   允许对本节点setAcl 操作
             *ALL = READ | WRITE | CREATE | DELETE | ADMIN;
             */
            List<ACL> aclList = new ArrayList<>();
            aclList.add(new ACL(ZooDefs.Perms.ADMIN, ADMIN_IDS));
            //aclList.add(new ACL(ZooDefs.Perms.CREATE | ZooDefs.Perms.READ | ZooDefs.Perms.WRITE | ZooDefs.Perms.DELETE, USER_IDS));
            aclList.add(new ACL(ZooDefs.Perms.READ, ANYONE_ID_UNSAFE));

            /**
             * 创建根节点
             * 不支持递归创建，即无法在父节点不存在的情况下创建一个子节点
             * 一个节点已经存在了，那么创建同名节点的时候，会抛出NodeExistException异常。如果是顺序节点,那么永远不会抛出NodeExistException异常
             * 临时节点不能有子节点
             *
             * CreateMode
             * PERSISTENT : 持久节点
             * PERSISTENT_SEQUENTIAL : 持久顺序节点
             * EPHEMERAL : 临时节点
             * EPHEMERAL_SEQUENTIAL : 临时顺序节点
             * CONTAINER : 容器节点
             * PERSISTENT_WITH_TTL : 持久TTL节点
             * PERSISTENT_SEQUENTIAL_WITH_TTL : 持久顺序TTL节点
             */
            zk.create(path, "宁川".getBytes(), aclList, CreateMode.PERSISTENT);

            //设置Acl
            zk.setACL(path, Collections.singletonList(new ACL(ZooDefs.Perms.CREATE | ZooDefs.Perms.READ | ZooDefs.Perms.WRITE | ZooDefs.Perms.DELETE, USER_IDS)), -1);
        }

        //进行认证
        zk.addAuthInfo("digest", "user:user".getBytes());

        //查看ACL权限
        Stat stat = null;
        if ((stat = zk.exists(path, true)) != null) {
            System.out.println("查看节点权限：" + zk.getACL(path, stat));
        }

        //获取子节点数量
        System.out.println("获取子节点数量：" + zk.getAllChildrenNumber(path));

        //获取所有的节点
        System.out.println("所有的节点:" + zk.getChildren("/", false));

        //获取节点的值，并设置监听
        System.out.println("获取节点的值：" + new String(zk.getData(path, (event -> {
            System.out.println("getWatch：" + event.toString());
        }), stat)));

        //添加监听
        zk.addWatch(path, (event) -> {
            System.out.println("addWatch：" + event.toString());
        }, AddWatchMode.PERSISTENT);

        //设置节点数据
        // 1 -自动维护
        System.out.println("设置节点数据：" + zk.setData(path, "123".getBytes(), -1));

        /**
         * 异步创建一个 临时顺序节点，ACL为 ip:127.0.0.1:c
         */
        zk.create("/node",
                "123".getBytes(),
                Collections.singletonList(new ACL(ZooDefs.Perms.CREATE, new Id("ip", "127.0.0.1"))),
                CreateMode.EPHEMERAL_SEQUENTIAL,
                //new AsyncCallback.StringCallback()
                (rc, path1, ctx, name) -> {
                    System.out.println("rc:" + rc);
                    System.out.println("path:" + path1);
                    System.out.println("ctx:" + ctx);
                    System.out.println("name:" + name);
                }, "传给服务端的内容,会在异步回调时传回来");
        //等待执行结果
        Thread.sleep(2000);

        //删除节点
        if ((stat = zk.exists(path, true)) != null) {
            List<String> subPaths = zk.getChildren(path, false);
            if (subPaths.isEmpty()) {
                zk.delete(path, stat.getVersion());
            } else {
                for (String subPath : subPaths) {
                    zk.delete(path + "/" + subPath, -1);
                }
            }
        }
    }
}
```

#### .3. CreateModel

```java
public enum CreateMode {

    /**
     * 持久节点
     * The znode will not be automatically deleted upon client's disconnect.
     */
    PERSISTENT(0, false, false, false, false),

    /**
     * 持久顺序节点
     * The znode will not be automatically deleted upon client's disconnect,
     * and its name will be appended with a monotonically increasing number.
     */
    PERSISTENT_SEQUENTIAL(2, false, true, false, false),

    /**
     * 临时节点
     * The znode will be deleted upon the client's disconnect.
     */
    EPHEMERAL(1, true, false, false, false),

    /**
     * 临时顺序节点
     * The znode will be deleted upon the client's disconnect, and its name
     * will be appended with a monotonically increasing number.
     */
    EPHEMERAL_SEQUENTIAL(3, true, true, false, false),

    /**
     * 容器节点
     * The znode will be a container node. Container
     * nodes are special purpose nodes useful for recipes such as leader, lock,
     * etc. When the last child of a container is deleted, the container becomes
     * a candidate to be deleted by the server at some point in the future.
     * Given this property, you should be prepared to get
     * {@link org.apache.zookeeper.KeeperException.NoNodeException}
     * when creating children inside of this container node.
     */
    CONTAINER(4, false, false, true, false),

    /**
     * 持久TTL节点
     * The znode will not be automatically deleted upon client's disconnect.
     * However if the znode has not been modified within the given TTL, it
     * will be deleted once it has no children.
     */
    PERSISTENT_WITH_TTL(5, false, false, false, true),

    /**
     * 持久顺序TTL节点
     * The znode will not be automatically deleted upon client's disconnect,
     * and its name will be appended with a monotonically increasing number.
     * However if the znode has not been modified within the given TTL, it
     * will be deleted once it has no children.
     */
    PERSISTENT_SEQUENTIAL_WITH_TTL(6, false, true, false, true);
}
```

#### .4. Zoodefs.Ids

```java
public interface Ids {

        /**
         * world:anyone:adrwa
         * This Id represents anyone.
         */
        Id ANYONE_ID_UNSAFE = new Id("world", "anyone");

        /**
         * 认证后可操作
         * This Id is only usable to set ACLs. It will get substituted with the
         * Id's the client authenticated with.
         */
        Id AUTH_IDS = new Id("auth", "");

        /**
         * 完全开放的ACL，任何连接的客户端都可以操作该属性znode
         * This is a completely open ACL .
         */
        @SuppressFBWarnings(value = "MS_MUTABLE_COLLECTION", justification = "Cannot break API")
        ArrayList<ACL> OPEN_ACL_UNSAFE = new ArrayList<ACL>(Collections.singletonList(new ACL(Perms.ALL, ANYONE_ID_UNSAFE)));

        /**
         * 只有创建者才有ACL权限
         * This ACL gives the creators authentication id's all permissions.
         */
        @SuppressFBWarnings(value = "MS_MUTABLE_COLLECTION", justification = "Cannot break API")
        ArrayList<ACL> CREATOR_ALL_ACL = new ArrayList<ACL>(Collections.singletonList(new ACL(Perms.ALL, AUTH_IDS)));

        /**
         * 只读ACL
         * This ACL gives the world the ability to read.
         */
        @SuppressFBWarnings(value = "MS_MUTABLE_COLLECTION", justification = "Cannot break API")
        ArrayList<ACL> READ_ACL_UNSAFE = new ArrayList<ACL>(Collections.singletonList(new ACL(Perms.READ, ANYONE_ID_UNSAFE)));
}
```

#### .5. Watch

> 一个Watch事件是一个一次性的触发器，当被设置了Watch的数据发生了改变的时候，则服务器将这个改变发送给设置了Watch的客户端，以便通知它们。
>
> watch机制的特点：
>
> - 一次性触发 数据发生改变时，一个watcher event会被发送到client，但是client只会收到一次这样的信息。
> - watcher event异步发送 watcher 的通知事件从server发送到client是异步的，这就存在一个问题，不同的客户端和服务器之间通过socket进行通信，由于网络延迟或其他因素导致客户端在不通的时刻监听到事件，由于Zookeeper本身提供了ordering guarantee，即客户端监听事件后，才会感知它所监视znode发生了变化。
> - 数据监视 Zookeeper有数据监视和子数据监视 getdata() and exists() 设置数据监视，getchildren()设置了子节点监视
> - 注册watcher getData、exists、getChildren
> - 触发watcher create、delete、setData

```java
nodedatachanged # 节点数据改变
nodecreate # 节点创建事件
nodedelete #节点删除事件
nodechildrenchanged # 子节点改变事件
package cn.sivan.test;

import org.apache.zookeeper.AddWatchMode;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.ZooDefs;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.ACL;
import org.apache.zookeeper.data.Id;
import org.apache.zookeeper.data.Stat;
import org.apache.zookeeper.server.auth.DigestAuthenticationProvider;

import java.util.Collections;

public class WatchTest {
    public static void main(String[] args) throws Exception {

        String path = "/chodble2";
        String pathValue = "root-value";

        String subPath = path + "/node";
        String subPathValue = "sbu-value";

        String address = "127.0.0.1:2181";
        int timeout = 3000;

        Connect zkConnect = new Connect();
        ZooKeeper zk = zkConnect.connect(address, timeout);

        //判断节点是否存在
        Stat stat = null;
        if ((stat = zk.exists(path, true)) == null) {

            //创建节点
            zk.create(path, pathValue.getBytes(), Collections.singletonList(new ACL(ZooDefs.Perms.ALL, ZooDefs.Ids.ANYONE_ID_UNSAFE)), CreateMode.PERSISTENT);
        }

        //进行认证
        zk.addAuthInfo("digest", "user:user".getBytes());

        //设置节点监听
        zk.addWatch(path, event -> {
            System.out.println("addWatch-1：" + event);
        }, AddWatchMode.PERSISTENT);

        //获取值
        zk.getData(path, true, new Stat());

        //设置值
        zk.setData(path, "new-data".getBytes(), -1);

        //测试通知
        for (int i = 0; i < 10; i++) {
            //重新设置值
            zk.setData(path, ("new-data" + i).getBytes(), -1);
        }

        //设置权限
        zk.setACL(path, Collections.singletonList(new ACL(ZooDefs.Perms.ALL, new Id("digest", DigestAuthenticationProvider.generateDigest("user:user")))), -1);

        //创建子节点
        zk.create(subPath, subPathValue.getBytes(), Collections.singletonList(new ACL(ZooDefs.Perms.ALL, ZooDefs.Ids.AUTH_IDS)), CreateMode.PERSISTENT);

        //获取子节点的值
        zk.getData(subPath, true, new Stat());

        //设置子节点的值
        zk.setData(subPath, "sub-new-data".getBytes(), -1);

        //设置子节点权限
        zk.setACL(subPath, Collections.singletonList(new ACL(ZooDefs.Perms.ALL, new Id("digest", DigestAuthenticationProvider.generateDigest("user:user")))), -1);

        //获取子节点的个数
        zk.getAllChildrenNumber(path);

        //列出所有的子节点
        zk.getChildren(path, true);

        //查看节点ACl
        if ((stat = zk.exists(path, true)) != null) {
            System.out.println("查看节点权限：" + zk.getACL(path, stat));
        }

        //删除子节点
        zk.delete(subPath, -1);

        //删除节点
        zk.delete(path, -1);
    }
}
```

### Resource

- http://www.uml.org.cn/zjjs/202006062.asp
- https://www.cnblogs.com/sivanchan/p/13763030.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/zookeeper/  

