# Kubernetes-etcd使用


> etcd 是一个`分布式的、可靠的 key-value 存储系统`，它用于存储分布式系统中的`关键数据`，这个定义非常重要。 
>
> 一个 etcd 集群，通常会由 3 个或者 5 个节点组成，多个节点之间通过 Raft 一致性算法的完成分布式一致性协同，算法会选举出一个主节点作为 leader，由 leader 负责数据的同步与数据的分发。当 leader 出现故障后系统会自动地选取另一个节点成为 leader，并重新完成数据的同步。客户端在多个节点中，仅需要选择其中的任意一个就可以完成数据的读写，内部的状态及数据协同由 etcd 自身完成。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16eeec569c66ba18tplv-t2oaga2asx-zoom-in-crop-mark3024000.awebp)

### API

- 第一组是 Put 与 Delete。上图可以看到 put 与 delete 的操作都非常简单，只需要提供一个 key 和一个 value，就可以向集群中写入数据了，删除数据的时候只需要指定 key 即可；
- 第二组是查询操作。etcd 支持两种类型的查询：第一种是指定单个 key 的查询，第二种是指定的一个 key 的范围；
- 第三组是数据订阅。etcd 提供了 Watch 机制，我们可以利用 watch 实时订阅到 etcd 中增量的数据更新，watch 支持指定单个 key，也可以指定一个 key 的前缀，在实际应用场景中的通常会采用第二种形势；
- 第四组事务操作。etcd 提供了一个简单的事务支持，用户可以通过指定一组条件满足时执行某些动作，当条件不成立的时候执行另一组操作，类似于代码中的 if else 语句，etcd 确保整个操作的原子性；
- 第五组是 Leases 接口。Leases 接口是分布式系统中常用的一种设计模式，其用法后面会具体展开。

### 数据版本机制

对于每一个 KeyValue 数据节点，etcd 中都记录了三个版本：

- 第一个版本叫做 create_revision，是 KeyValue 在创建时对应的 revision；
- 第二个叫做 mod_revision，是其数据被操作的时候对应的 revision；
- 第三个 version 就是一个计数器，代表了 KeyValue 被修改了多少次。

### MVCC&Streaming watch

`etcd 中所有的数据都存储在一个 b+tree 中（灰色）`，该 b+tree 保存`在磁盘`中，并通过 mmap 的方式映射到内存用来支持快速的访问。`灰色的 b+tree 中维护着 revision 到 value 的映射关系`，支持通过 revision 查询对应的数据。因为 revision 是单调递增的，当我们`通过 watch 来订阅指定 revision 之后的数据时，仅需要订阅该 b+ tree 的数据变化即可`。

在 etcd 中会运行一个**周期性的 Compaction 的机制**来清理历史数据，将一段时间之前的同一个 Key 的多个历史版本数据清理掉。最终的结果是灰色的 b+tree 依旧保持单调递增，但可能会出现一些空洞。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16eeec56c2ade062tplv-t2oaga2asx-zoom-in-crop-mark3024000.awebp)

### 分布式租约--lease

- 在分布式系统中需要去检测一个节点是否存活的时，就需要租约机制。
- 如果希望这个租约永不过期，需要周期性的调用 KeeyAlive 方法刷新租约。比如说需要检测分布式系统中一个进程是否存活，可以`在进程中去创建一个租约，并在该进程中周期性的调用 KeepAlive 的方法`。如果一切正常，该节点的租约会一致保持，如果这个进程挂掉了，最终这个租约就会自动过期。

### 应用场景

#### 元数据存储

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16eeec56cacb26a0tplv-t2oaga2asx-zoom-in-crop-mark3024000.awebp)

#### 服务发现

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16eeec56cdbaa6d2tplv-t2oaga2asx-zoom-in-crop-mark3024000.awebp)

#### 分布式选举

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16eeec56ce76b1b6tplv-t2oaga2asx-zoom-in-crop-mark3024000.awebp)

### etcd 命令

```
存储:
    curl http://127.0.0.1:4001/v2/keys/testkey -XPUT -d value='testvalue'
    curl -s http://127.0.0.1:4001/v2/keys/message2 -XPUT -d value='hello etcd' -d ttl=5

获取:
    curl http://127.0.0.1:4001/v2/keys/testkey

查看版本:
    curl  http://127.0.0.1:4001/version

删除:
    curl -s http://127.0.0.1:4001/v2/keys/testkey -XDELETE

监视:
    窗口1：curl -s http://127.0.0.1:4001/v2/keys/message2 -XPUT -d value='hello etcd 1'
          curl -s http://127.0.0.1:4001/v2/keys/message2?wait=true
    窗口2：
          curl -s http://127.0.0.1:4001/v2/keys/message2 -XPUT -d value='hello etcd 2'

自动创建key:
    curl -s http://127.0.0.1:4001/v2/keys/message3 -XPOST -d value='hello etcd 1'
    curl -s 'http://127.0.0.1:4001/v2/keys/message3?recursive=true&sorted=true'

创建目录：
    curl -s http://127.0.0.1:4001/v2/keys/message8 -XPUT -d dir=true

删除目录：
    curl -s 'http://127.0.0.1:4001/v2/keys/message7?dir=true' -XDELETE
    curl -s 'http://127.0.0.1:4001/v2/keys/message7?recursive=true' -XDELETE

查看所有key:
    curl -s http://127.0.0.1:4001/v2/keys/?recursive=true

存储数据：
    curl -s http://127.0.0.1:4001/v2/keys/file -XPUT --data-urlencode value@upfile


使用etcdctl客户端：
存储:
    etcdctl set /liuyiling/testkey "610" --ttl '100'
                                         --swap-with-value value

获取：
    etcdctl get /liuyiling/testkey

更新：
    etcdctl update /liuyiling/testkey "world" --ttl '100'

删除：
    etcdctl rm /liuyiling/testkey

使用ca获取：
etcdctl --cert-file=/etc/etcd/ssl/etcd.pem   --key-file=/etc/etcd/ssl/etcd-key.pem  --ca-file=/etc/etcd/ssl/ca.pem get /message

目录管理：
    etcdctl mk /liuyiling/testkey "hello"    类似set,但是如果key已经存在，报错

    etcdctl mkdir /liuyiling 

    etcdctl setdir /liuyiling  

    etcdctl updatedir /liuyiling      

    etcdctl rmdir /liuyiling    

查看：
    etcdctl ls --recursive

监视：
    etcdctl watch mykey  --forever         +    etcdctl update mykey "hehe"

    #监视目录下所有节点的改变

    etcdctl exec-watch --recursive /foo -- sh -c "echo hi"

    etcdctl exec-watch mykey -- sh -c 'ls -al'    +    etcdctl update mykey "hehe"

    etcdctl member list


集群启动步骤

1.启动一个etcd，任意机器，如192.168.1.1:2379

2.curl -X PUT http://192.168.1.1:2379/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0f222/_config/size -d value=3

3.etcd -name machine1 -initial-advertise-peer-urls http://127.0.0.1:2380 -listen-peer-urls http://127.0.0.1:2380 -discovery http://192.168.1.1:2379/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0f222

4.如果是在三台不同的服务器上，则重复上面的命令3次，否则重复上面的命令1次+下面的命令2次
etcd -name machine2 -discovery http://192.168.1.1:2379/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0f222 -addr 127.0.0.1:2389 -bind-addr 127.0.0.1:2389 -peer-addr 127.0.0.1:2390 -peer-bind-addr 127.0.0.1:2390

etcd -name machine3 -discovery http://192.168.1.1:2379/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0f222 -addr 127.0.0.1:2409 -bind-addr 127.0.0.1:2409 -peer-addr 127.0.0.1:2490 -peer-bind-addr 127.0.0.1:2490

5.curl -L http://localhost:2379/v2/members | python -m json.tool



```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes_etcd%E4%BD%BF%E7%94%A8/  

