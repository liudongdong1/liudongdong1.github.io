# 通信机制

一对一的交互方式：

- Request/response： 标准的请求/应答方式。客户端发送请求到服务并等待应答。客户端期待应答在适当的时间内到达。在基于线程的程序中，发起请求的线程在等待时可能阻塞。
- Notification(或者单路请求)： 客户端发送请求到服务但是不期待有答复。
- Request/async response： 客户端发送请求到服务，服务器端异步给回复。客户端在等待时不阻塞，而在设计上通常是假定可能不会很快就有应答。

一对多的交互方式：

- Publish/subscribe： 客户端发布通知消息，然后被0个或者多个关注的服务消费
- Publish/async responses： 客户端发布请求消息，然后等待一定数量的从关注的服务返回的应答

|              | One-to-One             | One-to-Many             |
| ------------ | ---------------------- | ----------------------- |
| Synchronous  | Request/response       |                         |
| Asynchronous | Notification           | Publish/subscribe       |
| Asynchronous | Request/async response | Publish/async responses |

### 1. Rest
>Spring MVC, Spark Framework, Spring hateoas(服务器使用Spring MVC，客户端使用Jersey)

### 2. RPC
>新 gRPC, 稳 thrift，旧 hessian2

### 3. 定制

#### .1.  网络通讯--netty
>Netty提供异步的、事件驱动的网络应用程序框架和工具，用以快速开发高性能、高可靠性的网络服务器和客户端程序。

#### .2. 通信协议--HTTP/TCP/UDP/socket
>An HTTP & HTTP/2 client for Android and Java applications, 支持HTTP/2, Android;
#### .3. 编码方案--Rest/Protocol、buffer/thrift

### 4. 服务注册&发现

#### .1. Zookeeper
>来自 Apache，以Java语言编写，强一致性(CP), 使用 Zab 协议 (基于PAXOS)。使用zookeeper时，不要直接使用zk的Java api，请尽量使用 Curator ！
#### .2. Etcd

>据说是受 zookeeper 和 Doozer 启发，用 Go 语言编写，使用 Raft 协议。etcd 2.* 版本提供 HTTP+JSON 的 API，而最新的 etcd 3.0 版本修改为 gRPC，效率大为提升。

#### .3. [Eureka](https://github.com/Netflix/eureka)

> 来自 Netflix， 服务器端和客户端都是用 Java 语言编写，因此只能用于Java和基于JVM的语言。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/implementation_list.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E9%80%9A%E4%BF%A1%E6%9C%BA%E5%88%B6/  

