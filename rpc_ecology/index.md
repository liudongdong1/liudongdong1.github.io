# rpc_ecology


> `RPC（Remote Procedure Call）`—远程过程调用，它是一种`通过网络从远程计算机程序上请求服务，而不需要了解底层网络技术的协议`。这里的屏蔽底层网络技术包括 传输协议就是`屏蔽TCP/UDP`，`序列化方式等`。RPC框架的目标就是`让调用方像调用本地应用一样调用远程服务`，而不关心服务提供方在哪里。
>
> - RPC框架`一般使用长链接`，不必每次通信都要3次握手，减少网络开销;
> - RPC框架`一般都有注册中心`，有丰富的`监控管理`; `发布、下线接口、动态扩展等`，对调用方来说是无感知、统一化的操作; `协议私密，安全性较高`;
> - RPC 协议`更简单内容更小`，效率更高，服务化架构、服务化治理，RPC框架是一个强力的支撑。
> - RPC 会`隐藏底层的通讯细节`（不需要直接处理Socket通讯或Http通讯） RPC 是一个请求响应模型。

#### 1. 相关通信技术

##### 1. HTTP协议

> http协议是基于tcp协议的，tcp协议是流式协议，包头部分可以通过多出的\r\n来分界，包体部分如何分界呢？这是协议本身要解决的问题。目前一般有两种方式，第一种方式就是在包头中有个content-Length字段，这个字段的值的大小标识了POST数据的长度，服务器收到一个数据包后，先从包头解析出这个字段的值，再根据这个值去读取相应长度的作为http协议的包体数据。

##### 2. RESTful API (http+json)

> 网站即软件，而且是一种新型的软件，这种"互联网软件"`采用客户端/服务器模式`，建立在分布式体系上，通过互联网通信，具有高延时（high latency）、高并发等特点。它首次出现在 2000 年 [Roy Fielding 的博士论文](https://link.zhihu.com/?target=http%3A//www.ics.uci.edu/~fielding/pubs/dissertation/top.htm)中，他是 HTTP 规范的主要编写者之一。`Representational State Transfer，翻译是”表现层状态转化”`，通俗来讲就是：`资源在网络中以某种表现形式进行状态转移。`
>  总结一下什么是RESTful架构：
>  　（1）`每一个URI代表一种资源`；
>  　（2）客户端和服务器之间，`传递这种资源的某种表现层`，比如`用JSON，XML，JPEG等；`
>  　（3）客户端通过`四个HTTP动词`，对服务器端资源进行操作，实现"表现层状态转化"。
>
> URL定位**资源**，用HTTP动词（GET,POST,DELETE,DETC）描述操作。用HTTP协议里的动词来实现资源的添加，修改，删除等操作。即通过HTTP动词来实现资源的状态扭转：GET 用来获取资源，POST 用来新建资源（也可以用于更新资源)，PUT 用来更新资源，DELETE 用来删除资源。

##### 3. RPC

> 进程间通信（IPC，Inter-Process Communication），指`至少两个进程或线程间传送数据或信号的一些技术或方法`。进程是计算机系统分配资源的最小单位。每个进程都有自己的一部分独立的系统资源，彼此是隔离的。为了能使不同的进程互相访问资源并进行协调工作，才有了进程间通信。这些进程可以运行在同一计算机上或网络连接的不同计算机上。 进程间通信技术包括`消息传递、同步、共享内存和远程过程调用`。 IPC是一种标准的Unix通信机制。
>
> - `本地过程调用(LPC)`用在多任务操作系统中，使得同时运行的任务能互相会话。这些任务共享内存空间使任务同步和互相发送信息。
> - `远程过程调用(RPC)`类似于LPC，只是在网上工作。RPC开始是出现在Sun微系统公司和HP公司的运行UNIX操作系统的计算机中。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725122130536.png)

- 服务消费者(Client 客户端)通过本地调用的方式调用服务。
- 客户端存根(Client Stub)接收到调用请求后负责将方法、入参等信息序列化(组装)成能够进行网络传输的消息体。
- 客户端存根(Client Stub)找到远程的服务地址，并且将消息通过网络发送给服务端。
- 服务端存根(Server Stub)收到消息后进行解码(反序列化操作)。
- 服务端存根(Server Stub)根据解码结果调用本地的服务进行相关处理
- 服务端(Server)本地服务业务处理。
- 处理结果返回给服务端存根(Server Stub)。
- 服务端存根(Server Stub)序列化结果。
- 服务端存根(Server Stub)将结果通过网络发送至消费方。
- 客户端存根(Client Stub)接收到消息，并进行解码(反序列化)。
- 服务消费方得到最终结果。

#### 2. RPC概念

##### 1. 组成结构

> - `RPC 服务方`通过` RpcServer 去导出（export）远程接口方法`，而`客户方通过 RpcClient 去引入(import)远程接口方法`。客户方像调用本地方法一样去调用远程接口方法，RPC 框架提供接口的代理实现，实际的调用将`委托给代理RpcProxy` 。`代理封装调用信息并将调用转交给RpcInvoker 去实际执行`。`在客户端的RpcInvoker 通过连接器RpcConnector 去维持与服务端的通道RpcChannel`，并使用`RpcProtocol 执行协议编码（encode）`并将编码后的请求消息通过通道发送给服务方。
>
> - RPC 服务端接收器 RpcAcceptor 接收客户端的调用请求，同样使用RpcProtocol 执行协议解(decode)。解码后的调用信息传递给RpcProcessor 去控制处理调用过程，最后再`委托调用给RpcInvoker 去实际执行并返回调用结果。`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725124707440.png)

1. RpcServer: 负责`导出（export）远程接口`
2. RpcClient: 负责`导入（import）远程接口的代理实现`
3. RpcProxy: 远程接口的代理实现
4. RpcInvoker: 客户方实现：`负责编码调用信息和发送调用请求到服务方并等待调用结果返回`; 服务方实现：负责`调用服务端接口的具体实现并返回调用结果`
5. RpcProtocol: 负责协议编/解码
6. RpcConnector: 负责`维持客户方和服务方的连接通道和发送数据到服务方`
7. RpcAcceptor: 负责接收客户方请求并返回请求结果
8. RpcProcessor: 负责在服务方控制调用过程，包括管理调用线程池、超时时间等
9. RpcChannel: 数据传输通道

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725123019972.png)

##### 2. 服务调用方式

> RPC服务调用方式 分为  同步阻塞调用； 异步非阻塞调用

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725124540929.png)

###### .1. 导出远程接口

> 导出远程接口的意思是指`只有导出的接口可以供远程调用`，而未导出的接口则不能。

```java
DemoService demo   = new ...;
RpcServer   server = new ...;
server.export(DemoService.class, demo, options);

//可以导出整个接口，也可以更细粒度一点只导出接口中的某些方法, 如导出 DemoService 中签名为 hi(String s) 的方法
server.export(DemoService.class, demo, "hi", new Class<?>[] { String.class }, options);

//多态语义调用问题
DemoService demo   = new ...;
DemoService demo2  = new ...;
RpcServer   server = new ...;
server.export(DemoService.class, demo, options);
server.export("demo2", DemoService.class, demo2, options); //导出的时候添加标记
```

###### .2. 导入远程接口与客户端代理

```java
//代码生成的方式对跨语言平台 RPC 框架而言是必然的选择，而对于同一语言平台的 RPC 则可以通过共享接口定义来实现。在 java 中导入接口的代码片段可能如下：
RpcClient client = new ...;
DemoService demo = client.refer(DemoService.class);  //类似java import 实现导入功能
demo.hi("how are you?");
```

###### .3. 协议编码

> 编码的信息越少越好（传输数据少），编码的规则越简单越好（执行效率高）；
>
> - 调用编码 ：
>   1. `接口方法`： 包括接口名、方法名
>   2. `方法参数`： 包括参数类型、参数值
>   3. `调用属性`： 包括调用属性信息，例如调用附件隐式参数、调用超时时间等
> - 返回编码 ：
>    1. `返回结果`：接口方法中定义的返回值
>    2. 返回码：  异常返回码
>    3. 返回异常信息：调用异常信息

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725130101867.png)

> - 消息头 
>   - magic      : 协议魔数，为解码设计
>   - header size: 协议头长度，为扩展设计
>   - version    : 协议版本，为兼容设计
>   - st         : 消息体序列化类型
>   - hb         : 心跳消息标记，为长连接传输层心跳设计
>   - ow         : 单向消息标记，
>   - rp         : 响应消息标记，不置位默认是请求消息
>   - status code: 响应消息状态码
>   - reserved   : 为字节对齐保留
>   - message id : 消息 id
>   - body size  : 消息体长度
>
> - 消息体 :  采用序列化编码，常见有以下格式
>   - xml   : 如 webservie soap
>   - json  : 如 JSON-RPC
>   - binary: 如 thrift; hession; kryo 等

###### .4.  异常处理

1. `本地调用一定会执行，而远程调用则不一定`，调用消息可能因为网络原因并未发送到服务方。
2. `本地调用只会抛出接口声明的异常`，而`远程调用还会跑出 RPC 框架运行时的其他异常。`
3. 本地调用和远程调用的性能可能差距很大，这取决于 RPC 固有消耗所占的比重。

#### 3. 框架分类

- 支持`多语言的 RPC 框架`，比较成熟的有 Google 的 gRPC、Apache（Facebook）的 Thrift；

- 只支持`特定语言的 RPC 框架`，例如新浪微博的 Motan；
- 支持`服务治理等服务化特性的分布式服务框架`，其底层内核仍然是 RPC 框架, 例如阿里的 Dubbo。

##### 1. Dubbo

![img](https://gitee.com/github-25970295/blogpictureV2/raw/master/16caf43c8f281580)

> 从图中你能看到，Dubbo 的架构主要包含四个角色，其中 Consumer 是服务消费者，Provider 是服务提供者，Registry 是注册中心，Monitor 是监控系统。具体的交互流程是` Consumer 一端通过注册中心获取到 Provider 节点后`，`通过 Dubbo 的客户端 SDK 与 Provider 建立连接，并发起调用。`Provider 一端通过 Dubbo 的服务端 SDK 接收到 Consumer 的请求，处理后再把结果返回给 Consumer。

##### 2. Motan

![img](https://gitee.com/github-25970295/blogpictureV2/raw/master/16caf43c8ce67a63)

> Motan 与 Dubbo 的架构类似，都需要在 Client 端（服务消费者）和 Server 端（服务提供者）引入 SDK，其中 Motan 框架主要包含下面几个功能模块。
>
> - register：用来和注册中心交互，包括注册服务、订阅服务、服务变更通知、服务心跳发送等功能。
> - protocol：用来进行 RPC 服务的描述和 RPC 服务的配置管理，这一层还可以`添加不同功能的 filter 用来完成统计、并发限制等功能。`
> - serialize：`将 RPC 请求中的参数、结果等对象进行序列化与反序列化`
> - transport：用来`进行远程通信，默认使用 Netty NIO 的 TCP 长链接方式。`
> - cluster：请求时会根据不同的高可用与负载均衡策略选择一个可用的 Server 发起远程调用。

##### 3. Tars

Tars 是腾讯根据内部多年使用微服务架构的实践，总结而成的开源项目，仅支持 C++ 语言，它的架构图如下。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/16caf43c8de17c70)

> Tars 的架构交互主要包括以下几个流程：
>
> - `服务发布流程`：在 web 系统上传 server 的发布包到 patch，上传成功后，在 web 上提交发布 server 请求，由 registry 服务传达到 node，然后 node 拉取 server 的发布包到本地，拉起 server 服务。
> - `管理命令流程`：web 系统上的可以提交管理 server 服务命令请求，由 registry 服务传达到 node 服务，然后由 node 向 server 发送管理命令。
> - `心跳上报流程`：server 服务运行后，会定期上报心跳到 node，node 然后把服务心跳信息上报到 registry 服务，由 registry 进行统一管理。
> - `信息上报流程`：server 服务运行后，会定期上报统计信息到 stat，打印远程日志到 log，定期上报属性信息到 prop、上报异常信息到 notify、从 config 拉取服务配置信息。
> - `client 访问 server 流程`：client 可以通过 server 的对象名 Obj 间接访问 server，client 会从 registry 上拉取 server 的路由信息（如 IP、Port 信息），然后根据具体的业务特性（同步或者异步，TCP 或者 UDP 方式）访问 server（当然 client 也可以通过 IP/Port 直接访问 server）。

##### 4. Spring Cloud

Spring Cloud 利用 Spring Boot 特性整合了开源行业中优秀的组件，整体对外提供了一套在微服务架构中服务治理的解决方案。只支持 Java 语言平台，它的架构图可以用下面这张图来描述。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/16caf43c8f2215fd)

由此可见，Spring Cloud 微服务架构是由多个组件一起组成的，各个组件的交互流程如下。

- `请求统一通过 API 网关 Zuul 来访问内部服务`，`先经过 Token 进行安全认证`。
- 通过安全认证后，`网关 Zuul 从注册中心 Eureka 获取可用服务节点列表`。
- `从可用服务节点中选取一个可用节点，然后把请求分发到这个节点。`
- 整个请求过程中，`Hystrix 组件负责处理服务超时熔断，Turbine 组件负责监控服务间的调用和熔断相关指标，Sleuth 组件负责调用链监控，ELK 负责日志分析。`

##### 5. gRPC

先来看下 gRPC，它的原理是`通过 IDL（Interface Definition Language）文件定义服务接口的参数和返回值类型，然后通过代码生成程序生成服务端和客户端的具体实现代码`，这样在 gRPC 里，客户端应用可以像调用本地对象一样调用另一台服务器上对应的方法。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/16caf43c8e5b851e)

它的主要特性包括三个方面。

- 通信协议采用了` HTTP/2`，因为 HTTP/2 提供了`连接复用、双向流、服务器推送、请求优先级、首部压缩等机制`
- IDL 使用了ProtoBuf，ProtoBuf 是由 Google 开发的一种数据序列化协议，它的`压缩和传输效率极高，语法也简单`
- `多语言支持`，能够基于多种语言自动生成对应语言的客户端和服务端的代码。

##### 6. Thrift

再来看下 Thrift，`Thrift 是一种轻量级的跨语言 RPC 通信方案`，支持多达 25 种编程语言。为了支持多种语言，跟 gRPC 一样，Thrift 也有一套自己的接口定义语言 IDL，`可以通过代码生成器，生成各种编程语言的 Client 端和 Server 端的 SDK 代码`，这样就保证了不同语言之间可以相互通信。它的架构图可以用下图来描述。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/16caf43c8d711952)

从这张图上可以看出 Thrift RPC 框架的特性。

- `支持多种序列化格式`：如 Binary、Compact、JSON、Multiplexed 等。
- `支持多种通信方式`：如 Socket、Framed、File、Memory、zlib 等。
- `服务端支持多种处理方式`：如 Simple 、Thread Pool、Non-Blocking 等。

#### RPC  核心功能

##### .1. 服务寻址

> - 如果是本地调用，被调用的方法在同一个进程内，操作系统或虚拟机可以地址空间找到;
> - 但是在远程调用中，这是行不通的，因为两个进程的地址空间是完全不一样的，并且也无法知道远端的进程在何处。

要想实现远程调用，我们需要对服务消费者和服务提供者进行约束：

- 在远程过程调用中`所有的函数都必须有一个ID`，这个 ID 在整套系统中是唯一确定的。
- 服务消费者在做远程过程调用时，`发送的消息体中必须携带这个 ID`。
- 服务消费者和服务提供者`分别维护一个函数和 ID 的对应表`。

当服务消费者需要进行远程调用时，它就查一下这个表，找出对应的 ID，然后把它传给服务端，服务端也通过查表，来确定客户端需要调用的函数，然后执行相应函数的代码。要调用服务，首先你需要一个服务注册中心去查询对方服务都有哪些实例，然后根据负载均衡策略择优选一。

![](../../../../../blogimgv2022/f52de3a5c920b0baa36845f8719f810e.jpg)

##### .2. **数据编解码(序列化和反序列化)**

##### .3. 网络传输

- 数据编解码和网络传输可以有多种组合方式，比如常见的有：HTTP+JSON, Dubbo 协议+TCP 等。

#### 4. 技术需求分析

##### .1. 服务定义

> RPC是一个两方（消费方和提供方）的通信调用关系，这需要在两方之间建立一个通信契约，服务定义就是实现这个目的。`服务定义的内容主要包括服务名、入参请求和出参响应`。

```go
//gRpc/Thrift是以一个规范的文件格式定义服务，
syntax = “proto3”;
package helloworld;

// The greeting service definition.
Service Greeter {
  rpc SayHello (HelloRequest) returns (HelloReply) {}
}

message HelloRequest {
  string name = 1;
}

message HelloReply {
  string message = 1;
}
```

```java
//而Dubbo和Motan则使用Java Interface的形式。
Public interface Greeter {
    String sayHello(String name);
}
```

##### .2. 服务开发

> 大多数RPC框架都对服务的开发进行了非常友好的支持，例如`gRpc/Thrift提供代码的自动生成，开发者只需要继承生成的服务基类，添加相应的逻辑代码即可`，`Dubbo和Motan由于直接使用Java Interface`，开发也是非常容易。

```java
public class GreeterImpl implements Greeter {
    public String sayHello(String name) {
        return “Hello “ + name;
    }
}
```

##### .3.服务调用

`服务调用`比较多样化，主要考虑包括如下几个方面，

- `同步和异步`调用的支持。
- `对容错、负载均衡的支持`，对服务的`多版本和分组支持`，以保证高可用的目的。
- 对`服务熔断和服务降级的支持`，以保证服务故障隔离的目的。

容错策略解决的是，一旦发现远程调用失败，如果进行下一步操作以减少错误影响，常见的容错策略有，

- `FailFast 快速失败`：当消费者调用远程服务失败时，`立即报错，消费者只发起一次调用请求。`
- `FailOver 失败自动切换`：当消费者调用远程服务失败时，`重新尝试调用服务，重试的次数一般需要指定，防止无限次重试。`
- `FailSafe 失败安全`：当消费者调用远程服务失败时，`直接忽略，请求正常返回报成功`。一般用于可有可无的服务调用。
- `FailBack 失败自动恢复`：当消费者调用远程服务失败时，`定时重发请求。一般用于消息通知`。
- `Forking 并行调用`：消费者`同时调用多个远程服务，任一成功响应则返回`。

负载均衡机制解决的是如何在多个可用远程服务提供者中，选择下一个进行调用，常见的负载均衡策略有，

- `Random 随机选择`：在可用的远程服务提供者中，`随机选择一个进行调用。`
- `RoundRobin 轮询选择`：在可用的远程服务提供者中，依次轮询选择一个进行调用。
- `LeastActive 按最少活跃调用数选择`：在可用的远程服务提供者中，选择最少调用过的远程进行调用。
- `ConsistentHash 按一致哈希选择`：获取调用请求的哈希值，根据哈希值选择远程服务提供者，保证相同参数的请求总是发到同一提供者。

服务熔断是指一旦服务调用无法正常工作，在故障时间内，尽量减少对服务的调用，直接返回null或抛异常，直到服务恢复正常。各个RPC框架对服务调用的支持程度不一，但基本的高可用和熔断措施都会提供，而且一般会有相应的扩展点供自定义。在高并发量的情况下，对高可用和熔断措施会是相当大的考验，能否正常稳定运行，是一个RPC框架走向生产环境的关键，一般会和远程服务通信模块一并进行性能测试，详细的性能测试参数见后续的远程服务通信。

- **Motan以注解的形式提供服务**

```java
@MotanService(export = “demoMotan:8002”)
public class GreeterImpl implements Greeter {
    public String sayHello (String name) {
        return “Hello “ + name;
    }
}
```

- **Motan以注解的形式消费服务**

```java
public class HelloController {
    @MotanReferer(basicReferer = “clientConfig”, directUrl = “127.0.0.1:8002”)
    Greeter greeter;
    public String home() {
        return greeter.sayHello(“test”);
    }
}
```

##### .4. 服务通信

服务通信组件是底层，其`通信吞吐量和性能`，决定了整个微服务框架的高并发，其通信一般通过性能测试报告获得。一般的测试场景包括，

`数据传输性能`测试场景（考察最大通信吞吐量TPS，响应时间，10并发）

- 传入1k string，原样返回
- 传入50k string，原样返回
- 传入200k string，原样返回
- 传入1k POJO，原样返回，考察序列化模块
- 传入的string size在1k-200k随机变化，原样返回

`高并发性能测试`场景（考察最大通信`吞吐量TPS，响应时间，负载均衡`）

- 10并发1k string，原样返回
- 20并发 1k string，原样返回

`稳定性测试`场景（考察框架的`稳定性`, 通信`吞吐量TPS、成功率、响应时间的趋势`）

- 时长：24小时，7天，一个月
- 吞吐量：10并发 1k string，原样返回

`服务消费方性能`测试（考察消费方的稳定性和高可用）

- `熔断`：每隔1分钟触发一次熔断，1分钟后恢复，一次轮询24小时
- `负载均衡`：对每个负载均衡策略，持续运行24小时

##### .5. 服务监控运行

#### 5. 应用场景

- **分布式操作系统的进程间通讯**：进程间通讯是操作系统必须提供的基本设施之一,分布式操作系统必须提供分布于异构的结点机上进程间的通讯机制，`RPC是实现消息传送模式的分布式进程间通讯方式之一`。

- **构造分布式设计的软件环境**：由于分布式软件设计，服务与环境的分布性, 它的各个组成成份之间存在大量的交互和通讯, RPC是其基本的实现方法之一。Dubbo分布式服务框架基于RPC实现，Hadoop也采用了RPC方式实现客户端与服务端的交互。

- **远程数据库服务**：在分布式数据库系统中，数据库一般驻存在服务器上，客户机通过远程数据库服务功能访问数据库服务器，现有的远程数据库服务是使用RPC模式的。例如，Sybase和Oracle都提供了存储过程机制，系统与用户定义的存储过程存储在数据库服务器上，用户在客户端使用RPC模式调用存储过程。

- **分布式应用程序设计**：RPC机制与RPC工具为分布式应用程序设计提供了手段和方便, 用户可以`无需知道网络结构和协议细节`而直接使用RPC工具设计分布式应用程序。

- **分布式程序的调试**：RPC可用于分布式程序的调试。使用反向RPC使服务器成为客户并向它的客户进程发出RPC，可以调试分布式程序。例如，在服务器上运行一个远端调试程序，它不断接收客户端的RPC，当遇到一个调试程序断点时，它向客户机发回一个RPC，通知断点已经到达，这也是RPC用于进程通讯的例子。

#### 6. 存在问题

- 各个RPC框架`都对某一个开发语言有特定的支持`。
- 目前通过RPC框架开发微服务时，带来最大的问题便是`其框架侵入性，`开发者在项目的开发过程中，不仅引入框架组件，还不得不`在业务代码中添加各种“胶水”配置代码`。

####  Resource

- https://toutiao.io/posts/06d6hdl/preview
- https://www.jianshu.com/p/b0343bfd216e
- https://juejin.cn/post/6844903920402333709
- Dubbo的技术文档 https://www.gitbook.com/@dubbo
- Motan的技术文档 https://github.com/weibocom/motan/wiki
- https://gitee.com/huangyong/rpc



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/rpc_ecology/  

