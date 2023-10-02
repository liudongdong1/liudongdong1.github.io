# servicemesh


- From: https://zhuanlan.zhihu.com/p/61901608

> 微服务 (Microservices) 是一种软件架构风格，它是以`专注于单一责任与功能的小型功能区块` (Small Building Blocks) 为基础，利用模块化的方式组合出复杂的大型应用程序，各功能区块使用与***语言无关\*** (Language-Independent/Language agnostic) 的 API 集相互通信。

> 服务网格是一个***基础设施层\***，用于处理服务间通信。云原生应用有着复杂的服务拓扑，服务网格保证***请求在这些拓扑中可靠地穿梭\***。在实际应用当中，服务网格通常是由一系列轻量级的***网络代理\***组成的，它们与应用程序部署在一起，但***对应用程序透明\***。
>
> - `屏蔽分布式系统通信的复杂性`(负载均衡、服务发现、认证授权、监控追踪、流量控制等等)，服务只用关注业务逻辑；
> - 真正的`语言无关`，服务可以用任何语言编写，只需和Service Mesh通信即可；
> - 对`应用透明`，Service Mesh组件可以单独升级；
> - Service Mesh组件`以代理模式计算并转发请求`，`一定程度上会降低通信系统性能，并增加系统资源开销`；
> - Service Mesh组件`接管了网络流量`，因此服务的`整体稳定性依赖于Service Mesh`，同时额外引入的大量Service Mesh服务实例的运维和管理也是一个挑战；

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726090312349.png)

### 1. 迭代过程

#### .1. 原始通信时代

> 在TCP协议出现之前，服务需要自己处理网络通信所面临的丢包、乱序、重试等一系列流控问题，因此服务实现中，除了业务逻辑外，还夹杂着对网络传输问题的处理逻辑。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725174114554.png)

#### .2. TCP 时代

> 解决了网络传输中通用的流量控制问题，将技术栈下移，从服务的实现中抽离出来，成为操作系统网络层的一部分。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725174235475.png)

#### .3. 第一代微服务

> 以GFS/BigTable/MapReduce为代表的分布式系统得以蓬勃发展。这时，分布式系统特有的通信语义又出现了，如熔断策略、负载均衡、服务发现、认证和授权、quota限制、trace和监控等等，于是服务根据业务需求来实现一部分所需的通信语义。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725174318708.png)

#### .4. 第二代微服务

> 为了`避免每个服务都需要自己实现一套分布式系统通信的语义功能`，随着技术的发展，一些面向`微服务架构的开发框架`出现了，如Twitter的[Finagle](https://link.zhihu.com/?target=https%3A//finagle.github.io/)、Facebook的[Proxygen](https://link.zhihu.com/?target=https%3A//code.facebook.com/posts/1503205539947302)以及Spring Cloud等等，这些框架`实现了分布式系统通信需要的各种通用语义功能`：如负载均衡和服务发现等，因此一定程度上屏蔽了这些通信细节，使得开发人员使用较少的框架代码就能开发出健壮的分布式系统。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210725174444432.png)

#### .5. 第一代 Service Mesh

- 其一，虽然框架本身屏蔽了分布式系统通信的一些通用功能实现细节，但`开发者却要花更多精力去掌握和管理复杂的框架本身`，在实际应用中，去追踪和解决框架出现的问题也绝非易事；
- 其二，`开发框架通常只支持一种或几种特定的语言`，回过头来看文章最开始对微服务的定义，一个重要的特性就是语言无关，但那些没有框架支持的语言编写的服务，很难融入面向微服务的架构体系，想因地制宜的用多种语言实现架构体系中的不同模块也很难做到；
- 其三，框架`以lib库的形式和服务联编`，复杂项目`依赖时的库版本兼容问题非常棘手`，同时，`框架库的升级也无法对服务透明`，服务会因为和业务无关的lib库升级而被迫升级；

> 以`Linkerd，Envoy，NginxMesh`为代表的`代理模式（边车模式）`应运而生，这就是第一代Service Mesh，它将`分布式服务的通信抽象为单独一层`，在这一层中实现`负载均衡、服务发现、认证授权、监控追踪、流量控制等分布式系统所需要的功能`，作为一个和服务对等的代理服务，和服务部署在一起，接管服务的流量，通过代理之间的通信间接完成服务之间的通信请求，这样上边所说的三个问题也迎刃而解。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-e5660d35a311467c3323f10ebf2fb9a5_720w.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-8a9cc161a34d97f36ead06d0abc5b1fb_720w.jpg)

#### 6. 第二代 Service Mesh

> 第一代Service Mesh由一系列独立运行的单机代理服务构成，为了`提供统一的上层运维入口`，`演化出了集中式的控制面板`，所有的单机代理组件通过和控制面板交互进行网络拓扑策略的更新和单机数据的汇报。这就是以`Istio`为代表的第二代Service Mesh。还包括： Conduit/Linkerd 2.X; NGINXMESH;

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-546ed82e25d83a2cb404b0a3f526f9c6_720w.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-8686840abd3de29e5cb6e8dcfa78182f_720w.jpg)

### 2. 关键术语

- 服务网格（Service mesh）：服务间网络流量的逻辑边界。这个概念比较好理解，就是为使用 App mesh 的服务圈一个虚拟的边界。
- 虚拟服务（Virtual services）：是真实服务的抽象。真实服务可以是部署于抽象节点的服务，也可以是间接的通过路由指向的服务。
- 虚拟节点（Virtual nodes）：虚拟节点是指向特殊工作组（task group）的逻辑指针。例如 AWS 的 ECS 服务，或者 Kubernetes 的 Deployment。可以简单的把它理解为是物理节点或逻辑节点的抽象。
- Envoy：AWS 改造后的 Envoy（未来会合并到 Envoy 的官方版本），作为 App Mesh 里的数据平面，Sidecar 代理。
- 虚拟路由器（Virtual routers）：用来处理来自虚拟服务的流量。可以理解为它是一组路由规则的封装。
- 路由（Routes）：就是路由规则，用来根据这个规则分发请求。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/a32d3f907e4d3e2803089e3c53790875.png)

### 3. Google Cloud Platform

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726090822826.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726090842154.png)

### Resource

- https://zhuanlan.zhihu.com/p/61901608

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/servicemesh/  

