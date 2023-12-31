# 微服务5种设计


> - 整个应用程序被拆分成相互独立但包含多个内部模块的子进程。
> - 与模块化的单体应用（Modular Monoliths）或 SOA 相反，微服务`应用程序根据业务范围或领域垂直拆分`。
> - 微服务边界是外部的，`微服务之间通过网络调用（RPC 或消息）相互通信`。
> - 微服务是独立的进程，它们可以独立部署。
> - 它们以轻量级的方式进行通信，不需要任何智能通信通道。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815210145221.png)

### 1. 独享数据库（Database per Microservice）

> 为每个微服务提供自己的数据存储，这样服务之间在数据库层就不存在强耦合。这里我使用数据库这一术语来表示逻辑上的数据隔离，也就是说微服务可以共享物理数据库，但应`该使用分开的数据结构、集合或者表`，这还将有助于确保微服务是按照[领域驱动设计](https://en.wikipedia.org/wiki/Domain-driven_design)的方法正确拆分的。
>
> 缺点：
>
> - 服务间的数据共享变得更有挑战性。
> - 在应用范围的保证 ACID 事务变得困难许多。
> - 细心设计如何拆分单体数据库是一项极具挑战的任务。

![image-20210815210342898](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815210342898.png)

### 2. 事件源（Event Sourcing）

> 可以基于事件的架构使用事件源模式。在传统数据库中，直接存储的是业务实体的当前“状态”，而`在事件源中任何“状态”更新事件或其他重要事件都会被存储起来`，而不是直接存储实体本身。这意味着`业务实体的所有更改将被保存为一系列不可变的事件`。因为数据是作为一系列事件存储的，而非直接更新存储，所以各项服务可以通过重放事件存储中的事件来计算出所需的数据状态。
>
> **缺点**
>
> - `从事件存储中读取实体成为新的挑战，通常需要额外的数据存储（CQRS 模式）`。
> - 系统整体复杂性增加了，通常需要[领域驱动设计](https://en.wikipedia.org/wiki/Domain-driven_design)。
> - 系统需要处理事件重复（幂等）或丢失。
> - 变更事件结构成为新的挑战。
>
> **何时使用事件源**
>
> - 使用`关系数据库的、高可伸缩的事务型系统`。
> - 使用` NoSQL 数据库的事务型系统`。
> - `弹性高可伸缩微服务架构`。
> - 典型的`消息驱动或事件驱动系统`（电子商务、预订和预约系统）。
>
> **何时不宜使用事件源**
>
> - 使用 SQL 数据库的低可伸缩性事务型系统
> - 在服务可以同步交换数据（例如，通过 API）的简单微服务架构中。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815211051199.png)

> 事件存储： [EventStoreDB](https://www.eventstore.com/)， [Apache Kafka](https://kafka.apache.org/)， [Confluent Cloud](https://www.confluent.io/confluent-cloud)， [AWS Kinesis](https://aws.amazon.com/kinesis/)， [Azure Event Hub](https://azure.microsoft.com/en-us/services/event-hubs/)， [GCP Pub/Sub](https://cloud.google.com/pubsub)， [Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)， [MongoDB](https://www.mongodb.com/)， [Cassandra](https://cassandra.apache.org/). [Amazon DynamoDB](https://aws.amazon.com/dynamodb/?trk=ps_a134p000004f2XeAAI&trkCampaign=acq_paid_search_brand&sc_channel=PS&sc_campaign=acquisition_EMEA&sc_publisher=Google&sc_category=Database&sc_country=EMEA&sc_geo=EMEA&sc_outcome=acq&sc_detail=amazon dynamodb&sc_content=DynamoDB_e&sc_matchtype=e&sc_segment=468764879940&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Database|DynamoDB|EMEA|EN|Text|xx|EU&s_kwcid=AL!4422!3!468764879940!e!!g!!amazon dynamodb&ef_id=CjwKCAiAq8f-BRBtEiwAGr3DgRRqVmhD5PL323QFmdBJvvOwzxU1nvrGFdbM8ra-DQViD8jjGn-PGBoCWJYQAvD_BwE:G:s&s_kwcid=AL!4422!3!468764879940!e!!g!!amazon dynamodb)
>
> 框架： [Lagom](https://www.lagomframework.com/)， [Akka](https://akka.io/)， [Spring](https://spring.io/)， [akkatecture](https://akkatecture.net/)， [Axon](https://axoniq.io/)，[Eventuate](https://eventuate.io/)

### 3. 命令和查询职责分离（CQRS）

> 在该模式中，系统的数据修改部分（命令）与数据读取部分（查询）是分离的。而 CQRS 模式有两种容易令人混淆的模式，分别是简单的和高级的。
>
> **缺点**
>
> - `读数据存储是弱一致性的`（最终一致性）。
> - 整个系统的复杂性增加了，混乱的 CQRS 会显着危害整个项目。
>
> **何时使用 CQRS**
>
> - 在`高可扩展`的微服务架构中使用事件源。
> - 在复杂领域模型中，`读操作需要同时查询多个数据存储`。
> - 在`读写操作负载差异明显的系统中`。

> 写存储： [EventStoreDB](https://www.eventstore.com/)， [Apache Kafka](https://kafka.apache.org/)， [Confluent Cloud](https://www.confluent.io/confluent-cloud)， [AWS Kinesis](https://aws.amazon.com/kinesis/)， [Azure Event Hub](https://azure.microsoft.com/en-us/services/event-hubs/)， [GCP Pub/Sub](https://cloud.google.com/pubsub)， [Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)， [MongoDB](https://www.mongodb.com/)， [Cassandra](https://cassandra.apache.org/). [Amazon DynamoDB](https://aws.amazon.com/dynamodb/)
>
> 读存储：[Elastic Search](https://www.elastic.co/)， [Solr](https://lucene.apache.org/solr/features.html)， [Cloud Spanner](https://cloud.google.com/spanner)， [Amazon Aurora](https://aws.amazon.com/rds/aurora/)， [Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)， [Neo4j](https://neo4j.com/)
>
> 框架：[Lagom](https://www.lagomframework.com/)， [Akka](https://akka.io/)， [Spring](https://spring.io/)， [akkatecture](https://akkatecture.net/)， [Axon](https://axoniq.io/)， [Eventuate](https://eventuate.io/)

- **简单模式**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815211341109.png)

- **复杂模式**

> 对于`读频繁的应用程序或微服务架构`，OLTP 数据库（任何提供 ACID 事务保证的关系或非关系数据库）或分布式消息系统都可以被用作写存储。对于`写频繁`的应用程序（写操作高可伸缩性和大吞吐量），需要使用`写可水平伸缩的数据库`（如全球托管的公共云数据库）。标准化的数据则保存在写数据存储中。
>
> 对搜索（例如 Apache Solr、Elasticsearch）或读操作（KV 数据库、文档数据库）进行优化的非关系数据库常被用作读存储。许多情况会在需要 SQL 查询的地方使用读可伸缩的关系数据库。非标准化和特殊优化过的数据则保存在读存储中。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815211435652.png)

### 4. Saga

> Saga 是 1987 年开发的一种古老模式，是关系数据库中关于大事务的一个替代概念。但这种模式的一种现代变种对分布式事务也非常有效。Saga 模式是一个`本地事务序列`，`其每个事务在一个单独的微服务内更新数据存储并发布一个事件或消息`。Saga 中的`首个事务是由外部请求（事件或动作）初始化的，一旦本地事务完成（数据已保存在数据存储且消息或事件已发布），那么发布的消息或事件则会触发 Saga 中的下一个本地事务`。[Axon](https://axoniq.io/)， [Eventuate](https://eventuate.io/)， [Narayana](https://narayana.io/)
>
> Saga 事务协调管理主要有两种形式：
>
> - *事件编排 Choreography*：分散协调，`每个微服务生产并监听其他微服务的事件或消息然后`决定是否执行某个动作。
> - *命令编排 Orchestration*：集中协调，`由一个协调器告诉参与的微服务哪个本地事务需要执行`。
>
> **优点**
>
> - 为高可伸缩或松耦合的、事件驱动的微服务架构提供一致性事务。
> - 为使用了不支持 2PC 的非关系数据库的微服务架构提供一致性事务。
>
> **缺点**
>
> - 需要处理瞬时故障，并且提供等幂性。
> - 难以调试，而且复杂性随着微服务数量增加而增加。
>
> **何时使用 Saga**
>
> - 在使用了事件源的高可伸缩、松耦合的微服务中。
> - 在使用了分布式非关系数据库的系统中。
>
> **何时不宜使用 Saga**
>
> - 使用关系数据库的低可伸缩性事务型系统。
> - 在服务间存在循环依赖的系统中。

### 5. 面向前端的后端 （BFF）

> 面向前端的后端模式适用于需要为特殊 UI 定制单独后端的场景。它还提供了其他优势，比如`作为下游微服务的封装`，从而减少 UI 和下游微服务之间的频繁通信。此外，在高安全要求的场景中，BFF 为部署在 DMZ 网络中的下游微服务提供了更高的安全性。
>
> **优点**
>
> - 分离 BFF 之间的关注点，使得我们可以为具体的 UI 优化他们。
> - 提供更高的安全性。
> - 减少 UI 和下游微服务之间频繁的通信。
>
> **缺点**
>
> - BFF 之间代码重复。
> - 大量的 BFF 用于其他用户界面（例如，智能电视，Web，移动端，PC 桌面版）。
> - 需要仔细的设计和实现，BFF 不应该包含任何业务逻辑，而应只包含特定客户端逻辑和行为。
>
> **何时使用 BFF**
>
> - 如果`应用程序有多个含不同 API 需求的 UI`。
> - 出于`安全需要，UI 和下游微服务之间需要额外的层`。
> -  如果`在 UI 开发中使用微前端`。
>
> **何时不宜使用 BFF**
>
> - 如果应用程序虽有多个 UI，但使用的 API 相同。
> - 如果核心微服务不是部署在 DMZ 网络中。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815211810670.png)

### 6. API 网关

> API 网关位于`客户端 APP 和后端微服务之间充当 facade`，它可以是反向代理，将客户端请求路由到适当的后端微服务。它还支持将客户端请求扇出到多个微服务，然后将响应聚合后返回给客户端。它还支持必要的横切关注点。
>
> **优点**
>
> - 在前端和后端服务之间提供松耦合。
> - 减少客户端和微服务之间的调用次数。
> - 通过 SSL 终端、身份验证和授权实现高安全性。
> - 集中管理的横切关注点，例如，日志记录和监视、节流、负载平衡。
>
> **缺点**
>
> - 可能导致微服务架构中的`单点故障`。
> - `额外的网络调用带来的延迟增加。`
> - 如果不进行扩展，它们`很容易成为整个企业应用的瓶颈`。
> - 额外的维护和开发费用。
>
> **何时使用 API 网关**
>
> - 在复杂的微服务架构中，它几乎是必须的。
> - 在大型企业中，API 网关是中心化安全性和横切关注点的必要工具。
>
> **何时不宜使用 API 网关**
>
> - 在`安全和集中管理不是最优先要素的私人项目或小公司中`。
> - 如果微服务的数量相当少。
>
> **可用技术示例**
>
> [Amazon API 网关](https://aws.amazon.com/api-gateway/)， [Azure API 管理](https://docs.microsoft.com/en-us/azure/api-management/)， [Apigee](https://cloud.google.com/apigee)， [Kong](https://konghq.com/kong/)， [WSO2 API 管理器](https://wso2.com/api-management/)
>
> [微服务模式： API 网关模式](https://microservices.io/patterns/apigateway.html)
>
> [API 网关-Azure 架构中心](https://docs.microsoft.com/en-us/azure/architecture/microservices/design/gateway)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815212435012.png)

### 7. Strangler

> Strangler 模式意味着通过`使用新的微服务逐步替换特定功能`，将单体应用程序增量地迁移到微服务架构。此外，`新功能只在微服务中添加，而不再添加到遗留的单体应用中。`然后配置一个 Facade （API 网关）来路由遗留单体应用和微服务间的请求。当某个功能从单体应用迁移到微服务，Facade 就会拦截客户端请求并路由到新的微服务。`一旦迁移了所有的功能，遗留单体应用程序就会被“扼杀（Strangler）”，即退役`。
>
> **优点**
>
> - 安全的迁移单体应用程序到微服务。
> - 可以并行地迁移已有功能和开发新功能。
> - 迁移过程可以更好把控节奏。
>
> **缺点**
>
> - 在现有的单体应用服务和新的微服务之间共享数据存储变得具有挑战性。
> - 添加 Facade （API 网关）将增加系统延迟。
> - 端到端测试变得困难。
>
> **何时使用 Strangler**
>
> - 将大型后端单体应用程序的增量迁移到微服务。
>
> **何时不宜使用 Strangler**
>
> - 如果后端单体应用很小，那么全量替换会更好。
> - 如果无法拦截客户端对遗留的单体应用程序的请求。
>
> [bliki： StranglerFig 应用程序](https://martinfowler.com/bliki/StranglerFigApplication.html)
>
> [Strangler 模式 - 云设计模式](https://docs.microsoft.com/en-us/azure/architecture/patterns/strangler)
>
> [微服务模式：Strangler 应用程序](https://microservices.io/patterns/refactoring/strangler-application.html)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815212815745.png)

### 8. 断路器

> 微服务通过同步调用其他服务来满足业务需求。`服务调用会由于瞬时故障（网络连接缓慢、超时或暂时不可用） 导致失败`，这种情况重试可以解决问题。然而，如果出现了严重问题（微服务完全失败），那么微服务将长时间不可用，这时重试没有意义且浪费宝贵的资源（线程被阻塞，CPU 周期被浪费）。此外，一个服务的故障还会引发整个应用系统的级联故障。这时快速失败是一种更好的方法。
>
> 在这种情况，可以使用断路器模式挽救。`一个微服务通过代理请求另一个微服务`，其工作原理类似于**电气断路器**，代理通过统计最近发生的故障数量，并使用它来决定是继续请求还是简单的直接返回异常。
>
> - *关闭：*`断路器将请求路由到微服务`，`并统计给定时段内的故障数量，如果超过阈值，它就会触发并进入打开状态。`
> - *打开*：来自微服务的`请求会快速失败并返回异常`。在超时后，断路器进入半开启状态。
> - *半开*：`只有有限数量的微服务请求被允许通过并进行调用`。如果这些请求成功，断路器将进入闭合状态。如果任何请求失败，断路器则会进入开启状态。
>
> **缺点**
>
> - 需要复杂的异常处理。
> - 日志和监控。
> - 应该支持人工复位。
>
> **何时使用断路器**
>
> - 在微服务间使用同步通信的紧耦合的微服务架构中。
> - 如果微服务依赖多个其他微服务。
>
> API 网关，服务网格，各种断路器库（[Hystrix](https://github.com/Netflix/Hystrix/wiki/How-it-Works)， [Reselience4J](https://github.com/resilience4j/resilience4j)， [Polly](http://www.thepollyproject.org/)）。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210815214709327.png)

### 9. 外部化配置

> `每个业务应用都有许多用于各种基础设施的配置参数`（例如，数据库、网络、连接的服务地址、凭据、证书路径）。此外`在企业应用程序通常部署在各种运行环境`（Local、 Dev、 Prod）中，实现这些的一个方法是通过内部配置。这是一个致命糟糕实践，它会导致严重的安全风险，因为生产凭证很容易遭到破坏。此外，`配置参数的任何更改都需要重新构建应用程序`，这在在微服务架构中会更加严峻，因为我们可能拥有数百个服务。`将所有配置外部化，使得构建过程与运行环境分离，生产的配置文件只在运行时或通过环境变量使用`，从而最小化了安全风险。
>
> [**微服务模式：外部化配置**](https://microservices.io/patterns/externalized-configuration.html)
>
> [**一次构建，到处运行：外部化你的配置**](https://reflectoring.io/externalize-configuration/)

### 10. 消费端驱动的契约测试

> 在微服务架构中，通常有许多有不同团队开发的微服务。这些微型服务协同工作来满足业务需求（例如，客户请求），并相互进行同步或异步通信。**消费端**微服务的集成测试具有挑战性，通常用 **TestDouble** 以获得更快、更低成本的测试运行。但是 TestDouble 通常并不能代表真正的微服务提供者，而且如果微服务提供者更改了它的 API 或 消息，那么 TestDouble 将无法确认这些。另一种选择是进行端到端测试，尽管它在生产之前是强制性的，但却是脆弱的、缓慢的、昂贵的且不能替代集成测试（[Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)）。
>
> 在这方面消费端驱动的契约测试可以帮助我们。在这里，负责消费端微服务的团队针对特定的服务端微服务，编写一套包含了其请求和预期响应（同步）或消息（异步）的测试套件，这些测试套件称为显式的约定。对于微服务服务端，将其消费端所有约定的测试套件都添加到其自动化测试中。当特定服务端微服务的自动化测试执行时，它将一起运行自己的测试和约定的测试并进行验证。通过这种方式，契约测试可以自动的帮助维护微服务通信的完整性。[Pact](https://docs.pact.io/)， [Postman](https://www.postman.com/)， [Spring Cloud Contract](https://spring.io/guides/gs/contract-rest/)

### Resource

- [软件分层设计思考](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247522953&idx=3&sn=d1bb2998b201285d0d63d7573e1babe8&chksm=fa4ae538cd3d6c2e70e78f4163b42e02161da03bbe32539f038ee088b24a0b8e0939cde652e0&mpshare=1&scene=24&srcid=0806pe5wGlLJxcxtFMcKSt6V&sharer_sharetime=1628218961145&sharer_shareid=ed830dd7498411a03cfe8154944facae#rd)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%BE%AE%E6%9C%8D%E5%8A%A15%E7%A7%8D%E8%AE%BE%E8%AE%A1/  

