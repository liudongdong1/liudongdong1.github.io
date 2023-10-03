# 分布式存储技能


> 即这个知识产生的过程，它解决了什么问题，它是怎么样解决的并且它带来了哪些问题
>
> - 分布式系统解决了什么问题？
>   - 单机性能瓶颈导致的成本问题
>   - 用户量和数据量爆炸性的增大导致的成本问题
>   - 业务高可用的要求
> - 分布式系统是怎么来解决单机系统面临的成本和高可用问题呢？
>   - 分布式系统是由一组通过网络进行通信、为了完成共同的任务而协调工作的计算机节点组成的系统。
> - 分布式带来了那些问题？
>   - 对分布式系统内部工作节点的协调问题？
>     - 怎样找到服务
>     - 曾阳找到发我那个服务的哪一个实例？ 
>       - 如果服务实例完全对等，采用负责均衡（轮询，权重，hash, 一致性hash）
>       - 如果服务不对等，通过路由服务（元数据服务）
>     - 怎样避免雪崩？
>       - 快速失败和降级机制（熔断，降级，限流等）
>       - 弹性扩容机制
>     - 怎样进行监控告警？
>   - 分布式存储怎么进行内部协调？
>     - CAP理论
>     - 怎样做数据分片（jump hash,ring hash, crush）
>     - 怎样做数据复制（中心化方案：主从复制，一致性协议）去中心化方案（Quorum和vector clock）
>     - 怎样做分布式事务？ 事务ID后，通过 2PC 或者 3PC 协议来实现分布式事务的原子性

**一、分布式存储工程师技能栈**
**1、能力基础**
**基础知识：**如[数据结构](https://www.zhihu.com/search?q=数据结构&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2267909637})与算法、网络通信协议、计算机系统结构、[软件工程](https://www.zhihu.com/search?q=软件工程&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2267909637})，设计模式等。

- btree，skiplist，LSM
- Blobfs Bluefs Ext4 Btrfs
- 本地存储引擎：RocksDB LevelDB InnoDB PostgreSQL
- 成员管理：SWIM Gossip Serf Consul Zookeeper Etcd
- 公式协议：Paxos Raft EPaxos Zab Quorum XPaxos
- 分布式文件系统：GFS HDFS Pangu CephFS GlusterFS Lustre JuiceFS PolarFS
- Lamport大师的论文
- AmazonDynamoDB那篇著名的论文

**基本技能：**IDE与编码基础、GIT代码仓日常操作、代码调试工具、性能分析与优化、OS基本操作，常见[中间件](https://www.zhihu.com/search?q=中间件&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2267909637})、开源软件、数据库的使用等。

**2、分布式系统**
**基础知识：**CAP定理、分布式Cache、分布式事务、Paxos与Raft算法、数据分布sharding、负载均衡、消息队列等。
**基本技能：**分布式系统的不确定性处理、任务高效协同、性能分析优化实践等；Nginx、kubernetes、redis、zookeeper、kafka、mysql等常见组件的应用集成、二次开发、部署与维护等。

**3、存储系统**
**基础知识：**各种存储介质，存储接口协议、存储网络协议，存储服务器硬件、硬件冗余结构，软件高可靠、高可用设计，块、文件、对象存储服务，垃圾回收，数据缩减，Qos，磨损均衡、快照、复制、备份，SCM、PIM，NDP等。
**基本技能：**RDMA网络、SPDK/PMDK，熟悉开源Ceph、GFS、Swift、hadoop等使用。

**二、分布式系统可能工作方向**
**从功能上看：**分布式系统按照其任务功能可以分为分布式存储系统（如Ceph）、分布式计算系统（如MapReduce）和[分布式管理系统](https://www.zhihu.com/search?q=分布式管理系统&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2267909637})（如zookeeper）。
**从职业上看：**分布式系统相关的岗位可以是开发、测试、文档交付实施、维护、产品管理、解决方案、市场开拓、产业发展、规划等，可涵盖产品和商业全流程。

**三、参考材料**

**书籍类：**
《设计模式：可复用面向对象软件的基础》
《大规模分布式存储系统：原理解析与架构实战》
《Paxos到Zookeeper：分布式一致性原理与实践》
《[分布式系统：概念与设计](https://www.zhihu.com/search?q=分布式系统：概念与设计&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2267909637})》
《大话存储I&II》
《存储技术原理分析》
《[海量网络存储系统原理与设计](https://www.zhihu.com/search?q=海量网络存储系统原理与设计&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2267909637})》
《信息存储与管理》

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%88%86%E5%B8%83%E5%BC%8F%E5%AD%98%E5%82%A8%E6%8A%80%E8%83%BD/  

