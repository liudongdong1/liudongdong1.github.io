# DistributionAlgorithm


From: https://juejin.cn/post/6874218041136300040

## 1. 分布式理论：一致性算法 Paxos

> 在常见的分布式系统中，总会发生诸如`机器宕机或者网络异常`等情况。Paxos算法需要解决的问题就是`如何在一个可能发生上述异常的分布式系统中`：快速且正确的在集群内部对某个数据的值达成一致，并且保证不论发生以上任何异常，都不会破坏整个系统的一致性
>
> 分布式系统才用多副本进行存储数据 , 如果对多个副本执行序列不控制, 那多个副本执行更新操作,由于网络延迟 超时 等故障到值各个副本的数据不一致. 我们希望每个副本的执行序列是`[ op1 op2 op3 .... opn ]`不变的, 相同的. `Paxos 一次来确定不可变变量 opi的取值 , 每次确定完Opi之后,各个副本执行opi操作,一次类推。`

在一个集群环境中，要求所有机器上的状态是一致的，其中有2台机器想修改某个状态，`机器A 想把状态改为 A，机器 B 想把状态改为 B`，那么到底听谁的呢？

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/6fdaf662c0a947b8b6a5ab3aa4a16cc9~tplv-k3u1fbpfcp-zoom-1.image)

那么要是协调者蹦了呢？ 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/9586cf6a00864f99a1cffab42f3585d3~tplv-k3u1fbpfcp-zoom-1.image)

所以需要对协调者也做备份，也要做集群。这时候，问题来了，这么`多协调者，听谁的`呢？ 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/e9eae1dc32fb4827a98a5051d7a7d5ef~tplv-k3u1fbpfcp-zoom-1.image)

#### 基本概念-提案（Proposal）

最终要达成一致的value就`在提案里Proposal信息包括提案编号 (Proposal ID) 和提议的值 (Value)`

#### 基本概念-4角色

- `Client`：客户端
  - 客户端向分布式系统发出请求，并等待响应。例如，对分布式文件服务器中文件的写请求。
- `Proposer`：提案发起者
  - 提案者提倡客户请求，试图说服Acceptor对此达成一致，并在发生冲突时充当协调者以推动协议向前发展
- `Acceptor`：决策者，可以批准提案
  - Acceptor可以接受（accept）提案；如果某个提案被选定（chosen），那么该提案里的value就被选定了
- `Learners`：最终决策的学习者
  - 学习者充当该协议的复制因素

这里需要说明的是，Proposer，Acceptor，Learners 会存在多份实例，一个进程可能充当不只一种角色

他们之间协作的流程是： `Proposer提出提案，Accepter接收建议，然后Accepter之间 选定出一个最终提案Proposal `![](https://gitee.com/github-25970295/blogimgv2022/raw/master/00b22161571541db81502f899cb2df9d~tplv-k3u1fbpfcp-zoom-1.image)

### 问题描述

> 假设有一组可以提出提案的进程集合，那么对于一个一致性算法需要保证以下几点：
>
> - 在这些被提出的提案中，`只有一个会被选定`
> - 如果没有提案被提出，就不应该有被选定的提案。
> - 当`一个提案被选定后，那么所有进程都应该能学习（learn）到这个被选定的value`

### 推导过程

#### 最简单的方案——只有一个Acceptor

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/4ba0d722b8a6412aa98dd7f9ede4fdf6~tplv-k3u1fbpfcp-zoom-1.image) 

#### 多个Proposer和多个Acceptor

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/4990354c31984466985e1bf957153310~tplv-k3u1fbpfcp-zoom-1.image)

> P1：一个Acceptor必须接受它收到的第一个提案 【An acceptor must accept the first proposal that it receives.】

但是，这又会引出另一个问题：如果每个Proposer分别提出不同的value，发给不同的Acceptor。根据P1， Acceptor分别接受自己收到的第一个提案，就导致不同的value被选定。出现了不一致。如下图： 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/507b58cbaeb141bb8e86856d7670960b~tplv-k3u1fbpfcp-zoom-1.image) 

> 规定：`一个提案被选定需要被半数以上的Acceptor接受`
>
> ​          『一个Acceptor必须能够接受不止一个提案！』不然可能导致最终没有value被选定。

所以在这种情况下，我们使用一个`全局的编号`来标识每一个Acceptor批准的提案，当一个具有某value值的提案被 `半数以上`的Acceptor批准后，我们就认为该value被选定了.

> P2：如果`某个value为v的提案被选定了，那么每个编号更高的被选定提案的value必须也是v`。【If a proposal with value v is chosen, then every higher-numbered proposal that is chosen has value v.】

> P2a：如果`某个value为v的提案被选定了，那么每个编号更高的被Acceptor接受的提案的value必须也是v`【If a proposal with value v is chosen, then every higher-numbered proposal accepted by any acceptor has value v.】

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/2bf6e71f98ff4bb081656a325cd7f87d~tplv-k3u1fbpfcp-zoom-1.image)

但是，考虑如下的情况：假设总的有5个Acceptor。

1. Proposer2提出[M1,V1]的提案，
2. Acceptor2~5（半数以上）均接受了该提案
3. 于是对于Acceptor2~5和Proposer2来讲，它们都认为V1被选定。
4. Acceptor1刚刚从宕机状态恢复过来（之前Acceptor1没有收到过任何提案）
5. 此时Proposer1向Acceptor1发送了[M2,V2]的提案（V2≠V1且M2>M1）
6. 对于Acceptor1来讲，这是它收到的第一个提案。根据P1（一个Acceptor必须接受它收到的第一个提

案。）,Acceptor1必须接受该提案！同时Acceptor1认为V2被选定。这就出现了两个问题：

(1) Acceptor1认为`V2被选定`，Acceptor2~5和Proposer2认为`V1被选定`。出现了不一致。

(2) `V1被选定了，但是编号更高的被Acceptor1接受的提案[M2,V2]的value为V2，且V2≠V1`。这就跟P2a（如果某 个value为v的提案被选定了，那么每个编号更高的被Acceptor接受的提案的value必须也是v）矛盾了。

> P2b：如果`某个value为v的提案被选定了，那么之后任何Proposer提出的编号更高的提案的value必须也是v。`【If a proposal with value v is chosen, then every higher-numbered proposal issued by any proposer has value v.】

> P2c：对于`任意的Mn和Vn,如果提案[Mn,Vn]被提出，那么肯定存在一个由半数以上的Acceptor组成的集合S`，满足以下 两个条件中的任意一个：
>
> - 要么`S中每个Acceptor都没有接受过编号小于Mn的提案`。
> - 要么`S中所有Acceptor批准的所有编号小于Mn的提案中，编号最大的那个提案的value值为Vn`

### Proposer生成提案

> 这里有个比较重要的思想：Proposer生成提案之前，应该先去`『学习』`已经被选定或者可能被选定的value，然后 以该value作为自己提出的提案的value。`如果没有value被选定，Proposer才可以自己决定value的值`。这样才能达成一致。这个学习的阶段是通过一个`『Prepare请求』`实现的。

于是我们得到了如下的提案生成算法：

`Proposer选择一个新的提案编号N`，然后`向某个Acceptor集合（半数以上）发送请求`，要求该集合中的每个

Acceptor做出如下响应（response）

- ` Acceptor向Proposer承诺保证不再接受任何编号小于N的提案。`
-  如果`Acceptor已经接受过提案`，那么`就向Proposer反馈已经接受过的编号小于N的，但为最大编号的提案的值。`

我们将该请求称为编号为N的Prepare请求。如果Proposer收到了`半数以上的Acceptor的响应`，那么它就可以生成编号为N，Value为V的提案[N,V]。这里的`V是所有的响应中编号最大的提案的Value`。如果所有的响应中都没有提案，那 么此时V就可以由Proposer 自己选择。生成提案后，Proposer将该提案发送给半数以上的Acceptor集合，并期望这些Acceptor能接受该提案。我们 称该请求为Accept请求。

### Acceptor接受提案

刚刚讲解了Paxos算法中Proposer的处理逻辑，怎么去生成的提案，下面来看看Acceptor是如何批准提案的

根据刚刚的介绍，`一个Acceptor可能会受到来自Proposer的两种请求，分别是Prepare请求和Accept请求`，对这两 类请求作出响应的条件分别如下

- Prepare请求`：Acceptor可以在任何时候响应一个Prepare请求`
- Accept请求：`在不违背Accept现有承诺的前提下，可以任意响应Accept请求`

> P1a：`一个Acceptor只要尚未响应过任何编号大于N的Prepare请求，那么他就可以接受这个编号为N的提案。`

### 算法优化

分别从Proposer和Acceptor对提案的生成和批准两方面来讲解了Paxos算法在提案选定过程中的算 法细节，同时也在提案的编号全局唯一的前提下，获得了一个提案选定算法，接下来我们再对这个初步算法做一个 小优化，尽可能的忽略Prepare请求

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/819c429750bf46d0b75d84908d76ce78~tplv-k3u1fbpfcp-zoom-1.image)

> 如果`Acceptor收到一个编号为N的Prepare请求`，`在此之前它已经响应过编号大于N的Prepare请求`。根据P1a，`该 Acceptor不可能接受编号为N的提案。因此，该Acceptor可以忽略编号为N的Prepare请求。`

通过这个优化，每个Acceptor只需要记住它`已经批准的提案的最大编号以及它已经做出Prepare请求响应的提案的 最大编号`，以便出现故障或节点重启的情况下，也能保证P2c的不变性，而对于Proposer来说，只要它可以保证不 会产生具有相同编号的提案，那么就可以丢弃任意的提案以及它所有的运行时状态信息

### Paxos算法描述

综合前面的讲解，我们来对Paxos算法的提案选定过程进行下总结，那结合Proposer和Acceptor对提案的处理逻 辑，就可以得到类似于两阶段提交的算法执行过程

Paxos算法分为两个阶段。具体如下： ![](https://gitee.com/github-25970295/blogimgv2022/raw/master/f70ca5f479734db29e0166dcd03f9761~tplv-k3u1fbpfcp-zoom-1.image)

- **阶段一**：

  - Proposer选择一个提案编号`N`，然后向`半数以上`的Acceptor发送编号为`N`的Prepare请求。
  - 如果一个Acceptor收到一个`编号为N`的Prepare请求，且`N大于该Acceptor已经响应过的所有Prepare请求的编号`，那么它就会将它已经接受过的编号最大的提案（如果有的话）作为响应反馈给Proposer，同时`该Acceptor承诺不再接受任何编号小于N的提案。`

- **阶段二**：

  -  如果Proposer收到`半数以上Acceptor对其发出的编号为N的Prepare请求的响应`，那么它就会发送一个针对`[N,V]提案的Accept请求`给半数以上的Acceptor。注意：`V就是收到的响应中编号最大的提案的value`，如果 响应中不包含任何提案，那么V就由Proposer自己决定。

  - 如果Acceptor收到一个针对编号为N的提案的Accept请求，只`要该Acceptor没有对编号大于N的Prepare请求做出过响应，它就接受该提案。`

### Learner学习被选定的value

方案一：Learner获取一个已经被选定的提案的前提是，该提案已经被`半数以上的Acceptor批准`，因此，最简单的 做法就是一旦Acceptor批准了一个提案，就将该提案发送给所有的Learner 很显然，这种做法虽然可以让Learner尽快地获取被选定的提案，但是却需要让每个Acceptor与所有的Learner逐 个进行一次通信，通信的次数至少为二者个数的乘积

方案二：另一种可行的方案是，我们可以让所有的Acceptor将它们对提案的批准情况，统一发送给一个特定的`Learner（称 为主Learner）`, 各个Learner之间可以通过消息通信来互相感知提案的选定情况，基于这样的前提，当主Learner 被通知一个提案已经被选定时，它会负责通知其他的learner 在这种方案中，Acceptor首先会将`得到批准的提案发送给主Learner`,再由其`同步给其他Learner`.因此较方案一而 言，方案二虽然需要多一个步骤才能将提案通知到所有的learner，但其通信次数却大大减少了，通常只是 Acceptor和Learner的个数总和，但同时，该方案引入了一个新的不稳定因素：`主Learner随时可能出现故障`

方案三：在讲解方案二的时候，我们提到，方案二最大的问题在于`主Learner存在单点问题`，即主Learner随时可能出现故 障，因此，对方案二进行改进，可以将`主Learner的范围扩大`，即Acceptor可以将批准的提案发送给一个`特定的 Learner集合`，该集合中每个Learner都可以在一个提案被选定后通知其他的Learner。这个Learner集合中的 Learner个数越多，可靠性就越好，但同时网络通信的复杂度也就越高.

![img](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/31c5626cba0a4d23b53901f24cc05751~tplv-k3u1fbpfcp-zoom-1.image)

### 如何保证Paxos算法的活性

根据前面的内容讲解，我们已经基本上了解了Paxos算法的核心逻辑，那接下来再来看看Paxos算法在实际过程中 的一些细节 活性：最终一定会发生的事情：最终一定要选定value

假设存在这样一种极端情况，`有两个Proposer依次提出了一系列编号递增的提案，导致最终陷入死循环，没有 value被选定`,具体流程如下: ![img](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/de0c03f3df2442bc9befb6ead44425c3~tplv-k3u1fbpfcp-zoom-1.image)

![img](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5e8a0edd91f3427c8b4bdef5952afa94~tplv-k3u1fbpfcp-zoom-1.image) 下面来详细描述下这个场景：

- 提案者1 发出`编号为1`的Prepare请求，收到过半请求，完成阶段一流程  --> 【决策者集群保证`不再接受编号小于1`的提案】
- 提案者2 发出`编号为2`的Prepare请求，收到过半请求，完成阶段一流程  --> 【决策者集群保证`不再接受编号小于2`的提案】
- 提案者1 进入第二阶段的时候【提案为1】，发送的Accept请求被Acceptor忽略
- 提案者1 发出`编号为3`的Prepare请求，收到过半请求，完成阶段一流程  --> 【决策者集群保证`不再接受编号小于3`的提案】
- 提案者2 进入第二阶段的时候【提案为2 】，发送的Accept请求被Acceptor忽略
- ......进入死循环中

解决：通过`选取主Proposer`，并规定只有`主Proposer才能提出议案`。这样一来`只要主Proposer和过半的Acceptor 能够正常进行网络通信，那么但凡主Proposer提出一个编号更高的提案，该提案终将会被批准，这样通过选择一个 主Proposer，整套Paxos算法就能够保持活性`

## 2. 分布式理论：一致性算法 Raft

> Raft 是一种为了`管理复制日志的一致性算法` 。 Raft提供了和Paxos算法相同的功能和性能，但是它的算法结构和Paxos不同。Raft算法更加容易理解并且更容易构建实际的系统。分解成了3模块: 领导人选举，日志复制，安全性；Raft算法分为两个阶段，首先是选举过程，然后在选举出来的领导人带领进行正常操作，比如日志复制等。

### 领导人Leader选举

做法：`Raft 通过选举一个领导人`，然后给予他全部的管理复制日志的责任来实现一致性。 在Raft中，任何时候一个服务器都可以扮演下面的角色之一：

- 领导者(leader)：处理`客户端交互，日志复制等动作`，一般一次只有一个领导者
- 候选者(candidate)：`候选者就是在选举过程中提名自己的实体，一旦选举成功，则成为领导者`
- 跟随者(follower)：类似`选民，完全被动的角色，这样的服务器等待被通知投票`

而影响他们身份变化的则是 选举。

 ![](https://gitee.com/github-25970295/blogimgv2022/raw/master/16b28da0e98f4468975ba2b51271c919~tplv-k3u1fbpfcp-zoom-1.image)

**Raft使用心跳机制来触发选举**

- 当server启动时，初始状态都是follower。
- 每一个s`erver都有一个定时器，超时时间为election timeout（一般为150-300ms）`
  - 如果`在超时时间内,收到来自领导者或者候选者的任何消息，重启定时器`
  - 如果到`达了超时时间，还没有收到其他领导发过来的消息，会认为现在就没有领导，它就开始一次选举，就开始向别的服务器发送消息，让他们投自己一票。`

[thesecretlivesofdata.com/raft/](https://link.juejin.cn/?target=http%3A%2F%2Fthesecretlivesofdata.com%2Fraft%2F) 动画演示

#### 节点异常

集群中各个节点的状态随时都有可能发生变化。从实际的变化上来分类的话，节点的异常大致可以分为四种类型：

- leader 不可用；
- follower 不可用；
- 多个 candidate 或多个 leader；
- 新节点加入集群。

##### leader 不可用

下面将说明当集群中的 leader 节点不可用时，raft 集群是如何应对的。

➢ 一般情况下，`leader 节点定时发送 heartbeat 到 follower 节点。 `

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/bb5d47699bc148119aadf0d555b3d3b5~tplv-k3u1fbpfcp-zoom-1.image)

➢ 由于某些异常导致 leader 不再发送 heartbeat ，或 follower 无法收到 heartbeat 。 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/6fdf6509439641a3ab58091b09415b8e~tplv-k3u1fbpfcp-zoom-1.image)

➢ 当某一 follower 发生 election timeout 时，其状态变更为 candidate，并向其他 follower 发起投票。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/92ebcc0e21054e3987064156685e2eee~tplv-k3u1fbpfcp-zoom-1.image)

➢ 当超过半数的 follower 接受投票后，这一节点将成为新的 leader，leader 的`步进数加 1 `并开始向 follower 同 步日志。 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/f392dc714a834f1aa758c60d63b16ce7~tplv-k3u1fbpfcp-zoom-1.image)

➢ 当一段时间之后，如果之前的 leader 再次加入集群，`则两个 leader 比较彼此的步进数`，`步进数低的 leader 将 切换自己的状态为 follower。` 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/808e0dabb56c43db9ea4d5e2eeb173ff~tplv-k3u1fbpfcp-zoom-1.image)

➢ 较早前 leader 中不一致的日志将被清除，并与现有 leader 中的日志保持一致。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/7e4cf3847fad4a2a8e01f7bc8890e48b~tplv-k3u1fbpfcp-zoom-1.image)

##### follower 节点不可用

> follower 节点不可用的情况相对容易解决。因为集群中的日志内容始终是从 leader 节点同步的，只要这一节点`再次加入集群时重新从 leader 节点处复制日志即可`。

➢ 集群中的某个 follower 节点发生异常，不再同步日志以及接收 heartbeat

 ![](https://gitee.com/github-25970295/blogimgv2022/raw/master/e495806909c645bea55d0448ed959bc1~tplv-k3u1fbpfcp-zoom-1.image)

➢ 经过一段时间之后，原来的 follower 节点重新加入集群。这个时候他很懵逼，究竟发生了什么？我是谁，我在哪里？

 ![](https://gitee.com/github-25970295/blogimgv2022/raw/master/d17550c4847e42da9b14989443f723d4~tplv-k3u1fbpfcp-zoom-1.image)

➢ 这一节点的日志将从当时的 leader 处同步。直接认当前的君主为王就行了，别的也不考虑这么多

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/626a69c3d9b8412ab0ef75128e5031dd~tplv-k3u1fbpfcp-zoom-1.image)

### 日志复制（保证数据一致性）

Leader选出后，就开始接收客户端的请求。`Leader把请求作为日志条目（Log entries）加入到它的日志中， 然后并行的向其他服务器发起 AppendEntries RPC复制日志条目`。当这条日志被复制到大多数服务器上，Leader 将这条日志应用到它的状态机并向客户端返回执行结果。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/c9326506c4694dab8a8dc9dce0d26b33~tplv-k3u1fbpfcp-zoom-1.image)

- 客户端的`每一个请求都包含被复制状态机执行的指令`。
- leader把这个指令作为一条新的日志条目添加到日志中，然后`并行发起 RPC 给其他的服务器`，让他们复制这条信息。

- `跟随者响应ACK`,如果` follower 宕机或者运行缓慢或者丢包，leader会不断的重试，直到所有的 follower 最终都复制了所有的日志条目`。

- `通知所有的Follower提交日志，同时领导人提交这条日志到自己的状态机中，并返回给客户端。`

直到`第四步骤，整个事务才会达成。中间任何一个步骤发生故障，都不会影响日志一致性`

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/distributionalgorithm/  

