# CloudComputing_Relative


> Wang, F., Zhang, C., Liu, J., Zhu, Y., Pang, H., & Sun, L. (2019, April). Intelligent edge-assisted crowdcast with deep reinforcement learning for personalized QoE. In *IEEE INFOCOM 2019-IEEE Conference on Computer Communications* (pp. 910-918). IEEE.

# Paper:  Edge-Assisted Crowdcast

<div align=center>
<br/>
<b>Intelligent Edge-Assisted Crowdcast with Deep Reinforcement Learning for Personalized QoE</b>
</div>

#### Phenomenon&Challenge:

1.  crowdcast is featured with tremendous video contents at the broadcaster side, highly diverse viewer side content watching environments/preferences as well as viewers’ personalized quality** of experience (QoE) demands (e.g., individual preferences for*streaming delays, channel switching latencies and bitrates). This imposes unprecedented key challenges on how to flexibly and cost-effectively accommodate the heterogeneous and personalized QoE demands for the mass of viewers.
2.  Usually, the source streaming will be sent to the cloud and CDN servers for transcoding and then delivering to massive viewers with different bitrates 
3.  how to flexibly and cost-effectively accommodate the heterogeneous and personalized quality of experience (QoE) demands (such as individual preferences for streaming delays, channel switching latencies and bitrates) for different viewers. 

#### Contribution:

1. makes intelligent decisions at edges based on the massive amount of real-time information from the network and viewers to accommodate personalized QoE with minimized system cost.

2. propose a data-driven deep reinforcement learning (DRL) based solution that can automatically learn the best  suitable strategies for viewer scheduling and transcoding selection.

3. propose a novel edge-assisted framework called DeepCast that is customized for crowdcast services. We 

   comprehensively consider the personalized QoE targets (e.g., different preferences in streaming delay, channel switching latency and bitrate) and system cost (e.g., computation and bandwidth cost), and integrate them into a viewer scheduling optimization.

4. propose a DRL based experience-driven solution in DeepCast, which can effectively learn the best suitable strategy of viewer scheduling and transcoding selection to achieve high personalized QoE and low system cost.

#### Chart&Analyse:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566095227527.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566095563830.png)

> Yang, J., Zhang, L., & Wang, X. A. (2015, November). On Cloud Computing Middleware Architecture. In *2015 10th International Conference on P2P, Parallel, Grid, Cloud and Internet Computing (3PGCIC)* (pp. 832-835). IEEE.

# Paper:  Middleware Architecture

<div align=center>
<br/>
<b>On Cloud Computing Middleware Architecture</b>
</div>

#### Contribution:

1. The middleware is the service-oriented  system architecture of the cloud computing platform. Innovation&consolution:
2. REST.the Representational State Transfer Technology,  can conveniently offer part of the service supporting by the  middleware to callers. 
3. Multiple tenants. It can make one individual system  work for many organizations with good isolation and better  safety. This technology can effectively reduce the purchase and maintenance cost of the application. 
4. Parallel processing. It can process mass data.
5. Application server. Based on the original AS, it is  optimized for the cloud computing system. 
6. Distributed cache. This distributed cache can not only  effectively reduce the pressure of background, but increase the  response speed.  

#### Chart&Analyse:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566907225904.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566907354393.png)

#  OpenFog:雾计算联盟

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566910129634.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566910189211.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566910242151.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566910316727.png)

1. **OpenFog Fabric** is composed of building blocks which allow the construction of a homogenous computational infrastructure on which usefulservices can be delivered to the surrounding ecosystem (e.g. devices,protocol gateways and other fog nodes). The homogenous infrastructure is
   generally built upon heterogeneous hardware and platforms supplied bymultiple vendors.
2. **OpenFog Service**s are built upon the OpenFog fabric infrastructure. These
   services may include network acceleration, NFV, SDN, content delivery,
   device management, device topology, complex event processing, video
   encoding, field gateway, protocol bridging, traffic offloading, crypto,
   compression, analytics platform, analytics algorithms/libraries etc. This is
   an example of a micro-service architecture.
3. **Devices/Applications** are edge sensors, actuators, and applications
   running standalone, within a fog deployment, or spanning fog deployments.
   This is addressed by the OpenFog service layer.
4. **Cloud Services** may take advantage of the cloud for computational work
   that needs to operate on a larger data scope or pre-processed edge data to
   establish policies. These should be leveraged in ways that don’t impede
   operational autonomy.
5. **Security** is fundamental to OpenFog deployments. Discrete units of
   functionality within each architecture layer are wrapped with discretionary
   access control mechanisms so that the OpenFog deployment and the surrounding ecosystem operate in a safe and secure environment. The
   OpenFog architecture will ensure all the data transfers between the
   participating endpoints are secured through the state of the art information
   security practices.
6. **DevOps** are driven by automation enabled by operationally efficient set of
   standard DevOps processes and frameworks. The DevOps in OpenFog drives
   the agility of software upgrades and patching through controlled continuous
   integration processes.

---

> Zhu, Y. J., Yao, J. G., & Guan, H. B. (2020). Blockchain as a Service: Next Generation of Cloud Services [J]. *Journal of Software*, *31*(1), 1-19.

# Paper:  BlockChain Service

<div align=center>
<br/>
<b>Blockchain as a Service: Next Generation of Cloud Services</b>
</div>

#### Contribution:

- `区块链即服务（blockchain as a service）`则是把区块链当作基础设施，并在其上搭建各种满足普通用户需求的应用，向用户提供服务
- 研究了区块链即服务最新的技术发展状况，结合行业研究和企业实践探索，对区块链即服务的架构以及各模块功能进行了概要设计说明，为区块链即服务的发展提供了通用架构模型.
- 分析了结合区块链即服务的云计算相关技术特点，并给出了可能的攻击模型.最后，结合行业区块链即服务的应用.

#### Block Chain:

- **区块链定义：**区块链是一种按照时间顺序将数据区块以链表的方式组合成特定数据结构, 并以密码学方式保证的不可篡改和不可伪造的去中心化共享总账(decentralized shared ledger), 能够安全存储简单的、有先后关系的、能在系统内验证的数据.

- **区块链种类**:`共有链、联盟链、私有链`.共有链是完全去中心化的区块链, 分布式系统中的所有节点均可参与共识机制和交易, 且可以随时加入或退出; 联盟链是部分去中心化的区块链, 区块链只由特定组织团体维护, 预先指派部分节点负责维护共识机制, 新的维护节点加入需要提交申请并且通过身份认证, 但全网所有节点均可参与交易, 查看账本记录; 私有链是完全中心化的区块链, 维护在特定机构内部, 数据的读取权限选择性地对外开放, 类似传统大型企业分布式系统, 但能提供更强的鲁棒性、稳定性[[4](http://www.jos.org.cn/html/2020/1/5891.htm#b4)].

- **区块链工作过程：**区块链是`新型去中心化协议`, 必须基于分布式系统进行维护.它记录着所有历史交易记录, 随着维护节点(矿工)持续生成新区块, 数据记录不断增长.所有区块按时间先后顺序组成链式结构, 整体架构具有可追溯性和可验证性.利用特定的激励机制, 区块链技术保证分布式系统中的节点均会积极参与数据验证过程.同时, 系统通过分布式共识算法决定最新的有效区块.另外, 区块链技术利用非对称密码学算法对数据进行加密, 并通过特殊的共识算法抵御外部攻击, 保证了区块链数据的不可篡改和不可伪造.在多方无需相互信任的环境下, 区块链利用密码学技术, 让分布式系统中所有节点相互协作, 共同维护一个可靠的数据日志.

- **发展阶段**:以`数字货币`(如比特币)为主要特征的`区块链1.0模式`; 以`数字资产和智能合约`为核心的区块链2.0模式, 这一阶段主要触及金融领域, 革新传统的债券发行、股权众筹、证券交易; 以`智能社会`为主要特征的区块链3.0模式, 这一阶段区块链被用于改善社会基础架构, 例如身份认证、医疗、域名、签证, 被称为“万物互联”的最底层协议.

- **区块链及服务：**区块链作为底层分布式账本技术, 可替代如今的数据存储、数据传输系统模块, 并作为底层架构向公众提供服务.

  - ●  `去中心化`:系统依靠的是网络上多个参与者的公平约束, 没有中心决策者, 所以任意每几个节点的权利和义务都是均等的, 而且每一个节点都会储存系统上所有数据.即使单个节点被损坏或遭受攻击, 系统服务依旧能稳定运行;

    ●  `高可用性`:区块链即服务的底层共识算法采用了拜占庭容错(BFT)共识算法, 该算法支持节点动态加入和退出, 实现系统的高可用性, 保证业务不间断运行;

    ●  `扩展性`:区块链即服务系统支持大规模场景下部署和管理的能力, 可以快速进行扩展;

    ●  `透明性`:区块链上所有记录均是可追溯的、全历史的、防篡改的, 并且每一个节点都会储存系统上的全量数据, 保证了系统整体的透明性[[8](http://www.jos.org.cn/html/2020/1/5891.htm#b8)].

#### Structure

##### **微软Bletchley(私有链即服务)架构**

 	    Bletchley对多个区块链机制提供支持, 支持智能合约机制或者未花费交易输出(UTXO)机制.基于智能合约机制的区块链包括Ethereum、Eris, 基于UTXO机制的区块链包括Hyperledger.基础平台层提供了区块链的基础架构, 包括共识协议、网络、数据存储这3部分.Bletchley结合平台即服务对外提供多种服务, 包括:

- 身份认证服务:可以为个人、组织、关键交易、合同、物品创建身份认证, 这个服务可被用于提供纵向服务, 比如了解你的客户(KYC)服务、资产注册;
- 加密服务:有偿加密服务, 机密数据只对拥有者和交易方可见;
- 加密书签服务:当区块链需要与部分外界数据(如时间、市场消息)交互时, 就需要加密书签的参与.共有两种加密书签:效用书签、合同书签.效用书签是处理日期、时间记录、加密功能、外部数据访问; 合同书签是由智能合约自动地创建, 像智能合约一样, 也是一个全代理引擎, 不需要外界的干涉.加密书签服务使得加密书签能被智能合约或UTXO适配器的加密书签代理安全调用;
- 区块链门服务:该服务允许智能合约或者标记化的物品能够在不同账本系统之间传递.它提供了账本间交易传输的完整性;
- 数据服务:核心数据服务, 包括数据分析、数据存储等;
- 管理运作服务:企业联盟分布式账本的部署、管理、运作工具.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113071817004.png)

##### **IBM Hyperledger架构**

- 区块链服务(blockchain):利用分布式共识协议管理分布式账本, 维护一个区块链基础设施, 并通过高效的哈希算法维护世界观(world state)副本;
-  链码服务(chaincode):该服务提供一种安全且轻量级的沙盒运行模式, 是运营智能合约的机制, 用于在确认节点(validating nodes)之间的沟通服务;
- 成员权限管理(membership):该服务用于管理节点身份, 保护用户隐私, 保证网络上的机密性和可审计性.该服务基于公钥基础设施, 引入交易认证授权, 利用证书对接入节点和客户端能力进行了限制.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113071932867.png)

##### **以太坊架构**

> 以太坊的核心部分也是区块链协议, 包括共识、点到点(P2P)网络、区块链.区块链负责维护基础的数据记录存储服务, P2P网络负责节点之间的交互, 共识负责保证网络节点状态的一致性.为了支持分布式应用, 增强以太坊的平台功能, 以太坊还定义了以太坊协议.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113072154712.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113072320465.png)

![产业生态](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112220144824.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112220232114.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112220244246.png)

> Ghosh, A. M., & Grolinger, K. (2020). Edge-Cloud Computing for Internet of Things Data Analytics: Embedding Intelligence in the Edge With Deep Learning. *IEEE Transactions on Industrial Informatics*, *17*(3), 2191-2200.

# Paper:  Edge-Cloud

<div align=center>
<br/>
<b>Edge-Cloud Computing for Internet of Things
Data Analytics: Embedding Intelligence
in the Edge With Deep Learning</b>
</div>

#### Contribution:

- Reversible: the approaches that reduce data with ability to reproduce the original data from the reduced representations.

![Edge-cloud ML system](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113094034748.png)

![Location-based data reduction](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113094329700.png)

![Similarity-based data reduction](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210113094347561.png)

# Resource

- [中国云计算图谱](https://www.analysys.cn/article/detail/1000470)
- [云计算一些概念介绍](https://blog.csdn.net/fengbingchun/article/details/79719338#:~:text=(1)%E3%80%81%E4%BA%91%E8%AE%A1%E7%AE%97%E6%98%AF,%E6%8F%90%E4%BE%9B%E7%BD%91%E7%BB%9C%E8%AE%BF%E9%97%AE%E7%9A%84%E6%A8%A1%E5%BC%8F%E3%80%82&text=(2)%E3%80%81%E4%BA%91%E8%AE%A1%E7%AE%97%E6%98%AF,%E7%A7%8D%E7%BB%88%E7%AB%AF%E5%92%8C%E5%85%B6%E5%AE%83%E8%AE%BE%E5%A4%87%E3%80%82)

- [区块链与云计算关系](http://www.ctiforum.com/news/guandian/537009.html)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/cloudcomputing_relative/  

