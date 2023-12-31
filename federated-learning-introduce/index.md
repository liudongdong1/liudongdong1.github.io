# Federated Learning Introduce


### 1. 背景

- 数据孤岛问题严重，由于安全问题、竞争关系和审批流程等因素，数据在行业、甚至是在公司内部以“孤岛”的形式存在。

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523161326465.png)

- 重视数据隐私和安全已经成为世界性的趋势，在国外，2018年5月，欧盟的[《通用数据保护条例》](https://link.zhihu.com/?target=https%3A//gdpr-info.eu/)(General Data Protection Regulation,GDPR)正式开始生效,该条例对于数据保护做出了严格规定。同时在国内，对于数据保护的力度越来越严格，国家先后发布《网络安全法》、[《信息](https://link.zhihu.com/?target=https%3A//www.tc260.org.cn/upload/2018-01-24/1516799764389090333.pdf)[安全技术 个人信息安全规范》](https://link.zhihu.com/?target=https%3A//www.tc260.org.cn/upload/2018-01-24/1516799764389090333.pdf)和[《互联网个人信息安全保护指南》](https://link.zhihu.com/?target=http%3A//www.beian.gov.cn/portal/topicDetail%3Fid%3D88)等法律法规，同时公安部也在严厉打击数据安全犯罪行为。在这样的背景之下，即便行业有意共享数据，也面临政策、法律合规的严峻问题。

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523161357774.png)

- 传统机器学习：数据集中的过程中有出现数据泄露的风险。基于云的AI解决方案以及API，这种模式使用户无法控制AI产品的使用以及个人隐私数据，而通过数据集中公司却可以做到垄断数据。

### 2. 解决方案

#### .1. 基于硬件可信执行环境技术的可行计算（TEE： Trusted Execution Environment)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200409213309662.png)

#### .2. 基于密码学的多方安全计算（MPC：Multi-party Computation）

MPC 方案的大致原理如下图所示：<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200409213309662.png" style="zoom:150%;" />

- 混淆电路（Garbled Circuit）：任意函数最后在计算机语言内部都是由加法器、乘法器、移位器、选择器等电路表示，而这些电路最后都可以仅由 AND 和 XOR 两种逻辑门组成。一个门电路其实就是一个真值表，假设我们把门电路的输入输出都使用不同的密钥加密，设计一个加密后的真值表，这个门从控制流的角度来看还是一样的，但是输入输出信息都获得了保护。
- 秘密分享（Secret Sharing）：将每个数字随机拆散成多个数并分发到多个参与方那里。然后每个参与方拿到的都是原始数据的一部分，一个或少数几个参与方无法还原出原始数据，只有大家把各自的数据凑在一起时才能还原真实数据。![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429183836278.png)

#### .3. 联邦学习

> 本质：联邦学习本质上是一种**分布式**机器学习技术，或机器学习**框架**。
>
> 目标：联邦学习的目标是在保证数据隐私安全及合法合规的基础上，实现共同建模，提升AI模型的效果。
>
> 优点：
>
> - 数据隔离：联邦学习的整套机制在合作过程中，数据不会传递到外部。
> - 无损：通过联邦学习分散建模的效果和把数据合在一起建模的效果对比，几乎是无损的。
> - 对等：合作过程中，合作双方是对等的，不存在一方主导另外一方。
> - 共同获益：无论数据源方，还是数据应用方，都能获取相应的价值。

##### .1. 分类

- 横向联邦学习：比如不同地区的银行间，他们的业务相似（特征相似），但用户不同（样本不同）

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523160032949.png)

- 纵向联邦学习：比如同一地区的商超和银行，他们触达的用户都为该地区的居民（样本相同），但业务不同（特征不同）。

  ![image-20210523160125470](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523160125470.png)

- 联邦迁移学习：如不同地区的银行和商超间的联合。主要适用于以深度神经网络为基模型的场景。迁移学习的核心是，找到`源领域和目标领域之间的相似性（不变量）`。

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523160717864.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523155802328.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429184116690.png)

##### .2.  挑战

- **Expensive Communication**

> 联邦网络可能由大量设备组成，例如数百万部智能手机，网络中的通信速度可能比本地计算慢很多个数量级。为了使模型与联邦网络中的设备生成的数据相匹配，因此有必要开发通信效率高的方法，作为训练过程的一部分，迭代地发送小消息或模型更新，而不是通过网络发送整个数据集。为了在这种情况下进一步减少通信，需要考虑的两个关键方面是：（i）`减少通信回合的总数`，或（ii）`在每一回合减少发送的消息大小`。

- **Systems Heterogeneity**

> 由于硬件（CPU，内存）、网络连接（3G，4G，5G，wifi）和电源（电池电量）的变化，联邦网络中每个设备的存储、计算和通信能力可能不同。此外，每个设备上的网络大小和系统相关限制导致同时活跃的设备通常仅占一小部分，例如，一百万个设备网络中的数百个活跃设备。每个设备也可能不可靠，并且由于连接性或能量限制，活跃设备在给定迭代中随机失活的情况并不少见。这些系统级特性极大地加剧了诸如掉队者缓解和容错等挑战。因此，开发和分析的联邦学习方法必须：(i) `预计参与人数较少`，(ii)` 容忍异构硬件`，以及(iii)` 对网络中的已下线设备具有鲁棒性`。

- **Statistical Heterogeneity**

> 设备经常以non-IID的方式在网络上生成和收集数据，例如，移动电话用户在下一个单词预测任务的上下文中使用了不同的语言。此外，跨设备的数据点的数量可能有很大的变化，并且可能存在捕获设备之间的关系及其相关分布的底层结构。这种数据生成范例违反了分布式优化中经常使用的独立同分布（I.I.D）假设，增加了掉队者的可能性，并且可能在建模、分析和评估方面增加复杂性。事实上，虽然标准的联邦学习问题旨在学习一个单一的全局模型，但是存在其他选择，例如同时通过多任务学习框架学习不同的局部模型。在这方面，联邦学习和元学习的主要方法之间也有密切的联系。多任务和元学习视角都支持个性化或特定于设备的建模，这通常是处理数据统计异质性的更自然的方法。

- **Privacy Concerns**

> **联邦学习通过共享模型更新（例如梯度信息）而不是原始数据，**朝着保护在每个设备上生成的数据迈出了一步。**然而，在整个训练过程中进行模型更新的通信仍然可以向第三方或中央服务器显示敏感信息。**虽然最近的方法旨在`使用安全多方计算或差异隐私等工具增强联邦学习的隐私性`，但这些方法通常`以降低模型性能或系统效率为代价`提供隐私。在理论和经验上理解和平衡这些权衡是实现私有联邦学习系统的一个相当大的挑战。

- **Incentives Mechanism**

> 模型建立后，模型的性能将在实际应用中体现出来，这种性能可以记录在永久数据记录机制（如区块链）中。提供更多数据的组织会更好，模型的有效性取决于数据提供者对系统的贡献。这些模型的有效性基于联邦机制分发给各方，并继续激励更多组织加入数据联邦。不仅考虑了`多个组织之间协作建模的隐私保护和有效性`，还考虑了`如何奖励贡献更多数据的组织，以及如何通过共识机制实施激励`。因此，联邦学习是一种“闭环”学习机制。

### 2. 案例蚂蚁金融共享学习

#### 2.1 基于TEE的共享学习

- 底层使用 Intel 的 SGX 技术，并可兼容其它 TEE 实现。目前，基于 SGX 的共享学习已支持集群化的模型在线预测和离线训练。
- 在线预测

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429183704788.png)

- 与传统分布式框架不同的地方在于，每个服务启动时会到集群管理中心（ClusterManager，简称 CM）进行注册，并维持心跳，CM 发现有多个代码相同的 Enclave 进行了注册后，会通知这些 Enclave 进行密钥同步，Enclave 收到通知后，会通过远程认证相互确认身份。当确认彼此的 Enclave 签名完全相同时，会通过安全通道协商并同步密钥。
- 通过集群化方案解决了在线服务的负载均衡，故障转移，动态扩缩容，机房灾备等问题；
- 通过多集群管理和 SDK 心跳机制，解决代码升级，灰度发布，发布回滚等问题；
- 通过 ServiceProvider 内置技术配合 SDK，降低了用户的接入成本；
- 通过提供易用性的开发框架，使得用户在开发业务逻辑时，完全不需要关心分布式化的逻辑；
- 通过提供 Provision 代理机制，确保 SGX 机器不需要连接外网，提升了系统安全性。

- 模型离线训练：![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429184014744.png)
- 步骤流程：

  - 机构用户从 Data Lab 下载加密工具
  - 使用加密工具对数据进行加密，加密工具内嵌了 RA 流程，确保加密信息只会在指定的 Enclave 中被解密
  - 用户把加密数据上传到云端存储
  - 用户在 Data Lab 的训练平台进行训练任务的构建
  - 训练平台将训练任务下发到训练引擎
  - 训练引擎启动训练相关的 Enclave，并从云端存储读取加密数据完成指定的训练任务。

#### **2.2 基于 MPC 的共享学习**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429184033960.png)

- 安全技术层：安全技术层提供基础的安全技术实现，比如在前面提到的秘密分享、同态加密、混淆电路，另外还有一些跟安全密切相关的，例如差分隐私技术、DH 算法等等；

- 基础算子层：在安全技术层基础上，我们会做一些基础算子的封装，包括多方数据安全求交、矩阵加法、矩阵乘法，以及在多方场景下，计算 sigmoid 函数、ReLU 函数等等；同一个算子可能会有多种实现方案，用以适应不同的场景需求，同时保持接口一致；

- 安全机器学习算法：有了基础算子，就可以很方便的进行安全机器学习算法的开发，这里的技术难点在于，如何尽量复用已有算法和已有框架，我们在这里做了一些有益的尝试，但也遇到了很大的挑战。

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429184047309.png)

- 机构用户从 Data Lab 下载训练服务并本地部署

- 用户在 Data Lab 的训练平台上进行训练任务的构建

- 训练平台将训练任务下发给训练引擎

- 训练引擎将任务下发给机构端的训练服务器 Worker

- Worker 加载本地数据

- Worker 之间根据下发的训练任务，通过多方安全协议交互完成训练任务

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200429184100859.png)

### 4. Tensorflow框架[分布式训练策略](https://cloud.tencent.com/developer/article/1421382)

#### 4.1 模型并行

- 将模型部署到很多设备上（设备可能分布在不同机器上，下同）运行，比如多个机器的GPUs。当神经网络模型很大时，由于显存限制，它是难以完整地跑在单个GPU上，这个时候就需要把模型分割成更小的部分，不同部分跑在不同的设备上，例如将网络不同的层运行在不同的设备上。由于模型分割开的各个部分之间有相互依赖关系，因此计算效率不高。所以在模型大小不算太大的情况下一般不使用模型并行。

#### 4.2 数据并行

- 数据并行在多个设备上放置相同的模型，各个设备采用不同的训练样本对模型训练。每个Worker拥有模型的完整副本并且进行各自单独的训练。![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410092017809.png)
- **异步并行**：梯度失效问题（stale gradients）

  - 各个设备完成一个mini-batch训练之后，不需要等待其它节点，直接去更新模型的参数。从下图中可以看到，在每一轮迭代时，不同设备会读取参数最新的取值，但因为不同设备读取参数取值的时间不一样，所以得到的值也有可能不一样。根据当前参数的取值和随机获取的一小部分训练数据，不同设备各自运行反向传播的过程并独立地更新参数。![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410092457696.png)
- 同步并行：

  - 所有的设备都是采用相同的模型参数来训练，等待所有设备的mini-batch训练完成后，收集它们的梯度后执行模型的一次参数更新。在同步模式下，所有的设备同时读取参数的取值，并且当反向传播算法完成之后同步更新参数的取值。单个设备不会单独对参数进行更新，而会等待所有设备都完成反向传播之后再统一更新参数 。

<div style="float:left;border:solid 1px 000;margin:2px;">
    <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410092617194.png" width=40% height="200" >
    <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410092726202.png" width=40% height="200" >
</div>


### 5. 分布式训练架构

- **分布式并行模式**：深度学习模型的训练是一个迭代的过程，如图2所示。在每一轮迭代中，前向传播算法会根据当前参数的取值计算出在一小部分训练数据上的预测值，然后反向传播算法再根据损失函数计算参数的梯度并更新参数。在并行化地训练深度学习模型时，不同设备（GPU或CPU）可以在不同训练数据上运行这个迭代的过程，而不同并行模式的区别在于不同的参数更新方式。

  <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410092208013.png" width="450" height="200" >

#### 5.1 Parameter Server架构（PS）

- 在PS架构中，集群中的节点被分为两类：parameter server和worker。其中parameter server存放模型的参数，而worker负责计算参数的梯度。在每个迭代过程，worker从parameter sever中获得参数，然后将计算的梯度返回给parameter server，parameter server聚合从worker传回的梯度，然后更新参数，并将新的参数广播给worker。<font color=red>当worker数量较多时，ps节点的网络带宽将成为系统的瓶颈</font>

  <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410093002069.png" width="450" height="200" >

#### **5.2 Ring AllReduce架构**

- 各个设备都是worker，没有中心节点来聚合所有worker计算的梯度。Ring AllReduce算法将 device 放置在一个逻辑环路（logical ring）中。每个 device 从上行的device 接收数据，并向下行的 deivce 发送数据，因此可以充分利用每个 device 的上下行带宽。

 <figure class="third">
     <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410093340868.png" width=33%>
     <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410093714868.png" width=33%>
     <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200410093728287.png" width=30%>
</figure>




- 使用 Ring Allreduce 算法进行某个稠密梯度的平均值的基本过程如下：
  - 将每个设备上的梯度 tensor 切分成长度大致相等的 num_devices 个分片；
  - ScatterReduce 阶段：通过 num_devices - 1 轮通信和相加，在每个 device 上都计算出一个 tensor 分片的和；
  - AllGather 阶段：通过 num_devices - 1 轮通信和覆盖，将上个阶段计算出的每个 tensor 分片的和广播到其他 device；
  - 在每个设备上合并分片，得到梯度和，然后除以 num_devices，得到平均梯度；

### 6. 开源框架

- FATE：微众银行AI团队推出的工业级联邦学习框架
- MORSE：蚂蚁区块链打造的数据安全共享基础设施
- PrivPy：华控清交研发的安全多方计算平台
- FMPC：富数科技推出的私有化部署联邦建模平台
- 蜂巢系统：平安科技自主研发的联邦智能系统
- 点石平台：百度研发的可信云端计算及联合建模平台

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523161215978.png)

#### 6.1 FATE联邦学习框架

##### 系统架构

1）FATE技术架构如（图1）所示，部分模块简介如下：

- EggRoll：分布式计算和存储的抽象；
- Federated Network：跨域跨站点通信的抽象；
- FATE FederatedML：联邦学习算法模块，包含了目前联邦学习所有的算法功能；
- FATE-Flow | FATE-Board：完成一站式联邦建模的管理和调度以及整个过程的可视化；
- FATE-Serving：联邦学习模型API模块。

<img src="https://pic4.zhimg.com/80/v2-2841daf33f2802a25cad14a50b2b2927_720w.jpg" alt="img" style="zoom:80%;" />

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200411105256021.png" alt="image-20200411105256021" style="zoom:80%;" />

##### 安装部署  

- docker 镜像安装/ 主机安装/ docker从源代码编译（文档Fate/standalone-deploy/readme.md)

docker安装：

1. 主机需要能够访问外部网络，从公共网络中拉取安装包和docker镜像。

2. 依赖[docker](https://download.docker.com/linux/)和[docker-compose](https://github.com/docker/compose/releases/tag/1.24.0)，docker建议版本为18.09，docker-compose建议版本为1.24.0，您可以使用以下命令验证docker环境：docker --version和docker-compose --version，docker的起停和其他操作请参考docker --help。

3. 执行之前，请检查8080、9060和9080端口是否已被占用。 如果要再次执行，请使用docker命令删除以前的容器和镜像。

   请按照以下步骤操作:


```shell
#获取安装包
FATE $ wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone-fate-1.3.0.tar.gz
FATE $tar -xvf docker_standalone-fate-1.3.0.tar.gz

#执行部署
FATE $ cd docker_standalone-fate-1.3.0
FATE $ bash install_standalone_docker.sh

#验证和测试
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ bash ./federatedml/test/run_test.sh
# 控制面板  http://hostip:8080
#测试 test目录下
python run_test.py default_env.json -s ./demo/temp_testsuite.json
#example min_test_task  是最小安装测试程序
```

##### 模型训练 Fate-Flow：

1） FAETE 运行job主要通过fate_flow模块来完成，<font color=red>FATE-Flow是用于联邦学习的端到端Pipeline系统</font>，它由一系列高度灵活的组件构成,专为高性能的联邦学习任务而设计。其中包括数据处理、建模、训练、验证、发布和在线推理等功能.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/federated_learning_pipeline.png)

2）支持以下功能

- <font color=red>使用DAG定义Pipeline；</font>
- 使用 **JSON** 格式的 **FATE-DSL** 描述DAG；
- FATE具有大量默认的联邦学习组件, 例如Hetero LR/Homo LR/Secure Boosting Tree等；
- 开发人员可以使用最基本的API轻松实现自定义组件, 并通过DSL构建自己的Pipeline；
- 联邦建模任务生命周期管理器, 启动/停止, 状态同步等；
- 强大的联邦调度管理, 支持DAG任务和组件任务的多种调度策略；
- <font color=red>运行期间实时跟踪数据, 参数, 模型和指标；</font>
- 联邦模型管理器, 模型绑定, 版本控制和部署工具；
- 提供HTTP API和命令行界面；
- <font color=red>提供可视化支持, 可在 **FATE-Board** 上进行可视化建模。</font>

3）DSL编写

- 定义此组件的模块
- 定义输入, 包括数据, 模型或isometric_model(仅用于FeatureSelection)
- 定义输出, 包括数据和模型

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200411104629489.png" style="zoom:50%;" />

- DSL 编写规则

  - module：用来指定使用的模块

  - input: 分为data， model

    Data: 有三种可能的输入类型

    - data: 一般被用于 data_io 模块, feature_engineering 模块或者 evaluation 模块

    - train_data: 一般被用于 homo_lr, heero_lr 和 secure_boost 模块。如果出现了 train_data 字段，那么这个任务将会被识别为一个 fit 任务
    - eval_data: 如果存在 train_data 字段，那么该字段是可选的。如果选择保留该字段，则 eval_data 指向的数据将会作为 validation set。若不存在 train_data 字段，则这个任务将被视作为一个 predict 或 transform 任务。 

    Model: 有两种可能的输入类型：

    - model: 用于同种类型组件的模型输入。
    - isometric_model: 用于指定继承上游组件的模型输入。

  - output: 和 input 一样，有 data 和 model 两种类型。

- Pipeline 运行实例

  <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200411105052994.png" alt="image-20200411105052994" style="zoom:50%; align=center;" />

- Fate-Flow部署在``$PYTHONPATH/fate_flow/``中，它依赖两个配置文件：``$PYTHONPATH/arch/conf/server.conf``, ``$PYTHONPATH/fate_flow/settings.py``

  - **server.conf**: f配置所有FATE服务的地址，不同部署模式的FATE-Flow需要不同的Fate服务。 service.sh:  启动/停止/重启服务   

  - **settings.py**: 

    | 配置项                      | 配置项含义                           | 配置项值                 |
    | --------------------------- | ------------------------------------ | ------------------------ |
    | IP                          | FATE-Flow 的监听地址                 | 默认0.0.0.0              |
    | GRPC_PORT                   | 监听 FATE-Flow grpc 服务的端口       | 默认9360                 |
    | HTTP_PORT                   | FATE-Flow的http服务器的侦听端口      | 默认9380                 |
    | WORK_MODE                   | FATE-Flow的工作模式                  | 0(单机模式), 1(群集模式) |
    | USE_LOCAL_DATABASE          | 是否使用本地数据库(sqlite)           | False表示否, True表示是  |
    | USE_AUTHENTICATION          | 是否启用身份验证                     | False表示否, True表示是  |
    | USE_CONFIGURATION_CENTER    | 是否使用Zookeeper                    | False表示否,True表示是   |
    | MAX_CONCURRENT_JOB_RUN      | 同时并行执行的Pipeline作业(job) 数量 | 默认5                    |
    | MAX_CONCURRENT_JOB_RUN_HOST | 最大运行作业(job) 数量               | 默认值10                 |
    | 数据库                      | mysql数据库的配置                    | 定制配置                 |
    | REDIS                       | Redis的配置                          | 定制配置                 |
    | REDIS_QUEUE_DB_INDEX        | Redis队列的Redis数据库索引           | 默认值0                  |


3）[Fate-Flow 客户端用法](https://github.com/FederatedAI/FATE/blob/master/fate_flow/doc/fate_flow_rest_api.md) 

- <font color=red>Upload data config file: for upload data</font>

  - file: file path
  - head: Specify whether your data file include a header or not
  - <font color=red>partition: Specify how many partitions used to store the data</font>
  - <font color=red>table_name & namespace: Indicators for stored data table.</font>
  - work_mode: Indicate if using standalone version or cluster version. 0 represent for standalone version and 1 stand for cluster version.

- <font color=red>DSL config file: for defining your modeling task</font>

  - module: specify which component to use, the field should be one of the algorithm modules Fate support.

- <font color=red>Submit runtime conf: for setting parameters for each component</font>

  - initiator: 发起者角色何参与制id
  - role: indicate all the party ids for all roles
  - role_parameters: 
  - algorithm_parameters: those parameters are same among all parties are here.

- 数据上传

  ```shell
  python fate_flow_client.py -f upload -c examples/upload_guest.json
  python fate_flow_client.py -f upload -c examples/upload_host.json
  #Upload data是上传到eggroll里面，变成后续算法可执行的DTable格式。
  ```

- 提交任务

  ```shell
  python fate_flow_client.py -f submit_job -d examples/test_hetero_lr_job_dsl.json -c examples/test_hetero_lr_job_conf.json
  #the table_name and namespace in the conf file match with upload_data conf
  ```

- 查询作业

  ```shell
  python fate_flow_client.py -f query_job -r guest -p 10000 -j $job_id
  ```

- 发布模型

  ```shell
  python fate_flow_client.py -f load -c examples/publish_load_model.json
  #load可以理解为发送模型到模型服务上, 而bind是绑定一个模型到模型服务
  ```

- arbiter是用来辅助多方完成联合建模的，它的主要作用是聚合梯度或者模型。比如纵向lr里面，各方将自己一半的梯度发送给arbiter，然后arbiter再联合优化等等。

- <font color=red>guest代表数据应用方。</font>

- <font color=red>host是数据提供方。</font>

- <font color=red>local是指本地任务, 该角色仅用于upload和download阶段中。</font>

##### Train Model

- add/modify "need_deploy" field for those modules that need to deploy in predict stage. <font color=red>The "need_deploy" field is True means this module should run a "fit" process and the fitted model need to be deployed in predict stage.</font>

- <font color=red>config to have a model output except Intersect module</font> to store the trained model and make it usable in inference stage.

- <font color=red>Get training model's model_id and model_version</font>. There are two ways to get this

  ```
  python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f job_config -j ${jobid} -r guest -p ${guest_partyid}  -o ${job_config_output_path}
   
  where
  $guest_partyid is the partyid of guest (the party submitted the job)
  $job_config_output_path: path to store the job_config
  ```

##### Predict Config:

- job_parameters: the job_type: predict.
- role_parameters: the "eval_data", which means data going to be predicted, should be filled for both Guest and Host parties.
- download the model:

```
python ${your_fate_install_path}/fate_flow/fate_flow_client.py -f component_output_data -j ${job_id} -p ${party_id} -r ${role} -cpn ${component_name} -o ${predict_result_output_dir}

where
${job_id}: predict task's job_id
${party_id}: the partyid of current user.
${role}: the role of current user. Please keep in mind that host users are not supposed to get predict results in heterogeneous algorithm.
${component_name}: the component who has predict results
${predict_result_output_dir}: the directory which use download the predict result to.
```

执行任务后可在FATE-Board面板查看任务信息及相关日志

http://{Your fate-board ip}:{your fate-board port}/index.html#/history

![](https://pic4.zhimg.com/80/v2-6f6ef7836a37877e90ee8a571a859dc7_720w.jpg)

#### 6.2 FMPC联邦建模平台 

**1、系统架构**
技术架构如（图6）所示，部分模块简介如下：

![](https://pic3.zhimg.com/80/v2-2dd1fa512fc1fa0e1870338836fab5ce_720w.jpg)

### 7. 资源

- ###### [联邦学习论文资源](https://github.com/poga/awesome-federated-learning)

- ###### [Android 联邦训练](https://github.com/zouyu4524/fl-android.git)

- ###### [分布式训练](https://mp.weixin.qq.com/s/LRxyvVazRAOR_B0as7ujvg)

- https://tensorflow.google.cn/federated/get_started
  https://blog.csdn.net/Mr_Zing/article/details/100051535

- https://liudongdong.blog.csdn.net/article/details/105464444

- https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247493607&idx=1&sn=16becab2acb865799d9ce5e54b4feb2c&chksm=fbd7558bcca0dc9db7877086285e2543cc60bb2ca5de075e7bca0e6636c394c4222baff4dd1c&scene=27#wechat_redirect

- https://www.cnblogs.com/wt869054461/p/12375011.html



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/federated-learning-introduce/  

