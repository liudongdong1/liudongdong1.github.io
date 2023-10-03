# MeetingDirection


>- 云数据库：除了Eric Brewer关于Kubernetes的keynote，还有一个关于云数据库的industry session。Amazon Aurora从理论和实践上证明公有云场景可以重构关系数据库底层架构，Vertica这样的老牌OLAP系统也上云了。
>- 新型硬件：存储介质从SSD、大内存到非易失性内存，RDMA、GPU、FPGA这样的硬件技术逐步普及。这次SIGMOD除了一场关于新硬件的research session，即“Databases for Emerging Hardware”，还有一场专门的workshop DaMoN（Data Management on New Hardware）。Oracle Labs也有一个RAPID项目尝试做一个基于专用的数据处理芯片的SQL执行优化框架，不过通过内部交流了解到，专用硬件在Oracle Labs前景也不明确，RAPID的项目不会集成到Oracle内核，只会考虑MySQL。
>- 自治数据库：自治数据库在学术界和工业界都很热，Oracle数据库最近几年最重要的研发工作就是自治数据库。CMU的Andrew Pavlo团队这几年在学术界特别活跃，每年都能抓住热点并在SIGMOD和VLDB发表多篇论文，今年的最佳论文就是出自这个团队，他们在自治数据库上做的工作发表在“Query-based Workload Forecasting for Self-Driving Database Management Systems”。
>- AI+数据库：这几年国内很多高校的数据库团队都转型大数据和AI了，北大数据库实验室在这次SIGMOD会议上也发表了两篇AI方向的论文，可见AI的火热。AI + System是这两年兴起的一个热门方向，Google Jeff Dean团队提出的Learned Index很好地利用AI统计实际数据的规律来设计更加高效的索引结构。这次SIGMOD除了Learned Index相关论文外，还有一个专门的关于机器学习的research session“Machine Learning & Knowledge-base Construction”。目前的AI和数据库的结合还是比较浅的，未来是否可以用AI的方法来改进数据库内核的核心组件，例如优化器，这点我不太确定但充满期待。
>- 图数据库：去年SIGMOD的best paper就是关于图计算的“Parallelizing Sequential Graph Computations”，第一作者是Wenfei Fan，今年他们团队又发表了多篇关于图数据库的paper，今年SIGMOD的research session和industry session都有关于图数据库的论文。

### 1. SIGIR 会议

> ACM SIGIR 2021是CCF A类会议，人工智能领域智能信息检索（ Information Retrieval，IR）方向最权威的国际会议。会议专注于信息的存储、检索和传播等各个方面，包括研究战略、输出方案和系统评估等等。

> 根据长文和短文的标题绘制如下词云图，可以看到今年研究方向主要集中在`Recommendation`和`Retrieval`两个方向，也包括Summarization、Conversations等NLP方向；主要任务包括：Ranking、Cross-domain、Multi-Model/Behavior、Few-Shot、User modeling、Personalization等；热门技术包括：Neural Networks、Knowledge Graph、GNN、Attention、Meta Learning等，其中基于Graph的一类方法依旧是今年的研究热点。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/641)

> Collaborative Filtering，  Sequential/Session-based Recommendations，  Conversational Recommender System，  News Recommendations，  Cross-domain/Multi-behavior Recommendations， Social Recommendation

>Graph-based,  VAE-based,  GAN-based,  FM-based

### 2. VLDB

>VLDB会议的全称是Very Large Data Bases Conferences，由 VLDB Endowment 主办，来自全球各地的数据库相关领域研究人员、供应商、参与者、应用开发者等共同参与和关注的国际重大学术会议。

![img](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-68ea62f88e9eca8f7be379d7d3b9b9fc_720w.jpg)

![RDMS](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-2f6d2b9f1f4df1572cfcdf459ec8f63c_720w.jpg)



### 3. SIGKDD

>ACM SIGKDD（国际知识发现与数据挖掘大会，简称KDD），是机器学习领域的顶级国际学术会议，由ACM（国际计算机学会）创办于1995年，已经连续举办了26届，今年将于8月14日至18日在新加坡举办。目前KDD已经发展成为AI领域最具活力、影响最大的国际学术组织之一，代表了学术界和工业界的研究方向，被中国计算机协会推荐为A类会议。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/JkkpaA6B1TtPQ)

### 4. SIGMOD

> Benchmarking and performance evaluation,  Crowd sourcing,   Data models, semantics, query languages,   Data provenance,   Data visualization,   Data warehousing, OLAP, SQL Analytics,   Database monitoring and tuning,   Database security, privacy, access control,   Database usability,   Databases for emerging hardware,   Distributed and parallel databases,   Graph data management, RDF, social networks,   Information extraction,    Information retrieval and text mining,    Knowledge discovery, clustering, data mining,    Query processing and optimization,   Schema matching, data integration, and data cleaning,    Scientific databases,    Semi-structured data,   Spatio-temporal databases,   Storage, indexing, and physical database design,    Streams, sensor networks, complex event processing,    Transaction processing,    Uncertain, probabilistic, and approximate databases,    Machine learning for data management and vice versa

### Resource

- https://zhuanlan.zhihu.com/p/80962482


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/meetingdirection/  

