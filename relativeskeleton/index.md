# RelativeSkeleton


### 基于Hadoop 58同城离线计算平台

![img](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPjdVTp7C5IqicEiafiaGSTIUaNeXvpV3vHXDohd6pHrK8mDU3qXwqSrXia4nQIicibBmPUN1LRPSrdeiaibzg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- **数据接入：**文本的收集，我们采用 flume 接入，然后用 kafka 做消息缓冲，我们基于 kafka client 打造了一个实时分发平台，可以很方便的把 kafka 的中间数据打到后端的各种存储系统上。
- **离线计算：**我们主要基于 Hadoop 生态的框架做了二次定制开发。包括 HDFS、YARN、MR、SPARK。
- **实时计算：**目前主要是基于 Flink 打造了一个一栈式的流式计算开发平台 Wstream。
- **多维分析：**我们主要提供两组多维分析的解决方案。离线的使用 Kylin，实时的使用 Druid。
- **数据库：**在数据库的这个场景，我们主要还是基于 HBase 的这个技术体系来打造了出来，除了 HBase 提供海量的 K-V 存储意外，我们也基于 HBase 之上提供 OpenTSDB 的时序存储、JanusGraph 图存储。



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/relativeskeleton/  

