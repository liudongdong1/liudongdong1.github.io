# StreamAnalyse-BloomFilter


> A **Bloom filter** is a space-efficient [probabilistic](https://en.wikipedia.org/wiki/Probabilistic) [data structure](https://en.wikipedia.org/wiki/Data_structure), conceived by [Burton Howard Bloom](https://en.wikipedia.org/w/index.php?title=Burton_Howard_Bloom&action=edit&redlink=1) in 1970, that is used to test whether an [element](https://en.wikipedia.org/wiki/Element_(mathematics)) is a member of a [set](https://en.wikipedia.org/wiki/Set_(computer_science)). a query returns either "possibly in set" or "definitely not in set". the shortcoming of this structure is that the more elements that are added to the set, the larger the probability of false positives. and , Bloom filters do not store the data items at all, and a separate solution must be provided for the actual storage

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572229700748.png)

**level**: IEEE International Conference on Computer Communications  INFCOM CCFA
**author**: Tong Yang
**date**: '2019-4'
**keyword**:

- Size measurement, Hash function

------

### Paper: SA Counter

<div align=center>
<br/>
<b>A Generic Technique for Sketches to Adapt to Different Counting Ranges</b>
</div>


#### Summary

使用概率的方法，在可以有一定的容错下，自适应不同的counting range。

- 静态模型中概率 如何选择？文中是predefined
- 文中说这种概率计数所带来的误差在大流是是可以忽略的？ 理论界限是多少
- 刚开始 静态模型中定义 length of the sign and counting part.

#### Research Objective

- **Application Area**: net work measurement provide indispensable information for congestion control,DDoS attack detection,heavy hitter identification,heavy change detection
- **Purpose**: in order to balance the number of counters and the size of each counter  to achieve high and constant processing speed to keep up with line rate and memory efficiency, we propose self-adaptive counters.

#### Problem&Challenge Statement

- skewed distribution flow
- high speed of flows
- have no idea of the approximate flow size of elephant flows beforehand

previous work:

- sampling   low accuracy
- a compact data structure called sketch (Count,CM,CU,C Sketch,Sophisticated Sketches)

#### Methods

【定义问题1】 each small counter has to be able to represent the size of both mouse and elephant flow.

use two version: Static Sign Bits version and Dynamic Sign Bits version, split each counter into two parts,sign bits ,and counting bits,when sign bits are all 0,increment the counting bits normally,else increment the counting bits with a probability calculated by the value of sign bits.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572159985155.png)

for Static Version: the buckets structure as follows:
![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572160032609.png)

so the total count of the buckets is:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572160089664.png)

insertion algorithm:
![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572160126512.png)

query algorithm:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572160151034.png)

【定义子问题2】how to determine how many bits should be assigned for the sign bits?  Using the dynamic sign bit version .

for dynamic counter structure:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572160188069.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572160351224.png)

#### Evaluation

- Environment set up: 
- Dataset :  IP trace Datasets ,Synthetic Datasets   using C++ to implemention

#### Conclusion

- use `small counters` to accurately `record the sizes of both elephant and mouse flow` ,achieving memory efficiency and constant fast speed. and applied to CM,CU,C
- according to experiment based on two real datasets and one synthetic dataset ,self-adaptive counters have superior performance.

#### Notes

- Flow identifiers are selected from the header fields of packets,(source ip address,port,destination ip,address,protocol)
- Flow size is defined as the number of packets in a flow
- Flow volume is defined as the number of bytes in the flow

------

**level**: IEEE/ACM TRANSACTIONS ON NETWORKING  计算机网络 CCF A类
**author**: Tong Yang
**date**: '2019'
**keyword**:

- sketch,network measurements,elephant flow

### Paper：HeavyKeeper

<div align=center>
<br/>
<b>HeavyKeeper:An Accurate Algorithm for Finding Top-k Elephant Flows</b>
</div>


#### Summary

这篇文章主要讲述了 如何在网络流找出Top-k个大象流问题，使用decay概率方式有效的过滤大部分老鼠流，在fingerprint collsion方面提出的理论可以学习使用。

- ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572153854005.png)这里每个bucket 中fingerprintfield counterfield 各占多少字节是固定的，文中提到如果counterfield 都比较大时 添加新的array，这里能不能采用动态算法调节FP，C 间的存储空间
- only focus  on handle top-k flows detection, cann't handle other flow measurement tasks(flow size estimation,entropy detection),cant support weighted updates.
- Fingerprint 值文中没具体指出用什么来计算
- Top-k里面存储的是流的ID，当流很大的情况下记录那个ID会不会溢出，虽然只有K个ID值，能不能也用新的hash函数计算
- 没有考虑满足Case2的情况下：发生fingerprint collision 导致错误的增加

#### Research Objective

- **Application Area**: dynamically scheduling elephant flows,network capacity planning ,anomaly detection ,caching of forwarding table entries. data mining,information retrieval,databases,security
- **Purpose**: Finding the Top-k elephant flows  and achieve space-accuracy balance by actively removing small flows through decaying while minimizing the impact on large flows. 
- ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572145566768.png)

#### Problem&Challenge Statement

- the distribution of flow sizes is highly skewed ,the majority are mouse flows,while minority are elephant flows
- line rates of modern networks make it practically impossible to accurately track the information of all flows.
- on-chip memory is small.

previous work:

-  <font color="red">count-all strategy</font> relies on  a sketch to measure the sizes of all flows, while using a min-heap to keep track of the top-k flows. <font color="red">not memory efficient</font>
-  <font color="red">admit-all-count-some strategy</font> (Lossy Counting,Space-Saving,CSS) maintains a data structure called stream-Summary to count only some flows.   <font color="red">but assumes every new incoming flow is an elephant flow,and expels the smallest one in the summary to make room for the new one,which can causes significant error,under tight memory</font>
-  <font color="red">new strategy</font> like Elastic sketch,Heavy Guardian ,Cold Filter ,Counter tree ,这里没有作比较 

#### Methods

A new algorithm HeavyKeeper which uses the similar strategy called count-with-exponential decay. Keep all  elephant flows while drastically reducing space wasted on mouse flows. 

【定义问题1】 how to keep the flow ?  design a data structure called HeavyKeeper, and using Min-heap to store the K-top elephant flows

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572145776329.png)

【定义问题2】how to insert and query package ？

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572145990086.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572146007970.png)

【定义问题3】how to distinguish the elephant flow and mouse flow ? using decay Probability.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572146396956.png)

【定义问题4】how to deal with the fingerprint collision detection caused by hash collision?

a mouse flow fj mapped to the same bucket has the same fingerprint as fi, i.e., Fi = Fj due to hash
collisions. estimated size is drastically over-estimated  if flow has a fingerprint collision in all d arrays,
the mouse flow fj will probably be inserted into the min-heap.    using IDs of flow instead of fingerprints(memory limited)   Still using  fingerprint ,propose a solution based on the Theorem. 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572146666813.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572146837647.png)

【定义问题5】in the worst case, when a new flow arrives,if all values of its mapped d counters are large enough, it could never be inserted into some buckets  using extra global counter to record ,if >threshold,add a new array.

【定义问题6】how to deal with unnecessary and unhelpful to decay large counters?  Minimum Decay ,decay the smallest one instead of decaying all the mapped counters.

插入流程：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572147118119.png)

#### Evaluation

- Environment set up: platform : a server with dual 6-core CPUs (24 threads, Intel Xeon CPU E5-2620 @2 GHz) and 62 GB total system memory. Each core has two L1 caches with 32KB memory (one instruction cache and one data cache) and one 256KB L2 cache. All cores share one 15MB L3 cache.
- Dataset : Campus Dataset, CAIDA Dataset ,Synthetic Datasets.  using c++ to implemetation.

#### Conclusion

- propose a novel data structure, called Heavy Keeper, which achieves a much higher precision on top-k queries and a much lower error rate on flow size estimation, compared to previous algorithms.
- intelligently omits mouse flows, and focuses on recording the information of elephant flows by using the exponential-weakening decay strategy.
- Heavy Keeper achieves 99.99% precision for finding the top-k elephant flows, while
  also achieving a reduction in the error rate of the estimated flow size by about 3 orders of magnitude compared to the state-of-the-art algorithms

#### Notes

- code : https://github.com/papergitkeeper/heavykeeper-project  
- 交换机 line-rate 用于端口限速，主要用于出端口，traffic-limit 用于流限速，主要用于入端口，对应line-rate 端口是出流量限制，有芯片内报文缓存，可以让流量稳定在设置的值，没有丢包。

**level**: SIGMOD’ 18: 2018 International Conference on Management of Data  CCF A 类
**author**:  Tong Yang
**date**: '2018'
**keyword**:

- computing methodologies Distributed algorithms ,Quantile Sketch,Frequency Sketch,Quantification

------

### Paper: SketchML

<div align=center>
<br/>
<b>Accelerating Distributed Machine Learning with Data Sketches</b>
</div>


#### Summary

the eye-catching point of this article are as follows:

- design a quantile-bucket quantification method that uses a quantile sketch to sort gradient values into buckets and encodes them with the buckets indexes
- MinMaxSketch builds a set of hash tables and solves hash collisions with a MinMax strategy
- a delta-binary encoding method that calculates the increment of the gradient keys and stores them with fewer bytes.

#### Research Objective

- **Motivation**: partition a training dataset over workers and make each worker independently propose gradients,how to efficiently exchange gradients among workers since the communication often dominates the total cost.
- **Application area** : Large Model training; Cloud environment(to minimize the transmission through network); Geo-Distributed ML(data movement over WAN is much slower than LAN), <font color="red">IOT(communication between end devices)</font>

- **Purpose**: Distributed ML algorithms trained by stochastic gradient descent involve commucating gradients through the network,how to efficiently handle a sparse and nonuniform gradient consisting of key-value pairs?

#### Proble Statement

- Data Model

![1572313895442](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572313895442.png)

- System Architecture

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572314058134.png)

- 【question 1】 how to compress Gradient values ?

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1572314125372.png)

#### Notes

- 近似框架（approximate framework），重组了在文献[17,2,22]中提出的思想，如算法2所示。为了进行总结(summarize)，**该算法会首先根据特征分布的百分位数(percentiles of feature distribution)，提出候选划分点(candidate splitting points)。接着，该算法将连续型特征映射到由这些候选点划分的分桶(buckets)中，聚合统计信息，基于该聚合统计找到在建议（proposal）间的最优解**。



## Cardinality - HyperLogLog

> ​        HyperLogLog is a streaming algorithm used for estimating the number of<font color="red"> distinct elements</font> (the cardinality) of very large data sets. HyperLogLog counter can count one billion distinct items with an accuracy of 2% using only 1.5 KB of memory. It is based on the bit pattern observation that for a stream of randomly distributed numbers, if there is a number x with the maximum of leading 0 bits k, the cardinality of the stream is very likely equal to 2^k.
>

## Frequency - Count-Min Sketch

> ​         Count-Min sketch is a probabilistic sub-linear space streaming algorithm. It is somewhat similar to bloom filter. The main difference is that<font color="red"> bloom filter represents a set as a bitmap, while Count-Min sketch represents a multi-set which keeps a frequency distribution summary.</font>
>

## T-Digest

> ​         [T–Digest](https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf) 是一种支持精确排序统计的数据结构，如在常数空间复杂度下计算百分位数和中位数。以很小的误差达到高效的性能，使得T-Digest很适合用于流的排序计算，这要求输入是有限且有序以达到高度的精确性。T-Digest本质上是一种在元素被添加时智能调整桶和频数的自适应的直方图

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/streamanalyse-bloomfilter/  

