# IoTWorkRelated


- **Concentrate your bets:** focus on select use cases and tackle barriers to adoption such as security, ROI, IT and operational technology integration. Package IoT solutions into scalable products that you then can roll out to customers.
- **Don't try alone:** partnerships tend to be more effective with selective approach based on the use case;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200913201309487.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200913201425666.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200913201520549.png)

**level**: 
**author**: Kartikeya Bhardwaj, CA,USA; WeiChen(Carnegie Mellon University); Radu Marculescu(Texas at Austing)
**date**: 2020
**keyword**:

- Federated Learning; Data-independent model compression; communication-aware model compression

> Bhardwaj, Kartikeya, Wei Chen, and Radu Marculescu. "New Directions in Distributed Deep Learning: Bringing the Network at Forefront of IoT Design." *arXiv preprint arXiv:2008.10805* (2020).

------

## Paper: IoTDesign

<div align=center>
<br/>
<b>Invited: new Directions in Distributed Deep Learning: Bringing the network at Forefront of IOT Design</b>
</div>

#### Summary

- three challenges to large-scale adoption of deep learning at the edge:

> - **Hardware-constrained IoT devices**: memory-limited and run at low operating frequencies for high energy efficiency; 
> - **Data security and privacy:** operate locally on user devices;
> - **Network-aware deep learning:** 

- three major directions:

> - Federated learning for training deep networks;
> - Data-independent deployment of learning algorithms;
> - Communication-aware distributed inference;

#### Methods

- **Data privacy and network of devices in regards to training**

> - **statistical heterogeneity** : when data is not independent and identically distributed(non-iid) across users;
> - **systems heterogeneity**: devices that are training have different computational and communication capabilities;

- **Hardware constraints and data privacy in regards to inference**

> using data-independent model compression techniques[4] which not rely on the privacy datasets of third parties;

- **Hardware constraints and network of devices in regards to inference**

> communication aware model compression[3] which account for the hardware constraints and communication costs resulting from distributed inference;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831162216.png)

>  (a) Federated Learning is a new distributed training paradigm at the intersection of data privacy and network of devices, (b) Data-independent model compression aims to compress deep networks without using private datasets – This inference problem is at the intersection of hardware constraints and data privacy, (c) Communication-aware model compression is a new distributed inference paradigm at the intersection of hardware constraints and network of devices: Exploit the network of devices to collaboratively obtain low-latency, high-accuracy intelligence at the edge, (d) Prior model compression methods focus on the hardware constraints at device-level, but not the other two critical challenges: Key techniques include Pruning, Quantization, and Knowledge Distillation (KD).

【**Direction One**】**Federated Learning**(FedMax&&FedAvg)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831162519.png)

**【Direction two】Data-Independent Model Compression**

>  when deep learning models are trained on private datasets, the models deploy on edge devices cannot use the original datasets for model compression;

- **Dream Distillation**
  - Data-Free Knowledge Distillation: using metadata from a single layer makes the problem under-constrained, leading poor accuracy;
  - DeepDream[2]:
    - Dream generation: create custom objectives from metadata.
    - add small amount of Gaussian noise to the cluster-centroids along the principal component directions to create target activations;
    - using synthetic images generated before  to train the KD netwrok;

> In Knowledge Distillation, given a pretrained teacher model, a student model is trained using either the same dataset or using some unlabeled data.[4] For example, assume that neither alternate data, nor original training set is available, only amount of metadata is given:
>
> - use k-means algorithm to cluster real activations at the average-pool layer of the teacher network for just 10% of the real CIFAR-10 images, these cluster-centroids are used as metadata;
> - the centroids represent the average activations in a cluster, which is the mean activations, and the activation from real images are not used;
> - metadata also contains principle components for each cluster.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831163643.png)

**【Direction Three】Communication-aware model compression**

- SplitNet aims to split a deep learning model into disjoint subsets without considering the strict memory- and Flop-budgets for IoT devices;
- MoDNN[21] aims to reduce the number of FLOPS during distributed inference, but assume that the entire model can fit on each device, and does not consider any model compression;
- to improve the accuracy without increasing the communication latency,  such a communication-aware model compression, present **NoNN** network;

> a NoNN consists of a collection of multiple, disjoint student modules, which focus only on a part of teacher's knowledge, individual students are deployed on separate edge devices to collaboratively perform the distributed inference.
>
> - features for various classes are learned at different filters in CNNs, these activation patterns reveal how teacher's knowledge gets distributed at the final convolution layer, which can be used to create a filter activation network[3] that represent <font color=red>how the teacher's knowledge about various classes is organized into filters, and then partition this network via community detection; </font>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831172202.png)

>  (a) Prior KD [12]: Distributing large student models that do not ﬁt on a single memory-limited IoT device leads to heavy communication a teach layer. (b) NoNN [3] results in disjoint students that can ﬁt on individual devices: Filters at teacher’s ﬁnal convolution layer (representing knowledge about different classes) can be partitioned to train individual students which results in minimal communication until the ﬁnal layer (Figure adopted from [5]).

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831172425.png)

#### Notes <font color=orange>去加强了解</font>

  - [ ] data-independent model compression techniques[4]
  - [ ] network science[22],  [3]
  - [ ]  these activation patterns reveal how teacher's knowledge gets distributed at the final convolution layer  ???
  - [ ]  Dream Distillation: A Data Independent Model Compression Framework
  - [ ]  Knowledge distillation using unlabeled mismatched images
  - [ ]  FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efﬁcient Federated Learning
  - [ ] Federated learning: Challenges, methods, and future directions
  - [ ] Distilling the knowledge in a neural network.
  - [ ] Splitnet: Learning to semantically split deep networks for parameter reduction and model parallelization







## Paper《SAT-IoT: An Architectural Model for a High-Performance Fog/Edge/Cloud IoT Platform》

**keyword:**                                                     **author:**Madrid, Spain
miguelangel.lopez@satec.es

#### RelatedWork:

1. connectivity device managemnet data access and databases data processing and management of actions data analytics external interfaces including human machine interface (HMI)

#### Contribution:

1. the paradigm of edge/cloud computing transparency that lets the computation nodes change dynamically without administrator intervention
2. the IOT computing topology management that gives an IOT system global view form the hardware and communication infrastructures to the software deployed on them 
3. the automation and integration of IOT visualization systems for real time data visualization ,current IOT topology and current paths of data flows.

#### Chart&Analyse:

![1571877725494](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1571877725494.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1571877760207.png)

1. NVNet Framework

   ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1571878362198.png)



**level**:  CCF_A
**author**: Jozef Mocnej
**date**: 2018
**keyword**:

- Internet of Things ,decentralised architecture , Resource utilisation

------

## Paper: De-centralised IOT Architecture

<div align=center>
<br/>
<b>Decentralised IoT Architecture for Efficient Resources Utilisation</b>
</div>


#### Summary

1. <font color=red>identify the optimal set of features required by a generalised form of decentralised IoT platform </font>
2. <font color=red>describe a novel approach for efficient resource utilisation</font>
3. <font color=red>specify the structure of the overall architecture</font>

#### Proble Statement

- Growing number of devices connected to the Internet ,Diversity of the Internet of Things, the variety of IoT protocol stacks 

- current implementations of IoT platforms are based on centralised architectures 

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202124407187.png)

#### Methods

**Features:**

1. Multi-network approach
2. Scalable and interoperable implementation
3. Low power consumption
4. Intuitive data and device management			
5. Artificial Intelligence at the edge						

**system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202124802126.png)

- Monitoring the platform in runtime to abtain insight into the system and its processes
  - Quality of Device (battery life , precision , and sending rate...)
  - Quality of Service (bandwidth , packet loss to monitor the performance characteristics of the network)
  - Quality of Information (accuracy ,precision, and freshness)
- Ensuring the desired quality of the output by taking the appropriate measures; an example output
  - Value of Information (deals with an assessment of the information utility for a specific use case scenario using attributes such as relevance ,integrity ,timeliness,and understandability)
- Optimising the utilisation of available resources
  - Connectivity abstraction layer 
  - Devices services layer ( data filtering , communication service , power plan ,Over-The-Air(permits updates over the air to always keep a device’s software up-to-date))
- Custom application layer

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202124729502.png)



**Gateway Architecure**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202124820316.png)

**EndDevice Architecture:**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202124844922.png)



## Article Reading

### 多融合Slam

- landmark在实际应用中的复杂性（由于环境的复杂性引起）；
- 由于sensor的不同，也会存在一些其他的问题。在纹理少、四季天气变化、光照剧烈变化、车载条件IMU退化、长走廊、机器人剧烈运动等情况下
- 多传感器融合之后，传感器间数据如何同步？外参关系如何标定？
- 引入大量传感器之后，数据处理非常耗时，这与我们希望实现轻量级、快速响应且紧凑的SLAM系统相矛盾。而且传感器根据原理的不同，有些观测相互耦合，信息有一定的冗余，所以如何实现多个传感器之间的有效耦合
- 多特征基元的融合，在做线、面特征提取时，这是很工程化而且很依赖技巧，目前没有特别好的方法能够把线面的特征做到特别的通用、鲁棒。在另外一个层面，当我们进行参数化或者数据关联时，如果使用很差的原始输入，会产生很多误差和干扰。除此之外，图像特征和激光特征如何进行提取？几何特征之间如何实现紧耦合，这都是进行多源耦合会面临的问题
- 语义信息融合到传统或者经典SLAM框架里，甚至能够像人一样地对语义进行认知

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200414112018699.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200414112039578.png)

#### 多传感器

- Paper: LIC-Fusion
- 完成了传感器的标定工作，并且提出了环视鱼眼相机、轮速计陀螺仪融合方法，实现了基于环视相机和陀螺仪的融合估计。之后还完成了激光、相机、加速计和陀螺仪紧耦合的多传感器融合的里程计框架，叫LIC-Fusion。还完成了先验激光地图和相机、加速度计和陀螺仪所构成的定位系统。![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200414122817448.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202131128445.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20191202131200265.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/iotworkrelated/  

