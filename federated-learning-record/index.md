# Federated Learning Record


**level**: 
**author**: 	, Anit Kumar Sahu(Bosch Center for AI)
**date**: 2019

------

# Paper: Federated Learning

<div align=center>
<br/>
<b>Federated Learning: Challenges, Methods, and future Directions</b>
</div>

#### Summary

1. discuss the unique characteristics and challenges of federated learning, providing a broad overview of current approaches, and outline several directions of future work that are relevant to a wide range of research communities.

#### Research Objective

  - **Application Area**: learning sentiment, semantic location, activities of mobile phone users, adapting to pedestrian behavior in autonomous vehicles, predicting health events like heart attack risk from wearable devices; 
      - **Smart phones:** next-word prediction; voice recognition; while protect their personal privacy or to save the limited bandwidth/battery power of their phone;
      - **Organizations:** reduce strain on the network and enable private learning between divices/organizations;
      - **Internet of things:** Modern IoT networks, such as wearable devices, autonomous vehicles, or smart homes containing numerous sensors.

#### Proble Statement

- aim to learn the model under the constraint that device-generated data is stored and processed locally, with only intermediate updates being communicated periodically with a central server.
  - m: the total number of devices;
  - $p_k>0 $  and $\sum_kp_k=1$, and $F_k$ is the local objective function for the kth device;  
  - $p_k$: the relative impact of each device, like $p_k=1/n$ or $p_k=n_k/n$; 
  - $n=\sum_kn_k$:  is the total number of samples;

$$
min_wF(w), where F(w):=\sum_{k=1}^{m}p_kF_k(w)\\
F_k(w)=1/n_k\sum_{j_k=1}^{n_k}f_{j_k}(w;x_{j_k};y_{j_k})
$$

#### Core Problem :

- **Expensive Communications:** to develop communication-efficient methods that iteratively send small messages or model updates as part of the training process, as for reducing communication:
  - reducing the total number of communication rounds;
  - reducing the size of transmitted message at each round;
- **Systems Heterogeneity:**  the storage, computational and communication capabilities of each device in FL may differ due to variability in hardware, network connectivity, and power.
  - anticipate a low amount of participation;
  - tolerate heterogeneous hardware;
  - be robust to dropped devices in the network;
- **Statistical Heterogeneity:** devices frequently generate and collect data in a non-identically distributed manner across the netwok;
  - multi-task and meta-learning perspectives enable personalized or device-specific modeling;
- **Privacy Concerns:** tools like secure multiparty computation or differential privacy, which provide privacy at the cost of reduced model performance or system efficiency which should be understand and balance the trade-off;

#### Previous work:

**【Communication-efficiency】**

- **Local Updating methods**

> - Mini-batch optimization methods to process multiple data points at a time;
> - distributed  approaches like ADMM[4] in real-word data center environments;
>
> - variable number of local updates to be applied on each machine in parallel at each communication round,

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831230510.png)

- **Compression schemes**

> - model compression such as sparsification, subsampling, and quantization; [119], [135] for detail;
> - the low participation of devices, non-identically distributed local data, and local updating schemes pose noval challenges to these model compression approaches;
>   - errors accumulated locally may be stale if the devices are not frequently sampled;
>   - forcing the updating models to be sparse and low-rank;
>   - performing quantization with structured random rotations;
>   - using lossy compression and dropout to reduce server-device communication;
>   - applying Golomb lossless encoding;

- **Decentralized training**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200831231146.png)

**【Systems Heterogeneity】**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200901083436.png)

- **Asynchronous Communication**: 

> bounded-delay assumptions can be unrealistic in federated settings, where the delay may be on the order of hour to days, or completely unbounded.

- **Active Sampling**

> - activately selecting participating devices at each round;
>   - based on systems resources, with the aim being for the server to aggregate as many device updates within pre-defined time window.
>   - based on the data quality.
>   - <font color=red>how to extend these approaches to handle real-time, device-specific fluctuations in computation and communication delays</font>
>   - <font color=red>how to actively sampling a set of small but sufficiently representative devices based on the underlying statistical structure</font>

- **Fault Tolerance**

> - ignore such device failure, which may introduce bias;
> - code computation by introducing algorithmic redundancy:
>   - using gradient coding and its variants

**【Statistical Heterogeneity】**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200901090728.png)

> the data is not identically distributed across devices, both in terms of modeling the data and in analyzing the convergence behavior of associated training procedures.

- **Modeling Heterogeneous Data**  统计异质性

> using meta-learning and multi-task learning.
>
> MOCHA[106], an optimization framework designed for the federated setting, allow for personalization by learning separate but relate models for each device while leveraging a shared representation via multi-task learning;
>
> [26] models the star topology and perform variational inference;

- **Convergence Guarantees for Non-IID Data**

> FedProx makes a small modification to the FedAvg method to help ensure convergence, both theoretically and in practice. FedProx can also be interpreted as a generalized, reparameterized version of FedAvg that has practical ramifications in the context of accounting for systems heterogeneity across devices.

**【Privacy】**

> sharing other information such as model updates can also leak sensitive user information;

- **Privacy in Machine Learning**

> - differential privacy to communicate noisy data sketches;
> - homomorphic encryption to operate on encrypted data;
> - secure function evaluation or multi-party computation;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200901091216.png)

- **Privacy in FL**

> develop methods that are computationally cheap, commucation-efficient and tolerant to dropped devices, all without overly compromising accuracy.
>
> - SMC is a lossless method, retain the original accuracy with a very high privacy guarantee, incuring significant extra communication cost;
> - differential privacy: using hyperparameters that affect communication and accuracy that must be carefully chosen.

#### Future Directions

- **Extreme communication schemes:** 
- **Communication reduction and the Pareto frontier**
- **Novel medels of asynchrony**
- **Heterogeneity diagnostics**
- **Granular privacy constraints**
- **Beyond supervised learning**
- **Productionizing federated learning**
  - concept drift(the underlying data-generation model changes over time)
  - diurnal variations( the devices exhibit different behavior at different times of the day or week)
  - cold start problems(new devices enter the network)
- **Benchmarks**

#### Notes <font color=orange>去加强了解</font>

  - non-convex problems
  - [114]
  - differential privacy[32, 33, 34]
      - . A firm foundation for private data analysis
      - The algorithmic foundations of differential privacy
      - Calibrating noise to sensitivity in private data analysis



**level:**  cited by 750
**author**: H.Brendan McMahan  ,Google Inc

**date**: 2017,2,28
**keyword**:

- federal learning, distributed optimization and estimation, communication efficiency

------

## Paper: Communication-Efficient

<div align=center>
<br/>
<b>Communication-Efﬁcient Learning of DeepNetworks from Decentralized Data
</b>
</div>





#### Summary

1. <font color=red>introduce the FederatedAveraging algorithm, which combines local SGD on each client with a server that performs model averaging.</font>
2. learned the base concept of Federated learning.  learned the code but don't run it successfully.

#### Research Objective

  - **Application Area**: Modern mobile devices (phones,tablets) have access to a wealth of data suitable for learning models, which in turn improve the user experience on the devices.
    - <font color=red>Language models:</font> improve speech recognition and text entry on touch-screen keyboards by improving decoding,next-word-prediction,even the whole replies.
    - <font color=red>Image modes: </font> automatically select good photos
- **Purpose**:  advocate an alternative that leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates.

#### Proble Statement

**Opportunities:**

- the device we carried contained many <font color=red>sensors</font>, create much private data.
- the data driven many  <font color=red>intelligent applications</font> to improve usability.
- <font color=red>centrally store these data cost a lot</font>, eg: memory, communicating.
- decouple the model training from the need for direct access to the raw data,  <font color=red>reduce privacy and security risks by limiting the attack surface to only the device.</font>

**Previous work:**

- **Federated Learning:** 	
  - training on real-world data from mobile devices provides a distinct advantage over training on proxy data that is generally available in the data center.
  - data is privacy sensitive or large in size
  - <font color=red>for supervised tasks, labels on the data can be inferred naturally from user interaction.</font>
- **Privacy:**  the information transmitted for federated learning is the minimal update necessary to improve a particular model.   (有本书是关于数据隐私的，以及各种保护隐私算法)
- **Federated Optimization:** several key properties that differentiate it from typical distributed optimization problem.
  - <font color=red>Non-IID: </font>the data based on the usage of particular user will not presentative of the population distribution
  - <font color=red>Unbalanced: </font>the usage of device is different, leading to different amount of data.
  - <font color=red>Massively distributed:</font> the number of clients participating in an optimization is much larger than average number of examples per client.
  - <font color=red>Limited communication:</font> mobile devices are frequently offline or on slow or expensive connections.
- previous work don't consider unbalanced and non-IID data and the empirical evaluation is limited, we focus on the noe-IID and unbalanced properties of optimization, communication constraints.

#### Methods

- **system overview**:

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410084647545.png" height=250 width=100%>

【**Challenge** 1】<font color=red>client availability and unbalanced and non-IID data</font>

- assume a synchronous update scheme that proceeds in rounds of communication.

- propose Federated Averaging algorithm

  - <img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409171050998.png" height=350 width=80%>

  - Federated SGD &&Federated AVG

    <img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409184754940.png" height=280 width=100%>

  

  【**Challenge** 2】<font color=red>while in data center optimization, communication costs are relatively small, and computation costs dominate with GPU to low down, in federated optimization, communication costs dominate.</font>

  - the clients will only volunteer to participate in the optimization when they are charged, plugged-in, and on an unmetered wifi connection.
  - use additional computation to decrease the number of rounds of communication
    - increase parallelism
    - increase computation on each client: client perform more complex computation between each communication round.

  【**Challenge** 3】Non-Convex的问题，会得到任意坏的结果 **每一轮**的**各个客户端**的**起始参数值相同**（也就是**前一轮的全局参数值**）

  <img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409171145966.png" height=500 width=80%>

#### Evaluation

- Accuracy&&Communication rounds

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409174338562.png)

- local epoch test:

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409174644230.png)

- CIFAR experiments:

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409173229481.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409173127571.png)

- Large-scaleLSTMexperiments

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409173430010.png) 

#### Conclusion

- the identification of the problem of training on decentralized data from mobile devices as an important research direction
- the selection of straightforward and practical algorithm that can be applied to this setting
- an extensive empirical evaluation of the proposed approach.

#### Notes 

- [x] 传统的分布式优化问题的独立同分布假设
  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200409184415543.png)

**level**:  SysML Conference
**author**: Keith Bonawitz
**date**: 2019, 5, 22
**keyword**:

- Federated Learning, System Design

------

## Paper: FL System Design

<div align=center>
<br/>
<b>Towards Federated Learning at Scale：System Design</b>
</div>

#### Summary

1. 谷歌<font color=red>基于TensorFlow构建了全球首个产品级可扩展的大规模移动端联合学习系统，目前已在数千万台手机上运行</font>。这些手机能协同学习一个共享模型，所有的训练数据都留在设备端，确保了个人数据安全，手机端智能应用也能更快更低能耗更新。
2. **Federated Learning progress:**![](https://inews.gtimg.com/newsapp_bt/0/7635343934/1000)
3. ![](https://inews.gtimg.com/newsapp_bt/0/7635343935/1000)
4. **Server elements:**![](https://inews.gtimg.com/newsapp_bt/0/7635343937/1000)



**level**: 
**author**: AndrewHard,    GoogleLLC
**date**: 2019, 2,28
**keyword**:

- Federated Learning, language modeling, CIFG, NLP， next word prediction

------

## Paper: Mobile Key Prediction

<div align=center>
<br/>
<b>FEDERATED LEARNING FOR MOBILE KEY BOARD PREDICTION</b>
</div>





#### Summary

1. Gboard provides auto-correction, word completion, and next-word prediction features.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200327114728675.png)

- clients process their local data and share model updates with the server, weights from a large population of clients are aggregated by the server and combined to create an improved global model.
  - for every client: learnign rate $\epsilon $,the local client update $w_{t+1}^k=w_t-\epsilon g_k$ , $g_k$ is the average gradient.
  - for server: $w_{t+1}=\sum_{k=1}^K(n_k/N)w_{t+1}^k$
- updates are processed in memory and are immediately discarded after accumulation in a weight vector.

#### Notes <font color=orange>去加强了解</font>

  - [x] FSA是一个FSM(有限状态机)的一种，特性如下:
    - 确定：意味着指定任何一个状态，只可能最多有一个转移可以访问到。
    - 无环： 不可能重复遍历同一个状态
    - 接收机：有限状态机只“接受”特定的输入序列，并终止于final状态。
    - 查找这个key是否在集合内的时间复杂度，取决于key的长度，而不是集合的大小。
    - 构建上： TRIE只共享前缀，而FSA不仅共享前缀还共享后缀。
- [x] FST是也一个有限状态机（FSM）,具有这样的特性：
  - 确定：意味着指定任何一个状态，只可能最多有一个转移可以遍历到。
  - 无环： 不可能重复遍历同一个状态
  - transducer：接收特定的序列，终止于final状态，同时会**输出一个值**。
  - 不但能**共享前缀**还能**共享后缀**。不但能判断查找的key是否存在，还能给出响应的输入output
  - 构建： 在FSA上，多了转移上**放置和共享outputs**
- [ ] paper 2, “Finite-state transducers in language and speech processing
- [ ] 3 “Mobile keyboard input decoding with ﬁnite-state transducers
- [ ] CIFG for next-word prediction
- [ ] Character-aware neural language models

**level**: CCF_A     AAAI
**author**: Yang Liu, Anbu Huang, Yun Luo, He Huang, Youzhi Liu, Yuanyuan Chen
**date**: 2020, 1,17
**keyword**:

- Federated Learning, Object Detection

------

## Paper: Fed-Vision

<div align=center>
<br/>
<b>FedVision:An Online Visual Object Detection Platform Powered by Federated Learning </b>
</div>



#### Research Objective

  - **Application Area**: federated learning in Computer Vision
- **Purpose**:  use federated learning to solve the problem in computor vision.

#### Proble Statement

- Computer vision: privacy concerns, high cost of transmitting video data

- traditional typical workflow for centralized training of an object detector.

  - difficult to share data across organizations due to liability concerns

  - the whole process takes a long time and depends on when the next round of off-line training occurs, new data must wait for next round of training.

  - communication cost in transmitting data.

    <img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410170818514.png" width=80% height=250>

#### Methods

- **system overview**: HFL （horizontal federated learning)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410171127248.png)

**【Module 1】 Crowdsourced Image Annotation**

- use annotate the picture with { label, x, y, w, h}

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410171228763.png" width=60% height=250>

**【Module 2】Federated ModelTraining**

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410171556656.png" width=100% height=300>

- Configuration： configure training information, eg. number of iterations, reconnections, the server URL for uploading...
- Task Scheduler: global dispatch scheduling, coordinate communications between the server and clients to balance the utilization of local computational resources.
- Task Manager: coordinates the concurrent federated model training processes.
- Explorer: monitors the resource utilization situation on the client side, eg: CPU usage, memory usage, network load, etc. to inform the task scheduler on its load-balancing decisions.
- FL_SERVER: responsible for model parameter uploading, model aggregation, and model dispatch
- FL_CLIENT: hosts the task manager and explorer components and performs local model training.

**【Module 2】Federated Model Update**

- the number of model parameter files, and thus the storage size required, increases with the rounds of training operations

- Using Cloud Object Storage(COS) to store practically limitless amounts of data easily and at an affordable cost

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410172540671.png)

**【Module 3】Computer Vision Object Detect FedYOLOv3**

- in general two-stage approaches produce more accurate object detecion results(r-cnn), while one-stage approaches are more efficient,YOLOv3
- process of YOLOv3:
  - Predicting the positions of B bounding boxes<x,y,w,h>
  - Estimating the confidence score for the B predicted bounding boxes, whether a bounding box contains the target object $p(obj)=1$ meaning contain object else 0; how precise the boundary of box is. $\theta=p(obj)*IOU$, IOU meaning the intersection-over-union.
  - computing the class conditional probability,$p(c_i|obj)\epsilon[0,1]$ for each of the C classes.
  - The loss function:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410173637335.png)

**【Module 4】Model Compression**

- $M^{i,k}$ define the i-th user after the k-th iteration of FL training, $M_j^{i,k}$ define the j-th layer of $M^{i,k}$, $|\sum M_j^{i,k}|$  the sum of the absolute values of parameters in the j-th layer, the j-th layer to the overall model perfomance: $v(j)=|\sum M_j^{i,k}-\sum M_j^{i,k-1}|$, the larger the value of $v(j)$ , the greater the impact of layer j on the model, FL_CLIENT ranks the $v(j) values of ally layers in the model in descending order and selects only the parameters of  first n layers to upload for aggregation. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200410173826315.png)

#### Evaluation

  - Experiment company introduce:   
    - CRC has business interests in consumer products, health-care, energy services, urban construction and operation, technology and ﬁnance. <font color=red> detect multiple types of safety hazards via cameras in more than 100 factories</font>.
    - GRG Banking has more than 300,000 equipment deployed in over 80 countries. <font color=red>monitor suspicious transaction behaviours via cameras on the equipment</font>
    - SPIC the world's largest photovoltaic power generation company. <font color=red>monitor the safety of more than 10,000 square meters of photovoltaic panels</font>
- get good performance in Efficiency, data privacy, cost.

#### Conclusion

- report FedVision-- a machine learning engineering platform to support the development of federated learning powered computer vision applications.
- the first industrial application platform in computer vision-based tasks developed by webank and Extreme Vision to help customers develop computer vision based safety monitoring solutions in smart city applications.
- the platform help improve their operational efficiency and reduce their costs, while eliminating the need to transmit sensitive data around.

**level**: ICLR 2020
**author**: Hongyi Wang*(Wisconsin-Madison), Mikhail Yurochkin(MIT-IBM Watson AI lab), Yuekai Sun(Michigan University)
**date**: 2020
**keyword**:

- data privacy, federated learning

------

# Paper: FedMatchedAvg

<div align=center>
<br/>
<b>Federated Learning with Matched Averaging</b>
</div>

#### Summary

1. FedMA constructs the shared global model in a layer-wise manner by matching and averaging hidden elements(i.e. channels for convolutional layers; hidden states for LSTM; neurons for fully connected layers) with similar feature extraction signatures.
2. demostrate how PFNM can be applied to CNNs and LSTMs, but find it only gives very minor improvements over weight averging.
3. propose FedMA(Federated Matched Averaging) a new layer-wise federated learning algorithm for modern CNNs and LSTMs that appeal to Bayesian nonparametric methods to adapt to heterogeniety in the data.

#### Proble Statement

- coordinate-wise averaging of weights may have drastic detrimental effects on the performance of the averaged model and adds significantly to the communication burden.(<font color=red>permutation invariance of NN parameters</font>)

previous work:

- FedAvg(MCMahan et al.2017): the parameters of local models are averaged element-wise with weights proportional to sizes of the client datasets.
- FedProx(Sahu et al.) adds a proximal term to the client cost functions, limiting the impact of local updates by keeping them close to the global model.
- Agnostic Federated Learning(Mohri et al.): optimizes a centralized distribution that is a mixture of the client distributions.
- Probabilistic Federated Neural Matching(Yurochkin et al.2019): matching the neurons of client NNs before averaging them,using Bayesian non-parametric methods to adapt to global model size and to heterogeneity in the data. <font color=red> only work with simple architectures</font>

#### Methods

- **Problem Formulation**:  <font color=red>这里不懂，先记录着</font>

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705163648667.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705164501351.png)

- **system overview**:

> First, data center gathers only the weights of the first layers from the clients and performs one-layer matching described previously to obtain the first layer weights of the federeated model. Data center then broadcasts these weights to the clients, which proceed to train all consecutive layers on their datasets, keeping the matched federated layers frozen. And repeated up to the last layer for which we conduct a weighted averaging based on the class proportions of data points per client.
>
> FedMA with communication, where local clients receive the matched global model at the beginning of a new round and reconstruct their local models with the size equal to the original local models based on the matching results of the previous round.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705165047111.png)

#### Evaluation

  - **Environment**:   
    - Dataset: 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705165157271.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705165215850.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705165252982.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705165314632.png)

#### Conclusion

- <font color=red>present Federated Matched Averaging(FedMA), a layer-wise federated learning algorithm designed for modern CNNs and LSTMs architectures that accounts for permutation invariance of the neurons and permits global model size adaptation.</font>

#### Notes <font color=orange>去加强了解</font>

  - https://github.com/cheind/py-lapsolver

**level**: 
**author**: Wei Chen(Carnegie Mellon Univertity)  Radu Mar
**date**: 2020
**keyword**:

- Federated Learning; activation-divergence issue;  Maximum Entropy; Non-IID;

> Chen, Wei, Kartikeya Bhardwaj, and Radu Marculescu. "FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning." *arXiv preprint arXiv:2004.03657* (2020).

------

# Paper:  FedMax

<div align=center>
<br/>
<b>FedMAx: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning</b>
</div>

#### Summary

1. introduce a prior based on the principle of maximum entropy, which assumes minimal information about the per-device activation vectors and aims at making the activation vectors of same classes as similar as possible across multiple devices.
2. makes activation vector across multiple devices more similar(for same classes); improving the accuracy;
3. significantly reduces the number of total communication rounds needed to save energy when training on edge devices.

#### Proble Statement

- the activation vectors in FL can diverge, even if subsets of users share a few common classes with data residing on different devices.

previous work:

- **FedAvg:**  not designed to handle the statistical heterogeneity in federated settings, when data is not independent and identically distributed across diferent devices.
- **Data-sharing strategy:** distributes global data across the local devices, but obtaining this common global data is problematic in practice.
- **FedProx[4]:** targets the weight-divergence problem, the local-weights diverge from the global model due to non-IID data at local devices; by introducing a new loss function which constrains the local models to stay close to the global model.

#### Methods

- **Problem Formulation**:
  - $g_k(w_k)$: local objective which is typically the loss function of the prediction made with model parameters w;
  - m: the number of devices selected at any given communication round; m=C*M
  - C: the proportion of selected devices;
  - $\sum_{k=1}^Mp_k=1,p_k=n_k/n$,;
  - $n_k$: is the number of samples available at the device k;
  - $n=\sum_{k=1}^Mn_k$: total number of samples;

$$
min_w g(w)=\sum_{k=1}^m P_k*g_k(w_k)
$$

- **system overview**:

  - $L^2$ **Norm regularization:** reduce the activation-divergence across different devices by preventing the activation vectors from taking large values;

    - $F_k(w_k)$: the cross entropy loss on local data;
    - $\alpha_i^k$: the activation vectors at the input of the last fully-connected layer.

    $$
    min_wg_k(w_k)=F_k(w_k)+\beta ||a_i^k||_2\\
    $$

  - **Maximum Entropy Regularization:**  for we don't know which users have data from which classes, in other words, we don't have any prior information about how the activation vectors at different user should be distributed.

    - N: the min-batch size of local training data;
    - $H$: the entropy of activation vectors;
    - U: the uniform distribution over the activation vectors;
    - KL(.||.): the KL divergence;

    $$
    min_w g_k(w_k)=F_k(w_k)-\beta 1/N \sum_{i=1}^NH(\alpha_i^k)\\
    using Kullback-Leibler: min_w g_k(w_k)=F_k(w_k)+\beta 1/N \sum_{i=1}^N KL(\alpha_i^k||U)
    $$

    

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913184242778.png)

- $w^0$: the initial model and weights generated on a remote server;
- $E$: local epochs;
- $B$: the local training batch size;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913185949.png)

#### Evaluation

- **Effects of maximum entropy regularization with different distribution of synthetic data**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913190823.png)

- **Comparison of $L^2$-norm against Maximum Entropy**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913191102135.png)

- **Different Dataset Comparison**

![3000 rounds](https://gitee.com/github-25970295/blogImage/raw/master/img/20200913191304.png)

![600 round](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913191408146.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913191609259.png)

- **Medical data experiment-**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913191758843.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913191829604.png)

- **Mitigating Activation Divergence**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913192202686.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913192226909.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200913192251229.png)

#### Notes <font color=orange>去加强了解</font>

  - [ ] differential privacy [15]
  - [ ] KL divergence 

**level**: ICLR2020, CCF_A
**author**: Sean Augenstein(Google Inc), H.Brendan McMahan(Google Inc)
**date**: 2020
**keyword**:

- federated learning

------

# Paper: EffectiveML on Private

<div align=center>
<br/>
<b>Generative Models For Effective ML on Private, Decentralized DataSets</b>
</div>



#### Summary

1. demostrates that generative models-trained using federated methods and with formal differential privacy guarantees--can be used effectively to debug many commonly occurring data issues even when the data cannot be directly inspected.
2. Identifying key challenges in implementing end-to-end workflows with non-inspectable data, e.g. for debugging a "primary" ML model used in a mobile application.
3. Proposing a methodology that allows auxiliary generative models to resolve these challenges.
4. demonstrating how privacy preserving federated generative models-RNNs for text and GANs for images-can be trained to high enough fidelity to discover introduced data errors matching those encountered in real world scenarios.

#### Proble Statement

- challenges in model development and debugging.

previous work:

- manual data inspection is problematic for privacy-sensitive datasets:
  - identifying and fixing problems in the data
  - generating new modeling hypotheses
  - assigning or refining human-provided labels

#### Methods

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200705204644162.png)

- <font color=red>没有看懂</font>



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/federated-learning-record/  

