# AnomalyDetection


> 异常检测（Anomaly Detection 或 Outlier Detection）指的是通过数据挖掘手段识别数据中的“异常点”.

### 1.异常类型

#### .1.  单点异常

> 单点异常（Global Outliers）：也可以称为全局异常，即`某个点与全局大多数点都不一样`，那么这个点构成了单点异常。

#### .2. 上下文异常

> 多为时间序列数据中的异常，即某个时间点的表现与前后时间段内存在较大的差异，那么该异常为一个上下文异常点。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210520234405687.png)

#### .3. 集体异常

> 由多个对象组合构成的，即单独看某个个体可能并不存在异常，但这些个体同时出现，则构成了一种异常。集体异常可能存在多种组成方式，可能是由若干个单点组成的，也可能由几个序列组成。

### 2. 困境

- 数据本身是没有标签的，也存在一些数据集有标签，但标签的可信度非常低，导致放入模型后效果很差，这就导致我们无法直接使用一些成熟的有监督学习方法。
- 常常存在噪音和异常点混杂在一起的情况，难以区分。
- 在一些欺诈检测的场景中，多种诈骗数据都混在一起，很难区分不同类型的诈骗，因为我们也不了解每种诈骗的具体定义。

> 将无监督学习方法和专家经验相结合，基于无监督学习得到检测结果，并让领域专家基于检测结果给出反馈，以便于我们及时调整模型，反复进行迭代，最终得到一个越来越准确的模型。

### 3. 检测算法分类

#### .1. 时序相关&时序独立

> - 该场景的异常是否与时间维度相关。在时序相关问题中，我们假设异常的发生与时间的变化相关；
>
> - 时序独立问题中，我们假设时间的变化对异常是否发生是无影响的，在后续的分析建模中，也就不会代入时间维度。

#### .2. 全局检测&局部检测

> - 在全局检测方法中，`针对每个点进行检测时，是以其余全部点作为参考对象的`，其基本`假设是正常点的分布很集中，而异常点分布在离集中区域较远的地方`。这类方法的缺点是，在针对每个点进行检测时，其他的异常点也在参考集内，这可能会导致结果可能存在一定偏差。
> - 局部检测方法仅以部分点组成的子集作为参考对象，基于的假设是，正常点中可能存在多种不同的模式，与这些模式均不同的少数点为异常点。该类方法在使用过程中的缺点是，参考子集的选取比较困难。

##### 1. 基于时间序列分解

STL是一种单维度时间指标异常检测算法。大致思路是：

（1）先将指标做STL时序分解，得到seasonal、trend、residual成分。

（2）用ESD算法对trend+residual成分进行异常检测。

（3）为增强对异常点的鲁棒性，将ESD算法中的mean、std等统计量用median, MAD替换。

（4）异常分输出：abnorm_score = (value - median)/MAD，负分表示异常下跌，正分表示异常上升。

当然，还有其他的时间序列分解算法，例如STL、X12-ARIMA、STAMP等。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521081637515.png)

##### 2. 基于统计学模型预测

移动*均MA是一种分析时间序列的常用工具，它可过滤高频噪声和检测异常点。

- 根据计算方法的不同，常用的移动*均算法包括简单移动*均、加权移动*均、指数移动*均。
- 在序列取值随时间波动较小的场景中，上述移动均值与该时刻的真实值的差值超过一定阈值，则判定该时刻的值异常。

当然，还有ARMA、ARIMA、SARIMA等适用于时间序列分析的统计学模型，可以预测信号并指出其中的异常值。

##### 3. 基于同比和环比

适合数据呈周期性规律的场景中。例如：

- 监控APP的DAU的环比和同比，及时发现DAU上涨或者下跌
- 监控实时广告点击、消耗的环比和同比，及时发现变化

当上述比值超过一定阈值，则判定出现异常。

#### .3.**标签 VS 异常分数**

> 根据模型的输出形式，即直接输出标签还是异常分数。

- 区间判定： 给出一个阈值区间，实际观测落在区间之外则判定为异常。例如，在时间序列和回归分析中，预测值与真实值的残差序列便可构建这样一个区间。
- 二分判定：二分判定的前提是数据包含人工标注。异常值标注为 1，正常值标注为 0，通过机器学习方法给出观测为异常的概率。

#### .4. 模型特征

##### .1. 统计检验方法 

> 正常的数据是遵循特定分布形式的，并且占了很大比例，而异常点的位置和正常点相比存在比较大的偏移。**3 Sigma** 方法。

###### .1. 标准差分布， 方差分析，卡方检验等

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210520231802510.png)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
anomalies = []
normal = []
# 生成一些数据
data = np.random.randn(50000)  * 20 + 20

# 在一维数据集上检测离群点的函数
def find_anomalies(random_data):
    # 将上、下限设为3倍标准差
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3

    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    print("下限： ",lower_limit)
    print("上限： ",upper_limit)
    # 异常
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
        else:
            normal.append(outlier)
    return pd.DataFrame(anomalies,columns=["异常值"]),pd.DataFrame(normal,columns=["正常值"])

anomalies,normal = find_anomalies(data)
```

###### .2. 四分图检测

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
anomalies = []
normal = []
# 生成一些数据
data = np.random.randn(50000)  * 20 + 20

# 在一维数据集上检测离群点的函数
def find_anomalies(random_data):
    # 将上、下限设为3倍标准差
    iqr_25 = np.percentile(random_data, [25])
    iqr_75 = np.percentile(random_data, [75])

    lower_limit  = iqr_25 - 1.5 * (iqr_75 - iqr_25) 
    upper_limit = iqr_25 + 1.5 * (iqr_75 - iqr_25)
    print("下限： ",lower_limit)
    print("上限： ",upper_limit)
    # 异常
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
        else:
            normal.append(outlier)
    return pd.DataFrame(anomalies,columns=["异常值"]),pd.DataFrame(normal,columns=["正常值"])

anomalies,normal = find_anomalies(data)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210520233333173.png)

###### .3.  BOX-COX 转化

> 当原始数据的分布是有偏的，不满足正态分布时，可通过 BOX-COX 转化，在一定程度上修正分布的偏态。转换无需先验信息，但需要搜寻最优的参数λ。

##### .2.基于深度方法

> 从点空间的边缘定位异常点，按照不同程度的需求，决定层数及异常点的个数。如下图所示，圆中密密麻麻的黑点代表一个个数据点，基于的假设是点空间中心这些分布比较集中、密度较高的点都是正常点，而异常点都位于外层，即分布比较稀疏的地方。
>
> - 孤立森林算法

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210520231842604.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210520231859860.png)

##### .3.基于偏差方法

> 给定一个数据集后，对每个点进行检测，如果一个点自身的值与整个集合的指标存在过大的偏差，则该点为异常点。
>
> - 定义一个指标 SF（Smooth Factor），这个指标的含义就是当把某个点从集合剔除后方差所降低的差值，我们通过设定一个阈值，与这些偏差值进行比较来确定哪些点存在异常。

##### .4.基于距离&角度方法

> 计算每个点与周围点的距离，来判断一个点是不是存在异常。基于的假设是正常点的周围存在很多个近邻点，而异常点距离周围点的距离都比较远。
>
> - KNN， KMeans

##### .5.基于密度方法

> 针对所研究的点，计算它的周围密度和其临近点的周围密度，基于这两个密度值计算出相对密度，作为异常分数。即相对密度越大，异常程度越高。基于的假设是，正常点与其近邻点的密度是相近的，而异常点的密度和周围的点存在较大差异。

###### 1. LOF

首先对于每一个数据点，找出它的K个*邻，然后计算LOF得分，得分越高越可能是异常点。

LOF是一个比值，分子是K个*邻的*均局部可达密度，分母是该数据点的局部可达密度。

- 可达密度是一个比值，分子是K-*邻的个数，分母是K-*邻可达距离之和。
- A到B的可达距离定义：A和B的真实距离与B的k-*邻距离的最大值。

###### 2. COF

LOF中计算距离是用的欧式距离，也是默认了数据是球状分布，而COF的局部密度是根据最短路径方法求出的，也叫做链式距离。

###### 3. INFLO

LOF容易将边界处的点判断为异常，INFLO在计算密度时，利用k*邻点和反向*邻集合，改善了LOF的这个缺点。

###### 4. LoOP

将LOF中计算密度的公式加了*方根，并假设*邻距离的分布符合正态分布。

##### .6.  无监督学习

- IsolationForest
- DBSCAN
- Local Outlier Factor（LOF）

##### .8. 半监督学习

- Local Outlier Factor（LOF）
- One-Class SVM

##### .9. 基于深度学习方法

> 在仅有负样本（正常数据）或者少量正样本情况下：
>
> **训练阶段：**
>
> ​       可以通过网络仅仅学习负样本（正常数据）的数据分布，得到的模型G只能生成或者重建正常数据。
>
> **测试阶段：**
>
> ​       使用测试样本输入训练好的模型G，如果G经过重建后输出和输入一样或者接近，表明测试的是正常数据，否则是异常数据。
>
> **模型G的选择：**
>
> ​       一个重建能力或者学习数据分布能力较好的**生成模型**，例如GAN或者VAE，甚至encoder-decoder。

- ###### AE（AutoEncoder）

- VAE（Variational Autoencoder）

- GAN

##### .10. 基于行为序列

> 用户在搜索引擎上有5个行为状态：页面请求（P），搜索（S），自然搜索结果（W），广告点击（O），翻页（N）。状态之间有转移概率，由若干行为状态组成的一条链可以看做一条马尔科夫链。
>
> 统计正常行为序列中任意两个相邻的状态，然后计算每个状态转移到其他任意状态的概率，得状态转移矩阵。针对每一个待检测用户行为序列，易得该序列的概率值，概率值越大，越像正常用户行为。

##### .11. 基于图模型

### 4. 应用领域

#### .1. 视觉研究

> **Flip Library (LinkedAI)**：https://github.com/LinkedAi/flip
>
> Flip是一个python库，允许你从由背景和对象组成的一小组图像(可能位于背景中的图像)中在几步之内生成合成图像。它还允许你将结果保存为jpg、json、csv和pascal voc文件。

- 检测包含异常视频中的特定帧
- 检测异常事件
  - 基于轨迹
  - 基于时空方法： 基于光流特征

#### .2. 金融领域：

- 从金融数据中识别”欺诈案例“，如识别信用卡申请欺诈、虚假信贷等；

#### .3. 网络安全

> 从流量数据中找出”入侵者“，并识别新的网络入侵模式；

- 特征 1. 内存使用量
- 特征 2. 磁盘每秒访问次数
- 特征 3. CPU 负载
- 特征 4. 网络流量

#### .4. 电商领域

- 从交易数据中识别”恶意买家“，如羊毛党、恶意刷屏团伙；

#### .5. 生态灾难预警

- 基于对风速、降雨量、气温等指标的预测，判断未来可能出现的极端天气；

#### .6. 工业界

- 结构缺陷
- 设备故障
- 颜色&纹理， 划痕，错位，缺件，比例错误。

| 对比维度                               | 百度运维部         | 滴滴出行                   | Metis                                               | 阿里巴巴                       | 华为消费者BG               | 清华大学                 |
| :------------------------------------- | :----------------- | :------------------------- | :-------------------------------------------------- | :----------------------------- | :------------------------- | :----------------------- |
| 年份/地点                              | 2017年9月/北京     | 2017年9月/北京             | 2017年8月-至今                                      | 2018年9月/上海                 | 2018年9月/上海             | 2015年开始-至今          |
| 整体技术框架                           | 先分类，再检测     | 先分类，再检测             | 直接检测，并做分类特征                              | 先分类，再检测                 | 单条时间序列建模           | 先分类，再检测           |
| 机器学习模型（异常检测）               | 同环比模型阈值模型 | 同环比模型阈值模型趋势模型 | 无监督模型，有监督模型控制图理论，多项式拟合XGBoost | 同环比模型阈值模型Holt-Winters | 无监督模型Isolation Forest | 有监督模型 Random Forest |
| 深度学习（单条时间序列）               | 无                 | 无                         | 自编码器，VAE                                       | VAE                            | 无                         | VAE                      |
| 深度学习（海量时间序列）（端到端训练） | 无                 | 无                         | DNN，LSTM                                           | 无                             | 无                         | 无                       |
| 单调性/定时任务                        | 无                 | 无                         | 线性拟合/周期性识别算法                             | 无                             | 无                         | 无                       |
| 开源                                   | 只有打标工具       | 无                         | 打标工具无监督模型，有监督模型                      | 无                             | 无                         | 只有打标工具             |

#### .7. 个人健康

- 手环心率异常
- 老人摔倒
- 中分检测

#### .8. 制药领域

### 5. 开源工具

#### .1. 异常检测工具库**[Pyod](https://pyod.readthedocs.io/en/latest/)**

> 用于检测数据中异常值的库，它能对20多种不同的算法进行访问，以检测异常值，并能够与Python 2和Python 3兼容。
>
> - 包括近20种常见的异常检测算法，比如经典的LOF/LOCI/ABOD以及最新的深度学习如对抗生成模型(GAN)和集成异常检测(outlier ensemble);
> - 支持不同版本的Python：包括2.7和3.5+;支持多种操作系统：windows，macOS和Linux;
> - 简单易用且一致的API，只需要几行代码就可以完成异常检测，方便评估大量算法;
> - 使用即时编译器(JIT)和并行化(parallelization)进行优化，加速算法运行及扩展性(scalability)，可以处理大量数据;

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521082605123.png)

#### .2. [PyOdds](https://github.com/datamllab/pyodds)

> 一个端到端的异常检测Python组件。PyODDS提供异常检测算法，满足不同领域的用户需求，无论是数据科学还是机器学习背景。PyODDS提供了在数据库中执行机器学习算法的能力，而无需将数据移出数据库服务器。它还提供了大量基于统计和深度学习的异常检测算法。
>
> - 全栈服务，支持从轻量级的基于SQL的数据库到后端机器学习算法的操作和维护，使得吞吐量速度更快;
> - 先进的异常检测方法，包括统计、机器学习、深度学习模型与统一的API和详细的文档;
> - 强大的数据分析机制，支持静态和时间序列数据分析与灵活的时间片(滑动窗口)分割;
> - 自动化机器学习,首次尝试将自动机器学习与异常检测结合起来，并属于将自动机器学习概念扩展到现实世界数据挖掘任务的尝试之一。

#### .3. Arundo开源的[ADTK](https://github.com/arundo/adtk)

> 一个用于非监督、基于规则的时间序。列异常检测的Python包。这个软件包提供了一组具有统一通用检测器、转换器和聚合器的API，以及将它们连接到模型中的管道类。它还提供了一些处理和可视化时间序列和异常事件的功能。

#### .4. [LoudML](https://github.com/regel/loudml)

> Loud ML是一个建立在TensorFlow之上的开源时间序列推理引擎。该工具有助于预测数据、检测异常值，并使用先验的知识使异常检测过程自动化。
>
> - 内置HTTP API，方便与其他应用系统集成;
> - 可以通过机器学习引擎处理来自不同数据源的异常数据;
> - 支持ElasticSearch、InfluxDB、MongoDB、OpenTSDB等数据库;
> - 支持JSON配置安装和管理;
> - 近乎实时的数据处理，并提供给推理引擎以返回结果。

#### .5.**Linkedin开源的[luminol](https://github.com/linkedin/luminol)**

> Luminol是一个轻量级的时间序列数据分析python库。它支持的两个主要功能是异常检测和关联。它可以用来计算异常的可能原因。给定一个时间序列，检测数据是否包含任何异常，并返回异常发生的时间窗口、异常达到其严重程度的时间戳，以及指示该异常与时间序列中的其他异常相比有多严重的分数。给定两个时间序列，帮助求出它们的相关系数。
>
> 可以建立一个异常检测分析的逻辑流程。例如，假设网络延迟出现峰值：异常检测可以发现网络延迟时间序列中的峰值，并获取峰值的异常周期，之后与同一时间范围内的其他系统指标(如GC、IO、CPU等)关联获得相关指标的排序列表，根源候选项很可能位于最前面。

#### .6. **Twitter开源的[AnomalyDetection](https://github.com/twitter/AnomalyDetection)**

> 是一个R语言程序包，Twitter通常会在重大新闻和体育赛事期间用AnomalyDetection扫描入站流量，发现那些使用僵尸账号发送大量垃圾(营销)信息的机器人。

#### .7. Metis时间序列异常检测

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521111126834.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521111153239.png)

![率值](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521111202465.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521082850142.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521082909922.png)

### 6. 数据集

- [Alibaba/clusterdata](https://github.com/alibaba/clusterdata)
- [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)
- [Google/cluster-data](https://github.com/google/cluster-data)
- [The Numenta Anomaly Benchmark(NAB)](https://github.com/numenta/NAB)
- [Yahoo: A Labeled Anomaly Detection Dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)
- [港中文loghub数据集](https://github.com/logpai/loghub)
- [2018 AIOPS挑战赛预赛测试集](http://iops.ai/dataset_detail/?id=7) [2018 AIOPS挑战赛预赛训练集](http://iops.ai/dataset_detail/?id=6)
- Numenta's [NAB](https://github.com/numenta/NAB)

> NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications. It is comprised of over 50 labeled real-world and artificial timeseries data files plus a novel scoring mechanism designed for real-time applications.

- Yahoo's [Webscope S5](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)

> The dataset consists of real and synthetic time-series with tagged anomaly points. The dataset tests the detection accuracy of various anomaly-types including outliers and change-points.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521083113842.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521083129476.png)

### 资源

- https://zhuanlan.51cto.com/art/202101/641087.htm
- AIOps白皮书：https://www.rizhiyi.com/assets/docs/AIOps.pdf
- AIOps手册：https://github.com/chenryn/aiops-handbook
- awesome-AIOps: https://github.com/linjinjin123/awesome-AIOps
- Anomaly Detection Learning Resources https://github.com/yzhao062/anomaly-detection-resources
- awesome-TS-anomaly-detection https://github.com/rob-med/awesome-TS-anomaly-detection
- 擎创科技 夏洛克AIOps： https://max.book118.com/html/2019/0627/6122050055002042.shtm
- Gartner-Market-Guide-for-AIOps-Platforms : https://tekwurx.com/wp-content/uploads/2019/05/Gartner-Market-Guide-for-AIOps-Platforms-Nov-18.pdf https://www.gartner.com/doc/reprints?id=1-1XS12Z80&ct=191118&st=sb
- 知乎/博客：
  - [深度好文：腾讯运维的 AI 实践](https://mp.weixin.qq.com/s/1a45t6H-tP_la8lgvirfqw)
  - [腾讯云智能运维(AIOps)项目实践](https://cloud.tencent.com/developer/article/1538908?from=10680)
  - [时间序列异常检测——学习笔记](https://zhuanlan.zhihu.com/p/142320349)
  - [机器学习之：异常检测](https://zhuanlan.zhihu.com/p/25753926)
  - [基于时间序列的异常检测算法小结](https://blog.csdn.net/Jasminexjf/article/details/88527966)
  - [时间序列异常检测算法](https://juejin.im/post/5c19f4cb518825678a7bad4c)
  - [异常检测的N种方法，阿里工程师都盘出来了](http://www.shujuren.org/article/993.html)
  - [时间序列异常检测算法S-H-ESD](https://www.cnblogs.com/en-heng/p/9202654.html)
  - [基于时间序列的单指标异常检测_雅虎流量数据](http://bbs.learnfuture.com/topic/9566)
  - [阿里巴巴国际站之异常检测](https://www.alibabacloud.com/blog/alibaba-engineers-have-worked-out-loads-of-methods-to-detect-anomalies_595452)

-  ppt类：
  - [异常检测在苏宁的实践](https://www.slideshare.net/ssuserbefd12/ss-164777085?from_action=save)
  - [ClickHouse在新浪的最佳实践](https://www.slideshare.net/jackgao946/clickhousemeetup-clickhouse-best-practice-sina?qid=af52dd07-957b-4233-954f-4e639c8a07c3&v=&b=&from_search=1)
  - [AS深圳2018 《织云Metis时间序列异常检测全方位解析》](https://myslide.cn/slides/9775)

- 代码类：
  - [keras-anomaly-detection](https://github.com/chen0040/keras-anomaly-detection)
  - [Keras的LSTM多变量时间序列预测](https://zhuanlan.zhihu.com/p/28746221)
  - [基于机器学习算法的时间序列价格异常检测（附代码）](https://cloud.tencent.com/developer/article/1395760?from=10680)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/anomalydetection/  

