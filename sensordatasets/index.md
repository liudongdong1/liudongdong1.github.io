# SensorDatasets


## 1. 加速度

### 1.1. [UCI dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

> 来源于UCI（即UC Irvine，加州大学欧文分校）。数据由年龄在19-48岁之间的30位志愿者，`智能手机固定于他们的腰部，执行六项动作，即行走、上楼梯、下楼梯、坐、站立、躺下`，同时在手机中存储传感器（加速度传感器和陀螺仪）的三维（XYZ轴）数据。传感器的频率被设置为`50HZ`（即每秒50次记录）。对于所输出传感器的维度数据，进行噪声过滤（Noise Filter），`以2.56秒的固定窗口滑动，同时窗口之间包含50%的重叠`，即每个窗口的数据维度是128（2.56*50）维，根据不同的运动类别，将数据进行标注。传感器含有三类：身体（Body）的加速度传感器、整体（Total）的加速度传感器、陀螺仪。

- [下载链接](https://www.cis.fordham.edu/wisdm/dataset.php) https://archive.ics.uci.edu/ml/machine-learning-databases/00240/

- [识别代码](https://github.com/linw7/Activity-Recognition)

### 1.2. CoreMotion

- 功能： 
  - 通过测量三个轴的加速度大小来判断人体运动 
  - 通过测量设备周围地磁场的强度和方向来判断朝向
  - 通过测量三个轴的旋转速率来判断朝向 
  - 无须物理接触就判断附近物体的存在
- 主要局限性
  - 受重力干扰大，瞬时误差大
  - 误差大， 容易受其他磁场和金属物体影响。主要用于校正其他设备
  - 误差会累积，长时间读数的准确性差
  - 不通用，大多数只针对几种材质

- [特征提取](https://github.com/jindongwang/activityrecognition/tree/master/code)：
  - 滑动窗口： 窗口大小，滑动步长
  - 合成加速度： 常规采集的都是三个方向的加速度，在处理过程中，会用到三轴加速度合成一个加速度(为了减少计算性)
  - 时域特征： 均值，标准差，众数，MAX/MIN, Range，相关系数，信号幅值面积SMA,过零点个数，最大值与最小值之差，众数
  - 频域特征：  直流分量，幅度，功率谱密度PSD, 图形的均值、方差、标准差、斜度、峭度，幅度的均值、方差、标准差、斜度

## 2. 姿态求解

### 2.1. [AHRS](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201125155901178.png)

## 3.  相关ADR论文

| 题目 | 目的                                                         | 传感器                                                       | 算法                                                         | 会议                       | 年份                                                | 引用 |                                                              |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | --------------------------------------------------- | ---- | ------------------------------------------------------------ |
| 1    | Quantitative study of music listening behavior in a social and affective context | 推测用户听音乐的行为                                         | 加速度计、磁力计、方向感应计、旋转感应计、光感应器、距离感应器、位置感应 | SVM,Open Sensing Framework | ACM Transactions on Interactive Intelligent Systems | 2015 | Yang Y H, Liu J Y. Quantitative study of music listening behavior in a social and affective context[J]. Multimedia, IEEE Transactions on, 2013, 15(6): 1304-1315. |
| 2    | Predicting User Traits From a Snapshot of Apps Installed on a Smartphone | 通过手机上装的应用推测用户的特征（宗教、关系、语言。。。）   | 手机屏幕                                                     | SVM                        | Mobile Computing and Communications Review          | 2014 | Seneviratne S, Seneviratne A, Mohapatra P, et al. Predicting user traits from a snapshot of apps installed on a smartphone[J]. ACM SIGMOBILE Mobile Computing and Communications Review, 2014, 18(2): 1-8. |
| 3    | Understanding in-car smartphone usage pattern with an un-obfuscated observation | 挖掘用户在开车时使用手机的习惯                               | 无                                                           | 频繁模式挖掘               | CHI                                                 | 2014 | Oh C, Lee J. Understanding in-car smartphone usage pattern with an un-obfuscated observation[C]//CHI'14 Extended Abstracts on Human Factors in Computing Systems. ACM, 2014: 1795-1800. |
| 4    | Preference, context and communities: a multi-faceted approach to predicting smartphone app usage patterns | 通过挖掘手机的使用特征看用户使用手机的习惯                   | 手机本身                                                     | NNC                        | ISWC                                                | 2013 | Xu Y, Lin M, Lu H, et al. Preference, context and communities: a multi-faceted approach to predicting smartphone app usage patterns[C]//Proceedings of the 2013 International Symposium on Wearable Computers. ACM, 2013: 69-76. |
| 5    | Driving Behavior Analysis for Smartphone-based Insurance Telematics | 通过手机来判断车内的用户行为                                 | 加速度计、陀螺仪                                             | 无                         | WPA                                                 | 2015 | Wahlstr?m J, Skog I, H?ndel P. Driving Behavior Analysis for Smartphone-based Insurance Telematics[C]//Proceedings of the 2nd workshop on Workshop on Physical Analytics. ACM, 2015: 19-24. |
| 6    | My Smartphone Knows I am Hungry                              | 在手机上装studentlife应用来收集用户的信息，通过推测的用户购买行为和位置行为为推测用户什么时候吃饭 | 加速度计、距离感应、光感应、麦克风、位置、应用使用情况       | 决策树                     | WPA                                                 | 2014 | Chen F, Wang R, Zhou X, et al. My smartphone knows i am hungry[C]//Proceedings of the 2014 workshop on physical analytics. ACM, 2014: 9-14. |
| 7    | MoodScope: Building a Mood Sensor from smartphone usage patterns | 通过统计应用使用情况分析用户使用手机的情绪                   | 应用使用情况                                                 | 线性回归                   | MobiSys                                             | 2013 | LiKamWa R, Liu Y, Lane N D, et al. Moodscope: Building a mood sensor from smartphone usage patterns[C]//Proceeding of the 11th annual international conference on Mobile systems, applications, and services. ACM, 2013: 389-402. |
| 8    | A smartphone based method to enhance road pavement anomaly detection by analyzing the driver behavior | 用手机的传感器识别车内用户的行为来检测行车的异常             | 加速度计                                                     | na                         | ISWC                                                | 2015 | SERAJ F, ZHANG K, TURKES O, MERATNIA N, HAVINGA P J M，. A smartphone based method to enhance road pavement anomaly detection by analyzing the driver behavior[C]//ACM Press, 2015: 1169�1177. |
| 9    | A smartphone-based sensing platform to model aggressive driving behaviors | 用手机的传感器侦测有过激行为的开车行为                       | 加速度计                                                     | 朴素贝叶斯                 | CHI                                                 | 2014 | Jin-Hyuk Hong, Ben Margines, and Anind K. Dey. 2014. A smartphone-based sensing platform to model aggressive driving behaviors. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '14). ACM, New York, NY, USA, 4047-4056 |
| 10   | DOWELL: Dwell-time Based Smartphone Control Solution for People with Upper Limb Disabilities | 智能手机界面辅助上肢残疾的人进行界面操作                     | 无                                                           | 无                         | CHI                                                 | 2015 | Ahn H, Yoon J, Chung G, et al. DOWELL: Dwell-time Based Smartphone Control Solution for People with Upper Limb Disabilities[C]//Proceedings of the 33rd Annual ACM Conference Extended Abstracts on Human Factors in Computing Systems. ACM, 2015: 887-892. |
| 11   | Estimating heart rate variation during walking with smartphone | 通过手机加速度计来判断心跳                                   | 加速度计                                                     | 神经网络                   | Ubicomp                                             | 2013 | Sumida M, Mizumoto T, Yasumoto K. Estimating heart rate variation during walking with smartphone[C]//Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing. ACM, 2013: 245-254. |
| 12   | Evaluating tooth brushing performance with smartphone sound data | 用手机中的麦克风录音来分析刷牙的动作标准程度并给出预测       | 麦克风                                                       | HMM                        | Ubicomp                                             | 2015 | KORPELA J, MIYAJI R, MAEKAWA T, NOZAKI K, TAMAGAWA H，. Evaluating tooth brushing performance with smartphone sound data[C]//ACM Press, 2015: 109�120. |
| 13   | Oh app, where art thou?: on app launching habits of smartphone users | 分析启动手机应用的习惯                                       | 无                                                           | 无                         | MobileCHI                                           | 2013 | HANG A, DE LUCA A, HARTMANN J, HUSSMANN H，. Oh app, where art thou?: on app launching habits of smartphone users[C]//ACM Press, 2013: 392. |
| 14   | Smartphone-based monitoring system for activities of daily living for elderly people and their relatives etc. | 通过手机记录老人的行动log反馈给亲属                          | 加速度计、GPS、麦克风                                        | 无                         | Ubicomp                                             | 2013 | OUCHI K, DOI M，. Smartphone-based monitoring system for activities of daily living for elderly people and their relatives etc.[C]//ACM Press, 2013: 103�106. |

### 3.2. 位置相关识别

| 序号 | 方法                                                         | 传感器          | 位置                                                         | 方法与数据                  | 引文                                                         |
| ---- | ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| 1    | 用6个eWatch放在不同的6个位置，只是用了不同位置的传感器来组合识别行为，并没有迁移，效果很好 | 2轴加速度、光感 | 左手，要带，脖子，右裤口袋，上衣口袋，包                     | 6个人，6种行为，KNN等4种    | Maurer U, Smailagic A, Siewiorek D P, et al. Activity recognition and monitoring using multiple sensors on different body positions[C]//International Workshop on Wearable and Implantable Body Sensor Networks (BSN'06). IEEE, 2006: 4 pp.-116. |
| 2    | 通过手机放在口袋里不同位置和不同朝向来识别行为，用的是SVM。没有进行迁移，只是用了不同位置单独的数据，以及组合。 | 加速度          | 口袋的6个位置                                                | SVM，7个人，7种行为         | Sun L, Zhang D, Li B, et al. Activity recognition on an accelerometer embedded mobile phone with varying positions and orientations [C]//International Conference on Ubiquitous Intelligence and Computing. Springer Berlin Heidelberg, 2010: 548-562. |
| 3    | 不同的传感器位置对行为识别影响很大，文章设计了基于LDA的方法，对位置鲁棒。没有涉及到位置迁移识别。 | 加速度          | 口袋的5个位置                                                | LDA，7种行为                | Khan A M, Lee Y K, Lee S, et al. Accelerometer’s position independent physical activity recognition system for long-term activity monitoring in the elderly[J]. Medical & biological engineering & computing, 2010, 48(12): 1271-1279. |
| 4    | 加速度                                                       |                 | 有研究检测最优的行为检测位置；目前的研究倾向于研究最优的放置位置 |                             | Lara O D, Labrador M A. A survey on human activity recognition using wearable sensors[J]. IEEE Communications Surveys & Tutorials, 2013, 15(3): 1192-1209. |
| 5    | 稳重指出目前尚未有研究来说明传感器的最优位置对行为识别的影响。文章主要是在不同的位置放了传感器，来看对高层行为、低层行为、过度行为等的识别率，从中分析出哪些地方可能对哪些行为有特定的精度，不是迁移。 | 加速度          |                                                              | 15种行为，7个人             | Atallah L, Lo B, King R, et al. Sensor positioning for activity recognition using wearable accelerometers[J]. IEEE transactions on biomedical circuits and systems, 2011, 5(4): 320-329. |
| 6    | recognizes actions by measuring the circumference of body parts。用了柔性材料的衣服，布满传感器，通过测量衣服的形变与肢体的圆周关系来识别行为。 | 加速度          | Wrist，waist，ankle                                          | 识别率很高                  | Tsubaki K, Terada T, Tsukamoto M. An Activity Recognition Method by Measuring Circumference of Body Parts[C]//Proceedings of the 7th Augmented Human International Conference 2016. ACM, 2016: 13. |
| 7    | 识别传感器在身体的不同位置，同时提出了位置无关的行为识别算法。没有涉及到迁移。 | 加速度          | 7个：head, chest, upper arm, waist, forearm, thigh, and shin | 8种行为，15个人，数据集公开 | Sztyler T, Stuckenschmidt H. On-body localization of wearable devices: An investigation of position-aware activity recognition[C]//2016 IEEE International Conference on Pervasive Computing and Communications (PerCom). IEEE, 2016: 1-9. |
| 8    | 先识别不同位置，然后再进行行为识别，精度还可以.对行为进行分组，根据相似性进行分组 | 加速度          | 4个：hand, coat pocket, trouser pocket and the rear pocket   | 几种日常行为                | Guo Q, Liu B, Chen C W. A two-layer and multi-strategy framework for human activity recognition using smartphone[C]//2016 IEEE International Conference on Communications (ICC). IEEE, 2016: 1-6. |
| 9    | 通过实验看不同的位置对行为识别的影响度                       | 加速度、陀螺仪  | 身体和躯干的几种不同位置                                     |                             | Kunze K, Lukowicz P. Sensor placement variations in wearable activity recognition[J]. IEEE Pervasive Computing, 2014, 13(4): 32-41. |

### 3.3. 迁移

| 编号 | 迁移方法                                                     | 数据集                                                       | 对比方法                                                     | 实验方法                                                     | 文章                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Adaptive Multiple Kernel Learning (A-MKL)加上SVM。用pyramid match算法，首先将两条视频的距离降到最短，然后用若干个kernel SVM去学习 | (1) Kodak Consumer Video Benchmark Data(2) 从youtube上下载的部分web video | 1.Adaptive-SVM，2.domain transfer SVM，3.multiple kernel learning | 从第一部分video向第二部分迁移                                | Visual event recognition in videos by learning from web data |
| 2    | Importance weighted least-squares probabilistic classifier (IWLSPC)。一中基于adaptive采样的概率方法。Instance transfer【有源代码】 | Alkan加速度数据集：由ipod touch收集，有手中的有口袋里的。计算时提取了与位置无关的特征（均值、方差等5个）. | LapRLS+CV LapRLS+IWCV KLR+CV IWKLR+IWCV LSPC+CV IWLSPC+IWCV  | 2000个labeled数据，800个无label数据                          | Importance weighted least-squares probabilistic classifier for covariate shiftAdaptation with application to human activity recognition |
| 3    | Cost sensitive的boosting方法。目标是，给定同分部和不同分布的样本，预测同分布的部分的精确度。 | 2轴加速度计，在实验室和家里分别收集5种手势。                 | AdaBoost，TrAdaBoost                                         | 把数据分成两个环境都有的几部分，然后根据现有的两种环境数据去预测其他的 | Cost-sensitive Boosting for Concept Drift                    |
| 4    | TrAdaBoost：减少对不同分布数据的权重。解决问题：少量有label数据，分为同分布和不同分布的部分，去预测一个无label数据 | 新闻数据集3个                                                | TSVM。SVM                                                    | 不同的3个数据集之间迁移                                      | Boosting for Transfer Learning                               |
| 5    | 用可调整权重的SVM                                            | Youtube的视频数据                                            | 用户评价参与度                                               | 分成两部分进行迁移，正常迁移                                 | Interactive Event Search Through Transfer Learning           |
| 6    | 用两个不同domain的label信息的相似度去获取两个domain样本的相似度。Label信息相似度由web search获取。然后用一个加权SVM去做。 | 1. Amsterdam数据集（1个人生活，14个状态传感器）2. MIT PLIA13. Intel | 没有方法对比，仅多做了MMD和余弦相似度的对比                  | 每一个数据集中，一部分label迁移到另一部分label               | Cross-Domain Activity Recognition                            |
| 7    | 用基于HMM的迁移学习模型去做迁移。将两个房子的传感器进行映射，然后用EM算法去学习HMM的参数。 | 2个房子的数据                                                | 没有方法对比                                                 | 一个迁移到另一个                                             | Recognizing Activities in Multiple Contexts using Transfer Learning |
| 8    | 可以对特征空间、特征分布、label空间的不同做迁移。用概率的方法，把问题分成两个部分。 | 1.MIT数据集，2。1个人房子数据                                | 对比了不同参数下的精度                                       | 一个迁移到另一个                                             | Transfer Learning for Activity Recognition via Sensor Mapping |
| 9    | 用二部图的匹配进行迁移，挖掘图像的高层特征，这些特征可以被共享。 | 图像数据IXMAS多视角数据                                      | 不同的其他三种cross view方法                                 | 一个视角迁移到另一个                                         | Cross-View Action Recognition via View Knowledge Transfer    |
| 10   | 针对特征分布不一样的问题，用特征迁移，不需要label，把两部分映射到一个重构希尔伯特空间中最小化两都之间的距离 | Wifi定位                                                     | KPCA、KMM                                                    | 不同的设置相互迁移                                           | Domain adaptation via transfer component analysis            |
| 11   | 用ISOMAP，将source和target降维到同样的空间，然后选择置信度最高的标签进行 | SEMG数据                                                     | KE、TCA、LWE                                                 | SEMG数据的迁移                                               | Topology Preserving Domain Adaptation for Addressing Subject Based Variability in SEMG Signal |
| 12   | 用了层次化的复杂行为感知。先感知低层次的行为，做准确识别，然后将这些低层次行为进行组合，识别高层的行为。与HMM结合。 | 1.BookShelf数据，人身上安装3个传感器进行安装书架，2.Mirror数据 | 在bookshelf中识别简单子行为，在mirror数据进行迁移复杂行为    | 没有对比                                                     | Remember and Transfer what you have Learned �Recognizing Composite Activities based on Activity Spotting |
| 13   | 用HMM算法来做迁移。迁移的是meta-feature                      | 3个房间的生活数据                                            | 不同的房间相互迁移                                           | Meta-feature和sensor-feature的对比、迁移与不迁移的对比       | Transferring Knowledge of Activity Recognition across Sensor Networks |
| 14   | 把行为建模成传感器、时间、空间模型，然后进行source和target中传感器的映射 | 3个房间的数据                                                | 不同的房间相互迁移                                           | 不同数量的target data标记的对比                              | Activity Recognition Based on Home to Home Transfer Learning |
| 15   | 多类SVM进行迁移                                              | 几个房间不同人动作的数据视频                                 | 不同camera相互迁移                                           | 不同增量数量的对比                                           | Transferring Activities: Updating Human Behavior Analysis    |
| 16   | 第1步：用label数据训练一个模型,第2步：用这个模型去分类unlabeled数据，第3步：用这些数据反过来调整模型a，使得其适应unlabeled数据，形成模型b，对B进行下采样，用A进行预测，有了标签之后进行聚类，就有了label | 不同手机的数据                                               | 不同采样率                                                   | 不同采样率                                                   | Cross-mobile ELM based Activity Recognition                  |
| 17   | 用了一个决策树先对第一个人训练一个模型，然后识别第二个人，进行聚类 | 10个人用同样的手机                                           | SVM、NB                                                      | 不同人之间                                                   | Cross-People Mobile-Phone Based Activity Recognition         |
| 18   | 针对源和目标都无label的情况，利用彼此之间的知识训练3个聚类算法，精度很不错 | 图像数据                                                     | Co-clustering                                                | 不同图像                                                     | Self-taught clustering                                       |
| 19   | 从无label数据中自学习                                        | 图像、文本等                                                 | PCA                                                          | 不同域之间                                                   | Self-taught Learning: Transfer Learning from Unlabeled Data  |
| 20   | 无监督的迁移降维方法                                         | 人脸识别                                                     | LWF，PCA，LPP，DisKmeans                                     | 不同人脸之间                                                 | Transferred Dimensionality Reduction                         |
| 21   | 用markov logic进行迁移，属于关系之间的迁移                   | 蛋白质和社交网络                                             | 几种不同的参数设置                                           | 不同数据集之间                                               | Deep Transfer via Second-Order Markov Logic                  |
| 22   | 用构造方法对新来传感器迁移已经学习到的模型                   | 自己采集的动作数据                                           | KNN、SVM                                                     | 新来的传感器                                                 | Automatic transfer of activity recognition capabilities between body-worn motion sensors: training newcomers to recognize locomotion |
| 23   | 用GMM对数据进行建模，然后进行GMM参数的迁移                   | 图像数据集                                                   | 一些已有的方法                                               | 不同数据集之间                                               | Cross-Dataset Action Detection                               |
| 24   | 用EM和CRF做迁移                                              | 生理数据辅助进行行为识别                                     | 一些已有的基于CRF的方法                                      | 不同数据之间                                                 | Activity Recognition from Physiological Data using Conditional Random Fields |
| 25   | 用的RBM，不同的特征空间进行迁移                              | 行为数据和文本数据                                           | SCL                                                          | 文本数据辅助行为识别                                         | Heterogeneous Transfer Learning with RBMs                    |
| 26   | 用了NB和SVM混合，对new user有比较好的预测精度。              | 自己收集的28个人数据                                         | NB、SVM                                                      | 新来人的行为预测                                             | Hong J H, Ramos J, Dey A K. Toward Personalized Activity Recognition Systems With a Semipopulation Approach[J]. IEEE Transactions on Human-Machine Systems, 2016, 46(1): 101-112. |
| 27   | 深度迁移学习方面的第一篇文章                                 | OPP、Skoda                                                   | 无                                                           | `不同用户、不同设备、不同位置等`行为预测                     | Morales F J O, Roggen D. Deep convolutional feature transfer across mobile activity recognition domains, sensor modalities and locations[C]//Proceedings of the 2016 ACM International Symposium on Wearable Computers. ACM, 2016: 92-99. |

### 3.4. others

1. [A Class Incremental Extreme Learning Machine for Activity Recognition](https://mega.nz/#!hegygZwL!l25twe-8Krjl-6QxAKtpMxiInO4issQhtuvyeZguQA0) 提出一种类增量ELM算法，算法可以学习新出现的行为类别。
2. [A Framework for Wireless Sensor Network Based Mobile Mashup Applications](https://mega.nz/#!cXAjEAyY!Pcu1oqClxnPnP8qsT6Y-8EOsgfR3Y-RwcUMxXQCOX6M) 提出一种框架，包含无线传感器、智能手机作为网关以及服务器三者组成的通用的采集提取传感器数据的框架。
3. [A MOBILE DEVICE ORIENTED FRAMEWORK FOR CONTEXT INFORMATION MANAGEMENT](https://mega.nz/#!5eZwBLLC!5qenQvSzf7u7-_-N9p8WWaeZlER1pIKKYz0lt1QIUM4) 提出一种智能手机为中心的管理传感器网络上下文的框架
4. [A Nonintrusive and Single-Point Infrastructure-Mediated Sensing Approach for Water-Use Activity Recognition](https://mega.nz/#!AbYnCRQY!20tk8gbG1zbl1vRSOiSJVKJswAZscSFZ2AziEEE8hEU) 通过在水管表面加装三轴加速度计来检测用户的用水行为（Bathing, Flushing toilet, Cooking and Washing）
5. [PPCare: A Personal and Pervasive Health Care System for the Elderly](https://mega.nz/#!MP5F1AwC!n7N0YZvULU5x1CMn4YnbsfWRev-KctpGJFULJEc_qlQ) 提出PPCare手机软件，检测老人行为，包括四大块：`运动，卡路里消耗，跌倒检测，身体指数追踪`
6. [Wearable Accelerometer Based Extendable Activity Recognition System](https://mega.nz/#!BDIxxaKD!jHsw_3Zg7sjGtECRBcjNlJIuNnvHJvmDcvmYtsKsuBI) 基于`加速度计的穿戴设备进行行为识别，能识别未知的行为`.
7. [b-COELM: A fast, lightweight and accurate activity recognition model for mini-wearable devices](https://mega.nz/#!EXID1YhB!ow8E1NRMSnGi53aUWauELhwhS2JWY3DUqRdG3WzAN58) 提出一种`基于ELM的算法来用于mini穿戴设备的行为识别`，解决现有`设备运算量小`的问题。

## 4.  [HAR datasets](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset%20description.md)

- 1.Opportunity

  - [1.1网址与下载](https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition)
  - 描述
    - 数据集包括4个用户的6种大类（track）的将近100种行为： 这6个track及其行为是： Unique index - Track name - Label name;
    - 一共使用了3大类传感器：惯性、物体传感器、环境传感器。
    - 数据分为24个子文件，每个用户有6个文件，分别以S-0X开头。最后一个文件是drill，表示是用户按照预先要求的动作序列做出相应的动作。
  - [1.3引用此数据集的文章](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset description.md#13引用此数据集的文章)

- 2.UCI daily and sports dataset

  - [2.1网址与下载](http://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)

  - 2.2描述

    （1）运动人数：8人(4 female, 4 male, between the ages 20 and 30)

    （2）运动时间：5分钟/人

    （3）行为类别：19种日常行为，分别用1-19这19个数据来表示

    （4）数据尺寸：每人记录5分钟的总运动时长

    （5）采样频率：25Hz

    （6）传感器：三轴加速度计、三轴陀螺仪、三轴磁力计

  - [2.3引用此数据集的文章](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset description.md#23引用此数据集的文章)

- 3.Activity recognition from single chest-mounted accelerometer data set

  - [3.3引用此数据集的文章](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset description.md#33引用此数据集的文章)

- 4.Gas sensors for home activity monitoring Data Set

  - [4.1网址与下载](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset description.md#41网址与下载)
  - [4.2描述](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset description.md#42描述)
  - [4.3引用此数据集的文章](https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset description.md#43引用此数据集的文章)

- 5.Human activity recognition using smartphones data set

  - [5.1网址与下载](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

> 采样：30个人，年龄在19-48之间
>
> 活动类型：WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
>
> 传感器：加速度传感器、陀螺仪
>
> 采样频率：50HZ
>
> 数据集：70%训练样本，30%测试样本
>
> 预处理：噪声滤波，滑动窗口（2.56sec，50%overlap），其中加速度传感器信号使用巴特沃斯低通滤波器（Butterworth low-pass filter）分离成身体加速度和重力。由于重力只有低频率成分,因此滤波器使用0.3HZ截止频率。在每个窗口,一个向量的特性是通过计算变量的时间和频率域得到的，共561个特征。
>
> 更新的数据集：Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set，在这个数据集中作者在已有的六个活动基础上增加了stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie, and lie-to-stand.这些状态改变的样本，并且此数据集的数据为原始数据，而不是预处理后的数据。

- 6.Heterogeneity activity recognition data set
  - [6.1网址与下载](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition)

> 样本数量：43930257
>
> 属性：16 Activities: ‘Biking’, ‘Sitting’, ‘Standing’, ‘Walking’, ‘Stair Up’ and ‘Stair down’.
>
> Sensors: Sensors: Two embedded sensors, i.e., Accelerometer and Gyroscope, sampled at the highest frequency the respective device allows.
>
> Devices: 4 smartwatches (2 LG watches, 2 Samsung Galaxy Gears)
>
> 8 smartphones (2 Samsung Galaxy S3 mini, 2 Samsung Galaxy S3, 2 LG Nexus 4, 2 Samsung Galaxy S+)
>
> Recordings: 9 users
>
> Activity recognition exp.zip包含了不同设备、不同传感器的行为数据
>
> Still exp.zip增加了手机放置的位置，不同位置下包含了多种设备所采集的行为数据

- 7.chest-mounted accelerometer dataset

  - 加速度数据未校准,52hz, 7  labels;

  - 1: Working at Computer

    2: Standing Up, Walking and Going updown stairs

    3: Standing

    4: Walking

    5: Going UpDown Stairs

    6: Walking and Talking with Someone

    7: Talking while Standing

- 8.[https://catalogue.data.govt.nz/dataset/fall-data](https://catalogue.data.govt.nz/dataset/fall-data)

## 5. 资源链接

- https://github.com/jindongwang/activityrecognition/blob/master/notes/dataset%20description.md
- https://catalogue.data.govt.nz/dataset/fall-data

### 

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/sensordatasets/  

