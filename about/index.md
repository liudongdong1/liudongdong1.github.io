# 

<!-- The (first) h1 will be used as the <title> of the HTML page -->

## 刘冬冬

<!-- The unordered list immediately after the h1 will be formatted on a single
line. It is intended to be used for contact details -->
- <3463264078@qq.com> | 17843124735 | [CSDN](https://blog.csdn.net/liudongdong19?type=blog) | [GitHub](https://github.com/liudongdong1) | [摄影](https://tuchong.com/17242296/)

<!-- The paragraph after the h1 and ul and before the first h2 is optional. It
is intended to be used for a short summary. -->


##  专业技能

- **技能：** 熟悉 Java ，JVM 基础， Mysql（JDBC, JPA框架使用），有C++/C#/python编程基础
- **技能：** 熟悉 Mysql 数据库包括索引、事务、redolog/undolog/binlog机制
- **技能：** 熟悉 SpringBoot、Flask 框架，以及 SpringJPA、Mybatis 使用
- **技能：** 熟悉 Linux 常用命令、存储&缓存结构、NUMA架构、文件系统栈，Maven、Gradle、CMake、GDB、Shell、Git，有 CMakeLists.txt 和 Shell脚本编写经验，DIY过自美魔镜，NAS存储阵列，无人车
- **技能：** 了解 Kubernetes 容器常用命令和基本原理，Helm工具以及Docker使用
- **技能：** 了解分布式对象存储系统DAOS/Ceph，分布式融合存储OpenEBS；2PC、RingHash&JumpHash原理



## 项目经历

### <span>实习项目：阿里云 | AIS服务器 | 混合云存储研发工程师</span> | 2022年06月 - 2022年08月

- **主要内容：** 与Intel SCG存储部门合作，就DAOS如何为云原生提供高性能存储服务进行初期探索验证

- **主要工作1：** 对DAOS进行模拟部署验证

  - 采用不同ECS实例探索模拟部署方案，**提交SPDK初始化bug**，并成功模拟NVMe和PMEM运行DAOS, 编译配置DPDK、SPDK、DAOS、RDMA编译配置，调研HugePage参数设置，**修改并编译IO500测试库**

  - 调研IO500榜单上性能与配置，进行ior和mdtest测试，通过本地NVMe和RPC网络测试，初步定位分析路线

- **主要工作2：** 研究DAOS原理并探索如何使用
  - 围绕如何使用DAOS和改进，阅读DAOS代码与论文，搭建三节点**K8s**集群，部署并对比**Ceph**对象存储和**OpenEBS**融合存储；通过libdfs接口编写**PyDAOS中间件**，实现 Python 列表操作, 提高**1.5倍**带宽写入

### 比赛项目：基于华为NBIoT的健身房管理系统 | 队长 | 2019年03月 - 2019年08月

- **技术栈**：Liteos系统+NBIOT+OceanConnect云平台+SpringBoot+SpringJPA+MySQL+JSP
- **内容**：编译烧录Liteos，通过NBIOT上传数据至华为云平台，通过RestAPI推送数据到web端 | [演示视频](https://www.bilibili.com/video/BV1bu41197Vk/?vd_source=d9dcbd1d6301ca5726a2f2b65b9c5a7b)
- **成果**：全国大学生物联网竞赛**东北赛区二等奖**

## 研究经历

### 研究经历一：Flex-based智能手套系统 | 独立开发 | 2021年03月 - 2021年10月

   - 研究并设计开发基于 Flex 传感器的智能手套，用于人-机协同控制，手势识别
   - **技术栈：** EmbedAI+BLE+Python+PyQT+PyTorch+Android | [Repository](https://github.com/liudongdong1/DataGlove) | 代码行数1w+
   - **内容：** 基于PyQT、Android开发实时系统，实现采集，量化，机械手控制，嵌入式AI推理，实时识别准确率达94%
   - **成果：** 荣获2022中国高校-C4网络技术挑战赛全国一等奖，创业赛道全国二等奖

### 研究经历二：基于RFID和CV的机器人图书盘点系统 | 项目参与者 | 2022年02月 - 2022年06月

   - 研究在RFID 标签密集场景，融合RFID和CV技术提高图书盘点的准确性
   - **技术栈：** CV + RFID + Python | [演示视频](https://www.youtube.com/watch?v=EfbT9QfQf50) | 代码行数**1w+**
   - **内容：** 设计了一个具有多个输入和混合数据的深度神经网络（DNN）模型，以滤除其他层RFID标签的干扰，并提出了一种视频信息提取方案，以准确提取书脊信息，并使用强链接来对齐和匹配RFID - 以及基于 CV 的时间戳与书名序列，以避免融合期间出现错误。系统的分层过滤平均准确率达到 98.4%，图书盘点平均准确率达到 98.9%。
   - **成果：** [Jiuwu Zhang, Xiulong Liu, Tao Gu, Bojun Zhang, **Dongdong Liu**, Zijuan Liu, Keqiu Li. “An RFID and Computer Vision Fusion System for Book Inventory using Mobile Robot.” in Proc. of IEEE INFOCOM 2022, CCF A](https://dl.acm.org/doi/10.1109/INFOCOM48880.2022.9796711) | 学生三作

### 研究经历三：RF-Camera 融合感知系统 | 独立开发 | ACM MobiCom 21（CCF A）2019年12月 - 2021年06月

   - 首次研究在多人多物场景下，融合RFID和CV识别用户是谁，手持哪种物品，绘制哪种动作

   - **技术栈：** C#主程序+WPF界面+AI人脸和手势识别 | [演示视频](https://www.youtube.com/watch?v=EfbT9QfQf50) | 代码行数**3w+**

   - **内容：** 负责RF-Camera实时系统开发包括实手势识别，人脸识别，物品识别与匹配，识别匹配准确率在90%以上 |  [简介](https://hub.baai.ac.cn/view/9544)

   - **成果：**[Liu Xiulong, Liu Dongdong et al. "RFID and camera fusion for recognition of human-object interactions." ACM MobiCom21, CCF A](https://dl.acm.org/doi/abs/10.1145/3447993.3483244) | 学生一作

### 研究经历四：吉林大学口腔医院室内实时多维定位与精准导航系统 | 项目负责人 | 2017年09月 - 2019年01月

- 研究Android端定位算法，并构建定位指纹库包括WiFi，地磁，视觉orb特征点，为导航提供调用接口
- **技术栈：** Android + Java + Python + AI | [演示视频](https://v.qq.com/x/page/r332242j7z4.html) | 代码行数**2w+**
- **内容：** 阅读并编译ORB-SLAM项目，移植到Android手机端实时运行，开发定位&导航演示demo 程序

- **成果：** 发明专利：[《一种室内定位方法、装置、系统及计算设备》](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=SCPD&dbname=SCPD202102&filename=CN110057355B&uniplatform=NZKPT&v=ZUCZ3coL34xtK3DpWxJC5-gWJriRiszNvFbtnnfKPp-7f8Q_bBEfV2HZAQAkE7YS)| 学生第一负责人 | 著作权：《orbslam3D云图系统》，《基于 or 视觉特征匹配定位系统》，《基于数据库的可视化地图编辑系统》，《基于opencv 的 Kinect 实时视频流处理系统》，《基于 svg 地图的信息提取裁剪系统》

## 教育背景

### 天津大学 | 计算机科学与技术 | 全日制硕士 | 智能与计算学部 | 2020年09月

- 荣誉奖项：2020 **学业奖特等奖学金** | 2021学业奖一等奖学金 | **研究生国家奖学金 |** 优秀学生干部 
- 云计算课程 93，大数据综合实验 96，IBMWebsphere 认证 97

### 吉林大学 | 物联网 | 全日制本科 | 计算机科学与技术学院 | 专业第三保研  2016年09月

- 荣誉奖项：2017国家励志奖学金 | 博世助学金 | 2018国家励志奖学金 | 华为奖学金 | 2019国家励志奖学金 | **CET6:551, CET4:573**


---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/about/  

