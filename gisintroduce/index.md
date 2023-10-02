# GIS Introduce


### 1. 概念介绍

**地理信息系统**（Geographic Information System或 Geo－Information system，GIS）有时又称为“地学信息系统”。它是一种特定的十分重要的空间信息系统。它是在[计算机](https://baike.baidu.com/item/计算机/140338)硬、软件系统支持下，对整个或部分[地球](https://baike.baidu.com/item/地球/6431)表层（包括大气层）空间中的有关[地理](https://baike.baidu.com/item/地理)分布[数据](https://baike.baidu.com/item/数据/33305)进行[采集](https://baike.baidu.com/item/采集/4843625)、[储存](https://baike.baidu.com/item/储存/2446499)、[管理](https://baike.baidu.com/item/管理/366755)、[运算](https://baike.baidu.com/item/运算/5866856)、[分析](https://baike.baidu.com/item/分析/4327108)、[显示](https://baike.baidu.com/item/显示/9985945)和[描述](https://baike.baidu.com/item/描述/8928757)的技术系统。

地理信息系统（GIS，Geographic Information System）是一门综合性学科，结合[地理学](https://baike.baidu.com/item/地理学/661412)与[地图学](https://baike.baidu.com/item/地图学/1749670)以及[遥感](https://baike.baidu.com/item/遥感/1240667)和计算机科学，已经广泛的应用在不同的领域，是用于输入、存储、查询、分析和显示[地理](https://baike.baidu.com/item/地理)数据的[计算机系统](https://baike.baidu.com/item/计算机系统/7210959)，随着GIS的发展，也有称GIS为“[地理信息科学](https://baike.baidu.com/item/地理信息科学/2553662)”（Geographic Information Science），近年来，也有称GIS为"地理信息服务"（Geographic Information service）。GIS是一种基于计算机的工具，它可以对空间信息进行分析和处理（简而言之，是对地球上存在的现象和发生的事件进行成图和分析）。 GIS 技术把地图这种独特的视觉化效果和地理分析功能与一般的数据库操作（例如查询和统计分析等）集成在一起。

在GIS中的两种地理数据成分：`空间数据`，与[空间要素](https://baike.baidu.com/item/空间要素)几何特性有关；`属性数据`，提供空间要素的信息。

地理信息系统（GIS）与[全球定位系统](https://baike.baidu.com/item/全球定位系统)(GPS)、[遥感系统](https://baike.baidu.com/item/遥感系统)(RS)合称[3S](https://baike.baidu.com/item/3S)系统。

地理信息系统(GIS) 是一种具有[信息系统](https://baike.baidu.com/item/信息系统)空间专业形式的[数据管理](https://baike.baidu.com/item/数据管理)系统。在严格的意义上, 这是一个具有集中、存储、操作、和显示地理参考信息的[计算机系统](https://baike.baidu.com/item/计算机系统)。例如，根据在[数据库](https://baike.baidu.com/item/数据库)中的位置对数据进行识别。实习者通常也认为整个GIS系统包括操作人员以及输入系统的数据。

地理信息系统（GIS）技术能够应用于科学调查、资源管理、财产管理、发展规划、绘图和路线规划。例如，一个地理信息系统(GIS)能使应急计划者在自然灾害的情况下较易地计算出应急反应时间，或利用GIS系统来发现那些需要保护不受污染的[湿地](https://baike.baidu.com/item/湿地)。

### 2. 特点

1. 公共的地理定位基础；
2. 具有采集、管理、分析和输出多种地理空间信息的能力；
3. 系统以分析模型驱动，具有极强的空间综合分析和动态预测能力，并能产生高层次的地理信息；
4. 以地理研究和地理决策为目的，是一个人机交互式的空间决策支持系统。

GIS数据以数字数据的形式表现了现实世界客观对象(公路、土地利用、海拔)。 现实世界客观对象可被划分为二个抽象概念: 离散对象(如房屋) 和连续的对象领域(如降雨量或海拔)。这二种抽象体在GIS系统中存储数据主要的二种方法为：[栅格](https://baike.baidu.com/item/栅格)（[网格](https://baike.baidu.com/item/网格)）和[矢量](https://baike.baidu.com/item/矢量)。

栅格（网格）数据由存放唯一值存储单元的行和列组成。它与栅格（网格）图像是类似的，除了使用合适的颜色之外，各个单元记录的数值也可能是一个分类组（例如土地使用状况）、一个连续的值（例如[降雨量](https://baike.baidu.com/item/降雨量)）或是当数据不是可用时记录的一个空值。栅格数据集的分辨率取决于地面单位的网格宽度。通常存储单元代表地面的方形区域，但也可以用来代表其它形状。栅格数据既可以用来代表一块区域，也可以用来表示一个实物。

### 3. 系统建模

- **【数据建模】**将湿地地图与在机场、电视台和学校等不同地方记录的降雨量关联起来是很困难的。然而，GIS能够描述 地表、地下和大气的二维三维特征。例如，`GIS能够将反映降雨量的雨量线迅速制图`。这样的图称为雨量线图。通过有限数量的点的量测可以估计出整个地表的特征，这样的方法已经很成熟。 一张二维雨量线图可以和GIS中相同区域的其它图层进行叠加分析。

- **【拓扑建模】**在过去的35年，在湿地边上有没有任何加油站或工厂经营过？有没有任何满足在2英里内且高出湿地的条件的这类设施？GIS可以识别并分析这种在数字化空间数据中的这种[空间关系](https://baike.baidu.com/item/空间关系)。这些拓扑关系允许进行复杂的空间建模和分析。`地理实体间的拓扑关系包括连接（什么和什么相连）、包含（什么在什么之中）、还有邻近（两者之间的远近）`。

- **【网络建模】**如果所有在湿地附近的工厂同时向河中排放化学物质，那么排入湿地的污染物的数量要多久就能达到破坏环境的数量？`GIS能模拟出污染物沿线性网络(河流)的扩散的路径。诸如坡度、速度限值、管道直径之类的数值可以纳入这个模型使得模拟得更精确。网络建模通常用于交通规划、水文建模和地下管网建模。`

### 4. 案例介绍

#### .1. 汽车导航系统

汽车导航系统是地理资讯系统的一个特例，它除了`一般的地理资讯系统`的内容以外，还包括了`各条道路的行车及相关信息的数据库`。这个数据库利用[矢量](https://baike.baidu.com/item/矢量)表示行车的`路线、方向、路段等信息`，又利用[网络拓扑](https://baike.baidu.com/item/网络拓扑)的概念来决定最佳行走路线。[地理数据文件](https://baike.baidu.com/item/地理数据文件)(GDF)是为导航系统描述地图数据的[ISO](https://baike.baidu.com/item/ISO)标准。[汽车导航系统](https://baike.baidu.com/item/汽车导航系统)组合了`地图匹配`、[GPS](https://baike.baidu.com/item/GPS)定位和来计算车辆的位置。地图资源数据库也用于`航迹规划、导航，并可能还有主动安全系统、辅助驾驶及位置定位服`务(Location Based Services, LBS)等高级功能。汽车导航系统的数据库应用了地图资源数据库管理。

![img](https://gitee.com/github-25970295/blogpictureV2/raw/master/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdWRvbmdkb25nMTk=,size_16,color_FFFFFF,t_70)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

#### .2. **[ arcgis-python-api](https://github.com/Esri/arcgis-python-api)**

将即用`型底图、业务图层和统计图表混合在一起，形成鲜活的动态地图`，以使用简单简明的方式共享地理内容。![](https://gitee.com/github-25970295/blogpictureV2/raw/master/02-fig-2-8-v2-16276341155155.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210730191401908.png)

#### .3. [cesium](https://github.com/CesiumGS/cesium)

> CesiumJS is a `JavaScript library` for creating `3D globes and 2D maps `in a web browser without a plugin. It uses` WebGL for hardware-accelerated graphics`, and is `cross-platform, cross-browser, and tuned for dynamic-data visualization`.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210730221049511.png)

#### .4. [QGIS](https://github.com/qgis/QGIS)

> QGIS is a free, open source, cross platform (lin/win/mac) geographical information system (GIS). [map visualization](https://www.flickr.com/groups/2244553@N22/pool/with/50355460063/?rb=1)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210730221313879.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210730221328730.png)

#### .5. OSMNX

> **OSMnx** is a Python package that lets you `download geospatial data from OpenStreetMap and model, project, visualize, and analyze real-world street networks and any other geospatial geometries`. You can download and model walkable, drivable, or bikeable urban networks with a single line of Python code then easily analyze and visualize them. You can just as easily download and work with other infrastructure types, amenities/points of interest, building footprints, elevation data, street bearings/orientations, and speed/travel time.

#### .6. [leaflet-geoman](https://github.com/geoman-io/leaflet-geoman)

> The most `powerful leaflet plugin` for drawing and editing geometry layers

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/68747470733a2f2f66696c652d676d65696c6571666d672e6e6f772e73682f)

### Resource

- 开源GIS浅谈  https://blog.csdn.net/happyduoduo1/article/details/51773850    
- GIS学习教程：https://www.esri.com/en-us/what-is-gis/overview#image6



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/gisintroduce/  

