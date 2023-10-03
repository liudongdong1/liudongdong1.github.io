# LaneDetection


#### 1. [Lanenet-Lane-Detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

- 检测车道线
- 检测其他物体：车辆、人、环境中的动物
- 跟踪检测到的对象
- 预测他们可能的运动
- 检测其他车辆是否在车道线内，并量度与他们的距离
- 检测邻近车道上是否有车辆的存在
- 了解弯曲道路的转弯半径

> LaneNet模型是一种两阶段车道线预测器。第一阶段是一个编码器-解码器模型，为车道线创建分割掩码。第二阶段是车道先定位网络，从掩码中提取的车道点作为输入，使用LSTM学习一个二次函数来预测车道线点。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210320155206329.png)

### 2. [Surface Detection](https://github.com/Charmve/Surface-Defect-Detection)

> 3C, automobiles, home appliances, machinery manufacturing, semiconductors and electronics, chemical, pharmaceutical, aerospace, light industry and other industries. 
>
> Compared with the clear classification, detection and segmentation tasks in computer vision, the requirements for defect detection are very general. In fact, its requirements can be divided into three different levels: "what is the defect" (**classification**), "where is the defect" (**positioning**) And "How many defects are" (**split**).

- The surface defect dataset released by Northeastern University (NEU) collects six typical surface defects of hot-rolled steel strips, namely rolling scale (RS), plaque (Pa), cracking (Cr), pitting surface (PS), inclusions (In) and scratches (Sc).

-  Solar Panels: elpv-dataset：https://github.com/zae-bayern/elpv-dataset
- Metal Surface: KolektorSDD
- PCB Inspection: DeepPCB
- Fabric Detects Datsets
- Aluminium Profile Surface Defect Dataset
- Industrial Optical Inspection: aimed at miscellaneous defects on textured backgrounds.
- RSDDs: Rail Surface Defect Datasets
- texture dataset

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/lanedetection/  

