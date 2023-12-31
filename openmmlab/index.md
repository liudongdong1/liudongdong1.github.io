# OpenMMLab


> [OpenMMLab](http://openmmlab.org/) 在Github上不是一个单独项目，除了大家所熟知的 Github 上万 star 目标检测库 MMDetection，还有其他方向的代码库和数据集,目前Github总星标超过 1.7 万。是CV方向系统性较强、社区活跃的开源平台。

## 1. [MMCV](https://github.com/open-mmlab/mmcv)

> MMCV是用于计算机视觉研究的基础Python库，支持OpenMMLab旗下其他开源库。主要功能是I/O、图像视频处理、标注可视化、各种CNN架构、各类CUDA操作算子。

## 2.[MMDetection](https://github.com/open-mmlab/mmdetection)

> MMDetection是基于PyTorch的开源目标检测工具箱。是OpenMMLab最知名的开源库，几乎是研究目标检测必备！
>
> 主要特点：
>
> - 模块化设计
> - 支持开箱即用的多方法
> - 高效率
> - SOTA

主持的主干网:

-  ResNet
-  ResNeXt
-  VGG
-  HRNet
-  RegNet
-  Res2Net

支持的算法:

-  RPN
-  Fast R-CNN
-  Faster R-CNN
-  Mask R-CNN
-  Cascade R-CNN
-  Cascade Mask R-CNN
-  SSD
-  RetinaNet
-  GHM
-  Mask Scoring R-CNN
-  Double-Head R-CNN
-  Hybrid Task Cascade
-  Libra R-CNN
-  Guided Anchoring
-  FCOS
-  RepPoints
-  Foveabox
-  FreeAnchor
-  NAS-FPN
-  ATSS
-  FSAF
-  PAFPN
-  Dynamic R-CNN
-  PointRend
-  CARAFE
-  DCNv2
-  Group Normalization
-  Weight Standardization
-  OHEM
-  Soft-NMS
-  Generalized Attention
-  GCNet
-  Mixed Precision (FP16) Training
-  InstaBoost
-  GRoIE
-  DetectoRS
-  Generalized Focal Loss

## 3. MMDetection3D

> 从CVPR2020 中也可以看出3D目标检测研究异常火热，该库是专门用于3D目标检测的开源库。
>
> 主要特点：
>
> - 支持开箱即用的多模态/单模态检测器
> - 支持开箱即用的室内/室外检测器
> - 与2D目标检测自然融合
> - 高效率

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200715090523.gif)

## 4. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

> MMSegmentation是一个基于PyTorch的开源语义分割工具箱.
>
> 主要特点：
>
> - 统一基准
> - 模块化设计
> - 支持开箱即用的多方法
> - 高效率

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200715090734.gif)

支持的骨干网:

-  ResNet
-  ResNeXt
-  HRNet

支持的算法:

-  FCN
-  PSPNet
-  DeepLabV3
-  PSANet
-  DeepLabV3+
-  UPerNet
-  NonLocal Net
-  EncNet
-  CCNet
-  DANet
-  GCNet
-  ANN
-  OCRNet

## 5. [MMClassification](https://github.com/open-mmlab/mmclassification)

> MMClassification是基于PyTorch的开源图像分类工具箱。
>
> 主要特点：
>
> - 各种骨干与预训练模型
> - Bag of training tricks
> - 大规模训练配置
> - 高效率与可扩展性

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200715090827.gif)

支持的骨干网:

-  ResNet
-  ResNeXt
-  SE-ResNet
-  SE-ResNeXt
-  RegNet
-  ShuffleNetV1
-  ShuffleNetV2
-  MobileNetV2
-  MobileNetV3

## 6. [MMPose](https://github.com/open-mmlab/mmpose)

> MMPose是一个基于PyTorch的开源姿势估计工具箱。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200715090944.gif)

## 7. [MMAction](https://github.com/open-mmlab/mmaction)

> MMAction是一个基于PyTorch开放源代码的工具箱，用于动作理解。
>
> 主要特点：
>
> - 可以解决以下任务：
>
> - 从剪辑视频中进行动作识别
> - 未剪辑视频中的时序动作检测（也称为动作定位）
> - 未剪辑视频中的时空动作检测。
>
> - 支持各种数据集
> - 支持多动作理解框架
> - 模块化设计

## 8. [MMAction2](https://github.com/open-mmlab/mmaction2)

> MMAction2是一个基于PyTorch开放源代码的工具箱，用于动作理解。
>
> 主要特点：
>
> - 模块化设计
> - 支持多种数据集
> - 支持多重动作理解框架
> - 完善的测试和记录
>
> MMAction2比MMAction支持的算法更多，速度更快，开发者也更活跃。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200715091056.gif)

支持的动作识别算法:

-  TSN
-  TSM
-  R(2+1)D
-  I3D
-  SlowOnly
-  SlowFast

支持的动作定位算法:

-  BMN
-  BSN

## 9. [MMSkeleton](https://github.com/open-mmlab/mmskeleton)

> MMSkeleton
>
> 用于人体姿势估计，基于骨架的动作识别和动作合成。
>
> 特点：
>
> - 高扩展性
> - 多任务

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20200715091133.gif)

## 10. [MMFashion](https://github.com/open-mmlab/mmfashion)

> MMFashion是一个基于PyTorch的开源视觉时尚分析工具箱。
>
> 特点：
>
> - 灵活：模块化设计，易于扩展
> - 友好：外行用户的现成模型
> - 全面：支持各种时装分析任务
>
> 支持应用：
>
> - 服饰属性预测
> - 服饰识别与检索
> - 服饰特征点检测
> - 服饰解析和分割
> - 服饰搭配推荐

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/微信图片_20200715091208.gif)

## 11. MMEditing

> MMEditing是基于PyTorch的开源图像和视频编辑工具箱
>
> 主要特点：
>
> - 模块化设计
> - 在编辑中支持多任务
> - SOTA

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715091256696.png)

## 12.**OpenPCDet**

> OpenPCDet 是一个清晰，简单，自成体系的开源项目，用于基于LiDAR的3D目标检测。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715091336543.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/openmmlab/  

