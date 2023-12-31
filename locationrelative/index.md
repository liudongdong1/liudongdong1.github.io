# ROS Relative Learning


> Chen, Xieyuanli, Thomas Läbe, Lorenzo Nardi, Jens Behley, and Cyrill Stachniss. "Learning an Overlap-based Observation Model for 3D LiDAR Localization."

------

# Paper: Overlap-based

<div align=center>
<br/>
<b>Learning an Overlap-based Observation Model for 3D LiDAR Localization</b>
</div>


#### Summary

1. **文章使用了OverlapNet作为蒙特卡洛定位算法（MCL）的观测模型**，实现了基于激光雷达传感器的高精度全局定位。**目前MCL最大的难题**就是如何去设计一个好的观测模型。**文章的创新点**是利用OverlapNet来训练了一个观测模型，然后把它集成到MCL中，提高了MCL的定位性能。
2. a approach for global localization  using 3D Lidar scans on road vehicles;
3. novel observation model that exploit the overlap  and yaw angle estimation;
4. using overlapNet2020 model;
5. 开源代码：**https://github.com/PRBonn/overlap_localization**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201203185627048.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201203190024140.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/locationrelative/  

