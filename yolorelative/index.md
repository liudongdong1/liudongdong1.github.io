# YoloRelative


### **1 Yolox相关基础知识点**

#### **1.1 Yolox的论文及代码**

> - Yolox论文名：《YOLOX: Exceeding YOLO Series in 2021》
> - Yolox论文地址：https://arxiv.org/abs/2107.08430
> - Yolox代码地址：https://github.com/Megvii-BaseDetection/YOLOX

#### **1.2. Yolox个版本网络结构图**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816084242907.png)

> 将**各个模型文件转换成onnx格式**，再用**netron工具打开的方式，** 对网络结构进行可视化学习。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816084252074.png)

##### 1.2.1 Netron工具

> 使用**netron可视化工具**，可以清晰的看到每一层的输入输出，网络总体的架构，而且支持各种不同网络框架，简单好用
>
> - 在线版本： https://lutzroeder.github.io/netron/
> - app版本： **https://github.com/lutzroeder/netron**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816084558971.png)

##### 1.2.2 各个Yolox的onnx文件

> 各个onnx文件，可以采用代码中的，tools/export_onnx.py脚本，进行转换

##### 1.2.3 各个Yolox网络结构图

- Yolox-Nano网络结构可视图： https://blog.csdn.net/nan355655600/article/details/119329864
- Yolox-Tiny网络结构可视图：https://blog.csdn.net/nan355655600/article/details/119329848
- Yolox-Darknet53网络结构可视图：https://blog.csdn.net/nan355655600/article/details/119329834
- Yolox-s网络结构可视图：https://blog.csdn.net/nan355655600/article/details/119329727
- Yolox-l网络结构可视图：https://blog.csdn.net/nan355655600/article/details/119329801
- Yolox-x网络结构可视图：https://blog.csdn.net/nan355655600/article/details/119329818

### **2 Yolox核心知识点**

#### **2.1 Yolov3&Yolov4&Yolov5网络结构图**

![YoloV3](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816085213132.png)

![YoloV4](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816085420553.png)

![YoloV5](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816085704126.png)

**（1）标准网络结构：** Yolox-s、Yolox-m、Yolox-l、Yolox-x、Yolox-Darknet53。

**（2）轻量级网络结构：** Yolox-Nano、Yolox-Tiny。



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/yolorelative/  

