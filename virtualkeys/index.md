# VirtualKeys


> 基于OpenCV，结合图像摄像头、980nm一字红外激光、980nm红外滤光片以及键盘投影激光组成，使用加装了980nm红外滤光片的图像摄像头检测由手指遮挡引起980nm一字红外激光漫反射生成的光点，通过`检测和定位这个光点轮廓的中心位置`，从而达到识别和检测`手指的位置`，然后映射到键盘位置，从而实现对应的键盘按键事件，使用OpenCV视觉库，可以很快捷查找由图像摄像头获取到的手指头轮廓和定位手指头的位置，以及校正由图像摄像头引起的图像曲面失真，使用OpenCV可以减少底层硬件驱动代码的编写，调用内部函数可以直接面向硬件编写程序.

### 1. 工作原理

> 当用户在桌上“按下”一个虚拟的按键后，手指上反射的激光信号会被摄像头捕捉。随后安装在PC/Mac上的信号处理软件就会进行最核心的工作：通过反射的激光光斑定位用户的指尖位置，并求出对应的按键。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119180051285.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119175248351.png)

- 红外激光：

> 使用红外线作为信号检测的光源，之所以选择红外激光是因为激光的具有低功耗和集成度效果好，而且其发出光的频率几乎专一，可以让人眼觉察不到，为了考虑到人身安全状况，故选择30mW的980nm红外激光，由于键盘是一个平面，所以不能使用传统的单束激光作为光源，所以只能选择一字形激光，其线角度为120°，只有调节好投影键盘的位置，一字激光可以完全覆盖到所有的键盘范围。

> 此时若手指接近桌面，则会阻挡住激光的通路，产生反射，反射的光点画面会被图中摄像头拍摄到。这是一个标准的三角测距的结构设置。细心的读者可能已经发现在前文给出的本制作摄像头拍摄到的画面中手指尖部的白色光斑，这正是安装了线激光器后被手指遮挡产生的反射效果。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119180006149.png)

- 识别键盘事件

> **1) 通过计算机视觉的方式，通过图像来识别**
>
> 通过摄像头捕捉键盘区域的画面并进行分析，判断出键盘输入事件。
>
> **2) 通过检测按键发出的声音来判断**
>
> 这里假设使用者在按键时会碰触桌面，产生一定的敲击声。通过检测该声音传播时间，可以进行定位。该方案在国外的一些研究机构已经实现。
>
> **3) 通过超声波雷达手段来判断**
>
> 通过发射超声波并检测反射波的传播时间查来检测目标物体（手指）的位置。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119175858968.png)

| **元件**                 | **核心参数**         |
| ------------------------ | -------------------- |
| **摄像头**               | 广角镜头，视角>120度 |
| **投射键盘画面的激光器** | 无特殊要求           |
| **一字线激光**           | 红外激光器，>50mW    |
| **红外带通滤光片**       | 800nm左右光谱带通    |





---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/virtualkeys/  
