# KinetRelativeProject


### 1. [Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)

> **Azure Kinect SDK** is a cross platform (Linux and Windows) user mode SDK to read data from your Azure Kinect device.
>
> - Depth camera access
> - RGB camera access and control (e.g. exposure and white balance)
> - Motion sensor (gyroscope and accelerometer) access
> - Synchronized Depth-RGB camera streaming with configurable delay between cameras
> - External device synchronization control with configurable delay offset between devices
> - Camera frame meta-data access for image resolution, timestamp and temperature
> - Device calibration data access

Azure Kinect DK 开发环境由以下多个 SDK 组成：

- 用于访问低级别传感器和设备的传感器 SDK。
- 用于跟踪 3D 人体的人体跟踪 SDK。
- 用于启用麦克风访问和基于 Azure 云的语音服务的语音认知服务 SDK。

可将认知视觉服务与设备 RGB 相机配合使用。

![Azure Kinect SDK 示意图](https://docs.microsoft.com/zh-cn/azure/Kinect-dk/media/quickstarts/sdk-diagram.jpg)

![](https://azure.microsoft.com/images/page/services/azure-kinect-dk/whats-inside.jpg?v=7cb127ac965b83c1e6840842244fcb5e31a9f4f9328acfb2a63b04538befa8b7)

![完整设备功能](https://docs.microsoft.com/zh-cn/azure/Kinect-dk/media/quickstarts/full-device-features.png)

- 100 万像素深度传感器，具有宽、窄视场角 (FOV) 选项，可帮助你针对应用程序进行优化
- 7 麦克风阵列，可用于远场语音和声音捕获
- 1200 万像素 RGB 摄像头，提供和深度数据匹配的彩色图像数据流
- 加速计和陀螺仪 (IMU)，可用于传感器方向和空间跟踪
- 外部同步引脚，可轻松同步多个 Kinect 设备的传感器数据流

**Azure Kinect 人体跟踪工具**

- 人体跟踪器提供一个查看器工具用于跟踪 3D 人体。3D 坐标系中的点以指标 [X,Y,Z] 坐标三元组的形式表示，单位为毫米。

![png](https://img-blog.csdn.net/20180703120303332?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmdiaW5feHU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210204100249440.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210204100307697.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210204100217254.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210204100337343.png)

### 2.[LiveScan3D](https://github.com/MarekKowalski/LiveScan3D)

> a system designed for real time 3D reconstruction using multiple AzureKinect or Kinect v2 depth sensors simultaneously at real time speed. For both sensors the produced 3D reconstruction is in the form of a coloured point cloud, with points from all of the Kinects placed in the same coordinate system. The point cloud stream can be visualized, recorded or streamed to a HoloLens or any Unity application. 
>
> - capturing an object’s 3D structure from multiple viewpoints simultaneously,
> - capturing a “panoramic” 3D structure of a scene (extending the field of view of one sensor by using many),
> - streaming the reconstructed point cloud to a remote location,
> - increasing the density of a point cloud captured by a single sensor, by having multiple sensors capture the same scene.
> - [video inroduce](https://www.youtube.com/watch?v=9y_WglwpJtE)

### 3.[nuitrack-sdk](https://github.com/3DiVi/nuitrack-sdk)

> **Nuitrack™** is a `3D tracking middleware` developed by **3DiVi Inc**. This is a solution for `skeleton tracking and gesture recognition` that enables capabilities of Natural User Interface (NUI) on **Android**, **Windows**, and **Linux**.
>
> **Nuitrack™ framework** is multi-language and cross-platform. **Nuitrack™ API**s include the set of interfaces for developing applications, which utilize Natural Interaction. The main purpose of **Nuitrack™** is to establish an API for communication with 3D sensors.
>
> The **Nuitrack™ module** is optimized for ARM based processors, which means that you can use it with Android devices and embedded platforms.
>
> - Features:
>   - `Full Body Skeletal Tracking (19 Joints);`
>   - `3D Point Cloud;`
>   - `User Masks;`
>   - `Gesture Recognition;`
>   - `Cross-platform SDK` for Android, Windows, and Linux;
>   - 3D Sensor independent:` supports Kinect v1, Asus Xtion, Orbbec Astra, Orbbec Persee, Intel RealSense;
>   - Unity and Unreal Engine Plugins;
>   - OpenNI 1.5 compatible: OpenNI module allows you to move your OpenNI based apps developed for Kinect and Asus Xtion to other platforms, including Android.
> - Application areas:
>   - Natural User Interface (NUI) for Windows/Linux/Android;
>   - Games and Training (Fitness, Dance Lessons);
>   - Medical Rehabilitation;
>   - Smart Home;
>   - Positional and Full Body Tracking for VR;
>   - Audience Analytics;
>   - Robot Vision.
>   - [video url](https://www.youtube.com/watch?v=VResoec14wQ)
>   - [tutorial](https://www.youtube.com/watch?v=F1Vx_pu9dbk)

### 4. [KinectTouch](https://github.com/robbeofficial/KinectTouch)

> Turns any surface into a giant touchpad using kinect

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119150933293.png)

### 5. [kinectron](https://github.com/kinectron/kinectron)

> Kinectron is a `node-based library that broadcasts Kinect2 data over a peer connection`. It builds on the Node Kinect2 and PeerJS libraries. Kinectron has two components--an electron application to broadcast Kinect2 data over a peer connection, and a client-side API to request and receive Kinect2 data over a peer connection.

### 6. [depthjs](https://github.com/doug/depthjs)

> DepthJS is an open-source browser extension and plugin (currently working for Chrome) that `allows the Microsoft Kinect to talk to any web page`. It provides the low-level raw access to the Kinect as well as high-level hand gesture events to simplify development.

### 7.[KinectFusionApp](https://github.com/chrdiller/KinectFusionApp)

>  implements cameras (for data acquisition from recordings as well as from a live depth sensor) as data sources. The resulting fused volume can then be exported into a pointcloud or a dense surface mesh.

### 8. [Kinect-Vision](https://github.com/soumik12345/Kinect-Vision)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210301202846740.png)

### 9. [rgbd-kinect-pose](https://github.com/rmbashirov/rgbd-kinect-pose)

> Bashirov R, Ianina A, Iskakov K, et al. Real-time RGBD-based Extended Body Pose Estimation[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2021: 2807-2816.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210822163959268.png)

### 10. [Azure-Kinect-DK-3D-reconstruction](https://github.com/luckyluckydadada/Azure-Kinect-DK-3D-reconstruction)

> 利用开源框架open3d的Reconstruction system实现Azure Kinect DK相机的三维重建。 

### 10. 相关资源

- https://azure.microsoft.com/zh-cn/services/kinect-dk/?cdn=disable

  https://www.youtube.com/watch?v=PcIG0qbOMRA   这个是关于Kinect3 视频

  https://brekel.com/  关于Kinect  LeapMotion 公司

  https://github.com/microsoft/Azure-Kinect-Samples
  
- 官网教程：https://docs.microsoft.com/en-us/azure/kinect-dk/azure-kinect-recorder

- pykinect: 



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kinectrelativeproject/  

