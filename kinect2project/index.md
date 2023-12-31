# kinect2project


> In the` depth image`, the` value of a pixels` relates to the `distance from the camera` as measured by time-of-flight. For the a`ctive infrared imag`e, the `value of a pixel` is determined by the `amount of infrared light reflected back to the camera`.
>
> - The Kinect uses the reflected IR to calculate time of flight but then also makes it available as an IR image.

### 1. **[OpenDepthSensor](https://github.com/jing-interactive/OpenDepthSensor)**

> kinect官方sdk自带demo，包含 AudioBeam, AudioBody, BodyIndex, ChromaKey, Color, CoordinateMapper, Depth, Face, FaceClip, FaceRecogition, Fusion, Gesture, HDFace, Infrared, Inpaint, JointSmooth, MultiSource, PointCloud, Speech 示例demo； [使用教程, ](http://brightguo.com/kinect2-official-sdk-samples/)
>
> - 声源定位，音源采集（控制台）
> - 身体基础，身体指数基础， 高清画面显示，控制基础，`坐标映射基础`, 深度基础例子, 不良姿势等基础知识
> - 颜色基础 D2D, 高清面部基础（用于实时捕捉显示面部表情 具有重置按钮）
> - 红外基础 D2D（显示红外线摄像头的实时捕捉画面 具有重置功能），红外基础 WPF
> -  Kinect融合基础，Kinect融合器的基础知识, 手势生成器查看器 [[配图介绍]](http://brightguo.com/kinect2-official-sdk-samples/)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/psb-2.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/psb-10.png)

![psb (14)](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/psb-14.png)

![psb (18)](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/psb-18.png)

![psb (26)](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/psb-26.png)

![kinect fusion Explorer-WPF ](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/psb-31.png)

### 2. **[Kinect-Gait-Analysis](https://github.com/ahhda/Kinect-Gait-Analysis)**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210725230139447.png)

### 3. **[ OpenDepthSensor](https://github.com/jing-interactive/OpenDepthSensor)**

- Kinect for Azure via [k4a SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) (Windows, Linux)
- Kinect V1 via KinectCommonBridge (Windows 7+)
- Kinect V2 via KinectCommonBridge-V2 (Windows 8+)
- Kinect V2 via [libfreenect2](https://github.com/jing-vision/libfreenect2) (Windows 7+, macOS, Linux)
- Intel RealSense sensors (R200, F200, SR300, LR200, ZR300) via librealsense SDK (Windows, macOS, Linux)
- OpenNI2-compatible sensors via OpenNI2 SDK (Windows, macOS, Linux, Android)
- Hjimi sensors via Imi SDK (Windows, Linux, Android)

### 4. **[kinect2.0-opencv](https://github.com/otnt/kinect2.0-opencv)**

> c++ code; Using Kinect 2.0 + OpenCV to get `Depth Data, Body Data, Hand State and Body Index Data.`

### 5. **[PyKinect2-PyQtGraph-PointClouds](https://github.com/KonstantinosAng/PyKinect2-PyQtGraph-PointClouds)**

> Creating `real-time dynamic Point Clouds` using PyQtGraph, Kinect 2 and the python library PyKinect2.  The `PointCloud.py` file contains the main class to `produce dynamic Point Clouds using the [PyKinect2](https://github.com/Kinect/PyKinect2) and the [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) libraries.` The main file uses the numpy library that runs in C, thus it is fully optimized and can produce dynamic Point Clouds with up to `60+ frames`, except for the point clouds `produces by the RGB camera that run in 10+ frames.` The library can also be used to create a PointCloud and save it as a .txt file containing the world point coordinates as: x, y, z . . . x, y, z It can also be used to view .ply or .pcd point cloud files or create PointClouds and save them as .ply or .pcd files. Instructions on how to use the main file are written in the **Instructions** chapter. In addition, there is a window with opencv track bars that can be used to dynamically change the color and the size of the points in the point cloud and the input flags.

```python
# rgb camera
pcl = Cloud(dynamic=True, color=True)
pcl.visualize()

# depth camera
pcl = Cloud(dynamic=True, depth=True)
pcl.visualize()

# body index
pcl = Cloud(dynamic=True, body=True)
pcl.visualize()

# skeleton cloud
pcl = Cloud(dynamic=True, skeleton=True)
pcl.visualize()
```

```python
# example 1 with color and depth point clouds
pcl = Cloud(dynamic=True, simultaneously=True, color=True, depth=True, body=False, skeleton=False, color_overlay=False)
pcl.visualize()

# example 2 with all the point clouds enabled (scroll out to see the point cloud)
pcl = Cloud(dynamic=True, simultaneously=True, depth=True, color=True, body=True, skeleton=True, color_overlay=True)
pcl.visualize()

# example 3 with depth and body index point cloud
pcl = Cloud(dynamic=True, simultaneously=True, depth=True, color=False, body=True, skeleton=False, color_overlay=True)
pcl.visualize()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image_6.png)

### 6. **[kinect2-opencv](https://github.com/m6c7l/kinect2-opencv)**

![kinect2-opencv](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/kinect2-opencv.png)

> - Camera: raw and altered images from RGB camera, IR and depth sensor
> - Tracker: ROI and color trackers
> - 3D: point clouds
> - Depth: denoised depth maps
> - IR: simple filters applied to IR images
> - Features: various feature extractions
> - BackSub: background substractions

- OpenKinect -- [Open source drivers for the Kinect for Windows v2 device (libfreenect2)](https://github.com/OpenKinect/libfreenect2)
- Ryuichi Yamamoto -- [A python interface for libfreenect2](https://github.com/r9y9/pylibfreenect2)
- Adrian Rosebrock -- [Ball Tracking with OpenCV](https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
- Satya Mallick -- [Object Tracking using OpenCV](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/)

### 7. **[ Tracking-Coordinate](https://github.com/Tacode/Tracking-Coordinate)**

![](https://github.com/Tacode/Tracking-Coordinate/raw/master/object.gif)

### 8. **[kinect2-utilities](https://github.com/rjgpinel/kinect2-utilities)**

```python
//Calibrate a camera using recorded images of a Charuco board (e.g. 50 images):
python -m kintools.calibration path_to_images/
//Markers pose estimation using camera calibration parameters:
 python -m kintools.pose_estimation cam_params.pkl path_to_image
//convert depth image from kinect2 to point cloud
python -m kintools.reconstruct -i depth/depth.npy -o cloud/res.npy
```

### 9. **[kinect2](https://github.com/mdleiton/kinect2)**

> 使用 freenect2 取捕获深度图，rgb图，红外图， RGBD图；

- https://github.com/r9y9/pylibfreenect2

> 安装错误记录：
>
> - 镜像源出现问题，无法install 和创建环境
> - 代码问题，无法建立连接
> - python install Cython
> - pip install numpy
> - pip install pylibfreenect2
> - pip install opencv-python

### 10. **[LibKinect2](https://github.com/sshh12/LibKinect2)**

> A Python API for interfacing with the Kinect2.

```python
from libkinect2 import Kinect2
from libkinect2.utils import depth_map_to_image
import numpy as np
import cv2

# Init Kinect2 w/2 sensors
kinect = Kinect2(use_sensors=['color', 'depth'])
kinect.connect()
kinect.wait_for_worker()

for _, color_img, depth_map in kinect.iter_frames():
    
    # Display color and depth data
    cv2.imshow('color', color_img)
    cv2.imshow('depth', depth_map_to_image(depth_map))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

kinect.disconnect()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/59576877-e8f68c00-9086-11e9-826b-eceb6eb80573.gif)

![](https://user-images.githubusercontent.com/6625384/59576903-088db480-9087-11e9-96f6-251240d25f0c.gif)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kinect2project/  

