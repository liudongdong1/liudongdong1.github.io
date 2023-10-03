# kinect2Relative


> - `红外图像`，像素值由反射回相机的红外光量确定。
> - `深度图像`也叫距离影像，是指将`从图像采集器到场景中各点的距离（深度）值作为像素值的图像`。获取方法有：`激光雷达深度成像法`、`计算机立体视觉成像`、`坐标测量机法`、`莫尔条纹法`、`结构光法`。
> - `点云`：当一束激光照射到物体表面时，所反射的激光会携带方位、距离等信息。若将激光束按照某种轨迹进行扫描，便会边扫描边记录到反射的激光点信息，由于扫描极为精细，则能够得到大量的激光点，因而就可形成激光点云。点云格式有*.las ;*.pcd; *.txt等。`深度图像经过坐标转换可以计算为点云数据`；`有规则及必要信息的点云数据可以反算为深度图像`。
> -  `TOF`是通过红外光发射器发射调制后的红外光脉冲，不停地打在物体表面，经反射后被接收器接收，通过`相位的变化来计算时间差`，进而结合光速计算出物体深度信息。不怎么受环境光干扰，缺点是分辨率暂时都做不高。
> - `结构光`是通过红外光发射器发射一束编码后的光斑到物体表面，光斑打在物体表面后，由于物体的形状、深度不同，光斑位置不同，通过光斑的编码信息与成像信息，进而计算出物体深度信息。`结构光在室外效果很差，光斑成像容易受环境光干扰`。

> - pykinect2使用demo： https://github.com/liudongdong1/pykinect2
> - `彩色空间坐标系统（Color Space）` ——ColorSpacePoint(x,y),彩色图像 
> - `深度空间坐标系统（Depth Space）` ——DepthSpacePoint(x,y),红外线图像、深度图像以及BodyIndex图像, x代表列，y代表行，（x，y）就表示深度图上的一个像素坐标。（0，0）对应于图片的左上角，而（511，423）代表着图片的右下角。`深度图和红外图都是一个传感器得到的`。`深度图上每个像素`对应的`彩色值`，你会用到`Coordinate mapping类来获得彩色图上对应的像素位置。`
>
> 以`左上角为原点`，往右是+X，往下是+Y，单位是像素
>
> - 摄像头空间坐标系统（Camera Space）——CameraSpacePoint(x,y,z),以感应器为原点的3维空间坐标系统，单位是米（m）,做人体骨架追踪需要用到这个坐标系统。

> 深度图上每个像素对应的彩色值，你会用到Coordinate mapping类来获得彩色图上对应的像素位置。
>
> - **映射**的概念，比如深度图映射到彩色图的意思是对于`深度图上的一个像素，找到彩色图上的一个像素与之对应`.
> - Depth Map 类似于灰度图像，只是它的`每个像素值是传感器距离物体的实际距离`。通常RGB图像和Depth图像是配准的，因而`像素点之间具有一对一的对应关系`。深度图像 = 普通的RGB三通道彩色图像 + Depth Map, **图像深度** 是指存储每个像素所用的位数，也用于量度图像的色彩分辨率。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610204607302.png)

> jpg格式属于有损压缩，而png为无损压缩。研究者当然希望传感器精度更高。
> 同理，在piv,dic等摄影测量的领域，为了保证优化的精度，直接拍摄的rgb图也不会采用jpg格式保存。
>
> ```python
> cv2.imwrite('./examples/savefig/rgb/image_r_{}.png'.format(str(i).zfill(5)), rgb_map)  cv2.imwrite('./examples/savefig/depth/Tbimage_d_{}.png'.format(str(0).zfill(5)), np.asarray(depth_map,np.uint16))
> ```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610202047942.png)

#### 1. Coordination

##### .1. 图像坐标系

> 图像坐标系分为像素和物理两个坐标系种类。数字图像的信息以矩阵形式存储，即一副像素的图像数据存储在维矩阵中。`图像像素坐标系以为原点、以像素为基本单位，U、V分别为水平、垂直方向轴`。图像`物理坐标系以摄像机光轴与图像平面的交点作为原点`、以米或毫米为基本单位，其X、Y轴分别与U、V轴平行。图2-4展示的是两种坐标系之间的位置关系：

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610214155636.png)

##### .2. 摄像机坐标系

> 摄像机坐标系由摄像机的光心及三条、、轴所构成。它的、轴对应平行于图像物理坐标系中的、轴，轴为摄像机的光轴，并与由原点、、轴所组成的平面垂直。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610205431255.png)

##### .3. 世界坐标系

> 考虑到摄像机位置具有不确定性，因此有必要采用世界坐标系来统一摄像机和物体的坐标关系。世界坐标系由原点及、、三条轴组成。世界坐标与摄像机坐标间有着（2-3）所表达的转换关系

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610205600991.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610205614248.png)

##### .4. [相机参数含义](https://bbs.huaweicloud.com/blogs/218498)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610213030659.png)

##### .5. 单目测距

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610212654526.png)

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Date: 18-10-29

import numpy as np      # 导入numpy库
import cv2              # 导入Opencv库

KNOWN_DISTANCE = 32   # 这个距离自己实际测量一下

KNOWN_WIDTH = 11.69     # A4纸的宽度
KNOWN_HEIGHT = 8.27

IMAGE_PATHS = ["Picture1.jpg", "Picture2.jpg", "Picture3.jpg"]   # 将用到的图片放到了一个列表中


# 定义目标函数
def find_marker(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图转化为灰度图
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)    # 高斯平滑去噪
    edged_img = cv2.Canny(gray_img, 35, 125)     # Canny算子阈值化
    cv2.imshow("降噪效果图", edged_img)          # 显示降噪后的图片
    # 获取纸张的轮廓数据
    img, countours, hierarchy = cv2.findContours(edged_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(countours))
    c = max(countours, key=cv2.contourArea)    # 获取最大面积对应的点集
    rect = cv2.minAreaRect(c)       # 最小外接矩形
    return rect


# 定义距离函数
def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


# 计算摄像头的焦距（内参）
def calculate_focalDistance(img_path):
    first_image = cv2.imread(img_path)      # 这里根据准备的第一张图片，计算焦距
    # cv2.imshow('first image', first_image)
    marker = find_marker(first_image)       # 获取矩形的中心点坐标，长度，宽度和旋转角度
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH  # 获取摄像头的焦距
    # print(marker[1][0])
    print('焦距(focalLength) = ', focalLength)        # 打印焦距的值
    return focalLength


# 计算摄像头到物体的距离
def calculate_Distance(image_path, focalLength_value):
    image = cv2.imread(image_path)
    # cv2.imshow("原图", image)
    marker = find_marker(image)     # 获取矩形的中心点坐标，长度，宽度和旋转角度， marke[1][0]代表宽度
    distance_inches = distance_to_camera(KNOWN_WIDTH, focalLength_value, marker[1][0])
    box = cv2.boxPoints(marker)
    # print("Box = ", box)
    box = np.int0(box)
    print("Box = ", box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)      # 绘制物体轮廓
    cv2.putText(image, "%.2fcm" % (distance_inches * 2.54), (image.shape[1] - 300, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    cv2.imshow("单目测距", image)

if __name__ == "__main__":
    img_path = "Picture1.jpg"
    focalLength = calculate_focalDistance(img_path)

    for image_path in IMAGE_PATHS:
        calculate_Distance(image_path, focalLength)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
```

##### .5. 双目测距

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726124419575.png)

#### 2. pykinect2

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610193508669.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610193650032.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610193928519.png)

- read_stream.py

> 展示并存储RGB以及深度图片信息，这里存在问题；
>
> ```python
> depthframesaveformat = np.copy(np.ctypeslib.as_array(depthframeD, shape=(kinect._depth_frame_data_capacity.value,)))
> pickle.dump(depthframesaveformat, depthfile)
> ```
>
> 

- [mapper.py](https://github.com/KonstantinosAng/PyKinect2-Mapper-Functions/blob/master/mapper.py)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610195517769.png)

#### 2. pylibfreenect2

```python
import numpy as np
import open3d as o3d
import cv2
import time
import sys
import datetime

# 导入 pylibfreenect2 库
import pylibfreenect2 as freenect2
from pylibfreenect2 import FrameType, Registration, FrameListener

from test_resnetv2 import *


# 定义颜色常量
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def get_hand_3d_coordinates():
    # 初始化 Kinect 传感器
    fn = freenect2.Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit()
    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial)
    types = FrameType.Ir | FrameType.Depth | FrameType.Color
    listener = FrameListener()
    device.setIrAndDepthFrameListener(listener)
    device.setColorFrameListener(listener)
    device.startStreams(*types)

    # 初始化变量
    hand_3d_coordinates = []

    # 初始化 Open3D 点云
    pcd = o3d.geometry.PointCloud()

    # 初始化 Kinect 注册器
    registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())
    
    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
    n = 0
    # 循环处理帧数据
    while True:
        # 获取深度数据
        frames = listener.waitForNewFrame()
        depth = frames["depth"]

        # 获取彩色图像
        color = frames["color"]

        # 将深度和彩色图像进行注册
        undistorted = FrameType.Depth
        registered = FrameType.Color
        registration.apply(color, depth, undistorted, registered)


        depth_pixel = depth.asarray() / 20
        over_index = depth_pixel > 72
        depth_pixel[over_index] = 0
        cv2.imwrite('/home/robot/Desktop/rh_workspace/workcoatdtest/data/input/0.png', depth_pixel)
        onepiece_data = test_resnetv2.testSingleImagefile

        # 获取相机姿态
        pose = registration.getRegistration()

        # 将彩色图像转换为 OpenCV 图像
        color_data = registered.asarray(np.uint8)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        # 获取深度图像的宽度和高度
        height, width = depth.height, depth.width

        # 创建空白图像
        blank_image = np.zeros((height, width, 3), np.uint8)

        # 阈值化深度图像
        threshold = 2000
        ret, thresholded = cv2.threshold(depth.asarray(np.float32), threshold, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓
        max_contour = None
        max_contour_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_contour_area:
                max_contour = contour
                max_contour_area = area

        # 如果找到了手部轮廓
        if max_contour is not None:
            # 获取手部轮廓的凸包
            hull = cv2.convexHull(max_contour)

            # 画出手部轮廓和凸包
            cv2.drawContours(color_data, [max_contour], 0, GREEN, 2)
            cv2.drawContours(color_data, [hull], 0, BLUE, 2)

            # 获取凸包的顶点
            hull_points = np.squeeze(hull)

            # 将凸包顶点转换为相机坐标系下的 3D 坐标
            points = []
            for point in hull_points:
                x = point[0]
                y = point[1]
                xyz = registration.getPointXYZ(depth, point[1], point[0])
                points.append(xyz)

            # 将点云数据转换为 Numpy 数组
            points = np.array(points)

            # 更新 Open3D 点云
            pcd.points = o3d.utility.Vector3dVector(points)

            # 显示 Open3D 点云
            o3d.visualization.draw_geometries([pcd])

            # 将手部 3D 坐标添加到列表中
            hand_3d_coordinates = points.tolist()

        # 显示彩色图像
        cv2.imshow("Color", color_data)
        n += 1
        # 等待按下 ESC 键退出程序
        key = cv2.waitKey(delay=16)
        if n > 0:
            break
        if key == ord('q'):
            break

        # 释放帧数据
        listener.release(frames)

    # 停止 Kinect 传感器
    device.stop()
    device.close()

    # 关闭所有窗口
    cv2.destroyAllWindows()

    # 返回肩部坐标， 手部 3D 坐标列表
    return onepiece_data, hand_3d_coordinates


def test_get_hand_3d_coordinates():
    # 调用 get_hand_3d_coordinates 函数
    onepiece_data, hand_3d_coordinates = get_hand_3d_coordinates()

    # 打印手部 3D 坐标
    print("Hand 3D Coordinates:")
    print(hand_3d_coordinates)

    # 打印肩部 3D 坐标
    print("Hand 3D Coordinates:")
    print(hand_3d_coordinates)

    # 如果手部 3D 坐标不为空，则绘制点云
    if hand_3d_coordinates:
        # 创建 Open3D 点云
        pcd = o3d.geometry.PointCloud()

        # 将手部 3D 坐标转换为 Numpy 数组
        points = np.array(hand_3d_coordinates)

        # 更新 Open3D 点云
        pcd.points = o3d.utility.Vector3dVector(points)

        # 显示 Open3D 点云
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    test_get_hand_3d_coordinates()
```



#### 3. C++ kinect2

```c++
#include <Windows.h>
#include <iostream>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef struct structBGR {
    BYTE blue;
    BYTE green;
    BYTE red;
    BYTE player;
} BGR;

bool tracked[NUI_SKELETON_COUNT] = { FALSE };
cv::Point skeletonPoint[NUI_SKELETON_COUNT][NUI_SKELETON_POSITION_COUNT] = { cv::Point(0, 0) };
cv::Point colorPoint[NUI_SKELETON_COUNT][NUI_SKELETON_POSITION_COUNT] = { cv::Point(0, 0) };

void getColorImage(HANDLE & colorStreamHandle, cv::Mat & colorImg);
BGR Depth2RGB(USHORT depthID);
void getDepthImage(HANDLE & depthStreamHandle, cv::Mat & depthImg, cv::Mat & mask);
void drawSkeleton(cv::Mat &img, cv::Point pointSet[], int which_one);
void getSkeletonImage(cv::Mat & skeletonImg, cv::Mat & colorImg);

int main(int argc, char* argv[])
{
    cv::Mat colorImg;
    colorImg.create(480, 640, CV_8UC3);
    cv::Mat depthImg;
    depthImg.create(240, 320, CV_8UC3);
    cv::Mat skeletonImg;
    skeletonImg.create(240, 320, CV_8UC3);
    cv::Mat mask;
    mask.create(240, 320, CV_8UC3);

    HANDLE colorEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    HANDLE depthEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    HANDLE skeletonEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    HANDLE colorStreamHandle = NULL;
    HANDLE depthStreamHandle = NULL;

    HRESULT hr;

    hr = NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX
        | NUI_INITIALIZE_FLAG_USES_SKELETON);
    if (FAILED(hr))
    {
        cout << "Nui initialize failed." << endl;
        return hr;
    }

    hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, colorEvent, &colorStreamHandle);
    if (FAILED(hr))
    {
        cout << "Can not open color stream." << endl;
        return hr;
    }

    hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, depthEvent, &depthStreamHandle);
    if (FAILED(hr))
    {
        cout << "Can not open depth stream." << endl;
        return hr;
    }

    hr = NuiSkeletonTrackingEnable(skeletonEvent, 0);
    if (FAILED(hr))
    {
        cout << "Can not enable skeleton tracking." << endl;
        return hr;
    }

    cv::namedWindow("mask", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("colorImg", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("depthImg", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("skeletonImg", CV_WINDOW_AUTOSIZE);

    while (1)
    {
        if (WaitForSingleObject(colorEvent, 0) == 0)
        {
            getColorImage(colorStreamHandle, colorImg);
        }

        if (WaitForSingleObject(depthEvent, 0) == 0)
        {
            getDepthImage(depthStreamHandle, depthImg, mask);
        }

        if (WaitForSingleObject(skeletonEvent, 0) == 0)
        {
            getSkeletonImage(skeletonImg, colorImg);
        }

        cv::imshow("mask", mask);
        cv::imshow("colorImg", colorImg);
        cv::imshow("depthImg", depthImg);
        cv::imshow("skeletonImg", skeletonImg);


        if (cv::waitKey(20) == 27)
            break;
    }

    NuiShutdown();
    cv::destroyAllWindows();


    return 0;
}

void getColorImage(HANDLE & colorStreamHandle, cv::Mat & colorImg)
{
    const NUI_IMAGE_FRAME * pImageFrame = NULL;

    HRESULT hr = NuiImageStreamGetNextFrame(colorStreamHandle, 0, &pImageFrame);
    if (FAILED(hr))
    {
        cout << "Could not get color image" << endl;
        NuiShutdown();
        return;
    }

    INuiFrameTexture * pTexture = pImageFrame->pFrameTexture;
    NUI_LOCKED_RECT LockedRect;

    pTexture->LockRect(0, &LockedRect, NULL, 0);

    if (LockedRect.Pitch != 0)
    {
        for (int i = 0; i < colorImg.rows; i++)
        {
            uchar *ptr = colorImg.ptr<uchar>(i);    //第i行的指针

                                            //每个字节代表一个颜色信息，直接使用uchar
            uchar *pBuffer = (uchar*)(LockedRect.pBits) + i * LockedRect.Pitch;
            for (int j = 0;j < colorImg.cols;j++)
            {
                //内部数据是4个字节，0-1-2是BGR，第4个现在未使用
                ptr[3 * j] = pBuffer[4 * j];
                ptr[3 * j + 1] = pBuffer[4 * j + 1];
                ptr[3 * j + 2] = pBuffer[4 * j + 2];
            }
        }
    }
    else
    {
        cout << "捕获彩色图像出错" << endl;
    }

    pTexture->UnlockRect(0);
    NuiImageStreamReleaseFrame(colorStreamHandle, pImageFrame);
}

// 处理深度数据的每一个像素，如果属于同一个用户的ID，那么像素就标为同种颜色，不同的用户，  
// 其ID不一样，颜色的标示也不一样，如果不属于某个用户的像素，那么就采用原来的深度值
BGR Depth2RGB(USHORT depthID)
{
    //每像素共16bit的信息，其中最低3位是ID（所捕捉到的人的ID），剩下的13位才是信息 
    USHORT realDepth = (depthID & 0xfff8) >> 3; //深度信息，高13位
    USHORT player = depthID & 0x0007;   //提取用户ID信息，低3位

                                        //因为提取的信息是距离信息，为了便于显示，这里归一化为0-255
    BYTE depth = (BYTE)(255 * realDepth / 0x1fff);

    BGR color_data;
    color_data.blue = color_data.green = color_data.red = 0;

    color_data.player = player;

    //RGB三个通道的值都是相等的话，就是灰度的  
    //Kinect系统能够处理辨识传感器前多至6个人物的信息，但同一时刻最多只有2个玩家可被追踪（即骨骼跟踪）
    switch (player)
    {
    case 0:
        color_data.blue = depth / 2;
        color_data.green = depth / 2;
        color_data.red = depth / 2;
        break;
    case 1:
        color_data.blue = depth;
        break;
    case 2:
        color_data.green = depth;
        break;
    case 3:
        color_data.red = depth;
        break;
    case 4:
        color_data.blue = depth;
        color_data.green = depth;
        color_data.red = depth / 4;
        break;
    case 5:
        color_data.blue = depth;
        color_data.green = depth / 4;
        color_data.red = depth;
        break;
    case 6:
        color_data.blue = depth / 4;
        color_data.green = depth;
        color_data.red = depth;
        break;
    }

    return color_data;
}

void getDepthImage(HANDLE & depthStreamHandle, cv::Mat & depthImg, cv::Mat & mask)
{
    const NUI_IMAGE_FRAME * pImageFrame = NULL;
    HRESULT hr = NuiImageStreamGetNextFrame(depthStreamHandle, 0, &pImageFrame);
    if (FAILED(hr))
    {
        cout << "Could not get depth image" << endl;
        NuiShutdown();
        return;
    }

    INuiFrameTexture * pTexture = pImageFrame->pFrameTexture;
    NUI_LOCKED_RECT LockedRect;

    pTexture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0)
    {
        for (int i = 0;i < depthImg.rows;i++)
        {
            uchar * ptr = depthImg.ptr<uchar>(i);
            uchar * ptr_mask = mask.ptr<uchar>(i);

            uchar *pBufferRun = (uchar*)(LockedRect.pBits) + i * LockedRect.Pitch;
            USHORT * pBuffer = (USHORT*)pBufferRun;

            for (int j = 0;j < depthImg.cols;j++)
            {
                // ptr[j] = 255 - (uchar)(255 * pBuffer[j] / 0x0fff);   //直接将数据归一化处理
                // ptr[j] = (uchar)(255 * pBuffer[j] / 0x0fff); //直接将数据归一化处理

                BGR rgb = Depth2RGB(pBuffer[j]);
                ptr[3 * j] = rgb.blue;
                ptr[3 * j + 1] = rgb.green;
                ptr[3 * j + 2] = rgb.red;

                switch (rgb.player)
                {
                case 0:
                    ptr_mask[3 * j] = 0;
                    ptr_mask[3 * j + 1] = 0;
                    ptr_mask[3 * j + 2] = 0;
                    break;
                case 1:
                    ptr_mask[3 * j] = 255;
                    ptr_mask[3 * j + 1] = 0;
                    ptr_mask[3 * j + 2] = 0;
                    break;
                case 2:
                    ptr_mask[3 * j] = 0;
                    ptr_mask[3 * j + 1] = 255;
                    ptr_mask[3 * j + 2] = 0;
                    break;
                case 3:
                    ptr_mask[3 * j] = 0;
                    ptr_mask[3 * j + 1] = 0;
                    ptr_mask[3 * j + 2] = 255;
                    break;
                case 4:
                    ptr_mask[3 * j] = 255;
                    ptr_mask[3 * j + 1] = 255;
                    ptr_mask[3 * j + 2] = 0;
                    break;
                case 5:
                    ptr_mask[3 * j] = 255;
                    ptr_mask[3 * j + 1] = 0;
                    ptr_mask[3 * j + 2] = 255;
                    break;
                case 6:
                    ptr_mask[3 * j] = 0;
                    ptr_mask[3 * j + 1] = 255;
                    ptr_mask[3 * j + 2] = 255;
                    break;
                default:
                    ptr_mask[3 * j] = 0;
                    ptr_mask[3 * j + 1] = 0;
                    ptr_mask[3 * j + 2] = 0;
                    break;
                }
            }
        }
    }
    else
    {
        cout << "捕获深度图像出错" << endl;
    }

    pTexture->UnlockRect(0);
    NuiImageStreamReleaseFrame(depthStreamHandle, pImageFrame);

}

void drawSkeleton(cv::Mat &img, cv::Point pointSet[], int which_one)
{
    cv::Scalar color;
    switch (which_one)
    {
    case 0:
        color = cv::Scalar(255, 0, 0);
        break;
    case 1:
        color = cv::Scalar(0, 255, 0);
        break;
    case 2:
        color = cv::Scalar(0, 0, 255);
        break;
    case 3:
        color = cv::Scalar(255, 255, 0);
        break;
    case 4:
        color = cv::Scalar(255, 0, 255);
        break;
    case 5:
        color = cv::Scalar(0, 255, 255);
        break;
    }

    // 脊柱
    if ((pointSet[NUI_SKELETON_POSITION_HEAD].x != 0 || pointSet[NUI_SKELETON_POSITION_HEAD].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_HEAD], pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_SPINE].x != 0 || pointSet[NUI_SKELETON_POSITION_SPINE].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER], pointSet[NUI_SKELETON_POSITION_SPINE], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_SPINE].x != 0 || pointSet[NUI_SKELETON_POSITION_SPINE].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_HIP_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_CENTER].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_SPINE], pointSet[NUI_SKELETON_POSITION_HIP_CENTER], color, 2);

    // 左上肢
    if ((pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_SHOULDER_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER], pointSet[NUI_SKELETON_POSITION_SHOULDER_LEFT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_SHOULDER_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_LEFT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_ELBOW_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_ELBOW_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_SHOULDER_LEFT], pointSet[NUI_SKELETON_POSITION_ELBOW_LEFT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_ELBOW_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_ELBOW_LEFT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_WRIST_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_WRIST_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_ELBOW_LEFT], pointSet[NUI_SKELETON_POSITION_WRIST_LEFT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_WRIST_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_WRIST_LEFT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_HAND_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_HAND_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_WRIST_LEFT], pointSet[NUI_SKELETON_POSITION_HAND_LEFT], color, 2);

    // 右上肢
    if ((pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_SHOULDER_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_SHOULDER_CENTER], pointSet[NUI_SKELETON_POSITION_SHOULDER_RIGHT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_SHOULDER_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_SHOULDER_RIGHT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_ELBOW_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_ELBOW_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_SHOULDER_RIGHT], pointSet[NUI_SKELETON_POSITION_ELBOW_RIGHT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_ELBOW_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_ELBOW_RIGHT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_WRIST_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_WRIST_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_ELBOW_RIGHT], pointSet[NUI_SKELETON_POSITION_WRIST_RIGHT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_WRIST_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_WRIST_RIGHT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_HAND_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_HAND_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_WRIST_RIGHT], pointSet[NUI_SKELETON_POSITION_HAND_RIGHT], color, 2);

    // 左下肢
    if ((pointSet[NUI_SKELETON_POSITION_HIP_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_CENTER].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_HIP_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_HIP_CENTER], pointSet[NUI_SKELETON_POSITION_HIP_LEFT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_HIP_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_LEFT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_KNEE_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_KNEE_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_HIP_LEFT], pointSet[NUI_SKELETON_POSITION_KNEE_LEFT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_KNEE_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_KNEE_LEFT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_ANKLE_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_ANKLE_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_KNEE_LEFT], pointSet[NUI_SKELETON_POSITION_ANKLE_LEFT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_ANKLE_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_ANKLE_LEFT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_FOOT_LEFT].x != 0 || pointSet[NUI_SKELETON_POSITION_FOOT_LEFT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_ANKLE_LEFT], pointSet[NUI_SKELETON_POSITION_FOOT_LEFT], color, 2);

    // 右下肢
    if ((pointSet[NUI_SKELETON_POSITION_HIP_CENTER].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_CENTER].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_HIP_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_HIP_CENTER], pointSet[NUI_SKELETON_POSITION_HIP_RIGHT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_HIP_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_HIP_RIGHT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_KNEE_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_KNEE_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_HIP_RIGHT], pointSet[NUI_SKELETON_POSITION_KNEE_RIGHT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_KNEE_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_KNEE_RIGHT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_ANKLE_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_ANKLE_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_KNEE_RIGHT], pointSet[NUI_SKELETON_POSITION_ANKLE_RIGHT], color, 2);

    if ((pointSet[NUI_SKELETON_POSITION_ANKLE_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_ANKLE_RIGHT].y != 0) &&
        (pointSet[NUI_SKELETON_POSITION_FOOT_RIGHT].x != 0 || pointSet[NUI_SKELETON_POSITION_FOOT_RIGHT].y != 0))
        cv::line(img, pointSet[NUI_SKELETON_POSITION_ANKLE_RIGHT], pointSet[NUI_SKELETON_POSITION_FOOT_RIGHT], color, 2);
}

void getSkeletonImage(cv::Mat & skeletonImg, cv::Mat & colorImg)
{
    NUI_SKELETON_FRAME skeletonFrame = { 0 };   //骨骼帧的定义 
    bool foundSkeleton = false;

    HRESULT hr = NuiSkeletonGetNextFrame(0, &skeletonFrame);
    if (SUCCEEDED(hr))
    {
        //NUI_SKELETON_COUNT是检测到的骨骼数（即，跟踪到的人数）
        for (int i = 0;i < NUI_SKELETON_COUNT;i++)
        {
            NUI_SKELETON_TRACKING_STATE trackingState = skeletonFrame.SkeletonData[i].eTrackingState;

            // Kinect最多检测到6个人，但只能跟踪2个人的骨骼，再检查是否跟踪到了
            if (trackingState == NUI_SKELETON_TRACKED)
            {
                foundSkeleton = true;
            }
        }
    }

    if (!foundSkeleton)
    {
        return;
    }

    NuiTransformSmooth(&skeletonFrame, NULL);
    skeletonImg.setTo(0);

    for (int i = 0;i < NUI_SKELETON_COUNT;i++)
    {
        // 判断是否是一个正确骨骼的条件：骨骼被跟踪到并且肩部中心（颈部位置）必须跟踪到
        if (skeletonFrame.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED && skeletonFrame.SkeletonData[i].eSkeletonPositionTrackingState[NUI_SKELETON_POSITION_SHOULDER_CENTER] != NUI_SKELETON_POSITION_NOT_TRACKED)
        {
            float fx, fy;
            // 拿到所有跟踪到的关节点的坐标，并转换为我们的深度空间的坐标，因为我们是在深度图像中  
            // 把这些关节点标记出来的  
            // NUI_SKELETON_POSITION_COUNT为跟踪到的一个骨骼的关节点的数目，为20
            for (int j = 0;j < NUI_SKELETON_POSITION_COUNT;j++)
            {
                NuiTransformSkeletonToDepthImage(skeletonFrame.SkeletonData[i].SkeletonPositions[j], &fx, &fy);
                skeletonPoint[i][j].x = (int)fx;
                skeletonPoint[i][j].y = (int)fy;
            }

            for (int j = 0;j < NUI_SKELETON_POSITION_COUNT;j++)
            {
                if (skeletonFrame.SkeletonData[i].eSkeletonPositionTrackingState[j] != NUI_SKELETON_POSITION_NOT_TRACKED)
                {
                    cv::circle(skeletonImg, skeletonPoint[i][j], 3, cv::Scalar(0, 255, 255), 1, 8, 0);
                    tracked[i] = true;

                    // 在彩色图中也绘制骨骼关键点
                    LONG color_x, color_y;
                    NuiImageGetColorPixelCoordinatesFromDepthPixel(NUI_IMAGE_RESOLUTION_640x480, 0,
                        skeletonPoint[i][j].x, skeletonPoint[i][j].y, 0, &color_x, &color_y);
                    colorPoint[i][j].x = (int)color_x;
                    colorPoint[i][j].y = (int)color_y;
                    cv::circle(colorImg, colorPoint[i][j], 4, cv::Scalar(0, 255, 255), 1, 8, 0);
                }
            }

            drawSkeleton(skeletonImg, skeletonPoint[i], i);
            drawSkeleton(colorImg, colorPoint[i], i);
        }
    }

}
```

#### 4. Project

- [Depth-Mask-RCNN](https://github.com/SuperBadCode/Depth-Mask-RCNN)
- **[ PyKinect2-PyQtGraph-PointClouds](https://github.com/KonstantinosAng/PyKinect2-PyQtGraph-PointClouds)**
- c++识别人体关键点：https://blog.csdn.net/baolinq/article/details/52373574
- c++识别人体关键点：https://blog.csdn.net/hongbin_xu/article/details/80896424
- kinect bodyindex, color, depth绘制： https://blog.csdn.net/hongbin_xu/article/details/80907403



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kinect2relative/  

