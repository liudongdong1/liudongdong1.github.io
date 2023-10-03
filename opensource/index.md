# OpenSource


#### 1. 	PySOT

> **PySOT** is a software system designed by SenseTime Video Intelligence Research team. It implements state-of-the-art single object tracking algorithms, including [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) and [SiamMask](https://arxiv.org/abs/1812.05050). It is written in Python and powered by the [PyTorch](https://pytorch.org) deep learning framework. This project also contains a Python port of toolkit for evaluating trackers.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/bag_demo.gif)

The goal of PySOT is to provide a high-quality, high-performance codebase for visual tracking *research*. It is designed to be flexible in order to support rapid implementation and evaluation of novel research. PySOT includes implementations of the following visual tracking algorithms:

- [SiamMask](https://arxiv.org/abs/1812.05050)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [SiamFC](https://arxiv.org/abs/1606.09549)

using the following backbone network architectures:

- [ResNet{18, 34, 50}](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

#### 2.  Darknet

- Paper Yolo v4: https://arxiv.org/abs/2004.10934

- More details: http://pjreddie.com/darknet/yolo/

#### 3. [MSRocket](http://aka.ms/rocket)

>  Rocket (http://aka.ms/rocket), a hybrid edge-cloud live video analytics software stack built on C# .NET Core, and introduce five different pipelines:How to setup and run the video analytics system
>
>  - Pipeline 1: Alerting on objects
>  - Pipeline 2: Detecting objects with cheap filters, and after-the-fact querying
>  - Pipeline 3: Detecting objects with cascaded DNNs
>  - Pipeline 4: Edge/Cloud split
>  - Pipeline 5: Edge/Cloud split + containers

代码学习进度：

1. 阅读了代码，了解了基本工作过程，

2. 提供了一些深度学习编译好的动态链接库，  (这里我搭建时出现 dll 加载格式不正确，我同学搭建正确，问题还没有解决）

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200418095136902.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)


​     ![img](https://img-blog.csdnimg.cn/20200417110721630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdWRvbmdkb25nMTk=,size_16,color_FFFFFF,t_70)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

​       ![img](https://img-blog.csdnimg.cn/20200417110744289.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdWRvbmdkb25nMTk=,size_16,color_FFFFFF,t_70)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210312222554975.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210312222726190.png)

#### 4. videoAnalyse

- 该代码使用yolo模型检测物体框；
- 通过计算新的目标框与之前存储目标框中的中性点位置来判断是不是新的待追踪物体。
- 刚开始通过鼠标选择监控物体窗口大小；



#### 5. [banodet](https://github.com/RangiLyu/nanodet)

>  Super fast and lightweight anchor-free object detection model. Real-time on mobile devices.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210321142731.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210321143238.png)

#### 6.  [quickDraw](https://github.com/uvipen/QuickDraw)

- 该项目使用Opencv 预定义颜色来追踪物品的轨迹，并将轨迹显示在canvas上，通过深度学习模型学习绘制的是什么形状；
- 如果扩展的或可以使用手部关键点追踪算法，聚标追踪算法；
- model

```python
import torch
import torch.nn as nn
from math import pow

class QuickDraw(nn.Module):
    def __init__(self, input_size = 28, num_classes = 15):
        super(QuickDraw, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        dimension = int(64 * pow(input_size/4 - 3, 2))
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
```

- 颜色追踪

```python
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

# from src.dataset import CLASSES
from src.config import *
from src.utils import get_images, get_overlay


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-c", "--color", type=str, choices=["green", "blue", "red"], default="green",
                        help="Color which could be captured by camera and seen as pointer")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-d", "--display", type=int, default=3, help="How long is prediction shown in second(s)")
    parser.add_argument("-s", "--canvas", type=bool, default=False, help="Display black & white canvas")
    args = parser.parse_args()
    return args


def main(opt):
    # Define color range
    if opt.color == "red":  # We shouldn't use red as color for pointer, since it
        # could be confused with our skin's color under some circumstances
        color_lower = np.array(RED_HSV_LOWER)
        color_upper = np.array(RED_HSV_UPPER)
        color_pointer = RED_RGB
    elif opt.color == "green":
        color_lower = np.array(GREEN_HSV_LOWER)
        color_upper = np.array(GREEN_HSV_UPPER)
        color_pointer = GREEN_RGB
    else:
        color_lower = np.array(BLUE_HSV_LOWER)
        color_upper = np.array(BLUE_HSV_UPPER)
        color_pointer = BLUE_RGB

    # Initialize deque for storing detected points and canvas for drawing
    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Load the video from camera (Here I use built-in webcam)
    camera = cv2.VideoCapture(0)
    is_drawing = False
    is_shown = False

    # Load images for classes:
    class_images = get_images("images", CLASSES)
    predicted_class = None

    # Load model
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=512)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False
        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # Blur image
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    # Check if the largest contour satisfy the condition of minimum area
                    if cv2.contourArea(contour) > opt.area:
                        x, y, w, h = cv2.boundingRect(contour)
                        image = canvas_gs[y:y + h, x:x + w]
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, None, :, :]
                        image = torch.from_numpy(image)
                        logits = model(image)
                        predicted_class = torch.argmax(logits[0])
                        # print (CLASSES[predicted_class])
                        is_shown = True
                    else:
                        print("The object drawn is too small. Please draw a bigger one!")
                        points = deque(maxlen=512)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Read frame from camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        # Detect pixels fall within the pre-defined color range. Then, blur the image
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check to see if any contours are found
        if len(contours):
            # Take the biggest contour, since it is possible that there are other objects in front of camera
            # whose color falls within the range of our pre-defined color
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)                        #通过找最大的轮廓，这里可能会存在一些问题;这里可以结合相关的追踪算法进行处理
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], WHITE_RGB, 5)    #这个canvas 是后来记录绘制的图形
                    cv2.line(frame, points[i - 1], points[i], color_pointer, 2)

        if is_shown:
            cv2.putText(frame, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)
            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60,60))


        cv2.imshow("Camera", frame)
        if opt.canvas:
            cv2.imshow("Canvas", 255-canvas)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)

```

- 鼠标绘制代码

```python
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

# from src.dataset import CLASSES
from src.config import *
from src.utils import get_images, get_overlay


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-c", "--color", type=str, choices=["green", "blue", "red"], default="green",
                        help="Color which could be captured by camera and seen as pointer")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-d", "--display", type=int, default=3, help="How long is prediction shown in second(s)")
    parser.add_argument("-s", "--canvas", type=bool, default=False, help="Display black & white canvas")
    args = parser.parse_args()
    return args


def main(opt):
    # Define color range
    if opt.color == "red":  # We shouldn't use red as color for pointer, since it
        # could be confused with our skin's color under some circumstances
        color_lower = np.array(RED_HSV_LOWER)
        color_upper = np.array(RED_HSV_UPPER)
        color_pointer = RED_RGB
    elif opt.color == "green":
        color_lower = np.array(GREEN_HSV_LOWER)
        color_upper = np.array(GREEN_HSV_UPPER)
        color_pointer = GREEN_RGB
    else:
        color_lower = np.array(BLUE_HSV_LOWER)
        color_upper = np.array(BLUE_HSV_UPPER)
        color_pointer = BLUE_RGB

    # Initialize deque for storing detected points and canvas for drawing
    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Load the video from camera (Here I use built-in webcam)
    camera = cv2.VideoCapture(0)
    is_drawing = False
    is_shown = False

    # Load images for classes:
    class_images = get_images("images", CLASSES)
    predicted_class = None

    # Load model
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=512)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False
        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # Blur image
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    # Check if the largest contour satisfy the condition of minimum area
                    if cv2.contourArea(contour) > opt.area:
                        x, y, w, h = cv2.boundingRect(contour)
                        image = canvas_gs[y:y + h, x:x + w]
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, None, :, :]
                        image = torch.from_numpy(image)
                        logits = model(image)
                        predicted_class = torch.argmax(logits[0])
                        # print (CLASSES[predicted_class])
                        is_shown = True
                    else:
                        print("The object drawn is too small. Please draw a bigger one!")
                        points = deque(maxlen=512)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Read frame from camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        # Detect pixels fall within the pre-defined color range. Then, blur the image
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check to see if any contours are found
        if len(contours):
            # Take the biggest contour, since it is possible that there are other objects in front of camera
            # whose color falls within the range of our pre-defined color
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)                        #通过找最大的轮廓，这里可能会存在一些问题;这里可以结合相关的追踪算法进行处理
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], WHITE_RGB, 5)    #这个canvas 是后来记录绘制的图形
                    cv2.line(frame, points[i - 1], points[i], color_pointer, 2)

        if is_shown:
            cv2.putText(frame, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)
            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60,60))


        cv2.imshow("Camera", frame)
        if opt.canvas:
            cv2.imshow("Canvas", 255-canvas)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)

```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/quickdraw.gif)

#### 7. video-object-remove

> 使用pysot库中 [SiamMask](https://github.com/foolwood/SiamMask) 对每个图片处理，找到每个图片的mask不存储到目录中，然后使用 [Deep-Video-Inpainting](https://github.com/mcahny/Deep-Video-Inpainting) 工具对图片进行恢复处理。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/skate.gif)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/opensource/  

