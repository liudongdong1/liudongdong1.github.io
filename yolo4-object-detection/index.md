# YOLO4 Object Detection


- Yolo5 自定义数据检测教程： https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
- google云盘教程： https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb#scrollTo=V0AJnSeCIHyJ

### 1. [Darknet](https://github.com/hgpvision/darknet)

​         darknet是一个较为轻型的完全基于C与CUDA的开源深度学习框架，其主要特点就是容易安装，没有任何依赖项（OpenCV都可以不用），移植性非常好，支持CPU与GPU两种计算方式。

相比于TensorFlow来说，darknet并没有那么强大，但这也成了darknet的优势：

1. darknet完全由C语言实现，没有任何依赖项;
2. darknet支持CPU（所以没有GPU也不用紧的）与GPU（CUDA/cuDNN，使用GPU当然更块更好了）；
3. 正是因为其较为轻型，没有像TensorFlow那般强大的API，所以给我的感觉就是有另一种味道的灵活性，适合用来研究底层，可以更为方便的从底层对其进行改进与扩展；
4. darknet的实现与caffe的实现存在相似的地方，熟悉了darknet，相信对上手caffe有帮助；

### 2. Yolo One-stage Detect

#### 2.1 目录结构

<div class="center">
    <div style="float:left; width:80%">
        <p>1.cfg：模型的架构，每个cfg文件类似与caffe的prototxt文件，通过该文件定义的整个模型的架构;<br>
            2.data：label文件，如coco9k的类别名等，和一些样例图（该文件夹主要为演示用，或者是直接训练coco等对应数据集时有用，如果要用自己的数据自行训练，该文件夹内的东西都不是我们需要的）<br>
            3.src：最底层的框架定义文件，所有层的定义等最基本的函数全部在该文件夹内。<br>
            4.examples：高层的一些函数，如检测函数，识别函数等，这些函数直接调用了底层的函数，我们经常使用的就是example中的函数；<br>
            5.include：头文件存放darknet 上传api接口<br>
            6.python：python对模型的调用方法，基本都在darknet.py中。当然，要实现python的调用，还需要用到darknet的动态库libdarknet.so；<br>
            7.scripts：一些脚本，如下载coco数据集，将voc格式的数据集转换为训练所需格式的脚本等<br>
		</p>
	</div>
    <div style="float:right; width:20%; height:120%"><img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200426191818182.png"></div>
</div>























其中：cfg文件：中括号+网络层的名字定义当前网络属于什么层。输入图像的信息；训练过程用到的信息；数据增强的一些信息；

<center class="half">
    <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200426193451234.png" width=35%/><img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200426193954588.png" width=35%/>
</center>


#### 2.2 目标检测的核心函数

YOLOv3的目标检测源代码，核心的函数包括：

detect_image() ;       generate();        yolo_eval();       yolo_model(); 

yolo_boxes_and_scores();     yolo_head();         yolo_correct_boxes()

#### 2.3 目标检测流程

YOLOv3的目标识别的源代码流程大致如下：

（1）设置缺省值并初始化：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514101324454.png)

（2）detect_image()将图片缩放成416x416大小，调用<font color=red>yolo_model()，生成13x13、26x26与52x52等3个feature map的输出</font>，对这3个feature map进行预测，调用<font color=red>yolo_eval()函数得到目标框、目标框得分和类别</font>，然后使用Pillow对发现的每一类对象的每一个目标框，绘制标签、框和文字：

###### ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514101345475.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514101414591.png)

（3）在yolo_eval()函数中调用<font color=red>yolo_boxes_and_scores()得到目标框、目标框得分和类别</font>。而在yolo_boxes_and_scores()函数中，先调用yolo_head()函数计算每一个网格的目标对象的中心点坐标box_xy和目标框的宽与高box_wh，以及目标框的置信度box_confidence和类别置信度box_class_probs；然后调用<font color=red>yolo_correct_boxes（），将box_xy, box_wh转换为输入图片上的真实坐标，输出boxes是框的左下、右上两个坐标(y_min, x_min, y_max, x_max)：</font>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514101442195.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514101458697.png)

（4）完整的代码流程图如下图所示：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514101559843.png)

#### 2.4 Darknet 安装使用

```shell
mkdir build-release
cd build-release
cmake ..
make
make install
#test install
`./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights`
#YOLO2
$ ./darknet detector test ./cfg/voc.data ./cfg/yolov2.cfg ./yolov2.weights data/dog.jpg -i 0 -thresh 0.2
#Yolo3
$ ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights data/dog.jpg -i 0 -thresh 0.25
$ ./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output
```

#### 2.5 Python Demo

```python
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()

```

### 3. Papar Record

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200426200103526.png)

**level**: CVPR  CCF_A
**author**:  Zhi Zhang, Tong He     amazon web services
**date**: 2019.2.15
**keyword**:

- yolo4, object detection

------

# Paper: YOLO4

<div align=center>
<br/>
<b>Bag of Freebies for Training Object Detection Neural Networks</b>
</div>




#### Summary

1. recognize the special property of multiple object detection task which favors spatial preserving transfroms, and thus proposed a visually coherent image mixup methods for object detection tasks.
2. explore detailed training pipelines including learning rate scheduling weight decay and synchronized BatchNorm.
3. learn some training tricks in AI.

#### Proble Statement

previous work:

- Scattering tricks from Image Classification:
  - Learning rate warm up heuristic: to overcome the negative effective of extremely large mini-batch size.
  - Label smoothing: modifies the hard ground truth labeling in cross entropyloss.
  - Mixup strategies: alleviate adversarial perturbation
  - Cosine annealing strategy : for learning rate decay compared to  traditional step policy.
  - Bag of tricks:  some training strategies.

#### Methods

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200426201633444.png)

**【Function1】Visually Coherant Image Mixup for object detection**

- continues increasing the blending ratio used in the mixup, the objects in resulting frames are more vibrant and coherent to the natural  presentations.
- use geometry preserved alignment for image mixup to avoid distort image at the initial steps.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200425103925238.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200425104052552.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200425104025525.png)

【**Function 2**】 **Classification head label smoothing**

- traditional detection networks often compute a probability distribution over all classes with softmax function:$p_i=e^{z_i}/\sum_je^{z_j}$, $z_is$ are the unnormalized logits directly from last linear layer for classification prediction. $p$ :ouput distibution, $q$: the ground truth distribution, often a one-hot distribution, causing $L=\sum_iq_ilogp_i$  to be too confident in its predictions and is prone to over-fitting.
- Label smoothing as a form of regularization is : $q_i'=(1-\epsilon)q_i+\epsilon/K$, $K$ is the total number of classes, $\epsilon$ is a small constant.

**【Function 3】** **Data Pre-processing**

1. **Random geometry transformation**： random cropping(with constraints), random expansion, random horizontal flip and random resize(with random interpolation)
2. **Random color jittering including**: brightness, hue, saturation, and contrast.

- For image classification domain, the network is extremely tolerant to image geometrical transformation.
- single-stage detector network: the final ouput are generated from every single pixel on feature map.
- multi-stage proposal and sampling based approached:  many ROI candidates are generated and sampled with a fixed number, needless for extensive geometric augmentations.

**【Function 4】** **Training Scheduler Revamping**

- **Step scheduler** has sharp learning rate transition which may cause the optimizer to re-stabilize the learning momentum in the next few iterations.
- **Cosine Schedule** scales the learning rate according to the value of cosine function on 0 to Pi, it starts with slowly reducing large learning rate, then reduces the learning rate quickly halfway ad finally ends up with tiny slope reducing small learning rate until it reaches 0.
- **Warm up learning rate**: to avoid gradient explosion during the initial training iterations.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200425110046052.png)

**【Function 4】** **Synchronized Batch Normalization**

- typical implementation of batch normalization working on multiple devices is fast(with no communication overhead), it inevitably reduces the size of bath size and causing slightly different statistics during computation.
- Peng et al. prove the importance of synchronized batch normalization in object detection with the analysis.

**【Function 5】** **Random shapes training for single-stage object detection networks**

- To fit memory limitation and allow simpler batching, the fixed shapes are needed.
- follow the approach of random shapes training as in Redmon et al.[1]

#### Conclusion

- first to systematically evaluate the various training heuristics applied in different object detection pipelines, providing valuable practice guidelines for future researches.
- proposed a visually coherent image mixup method designed for training object detection networks which is proved to be very effective in improving model generalization capabilities.
- achieved up to 5%-30% absolute average precision improvement based on existing models without modifying network structure and the loss function, with no extra inference cost.
- extended the research depth on object detection data augmentation domain that significantly strengthened the model generalization capability and help reduce over-fitting problems.

#### Notes <font color=orange>去加强了解</font>

  - this article tell more about train tricks, and detail tricks are not clearly understand.
  - [1] J.RedmonandA.Farhadi.Yolov3: An incremental improvement. arXivpreprintarXiv:1804.02767,2018.
  - batch normalization: 
    -  Megdet: A large mini-batch object detector
    -  Batch normalization: Accelerating deep network training by reducing internal covariate shift. 
- mixup:
  -  The elephant in the room
  -  mixup: Beyond empirical risk minimization

> Wu D, Liao M, Zhang W, et al. YOLOP: You Only Look Once for Panoptic Driving Perception[J]. arXiv preprint arXiv:2108.11250, 2021.

# Paper: YOLOP

<div align=center>
<br/>
<b>YOLOP: You Only Look Once for Panoptic Driving Perception</b>
</div>

#### Summary

- We put forward an efficient `multi-task network that can jointly handle three crucial tasks in autonomous driving`: `object detection, drivable area segmentation and lane detection` to save computational costs, reduce inference time as well as improve the performance of each task. Our work is the first to reach real-time on embedded devices while maintaining state-of-the-art level performance on the `BDD100K `dataset.
- The works we has use for reference including `Multinet` ([paper](https://arxiv.org/pdf/1612.07695.pdf?utm_campaign=affiliate-ir-Optimise media( South East Asia) Pte. ltd._156_-99_national_R_all_ACQ_cpa_en&utm_content=&utm_source= 388939),[code](https://github.com/MarvinTeichmann/MultiNet)）,`DLT-Net` ([paper](https://ieeexplore.ieee.org/abstract/document/8937825)）,`Faster R-CNN` ([paper](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf),[code](https://github.com/ShaoqingRen/faster_rcnn)）,`YOLOv5s`（[code](https://github.com/ultralytics/yolov5)) ,`PSPNet`([paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf),[code](https://github.com/hszhao/PSPNet)) ,`ENet`([paper](https://arxiv.org/pdf/1606.02147.pdf),[code](https://github.com/osmr/imgclsmob)) `SCNN`([paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16802/16322),[code](https://github.com/XingangPan/SCNN)) `SAD-ENet`([paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hou_Learning_Lightweight_Lane_Detection_CNNs_by_Self_Attention_Distillation_ICCV_2019_paper.pdf),[code](https://github.com/cardwing/Codes-for-Lane-Detection)). Thanks for their wonderful works.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016201750735.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016201802960.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016200912351.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/output1.gif)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/yolo4-object-detection/  

