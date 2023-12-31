# FaceRecognition


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716092214886.png)

## 1. 人脸检测问题

在实际工程应用中，常常会面临非常复杂的工况。一方面算法准确度会受到很多因素影响，例如目标遮挡、光线变化、小尺寸人脸等等。另一方面算法的推理时间也会受到很多因素的影响，例如硬件性能，目标数量，图片尺寸等等。下面是几种工程中常见的问题。

- **人脸遮挡**，或者人脸角度较大，都会直接导致目标不完整，对于检测算法召回率有很大影响
- **暗光**，光线不充足条件下，导致成像质量不高，会影响检测算法召回率
- **低分辨率**，低分辨率导致人脸尺寸过小
- **人脸数量过多**，图片中人脸数量多，对检测算法要求较高。例如多目标靠的太近，对于NMS算法会是一种考验，另外数量过多会影响某些算法(图像金字塔类型)的时间复杂度，例如MTCNN

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716092748448.png)

> **基于特征的算法**就是通过提取图像中的特征和人脸特征进行匹配，如果匹配上了就说明是人脸，反之则不是。提取的特征是人为设计的特征，例如Haar，FHOG，特征提取完之后，再利用分类器去进行判断。通俗的说就是采用模板匹配，就是用人脸的模板图像与待检测的图像中的各个位置进行匹配，匹配的内容就是提取的特征，然后再利用分类器进行判断是否有人脸

> **基于图像的算法**，将图像分为很多小窗口，然后分别判断每个小窗是否有人脸。通常基于图像的方法依赖于统计分析和机器学习，通过统计分析或者学习的过程来找到人脸和非人脸之间的统计关系来进行人脸检测。最具代表性的就是CNN，CNN用来做人脸检测也是目前效果最好，速度最快的。后面着重介绍CNN相关人脸检测算法。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219112426981.png)

## 2. 工业界常用算法

### 2.1. MTCNN

> MTCNN是kaipeng Zhang在本科阶段研究出来的，它是一个3级联的CNN网络，分为PNet，RNet，ONet，层层递进。PNet的输入是原图经过图像金字塔之后不同尺寸的图片，最后结果由ONet输出。

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716093313218.png" style="zoom: 67%;" />

### 2.2. **FaceBoxes、RetinaFace mnet、LFFD**

> 属于One Stage 算法，FaceBoxes类似于SSD算法框架，采用多尺度特征层融合方式，采用anchor proposal，在不同尺度特征层上进行检测，这样就顾及到多尺度的人脸检测，FaceBoxes的文章旨在CPU上实现实时检测。

### 2.3. CenterFace

> 最新开源的一个人脸检测算法，github上同名项目。目前从数据来看，效果最好。
>
> 在实际工程应用中，要根据部署环境来选择人脸检测算法。例如在多人脸抓拍的场景，就不能选择MTCNN这类级联的算法，因为级联网络的推理速度与人脸数成反比，受人脸数量影响较大，MTCNN适用于人脸考勤或者人证对比的场景，只可能出现固定数量人脸的场景。

### 2.4. [dlib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

> dlib 安装失败问题： 

```shell
pip install cmake
pip install dlib

#方法二
pip install dlib==19.6.1 
#方法三： 离线安装
pip install dlib-19.21.0.tar.gz
```



> 一个人脸算法库，并且开源。不管你是用c++还是python，都可以直接使用dlib来做检测.

```python
#opencv-python, dlib
import cv2
import numpy as np
import dlib
# Load the detector
detector = dlib.get_frontal_face_detector()
# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# read the image
img = cv2.imread("face.jpg")
# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point
    # Create landmark object
    landmarks = predictor(image=gray, box=face)
    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # Draw a circle
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
# show the image
cv2.imshow(winname="Face", mat=img)
# Delay between every fram
cv2.waitKey(delay=0)
# Close all windows
cv2.destroyAllWindows()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200826082312.png)

## 3. 论文阅读

**level**: CVPR
**author**: Valentin Bazarevsky  (GoogleResearch)
**date**: 2019
**keyword**:

- face detect

------

### Paper: BlazeFace

<div align=center>
<br/>
<b>BlazeFace: Sub-millisecond Neural FaceDetectionon Mobile GPUs
</b>
</div>

#### Summary

1. present a lightweight and well-performing face detector tailored for mobile GPU inference, run 200-1000+on flagshship devices.
2. supporting to any augmented reality pipeline that requires an accurate facial region of interest as an input for task-specific models, such as 2D/3D facial keypoint or geometry estimation, facial features or expression classification, and face region segmentation.
3. **Relative to speed:**
   - a very compact feature extractor convolutional neural network related in structure to MobleNetV1.
   - a novel GPU-friendly anchor scheme modified from SSD, aimed at effective GPU utilization.
4. **Related to prediction quality:** a tie resolution strategy alternative to non-maximum suppression that achieves stabler,smoother tie resolution between overlapping predictions.

#### System Overview

> BlazeFace model produces 6 facial keypoint coordinates (for eye centers, ear tragions, mouth center, and nose tip) that allow us to estimate **face rotation**, alleviating the requirement of significant translation and rotation invariance in subsequent processing steps;

【**Model architecture design**】

- **Enlarging the receptive field sizes:**  
  - increasing the kernel size of the depthwise part is relatively cheap, and employ 5*5 kernels in model architecture bottlenecks, trading the kernel size increase for the decrease in the total amount of such bottlenecks required to reach a particular receptive field size.  <font color=red>这块看不懂</font>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831120241.png)

- **Feature extractor:** the extractor takes an RGB input of 128*128 pixels and consists of a 2D convolution followed by 5 single BlazeBlocks and 6 double BlazeBlocks.
- **Anchor scheme:** SSD-like object detection models rely on pre-defined fixed-size base bounding boxes called priors, or anchors in Faster-R-CNN terminology.
  - adopt an alternative anchor scheme that stops at the 8*8 feature map dimensions without further downsampling,
  - replace 2 anchors per pixel in each of the 8\*8, 4\*4 and 2*2 resolutions by 6 anchors at  8\*8;
  - due to limited variance in human face aspect ratios, limiting the anchors to the 1:1 aspect ratio was found sufficient for accurate face detection;
- **Post-processing:** replacing the suppression algorithm wiht a blending strategy that estimates the regression parameters of a bounding box as a weighted mean between the overlapping predictions;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831121558.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831121622.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831121809.png)

#### Notes <font color=orange>去加强了解</font>

- [ ] 几种常见模型里面的计算

**level**:   CCF_A   CVPR
**author**:  FlorianSchroff fschroff@google.com GoogleInc
**date**: 2015 
**keyword**:

- AI, FaceRecognition

------

### Paper: FaceNet

<div align=center>
<br/>
<b>FaceNet: A Uniﬁed Embedding for FaceRecognition and Clustering
</b>
</div>
#### Summary

1. present a system, FaceNet that directly learns  a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity, further used by face recognition, verification, and clustering.
2. FaceNet directly trains its output to be a compact 128-D embedding using a triplet based loss function based on LMNN, the triplets consist of two matching face thumbnails and a non-matching face thumbnail and the loss aims  to separate the positive pair form the negative by a distance margin. The thumbnails are tight crops of the face area, no 2D or 3D alignment, other than scale and translation is performed.

#### Research Objective

**previous work:**

- <font color=red>Zeiler&Fergus[22] model </font>: multiple interleaved layers of convolutions, non-linear activations, local response normalizations, and max pooling layers.
- <font color=red>Inception model of Szegedy et al.</font> : use  mixed layers that run several different convolutional and pooling layers in parallel and concatenate their responses
- using a complex system of multiple stages combining the output of a deep convolutional network with PCA for dimensionality reduction and SVM for classification
- ZHanyao et al.  : employ deep network to warp faces into a canonical frontal view and then learn CNN that classifies each face as belonging to a known identity, PCA on the network output in conjunction with an ensemble of SVM is used
- Taigman et al. : multi-stage approach that aligns faces to a general 3D shape model.
- sun et al. : propose a compact and therefore relatively cheap to compute network.

#### Methods

- **Problem Formulation**:

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200301104544427.png)

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200301104603166.png)

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200301104625390.png)

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200301104451015.png)

**[FaceNet Model1]**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200301104704084.png)

**[FaceNet Model2]**based on GoogLeNet styleInceptionmodels[16]

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200301104856005.png)



#### Notes 

- curriculum learning : describes a type of learning in which you first start out with only easy examples of a task and then gradually increase the task difficulty.
  - code available: https://github.com/vkakerbeck/Progressively-Growing-Networks
- Progressively Growing Neural Networks:grow networks during training and to learn new image categories
- LMNN

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716091352035.png" alt="image-20200716091352035" style="zoom:50%;" />

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200305173050782.png)

- [triplet based loss](https://en.wikipedia.org/wiki/Triplet_loss)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200305175940694.png)

**level**:   CVPR   CCF_A
**author**: XinLiu (CAS)
**date**: 2016
**keyword**:

- FaceRecognition

------

### Paper: VIPLFaceNet

<div align=center>
<br/>
<b>VIPLFaceNet: AnOpenSourceDeepFaceRecognitionSDK</b>
</div>


#### Research Objective

- **Application Area**: Face recognition

#### Proble Statement

- a conventional face recognition system consists of four modules,face detection,face alignment, face representation, and identity classification.
- main challenges of face representation:
  - small inter-person appearance difference caused by similar facial configurations
  - large intra-person appearance variations due to large intrinsic variations and diverse extrinsic imaging factors, such as head pose, expression, aging, and illumination.

previous work:

- Face Representation before DL

- Hand Craft features: Gabor wavelets, local Binary Pattern, SIFT, Historgram of Oriented Gradients

- Deep learning Methods:

  - DeepFace : 1. 3D model based face alignment to frontalize facial images with large pose. 2.large scale training set with 4 million face images of 4000 identities. 3.Deep convolutional neural network with the local connected layer that learns separate kernel for each spatial position. 4.A siamese network architecture to learn deep metric based on the features of the deep convolutional network
  - DeepID, DeepID2, DeepID2+ 
  - Learning face representation from scratch.
  - FaceNet

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200304122129244.png)

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200304122158473.png)

【Optimation 1】Fast Normalization Layer

Data normalization can speed up convergence, which is recently extended as the batch normalization algorithms.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200304152727485.png)

**Face Detection:**  using face detection toolkit by VIPL lab of CAS,

**Facial Landmark Location: **coarse to fine auto-encoder networks(CFAN) to detect five facial landmarks in the face.

**Face Normalization:** the face image is normalized to 256*256 pixels using five facial landmarks.

#### Evaluation

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200304153238679.png)

#### Conclusion

- propose and release an open source deep face recognition model, VIPFaceNet, with high-accuracy and low computational cost.
- reduces 40% computation cost and cuts down 40% error rate on LFW compared with AlexNet
- pure C++ code

#### Notes <font color=orange>去加强了解</font>

- [ ] 学习使用AlexNet模型  simplest with 5 convolutional layer and 3 fully-connected layers
- [ ] 学习使用LFW模型
- [ ] Learning face representation from scratch. arXivpreprintarXiv:1411.7923,2014 
- [ ] GoogleNet 模型
- [ ] VGGNet模型

## 4. 开源项目

### [4.1. DBFace](https://github.com/dlunion/DBFace)

- DBFace 是一个轻量级的实时人脸识别方法，其有着更快的识别速度与更高的精度。下图展示了多种人脸检测方法在 WiderFace 数据集上的测试效果。可以看到不仅 DBFace 模型的大小最小，其在 Easy、medium、Hard 三个测试任务中均取得了最高的识别精度。
- WiderFace 是一个关于人脸检测的基准跑分数据集，其中包含 32,203 张图片以及在各方面剧烈的 393,703 张人脸，数据集具有从简单到困难等不同难度的任务。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716091843597.png)

![image-20200716091856713](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716091856713.png)

### 4.2. Rotation-Invariant FaceDetection

Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks

PCN 会抽选识别候选面部图像块，并将朝下的图像块翻转至正向，这样就会减半 RIP 的角度范围，即从 [−180° , 180° ] 到 [−90° , 90° ]。然后旋转过的面部图像块会进一步区分朝向并校准到垂直向的 [−45° , 45° ] 范围，这样又会减半 RIP 的角度范围。最后，PCN 会分辨到底这些候选图像块是不是人脸，并预测出精确的 RIP 角度。

通过将校准过程分割为几个渐进的步骤，且在早期校准步骤只预测粗略的朝向，PCN 最后能实现精确的校准。此外，每一个校准步骤可以简单地旋转-90°、90°和 180°，因此额外的计算量非常低，这也就是为什么该检测项目能在 CPU 上实时运行的重要原因。通过在逐渐降低的 RIP 范围内执行二元分类（是人脸或不是人脸），PCN 能在 360° RIP 旋转角度内准确地检测到人脸，而本项目重点就是实现这样旋转不变的人脸检测器。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200716091914764.png)



论文图 3：uepeng Shi 等研究者提出的 PCN 概览，它会逐渐降低旋转的角度范围，并最终预测人脸及其旋转的角度。这种能处理不同旋转方向的人脸检测器有非常高的准确率，因为它会先将候选人脸旋转至正向再预测。此外，这种方法同样有非常小的计算量，该 GitHub 项目表示它甚至可以在 CPU 上实时检测人脸。

### [4.3. OpenCV+OpenVINO实现人脸](https://software.intel.com/en-us/openvino-toolkit/choose-download?innovator=CONT-0026250)

- 支持35点分布表示出左眼、右眼、鼻子、嘴巴、左侧眉毛、右侧眉毛、人脸轮廓

```python
/ 加载LANDMARK
Net mkNet = readNetFromModelOptimizer(landmark_xml, landmark_bin);
mkNet.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
mkNet.setPreferableTarget(DNN_TARGET_CPU);

// 加载网络
Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
net.setPreferableTarget(DNN_TARGET_CPU);
Mat frame;
while (true) {
    bool ret = cap.read(frame);
    if (!ret) {
        break;
    }
    // flip(frame, frame, 1);
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300),
        Scalar(104.0, 177.0, 123.0), false, false);
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.5)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * w);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * h);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * w);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * h);
            Mat roi = frame(Range(y1, y2), Range(x1, x2));
            Mat blob = blobFromImage(roi, 1.0, Size(60, 60), Scalar(), false, false);
            mkNet.setInput(blob);
            Mat landmark_data = mkNet.forward();
            // printf("rows: %d \n, cols : %d \n", landmark_data.rows, landmark_data.cols);
            for (int i = 0; i < landmark_data.cols; i += 2) {
                float x = landmark_data.at<float>(0, i)*roi.cols+x1;
                float y = landmark_data.at<float>(0, i + 1)*roi.rows+y1;
                // mkList.push_back(Point(x, y));
                circle(frame, Point(x, y), 2, Scalar(0, 0, 255), 2, 8, 0);
            }
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 8);
        }
    }
    imshow("Face-Detection Demo", frame);
    char c = waitKey(1);
    if (c == 27) {
        break;
    }
}

```

### 4.4. [Face_Recognition 库使用](https://github.com/ageitgey/face_recognition)

```shell
pip install face_recognition
```

> Face Recognition 是一个基于 Python 的人脸识别库，它还提供了一个命令行工具，让你通过命令行对任意文件夹中的图像进行人脸识别操作。 该库使用 dlib 顶尖的深度学习人脸识别技术构建，在户外脸部检测数据库基准(Labeled Faces in the Wild benchmark)上的准确率高达 99.38%。

> batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128)： 
>
> 使用cnn面部检测器返回图像中二维人脸的边界框数组，如果您正在使用[GPU](https://cloud.tencent.com/product/gpu?from=10680)，这可以更快的给您结果，因为GPU可以一次处理批次的图像。如果您不使用GPU，则不需要此功能。
>
> ##### 参数：
>
> - images - 图像列表（每个作为numpy数组）
> - number_of_times_to_upsample - 用于对图像进行采样的次数。较高的数字找到较小的脸。
> - batch_size - 每个GPU处理批次中包含的图像数量。
>
> ##### 返回：
>
> 一个可以在css（上，右，下，左）顺序中找到的人脸位置的元组列表

> compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)： 将候选编码的面部编码列表进行比较，以查看它们是否匹配。
>
> ##### 参数：
>
> - known_face_encodings - 已知面部编码的列表
> - face_encoding_to_check - 与已知面部编码的列表进行比较的单面编码
> - tolerance - 面孔之间的距离要考虑多少。越小越严格， 0.6是典型的最佳性能。
>
> ##### 返回：
>
> 一个True / False值的列表，指出哪个known_face_encodings匹配要检查的面部编码

> face_distance(face_encodings, face_to_compare)： 给出面部编码列表，将其与已知的面部编码进行比较，并为每个比较的人脸获得欧几里得距离。距离告诉你面孔是如何相似的。
>
> ##### 参数：
>
> - face_encodings - 要比较的面部编码列表
> - face_to_compare - 要比较的面部编码
>
> ##### 返回：
>
> 一个numpy ndarray，每个面的距离与“faces”数组的顺序相同

> face_encodings(face_image, known_face_locations=None, num_jitters=1)： 
>
> 给定图像，返回图像中每个面部的128维面部编码。
>
> ##### 参数：
>
> - face_image - 包含一个或多个面的图像
> - known_face_locations - 可选 - 如果您已经知道它们，每个面的边框。
> - num_jitters - 计算编码时重新采样多少次。更高更准确，但更慢（即100是100倍慢）
>
> ##### 返回：
>
> 128个面部编码的列表（图像中的每个脸部一个）

> face_landmarks(face_image, face_locations=None)：给定图像，返回图像中每个脸部的脸部特征位置（眼睛，鼻子等）的指令
>
> ##### 参数：
>
> - face_image - 要搜索的图像
> - face_locations - 可选地提供要检查的面部位置的列表。
>
> ##### 返回：
>
> 面部特征位置（眼睛，鼻子等）的列表

> face_locations(img, number_of_times_to_upsample=1, model='hog')：
>
> 返回图像中人脸的边框数组
>
> ##### 参数：
>
> - img - 一个图像（作为一个numpy数组）
> - number_of_times_to_upsample - 用于对图像进行上采样的次数多少次。较高的数字找到较小的脸。
> - model - 要使用的面部检测模型。“hog”在CPU上不太准确，但速度更快。“cnn”是一个更准确的深入学习模式，GPU / CUDA加速（如果可用）。默认为“hog”。
>
> ##### 返回：
>
> 一个可以在css（上，右，下，左）顺序中找到的表面位置的元组列表

> load_image_file(file, mode='RGB')：
>
> 将图像文件（.jpg，.png等）加载到numpy数组中
>
> ##### 参数：
>
> - file - 要加载的图像文件名或文件对象
> - mode - 将图像转换为格式。只支持“RGB”（8位RGB，3声道）和“L”（黑白）。
>
> ##### 返回：
>
> 图像内容为numpy数组

- 人脸识别并检测绘制关键点

```python
import face_recognition
import cv2
import numpy as np
import os
import time
from threading import Thread
from PyQt5.QtCore import *

from PIL import Image, ImageDraw, ImageFont



class Face_Recognizer:
    def __init__(self, camera_id = 0):
        self.basefolder="../../data/face"
        self.faces,self.faceNames=self.initFaceData()
    
    '''
    @function: 初始化人脸数据，从已经存储的文件加载name和对应的人脸 encoding
    '''
    def initFaceData(self):
        known_faces=[]
        known_faceNames=[]
        for file in os.listdir(self.basefolder):
            filepath=os.path.join(self.basefolder,file)
            #print(filepath)
            image=face_recognition.load_image_file(filepath)
            try:
                imageEncoding=face_recognition.face_encodings(image)[0]
                known_faceNames.append(file.split('.')[0])
                known_faces.append(imageEncoding)
                print(known_faceNames[-1])
            except:
                print("file don't detect face")
        return known_faces,known_faceNames

    #This function will take a sample frame
    #save the picture of the given user in a folder
    #returns the path of the saved image
    def saveFaceImage(self,imgdata, face_name='user'):
        face_name = '{}.png'.format(face_name)
        facesavepath=os.path.join(self.basefolder, face_name)
        try:
            cv2.imwrite(facesavepath, imgdata)
            time.sleep(1)
        except:
            print("Can't Save File")
    
    '''
    @function: 如果图片数据中包含人脸，则添加人脸的编码和name信息
    @parameters：
        imgdata: 人脸数据
        face_names: 名称数据
    @return
        true: 录入人脸信息成功
        false: 录入人脸信息失败
    '''
    def faceRegister(self,originimage,face_name):
        #imgdata = face_recognition.load_image_file(self.face_image_path)
        # if face_name not in self.faceNames:
        #     imgdata = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
        #     face_encoding = face_recognition.face_encodings(imgdata)[0]
        #     self.faces.append(face_encoding)
        #     self.faceNames.append(face_name)
        #     print("faceRegister: add facecoding ok")
        # self.saveFaceImage(originimage,face_name)
        # print("faceRegister: save face ok")
        # return True
        try:
            if face_name not in self.faceNames:
                imgdata = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
                face_encoding = face_recognition.face_encodings(imgdata)[0]
                self.faces.append(face_encoding)
                self.faceNames.append(face_name)
                print("faceRegister: add facecoding ok")
            self.saveFaceImage(originimage,face_name)
            print("faceRegister: save face ok")
            return True
        except Exception as err:
            print("No face found in the image",err)
            return False



    '''
        @parameter: face_name: 用户名称
        @return: 如果列表包含用户名称，则返回 true； 否则返回false；
    '''
    def nameContain(self, face_name):
        registered = False
        namelist=os.listdir(self.basefolder)
        if '{}.png'.format(face_name) in namelist:
            registered=True
        return registered

    '''
    @function: 从一张图片中识别人脸信息
    @parameters: 
        targetPath: 待识别的图片路径
    @returns: 识别用户的姓名信息
    '''
    def getFaceNameFromFile(self,targetPath):
        name="None"
        image=face_recognition.load_image_file(targetPath)
        try:
            face_encoding=face_recognition.face_encodings(image)[0]
            matches = face_recognition.compare_faces(self.faces, face_encoding)
            face_distances = face_recognition.face_distance(self.faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            #print("最小距离： ",face_distances[best_match_index])
            #print(face_distances)
            if matches[best_match_index]:
                name = self.faceNames[best_match_index]
            return name
        except Exception as e:
            print("file don't detect face",e)
            return name   
    
    '''
    @function: 从一张图片编码中识别人脸信息
    @parameters: 
        targetEncoding: 待识别的图片 特征编码
    @returns: 识别用户的姓名信息
    '''
    def getFaceNameFromEncoding(self,targetEncoding):
        name="None"
        matches = face_recognition.compare_faces(self.faces, targetEncoding)
        face_distances = face_recognition.face_distance(self.faces, targetEncoding)
        best_match_index = np.argmin(face_distances)
        print("最小距离： ",face_distances[best_match_index])
        print(face_distances)
        if matches[best_match_index]:
            name = self.faceNames[best_match_index]
        return name

    def compareToDatabase(self, unknown_face_encoding=None):
        if not self.is_running:
            self.is_running = True
            self.m_thread = Thread(target= self._compareToDatabase )
            self.m_thread.start()

    def _compareToDatabase(self,originimage):
        authenticated = False
        #imgdata = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
        #imgdata=originimage
        small_frame = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        if(len(face_locations)>1):
            return "Unknown"
        face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]

        print(type(self.faces),type(face_encoding))
        matches = face_recognition.compare_faces(self.faces, face_encoding)
        face_distances = face_recognition.face_distance(self.faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        #print("最小距离： ",face_distances[best_match_index])
        #print(face_distances)
        if matches[best_match_index]:
            name = self.faceNames[best_match_index]
        print("_compareToDataset",name,face_locations)

        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                for pt_pos in face_landmarks[facial_feature]:
                        cv2.circle(originimage, (pt_pos[0]*4,pt_pos[1]*4), 1, (255, 0, 0), 2)
                        cv2.circle(originimage, (pt_pos[0]*4,pt_pos[1]*4), 5, color=(0, 255, 0))
            
        #process_this_frame = not process_this_frame
        top, right, bottom, left=face_locations[0]
        top=top*4
        right=right*4
        bottom=bottom*4
        left=left*4
        # Draw a box around the face
        cv2.rectangle(originimage, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        #cv2.rectangle(originimage, (left-20, bottom - 60), (right+20, bottom+20), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_PIL = Image.fromarray(originimage)
        font = ImageFont.truetype('../icons/方正康体简体.TTF', 40)
        # 字体颜色
        fillColor = (255,0,0)
        # 文字输出位置
        position = (left - 100, bottom - 30)
        textinfo = "欢迎{}登录".format(name)
        # 需要先把输出的中文字符转换成Unicode编码形式
        if not isinstance(textinfo, str):
            textinfo = textinfo.decode('utf8')
    
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, textinfo, font=font, fill=fillColor)
        # 使用PIL中的save方法保存图片到本地
        # img_PIL.save('02.jpg', 'jpeg')
        # 转换回OpenCV格式
        originimage = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

       # cv2.putText(originimage,"欢迎{}登录".format(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imwrite("tmp.png" , originimage)
        #self.im_s.new_image.emit(".tmp.png")
            # Display the resulting image
        #cv2.imshow('Video', imgdata)
        return originimage,name 


    def removeFaceData(self, face_name):
        pass

    def paint_chinese_opencv(self,im,chinese,pos,color):
        img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('NotoSansCJK-Bold.ttc',25)
        fillColor = color #(255,0,0)
        position = pos #(100,100)
        if not isinstance(chinese,str):
            chinese = chinese.decode('utf-8')
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position,chinese,font=font,fill=fillColor)

        img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        return img
#编写一个测试
def functionTest():
    capture = cv2.VideoCapture(0)
    faceServer=Face_Recognizer(0)
    i=0
    name="liudongdong"
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame,1)   #镜像操作
        
        key = cv2.waitKey(50)
        #print(key)
        if key  == ord('q'):  #判断是哪一个键按下
            if faceServer.faceRegister(frame,name):
                print("人脸信息录入成功")
            else: print("人脸信息录入失败")
        if key == ord('r'):
            frame,name=faceServer._compareToDatabase(frame)
            print("识别名称，",name)
        cv2.imshow("video", frame)

        #cv2.imshow('Video', imgdata)
        if key== ord('b'):
            break
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    functionTest()
 
    # while i<50:
    #     i=i+1
    #     ret, frame = video_capture.read()
    #     cv2.imshow('Video', frame)
    #     if i<10:
    #         if faceServer.faceRegister(frame,name):
    #             print("人脸信息录入成功")
    #         else: print("人脸信息录入失败")
    #     else:
    #         imgdata,name=faceServer._compareToDatabase(frame)
    #         cv2.imshow('Video', imgdata)
    #         print("识别姓名： name=",name)
'''
recognizer = Face_Recognizer()
recognizer.registerFace()
recognizer.saveFaceImage('Kareem')
'''
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210723155206990.png)

- 检测图像中所有人脸

```python
# -*- coding: utf-8 -*-
# 检测人脸
import face_recognition
import cv2
# 读取图片并识别人脸
img = face_recognition.load_image_file("1.png")
face_locations = face_recognition.face_locations(img)
print (face_locations)
# 调用opencv函数显示图片
img = cv2.imread("1.png")
cv2.namedWindow("原图")
cv2.imshow("原图", img)
# 遍历每个人脸，并标注
faceNum = len(face_locations)
for i in range(0, faceNum):
    top =  face_locations[i][0]
    right =  face_locations[i][1]
    bottom = face_locations[i][2]
    left = face_locations[i][3]
    start = (left, top)
    end = (right, bottom)
    color = (55,255,155)
    thickness = 3
    cv2.rectangle(img, start, end, color, thickness)
# 显示识别结果
cv2.namedWindow("识别")
cv2.imshow("识别", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- 人脸匹配

```python
# 导入库
import os
import face_recognition
# 制作所有可用图像的列表
images = os.listdir('images')
# 加载图像
image_to_be_matched = face_recognition.load_image_file('my_image.jpg')
# 将加载图像编码为特征向量
image_to_be_matched_encoded = face_recognition.face_encodings(
   image_to_be_matched)[0]
# 遍历每张图像
for image in images:
   # 加载图像
   current_image = face_recognition.load_image_file("images/" + image)
   # 将加载图像编码为特征向量
   current_image_encoded = face_recognition.face_encodings(current_image)[0]
   # 将你的图像和图像对比，看是否为同一人
   result = face_recognition.compare_faces(
       [image_to_be_matched_encoded], current_image_encoded)
   # 检查是否一致
   if result[0] == True:
       print ("Matched: " + image)
   else:
       print ("Not matched: " + image)
```

- 检测标记人脸特征

```python
# -*- coding: utf-8 -*-
# 自动识别人脸特征
from PIL import Image, ImageDraw
import face_recognition
# 将jpg文件加载到numpy 数组中
image = face_recognition.load_image_file("my_image.jpg")
#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)
#打印发现的脸张数
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
for face_landmarks in face_landmarks_list:
   #打印此图像中每个面部特征的位置
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]
    for facial_feature in facial_features:
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
   #让我们在图像中描绘出每个人脸特征！
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], width=5)
    pil_image.show()
```

### 4.5. [InsightFace](https://github.com/deepinsight/insightface)

> InsightFace is an open source 2D&3D deep face analysis toolbox, mainly based on MXNet. This module can help researcher/engineer to `develop deep face recognition algorithms quickly by only two steps`: `download the binary dataset and run the training script`.

- Face Detection

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219111717005.png)

### 4.6. [SeetaFace6](https://github.com/tensorflower/seetaFace6Python) 

```python
#类似face_recognition 库，可以快捷使用
#人脸检测
FACE_DETECT = 0x00000001
#人脸跟踪
FACE_TRACK = 0x00000002
#人脸识别（特征提取）
FACERECOGNITION = 0x00000004
#rgb活体检测
LIVENESS = 0x00000008
#人脸5点关键点检测
LANDMARKER5 = 0x00000010
#人脸68点关键点检测
LANDMARKER68 = 0x00000020
#带遮挡识别的人脸5点关键点检测
LANDMARKER_MASK = 0x00000040
#人脸姿态角度方向评估
FACE_POSE_EX = 0x00000080
#性别识别
FACE_GENDER = 0x00000100
#年龄识别
FACE_AGE = 0x00000200
def Predict(self, frame: np.array, face: SeetaRect,
                   points: List[SeetaPointF]) -> int:
        """
        单帧rgb活体检测
        :param frame: 原始图像
        :param face: 人脸区域
        :param points:  人脸关键点位置
        :return:  活体检测结果
        0:真实人脸
        1:攻击人脸（假人脸）
        2:无法判断（人脸成像质量不好）
        """
        self.check_init(LIVENESS)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._Predict(seetaImageData, face, points)
def compare_feature_np(self, feature1: np.array, feature2: np.array) -> float:
        """
        使用numpy 计算，比较人脸特征值相似度
       :param feature1: 人脸特征值1
        :param feature2: 人脸特征值2
        :return: 人脸相似度
        """
        dot = np.sum(np.multiply(feature1, feature2))
        norm = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        dist = dot / norm
        return float(dist)
def PredictAge(self,frame: np.array) -> int:
        """
        检测一张只有人脸的图片,识别出年龄
        :param frame: 原图
        :param face: 人脸检测框
        :param points: 人脸关键点
        :return: 年龄大小
        """
        self.check_init(FACE_AGE)
        if frame.shape[0]!=256 or frame.shape[1]!=256:
            seetaImageData = get_seetaImageData_by_numpy(cv2.resize(frame,(256,256)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictAge(seetaImageData)
```

### 4.7. [Ultra-Light-Generic-Face-Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

> - [ ] [Widerface test code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/widerface_evaluate)
> - [ ] [NCNN C++ inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/ncnn) ([vealocia](https://github.com/vealocia))
> - [ ] [MNN C++ inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN), [MNN Python inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN/python)
> - [ ] [Caffe model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/caffe/model) and [onnx2caffe conversion code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/caffe)
> - [ ] [Caffe python inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/caffe/ultra_face_caffe_inference.py) and [OpencvDNN inference code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/caffe/ultra_face_opencvdnn_inference.py)
> - [ ] 如果要使用的话，可以学习这个项目源代码

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210324090933.png)

### 4.8. [CompreFace](https://github.com/exadel-inc/CompreFace)

> CompreFace provides `REST API` for `face recognition`, `face verification`, `face detection`, `landmark detection`, `age`, and `gender recognition`. The solution also features a `role management system` that allows you to easily control who has access to your Face Recognition Services.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210611170456149.png)

### 4.9. [TFace](https://github.com/Tencent/TFace)

> 基于可信人脸识别的理念，TFace重点关注人脸识别领域的四个研究方向：精准、公平、可解释以及隐私。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210625104637590.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210625104702814.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210625104719037.png)

- **[CurricularFace](https://arxiv.org/abs/2004.00288)** 一种用于人脸识别基础模型训练的损失函数，发表于CVPR2020， 主要的思路是将课程学习的思想结合到常用的人脸识别损失函数，训练过程中自动挖掘困难样本，先易后难渐进学习，提升识别模型训练鲁棒性及难样本识别性能。
- **[DDL](https://arxiv.org/abs/2002.03662)** 一种用于提升特定场景下人脸识别性能的方法，发表于ECCV2020，主要的思路是针对某一特定场景的难样本，为其寻找一个合适的教师场景，通过拉近两种场景下的人脸相似度分布，从而提升该场景下困难样本的识别性能。
- **[CIFP]( https://arxiv.org/abs/2106.05519)** 一种提升个体识别公平性的方法，发表于CVPR2021, 提出了基于误报率惩罚的损失函数，即通过增加实例误报率（FPR）的一致性来减轻人脸识别偏差。
- **[SDD-FIQA](https://arxiv.org/abs/2103.05977)** 一种基于人脸识别相似度分布的无监督人脸质量评估方法，发表于CVPR2021, 通过计算同人和非同人相似度分布的韦氏距离作为目标图像的质量分伪标签， 最终通过图像+质量伪标签训练得到质量分模型。
- **[SCF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf)** 一种基于人脸特征置信度的人脸识别方法，发表于CVPR2021, 核心思想包含两点：a. 将人脸样本特征从确定向量升级为概率分布，从而获得额外刻画样本识别置信度的能力；b. 提出适配于超球流形r-radius von Mises Fisher分布建模特征，理论可解释性与方法收敛性较PFE更佳。

## 5. FaceSearching

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219124037238.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219112540276.png)

1. 单机由于内存和CPU性能限制，能够支持的人脸检索数始终都有上限，所以必须进行集群设计来提高容量。
2. 10亿级别的人脸库存储是一个问题，按每张图片50K的大小都会是TB级别了。
3. 10亿级别人脸库建模需要很长时间。
4. 10亿级别人脸库检索响应时间能否做到秒级。
5. 10亿级别人脸库检索TPS能到多少。

### 5.0. 基于GPU优化的检索方案

> **数据并行+模型并行**： 首先是数据并行，每个GPU上去预测它自己的数据batch，得到人脸特征，然后对特征进行一个多机汇总，得到完整的F。同时，我们把参数矩阵W均匀拆分到多机不同的显卡上，比如第一个GPU负责计算每张图属于第1-10万类的概率，下一GPU负责第10万到20万类，这样依次进行。可考虑使用浮点运算能力更高的GPU来实现.

### 5.1. **分布式人脸检索系统框图**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219113002358.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219123840814.png)

### 5.2. **人脸动态库方案**

> 在内部验证阶段，使用单机存储固定特征个数（可能是一千万个）的特征库，每个特征对应记录ID、时间戳、摄像机编号等信息。每天新增的特征形成一个单独的小特征库，每天定时把小特征库合并到大特征库，并把大特征库中最旧的同量特征删除，保持特征库的大小。在检索时先对全库进行1:N，根据阈值过滤出部分记录后，再抽取对应记录的额外信息，与页面检索条件进行匹配，返回结果。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219124313646.png)

### 5.3.  ES分布式人脸检索方案

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219124437959.png)

### 5.4. 基于RocksDB的分布式特征索引方案

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219124854164.png)

### 5.5.  基于小特征加速比对的检索方案

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201219125047175.png)

## 6. 数据集&学习链接

- [人脸数据集](https://dataware.cc/tag/%E4%BA%BA%E8%84%B8%E6%95%B0%E6%8D%AE%E9%9B%86/)
- [分布式检索](https://blog.csdn.net/yimin_tank/article/details/82703121)

- https://mp.weixin.qq.com/s/vugfNwlH8a7uOIpmhg2Nig

- https://zhuanlan.zhihu.com/p/35968767
- [人脸检索方案](https://segmentfault.com/a/1190000019224111)

> cv2.imread() 读取图片数据为空的问题： 
>
> im = cv2.imdecode(np.fromfile(file,dtype=np.uint8),-1)   使用该函数代替即可解决问题。

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/facerecognition/  

