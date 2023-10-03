# weekmodel


# 1. KeypointDetection

### 1.1. CharPointDetection

> 识别字符中的俩个关键点。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021182442366.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021182723440.png)

### 1.2. Facial-keypoints-detection

> 用于检测人脸的68个关键点示例。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021183012603.png)

### 1.3. Hourglass-facekeypoints

> 使用基于论文Hourglass 的模型实现人体关键点检测。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021183120052.png)

### 1.4. [Realtime-Action-Recognition](https://github.com/felixchenfy/Realtime-Action-Recognition)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210317162813547.png)



> containing:
>         - boxes (``Tensor[N, 4]``): the ground-truth boxes in ``[x0, y0, x1, y1]`` format, with values
>           between ``0`` and ``H`` and ``0`` and ``W``
>                 - labels (``Tensor[N]``): the class label for each ground-truth box
>                         - keypoints (``Tensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
>           format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.
>
> The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
> losses for both the RPN and the R-CNN, and the keypoint loss.

> During inference, the model requires only the input tensors, and returns the post-processed
>
>   predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
>
>   follows:
>
> ​    \- boxes (``Tensor[N, 4]``): the predicted boxes in ``[x0, y0, x1, y1]`` format, with values between
>
> ​     ``0`` and ``H`` and ``0`` and ``W``
>
> ​    \- labels (``Tensor[N]``): the predicted labels for each image
>
> ​    \- scores (``Tensor[N]``): the scores or each prediction
>
> ​    \- keypoints (``Tensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

# 2. Classification

### 2.1 image_classification

> 使用pretain resnet系列模型进行五种花卉识别。结果比较清晰。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201021183309747.png)

### 2.2. yolo5_helmetDetect

> 基于Yolo5 进行的安全帽检测;
>
> 学习了yolo5 模型的使用，学习了使用yolo5进行自定义数据集使用；学习yolo5数据集标注形式，并使用yolo5进行数据集标注生成；

### 2.3. pytorch_classification

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201122163550833.png)

### 2.4. 驾驶员状态检测

- https://github.com/Yifeng-He/Distracted-Driver-Detection-with-Deep-Learning
- The [dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) contains 22,424 images which belongs to one of the 10 classes given below；

```
c0: safe driving
c1: texting - right
c2: talking on the phone - right
c3: texting - left
c4: talking on the phone - left
c5: operating the radio
c6: drinking
c7: reaching behind
c8: hair and makeup
c9: talking to passenger
```

### 2.5. Pytorch GoodStructure

#### 2.5.1. CNN_LSTM

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210119143253282.png)

> 下次进行代码开发，可以采用这种方式。

### 2.6. 人脸识别

#### 2.6.1. Face_recognition 库

- G:\weakmodel\weeklystudy\faceRecognition

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210203103640953.png)

#### 2.6.2. SeetaFace6

- G:\weakmodel\weeklystudy\faceRecognition\seetaFace6Python

#### 2.6.3  眼部跟踪

- G:\weakmodel\weeklystudy\faceRecognition\eyetracking  该目录记录眼部追踪处理相关demo。

# 2. NLP relative

### 2.1. textclassify

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119221751378.png)

- 学习了 通过gensin库训练 word2Vec模型；
- 学习了通过sklearn.linear_model进行ML相关操作；
- movierecommand 使用pyspark进行ALS 推荐；
- movieComment 使用nltk库进行分词，使用sklearn.feature_extraction.text库进行文本特征处理，并使用相关模型；
- hospitalEmotion 使用sklearn.linear_model进行 pos，neg 分类；
- crimeanalyse 使用pyspark 进行犯罪类别的分类，使用了各种学习模型；
- jobrequ.ipynb: 自己学习编写的 pyspark ML 使用，以及自定义函数UDF使用；

# 3. 前端

## 3.1. Book-Management-System

- 基本代码都有了，学习只关注了flask那登录到主页那部分，其他的处理逻辑没有具体细看。如果不考虑css中 layui 如何使用；
- 以后自己写python的管理系统，基于这个代码修改；
- 登录，数据库连接，请求啥的都齐全了；
- 后续自己编写相关系统的时候可以具体在学习 block 块使用规则。
- 项目效果； 如果做`物联网可以添加卡片式的那种显示效果`；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201115094825916.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201115095655445.png)

## 3.2. BookManageSystem

- https://github.com/withstars/Books-Management-System
- 技术选择： spring+ spring mvc+ jdbc template

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201205110513614.png)

## 3.3  LibrayManageSystem

- [LibrarySystem](https://github.com/cckevincyh/LibrarySystem)
- 前端用bootstrap框架搭建ui+ajax异步请求，后台用SSH+Quartz框架搭建的图书管理系统。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20201205124206468.png)

## 3.4. 电子相册

#### 1.  3Dalbum-master

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210109095231005.png)

#### 2. 3DRotatePhote

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210109095333755.png)

## 3.5. python qt 教程

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210109101543347.png)

### 1. qt UI 主题

- G:\weakmodel\weeklystudy\QDarkStyleSheet

### 2. TCP&UDP通信

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210312220032756.png)

### 3. [基于PyQt 电影天堂 爬虫](https://github.com/LeetaoGoooo/MovieHeavens)

> 基本代码思路可以参考，原始代码爬虫出现问题；

# 4. Raspberry

## 4.1. Vision Related

- 目录结构功能如下：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201129145649893.png)

## 4.2. fruitnanny

- 基于树莓派婴儿监控系统

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201130160315923.png)

- 通过解析print数据实现python和js通信；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201130154815852.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20201130160236011.png)

## 4.3. [cortex-license-plate-reader-client](D:\work_OneNote\OneDrive - tju.edu.cn\work_project\hardware\cortex-license-plate-reader-client)

> - python 上传api请求推理结果
> - 图片进行编码上传
> - gps模块处理函数，使用线程的方式进行处理
> - 使用若干worker并行处理，多线程方式的使用

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210615181233476.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210615185933117.png)

# 5. GAN 

## 5.1. GAN_Minist_keras-master

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210109094625595.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/weekmodel/  

