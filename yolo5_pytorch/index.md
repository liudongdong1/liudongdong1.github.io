# Yolo5_pytorch


### 1. Introduce

- https://github.com/ultralytics/yolov5  

```
pip install -r requirements.txt
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201031133710134.png)

| Model                                                        | APval    | APtest   | AP50     | SpeedGPU  | FPSGPU  |      | params | FLOPS  |
| ------------------------------------------------------------ | -------- | -------- | -------- | --------- | ------- | ---- | ------ | ------ |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 37.0     | 37.0     | 56.2     | **2.4ms** | **416** |      | 7.5M   | 13.2B  |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 44.3     | 44.3     | 63.2     | 3.4ms     | 294     |      | 21.8M  | 39.4B  |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 47.7     | 47.7     | 66.5     | 4.4ms     | 227     |      | 47.8M  | 88.1B  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | **49.2** | **49.2** | **67.7** | 6.9ms     | 145     |      | 89.0M  | 166.4B |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0) + TTA | **50.8** | **50.8** | **68.9** | 25.5ms    | 39      |      | 89.0M  | 354.3B |
| [YOLOv3-SPP](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     |      | 63.0M  | 118.0B |

#### 1.1. Environment

> - **[Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb)** with free GPU
> - **Kaggle Notebook** with free GPU: https://www.kaggle.com/ultralytics/yolov5
> - **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
> - **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart)

```python
pip install -r requirements.txt
python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                 http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
#train
python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
# for example
python detect.py --source inference/images --weights yolov5s.pt --conf 0.25
```

### 2. Train Custom Data

- https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
- 标注工具：类似于 [Labelbox](https://labelbox.com/) 、[CVAT](https://github.com/opencv/cvat) 、[精灵标注助手](http://www.jinglingbiaozhu.com/) 

- Learning Example: Helmet Detect

> - Install YOLOv5 dependencies
> - Download Custom YOLOv5 Object Detection Data
> - Define YOLOv5 Model Configuration and Architecture
> - Train a custom YOLOv5 Detector
> - Evaluate YOLOv5 performance
> - Visualize YOLOv5 training data
> - Run YOLOv5 Inference on test images
> - Export Saved YOLOv5 Weights for Future Inference

#### 2.0. 环境搭建

```python
#print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
# pip install -r requirements.txt

# base ----------------------------------------
Cython
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
pillow
PyYAML>=5.3
scipy>=1.4.1
tensorboard>=2.2
torch>=1.6.0
torchvision>=0.7.0
tqdm>=4.41.0

# coco ----------------------------------------
# pycocotools>=2.0

# export --------------------------------------
# packaging  # for coremltools
# coremltools==4.0
# onnx>=1.7.0
# scikit-learn==0.19.2  # for coreml quantization

# extras --------------------------------------
# thop  # FLOPS computation
# seaborn  # plotting
```

#### 2.1. 数据准备,文件放置规范

数据集：[Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201031121449831.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201031120637196.png)

```yaml
#custom_data.yml --> create data
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./Dataset/Safety_Helmet_Train_dataset/score/images/train
val: ./Dataset/Safety_Helmet_Train_dataset/score/images/val
# number of classes
nc: 3
# class names
names: ['person', 'head', 'helmet']
```

- **Create Label**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201031121025402.png)

- **Anchor K-means 介绍**

> bounding box由左上角顶点和右下角顶点表示，即$(x_1,y_1,x_2,y_2)$; 对box做聚类时，我们只需要box的宽和高作为特征，并且由于数据集中图片的大小可能不同，还需要先使用图片的宽和高对box的宽和高做归一化;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201031123734283.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201031123805207.png)

> 1. 随机选取K个box作为初始anchor；
> 2. 使用IOU度量，将每个box分配给与其距离最近的anchor；
> 3. 计算每个簇中所有box宽和高的均值，更新anchor；
> 4. 重复2、3步，直到anchor不再变化，或者达到了最大迭代次数。

#### 2.3. 自定义模型

> 在文件夹 `./models` 下选择一个你需要的模型然后复制一份出来，将文件开头的 `nc = ` 修改为数据集的分类数，下面是借鉴 `./models/yolov5s.yaml`来修改的

```yaml
# parameters
nc: 3  # number of classes     <============ 修改这里为数据集的分类数
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors: # <============ 根据 ./data/gen_anchors/anchors.txt 中的 Best Anchors 修改，需要取整（可选）
  - [14,27, 23,46, 28,130] 
  - [39,148, 52,186, 62.,279] 
  - [85,237, 88,360, 145,514]

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

#### 2.4. 训练

```shell script
python train.py --img 640 --batch 16 --epochs 10 --data ./data/custom_data.yaml --cfg ./models/custom_yolov5.yaml --weights ./weights/yolov5s.pt
```

其中，`yolov5s.pt` 需要自行下载放在本工程的根目录即可，下载地址[官方权重](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)

#### 2.5. 看训练之后的结果

训练之后，权重会保存在 `./runs` 文件夹里面的每个 `exp` 文件里面的 `weights/best.py` ，里面还可以看到训练的效果
![](G:/%25E5%2591%25A8%25E8%25AE%25A1%25E5%2588%2592%25E5%25AD%25A6%25E4%25B9%25A0%25E6%25A8%25A1%25E5%259E%258B/Smart_Construction-master/doc/test_batch0_gt.jpg)

#### 2.6. 侦测

侦测图片会保存在 `./inferenct/output/` 文件夹下

```shell script
python detect.py --source   0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

#### 2.7. 生成ONNX

```
pip install onnx
python ./models/export.py --weights ./weights/helmet_head_person_s.pt --img 640 --batch 1
```

#### 2.8. 增加数据集的分类

关于增加数据集分类的方法：

`SHWD` 数据集里面没有 `person` 的类别，先将现有的自己的数据集执行脚本生成 yolov5 需要的标签文件 `.txt`，之后再用 `yolov5x.pt` 加上 `yolov5x.yaml` ，使用指令检测出人体

```shell script
python detect.py --save-txt --source ./自己数据集的文件目录 --weights ./weights/yolov5x.pt
```

`yolov5` 会推理出所有的分类，并在 `inference/output` 中生成对应图片的 `.txt` 标签文件；

修改 `./data/gen_data/merge_data.py` 中的自己数据集标签所在的路径，执行这个python脚本，会进行 `person` 类型的合并 

### 3. Relative work

- [基于Yolo5模型的卡片四角检测](https://mp.weixin.qq.com/s?__biz=MzA4MTk3ODI2OA==&mid=2650349206&idx=1&sn=8340458489e9878109367c320d875533&chksm=87810865b0f681730dac5f5b2c693a4681dc8d317b5ef56c44bb6059b3e1c6e9c1f3802f9b77&scene=126&sessionid=1607828524&key=b918ff6c28ca5e8171245a6fcdc041378743ee07922f777f919dfdcba6bbe9cd984b60b88edb9a93630f3927418c9779fc715efed8d74e1e156b8cf6347d8145a5902c4d1c3f733a2afc423d1de0306bea506eac7af303aeef48c4f242353f92c02b33a6d1e872e9e18a0c26533788f73285c83e4dcdb3ce70d8170b4d3f173f&ascene=1&uin=MzE0ODMxOTQzMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A%2BQ5146voHbPHC9I80kUtdc%3D&pass_ticket=w%2Fpc6C5KDtVj%2Beh2vLjGFeKNhX9PO7R%2BDceH7UrCSuY6uEGbKjF5cq30Ri5W20h2&wx_header=0)

### 4. Resource

- [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
- [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
- [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
- [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
- [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
- [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/yolo5_pytorch/  

