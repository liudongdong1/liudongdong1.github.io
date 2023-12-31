# TorchVision_Models


> Torchvision.Models contain different models like: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.

### 1. Classification Models

- [AlexNet](https://arxiv.org/abs/1404.5997)  [VGG](https://arxiv.org/abs/1409.1556)[ResNet](https://arxiv.org/abs/1512.03385)[SqueezeNet](https://arxiv.org/abs/1602.07360)[DenseNet](https://arxiv.org/abs/1608.06993)[Inception](https://arxiv.org/abs/1512.00567) v3[GoogLeNet](https://arxiv.org/abs/1409.4842)[ShuffleNet](https://arxiv.org/abs/1807.11164) v2[MobileNet](https://arxiv.org/abs/1801.04381) v2[ResNeXt](https://arxiv.org/abs/1611.05431)[Wide ResNet](https://pytorch.org/docs/stable/torchvision/models.html#wide-resnet)[MNASNet](https://arxiv.org/abs/1807.11626)

#### 1.1.  Random weights

```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()
```

#### 1.2. Using pretrained weights

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201017142036494.png)

### 2. Semantic Segmentation

- [FCN ResNet50, ResNet101](https://arxiv.org/abs/1411.4038)
- [DeepLabV3 ResNet50, ResNet101](https://arxiv.org/abs/1706.05587)

### 3. Object Detection, Instance Segmentation, Person Keypoint Detection

- [Faster R-CNN ResNet-50 FPN](https://arxiv.org/abs/1506.01497)
- [Mask R-CNN ResNet-50 FPN](https://arxiv.org/abs/1703.06870)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201017143001110.png)

### 4. Video Classification

- ResNet 3D 18 ; ResNet MC 18;   ResNet (2+1)D

### 5. 后序Todo

- [ ] 积累学习使用使用torch提供的模型进行具体的实用教程；
- [ ] 如果学习一个，理解完整的工作过程，训练效果以及可能迁移使用的场景；

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/torchvision_models/  

