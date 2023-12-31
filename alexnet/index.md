# Alexnet


> lexNet是2012年ILSVRC 2012（ImageNet Large Scale Visual Recognition Challenge）竞赛的冠军网络，分类准确率由传统方法的 70%+提升到 80%+。它是由Hinton和他的学生Alex Krizhevsky设计的。

- 采用Relu激活函数：替换饱和激活函数，减轻梯度消失
- 采用LRN（局部响应归一化）：对数据归一化，减轻梯度消失
- Dropout：提高了全连接层的鲁棒性，增加网络的泛化能力
- 数据增强：TenCrop，色彩修改

### 1. Alexnet 网络模型

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210512225816.png)

> **N = (W − F + 2P ) / S + 1**:  其中W是输入图片大小，F是卷积核或池化核的大小， P是padding的像素个数 ，S是步距

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210512230856.png)

- 卷积层1

> Conv1: kernels:48*2=96；kernel_size:11；padding:[1, 2] ；stride:4
>
> 其中`kernels代表卷积核的个数`，`kernel_size代表卷积的尺寸`，`padding代表特征矩阵上下左右补零的参数`，`stride代表步距`
>
> 输入的图像shape: [224, 224, 3]， 输出特征矩阵的shape: [55, 55, `96`]  维度代表卷积核个数。
>
> shape计算：N = (W − F + 2P ) / S + 1 = [ 224 - 11 + (1 + 2)] / 4 + 1 = 55

### 2. 代码

- pytorch_alexnet 可视化

```python
import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torchvision.utils as utils
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
# function for visualizing the feature maps
def visualize_activation_maps(input, model):
    I = utils.make_grid(input, nrow=1, normalize=True, scale_each=True)
    img = I.permute((1, 2, 0)).cpu().numpy()

    conv_results = []
    x = input
    acti_map = []
    for idx, operation in enumerate(model.features):
        x = operation(x)
        if idx in {1, 4, 7, 9, 11}:
            conv_results.append(x)
    for i in range(5):
        conv_result = conv_results[i]
 
        mean_acti_map = torch.mean(conv_result, dim=1, keepdim=True)
   
        mean_acti_map = F.interpolate(mean_acti_map, size=[224,224], mode='bilinear', align_corners=False)
        
        map_grid = utils.make_grid(mean_acti_map, nrow=1, normalize=True, scale_each=True)

        map_grid = map_grid.permute((1, 2, 0)).mul(255).byte().cpu().numpy()

        map_grid = cv2.applyColorMap(map_grid, cv2.COLORMAP_JET)
        map_grid = cv2.cvtColor(map_grid, cv2.COLOR_BGR2RGB)
        map_grid = np.float32(map_grid) / 255

        visual_acti_map = 0.6 * img + 0.4 * map_grid
        tensor_visual_acti_map = torch.from_numpy(visual_acti_map).permute(2, 0, 1)
        acti_map.append(tensor_visual_acti_map)
        file_name_visual_acti_map = 'conv{}_activation_map.jpg'.format(i+1)
        utils.save_image(tensor_visual_acti_map, file_name_visual_acti_map)

    final_tensor=torch.stack(acti_map,0) 
    image_batch = utils.make_grid(final_tensor,padding = 0)
    plt.imshow(np.transpose(image_batch, (1, 2, 0))) 
    plt.show()
    return 0

# main 
if __name__ == "__main__":
    """
    data transforms, for pre-processing the input testing image before feeding into the net
    """
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

    # obtain the file path of the testing image
    test_image_dir = './alexnet_images'
    test_image_filepath = os.path.join(test_image_dir, 'panda.jpg')
    #print(test_image_filepath)

    # open the testing image
    img = Image.open(test_image_filepath)

    # pre-process the input
    transformed_img = data_transforms(img)

    # form a batch with only one image
    batch_img = torch.unsqueeze(transformed_img, 0)


    # load pre-trained AlexNet model

    alexnet = models.alexnet(pretrained=True)
    # put the model to eval mode for testing
    alexnet.eval()

    # obtain the output of the model
    output = alexnet(batch_img)

    
    # obtain the activation maps
    visualize_activation_maps(batch_img, alexnet)

    # map the class no. to the corresponding label
    with open('class_names_ImageNet.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]
    

    
    # sort the probability vector in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = F.softmax(output, dim=1)[0] * 100.0
    # obtain the first 5 classes (with the highest probability) the input belongs to
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
    for i in range(5):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))
```

- pytorch_alexnet

```python
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
'''
modified to fit dataset size
'''
NUM_CLASSES = 10

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model
```

- TFLearn

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

epochs = 4

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=3,tensorboard_dir='D:/tmp/')
history = model.fit(X, Y, n_epoch=epochs, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=20,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
model.save('my_model.tflearn')
# Load a model
model.load('my_model.tflearn')
scores = model.evaluate(X, Y)
print(scores)
```

### 3. 学习资源

- https://github.com/liudongdong1/deep-learning-for-image-processing
- 代码记录（TFlearn 版本，pytorch_各种模型，alexnet可视化）： D:\projectBack\Alexnet

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/alexnet/  

