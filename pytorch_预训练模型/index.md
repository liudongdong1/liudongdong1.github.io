# Pytorch_预训练模型


### 1. 模型[下载](http://www.cxyzjd.com/article/Jorbo_Li/106248808)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210716115238769.png)

```python
import re
import os
import glob
import torch
from torch.hub import download_url_to_file
from torch.hub import urlparse
import torchvision.models as models

def download_model(url, dst_path):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    
    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = HASH_REGEX.search(filename).group(1)
    if os.path.exists(os.path.join(dst_path, filename)):
        return filename
    download_url_to_file(url, os.path.join(dst_path, filename), hash_prefix, True)
    return filename


def saveToFolder(path):
    #其他各种模型可以在这个目录下进行搜索查看 https://github.com/pytorch/vision/tree/master/torchvision/models
    # model_urls = {
    #     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    #     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    #     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    #     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    #     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    #     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    #     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    #     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # }
    model_urls={
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    }
    if not (os.path.exists(path)):
        os.makedirs(path)

    for url in model_urls.values():
        download_model(url, path)

def load_model(model_name, model_dir):
    model  = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)
    model_path = glob.glob(path_format)[0]
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    path = '/home/iot/jupyter/root_dir/liudongdong/pytorch_demo/pretainedpth/vgg16'
    saveToFolder(path)
    model = load_model('vgg16', path)
    print(model)
if __name__ == "__main__":
    main()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210716115155027.png)

### 2. 模型查看

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210516230215346.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210516230252202.png)

```python
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 100)
print(resnet)   #将会输出网络每一层结构, print顺序可能不是最终网络的数据
# 或者采用torchviz模块，对网络结构进行可视化， 将会生成一个pdf 网络结构图
import torch
import torchvision
from torchviz import make_dot
x = torch.randn(10, 3, 224, 224).requires_grad_(True)
model50 = torchvision.models.resnet50()
y = model50(x)
vis_graph = make_dot(y, params=dict(list(resnet.named_parameters()) + [('x', x)]))
vise_graph.view()
#保存成文件形式
vise_graph.render(filename='resnet50', view=False, format='pdf')
```

- 方式二：以列表的形式

```python
from torchsummary import summary
summary(model50, (3, 224, 224)) #模型参数，输入数据的格式
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211011104049510.png)

### 3. 模型初始化

> 适当的权值初始化可以加速模型的训练和模型的收敛，而错误的权值初始化会导致梯度消失/爆炸，从而无法完成网络的训练，因此需要控制网络输出值的尺度范围。torch.nn.init中提供了常用的初始化方法函数，1. Xavier，kaiming系列；2. 其他方法分布

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210517095212696.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210517095258942.png)

> **从上图中的公式可以看出，\**每传播一层，输出值数据的方差就会扩大n\**** ***\*倍\**，要想控制输出H的尺度范围，只需要控制H的方差为1，则无论经过多少层都可以维持在初始输入X的方差附近，因此\**权重w需要初始化方差为1/n\**（n为神经元的个数）**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210517100543137.png)

##### .1. Xavier 均匀分布

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210517095432537.png)

```python
import os
import torch
import random
import numpy as np
import torch.nn as nn
 
 
 
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
 
set_seed(1)  # 设置随机种子
 
 
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num
 
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.tanh(x)
 
            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break
 
        return x
 
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #xavier手动计算
                a = np.sqrt(6 / (self.neural_num + self.neural_num))
                tanh_gain = nn.init.calculate_gain('tanh')         #计算增益
                a *= tanh_gain
                nn.init.uniform_(m.weight.data, -a, a)
 
                #调用pytorch实现xavier初始化，适用于饱和激活函数
                # tanh_gain = nn.init.calculate_gain('tanh')
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
 
 
# flag = 0
flag = 1
 
if flag:
    layer_nums = 100
    neural_nums = 256
    batch_size = 16
 
    net = MLP(neural_nums, layer_nums)
    net.initialize()
 
    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1
 
    output = net(inputs)
    print(output)
```

> torch.nn.init.xavier_uniform_(tensor, gain=1)
>
> xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，
>
> 这里有一个gain，增益的大小是依据激活函数类型来设定
>
> ```python
> eg：nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
> ```

##### .2. Xavier正态分布

> torch.nn.init.xavier_normal_(*tensor*, *gain=1*)
>
> xavier初始化方法中服从正态分布，
>
> mean=0,std = gain * sqrt(2/fan_in + fan_out)

##### .3. kaiming均匀分布

> torch.nn.init.kaiming_uniform_(*tensor*, *a=0*, *mode='fan_in'*, *nonlinearity='leaky_relu'*)
>
> 此为均匀分布，U～（-bound, bound）, bound = sqrt(6/(1+a^2)*fan_in)
>
> 其中，a为激活函数的负半轴的斜率，relu是0
>
> mode- 可选为fan_in 或 fan_out, fan_in使正向传播时，方差一致; fan_out使反向传播时，方差一致
>
> nonlinearity- 可选 relu 和 leaky_relu ，默认值为 。 leaky_relu
>
> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210517100453509.png)

```python
import os
import torch
import random
import numpy as np
import torch.nn as nn
 
 
 
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
 
set_seed(1)  # 设置随机种子
 
 
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num
 
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
 
            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break
 
        return x
 
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #kaiming初始化手动
                nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
 
                #kaiming初始化
                # nn.init.kaiming_normal_(m.weight.data)
 
 
# flag = 0
flag = 1
 
if flag:
    layer_nums = 100
    neural_nums = 256
    batch_size = 16
 
    net = MLP(neural_nums, layer_nums)
    net.initialize()
 
    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1
 
    output = net(inputs)
    print(output)
```



##### .4. kaiming 正态分布

> torch.nn.init.kaiming_normal_(*tensor*, *a=0*, *mode='fan_in'*, *nonlinearity='leaky_relu'*)
>
> 此为0均值的正态分布，N～ (0,std)，其中std = sqrt(2/(1+a^2)*fan_in)
>
> 其中，a为激活函数的负半轴的斜率，relu是0
>
> mode- 可选为fan_in 或 fan_out, fan_in使正向传播时，方差一致;fan_out使反向传播时，方差一致
>
> nonlinearity- 可选 relu 和 leaky_relu ，默认值为 。 leaky_relu
>
> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

##### .5. 均匀初始化分布

> torch.nn.init.uniform_(*tensor*, *a=0*, *b=1*)
>
> 使值服从均匀分布U(a,b)

##### .6. 正态初始化分布

> torch.nn.init.normal_(*tensor*, *mean=0*, *std=1*)
>
> 使值服从正态分布N(mean, std)，默认值为0，1

##### .7. 常数初始化

> torch.nn.init.constant_(*tensor*, *val*)
>
> 使值为常数val nn.init.constant_(w, 0.3)

##### .8. 单位矩阵初始化

> torch.nn.init.eye_(*tensor*)
>
> 将二维tensor初始化为单位矩阵（the identity matrix）

##### .9. 正交初始化

> torch.nn.init.orthogonal_(*tensor*, *gain=1*)
>
> 使得tensor是正交的，论文:Exact solutions to the nonlinear dynamics of learning in deep linear neural networks” - Saxe, A. et al. (2013)

##### .10. 稀疏初始化

> torch.nn.init.sparse_(*tensor*, *sparsity*, *std=0.01*)
>
> 从正态分布N～（0. std）中进行稀疏化，使每一个column有一部分为0
>
> sparsity- 每一个column稀疏的比例，即为0的比例
>
> nn.init.sparse_(w, sparsity=0.1)

> 注意 model.modules()和 model.children()的区别：**model.modules()**会迭代地遍历模型的**所有子层**，而**model.children()**只会遍历模型下的一层。

- **对网络中某一层进行初始化**

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
init.xavier_uniform(self.conv1.weight)
init.constant(self.conv1.bias, 0.1)
```

- **对网络整体进行初始化**

```python
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier(m.weight.data)
        xavier(m.bias.data)
net = Net()#构建网络
net.apply(weights_init) #apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。  
 #对所有的Conv层都初始化权重. 
```

- **权重初始化**

```python
# Common practise for initialization.
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)
```

- **对指定层进行Finetune**

```python
count = 0
para_optim = []
for k in model.children():
    count += 1
    # 6 should be changed properly
    if count > 6:
        for param in k.parameters():
            para_optim.append(param)
            else:
                for param in k.parameters():
                    param.requires_grad = False
optimizer = optim.RMSprop(para_optim, lr)
```

- **对固定部分参数训练**

```python
# 只有True的才训练
optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
		#前面的参数就是False，而后面的不变
        for p in self.parameters():
            p.requires_grad=False
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

- **优化**

```python
optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], betas=(args['momentum'], 0.999))
```

- 加载部分权重

```python
# 获得模型的键值
keys=[]
for k,v in desnet.state_dict().items():
    if v.shape:
        keys.append(k)
    print(k,v.shape)  
# 从预训练文件中加载权重
state={}
pretrained_dict = torch.load('/home/lulu/pytorch/Paper_Code/weights/densenet121-a639ec97.pth')
for i,(k,v) in enumerate(pretrained_dict.items()):
    if 'classifier' not in k:
        state[keys[i]] = v
# 保存权重
torch.save(state,'/home/lulu/pytorch/Paper_Code/weights/densenet121.pth')
```

### 4. 构建模型

> - Sequential：`顺序性，各网络层之间严格按照顺序执行，常用语block构建`
> - ModuleList：`迭代性，常用于大量重复网络构建，通过for循环实现重复构建`
> - ModuleDict：`索引性，常用于可选择的网络层`

##### .1. nn.Sequential

```python
# ============================ Sequential
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)
 
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),)
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
```

##### .2. nn.ModuleList

> 功能：`像python的**list**一样包装多个网络层，以迭代的方式调用网络层`
>
> - append（）：在modulelist后面**添加**网络层
> - extend（）：**拼接**两个modulelist
> - insert（）：在modulelist中指定位置**插入**网络层

```python
class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])
    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x
net = ModuleList()
print(net)
fake_data = torch.ones((10, 10))
output = net(fake_data)
print(output)
```

##### .3. nn.ModuleDict

> 功能：`像python的dict一样包装多个网络层（每一个给一个key，可通过key索引网络层）`
>
> - clear（）：清空moduleDict
> - items（）：返回可迭代的键值对（key-value pairs）
> - keys（）：返回字典的key
> - values（）：返回字典的value
> - pop（）：返回一对键值，并从字典中删除

```python
# ============================ ModuleDict
class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })
    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
net = ModuleDict()
fake_img = torch.randn((4, 10, 32, 32))
output = net(fake_img, 'conv', 'relu')
#prelu输出结果有负值，改为relu后输出没有负数，可以检查是不是按照我们的想法运行的
print(output)
```

### 5. 使用预训练模型

#### .0. AlexNet 预训练模型修改

###### 1. 直接使用AlexNet，并添加可视化

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
import argparse

"""
input commands
"""
paser = argparse.ArgumentParser()
paser.add_argument("--test_img", type=str, default='whippet.jpg', help="testing image")
opt = paser.parse_args()

# function for visualizing the feature maps
def visualize_activation_maps(input, model):
    I = utils.make_grid(input, nrow=1, normalize=True, scale_each=True)
    img = I.permute((1, 2, 0)).cpu().numpy()

    conv_results = []
    x = input
    for idx, operation in enumerate(model.features):
        x = operation(x)
        if idx in {1, 4, 7, 9, 11}:
            conv_results.append(x)
    
    for i in range(5):
        conv_result = conv_results[i]
        N, C, H, W = conv_result.size()

        mean_acti_map = torch.mean(conv_result, 1, True)
        mean_acti_map = F.interpolate(mean_acti_map, size=[224,224], mode='bilinear', align_corners=False)

        map_grid = utils.make_grid(mean_acti_map, nrow=1, normalize=True, scale_each=True)
        map_grid = map_grid.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
        map_grid = cv2.applyColorMap(map_grid, cv2.COLORMAP_JET)
        map_grid = cv2.cvtColor(map_grid, cv2.COLOR_BGR2RGB)
        map_grid = np.float32(map_grid) / 255

        visual_acti_map = 0.6 * img + 0.4 * map_grid
        tensor_visual_acti_map = torch.from_numpy(visual_acti_map).permute(2, 0, 1)

        file_name_visual_acti_map = 'conv{}_activation_map.jpg'.format(i+1)
        utils.save_image(tensor_visual_acti_map, file_name_visual_acti_map)

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
    test_image_filepath = os.path.join(test_image_dir, opt.test_img)
    #print(test_image_filepath)

    # open the testing image
    img = Image.open(test_image_filepath)
    print("original image's shape: " + str(img.size))
    # pre-process the input
    transformed_img = data_transforms(img)
    print("transformed image's shape: " + str(transformed_img.shape))
    # form a batch with only one image
    batch_img = torch.unsqueeze(transformed_img, 0)  #Returns a new tensor with a dimension of size one inserted at the specified position.
    print("image batch's shape: " + str(batch_img.shape))

    # load pre-trained AlexNet model
    print("\nfeed the input into the pre-trained alexnet to get the output")
    alexnet = models.alexnet(pretrained=True)
    # put the model to eval mode for testing
    alexnet.eval()

    # obtain the output of the model
    output = alexnet(batch_img)
    print("output vector's shape: " + str(output.shape))
    
    # obtain the activation maps
    visualize_activation_maps(batch_img, alexnet)

    # map the class no. to the corresponding label
    with open('class_names_ImageNet.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]
    
    # print the first 5 classes to see the labels
    print("\nprint the first 5 classes to see the lables")
    for i in range(5):
        print("class " + str(i) + ": " + str(classes[i]))
    
    # sort the probability vector in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = F.softmax(output, dim=1)[0] * 100.0
    # obtain the first 5 classes (with the highest probability) the input belongs to
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
    print("\nprint the first 5 classes the testing image belongs to")
    for i in range(5):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))

```

###### 2. 修改Alexnet最后一层

```python
import torchvision.models as models

model = models.AlexNet()
print(model)
#修改网络的第一个卷积层的输入为4通道，输出的结果预测为10个类别
model.features[0]=nn.Conv2d(4, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
model.classifier[6] = nn.Linear(4096,10)

print(model)
```

```python
model = cifar10_cnn.CIFAR10_Nettest()
pretrained_dict = torch.load('models/cifar10_statedict.pkl')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print(model)
new_model_dict = model.state_dict()
dict_name = list(new_model_dict)
for i, p in enumerate(dict_name):
    print(i, p)

print('before change:\n',new_model_dict['classifier.5.bias'])
model.classifier[5]=nn.Linear(1024,17)

change_model_dict = model.state_dict()
new_dict_name = list(change_model_dict)
print('after change:\n',change_model_dict['classifier.5.bias'])
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210716225130620.png)

```python
import torch.nn as nn
from torchvision import models

class BuildAlexNet(nn.Module):
    def __init__(self, model_type, n_output):
        super(BuildAlexNet, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            model = models.alexnet(pretrained=True)
            self.features = model.features
            fc1 = nn.Linear(9216, 4096)
            fc1.bias = model.classifier[1].bias
            fc1.weight = model.classifier[1].weight
            
            fc2 = nn.Linear(4096, 4096)
            fc2.bias = model.classifier[4].bias
            fc2.weight = model.classifier[4].weight
            
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    fc1,
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    fc2,
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, n_output))  
            # 
#            model.classifier[6]==nn.Linear(4096,n_output)
#            self.classifier = model.classifier
        if model_type == 'new':
            self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 11, 4, 2),
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(64, 192, 5, 1, 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(192, 384, 3, 1, 1),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(384, 256, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 0))
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(9216, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, n_output))
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out  = self.classifier(x)
        return out
```

```python
import numpy as np
from torch.autograd import Variable
import torch

if __name__ == '__main__':
    model_type = 'pre'
    n_output = 10
    alexnet = BuildAlexNet(model_type, n_output)
    print(alexnet)
    
    x = np.random.rand(1,3,224,224)
    x = x.astype(np.float32)
    x_ts = torch.from_numpy(x)
    x_in = Variable(x_ts)
    y = alexnet(x_in)
```

#### .1. ResNet参数修改

> resnet网络最后一层分类层fc是对1000种类型进行划分，对于自己的数据集，如果只有9类

```python
# coding=UTF-8
import torchvision.models as models
#调用模型
model = models.resnet50(pretrained=True)
#提取fc层中固定的参数
fc_features = model.fc.in_features
#修改类别为9
model.fc = nn.Linear(fc_features, 9)
```

#### .2. 增减卷积层

> 1、先建立好自己的网络（与预训练的模型类似，要不谈何fine-tune）
>
> 2、然后将预训练模型参数与自己搭建的网络不一致的部分参数去掉
>
> 3、将保留的合适的参数读入网络初始化，实现fine-tune的效果

```python
# -*- coding:utf-8 -*-
#####################
#建立自己的网络模型net
#####################

###然后读出预训练模型参数以resnet152为例，我不是利用程序下载的，我是习惯了下载好存储在文件夹中
pretrained_dict = torch.load(save_path)
model_dict = net.state_dict()   #(读出搭建的网络的参数，以便后边更新之后初始化）

####去除不属于model_dict的键值
pretrained_dict={ k : v for k, v in pretrained_dict.items() if k in model_dict}

###更新现有的model_dict的值
model_dict.update(pretrained_dict)

##加载模型需要的参数
net.load_state_dict(model_dict)
```

#### .3. ImageNet计算多层卷积特征

```python
class FeatureExtractor(torch.nn.Module):
    """Helper class to extract several convolution features from the given
    pre-trained model.
    Attributes:
        _model, torch.nn.Module.
        _layers_to_extract, list<str> or set<str>
    Example:
        >>> model = torchvision.models.resnet152(pretrained=True)
        >>> model = torch.nn.Sequential(collections.OrderedDict(
                list(model.named_children())[:-1]))
        >>> conv_representation = FeatureExtractor(
                pretrained_model=model,
                layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)
    """
    def __init__(self, pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)
    
    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
            return conv_representation
```

#### 4. 训练特定层，冻结其它层

> 将模型起始的一些层的权重保持不变，重新训练后面的层，得到新的权重。在这个过程中，可多次进行尝试，从而能够依据结果找到 frozen layers 和 retrain layers 之间的最佳搭配。 如何使用预训练模型，是由数据集大小和新旧数据集(预训练的数据集和自己要解决的数据集)之间数据的相似度来决定的。
>
> - `requires_grad`为False来冻结网络参数
> - **filter(lambda p: p.requires_grad, model.parameters())**过滤掉requires_grad=false的层

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210716230659569.png)

```python
#首先自己新定义一个网络
class CNN(nn.Module):
    def __init__(self, block, layers, num_classes=9): 
        #自己新定义的CNN与继承的ResNet网络结构大体相同，即除了新增层，其他层的层名与ResNet的相同。

        self.inplanes = 64 
        super(ResNet, self).__init__() #继承ResNet网络结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) 
        self.avgpool = nn.AvgPool2d(7, stride=1)

        #新增一个反卷积层 
        self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=False, dilation=1) 

        #新增一个最大池化层 
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 

        #将原来的fc层改成fclass层 
        self.fclass = nn.Linear(2048, num_classes) #原来的fc层：self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules(): #
            if isinstance(m, nn.Conv2d): 
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels #
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
                elif isinstance(m, nn.BatchNorm2d): 
                    m.weight.data.fill_(1) 
                    m.bias.data.zero_() 
                    def _make_layer(self, block, planes, blocks, stride=1): 
                        downsample = None 
                        if stride != 1 or self.inplanes != planes * block.expansion: 
                            downsample = nn.Sequential( 
                                nn.Conv2d(self.inplanes, planes * block.expansion, 
                                          kernel_size=1, stride=stride, bias=False), 
                                nn.BatchNorm2d(planes * block.expansion), 
                            ) 
                            layers = [ ] 
                            layers.append(block(self.inplanes, planes, stride, downsample)) 
                            self.inplanes = planes * block.expansion 
                            for i in range(1, blocks): 
                                layers.append(block(self.inplanes, planes)) 
                                return nn.Sequential(*layers) 
                            def forward(self, x): 
                                x = self.conv1(x) 
                                x = self.bn1(x) 
                                x = self.relu(x) 
                                x = self.maxpool(x) 
                                x = self.layer1(x) 
                                x = self.layer2(x) 
                                x = self.layer3(x) 
                                x = self.layer4(x) 
                                x = self.avgpool(x) 
                                #3个新加层的forward 
                                x = x.view(x.size(0), -1) 

                                #因为接下来的self.convtranspose1层的输入通道是2048
                                x = self.convtranspose1(x) 
                                x = self.maxpool2(x) 
                                x = x.view(x.size(0), -1)  

                                #因为接下来的self.fclass层的输入通道是2048 
                                x = self.fclass(x) 
                                return x
                            #加载model 
                            resnet50 = models.resnet50(pretrained=True) 
                            cnn = CNN(Bottleneck, [3, 4, 6, 3]) #创建一个自己新定义的网络对象cnn。
```

```python
#微调全连接层
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 100)  # Replace the last fc layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

#以较大的学习率微调全连接层，较小的学习率微调卷积层
model = torchvision.models.resnet18(pretrained=True)
finetuned_parameters = list(map(id, model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'params': conv_parameters, 'lr': 1e-3}, 
              {'params': model.fc.parameters()}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

#### .5. keypointrcnn_resnet50_fpn 模型使用

```python
import torch
import torchvision
import torch.nn as nn
def get_model(num_kpts,train_kptHead=False,train_fpn=True):
    is_available = torch.cuda.is_available()
    device =torch.device('cuda:0' if is_available else 'cpu')
    dtype = torch.cuda.FloatTensor if is_available else torch.FloatTensor
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    
    for i,param in enumerate(model.parameters()):
        param.requires_grad = False
        
    if train_kptHead!=False:
      for i, param in enumerate(model.roi_heads.keypoint_head.parameters()):
          if i/2>=model.roi_heads.keypoint_head.__len__()/2-train_kptHead:
            param.requires_grad = True

    if train_fpn==True:
      for param in model.backbone.fpn.parameters():
        param.requires_grad = True

    out = nn.ConvTranspose2d(512, num_kpts, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    model.roi_heads.keypoint_predictor.kps_score_lowres = out
    
    return model, device, dtype
#model, device, dtype=get_model(2)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201021161049583.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pytorch_%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/  

