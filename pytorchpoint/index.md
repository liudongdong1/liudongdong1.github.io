# PytorchPoint


## 1.代码片段

#### 1.1.导入配置

```python
import torch
import torch.nn as nn
import torchvision
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
```

#### 1.2. 显卡设置

```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#这只指定多张显卡
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#清除显存
torch.cuda.empty_cache()
```

#### 1.3. Tensor 处理

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200830084404.png)

```python
tensor = torch.randn(3,4,5)
print(tensor.type())  # 数据类型
print(tensor.size())  # 张量的shape，是个元组
print(tensor.dim())   # 维度的数量
```

- torch reshape操作

```python
d=torch.reshape(c,(5,2))
```

- 数据类型转化

```python
# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor
torch.set_default_tensor_type(torch.FloatTensor)
# 类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()
```

- torch.Tensor&& np.ndarray

```python
#除了CharTensor，其他所有CPU上的张量都支持转换为numpy格式然后再转换回来。
ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float() # If ndarray has negative stride
```

- torch.Tensor&&PIL.Image

```python
# pytorch中的张量默认采用[N, C, H, W]的顺序，并且数据范围在[0,1]，需要进行转置和规范化
# torch.Tensor -> PIL.Image
image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0, max=255).byte().permute(1,2,0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way

# PIL.Image -> torch.Tensor
path = r'./figure.jpg'
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) # Equivalently way
```

- np.ndarray&&PIL.Image

```python
image=PIL.Image.fromarray(ndarray.astype(np.uint8))
ndarray=np.asarray(PIL.Image.open(path))
```

```python
# 从只包含一个元素的张量中取值
value=torch.rand(1).item()
#张量形变
#相比于torch.view, torch.reshape可以自动处理张量不连续的情况
tensor=torch.rand(2,3,4)
shape=(6,4)
tensor=torch.reshape(tensor,shape)
#打乱顺序
tensor=tensor[torch.randperm(tensor.size(0))] #打乱第一维度
```

- 张量复制

| Operation             | New/Shared memory | Still in computation graph |
| --------------------- | ----------------- | -------------------------- |
| tensor.clone()        | New               | Yes                        |
| tensor.detach()       | shared            | no                         |
| tensor.detach.clone() | new               | no                         |

- 张量拼接

```python
'''
注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，
而torch.stack会新增一维。例如当参数是3个10x5的张量，torch.cat的结果是30x5的张量，
而torch.stack的结果是3x10x5的张量。
'''
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)
```

- one-hot编码

```python
# pytorch的标记默认从0开始
tensor = torch.tensor([0, 2, 1, 3])
N = tensor.size(0)
num_classes = 4
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
```

- 张量相等

```python
torch.allclose(tensor1, tensor2)  # float tensor
torch.equal(tensor1, tensor2)     # int tensor
```

- 张量乘法

```python
# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).
result = torch.mm(tensor1, tensor2)
# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)
result = torch.bmm(tensor1, tensor2)
# Element-wise multiplication.
result = tensor1 * tensor2
```

#### 1.4. 模型定义操作

- 俩层卷积

```python
# convolutional neural network (2 convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
model = ConvNet(num_classes).to(device)
```

- 将已有网络的所有BN层改为同步BN层

```python
def convertBNtoSyncBN(module, process_group=None):
    '''Recursively replace all BN layers to SyncBN layer.

    Args:
        module[torch.nn.Module]. Network
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        sync_bn = torch.nn.SyncBatchNorm(module.num_features, module.eps, module.momentum, 
                                         module.affine, module.track_running_stats, process_group)
        sync_bn.running_mean = module.running_mean
        sync_bn.running_var = module.running_var
        if module.affine:
            sync_bn.weight = module.weight.clone().detach()
            sync_bn.bias = module.bias.clone().detach()
        return sync_bn
    else:
        for name, child_module in module.named_children():
            setattr(module, name) = convert_syncbn_model(child_module, process_group=process_group))
        return module
```

- 查看网络参数

```python
params = list(model.named_parameters())
(name, param) = params[28]
print(name)
print(param.grad)
```

- [模型可视化](szagoruyko/pytorchvizgithub.com)
- 模型权重初始化

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

- 提取网络中某一层

```python
# 取模型中的前两层
new_model = nn.Sequential(*list(model.children())[:2] 
# 如果希望提取出模型中的所有卷积层，可以像下面这样操作：
for layer in model.named_modules():
    if isinstance(layer[1],nn.Conv2d):
         conv_model.add_module(layer[0],layer[1])
```

- 模型加载

```python
model.load_state_dict(torch.load('model.pth'), strict=False)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
#导入另一个模型的相同部分到新的模型
# model_new代表新的模型
# model_saved代表其他模型，比如用torch.load导入的已保存的模型
model_new_dict = model_new.state_dict()
model_common_dict = {k:v for k, v in model_saved.items() if k in model_new_dict.keys()}
model_new_dict.update(model_common_dict)
model_new.load_state_dict(model_new_dict)
```

#### 1.5. 数据处理

- 计算数据集均值&标准差

```python
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
def compute_mean_and_std(dataset):
    # 输入PyTorch的dataset，输出均值和标准差
    mean_r = 0
    mean_g = 0
    mean_b = 0

    for img, _ in dataset:
        img = np.asarray(img) # change PIL Image to numpy array
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:
        img = np.asarray(img)

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std
```

后续学习链接: https://mp.weixin.qq.com/s/JnIO_HjTrC0DCWtKrkYC8A

# pytorch 书籍


![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPotLDYAianDupMzw2shaQ9voSo3EpvNMV5YTHKSRDMFapGheP3eARD9Ew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPol4iavYh9dyghIC59B7G0IFyROII0odKCLLicSJEVUAsKk8PMXYRFRGsA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPohOYu7cL2ia33q4kk840JYZJWU06mFcknicj4eFD05jSs35EviaEiad9RFw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPoP5e4Y8dgHvwYZOjiabvBzHlhpUTTEYicJqibGHqKPz30BWlgyictjYT2Tg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPo0rFPgzldhfoqZ6n5TW1fZk4icpUgF127moNSrO65S3sgMJCzP5Bz5uA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPozKO6je8bjk5REukz24LP7I19wFCe37v0vokI0mN1ABatbwo4a5Dpcw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/teF4oHzZ4IRTj9icBYLjWTcTrM8QTCtPocADHMqrwKwmZD80ibb3xQOicf8qMpLfX2w4VPOHgl6U6gZlTg3sFFsCw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### learning from:  https://www.learnopencv.com/


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/pytorchpoint/  

