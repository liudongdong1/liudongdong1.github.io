# tensorboard


### 1. 仪表板

- **Scalars** 显示`损失和指标`在每个时期如何变化。 您还可以使用它来跟踪`训练速度，学习率和其他标量值。`
- **Graphs** 可帮助您`可视化模型`。 在这种情况下，将显示层的Keras图，这可以帮助您确保正确构建。
- **Distributions** 和 **Histograms** 显示`张量随时间的分布`。 这对于可视化权重和偏差并验证它们是否以预期的方式变化很有用。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210512140806.png)

### 2. 调用

#### .1. keras

- 通过tf.keras.callbacks.TensorBoard 回调函数

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
```

### 3. 错误记录

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210512133910100.png)

```shell
tensorboard --logdir logs/scalars
```

1. 检查是否安装对应的环境

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210512133956.png)

2. --logdir 路径是否写对

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210512140722.png)

### 4. pytorch 使用[tensorboardX](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding) [官网api](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding)

#### .1. SummaryWriter

> `torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')`功能：提供创建 event file 的高级接口
>
> - log_dir：`event file 输出文件夹`，默认为`runs`文件夹
> - comment：不指定 log_dir 时，`runs`文件夹里的`子文件夹后缀`
> - filename_suffix：`event_file 文件名后缀`

```python
log_dir = "./train_log/test_log_dir"
writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix="12345678")
# writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")
for x in range(100):
    writer.add_scalar('y=pow_2_x', 2 ** x, x)
    writer.close()
```

运行后会生成`train_log/test_log_dir`文件夹，里面的 event file 文件名后缀是`12345678`。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/L3Byb3h5L2h0dHBzL2ltYWdlLnpoYW5neGlhbm4uY29tLzIwMjAwNzAzMTcyMzA5LnBuZw==.jpg)

但是我们指定了`log_dir`，`comment`参数没有生效。如果想要`comment`参数生效，把`SummaryWriter`的初始化改为`writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")`，生成的文件夹如下，`runs`里的子文件夹后缀是`_scalars`。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/L3Byb3h5L2h0dHBzL2ltYWdlLnpoYW5neGlhbm4uY29tLzIwMjAwNzAzMTcyODMyLnBuZw==.jpg)

#### .2. add_scalar

> `add_scalar(tag, scalar_value, global_step=None, )`：将我们所需要的数据保存在文件里面供可视化使用
>
> - tag（字符串）：保存`图的名称`
> - scalar_value（浮点型或字符串）：`y轴数据（步数）`
> - global_step（int）：`x轴数据`

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from common_tools import set_seed
max_epoch = 100
writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
for x in range(max_epoch):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow_2_x', 2 ** x, x)
    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x)}, x)
    writer.close()
```

#### .3. add_image

> `add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats=‘CHW’)`:绘制图片，可用于检查模型的输入，`监测 feature map 的变化，或是观察 weight`。
>
> - tag：就是保存图的名称
> - img_tensor:`图片的类型要是torch.Tensor, numpy.array, or string这三种`
> - global_step：`第几张图片`
> - dataformats=`‘CHW’，默认CHW，tensor是CHW，numpy是HWC`

```python
#image_PIL 读取
from torch.utils.tensorboard import SummaryWriter
import  numpy as np
from PIL import Image
writer = SummaryWriter("logs")
image_path = "C:/Users/msi/Desktop/20200103_212904.jpg"
image_PIL = Image.open(image_path)
img = np.array(image_PIL)
print(img.shape)
writer.add_image("test", img, 1, dataformats='HWC')
writer.close()

#opencv 读取
from torch.utils.tensorboard import SummaryWriter
import cv2
from torchvision import transforms
writer = SummaryWriter("logs")
image = cv2.imread('C:/Users/msi/Desktop/20200103_212904.jpg')
print(image.shape)
tran = transforms.ToTensor()
img_tensor = tran(image)
print(img_tensor.shape)
writer.add_image("test", img_tensor, 1)
writer.close()
```

#### .4. make_grid

> `torchvision.utils.make_grid(tensor: Union[torch.Tensor, List[torch.Tensor]], nrow: int = 8, padding: int = 2, normalize: bool = False, range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0)`
>
> - tensor：图像数据，$B \times C \times H \times W$ 的形状
> - nrow：行数(列数是自动计算的，为：$\frac{B}{nrow}$)
> - padding：图像间距，单位是像素，默认为 2
> - normalize：是否将像素值标准化到 [0, 255] 之间
> - range：标准化范围，例如原图的像素值范围是 [-1000, 2000]，设置 range 为 [-600, 500]，那么会把小于 -600 的像素值变为 -600，那么会把大于 500 的像素值变为 500，然后标准化到 [0, 255] 之间
> - scale_each：是否单张图维度标准化
> - pad_value：间隔的像素值
>
> 下面的代码是人民币图片的网络可视化，batch_size 设置为 16，nrow 设置为 4，得到 4 行 4 列的网络图像

```python
writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
split_dir = os.path.join(enviroments.project_dir, "data", "rmb_split")
train_dir = os.path.join(split_dir, "train")
# train_dir = "path to your training data"
# 先把宽高缩放到 [32， 64] 之间，然后使用 toTensor 把 Image 转化为 tensor，并把像素值缩放到 [0, 1] 之间
transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
data_batch, label_batch = next(iter(train_loader))
img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
# img_grid = vutils.make_grid(data_batch, nrow=4, normalize=False, scale_each=False)
writer.add_image("input img", img_grid, 0)
writer.close()
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210716202400276.png)

#### .5. AlexNet 卷积核与特征图可视化

```python
writer = SummaryWriter(comment='test_your_comment',filename_suffix="_test_your_filename_suffix")
alexnet = models.alexnet(pretrained=True)
# 当前遍历到第几层网络的卷积核了
kernel_num = -1
# 最多显示两层网络的卷积核:第 0 层和第 1 层
vis_max = 1
# 获取网络的每一层
for sub_module in alexnet.modules():
    # 判断这一层是否为 2 维卷积层
    if isinstance(sub_module, nn.Conv2d):
        kernel_num += 1
        # 如果当前层大于1，则停止记录权值
        if kernel_num > vis_max:
            break
            # 获取这一层的权值
            kernels = sub_module.weight
            # 权值的形状是 [c_out, c_int, k_w, k_h]
            c_out, c_int, k_w, k_h = tuple(kernels.shape)
            # 根据输出的每个维度进行可视化
            for o_idx in range(c_out):
                # 取出的数据形状是 (c_int, k_w, k_h)，对应 BHW; 需要扩展为 (c_int, 1, k_w, k_h)，对应 BCHW
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)   # make_grid需要 BCHW，这里拓展C维度
                # 注意 nrow 设置为 c_int，所以行数为 1。在 for 循环中每 添加一个，就会多一个 global_step
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)
                # 因为 channe 为 3 时才能进行可视化，所以这里 reshape
                kernel_all = kernels.view(-1, 3, k_h, k_w)  #b, 3, h, w
                kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
                writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=kernel_num+1)
                print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))
                writer.close()
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210716202556959.png)

#### .6. AlexNet第一个卷积层的输出进行可视化

> 把 AlexNet 的第一个卷积层的输出进行可视化，首先对图片数据进行预处理(resize，标准化等操作)。由于在定义模型时，网络层通过nn.Sequential() 堆叠，保存在 features 变量中。因此通过 features 获取第一个卷积层。把图片输入卷积层得到输出，形状为 (1, 64, 55, 55)，需要转换为 (64, 1, 55, 55)，对应 (B, C, H, W)，nrow 设置为 8，最后进行可视化

```python
writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
# 数据
path_img = "./lena.png"     # your path to image
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
norm_transform = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transform
])
img_pil = Image.open(path_img).convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)    # chw --> bchw   这个后期可能经常用到
    # 模型
    alexnet = models.alexnet(pretrained=True)
    # forward
    # 由于在定义模型时，网络层通过nn.Sequential() 堆叠，保存在 features 变量中。因此通过 features 获取第一个卷积层
    convlayer1 = alexnet.features[0]
    # 把图片输入第一个卷积层
    fmap_1 = convlayer1(img_tensor)
    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)
    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    writer.close()
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210716202936857.png)

#### .7. add_graph

> `add_graph(model, input_to_model=None, verbose=False)`
>
> - model：模型，`必须继承自 nn.Module`
> - input_to_model：`输入给模型的数据，形状为 BCHW`
> - verbose：`是否打印图结构信息`

```python
writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
    # 模型
    fake_img = torch.randn(1, 3, 32, 32)
    lenet = LeNet(classes=2)
    writer.add_graph(lenet, fake_img)
    writer.close()
```

#### .8. summary

> `torchsummary.summary(model, input_size, batch_size=-1, device="cuda")`
>
> - model：`pytorch 模型，必须继承自 nn.Module`
> - input_size：`模型输入 size，形状为 CHW`
> - batch_size：`batch_size，默认为 -1，在展示模型每层输出的形状时显示的 batch_size`
> - device：`"cuda"或者"cpu"`

```python
# 模型计算图的可视化还是比较复杂，不够清晰。而torchsummary能够查看模型的输入和输出的形状，可以更加清楚地输出模型的结构。
lenet = LeNet(classes=2)
print(summary(lenet, (3, 32, 32), device="cpu"))
```

```shell
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             456
            Conv2d-2           [-1, 16, 10, 10]           2,416
            Linear-3                  [-1, 120]          48,120
            Linear-4                   [-1, 84]          10,164
            Linear-5                    [-1, 2]             170
================================================================
Total params: 61,326
Trainable params: 61,326
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.05
Params size (MB): 0.23
Estimated Total Size (MB): 0.30
----------------------------------------------------------------
None
```

#### .9. Optimizer

> Optimizer 类：
>
> - defaults：优化器的超参数，如 weight_decay，momentum
> - state：参数的缓存，如 momentum 中需要用到前几次的梯度，就缓存在这个变量中
> - param_groups：管理的参数组，是一个 list，其中每个元素是字典，包括 momentum、lr、weight_decay、params 等。
> - _step_count：记录更新 次数，在学习率调整中使用

#### .10. add_histogram

> `add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)`
>
> 功能：统计直方图与多分位折线图
>
> - tag：图像的标签名，图的唯一标识
> - values：`要统计的参数，通常统计权值、偏置或者梯度`
> - global_step：第几个子图
> - bins：`取直方图的 bins`

```python
writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
for x in range(2):
    np.random.seed(x)
    data_union = np.arange(100)
    data_normal = np.random.normal(size=1000)
    writer.add_histogram('distribution union', data_union, x)
    writer.add_histogram('distribution normal', data_normal, x)
    plt.subplot(121).hist(data_union, label="union")
    plt.subplot(122).hist(data_normal, label="normal")
    plt.legend()
    plt.show()
    writer.close()
```

#### .11. **Linear 案例**

```python
# -*- coding: utf-8 -*-
# @Author  : Miaoshuyu
# @Email   : miaohsuyu319@163.com
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

input_size = 1
output_size = 1
num_epoches = 60
learning_rate = 0.01
writer = SummaryWriter(comment='Linear')
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    output = model(inputs)
    loss = criterion(output, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 保存loss的数据与epoch数值
    writer.add_scalar('Train', loss, epoch)
    if (epoch + 1) % 5 == 0:
        print('Epoch {}/{},loss:{:.4f}'.format(epoch + 1, num_epoches, loss.item()))

# 将model保存为graph
writer.add_graph(model, (inputs,))

predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
writer.close()
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210716195633883.png)

#### .12. **官网案例**

```python
# -*- coding: utf-8 -*-
# @Author  : Miaoshuyu
# @Email   : miaohsuyu319@163.com
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

for n_iter in range(100):
    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)

    dummy_img = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter % 10 == 0:
        x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)

        dummy_audio = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # amplitude of sound should in [-1, 1]
            dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
        writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

        for name, param in resnet18.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

        # needs tensorboard 0.4RC or later
        writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]

features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
```

#### .13. add_embedding

> `add_embedding`(*mat*, *metadata=None*, *label_img=None*, *global_step=None*, *tag='default'*, *metadata_header=None*)
>
> - **mat** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or* *numpy.array*) – A matrix which each row is the feature vector of the data point,(N, D)(*N*,*D*), where N is number of data and D is feature dimension
> - **metadata** ([*list*](https://docs.python.org/3/library/stdtypes.html#list)) – A list of labels, each element will be convert to string
> - **label_img** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Images correspond to each data point, label_img: (N, C, H, W)(*N*,*C*,*H*,*W*)
> - **global_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Global step value to record
> - **tag** (*string*) – Name for the embedding

```python
import keyword
import torch
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0

writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.add_embedding(torch.randn(100, 5), label_img=label_img)
writer.add_embedding(torch.randn(100, 5), metadata=meta)
```



### 5. pytorch 模板代码

```python
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from matplotlib import pyplot as plt
from model.lenet import LeNet
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed

set_seed()  # 设置随机种子
rmb_label = {"1": 0, "100": 1}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1

# ============================ step 1/5 数据 ============================

split_dir = os.path.join("..", "..", "data", "rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

net = LeNet(classes=2)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

iter_count = 0

# 构建 SummaryWriter
writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, data in enumerate(train_loader):

        iter_count += 1

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

        # 记录数据，保存于event file
        writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
        writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

    # 每个epoch，记录梯度，权值
    for name, param in net.named_parameters():
        writer.add_histogram(name + '_grad', param.grad, epoch)
        writer.add_histogram(name + '_data', param, epoch)

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            valid_curve.append(loss.item())
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct / total))

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
            writer.add_scalars("Accuracy", {"Valid": correct_val / total_val}, iter_count)

train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()
```



### Resource

- 各种demo使用python examples/： https://github.com/lanpa/tensorboardX


---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: liudongdong1.github.io/tensorboard/  

