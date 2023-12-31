# PytorchSegmentCode


### 0. 基础配置

#### 0.1. 设置随机种子

```python
def set_seeds(seed, cuda):
    """ Set Numpy and PyTorch seeds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    print ("==> Set NumPy and PyTorch seeds.")
```

#### 0.2. 张量处理与转化

```python
tensor.type()   # Data type
tensor.size()   # Shape of the tensor. It is a subclass of Python tuple
tensor.dim()    # Number of dimensions.

# Type convertions.
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()

#tensor 与python数据类型转化
#Tensor ----> 单个Python数据，使用data.item()，data为Tensor变量且只能为包含单个数据
#Tensor ----> Python list，使用data.tolist()，data为Tensor变量，返回shape相同的可嵌套的list

#CPU&GPU 位置
#CPU张量 ---->  GPU张量，使用data.cuda()
#GPU张量 ----> CPU张量，使用data.cpu()

#tensor 与np.ndarray
ndarray = tensor.cpu().numpy()
ndarray = tensor.numpy()
tensor.cpu().detach().numpy().tolist()[0]
# np.ndarray -> torch.Tensor.
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float()  # If ndarray has negative stride
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255
    ).byte().permute(1, 2, 0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way
# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))
    ).permute(2, 0, 1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))  # Equivalently way
# np.ndarray -> PIL.Image.
image = PIL.Image.fromarray(ndarray.astypde(np.uint8))
# PIL.Image -> np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))

#复制张量
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No                |
#reshape 操作
tensor = torch.reshape(tensor, shape)
# Expand tensor of shape 64*512 to shape 64*512*7*7.
torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)

#向量拼接 注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，而torch.stack会新增一维。例如当参数是3个10×5的张量，torch.cat的结果是30×5的张量，而torch.stack的结果是3×10×5的张量。
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)

#得到0/非0 元素
torch.nonzero(tensor)               # Index of non-zero elements
torch.nonzero(tensor == 0)          # Index of zero elements
torch.nonzero(tensor).size(0)       # Number of non-zero elements
torch.nonzero(tensor == 0).size(0)  # Number of zero elements

#向量乘法
# Matrix multiplication: (m*n) * (n*p) -> (m*p).
result = torch.mm(tensor1, tensor2)
# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p).
result = torch.bmm(tensor1, tensor2)
# Element-wise multiplication.
result = tensor1 * tensor2

#计算两组数据之间的两两欧式距离
# X1 is of shape m*d.
X1 = torch.unsqueeze(X1, dim=1).expand(m, n, d)
# X2 is of shape n*d.
X2 = torch.unsqueeze(X2, dim=0).expand(m, n, d)
# dist is of shape m*n, where dist[i][j] = sqrt(|X1[i, :] - X[j, :]|^2)
dist = torch.sqrt(torch.sum((X1 - X2) ** 2, dim=2))

#卷积核
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
```

#### 0.3. pytorch 版本

```python
torch.__version__               # PyTorch version
torch.version.cuda              # Corresponding CUDA version
torch.backends.cudnn.version()  # Corresponding cuDNN version
torch.cuda.get_device_name(0)   # GPU type
```

#### 0.4. GPU指定

```python
torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

### 1. 数据加载分割

#### 1.0. Transform 变化

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201020213213586.png)

> 其中`ToTensor操作会将PIL.Image或形状为H×W×D，数值范围为[0, 255]的np.ndarray转换为形状为D×H×W`，数值范围为[0.0, 1.0]的torch.Tensor。  Normalize 需要注意数据的维度，否则容易报错。

```python
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224,
                                             scale=(0.08, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
 ])
 val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
])
```

#### 1.1. 自定义dataset类

```python
class CharDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        # args: path to csv file with keypoint data, directory with images, transform to be applied
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        # return size of dataset
        return len(self.key_pts_frame.shape)
    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)
        # removing alpha color channel if present
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        key_pts = self.key_pts_frame.iloc[idx, 1:].values()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}
        # apply transform
        if self.transform:
            sample = self.transform(sample)
        return sample
if __name__ == "__main__":
    chardata=CharDataset("D:\\Model\\CharPointDetection\\data\\test\\")
    print(len(chardata))    #1198
    print(chardata[0].get("image").shape)  #(96, 96)  最大值1， 最小值0
```

- dataset

```python
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import cv2
from util.imageUtil import *
from util.config import *
class DatasetCustom(Dataset):
    def __init__(self, rootcsv, imgroot,train=True, transform = None,ratio=0.7):
        self.train = train
        self.transform = transform
        self.allItem=self.readcsv(rootcsv)
        self.imgroot=imgroot
        #todo 添加打乱操作 训练和测试数据集进行分割处理
        if self.train :
            self.labelItem=self.allItem[:int(len(self.allItem)*ratio)]
        else:
            self.labelItem=self.allItem[int(len(self.allItem)*ratio)+1:]


    def readcsv(self,filename):
        '''
            读取CSV中clothdata数据
        '''
        with open(filename,encoding = 'utf-8') as f:
            data = np.loadtxt(f,str,delimiter = ",", skiprows = 1)
            data=data[::2,:]     #或取csv 文件数据
            return data

    def __getitem__(self, index):
        index=index%self.__len__()
        img_name = self.labelItem[index][0].split('_')  # 或取图片对于路径
        imgpath="{}/camera{}_{}_{}_{}.jpg".format(self.imgroot,img_name[0],img_name[1],0-int(img_name[1]),img_name[2])
        ratioW,ratioH,img=imageloadCV(imgpath,RESIZE)  #图片大小进行了resize处理，对于x,y也进行缩放处理
        keypoints = self.labelCoordinateHandle(self.labelItem[index][10:],ratioW,ratioH)
        if self.transform is not None:
            img = self.transform(img)
        # return img, keypoints     对于这种枚举方式：for step ,(b_x,b_y) in enumerate(train_loader):
        # return {                                           
        #     'image': torch.tensor(img, dtype=torch.float),
        #     'keypoints': torch.tensor(keypoints, dtype=torch.float),
        # }   
        # 对应代码枚举方式                        
        # for i, data in tqdm(enumerate(dataloader), total=num_batches):
        #     image, keypoints = data['image'].to(DEVICE), data['keypoints'].to(DEVICE)                       
        return {
            'image': img,
            'keypoints': keypoints,
        }

    def labelCoordinateHandle(self,data,ratioW,ratioH):
        '''
            对图片的长宽进行了相应的缩放处理
        '''
        data=[float(i) for i in data]
        data[0]=data[0]*ratioW
        data[1]=data[1]*ratioH
        data[3]=data[3]*ratioW
        data[4]=data[4]*ratioH
        return np.array(data, dtype='float32')

    def __len__(self):
        return len(self.labelItem) 

 
if __name__ == '__main__':
    train_dataset =DatasetCustom(rootcsv=ROOT_CSV,imgroot=IMG_ROOT,train=True,transform=transforms.ToTensor(),ratio=0.7)
    test_dataset = DatasetCustom(rootcsv=ROOT_CSV,imgroot=IMG_ROOT,train=False,transform=transforms.ToTensor(),ratio=0.7)
    
    #single record
    data= train_dataset.__getitem__(1)     #toTensor中进行了转化  img = torch.from_numpy(pic.transpose((2, 0, 1)))
    img, label = data['image'], data['keypoints']
    img = np.transpose(img.numpy(),(1,2,0))
    plt.imshow(img)
    plt.show()
    print("label",label)

    #DataLoader查看
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=False)
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print('num_of_trainData:', len(train_loader))
    print('num_of_testData:', len(test_loader))
    #显示要给batch 中图片内容
    for step ,(b_x,b_y) in enumerate(train_loader):
        #print("step:",step)
        if step < 1:
            imgs = utils.make_grid(b_x)
            print(imgs.shape)
            imgs = np.transpose(imgs,(1,2,0))
            print(imgs.shape)
            plt.imshow(imgs)
            plt.show()
            break
```

#### 1.2. 数据分割获取

```
Dataset = CharDataset(rootdir)  # 自定义的dataset 类
l=Dataset.__len__()
test_percent=5
torch.manual_seed(1)
indices = torch.randperm(len(Dataset)).tolist()
dataset = torch.utils.data.Subset(Dataset, indices[:-int(np.ceil(l*test_percent/100))])
dataset_test = torch.utils.data.Subset(Dataset, indices[int(-np.ceil(l*test_percent/100)):])
# define training and validation data loaders
import utils
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, 
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=(1), shuffle=False, 
    collate_fn=utils.collate_fn)
for batch_i, data in enumerate(data_loader):
    images = data['image']
    key_pts = data['keypoints']
```

#### 1.3. 视频图像数据

```python
import cv2
video = cv2.VideoCapture(mp4_path)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video.release()
```

#### 1.4. ImageFolder等类

```python
import torchvision.datasets as dset
dataset = dset.ImageFolder('./data/dogcat_2') #没有transform，先看看取得的原始图像数据
print(dataset.classes)  #根据分的文件夹的名字来确定的类别
print(dataset.class_to_idx) #按顺序为这些类别定义索引为0,1...
print(dataset.imgs) #返回从所有文件夹中得到的图片的路径以及其类别


#获取图片
datalength=min(len(os.listdir(os.path.join(imageFolder,'protectivesuit'))),len(os.listdir(os.path.join(imageFolder,'whitecoat'))))
print("数据划分:",[int(datalength*0.7), int(datalength*0.2), int(datalength*0.1)])
all_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
# 使用random_split实现数据集的划分，lengths是一个list，按照对应的数量返回数据个数。
# 这儿需要注意的是，lengths的数据量总和等于all_dataset中的数据个数，这儿不是按比例划分的
train, test, valid = torch.utils.data.random_split(dataset= all_dataset, lengths=[int(datalength*0.7), int(datalength*0.2), int(datalength*0.1)])
# 接着按照正常方式使用DataLoader读取数据，返回的是DataLoader对象
train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
test  = DataLoader(test,  batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
valid = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
print(train.classes)  #根据分的文件夹的名字来确定的类别
print(train.class_to_idx) #按顺序为这些类别定义索引为0,1...
print(train.imgs) #返回从所有文件夹中得到的图片的路径以及其类别
```

#### 1.5. OneHot 编码

```python
# pytorch的标记默认从0开始
tensor = torch.tensor([0, 2, 1, 3])
N = tensor.size(0)
num_classes = 4
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
```

### 2. 训练基本框架

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)  #这里以及进行了平均处理
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

```python
for t in epoch(80):
    for images, labels in tqdm.tqdm(train_loader, desc='Epoch %3d' % (t + 1)):
        images, labels = images.cuda(), labels.cuda()
        scores = model(images)
        loss = loss_function(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#计算 softmax 输出准确率
score = model(images)
prediction = torch.argmax(score, dim=1)   # 按行 返回每行最大值在的该行索引， 如果没有dim 则按照一维数组计算
num_correct = torch.sum(prediction == labels).item()
accuruacy = num_correct / labels.size(0)
```

- Label One-hot编码时

```python
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes.
    smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=0.9)

    score = model(images)
    log_prob = torch.nn.functional.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3. 模型保存与加载

> 注意，torch.load函数要确定存储的位置：map_location='cpu'
>
> torch.sava有俩种方式：
>
> - `保存权重和模型，但是文件结果不能改变，否则报错`；
> - `保存权重，加载时，先初始化类，然后加载权重信息。`

```python
# 保存整个网络
torch.save(net, PATH) 
# 保存网络中的参数, 速度快，占空间少
torch.save(net.state_dict(),PATH)
#--------------------------------------------------
#针对上面一般的保存方法，加载的方法分别是：
model_dict=torch.load(PATH)
model_dict=model.load_state_dict(torch.load(PATH))
mlp_mixer.load_state_dict(torch.load(Config.MLPMIXER_WEIGHT,map_location='cpu'))

#save model
def save_models(tempmodel,save_path):
    torch.save("./model/"+tempmodel.state_dict(), save_path)
    print("Checkpoint saved")
# load model
model=Net()  #模型的结构
model.load_state_dict(torch.load(Path("./model/95.model")))
model.eval()  #运行推理之前，必须先调用以将退出和批处理规范化层设置为评估模式。不这样做将产生不一致的推断结果。

#断点保存
# Save checkpoint.
is_best = current_acc > best_acc
best_acc = max(best_acc, current_acc)
checkpoint = {
    'best_acc': best_acc,    
    'epoch': t + 1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
model_path = os.path.join('model', 'checkpoint.pth.tar')
torch.save(checkpoint, model_path)
if is_best:
    shutil.copy('checkpoint.pth.tar', model_path)
 
# Load checkpoint.
if resume:
    model_path = os.path.join('model', 'checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Load checkpoint at epoch %d.' % start_epoch)
```

### 4. 计算准确率，查准率，查全率

```python
# data['label'] and data['prediction'] are groundtruth label and prediction 
# for each image, respectively.
accuracy = np.mean(data['label'] == data['prediction']) * 100
 
# Compute recision and recall for each class.
for c in range(len(num_classes)):
    tp = np.dot((data['label'] == c).astype(int),
                (data['prediction'] == c).astype(int))
    tp_fp = np.sum(data['prediction'] == c)
    tp_fn = np.sum(data['label'] == c)
    precision = tp / tp_fp * 100
    recall = tp / tp_fn * 100
    
# data['label'] and data['prediction'] are groundtruth label and prediction 
# for each image, respectively.
accuracy = np.mean(data['label'] == data['prediction']) * 100
 
# Compute recision and recall for each class.
for c in range(len(num_classes)):
    tp = np.dot((data['label'] == c).astype(int),
                (data['prediction'] == c).astype(int))
    tp_fp = np.sum(data['prediction'] == c)
    tp_fn = np.sum(data['label'] == c)
    precision = tp / tp_fp * 100
    recall = tp / tp_fn * 100
```

> 建议有参数的层和汇合（pooling）层使用torch.nn模块定义，激活函数直接使用torch.nn.functional。torch.nn模块和torch.nn.functional的区别在于，torch.nn模块在计算时底层调用了torch.nn.functional，但torch.nn模块包括该层参数，还可以应对训练和测试两种网络状态。model(x)前用model.train()和model.eval()切换网络状态。loss.backward()前用optimizer.zero_grad()清除累积梯度。optimizer.zero_grad()和model.zero_grad()效果一样。

### 5. 可视化部分

> 有 Facebook 自己开发的 Visdom 和 Tensorboard 两个选择。
> https://github.com/facebookresearch/visdom
> https://github.com/lanpa/tensorboardX

```python
# Example using Visdom.
vis = visdom.Visdom(env='Learning curve', use_incoming_socket=False)
assert self._visdom.check_connection()
self._visdom.close()
options = collections.namedtuple('Options', ['loss', 'acc', 'lr'])(
    loss={'xlabel': 'Epoch', 'ylabel': 'Loss', 'showlegend': True},
    acc={'xlabel': 'Epoch', 'ylabel': 'Accuracy', 'showlegend': True},
    lr={'xlabel': 'Epoch', 'ylabel': 'Learning rate', 'showlegend': True})

for t in epoch(80):
    tran(...)
    val(...)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([train_loss]),
             name='train', win='Loss', update='append', opts=options.loss)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([val_loss]),
             name='val', win='Loss', update='append', opts=options.loss)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([train_acc]),
             name='train', win='Accuracy', update='append', opts=options.acc)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([val_acc]),
             name='val', win='Accuracy', update='append', opts=options.acc)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([lr]),
             win='Learning rate', update='append', opts=options.lr)

```

- pytorch   graphviz

> pip install torchviz

```python
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
```

![](https://user-images.githubusercontent.com/13428986/110844921-ff3f7500-8277-11eb-912e-3ba03623fdf5.png)

- 显示图片中的关键点

```python
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pytorchsegmentcode/  

