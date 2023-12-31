# FlowerClassifyPytorch


> The images above were from the **Kaggle’s** dataset “[Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)” by Alexander. The title of each image consists its class name and index number in the dataset. This dataset contains **4242** images of flowers. The pictures are divided into five classes: **daisy, tulip, rose, sunflower and dandelion**. For each class there are about 800 photos. Photos are not in high resolution, 320x240 pixels. 

### 0. Python api

- **[argmax（）](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)**

```python
a = np.arange(6).reshape(2,3) + 10
>>> a
array([[10, 11, 12],
       [13, 14, 15]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201010104858838.png)



- **[Torch.max()](https://pytorch.org/docs/stable/generated/torch.max.html), 注意这里返回俩个值**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201010105908365.png)

- **Torch.sum()**   0: 按列sum； 1: 按行sum；

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201010110825280.png)

- **BATCHNORM2D**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201021110219308.png)

### 1. model.py

```python
import torch
from torch import nn
import torchvision
import torchvision.models as models
from util import *
import numpy as np
from torch.autograd import Variable
class BuildVGG16Net(nn.Module):
    def __init__(self, model_type, n_output):
        super(BuildVGG16Net, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            path='/home/iot/jupyter/root_dir/liudongdong/pytorch_demo/pretainedpth/vgg16'
            model=load_model('vgg16', path)
            self.features = model.features

            fc1 = nn.Linear(9216, 4096)
            fc1.bias = model.classifier[0].bias
            fc1.weight = model.classifier[0].weight
            
            fc2 = nn.Linear(4096, 4096)
            fc2.bias = model.classifier[3].bias
            fc2.weight = model.classifier[3].weight
            
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    fc1,
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    fc2,
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, n_output))  
        # model.classifier[6]=nn.Linear(4096,n_output)
        # self.classifier = model.classifier
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

if __name__ == "__main__":
    model_type = 'pre'
    n_output = 2
    vggnet = BuildVGG16Net(model_type, n_output)
    print(vggnet)
    
    x = np.random.rand(50,3,224,224)
    x = x.astype(np.float32)
    x_ts = torch.from_numpy(x)
    x_in = Variable(x_ts)
    y = vggnet(x_in)
    print(y)
```

```python
from multiprocessing import freeze_support
import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam

from model import BuildVGG16Net
import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path
import os

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

#代码一些参数设置
num_classes = 2    
num_epochs=6
batch_size = 10
num_of_workers = 5
imageFolder='/home/iot/jupyter/root_dir/liudongdong/data/depth'


DATA_PATH_TRAIN = Path(imageFolder)
trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
# print(len(os.listdir(os.path.join(imageFolder,'protectivesuit'))),len(os.listdir(os.path.join(imageFolder,'whitecoat'))))
# #获取图片  
# datalength=min(len(os.listdir(os.path.join(imageFolder,'protectivesuit'))),len(os.listdir(os.path.join(imageFolder,'whitecoat'))))
#print("数据划分:",datalength,[int(datalength*0.7), int(datalength*0.2), int(datalength*0.1)])
#14366
#(sample, target) where target is class_index of the target class.
all_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
# print(all_dataset.__len__,type(all_dataset))
# 使用random_split实现数据集的划分，lengths是一个list，按照对应的数量返回数据个数。
# 这儿需要注意的是，lengths的数据量总和等于all_dataset中的数据个数，这儿不是按比例划分的  28732
#Randomly split a dataset into non-overlapping new datasets of given lengths
train, test, valid = torch.utils.data.random_split(dataset= all_dataset, lengths=[26000,2000,732])

# 接着按照正常方式使用DataLoader读取数据，返回的是DataLoader对象
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()
# Create model, optimizer and loss function
model = BuildVGG16Net('pre', num_classes)

dummy_input = torch.rand(batch_size,3,224,224)
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model, (dummy_input,))

# if cuda is available, move the model to the GPU
if cuda_avail:
    model.cuda()

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

def train(num_epoch):
    best_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        counter=0
        for i, (images, labels) in enumerate(train_loader):
            counter=counter+1
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            #print("loss:",loss)  # float tensor(0.6957, device='cuda:0', grad_fn=<NllLossBackward>)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
			##对比后相同的值会为1，不同则会为0;# 更新正确分类的图片的数量
            train_acc += torch.sum(prediction == labels.data).float()
            if counter%50==0:
                writer.add_scalar('Train_batch_acc', train_acc/counter / batch_size, counter)
                writer.add_scalar('Train_batch_loss', train_loss/counter, counter)
                print(f"batch {counter}, Train Accuracy: {train_acc/counter / batch_size} , TrainLoss: {train_loss/counter} ")

        # Call the learning rate adjustment function
        #adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / counter / batch_size
        train_loss = train_loss / counter

        writer.add_scalar('Train_acc', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)

        # Evaluate on the test set
        test_acc = test()
        writer.add_scalar('Test_acc', test_acc, epoch)
        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Test Accuracy: {test_acc}")

def test():
    model.eval()
    test_acc = 0.0
    counter=0
    for i, (images, labels) in enumerate(test_loader):
        counter=counter+1
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)

        test_acc += torch.sum(prediction == labels.data).float()
    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / counter / batch_size

    return test_acc

def save_models(epoch):
    torch.save(model.state_dict(), f"{epoch}.model")
    print("Checkpoint saved")

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img[0].numpy(), (1, 2, 0)))
    plt.show()

def main():
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # show images
    imshow(images)


if __name__ == '__main__':
    freeze_support()
    train(num_epochs)
```

```python
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from model import BuildVGG16Net
from pathlib import Path
from torch.autograd import Variable
from torchvision import datasets
num_classes=2
lable={0:'防护服',1:'白大褂'}

model = BuildVGG16Net('pre', num_classes)
checkpoint = torch.load(Path('0.model'))
model.load_state_dict(checkpoint)
cuda_avail = torch.cuda.is_available()
if cuda_avail:
    model.cuda()
model.eval()

trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
#这里是需要进行预测的图片
def testSingleImage(picturePath):
    image = Image.open(Path(picturePath)).convert('RGB')
    input_image = trans(image)
    #如果不使用reshape操作，可以使用unsqueeze操作
    #batch_img=torch.unsqueeze(input_image,0)
    input_image=input_image.reshape((1,3,224,224))
    print(type(input_image),input_image.shape)
    if cuda_avail:
        images = Variable(input_image.cuda())
        output = model(images)
        prediction = output.cpu().data.numpy().argmax()
        print(lable[prediction])

# picturepath='/home/iot/jupyter/root_dir/liudongdong/data/realface/protectivesuit/2depth.png'
# testSingleImage(picturepath)
batch_size=20
DATA_PATH_TRAIN='/home/iot/jupyter/root_dir/liudongdong/data/realface/'
all_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
test_loader  = DataLoader(all_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

test_acc = 0.0
counter=0
for i, (images, labels) in enumerate(test_loader):
    counter=counter+1
    if cuda_avail:
        images = Variable(images.cuda())
        labels=Variable(labels.cuda())
    # Predict classes using images from the test set
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)
    print(prediction)
    test_acc += torch.sum(prediction == labels.data).float()
# Compute the average acc and loss over all 10000 test images
test_acc = test_acc / counter /batch_size
print(test_acc)
```

- util.py

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


def removefile(folder):
    #14366
    filelist=os.listdir(folder)
    for i in range(14366,len(filelist)):
        if os.path.exists(os.path.join(folder,filelist[i])):
            os.remove(os.path.join(folder,filelist[i]))
    print("删除后文件长度为:",len(os.listdir(folder)))


import imghdr
def checkFile(folder):
    for file in os.listdir(folder):
        filepath=os.path.join(folder,file)
        img = imghdr.what(filepath)
        if img==None:
            os.remove(filepath)
            print("delete:",filepath)
 


if __name__ == "__main__":
    #main()
    removefile('/home/iot/jupyter/root_dir/liudongdong/data/depth/protectivesuit')
    checkFile('/home/iot/jupyter/root_dir/liudongdong/data/realface/protectivesuit')
    checkFile('/home/iot/jupyter/root_dir/liudongdong/data/realface/whitecoat')
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/flowerclassifypytorch/  

