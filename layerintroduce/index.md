# layerIntroduce


### 1. 卷积层

> 卷积运算：卷积核在输入图像上滑动，`相应位置上进行相加`。卷积过程`类似于用一个模板去图像上寻找与他相似的区域`，与卷积核模式越相似，激活值越高，从而实现特征提取。
>
> 卷积核：可以认为是`某种模式或某种特征`
>
> 卷积维度：一般情况下，卷积核在几个维度上滑动就是几维卷积

### 2. 转至反卷积

> 功能：用于`对图像进行上采样`，`物体检测任务经常用到`（不可逆过程，转置卷积得到的图像与原图不相等）

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210516232024368.png)

```python
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from tools.common_tools import transform_invert, set_seed
 
set_seed(1)  # 设置随机种子
 
# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255
 
# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W
 
# ================================= create convolution layer ==================================
# ================ transposed
flag = 1
# flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)
 
    # calculation
    img_conv = conv_layer(img_tensor)
 
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210516232535321.png)

### 3. 池化层

> 池化运算：对输入信号（图像）进行“收集”（多变少）并“总结”（max，mean），类似水池收集水资源。

### 4. 反最大值池化

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210516232719790.png)

### 5. 全连接层

> 每一个神经元与上一层所有神经元相连，实现对上一层的线性组合。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210516232815261.png)

### 6. 激活函数![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210516232910664.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/layerintroduce/  

