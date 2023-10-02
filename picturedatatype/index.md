# PictureDataType


### 1. 图像读取与存储

#### 1.1. PIL

```python
from PIL import Image
import numpy as np
def imageloadDirectlyResize(path,resize):
    '''
        直接通过PIL image 加载图像，resize操作
    '''
    img = Image.open(path)
    img=img.resize((196,196))
    img.show()
    #print(type(img))  #<class 'PIL.Image.Image'>
    #img.save("1.jpg")
    return np.asarray(img)   #196*196*3
```

#### 1.2. skimage

```python
from skimage import io
img=io.imread('img.jpg')
io.imshow(img)
io.show()
io.imsave('img_copy.jpg', img)
```

#### 1.3. matplotlib

```python
import matplotlib.pyplot as plt
img = plt.imread('img.jpg')
print(type(img))  # <class 'numpy.ndarray'>
print(img.shape)  # (531, 742, 3)
plt.imshow(img)
plt.show()
plt.imsave('img_copy.jpg', img，cmap=plt.cm.gray)
#从数据数据显示
data=handle.img2arr("./temp/"+file)
#data=handle.addGlitch(data)
#plt.rcParams['figure.figsize'] = (0.96, 0.96)
plt.clf()
plt.axis('off')
#cv2.imwrite("2.png",data)
#plt.figure("head")
plt.imshow(data,cmap=plt.cm.gray)
plt.savefig("./temp/1"+file+".png")
```

#### 1.4. opencv

```python
def imageloadCV(path,resize):
    img = cv2.imread(path)
    orig_w, orig_h,_ = img.shape
    img=cv2.resize(img,(resize,resize),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #print(img_gray.shape)  #(196, 196, 3)   转化灰度图之后(196*196)
    #print(type(img))  # <class 'numpy.ndarray'>
    cv2.imwrite("imageloadCV.jpg",img)
    return resize/orig_w,resize/orig_h,img_gray
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210516154604079.png)

#### 1.5. numpy

```python
import cv2
img = cv2.imread('img.jpg')
print(type(img))  # <class 'numpy.ndarray'>
print(img.shape)  # (531, 742, 3)
cv2.imshow('image',img)
cv2.imwrite('img_copy.jpg', img)
```

### 2. 图像格式

#### 2.1. 二值图像

> 只有两个值，0和1，0代表黑，1代表白，或者说0表示背景，而1表示前景。 [1,1,1,1]表示白色 [0,0,0,0]表示黑色，$\# 96*96*4$, 或者$\# 96*96*1$ 注意数据格式。

#### 2.2. 灰度图像

> 灰度图像（gray image）是每个像素只有一个采样颜色的图像，这类图像通常显示为从最暗黑色到最亮的白色的灰度，尽管理论上这个采样可以任何颜色的不同深浅，甚至可以是不同亮度上的不同颜色。灰度图像与黑白图像不同，在计算机图像领域中黑白图像只有黑色与白色两种颜色；但是，灰度图像在黑色与白色之间还有许多级的颜色深度。灰度图像经常是在单个电磁波频谱如可见光内测量每个像素的亮度得到的，用于显示的灰度图像通常用每个采样像素8位的非线性尺度来保存，这样可以有256级灰度（如果用16位，则有65536级）。 

- 识别物体，最关键的因素是梯度（现在很多的特征提取，SIFT,HOG等等本质都是梯度的统计信息），梯度意味着边缘，这是最本质的部分，而计算梯度，自然就用到灰度图像了。颜色本身，非常容易受到光照等因素的影响，同类的物体颜色有很多变化。所以颜色本身难以提供关键信息。
- 灰度化之后矩阵维数下降，运算速度大幅度提高，并且梯度信息仍然保留。

#### 2.3. 彩色图像

> 彩色图像，每个像素通常是由红（R）、绿（G）、蓝（B）三个分量来表示的，分量介于（0，255）。

```python
# -*- coding: utf-8 -*-
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
'''
彩色图像的灰度化、二值化
'''
img = io.imread("1.jpg")  # (1080, 1920, 3)
io.imshow(img)
io.show()
# 灰度化
img_gray = rgb2gray(img)  # (1080, 1920)
io.imshow(img_gray)
io.show()

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
 
img_binary = np.where(img_gray >= 0.5, 1, 0)  # (1080, 1920)
print(img_binary)
io.imshow(img_binary,cmap=plt.cm.gray)
io.show()
```

**[cmap](https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html)**

> **gray**：0-255 级灰度，0：黑色，1：白色，黑底白字；
>
> **gray_r**：翻转 gray 的显示，如果 gray 将图像显示为黑底白字，gray_r 会将其显示为白底黑字；

### 3. 像素访问

> row == heigh == Point.y
>
> col == width == Point.x
>
> Mat::at(Point(x, y)) == Mat::at(y,x)
>
> 注意： 一般是以左上方为原点，也可能是左下方

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027093952408.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027132339343.png)

### 4. 颜色空间

> 颜色是一种连续的现象，它意味着有无数种颜色。但是，人类的眼睛和感知能力是有限的。所以，为了识别这些颜色，我们需要一种媒介或这些颜色的表示，这种颜色的表示被称为色彩空间。在技术术语中，一个颜色模型或颜色空间是一个特定的3-D坐标系统以及该系统中的一个子空间，其中每一种颜色都由一个单点表示。

- RGB(Red Green Blue)
- HSL(Hue Saturation Lightness)
- HSV(Hue Saturation Value)
- YUV(Luminance, blue–luminance, red–luminance)
- CMYK(Cyan, Magenta, Yellow, Key)

#### 4.1.  RGB

> **RGB**将颜色描述为由三个部分组成的元组。每个部分都可以取0到255之间的值，其中元组`(0,0,0)表示黑色`，元组`(255,255,255)表示白色`。元组的第0、第1和第2个部分分别表示红、绿、蓝的分量。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213103931034.png)

#### 4.2.  HSL

> HSL的一般含义是色调、饱和度和明度。你可以将HSL以圆柱体的形式可视化，如图2(a)所示。围绕圆柱体的是不同的颜色，比如绿色、黄色、红色等等(我们真正想要的颜色)。饱和度是指颜色的多少，而明度是指颜色有多暗或多亮。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213104224305.png)

#### 4.3. HSV

> HSV这个名字来自于颜色模型的三个坐标，即`色相、饱和度和值`。它也是一个圆柱形的颜色模型，圆柱体的半径表示饱和度，垂直轴表示值，角度表示色调。对于观察者，`色调是占主导地位的`，饱和度是混合到色调中的白光的数量，value是chrome的强度，value较低颜色变得更加类似于黑色，value越高，颜色变得更加像颜色本身。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213104334553.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/picturedatatype/  

