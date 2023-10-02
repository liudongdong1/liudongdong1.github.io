# PictureDataEnhance


> 数据增强中的一些基本操作，例如裁剪图片大小，图片正则话标准化处理，图片数据转tensor向量，图片随机裁剪，旋转，过滤，图片锐化，以及图片模糊等。

### 1. Resize

```python
class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, X, Y):
        _X = cv2.resize(X, self.output_size)
        w, h = self.output_size
        c = Y.shape[-1]
        _Y = np.zeros((h, w, c))
        for i in range(Y.shape[-1]):
            _Y[..., i] = cv2.resize(Y[..., i], self.output_size)
        return _X, _Y
```

### 2. Clip

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200814102802185.png)

```python
class Clip(object):
    def __init__(self, mini, maxi=None):
        if maxi is None:
            self.mini, self.maxi = 0, mini
        else:
            self.mini, self.maxi = mini, maxi

    def __call__(self, X, Y):
        mini_mask = np.where(X < self.mini)
        maxi_mask = np.where(X > self.maxi)
        X[mini_mask] = self.mini
        X[maxi_mask] = self.maxi
        return X, Y
```

### 3. Normalize or Standardize

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200814102952673.png)

```python
class Normalize(object):
    def __init__(self, axis=None):
        self.axis = axis
    def __call__(self, X, Y):
        mini = np.min(X, self.axis)
        maxi = np.max(X, self.axis)
        X = (X - mini) / (maxi - mini)
        return X, Y
class Standardize(object):
    def __init__(self, axis=None):
        self.axis = axis
    def __call__(self, X, Y):
        mean =  np.mean(X, self.axis)
        std = np.std(X, self.axis)
        X = (X - mean) / std
        return X, Y
```

### 4. ToTensor

如果您使用的是`Pytorch`，则需要将图像转换为`Torch.Tensor`。 唯一需要注意的是，使用`Pytorch`，我们的图像维度中**首先是通道**，而不是最后是通道。 最后，我们还可以选择张量的**输出类型**。

```python
class ToTensor(object):
    def __init__(self, X_type=None, Y_type=None):
        # must bu torch types
        self.X_type = X_type
        self.Y_type = Y_type
    def __call__(self, X, Y):
        # swap color axis because
        # numpy img_shape: H x W x C
        # torch img_shape: C X H X W
        X = X.transpose((2, 0, 1))
        Y = Y.transpose((2, 0, 1))
        # convert to tensor
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        if self.X_type is not None:
            X = X.type(self.X_type)
        if self.Y_type is not None:
            Y = Y.type(self.Y_type)
        return X, Y
```

### 5. Flip

```python
class Flip(object):
    def __call__(self, X, Y):
        for axis in [0, 1]:
            if np.random.rand(1) < 0.5:
                X = np.flip(X, axis)
                Y = np.flip(Y, axis)
        return X, Y
```

### 6. Random crop

```python
class Crop(object):
    def __init__(self, min_size_ratio, max_size_ratio=(1, 1)):
        self.min_size_ratio = np.array(list(min_size_ratio))
        self.max_size_ratio = np.array(list(max_size_ratio))

    def __call__(self, X, Y):
        size = np.array(X.shape[:2])
        mini = self.min_size_ratio * size
        maxi = self.max_size_ratio * size
        # random size
        h = np.random.randint(mini[0], maxi[0])
        w = np.random.randint(mini[1], maxi[1])
        # random place
        shift_h = np.random.randint(0, size[0] - h)
        shift_w = np.random.randint(0, size[1] - w)
        X = X[shift_h:shift_h+h, shift_w:shift_w+w]
        Y = Y[shift_h:shift_h+h, shift_w:shift_w+w]

        return X, Y
```

### 7. Filter

```python
class CustomFilter(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, Y):
        X = cv2.filter2D(X, -1, self.kernel)
        return X, Y
```

### 8. Sharpen

```python
class Sharpen(object):
    def __init__(self, max_center=4):
        self.identity = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]])
        self.sharpen = np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]]) / 4

    def __call__(self, X, Y):

        sharp = self.sharpen * np.random.random() * self.max_center
        kernel = self.identity + sharp

        X = cv2.filter2D(X, -1, kernel)
        return X, Y
```

### 9. Blur

> 本质上，它是一种[数据平滑技术](http://en.wikipedia.org/wiki/Smoothing)（data smoothing）；所谓"模糊"，可以理解成每一个像素都取周边像素的平均值。图像都是连续的，越靠近的点关系越密切，越远离的点关系越疏远。因此，加权平均更合理，距离越近的点权重越大，距离越远的点权重越小。权重的分配使用高斯模型；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027094643798.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027094436977.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027094542433.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027094506667.png)

```python
class GaussianBlur(object):
    def __init__(self, max_kernel=(7, 7)):
        self.max_kernel = max_kernel

    def __call__(self, X, Y):
        kernel_size = (
            np.random.randint(1, self.max_kernel[0] + 1),
            np.random.randint(1, self.max_kernel[1] + 1),
        )
        X = cv2.GaussianBlur(X, kernel_size, 0)
        return X, Y
'''
   高斯模糊处理
'''
def GaussianBlurHandle(self,imagepath):
    kernel_size=(5,5)
    sigma=1.5
    img=cv2.imread(imagepath)
    img=cv2.GaussianBlur(img,kernel_size,sigma)
    cv2.imwrite("./temp/gaosi.png",img)
```

转载：

- https://zhuanlan.zhihu.com/p/158854758
- http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.html



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/picturedataenhance/  

