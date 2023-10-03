# TorchVision_Transforms


> Offering transformation pipeline;

```python
 transforms.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),
 ])
```

### 1. PIL Image Op

```python
CenterCrop(size) # Crops the given PIL Image at the center.
ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)#Randomly change the brightness, contrast and saturation of an image.
FiveCrop(size) #Crop the given PIL Image into four corners and the central crop
Grayscale(num_output_channels=1) # convert image to grayscale
RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)#Random affine transformation of the image keeping center invariant
RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')#Crop the given PIL Image at a random location.
RandomGrayscale(p=0.1)#Randomly convert image to grayscale with a probability of p (default 0.1).
RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)#Rotate the image by angle.
Resize(size, interpolation=2)#Resize the input PIL Image to the given size.
```

### 2. Torch.*Tensor Op

```python
LinearTransformation(transformation_matrix, mean_vector)#
Normalize(mean, std, inplace=False)#Normalize a tensor image with mean and standard deviation. Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, this transform will normalize each channel of the input torch.*Tensor i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]

ToPILImage(mode=None)#Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
ToTensor #Convert a PIL Image or numpy.ndarray to tensor.
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201017150852446.png)

### 3. TORCHVISION.UTILS

#### 3.1. make_grid()   

```python
torchvision.utils.make_grid(tensor: Union[torch.Tensor, List[torch.Tensor]], nrow: int = 8, padding: int = 2, normalize: bool = False, range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0) → torch.Tensor
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201017151754444.png)

```python
torchvision.utils.save_image(tensor: Union[torch.Tensor, List[torch.Tensor]], fp: Union[str, pathlib.Path, BinaryIO], nrow: int = 8, padding: int = 2, normalize: bool = False, range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0, format: Optional[str] = None) → None
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/torchvision_transforms/  

