# TorchVision_DataLoad


> All datasets are subclasses of [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) i.e, they have `__getitem__` and `__len__` methods implemented. Hence, they can all be passed to a [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) which can load multiple samples parallelly using `torch.multiprocessing` workers.
>
> - Dataloader： offer a way to parallelly load data, batch load, and offer shuffle policy.
> - Dataset: the dataset entry, offer __getitem__ function., Transformer function 在这里执行。
## 0. DataLoader

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

- **dataset** ([*Dataset*](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)) – dataset from which to load the data.
- **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many samples per batch to load (default: `1`).
- **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – set to `True` to have the data reshuffled at every epoch (default: `False`).
- **sampler** ([*Sampler*](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Sampler) *or* *Iterable**,* *optional*) – defines the strategy to draw samples from the dataset. Can be any `Iterable` with `__len__` implemented. If specified, `shuffle` must not be specified.
- **batch_sampler** ([*Sampler*](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Sampler) *or* *Iterable**,* *optional*) – like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.
- **num_workers** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)
- **collate_fn** (*callable**,* *optional*) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
- **pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your `collate_fn` returns a batch that is a custom type, see the example below.
- **drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – set to `True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If `False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: `False`)
- **timeout** (*numeric**,* *optional*) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: `0`)
- **worker_init_fn** (*callable**,* *optional*) – If not `None`, this will be called on each worker subprocess with the worker id (an int in `[0, num_workers - 1]`) as input, after seeding and before data loading. (default: `None`)
- **prefetch_factor** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional**,* *keyword-only arg*) – Number of samples loaded in advance by each worker. `2` means there will be a total of 2 * num_workers samples prefetched across all workers. (default: `2`)
- **persistent_workers** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True`, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: `False`)

### .1. Map-style Dataset

> implements the `__getitem__()` and `__len__()` protocols, and represents a map from (poissibly non-integral) indices/keys to data samples. dataset[idx]={image, label}

### .2. Iterable-style dataset

>  implements the `__iter__()` protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.

```python
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])


    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py
```

## 1. Available Datasets

- [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)         [QMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#qmnist)       [FakeData](https://pytorch.org/docs/stable/torchvision/datasets.html#fakedata)   [COCO](https://pytorch.org/docs/stable/torchvision/datasets.html#coco)   [Captions](https://pytorch.org/docs/stable/torchvision/datasets.html#captions)   [Detection](https://pytorch.org/docs/stable/torchvision/datasets.html#detection)[LSUN](https://pytorch.org/docs/stable/torchvision/datasets.html#lsun)   [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)   [DatasetFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder)   [ImageNet](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)[CIFAR](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)   [STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#stl10)   [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn)   [PhotoTour](https://pytorch.org/docs/stable/torchvision/datasets.html#phototour)   [SBU](https://pytorch.org/docs/stable/torchvision/datasets.html#sbu)   [Flickr](https://pytorch.org/docs/stable/torchvision/datasets.html#flickr)   [VOC](https://pytorch.org/docs/stable/torchvision/datasets.html#voc)   [Cityscapes](https://pytorch.org/docs/stable/torchvision/datasets.html#cityscapes)   [SBD](https://pytorch.org/docs/stable/torchvision/datasets.html#sbd)   [USPS](https://pytorch.org/docs/stable/torchvision/datasets.html#usps)   [Kinetics-400](https://pytorch.org/docs/stable/torchvision/datasets.html#kinetics-400)   [HMDB51](https://pytorch.org/docs/stable/torchvision/datasets.html#hmdb51)   [UCF101](https://pytorch.org/docs/stable/torchvision/datasets.html#ucf101)   [CelebA](https://pytorch.org/docs/stable/torchvision/datasets.html#celeba)   [Fashion-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)   [KMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#kmnist)   [EMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist)

> All the datasets have almost similar API. They all have two common arguments: `transform` and `target_transform` to transform the input and target respectively.
Take MNIST for example:

![image-20201017124252716](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201017124252716.png)

```python
from .vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
from .utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg
[docs]class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets
    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets
    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data
    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
        # process and save as torch files
        print('Processing...')
        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
```

## 2. Generic Dataloader

### 2.1. ImageFolder

> A generic data loader where the images are arranged in this way:
>
> torchvision.datasets.ImageFolder(*root*, *transform=None*, *target_transform=None*, *loader=<function default_loader>*, *is_valid_file=None*)
![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201017124740466.png)

`__getitem__`(*index*)

- Parameters

  **index** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Index

- Returns

  (sample, target) where target is class_index of the target class.

- Return type

  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

```python
def load_data(root_path, dir, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[phase])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    return data_loader 
```

### 2.2. DatasetFolder

> `torchvision.datasets.``DatasetFolder`(*root*, *loader*, *extensions=None*, *transform=None*, *target_transform=None*, *is_valid_file=None*)
![image-20201017125057711](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201017125057711.png)

### 3. Examples

```python
from multiprocessing import freeze_support
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# Hyperparameters.
num_epochs = 20
num_classes = 5
batch_size = 100
learning_rate = 0.001
num_of_workers = 5
DATA_PATH_TRAIN = Path('C:/Users/Aeryes/PycharmProjects/simplecnn/images/train/')
DATA_PATH_TEST = Path('C:/Users/Aeryes/PycharmProjects/simplecnn/images/test/')
MODEL_STORE_PATH = Path('C:/Users/Aeryes/PycharmProjects/simplecnn/model')
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
# Flowers dataset.
train_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
test_dataset = datasets.ImageFolder(root=DATA_PATH_TEST, transform=trans)
# Create custom random sampler class to iter over dataloader.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_of_workers)
for i, (images, labels) in enumerate(train_loader):   
    # Move images and labels to gpu if available
    if cuda_avail:
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())
```

## Learning Resources

- https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
- https://pytorch.org/docs/stable/torchvision/index.html



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/torchvision_dataload/  

