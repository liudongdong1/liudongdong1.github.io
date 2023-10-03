# ROSRelative


#### Caffe Introduce

- a deep learning framework that provides an easy and straightforward way to experiment with deep learning and leverage community contributions of new models and algorithms.
- Caffe2 is built to excel at utilizing both multiple GPUs on a single-host and multiple hosts with GPUs. PyTorch is great for research, experimentation and trying out exotic neural networks, while Caffe2 is headed towards supporting more industrial-strength applications with a heavy focus on mobile. 
- easy to converting from Pytorch
- <font color=red>easy, built-in distributed training. This means that you can very quickly scale up or down without refactoring your design.</font>

Caffe2 improves Caffe 1.0 in a series of directions:

- first-class support for large-scale distributed training
- mobile deployment
- new hardware support (in addition to CPU and CUDA)
- flexibility for future directions such as quantized computation
- stress tested by the vast scale of Facebook applications

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191208094933945.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191208095158661.png)

#### Distribute Training

- [Gloo](https://github.com/facebookincubator/gloo): Caffe2 leverages, Gloo, a communications library for multi-machine training.
- [NCCL](https://github.com/nvidia/nccl): Caffe2 also utilize’s NVIDIA’s NCCL for multi-GPU communications.
- [Redis](https://redis.io/) To facilitate management of nodes in distributed training, Caffe2 can use a simple NFS share between nodes, or you can provide a Redis server to handle the nodes’ communications.   <font color=red>实践的时候在细看</font>

#### DataSets:

| NAME                                                         | TYPE                                                         | DOWNLOAD                                                     |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [AlexNet-Places205](http://places.csail.mit.edu/index.html)  | ![images > places recognition](https://caffe2.ai/static/images/boathouse.png) | [![download](https://caffe2.ai/static/images/download.png)](http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz) |
| [AN4](http://www.speech.cs.cmu.edu/databases/an4/): 948 training and 130 test utterances | ![speech](https://caffe2.ai/static/images/landing-audio.png) | [![download](https://caffe2.ai/static/images/download.png)](http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz) |
| [BSDS (300/500)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/): 12k labeled segmentations | ![image segmentation](https://caffe2.ai/static/images/wolf.jpg) | [![download](https://caffe2.ai/static/images/download.png) images](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz) [![download](https://caffe2.ai/static/images/download.png) segmentations](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-human.tgz) |
| [Celeb-A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): 200k+ celebrity images, 10k+ identities | ![celebrity images](https://caffe2.ai/static/images/celebrity.png) | [![download](https://caffe2.ai/static/images/download.png)](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0) |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60k tiny (32x32) tagged images | ![tiny images](https://caffe2.ai/static/images/cifar-tiny.png) | [![download](https://caffe2.ai/static/images/download.png)](https://www.cs.toronto.edu/~kriz/cifar.html) |
| [COCO](http://mscoco.org/dataset/): A large image dataset designed for object detection, segmentation, and caption generation. | ![coco](https://caffe2.ai/static/images/coco.png)            | [![download](https://caffe2.ai/static/images/download.png)](http://mscoco.org/dataset/#download) |
| [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html): 136k+ car images & 1716 car models | ![cars](https://caffe2.ai/static/images/cars.png)            | [![download](https://caffe2.ai/static/images/download.png)](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) |
| [Oxford 102 Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html): 102 flower categories | ![flowers](https://caffe2.ai/static/images/flowers.png)      | [![download](https://caffe2.ai/static/images/download.png) images](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) [![download](https://caffe2.ai/static/images/download.png) segmentations](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz) |
| [ImageNet](http://image-net.org/): 14,197,122 images, 21841 synsets indexed | ![large range of images](https://caffe2.ai/static/images/imagenet.jpg) | [![download](https://caffe2.ai/static/images/download.png)](http://image-net.org/download) |
| [ImageNet ILSVRC](http://www.image-net.org/challenges/LSVRC/): Competition datasets | ![large range of images](https://caffe2.ai/static/images/imagenet.jpg) | [![download](https://caffe2.ai/static/images/download.png)](http://www.image-net.org/challenges/LSVRC/) |
| [Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set)   | ![flowers > classification](https://caffe2.ai/static/images/iris.jpg) | [![download](https://caffe2.ai/static/images/download.png)](https://en.wikipedia.org/wiki/Iris_flower_data_set) |
| [LSUN Scenes](http://lsun.cs.princeton.edu/2016/) millions of indoor/outdoor building scenes | ![scene classification](https://caffe2.ai/static/images/kitchen.jpg) | [![download](https://caffe2.ai/static/images/download.png)](https://github.com/fyu/lsun/blob/master/download.py) |
| [LSUN Room Layout](http://lsun.cs.princeton.edu/2016/) 4000 indoor scenes | ![scene classification](https://caffe2.ai/static/images/layout.png) | [![download](https://caffe2.ai/static/images/download.png)](https://github.com/fyu/lsun/blob/master/download.py) |
| [MNIST](http://yann.lecun.com/exdb/mnist/) 60k handwriting training set, 10k test images | ![handwriting](https://caffe2.ai/static/images/mnist.png)    | [![download](https://caffe2.ai/static/images/download.png)](http://yann.lecun.com/exdb/mnist/) |
| [Multi-Salient-Object (MSO)](https://cs-people.bu.edu/jmzhang/sos.html) 1224 tagged salient object images | ![tagged objects](https://caffe2.ai/static/images/mso.png)   | [![download](https://caffe2.ai/static/images/download.png)](https://www.cs.bu.edu/groups/ivc/data/SOS/MSO.zip) |
| [OUI-Adience Face Image](http://www.openu.ac.il/home/hassner/Adience/data.html#agegender) 26,580 age & gender labeled images | ![age, gender](https://caffe2.ai/static/images/age.png)      | [![download](https://caffe2.ai/static/images/download.png)](http://www.openu.ac.il/home/hassner/Adience/data.html#agegender) |
| [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) 11,530 images w/ 27,450 ROI annotated objects and 6,929 segmentations | ![images > object recognition](https://caffe2.ai/static/images/voc.png) | [![download](https://caffe2.ai/static/images/download.png)](http://host.robots.ox.ac.uk/pascal/VOC/) |
| [PCAP](http://www.netresec.com/?page=PcapFiles) Network captures of regular internet traffic and attack scenario traffic | ![network capture](https://caffe2.ai/static/images/defcon.jpg) | [![download](https://caffe2.ai/static/images/download.png)](http://www.netresec.com/?page=PcapFiles) |
| [Penn Treebank (PTB)](http://www.fit.vutbr.cz/~imikolov/rnnlm/) statistical language modeling | ![language](https://caffe2.ai/static/images/landing-audio.png) | [![download](https://caffe2.ai/static/images/download.png)](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz) |
| [UCF11/YouTube Action](http://crcv.ucf.edu/data/UCF_YouTube_Action.php) 11 action categories: basketball shooting, biking/cycling, diving, golf swinging, horse back riding, soccer juggling, swinging, tennis swinging, trampoline jumping, volleyball spiking, and walking with a dog | ![video > action](https://caffe2.ai/static/images/action.jpg) | [![download](https://caffe2.ai/static/images/download.png)](http://crcv.ucf.edu/data/UCF_YouTube_Action.php) |
| [UCI Datasets](https://archive.ics.uci.edu/ml/datasets.html) | ![variety](https://caffe2.ai/static/images/caffe2variety.png) | [![download](https://caffe2.ai/static/images/download.png)](https://archive.ics.uci.edu/ml/datasets.html) |
| [US Census](https://catalog.data.gov/dataset): demographic data | ![line graph](https://caffe2.ai/static/images/linegraph.png) | [![download](https://caffe2.ai/static/images/download.png)](https://catalog.data.gov/dataset) |
| [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) millions of faces | ![faces](https://caffe2.ai/static/images/faces.jpg)          | [![download](https://caffe2.ai/static/images/download.png)](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz) |
| [LibriSpeech](https://www.openslr.org/12/) 1000 hours free speech recognition traning dataset | ![language](https://caffe2.ai/static/images/landing-audio.png) | [![download](https://caffe2.ai/static/images/download.png)](https://www.openslr.org/12/) |

#### Downloading and Importing Caffe2 Models

- simpe predictions:  
  1. a protobuf that defines the network
  2. a protobuf that has all of the network weights

```python
with open(path_to_INIT_NET) as f:
    init_net = f.read()
with open(path_to_PREDICT_NET) as f:
    predict_net = f.read()
p = workspace.Predictor(init_net, predict_net)

#downlaod a Model as a module
python -m caffe2.python.models.download --install squeezenet 
#error try;sudo PYTHONPATH=/usr/local python -m caffe2.python.models.download --install
#If the above download worked then you should have a copy of squeezenet in your model folder or if you used the -i flag it will have installed the model locally in the /caffe2/python/models folder.
from caffe2.python import workspace
from caffe2.python.models import squeezenet as mynet
import numpy as np
init_net = mynet.init_net
predict_net = mynet.predict_net
# you must name it something
predict_net.name = "squeezenet_predict"

# Dummy batch
data = np.random.rand(1, 3, 227, 227)
workspace.FeedBlob("data", data)

workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
p = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/cafferelative/  

