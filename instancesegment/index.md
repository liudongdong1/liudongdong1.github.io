# InstanceSegment


- present a simple, fully-convolutional model for real-time instance segmentation that faster than any previous competitive approach.
- `generating a set of prototype masks` and `predicting per-instance mask coefficients`. And produce instance masks by linearly combining the prototypes with the mask coefficients.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308002640634.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308003525274.png)

- prototype generation: predicts a set of k prototype masks for the entire image.
  - taking protonet from deeper backbone features produces more robust masks, and higher resolution prototypes result in both higher quality masks and better performance on smaller objects.
  - predicts k mask coefficients, one corresponding to each prototype.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308090006101.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308091343761.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308091420813.png)

Relative

- focus primarily on performance over speed, leaving the scene devoid of instance segmentation parrallels to real-time object detectors like SSD, and YOLO.
- Instance Segmentation: 
  - Mask-RCNN: two-stage instance segmentation approach that generates candidate region-of-interests and then classifies and segments the ROIs.
  - semantic segmentation followed by boundary detection, pixel clustering or learn an embedding to form instance masks.
- Real-time Instance Segmentation:
  - Mask R-CNN remains one of the fastest instance segmentation methods on semantically challenging datasets like COCO;

#### Code

- backbone结构

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308093910226.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308092221562.png)

- FPN 结构

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308092453440.png)

- proto结构

> Protonet 的设计是受到了 Mask R-CNN 的启发，它由若干卷积层组成。其输入是0，其输出的 mask 维度是 138*138*32，即 32 个 prototype mask，每个大小是 138*138。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20200120163709273.png)

- predict_head

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308094136106.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20200120172944570.png)

> 'loc'：每个anchorbox的预测偏移量，形状为（1，19248，4）
>
> 'conf'：每个anchorbox的类别预测，形状为（1，19248，num_class）
>
> 'mask'：就是论文中指出的mask系数，形状为（1，19248，32）
>
> 'priors'：预设的anchorbox的坐标，形状为（19248，4）
>
> 'proto'：与mask系数配合使用的分割特征图，形状为（1，138，138，32）
>
> 'segm'：得到一个类似分割的热度图，这里的形状为（1，num_class-1，69，69），我估计segm是为了使得网络快速收敛用的。

- 损失函数

> net：就是上面的网络结构。
>
> preds：是一个字典，上面红色字体部分就是preds中字典的内容。
>
> targets：一般的形状为（batch，n，5），batch就是输入了常规的batchsize，n表示一张图片中有几个目标物体，5当中前4个表示目标物体的坐标，第5个数字表示该目标物体的类别。
>
> masks：一般形状为（batch，n，550，550），这个n不是固定的，batch中的每张图片得到的目标物体数量都不相同，这个mask和maskrcnn那个是一样的。
>
> num_crowds：（batch，）表示拥挤程度，0表示不拥挤，1表示拥挤，一般都是0。

- python 实现代码： https://github1s.com/hpc203/yolact-opencv-dnn-cpp-python/blob/HEAD/main_yolact.py


### 2. Railroad is not a Train

> Lee, Seungho, et al. "Railroad is not a Train: Saliency as Pseudo-pixel Supervision for Weakly Supervised Semantic Segmentation." *arXiv preprint arXiv:2105.08965* (2021).

> 本次工作所提出方案：提出 Explicit Pseudo-pixel Supervision（EPS），通过结合两个弱监督从像素级反馈中学习；图像级标签通过 localization map，以及来自现成的显著检测模型提供丰富边界的 saliency map 来提供目标身份。作者进而又设计一种联合训练策略，可以充分利用两种信息之间的互补关系。所提出方法可以获得准确的物体边界，并摒弃共同出现的像素，从而显著提高 pseudo-masks 的质量。
>
> - 论文链接：https://arxiv.org/abs/2105.08965
> - 项目链接：https://github.com/halbielee/EPS

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210528104226031.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/instancesegment/  

