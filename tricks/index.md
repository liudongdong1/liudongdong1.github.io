# Tricks


### 1. 数据增强

> 数据增强是增加深度模型鲁棒性和泛化性能的常用手段，随机翻转、随机裁剪、添加噪声等也被引入到检测任务的训练中来，个人认为数据（监督信息）的适时传入可能是更有潜力的方向。

#### .1. 图像增强

- 源码在mmdet/datasets/extra_aug.py里面，包括RandomCrop、brightness、contrast、saturation、ExtraAugmentation等等图像增强方法。
- 添加位置是train_pipeline或test_pipeline这个地方（一般train进行增强而test不需要），例如数据增强RandomFlip，flip_ratio代表随机翻转的概率：

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

#### .2. Bbox 增强

- 源码在mmdet/datasets/custom.py里面，增强源码为：

```python
def pre_pipeline(self, results):        
    results['img_prefix'] = self.img_prefix        
    results['seg_prefix'] = self.seg_prefix        
    results['proposal_file'] = self.proposal_file        
    results['bbox_fields'] = []        
    results['mask_fields'] = []
```

### 2. **Multi-scale Training/Testing 多尺度训练/测试**

> 通过输入更大、更多尺寸的图片进行训练，能够在一定程度上提高检测模型对物体大小的鲁棒性，仅在测试阶段引入多尺度，也可享受大尺寸和多尺寸带来的增益。
>
> 训练时，预先定义几个固定的尺度，每个epoch随机选择一个尺度进行训练。测试时，生成几个不同尺度的feature map，对每个Region Proposal，在不同的feature map上也有不同的尺度，我们选择最接近某一固定尺寸（即检测头部的输入尺寸）的Region Proposal作为后续的输入。
>
> 在[2]中，选择单一尺度的方式被Maxout（element-wise max，逐元素取最大）取代：随机选两个相邻尺度，经过Pooling后使用Maxout进行合并，如下图所示。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210518222638.png)

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True), #这里可以更换多尺度[(),()]
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

### 3.**Box Refinement/Voting 预测框微调/投票法**

- box voting 的阈值，
- 不同的输入中**这个框至少出现了几次来**允许它输出，
- 得分的阈值，一个目标框的得分低于这个阈值的时候，就删掉这个目标框。

### 4.**随机权值平均（SWA**）

 SWA 的工作原理。它只保存两个模型，而不是许多模型的集成：

1. 第一个模型保存模型权值的平均值（WSWA）。在训练结束后，它将是用于预测的最终模型。
2. 第二个模型（W）将穿过权值空间，基于周期性学习率规划探索权重空间。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210518223011.png)

### 5. 其他

- 更好的先验（YOLOv2）：使用聚类方法统计数据中box标注的大小和长宽比，以更好的设置anchor box的生成配置
- 更好的pre-train模型：检测模型的基础网络通常使用ImageNet（通常是ImageNet-1k）上训练好的模型进行初始化，使用更大的数据集（ImageNet-5k）预训练基础网络对精度的提升亦有帮助
- 超参数的调整：部分工作也发现如NMS中IoU阈值的调整（从0.3到0.5）也有利于精度的提升，但这一方面尚无最佳配置参照

```python
# model settings
model = dict(
  type='FasterRCNN',                         # model类型
    pretrained='modelzoo://resnet50',          # 预训练模型：imagenet-resnet50
    backbone=dict(
        type='ResNet',                         # backbone类型
        depth=50,                              # 网络层数
        num_stages=4,                          # resnet的stage数量
        out_indices=(0, 1, 2, 3),              # 输出的stage的序号
        frozen_stages=1,                       # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        style='pytorch'),                      # 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
    neck=dict(
        type='FPN',                            # neck类型
        in_channels=[256, 512, 1024, 2048],    # 输入的各个stage的通道数
        out_channels=256,                      # 输出的特征层的通道数
        num_outs=5),                           # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',                        # RPN网络类型
        in_channels=256,                       # RPN网络的输入通道数
        feat_channels=256,                     # 特征层的通道数
        anchor_scales=[8],                     # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
        anchor_ratios=[0.5, 1.0, 2.0],         # anchor的宽高比
        anchor_strides=[4, 8, 16, 32, 64],     # 在每个特征层上的anchor的步长（对应于原图）
        target_means=[.0, .0, .0, .0],         # 均值
        target_stds=[1.0, 1.0, 1.0, 1.0],      # 方差
        use_sigmoid_cls=True),                 # 是否使用sigmoid来进行分类，如果False则使用softmax来分类
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',                                   # RoIExtractor类型
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),   # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
        out_channels=256,                                            # 输出通道数
        featmap_strides=[4, 8, 16, 32]),                             # 特征图的步长
    bbox_head=dict(
        type='SharedFCBBoxHead',                     # 全连接层类型
        num_fcs=2,                                   # 全连接层数量
        in_channels=256,                             # 输入通道数
        fc_out_channels=1024,                        # 输出通道数
        roi_feat_size=7,                             # ROI特征层尺寸
        num_classes=81,                              # 分类器的类别数量+1，+1是因为多了一个背景的类别
        target_means=[0., 0., 0., 0.],               # 均值
        target_stds=[0.1, 0.1, 0.2, 0.2],            # 方差
        reg_class_agnostic=False))                   # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',            # RPN网络的正负样本划分
            pos_iou_thr=0.7,                  # 正样本的iou阈值
            neg_iou_thr=0.3,                  # 负样本的iou阈值
            min_pos_iou=0.3,                  # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),               # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',             # 正负样本提取器类型
            num=256,                          # 需提取的正负样本数量
            pos_fraction=0.5,                 # 正样本比例
            neg_pos_ub=-1,                    # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False),       # 把ground truth加入proposal作为正样本
        allowed_border=0,                     # 允许在bbox周围外扩一定的像素
        pos_weight=-1,                        # 正样本权重，-1表示不改变原始的权重
        smoothl1_beta=1 / 9.0,                # 平滑L1系数
        debug=False),                         # debug模式
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',            # RCNN网络正负样本划分
            pos_iou_thr=0.5,                  # 正样本的iou阈值
            neg_iou_thr=0.5,                  # 负样本的iou阈值
            min_pos_iou=0.5,                  # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),               # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',             # 正负样本提取器类型
            num=512,                          # 需提取的正负样本数量
            pos_fraction=0.25,                # 正样本比例
            neg_pos_ub=-1,                    # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=True),        # 把ground truth加入proposal作为正样本
        pos_weight=-1,                        # 正样本权重，-1表示不改变原始的权重
        debug=False))                         # debug模式
test_cfg = dict(
    rpn=dict(                                 # 推断时的RPN参数
        nms_across_levels=False,              # 在所有的fpn层内做nms
        nms_pre=2000,                         # 在nms之前保留的的得分最高的proposal数量
        nms_post=2000,                        # 在nms之后保留的的得分最高的proposal数量
        max_num=2000,                         # 在后处理完成之后保留的proposal数量
        nms_thr=0.7,                          # nms阈值
        min_bbox_size=0),                     # 最小bbox尺寸
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)   # max_per_img表示最终输出的det bbox数量
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)            # soft_nms参数
)
# dataset settings
dataset_type = 'CocoDataset'                # 数据集类型
data_root = 'data/coco/'                    # 数据集根目录
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)   # 输入图像初始化，减去均值mean并处以方差std，to_rgb表示将bgr转为rgb
data = dict(
    imgs_per_gpu=2,                # 每个gpu计算的图像数量
    workers_per_gpu=2,             # 每个gpu分配的线程数
    train=dict(
        type=dataset_type,                                                 # 数据集类型
        ann_file=data_root + 'annotations/instances_train2017.json',       # 数据集annotation路径
        img_prefix=data_root + 'train2017/',                               # 数据集的图片路径
        img_scale=(1333, 800),                                             # 输入图像尺寸，最大边1333，最小边800
        img_norm_cfg=img_norm_cfg,                                         # 图像初始化参数
        size_divisor=32,                                                   # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
        flip_ratio=0.5,                                                    # 图像的随机左右翻转的概率
        with_mask=False,                                                   # 训练时附带mask
        with_crowd=True,                                                   # 训练时附带difficult的样本
        with_label=True),                                                  # 训练时附带label
    val=dict(
        type=dataset_type,                                                 # 同上
        ann_file=data_root + 'annotations/instances_val2017.json',         # 同上
        img_prefix=data_root + 'val2017/',                                 # 同上
        img_scale=(1333, 800),                                             # 同上
        img_norm_cfg=img_norm_cfg,                                         # 同上
        size_divisor=32,                                                   # 同上
        flip_ratio=0,                                                      # 同上
        with_mask=False,                                                   # 同上
        with_crowd=True,                                                   # 同上
        with_label=True),                                                  # 同上
    test=dict(
        type=dataset_type,                                                 # 同上
        ann_file=data_root + 'annotations/instances_val2017.json',         # 同上
        img_prefix=data_root + 'val2017/',                                 # 同上
        img_scale=(1333, 800),                                             # 同上
        img_norm_cfg=img_norm_cfg,                                         # 同上
        size_divisor=32,                                                   # 同上
        flip_ratio=0,                                                      # 同上
        with_mask=False,                                                   # 同上
        with_label=False,                                                  # 同上
        test_mode=True))                                                   # 同上
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)   # 优化参数，lr为学习率，momentum为动量因子，weight_decay为权重衰减因子
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))          # 梯度均衡参数
# learning policy
lr_config = dict(
    policy='step',                        # 优化策略
    warmup='linear',                      # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=500,                     # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3,                 # 起始的学习率
    step=[8, 11])                         # 在第8和11个epoch时降低学习率
checkpoint_config = dict(interval=1)      # 每1个epoch存储一次模型
# yapf:disable
log_config = dict(
    interval=50,                          # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),      # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12                               # 最大epoch数
dist_params = dict(backend='nccl')              # 分布式参数
log_level = 'INFO'                              # 输出信息的完整度级别
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x' # log文件和模型文件存储路径
load_from = None                                # 加载模型的路径，None表示从预训练模型加载
resume_from = None                              # 恢复训练模型的路径
workflow = [('train', 1)]                       # 当前工作区名称
```

- 学习链接：https://mp.weixin.qq.com/s/W2_AK8p-CX2yNHxbJo77Kg

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/tricks/  

