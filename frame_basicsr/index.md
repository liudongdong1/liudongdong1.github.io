# Frame_BasicSR


> BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源图像视频复原工具箱, 比如 超分辨率, 去噪, 去模糊, 去 JPEG 压缩噪声等.
>
> - [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): 通用图像复原的实用算法
> - [**GFPGAN**](https://github.com/TencentARC/GFPGAN): 真实场景人脸复原的实用算法
> - [facexlib](https://github.com/xinntao/facexlib): 提供实用的人脸相关功能的集合
> - [HandyView](https://github.com/xinntao/HandyView): 基于PyQt5的 方便的看图比图工具
>
> <sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
> <sub>([HandyView](https://gitee.com/xinntao/HandyView), [HandyFigure](https://gitee.com/xinntao/HandyFigure), [HandyCrawler](https://gitee.com/xinntao/HandyCrawler), [HandyWriting](https://gitee.com/xinntao/HandyWriting))</sub>

![overall_structure](https://gitee.com/github-25970295/blogpictureV2/raw/master/picture/overall_structure.png)

> 1. 准备数据. 参见 [DatasetPreparation_CN.md](DatasetPreparation_CN.md)
> 1. 修改Config文件. Config文件在 `options` 目录下面. 具体的Config配置含义, 可参考 [Config说明](Config_CN.md)
> 1. [Optional] 如果是测试或需要预训练, 则需下载预训练模型, 参见 [模型库](ModelZoo_CN.md)
> 1. 运行命令. 根据需要，使用 [训练命令](#训练命令) 或 [测试命令](#测试命令)

### 1. Dataset

#### .1. Category

| 类                                                           |   任务   | 训练/测试 |                         描述                          |
| :----------------------------------------------------------- | :------: | :-------: | :---------------------------------------------------: |
| [PairedImageDataset](../basicsr/data/paired_image_dataset.py) | 图像超分 |   训练    |                支持读取成对的训练数据                 |
| [SingleImageDataset](../basicsr/data/single_image_dataset.py) | 图像超分 |   测试    | 只读取low quality的图像, 用在没有Ground-Truth的测试中 |
| [REDSDataset](../basicsr/data/reds_dataset.py)               | 视频超分 |   训练    |                   REDS的训练数据集                    |
| [Vimeo90KDataset](../basicsr/data/vimeo90k_dataset.py)       | 视频超分 |   训练    |                 Vimeo90K的训练数据集                  |
| [VideoTestDataset](../basicsr/data/video_test_dataset.py)    | 视频超分 |   测试    |      基础的视频超分测试集, 支持Vid4, REDS测试集       |
| [VideoTestVimeo90KDataset](../basicsr/data/video_test_dataset.py) | 视频超分 |   测试    |     继承`VideoTestDataset`, Vimeo90K的测试数据集      |
| [VideoTestDUFDataset](../basicsr/data/video_test_dataset.py) | 视频超分 |   测试    | 继承`VideoTestDataset`, 方法DUF的测试数据集, 支持Vid4 |
| [FFHQDataset](../basicsr/data/ffhq_dataset.py)               | 人脸生成 |   训练    |                   FFHQ的训练数据集                    |

#### .2. load type

##### 1. 直接读取硬盘数据

```yml
type: PairedImageDataset
dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
io_backend:
  type: disk
```

##### 2. 使用LMDB

```yml
type: PairedImageDataset
dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
io_backend:
  type: lmdb
```

##### 3. 使用Memcached

```yml
type: PairedImageDataset
dataroot_gt: datasets/DIV2K_train_HR_sub
dataroot_lq: datasets/DIV2K_train_LR_bicubicX4_sub
io_backend:
  type: memcached
  server_list_cfg: /mnt/lustre/share/memcached_client/server_list.conf
  client_cfg: /mnt/lustre/share/memcached_client/client.conf
  sys_path: /mnt/lustre/share/pymc/py3
```

### 2. Models

#### .1. List

-  **ECBSR training and testing** codes: [ECBSR](https://github.com/xindongzhang/ECBSR)
-  **SwinIR training and testing** codes: [SwinIR](https://github.com/JingyunLiang/SwinIR) 
- **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
- **dual-blind face restoration** codes: [HiFaceGAN](https://github.com/Lotayou/Face-Renovation) codes by [Lotayou](https://lotayou.github.io/).
- **ESRGAN** and **DFDNet** [colab demo](../colab)
- **blind face restoration** inference codes: [DFDNet](https://github.com/csxmli2016/DFDNet).
- **StyleGAN2 training and testing** codes: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).

| Class                                                   |                 Description                  |           Supported Algorithms            |
| :------------------------------------------------------ | :------------------------------------------: | :---------------------------------------: |
| [BaseModel](../basicsr/models/base_model.py)            | Abstract base class, define common functions |                                           |
| [SRModel](../basicsr/models/sr_model.py)                |             Base image SR class              | SRCNN, EDSR, SRResNet, RCAN, RRDBNet, etc |
| [SRGANModel](../basicsr/models/srgan_model.py)          |             SRGAN image SR class             |                   SRGAN                   |
| [ESRGANModel](../basicsr/models/esrgan_model.py)        |            ESRGAN image SR class             |                  ESRGAN                   |
| [VideoBaseModel](../basicsr/models/video_base_model.py) |             Base video SR class              |                                           |
| [EDVRModel](../basicsr/models/edvr_model.py)            |             EDVR video SR class              |                   EDVR                    |
| [StyleGAN2Model](../basicsr/models/stylegan2_model.py)  |          StyleGAN2 generation class          |                 StyleGAN2                 |

```
BaseModel
├── SRModel
│   ├── SRGANModel
│   │   └── ESRGANModel
│   └── VideoBaseModel
│       ├── VideoGANModel
│       └── EDVRModel
└── StyleGAN2Model
```

#### .2. ModelZoo

##### .0.  使用Pretrained model

**[下载官方提供的预训练模型]** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g))
你可以使用以下脚本从Google Drive下载预训练模型.

```python
python scripts/download_pretrained_models.py ESRGAN
# method can be ESRGAN, EDVR, StyleGAN, EDSR, DUF, DFDNet, dlib
```

**[下载复现的模型和log]** ([Google Drive](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing), [百度网盘](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ))，[wandb](https://www.wandb.com/) 上拥有模型训练的过程和曲线

##### 1. 图像超分官方模型

| Exp Name                                 | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | DIV2K100 (PSNR/SSIM) |
| :--------------------------------------- | :--------------: | :---------------: | :------------------: |
| EDSR_Mx2_f64b16_DIV2K_official-3ba7b086  | 35.7768 / 0.9442 | 31.4966 / 0.8939  |   34.6291 / 0.9373   |
| EDSR_Mx3_f64b16_DIV2K_official-6908f88a  | 32.3597 / 0.903  | 28.3932 / 0.8096  |   30.9438 / 0.8737   |
| EDSR_Mx4_f64b16_DIV2K_official-0c287733  | 30.1821 / 0.8641 | 26.7528 / 0.7432  |   28.9679 / 0.8183   |
| EDSR_Lx2_f256b32_DIV2K_official-be38e77d | 35.9979 / 0.9454 | 31.8583 / 0.8971  |   35.0495 / 0.9407   |
| EDSR_Lx3_f256b32_DIV2K_official-3660f70d |  32.643 / 0.906  |  28.644 / 0.8152  |    31.28 / 0.8798    |
| EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f | 30.5499 / 0.8701 | 27.0011 / 0.7509  |   29.277 / 0.8266    |

##### 2. 图像超分复现模型

| Exp Name                                                    | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | DIV2K100 (PSNR/SSIM) |
| :---------------------------------------------------------- | :--------------: | :---------------: | :------------------: |
| 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb             | 30.2468 / 0.8651 | 26.7817 / 0.7451  |   28.9967 / 0.8195   |
| 002_MSRResNet_x2_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 35.7483 / 0.9442 | 31.5403 / 0.8937  |   34.6699 / 0.9377   |
| 003_MSRResNet_x3_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 32.4038 / 0.9032 | 28.4418 / 0.8106  |   30.9726 / 0.8743   |
| 004_MSRGAN_x4_f64b16_DIV2K_400k_B16G1_wandb                 | 28.0158 / 0.8087 | 24.7474 / 0.6623  |   26.6504 / 0.7462   |
| 201_EDSR_Mx2_f64b16_DIV2K_300k_B16G1_wandb                  | 35.7395 / 0.944  | 31.4348 / 0.8934  |   34.5798 / 0.937    |
| 202_EDSR_Mx3_f64b16_DIV2K_300k_B16G1_201pretrain_wandb      | 32.315 / 0.9026  | 28.3866 / 0.8088  |   30.9095 / 0.8731   |
| 203_EDSR_Mx4_f64b16_DIV2K_300k_B16G1_201pretrain_wandb      | 30.1726 / 0.8641 |  26.721 / 0.743   |   28.9506 / 0.818    |
| 204_EDSR_Lx2_f256b32_DIV2K_300k_B16G1_wandb                 | 35.9792 / 0.9453 | 31.7284 / 0.8959  |   34.9544 / 0.9399   |
| 205_EDSR_Lx3_f256b32_DIV2K_300k_B16G1_204pretrain_wandb     | 32.6467 / 0.9057 | 28.6859 / 0.8152  |   31.2664 / 0.8793   |
| 206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb     | 30.4718 / 0.8695 | 26.9616 / 0.7502  |   29.2621 / 0.8265   |

##### 3. 视频超分辨率

###### 1. EDVR

EDVR\_(training dataset)\_(track name)\_(model complexity)

- track name. There are four tracks in the NTIRE 2019 Challenges on Video Restoration and Enhancement:
  - **SR**: `super-resolution` with a fixed downsampling kernel (MATLAB bicubic downsampling kernel is frequently used). Most of the previous video SR methods focus on this setting.
  - **SRblur**: `the inputs are also degraded with motion blur.`
  - **deblur**: `standard deblurring (motion blur)`.
  - **deblurcomp**: `motion blur + video compression artifacts.`
- model complexity
  - **L** (Large): # of channels = 128, # of back residual blocks = 40. This setting is used in our competition submission.
  - **M** (Moderate): # of channels = 64, # of back residual blocks = 10.

|       Model name       |                     [Test Set] PSNR/SSIM                     |
| :--------------------: | :----------------------------------------------------------: |
|   EDVR_Vimeo90K_SR_L   | [Vid4] (Y<sup>1</sup>) 27.35/0.8264 [[↓Results]](https://drive.google.com/open?id=14nozpSfe9kC12dVuJ9mspQH5ZqE4mT9K)<br/> (RGB) 25.83/0.8077 |
|     EDVR_REDS_SR_M     | [REDS] (RGB) 30.53/0.8699 [[↓Results]](https://drive.google.com/open?id=1Mek3JIxkjJWjhZhH4qVwTXnRZutKUtC-) |
|     EDVR_REDS_SR_L     | [REDS] (RGB) 31.09/0.8800 [[↓Results]](https://drive.google.com/open?id=1h6E0QVZyJ5SBkcnYaT1puxYYPVbPsTLt) |
|   EDVR_REDS_SRblur_L   | [REDS] (RGB) 28.88/0.8361 [[↓Results]](https://drive.google.com/open?id=1-8MNkQuMVMz30UilB9m_d0SXicwFEPZH) |
|   EDVR_REDS_deblur_L   | [REDS] (RGB) 34.80/0.9487 [[↓Results]](https://drive.google.com/open?id=133wCHTwiiRzenOEoStNbFuZlCX8Jn2at) |
| EDVR_REDS_deblurcomp_L |           [REDS] (RGB) 30.24/0.8567 [[↓Results]](            |

|         Model name          |     [Test Set] PSNR/SSIM      |
| :-------------------------: | :---------------------------: |
|     EDVR_REDS_SR_Stage2     | [REDS] (RGB) / [[↓Results]]() |
|   EDVR_REDS_SRblur_Stage2   | [REDS] (RGB) / [[↓Results]]() |
|   EDVR_REDS_deblur_Stage2   | [REDS] (RGB) / [[↓Results]]() |
| EDVR_REDS_deblurcomp_Stage2 | [REDS] (RGB) / [[↓Results]]() |

###### 2. DUF&TOF

### 3. 配置文件

#### .1. 实验命名

以`001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb`为例:

- `001`: 我们一般给实验标号, 方便实验管理
- `MSRResNet`: 模型名称, 这里是Modified SRResNet
- `x4_f64b16`: 重要配置参数, 这里表示放大4倍; 中间feature通道数是64, 使用了16个Residual Block
- `DIV2K`: 训练数据集是DIV2K
- `1000k`: 训练了1000k iterations
- `B16G1`: Batch size为16, 使用一个GPU训练
- `wandb`: 使用了wandb, 训练过程上传到了wandb云服务器

#### .2. 训练配置文件

```yml
#################
# 以下为通用的设置
#################
# 实验名称, 具体可参见 [实验名称命名], 若实验名字中有debug字样, 则会进入debug模式
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# 使用的model类型, 一般为在`models`目录下定义的模型的类名
model_type: SRModel
# 输出相比输入的放大比率, 在SR中是放大倍数; 若有些任务没有这个配置, 则写1
scale: 4
# 训练卡数
num_gpu: 1  # set num_gpu: 0 for cpu mode
# 随机种子设定
manual_seed: 0

#################################
# 以下为dataset和data loader的设置
#################################
datasets:
  # 训练数据集的设置
  train:
    # 数据集的名称
    name: DIV2K
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    # 文件名字模板, 一般LQ文件会有类似`_x4`这样的文件后缀, 这个就是来处理GT和LQ文件后缀不匹配的问题的
    filename_tmpl: '{}'
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk

    # 训练中Ground-Truth的Training patch的大小
    gt_size: 128
    # 是否使用horizontal flip, 这里的flip特指 horizontal flip
    use_flip: true
    # 是否使用rotation, 这里指的是每隔90°旋转
    use_rot: true

    #### 下面是data loader的设置
    # data loader是否使用shuffle
    use_shuffle: true
    # 每一个GPU的data loader读取进程数目
    num_worker_per_gpu: 6
    # 总共的训练batch size
    batch_size_per_gpu: 16
    # 扩大dataset的倍率. 比如数据集有15张图, 则会重复这些图片100次, 这样一个epoch下来, 能够读取1500张图
    # (事实上是重复读的). 它经常用来加速data loader, 因为在有的机器上, 一个epoch结束, 会重启进程, 往往会很慢
    dataset_enlarge_ratio: 100

  # validation 数据集的设置
  val:
    # 数据集名称
    name: Set5
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/Set5/LRbicx4
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk

#####################
# 以下为网络结构的设置
#####################
# 网络g的设置
network_g:
  # 网络结构 (Architecture)的类型, 一般为在`basicsr/archs`目录下定义的dataset的类名
  type: MSRResNet
  #### 以下属性是灵活的, 可以在相应类的说明文档中获得
  # 输入通道数目
  num_in_ch: 3
  # 输出通道数目
  num_out_ch: 3
  # 中间特征通道数目
  num_feat: 64
  # 使用block的数目
  num_block: 16
  # SR的放大倍数
  upscale: 4

######################################
# 以下为路径和与训练模型、重启训练的设置
######################################
path:
  # 预训练模型的路径, 需要以pth结尾的模型
  pretrain_network_g: ~
  # 加载预训练模型的时候, 是否需要网络参数的名称严格对应
  strict_load_g: true
  # 重启训练的状态路径, 一般在`experiments/exp_name/training_states`目录下
  # 这个设置了, 会覆盖  pretrain_network_g 的设定
  resume_state: ~


#################
# 以下为训练的设置
#################
train:
  # 优化器设置
  optim_g:
    # 优化器类型
    type: Adam
    ##### 以下属性是灵活的, 根据不同优化器有不同的设置
    # 学习率
    lr: !!float 2e-4
    weight_decay: 0
    # Adam优化器的 beta1 和 beta2
    betas: [0.9, 0.99]

  # 学习率的设定
  scheduler:
    # 学习率Scheduler的类型
    type: CosineAnnealingRestartLR
    #### 以下属性是灵活的, 根据学习率Scheduler有不同的设置
    # Cosine Annealing的周期
    periods: [250000, 250000, 250000, 250000]
    # Cosine Annealing每次Restart的权重
    restart_weights: [1, 1, 1, 1]
    # Cosine Annealing的学习率最小值
    eta_min: !!float 1e-7

  # 总共的训练迭代次数
  total_iter: 1000000
  # warm up的iteration数目, 如是-1, 表示没有warm up
  warmup_iter: -1  # no warm up

  #### 以下是loss的设置
  # pixel-wise loss的options
  pixel_opt:
    # loss类型, 一般为在`basicsr/models/losses`目录下定义的loss类名
    type: L1Loss
    # loss 权重
    loss_weight: 1.0
    # loss reduction方式
    reduction: mean


#######################
# 以下为Validation的设置
#######################
val:
  # validation的频率, 每隔 5000 iterations 做一次validation
  val_freq: !!float 5e3
  # 是否需要在validation的时候保存图片
  save_img: false

  # Validation时候使用的metric
  metrics:
    # metric的名字, 这个名字可以是任意的
    psnr:
      # metric的类型, 一般为在`basicsr/metrics`目录下定义的metric函数名
      type: calculate_psnr
      #### 以下属性是灵活的, 根据metric有不同的设置
      # 计算metric时, 是否需要crop border
      crop_border: 4
      # 是否转成在Y(CbCr)空间上计算metric
      test_y_channel: false

####################
# 以下为Logging的设置
####################
logger:
  # 屏幕上打印的logger频率
  print_freq: 100
  # 保存checkpoint的频率
  save_checkpoint_freq: !!float 5e3
  # 是否使用tensorboard logger
  use_tb_logger: true
  # 是否使用wandb logger, 目前wandb只是同步tensorboard的内容, 因此要使用wandb, 必须也同时使用tensorboard
  wandb:
    # wandb的project. 默认是 None, 即不使用wandb.
    # 这里使用了 basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # 如果是resume, 可以输入上次的wandb id, 则log可以接起来
    resume_id: ~

#############################################################
# 以下为distributed training的设置, 目前只有在Slurm训练下才需要
#############################################################
dist_params:
  backend: nccl
  port: 29500
```

#### .3. 测试配置文件

```yml
# 实验名称
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# 使用的model类型, 一般为在`models`目录下定义的模型的类名
model_type: SRModel
# 输出相比输入的放大比率, 在SR中是放大倍数; 若有些任务没有这个配置, 则写1
scale: 4
# 测试卡数
num_gpu: 1  # set num_gpu: 0 for cpu mode

#################################
# 以下为dataset和data loader的设置
#################################
datasets:
  # 测试数据集的设置, 后缀1表示第一个测试集
  test_1:
    # 数据集的名称
    name: Set5
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/Set5/LRbicx4
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk
  # 测试数据集的设置, 后缀2表示第二个测试集
  test_2:
    name: Set14
    type: PairedImageDataset
    dataroot_gt: datasets/Set14/GTmod12
    dataroot_lq: datasets/Set14/LRbicx4
    io_backend:
      type: disk
  # 测试数据集的设置, 后缀3表示第三个测试集
  test_3:
    name: DIV2K100
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_valid_HR
    dataroot_lq: datasets/DIV2K/DIV2K_valid_LR_bicubic/X4
    filename_tmpl: '{}x4'
    io_backend:
      type: disk

#####################
# 以下为网络结构的设置
#####################
# 网络g的设置
network_g:
  # 网络结构 (Architecture)的类型, 一般为在`basicsr/archs`目录下定义的dataset的类名
  type: MSRResNet
  #### 以下属性是灵活的, 可以在相应类的说明文档中获得
  # 输入通道数目
  num_in_ch: 3
  # 输出通道数目
  num_out_ch: 3
  # 中间特征通道数目
  num_feat: 64
  # 使用block的数目
  num_block: 16
  # SR的放大倍数
  upscale: 4

#############################
# 以下为路径和与训练模型的设置
#############################
path:
  # 预训练模型的路径, 需要以pth结尾的模型
  pretrain_network_g: experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth
  # 加载预训练模型的时候, 是否需要网络参数的名称严格对应
  strict_load_g: true

##################################
# 以下为Validation (也是测试)的设置
##################################
val:
  # 是否需要在测试的时候保存图片
  save_img: true
  # 对保存的图片添加后缀，如果是None, 则使用exp name
  suffix: ~

  # 测试时候使用的metric
  metrics:
    # metric的名字, 这个名字可以是任意的
    psnr:
      # metric的类型, 一般为在`basicsr/metrics`目录下定义的metric函数名
      type: calculate_psnr
      #### 以下属性是灵活的, 根据metric有不同的设置
      # 计算metric时, 是否需要crop border
      crop_border: 4
      # 是否转成在Y(CbCr)空间上计算metric
      test_y_channel: false
    # 另外一个metric
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```

### 4. Logging

- 文本屏幕日志： `experiments/exp_name/train_exp_name_timestamp.txt`

#### .1. Tensorboard日志

- 开启. 在 yml 配置文件中设置 `use_tb_logger: true`:

  ```yml
  logger:
    use_tb_logger: true
  ```

- 文件位置: `tb_logger/exp_name`

- 在浏览器中查看:

  ```bash
  tensorboard --logdir tb_logger --port 5500 --bind_all
  ```

#### .2. Wandb日志

[wandb](https://www.wandb.com/) 类似tensorboard的云端版本, 可以在浏览器方便地查看模型训练的过程和曲线. 我们目前只是把tensorboard的内容同步到wandb上, 因此要使用wandb, 必须打开tensorboard logger.

```yml
logger:
  # 是否使用tensorboard logger
  use_tb_logger: true
  # 是否使用wandb logger, 目前wandb只是同步tensorboard的内容, 因此要使用wandb, 必须也同时使用tensorboard
  wandb:
    # wandb的project. 默认是 None, 即不使用wandb.
    # 这里使用了 basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # 如果是resume, 可以输入上次的wandb id, 则log可以接起来
    resume_id: ~
```

### 5. Metrics [GAN评价指标](https://zhuanlan.zhihu.com/p/109342043)

#### .1. SSIM

> structural similarity index是一种用以`衡量两张数字图像相似度的指标`。当其中一张为无失真的图片，另一张是失真图片时，二者的SSIM可以看成是`失真图片的影响品质衡量指标`。相比于PSNR，SSIM在图片品质的衡量上更能符合人眼对图片质量的判断。
>
> 结构相似性的基本观点为图片是高度结构化的，相邻像素之间有很强的关联性。而这样的关联性体现了图像中物体的结构信息。人类的视觉系统在观看图片时习惯于抽取这样的结构信息。因此在设计品质用以衡量图片失真程度时，结构性失真的衡量是很重要的一环。` SSIM取值范围[0,1]，值越大，表示图像失真越小`

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211022113012993.png)

#### .2. MSSIM

> MSSIM（Mean Structural Similarity ）平均结构相似性,用滑动窗将图像分为N块，加权计算每一窗口的均值、方差以及协方差，权值wij满足∑i∑jwij=1，通常采用高斯核，然后计算对应块的结构相似度SSIM，最后将平均值作为两图像的结构相似性度量。

#### .3. FID

> FID（Fréchet Inception Distance）是一种评价GAN的指标，于2017年提出，它的想法是这样的：分别把`生成器生成的样本`和`判别器生成的样本`送到分类器中（例如Inception Net-V3或者其他CNN等），抽取分类器的中间层的抽象特征，并假设该抽象特征符合多元高斯分布，估计生成样本高斯分布的均值 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_g) 和方差 ![[公式]](https://www.zhihu.com/equation?tex=%5CSigma_g) ，以及训练样本 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_%7Bdata%7D) 和方差 ![[公式]](https://www.zhihu.com/equation?tex=%5CSigma_%7Bdata%7D) ，计算`两个高斯分布的弗雷歇距离，此距离值即FID`.
>
> `FID的数值越小，表示两个高斯分布越接近，GAN的性能越好`。实践中发现，FID对噪声具有比较好的鲁棒性，能够对生成图像的质量有比较好的评价，其给出的分数与人类的视觉判断比较一致，并且FID的计算复杂度并不高，虽然FID只考虑的样本的一阶矩和二阶矩，但整体而言，FID还是比较有效的，其理论上的不足之处在于：高斯分布的简化假设在实际中并不成立。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211022150812381.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211022150755790.png)

#### .4. PSNR

> Peak signal-to-noise ratio, 峰值信噪比，是一个表示`信号最大可能功率和影响它的表示精度的破坏性噪声功率的比值的工程术语`。峰值信噪比经常用作图像压缩等领域中信号重建质量的测量方法，它常简单地通过均方误差（MSE）进行定义。`数值越大表示失真越小`.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211022145737268.png)

### 6. 训练测试命令

#### .1. 单GPU

```python
#训练命令
PYTHONPATH="./:${PYTHONPATH}" \\\
CUDA_VISIBLE_DEVICES=0 \\\
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml
#测试命令
PYTHONPATH="./:${PYTHONPATH}" \\\
CUDA_VISIBLE_DEVICES=0 \\\
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml
```

#### .2. Colab Demo

-   [BasicSR_inference_DFDNet.ipynb](https://colab.research.google.com/drive/1RoNDeipp9yPjI3EbpEbUhn66k5Uzg4n8?usp=sharing)
- [BasicSR_inference_ESRGAN.ipynb](https://colab.research.google.com/drive/1JQScYICvEC3VqaabLu-lxvq9h7kSV1ML?usp=sharing)

### 8. 代码使用解读

- 环境安琥
  - local clone， 然后python [setup.py](http://link.zhihu.com/?target=http%3A//setup.py) develop/instal
  - pip 安装

#### .1. 代码目录

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-af0541341b79960a452ed9a4d756beef_b.jpg)

- basicsr目录

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-8f043bb9b91a5f2c3d2cd42b2d7f1fef_b.jpg)

- scripts 文件

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-1434197ea0db5a763191383be47d2942_b.jpg)

### Resource

- https://www.zhihu.com/column/c_1295528110138163200
- https://github.com/xinntao/BasicSR

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/frame_basicsr/  

