# SuperResolution


> Tian Y, Zhang Y, Fu Y, et al. Tdan: Temporally-deformable alignment network for video super-resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 3360-3369.

### Paper: Tdan

<div align=center>
<br/>
<b> Tdan: Temporally-deformable alignment network for video super-resolution</b>
</div>


#### Summary

1. propose a temporally-deformable alignment network(TDAN) to `adaptively align the reference frame and each supporting frame a the feature level without computing optical flow.`
2. use features from both the reference frame and each supporting frame to `dynamically predict offsets of sampling convolution kernels`, to transforms `supporting frames to align with the reference frame`.
3. taking aligned frames and the reference frame to predict the HR video frame.

#### Research Objective

  - **Application Area**: `Video super-resolution` aims to restore a photo-realistic high-resolution video frame from both its` corresponding low-resolution frame (reference frame)` and `multiple neighboring frames (supporting frames).`
      - varying motion of cameras, or objects, the reference frame and each support frame are not alighned.
- **Relative work**:
  - optical flow to `predict motion fields` between the reference frame and supporting frames, then warp the supporting frames using their corresponding motion fields.  

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211016224337688.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211016223010811.png)

- TDAN to `align each supporting frame with the reference frame`, a LR supproting frame, and referencfe frame-->feeding 2N support frames to get  2N corresponding aligned LR frames.

  - feature extraction: use one convolutional layer amd k1 residual blocksto extracts visual features.
  - deformable alignment: takes the features mentioned above to predict sampling parameters. (refers to the offsdets o fthe convolution kernels.)  the feature of the reference frame is only used for computing the offset, its information will not propagated into the aligned feature of the supporting frame.  The adaptively-learned offset will implicitly capture motion cues and explore neighboring features within the sma eimage structures for alignment.
  - aligned frame reconstruction: restore an aligned LR frame and utilize an alignment loss to enforce the deformable alignment module to sample useful features for accurate temporal alignment.

- supre resolution reconstruction network to predict the HR frame:  2N corresponding aligned LR frames+ reference frame --> reconstruct the HR video frame.

  - Temporal Fusion: concatenate the 2N+1 frames and then feed them into a 3*3 convolutional layer to output fused feature map;
  - Nonlinear Mapping: take th eshadow fused features as input to predict deep features.
  - utilize an upscaling layer to increase the resolution of th efeature map with a sub-pixel convolution.

- Loss Function:

  - utilize the reference frame as the label and make the aligned LR frames close to the reference frame.
  - utilize the final HR video estimated frame with HR video frame.

  ![Lalign](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017123513481.png)

  ![Lsr](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017123619469.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211022212648591.png)

> the aligned frame is reconstructed from features from the reference and supporting frames. Green points in the supporting frame indicate sampling positions for predicting corresponding pixels labeled withred color in the aligned frame.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211016223232945.png)

> the TDAN can expoit rich image contexts containing similar content (green regions) as target pixels (red points) from the supporting frame to employ accurately temporal alignment.

![Temporal alignment](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211016223752368.png)

#### Code

```python
class TDAN_VSR(nn.Module):
    def __init__(self):
        super(TDAN_VSR, self).__init__()
        self.name = 'TDAN'
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)

        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        # deformable
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]

        self.fea_ex = nn.Sequential(*fea_ex)
        self.recon_layer = self.make_layer(Res_Block, 10)     
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),      #？？
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up = nn.Sequential(*upscaling)  

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)  # 按dim 维度进行拼接，
            fea = self.cr(fea)
            # feature trans
            offset1 = self.off2d_1(fea)
            fea = (self.dconv_1(fea, offset1))
            offset2 = self.off2d_2(fea)
            fea = (self.deconv_2(fea, offset2))
            offset3 = self.off2d_3(fea)
            fea = (self.deconv_3(supp, offset3))
            offset4 = self.off2d(fea)
            aligned_fea = (self.dconv(fea, offset4))
            im = self.recon_lr(aligned_fea).unsqueeze(1)  #去掉维数为1的的维度，比如是一行或者一列这种
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):

        batch_size, num, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        center = num // 2
        # extract features
        y = x.view(-1, ch, w, h)     #这个y作用是什么？原始图像，和recon_lr 特征提取后的数据融合在一起           # batch_size*num, ch, w, h
        # y = y.unsqueeze(1)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)

        # align supporting frames
        lrs = self.align(out, x_center) # motion alignments
        y = lrs.view(batch_size, -1, w, h)
        # reconstruction
        fea = self.fea_ex(y)

        out = self.recon_layer(fea)
        out = self.up(out)
        return out, lrs
```

> Lim B, Son S, Kim H, et al. Enhanced deep residual networks for single image super-resolution[C]//Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017: 136-144.  cite  [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1707.02921.pdf)

------

### Paper: EDSR

<div align=center>
<br/>
<b>Enhanced deep residual networks for single image super-resolution</b>
</div>
#### Summary

1. optimize the SRResNet architecture by analyzing and removing unnecessary modules to simplify the network architecture. Train the network with appropricate loss function and careful model modification upon training.
2. propose a new multi-scale architecture that shares most of the parameters across different scales.

#### previous work:

- #### `interpolation techniques` based on sampling theory limites in predicting detailed, realistic textures.
- learn the mapping functions between $I^{LR}$ to $I^{HR}$​, including neighbor embedding, to sparse coding.

#### Methods

- **system overview**:

![Single-scale model](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211023172415860.png)

> building a multi-scale architecture that takes the advantage of inter-scale correlation as VDSR, and introduce scale specific processing modules to handle the super-resolution at multiple scales.
>
> - `pre-processing modules` are located at the head of networks to reduce the variance from input images of different scales. each consists of two residual blocks with 5*5 kernels to keep the scale-specific part shallow while the larger receptive field is covered in early stages of networks.
> - at the end of the multi-scale model, `scale-specific upsampling modules` are located in parallel to handle multi-scale reconstruction.

![MDSR](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211024085357018.png)

> batch normalization layers normalize the features, `they get rid of range flexibility` from networks by normalizing the features. GPU memory usage sufficiently reduced.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211023172200022.png)

#### Evaluation

- Dataset:  
  - DIV2K dataset is a newly proposed high-quality(2K resolution) image dataset for image restoration tasks, consisting 800 training images, 100 valication images, and 100 test images.
- use the RGB input patches of size 48*48 from LR image with the corresponding HR patches.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211024090426859.png)

> public benchmark test results and DIV2K validation results( PSNR(db)/SSIM), red indicates the best performance and the blue indicates the second best.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211024090558584.png)

#### Code

- single-scale EDSR network

```python
@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
```

> Wang X, Chan K C K, Yu K, et al. Edvr: Video restoration with enhanced deformable convolutional networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2019: 0-0.  [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1905.02716.pdf) 

------

### Paper: Edvr

<div align=center>
<br/>
<b> Edvr: Video restoration with enhanced deformable convolutional networks</b>
</div>


#### Summary

1. devise a Pyramid, Cascading and Deformable alignment module, in which frame alignment is done at the feature level using deformable convolutions in a coarse-to-fine manner, to handle large motions.
2. propose a Temporal and Spatial Attention fusion module, in which attention is applied both temporally and spatially, so as to emphasize important features for subsequenct restoration.

#### Proble Statement

- how to align multiple frames given large motions?
- how to effectively fuse different frames with diverse motion and blur?

#### Relative Work

- **Video Super-Resolution**: RCAN, DeepSR, BayesSR, VESPCN, SPMC, TOFlow, FRVSR, DUF, RBPN on three testing datasets, Vid4, Vimeo-90K-T, REDS4.
- **Video Deblurring:** DeepDeblur, DeblurGAN, SRNDEblur, DBN on the REDS4 dataset.

#### Methods

- **system overview**:

> Given 2N+1 consecutive low-quality frames $I_{t-N:t+N}$ , denote the middle frame $I_t$ as the reference frame and the other frames as neighboring frames, to estimate a high-quality reference frame $Q_t$​. 
>
> - each neighboring frame is aligned to the reference one by the PCD alignment module at the feature level.
> - TSA fusion module fuses image information of different frames.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025201730909.png)

```python
@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(EDVR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)  # 到这里每一张图片和reference都有一个对应关系

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out
```

- **Alignment with Pyramid, Cascading and Deformable convolution:** 

> To generate feature $F^l_{t+i}$​​ at the l-th level, use strided convolution filters to downsample the features at the (l-1)-th pyramid level by a factor of 2, obtaining L-level pyramids of feature representation. At the l-th level, offsets and aligned features are predicted also with the *2 upsampled offsets and aligned features from the upper (l+1)-th level.  (`下面这个公式没有看懂`)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025210506574.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025221603992.png)

```python
class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat
```

- **Fusion with Temporal and Spatial Attention**

> Inter-frame temporal relation and intra-frame spatial relation are critical in fusion:
>
> - `different neighboring frames are not equally informative` due to occlusion, blurry regions and parallax problems.
> - `misalignment and unalignment` arising from the preceding alignment stage adversely affect the subsequent reconstruction performance.
>
> propose TSA fusion module to assign pixel-level aggregation weights on each frame, adopt temporal and spatial attentions during the fusion process.
>
> - temporal attention is to compute frame similarity in an embedding space. In an embedding space, a neighboring frame that is more similar to the reference one, should be paid more attention.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025212147206.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025211618576.png)

```python
class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat

```

#### Evaluation

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025213641309.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025213721167.png)

> Timofte R, Rothe R, Van Gool L. Seven ways to improve example-based single image super resolution[C] //Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 1865-1873.  [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780575&tag=1)

------

### Paper: Seven ways

<div align=center>
<br/>
<b>Seven ways to improve example-based single image super resolution</b>
</div>


#### Summary

1. present seven techniques that everybody should know to improve example-based single image supre resolution. 

#### Ways

- Augmentation of training data

> If we rotate the original images by 90,180,270, and flip them upside-down, we get images without altered content. Using an interpolation for other rotation angles can corrupt edges and impact the performance.

- large dictionary and hierarchical search

> if the dictionary size(basis of samples/anchoring points) is increased, the performance for sparse coding or anchoed methods improves, as the learned model generalizes better.

### Project

#### 1. **[image-super-resolution 3k](https://github.com/idealo/image-super-resolution)**

This project contains `Keras implementations of different Residual Dense Networks for Single Image Super-Resolution (ISR)` as well as scripts to train these networks using content and adversarial loss components.

The implemented networks include:

- The super-scaling Residual Dense Network described in [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797) (Zhang et al. 2018)
- The super-scaling Residual in Residual Dense Network described in [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) (Wang et al. 2018)
- A multi-output version of the Keras VGG19 network for deep features extraction used in the perceptual loss
- A custom discriminator network based on the one described in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGANS, Ledig et al. 2017)

Read the full documentation at: https://idealo.github.io/image-super-resolution/.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017171103641.png)

#### 2. [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) 5k

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017171647835.png)

### Survey

> Anwar S, Khan S, Barnes N. A deep journey into super-resolution: A survey[J]. ACM Computing Surveys (CSUR), 2020, 53(3): 1-34. cite 107 [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.07523.pdf)

> Liu A, Liu Y, Gu J, et al. Blind image super-resolution: A survey and beyond[J]. arXiv preprint arXiv:2107.03055, 2021. cite 2, [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2107.03055.pdf)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/superresolution/  

