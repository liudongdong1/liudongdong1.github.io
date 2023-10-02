# Deformable_conv


> Dai J, Qi H, Xiong Y, et al. Deformable convolutional networks[C]//Proceedings of the IEEE international conference on computer vision. 2017: 764-773.  [[code 3.6k](https://github.com/msracver/Deformable-ConvNets)]

------

# Paper: Deformable Conv

<div align=center>
<br/>
<b>Deformable convolutional networks</b>
</div>

#### Summary

1. introduce two new modules to enhance the transformation modeling capability of CNNS, `deformable convolution` and `deformable RoI pooling`, based on the idea of `augmenting the spatial and learning the offsets ` from the target tasks without supervision.

#### Research Objective

  - **Application Area**:
- **Purpose**:  

#### Proble Statement

- 

previous work:

- build the training datasets with sufficient desired variations by augmenting the existing data samples. expensive training and complex model parameters.
- use transformation-invariant features and algorithm.  (SIFT and sliding window based object detection paradigm)
- **Sparial Transform Networks (STN)**: learn `spatial transformation from data`, it `warps the feature map via a gloval parametric transformation such as affine transformation.`
- **Active Convolution:** it augments the sampling locations in the convolution with offsets and learns the offsets via back-propagation end-to-end. ps: 1. it shares the offsets all over the different spatial locations. 2. the offsets are static model parameters that are learnt per task or per training.
- **Effective Receptive Field:** the pixels near the center have much larger impact, the effective receptive field only occupies a small fraction of the theoretical receptive field and ahs a Gaussian distribution.
- **Atrous convolution:** it increases a normal filter's stride to be larger than 1 an dkeeps th eoriginal weights at sparsified sampling locations.
- **Deformable Part Models:** learn the spatial deformation of object parts to maximize the classification score.
- **Spatial manipulation in RoI pooling:** spatial pyramid pooling uses hand crafted pooling regions over scales.
- **Transformation invariant features and their learning:** SIFT, ORB.
- **Combination of low level filters:** Gaussian filters and its smooth derivatives are widely used to extract low level image structures such as corners, edges, T-junction. 

#### Methods

- **system overview**:

![Dilated conv](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017154840462.png)

(a) 普通卷积，1-dilated convolution，卷积核的感受野为3×3

(b) 扩张卷积，2-dilated convolution，卷积核的感受野为7×7

(c) 扩张卷积，4-dilated convolution，卷积核的感受野为15×15

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017152143221.png)

- (a):  regular sampling grid(green points) of stardard convolution
- (b): deformed sampling locations (dark blue points) with augmented offsets (light blue arraws) in deformable convolution.
- (c)(d): deformable convolution generalizes various transformations for scale, aspect ratio and rotation.

![Deformable Convolution](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017152518659.png)

```python
res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu,
                                              num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset,
                                                         num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                         stride=(1, 1), dilate=(2, 2), no_bias=True)
```

![Deformable RoI pooling](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017152930371.png)

```python
def get_deformable_roipooling(self, name, data, rois, output_dim, spatial_scale, param_name, group_size=1, pooled_size=7,
                              sample_per_part=4, part_size=7):
    offset = mx.contrib.sym.DeformablePSROIPooling(name='offset_' + name + '_t', data=data, rois=rois, group_size=group_size, pooled_size=pooled_size,
                                                   sample_per_part=sample_per_part, no_trans=True, part_size=part_size, output_dim=output_dim,
                                                   spatial_scale=spatial_scale)
    offset = mx.sym.FullyConnected(name='offset_' + name, data=offset, num_hidden=part_size * part_size * 2, lr_mult=0.01,
                                   weight=self.shared_param_dict['offset_' + param_name + '_weight'], bias=self.shared_param_dict['offset_' + param_name + '_bias'])
    offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, part_size, part_size), name='offset_reshape_' + name)
    output = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool_' + name, data=data, rois=rois, trans=offset_reshape, group_size=group_size,
                                                   pooled_size=pooled_size, sample_per_part=sample_per_part, no_trans=False, part_size=part_size, output_dim=output_dim,
                                                   spatial_scale=spatial_scale, trans_std=0.1)
    return output
```



- `sampling using a regular grid R over the input feature map x;` the grid R defines the receptive field size and dilation.
- `summation of sampled values weighted by w.`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211017153049167.png)


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/deformable_conv/  

