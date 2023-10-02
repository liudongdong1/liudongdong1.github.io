# Layernorm


> - BN，LN，IN，GN从学术化上解释差异：
> - **BatchNorm**：**batch方向做归一化**，算NHW的均值，对小batchsize效果不好；BN主要缺点是**对batchsize的大小比较敏感**，由于每次计算均值和方差是在一个batch上，所以**如果batchsize太小，则计算的均值、方差不足以代表整个数据分布**
> - **LayerNorm**：**channel方向做归一化**，算CHW的均值，主要**对RNN作用明显；**
> - **InstanceNorm**：**一个channel内做归一化**，算H*W的均值，用在**风格化迁移**；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。**可以加速模型收敛，并且保持每个图像实例之间的独立**。
> - **GroupNorm：**将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值；这样与batchsize无关，不受其约束。
> - **SwitchableNorm**是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210531082356191.png)

```python
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

> num_features： 来自期望输入的特征数，该期望输入的大小为’batch_size x num_features [x width]’
> eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
> momentum： 动态均值和动态方差所使用的动量。默认为0.1。
> affine： 布尔值，当设为true，给该层添加可学习的仿射变换参数。
> track_running_stats：布尔值，当设为true，记录训练过程中的均值和方差；

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210529193401433.png)

```python
class LayerNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...] 
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/layernorm/  

