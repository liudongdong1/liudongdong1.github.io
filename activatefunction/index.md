# ActivateFunction


> 神经网络中的每个神经元节点接受上一层神经元的输出值作为本神经元的输入值，并将输入值传递给下一层，输入层神经元节点会将输入属性值直接传递给下一层（隐层或输出层）。在多层神经网络中，上层节点的输出和下层节点的输入之间具有一个函数关系，这个函数称为激活函数（又称激励函数）。如果`不用激励函数`（其实相当于激励函数是f(x) = x），在这种情况下你`每一层节点的输入都是上层输出的线性函数`，很容易验证，无论你神经网络有多少层，`输出都是输入的线性组合，与没有隐藏层效果相当`，这种情况就是最原始的感知机（Perceptron）了，那么网络的逼近能力就相当有限。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524085500389.png)



![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-506351bd86c341e4bb52ebab3b1a3f66_r.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-e6d254150ab20084be23756e49b52118_r.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-1cb1c6307cba97557a20b617e38e2ef4_r.jpg)

### 0. Softmax

> softmax函数的本质就是将一个K维的任意实数向量压缩（映射）成另一个K维的实数向量，其中向量中的每个元素取值都介于（0，1）之间。经常用在神经网络的最后一层，作为输出层，进行多分类。
>
> - softmax建模使用的分布是多项式分布，而logistic则基于伯努利分布
> - softmax回归进行的多分类，类与类之间是互斥的，即一个输入只能被归为一类; 多个logistic回归进行多分类，输出的类别并不是互斥的，即"苹果"这个词语既属于"水果"类也属于"3C"类别。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524111833629.png)

### 1. Sigmoid&Logistic

> 在分类任务中，sigmoid 正逐渐被 Tanh 函数取代作为标准的激活函数，因为后者为奇函数（关于原点对称）。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524090921933.png)

###2. Sinusoid

> 如同余弦函数，`Sinusoid（或简单正弦函数）激活函数为神经网络引入了周期性`。该函数的`值域为 [-1,1]`，且`导数处处连续`。此外，Sinusoid 激活函数为零点对称的奇函数。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524092309234.png)

### 2. Sinc

> Sinc 函数（全称是 Cardinal Sine）在信号处理中尤为重要，因为它`表征了矩形函数的傅立叶变换（Fourier transform）`。作为一种激活函数，它的优势在于`处处可微和对称的特性`，不过它比较`容易产生梯度消失的问题`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524092412368.png)

### 3. Symmetrical Sigmoid

> Symmetrical Sigmoid 是另一个 Tanh 激活函数的变种（实际上，它`相当于输入减半的 Tanh`）。和 Tanh 一样，它是反对称的、零中心、可微分的，值域在 -1 到 1 之间``。它更`平坦的形状和更慢的下降派生表明它可以更有效地进行学习`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524092014307.png)

### 4. LogLog

> Log Log 激活函数（由上图 f(x) 可知该函数为以 e 为底的嵌套指数函数）的值域为 [0,1]，Complementary Log Log 激活函数`有潜力替代经典的 Sigmoid 激活函数`。该函数`饱和地更快，且零点值要高于 0.5`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524092109602.png)

### 5. softmoid

> 特点：它能够把输入的连续实值变换为`0和1之间的输出`，特别的，如果是非常大的负数，那么输出就是0；如果是非常大的正数，输出就是1.
>
> - 缺点：sigmoid函数曾经被使用的很多，不过近年来，用它的人越来越少了。
>   - 在深度神经网络中梯`度反向传递时导致梯度爆炸和梯度消失`，其中梯度爆炸发生的概率非常小，而`梯度消失发生的概率比较大`。梯度从后向前传播时，`每传递一层梯度值都会减小为原来的0.25倍`，如果神经网络隐层特别多，那么梯度在穿过多层后将变得非常小接近于0，即出现梯度消失现象；当网络`权值初始化为 ( 1 , + ∞ )区间内的值`，则会出现梯度爆炸情况。
>   - 其解析式中含有幂运算，计算机求解时相对来讲比较耗时。对于规模比较大的深度网络，这会较大地增加训练时间。
>   - Sigmoid 的 output 不是0均值（即zero-centered）.如x &gt; 0 ,   f = w T x + b x&gt 0,那么对w求局部梯度则都为正，这样在反向传播的过程中w要么都往正方向更新，要么都往负方向更新，导致有一种捆绑的效果，使得收敛缓慢。 

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524084008547.png)

### 6. tanh

> 解决了Sigmoid函数的不是zero-centered输出问题，然而，`梯度消失（gradient vanishing）的问题和幂运算的问题仍然存在`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524084539502.png)

### 7. Hard Tanh

> Hard Tanh 是 Tanh 激活函数的线性分段近似。相较而言，`它更易计算`，这使得学习计算的速度更快，尽管`首次派生值为零可能导致静默神经元/过慢的学习速率`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091427502.png)

### 8. LeCun Tanh

> 是 Tanh 激活函数的扩展版本。它具有以下几个可以改善学习的属性：f(± 1) = ±1；二阶导数在 x=1 最大化；且有效增益接近 1。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091510721.png)

### 9. ArcTan

> ArcTan 激活函数更加平坦，这让它比其他双曲线更加清晰。在默认情况下，`其输出范围在-π/2 和π/2 之间`。其`导数趋向于零的速度也更慢`，这意味着学习的效率更高。但这也意味着，`导数的计算比 Tanh 更加昂贵`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091542870.png)

###10.  SoftSign

> Softsign 是 Tanh 激活函数的另一个替代选择。就像 Tanh 一样，Softsign 是反对称、去中心、可微分，并返回-1 和 1 之间的值。其更平坦的曲线与更慢的下降导数表明它可以更`高效地学习`。另一方面，`导数的计算比 Tanh 更麻烦。`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091708235.png)

### 11. SoftPlus

> 作为` ReLU 的一个不错的替代选择`，SoftPlus 能够`返回任何大于 0 的值`。与 ReLU 不同，SoftPlus 的导数是连续的、非零的，无处不在，从而`防止出现静默神经元`。然而，SoftPlus 另一个不同于 ReLU 的地方在于其`不对称性，不以零为中心，这兴许会妨碍学习`。此外，`由于导数常常小于 1，也可能出现梯度消失的问题`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091816118.png)

### 12. Relu

> - 优点：
>   - 解决了gradient vanishing 问题（在正区间）
>   - 计算速度非常快，只需要判读输入是否大于0
>   - 收敛速度远快于sigmoid和tanh
> - 缺点：
>   - ReLU 输出的不是zero-centered
>   - 当`输入为负值的时候`，ReLU 的学习速度可能会变得很慢，甚至使神经元直接无效，因为此时输入小于零而梯度为零，从而其权重无法得到更新，在剩下的训练过程中会一直保持静默。
>   - Dead ReLU Problem，指的是某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。
>     - 非常不幸的参数初始化，这种情况比较少见
>     -  learning rate太高导致在训练过程中参数更新太大，不幸使网络进入这种状态
>     - 可以采用`Xavier初始化方法`，以及`避免将learning rate设置太大`或`使用adagrad等自动调节learning rate的算法`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524084629494.png)

### 13. Leaky Relu

> Leaky ReLU有ReLU的所有优点，外加不会有Dead ReLU问题，但是在实际操作当中，并没有完全证明Leaky ReLU总是好于ReLU。
>
> - 带泄露修正线性单元（Leaky ReLU）的输出对负值输入有很小的坡度。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524085022288.png)

### 14. PReLU

> 参数化修正线性单元（Parameteric Rectified Linear Unit，PReLU）`属于 ReLU 修正类激活函数的一员`。它`和 RReLU 以及 Leaky ReLU 有一些共同点`，即`为负值输入添加了一个线性项`。而最关键的区别是，这个线性项的斜率实际上是在模型训练中学习到的。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091056192.png)

### 15. RReLu

> 随机带泄露的修正线性单元（Randomized Leaky Rectified Linear Unit，RReLU）也属于 ReLU 修正类激活函数的一员。和 Leaky ReLU 以及 PReLU 很相似，为负值输入添加了一个线性项。而最关键的区别是，`这个线性项的斜率在每一个节点上都是随机分配的（通常服从均匀分布）。`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091157083.png)

### 16. SReLU

> S 型整流线性激活单元（S-shaped Rectified Linear Activation Unit，SReLU）属于以 ReLU 为代表的整流激活函数族。它由三个分段线性函数组成。其中`两种函数的斜度，以及函数相交的位置会在模型训练中被学习`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091318519.png)

### 17. ELU(Exponential Linear Units)

> ELU也是为解决ReLU存在的问题而提出，显然，ELU有ReLU的基本所有优点:
>
> - 不会有Dead ReLU问题 
> - 输出的均值接近0，zero-centered
> - `计算量稍大`。类似于Leaky ReLU，理论上虽然好于ReLU，但在实际使用中目前并没有好的证据ELU总是优于ReLU。
> - 和其它修正类激活函数不同的是，它包括一个`负指数项，从而防止静默神经元出现，导数收敛为零`，从而提高学习效率。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524085125447.png)

### 18. SELU

> 扩展指数线性单元（Scaled Exponential Linear Unit，SELU）是激活函数指数线性单元（ELU）的一个变种。其中`λ和α是固定数值（分别为 1.0507 和 1.6726）`。这些值背后的推论（零均值/单位方差）构成了自归一化神经网络的基础（SNN）。

### 19. **Hard-Swish 或 H-Swish**

> 几乎类似于 swish 函数，但计算成本却比 swish 更低，因为它用线性类型的 ReLU 函数取代了指数类型的 sigmoid 函数。
>
> - 在归一化时，网络层数较深，性能由于relu。

![image-20210524090502344](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524090502344.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524090302779.png)

### 20. MaxOut

> `ReLu 和 Leaky ReLu 都是它的特殊形式`，所以它有 ReLu 的优点却没有 ReLu 的缺点。坏处是它使得`参数翻倍，导致总参数量非常大`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524090431806.png)

### 21. Step

> 激活函数 Step 更倾向于理论而不是实际，它模仿了生物神经元要么全有要么全无的属性。它无法应用于神经网络，因为其导数是 0（除了零点导数无定义以外），这意味着基于梯度的优化方法并不可行。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524090635449.png)

### 22. Identity

> 过激活函数 Identity，节点的输入等于输出。它完美适合于潜在行为是线性（与线性回归相似）的任务。当存在非线性，单独使用该激活函数是不够的，但它依然可以在最终输出节点上作为激活函数用于回归任务。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524090707832.png)

###  23. Bent Identity

> 激活函数 Bent Identity 是`介于 Identity 与 ReLU 之间的一种折衷选择`。它允许非线性行为，尽管其`非零导数有效提升了学习并克服了与 ReLU 相关的静默神经元的问题。`由于其导数`可在 1 的任意一侧返回值，因此它可能容易受到梯度爆炸和消失的影响。`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524091930958.png)

### 24. Gaussian

> 高斯激活函数（Gaussian）并不是径向基函数网络（RBFN）中常用的高斯核函数，高斯激活函数在多层感知机类的模型中并不是很流行。该函数处处可微且为偶函数，但一阶导会很快收敛到零。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210524092218128.png)

### 25. [GELU](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#GELU)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210529192110046.png)

```python
class GELU(Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210529192054909.png)

### 选择建议

- 训练深度学习网络`尽量使用zero-centered数据` (可以经过数据预处理实现) 和`zero-centered输出`。所以要尽量选择输出具有zero-centered特点的激活函数`以加快模型的收敛速度`。

- 如果`使用 ReLU`，那么一定要`小心设置 learning rate`，而且要注意不要让网络出现很多 “dead” 神经元，如果这个问题不好解决，那么可以试试 Leaky ReLU、PReLU 或者 Maxout.

- `最好不要用 sigmoid，你可以试试 tanh`，不过可以预期它的效果会比不上 ReLU 和 Maxout.

- `sigmoid 只会输出正数，以及靠近0的输出变化率最大`，tanh和sigmoid不同的是，`tanh输出可以是负数`，`ReLu是输入只能大于0,如果你输入含有负数，ReLu就不适合`，如果你的输入是图片格式，ReLu就挺常用的。

- `zere-centered:` 越接近0为中心，SGD会越接近 natural gradient（一种二次优化技术），从而降低所需的迭代次数。

- 用于`分类器时`，`Sigmoid函数及其组合通常效果更好`。

- 由于`梯度消失问题`，有时要`避免使用sigmoid和tanh函数`。

- `ReLU函数是一个通用的激活函数`，目前在大多数情况下使用, `ReLU函数只能在隐藏层中使用`, **从ReLU函数开始，如果ReLU函数没有提供最优结果，再尝试其他激活函数**

- 如果神经网络中`出现死神经元`，那么`PReLU函数`就是最好的选择。

- 组合使用：

  - **ReLU + MSE**

  > `均方误差损失函数无法处理梯度消失问题`，而使用Leak ReLU激活函数能够减少计算时梯度消失的问题，因此在神经网络中如果需要使用均方误差损失函数，**一 般采用Leak ReLU等可以减少梯度消失的激活函数。**另外，由于均方误差具有普遍性，一般作为衡量损失值的标准，因此使用均方误差作为损失函数表现既不会太好也不至于太差。

  - **Sigmoid + Logistic**

  > Sigmoid函数会引起梯度消失问题：根据链式求导法，Sigmoid函数求导后由多个[0,1]范围的数进行连乘，如其导数形式为当其中一个数很小时， 连成后会无限趋近于零直至最后消失。而类Logistic损失函数求导时，加上对数后 连乘操作转化为求和操作，在`一定程度上避免了梯度消失`，**所以我们经常可以看到 Sigmoid激活函数+交叉摘损失函数的组合。**

  - **Softmax + Logisitc**

  > 在数学上，`Softmax激活函数会返回输出类的互斥概率分布`，也就是能`把离散的 输出转换为一个同分布互斥的概率`，如（0.2, 0.8)。另外`，类Logisitc损失函数是基 于概率的最大似然估计函数而来的`，因此输出概率化能够更加方便优化算法进行求 导和计算，所以我们经常可以看到输出层使用Softmax激活函数+交叉熵损失函数 的组合。


### Resouces

- https://www.jiqizhixin.com/articles/2017-10-10-3


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/activatefunction/  

