# BatchNormalization


## 1. Internal Covariate Shift

统计学习中的一个很重要的假设就是输入的分布是相对稳定的。如果这个假设不满足，则模型的收敛会很慢，甚至无法收敛。所以，对于一般的统计学习问题，在训练前将数据进行归一化或者白化（whitening）是一个很常用的trick。

但这个问题在深度神经网络中变得更加难以解决。<font color=red>在神经网络中，网络是分层的，可以把每一层视为一个单独的分类器，将一个网络看成分类器的串联。这就意味着，在训练过程中，随着某一层分类器的参数的改变，其输出的分布也会改变，这就导致下一层的输入的分布不稳定。分类器需要不断适应新的分布，这就使得模型难以收敛。对数据的预处理可以解决第一层的输入分布问题，而对于隐藏层的问题无能为力，这个问题就是**Internal Covariate Shift**。而Batch Normalization其实主要就是在解决这个问题。</font>

除此之外，一般的神经网络的梯度大小往往会与参数的大小相关（仿射变换），且随着训练的过程，会产生较大的波动,这就导致学习率不宜设置的太大。**Batch Normalization使得梯度大小相对固定，**一定程度上允许我们使用更高的学习率。

## 2. Batch Normalization

- $x_i$: mini-batch with size about N ；
- $y, \beta$ 的大小为特征长度，与 $x_i$ 相同；

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831152445.png)

> BN层通常添加在隐藏层的激活函数之前，线性变换之后。如果我们把(2.4)和之后的激活函数放在一起看，可以将他们视为一层完整的神经网络（线性+激活）。（注意BN的线性变换和一般隐藏层的线性变换仍有区别，前者是element-wise的，后者是矩阵乘法。）
>
> 此时， 可以视为这一层网络的输入，而 是拥有固定均值和方差的。这就解决了Covariate Shift.
>
> 另外， 还具有保证数据表达能力的作用。 在normalization的过程中，不可避免的会改变自身的分布，而这会导致学习到的特征的表达能力有一定程度的丢失。通过引入参数γ和β，极端情况下，网络可以将γ和β训练为原分布的标准差和均值来恢复数据的原始分布。这样保证了引入BN，不会使效果更差。
>
> 在训练过程中，还需要维护一个移动平均的均值和方差，这两个移动平均会用于推断过程。

## 3. 待完成

- [ ] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- [ ] Deriving the Gradient for the Backward Pass of Batch Normalization
- [ ] CS231n Convolutional Neural Networks for Visual Recognition



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/batchnormalization/  

