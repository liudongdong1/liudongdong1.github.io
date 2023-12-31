# LossFunction


### 1. logSoftmax

> *CLASS*`torch.nn.``LogSoftmax`(*dim=None*)
>
> - Input: (*)(∗) where * means, any number of additional dimensions
> - Output: (*)(∗) , same shape as the input
> - **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – A dimension along which LogSoftmax will be computed.
>   - 参数dim=1表示对每一行求softmax，那么每一行的值加起来都等于1。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210531074733359.png)

```python
class LogSoftmax(Module):
    """
        >>> m = nn.LogSoftmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)
```

### 2. [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html?highlight=nllloss#torch.nn.NLLLoss)

> *CLASS*`torch.nn.``NLLLoss`(*weight=None*, *size_average=None*, *ignore_index=-100*, *reduce=None*, *reduction='mean'*)
>
> -  `weight` should be a 1D Tensor assigning weight to each of the classes. 
> - input has to be a Tensor of size either (minibatch, C) or (minibatch, C, d_1, d_2, ..., d_K) with K≥1 for the K-dimensional case
> - The target that this loss expects should be a class index in the range [0, C-1][0,*C*−1] where C = number of classes
> - nn.NLLLoss的输入target是类别值，并不是one-hot编码格式

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210531075700185.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210531080334224.png)

> Softmax计算出来的值范围在[0, 1]，`值的含义表示对应类别的概率`，也就是说，每行（代表每张图）中最接近于1的值对应的类别，就是该图片概率最大的类别，那么`经过log求值取绝对值之后，就是最接近于0的值`，如果此时`每行中的最小值对应的类别值与Target中的类别值相同`，那么`每行中的最小值求和取平均就是最小`，极端的情况就是0。总结一下就是，`input的预测值与Target的值越接近，NLLLoss求出来的值就越接近于0`，这不正是损失值的本意所在吗，所以NLLLoss可以用来求损失值。

```python
>>> m = nn.LogSoftmax(dim=1)
>>> loss = nn.NLLLoss()
>>> # input is of size N x C = 3 x 5
>>> input = torch.randn(3, 5, requires_grad=True)
>>> # each element in target has to have 0 <= value < C
>>> target = torch.tensor([1, 0, 4])
>>> output = loss(m(input), target)
>>> output.backward()
>>>
>>>
>>> # 2D loss example (used, for example, with image inputs)
>>> N, C = 5, 4
>>> loss = nn.NLLLoss()
>>> # input is of size N x C x height x width
>>> data = torch.randn(N, 16, 10, 10)
>>> conv = nn.Conv2d(16, C, (3, 3))
>>> m = nn.LogSoftmax(dim=1)
>>> # each element in target has to have 0 <= value < C
>>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
>>> output = loss(m(conv(data)), target)
>>> output.backward()
```

### 3. CrossEntropyLoss

> *CLASS*`torch.nn.``CrossEntropyLoss`(*weight=None*, *size_average=None*, *ignore_index=-100*, *reduce=None*, *reduction='mean'*)
>
> -  `weight` should be a 1D Tensor assigning weight to each of the classes.
> -  input is expected to contain raw, unnormalized scores for each class., input has to be a Tensor of size either (minibatch, C) or (minibatch, C, d_1, d_2, ..., d_K)with  K*≥1 for the K-dimensional case (described later)
> - a class index in the range [0, C-1][0,*C*−1] as the target for each value of a 1D tensor of size minibatch;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210531080926887.png)

### 4. One-Hot 编码orNot

```python
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F

# logsoft-max + NLLLoss
m = nn.LogSoftmax()
loss = nn.NLLLoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
print('logsoftmax + nllloss output is {}'.format(output))
# crossentripyloss
loss = nn.CrossEntropyLoss()
# input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(input, target)
print('crossentropy output is {}'.format(output))
# one hot label loss
C = 5
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
print('target is {}'.format(target))
N = target .size(0)
# N 是batch-size大小
# C is the number of classes.
labels = torch.full(size=(N, C), fill_value=0)
print('labels shape is {}'.format(labels.shape))
labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)  #one-hot 编码
print('labels is {}'.format(labels))

log_prob = torch.nn.functional.log_softmax(input, dim=1)
loss = -torch.sum(log_prob * labels) / N
print('N is {}'.format(N))
print('one-hot loss is {}'.format(loss))
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/lossfunction/  

