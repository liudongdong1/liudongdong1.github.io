# LinearLayer


> `torch.nn.``Linear`(*in_features*, *out_features*, *bias=True*)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210529191622728.png)

```python
[docs]class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```

```python
>>> import torch
>>> nn1 = torch.nn.Linear(100, 50)
>>> input1 = torch.randn(140, 100)
>>> output1 = nn1(input1)
>>> output1.size()
torch.Size([140, 50])
#具体计算为[140,100] × [50,100]的转置 + bias = [140,100] × [100,50] + bias最后得到的维度为[140,50]。
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/linearlay/  

