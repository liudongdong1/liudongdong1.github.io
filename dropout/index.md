# Dropout


> `torch.nn.``Dropout`(*p=0.5*, *inplace=False*)
>
> During training, `randomly zeroes` some of the elements of the input tensor with probability `p` using samples from a Bernoulli distribution`.
>
> -  proven to be an` effective technique for regularization and preventing the co-adaptation of neurons` as described in the paper [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) .

```python
[docs]class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/dropout/  

