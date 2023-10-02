# Embedding


### 1. [Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

> *CLASS*`torch.nn.``Embedding`(*num_embeddings*, *embedding_dim*, *padding_idx=None*, *max_norm=None*, *norm_type=2.0*, *scale_grad_by_freq=False*, *sparse=False*, *_weight=None*)
>
> - **num_embeddings** ([*int*](https://docs.python.org/3/library/functions.html#int)) – size of the dictionary of embeddings，nn.embedding的输入只能是编号，不能是隐藏变量，比如one-hot，或者其它。
> - **embedding_dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – the size of each embedding vector，
>   - 如果你`指定了padding_idx`，注意这个padding_idx也是在num_embeddings尺寸内的，比如符号总共有500个，指定了padding_idx，那么num_embeddings应该为501。
>   - embedding_dim的`选择`要注意，根据自己的符号数量，举个例子，如果你的词典尺寸是1024，那么极限压缩（用二进制表示）也需要10维，再考虑词性之间的相关性，怎么也要在15-20维左右，虽然embedding是用来降维的，但是>- 也要注意这种极限维度，结合实际情况，合理定义
> - **padding_idx** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding vector at `padding_idx` will default to all zeros, but can be updated to another value to be used as the padding vector.
> - **Embedding.weight** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from \mathcal{N}(0, 1)N(0,1)
>
> 作为训练的一层，随模型训练得到适合的词向量。

> A simple `lookup table` that stores` embeddings of a fixed dictionary and size`.
>
> This module is often used to `store word embeddings` and` retrieve them using indices`. The input to the module is a list of indices, and the output is the corresponding word embeddings.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601101544902.png)

```python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/embedding/  

