# 炼丹


### 1. 学习率

>越大的batch-size使用越大的学习率。原理很简单，越大的`batch-size`意味着我们学习的时候，收敛方向的`confidence`越大，我们前进的方向更加坚定，而小的`batch-size`则显得比较杂乱，毫无规律性，因为相比批次大的时候，批次小的情况下无法照顾到更多的情况，所以需要小的学习率来保证不至于出错。可以看下图`损失Loss`与`学习率Lr`的关系：

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210906215354829.png)

### 2. 零碎点

>- 使用时torch.utils.data.DataLoader，请设置num_workers > 0，而不是默认值0，和pin_memory=True，而不是默认值False。在选择worker数量时，建议将设置为可用GPU数量的`四倍`。
>
>- 使用GPU内存允许的最大批处理量可以加快训练速度, 一般来说，将批量大小增加一倍，学习率也提高一倍。
>- 使用自动混合精度：AMP, 在NVIDIA V100 GPU上对一些常见的语言和视觉模型进行基准测试时，使用AMP要比常规的FP32训练的速度提升2倍，最高可提升5.5倍。
>- 你的模型架构保持固定，输入大小保持不变，则可以`设置torch.backends.cudnn.benchmark = True，启动 cudNN 自动调整器。`
>- 注意要经常使用tensor.cpu()将tensors从GPU传输到CPU，.item()和.numpy()也是如此，使用.detach()代替。
>- torch.tensor() 总是复制数据。如果你有一个要转换的 numpy 数组，使用 torch.as_tensor() 或 torch.from_numpy() 来避免复制数据。
>- `在验证期间设置torch.no_grad() `
>- 只有没有预训练模型的领域会自己初始化权重，或者在模型中去初始化神经网络最后那几个全连接层的权重。常用的权重初始化算法是 **「kaiming_normal」** 或者 **「xavier_normal」**
>- `Dropout`一般适合于`全连接层部分`，而卷积层由于其参数并不是很多，所以不需要dropout，加上的话对模型的泛化能力并没有太大的影响。一般在网络的最开始和结束的时候使用全连接层，而hidden layers则是网络中的卷积层。所以一般情况，`在全连接层部分，采用较大概率的dropout`而`在卷积层采用低概率或者不采用dropout。`

### 3. 融合Ensemble

>1. model1 probs + model2 probs + model3 probs ==> final label
>2. model1 label , model2 label , model3 label ==> voting ==> final label
>3. model1_1 probs + ... + model1_n probs ==> mode1 label, model2 label与model3获取的label方式与1相同 ==> voting ==> final label

### 4. **差分学习率与迁移学习**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210906220237413.png)

>随着层数的增加，神经网络学习到的特征越抽象。因此，下图中的卷积层和全连接层的学习率也应该设置的不一样，一般来说，卷积层设置的学习率应该更低一些，而全连接层的学习率可以适当提高。
>
>大多数采用的优化算法还是是adam和SGD+monmentum。大多数采用的优化算法还是是adam和SGD+monmentum。
>
>- 超参上，learning rate 最重要，推荐了解 cosine learning rate 和 cyclic learning rate，其次是 batchsize 和 weight decay。当你的模型还不错的时候，可以试着做数据增广和改损失函数锦上添花了。
>- 如果你的`模型包含全连接层（MLP）`，并且`输入和输出大小一样`，可以考虑将MLP替换成`Highway Network`,我尝试对结果有一点提升，建议作为最后提升模型的手段，原理很简单，就是给输出加了一个gate来控制信息的流动，详细介绍请参考论文: http://arxiv.org/abs/1505.00387[17]
>- rnn的dim和embdding size,`一般从128上下开始调整`. batch size,一般从128左右开始调整. batch size合适最重要,并不是越大越好.
>- dropout的位置比较有讲究, 对于RNN,建议放到输入->RNN与RNN->输出的位置.关于RNN如何用dropout,可以参考这篇论文:http://arxiv.org/abs/1409.2329[15], dropout 对小数据防止过拟合有很好效果，值一般设为0.5

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210906220852768.png)









---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E7%82%BC%E4%B8%B9/  

