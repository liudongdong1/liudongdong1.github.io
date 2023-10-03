# GRU_LSTM


## 0. [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210601233048262.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601233227000.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601233315122.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601233342773.png)

### .2. conditional RNN

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601234205991.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210601232942714.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715152425359.png)
$$
h_t=f(x_t,h_{t-1})\\
h_t:=tanh(W_{xh}x_t+W_{hh}h_{t-1})
$$

- 计算目标：反向传播时，损失函数$l$ 对$t$ 时刻隐含状态向量$h_t$的偏导。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715153026772.png)

![奇异值分解](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715153116147.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20200715153213869.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715152805715.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715153248257.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715153319948.png)

## 1. GRU

> GRU（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。

【输入输出结构】

![GRU 输入输出结构](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715150335901.png)

【内部结构】

- r： 控制重置门控；
-  z： 为控制更新门控；门控信号越接近1，代表”记忆“下来的数据越多；而越接近0则代表”遗忘“的越多。

![r、z门控](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715150451699.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715150732447.png)

![GRU内部结构图](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715150823373.png)

【更新表达式】
$$
h^t=(1-z)\Theta h^{t-1}+z\Theta h'
$$

> - $(1-z)\Theta h^{t-1}$：表示对原本隐藏状态的选择性“遗忘”。这里的 $1-z$可以想象成遗忘门（forget gate），忘记 $h^{t-1}$维度中一些不重要的信息。
> - $z\Theta h^{t-1}$ ： 表示对包含当前节点信息的 $h'$进行选择性”记忆“。与上面类似，这里的 $(1-z)$ 同理会忘记  $h'$维度中的一些不重要的信息。或者，这里我们更应当看做是对 $h'$维度中的某些信息进行选择。
> - $h^t=(1-z)\Theta h^{t-1}+z\Theta h'$ ：结合上述，这一步的操作就是忘记传递下来的 $h^{t-1}$ 中的某些维度信息，并加入当前节点输入的某些维度信息。

### .1. [pytorch API](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#torch.nn.GRU)

> *CLASS*`torch.nn.``GRU`(**args*, ***kwargs*)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601232328315.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601232437633.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601232131136.png)

```python
>>> rnn = nn.GRU(10, 20, 2)
>>> input = torch.randn(5, 3, 10)
>>> h0 = torch.randn(2, 3, 20)
>>> output, hn = rnn(input, h0)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601232030199.png)

学习于：https://zhuanlan.zhihu.com/p/32481747

## 2. LSTM

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200715152100266.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210601111032053.png)

> *CLASS*`torch.nn.``LSTM`(**args*, ***kwargs*)
>
> - **input_size** – The `number` of expected `features in the input x`
> - **hidden_size** – The `number of features in the hidden state h`
> - **num_layers** – `Number of recurrent layers`. E.g., setting `num_layers=2` would mean stacking two LSTMs together to form a stacked LSTM, with the `second LSTM taking in outputs of the first LSTM` and computing the final results. Default: 1
> - **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
> - **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False`
> - **dropout** – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`. Default: 0
> - **bidirectional** – If `True`, becomes a bidirectional LSTM. Default: `False`
> - **proj_size** – If `> 0`, will use LSTM with projections of corresponding size. Default: 0

> - **input** of shape` (seq_len, batch, input_size):` tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See [`torch.nn.utils.rnn.pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) or [`torch.nn.utils.rnn.pack_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence) for details.
>
> - **h_0** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1. If `proj_size > 0` was specified, the shape has to be (num_layers * num_directions, batch, proj_size).
>
> - **c_0** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
>
>   `If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero`.

> Outputs: output, (h_n, c_n)
>
> - **output** of shape `(seq_len, batch, num_directions * hidden_size): `tensor containing the output features (h_t) from the last layer of the LSTM, for each t. If a [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) has been given as the input, the output will also be a packed sequence. If `proj_size > 0` was specified, output shape will be (seq_len, batch, num_directions * proj_size).
>
>   For the unpacked case, the directions can be separated using `output.view(seq_len, batch, num_directions, hidden_size)`, with forward and backward being direction 0 and 1 respectively. Similarly, the directions can be separated in the packed case.
>
> - **h_n** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len. If `proj_size > 0` was specified, `h_n` shape will be (num_layers * num_directions, batch, proj_size).
>
>   Like *output*, the layers can be separated using `h_n.view(num_layers, num_directions, batch, hidden_size)` and similarly for *c_n*.
>
> - **c_n** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len.

## 3.LSTM_paper

### Paper《Long Short-Term Memory RNN Architectures for Large Scale Acoustic Modeling》

**Note**:

1.  first distribute training of LSTM RNNs using asynchronous stochastic gradient descent optimization on a large cluster of machine.
2.  <font color=red>speech database TIMIT</font>   ，and the test it on a large vocabulary speech recognition task Google Voice Search Task.
3.  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175908610.png)
4.  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175932338.png)
5.  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107180015999.png)
6.  <font color="red">how to calculate the total number of parameters and the computational complexity with a moderate number of inputs.</font>
7.  <font color=red>Eigen matrix library</font> c++ 矩阵计算库

### Paper《Convolutional,Long Short-Term Memory Fully Connected Deep Neural Networks》

**Note**:

1. ​     CNNs are good at reducing frequency variations, LSTMs are good at temporal modeling, and DNNs are appropriate for mapping features to a more separable space.
2. take advantage of the complementarity of CNNs,LSTMs,DNNs by combining them into one unified architecture,and proposed architecture CLDNN on a variety of large vocabulary tasks.[LVCSR]
3. previous paper train the three models separately and then the ouput were combined through a combination layer.In this we train in a unified structure.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107164140509.png)

### Paper《Improved Semantic Representations from Tree-Structured LSTM》

**Note**:                                         author: Kai Sheng Tai        stanford.edu

1. <font color="red">models where real valued vectors are used to represent meaning fall into three class:1.Bag-of-Word  2.sequence models  3. tree structured modes</font>

2. Test on two task: semantic relatedness prediction on sentence pairs    sentiment classification of sentences drawn from movie reviews  

3. <font color="red">available code: https://github.com/stanfordnlp/treelsm </font> ,project : https://nlp.stanford.edu/projects/glove/ 

4. previous work: 

   1.  a problem with RNNs with transition functions of this form is that during training components of the gradient vector can gow or decay exponentially over long sequences.<font color="red">exploding or vanishing gradients</font> make it difficult for the RNN to learn long-distance correlations in a sequence.
   2.  Bidirectional LSTM (stacked LSTM) allow the hidden state to capture both past and future information,Multilayer LSTM( deep LSTM) let the higher layers capture longer-term dependencies of the input sequences.  <font color=red>they only allow for strictly sequential information propagation</font>

5. Datastructure:

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107170547474.png)

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175047377.png)

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175004057.png)

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175029530.png)

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175216506.png)

   ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107175316285.png)

   

   1. different from LSTM is that gating vectors and memory cell updates are dependent on the states of possibly many child units. Tree-LSTM unit contains one forget gate Fjk for each child k. to selective incorporate information from each child.
   2. <font color=red>Classification model:</font>![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191107181300853.png)
   3. <font color=red>Semantic Relatedness of Sentence Pairs</font> : given a sentence pair,predict a real-valued similarity score in some range.



### Paper《Learning to Forget:Continual Prediction with LSTM》 

**cited:**                                                                                **keyword:** 

#### Phenomenon&Challenge:

1. backpropagated error quickly either vanishes or blows up

#### Chart&Analyse:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566270826264.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296477248.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296655723.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296724323.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296763577.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296840233.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296878693.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566296938089.png)

#### Code:

#### Shortcoming&Confusion:

1. embedded Reber  grammar
2. 没怎么看懂公式推导过程



### Paper《**SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS**》 

**cited:**                                                                                **keyword:** 

#### Chart&Analyse:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566298936435.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566298989314.png)

Ft表示遗忘门限，It表示输入门限， ̃Ct表示前一时刻cell状态、Ct表示cell状态（这里就是循环发生的地方），Ot表示输出门限，Ht表示当前单元的输出，Ht-1表示前一时刻单元的输出

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566298945343.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566298976892.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566299008061.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1566299037440.png)







---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/gru_lstm/  

