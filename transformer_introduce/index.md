# Transformer_Introduce


> The Transformer starts by generating initial representations, or embeddings, for each word. These are represented by the unfilled circles. Then, using self-attention, it aggregates information from all of the other words, generating a new representation per word informed by the entire context, represented by the filled balls. This step is then repeated multiple times in parallel for all words, successively generating new representations.

- 代码讲解地址：http://nlp.seas.harvard.edu/2018/04/03/attention.html

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715201421532.png)

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715172733519.png" style="zoom:50%;" />

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715173110289.png" alt="image-20200715173110289" style="zoom:50%;" />

## 1. Embedding

> After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.
>
> The word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715173416418.png" style="zoom:50%;" />

## 2. **Encode**

> an encoder receives a list of vectors as input. It processes this list by passing these vectors into a ‘self-attention’ layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.

- **Self-Attention**

  - create vectors from each of the encoder’s input vectors (in this case, the embedding of each word). <font color=red>For each word, we create a Query vector, a Key vector, and a Value vector.</font>

  <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715174741652.png" style="zoom:50%;" />

  - calculating self-attention is to calculate a score <font color=red>这一步具体是怎么实现的</font>

  <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715175054672.png" style="zoom:67%;" />

  > Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.
  - **third and forth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.

  <img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200831200422.png" style="zoom:67%;" />

  - **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).
  - **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

## 3. **Matrix Calculation of Self-Attention**

- **the first step** is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715175943680.png" alt="Every row in the X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure)" style="zoom:67%;" />

-  condense steps two through six in one formula to calculate the outputs of the self-attention layer.

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715180124848.png" alt="The self-attention calculation in matrix form" style="zoom:67%;" />



> RNNs maintain a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.

## 4. The Beast With Many Heads

- **“multi-headed” attention**

> 1. <font color=red>It expands the model’s ability to focus on different positions.</font> Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, we would want to know which word “it” refers to.
> 2. <font color=red>It gives the attention layer multiple “representation subspaces”.</font> As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715180348165.png" style="zoom:67%;" />

> If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715180613767.png)

- concat the matrices then multiple them by an additional weights matrix WO.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715180644829.png)

- Multi-Headed self-attention visualization

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715180725827.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715180900704.png)

> As we encode the word "it", <font color=red>one attention head</font> is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715181015153.png)

## 5. Representing The Order of The Sequence Using Positional Encoding

> helps it determine the position of each word, or the distance between different words in the sequence. 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715181211222.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715181250522.png)

> In the following figure, each row corresponds the a positional encoding of a vector. So the first row would be the vector we’d add to the embedding of the first word in an input sequence. Each row contains 512 values – each with a value between 1 and -1. We’ve color-coded them so the pattern is visible.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715181350421.png)

## 6. Residuals

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715181554087.png)

![Transformer of 2 stacked encoders and decoders](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715181715394.png)

## 7. Decoder Side

![transformer_decoding_1](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/transformer_decoding_1.gif)

> The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

![transformer_decoding_2](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/transformer_decoding_2.gif)

## 8. Final Linear and Softmax Layer

> The decoder stack outputs a vector of floats.The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715182513674.png)

## 9.Go Forth And Transform

- Watch [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details
- Play with the [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
- Explore the [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

Follow-up works:

- [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
- [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
- [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
- [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
- [Image Transformer](https://arxiv.org/abs/1802.05751)
- [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

转载学习于：

- https://jalammar.github.io/illustrated-transformer/
- 视频介绍：https://www.youtube.com/watch?v=rBCqOTEfxvg

**author**: Google Brain; Google Research
**date**: 2017
**keyword**:

- model

> Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems*. 2017.  cited by 11535

------

# Paper: Attention

<div align=center>
<br/>
<b>Attention Is All You Need</b>
</div>

#### Summary

1. propose the transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.
2. the Transformer allows for significantly more parallelization and can reach a new state o fthe art in translation quality.

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715201421532.png)

**【Module One】 Encoder and Decoder Stacks**

- **Encoder**

> - composed of a stack of N=6 identical layers
> - each layer has multi-head self-attention mechanism, and position-wise fully connected feed-forward network.
> - employ a residual connection around each of the two sub-layers, followed by layer normalization.

- **Decoder**

> - composed of stack of N=6 identical layers;
> - the decoder also inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.
> - modify the self-attention sub-layer in the decoder stack to pervent positions from attending to subsequent positions, called masking, ensures that the predictions for position i can depend only on the knownn outputs at positions less than i.

**【Attention】**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200901102523.png)

- **Scaled Dot-Product Attention**

$$
Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
$$

- **Multi-Head Attention**

$$
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^o\\
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)\\
W_i^Q \epsilon R^{d_{model}*d_k}\\
W^o \epsilon R^{hd_v*d_{model}}
$$

**【Application of Attention 】**

- allow every position in the decoder to attend over all positions in the input sequence;
- each position in the encoder can attend to all positions in the previous layer of the encoder;

**【Position-wise Feed-Forward Networks 】**
$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/transformer_introduce/  

