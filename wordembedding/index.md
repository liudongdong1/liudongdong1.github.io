# WordEmbedding


> **TEXT processing** deals with humongous amount of text to perform different range of tasks like clustering in the g    oogle search example, classification in the second and Machine Translation. How to create a representation for words that capture their *meanings*, *semantic relationships* and the different types of contexts they are used in.

> - 作为 Embedding 层嵌入到深度模型中，实现将高维稀疏特征到低维稠密特征的转换（如 Wide&Deep、DeepFM 等模型）；
> - 作为预训练的 Embedding 特征向量，与其他特征向量拼接后，一同作为深度学习模型输入进行训练（如 FNN）；
> - 在召回层中，通过计算用户和物品的 Embedding 向量相似度，作为召回策略（比 Youtube 推荐模型等）；
> - 实时计算用户和物品的 Embedding 向量，并将其作为实时特征输入到深度学习模型中（比 Airbnb 的 embedding 应用）。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721081514640.png)

## 1. Item Embedding

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721081648881.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721083502813.png)

## 2. Img Embedding

> 图片作为文章的门面特征，对推荐也很重要，可以通过 resnet 得到图片的向量，还可以通过 image caption  得到对一张图片的中文描述，对于娱乐类的新闻，还可以利用 facenet 识别出组图中，哪一张包含明星，对于动漫类类的新闻可以利用 OCR 识别出漫画里的文字，对于年龄，性别有明显倾向的场景还可以利用 resnet 改变图片的风格。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721084331320.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721084622194.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721084850357.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200721090031252.png)

## 3. Methods 

### 3.1. Frequency based Embedding

- **Count Vector:** using top numbers words based on frequency and then prepare a dictionary.

> Consider a Corpus C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens will form our dictionary and the size of the Count Vector matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200716234329746.png)

Prediction based Embedding

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200716234619560.png)

- **TF-IDF vectorization:**takes into account not just the occurrence of a word in a single document but in the entire corpus. (1）搜索引擎；（2）关键词提取；（3）文本相似性；（4）文本摘要

  - TF词频

  $$
  tf_{ij}=n_{i,j}/\sum_kn_{k,j}\\
  $$

  $n_{i,j}$ 是该词再文件$d_j$ 中出现的次数，分母是文件$d_j$中所有词汇出现的次数和。

  - IDF逆向文件频率：某一特定词语的IDF，可以由**总文件数目除以包含该词语的文件的数目**，**再将得到的商取对数得到**。

  $$
  idf_i=log|D|/|{j:t_i\epsilon d_j}|
  $$

  |D|是预料库中文件总数，$|{j:t_i\epsilon d_j}|$ 表示包含词语$t_i$ 的文件数目。
  $$
  TF-IDF=TF*IDF
  $$

- #### Co-Occurrence Matrix with a fixed context window

  > Similar words tend to occur together and will have similar context for example .Co-occurrence matrix is decomposed using techniques like PCA, SVD etc. into factors and combination of these factors forms the word vector representation.

  - Co-occurrence – For a given corpus, the co-occurrence of a pair of words say w1 and w2 is the number of times they have appeared together in a Context Window.
  - Context Window – Context window is specified by a number and the direction. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200717101404690.png)

- Advantage:
  - `preserves the semantic relationship` between words,
  - uses SVD at its core, producing more accurate word vector representations.
  - uses factorization which is a well-defined problem and can be efficiently solved.
  - `computed once and can be use anytime once computed`.
- **Disadvantages:** requires huge memory to store the co-occurrence matrix.

### 3.2. Bag of Words(BOW)

> 1. Tokenize the text into sentences
> 2. Tokenize sentences into words
> 3. Remove punctuation or stop words
> 4. Convert the words to lower text
> 5. `Create the frequency distribution of words`

```python
#Creating frequency distribution of words using nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
text="""Achievers are not afraid of Challenges, rather they relish them, thrive in them, use them. Challenges makes is stronger.
        Challenges makes us uncomfortable. If you get comfortable with uncomfort then you will grow. Challenge the challenge """
#Tokenize the sentences from the text corpus
tokenized_text=sent_tokenize(text)
#using CountVectorizer and removing stopwords in english language
cv1= CountVectorizer(lowercase=True,stop_words='english')
#fitting the tonized senetnecs to the countvectorizer
text_counts=cv1.fit_transform(tokenized_text)
# printing the vocabulary and the frequency distribution pf vocabulary in tokinzed sentences
print(cv1.vocabulary_)
print(text_counts.toarray())
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201024214303559.png)

> each document is represented as a word-count vector, these counts can be `binary counts or absolute counts`, but the size equal to the size(Voc);
>
> - huge amount of weights;
> - computationally intensive;
> - `lack of meaningful relations and no consideration for order of words`;

### 3.3. 静态向量 

>  `translate large sparse vectors into a lower-dimensional space that preserves semantic relationships`.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201024215025229.png)

#### 3.3.1.  Latent Semantic Analysis (LSA)

#### 3.3.2.  Latent Dirichlet Allocation (LDA)

#### 3.3.3. PCA

#### 3.3.4. Word2Vector(CBOW, Skip-ngram)

In paper:"**Efficient Estimation of Word Representations in Vector Space**":  the training complexity is proportional to $O=E*T*Q$; E is the epoch; T is the number of words in the training set, and Q is the defined further for each model architecture;

- **NNLM(Feedforward Neural Net Language Model):** consists of input, projection, hidden and output layers; N previous words are encoded using 1-of-V coding, V: the vocabulary size; $N*D:$ the projection layer dimensionality; H: the hidden  size; the computational complexity per each example: $Q=N*D+N* D* H+H* V$

- **Recurrent Neural Net Language Model(RNNLM):** the word representations D have the same dimensionality as the hidden layer H, $Q=H*H+H*V$;

- **New Log-linear Models:**  neural network language model can be successfully trained in two steps: first `continuous word vectors are learned using simple model`, and then `N-gram NNLM is trained on the top of these distributed representations of words`;

  - **`Continuous Bag-of-Words Models`:** similar to the feed forward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words, all words projected into the same position(their vectors are averaged), the order of words in the history does not influence the projection. $Q=N*D+D*log_2(V)$;

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025125100209.png)

  - **`Continuous Skip-gram Model`:** instead of predicting the current word based on the context, it tries to maximize classification of a word based on another word in the same sentence. In other words: use each current word as an input to a log-linear classifier with continuous projection layer and predict words within a certain range before and after the current word; $Q=C*(D+D*log_2(V))$; C: the maximum distance of the words;

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025123844292.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025123923520.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025124909812.png)

  

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025122012098.png)

  > New model architectures. The CBOW architecture predicts the current word based on the context, and the Skip-gram predicts surrounding words given the current word. 
  >
  > - support algebraic operations with the vector representation of words, Vector=vector("biggest")-vector("big")+vector("small")=vector("smallest")
  > - when train high dimensional word vectors on large amount data, the resulting vectors can be used to answer very subtle semantic relationships between words, which is good for machine translation, information retrieval and question answering systems;

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025122757936.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025122930252.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025123038373.png)

#### 3.3.5. GloVe Embedding

> In paper: "GloVe: Global Vectors for Word Representation": 
>
> - analyze and make explicit the model properties needed for such regularities to emerge in word vectors, and propose a new global log-bilinear regression model that combines the advantages of the two major model families in the literature: `global matrix factorization and local context window methods`;
> - efficiently leverages statistical information by training only on the nonzero elements in a `word-word co-occurence matrix`;
> - open source: http://nlp.stanford.edu/projects/glove/.

- $X$:  the matrix of word-word co-occurrence counts, $X_{ij}$: the number of times word j occurs in the context of word i;
- $X_i=\sum_k X_{ik}$ : the number of times any word appears in the context of word i;
- $P_{ij}=P(j|i)=x_{ij}/X_i$: the probability that word j appear in the context of word i;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025131020291.png)

> The ratio is better able to distinguish relevant words (solid and gas) from irrelevant words (water and fashion) and it is also better able to discriminate between the two relevant words.

- $F(w_i, w_j, w_k)=P_{ik}/P_{jk}$:   w: the word vector; $w_k$: the separate context word vectors;
- loss function:  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025134201555.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025131931244.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025132137995.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025132259638.png)

**改良 SkipGram**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025132437364.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201025132513525.png)

> - word2vec是局部语料库训练的，其特征提取是基于滑窗的；而glove的滑窗是为了构建co-occurance matrix，是基于全局语料的，可见glove需要事先统计共现概率；因此，word2vec可以进行在线学习，glove则需要统计固定语料信息。
> - word2vec是无监督学习，同样由于不需要人工标注；glove通常被认为是无监督学习，但实际上glove还是有label的，即共现次数![[公式]](https://www.zhihu.com/equation?tex=log%28X_%7Bij%7D%29)。
> - word2vec损失函数实质上是带权重的交叉熵，权重固定；glove的损失函数是最小平方损失函数，权重可以做映射变换。
> - 总体来看，**glove可以被看作是更换了目标函数和权重函数的全局word2vec**。

#### 3.3.6. FastText

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126120608705.png)

### 3.4.  动态向量

> 由于静态向量表示中每个词被表示成一个固定的向量，无法有效解决一词多义的问题。在动态向量表示中，模型不再是向量对应关系，而是一个训练好的模型。

#### 3.4.1. ELMo

- (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy);
- Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pretrained on a large text corpus.
- ELMo（Embeddings from Language Models）是2018年3月发表，获得了NAACL18的Best Paper

> 1. 预训练biLM模型，通常由两层bi-LSTM组成，之间用residual connection连接起来。
> 2. 在任务语料上fine tuning上一步得到的biLM模型，这里可以看做是biLM的domain transfer。
> 3. `利用ELMo提取word embedding`，将word embedding`作为输入`来对任务进行训练。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126120802633.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126121055516.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126121352272.png)

#### 3.4.2. GPT

> GPT-1（Generative Pre-Training）是OpenAI在2018年提出的，采用pre-training和fine-tuning的下游统一框架，将预训练和finetune的结构进行了统一，解决了之前两者分离的使用的不确定性（例如ELMo）。此外，GPT使用了Transformer结构克服了LSTM不能捕获远距离信息的缺点。GPT主要分为两个阶段：pre-training和fine-tuning.

- **Pre-training**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126132635179.png)

- **Fine-tuning** :采用无监督学习预训练好模型后后，可以把模型模型迁移到新的任务中，并根据新任务来调整模型的参数。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126132849556.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126132916870.png)

#### 3.4.3. BERT(Bidirectional Encoder Representations from Transformers)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126133039218.png)

> BERT进一步增强了词向的型泛化能力，充分描述字符级、词级、句子级甚至句间的关系特征。BERT的输入的编码向量（长度为512）是3种Embedding特征element-wise和.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126133152652.png)

- **Input Features:**
  - Token Embedding (WordPiece)：将`单词划分成一组有限的公共词单元`，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。如图中的“playing”被拆分成了“play”和“ing”；
  - Segment Embedding：用于`区分两个句子`，如B是否是A的下文（对话场景，问答场景等）。对于`句子对，第一个句子的特征值是0，第二个句子的特征值是1`；
  - Position Embedding：`将单词的位置信息编码成特征向量`，Position embedding能有效将单词的位置关系引入到模型中，提升模型对句子理解能力；
  - `[CLS]`表示该特征用于分类模型，对非分类模型，该符合可以省去。`[SEP]`表示分句符号，用于断开输入语料中的两个句子。

- **Masked Language Model(MLM)**

> 是指在`训练时随机从输入语料中mask掉一些单词`，然后通过该词上下文来预测它（非常像让模型来做完形填空）。80%`概率直接替换为`[MASK]； `10%`概率替换为其他任意Token； `10%`概率保留为原始Token 。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126133616522.png)

- **Next Sentence Predictions（NSP）**

> BERT采用NSP任务来增强模型对句子关系的理解，即给出两个句子A、B，模型预测B是否是A的下一句。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126133832576.png)

- **Fine-tuning**
  - 句对关系判断：第一个起始符号[CLS]经过编码后，增加Softmax层，即可用于分类；
  - 单句分类任务：实现同“句对关系判断”；
  - `问答类任务`：问答系统输入文本序列的question和包含answer的段落，并在序列中标记answer，让BERT模型学习标记answer开始和结束的向量来训练模型；
  - 序列标准任务：识别系统输入标记好实体类别（人、组织、位置、其他无名实体）文本序列进行微调训练，识别实体类别时，将序列的每个Token向量送到预测NER标签的分类层进行识别。

#### 3.4.5. UniLM

给定一个输入序列$x=x_1...x_n$，UniLM 通过下图的方式获取每个词条的基于上下文的向量表示。整个预训练过程利用单向的语言建模（unidirectional LM），双向的语言建模（bidirectional LM）和 Seq2Seq 语言建模（sequence-to-sequence LM）优化共享的 Transformer 网络。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128145004244.png)

### 3.5. Graph Embedding

> Graph Embedding是一种将图结构数据映射为低微稠密向量的过程，从而捕捉到图的拓扑结构、顶点与顶点的关系、以及其他的信息。目前，Graph Embedding方法大致可以分为两大类：1）浅层图模型；2）深度图模型。**图嵌入（Graph / Network Embedding）**和**图神经网络（Graph Neural Networks, GNN）**是两个类似的研究领域。`图嵌入旨在将图的节点表示成一个低维向量空间`，同时`保留网络的拓扑结构和节点信息`，以便在后续的图分析任务中可以直接使用现有的机器学习算法。

#### 3.5.1. 浅层图

> 浅层图模型主要是采用`random-walk + skip-gram`模式的embedding方法。主要是通过在图中采用`随机游走策略来生成多条节点列表`，然后将`每个列表相当于含有多个单词（图中的节点）的句子`，再用`skip-gram模型`来训练每个`节点的向量`。这些方法主要包括`DeepWalk、Node2vec、Metapath2vec`等。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128145849897.png)

##### 1. DeepWalk

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126134723219.png)

到达节点后，下一步遍历其邻居节点的概率：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126134840114.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126134933080.png)

##### 2. Node2vec

> 该模型通过调整random walk权重的方法使得节点的embedding向量更倾向于体现网络的同质性或结构性。Node2vec无法指定游走路径，且仅适用于解决只包含一种类型节点的同构网络，无法有效表示包含多种类型节点和边类型的复杂网络。

- `同质性`：指得是`距离相近的节点的embedding向量应近似`，如下图中，与节点相连的节点s1、s2、s3和s4的embedding向量应相似。为了使embedding向量能够表达网络的同质性，需要让随机游走更倾向于`DFS`，因为DFS更有可能通过多次跳转，到达远方的节点上，使游走序列集中在一个较大的集合内部，使得在一个集合内部的节点具有更高的相似性，从而表达图的同质性。
- `结构性`：`结构相似的节点的embedding向量应近似`，如下图中，与节点结构相似的节点的embedding向量应相似。为了表达结构性，需要随机游走更倾向于`BFS`，因为BFS会更多的在当前节点的邻域中游走，相当于对当前节点的网络结构进行扫描，从而使得embedding向量能刻画节点邻域的结构信息。

##### ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126135506214.png)3. Metapath2vec

> 主要是在随机游走上使用基于meta-path的random walk来构建节点序列，然后用Skip-gram模型来完成顶点的Embedding。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126140116288.png)

##### 4. APP

> DeepWalk，node2vec 等，都无法保留图中的非对称信息。然而非对称性在很多问题，例如：社交网络中的链路预测、电商中的推荐等，中至关重要。

#### 3.5.2. 深度图

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608161356940.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/propa_step.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/table-1623202159387.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/gnn_table.png)

> 将图与深度模型结合，实现end-to-end训练模型，从而在图中提取拓扑图的空间特征。主要分为四大类：`Graph Convolution Networks (GCN)`，`Graph Attention Networks (GAT)`，`Graph AutoEncoder (GAE)`和`Graph Generative Networks (GGN)`。

- 基于**spatial domain**：基于`空域卷积`的方法直接`将卷积操作定义在每个结点的连接关系上`，跟传统的卷积神经网络中的卷积更相似一些。主要有两个问题：1）按照什么条件去`找`中心节点的邻居，也就是如何确定receptive field；2）按照什么方式`处理`包含不同数目邻居的特征。
- 基于**spectral domain**：借助卷积定理可以通过定义`频谱域上`的`内积`操作来得到`空间域图`上的卷积操作。

##### 0. GNN

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128151414174.png)

令 $H,O,X,X_n$  分别表示为 状态， 输出，特征，所有节点特征的向量:
$$
\begin{align}
H&7=F(H,X)\\
O&=G(H,X_n)\\
H^{t+1}&=F(H^t,X) \\
loss&=\sum{i=1}^p(t_i-o_i)
\end{align}
$$

##### 1. GCN

> 核心思想在于学习一个函数 $f$，通过聚合节点 $v_i$ `自身的特征` $X_i$ 和`邻居的特征` $X_j $获得节点的表示，其中 $j∈N(v_i) $为节点的邻居。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128151958945.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128152016958.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128152040805.png)

###### 1.1. 基于频谱（Spectral Methods）

> (1)`对图结构的小小扰动将会导致不同的特征基`;(2)特征分解需要`较为庞大的计算代价`;(3)学习到`的滤波器是针对特定问题的,不能够将其进行推广到更丰富的图结构上`.ChebNet及其一阶近似是局部卷积操作,从而可以在图的不同位置共享相同的滤波器参数.  基于频谱方法的一个关键缺陷是其需要将整个图的信息载入内存中,这使得其在大规模的图结构(如大规模的社交网络分析)上不能有效的进行应用.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128152717815.png)

###### .1. 频谱卷积神经网络

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608160028439.png)

###### .2. 契比雪夫频谱卷积网络(ChebNet)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608160042565.png)

###### .3. ChebNet的一阶近似

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608160110144.png)

###### 1.2. 基于空间的方法（Spatial Methods）

> 基于空间的方法通过节点的空间关系来定义图卷积操作。为了将图像和图关联起来，可以将图像视为一个特殊形式的图，每个像素点表示一个节点.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128152918580.png)

##### 2. GRN

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128153113989.png)

节点 $v$ 首先从邻居汇总信息，其中 $A_v $为图邻接矩阵$ A$ 的子矩阵表示节点$ v$ 及其邻居的连接。类似 GRU 的更新函数，通过结合其他节点和上一时间的信息更新节点的隐状态。$a $用于获取节点 $v $邻居的信息，$z $和 $r$ 分别为更新和重置门。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128153323524.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128153337702.png)

##### 3. Graph Attention

> 与 GCN 对于节点所有的邻居平等对待相比，注意力机制可以`为每个邻居分配不同的注意力评分`，从而识别更重要的邻居。

##### 4. **GraphSAGE**

> GraphSAGE（Graph SAmple and aggreGatE）是基于空间域方法，其思想与基于频谱域方法相反，是直接在图上定义卷积操作，对`空间上相邻的节点`上进行运算。其计算流程主要分为三部：
>
> - `对图中每个节点领节点进行采样`
> - 根据`聚合函数聚合邻居节点信息（特征）`
> - 得到图中`各节点的embedding向量`，供下游任务使用

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126145200343.png)

##### 5. DNGR

> 一种利用基于 Stacked Denoising Autoencoder（SDAE）提取特征的网络表示学习算法。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128150733829.png)

#### 3. Research Area

> 图分析任务可以大致抽象为以下四类: ( a )节点分类，( b )链接预测，( c )聚类，以及( d )可视化。

#### .1. 姿态识别&预测

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128150910711.png)

#### .2. 超图

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608161529742.png)

#### .3. 图构建

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608161558819.png)

#### .4. 子图嵌入

> 图嵌入（Graph Embedding，也叫Network Embedding）是一种将图数据（通常为高维稠密的矩阵）映射为低微稠密向量的过程，能够很好地解决图数据难以高效输入机器学习算法的问题。图嵌入是将属性图转换为向量或向量集。嵌入应该捕获图的**拓扑结构、顶点到顶点的关系以及关于图、子图和顶点的其他相关信息**。
>
> - 节点的分布式表示；节点之间的相似性表示链接强度； 编码网络信息并生成节点表示
>
> - 顶点嵌入:每个顶点(节点)用其自身的向量表示进行编码。这种嵌入一般用于在顶点层次上执行可视化或预测。比如，在2D平面上显示顶点，或者基于顶点相似性预测新的连接。
> - 图嵌入:用单个向量表示整个图。这种嵌入用于在图的层次上做出预测，可者想要比较或可视化整个图。例如，比较化学结构。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128150931910.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201128150955897.png)

## 4. 视文分析

### 4.1. 预训练模型

- **模型越来越大。**比如 Transformer 的层数变化，从12层的 Base 模型到24层的 Large 模型。导致模型的参数越来越大，比如 GPT 110 M，到 GPT-2 是1.5 Billion，图灵是 17 Billion，而 GPT-3 达到了惊人的 175 Billion。一般而言模型大了，其能力也会越来越强，但是训练代价确实非常大。
- **预训练方法也在不断增加**，从自回归 LM，到自动编码的各种方法，以及各种多任务训练等。
- **，还有从语言、多语言到多模态不断演进**。最后就是模型压缩，使之能在实际应用中经济的使用，比如在手机端。这就涉及到知识蒸馏和 teacher-student models，把大模型作为 teacher，让一个小模型作为 student 来学习，接近大模型的能力，但是模型的参数减少很多。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126150533100.png)

#### 4.1.1. LayoutLM

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126150648546.png)

- `二维位置嵌入 2-D Position Embedding`：根据 `OCR` 获得的文本边界框 (Bounding Box)，能获取文本在文档中的具体位置。在将对应坐标转化为虚拟坐标之后，则可以计算该坐标对应在 `x、y、w、h` 四个 Embedding 子层的表示，最终的` 2-D Position Embedding 为四个子层的 Embedding 之和`。
- `图嵌入 Image Embedding`：将每个文本相应的边界框 (Bounding Box) 当作 Faster R-CNN 中的候选框（Proposal），从而提取对应的局部特征。其特别之处在于，由于 `[CLS] 符号用于表示整个输入文本的语义`，所以同样`使用整张文档图像作为该位置的 Image Embedding`，从而保持模态对齐。

- 预训练：
  - `掩码视觉语言模型`（Masked Visual-Language Model，MVLM）：大量实验已经证明 MLM 能够在预训练阶段有效地进行自监督学习。研究员们在此基础上进行了修改：`在遮盖当前词之后，保留对应的 2-D Position Embedding 暗示，让模型预测对应的词`。在这种方法下，模型根据已有的上下文和对应的视觉暗示预测被掩码的词，从而让模型更好地学习文本位置和文本语义的模态对齐关系。
  - `多标签文档分类`（Multi-label Document Classification，MDC）：MLM 能够有效的表示词级别的信息，但是对于文档级的表示，还需要将文档级的预训练任务引入更高层的语义信息。在预训练阶段研究员们使用的 `IIT-CDIP 数据集为每个文档提供了多标签的文档类型标注`，并引入 `MDC 多标签文档分类任务`。该任务使得模型可以利用这些监督信号，聚合相应的文档类别并捕捉文档类型信息，从而获得更有效的高层语义表示。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201126151137411.png)

## 5. 学习链接

- https://zhuanlan.zhihu.com/p/39562499

- https://zhuanlan.zhihu.com/p/101179171

- [用万字长文聊一聊 Embedding 技术 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzIyNDY5NjEzNQ==&mid=2247486604&idx=2&sn=4805abb34e3a94243bb182ec74fff550&chksm=e80a4ea4df7dc7b244e39535ae6adf34eece9d10e428e07d52c9675d2629fe111c58e158127c&scene=126&sessionid=1606356123&key=d29f68c0cc2770c7a295460b63af1f2b438bd8d34e43fd923ffab1503f5419358cb801a0ec1cb521815173b06edbb8b4391e82bac9ca99bd324a470299662d0074b19f1c710ebb605f21ad6653f596049aeffc3ecf67018c3a6e1f06e4d967a80aa1117ee0badb2637c6f98ab2b9a0158f340990857e17d4ffc094708a5f04b9&ascene=1&uin=MzE0ODMxOTQzMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A6O1Vab3FB0dKfbgjPLgUOI%3D&pass_ticket=dWuSAMKgl2YK7zg1wPn7XPBZPohIpbR0IPLY%2Fi1CvZ%2B0Hp9NIxue%2FHPzD4K1r4vD&wx_header=0)

- “Document Visual Question Answering”：https://medium.com/@anishagunjal7/document-visual-question-answering-e6090f3bddee

- LayoutLM 论文：https://arxiv.org/abs/1912.13318

- LayoutLM 代码&模型：https://aka.ms/layoutlm

- https://leovan.me/cn/2020/04/graph-embedding-and-gnn/

  



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/wordembedding/  

