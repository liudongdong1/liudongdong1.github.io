# Torchtext


> torchtext.data.Example : 用来表示一个样本，数据+标签
>
> torchtext.vocab.Vocab: 词汇表相关
>
> torchtext.data.Datasets: 数据集类，**getitem** 返回 Example实例
>
> torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
>
> - 创建 Example时的预处理
>
> - batch 时的一些处理操作。
>
> torchtext.data.Iterator: 迭代器，用来生成 batch

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027162627905.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027180456476.png)

### 1.Data 相关类

#### 1.1. Dataset, Batch, Example,Field

```python
# defines a dataset composed of examples along with its fields;
torchtext.data.Dataset(examples, fields, filter-Pred=None);
#Defines a Dataset of columns stored in CSV, TSV, or JSON format.
torchtext.data.TabularDataset(path, format, fields, skip_header=False,scv_reader_params=[],**kwargs);
#Defines a batch of examples along with its Fields.
torchtext.data.Batch(data=None,dataset=None,device=None)
#Defines a single training or test example.Stores each column of the example as an attribute.
torchtext.data.Example
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027162219368.png)

- > **Field**: Torchtext采用了一种声明式的方法来加载数据：你来告诉Torchtext你希望的数据是什么样子的，剩下的由torchtext来处理。field在默认的情况下都期望一个输入是一组单词的序列，并且将单词映射成整数。这个映射被称为vocab。如果一个field已经被数字化了并且不需要被序列化，可以将参数设置为use_vocab=False以及sequential=False。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027174530628.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027173439055.png)

####  1.2.  具体使用

```python
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm

tokenize = lambda x: x.split()
# fix_length指定每条文本的长度，截断补长
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)

train_data = pd.read_csv('data/train_one_label.csv')
valid_data = pd.read_csv('data/valid_one_label.csv')
test_data = pd.read_csv('data/test.csv')
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)
 
# get_dataset构造并返回Dataset所需的examples和fields
def get_dataset(csv_data, text_field, label_field, test=False):
    fields = [('id', None), ('comment_text', text_field), ('toxic', label_field)]
    examples = [] 
    if test:
        for text in tqdm(csv_data['comment_text']):
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields
 
# 得到构建Dataset所需的examples和fields
train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
valid_examples, valid_fields = get_dataset(valid_data, TEXT, LABEL)
test_examples, test_fields = get_dataset(test_data, TEXT, None, True)
 
# 构建Dataset数据集
train = data.Dataset(train_examples, train_fields)
valid = data.Dataset(valid_examples, valid_fields)
test = data.Dataset(test_examples, test_fields)
```

#### 1.3. 自定义数据

```python
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import random
import os
 
train_path = 'data/train_one_label.csv'
valid_path = 'data/valid_one_label.csv'
test_path = 'data/test.csv'
 
class MyDataset(data.Dataset):
    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None), ("comment_text", text_field), ("toxic", label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))
         
        if test:
            for text in tqdm(csv_data['comment_text']):
                examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
                if aug:
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([None, text, label-1], fields)
        super(MyDataset, self).__init__(examples, fields, **kwargs)
     
    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)
         
    def dropout(self, text, p=0.5):
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        retrurn ' '.join(text)
train = MyDataset(train_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
valid = MyDataset(valid_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
test = MyDataset(test_path, text_field=TEXT, label_field=LABEL, test=True, aug=1)
```

#### 1.4.  pytorch 自带数据集使用

> All datasets are subclasses of [`torchtext.data.Dataset`](https://pytorch.org/text/data.html#torchtext.data.Dataset), which inherits from [`torch.utils.data.Dataset`](https://pytorch.org/docs/0.3.0/data.html#torch.utils.data.Dataset) i.e, they have `split` and `iters` methods implemented.

```python
#      使用方式一
# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)
# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train)
# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=3, device=0)
#      使用方式二
# use default configurations
train_iter, test_iter = datasets.IMDB.iters(batch_size=4)
```

### 2. 构建词表

#### 2.1. tokenizer

```python
# Generate tokenizer function for a string sentence.
torchtext.data.utils.get_tokenizer(*tokenizer*, *language='en'*)
# return an iterator that yields the given tokens and their ngrams
torchtext.data.utils.ngrams_iterator(token_list,ngrams)

token_list = ['here', 'we', 'are']
list(ngrams_iterator(token_list, 2))
# >>> ['here', 'here we', 'we', 'we are', 'are']
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027181930226.png)

> 所谓构建词表，即需要给每个单词编码，也就是用数字表示每个单词，这样才能传入模型。

```python
#   方式一： 使用build_vocab()方法构建词表
TEXT.build_vocab(train)
# 统计词频
TEXT.vocab.freqs.most_common(10)
#   方式二： 使用预训练的词向量
#         2.1. 使用pytorch默认支持的预训练词向量
from torchtext.vocab import GloVe
from torchtext import data
TEXT = data.Field(sequential=True)
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
#默认情况下，会自动下载对应的预训练词向量文件到当前文件夹下的.vector_cache目录下
TEXT.build_vocab(train, vectors="glove.6B.300d")
#          2.2. 使用外部预训练好的词向量
if not os.path.exists('.vector_cache'):
    os.mkdir('.vector_cache')
vectors = Vectors(name='myvector/glove/glove.6B.200d.txt')
TEXT.build_vocab(train, vectors=vectors)
#         2.3 vetor 使用
examples = ['chip', 'baby', 'Beautiful']
vec = text.vocab.GloVe(name='6B', dim=50)
#Look up embedding vectors of tokens.
ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027175126164.png)

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027175126164.png" alt="image-20201027175126164" style="zoom:33%;" />

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027175043317.png)

### 3. 构建迭代器

```python
from torchtext.data import Iterator, BucketIterator
# 若只对训练集构造迭代器
# train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
#BucketIterator相比Iterator的优势是会自动选取样本长度相似的数据来构建批数据。但是在测试集中一般不想改变样本顺序，因此测试集使用Iterator迭代器来构建。
# 若同时对训练集和验证集进行迭代器构建
train_iter, val_iter = BucketIterator.splits(
        (train, valid),
        batch_size=(8, 8),
        device=-1, # 如果使用gpu，将-1更换为GPU的编号
        sort_key=lambda x: len(x.comment_text),
        sort_within_batch=False,
        repeat=False
)
test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
for idx, batch in enumerate(train_iter):
    text, label = batch.comment_text, batch.toxic
```

### 4. 使用torchtext构建的数据集用于LSTM

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
weight_matrix = TEXT.vocab.vectors
 
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)
        self.word_embeddings.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layer=1)
        self.decoder = nn.Linear(128, 2)
     
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]
        final = lstm_out[-1]
        y = self.decoder(final)
        return y
 
def main():
    model = LSTM()
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_funciton = F.cross_entropy
    for epoch, batch in enumerate(train_iter):
        optimizer.zero_grad()
        start = time.time()
        predicted = model(batch.comment_text)
         
        loss = loss_function(predicted, batch.toxic)
        loss.backward()
        optimizer.step()
        print(loss)
```

### 5. Evaluation机制

#### 5.1. Bleu_score

> Computes the BLEU score between a candidate translation corpus and a references translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

```python
from torchtext.data.metrics import bleu_score
candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
bleu_score(candidate_corpus, references_corpus)
    0.8408964276313782
```

#### 5.2. ROUGE

> ROUGEROUGE 由 Chin-Yew Lin 在 2004 年的论文[《ROUGE: A Package for Automatic Evaluation of Summaries》](https://www.aclweb.org/anthology/W04-1013.pdf)中提出。与 BLEUBLEU 类似，通过统计生成的摘要与参考摘要集合之间重叠的基本单元（nn 元组）的数目来评估摘要的质量，该方法已成为自动文摘系统评价的主流方法。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027183228807.png)

### 6. MultiHeadAttention

- https://pytorch.org/text/nn_modules.html
- 学习链接：https://blog.nowcoder.net/n/3a8d2c1b05354f3b942edfd4966bb0c1
- https://xiaosheng.run/2020/08/13/article184/

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/torchtext_vocab/  

