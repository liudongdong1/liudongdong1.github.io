# SentenceDistance


### 0. Relative API

- CountVectorizer() 词频统计

```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.toarray())
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
>>> vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
>>> X2 = vectorizer2.fit_transform(corpus)
>>> print(vectorizer2.get_feature_names())
['and this', 'document is', 'first document', 'is the', 'is this',
'second document', 'the first', 'the second', 'the third', 'third one',
 'this document', 'this is', 'this the']
 >>> print(X2.toarray())
 [[0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]]
```

### 1. Word2Vec

> 将每一个词转换为向量的过程

```python
import gensim
import jieba
import numpy as np
from scipy.linalg import norm
  
model_file = './word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
  
def vector_similarity(s1, s2):
           
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
     
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))
 
s1 = '你在干嘛'
s2 = '你正做什么'
print(vector_similarity(s1, s2))
 
strings = [
    '你在干什么',
    '你在干啥子',
    '你在做什么',
    '你好啊',
    '我喜欢吃香蕉'
]
  
target = '你在干啥'
for string in strings:
    print(string, vector_similarity(string, target))
```

### 2. Jaccard index

> 杰卡德系数，英文叫做 Jaccard index，又称为 Jaccard 相似系数，用于`比较有限样本集之间的相似性与差异性`。`Jaccard 系数值越大，样本相似度越高`.实际上它的计算方式非常简单，`就是两个样本的交集除以并集得到的数值`，当两个样本完全一致时，结果为 1，当两个样本完全不同时，结果为 0

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211005195526760.png)

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
  
def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(s)
     
    # 将字中间加入空格
    s1, s2 = add_space(s1),  add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    print(cv.get_feature_names())
    print(vectors)
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    print(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    print(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator
  
s1 = '你在干嘛呢'
s2 = '你在干什么呢'
print(jaccard_similarity(s1, s2))
```

### 3. TF 计算

> 直接计算 TF 矩阵中两个向量的相似度,cosθ``=``a·b``/``|a|``*``|b|

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm
  
def tf_similarity(s1, s2):
           
    def add_space(s):
        return ' '.join(s)
     
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
  
s1 = '你在干嘛呢'
s2 = '你在干什么呢'
print(tf_similarity(s1, s2))
```

### 4. TFIDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
 
def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(s)
     
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
s1 = '你在干嘛呢'
s2 = '你在干什么呢'
print(tfidf_similarity(s1, s2))
```

### 5. 编辑距离

> 编辑距离，英文叫做 **Edit Distance，又称 Levenshtein 距离**，是指两个字串之间，由一个转成另一个所需的最少编辑操作次数.

```python
import distance
def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)
strings = [
    '你在干什么',
    '你在干啥子',
    '你在做什么',
    '你好啊',
    '我喜欢吃香蕉'
]
target = '你在干啥'
results = list(filter(lambda x: edit_distance(x, target) <= 2, strings))
print(results)
```

- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer

### 6. Hamming distance 

### 7. Cosine distance

> Cosine distance：cos(theta) =A*B/|A||B| ，常用来判断两个向量之间的夹角，夹角越小，表示它们越相似。理解：利用随机的超平面（random hyperplane）将原始数据空间进行划分，每一个数据被投影后会落入超平面的某一侧，经过多个随机的超平面划分后，原始空间被划分为了很多cell，而位于每个cell内的数据被认为具有很大可能是相邻的（即原始数据之间的cosine distance很小）。

### 8. normal Euclidean distance

> Euclidean distance是衡量D维空间中两个点之间的距离的一种距离度量方式。


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/sentencedistance/  

