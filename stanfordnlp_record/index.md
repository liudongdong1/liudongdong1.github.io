# Standfordnlp


> NLTK 是一款著名的 Python 自然语言处理(Natural Language Processing, NLP)工具包，在其收集的大量公开数据集、模型上提供了全面、易用的接口，涵盖了分词、词性标注(Part-Of-Speech tag, POS-tag)、命名实体识别(Named Entity Recognition, NER)、句法分析(Syntactic Parse)等各项 NLP 领域的功能。

> Stanford NLP 是由斯坦福大学的 NLP 小组开源的 Java 实现的 NLP 工具包，同样对 NLP 领域的各个问题提供了解决办法。斯坦福大学的 NLP 小组是世界知名的研究小组，如果能将 NLTK 和 Stanford NLP 这两个工具包结合起来使用，那自然是极好的！在 2004 年 Steve Bird 在 NLTK 中加上了对 Stanford NLP 工具包的支持，通过调用外部的 jar 文件来使用 Stanford NLP 工具包的功能。现在的 NLTK 中，通过封装提供了 Stanford NLP 中的以下几个功能:
>
> 1. 分词
> 2. 词性标注
> 3. 命名实体识别
> 4. 句法分析
> 5. 依存句法分析

### 1. 命名实体识别

> 命名实体识别（Named Entity Recognition，简称NER）是信息提取、问答系统、句法分析、机器翻译等应用领域的重要基础工具，在自然语言处理技术走向实用化的过程中占有重要地位。一般来说，**命名实体识别的任务就是识别出待处理文本中三大类（实体类、时间类和数字类）、七小类（人名、机构名、地名、时间、日期、货币和百分比）命名实体。**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116085824037.png)

### 1.1. NLTK

```python
pip install nltk
import nltk
nltk.download()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116094247020.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116094708285.png)

#### 1.1.1. 语料库

| **语料库**    | **说明**                                                     |
| ------------- | ------------------------------------------------------------ |
| **gutenberg** | **一个有若干万部的小说语料库，多是古典作品**                 |
| **webtext**   | **收集的网络广告等内容**                                     |
| **nps_chat**  | **有上万条聊天消息语料库，即时聊天消息为主**                 |
| **brown**     | **一个百万词级的英语语料库，按文体进行分类**                 |
| **reuters**   | **路透社语料库，上万篇新闻方档，约有1百万字，分90个主题，并分为训练集和测试集两组** |
| **inaugural** | **演讲语料库，几十个文本，都是总统演说**                     |

```python
from nltk.corpus import brown
print(brown.categories())   #输出brown语料库的类别
print(len(brown.sents()))   #输出brown语料库的句子数量
print(len(brown.words()))   #输出brown语料库的词数量
 
'''
结果为：
['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 
'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 
'science_fiction']
57340
1161192
'''
```

#### 1.1.2. 词频统计(frequency)

| **方法**                         | **作用**                                             |
| -------------------------------- | ---------------------------------------------------- |
| **B()**                          | **返回词典的长度**                                   |
| **plot(title,cumulative=False)** | **绘制频率分布图，若cumu为True，则是累积频率分布图** |
| **tabulate()**                   | **生成频率分布的表格形式**                           |
| **most_common()**                | **返回出现次数最频繁的词与频度**                     |
| **hapaxes()**                    | **返回只出现过一次的词**                             |

```python
import nltk
tokens=[ 'my','dog','has','flea','problems','help','please',
         'maybe','not','take','him','to','dog','park','stupid',
         'my','dalmation','is','so','cute','I','love','him'  ]
#统计词频
freq = nltk.FreqDist(tokens)
#输出词和相应的频率
for key,val in freq.items():
    print (str(key) + ':' + str(val))
#可以把最常用的5个单词拿出来
standard_freq=freq.most_common(5)
print(standard_freq)
#绘图函数为这些词频绘制一个图形
freq.plot(20, cumulative=False)
```

#### 1.1.3. 停用分词(stopwords)

```python
#英文停用分词
from nltk.corpus import stopwords
tokens=[ 'my','dog','has','flea','problems','help','please',
         'maybe','not','take','him','to','dog','park','stupid',
         'my','dalmation','is','so','cute','I','love','him'  ]
clean_tokens=tokens[:]
stwords=stopwords.words('english')
for token in tokens:
    if token in stwords:
        clean_tokens.remove(token)
print(clean_tokens)
```

#### 1.1.4. 分词&&分句(tokenize)

```python
#--- 分句
from nltk.tokenize import sent_tokenize
mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
print(sent_tokenize(mytext))
#--- 分词
from nltk.tokenize import word_tokenize
mytext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
print(word_tokenize(mytext))
```

#### 1.1.5. 词干提取（Stemming)

> 单词词干提取就是`从单词中去除词缀并返回词根`。（`比方说 working 的词干是 work。`）搜索引擎在索引页面的时候使用这种技术，所以很多人通过同一个单词的不同形式进行搜索，返回的都是相同的，有关这个词干的页面。词干提取的算法有很多，但最常用的算法是 **Porter 提取算法**。NLTK 有一个 PorterStemmer 类，使用的就是 Porter 提取算法。

```python
#    PorterStemmer算法
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('working'))
#结果为：work 
#    LancasterStemmer算法
from nltk.stem import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('working'))
#结果为：work 
```

#### 1.1.6. 词干还原（Lemmatization）

```python
#词形还原与词干提取类似， 但不同之处在于词干提取经常可能创造出不存在的词汇，词形还原的结果是一个真正的词汇
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('playing', pos="v"))
print(lemmatizer.lemmatize('playing', pos="n"))
print(lemmatizer.lemmatize('playing', pos="a"))
print(lemmatizer.lemmatize('playing', pos="r"))
'''
结果为：
play
playing
playing
playing
'''
```

#### 1.1.7. 词性标注（PosTag）

> **词性标注是把一个句子中的单词标注为名词，形容词，动词等。**

| **标记（Tag）** | **含义（Meaning）**                | **例子（Examples）**                    |
| --------------- | ---------------------------------- | --------------------------------------- |
| **ADJ**         | **形容词（adjective）**            | **new，good，high，special，big**       |
| **ADV**         | **副词（adverb）**                 | **really,，already，still，early，now** |
| **CNJ**         | **连词（conjunction）**            | **and，or，but，if，while**             |
| **DET**         | **限定词（determiner）**           | **the，a，some，most，every**           |
| **EX**          | **存在量词（existential）**        | **there，there's**                      |
| **FW**          | **外来词（foreign word）**         | **dolce，ersatz，esprit，quo，maitre**  |
| **MOD**         | **情态动词（modal verb）**         | **will，can，would，may，must**         |
| **N**           | **名词（noun）**                   | **year，home，costs，time**             |
| **NP**          | **专有名词（proper noun）**        | **Alison，Africa，April，Washington**   |
| **NUM**         | **数词（number）**                 | **twenty-four，fourth，1991，14:24**    |
| **PRO**         | **代词（pronoun）**                | **he，their，her，its，my，I，us**      |
| **P**           | **介词（preposition）**            | **on，of，at，with，by，into，under**   |
| **TO**          | **词 to（the word to）**           | **to**                                  |
| **UH**          | **感叹词（interjection）**         | **ah，bang，ha，whee，hmpf，oops**      |
| **V**           | **动词（verb）**                   | **is，has，get，do，make，see，run**    |
| **VD**          | **过去式（past tense）**           | **said，took，told，made，asked**       |
| **VG**          | **现在分词（present participle）** | **making，going，playing，working**     |
| **VN**          | **过去分词（past participle）**    | **given，taken，begun，sung**           |
| **WH**          | **wh限定词（wh determiner）**      | **who，which，when，what，where**       |

```python
import nltk
text=nltk.word_tokenize('what does the fox say')
print(text)
print(nltk.pos_tag(text))
'''
结果为：
['what', 'does', 'the', 'fox', 'say']
输出是元组列表，元组中的第一个元素是单词，第二个元素是词性标签
[('what', 'WDT'), ('does', 'VBZ'), ('the', 'DT'), ('fox', 'NNS'), ('say', 'VBP')]
'''
```

#### 1.1.8. wordnet

> **wordnet** 是为自然语言处理构建的数据库。它包括部分词语的一个同义词组和一个简短的定义和反义词。

```python
from nltk.corpus import wordnet
syn = wordnet.synsets("pain")  #获取“pain”的同义词集
print(syn[0].definition())  #定义
print(syn[0].examples())# 例句

synonyms = []
for syn in wordnet.synsets('Computer'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name()) #同义词
print(synonyms)


from nltk.corpus import wordnet
antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():   #判断是否是正确的反义词
            antonyms.append(l.antonyms()[0].name())
print(antonyms)
```

#### 1.1.9.  命名实体识别

```python
import re
import pandas as pd
import nltk

def parse_document(document):
   document = re.sub('\n', ' ', document)
   if isinstance(document, str):
       document = document
   else:
       raise ValueError('Document is not string!')
   document = document.strip()
   sentences = nltk.sent_tokenize(document)
   sentences = [sentence.strip() for sentence in sentences]
   return sentences

# sample document
text = """
FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, 
Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland. Headquartered in Zürich, its 
membership now comprises 211 national associations. Member countries must each also be members of one of 
the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America 
and the Caribbean, Oceania, and South America.
"""

# tokenize sentences
sentences = parse_document(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# tag sentences and use nltk's Named Entity Chunker
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
ne_chunked_sents = [nltk.ne_chunk(tagged) for tagged in tagged_sentences]
# extract all named entities
named_entities = []
for ne_tagged_sentence in ne_chunked_sents:
   for tagged_tree in ne_tagged_sentence:
       # extract only chunks having NE labels
       if hasattr(tagged_tree, 'label'):
           entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #get NE name
           entity_type = tagged_tree.label() # get NE category
           named_entities.append((entity_name, entity_type))
           # get unique named entities
           named_entities = list(set(named_entities))

# store named entities in a data frame
entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
# display results
print(entity_frame)
```

- NLTK 中集成了standordnlp

> StanfordNERTagger('./stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
>                        path_to_jar='./stanford-ner/stanford-ner.jar')

```python
import re
from nltk.tag import StanfordNERTagger
import os
import pandas as pd
import nltk

def parse_document(document):
   document = re.sub('\n', ' ', document)
   if isinstance(document, str):
       document = document
   else:
       raise ValueError('Document is not string!')
   document = document.strip()
   sentences = nltk.sent_tokenize(document)
   sentences = [sentence.strip() for sentence in sentences]
   return sentences

# sample document
text = """
FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, 
Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland. Headquartered in Zürich, its 
membership now comprises 211 national associations. Member countries must each also be members of one of 
the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America 
and the Caribbean, Oceania, and South America.
"""
#C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;C:\Program Files (x86)\Common Files\Intel\Shared Libraries\redist\intel64\compiler;C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\Program Files\dotnet\;C:\Program Files (x86)\Git\cmd;C:\Users\dell\Anaconda3;C:\Users\dell\Anaconda3\Scripts;C:\Users\dell\Anaconda3\Library\bin;C:\Program Files\nodejs\;C:\Program Files (x86)\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files\Microsoft SQL Server\110\DTS\Binn\;C:\Users\dell\AppData\Local\Android\Sdk\tools\;C:\Users\dell\AppData\Local\Android\Sdk\platform-tools\;D:\latex\texlive\2020\bin\win32;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\MATLAB\R2015b\runtime\win64;C:\Program Files\MATLAB\R2015b\bin;C:\Program Files\MATLAB\R2015b\polyspace\bin;C:\msys64\usr\bin;C:\Users\dell\Downloads\Programs\bazel-3.4.1-windows-x86_64_2.exe;
sentences = parse_document(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# set java path in environment variables
java_path = r'C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe'
os.environ['JAVAHOME'] = java_path
# load stanford NER
sn = StanfordNERTagger('./stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
                       path_to_jar='./stanford-ner/stanford-ner.jar')

# tag sentences
ne_annotated_sentences = [sn.tag(sent) for sent in tokenized_sentences]
# extract named entities
named_entities = []
for sentence in ne_annotated_sentences:
   temp_entity_name = ''
   temp_named_entity = None
   for term, tag in sentence:
       # get terms with NE tags
       if tag != 'O':
           temp_entity_name = ' '.join([temp_entity_name, term]).strip() #get NE name
           temp_named_entity = (temp_entity_name, tag) # get NE and its category
       else:
           if temp_named_entity:
               named_entities.append(temp_named_entity)
               temp_entity_name = ''
               temp_named_entity = None

# get unique named entities
named_entities = list(set(named_entities))
# store named entities in a data frame
entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
# display results
print(entity_frame)
```

#### 1.1.10. 文本分类

#### 1.1.11. 情感分类

- http://ir.dlut.edu.cn/EmotionOntologyDownload 

#### 1.1.12. 事件抽取 

- https://github.com/twjiang/fact_triple_extraction

## 2.StanfordNlp

> 斯坦福 NER 标记器的一大优势是，为我们提供了几种不同的模型来提取命名实体。我们可以使用以下任何一个：
>
> - 三类模型，用于识别位置，人员和组织
> - 四类模型，用于识别位置，人员，组织和杂项实体
> - 七类模型，识别位置，人员，组织，时间，金钱，百分比和日期

```python
#!/usr/bin/env python
from __future__ import print_function
import os
import pickle
from argparse import ArgumentParser
from platform import system
from subprocess import Popen
from sys import argv
from sys import stderr

#IS_WINDOWS = True if system() == 'Windows' else False
JAVA_BIN_PATH = 'java.exe' if IS_WINDOWS else 'java'
STANFORD_NER_FOLDER = 'stanford-ner'
def arg_parse():
    arg_p = ArgumentParser('Stanford NER Python Wrapper')
    arg_p.add_argument('-f', '--filename', type=str, default=None)
    arg_p.add_argument('-v', '--verbose', action='store_true')
    return arg_p
def debug_print(log, verbose):
    if verbose:
        print(log)
def process_entity_relations(entity_relations_str, verbose=True):
    # format is ollie.
    entity_relations = list()
    for s in entity_relations_str:
        entity_relations.append(s[s.find("(") + 1:s.find(")")].split(';'))
    return entity_relations


def stanford_ner(filename, verbose=True, absolute_path=None):
    out = 'out.txt'

    command = ''
    if absolute_path is not None:
        command = 'cd {};'.format(absolute_path)
    else:
        filename = '../{}'.format(filename)
#java -mx1g -cp "*:lib/*" edu.stanford.nlp.ie.NERClassifierCombiner -textFile sample.txt -ner.model classifiers/english.all.3class.distsim.crf.ser.gz,classifiers/english.conll.4class.distsim.crf.ser.gz,classifiers/english.muc.7class.distsim.crf.ser.gz
    command += 'cd {}; {} -mx1g -cp "*:lib/*" edu.stanford.nlp.ie.NERClassifierCombiner ' \
               '-ner.model classifiers/english.all.3class.distsim.crf.ser.gz ' \
               '-outputFormat tabbedEntities -textFile {} > ../{}' \
        .format(STANFORD_NER_FOLDER, JAVA_BIN_PATH, filename, out)

    if verbose:
        debug_print('Executing command = {}'.format(command), verbose)
        java_process = Popen(command, stdout=stderr, shell=True)
    else:
        java_process = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ner exited with a non-zero code status.'

    if absolute_path is not None:
        out = absolute_path + out

    with open(out, 'r') as output_file:
        results_str = output_file.readlines()
    os.remove(out)

    results = []
    for res in results_str:
        if len(res.strip()) > 0:
            split_res = res.split('\t')
            entity_name = split_res[0]
            entity_type = split_res[1]
            if len(entity_name) > 0 and len(entity_type) > 0:
                results.append([entity_name.strip(), entity_type.strip()])
    if verbose:
        pickle.dump(results_str, open('out.pkl', 'wb'))
        debug_print('wrote to out.pkl', verbose)
    return results
def main(args):
    arg_p = arg_parse().parse_args(args[1:])
    filename = arg_p.filename
    verbose = arg_p.verbose
    debug_print(arg_p, verbose)
    if filename is None:
        print('please provide a text file containing your input. Program will exit.')
        exit(1)
    if verbose:
        debug_print('filename = {}'.format(filename), verbose)
    entities = stanford_ner(filename, verbose)
    print('\n'.join([entity[0].ljust(20) + '\t' + entity[1] for entity in entities]))

if __name__ == '__main__':
    exit(main(argv))
```

## 3. jieba 分词

```shell
pip3 install jieba
import jieba
```

### 3.1. 分词

> ##### jieba.cut 和jieba.lcut；  `lcut` 将返回的对象转化为`list对象`返回·

```
def cut(self, sentence, cut_all=False, HMM=True, use_paddle=False):
# sentence: 需要分词的字符串;
# cut_all: 参数用来控制是否采用全模式；
# HMM: 参数用来控制是否使用 HMM 模型;
# use_paddle: 参数用来控制是否使用paddle模式下的分词模式，paddle模式采用延迟加载方式，通过enable_paddle接口安装paddlepaddle-tiny
```

###### 1）精准模式（默认）:

> 试图将句子最精确地切开，适合文本分析

```python
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("精准模式: " + "/ ".join(seg_list))  # 精确模式
# -----output-----
精准模式: 我/ 来到/ 北京/ 清华大学
```

###### 2）全模式:

> 把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；

```python
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("全模式: " + "/ ".join(seg_list))  # 全模式
# -----output-----
全模式: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
```

###### 3）paddle模式

> 利用PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词。同时支持词性标注。
> paddle模式使用需安装paddlepaddle-tiny，pip install paddlepaddle-tiny==1.6.1。
> 目前paddle模式支持jieba v0.40及以上版本。
> jieba v0.40以下版本，请升级jieba，pip installjieba --upgrade。 [PaddlePaddle官网](https://www.paddlepaddle.org.cn/)

```python
import jieba
# 通过enable_paddle接口安装paddlepaddle-tiny，并且import相关代码；
jieba.enable_paddle()  # 初次使用可以自动安装并导入代码
seg_list = jieba.cut(str, use_paddle=True)
print('Paddle模式: ' + '/'.join(list(seg_list)))
# -----output-----
Paddle模式: 我/来到/北京清华大学
```

### 3.2. 搜索引擎模式

> 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词

```python
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))
# -----output-----
小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, ，, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
```

##### 1) jieba.Tokenizer(dictionary=DEFAULT_DICT)

> 新建自定义分词器，可用于同时使用不同词典。jieba.dt 为默认分词器，所有全局分词相关函数都是该分词器的映射。

```python
import jieba
test_sent = "永和服装饰品有限公司"
# jieba.load_userdict(dict_path)    # dict_path为文件类对象或自定义词典的路径。
result = jieba.tokenize(test_sent) ##Tokenize：返回词语在原文的起始位置
print(result)
for tk in result:
    # print ("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2])    )
    print (tk)
   
# -----output-----
<generator object Tokenizer.tokenize at 0x7f6b68a69d58>
('永和', 0, 2)  #词语、词频（可省略）、词性（可省略）
('服装', 2, 4)
('饰品', 4, 6)
('有限公司', 6, 10)    
```

###### 2）使用自定义词典文件

```python
import jieba

test_sent = "中信建投投资公司投资了一款游戏,中信也投资了一个游戏公司"
jieba.load_userdict("userdict.txt")
words = jieba.cut(test_sent)
print(list(words))
#-----output------
['中信建投', '投资公司', '投资', '了', '一款', '游戏', ',', '中信', '也', '投资', '了', '一个', '游戏', '公司']
```

###### 3）使用 jieba 在程序中动态修改词典

```python
import jieba

# 定义示例句子
test_sent = "中信建投投资公司投资了一款游戏,中信也投资了一个游戏公司"
#添加词
jieba.add_word('中信建投')
jieba.add_word('投资公司')
# 删除词
jieba.del_word('中信建投')
words = jieba.cut(test_sent)
print(list(words))
#-----output------
['中信', '建投', '投资公司', '投资', '了', '一款', '游戏', ',', '中信', '也', '投资', '了', '一个', '游戏', '公司']
```

### 3.3. 关键词提取

###### 1）TF-IDF接口和示例

```python
import jieba.analyse
```

- jieba.analyse.extract_tags(sentence, topK=20, withWeight=False,allowPOS=())

  其中需要说明的是：

  - 1.sentence 为待提取的文本
  - 2.topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
  - 3.withWeight 为是否一并返回关键词权重值，默认值为 False
  - 4.allowPOS 仅包括指定词性的词，默认值为空，即不筛选

- jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件

```python
import jieba
import jieba.analyse
#读取文件,返回一个字符串，使用utf-8编码方式读取，该文档位于此python同以及目录下
content  = open('data.txt','r',encoding='utf-8').read()
tags = jieba.analyse.extract_tags(content,topK=10,withWeight=True,allowPOS=("nr")) 
print(tags)

# ----output-------
[('虚竹', 0.20382572423643955), ('丐帮', 0.07839419568792882), ('什么', 0.07287469641815765), ('自己', 0.05838617200768695), ('师父', 0.05459680087740782), ('内力', 0.05353758008018405), ('大理', 0.04885277765801372), ('咱们', 0.04458784837687502), ('星宿', 0.04412126568280158), ('少林', 0.04207588649463058)]
123456789
```

###### 2）Stop Words

- 用法： jieba.analyse.set_stop_words(file_name) # file_name为自定义语料库的路径
- 自定义语料库示例：

```python
import jieba
import jieba.analyse
#读取文件,返回一个字符串，使用utf-8编码方式读取，该文档位于此python同以及目录下
content  = open(u'data.txt','r',encoding='utf-8').read()
jieba.analyse.set_stop_words("stopwords.txt")
tags = jieba.analyse.extract_tags(content, topK=10)
print(",".join(tags))
```

### 3.4. 词性标注

- jieba.posseg.POSTokenizer(tokenizer=None) 新建自定义分词器，tokenizer参数可指定内部使用的 jieba.Tokenizer 分词器。 jieba.posseg.dt 为默认词性标注分词器。
- 标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法。
- 用法示例

```python
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))
# ----output--------
我 r
爱 v
北京 ns
天安门 ns
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116165024512.png)

## 4. nlp-pyltp

> LTP 是哈工大社会计算与信息检索研究中心历时十年开发的一整套中文语言处理系统。LTP 制定了基于 XML 的语言处理结果表示，并在此基础上提供了一整套自底向上的丰富而且高效的中文语言处理模块 （包括词法、句法、语义等6项中文处理核心技术），以及基于动态链接库（Dynamic Link Library, DLL）的应用程序接口，可视化工具，并且能够以网络服务（Web Service）的形式进行使用。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116214442582.png)

```python
# 分句子
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')  # 分句
print ('\n'.join(sents))
```

```python
#---- 分词
import os
from pyltp import Segmentor
LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
word1 = segmentor.segment('中信建投证券投资有限公司')  # 分词
word2 = segmentor.segment('中信今天投资了一款游戏')  # 分词
print(type(word1))
print ('\t'.join(word1))
print ('\t'.join(word2))
segmentor.release()  # 释放模型
```

```python
# 词性标注
import os
LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

from pyltp import Postagger
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

word1 = ["中信建投","证券","投资","有限公司"]
word2 = ["中信","	今天","投资","了","一款","游戏"]

postags1 = postagger.postag(word1)  # 词性标注
postags2 = postagger.postag(word2)  # 词性标注

print ('\t'.join(postags1))
print ('\t'.join(postags2))

postagger.release()  # 释放模型
```

```python
#--   命名实体识别
import os
LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型
word1 = ["中信建投","证券","投资","有限公司"]
word2 = ["中信","今天","投资","了","一款","游戏"]
postags1  = ["j","n","v","n"]
postags2 = ["j","nt","v","u","m","n"]
netags1 = recognizer.recognize(word1, postags1)  # 命名实体识别
netags2 = recognizer.recognize(word2, postags2)  # 命名实体识别
print('\t'.join(netags1))
print('\t'.join(netags2))
recognizer.release()  # 释放模型
```

> `依存句法` (Dependency Parsing, DP) 通过分析语言单位内成分之间的依存关系揭示其句法结构。 直观来讲，依存句法分析识别句子中的`“主谓宾”、“定状补”`这些语法成分，并分析`各成分之间的关 系`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116215205218.png)

```python
# 1）依存句法分析
import os
LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

from pyltp import Parser
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型

words = ['中信建投', '证券', '投资', '有限公司',"今天","投资","了","一款","雷人","游戏"]
postags = ["j","n","v","n","nt","v","u","m","n","n"]
arcs = parser.parse(words, postags)  # 句法分析
print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
parser.release()  # 释放模型

# 2）语义角色标注
import os
LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
from pyltp import SementicRoleLabeller
labeller = SementicRoleLabeller() # 初始化实例
labeller.load(srl_model_path)  # 加载模型
words = ['中信建投', '证券', '投资', '有限公司',"今天","投资","了","一款","雷人","游戏"]
postags = ["j","n","v","n","nt","v","u","m","n","n"]
# arcs 使用依存句法分析的结果
roles = labeller.label(words, postags, arcs)  # 语义角色标注
# 打印结果
for role in roles:
    print (role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
labeller.release()  # 释放模型
```

> `语义角色标注 (Semantic Role Labeling, SRL) `是一种浅层的语义分析技术，标注句子中某些短语为给定谓词的论元 (语义角色) ，如施事、受事、时间和地点等。其能够对问答系统、信息抽取和机器翻译等应用产生推动作用。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116215730714.png)

> `语义依存分析 (Semantic Dependency Parsing, SDP)`，分析句子各个语言单位之间的语义关联，并将语义关联以依存结构呈现。 使用语义依存刻画句子语义，好处在于不需要去抽象词汇本身，而是通过词汇所承受的语义框架来描述该词汇，而论元的数目相对词汇来说数量总是少了很多的。语义依存分析目标是跨越句子表层句法结构的束缚，直接获取深层的语义信息。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201116220050224.png)

##  5. 学习链接

- http://www.ltp-cloud.com/intro#dp_how
- http://www.ltp-cloud.com/document2#api2_python_interface
- [技术博客](http://www.ltp-cloud.com/blog/)

- [Stanfordnlp](http://nlp.stanford.edu:8080/ner/process)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stanfordnlp_record/  

