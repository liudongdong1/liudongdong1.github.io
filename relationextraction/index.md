# RelationExtraction


## 1. Challenges

- **数据规模问题：**人工精准地标注句子级别的数据代价十分高昂，需要耗费大量的时间和人力。在实际场景中，面向数以千计的关系、数以千万计的实体对、以及数以亿计的句子，依靠人工标注训练数据几乎是不可能完成的任务。
  - 远程监督：如果两个实体在知识图谱中被标记为某个关系，那么我们就认为同时包含这两个实体的所有句子也在表达这种关系。再以（清华大学，位于，北京）为例，我们会把同时包含“清华大学”和“北京”两个实体的所有句子，都视为“位于”这个关系的训练样例。
  - 构建了一套强化学习机制来筛除噪音数据。
- **学习能力问题：**在实际情况下，实体间关系和实体对的出现频率往往服从`长尾分布，存在大量的样例较少的关系或实体对`。神经网络模型的效果需要依赖大规模标注数据来保证，存在”举十反一“的问题。如何提高深度模型的学习能力，实现”`举一反三`“，是关系抽取需要解决的问题。
  - 少次学习方法包括度量学习（Metric learning）、元学习（Meta learning）、参数预测（Parameter prediction）等，评测表明即使是效果最佳的原型网络（Prototypical Networks）模型，在少次关系抽取上的性能仍与人类表现相去甚远。
- **复杂语境问题：**现有模型主要`从单个句子中抽取实体间关系，要求句子必须同时包含两个实体`。实际上，`大量的实体间关系往往表现在一篇文档的多个句子中，甚至在多个文档中`。如何在更复杂的语境下进行关系抽取，也是关系抽取面临的问题。、
  - 文档级关系抽取
- **开放关系问题：**现有任务设定一般`假设有预先定义好的封闭关系集合`，将`任务转换为关系分类问题`。这样的话，文本中蕴含的实体间的新型关系无法被有效获取。如何利用深度学习模型自动发现实体间的新型关系，实现开放关系抽取，仍然是一个”开放“问题。

## 2. Paper Record

**keyword**:

- NER（named entities recognizers)

------

### 2.1. Paper:  review of RE

<div align=center>
<br/>
<b>A Review of Relation Extraction
</b>
</div>

#### Research Objective

  - **Application Area**: question answering， biotext mining

#### Supervised Methods

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125133919042.png)

- difficult to extend to new entity-relation types for want of labeled data;
- extensions to higher order entity relations are difficult as well;
- computationally burdensome and don't scale well with increasing amounts of input data;
- require pre-processed input data in the form of parse tree, dependency parse trees etc.

##### Feature based methods:

> Given a set of positive and negative relation examples, syntactic and semantic features can be extracted from the text,`deciding whether the entities in a sentence are related or not`.

- syntactic features:
  - the entities themselves;
  - the types of two entities;
  - word sequence between the entities;
  - number of words between the entities;
  - path in the parse tree containing the two entities;
- semantic cues:
  - the path between the two entities in the dependency parse;

##### Kernel Methods

>  Given two strings x and y, the `string-kernel `computes their` similarity `based `on the number of subsequences that are common to both of them.`

- **Bag of features Kernel**

> . A sentence s = w1, ..., e1, ..., wi , ..., e2, ..., wn containing related entities e1 and e2 can be described as s = sb e1 sm e2 sa. Here sb, sm and sa are portions of word-context before, middle and after the related entities respectively. Now given a test sentence t containing entities e 0 1 and e 0 2 , the similarity of its before, middle and after portions with those of sentence s is computed using the sub-sequence kernel based on (4)

- **Tree Kernels**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125134001281.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125134158391.png)

#### Semi-Supervised

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125134900608.png)

- ##### DIPRE

> s. Assume that our current seed set has only `one seed (Arthur Conan Doyle, The Adventures of Sherlock Holmes)`. The system crawls the Internet to look for pages containing both instances of the seed. To learn `patterns DIPRE uses a tuple of 6 elements [order , author , book, prefix , suffix , middle]` where order is `1 if the author string occurs before the book string and 0 otherwise`, `prefix and suffix` are strings contain the 10 characters occurring to the left/right of the match, middle is the string occurring between the author and book

- **Snowball**

> The task is to identify (organization, location) relation on regular text. Snowball also starts with a seed set of relations and attaches a confidence of 1 to them. Snowball represents each tuple as a vector and uses a similarity function to group tuples. `Snowball extracts tuples in form [prefix , orgnization, middle, location, suffix ]`. Prefix , suffix , and middle are feature vectors of tokenized terms occurring in the pair.
>
> work pipline: The system matches each relation candidate with all patterns and only keeps candidates that have similarity score greater than some threshold. Next, Snowball assigns a high confidence score for a relation candidate when the candidate is matched by many tuples with high similarity to pattern that the system confident in. Finally, the new relation is added to the seed set and the process is repeated iteratively

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125135623816.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125135738664.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125135819932.png)

- **KnowItAll**

> When instantiated for a particular relation, these generic patterns yield relation-specific extraction rules which are then used to learn domain-specific extraction rules.

- **Text Runner**

>  TextRunner learns the relations, classes, and entities from the text in its corpus in a self-supervised fashion.TextRunner 算法主要分为三个过程，
> 1）self-Supervised Learner 自监督学习过程，目标是构建样集合分类器。构建样本集的大致为：先通过一个自然语言分析器提取出样本中存在的三元组，指定一些规则对这些三元组标记为正或者负。标记完成后利用这些数据训练出一个分类器，对三元组打分，该阶段称为训练阶段。
> 2） Single-Pass Extractor: 抽取阶段，对待抽取句子进行浅层语义分析，标注其中的词性和名词短语。对名词短语，如果满足某种条件，则作为候选三元组，之后利用第一步的分类器进行分类，打分。过滤出可信的三元组，同时记录其出现的频次。
> 3）Redundancy-based Assesor , 基于冗余的评价器，对抽取的三元组进行评价，进一步提高效果

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201125140820963.png)

#### Beyond Binary Relations

- started by recognizing binary relation instances that appear to be arguments of the relation of interest;
- extracted binary relations can be treated as the edges of graph with entity mention as nodes;
- reconstruct complex relations by making tuples from selected maximal cliques in the graph.



测试数据集合SemEval-2010 Task-8 

![image-20201123160420523](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201123160420523.png)

## 3. Project

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201123163752444.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201123163856515.png)

学习资料：

- https://www.jianshu.com/p/4c3fc1f0b5f1
- project: [Entity and Relation Extraction Based on TensorFlow](https://github.com/yuanxiaosc/Entity-Relation-Extraction)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/relationextraction/  

