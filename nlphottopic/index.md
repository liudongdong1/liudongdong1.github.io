# NLPHotTopic


> This week i get a summary knowledge of NLP, and learn some direction for further learning. And in this blog, i will record what i learned this weak by searching some information on Internet, the content is organized as follows: the Preparatory knowledge which need to be master in the following years, and some direction in NLP areas from model sides, application sides and the scene task, and some paper and learning resource recording.

### 0.  Preparatory knowledge

- Probability& Statistics

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201008165645446.png)

- **Machine Learning**![](https://gitee.com/github-25970295/blogImage/raw/master/img/ml.png)

- **Text Mining**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/textmining.png)

- **NLP**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/prob.png)

### 1. Model sides

#### 1.1. Transformers and pre-trained language models

-  “Attention is all you need” ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
- “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805))

> **Theory-proving side:**  ([Shi et al., 2020](https://arxiv.org/abs/2002.06622); [Brunner et al., 2020](https://arxiv.org/abs/1908.04211); [Yun et al., 2019](https://arxiv.org/abs/1912.10077); [Cordonnier et al., 2019](https://arxiv.org/abs/1911.03584)).

> **improving the task performances of Transformers and pre-trained language models:**([Wang et al. 2019](https://arxiv.org/abs/1908.04577); [Lee et al., 2019](https://arxiv.org/abs/1909.11299)).

> **Reducing the size of models or the time of training:**([Wu et al., 2020](https://arxiv.org/abs/2004.11886); [Lan et al., 2019](https://arxiv.org/abs/1909.11942); [Kitaev et al., 2020](https://arxiv.org/abs/2001.04451); [Clark et al., 2020](https://arxiv.org/abs/2003.10555); [Rae et al., 2019](https://arxiv.org/abs/1911.05507); [Fan et al., 2019](https://arxiv.org/abs/1909.11556); [You et al., 2019](https://arxiv.org/abs/1904.00962)).
>
> - Model Compression/Pruning ([Zhang et al., 2019](https://arxiv.org/abs/1912.00120))

#### 1.2. Multilingual/Cross-lingual tasks:

-  ([Karthikeyan et al., 2019](https://arxiv.org/abs/1912.07840); [Berend 2020](https://openreview.net/forum?id=HyeYTgrFPB); [Cao et al., 2020](https://arxiv.org/abs/2002.03518); [Wang et al., 2019](https://arxiv.org/abs/1910.04708))
- Multimodal models([Su et al., 2019](https://arxiv.org/abs/1908.08530))
- [**Cross-Lingual Ability of Multilingual BERT: An Empirical Study**](https://openreview.net/forum?id=HJeT3yrtDr)
- [**On the Relationship between Self-Attention and Convolutional Layers**](https://openreview.net/forum?id=HJlnC1rKPB)

#### 1.3. Reinforcement learning and NLP

- ([Yu et al., 2019](https://arxiv.org/abs/1906.02768); [Clift et al., 2019](https://arxiv.org/abs/1909.00668))

> **Session 4：The Machine Learning in NLP**
>
> -  Learning Sparse Sharing Architectures for Multiple Tasks
>
> -  Reinforcement Learning from Imperfect Demonstrations under Soft Expert Guidance
>
> -  Shapley Q-value: A Local Reward Approach to Solve Global Reward Games
>
> - Measuring and relieving the over-smoothing problem in graph neural networks from the topological view
>
> - Neighborhood Cognition Consistent Multi-Agent Reinforcement Learning
>
> - Neural Snowball for Few-Shot Relation Learning
>
> - Multi-Task Self-Supervised Learning for Disfluency Detection
>
> - Constructing Multiple Tasks for Augmentation: Improving Neural Image Classification With K-means Features
>
> - Graph-propagation based correlation learning for fine-grained image classification
>
> - End-to-End Bootstrapping Neural Network for Entity Set Expansion

### 2. Application sides

#### 2.1. Natural language generation

- Generation of realistic, rhymed and theme based poetry (creative writing)
-  Generation of theme based short stories (creative writing)
- Generation of theme based novels (creative writing)
- Generation of news / short articles based on numerical / audio / video data
- Generation of research papers based on a topic. 

#### 2.2. Natural language understanding

- **Sentiment Analysis** 

> Deriving sentiments in sentences (positive, negative, neutral), and also in articles (though that will be more appropriate like bag of sentence sentiments). The future is to include emotions (attributes) in that, like the attributes now on Facebook posts - Love, Like, Angry, Surprised, Sad, Hilarious. These attributes make a lot more sense for sentiments going forward.

- **Text Summarization（汇总）**

> Summarizing a single or many articles according to a particular theme.

- **Textual entailment（语篇蕴涵）**

>  Inferring directional causal relationships between textual fragments. This can be challenging in a long article.
>
> - Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets
> - Multi-Scale Self-Attention for Text Classification
> -  Learning Multi-level Dependencies for Robust Word Recognition

- **Information Extraction** or **Relationship Extraction** or **Knowledge Graph**

> Find structured information from unstructured data, like entities, relationships, co-reference resolution. This at a basic level is very useful for algorithmic trading. An extension of this is a global form of extracting logic structures (first order and higher order).

- **Topic Segmentation**

> Topic Extraction (with regions). Normally, there will be overlapping regions.

- <font color=red>**Question Answering** or **NLP-based voice assistant**</font>

> Answer the questions to both closed (specific) and open questions (subjective). Answers to subjective questions is the main challenge for the likes of realistic Virtual Assistants.
>
> - Modeling Fluency and Faithfulness for Diverse Neural Machine Translation
> - Minimizing the Bag-of-Ngrams Difference for Non-Autoregressive Neural Machine Translation
> - Neural Machine Translation with Joint Representation
> - Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context
> - A pre-training based personalized dialogue generation model with persona-sparse data
> - Knowledge Graph Grounded Goal Planning for Open-Domain Conversation Generation

- **Parsing**

> Parsing natural language generally in the form a tree. This involves hierarchical segmentation of the language involving the grammar rules.

- **Prediction**

> Given a short text, predict what happens next. The prediction problem is beginning to be targeted in vision, but it has never ever gained paths for realistic products. For closed and deterministic prediction (not innovative else that would fall under the paradigm of creative writing), this can be a useful task for prediction of future events based on past evidences and analysis. This can be then very useful for finance sectors.

- **Part of Speech Tagging(词性标注)**

> Tagging words whether they are nouns, verbs or adjectives.

- **Translation**

> Translate one language to another. This can be very challenging given the nature of the language, and the grammar. Normally, under probabilistic models, this assumes that the underlying grammar is mostly the same, and thus, models normally fail for Sanskrit.

- **Query Expansion**

> Expand query in possible ways for making the search results more meaningful. This is normally an issue with search engines, where people do not know what all keywords (or query sentences) to include to cover the entire gamut of relevancy.

- **Argumentation Mining(论证分析挖掘）**

> Evolving field of NLP, where one wants to analyse discussions and arguments.

- **Interestingness(趣味性挖掘）**

#### 2. 3. NLP and CV

-  **Visual Question Answering**
- **Automated Image Captioning（自动图像字幕）**
- **OCR**

> - DualVD: An Adaptive Dual Encoding Model for Deep Visual  Understanding  in Visual Dialogue
> - Storytelling from an Image Stream Using Scene Graphs

#### 2.4. Voice and NLP

- **speech to text**

> Analysts predict speech recognition technologies will be substantially improved in the near future thanks to natural language processing. This will involve minimization of errors, recognition of what several individuals are saying despite different accents and a noisy environment. 

### 3. Scene task

- Integrated Chatbot

-  Human-to-machine Interaction

> conversing with a machine is as simple as conversing with a human. 

- Company monitoring

> Banks and other monetary organizations can utilize NLP to find and parse client sentiment by checking social media and analyzing discussions about their services and strategies. With the capacity to get to significant, separated data, financial services analysts can compose increasingly definite reports and give better advice to customers and internal decision makers.

- Business intelligence

> getting business intelligence from raw business information, including product information, marketing and sales information, customer service, brand notoriety and the present talent pool of a company. This implies NLP will be the way to moving numerous legacy organizations from data-driven to intelligence-driven platforms, helping humankind rapidly get the insights to make decisions.

> - 搜索是NLP技术最早得到大规模应用的技术，例如百度搜索、知乎话题搜索以及各大互联网公司的query搜索技术，都涉及到语义匹配或文本分类技术。此外，大型的搜索引擎，知识图谱的搭建是必须的。
>
> - 推荐系统在一定层面来说是跟搜索场景相反的。搜索是基于用户的意图，在文本库中寻找匹配项；推荐则相反，通常基于积累的用户信息，给用户推荐可能感兴趣的内容。推荐系统常常涉及用户画像、标签定义等过程，需要一定程度的依赖NLP技术。
>
> - 聊天机器人是目前NLP技术应用最多的场景，基于NLP技术构建一个能够替代客服、销售、办公文员是这一任务的终极目标。目前，聊天机器人已经以各种形态出现在人们面前，有站在银行门口迎接顾客的迎宾机器人，有放在卧室床头的智能音箱，有呆在各个APP首页的助手机器人等等。在聊天机器人中，运用了文本分类、语义匹配、对话管理、实体识别等大量的NLP技术。要做好是一件难度大、超复杂的任务。
>
> -  知识图谱是AI时代一个非常重要基础设施，大规模结构化的知识网络的搭建，能够重塑很多的智能场景。

### 4. Paper & Relative Article

**4.1. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**

> **About:** In this paper, researchers from Carnegie Mellon University and Google Brain proposed a novel neural architecture known as Transformer-XL that enables learning dependency beyond a fixed-length without disrupting temporal coherence. According to the researchers, TransformerXL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation.

**4.2. Bridging The Gap Between Training & Inference For Neural Machine Translation** 

> **About:** This paper is one of the top [NLP papers](https://analyticsindiamag.com/6-top-nlp-papers-from-acl-2019-you-should-read/) from the premier conference, Association for Computational Linguistics (ACL). This paper talks about the error accumulation during Neural Machine Translation. The researchers addressed such problems by sampling context words, not only from the ground truth sequence but also from the predicted sequence by the model during training, where the predicted sequence is selected with a sentence-level optimum. According to the researchers, this approach can achieve significant improvements in multiple datasets. 

**4.3. BERT: Pre-training Of Deep Bidirectional Transformers For Language Understanding**

> BERT by Google AI is one of the most popular language representation models. Several organisations, including Facebook as well as academia, have been researching NLP using this transformer model. BERT stands for Bidirectional Encoder Representations from Transformers and is designed to pre-train deep bidirectional representations from the unlabeled text by jointly conditioning on both left and right context in all layers. The model obtained new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5%, MultiNLI accuracy to 86.7%, and much more.

**4.4. Emotion-Cause Pair Extraction: A New Task To Emotion Analysis In Texts**

> Emotion cause extraction (ECE) is a task that is aimed at extracting the potential causes behind certain emotions in text. In this paper, researchers from China proposed a new task known as emotion-cause pair extraction (ECPE), which aims to extract the potential pairs of emotions and corresponding causes in a document. The experimental results on a benchmark emotion cause corpus that prove the feasibility of the ECPE task as well as the effectiveness of this approach. 

**4.5. Improving Language Understanding By Generative Pre-Training**

> This paper is published by OpenAI, where the researchers talked about natural language understanding and how it can be challenging for discriminatively trained models to perform adequately. The researchers demonstrated the effectiveness of the approach on a wide range of benchmarks for natural language understanding. They proposed a general task-agnostic model, which outperformed discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon state-of-the art in 9 out of the 12 tasks studied. 

**4.6. Neural Approaches To Conversational AI** 

>  This research paper by Microsoft Research surveys neural approaches to conversational AI that have been developed in the last few years. In this paper, the researchers grouped conversational systems into three categories, which are question answering agents, task-oriented dialogue agents, and chatbots. For each category, a review of state-of-the-art neural approaches is presented, drawing the connection between them and traditional approaches, as well as discussing the progress that has been made and challenges still being faced, using specific systems and models as case studies.

**Session 1：翻译、对话与文本生成**

(1) Modeling Fluency and Faithfulness for Diverse Neural Machine Translation

(2) Minimizing the Bag-of-Ngrams Difference for Non-Autoregressive Neural Machine Translation

(3) Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context

(4) A pre-training based personalized dialogue generation model with persona-sparse data

(5) Synchronous Speech Recognition and Speech-to-Text Translation with Interactive Decoding

(6) SPARQA: Skeleton-based Semantic Parsing for Complex Questions over Knowledge Bases

(7) Knowledge Graph Grounded Goal Planning for Open-Domain Conversation Generation

(8) Neural Machine Translation with Joint Representation 

**Session 2：文本分析与内容挖掘**

(9) Multi-Scale Self-Attention for Text Classification

(10) Learning Multi-level Dependencies for Robust Word Recognition

(11) Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets

(12) Cross-Lingual Low-Resource Set-to-Description Retrieval for Global E-Commerce

(13) Integrating Relation Constraints with Neural Relation Extractors

(14) Capturing Sentence Relations for Answer Sentence Selection with Multi-Perspective Graph Encoding

(15) Replicate, Walk, and Stop on Syntax: an Effective Neural Network Model for Aspect-Level Sentiment Classification

(16) Cross-Lingual Natural Language Generation via Pre-Training

**Session 3：知识理解与NLP应用**

(17) Hyperbolic Interaction Model For Hierarchical Multi-Label Classification

(18) Multi-channel Reverse Dictionary Model

(19) Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement

(20) Logo-2K+: A Large-Scale Logo Dataset for Scalable Logo Classification

(21) DMRM: A Dual-channel Multi-hop Reasoning Model for Visual Dialog

(22) DualVD: An Adaptive Dual Encoding Model for Deep Visual  Understanding  in Visual Dialogue

(23) Storytelling from an Image Stream Using Scene Graphs

(24) Draft and Edit: Automatic Storytelling Through Multi-Pass Hierarchical Conditional Variational Autoencoder 

[【NLP-词向量】词向量的由来及本质](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035757%26idx%3D1%26sn%3Dcaaf1d3f78e65a4df46fcffa0720f931%26chksm%3D8712ad90b065248603f8db9fbdc18a19ee2af5900bcc55ddc381833467ff1cb7e15120d504c1%26scene%3D21%23wechat_redirect)

[【NLP-词向量】从模型结构到损失函数详解word2vec](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035906%26idx%3D1%26sn%3D24df0e979ad2761a763c4f073ea92ac2%26chksm%3D8712aaffb06523e933148d4146cc343e12275ae91ea81710b4e2bab3bb6435e8d35b13223445%26scene%3D21%23wechat_redirect)

[【NLP】 聊聊NLP中的attention机制](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649034734%26idx%3D1%26sn%3D78b209c04b3f69387240efa1a904278e%26chksm%3D8712b193b0653885b808090c5c8e96ba4c7dac75fa013b1e4f72ef0027b6035155baae41c397%26scene%3D21%23wechat_redirect)

[【NLP】 理解NLP中网红特征抽取器Tranformer](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649034901%26idx%3D2%26sn%3D5a12aff786df3f305a5a05595fb6b8b8%26chksm%3D8712aee8b06527fee9a62c070313c47067e2bc00cb1a39b19401b4bf8d0e364eb88e28826667%26scene%3D21%23wechat_redirect)

[【NLP】 深入浅出解析BERT原理及其表征的内容](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035055%26idx%3D1%26sn%3Dc49f6919ec8d0fef269f751680819edf%26chksm%3D8712af52b06526443ed01d2ec3bb9d8621ec4ef714b132dfa88020bbda268fdc22ab2e598f78%26scene%3D21%23wechat_redirect)

[【NLP】GPT：第一个引入Transformer的预训练模型](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035554%26idx%3D2%26sn%3D61cbd0046aa055b16dd2e74f6a625a4d%26chksm%3D8712ad5fb06524495d663310836fd222c9e89c002ff778cba7996c90c27ca4f41b85a050cd6b%26scene%3D21%23wechat_redirect)

[【NLP】XLnet：GPT和BERT的合体，博采众长，所以更强](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035407%26idx%3D2%26sn%3De84f0f9f2c7458658514bf9a4e934324%26chksm%3D8712acf2b06525e41d82fbc5a9b60efeca91a0eec1a1f4c96f44800b30d2fe08fb0bf5f67917%26scene%3D21%23wechat_redirect)

[【NLP-NER】什么是命名实体识别？](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036099%26idx%3D1%26sn%3D7671dfd7c4f748c3aa0d12f57956fabf%26chksm%3D8712ab3eb065222862a03a0f18ec62cce6a6a8166656a3477c7c2f492c9749b7b68b75679693%26scene%3D21%23wechat_redirect)

[【NLP-NER】命名实体识别中最常用的两种深度学习模型](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036142%26idx%3D1%26sn%3D00b0a2588b0e4eb1f67f4e0997562c53%26chksm%3D8712ab13b0652205517ecf622982410ab81dd22ff7bc23f0eed2c89525ce95d8cf851000efff%26scene%3D21%23wechat_redirect)

[【NLP-NER】如何使用BERT来做命名实体识别](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036272%26idx%3D1%26sn%3D3ce6800462c6ea8d911909489bef4ed0%26chksm%3D8712ab8db065229bc8ea68f94332be9e03a06eb54b84c42d7e1ce8dc3d366a7e03035d0e3ae1%26scene%3D21%23wechat_redirect)

[【NLP实战系列】Tensorflow命名实体识别实战](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036559%26idx%3D1%26sn%3D97bd5d699ceffd7f5f98b831cc26ec1b%26chksm%3D8712a972b065206481a853852939ba4c7f4e8a713197ca4f2c3b55e98fbeb405af339ea3644c%26scene%3D21%23wechat_redirect)

[【每周NLP论文推荐】 NLP中命名实体识别从机器学习到深度学习的代表性研究](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035102%26idx%3D2%26sn%3D75957ee0aec259c1ada9b9015fc93828%26chksm%3D8712af23b0652635c255bd3e1c58d998e6b01cbd1513d8ce4e40e0a4dc0841560b1710699c54%26scene%3D21%23wechat_redirect)

[【NLP实战系列】朴素贝叶斯文本分类实战](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036470%26idx%3D1%26sn%3Dcc44bc3babdb25b959fb644975382156%26chksm%3D8712a8cbb06521dd17e2848a567b69b91b42baa4d0527186a64e0cbd059576bd99ee40e7ac86%26scene%3D21%23wechat_redirect)

[【NLP实战】基于ALBERT的文本相似度计算](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036836%26idx%3D1%26sn%3Da4e0b73a4ed227b53c305494b848e094%26chksm%3D8712a659b0652f4fab5613d0c2a85da6ee898159d1648e13d5be4fc8484692334bced37d0dd7%26scene%3D21%23wechat_redirect)

[【文本信息抽取与结构化】目前NLP领域最有应用价值的子任务之一](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649037438%26idx%3D1%26sn%3Dc68e8734c19bade085f7a5a23a5401a7%26chksm%3D8712a403b0652d15b3c5d8a721c6a3a838e100118116a53c5d6a443daf05c9d285f950ce956c%26scene%3D21%23wechat_redirect)

[【文本信息抽取与结构化】详聊文本的结构化【上】](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649037522%26idx%3D2%26sn%3D4c3b77627fd6a879d34781476bfd194f%26chksm%3D8712a4afb0652db964a5f7e1de3c927eb99ec0e9f5fad6588d4d1243f356392b4232ac3fc001%26scene%3D21%23wechat_redirect)

[【文本信息抽取与结构化】详聊文本的结构化【下】](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649037655%26idx%3D2%26sn%3Dd83d62d87227e34d9324faeb6d536cd1%26chksm%3D8712a52ab0652c3c6f28e08456ef9f740f2fbd42b4f46ca5f71914103a20b28a8bc88ad2497e%26scene%3D21%23wechat_redirect)

[【文本信息抽取与结构化】详聊如何用BERT实现关系抽取](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649037990%26idx%3D2%26sn%3D76b2b4c32f72aaddfec60a3a04dac90a%26chksm%3D8712a2dbb0652bcd9ae280267ca62bbe1f2c62430fca96534eb853db298efa1dc43f73e8a7bd%26scene%3D21%23wechat_redirect)

[【每周NLP论文推荐】 掌握实体关系抽取必读的文章](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035654%26idx%3D2%26sn%3Df9f8020da1faa66390c424c5faec3260%26chksm%3D8712adfbb06524ed14db1a5e35a62cb1db9ed7cf1b3298771969cf95731b1410ac08740891e2%26scene%3D21%23wechat_redirect)

[【NLP-ChatBot】我们熟悉的聊天机器人都有哪几类？](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036303%26idx%3D1%26sn%3D15d4cd20640fae64535ef5bff08ca1fa%26chksm%3D8712a872b065216466b0be4f44567c8470746c673f4113da0269a14f5342044f93a9b07d3e76%26scene%3D21%23wechat_redirect)

[【NLP-ChatBot】搜索引擎的最终形态之问答系统（FAQ）详述](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036396%26idx%3D1%26sn%3D55370a63f225ae9d734fdc31dad5869f%26chksm%3D8712a811b0652107ef7152df9ee001cc90aafb4db332b88bcb29a5f3ca2c0da04b30250ecbb7%26scene%3D21%23wechat_redirect)

[【NLP-ChatBot】能干活的聊天机器人-对话系统概述](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036508%26idx%3D1%26sn%3Dddd9a454497b7a766ca7246448fe2eb2%26chksm%3D8712a8a1b06521b789ca66143391efef8ba88e243d306ce6a45040213ff31fb2493635f93994%26scene%3D21%23wechat_redirect)

[【每周NLP论文推荐】 对话管理中的标志性论文介绍](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036142%26idx%3D2%26sn%3Df3ce4c20b0827b9f08babfd225a93aa9%26chksm%3D8712ab13b065220512ebf7339b58395e04634081904e6e6fe3a20237051ef6c2a846972f1fb5%26scene%3D21%23wechat_redirect)

[【每周NLP论文推荐】 开发聊天机器人必读的重要论文](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649035491%26idx%3D2%26sn%3D4ec519ce322949e9d5117bbfc7bd074e%26chksm%3D8712ac9eb0652588b7218a396140563553dc399e00136b6cd5403d675f852ef5a56616da3535%26scene%3D21%23wechat_redirect)

[【知识图谱】人工智能技术最重要基础设施之一，知识图谱你该学习的东西](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036786%26idx%3D1%26sn%3Dbf010d6a8c561b80d163f5c51598030f%26chksm%3D8712a98fb06520991297e3ac2b710643ce91d881d38d01cd9ba8ed80055826a1fcc9c14c0383%26scene%3D21%23wechat_redirect)

[【知识图谱】知识表示：知识图谱如何表示结构化的知识？](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036864%26idx%3D1%26sn%3D78c14394b20f80d481004cca5156a776%26chksm%3D8712a63db0652f2b513810ce6190c5bf1a7c98746f7d1188710cfa44d6249272a66aef893451%26scene%3D21%23wechat_redirect)

[【知识图谱】如何构建知识体系：知识图谱搭建的第一步](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649036959%26idx%3D1%26sn%3Ddce0df28080545324e40cabcb5c9e1e6%26chksm%3D8712a6e2b0652ff488e691c1db605570d6f4d888ac516548e843c9324e95e2ad42cb8fc7fe34%26scene%3D21%23wechat_redirect)

[【知识图谱】获取到知识后，如何进行存储和便捷的检索？](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649037188%26idx%3D1%26sn%3D2ac2099c02c1fadd2455a71601d3921f%26chksm%3D8712a7f9b0652eefaa14d21f186bfd0609ae220ed1a30dfc36a9dcc1fea26d4791ef47776553%26scene%3D21%23wechat_redirect)

[【知识图谱】知识推理，知识图谱里最“人工智能”的一段](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649037252%26idx%3D1%26sn%3D668affc58c11e731ad2f2488311c3df4%26chksm%3D8712a7b9b0652eaf980bc40311cd20370e93f49904e08d788a20f51ab2878220a69bd84b3627%26scene%3D21%23wechat_redirect)

### 5. Reference&Learning Resource

- https://github.com/graykode/nlp-roadmap
- <font color=red>[A comprehensive reference for all topics related to Natural Language Processing](https://github.com/ivan-bilan/The-NLP-Pandect )</font>
- [A curated list of resources dedicated to Natural Language Processing](https://github.com/keon/awesome-nlp)

- [YSDA course in Natural Language Processing](https://github.com/yandexdataschool/nlp_course)
- [NLP Learning journey.](https://github.com/makcedward/nlp)
- [NLP(Natural Language Processing) tutorials Pytorch example](https://github.com/lyeoni/nlp-tutorial)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/nlphottopic/  

