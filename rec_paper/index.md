# Rec_paper


> Liu, Hongtao, et al. "NRPA: Neural Recommendation with Personalized Attention." *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2019.

------

# Paper: NRPA

<div align=center>
<br/>
<b>NRPA: Neural Recommendation with Personalized Attention</b>
</div>


#### Summary

1. propose a neural recommendation approach with personalized attention to learn personalized representations of users and items from reviews, `to select different important words and reviews for different users/items.`
2. `review encoder `to learn representations of reviews from words and `user/item encoder` to learn representations of  users or items from reviews.

#### Research Objective

  - **Application Area**: learn user's interests and hobbies based on their historical behavior records, and predict users preference or ratings for items;

#### Proble Statement

- different users have different preference and different items have different characteristics, the same words or the similarity reviews may have different informativeness for different users and items.

previous work:

- `Collaborative Filtering(CF)`:  many based on matrix-factorization that decomposes the use-item rating matrix into two matrices corresponding to latent features of users and items.
  - only based on numeric ratings, sparsity.  -> using text reviews to model user preference and item features.
- `ConvMvMF`: integrates cnn into probablistic matrix factorization to exploit both ratings and item description documents.
- `TARMF`： using `attention-based rnn `to extract topical information from reviews and integrates textual features into probabilistic matrix factorization.
- `PMF [5]` ： models the latent factors for users and items by `introducing Gaussian distribution`. 
- ` CTR [7] `:     learns interpretable latent structure from user generated content to integrate probabilistic modeling into collaborative filtering. 
- ` ConvMF+ [2] `:  incorporates convolutional neural network into Matrix Factorization to `learn item features from item review documents`. 
- ` DeepCoNN [10]`:   models users and items via `combining all their associated reviews by convolutional neural network`. 
- `NARRE [1]` is a newly proposed method that introduces neural `attention mechanism` to build the recommendation model and `select highly-useful reviews simultaneously`. 

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212200315957.png)

- **Review encoder:** using cnn to extract semantic features of reviews from words, and use personalized word-level attention to select more important words in a review for each user/item.
  - User: $U$;  Item: $I$; rating matrix: $D \in R^{|U|*|I|}$;  text review collection: $D \in R^{|U|*|I|}$; single review written by user u for item i: $d_{u,i}=\{w_1,w_2,...,w_t\}$; 
  - utilize word embedding to map each word into low-dimensional vectors; 
    - transform the review $d_{u,i}$ into a matrix $M_{u,i}=[w_1,...,w_t]$, $w_k$ via word embeddign of word $w_k$;
  - using CNN to extract the semantic features of text reviews;
    - K is the number of filters and $W_j$ is the weight matrix of the j-th filter. feature matrix of the review $C\in R^{k*T}$, each column in C represents the semantic feature of the k-th word in the review.
    - ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212201708010.png)
  - Personalized Attention Vector over Word Level;
    - represent all users and items into low-dimensional vectors via an embedding layer based on their IDs($u_{id}$;![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212202027350.png)
  - user-specific attention over word  level
    - not all words of a review are equally important for the representation of the review meaning.![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212202705873.png)
    - A is the harmony matrix in attention, $q_w^u$ : is the attention vector for user u obtained , $z_k$: is the representations of the k-th word above, $\alpha_i$ is the attention weight of the i-th word in the review. and obtain the `representation of the i-th review of the user u` via aggregating feature vectors of all words：![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212203041219.png)
- **user/item encoder**: apply personalized review-level attention to learn the user/item representation via aggregating all the reviews representations according to their weights.
  - Personalized attention vectors over review level: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212203319737.png)
  - User-specific attention over review level: 
    - review set: $d_u=\{d_{u,1},...,d_{u,N}\}$; 
    - apply attention to highlight those informative reviews and de-emphasize those meaningless. To compute the weight $\beta_j$ of the j-th review of the i-th user:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212203619411.png)
    - obtain the text feature $Pu$ of user u :![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212203633841.png)
- **Rating Prediction:** predict ratings based on the Pu, Pi, and concatenate them feed into FM;![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212203833443.png)

> Wu, C., Wu, F., An, M., Huang, J., Huang, Y., & Xie, X. (2019). Neural news recommendation with attentive multi-view learning. *arXiv preprint arXiv:1907.05576*.

------

# Paper: attentive multi-view 

<div align=center>
<br/>
<b>Neural news recommendation with attentive multi-view learning</b>
</div>


#### Summary

1. learn informative representations of users and news by exploiting different kinds of news encoder and a user encoder.
2. news encoder:
   - an attentive multi-view learning model to learn unified news representations from titles, bodies and topic categories by regarding them as different views of news
   - apply both word-level and view level attention mechanism to news encoder to select important words and views for learning informative news representations.
3. user encoder:
   - learn the representation of users based on their browsed news and apply attention mechanism to select informative news for user representation learning.

#### Research Objective

  - **Application Area**:
- **Purpose**:  

#### Proble Statement

- how to learn representations of new and users.
- a new article usually contains different kinds of information.
- different kinds of news information have different characteristics;
- different news information may have different informativeness for different news. and different words in the same news may have different importance.

previous work:

- Okura et al.: learn news representations from body of news articles via `auto-encoders` and then learn user representation from `news representations by applying GRU to their browsed news`
- Wang et al.: learn news representation via a `konwledge-aware CNN from the titles of news articles` and then learn user representations from news representations `based on the similarity between candidate news and each browsed news`.

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212205816231.png)

- **News Encoder:** learn representation of news from different kinds of news information, such as titles, bodies and topic categories.
  - title encoder:  
    - word embedding to converted word sequence of title into a sequence of word vectors$[e_1^t,...,e_M^t]$;
    -  CNN to learn contextual word representations by capturing their local contexts. and the sequence of contextual word representations defined: $[c_1^t,...,c_M^t]$;![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212210418225.png)
    - word-attention network: to select important words within the context of each news title, the attention weight of the $i_{th}$ word in a news title as $\alpha_i^t$: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212210743107.png)
    - summation of the contextual representations of its words weighted by their attention weights: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212210857061.png)
  - Body encoder:  word embedding; CNN, attention network;  and the workflow like above handle;
  - category encoder: incorporate both the category and subcategory information for news representation learning;
    - input: the ID of the category and the ID of the subcategory;![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212211159499.png)
  - attentive pooling: 
    - propose a view-level attention network to model the informativeness of different kinds of news information for learning news representations.![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212211542126.png)
- **User Encoder:**  apply a news attention network to learn more informative user representations by selecting important news. the attention weight of $i_{th}$ news browsed by a user $\alpha_i^n$ as : ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212211752267.png)
  - user presentation: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212211827628.png)
- **Click Predictor: **
  - the representation of a candidate news $D^c$ as $r_c$;
  - the representation of user u as u;
  - the click probability score $y=u^Tr_c$;
- **Model Training: **

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212212253683.png)

> Wu, F., Qiao, Y., Chen, J. H., Wu, C., Qi, T., Lian, J., ... & Zhou, M. (2020, July). Mind: A large-scale dataset for news recommendation. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 3597-3606).

------

# Paper: MIND

<div align=center>
<br/>
<b>MIND: A Large-scale Dataset for News Recommendation</b>
</div>


#### Summary

1. present a large-scale dataset named MIND for news recommendation;
2. contains 1 million users and more than 160k English news articles, each of which has rich textural content such as title, abstract and body;
3. both effective text representation methods and pre-trained language models can contribute to the performance improvement of news recommendation.
4. appropriate modeling of user interest is also useful;

#### Proble Statement

- news articles on news websites update very quickly;
- news articles contain rich textural information such as title and body;
- there is no explicit rating of news articles posted by users on news platforms.
- news is a common form of texts, and text modeling techniques such as CNN and Transformer can be naturally applied to represent news articles;
- learning user interest representation from previously clicked news article has similarity with learning document representation from its sentences.
- news recommendation can be formulated as a special text matching problem.

previous work:

- Amazon dataset for product recommendation;
- MovieLens dataset for movie recommendation;

#### Common Methods

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212232033298.png)

##### General Reconmendation

- **LibFM(Rendle, 2012):** based on factorization machine; userId, newsId, features;
- **DSSM(huang et al 2013):** uses tri-gram hashes and mutliple feed-forward neural networks for `query-document matching`;
- **Wide&Deep(cheng et al 2016):** using linear transformation channel and deep neural network channel, and use same contend features of users and candidate news for both channels;
- **DeepFM**

##### News Recommendation

- **DFM(Lian et al.2018):** `deep fusion model`, uses an `inception network` to combine nn with different depths to capture the complex interactions between features;
- **GRU(Okura et al 2017):** uses `auto-encoder` to learn latent news representations from news content and use `GRU to learn user representations from the sequence of clicked news`;
- **DKN(wang et al 2018):** `knowledge-awar`e news recommendation method, uses CNN to learn news representations from news titles with `both word embeddings and entity embeddings(infered from knowledge graph)`, and learns user representation based on the `similarity between candidate news and previous clicked news`;
- **NPA(wu et al, 2019b):** nn with `personalized attention mechanism` to select important words and news articles based on user preferences to learn more informative news and user representations;
- **NAML(wu et al.2019):** nn with `attentive multi-view` to incorporate different kinds of news information into the representation of new articles;
- **LSTUR(An et al 2019):** nn with `long- and short-term user interest`s, models `short-term user interest from recently clicked news with GRU` and model long term user interest from the whole click history;
- **NRMS(Wu et al.2019):** use `multi-head self-attention` to learn news representations from the words in news text and learn user representation from previously clicked news article;

##### News Understanding

- **Representation Methods**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212234510899.png)

- **Pre-trained Language Models**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212234345004.png)



- **Different News Information**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212234622253.png)

- **User Interest Modeling**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212234657757.png)

> Graham, S., Min, J. K., & Wu, T. (2019, September). Microsoft recommenders: tools to accelerate developing recommender systems. In *Proceedings of the 13th ACM Conference on Recommender Systems* (pp. 542-543).

------

# Paper: Microsoft recommenders

<div align=center>
<br/>
<b>Microsoft recommenders: tools to accelerate developing recommender systems</b>
</div>


#### Summary

1. reduce the time involved in developing recommender systems.
2. accelerate the process of designing, evaluating and deploying recommender systems.
3. Recommender Utilities:
   - **Common Utilities**: defining constants, helper functions for managing aspects of different frameworks, dealing with Pandas DataFrames, TensorFlow or timing algorithm performance;
   - **Dataset**: splitting data; validation based on random sampling,.
   - **Evaluation**: calculating common metrics;
   - **Tuning:** Hyperparameter tuning tools;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212235443285.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201212235515215.png)

> Wang, H., Zhang, F., Xie, X., & Guo, M. (2018, April). DKN: Deep knowledge-aware network for news recommendation. In *Proceedings of the 2018 world wide web conference* (pp. 1835-1844).

------

# Paper: DKN

<div align=center>
<br/>
<b>DKN: Deep knowledge-aware network for news recommendation</b>
</div>


#### Summary

1. propose DKN that incorporates knowledge graph representation into news recommendation.
2. DKN is a content-based deep recommendation frame-work for click-through rate prediction.
   1. multi-channel and word-entity-aligned knowledge-aware convolutional neural network that fuse semantic-level and knowledge-level representations of news.
   2. KCNN treats words and entities as multiple channels, and keeps their alignment relationship during convolution.
   3. use attention module in DKN to dynamically aggregate a user's history with respect to current candidate news.

#### Proble Statement

- news languages is highly condensed, full of knowledge entities and common sense;
- news are highly time-sensitive and their relevance expires quickly within a short period.
- people are topic-sensitive in news reading as they are usually interested in multiple specific news categories.
- news language is usually highly condensed and comprised of a large amount of knowledge entities and common sense.

**previous work:**

- traditional semantic models of topic models can only find their relatedness based on co-occurrence or clustering structure of words, ignoring the latent knowledge-level connection.

- **Knowledge Graph:**

  - academic knowledge graphs such as NELL, DBpedia;
  - commercial knowledge graphs: Google Knowledge graph, microsoft satori;
    - machine reading,  text classification; word embedding;

- **Knowledge graph embedding:**  translation-based methods

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213090211799.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213091159628.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213091303062.png)

- **CNN representation**

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213091353118.png)

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213091656629.png)

#### Methods

- **problem statement:** 

> for a given user i in the online news platform, denote his click history as $\{t_i^i,t_2^i,...,t_N^i\}$;     $t_j^i$ is the title of the j-th news clicked by user i, and N is the total number of user i's clicked news. Each news title t is composed of a sequence of words, t=[w1,w2,...], and each word w may be associated with an entity e in the knowledge graph;
>
> Given users' click history and connection between words in news titles and entities in the knowledge graph, to predict whether user i will click a candidate news $t_j$ that has not seen before;

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213092412404.png)

- **Knowledge Distillation:** 
  - distinguish knowledge entities in news content, using  [entity linking](http://nlpprogress.com/english/entity_linking.html) to disambiguate mentions in texts by associating them with predefined entities in knowledge graph;
  - construct a sub-graph and extract all relational links among them from the original knowledge graph;
  - expand the knowledge sub-graph to all entities within one hop of identified ones;
  - using graph embedding methods to learn the entity embeddings;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213092613584.png)

- **Extracting additional contextual information for each entity**

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213093326382.png)

  - given the context of entity e, the context embedding is calculated as the average of its contextual entities:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213093511606.png)

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213093531326.png)

- **Knowledge-aware CNN**

  - the concatenating strategy breaks up the connection between words and associated entities and is unaware of their alignment;

  - word embeddings and entity embeddings are learned by different methods, not suitable to convolute them together in a single vector space;

  - the optimal dimensions for word and entity embeddings may differ from each other;

  - propose a multi-channel and word-entity-aligned KCNN for combining word semantics and knowledge information;

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094058038.png)

  - **Attention-based User Interest Extraction:**

      - a user's interest in news topics may be various, and user's clicked items are supposed to have different impacts on the candidate news $t_j$;

        ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094359078.png)

    - the embedding of user i with respect to the candidate news $t_j$ : ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094500760.png)

    - given user i embedding e(i), and candidate news tj embedding e(tj), the probability of user clicking news tj is predicted: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094617934.png)

#### Experiment

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094706080.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094737690.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213094820735.png)

> He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. *arXiv preprint arXiv:2002.02126*.

------

# Paper: LightGCN

<div align=center>
<br/>
<b>LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation</b>
</div>


#### Summary

1. existing work that adapts GCN to recommendation lacks thorough ablation analyses on GCN, which is originally designed for graph classification tasks and equipped with many neural network operations.
2. simplify the design of GCN to make it more concise and appropriate for recommendation, propose LightGCN, including only the most essential component in GCN-neighborhood aggregation for collaborative filtering.
3. LightGCN learns user and item embeddings by linearly propagating them on the user-item and interaction graph, and uses the weighted sum of the embeddings learned at all layers as the final embeddings;
4. empirically show that two common design in GCN, feature transformation and nonlinear activation, have no positive effect on the effectiveness of collaborative filtering;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201213100910274.png)



> Zhou, K., Wang, X., Zhou, Y., Shang, C., Cheng, Y., Zhao, W. X., ... & Wen, J. R. (2021). CRSLab: An Open-Source Toolkit for Building Conversational Recommender System. *arXiv preprint arXiv:2101.00939*.

----

# Paper:  CRSLab

<div align=center>
<br/>
<b>CRSLab: An Open-Source Toolkit for Building Conversational Recommender System</b>
</div>

#### Summary

1. existing studies on CRS vary in scenarios, goals and techniques, lacking unified, standardized implementation or comparison.
2. an open-source CRS toolkit [CRSLab](https://
   github.com/RUCAIBox/CRSLab), providing a unified and extensible framework with highly-decoupled modules to develop CRSs.
3. unify the task description of existing works for CRS into `three sub-tasks,  namely recommendation, conversation and policy`, covering the common functional requirements of mainstream CRSs;
   1. given the dialog context(i.e. historical utterances) and other useful side information(e.g. interaction history and knowledge graph); 
   2. predict user-preferred items (recommendation);
   3. generate a proper response (conversation);
   4. select proper inter-active action (policy);

#### System framework:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112201103585.png)

- **Configuration:** select & modify the experiment setup ( dataset, model, and hyperparameters).

- **Data Modules:** raw public dataset->preprocessed dataset -> dataset -> dataloader -> system;

  - data preprocessing: 

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112201518616.png)

  - Dataset Class: focuses on processing the input data into a unified format;
  - DataLoader Class:   reformulates data for supporting various models.

- ​	**Model Modules: **

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112202023902.png)

- **System Class:**
  
  - aims to set up models for accomplishing the CRS task, `distribute the tensor data from dataloader to corresponding models,` `train the models with proper optimization strategy` and conduct `evaluation with specified protocals`.
- **Evaluation Modeules: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112202505598.png)**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210112202555907.png)



# TodoList:

- [ ] https://github.com/RUCAIBox/CRSLab 学习使用，设计代码复用

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/rec_paper/  

