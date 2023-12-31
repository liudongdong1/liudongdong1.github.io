# VisionNLPCommend


**level**: CCF_A  CVPR
**author**: Amaia Salvador1(FaceBook Al Research)
**date**: 2019
**keyword**:

- image understanding; information retrieval

> Salvador, Amaia, et al. "Inverse cooking: Recipe generation from food images." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019.

------

# Paper: Inverse Cooking

<div align=center>
<br/>
<b>Inverse Cooking: Recipe Generation from Food Images
</b>
</div>

#### Summary

1. introduce an inverse cooking system that recreates cooking recipes given food images.
2. predicts ingredients as sets by means of a novel architecture, modeling their dependencies without imposing any order, and then generates cooking instructions by attending to both image and its inferred ingredients simultaneously.
3. for dataset constraints, generates a cooking recipe containing a title, ingredients, and cooking instructions directly from image.

#### Research Objective

  - **Application Area**: information retrieval, food image understand and recommendation.
- **Purpose**:  recognize the type of meal or its ingredients, but also understand its preparation process.

#### Proble Statement

- food and its components have high intraclass variability and present heavy deformations that occur during cooking process.
- ingredients are frequently occluded in a cooked dish and come in a variety of coloers, forms and textures;
- visual ingredient detection requires high level reasoning and prior knowledge.

previous work:

- **Food Understanding:** Food-101, Recipe1M datasets; with focus in image classification, estimating the number of calories given a food image, estimating food quantities, predicting the list of present ingredients and finding the recipe for a given image.
  - [34] provides a detailed corss-region analysis of food recipes, considering images, attributes and recipes, considering images, attributes and recipe ingredients.

#### Methods

- **Problem Formulation**: generating a recip

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200821222228.png)

【Module One】Generating recipes from images

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200822110938.png)

#### Evaluation

#### Conclusion

- present an inverse cooking system, which generates cooking instructions conditioned on an image and its ingredients, exploring different attention strategies to reason about both modalities simultaneously.
- exhaustively study ingredients as both a list and a set, and propose a new architecture for ingredient prediction that exploits co-dependencies among ingredients without imposing order.
- ingredient prediction is indeed a difficult task and demonstrate the superiority of our proposed system against image-to-recipe retrieval approaches.

#### Notes <font color=orange>去加强了解</font>

- [ ] 论文关键部分没有看完，看不懂

**level**:  
**author**: Valentin Gabeur, Chen Sun(Google Research)
**date**: 2020
**keyword**:

- retrieval, caption-to-video, video-to-caption

> Gabeur, Valentin, Chen Sun, Karteek Alahari, and Cordelia Schmid. "Multi-modal Transformer for Video Retrieval." *arXiv preprint arXiv:2007.10639* (2020).

------

# Paper: Multi-modal Transformer

<div align=center>
<br/>
<b>Multi-modal Transformer for Video Retrieval
</b>
</div>
#### Summary
1. introduce a novel video encoder architecture for retrieval: the multi-modal transformer process effectively multiple modality features extracted at different times;
2. thoroughly investigate different architectures for languages embedding, and show the superiority of the BERT model for the task of video retrieval;
3. outperform prior state of the art for the task of video retrievval on MSRVTT[30], ActivityNet[12], and LSMDC[21] datasets, and winning solution in the CVPR 2020 video Pentathlon Challenge[4];
#### Proble Statement
- most of the existing methods for this caption-to-video retrieval problem don't fully exploit cross-modal cues present in video;
- most aggregate per-frame visual features with limited or no temporal information;
![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200906191856.png)
previous work:
- **Language representations:** Word2Vec[18], LSTM[7], BERT[3]
- **Video representations:** S3D, VideoBERT, CBT;
- **Visual-language retrieval:** 
  - Harwath [5] perform image and audio-caption retrieval by emdedding audio segments and image regions in the same space and requiring high similarity between each audio segment and its corresponding image region;
  - **JSFusion[13]:**  estimates video-caption similarity through dense pairwise comparisons between each word of the caption and each frame of the video;
  - Zhang[33] perform paragraph-to-video retrieval by assuming a hierarchical decomposition of the video and paragraph.
  - <font color=red>don't pre-process the sentences but encode them directly through BERT</font>;

#### Methods

- **Problem Formulation**:
  - how to learn accurate representations of both caption and video to base our similarity estimation on?
  - video data varies in terms of appearance, motion, audio, overlaid text, and speech,etc;

> given a dataset of $n$ video-caption pairs ${(v1, c1), ...,(vn, cn)}$, the goal of the learnt similarity function $s(vi , cj )$, between video $vi$ and caption $cj$ , is to provide a high value if $i = j$, and a low one if $i != j$. 

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200906192207.png)

**【Module one】Video Representation**
$$
\Omega(v)=F(v)+E(v)+T(v)
$$

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201016164807866.png)

**【Module two】Caption Representation**: obtain an embedding h(c) of the caption, and then project it with a function g into N different spaces as $\varphi=g*h$;
$$
\varphi(c)=\{\varphi\}_{i=1}^N
$$
**【Module three】Similarity estimation**
$$
s(v,c)=\sum_{i=1}^Nw_i(c)(\varphi^i,\psi_{agg}^i)\\
w_i(c)=e^{h(c)^Ta_i}/\sum^N_{j=1}e^{h(c)^Ta_j}
$$

#### Notes <font color=orange>去加强了解</font>

  -  http://thoth.inrialpes.fr/research/MMT
  - similarity learning[29]

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/visionnlpcommend/  

