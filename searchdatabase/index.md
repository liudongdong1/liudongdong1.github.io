# SearchDatabase


> Lux, Mathias, and Savvas A. Chatzichristofis. "Lire: lucene image retrieval: an extensible java cbir library." *Proceedings of the 16th ACM international conference on Multimedia*. 2008. cited by 396.

------

# Paper: LIRe

<div align=center>
<br/>
<b>LIRe: Lucene Image Retrieval - An Extensible Java CBIR
Library
</b>
</div>


#### Summary

1. LiRe(Lucence Image Retrieval) is a light weight open source java library for content based image retrival.
2. provides common and state of the art global image features and offers means for indexing and retrieval.
3. images features:
   - color histograms in RGB and HSV space;
   - MPEG-7 descriptors scalable color, color layout and edge histogram;
   - the Tamura texture features coarseness, contrast and directionality;
   - color and edge directivity descriptor, CEDD;
   - Fuzzy color and texture histogram, FCTH;
   - Auto color correlation feature defined by Huang et.
4. `Indexing`: the signatures or vectors extracted by the feature implementations are wrapped int the documents as text, and add to the lucene index;
5. `Search`:  ImageSearcher either takes the given query feature or extracts the feature from a query image, then reads documents from the index sequentially and compares them to the query image.

> Yang, Peilin, Hui Fang, and Jimmy Lin. "Anserini: Enabling the use of Lucene for information retrieval research." *Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2017.

------

# Paper: Anserini

<div align=center>
<br/>
<b>Anserini: Enabling the Use of Lucene
for Information Retrieval Research
</b>
</div>


#### Summary

1. Lucene can handle heterogeneous web collections at scale, but lacks systematic support for evaluation over standard test collections.
2. introduces Anserini, a new information retrieval toolkit that aims to provide  the best of both worlds, to better align information retrieval practice and research.
3. focused on `scalable, multi-threaded inverted indexing` to handle modern web-scale collections, streamlined IR evaluation for ad hoc retrieval on standard test collecitions, and an extensible architecture for multi-stage ranking;
4. **Multi-threaded indexing(wrapper):**  lucene only provides access to a collection of indexing components that researchers need to assemble together to build and end-to-end indexer, eg. write from scatch custom document processing pipelines, code for managing individual indexing threads, and implementations of load balancing and synchronization procedures.
5. **Streamlined IR evaluation:** parsers for different query formats, a unified driver program for ad hoc experiments that outputs standard trec_eval format.

### 2. 学习资源

- [开源搜索引擎介绍](https://my.oschina.net/javaeye/blog/3026578)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/searchdatabase/  

