# GraphPaper


### 1. DDGK

> Al-Rfou R, Perozzi B, Zelle D. Ddgk: Learning graph representations for deep divergence graph kernels[C]//The World Wide Web Conference. 2019: 37-48.

> - `end-to-end supervised graph classification`: learn a intermediate representation of an entire graph as precondition in order to solve the classification task;
> - `graph representation learning`:  
>   - **feature engineering**:  graph's `clustering coefficient`,` its motif distribution`,` its spectral decomposition ` , limited to composing only known graph
>   - encode algorithmic heuristics from graph isomorphism literation

> DDGK capture the attributes of graphs by usign them as features for several classificaiton problems, capturing the local similarity of graph pairs and the global similarity across families of graphs.
>
> - Deep divergence graph kernel:  learnable kernel does not depend on feature engineering or domain knowledge.
> - Isomorphism Attention: cross-graph attention mechanism to probabilisticly align representations of nodes between graph pairs.

- **Embedding based kernels**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210609150700077.png)

#### .1. Graph Encoding

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210609153508184.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210609153829280.png)

> an encoder is given asingle vertex and it is expected to predict its neighbors

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210609151146576.png)

#### .2. Cross-Graph Attention

> an attention mechanism for aligning graphs basedon a set of encoded graph representations.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210609151931270.png)

#### .3. DDGK

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210609152603353.png)

### Resouces

- https://github.com/benedekrozemberczki/awesome-graph-classification/blob/master/chapters/deep_learning.md

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/graphpaper/  

