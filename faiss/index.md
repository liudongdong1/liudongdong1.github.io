# Faiss


> Faiss是Facebook AI团队开源的针对`聚类和相似性搜索库`，为稠密向量提供高效相似度搜索和聚类，支持十亿级别向量的搜索，是目前最为成熟的近似近邻搜索库。它包含多种搜索任意大小向量集（备注：向量集大小由RAM内存决定）的算法，以及用于算法评估和参数调整的支持代码。Faiss用C++编写，并提供与Numpy完美衔接的Python接口。除此以外，对一些核心算法提供了GPU实现。[参考论文](https://arxiv.org/pdf/1702.08734.pdf)  [源码地址](https://github.com/huaxz1986/faiss)

- faiss的核心就是索引（index）概念，它封装了一组向量，并且可以选择是否进行预处理，帮忙高效的检索向量。faiss中由多种类型的索引.
- 精确搜索：`faiss.indexFlatL2(欧式距离)` `faiss.indexFlatIP(内积)`;
- 一种加速搜索的方法indexIVFFlat（倒排文件）。起始就是使用k-means建立聚类中心，然后通过查询最近的聚类中心，然后比较聚类中所有向量得到相似的向量。
- 在建立 IndexFlatL2 和IndexIVFFlat都会全量存储所有向量在内存中，为了满足大的数据需求，faiss提供了一种基于 Product Quantizer（乘积量化）的压缩算法，编码向量大小到指定的字节数。此时，存储的向量是压缩过的，查询的距离也是近似的

![faiss(1)：简介 安装 与 原理3](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/7f69e05a5138467d79e0600f65a131f71603442485964.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/d397f39e81914aa043c27c99593c93bf1603442485963.png)

### 1. 相似搜索方法

> - 1~2维： Voronoi diagrams
> -  10维： kd trees，metric trees, ball-trees，spill trees
> - 高维大规模最近邻搜索： 近似搜索算法，LSH，矢量量化方法等

> - 基于树的方法
>   - KD树是其下的经典算法。一般而言，在`空间维度比较低时`，KD树的查找性能还是比较高效的；但当空间维度较高时，该方法会退化为暴力枚举，性能较差，这时一般会采用下面的哈希方法或者矢量量化方法。
> - 哈希方法
>   - LSH(Locality-Sensitive Hashing)是其下的代表算法。文献[7]是一篇非常好的LSH入门资料。
>   - 哈希算法：DH&SDH、CNNH、NINH、DSRH、DRSCH、DLBHC等。
>   - 对于小数据集和中规模的数据集`(几个million-几十个million)`，基于LSH的方法的效果和性能都还不错。这方面有2个开源工具`FALCONN和NMSLIB`。
> - 矢量量化方法
>   - 矢量量化方法，即vector quantization。在矢量量化编码中，关键是`码本的建立和码字搜索算法`。比如常见的聚类算法，就是一种矢量量化方法。而在相似搜索中，向量量化方法又以`PQ方法`最为典型。
>   - 对于大规模数据集(几百个million以上)，基于矢量量化的方法是一个明智的选择，可以用用`Faiss开源工具`。

### 2. Vector quantization

> 将一个`向量空间中的点`用其中的`一个有限子集`来进行编码的过程， 比较常见的聚类方法都可以用来做矢量量化。以Kmeans算法为例， 假设数据集一个包含N个元素， 每个元素是一个D维向量， 使用Kmeans方法进行聚类，最终产生K个聚类中心， 比如K=256， 此时需要8bit表示， 0-255 每个cluster_id，` 每个元素使用8bit表示 记录了该元素所属的cluster_id, 然后通过cluster_id 查到 中心的向量`， 用中心向量近作为该元素的近似表示。压缩rate = 8bit / (D * 4 * 8bit) 假设每个元素时D维的浮点数向量

### 3. Product quantization

> 指把原来的向量空间分解为若干个低维向量空间的笛卡尔积，并对分解得到的低维向量空间分别做量化（quantization）。这样每个向量就能由多个低维空间的量化code组合表示。
>
> - 如果全部实体的个数是n，n是千万量级甚至是上亿的规模，每个实体对应的向量是D，那么当要从这个实体集合中寻找某个实体的相似实体，暴力穷举的计算复杂度是O(n×D)，这是一个非常大的计算量，该方法显然不可取。所以对大数据量下高维度数据的相似搜索场景，我们就需要一些高效的`相似搜索技术`，而PQ就是其中一类方法。
> - PQ算法可以理解为是对vector quantization做了一次分治，首先`把原始的向量空间分解为m个低维向量空间的笛卡尔积`，并对分解得到的`低维向量空间分别做量化`.
> - 把原始D维向量（比如D=128）`分成m组`（比如m=4），每组就是D∗=D/m维的子向量（比如D∗=D/m=128/4=32)，各自用kmeans算法学习到一个码本，然后`这些码本的笛卡尔积就是原始D维向量对应的码本`。用qj表示第j组子向量，用Cj表示其对应学习到的码本，那么原始D维向量对应的码本就是C=C1×C2×…×Cm。用k∗表示子向量的聚类中心点数或者说码本大小，那么原始D维向量对应的聚类中心点数或者说码本大小就是k=(k∗)m。可以看到m=1或者m=D是PQ算法的2种极端情况，对m=1，PQ算法就回退到vector quantization，对m=D，PQ算法相当于对原始向量的每一维都用kmeans算出码本。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211005191301786.png)

> 基于这些量化器做相似搜索。有2种方法做相似搜索，一种是SDC(symmetric distance computation)，另一种是ADC(asymmetric distance computation)。SDC算法和ADC算法的区别在于是否要对查询向量x做量化;
>
> - x是查询向量(query vector)，y是数据集中的某个向量，目标是要在数据集中找到x的相似向量。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211005185722040.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211005190115941.png)

![SDC 计算流程](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211005192528860.png)

![ADC 计算流程](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211005192552954.png)

#### .1. 改进算法

> IVFADC算法，一种基于倒排索引的ADC算法。
>
> - **coarse quantizer**：对数据库中的所有特征采用K-means聚类，得到粗糙量化的类中心，比如聚类成1024类，并记`录每个类的样本数和各个样本所属的类别`。这个类中心的个数就是inverted list的个数。`把所有类中心保存到一张表中，叫coarse_cluster表，表中每项是d维`。
> - **product quantizer**: 计算y的余量，这里写图片描述，`用y减去y的粗糙量化的结果得到r(y)`。r(y)维数与y一样，然后对所有r(y)的特征分成m组，采用乘积量化，每组内仍然使用k-means聚类，这时结果是一个m维数的向量，这就是上篇文章中提到的内容。把所有的乘积量化结果保存到一个表中，叫pq_centroids表，表中每项是m维。
> - **append to inverted list**: 前面的操作中记录下`y在coarse_cluster表的索引id，在pq_centroids表中的索引j`，那么插入inverted list时，把（id，j）插入到第i个倒排索引Li中，id是y的标识符，比如文件名。list的长度就是属于第i类的样本y的数目。处理不等长list有些技巧。

> 检索过程：
>
> - 粗糙量化：对查询图像x的`特征进行粗糙量化`，即采用KNN方法将x分到某个类或某几个类，`分到几个类的话叫做multiple assignment`。过程同对数据集中的y分类差不多。
> - 计算余量： 计算x的粗糙量化误差r(x)==x−qc(x)。
> - 计算d(x,y): 对r(x)分组，计算每组中r(x)的特征子集到pq_centroids的距离。根据ADC的技巧，计算x与y的距离可以用计算x与q(y)的距离，而q(y)就是pq_centroids表中的某项，因此已经得到了x到y的近似距离。
> - 最大堆排序： 堆中每个元素代表数据库中y与x的距离，堆顶元素的距离最大，只要是比堆顶元素小的元素，代替堆顶元素，调整堆，直到判断完所有的y。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/2017-08-05-understanding-product-quantization-figure5.jpg)

> 对IVFADC的3点补充说明：
>
> 1. 考虑到在coarse quantization中，x和它的近邻不一定落在同一个簇中，所以在查询coarse quantization时，会同时取出w个倒排链。
> 2. 对取出的每个倒排链，还是用第3节介绍的PQ算法把近邻给找出。
> 3. 考虑当n>k∗D∗时，朴素的ADC算法的复杂度是O(n×m)，而IVFADC算法的复杂度会降低为O((n×w/k′)×m)。

### 4. LSH 算法

> 局部敏感是哈希函数的一种性质：如果相近的样本点对经过哈希后比相远的样本点对更容易发生碰撞。
>
> d(x,y)表示x和y之间的距离， h(x)和h(y)分别表示对x和y进行hash变换，d1<d2
>
> ​	1）如果d(x,y) ≤ d1， 则h(x) = h(y)的概率至少为p1；
>
> ​	2）如果d(x,y) ≥ d2， 则h(x) = h(y)的概率至多为p2；
>
> 满足以上两个条件的hash functions称为(d1,d2,p1,p2)-sensitive。
>
> 通过一个或多个(d1,d2,p1,p2)-sensitive的hash function对原始数据集合进行hashing生成一个或多个hash table的过程称为Locality-sensitive Hashing。
>
> - K， 表示一个Hash表被哈希函数划分得到的空间数，即`一个hash函数得到的桶的数目`；
> - L， hash表的数目，`一个hash表可以分为K个空间`；
> - T，` 查找桶的个数`，所有的hash表中，一共可以在多少个桶中进行查找。
>
> 首先，根据可使用的内存大小选取L，然后在K和T之间做出折中：哈希函数数目K越大，每个桶中的元素就越少，近邻哈希桶的数目的数目T也应该设置得比较大，反之K越小，L也可以相应的减小。获取K和L最优值的方式可以按照如下方式进行：对于每个固定的K，如果在查询样本集上获得了我们想要的精度，则此时T的值即为合理的值。在对T进行调参的时候，我们不需要重新构建哈希表，甚至我们还可以采用二分搜索的方式来加快T参数的选取过程。

#### .1.**离线建立索引**

（1）选取满足(d1,d2,p1,p2)-sensitive的`LSH hash functions`；

（2）根据对查找结果的准确率（即相邻的数据被查找到的概率）确定hash table的个数L，每个table内的hash functions的个数K，以及跟LSH hash function自身有关的参数；

（3）将所有数据经过LSH hash function哈希到相应的桶内，构成了一个或多个hash table；

#### .2. **在线查找**

（1）将查询数据经过LSH hash function哈希得到相应的桶号；

（2）`将桶号中对应的数据取出`；（为了保证查找速度，通常只需要取出前2L个数据即可）；

（3）计算`查询数据与这2L个数据之间的相似度或距离，返回最近邻的数据`；

#### .3. 应用场景

（1）`查找网络上的重复网页`：互联网上由于各式各样的原因（例如转载、抄袭等）会存在很多重复的网页，因此为了提高搜索引擎的检索质量或避免重复建立索引，需要查找出重复的网页，以便进行一些处理。其大致的过程如下：将互联网的文档用一个集合或词袋向量来表征，然后通过一些hash运算来判断两篇文档之间的相似度，常用的有minhash+LSH、simhash。

（2）`查找相似新闻网页或文章`: 与查找重复网页类似，可以通过hash的方法来判断两篇新闻网页或文章是否相似，只不过在表达新闻网页或文章时利用了它们的特点来建立表征该文档的集合。

（3）`图像检索`: 在图像检索领域，每张图片可以由一个或多个特征向量来表达，为了检索出与查询图片相似的图片集合，我们可以对图片数据库中的所有特征向量建立LSH索引，然后通过查找LSH索引来加快检索速度。

（4）`音乐检索`: 对于一段音乐或音频信息，我们提取其音频指纹（Audio Fingerprint）来表征该音频片段，采用音频指纹的好处在于其能够保持对音频发生的一些改变的鲁棒性，例如压缩，不同的歌手录制的同一条歌曲等。为了快速检索到与查询音频或歌曲相似的歌曲，我们可以对数据库中的所有歌曲的音频指纹建立LSH索引，然后通过该索引来加快检索速度。

（5）`指纹匹配`: 一个手指指纹通常由一些细节来表征，通过对比较两个手指指纹的细节的相似度就可以确定两个指纹是否相同或相似。类似于图片和音乐检索，我们可以对这些细节特征建立LSH索引，加快指纹的匹配速度。


### Resource

- [图像检索相关开源工作](https://mp.weixin.qq.com/s?__biz=Mzg2ODUzMzEzMg==&mid=2247489764&idx=2&sn=963047141b3772ce5d090d75f86ab790&chksm=ceab8e07f9dc0711861d5f6a08488e555bdb494f90d134082276ad27ab98152c4cc7ba0c54e3&scene=126&sessionid=1608019331&key=b918ff6c28ca5e81e3b180c8f362f94782b109575d9be1effdff2c6cd502f28fb98433f57174a788173afd99cb413acac90515b8f60a90605ccef4d0577d32dd418d92710a682c6296671baf4fb751a0889b2cada6b0ae7cba37a471e86d06b58767447f3dd8629e2acad6d94c52ccde8931293d4fae7169c2885571502e6d05&ascene=1&uin=MzE0ODMxOTQzMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A5jnIpd5ghQLNeNL1TsIs0I%3D&pass_ticket=mPiO6DFMfWSiTph6CkYgb%2B5c8bxVphy6As8Rr%2BazGGavgw2zPtc5evacG1a1R91H&wx_header=0&fontgear=2)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/faiss/  

