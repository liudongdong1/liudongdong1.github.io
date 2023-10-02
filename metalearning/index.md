# MetaLearning


> Few-shot classification aims to learn a classifier to `recognize unseen classes` during `training with limited labeled examples. `
>
> - **meta-learning paradigm:** transferable knowledge is extracted and propagated from a collection of tasks to prevent over fitting and improve `generalization`.
>   - model initialization based methods;
>   - metric learning methods
>   - hallucination based methods
> - **directly predicting the weighs of the classifiers for novel classes**
>
> **Relative Work**
>
> - **Initialization based methods:**
>   - good model initialization:  to learn to fine-tune, learn with limited number of labeled examples and small number of gradient update steps;
>   - learning an optimizer: LSTM-based meta-learner for replacing the stochastic gradient decent optimizer.
>   - these methods have difficulty in handling domain shifts between base and novel classes.
> - **Distance metric learning based methods:** learn to compare, make their prediction conditioned on distance or metric to few labeled instances during training process.  [MatchingNet](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.vinyals2016matching)  [ProtoNet](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.snell2017prototypical) [RelationNet](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.sung2018learning) [MAML](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.finn2017model)
>   - [cosine similarity](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.vinyals2016matching)
>   - [Euclidean distance to class-mean representation](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.snell2017prototypical)
>   - [CNN-based relation module](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.sung2018learning)
>   - [ridge regression](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.bertinetto2019meta)
>   - [graph neural network](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.garcia2018few)
>   - simply `reducing intra-class variation` in a baseline methods using base class data
> - **Hallucination based methods:** learn to augment, learns a generator from data in the base classes, and use the learned generator to hallucinated new novel class data for data augmentation.
>   - transfer appearance variations exhibited in the base classes:
>     - transfer variance in base classes to novel classes, use GAN to transfer style
>   - directly integrate the generator into a meta-learning algorithm.
> - **Domain adaption:** reduce the domain shifts between source and target domain, or novel tasks in a different domain.  [Dong&Xing](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf#cite.dong2018domain)
>
> **Limitations:**
>
> - discrepancy of the implementation details among multiple few-shot learning algorithms obscures the relative performance gain;
> - the novel classes are sampled from the same dataset, lack of domain shift between the base and novel classes makes the evaluation scenarios unrealistic.
>
> **Solution:**
>
> - using a deep backbone shrinks the performance gap between different methods in the setting of  limited domain differences between base and novel classes.
> - by replacing the linear classifier with a distance-based classifier is surprisingly competitive to exiting methods;
> - practical evaluation setting where there exists domain shift between base and novel classes.
>
> **Datasets&Scenarios**
>
> - **Scenarios**: `generic object recognition`,` fine-grained image classification`, `cross-domain adaptation`
> - **Dataset**
>   - mini-ImageNet:  a subset of 100 classes,contains 600 images for each class
>   - CUB-200-2011 dataset: contains  200  classes  and  11,788  images  in  total

### 1. Introdce

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221116253.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221151526.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221218773.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221238574.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221250503.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221318618.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221355339.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221439417.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221635156.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221805199.png)

Triplet loss

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607221955803.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607222025711.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607222116977.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607222622898.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607222705125.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607222741142.png)

![image-20210607222802846](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607222802846.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607223003197.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607223133699.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607223305277.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607223349000.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607223413399.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210607223454513.png)

### 2. Base Model

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608143334567.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608143813515.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608145114206.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608145154925.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608145833634.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210608145846609.png)

### Resource

- Chen, Wei-Yu, et al. "A closer look at few-shot classification." *arXiv preprint arXiv:1904.04232* (2019). [[pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.04232.pdf)]



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/metalearning/  

