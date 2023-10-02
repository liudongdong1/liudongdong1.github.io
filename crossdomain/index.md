# CrossDomain


> UDA refers to a set of transfer learning methods for transferring knowledge learned from th source domain to the target domain under the assumption of domain discrepancy.
>
> Domain adaptation generally assumes that the two domains have the same conditional distributions, but different marginal distributions.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006201042249.png)

## 1. Resource

- Paper List: https://github.com/zhaoxin94/awesome-domain-adaptation
- Project List: https://github.com/jindongwang/transferlearning

> Na J, Jung H, Chang H J, et al. FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation[C] //Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1094-1103.

------

## Paper: FixBi

<div align=center>
<br/>
<b>FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation</b>
</div>


#### Summary

1. introduce a fixed ratio-based mixup to `augment multiple intermediate domains` between the source and target domain.
2. propose confidence-based learning methodologies: a bidirectional matching and a sel-penalization using positive and negative pseudo-labels, respectively.

#### previous work:

- GAN-based DA methods: to generate transferable representations to minimize domain discrepancy.
- `Semi-supervised learning(SSL):`  leverages unlabeled data to improve a model's performance when limited labeled data is provideed, alleviating the expensive labeling process efficiently.
  - MixMatch: used low-entropy labels for data-augmented unlabeled instances, and mixed labeled and unlabeled data for semi-supervised learning.
  - FixMatch: generates pseudo-labels using the model's predictions on weakly augmented unlabeled images. And then the examples have high-confidence prediction, they train the model using strong-augmented images. (labeled and unlabeled data have similar domains or feature distributions)
  - ReMixMatch
- `Unserpervised Domain Adaption(UDA)`: based on domain alighment and discriminative domain-invariant feature learning methods.
  - a domain adversarial neural network(DANN) learned a domain invariant representation by back-propagating the recerse gradients o fth edomain classifier.
  - Adversarial discriminative domain adaption(ADDA): learned a discriminatie representation using the source labels, and then `a separate encoding that maps the target data to the same space` based on a domain-adversarial loss is used.
  - Maximum classification discrepancy(MCD): tried to `align the distribution of a target domain` by considering task-specific desicion boundaries by maximizing the discrepancy on the target samples and then generating features that minimize this descrepancy.

#### Methods

- **Problem Formulation**:

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006164014178.png)

- **system overview**:

> The proposed method consists of (a) fixed ratio-based mixup, (b) confidence-based learning, e.g., bidirectional matching with the positive pseudo-labels and self-penalization with the negative pseudo-labels, and (c) consistency regularization.   

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006160100542.png)

**【Module One】** **Fixed Ratio-based Mixup**

- a data augmentation method to increase the robustness of model when learning from corrupt labels.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006165030071.png)

**【Module Two】** **Confidence-based learning**

> to utilize the two models as `bridges from the sourve domain to the target domain`, propose a confidence-based learning where `one model teaches the other model using the positive pseudo-labels or teach itself using the negative pseudo-labels`.    答： `一个网络的输出修改另一个网络分类任务的输出，用于交叉熵计算loss。这几个loss采用梯度下降的方法达到最优。`

- Bidirectinal Matching with positive pseudo-labels:
  - when one network assigns the class probability of input above a certain threshold r, we assume that this predicted label as a pseudo-label(positive pseudo-labels).

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006183316923.png)

- self-penalization with negative pseudo-labels:
  - since the nagetive psudo-label is unlikely to be a correct label, we need to increase the proballibity values of all other classes except for this negative pseudo-label. And optimize the output probability corresponding to the nagative pseudo-label to be close to zero.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006183823708.png)

**【Module Three】Consistency Regularization**

> to ensure a stable convergence of training both models, and the trained models should be regularized to have consistent results in the same space. helps to construct the domain bridging by ensuring that two models trained from the different domain spaces maintain consistency in the same area between the source and target domain.  ？？ 为什么同一个图片，俩个模型输出的结果都要求对呢？ `回答： 这里的xi是俩这个域混合之后的图像。loss=fixmix_sd_loss+fixmix_td_loss+bim_sd_loss+bim_td_loss+cr_loss`.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006184724627.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006185403466.png)

#### Evaluation

  - **Environment**:   
    - Office-31 [29] is the most popular dataset for real-world domain adaptation. It contains 4,110 images of 31 categories in three domains: Amazon (A), Webcam (W), DSLR (D). We evaluated all methods on six domain adaptation tasks.
    - Office-Home [40] is a more challenging benchmark than Office-31. It consists of images of everyday objects organized into four domains: artistic images (Ar), clip art (Cl),product images (Pr), and real-world images (Rw). It contains 15,500 images of 65 classes  
    - VisDA-2017 [27] is a large-scale dataset for syntheticto-real domain adaptation. It contains 152,397 synthetic
      images for the source domain and 55,388 real-world images for the target domain  





---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/crossdomain/  

