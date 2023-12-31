# HighDimensionClassify


> Zhu Q, Deng W, Zheng Z, et al. A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification[J]. IEEE Transactions on Cybernetics, 2021. [code ](https://github.com/dengweihuan/SSDGL) [[pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fftp%2Farxiv%2Fpapers%2F2105%2F2105.14327.pdf)]

# Paper:

<div align=center>
<br/>
<b>A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification</b>
</div>


#### Summary

1. a` spectral-spatial dependent global learning (SSDGL) framework` based on `global convolutional long short-term memory (GCL)` and `global joint attention mechanism (GJAM)` is proposed for insufficient and imbalanced HSI classification.
2. in SSDGL, the `hierarchically balanced(H-B) sampling strategy` and the `weighted softmax loss` are proposed to address the imbalanced sample problem.
3. the GCL module is introduced to extract the `long-short-term dependency of spectral features` to effectively `distinguish similar spectral characteristics of land cover types.`
4. the GJAM module is proposed to extract attention areas, learning the most most discriminative feature representations.

### Problem

- to extract the deep spectral-spatial features and solve the sample problem of insufficiency and imbalance.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016205251462.png)

![GCLAM](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016205712811.png)

![GCL](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016205741143.png)

> Two-dimensional t-SNE visualization of features. Data distributions of the labeled samples in the original feature space(the first row) and the convolutional feature space (the second row).

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211016205448820.png)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/highdimensionclassify/  

