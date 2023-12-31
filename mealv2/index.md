# MEALv2


**level**: 
**author**: Zhiqiang Shen Marios Savvides(Carnegie Mello University)
**date**: 2020,9,17
**keyword**:

- knowledge distillation; discriminators;

------

# Paper: MEALv2

<div align=center>
<br/>
<b>MEAL V2: Boosting Vanilla ResNet-50 to 80%+ Top-1 Accuracy
on ImageNet without Tricks∗</b>
</div>


#### Summary

1. simplify MEAL by several methods:
   1. adopting the similarity loss and discriminator only on the final outputs;
   2. using the average of softmax probabilities from all teacher ensembles as the stronger supervision for distillation;
2. the first to boost vanilla resnet-50 to surpass 80% on ImageNet without architecture modification or additional training data;
3. only relies on teacher-student paradigm,
   1. no architecture modification;
   2. no outside training data;
   3. no cosine learning rate;
   4. no extra data augmentation;
   5. no label smoothing;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200919085240930.png)

#### Methods

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200919085120551.png)

**【Module One】Teachers Ensemble**: adopt the average of softmax probabilities from multiple pre-trained teachers as an ensemble;

- $p_t^{T_\theta}$: the t-th teacher's softmax prediction;
- X: the inout image;
- K: the number of total teachers;

$$
p_e^{T_\theta}(x)=1/K\sum_{t=1}^Kp_t^{T_\theta}(x)
$$

**【Module Two】KL-divergence**: measure metric of how one probability distribution is different from another reference distribution;

- $p^{s_\theta}(x_i)$: the student output probability;
- N: the number of samples;

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200919090331634.png)

**【Module Three】Discriminator**: a binary classifier to distinguish the input feature are from teacher ensemble or student network, consist of a sigmoid function following the binary cross entropy loss:

- $x_t,x_s$: the teacher and student input features;
- $f_\theta$: a three-fc-layer subnetwork;
- $\sigma(x)=1/(1+exp(-x))$: logistic function;
- $y\epsilon[0,1]$: the label;

$$
P^D(x;\theta)=\sigma(f_\theta({x_t,x_s}))\\
L_D=-1/N\sum_{i=1}^N[y_i*logp_i^D+(1-y_i)*log(1-P^D_i)]
$$

#### Evaluation

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200919091306706.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200919091319708.png)

#### Notes <font color=orange>去加强了解</font>

  - https://github.com/szq0214/MEAL-V2.
  - 学习代码，学习知识蒸馏网络结构

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mealv2/  

