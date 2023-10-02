# AnchorIntroduce


> 在网络最后的输出中，对于每个grid cell产生3个bounding box，每个bounding box的输出有三类参数：一个是对象的box参数，一共是四个值，即**box的中心点坐标（x,y）和box的宽和高（w,h）**;一个是**置信度**，这是个区间在[0,1]之间的值；最后一个是**一组条件类别概率**，都是区间在[0,1]之间的值，代表概率。假如一个图片被分割成S*∗*S个grid cell，我们有B个anchor box，也就是说每个grid cell有B个bounding box, 每个bounding box内有4个位置参数，1个置信度，classes个类别概率，输出维数是:S∗S∗[B∗(4+1+classes)]。

## 1. anchor box

### 1.1 对anchor box的理解

anchor box其实就是从训练集的所有ground truth box中统计(使用k-means)出来的<font color=red>在训练集中最经常出现的几个box形状和尺寸</font>。比如，在某个训练集中最常出现的box形状有扁长的、瘦高的和宽高比例差不多的正方形这三种形状。我们可以预先将这些统计上的先验（或来自人类的）经验加入到模型中，这样模型在学习的时候，瞎找的可能性就更小了些，当然就**有助于模型快速收敛**了。以前面提到的训练数据集中的ground truth box最常出现的三个形状为例，当模型在训练的时候我们可以告诉它，你要在grid cell 1附件找出的对象的形状要么是扁长的、要么是瘦高的、要么是长高比例差不多的正方形，你就不要再瞎试其他的形状了。**anchor box其实就是对预测的对象范围进行约束，并加入了尺寸先验经验，从而实现多尺度学习的目的。**

- **量化anchor box**
  要在模型中使用这些形状，总不能告诉模型有个形状是瘦高的，还有一个是矮胖的，我们需要量化这些形状。YOLO的做法是想办法找出分别代表这些形状的**宽和高**，有了宽和高，尺寸比例即形状不就有了。<font color=red>YOLO作者的办法是使用k-means算法在训练集中所有样本的ground truth box中聚类出具有代表性形状的宽和高，作者将这种方法称作维度聚类</font>（dimension cluster）。细心的读者可能会提出这个问题：**到底找出几个anchor box算是最佳的具有代表性的形状**。YOLO作者方法是做实验，聚类出多个数量不同anchor box组，分别应用到模型中，最终找出最优的在模型的复杂度和高召回率(high recall)之间折中的那组anchor box。作者在COCO数据集中使用了9个anchor box，我们前面提到的例子则有3个anchor box。
- **怎么在实际的模型中加入anchor box的先验经验呢？**
  最终负责预测grid cell中对象的box的最小单元是bounding box,那我们可以让一个grid cell输出（预测）多个bounding box，然后每个bounding box负责预测不同的形状不就行了？比如前面例子中的3个不同形状的anchor box，我们的一个grid cell会输出3个参数相同的bounding box，第一个bounding box负责预测的形状与anchor box 1类似的box，其他两个bounding box依次类推。**作者在YOLOv3中取消了v2之前每个grid cell只负责预测一个对象的限制，也就是说grid cell中的三个bounding box都可以预测对象，当然他们应该对应不同的ground truth**。那么如何在**训练中**确定哪个bounding box负责某个ground truth呢？方法是<font color=red>求出每个grid cell中每个anchor box与ground truth box的IOU(交并比)，IOU最大的anchor box对应的bounding box就负责预测该ground truth，也就是对应的对象</font>，后面还会提到负责预测的问题。
- **怎么告诉模型第一个bounding box负责预测的形状与anchor box 1类似，第二个bounding box负责预测的形状与anchor box 2类似？**
  YOLO的做法是**不让bounding box直接预测实际box的宽和高**(w,h)，而是将预测的宽和高分别与anchor box的宽和高绑定，这样不管一开始bounding box输出的(w,h)是怎样的，经过转化后都是与anchor box的宽和高相关，这样经过很多次惩罚训练后，每个bounding box就知道自己该负责怎样形状的box预测了。这个**绑定的关系**是什么？就涉及到了anchor box的计算。

### 1.2 anchor box的计算

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825115832.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825120038.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825120138.png)

## 2. 至信度

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200825120443.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/anchorintroduce/  

