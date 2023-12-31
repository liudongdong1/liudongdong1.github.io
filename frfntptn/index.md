# FRFNTPTN


### 0. python统计函数

```python
import numpy as np
layerData = np.loadtxt(open('./data/layer_test_acc.csv', "r"), delimiter=",", skiprows=1,dtype=str)

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print(len(ground),len(predict))
C2= confusion_matrix(ground, predict, labels=[0,1])
#print(ground)
#print(C2)
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
result=C2.ravel()
print(C2.ravel(),C2.ravel()/len(ground))
print("speed:{}- layer:{}- distance:{} -slop:{} -accuracy:{} -TN:{} -FP:{} -FN:{} -TP:{} -Pr:{} -Recal:{}".format(0.15,2,0.3,0,count/len(ground),result[0],result[1],result[2],result[3],precision_score(ground,predict),recall_score(ground,predict)))
sns.heatmap(C2,annot=True)
# f, (ax1,ax2) = plt.subplots(figsize = (10, 8),nrows=2)
# ax2.set_title('sns_heatmap_confusion_matrix')
# ax2.set_xlabel('Pred')
# ax2.set_ylabel('True')
# f.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210731195028019.png)

#### 1. [FP、FN、TP、TN](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210727181314595.png)

```python
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
```

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
 
f, (ax1,ax2) = plt.subplots(figsize = (10, 8),nrows=2)
y_true = ["dog", "dog", "dog", "cat", "cat", "cat", "cat"]
y_pred = ["cat", "cat", "dog", "cat", "cat", "cat", "cat"]
C2= confusion_matrix(y_true, y_pred, labels=["dog", "cat"])
print(C2)
print(C2.ravel())
sns.heatmap(C2,annot=True)
 
ax2.set_title('sns_heatmap_confusion_matrix')
ax2.set_xlabel('Pred')
ax2.set_ylabel('True')
f.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
```

#### 2. 精确率(Precision),召回率(Recall),准确率(Accuracy)

- `准确率(Accuracy):`这三个指标里最直观的就是准确率: 模型判断正确的数据(TP+TN)占总数据的比例,"
- `召回率(Recall):` 针对数据集中的所有正例label(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例；FN表示被模型误认为是负例但实际是正例的数据；召回率也叫查全率，以物体检测为例,我们往往把图片中的物体作为正例，此时召回率高代表着模型可以找出图片中更多的物体!
- `精确率(Precision)`:针对模型判断出的所有正例(TP+FP)而言，其中真正例(TP)占的比例。精确率也叫查准率,还是以物体检测为例，精确率高表示模型检测出的物体中大部分确实是物体，只有少量不是物体的对象被当成物体。

```
Accuracy: "+str(round((tp+tn)/(tp+fp+fn+tn), 3))
"Recall: "+str(round((tp)/(tp+fn), 3))
"Precision: "+str(round((tp)/(tp+fp), 3))
("Sensitivity: "+str(round(tp/(tp+fn+0.01), 3)))
("Specificity: "+str(round(1-(fp/(fp+tn+0.01)), 3)))
("False positive rate: "+str(round(fp/(fp+tn+0.01), 3)))
("Positive predictive value: "+str(round(tp/(tp+fp+0.01), 3)))
("Negative predictive value: "+str(round(tn/(fn+tn+0.01), 3)))
```

#### 3. ROC 曲线绘制

```python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import cross_validation

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

##变为2分类
X, y = X[y != 2], y[y != 2]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3,random_state=0)

# Learn to predict each class against the other
svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = svm.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```

#### 4. 多分类问题ROC

```python
# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
 
# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 将标签二值化
y = label_binarize(y, classes=[0, 1, 2])
# 设置种类
n_classes = y.shape[1]
 
# 训练模型并预测
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
 
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)
 
# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
 
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
 
# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
 
# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
 
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210727182007386.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/frfntptn/  

