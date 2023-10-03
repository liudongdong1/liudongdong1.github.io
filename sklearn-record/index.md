# SkLearn Record


- Supervised learning
  - [1.1. Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
  - [1.2. Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
  - [1.3. Kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html)
  - [1.4. Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
  - [1.5. Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html)
  - [1.6. Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
  - [1.7. Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
  - [1.8. Cross decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html)
  - [1.9. Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
  - [1.10. Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
  - [1.11. Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
  - [1.12. Multiclass and multioutput algorithms](https://scikit-learn.org/stable/modules/multiclass.html)
  - [1.13. Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
  - [1.14. Semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html)
  - [1.15. Isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html)
  - [1.16. Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
  - [1.17. Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- Unsupervised learning
  - [2.1. Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html)
  - [2.2. Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
  - [2.3. Clustering](https://scikit-learn.org/stable/modules/clustering.html)
  - [2.4. Biclustering](https://scikit-learn.org/stable/modules/biclustering.html)
  - [2.5. Decomposing signals in components (matrix factorization problems)](https://scikit-learn.org/stable/modules/decomposition.html)
  - [2.6. Covariance estimation](https://scikit-learn.org/stable/modules/covariance.html)
  - [2.7. Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
  - [2.8. Density Estimation](https://scikit-learn.org/stable/modules/density.html)
  - [2.9. Neural network models (unsupervised)](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html)

### 1. SVC

> - [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html?highlight=svc#sklearn.svm.LinearSVC)
> - [sklearn.svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html?highlight=svc#sklearn.svm.NuSVC)
> - [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC)

#### 1.1. 函数介绍

`sklearn.svm.``SVC`(*, *C=1.0*, *kernel='rbf'*, *degree=3*, *gamma='scale'*, *coef0=0.0*, *shrinking=True*, *probability=False*, *tol=0.001*, *cache_size=200*, *class_weight=None*, *verbose=False*, *max_iter=- 1*, *decision_function_shape='ovr'*, *break_ties=False*, *random_state=None*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/svm/_classes.py#L443)[¶](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC)

> The implementation is based on `libsvm`. The fit time scales at least quadratically with the number of samples and may be `impractical beyond tens of thousands of samples`. For large datasets consider using [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) or [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) instead, possibly after a [`Nystroem`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem) transformer.

- C : float, optional (default=1.0)

      误差项的惩罚参数，一般取值为10的n次幂，如10的-5次幂，10的-4次幂。。。。10的0次幂，10，1000,1000，在python中可以使用pow（10，n） n=-5~inf
      C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样会出现训练集测试时准确率很高，但泛化能力弱。
      C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强。

- kernel : string, optional (default=’rbf’)

  ```
  svc中指定的kernel类型。
  可以是： ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 或者自己指定。 默认使用‘rbf’ 
  ```

- degree : int, optional (default=3)

  ```
   当指定kernel为 ‘poly’时，表示选择的多项式的最高次数，默认为三次多项式。
   若指定kernel不是‘poly’,则忽略，即该参数只对‘poly’有作用。
  ```

- gamma : float, optional (default=’auto’)

  ```
  当kernel为‘rbf’, ‘poly’或‘sigmoid’时的kernel系数。
  如果不设置，默认为 ‘auto’ ，此时，kernel系数设置为：1/n_features
  ```

- probability : boolean, optional (default=False)

  ```
  是否采用概率估计。
  必须在fit（）方法前使用，该方法的使用会降低运算速度，默认为False。
  ```

- tol : float, optional (default=1e-3)

  ```
  误差项达到指定值时则停止训练，默认为1e-3，即0.001。
  ```

- max_iter : int, optional (default=-1)

  ```
  强制设置最大迭代次数。
  默认设置为-1，表示无穷大迭代次数。
  Hard limit on iterations within solver, or -1 for no limit.
  ```

- 松弛变量：

  ```
  若所研究的线性规划模型的约束条件全是小于类型，那么可以通过标准化过程引入M个非负的松弛变量。
  松弛变量的引入常常是为了便于在更大的可行域内求解。若为0，则收敛到原有状态，若大于零，则约束松弛。
  ```

#### 1.2. 数学推导

### 2. [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier)

#### 2.1. 函数介绍

*class* `sklearn.neighbors.``KNeighborsClassifier`(*n_neighbors=5*, ***, *weights='uniform'*, *algorithm='auto'*, *leaf_size=30*, *p=2*, *metric='minkowski'*, *metric_params=None*, *n_jobs=None*, ***kwargs*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/neighbors/_classification.py#L24)

- n_neighbors ： int，optional(default = 5)
  默认情况下kneighbors查询使用的邻居数。就是k-NN的k的值，选取最近的k个点。
- weights ： str或callable，可选(默认=‘uniform’)
  默认是uniform，参数可以是uniform、distance，也可以是用户自己定义的函数。uniform是均等的权重，就说所有的邻近点的权重都是相等的。`distance是不均等的权重，距离近的点比距离远的点的影响大`。用户自定义的函数，接收距离的数组，返回一组维数相同的权重。
- algorithm ： {‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选
  `快速k近邻搜索算法`，默认参数为auto，可以理解为`算法自己决定合适的搜索算法`。除此之外，用户也可以自己指定搜索算法`ball_tree`、`kd_tree`、`brute方法`进行搜索，brute是蛮力搜索，也就是线性扫描，当训练集很大时，计算非常耗时。kd_tree，构造kd树存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个结点是一个超矩形，`在维数小于20时效率高`。`ball tree是为了克服kd树高纬失效而发明的`，其构造过程是以质心C和半径r分割样本空间，每个节点是一个超球体。
- leaf_size ： int，optional(默认值= 30)
  默认是30，这个是构造的kd树和ball树的大小。这个值的设置会影响树构建的速度和搜索速度，同样也影响着存储树所需的内存大小。需要根据问题的性质选择最优的大小。
- p ： 整数，可选(默认= 2)
  `距离度量公式`。在上小结，我们使用欧氏距离公式进行距离度量。除此之外，还有其他的度量方法，例如曼哈顿距离。这个参数默认为2，也就是默认使用欧式距离公式进行距离度量。也可以设置为1，使用曼哈顿距离公式进行距离度量。
- metric ： 字符串或可调用，默认为’minkowski’
  用于距离度量，默认度量是minkowski，也就是p=2的欧氏距离(欧几里德度量)。
- n_jobs ： int或None，可选(默认=None)
  并行处理设置。默认为1，临近点搜索并行工作数。如果为-1，那么CPU的所有cores都用于并行工作。

### 3. [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier)

#### 3.1. 函数介绍

*class* `sklearn.ensemble.``RandomForestClassifier`(*n_estimators=100*, ***, *criterion='gini'*, *max_depth=None*, *min_samples_split=2*, *min_samples_leaf=1*, *min_weight_fraction_leaf=0.0*, *max_features='auto'*, *max_leaf_nodes=None*, *min_impurity_decrease=0.0*, *min_impurity_split=None*, *bootstrap=True*, *oob_score=False*, *n_jobs=None*, *random_state=None*, *verbose=0*, *warm_start=False*, *class_weight=None*, *ccp_alpha=0.0*, *max_samples=None*)

> A random forest is a meta estimator that `fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting`. The sub-sample size is controlled with the `max_samples` parameter if `bootstrap=True` (default), otherwise the whole dataset is used to build each tree.

1. n_estimators：森林中`决策树的数量`。默认100
2. criterion：`分裂节点所用的标准`，可选“gini”, “entropy”，默认“gini”。
3. max_depth：`树的最大深度`。如果为None，则将节点展开，直到所有叶子都是纯净的(只有一个类)，或者直到所有叶子都包含少于min_samples_split个样本。默认是None。
4. min_samples_split：拆分内部节点所需的最少样本数：如果为int，则将min_samples_split视为最小值。如果为float，则min_samples_split是一个分数，而ceil（min_samples_split * n_samples）是每个拆分的最小样本数。默认是2。
5. min_samples_leaf：在叶节点处需要的最小样本数。仅在任何深度的分割点在左分支和右分支中的每个分支上至少留下min_samples_leaf个训练样本时，才考虑。这可能具有平滑模型的效果，尤其是在回归中。如果为int，则将min_samples_leaf视为最小值。如果为float，则min_samples_leaf是分数，而ceil（min_samples_leaf * n_samples）是每个节点的最小样本数。默认是1。
6. min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。
7. max_features：`寻找最佳分割时要考虑的特征数量`：如果为int，则在每个拆分中考虑max_features个特征。如果为float，则max_features是一个分数，并在每次拆分时考虑int（max_features * n_features）个特征。如果为“auto”，则max_features = sqrt（n_features）。如果为“ sqrt”，则max_features = sqrt（n_features）。如果为“ log2”，则max_features = log2（n_features）。如果为None，则max_features = n_features。注意：在找到至少一个有效的节点样本分区之前，分割的搜索不会停止，即使它需要有效检查多个max_features功能也是如此。
8. max_leaf_nodes：最大叶子节点数，整数，默认为None
9. min_impurity_decrease：如果分裂指标的减少量大于该值，则进行分裂。
10. min_impurity_split：决策树生长的最小纯净度。默认是0。自版本0.19起不推荐使用：不推荐使用min_impurity_split，而建议使用0.19中的min_impurity_decrease。min_impurity_split的默认值在0.23中已从1e-7更改为0，并将在0.25中删除。
11. bootstrap：`是否进行bootstrap操作`，bool。默认True。如果`bootstrap==True，将每次有放回地随机选取样本，只有在extra-trees中，bootstrap=False`
12. oob_score：是否使用袋外样本来估计泛化精度。默认False。
13. n_jobs：并行计算数。默认是None。
14. random_state：`控制bootstrap的随机性以及选择样本的随机性`。
15. verbose：在拟合和预测时控制详细程度。默认是0。
16. warm_start：不常用
17. class_weight：每个类的权重，可以用字典的形式传入{class_label: weight}。如果选择了“balanced”，则输入的权重为n_samples / (n_classes * np.bincount(y))。
18. ccp_alpha：将选择成本复杂度最大且小于ccp_alpha的子树。默认情况下，不执行修剪。
19. max_samples：如果bootstrap为True，则从X抽取以训练每个基本分类器的样本数。如果为None（默认），则抽取X.shape [0]样本。如果为int，则抽取max_samples样本。如果为float，则抽取max_samples * X.shape [0]个样本。因此，max_samples应该在（0，1）中。是0.22版中的新功能。

#### 4. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

> | [`tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)(*[, criterion, …]) | A decision tree classifier.              |
> | ------------------------------------------------------------ | ---------------------------------------- |
> | [`tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)(*[, criterion, …]) | A decision tree regressor.               |
> | [`tree.ExtraTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier)(*[, criterion, …]) | An extremely randomized tree classifier. |
> | [`tree.ExtraTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor)(*[, criterion, …]) | An extremely randomized tree regressor.  |

#### 4.1. 函数介绍

*class* `sklearn.tree.``DecisionTreeClassifier`(***, *criterion='gini'*, *splitter='best'*, *max_depth=None*, *min_samples_split=2*, *min_samples_leaf=1*, *min_weight_fraction_leaf=0.0*, *max_features=None*, *random_state=None*, *max_leaf_nodes=None*, *min_impurity_decrease=0.0*, *min_impurity_split=None*, *class_weight=None*, *ccp_alpha=0.0*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/tree/_classes.py#L607)

- criterion：`分裂节点所用的标准`，可选“gini”, “entropy”，默认“gini”。
- splitter：用于在`每个节点上选择拆分的策略`。可选“best”, “random”，默认“best”。
- max_depth：树的最大深度。如果为None，则将节点展开，直到所有叶子都是纯净的(只有一个类)，或者直到所有叶子都包含少于min_samples_split个样本。默认是None。
- min_samples_split：`拆分内部节点所需的最少样本数`：如果为int，则将min_samples_split视为最小值。如果为float，则min_samples_split是一个分数，而ceil（min_samples_split * n_samples）是每个拆分的最小样本数。默认是2。
- min_samples_leaf：在叶节点处需要的最小样本数。仅在任何深度的分割点在左分支和右分支中的每个分支上至少留下min_samples_leaf个训练样本时，才考虑。这可能具有平滑模型的效果，尤其是在回归中。如果为int，则将min_samples_leaf视为最小值。如果为float，则min_samples_leaf是分数，而ceil（min_samples_leaf * n_samples）是每个节点的最小样本数。默认是1。
- min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。
- max_features：`寻找最佳分割时要考虑的特征数量`：如果为int，则在每个拆分中考虑max_features个特征。如果为float，则max_features是一个分数，并在每次拆分时考虑int（max_features * n_features）个特征。如果为“auto”，则max_features = sqrt（n_features）。如果为“ sqrt”，则max_features = sqrt（n_features）。如果为“ log2”，则max_features = log2（n_features）。如果为None，则max_features = n_features。注意：在找到至少一个有效的节点样本分区之前，分割的搜索不会停止，即使它需要有效检查多个max_features功能也是如此。
- random_state：随机种子，负责控制分裂特征的随机性，为整数。默认是None。
- max_leaf_nodes：`最大叶子节点数`，整数，默认为None
- min_impurity_decrease：如果`分裂指标的减少量大于该值`，则进行分裂。
- min_impurity_split：决策树生长的最小纯净度。默认是0。自版本0.19起不推荐使用：不推荐使用min_impurity_split，而建议使用0.19中的min_impurity_decrease。min_impurity_split的默认值在0.23中已从1e-7更改为0，并将在0.25中删除。
- class_weight：每个类的权重，可以用字典的形式传入{class_label: weight}。如果选择了“balanced”，则输入的权重为n_samples / (n_classes * np.bincount(y))。
- presort：此参数已弃用，并将在v0.24中删除。
- ccp_alpha：将选择成本复杂度最大且小于ccp_alpha的子树。默认情况下，不执行修剪。

#### 4.3. 示例代码

```python
# -*- coding: utf-8 -*-

# 引入数据
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print("Class labels:",np.unique(y))  #打印分类类别的种类
## 画出决策边界图(只有在2个特征才能画出来)
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.colors import ListedColormap

def plot_decision_region(X,y,classifier,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                         np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    # plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                   y = X[y==cl,1],
                   alpha=0.8,
                   c=colors[idx],
                   marker = markers[idx],
                   label=cl,
                   edgecolors='black')
# 切分训练数据和测试数据
from sklearn.model_selection import train_test_split
## 30%测试数据，70%训练数据，stratify=y表示训练数据和测试数据具有相同的类别比例
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
## 估算训练数据中的mu和sigma
sc.fit(X_train)
## 使用训练数据中的mu和sigma对数据进行标准化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

## 决策树分类器
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train_std,y_train)
plot_decision_region(X_train_std,y_train,classifier=tree,resolution=0.02)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
tree_fit=tree.fit(X_train_std,y_train)
tree_fit.classes_  #array([0, 1, 2])
tree_fit.feature_importances_  #array([0.42708333, 0.57291667])
tree_fit.max_features_  #2
tree_fit.n_classes_  #3
tree_fit.n_features_  #2
tree_fit.n_outputs_  #1
tree_fit.tree_
```

```python
# conda install -c conda-forge pydotplus
## 决策树可视化
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree,filled=True,class_names=['Setosa','Versicolor','Virginica'],
                          feature_names=['petal_length','petal_width'],out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('D:\\Users\\Desktop\\一部二部文件\\tree.png')
```

### 5. [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradientboostingclassifier#sklearn.ensemble.GradientBoostingClassifier)

> - [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradientboostingclassifier#sklearn.ensemble.GradientBoostingClassifier)
> - [sklearn.ensemble.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html?highlight=gradientboostingclassifier#sklearn.ensemble.HistGradientBoostingClassifier)
> - `sklearn.ensemble`.GradientBoostingClassifier
> - `sklearn.ensemble`.AdaBoostClassifier
> - [`sklearn.ensemble`.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html?highlight=gradientboostingclassifier)

#### 5.1. 参数介绍

*class* `sklearn.ensemble.``GradientBoostingClassifier`(***, *loss='deviance'*, *learning_rate=0.1*, *n_estimators=100*, *subsample=1.0*, *criterion='friedman_mse'*, *min_samples_split=2*, *min_samples_leaf=1*, *min_weight_fraction_leaf=0.0*, *max_depth=3*, *min_impurity_decrease=0.0*, *min_impurity_split=None*, *init=None*, *random_state=None*, *max_features=None*, *verbose=0*, *max_leaf_nodes=None*, *warm_start=False*, *validation_fraction=0.1*, *n_iter_no_change=None*, *tol=0.0001*, *ccp_alpha=0.0*)

-  **n_estimators**: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。在`实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑`。
-  **learning_rate**: 即每个弱学习器的权重缩减系数νν，也称作步长，在原理篇的正则化章节我们也讲到了，加上了正则化项，我们的强学习器的迭代公式为fk(x)=fk−1(x)+νhk(x)fk(x)=fk−1(x)+νhk(x)。νν的取值范围为0<ν≤10<ν≤1。对于同样的训练集拟合效果，较小的νν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的νν开始调参，默认是1。
-  **subsample**: 即我们在原理篇的正则化章节讲到的子采样，取值为(0,1]。注意这里的子采样和随机森林不一样，`随机森林使用的是放回抽样，而这里是不放回抽样`。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在`[0.5, 0.8]之间`，默认是1.0，即不使用子采样。
-  **init**: 即我们的初始化的时候的弱学习器，拟合对应原理篇里面的f0(x)f0(x)，如果不输入，则用训练集样本来做样本集的初始化分类回归预测。否则用init参数提供的学习器做初始化分类回归预测。一般用在我们对数据有先验知识，或者之前做过一些拟合的时候，如果没有的话就不用管这个参数了。
-  **loss:** 即我们GBDT算法中的损失函数。分类模型和回归模型的损失函数是不一样的。

​               对于分类模型，有对数似然损失函数"deviance"和指数损失函数"exponential"两者输入选择。默认是对数似然损失函数"deviance"。在原理篇中对这些分类损失函数有详细的介绍。一般来说，推荐使用默认的"deviance"。它对二元分离和多元分类各自都有比较好的优化。而指数损失函数等于把我们带到了Adaboost算法。

​            对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。

- **alpha：**这个参数只有GradientBoostingRegressor有，当我们使用Huber损失"huber"和分位数损失“quantile”时，需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。

### 6. [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb#sklearn.naive_bayes.GaussianNB)

#### 6.1. 函数介绍

*class* `sklearn.naive_bayes.``GaussianNB`(***, *priors=None*, *var_smoothing=1e-09*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/naive_bayes.py#L118)

> perform online updates to model parameters via [`partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb#sklearn.naive_bayes.GaussianNB.partial_fit). For [details](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf) on algorithm used to update feature means and variance online.

- `priors`:先验概率大小，如果没有给定，模型则根据样本数据自己计算（利用极大似然法）。

- `class_prior_`:每个样本的概率
- `class_count`:每个类别的样本数量
- `theta_`:每个类别中每个特征的均值
- `sigma_`:每个类别中每个特征的方差

#### 6.2. **多项式分布贝叶斯**

##### 6.2.1. 函数介绍

**class** sklearn.naive_bayes.**MultinomialNB**(**alpha**=1.0, **fit_prior**=**True**, **class_prior**=**None**)

- `alpha`:先验平滑因子，默认等于1，当等于1时表示拉普拉斯平滑。
- `fit_prior`:是否去学习类的先验概率，默认是True
- `class_prior`:各个类别的先验概率，如果没有指定，则模型会根据数据自动学习， 每个类别的先验概率相同，等于类标记总个数N分之一。

- `class_log_prior_`:每个类别平滑后的先验概率
- `intercept_`:是朴素贝叶斯对应的线性模型，其值和class_log_prior_相同`feature_log_prob_`:给定特征类别的对数概率(条件概率)。 特征的条件概率=（指定类下指定特征出现的次数+alpha）/（指定类下所有特征出现次数之和+类的可能取值个数*alpha）`coef_`: 是朴素贝叶斯对应的线性模型，其值和feature_log_prob相同
- `class_count_`: 训练样本中各类别对应的样本数
- `feature_count_`: 每个类别中各个特征出现的次数

#### 6.3. [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html?highlight=bernoullinb#sklearn.naive_bayes.BernoulliNB)

##### 6.3.1. 函数介绍

> this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, `BernoulliNB is designed for binary/boolean features`.

class sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

- `alpha`:平滑因子，与多项式中的alpha一致。

- `binarize`:样本特征二值化的阈值，默认是0。如果不输入，则模型会认为所有特征都已经是二值化形式了；如果输入具体的值，则模型会把大于该值的部分归为一类，小于的归为另一类。
- `fit_prior`:是否去学习类的先验概率，默认是True
- `class_prior`:各个类别的先验概率，如果没有指定，则模型会根据数据自动学习， 每个类别的先验概率相同，等于类标记总个数N分之一。

- `class_log_prior_`:每个类别平滑后的先验对数概率。
- `feature_log_prob_`:给定特征类别的经验对数概率。
- `class_count_`:拟合过程中每个样本的数量。
- `feature_count_`:拟合过程中每个特征的数量。

#### 6.4. [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html?highlight=multinomialnb#sklearn.naive_bayes.MultinomialNB)

##### 6.4.1. [函数介绍](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html?highlight=multinomialnb#sklearn.naive_bayes.MultinomialNB)

*class* `sklearn.naive_bayes.``MultinomialNB`(***, *alpha=1.0*, *fit_prior=True*, *class_prior=None*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/naive_bayes.py#L669)[
  ](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html?highlight=multinomialnb#sklearn.naive_bayes.MultinomialNB)

> The multinomial Naive Bayes classifier is suitable for `classification with discrete features (e.g., word counts for text classification)`. The `multinomial distribution normally requires integer feature counts`. However, in practice, fractional counts such as tf-idf may also work.

#### 6.5. [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=adaboostclassifier#sklearn.ensemble.AdaBoostClassifier)

##### 6.5.1. 函数介绍

*class* `sklearn.ensemble.``AdaBoostClassifier`(*base_estimator=None*, ***, *n_estimators=50*, *learning_rate=1.0*, *algorithm='SAMME.R'*, *random_state=None*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/ensemble/_weight_boosting.py#L285)

> a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

- **base_estimator：** 可选参数，默认为DecisionTreeClassifier。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的一般是CART决策树或者神经网络MLP。默认是决策树，即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba。
- **algorithm：** 可选参数，默认为SAMME.R。scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，SAMME使用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重。由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R。我们一般使用默认的SAMME.R就够了，但是要注意的是使用了SAMME.R， 则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制。
- **n_estimators：** 整数型，可选参数，默认为50。弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是50。在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。
- **learning_rate：** 浮点型，可选参数，默认为1.0。每个弱学习器的权重缩减系数，取值范围为0到1，对于同样的训练集拟合效果，较小的v意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的v开始调参，默认是1。
- **random_state：** 整数型，可选参数，默认为None。如果RandomState的实例，random_state是随机数生成器; 如果None，则随机数生成器是由np.random使用的RandomState实例。

#### 6.6. 函数使用

- `fit(X,Y)`:在数据集(X,Y)上拟合模型。
- `partial_fit`(*X*, *y*, *classes=None*, *sample_weight=None*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/naive_bayes.py#L289)[
    ](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb#sklearn.naive_bayes.GaussianNB.partial_fit)： 当数据很大的时候
- `get_params()`:获取模型参数。
- `predict(X)`:对数据集X进行预测。
- `predict_log_proba(X)`:对数据集X预测，得到每个类别的概率对数值。`predict_proba(X)`:对数据集X预测，得到每个类别的概率。
- `score(X,Y)`:得到模型在数据集(X,Y)的得分情况。

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/sklearn-record/  

