# SkLearn Optimize


> 使用scikit-learn时提高速度的三种主要方法是：使用**joblib和Ray并行化或分发培训**，**使用不同的超参数优化技术**（网格搜索，随机搜索，提前停止），以及**更改优化功能**（求解器）。
>
> 1.**随机参数选择模型（RandomizedSearchCV）可以帮助我们快速的确定参数的范围。**
>
> 2.**对于随机参数选择模型而言，初始的特征空间选择特别重要。如果初始的特征空间选择不对，则后面的调参工作都可能是徒劳。我们可参考一些经验值或者做一些对比试验，来确定模型的参数空间。**
>
> 3.**RandomizedSearchCV 和 GridSearchCV 搭配使用，`先找大致范围，再精确搜索**`。
>
> 4.通过优化模型参数，虽然每次的提升幅度不是很大，但是通过多次的优化，这些小的提升累加在一起就是很大的提升。
>
> 5.**遇到不懂的问题，多查看sklearn官方文档，这是一个逐渐积累和提升的过程**。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210414094200.png)

### 1. [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=grid%20search#sklearn.model_selection.GridSearchCV)

#### 1.1. 函数介绍

*class* `sklearn.model_selection.``GridSearchCV`(*estimator*, *param_grid*, ***, *scoring=None*, *n_jobs=None*, *refit=True*, *cv=None*, *verbose=0*, *pre_dispatch='2\*n_jobs'*, *error_score=nan*, *return_train_score=False*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_search.py#L971)

- estimator ：估计器对象，例如要找随机森林模型的最佳参数，就传随机森林模型。
- parameters：要选择的参数集合。
- cv: 用来指定交叉验证数据集的生成规则。假设=5表示每次计算都把数据集分成5份，拿其中一份作为交叉验证数据集，其他作为训练集。

> GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.

> GridSearch和CV，即网格搜索和交叉验证。网格搜索，搜索的是参数，即在指定的参数范围内，按步长依次调整参数，利用调整的参数训练学习器，从所有的参数中找到在验证集上精度最高的参数，这其实是一个训练和比较的过程,非常耗时。
>
> 网格搜索适用于三四个（或者更少）的超参数（`当超参数的数量增长时，网格搜索的计算复杂度会呈现指数增长，这时候则使用随机搜索`），用户列出一个较小的超参数值域，这些超参数至于的笛卡尔积（排列组合）为一组组超参数。网格搜索算法使用每组超参数训练模型并挑选验证集误差最小的超参数组合。

#### 1.2. 使用案例

##### .1. **DecisionTreeClassifier 调参数**

```python
# 数据分为训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

# 用GridSearchCV寻找最优参数（字典）  参数具体看API种 estimator 函数有哪些参数
param = [{'criterion':['gini'],'max_depth':[30,50,60,100],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.5]},
         {'criterion':['gini','entropy']},
         {'max_depth': [30,60,100], 'min_impurity_decrease':[0.1,0.2,0.5]}]
clf = GridSearchCV(DecisionTreeClassifier(),param_grid=param,cv=6)
clf.fit(x_train,y_train)
print('最优分类器:',grid.best_params_,'最优分数:', grid.best_score_)  # 得到最优的参数和分值
```

- 调参过程可视化

```python
clf.fit(x_train,y_train)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(min_samples_leaf,clf.cv_results_['mean_train_score']+clf.cv_results_['std_train_score'],
                 clf.cv_results_['mean_train_score']-clf.cv_results_['std_train_score'],color='b')
ax.fill_between(min_samples_leaf,clf.cv_results_['mean_test_score']+clf.cv_results_['std_test_score'],
                 clf.cv_results_['mean_test_score']-clf.cv_results_['std_test_score'],color='r')
ax.plot(min_samples_leaf,clf.cv_results_['mean_train_score'],'ko-')
ax.plot(min_samples_leaf,clf.cv_results_['mean_test_score'],'g*-')
plt.legend()
plt.title('GridSearchCV训练过程图')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']  # 设置正常显示中文
plt.show()
```

##### .2. Cross_validation 调参

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
```

##### .3. **XGBClassifier**

```python
#分类器使用 xgboost
clf1 = xgb.XGBClassifier()
 
#设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
        'n_estimators':range(80,200,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1)
        }
 
#GridSearchCV参数说明，clf1设置训练的学习器
#param_dist字典类型，放入参数搜索范围
#scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
#n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
#n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
grid = GridSearchCV(clf1,param_dist,cv = 3,scoring = 'neg_log_loss',n_iter=300,n_jobs = -1)
 
#在训练集上训练
grid.fit(traindata.values,np.ravel(trainlabel.values))
#返回最优的训练器
best_estimator = grid.best_estimator_
print(best_estimator)
#输出最优训练器的精度
print(grid.best_score_)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210414095030.png)

##### .4. **MLPClassifier**

```python
from sklearn import neural_network
mlp=neural_network.MLPClassifier(max_iter=1000)

param_grid = {
    'hidden_layer_sizes':[(10, ), (20, ), (5, 5)],
    'activation':['logistic', 'tanh', 'relu'],
    'alpha':[0.001, 0.01, 0.1, 0.4, 1]
}

gscv = model_selection.GridSearchCV(estimator=mlp,
                                   param_grid=param_grid,
                                   scoring='accuracy', # 打分
                                   cv=gkf.split(X,y,groups), # cv 方法
                                   return_train_score=True, # 默认不返回 train 的score
                                   refit=True, # 默认为 True, 用最好的模型+全量数据再次训练，用 gscv.best_estimator_ 获取最好模型
                                   n_jobs=-1)

gscv.fit(X,y)
gscv.cv_results_

gscv.best_score_
gscv.best_params_
best_model = gscv.best_estimator_
best_model.score(test_data, test_target)
```

##### .5. **SVC 调参数**

```python
#把要调整的参数以及其候选值 列出来；
param_grid = [
	{'kernel':['linear'],'C':[1,10,100,1000]},
	{'kernel':['poly'],'C':[1,10],'degree':[2,3]},
	{'kernel':['rbf'],'C':[1,10,100,1000],'gamma':[1,0.1, 0.01, 0.001]}]
print("Parameters:{}".format(param_grid))
grid_search = GridSearchCV(SVC(),param_grid,cv=5) #实例化一个GridSearchCV类
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=10)
grid_search.fit(X_train,y_train) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
print("Test set score:{:.2f}".format(grid_search.score(X_test,y_test)))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.2f}".format(grid_search.best_score_))
```

##### .6. Multi-metric Evaluation on Cross_val_score

```python
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

print(__doc__)
X, y = make_hastie_10_2(n_samples=8000, random_state=42)

# The scorers can be either one of the predefined metric strings or a scorer
# callable, like the one returned by make_scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# Setting refit='AUC', refits an estimator on the whole dataset with the
# parameter setting that has the best cross-validated AUC score.
# That estimator is made available at ``gs.best_estimator_`` along with
# parameters like ``gs.best_score_``, ``gs.best_params_`` and
# ``gs.best_index_``
gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid={'min_samples_split': range(2, 403, 10)},
                  scoring=scoring, refit='AUC', return_train_score=True)
gs.fit(X, y)
results = gs.cv_results_
plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 402)
ax.set_ylim(0.73, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210522185021612.png)

##### .7. Selecting dimensionality reduction with Pipeline

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

print(__doc__)

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', 'passthrough'),
    ('classify', LinearSVC(dual=False, max_iter=10000))
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
X, y = load_digits(return_X_y=True)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()
```

#### 1.3. 代码理解

> 该类在搜索参数空间的时候使用到：ParameterGrid:

```python
>>> from sklearn.model_selection import ParameterGrid
>>> param_grid = {'a': [1, 2], 'b': [True, False]}
>>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
            ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')
```

#### 1.4. 自定义模型使用

- 自定义评价函数

```python
import numpy as np
from sklearn.metrics import make_scorer
 
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll
 
#这里的greater_is_better参数决定了自定义的评价指标是越大越好还是越小越好
loss  = make_scorer(logloss, greater_is_better=False)
score = make_scorer(logloss, greater_is_better=True)
```

- 自定义模型

```python
class mymodel():
    def __init__(self, h=None, lam=1,maxiter=500, tol=1e-6):
        self.beta = None
        self.h = h
        self.dataset = None
        self.maxiter = maxiter
        self.tol = tol
        self.funvalue = None
        self.coef = None
        self.dataset = None
        self.lam=lam
        self.iteration=maxiter


    def fit(self, X_train, y_train):
		#用于训练模型参数，例如self.coef
        self.coef, self.funvalue = myfun(X_train, y_train)

    def predict(self, X_new):
		#用于根据X预测y，返回y的预测值数组
		#XXXXXXX
        return y_pre


    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['h','lam','maxiter','tol']:#这里是所用超参数的list
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    def score(self, X, y, sample_weight=None):
    	#如果这里不设置score函数，可以在GridSearchCV()的scoring参数中指定
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        return myloss_fun(y, self.predict(X), sample_weight=sample_weight)
```

### 2. [HalvingGridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV) & [HalvingRandomSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV)

> 新类使用锦标赛方法（tournament approach）选择最佳超参数。它们在观测数据的子集上训练超参数组合，得分最高的超参数组合会进入下一轮。在下一轮中，它们会在大量观测中获得分数。比赛一直持续到最后一轮。确定传递给 HalvingGridSearchCV 或 halvingAndomSearchCV 的超参数需要进行一些计算，你也可以使用合理的默认值。
>
> - 如果没有太多的超参数需要调优，并且 pipeline 运行时间不长，请使用 GridSearchCV；
> - 对于较大的搜索空间和训练缓慢的模型，请使用 HalvingGridSearchCV；
> - 对于非常大的搜索空间和训练缓慢的模型，请使用 HalvingRandomSearchCV。

>  [`HalvingGridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV) and [`HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV) can be used as drop-in replacement for [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) and [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV). Successive Halving is an iterative selection process illustrated in the figure below. `The first iteration is run with a small amount of resources`, where the resource typically corresponds to the number of training samples, but can also be an arbitrary integer parameter such as `n_estimators` in a random forest. `Only a subset of the parameter candidates are selected for the next iteration`, which will be run with an increasing amount of allocated resources. `Only a subset of candidates will last until the end of the iteration process`, and the best parameter candidate is the one that has the highest score on the last iteration.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210414101408.png)

#### 2.1. 函数介绍

*class* `sklearn.model_selection.``HalvingGridSearchCV`(*estimator*, *param_grid*, ***, *factor=3*, *resource='n_samples'*, *max_resources='auto'*, *min_resources='exhaust'*, *aggressive_elimination=False*, *cv=5*, *scoring=None*, *refit=True*, *error_score=nan*, *return_train_score=True*, *random_state=None*, *n_jobs=None*, *verbose=0*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_search_successive_halving.py#L335)[¶](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV)

> The search strategy starts evaluating all the candidates with a small amount of resources and iteratively selects the best candidates, using more and more resources.

*class* `sklearn.model_selection.``HalvingRandomSearchCV`(*estimator*, *param_distributions*, ***, *n_candidates='exhaust'*, *factor=3*, *resource='n_samples'*, *max_resources='auto'*, *min_resources='smallest'*, *aggressive_elimination=False*, *cv=5*, *scoring=None*, *refit=True*, *error_score=nan*, *return_train_score=True*, *random_state=None*, *n_jobs=None*, *verbose=0*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_search_successive_halving.py#L613)

> The search strategy starts evaluating all the candidates with a small amount of resources and iteratively selects the best candidates, using more and more resources.

#### 2.2. 使用案例

##### .1. **RandomForestClassifier**

```python
import numpy as np
from scipy.stats import randint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rng = np.random.RandomState(0)

X, y = make_classification(n_samples=700, random_state=rng)

clf = RandomForestClassifier(n_estimators=10, random_state=rng)

param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 11),
              "min_samples_split": randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rsh = HalvingRandomSearchCV(estimator=clf, param_distributions=param_dist,
                            factor=2, random_state=rng)
rsh.fit(X, y)
rsh.best_params_
```

- **可视化代码**

```python
results = pd.DataFrame(rsh.cv_results_)
results['params_str'] = results.params.apply(str)
results.drop_duplicates(subset=('params_str', 'iter'), inplace=True)
mean_scores = results.pivot(index='iter', columns='params_str',
                            values='mean_test_score')
ax = mean_scores.plot(legend=False, alpha=.6)

labels = [
    f'iter={i}\nn_samples={rsh.n_resources_[i]}\n'
    f'n_candidates={rsh.n_candidates_[i]}'
    for i in range(rsh.n_iterations_)
]
ax.set_xticks(range(rsh.n_iterations_))
ax.set_xticklabels(labels, rotation=45, multialignment='left')
ax.set_title('Scores of candidates over iterations')
ax.set_ylabel('mean test score', fontsize=15)
ax.set_xlabel('iterations', fontsize=15)
plt.tight_layout()
plt.show()
```

##### .2. TimeCompare visualization

```python
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
print(__doc__)

rng = np.random.RandomState(0)
X, y = datasets.make_classification(n_samples=1000, random_state=rng)
gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
Cs = [1, 10, 100, 1e3, 1e4, 1e5]
param_grid = {'gamma': gammas, 'C': Cs}
clf = SVC(random_state=rng)

tic = time()
gsh = HalvingGridSearchCV(estimator=clf, param_grid=param_grid, factor=2,
                          random_state=rng)
gsh.fit(X, y)
gsh_time = time() - tic

tic = time()
gs = GridSearchCV(estimator=clf, param_grid=param_grid)
gs.fit(X, y)
gs_time = time() - tic

#visualization
def make_heatmap(ax, gs, is_sh=False, make_cbar=False):
    """Helper to make a heatmap."""
    results = pd.DataFrame.from_dict(gs.cv_results_)
    results['params_str'] = results.params.apply(str)
    if is_sh:
        # SH dataframe: get mean_test_score values for the highest iter
        scores_matrix = results.sort_values('iter').pivot_table(
                index='param_gamma', columns='param_C',
                values='mean_test_score', aggfunc='last'
        )
    else:
        scores_matrix = results.pivot(index='param_gamma', columns='param_C',
                                      values='mean_test_score')

    im = ax.imshow(scores_matrix)

    ax.set_xticks(np.arange(len(Cs)))
    ax.set_xticklabels(['{:.0E}'.format(x) for x in Cs])
    ax.set_xlabel('C', fontsize=15)

    ax.set_yticks(np.arange(len(gammas)))
    ax.set_yticklabels(['{:.0E}'.format(x) for x in gammas])
    ax.set_ylabel('gamma', fontsize=15)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if is_sh:
        iterations = results.pivot_table(index='param_gamma',
                                         columns='param_C', values='iter',
                                         aggfunc='max').values
        for i in range(len(gammas)):
            for j in range(len(Cs)):
                ax.text(j, i, iterations[i, j],
                        ha="center", va="center", color="w", fontsize=20)

    if make_cbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_ylabel('mean_test_score', rotation=-90, va="bottom",
                           fontsize=15)


fig, axes = plt.subplots(ncols=2, sharey=True)
ax1, ax2 = axes

make_heatmap(ax1, gsh, is_sh=True)
make_heatmap(ax2, gs, make_cbar=True)

ax1.set_title('Successive Halving\ntime = {:.3f}s'.format(gsh_time),
              fontsize=15)
ax2.set_title('GridSearch\ntime = {:.3f}s'.format(gs_time), fontsize=15)

plt.show()
```



### 3. [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)

#### 3.1. 函数介绍

*class* `sklearn.model_selection.``RandomizedSearchCV`(*estimator*, *param_distributions*, ***, *n_iter=10*, *scoring=None*, *n_jobs=None*, *refit=True*, *cv=None*, *verbose=0*, *pre_dispatch='2\*n_jobs'*, *random_state=None*, *error_score=nan*, *return_train_score=False*)[[source\]](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_search.py#L1294)

- **estimator**：我们要传入的模型，如KNN,LogisticRegression,RandomForestRegression等。
- **params_distributions**：参数分布，字典格式。将我们所传入模型当中的参数组合为一个字典。
- n_iter：随机寻找参数组合的数量，默认值为10。
- **scoring**：**模型的评估方法**。在分类模型中有accuracy,precision,recall_score,roc_auc_score等，在回归模型中有MSE，RMSE等。
- **n_jobs**：并行计算时使用的计算机核心数量，默认值为1。当n_jobs的值设为-1时，则使用所有的处理器。
- iid：bool变量，默认为deprecated，返回值为每折交叉验证的值。当iid = True时，返回的是交叉验证的均值。
- **cv**：交叉验证的折数，最新的sklearn库默认为5。

#### 3.2. 使用案例

##### .1. XGBClassifier

```python
#分类器使用 xgboost
clf1 = xgb.XGBClassifier()
 
#设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
        'n_estimators':range(80,200,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1)
        }
 
#RandomizedSearchCV参数说明，clf1设置训练的学习器
#param_dist字典类型，放入参数搜索范围
#scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
#n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
#n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
grid = RandomizedSearchCV(clf1,param_dist,cv = 3,scoring = 'neg_log_loss',n_iter=300,n_jobs = -1)
 
#在训练集上训练
grid.fit(traindata.values,np.ravel(trainlabel.values))
#返回最优的训练器
best_estimator = grid.best_estimator_
print(best_estimator)
#输出最优训练器的精度
print(grid.best_score_)
```

##### .2. **RandomForestRegressor**  **随机+网格搜索案例**

```python
from sklearn.model_selection import RandomizedSearchCV

RF = RandomForestRegressor()
#设置初始的参数空间
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 2000,num = 10)]
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
max_depth = [5,8,10]
max_features = ['auto','sqrt']
bootstrap = [True,False]
#将参数整理为字典格式
random_params_group = {'n_estimators':n_estimators,
                      'min_samples_split':min_samples_split,
                      'min_samples_leaf':min_samples_leaf,
                      'max_depth':max_depth,
                      'max_features':max_features,
                      'bootstrap':bootstrap}
#建立RandomizedSearchCV模型
random_model =RandomizedSearchCV(RF,param_distributions = random_params_group,n_iter = 100,
scoring = 'neg_mean_squared_error',verbose = 2,n_jobs = -1,cv = 3,random_state = 0)
#使用该模型训练数据
random_model.fit(train_features,train_labels)
#获得Random_model最好的参数
random_model.best_params_
#{'n_estimators': 1200,
# 'min_samples_split': 5,
# 'min_samples_leaf': 4,
# 'max_features': 'auto',
# 'max_depth': 5,
# 'bootstrap': True}
RF = RandomForestRegressor(n_estimators = 1200,min_samples_split = 5,
                           min_samples_leaf = 4,max_features = 'auto',max_depth = 5,bootstrap = True)
RF.fit(train_features,train_labels)
predictions = RF.predict(test_features)

RMSE = np.sqrt(mean_squared_error(test_labels,predictions))

print('模型预测误差:',RMSE)
print('模型的提升效果:{}'.format(round(100*(5.06-4.96)/5.06),2),'%')

#使用网格搜索进行细化处理
from sklearn.model_selection import GridSearchCV
import time

param_grid = {'n_estimators':[1100,1200,1300],
             'min_samples_split':[4,5,6,7],
             'min_samples_leaf':[3,4,5],
             'max_depth':[4,5,6,7]}
RF = RandomForestRegressor()
grid = GridSearchCV(RF,param_grid = param_grid,scoring = 'neg_mean_squared_error',cv = 3,n_jobs = -1)
start_time = time.time()
grid.fit(train_features,train_labels)
end_time = time.time()
print('模型训练用时:{}'.format(end_time - start_time))
grid.best_params_
#{'max_depth': 5,
# 'min_samples_leaf': 5,
# 'min_samples_split': 6,
# 'n_estimators': 1100}
RF = RandomForestRegressor(n_estimators = 1100,min_samples_split = 6,
                           min_samples_leaf = 5,max_features = 'auto',max_depth = 5,bootstrap = True)
RF.fit(train_features,train_labels)
predictions = RF.predict(test_features)

RMSE = np.sqrt(mean_squared_error(test_labels,predictions))

print('模型预测误差:',RMSE)
```

#### 3.3. 代码理解

- 搜索策略：**ParameterSampler**

> （a）对于搜索范围是distribution的超参数，`根据给定的distribution随机采样`；
>
> （b）对于`搜索范围是list的超参数，在给定的list中等概率采样`；
>
> （c）对a、b两步中得到的`n_iter组采样结果，进行遍历`。
>
> （补充）如果给定的搜索范围均为list，则不放回抽样n_iter次。

#### 3.4. 自定义模型

### 4. 跨框架调参

#### 4.1. kears 转 sklearn进行调参

> - 使用scikit-learn封装Keras的模型
> - 使用scikit-learn对Keras的模型进行交叉验证
> - 使用scikit-learn，利用网格搜索调整Keras模型的超参

```python
"""
房价预测数据集 使用sklearn执行超参数搜索
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow_core.python.keras.api._v2 import keras # 不能使用 python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import reciprocal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 0.打印导入模块的版本
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, sklearn, pd, tf, keras:
  print("%s version:%s" % (module.__name__, module.__version__))


# 显示学习曲线
def plot_learning_curves(his):
  pd.DataFrame(his.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.show()


# 1.加载数据集 california 房价
housing = fetch_california_housing()

print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

# 2.拆分数据集 训练集 验证集 测试集
x_train_all, x_test, y_train_all, y_test = train_test_split(
  housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(
  x_train_all, y_train_all, random_state=11)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# 3.数据集归一化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)


# 创建keras模型
def build_model(hidden_layers=1, # 中间层的参数
        layer_size=30,
        learning_rate=3e-3):
  # 创建网络层
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(layer_size, activation="relu",
                 input_shape=x_train.shape[1:]))
 # 隐藏层设置
  for _ in range(hidden_layers - 1):
    model.add(keras.layers.Dense(layer_size,
                   activation="relu"))
  model.add(keras.layers.Dense(1))

  # 优化器学习率
  optimizer = keras.optimizers.SGD(lr=learning_rate)
  model.compile(loss="mse", optimizer=optimizer)

  return model


def main():
  # RandomizedSearchCV

  # 1.转化为sklearn的model
  sk_learn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)

  callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]

  history = sk_learn_model.fit(x_train_scaled, y_train, epochs=100,
                 validation_data=(x_valid_scaled, y_valid),
                 callbacks=callbacks)
  # 2.定义超参数集合
  # f(x) = 1/(x*log(b/a)) a <= x <= b
  param_distribution = {
    "hidden_layers": [1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2),
  }

  # 3.执行超搜索参数
  # cross_validation:训练集分成n份, n-1训练, 最后一份验证.
  random_search_cv = RandomizedSearchCV(sk_learn_model, param_distribution,
                     n_iter=10,
                     cv=3,
                     n_jobs=1)
  random_search_cv.fit(x_train_scaled, y_train, epochs=100,
             validation_data=(x_valid_scaled, y_valid),
             callbacks=callbacks)
  # 4.显示超参数
  print(random_search_cv.best_params_)
  print(random_search_cv.best_score_)
  print(random_search_cv.best_estimator_)

  model = random_search_cv.best_estimator_.model
  print(model.evaluate(x_test_scaled, y_test))

  # 5.打印模型训练过程
  plot_learning_curves(history)


if __name__ == '__main__':
  main()
```



### 学习资源

- https://scikit-learn.org/stable/user_guide.html
- https://scikit-learn.org/stable/auto_examples/index.html
- ToLearn List:
  - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
  - https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch9-use-keras-models-with-scikit-learn-for-general-machine-learning.html
  - https://www.guofei.site/2019/09/28/model_selection.html
  - `关于参数的选择有没有可以优化的地方`



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/sklearn-optimize/  

