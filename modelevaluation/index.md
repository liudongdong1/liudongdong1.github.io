# ModelEvaluation


### 0. Sklearn Metric

#### .1. Classification metrics

See the [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) section of the user guide for further details.

| [`metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)(y_true, y_pred, *[, …]) | Accuracy classification score.                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.auc`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)(x, y) | Compute Area Under the Curve (AUC) using the trapezoidal rule. |
| [`metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)(y_true, …) | Compute average precision (AP) from prediction scores.       |
| [`metrics.balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)(y_true, …) | Compute the balanced accuracy.                               |
| [`metrics.brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss)(y_true, y_prob, *) | Compute the Brier score loss.                                |
| [`metrics.classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)(y_true, y_pred, *) | Build a text report showing the main classification metrics. |
| [`metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score)(y1, y2, *[, …]) | Cohen’s kappa: a statistic that measures inter-annotator agreement. |
| [`metrics.confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)(y_true, y_pred, *) | Compute confusion matrix to evaluate the accuracy of a classification. |
| [`metrics.dcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.dcg_score.html#sklearn.metrics.dcg_score)(y_true, y_score, *[, k, …]) | Compute Discounted Cumulative Gain.                          |
| [`metrics.det_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.det_curve.html#sklearn.metrics.det_curve)(y_true, y_score[, …]) | Compute error rates for different probability thresholds.    |
| [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)(y_true, y_pred, *[, …]) | Compute the F1 score, also known as balanced F-score or F-measure. |
| [`metrics.fbeta_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score)(y_true, y_pred, *, beta) | Compute the F-beta score.                                    |
| [`metrics.hamming_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss)(y_true, y_pred, *[, …]) | Compute the average Hamming loss.                            |
| [`metrics.hinge_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss)(y_true, pred_decision, *) | Average hinge loss (non-regularized).                        |
| [`metrics.jaccard_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score)(y_true, y_pred, *[, …]) | Jaccard similarity coefficient score.                        |
| [`metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss)(y_true, y_pred, *[, eps, …]) | Log loss, aka logistic loss or cross-entropy loss.           |
| [`metrics.matthews_corrcoef`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef)(y_true, y_pred, *) | Compute the Matthews correlation coefficient (MCC).          |
| [`metrics.multilabel_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html#sklearn.metrics.multilabel_confusion_matrix)(y_true, …) | Compute a confusion matrix for each class or sample.         |
| [`metrics.ndcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html#sklearn.metrics.ndcg_score)(y_true, y_score, *[, k, …]) | Compute Normalized Discounted Cumulative Gain.               |
| [`metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)(y_true, …) | Compute precision-recall pairs for different probability thresholds. |
| [`metrics.precision_recall_fscore_support`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)(…) | Compute precision, recall, F-measure and support for each class. |
| [`metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)(y_true, y_pred, *[, …]) | Compute the precision.                                       |
| [`metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)(y_true, y_pred, *[, …]) | Compute the recall.                                          |
| [`metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)(y_true, y_score, *[, …]) | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. |
| [`metrics.roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)(y_true, y_score, *[, …]) | Compute Receiver operating characteristic (ROC).             |
| [`metrics.top_k_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html#sklearn.metrics.top_k_accuracy_score)(y_true, y_score, *) | Top-k Accuracy classification score.                         |
| [`metrics.zero_one_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html#sklearn.metrics.zero_one_loss)(y_true, y_pred, *[, …]) | Zero-one classification loss.                                |

#### .2. Regression metrics

See the [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) section of the user guide for further details.

| [`metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)(y_true, …) | Explained variance regression score function.                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.max_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error)(y_true, y_pred) | max_error metric calculates the maximum residual error.      |
| [`metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)(y_true, y_pred, *) | Mean absolute error regression loss.                         |
| [`metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)(y_true, y_pred, *) | Mean squared error regression loss.                          |
| [`metrics.mean_squared_log_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error)(y_true, y_pred, *) | Mean squared logarithmic error regression loss.              |
| [`metrics.median_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error)(y_true, y_pred, *) | Median absolute error regression loss.                       |
| [`metrics.mean_absolute_percentage_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error)(…) | Mean absolute percentage error regression loss.              |
| [`metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)(y_true, y_pred, *[, …]) | R2 (coefficient of determination) regression score function. |
| [`metrics.mean_poisson_deviance`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html#sklearn.metrics.mean_poisson_deviance)(y_true, y_pred, *) | Mean Poisson deviance regression loss.                       |
| [`metrics.mean_gamma_deviance`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html#sklearn.metrics.mean_gamma_deviance)(y_true, y_pred, *) | Mean Gamma deviance regression loss.                         |
| [`metrics.mean_tweedie_deviance`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html#sklearn.metrics.mean_tweedie_deviance)(y_true, y_pred, *) | Mean Tweedie deviance regression loss.                       |

#### .3. Multilabel ranking metrics

See the [Multilabel ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) section of the user guide for further details.

| [`metrics.coverage_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.coverage_error.html#sklearn.metrics.coverage_error)(y_true, y_score, *[, …]) | Coverage error measure.                  |
| ------------------------------------------------------------ | ---------------------------------------- |
| [`metrics.label_ranking_average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score)(…) | Compute ranking-based average precision. |
| [`metrics.label_ranking_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html#sklearn.metrics.label_ranking_loss)(y_true, y_score, *) | Compute Ranking loss measure.            |

#### .4. Clustering metrics

The [`sklearn.metrics.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster) submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:

- supervised, which uses a ground truth class values for each sample.
- unsupervised, which does not and measures the ‘quality’ of the model itself.

| [`metrics.adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score)(…[, …]) | Adjusted Mutual Information between two clusterings.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score)(labels_true, …) | Rand index adjusted for chance.                              |
| [`metrics.calinski_harabasz_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score)(X, labels) | Compute the Calinski and Harabasz score.                     |
| [`metrics.davies_bouldin_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score)(X, labels) | Computes the Davies-Bouldin score.                           |
| [`metrics.completeness_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score)(labels_true, …) | Completeness metric of a cluster labeling given a ground truth. |
| [`metrics.cluster.contingency_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html#sklearn.metrics.cluster.contingency_matrix)(…[, …]) | Build a contingency matrix describing the relationship between labels. |
| [`metrics.cluster.pair_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html#sklearn.metrics.cluster.pair_confusion_matrix)(…) | Pair confusion matrix arising from two clusterings.          |
| [`metrics.fowlkes_mallows_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score)(labels_true, …) | Measure the similarity of two clusterings of a set of points. |
| [`metrics.homogeneity_completeness_v_measure`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html#sklearn.metrics.homogeneity_completeness_v_measure)(…) | Compute the homogeneity and completeness and V-Measure scores at once. |
| [`metrics.homogeneity_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score)(labels_true, …) | Homogeneity metric of a cluster labeling given a ground truth. |
| [`metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score)(labels_true, …) | Mutual Information between two clusterings.                  |
| [`metrics.normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score)(…[, …]) | Normalized Mutual Information between two clusterings.       |
| [`metrics.rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score)(labels_true, labels_pred) | Rand index.                                                  |
| [`metrics.silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)(X, labels, *[, …]) | Compute the mean Silhouette Coefficient of all samples.      |
| [`metrics.silhouette_samples`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples)(X, labels, *[, …]) | Compute the Silhouette Coefficient for each sample.          |
| [`metrics.v_measure_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score)(labels_true, …[, beta]) | V-measure cluster labeling given a ground truth.             |

#### .5. Pairwise metrics

| [`metrics.pairwise.additive_chi2_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.additive_chi2_kernel.html#sklearn.metrics.pairwise.additive_chi2_kernel)(X[, Y]) | Computes the additive chi-squared kernel between observations in X and Y. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.pairwise.chi2_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.chi2_kernel.html#sklearn.metrics.pairwise.chi2_kernel)(X[, Y, gamma]) | Computes the exponential chi-squared kernel X and Y.         |
| [`metrics.pairwise.cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity)(X[, Y, …]) | Compute cosine similarity between samples in X and Y.        |
| [`metrics.pairwise.cosine_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances)(X[, Y]) | Compute cosine distance between samples in X and Y.          |
| [`metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics)() | Valid metrics for pairwise_distances.                        |
| [`metrics.pairwise.euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances)(X[, Y, …]) | Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors. |
| [`metrics.pairwise.haversine_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html#sklearn.metrics.pairwise.haversine_distances)(X[, Y]) | Compute the Haversine distance between samples in X and Y.   |
| [`metrics.pairwise.kernel_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics)() | Valid metrics for pairwise_kernels.                          |
| [`metrics.pairwise.laplacian_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html#sklearn.metrics.pairwise.laplacian_kernel)(X[, Y, gamma]) | Compute the laplacian kernel between X and Y.                |
| [`metrics.pairwise.linear_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html#sklearn.metrics.pairwise.linear_kernel)(X[, Y, …]) | Compute the linear kernel between X and Y.                   |
| [`metrics.pairwise.manhattan_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html#sklearn.metrics.pairwise.manhattan_distances)(X[, Y, …]) | Compute the L1 distances between the vectors in X and Y.     |
| [`metrics.pairwise.nan_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html#sklearn.metrics.pairwise.nan_euclidean_distances)(X) | Calculate the euclidean distances in the presence of missing values. |
| [`metrics.pairwise.pairwise_kernels`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html#sklearn.metrics.pairwise.pairwise_kernels)(X[, Y, …]) | Compute the kernel between arrays X and optional array Y.    |
| [`metrics.pairwise.polynomial_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html#sklearn.metrics.pairwise.polynomial_kernel)(X[, Y, …]) | Compute the polynomial kernel between X and Y.               |
| [`metrics.pairwise.rbf_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel)(X[, Y, gamma]) | Compute the rbf (gaussian) kernel between X and Y.           |
| [`metrics.pairwise.sigmoid_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html#sklearn.metrics.pairwise.sigmoid_kernel)(X[, Y, …]) | Compute the sigmoid kernel between X and Y.                  |
| [`metrics.pairwise.paired_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_euclidean_distances.html#sklearn.metrics.pairwise.paired_euclidean_distances)(X, Y) | Computes the paired euclidean distances between X and Y.     |
| [`metrics.pairwise.paired_manhattan_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_manhattan_distances.html#sklearn.metrics.pairwise.paired_manhattan_distances)(X, Y) | Compute the L1 distances between the vectors in X and Y.     |
| [`metrics.pairwise.paired_cosine_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_cosine_distances.html#sklearn.metrics.pairwise.paired_cosine_distances)(X, Y) | Computes the paired cosine distances between X and Y.        |
| [`metrics.pairwise.paired_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html#sklearn.metrics.pairwise.paired_distances)(X, Y, *[, …]) | Computes the paired distances between X and Y.               |
| [`metrics.pairwise_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances)(X[, Y, metric, …]) | Compute the distance matrix from a vector array X and optional Y. |
| [`metrics.pairwise_distances_argmin`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html#sklearn.metrics.pairwise_distances_argmin)(X, Y, *[, …]) | Compute minimum distances between one point and a set of points. |
| [`metrics.pairwise_distances_argmin_min`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html#sklearn.metrics.pairwise_distances_argmin_min)(X, Y, *) | Compute minimum distances between one point and a set of points. |
| [`metrics.pairwise_distances_chunked`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html#sklearn.metrics.pairwise_distances_chunked)(X[, Y, …]) | Generate a distance matrix chunk by chunk with optional reduction. |

#### .6. Plotting

| [`metrics.plot_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix)(estimator, X, …) | Plot Confusion Matrix.                              |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [`metrics.plot_det_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_det_curve.html#sklearn.metrics.plot_det_curve)(estimator, X, y, *[, …]) | Plot detection error tradeoff (DET) curve.          |
| [`metrics.plot_precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html#sklearn.metrics.plot_precision_recall_curve)(…[, …]) | Plot Precision Recall Curve for binary classifiers. |
| [`metrics.plot_roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html#sklearn.metrics.plot_roc_curve)(estimator, X, y, *[, …]) | Plot Receiver operating characteristic (ROC) curve. |

| [`metrics.ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay)(…[, …]) | Confusion Matrix visualization. |
| ------------------------------------------------------------ | ------------------------------- |
| [`metrics.DetCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DetCurveDisplay.html#sklearn.metrics.DetCurveDisplay)(*, fpr, fnr[, …]) | DET curve visualization.        |
| [`metrics.PrecisionRecallDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay)(precision, …) | Precision Recall visualization. |
| [`metrics.RocCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay)(*, fpr, tpr[, …]) | ROC Curve visualization.        |

![image-20210523215145923](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210523215145923.png)

### 1. IOU

> 预测框与标注框的交集与并集之比，数值越大表示该检测器的性能越好。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523213155615.png)

### 2. Precision

> 查准率或者是精确率,是指在所有系统判定的“真”的样本中，确实是真的的占比

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523215727400.png)

### 3. Accuracy

> accuracy针对所有样本

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210523213453110.png)

### 4. Recall

> 在所有确实为真的样本中，被判为的“真”的占比.

### 5. PRC图例

> 以查准率为Y轴，、查全率为X轴做的图。它是综合评价整体结果的评估指标。所以，哪总类型（正或者负）样本多，权重就大。在进行比较时，若一个学习器的PR曲线被另一个学习器的曲线完全包住，则可断言后者的性能优于前者。`比较PR曲线下的面积`，该指标在一定程度上表征了学习器在查准率和查全率上取得相对“双高”的比例。因为这个值不容易估算，所以人们引入`“平衡点”(BEP)`来度量，他表示“查准率=查全率”时的取值，`值越大表明分类器性能越好`

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210523213628048.png)

> F1-score 就是一个综合考虑precision和recall的指标，比BEP更为常用。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210523213914515.png)

### 6. ROC&AUC&K-S曲线

> ROC全称是“受试者工作特征”（Receiver Operating Characteristic）曲线，ROC曲线以“真正例率”（TPR）为Y轴，以“假正例率”（FPR）为X轴，对角线对应于“随机猜测”模型，而（0,1）则对应“理想模型”。ROC形式如下图所示。**针对二分类**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523214430826.png)

> 若一个学习器的ROC曲线被另一个学习器的曲线包住，那么我们可以断言后者性能优于前者；若两个学习器的ROC曲线发生交叉，则难以一般性断言两者孰优孰劣。此时若要进行比较，那么可以比较ROC曲线下的面积，即AUC，面积大的曲线对应的分类器性能更好。

> AUC（Area Under Curve）的值为ROC曲线下面的面积，若分类器的性能极好，则AUC为1。一般AUC均在0.5到1之间，AUC越高，模型的区分能力越好.**0.85 – 0.95：** 效果很好**0.95 – 1：** 效果非常好，但一般不太可能

- KS值越大，说明模型能将两类样本区分开的能力越大。

> 先将实例`按照模型输出值进行排序`，通过`改变不同的阈值得到小于（或大于）某个阈值时`，对应实例集合中正（负）样本占全部正（负）样本的`比例`（即TPR 和 FPR，和 ROC 曲线使用的指标一样，只是两者的横坐标不同）。由`小到大改变阈值从而得到多个点`，将这些点连接后分别得到`正、负实例累积曲线`。正、负实例累积曲线相减得到KS曲线， KS曲线的最高点即KS值，该点所对应的阈值划分点即模型最佳划分能力的点。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523222157838.png)

### 7. Confusion Matrix

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210523214926412.png)

### 8. 泛化能力

> 泛化能力指的是训练得到的模型**对未知数据的预测能力**。
>
> - 损失函数：度量预测错误程度的函数
> - 训练误差：训练数据集上的平均损失，虽然有意义，但本质不重要
> - 测试误差：测试数据集上的平均损失，反应了模型对未知数据的预测能力

### 9. 过拟合&欠拟合

> 当机器学习模型对训练集学习的太好的时候，此时表现为训练误差很小，而泛化误差会很大，这种情况我们称之为**过拟合**：
>
> - **模型记住了数据中的噪音** 意味着模型受到噪音的干扰，导致拟合的函数形状与实际总体的数据分布相差甚远。这里的噪音可以是标记错误的样本，也可以是少量明显偏离总体分布的样本（异常点）。通过清洗样本或异常值处理可以帮助缓解这个问题。
> - **训练数据过少** 导致训练的数据集根本无法代表整体的数据情况，做什么也是徒劳的。需要想方设法增加数据，包括人工合成假样本。
> - **模型复杂度过高** 导致模型对训练数据学习过度
>
> 当模型在数据集上学习的不够好的时候，此时训练误差较大，这种情况我们称之为**欠拟合**：
>
> - **模型过于简单** 即模型形式太简单，以致于无法捕捉到数据特征，无法很好的拟合数据

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523221850819.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523221348618.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523221554551.png)

### 10. [偏差和方差](https://machine-learning-from-scratch.readthedocs.io/zh_CN/latest/模型评估与模型调优.html#id47)

> **偏差：**the difference between your model’s expected predictions and the true values. `衡量了模型期望输出与真实值之间的差别`，刻画了模型本身的拟合能力。
>
> **方差：**refers to your algorithm’s sensitivity to specific sets of training data. High variance algorithms will produce drastically different models depending on the training set. `度量了训练集的变动所导致的学习性能的变化`，刻画了模型输出结果由于训练集的不同造成的波动。
>
> **噪音：**度量了在当前任务上任何学习算法所能达到的期望泛化误差的下界，刻画了学习问题本身的难度。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523221808263.png)

### 11. 回归度量

#### .1. 平均绝对误差MAE

> 缺点是该误差形式没有二阶导数，导致不能用某些方法优化。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523222358303.png)

#### .2. 均方根误差RMSE

> 对大误差的样本有更多的惩罚，因此也对离群点更敏感。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523222434448.png)

#### .3. 均方根对数误差RMSLE

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523222538084.png)

> 当真实值的分布范围比较广时（如：年收入可以从 0 到非常大的数），如果使用`MAE、MSE、RMSE` 等误差，这将使得模型更`关注于那些真实标签值较大的样本`。而`RMSLE` 关注的是预测误差的比例，使得`真实标签值较小的样本也同等重要`。当数据中存在标签较大的异常值时，`RMSLE` 能够降低这些异常值的影响。

### 12. PSI(模型稳定性)

> 稳定度指标(population stability index ,PSI)可衡量测试样本及模型开发样本评分的的分布差异，为最常见的模型稳定度评估指针。其实PSI表示的就是按分数分档后，针对不同样本，或者不同时间的样本.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523230309877.png)

![image-20210523230436644](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523230436644.png)

### 13. 验证测试集

- 测试集通常用于对模型的预测能力进行评估，它提供了模型预测能力的无偏估计。如果你不需要对模型预测能力的无偏估计，则不需要测试集。
- 验证集用于超参数的选择，因为模型依赖于超参数，而超参数依赖于验证集。因此验证集参与了模型的构建，这意味着模型已经考虑了验证集的信息。所以我们需要一份单独的测试集来估计模型的泛化能力。

- 如果未设置验证集，则将数据三七分：70% 的数据用作训练集、30% 的数据用作测试集。
- 如果设置验证集，则将数据划分为：60% 的数据用作训练集、20%的数据用过验证集、20% 的数据用作测试集。

- 必须保证验证集、测试集的`分布一致`，它们都要很好的代表你的真实应用场景中的数据分布。

- 如果训练集和验证集的分布一致，那么当训练误差和验证误差相差较大时，我们认为存在很大的方差问题。
- 如果训练集和验证集的分布不一致，那么当训练误差和验证误差相差较大时，有两种原因：
  - 第一个原因：模型只见过训练集数据，没有见过验证集的数据导致的，是数据不匹配的问题。
  - 第二个原因：模型本来就存在较大的方差。



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/modelevaluation/  

