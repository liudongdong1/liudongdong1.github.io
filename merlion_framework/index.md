# Merlion_Framework


> 首先时间序列的任务主要包括**时间序列异常点检测及时间序列的预测**，其中时间序列的预测又包括单变量时间序列预测，和多变量时间序列预测。在时间序列预测的方法论上，主要又分为传统的计量方法，如ARIMA等；及最近兴起的机器学习的方法，如LSTM、树模型（Random Forest，GBDT）及Transformer等。对于常用的时序研究工具包的功能对比如下：   [Merlion](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.html)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102163918222.png)

- [`merlion.models`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.models.html#module-merlion.models): A library of models unified under a single shared interface, with specializations for anomaly detection and forecasting. More specifically, we have
  - [`merlion.models.anomaly`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.models.anomaly.html#module-merlion.models.anomaly): Anomaly detection models
  - [`merlion.models.forecast`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.models.forecast.html#module-merlion.models.forecast): Forecasting models
  - [`merlion.models.anomaly.forecast_based`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.models.anomaly.forecast_based.html#module-merlion.models.anomaly.forecast_based): Forecasting models adapted for anomaly detection. Anomaly scores are based on the residual between the predicted and true value at each timestamp.
  - [`merlion.models.ensemble`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.models.ensemble.html#module-merlion.models.ensemble): Ensembles & automated model selection of models for both anomaly detection and forecasting.
  - [`merlion.models.automl`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.models.automl.html#module-merlion.models.automl): AutoML layers for various models
- [`merlion.transform`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.transform.html#module-merlion.transform): Data pre-processing layer which implements many standard data transformations used in time series analysis. Transforms are callable objects, and each model has its own configurable `model.transform` which it uses to pre-process all input time series for both training and inference.
- [`merlion.post_process`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.post_process.html#module-merlion.post_process): Post-processing rules to apply on the output of a model. Currently, these are specific to anomaly detection, and include
  - [`merlion.post_process.calibrate`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.post_process.html#module-merlion.post_process.calibrate): Rules to calibrate the anomaly scores returned by a model, to be interpretable as z-scores, i.e. as standard deviations of a standard normal random variable. Each anomaly detection model has a `model.calibrator` from this module, which can optionally be applied to ensure that the model’s anomaly scores are calibrated.
  - [`merlion.post_process.threshold`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.post_process.html#module-merlion.post_process.threshold): Rules to reduce the noisiness of an anomaly detection model’s outputs. Each anomaly detection model has a `model.threshold` from this module, which can optionally be applied to filter the model’s predicted sequence of anomaly scores.
- [`merlion.evaluate`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.evaluate.html#module-merlion.evaluate): Evaluation metrics & pipelines to simulate the live deployment of a time series model for any task.
- [`merlion.plot`](https://opensource.salesforce.com/Merlion/v1.0.1/merlion.html#module-merlion.plot): Automated visualization of model outputs for univariate time series

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102164955181.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102165004734.png)

### 1. Anomaly Detectors

1. Initializing an anomaly detection model (including ensembles)
2. Training the model
3. Producing a series of anomaly scores with the model
4. Quantitatively evaluating the model
5. Visualizing the model’s predictions
6. Saving and loading a trained model
7. Simulating the live deployment of a model using a `TSADEvaluator`

```python
import matplotlib.pyplot as plt
import numpy as np

from merlion.plot import plot_anoms
from merlion.utils import TimeSeries
from ts_datasets.anomaly import NAB

np.random.seed(1234)
# This is a time series with anomalies in both the train and test split.
# time_series and metadata are both time-indexed pandas DataFrames.
time_series, metadata = NAB(subset="realKnownCause")[3]

# Visualize the full time series
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(time_series)

# Label the train/test split with a dashed line & plot anomalies
ax.axvline(metadata[metadata.trainval].index[-1], ls="--", lw=2, c="k")
plot_anoms(ax, TimeSeries.from_pd(metadata.anomaly))
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102182612865.png)

```python
from merlion.utils import TimeSeries

# Get training split
train = time_series[metadata.trainval]
train_data = TimeSeries.from_pd(train)
train_labels = TimeSeries.from_pd(metadata[metadata.trainval].anomaly)

# Get testing split
test = time_series[~metadata.trainval]
test_data = TimeSeries.from_pd(test)
test_labels = TimeSeries.from_pd(metadata[~metadata.trainval].anomaly)
```

- model initialize

```python
# Import models & configs
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.anomaly.forecast_based.prophet import ProphetDetector, ProphetDetectorConfig

# Import a post-rule for thresholding
from merlion.post_process.threshold import AggregateAlarms

# Import a data processing transform
from merlion.transform.moving_average import DifferenceTransform

# All models are initialized using the syntax ModelClass(config), where config
# is a model-specific configuration object. This is where you specify any
# algorithm-specific hyperparameters, any data pre-processing transforms, and
# the post-rule you want to use to post-process the anomaly scores (to reduce
# noisiness when firing alerts).

# We initialize isolation forest using the default config
config1 = IsolationForestConfig()
model1  = IsolationForest(config1)

# We use a WindStats model that splits each week into windows of 60 minutes
# each. Anomaly scores in Merlion correspond to z-scores. By default, we would
# like to fire an alert for any 4-sigma event, so we specify a threshold rule
# which achieves this.
config2 = WindStatsConfig(wind_sz=60, threshold=AggregateAlarms(alm_threshold=4))
model2  = WindStats(config2)

# Prophet is a popular forecasting algorithm. Here, we specify that we would like
# to pre-processes the input time series by applying a difference transform,
# before running the model on it.
config3 = ProphetDetectorConfig(transform=DifferenceTransform())
model3  = ProphetDetector(config3)
```

- combine them in an ensemble

```python
from merlion.models.ensemble.anomaly import DetectorEnsemble, DetectorEnsembleConfig

ensemble_config = DetectorEnsembleConfig(threshold=AggregateAlarms(alm_threshold=4))
ensemble = DetectorEnsemble(config=ensemble_config, models=[model1, model2, model3])
```

- model train

```python
from merlion.evaluate.anomaly import TSADMetric

# Train IsolationForest in the default way, using the ground truth anomaly labels
# to set the post-rule's threshold
print(f"Training {type(model1).__name__}...")
train_scores_1 = model1.train(train_data=train_data, anomaly_labels=train_labels)

# Train WindStats completely unsupervised (this retains our anomaly detection
# default anomaly detection threshold of 4)
print(f"\nTraining {type(model2).__name__}...")
train_scores_2 = model2.train(train_data=train_data, anomaly_labels=None)

# Train Prophet with the ground truth anomaly labels, with a post-rule
# trained to optimize Precision score
print(f"\nTraining {type(model3).__name__}...")
post_rule_train_config_3 = dict(metric=TSADMetric.F1)
train_scores_3 = model3.train(
    train_data=train_data, anomaly_labels=train_labels,
    post_rule_train_config=post_rule_train_config_3)

# We consider an unsupervised ensemble, which combines the anomaly scores
# returned by the models & sets a static anomaly detection threshold of 3.
print("\nTraining ensemble...")
ensemble_post_rule_train_config = dict(metric=None)
train_scores_e = ensemble.train(
    train_data=train_data, anomaly_labels=train_labels,
    post_rule_train_config=ensemble_post_rule_train_config,
)

print("Done!")
```

- model inference

```python
# Here is a full example for the first model, IsolationForest
scores_1 = model1.get_anomaly_score(test_data)
scores_1_df = scores_1.to_pd()
print(f"{type(model1).__name__}.get_anomaly_score() nonzero values (raw)")
print(scores_1_df[scores_1_df.iloc[:, 0] != 0])
print()

labels_1 = model1.get_anomaly_label(test_data)
labels_1_df = labels_1.to_pd()
print(f"{type(model1).__name__}.get_anomaly_label() nonzero values (post-processed)")
print(labels_1_df[labels_1_df.iloc[:, 0] != 0])
print()

print(f"{type(model1).__name__} fires {(labels_1_df.values != 0).sum()} alarms")
print()

print("Raw scores at the locations where alarms were fired:")
print(scores_1_df[labels_1_df.iloc[:, 0] != 0])
print("Post-processed scores are interpretable as z-scores")
print("Raw scores are challenging to interpret")
```

- model quantitative evaluation

```python
from merlion.evaluate.anomaly import TSADMetric

for model, labels in [(model1, labels_1), (model2, labels_2), (model3, labels_3), (ensemble, labels_e)]:
    print(f"{type(model).__name__}")
    precision = TSADMetric.Precision.value(ground_truth=test_labels, predict=labels)
    recall = TSADMetric.Recall.value(ground_truth=test_labels, predict=labels)
    f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=labels)
    mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=labels)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"MTTD:      {mttd}")
```

- model visualization

```python
for model in [model1, model2, model3]:
    print(type(model).__name__)
    fig, ax = model.plot_anomaly(
        time_series=test_data, time_series_prev=train_data,
        filter_scores=True, plot_time_series_prev=True)
    plot_anoms(ax=ax, anomaly_labels=test_labels)
    plt.show()
    print()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211102183921939.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/merlion_framework/  

