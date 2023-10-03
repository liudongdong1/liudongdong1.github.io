# TFLearn


> TFlearn is a modular and transparent deep learning library built on top of Tensorflow. It was designed to provide a` higher-level API to TensorFlow` in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.

### 1. layers

| File                                                      | Layers                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| [core](http://tflearn.org/layers/core/)                   | input_data, fully_connected, dropout, custom_layer, reshape, flatten, activation, single_unit, highway, one_hot_encoding, time_distributed |
| [conv](http://tflearn.org/layers/conv/)                   | conv_2d, conv_2d_transpose, max_pool_2d, avg_pool_2d, upsample_2d, conv_1d, max_pool_1d, avg_pool_1d, residual_block, residual_bottleneck, conv_3d, max_pool_3d, avg_pool_3d, highway_conv_1d, highway_conv_2d, global_avg_pool, global_max_pool |
| [recurrent](http://tflearn.org/layers/recurrent/)         | simple_rnn, lstm, gru, bidirectionnal_rnn, dynamic_rnn       |
| [embedding](http://tflearn.org/layers/embedding_ops/)     | embedding                                                    |
| [normalization](http://tflearn.org/layers/normalization/) | batch_normalization, local_response_normalization, l2_normalize |
| [merge](http://tflearn.org/layers/merge_ops/)             | merge, merge_outputs                                         |
| [estimator](http://tflearn.org/layers/estimator/)         | regression                                                   |

```python
tflearn.conv_2d(x, 32, 5, activation='relu', name='conv1')
```

### 2. build-in Op

| File                                                  | Ops                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| [activations](http://tflearn.org/activations)         | linear, tanh, sigmoid, softmax, softplus, softsign, relu, relu6, leaky_relu, prelu, elu |
| [objectives](http://tflearn.org/objectives)           | softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss, roc_auc_score, weak_cross_entropy_2d |
| [optimizers](http://tflearn.org/optimizers)           | SGD, RMSProp, Adam, Momentum, AdaGrad, Ftrl, AdaDelta        |
| [metrics](http://tflearn.org/metrics)                 | Accuracy, Top_k, R2                                          |
| [initializations](http://tflearn.org/initializations) | zeros, uniform, uniform_scaling, normal, truncated_normal, xavier, variance_scaling |
| [losses](http://tflearn.org/losses)                   | l1, l2                                                       |

```python
# Activation and Regularization inside a layer:
fc2 = tflearn.fully_connected(fc1, 32, activation='tanh', regularizer='L2')
# Equivalent to:
fc2 = tflearn.fully_connected(fc1, 32)
tflearn.add_weights_regularization(fc2, loss='L2')
fc2 = tflearn.tanh(fc2)

# Optimizer, Objective and Metric:
reg = tflearn.regression(fc4, optimizer='rmsprop', metric='accuracy', loss='categorical_crossentropy')
# Ops can also be defined outside, for deeper customization:
momentum = tflearn.optimizers.Momentum(learning_rate=0.1, weight_decay=0.96, decay_step=200)
top5 = tflearn.metrics.Top_k(k=5)
reg = tflearn.regression(fc4, optimizer=momentum, metric=top5, loss='categorical_crossentropy')
```

### 3. Trainning, Evaluating&Predicting

```python
network = ... (some layers) ...
network = regression(network, optimizer='sgd', loss='categorical_crossentropy')

model = DNN(network)
model.fit(X, Y)
//----------------------
network = ...
model = DNN(network)
model.load('model.tflearn')
model.predict(X)
```

### 4. Visualization

- 0: Loss & Metric (Best speed).
- 1: Loss, Metric & Gradients.
- 2: Loss, Metric, Gradients & Weights.
- 3: Loss, Metric, Gradients, Weights, Activations & Sparsity (Best Visualization).

```python
model = DNN(network, tensorboard_verbose=3)

#tensorboard --logdir='/tmp/tflearn_logs'          #shell command
```

### 5. Persistence

```python
# Save a model
model.save('my_model.tflearn')
# Load a model
model.load('my_model.tflearn')

# Let's create a layer
fc1 = fully_connected(input_layer, 64, name="fc_layer_1")
# Using Tensor attributes (Layer will supercharge the returned Tensor with weights attributes)
fc1_weights_var = fc1.W
fc1_biases_var = fc1.b
# Using Tensor name
fc1_vars = tflearn.get_layer_variables_by_name("fc_layer_1")
fc1_weights_var = fc1_vars[0]
fc1_biases_var = fc1_vars[1]

input_data = tflearn.input_data(shape=[None, 784])
fc1 = tflearn.fully_connected(input_data, 64)
fc2 = tflearn.fully_connected(fc1, 10, activation='softmax')
net = tflearn.regression(fc2)
model = DNN(net)
# Get weights values of fc2
model.get_weights(fc2.W)
# Assign new random weights to fc2
model.set_weights(fc2.W, numpy.random.rand(64, 10))
```

### 6. Fine-tuning

> -  specify which layer's weights you want to be restored or not (when loading pre-trained model). 
>
> -  share variables among multiple layers and make TFLearn suitable for distributed training. All layers with inner variables support a 'scope' argument to place variables under; layers with same scope name will then share the same weights.

```python
# Weights will be restored by default.
fc_layer = tflearn.fully_connected(input_layer, 32)
# Weights will not be restored, if specified so.
fc_layer = tflearn.fully_connected(input_layer, 32, restore='False')
```

### 7. DataPreprocessing&Augmentation

```python
# Real-time image preprocessing
img_prep = tflearn.ImagePreprocessing()
# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center()
# STD Normalization (With std computed over the whole dataset)
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
# Random flip an image
img_aug.add_random_flip_leftright()

# Add these methods into an 'input_data' layer
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
```

### 8.Models

#### 8.1. DNN

> **tflearn.models.dnn.DNN** (network, clip_gradients=5.0, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/', checkpoint_path=None, best_checkpoint_path=None, max_checkpoints=None, session=None, best_val_accuracy=0.0)

- **network**: `Tensor`. Neural network to be used.
- **tensorboard_verbose**: `int`. Summary verbose level, it accepts different levels of tensorboard logs:

```
0: Loss, Accuracy (Best Speed).
1: Loss, Accuracy, Gradients.
2: Loss, Accuracy, Gradients, Weights.
3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.(Best visualization)
```

- **tensorboard_dir**: `str`. Directory to store tensorboard logs. Default: "/tmp/tflearn_logs/"
- **checkpoint_path**: `str`. Path to store model checkpoints. If None, no model checkpoint will be saved. Default: None.
- **best_checkpoint_path**: `str`. Path to store the model when the validation rate reaches its highest point of the current training session and also is above best_val_accuracy. Default: None.
- **max_checkpoints**: `int` or None. Maximum amount of checkpoints. If None, no limit. Default: None.
- **session**: `Session`. A session for running ops. If None, a new one will be created. Note: When providing a session, variables must have been initialized already, otherwise an error will be raised.
- **best_val_accuracy**: `float` The minimum validation accuracy that needs to be achieved before a model weight's are saved to the best_checkpoint_path. This allows the user to skip early saves and also set a minimum save point when continuing to train a reloaded model. Default: 0.0.

##### .1.1.  evaluate, load, save, predict

##### .1.2. Fit

> **fit** (X_inputs, Y_targets, n_epoch=10, validation_set=None, show_metric=False, batch_size=None, shuffle=None, snapshot_epoch=True, snapshot_step=None, excl_trainops=None, validation_batch_size=None, run_id=None, callbacks=[])

- **X_inputs**: array, `list` of array (if multiple inputs) or `dict` (with inputs layer name as keys). Data to feed to train model.
- **Y_targets**: array, `list` of array (if multiple inputs) or `dict` (with estimators layer name as keys). Targets (Labels) to feed to train model.
- **n_epoch**: `int`. Number of epoch to run. Default: None.
- **validation_set**: `tuple`. Represents data used for validation. `tuple` holds data and targets (provided as same type as X_inputs and Y_targets). Additionally, it also accepts `float` (<1) to performs a data split over training data.
- **show_metric**: `bool`. Display or not accuracy at every step.
- **batch_size**: `int` or None. If `int`, overrides all network estimators 'batch_size' by this value. Also overrides `validation_batch_size` if `int`, and if `validation_batch_size` is None.
- **validation_batch_size**: `int` or None. If `int`, overrides all network estimators 'validation_batch_size' by this value.
- **shuffle**: `bool` or None. If `bool`, overrides all network estimators 'shuffle' by this value.
- **snapshot_epoch**: `bool`. If True, it will snapshot model at the end of every epoch. (Snapshot a model will evaluate this model on validation set, as well as create a checkpoint if 'checkpoint_path' specified).
- **snapshot_step**: `int` or None. If `int`, it will snapshot model every 'snapshot_step' steps.
- **excl_trainops**: `list` of `TrainOp`. A list of train ops to exclude from training process (TrainOps can be retrieve through `tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)`).
- **run_id**: `str`. Give a name for this run. (Useful for Tensorboard).
- **callbacks**: `Callback` or `list`. Custom callbacks to use in the training life cycle

#### 8.2. SGM

> **tflearn.models.generator.SequenceGenerator** (network, dictionary=None, seq_maxlen=25, clip_gradients=0.0, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/', checkpoint_path=None, max_checkpoints=None, session=None)

- **network**: `Tensor`. Neural network to be used.
- **dictionary**: `dict`. A dictionary associating each sample with a key ( usually integers). For example: {'a': 0, 'b': 1, 'c': 2, ...}.
- **seq_maxlen**: `int`. The maximum length of a sequence.
- **tensorboard_verbose**: `int`. Summary verbose level, it accepts different levels of tensorboard logs:

```
0 - Loss, Accuracy (Best Speed).
1 - Loss, Accuracy, Gradients.
2 - Loss, Accuracy, Gradients, Weights.
3 - Loss, Accuracy, Gradients, Weights, Activations, Sparsity.(Best visualization)
```

- **tensorboard_dir**: `str`. Directory to store tensorboard logs. Default: "/tmp/tflearn_logs/"
- **checkpoint_path**: `str`. Path to store model checkpoints. If None, no model checkpoint will be saved. Default: None.
- **max_checkpoints**: `int` or None. Maximum amount of checkpoints. If None, no limit. Default: None.
- **session**: `Session`. A session for running ops. If None, a new one will be created. Note: When providing a session, variables must have been initialized already, otherwise an error will be raised.

##### 1.1. save, load, evaluate

##### 1.2. fit

> **fit** (X_inputs, Y_targets, n_epoch=10, validation_set=None, show_metric=False, batch_size=None, shuffle=None, snapshot_epoch=True, snapshot_step=None, excl_trainops=None, run_id=None)

- **X_inputs**: array, `list` of array (if multiple inputs) or `dict` (with inputs layer name as keys). Data to feed to train model.
- **Y_targets**: array, `list` of array (if multiple inputs) or `dict` (with estimators layer name as keys). Targets (Labels) to feed to train model. Usually set as the next element of a sequence, i.e. for x[0] => y[0] = x[1].
- **n_epoch**: `int`. Number of epoch to run. Default: None.
- **validation_set**: `tuple`. Represents data used for validation. `tuple` holds data and targets (provided as same type as X_inputs and Y_targets). Additionally, it also accepts `float` (<1) to performs a data split over training data.
- **show_metric**: `bool`. Display or not accuracy at every step.
- **batch_size**: `int` or None. If `int`, overrides all network estimators 'batch_size' by this value.
- **shuffle**: `bool` or None. If `bool`, overrides all network estimators 'shuffle' by this value.
- **snapshot_epoch**: `bool`. If True, it will snapshot model at the end of every epoch. (Snapshot a model will evaluate this model on validation set, as well as create a checkpoint if 'checkpoint_path' specified).
- **snapshot_step**: `int` or None. If `int`, it will snapshot model every 'snapshot_step' steps.
- **excl_trainops**: `list` of `TrainOp`. A list of train ops to exclude from training process (TrainOps can be retrieve through `tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)`).
- **run_id**: `str`. Give a name for this run. (Useful for Tensorboard).

##### 1.3. generate

> **generate** (seq_length, temperature=0.5, seq_seed=None, display=False)

### 9.学习资源

- API Document: http://tflearn.org/getting_started/
- source code: https://github1s.com/tflearn/tflearn/blob/master/tflearn/models/dnn.py

tensorboard --logdir='C:/Users/liudongdong/OneDrive - tju.edu.cn/桌面/Alexnet/tmp'



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/tflearn/  

