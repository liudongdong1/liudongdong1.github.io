# KerasRelative


### Introduce

1. Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano).Keras is compatible with: **Python 2.7-3.6**.
2. <font color=red>Keras 2.2.5</font> was the last release of Keras implementing the 2.2.* API. It was the last release to only support TensorFlow 1 (as well as Theano and CNTK).
3. The current release is <font color=red>Keras 2.3.0</font>, which makes significant API changes and add support for TensorFlow 2.0. The 2.3.0 release will be the last major release of multi-backend Keras. Multi-backend Keras is superseded by `tf.keras`.
4. pip install keras
5. [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (recommended if you plan on running Keras on GPU).
6. HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (required if you plan on saving Keras models to disk).
7. [graphviz](https://graphviz.gitlab.io/download/) and [pydot](https://github.com/erocarrera/pydot) (used by [visualization utilities](https://keras.io/visualization/) to plot model graphs).<font color=red>conda install python-graphviz</font>

### Get started

```python
from keras.models import Sequential
from keras.layers import Dense
model.add(Dense(units=64,activation='relu',input_dim=100))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True))
model.fit(x_train,y_train,epochs=5,batch_size=32)
#model.train_on_batch(x_batch,y_batch)
loss_and_metrics=model.evaluation(x_test,y_test,batch_size=128)
classes=model.predict(x_test,batch_size=128)
```

### Sequential Models (a linear stack of layers)

1. optimizer：字符串（预定义优化器名）或者优化器对象，，如 `rmsprop` 或 `adagrad`，也可以是 Optimizer 类的实例。详见：[optimizers](https://keras.io/zh/optimizers)。
2. loss：字符串（预定义损失函数名）或目标函数，模型试图最小化的目标函数，它可以是现有损失函数的字符串标识符，如`categorical_crossentropy` 或 `mse`，也可以是一个目标函数。详见：[losses](https://keras.io/zh/losses)
3. metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=[‘accuracy’]。评估标准可以是现有的标准的字符串标识符，也可以是自定义的评估标准函数。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
#构造方式一
model = Sequential([
    Dense(32, input_shape=(784,)),   # input shape it should expect.following layers can do automatic shape inference
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
#构造方式二
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
#编译介绍    这里需要进一步了解 optimizer，loss metrics 三者含义
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')
# For custom metrics
import keras.backend as K
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

```python
# For a single-input model with 2 classes (binary classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))  #numpy.random.random(size=None)Return random floats in the half-open interval [0.0, 1.0).
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32) #指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
```

```python
# For a single-input model with 10 classes (categorical classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))
# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

### Functional API		                        

-  layer instance is callable (on a tensor), and it returns a tensor
-  Input tensor(s) and output tensor(s) can then be used to define a `Model`
-  Such a model can be trained just like Keras `Sequential` models.

```python
from keras.layers import Input, Dense
from keras.models import Model
# This returns a tensor
inputs = Input(shape=(784,))
# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(64, activation='relu')(inputs)
output_2 = Dense(64, activation='relu')(output_1)
predictions = Dense(10, activation='softmax')(output_2)
# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

###### Turn an image classification model into a video classification model

```python
from keras.layers import TimeDistributed
# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))
# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

#### Multi-input and multi-output models

![](https://gitee.com/github-25970295/blogImage/raw/master/img/multi-input-multi-output-graph.png)

```python
from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
import numpy as np
np.random.seed(0)
main_input=Input(shape(100,),dtype='int32',name="main_input")
x=Embedding(output_dim=512,input_dim=10000,input_length=100)(main_input)
lstm_out=LSTM(32)(x)
auxiliary_output=Dense(1,activation='sigmoid',name='aux_output')(lstm_out)

auxiliary_input=Input(shape=(5,),name='aux_input')
x=keras.layers.concatenate([lstm_out,auxiliary_input])
x=Dense(64,activation='relu')(x)
x=Dense(64,activation='relu')(x)
x=Dense(64,activation='relu')(x)
main_output=Dense(1,activation='sigmoid',name='main_output')(x)

model=Model(inputs=[main_input,auxiliary_input],outputs=[main_output,auxiliary_output])
model.compile(optimizer='rmsprop',loss='binary_crossentropy',loss_weights=[1.,0.2])

headline_data=np.round(np.abs(np.random.rand(12,100)*100))
additional_data=np.random.randn(12,5)
headline_labels=np.random.randn(12,1)
additional_labels=np.random.randn(12,1)
model.fid([headline_data,additional_data],[headline_labels,additional_labels],epochs=50,batch_size=32)

# inputs and outputs are named (we passed them a "name" argument) alternative
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})
# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': headline_labels, 'aux_output': additional_labels},
          epochs=50, batch_size=32)

model.predict({'main_input':headline_data,'aux_input':additional_data})
```

#### The Concept of layer "node"

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))
conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)
# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)
conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

- https://github.com/keras-team/keras/tree/master/examples
- https://keras.io/#you-have-just-found-keras

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kerasrelative/  

