# ExamplesDL4J


- [dl4j-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/README.md) This project contains a set of examples that demonstrate use of the` high level DL4J API to build a variety of neural networks.` Some of these examples are end to end, in the sense they start with raw data, process it and then build and train neural networks on it.
- [tensorflow-keras-import-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/tensorflow-keras-import-examples/README.md) This project contains a set of examples that demonstrate how to `import Keras h5 models and TensorFlow frozen pb models into the DL4J ecosystem.` Once imported into DL4J these models can be treated like any other DL4J model - meaning you can continue to run training on them or modify them with the transfer learning API or simply run inference on them.
- [dl4j-distributed-training-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-distributed-training-examples/README.md) This project contains a set of examples that demonstrate how to do `distributed training, inference and evaluation in DL4J on Apache Spark`. DL4J distributed training employs a "hybrid" asynchronous SGD approach - further details can be found in the distributed deep learning documentation [here](https://deeplearning4j.konduit.ai/distributed-deep-learning/intro)
- [cuda-specific-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/cuda-specific-examples/README.md) This project contains a set of examples that demonstrate how to leverage `multiple GPUs for data-parallel training of neural networks for increased performance`.
- [samediff-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/samediff-examples/README.md) This project contains a set of examples that demonstrate the SameDiff API. SameDiff (which is part of the ND4J library) can be used to `build lower level auto-differentiating computation graphs`. An analogue to the SameDiff API vs the DL4J API is the low level TensorFlow API vs the higher level of abstraction Keras API.
- [data-pipeline-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/data-pipeline-examples/README.md) This project contains a set of examples that demonstrate `how raw data in various formats can be loaded, split and preprocessed to build serializable (and hence reproducible) ETL pipelines.`
- [nd4j-ndarray-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/nd4j-ndarray-examples/README.md) This project contains a set of examples that demonstrate how to manipulate `NDArrays. `The functionality of ND4J demonstrated here can be likened to NumPy.
- [arbiter-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/arbiter-examples/README.md) This project contains a set of examples that demonstrate usage of the` Arbiter library for hyperparameter tuning of Deeplearning4J neural networks.`
- [rl4j-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/rl4j-examples/README.md) This project contains examples of using RL4J, the` reinforcement learning library` in DL4J.
- [android-examples](https://github.com/eclipse/deeplearning4j-examples/blob/master/android-examples/README.md) This project contains an Android example project, that shows` DL4J being used in an Android application.`

### 1. N4Dj

> ND4J is a` scientific computing library for the JVM`. It is meant to be used in production environments rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements.
>
> - A versatile n-dimensional array object.
> - Linear algebra and signal processing functions.
> - Multiplatform functionality including GPUs.
>   - all major operating systems: win/linux/osx/android.
>   - architectures: x86, arm, ppc.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210720191735969.png)

### 2. Importing TensorFlow models

> Currently SameDiff supports the import of TensorFlow frozen graphs through the various SameDiff.importFrozenTF methods. TensorFlow documentation on frozen models can be found [here](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).

```java
import org.nd4j.autodiff.SameDiff.SameDiff;
SameDiff sd = SameDiff.importFrozenTF(modelFile);

sd.summary();
List<String> inputs = sd.inputs();
INDArray out = sd.batchOutput()
    .input(inputName, inputArray)
    .output(outputs)
    .execSingle();
```

### 3. Importing Keras models

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

model.save('simple_mlp.h5')
```

```java
String simpleMlp = new ClassPathResource("simple_mlp.h5").getFile().getPath();
MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);

INDArray input = Nd4j.create(DataType.FLOAT, 256, 100);
INDArray output = model.output(input);

model.fit(input, output);
```

- more model examples: https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/test/java/org/deeplearning4j/nn/modelimport/keras/e2e/KerasModelEndToEndTest.java

### 4. ND4J&Samediff ops

> All operations in ND4J and SameDiff are available in "Operation Namespaces". Each namespace is available on the `Nd4j` and `SameDiff` classes with its lowercase name. 

### 5. Spark

> Deeplearning4j supports neural network training on a cluster of CPU or GPU machines `using Apache Spark`. Deeplearning4j also supports distributed evaluation as well as distributed inference using Spark.
>
> - `Gradient sharing`, available as of 1.0.0-beta: Based on [this](http://nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf) paper by Nikko Strom, is an asynchronous SGD implementation with quantized and compressed updates implemented in Spark+Aeron
> - `Parameter averaging`: A synchronous SGD implementation with a single parameter server implemented entirely in Spark.

- **TrainingMaster**: Specifies how distributed training will be conducted in practice. Implementations include Gradient Sharing (SharedTrainingMaster) or Parameter Averaging (ParameterAveragingTrainingMaster)
- **SparkDl4jMultiLayer and SparkComputationGraph**: These are wrappers around the MultiLayerNetwork and ComputationGraph classes in DL4J that enable the functionality related to distributed training. For training, they are configured with a TrainingMaster.
- **`RDD<DataSet>`** **and** **`RDD<MultiDataSet>`**: A Spark RDD with DL4J's DataSet or MultiDataSet classes define the source of the training data (or evaluation data). Note that the recommended best practice is to preprocess your data once, and save it to network storage such as HDFS. Refer to the [Deeplearning4j on Spark: How To Build Data Pipelines]() section for more details.

### Resource

- [x] https://deeplearning4j.konduit.ai/distributed-deep-learning/howto
- [ ] To learn to use spark, and embed N4dj
- [x] a solution for java using keras models, instead of using sockets or some else to communicate between different applicaiton.

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/examplesdl4j/  

