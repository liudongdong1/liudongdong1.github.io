# DL4J


> [Eclipse Deeplearning4j](https://deeplearning4j.konduit.ai/getting-started/cheat-sheet) is the first `commercial-grade`, `open-source`, `distributed deep-learning library` written for Java and Scala.` Integrated with Hadoop and Apache Spark`, DL4J brings AI to business environments for use on distributed GPUs and CPUs.

### 1. DataPrepare

> Deeplearning4j works with a lot of different data types, such as images, CSV, plain text, images, audio, video and, pretty much any other data type you can think of.

#### .1. RecordReader

#### .2. DataSetIterator

- **ScoreIterationListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/ScoreIterationListener.java), Javadoc) - `Logs the loss function score every N training iterations`
- **PerformanceListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/PerformanceListener.java), Javadoc) - Logs performance (examples` per sec, minibatches per sec, ETL time), and optionally score, every N training iterations.`
- **EvaluativeListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/EvaluativeListener.java), Javadoc) - Evaluates network performance on a test set every N iterations or epochs. Also has a system for callbacks, to (for example) save the evaluation results.
- **CheckpointListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CheckpointListener.java), Javadoc) - S`ave network checkpoints periodically - based on epochs, iterations or time (or some combination of all three).`
- **StatsListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui-model/src/main/java/org/deeplearning4j/ui/stats/StatsListener.java)) - Main listener for DL4J's web-based network training user interface. See [visualization page]() for more details.
- **CollectScoresIterationListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/CollectScoresIterationListener.java), Javadoc) - Similar to ScoreIterationListener, but stores scores internally in a list (for later retrieval) instead of logging scores
- **TimeIterationListener** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/listeners/TimeIterationListener.java), Javadoc) - Attempts to estimate time until training completion, based on current speed and specified total number of iterations

### 2. DatasetLoad

#### .1. DataSet

#### .2. INDArray

### 3. [Model](https://deeplearning4j.konduit.ai/model-zoo/overview)

- **AlexNet** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/AlexNet.java))
- **Darknet19** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/Darknet19.java))
- **FaceNetNN4Small2** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/FaceNetNN4Small2.java))
- **InceptionResNetV1** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/InceptionResNetV1.java))
- **LeNet** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/LeNet.java))
- **ResNet50** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/ResNet50.java))
- **SimpleCNN** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/SimpleCNN.java))
- **TextGenerationLSTM** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TextGenerationLSTM.java))
- **TinyYOLO** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TinyYOLO.java))
- **VGG16** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/VGG16.java))
- **VGG19** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/VGG19.java))

#### 1. Layers

##### .1. [Feed-Forward Layers]()

- **DenseLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/dense/DenseLayer.java)) - A simple/standard `fully-connected layer`
- **EmbeddingLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/embedding/EmbeddingLayer.java)) - Takes `positive integer indexes as input`, `outputs vectors`. Only usable as first layer in a model. Mathematically equivalent (when bias is enabled) to DenseLayer with one-hot input, but more efficient. See also: EmbeddingSequenceLayer.

##### .2. [Output Layers]()

- **OutputLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/OutputLayer.java)) - Output layer for `standard classification/regression in MLPs/CNNs.` Has a fully connected DenseLayer built in. 2d input/output (i.e.,` row vector per example`).
- **LossLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LossLayer.java)) - Output layer `without parameters - only loss function and activation function`. 2d input/output (i.e., row vector per example). Unlike Outputlayer, restricted to n In = n Out.
- **RnnOutputLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnOutputLayer.java)) - Output layer for recurrent neural networks. `3d (time series) input and output.` Has time distributed fully connected layer built in.
- **RnnLossLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/RnnLossLayer.java)) - The 'no parameter' version of RnnOutputLayer. 3d (time series) input and output.
- **CnnLossLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CnnLossLayer.java)) - Used with CNNs, where a prediction must be made at each spatial location of the output (for example: segmentation or denoising).` No parameters, 4d input/output with shape [minibatch, depth, height, width]. ``When using softmax, this is applied depthwise at each spatial location.`
- **Cnn3DLossLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Cnn3DLossLayer.java)) - used with 3D CNNs, where a preduction must be made at each spatial location (x/y/z) of the output. Layer has no parameters, 5d data in either NCDHW or NDHWC ("channels first" or "channels last") format (configurable). Supports masking. When using Softmax, this is applied along channels at each spatial location.
- **Yolo2OutputLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/objdetect/Yolo2OutputLayer.java)) - Implentation of the` YOLO 2 model for object detection in images`
- **CenterLossOutputLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/CenterLossOutputLayer.java)) - A version of OutputLayer that also` attempts to minimize the intra-class distance of examples' activations` - i.e., "If example x is in class Y, ensure that embedding(x) is close to average(embedding(y)) for all examples y in Y"

##### .3. [Convolutional Layers]()

- **ConvolutionLayer** / Convolution2D - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.java)) - Standard 2d convolutional neural network layer.` Inputs and outputs have 4 dimensions with shape [minibatch,depthIn,heightIn,widthIn] and [minibatch,depthOut,heightOut,widthOut] respectively.`
- **Convolution1DLayer** / Convolution1D - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution1DLayer.java)) - Standard 1d convolution layer
- **Convolution3DLayer** / Convolution3D - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Convolution3D.java)) - Standard 3D convolution layer. Supports both `NDHWC ("channels last")` and `NCDHW ("channels first")` activations format.
- **Deconvolution2DLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/Deconvolution2DLayer.java)) - also known as transpose or fractionally strided convolutions. Can be considered a "reversed" ConvolutionLayer; output size is generally larger than the input, whilst maintaining the spatial connection structure.
- **SeparableConvolution2DLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/SeparableConvolution2DLayer.java)) - `depthwise separable convolution layer`
- **SubsamplingLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/subsampling/SubsamplingLayer.java)) - Implements standard` 2d spatial pooling for CNNs - with max, average and p-norm pooling available.`
- **Subsampling1DLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/Subsampling1DLayer.java)) - 1D version of the subsampling layer.
- **Upsampling2D** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling2D.java)) - Upscale CNN activations by repeating the row/column values
- **Upsampling1D** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/upsampling/Upsampling1D.java)) - 1D version of the upsampling layer
- **Cropping2D** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/convolutional/Cropping2D.java)) -` Cropping layer for 2D convolutional neural networks`
- **DepthwiseConvolution2D** ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DepthwiseConvolution2D.java))- 2d depthwise convolution layer
- **ZeroPaddingLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPaddingLayer.java)) - Very simple layer that `adds the specified amount of zero padding to edges of the 4d input activations.`
- **ZeroPadding1DLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/convolution/ZeroPadding1DLayer.java)) - 1D version of ZeroPaddingLayer
- **SpaceToDepth** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SpaceToDepthLayer.java)) - This operation `takes 4D array in, and moves data from spatial dimensions (HW) to channels (C) for given blockSize`
- **SpaceToBatch** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/SpaceToBatchLayer.java)) - Transforms data from a tensor from 2 spatial dimensions into batch dimension according to the "blocks" specified

##### .4. [Recurrent Layers]()

- **LSTM** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LSTM.java)) - LSTM RNN without peephole connections. Supports CuDNN.
- **GravesLSTM** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesLSTM.java)) - LSTM RNN with peephole connections. Does *not* support CuDNN (thus for GPUs, LSTM should be used in preference).
- **GravesBidirectionalLSTM** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesBidirectionalLSTM.java)) - A bidirectional LSTM implementation with peephole connections. Equivalent to Bidirectional(ADD, GravesLSTM). Due to addition of Bidirecitonal wrapper (below), has been deprecated on master.
- **Bidirectional** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/Bidirectional.java)) - A 'wrapper' layer - converts any standard uni-directional RNN into a bidirectional RNN (doubles number of params - forward/backward nets have independent parameters). Activations from forward/backward nets may be either added, multiplied, averaged or concatenated.
- **SimpleRnn** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/SimpleRnn.java)) - A standard/'vanilla' RNN layer. Usually not effective in practice with long time series dependencies - LSTM is generally preferred.
- **LastTimeStep** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/recurrent/LastTimeStep.java)) - A 'wrapper' layer - extracts out the last time step of the (non-bidirectional) RNN layer it wraps. 3d input with shape [minibatch, size, timeSeriesLength], 2d output with shape [minibatch, size].
- EmbeddingSequenceLayer: ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/EmbeddingSequenceLayer.java)) - A version of EmbeddingLayer that expects fixed-length number (inputLength) of integers/indices per example as input, ranged from 0 to numClasses - 1. This input thus has shape [numExamples, inputLength] or shape [numExamples, 1, inputLength]. The output of this layer is 3D (sequence/time series), namely of shape [numExamples, nOut, inputLength]. Can only be used as the first layer for a network.

##### .5. [Unsupervised Layers]()

- **VariationalAutoencoder** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational/VariationalAutoencoder.java)) - A variational autoencoder implementation with MLP/dense layers for the encoder and decoder. Supports multiple different types of [reconstruction distributions](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/variational)
- **AutoEncoder** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/AutoEncoder.java)) - `Standard denoising autoencoder layer`

##### .6. [Other Layers]()

- **GlobalPoolingLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GlobalPoolingLayer.java)) - Implements both pooling over time (for RNNs/time series - input size [minibatch, size, timeSeriesLength], out [minibatch, size]) and global spatial pooling (for CNNs - input size [minibatch, depth, h, w], out [minibatch, depth]). Available pooling modes: sum, average, max and p-norm.
- **ActivationLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/ActivationLayer.java)) - Applies an activation function (only) to the input activations. Note that most DL4J layers have activation functions built in as a config option.
- **DropoutLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/DropoutLayer.java)) - Implements dropout as a separate/single layer. Note that most DL4J layers have a "built-in" dropout configuration option.
- **BatchNormalization** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/BatchNormalization.java)) - Batch normalization for 2d (feedforward), 3d (time series) or 4d (CNN) activations. For time series, parameter sharing across time; for CNNs, parameter sharing across spatial locations (but not depth).
- **LocalResponseNormalization** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocalResponseNormalization.java)) - Local response normalization layer for CNNs. Not frequently used in modern CNN architectures.
- **FrozenLayer** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/misc/FrozenLayer.java)) - Usually not used directly by users - added as part of transfer learning, to freeze a layer's parameters such that they don't change during further training.
- **LocallyConnected2D** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocallyConnected2D.java)) - a 2d locally connected layer, assumes input is 4d data in NCHW ("channels first") format.
- **LocallyConected1D** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/LocallyConnected1D.java)) - a 1d locally connected layer, assumes input is 3d data in NCW ([minibatch, size, sequenceLength]) format

##### .7. [Graph Vertices]()

- **ElementWiseVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ElementWiseVertex.java)) - Performs an element-wise operation on the inputs - add, subtract, product, average, max
- **L2NormalizeVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2NormalizeVertex.java)) - normalizes the input activations by dividing by the L2 norm for each example. i.e., out <- out / l2Norm(out)
- **L2Vertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - calculates the L2 distance between the two input arrays, for each example separately. Output is a single value, for each input value.
- **MergeVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/L2Vertex.java)) - merge the input activations along dimension 1, to make a larger output array. For CNNs, this implements merging along the depth/channels dimension
- **PreprocessorVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/PreprocessorVertex.java)) - a simple GraphVertex that contains an InputPreProcessor only
- **ReshapeVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ReshapeVertex.java)) - Performs arbitrary activation array reshaping. The preprocessors in the next section should usually be preferred.
- **ScaleVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ScaleVertex.java)) - implements simple multiplicative scaling of the inputs - i.e., out = scalar * input
- **ShiftVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/ShiftVertex.java)) - implements simple scalar element-wise addition on the inputs - i.e., out = input + scalar
- **StackVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/StackVertex.java)) - used to stack all inputs along the minibatch dimension. Analogous to MergeVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels)
- **SubsetVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/SubsetVertex.java)) - used to get a contiguous subset of the input activations along dimension 1. For example, two SubsetVertex instances could be used to split the activations from an input array into two separate activations. Essentially the opposite of MergeVertex.
- **UnstackVertex** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/UnstackVertex.java)) - similar to SubsetVertex, but along dimension 0 (minibatch) instead of dimension 1 (nOut/channels). Opposite of StackVertex

##### .8. [InputPreProcessors]()

- **CnnToFeedForwardPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToFeedForwardPreProcessor.java)) - handles the activation reshaping necessary to `transition from a CNN layer (ConvolutionLayer, SubsamplingLayer, etc) to DenseLayer/OutputLayer etc`.
- **CnnToRnnPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/CnnToRnnPreProcessor.java)) - handles reshaping necessary to transition from a (effectively, time distributed) CNN layer to a RNN layer.
- **ComposableInputPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/ComposableInputPreProcessor.java)) - simple class that allows `multiple preprocessors to be chained + used on a single layer`
- **FeedForwardToCnnPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToCnnPreProcessor.java)) - handles activation reshaping to transition `from a row vector (per example) to a CNN layer.` Note that this transition/preprocessor only makes sense if the activations are actually CNN activations, but have been 'flattened' to a row vector.
- **FeedForwardToRnnPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/FeedForwardToRnnPreProcessor.java)) - handles transition from `a (time distributed) feed-forward layer to a RNN layer`
- **RnnToCnnPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToCnnPreProcessor.java)) - handles transition from a sequence of CNN activations with shape `[minibatch, depth*height*width, timeSeriesLength]` to time-distributed `[numExamples*timeSeriesLength, numChannels, inputWidth, inputHeight]` format
- **RnnToFeedForwardPreProcessor** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor/RnnToFeedForwardPreProcessor.java)) - handles transition from time series activations (shape `[minibatch,size,timeSeriesLength]`) to time-distributed feed-forward (shape `[minibatch*tsLength,size]`) activations.

#### 2. Configurations

##### .1. Activation

> Activation functions can be defined in one of two ways: (a) By passing an [Activation](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/Activation.java) enumeration value to the configuration - for example, `.activation(Activation.TANH)` (b) By passing an [IActivation](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/IActivation.java) instance - for example, `.activation(new ActivationSigmoid())`

- **CUBE** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationCube.java)) - `f(x) = x^3`
- **ELU** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationELU.java)) - Exponential linear unit ([Reference](https://arxiv.org/abs/1511.07289))
- **HARDSIGMOID** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardSigmoid.java)) - a piecewise linear version of the standard sigmoid activation function. `f(x) = min(1, max(0, 0.2*x + 0.5))`
- **HARDTANH** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationHardTanH.java)) - a piecewise linear version of the standard tanh activation function.
- **IDENTITY** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationIdentity.java)) - a 'no op' activation function: `f(x) = x`
- **LEAKYRELU** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationLReLU.java)) - leaky rectified linear unit. `f(x) = max(0, x) + alpha * min(0, x)` with `alpha=0.01` by default.
- **RATIONALTANH** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRationalTanh.java)) - `tanh(y) ~ sgn(y) * { 1 - 1/(1+|y|+y^2+1.41645*y^4)}` which approximates `f(x) = 1.7159 * tanh(2x/3)`, but should be faster to execute. ([Reference](https://arxiv.org/abs/1508.01292))
- **RELU** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationReLU.java)) - standard rectified linear unit: `f(x) = x` if `x>0` or `f(x) = 0` otherwise
- **RRELU** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRReLU.java)) - randomized rectified linear unit. Deterministic during test time. ([Reference](https://arxiv.org/abs/1505.00853))
- **SIGMOID** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSigmoid.java)) - standard sigmoid activation function, `f(x) = 1 / (1 + exp(-x))`
- **SOFTMAX** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftmax.java)) - standard softmax activation function
- **SOFTPLUS** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftPlus.java)) - `f(x) = log(1+e^x)` - shape is similar to a smooth version of the RELU activation function
- **SOFTSIGN** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSoftSign.java)) - `f(x) = x / (1+|x|)` - somewhat similar in shape to the standard tanh activation function (faster to calculate).
- **TANH** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationTanH.java)) - standard tanh (hyperbolic tangent) activation function
- **RECTIFIEDTANH** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationRectifiedTanh.java)) - `f(x) = max(0, tanh(x))`
- **SELU** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSELU.java)) - scaled exponential linear unit - used with [self normalizing neural networks](https://arxiv.org/abs/1706.02515)
- **SWISH** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/impl/ActivationSwish.java)) - Swish activation function, `f(x) = x * sigmoid(x)` ([Reference](https://arxiv.org/abs/1710.05941))

##### .2. Weight Initialization

> - Weight initialization are usually defined using the [WeightInit](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInit.java) enumeration.
>
> - Custom weight initializations can be specified using `.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))` for example. As for master (but not 0.9.1 release) `.weightInit(new NormalDistribution(0, 1))` is also possible, which is equivalent to the previous approach.

- **DISTRIBUTION**: Sample weights from a provided distribution (specified via `dist` configuration method
- **ZERO**: Generate weights as zeros
- **ONES**: All weights are set to 1
- **SIGMOID_UNIFORM**: A version of XAVIER_UNIFORM for sigmoid activation functions. U(-r,r) with r=4*sqrt(6/(fanIn + fanOut))
- **NORMAL**: Normal/Gaussian distribution, with mean 0 and standard deviation 1/sqrt(fanIn). This is the initialization recommented in [Klambauer et al. 2017, "Self-Normalizing Neural Network"](https://arxiv.org/abs/1706.02515) paper. Equivalent to DL4J's XAVIER_FAN_IN and LECUN_NORMAL (i.e. Keras' "lecun_normal")
- **LECUN_UNIFORM**: Uniform U[-a,a] with a=3/sqrt(fanIn).
- **UNIFORM**: Uniform U[-a,a] with a=1/sqrt(fanIn). "Commonly used heuristic" as per Glorot and Bengio 2010
- **XAVIER**: As per [Glorot and Bengio 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf): Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
- **XAVIER_UNIFORM**: As per [Glorot and Bengio 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf): Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
- **XAVIER_FAN_IN**: Similar to Xavier, but 1/fanIn -> Caffe originally used this.
- **RELU**: [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852). Normal distribution with variance 2.0/nIn
- **RELU_UNIFORM**: [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852). Uniform distribution U(-s,s) with s = sqrt(6/fanIn)
- **IDENTITY**: Weights are set to an identity matrix. Note: can only be used with square weight matrices
- **VAR_SCALING_NORMAL_FAN_IN**: Gaussian distribution with mean 0, variance 1.0/(fanIn)
- **VAR_SCALING_NORMAL_FAN_OUT**: Gaussian distribution with mean 0, variance 1.0/(fanOut)
- **VAR_SCALING_NORMAL_FAN_AVG**: Gaussian distribution with mean 0, variance 1.0/((fanIn + fanOut)/2)
- **VAR_SCALING_UNIFORM_FAN_IN**: Uniform U[-a,a] with a=3.0/(fanIn)
- **VAR_SCALING_UNIFORM_FAN_OUT**: Uniform U[-a,a] with a=3.0/(fanOut)
- **VAR_SCALING_UNIFORM_FAN_AVG**: Uniform U[-a,a] with a=3.0/((fanIn + fanOut)/2)

#### 3. Updaters

- **AdaDelta** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaDelta.java)) - [Reference](https://arxiv.org/abs/1212.5701)
- **AdaGrad** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaGrad.java)) - [Reference](http://jmlr.org/papers/v12/duchi11a.html)
- **AdaMax** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/AdaMax.java)) - A variant of the Adam updater - [Reference](https://arxiv.org/abs/1412.6980)
- **Adam** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Adam.java))
- **Nadam** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Nadam.java)) - A variant of the Adam updater, using the Nesterov mementum update rule - [Reference](https://arxiv.org/abs/1609.04747)
- **Nesterovs** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Nesterovs.java)) - Nesterov momentum updater
- **NoOp** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/NoOp.java)) - A 'no operation' updater. That is, gradients are not modified at all by this updater. Mathematically equivalent to the SGD updater with a learning rate of 1.0
- **RmsProp** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/RmsProp.java)) - [Reference - slide 29](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- **Sgd** - ([Source](https://github.com/eclipse/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/config/Sgd.java)) - Standard stochastic gradient descent updater. This updater applies a learning rate only.

#### 4. Learning Rate Schedules

- **ExponentialSchedule** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/ExponentialSchedule.java)) - Implements `value(i) = initialValue * gamma^i`
- **InverseSchedule** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/InverseSchedule.java)) - Implements `value(i) = initialValue * (1 + gamma * i)^(-power)`
- **MapSchedule** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/MapSchedule.java)) - Learning rate schedule based on a user-provided map. Note that the provided map must have a value for iteration/epoch 0. Has a builder class to conveniently define a schedule.
- **PolySchedule** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/PolySchedule.java)) - Implements `value(i) = initialValue * (1 + i/maxIter)^(-power)`
- **SigmoidSchedule** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/SigmoidSchedule.java)) - Implements `value(i) = initialValue * 1.0 / (1 + exp(-gamma * (iter - stepSize)))`
- **StepSchedule** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/schedule/StepSchedule.java)) - Implements `value(i) = initialValue * gamma^( floor(iter/step) )`

SaveLoad

- `MultiLayerNetwork.save(File)` and `MultiLayerNetwork.load(File)`;
- using the [ModelSerializer](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/util/ModelSerializer.java) class;

#### 5. Example

```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(1234)
    // parameters below are copied to every layer in the network
    // for inputs like dropOut() or activation() you should do this per layer
    // only specify the parameters you need
    .updater(new AdaGrad())
    .activation(Activation.RELU)
    .dropOut(0.8)
    .l1(0.001)
    .l2(1e-4)
    .weightInit(WeightInit.XAVIER)
    .weightInit(Distribution.TruncatedNormalDistribution)
    .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
    .gradientNormalizationThreshold(1e-3)
    .list()
    // layers in the network, added sequentially
    // parameters set per-layer override the parameters above
    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .build())
    .layer(new ActivationLayer(Activation.RELU))
    .layer(new ConvolutionLayer.Builder(1,1)
            .nIn(1024)
            .nOut(2048)
            .stride(1,1)
            .convolutionMode(ConvolutionMode.Same)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.IDENTITY)
            .build())
    .layer(new GravesLSTM.Builder()
            .activation(Activation.TANH)
            .nIn(inputNum)
            .nOut(100)
            .build())
    .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX)
            .nIn(numHiddenNodes).nOut(numOutputs).build())
    .pretrain(false).backprop(true)
    .build();

MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(conf);
```

### 4. Train

```java
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/");
FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
int numExamples = Math.toIntExact(fileSplit.length());
int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
InputSplit trainData = inputSplit[0];
InputSplit testData = inputSplit[1];

boolean shuffle = false;
ImageTransform flipTransform1 = new FlipImageTransform(rng);
ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
ImageTransform warpTransform = new WarpImageTransform(rng, 42);
List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
    new Pair<>(flipTransform1,0.9),
    new Pair<>(flipTransform2,0.8),
    new Pair<>(warpTransform,0.5));

ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);
DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

// training dataset
ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(trainData, null);
DataSetIterator trainingIterator = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numLabels);

// testing dataset
ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
recordReader.initialize(testData, null);
DataSetIterator testingIterator = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numLabels);

// early stopping configuration, model saver, and trainer
EarlyStoppingModelSaver saver = new LocalFileModelSaver(System.getProperty("user.dir"));
EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
    .evaluateEveryNEpochs(1)
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
    .scoreCalculator(new DataSetLossCalculator(testingIterator, true))     //Calculate test set score
    .modelSaver(saver)
    .build();

EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, neuralNetwork, trainingIterator);

// begin training
trainer.fit();
```



### 5. Evaluate

- **Evaluation** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/Evaluation.java)) - Used for the evaluation of `multi-class classifiers `(assumes standard one-hot labels, and softmax probability distribution over N classes for predictions). Calculates a number of metrics - accuracy, precision, recall, F1, F-beta, Matthews correlation coefficient, confusion matrix. Optionally calculates top N accuracy, custom binary decision thresholds, and cost arrays (for non-binary case). Typically used for softmax + mcxent/negative-log-likelihood networks.
- **EvaluationBinary** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/EvaluationBinary.java)) - A multi-label binary version of the Evaluation class. Each network output is assumed to be a `separate/independent binary class, with probability 0 to 1 independent of all other outputs. `Typically used for sigmoid + binary cross entropy networks.
- **EvaluationCalibration** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/EvaluationCalibration.java)) - Used to evaluation the calibration of a binary or multi-class classifier. Produces reliability diagrams, residual plots, and histograms of probabilities. Export plots to HTML using [EvaluationTools](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java).exportevaluationCalibrationToHtmlFile method
- **ROC** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/ROC.java)) - Used for single output binary classifiers only - i.e., networks with nOut(1) + sigmoid, or nOut(2) + softmax. Supports 2 modes: thresholded (approximate) or exact (the default). Calculates area under ROC curve, area under precision-recall curve. Plot ROC and P-R curves to HTML using [EvaluationTools](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-core/src/main/java/org/deeplearning4j/evaluation/EvaluationTools.java)
- **ROCBinary** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/ROCBinary.java)) - a version of ROC that is used for multi-label binary networks (i.e., sigmoid + binary cross entropy), where each network output is assumed to be an independent binary variable.  
- **ROCMultiClass** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/classification/ROCMultiClass.java)) - a version of `ROC that is used for multi-class (non-binary) networks (i.e., softmax + mcxent/negative-log-likelihood networks)`. As ROC metrics are only defined for binary classification, this treats the multi-class output as a set of 'one-vs-all' binary classification problems.
- **RegressionEvaluation** - ([Source](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/evaluation/regression/RegressionEvaluation.java)) - An evaluation class used for r`egression models (including multi-output regression models). `Reports metrics such as mean-squared error `(MSE), mean-absolute error,` etc for each output/column.

```java
// returns evaluation class with accuracy, precision, recall, and other class statistics
Evaluation eval = neuralNetwork.eval(testIterator);
System.out.println(eval.accuracy());
System.out.println(eval.precision());
System.out.println(eval.recall());

// ROC for Area Under Curve on multi-class datasets (not binary classes)
ROCMultiClass roc = neuralNetwork.doEvaluation(testIterator, new ROCMultiClass());
System.out.println(roc.calculateAverageAuc());
System.out.println(roc.calculateAverageAucPR());
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/dl4jusage/  

