# kafka_实时机器学习


> 在Kafka应用程序中部署一个分析模型来进行实时预测。模式训练和模型部署可以是两个独立的过程。但是相同的步骤也可应用于数据集成和数据预处理，因为模型训练和模型推导需要呈现同样的数据集成、过滤、充实和聚合
>
> - 有`RPC的服务端模型（RPCs）`
> - `本地嵌入Kafka客户端`应用的模型

### 1. **使用模型服务器和RPC进行流处理**

> 从`应用程序`到`模型服务器`的交流经常通过`请求-响应协议（HTTP）`或者谷歌`RPC（gRPC）等RPC框架`来完成。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210720230001510.png)

```java
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.KStream;

import com.github.megachucky.kafka.streams.machinelearning.TensorflowObjectRecogniser;

// Configure Kafka Streams Application
finalString bootstrapServers = args.length > 0 ? args[0] : "localhost:9092";
final Properties streamsConfiguration = new Properties();
// Give the Streams application a unique name. The name must be unique
// in the Kafka cluster against which the application is run.
streamsConfiguration.put(StreamsConfig.APPLICATION_ID_CONFIG, "kafka-streams-tensorflow-serving-gRPC-example");
// Where to find Kafka broker(s).
streamsConfiguration.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);

// 向TensorFlow服务端呈现RPC（如果RPC失败，则报备异常情况）
KStream<String, Object> transformedMessage = imageInputLines.mapValues(value -> {
    System.out.println("Image path: " + value);
    imagePath = value;
    TensorflowObjectRecogniser recogniser = new TensorflowObjectRecogniser(server, port);
    System.out.println("Image = " + imagePath);
    InputStream jpegStream;
    try {
        jpegStream = new FileInputStream(imagePath);
        // Prediction of the TensorFlow Image Recognition model:
        List<Map.Entry<String, Double>> list = recogniser.recognise(jpegStream);
        String prediction = list.toString();
        System.out.println("Prediction: " + prediction);
        recogniser.close();
        jpegStream.close();
        return prediction;
    } catch (Exception e) {
        e.printStackTrace();
        return Collections.emptyList().toString();
    }
});
 
// Start Kafka Streams Application to process new incoming images from the Input Topic
final KafkaStreams streams = new KafkaStreams(builder.build(), streamsConfiguration);
streams.start();
```

### 2. **嵌入式模型的流处理**

> 嵌入模型可以`通过Kafka本地处理的数据流应用`，以Kafka数据流为杠杆。该模型还可以`通过KSQL（一种SQL方言）或者Java、Scala、Python、Go.等Kafka客户端应用程序接口`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-46a3ba1ed4fa72cd41bee6207a269da6_720w.jpg)

```java
//输入Kafka和TensorFlowAPI
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.integration.utils.EmbeddedKafkaCluster;
import org.apache.kafka.streams.integration.utils.IntegrationTestUtils;
import org.apache.kafka.streams.kstream.KStream;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

//从数据存储（例如亚马逊S3链接）或者数据记忆（例如接受一个Kafkatopic级别参数）中加载TensorFlow模型
// Step 1: Load Keras TensorFlow Model using DeepLearning4J API
String simpleMlp = new ClassPathResource("generatedModels/Keras/simple_mlp.h5").getFile().getPath();
System.out.println(simpleMlp.toString());
MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);

// Configure Kafka Streams Application
Properties streamsConfiguration = new Properties();
streamsConfiguration.put(StreamsConfig.APPLICATION_ID_CONFIG, "kafka-streams-tensorflow-keras-integration-test");
streamsConfiguration.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, CLUSTER.bootstrapServers());
// Specify default (de)serializers for record keys and for record values
streamsConfiguration.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass().getName());
streamsConfiguration.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass().getName());

// 在数据流中应用TensorFlow模型
final KStream<String, String> inputEvents = builder.stream(inputTopic);
inputEvents.foreach((key, value) -> {
// Transform input values (list of Strings) to expected DL4J parameters (two Integer values):
String[] valuesAsArray = value.split(",");
INDArray input = Nd4j.create(Integer.parseInt(valuesAsArray[0]), Integer.parseInt(valuesAsArray[1]));
// Model inference in real time:
output = model.output(input);
prediction = output.toString();
});

// 启动kafka应用
final KafkaStreams streams = new TestKafkaStreams(builder.build(), streamsConfiguration);
streams.cleanUp();
streams.start();
```

### 3. **Kubernetes(K8s)云原生模型部署**

> 在云原生框架中，两种方法都可以获得好处。即使其他的云原生技术有相似的特征，下面仍用Kubernetes作为云原生环境。`将模型嵌入到Kafka应用中`，可以`获得独立pod数据结构的所有优势`。独立的pod数据结构是流式数据处理和模型推导的容器，不依赖外部的模型服务器。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-96591db76d443c192e445a5772b85a0a_720w.jpg)

> `边车设计模式`。Kubernetes支持将具有特定任务的其他容器添加到Pod中。在以下示例中，将Kafka Streams应用程序部署在一个容器中，而模型服务器作为边车部署在同一pod内的另一个容器中。

![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/v2-86630da7c459e79bdb504641c9a31e5a_720w.jpg)

### 4. **边缘模型部署**

- 边缘数据中心或者边缘设备/机器
- 边缘有一个Kafka应用集群，一个中介和一个Kafka应用客户端。
- 一个强大的客户端（比如KSQL或者Java）或者一个轻量级的客户端（比如C或者JavaScript）
- 一个嵌入模型或者RPC模型推导
- 本地或者远程训练
- 对法律和法规的影响

> 开源云基础架构软件堆栈StarlingX之类的框架实现的，该框架需要完整的OpenStack和Kubernetes集群以及对象存储。对于其他对象来说，“边缘”意味着移动设备、轻量级板或传感器，可以在其中部署非常小的轻量级C应用程序和模型的移动设备。
>
> 从Kafka的角度来看，有很多选择。可以使用librdkafka（本机Kafka C / C ++客户端库）完全构建轻量级的边缘应用程序，该库由Confluent完全支持。还可以使用JavaScript并利用REST代理或WebSocket集成进行 Kafka通信，将模型嵌入移动应用程序中。

### Resource

- https://zhuanlan.zhihu.com/p/105159188

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kafka_%E5%AE%9E%E6%97%B6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/  

