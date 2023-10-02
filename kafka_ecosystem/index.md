# kafka_ecosystem


From：https://dzone.com/articles/kafka-detailed-design-and-ecosystem

> Apache Kafka 的核心要素有`中介者，订阅主题，日志，分区还有集群，还包括像 MirrorMaker 这样的有关工具`。Kafka 生态系统由` Kafka Core`，`Kafka Streams`，`Kafka Connect`，`Kafka REST Proxy `和 `Schema Registry` 组成。Kafka 生态系统的其他组件多数都来自 Confluent，它们并不属于 Apache。
>
> - Kafka Stream 是一套用于`转换，聚集并处理来自数据流的记录并生成衍生的数据流的一套 API`;
> - Kafka Connect 是一套用于`创建可复用的生产者和消费者`（例如，来自 DynamoDB 的更改数据流）的连接器的 API;
> - Kafka REST Proxy 则用于`通过 REST（HTTP）生产者和消费者`;
> - Schema Registry 则用于`管理那些使用 Avro 来记录 Kafka 数据的模式`;
> - 而 Kafka MirrorMaker 用于`将集群的数据复制到另一个集群里去`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722100727487.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722101053177.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722101209532.png)

- Kafka Connect Sources 是 Kafka 记录的来源，而 Kafka Connect Sinks 则是这一记录的目的地。

### Kafka Connect

Kafka has a [built-in framework](http://docs.confluent.io/2.0.0/connect/index.html) called Kafka Connect `for writing sources and sinks that either continuously ingest data into Kafka or continuously ingest data in Kafka into external systems.` The connectors themselves for different applications or data systems are federated and maintained separately from the main code base. You can find a list of available connectors at the [Kafka Connect Hub](http://www.confluent.io/developers/connectors).

### Distributions & Packaging

- Confluent Platform - http://confluent.io/product/. Downloads - http://confluent.io/downloads/.
- Cloudera Kafka source (0.11.0) https://github.com/cloudera/kafka/tree/cdh5-1.0.1_3.1.0 and release http://archive.cloudera.com/kafka/parcels/3.1.0/
- Hortonworks Kafka source and release http://hortonworks.com/hadoop/kafka/
- Stratio Kafka source for ubuntu http://repository.stratio.com/sds/1.1/ubuntu/13.10/binary/ and for RHEL http://repository.stratio.com/sds/1.1/RHEL/
- IBM Event Streams - https://www.ibm.com/cloud/event-streams - Apache Kafka on premise and the public cloud
- Strimzi - http://strimzi.io/ - Apache Kafka Operator for Kubernetes and Openshift. Downloads and Helm Chart - https://github.com/strimzi/strimzi-kafka-operator/releases/latest 
- TIBCO Messaging - Apache Kafka Distribution - https://www.tibco.com/products/apache-kafka Downloads - https://www.tibco.com/products/tibco-messaging/downloads

### Stream Processing

- Kafka Streams

   \- the built-in stream processing library of the Apache Kafka project

  - [Documentation in Apache Kafka](http://kafka.apache.org/documentation.html#streams)
  - [Documentation in Confluent Platform](http://docs.confluent.io/current/streams/index.html)
  - [Kafka Streams code examples in Apache Kafka](https://github.com/apache/kafka/tree/trunk/streams/examples/src/main/java/org/apache/kafka/streams/examples)
  - [Kafka Streams code examples provided by Confluent](https://github.com/confluentinc/examples/tree/master/kafka-streams)

- Kafka Streams Ecosystem:

  - Complex Event Processing (CEP): https://github.com/fhussonnois/kafkastreams-cep.

- [Storm](http://storm-project.net/) - A stream-processing framework.

- [Samza](http://samza.incubator.apache.org/) - A YARN-based stream processing framework.

- [Storm Spout](https://github.com/HolmesNL/kafka-spout) - Consume messages from Kafka and emit as Storm tuples

- [Kafka-Storm](https://github.com/miguno/kafka-storm-starter) - Kafka 0.8, Storm 0.9, Avro integration

- [SparkStreaming](https://spark.apache.org/streaming/) - Kafka receiver supports Kafka 0.8 and above

- [Flink](http://data-artisans.com/kafka-flink-a-practical-how-to/) - Apache Flink has an integration with Kafka

- [IBM Streams](https://github.com/IBMStreams/streamsx.messaging) - A stream processing framework with Kafka source and sink to consume and produce Kafka messages 

- [Spring Cloud Stream](http://cloud.spring.io/spring-cloud-stream/) - a framework for building event-driven microservices, [Spring Cloud Data Flow](http://cloud.spring.io/spring-cloud-dataflow/) - a cloud-native orchestration service for Spring Cloud Stream applications

- [Apache Apex](http://apex.apache.org/) - Stream processing framework with connectors for Kafka as source and sink.

### Hadoop Integration

- [Confluent HDFS Connector](http://docs.confluent.io/3.0.0/connect/connect-hdfs/docs/index.html) - A sink connector for the Kafka Connect framework for writing data from Kafka to Hadoop HDFS
- [Camus](https://github.com/linkedin/camus) - LinkedIn's Kafka=>HDFS pipeline. This one is used for all data at LinkedIn, and works great.
- [Kafka Hadoop Loader](https://github.com/michal-harish/kafka-hadoop-loader) A different take on Hadoop loading functionality from what is included in the main distribution.
- [Flume](http://flume.apache.org/) - Contains Kafka source (consumer) and sink (producer)
- [KaBoom](https://github.com/blackberry/KaBoom) - A high-performance HDFS data loader

### Database Integration

- [Confluent JDBC Connector](http://docs.confluent.io/3.0.0/connect/connect-jdbc/docs/index.html) - A source connector for the Kafka Connect framework for writing data from RDBMS (e.g. MySQL) to Kafka
- [Oracle Golden Gate Connector](https://java.net/projects/oracledi/downloads/directory/GoldenGate/Oracle GoldenGate Adapter for Kafka Connect) - Source connector that collects CDC operations via Golden Gate and writes them to Kafka

### Search and Query

- [ElasticSearch](https://github.com/reachkrishnaraj/kafka-elasticsearch-standalone-consumer) - This project, Kafka Standalone Consumer will read the messages from Kafka, processes and index them in ElasticSearch. There are also several [Kafka Connect connectors for ElasticSeach](http://www.confluent.io/developers/connectors).
- [Presto](http://prestodb.io/docs/current/connector/kafka-tutorial.html) - The Presto Kafka connector allows you to query Kafka in SQL using Presto.
- [Hive ](http://hiveka.weebly.com/)- Hive SerDe that allows querying Kafka (Avro only for now) using Hive SQL

### Management Consoles

- [Kafka Manager](https://github.com/yahoo/kafka-manager) - A tool for managing Apache Kafka.
- [kafkat](https://github.com/airbnb/kafkat) - Simplified command-line administration for Kafka brokers.
- [Kafka Web Console](https://github.com/claudemamo/kafka-web-console) - Displays information about your Kafka cluster including which nodes are up and what topics they host data for.
- [Kafka Offset Monitor](http://quantifind.github.io/KafkaOffsetMonitor/) - Displays the state of all consumers and how far behind the head of the stream they are.
- [Capillary](https://github.com/keenlabs/capillary) – Displays the state and deltas of Kafka-based [Apache Storm](https://storm.incubator.apache.org/) topologies. Supports Kafka >= 0.8. It also provides an API for fetching this information for monitoring purposes.
- [Doctor Kafka](https://github.com/pinterest/doctorkafka) - Service for cluster auto healing and workload balancing.
- [Cruise Control](https://github.com/linkedin/cruise-control) - Fully automate the dynamic workload rebalance and self-healing of a Kafka cluster.
- [Burrow](https://github.com/linkedin/Burrow) - Monitoring companion that provides consumer lag checking as a service without the need for specifying thresholds.
- [Chaperone](https://github.com/uber/chaperone) - An audit system that monitors the completeness and latency of data stream.

### AWS Integration

- [Automated AWS deployment](https://github.com/nathanmarz/kafka-deploy)
- [Kafka -> S3 Mirroring tool](https://github.com/pinterest/secor) from Pinterest.
- Alternative [Kafka->S3 Mirroring](https://github.com/razvan/kafka-s3-consumer) tool

### Logging

- syslog (1M)
  - [syslog producer](https://github.com/stealthly/go_kafka_client/tree/master/sysloghttps://github.com/elodina/syslog-kafka) : A producer that supports both raw data and protobuf with meta data for deep analytics usage. 
  - syslog-ng (https://syslog-ng.org/) is one of the most widely used open source log collection tools, capable of filtering, classifying, parsing log data and forwarding it to a wide variety of destinations. Kafka is a first-class destination in the syslog-ng tool; details on the integration can be found at https://czanik.blogs.balabit.com/2015/11/kafka-and-syslog-ng/ .
- [klogd](https://github.com/leandrosilva/klogd) - A python syslog publisher
- [klogd2](https://github.com/leandrosilva/klogd2) - A java syslog publisher
- [Tail2Kafka](https://github.com/harelba/tail2kafka) - A simple log tailing utility
- [Fluentd plugin](https://github.com/htgc/fluent-plugin-kafka/) - Integration with [Fluentd](http://fluentd.org/)
- [Remote log viewer](https://github.com/Digaku/skywatchr)
- [LogStash integration](https://github.com/araddon/loges) - Integration with [LogStash](http://logstash.net/) and [Fluentd](http://fluentd.org/)
- [Syslog Collector](https://github.com/otoolep/syslog-gollector) written in Go
- [Klogger](https://github.com/blackberry/klogger) - A simple proxy service for Kafka.
- [fuse-kafka](https://github.com/yazgoo/fuse_kafka): A file system logging agent based on Kafka
- [omkafka](http://www.rsyslog.com/doc/master/configuration/modules/omkafka.html): Another syslog integration, this one in C and uses librdkafka library
- [logkafka](https://github.com/Qihoo360/logkafka/) - Collect logs and send lines to Apache Kafka

#### Flume - Kafka plugins

- [Flume Kafka Plugin](https://github.com/ewhauser/flume-kafka-plugin) - Integration with [Flume](http://flume.apache.org/)
- [Kafka as a sink and source in Flume](https://github.com/baniuyao/flume-kafka) - Integration with [Flume](http://flume.apache.org/)

### Metrics

- [Mozilla Metrics Service](https://github.com/mozilla-metrics/bagheera) - A Kafka and Protocol Buffers based metrics and logging system
- [Ganglia Integration](https://github.com/adambarthelson/kafka-ganglia)
- [SPM for Kafka](http://sematext.com/spm/index.html)
- [Coda Hale Metric Reporter to Kafka](https://github.com/stealthly/metrics-kafka/)
- [kafka-dropwizard-reporter](https://github.com/SimpleFinance/kafka-dropwizard-reporter) - Register built-in Kafka client and stream metrics to Dropwizard Metrics

### Packing and Deployment

- [RPM packaging](https://github.com/edwardcapriolo/kafka-rpm)
- [Debian packaging](https://github.com/tomdz/kafka-deb-packaging)https://github.com/tomdz/kafka-deb-packaging
- Puppet Integration
  - https://github.com/miguno/puppet-kafka
  - https://github.com/whisklabs/puppet-kafka
- [Dropwizard packaging](https://github.com/datasift/dropwizard-extra)

### Kafka Camel Integration

- https://github.com/ipolyzos/camel-kafka
- https://github.com/BreizhBeans/camel-kafka

### Misc.

- [Kafka Websocket](https://github.com/b/kafka-websocket) - A proxy that interoperates with websockets for delivering Kafka data to browsers.
- [KafkaCat ](https://github.com/edenhill/kafkacat)- A native, command line producer and consumer.
- [Kafka Mirror](https://github.com/ewhauser/flume-kafka-plugin) - An alternative to the built-in mirroring tool
- [Ruby Demo App](https://github.com/jpignata/kafka-demo) 
- [Apache Camel Integration](https://github.com/BreizhBeans/camel-kafka)
- [Infobright integration](https://github.com/cbeav/kafkaconsumerconsumer)
- [Riemann Consumer of Metrics](https://github.com/stealthly/metrics-kafka/)
- [stormkafkamom](https://github.com/otoolep/stormkafkamon) – curses-based tool which displays state of[ Apache Storm](https://storm.incubator.apache.org/) based Kafka consumers (Kafka 0.7 only).
- [uReplicator](https://github.com/uber/uReplicator) - Provides the ability to replicate across Kafka clusters in other data centers
- [Mirus](https://github.com/salesforce/mirus) - A tool for distributed, high-volume replication between Apache Kafka clusters based on Kafka Connect

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kafka_ecosystem/  

