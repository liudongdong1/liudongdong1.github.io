# kafka_Stream


From: http://www.jasongj.com/kafka/kafka_stream/

# 一、Kafka Stream背景

## 1. Kafka Stream是什么

Kafka Stream是Apache Kafka从0.10版本引入的一个新Feature。它是`提供了对存储于Kafka内的数据进行流式处理和分析的功能`。

Kafka Stream的特点如下：

- Kafka Stream提供了一个非常简单而轻量的Library，它可以非常`方便地嵌入任意Java应用中，也可以任意方式打包和部署`
- `除了Kafka外，无任何外部依赖`
- 充分`利用Kafka分区机制实现水平扩展和顺序性保证`
- 通过可容错的state store实现高效的状态操作（如windowed join和aggregation）
- 支持`正好一次处理语义`
- 提供`记录级的处理能力，从而实现毫秒级的低延迟`
- 支持`基于事件时间的窗口操作，并且可处理晚到的数据（late arrival of records）`
- 同时`提供底层的处理原语Processor（类似于Storm的spout和bolt）`，以及`高层抽象的DSL（类似于Spark的map/group/reduce）`

## 2. 什么是流式计算

一般流式计算会与批量计算相比较。在流式计算模型中，输入是持续的，可以认为在时间上是无界的，也就意味着，永远拿不到全量数据去做计算。同时，计算结果是持续输出的，也即计算结果在时间上也是无界的。流式计算一般对实时性要求较高，同时一般是先定义目标计算，然后数据到来之后将计算逻辑应用于数据。同时为了提高计算效率，往往尽可能采用增量计算代替全量计算。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823011917613-280268935.png)

批量处理模型中，一般先有全量数据集，然后定义计算逻辑，并将计算应用于全量数据。特点是全量计算，并且计算结果一次性全量输出。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823011948353-603485187.png)

## 3. 为什么要有Kafka Stream

当前已经有非常多的流式处理系统，最知名且应用最多的开源流式处理系统有`Spark Streaming`和`Apache Storm`。Apache Storm发展多年，应用广泛，提供记录级别的处理能力，当前也支持SQL on Stream。而Spark Streaming基于Apache Spark，可以非常方便与图计算，SQL处理等集成，功能强大，对于熟悉其它Spark应用开发的用户而言使用门槛低。另外，目前主流的Hadoop发行版，如MapR，Cloudera和Hortonworks，都集成了Apache Storm和Apache Spark，使得部署更容易。

既然Apache Spark与Apache Storm拥用如此多的优势，那为何还需要Kafka Stream呢？笔者认为主要有如下原因。

第一，`Spark和Storm都是流式处理框架，而Kafka Stream提供的是一个基于Kafka的流式处理类库`**。框架要求开发者按照特定的方式去开发逻辑部分，供框架调用。开发者很难了解框架的具体运行方式，从而使得调试成本高，并且使用受限。而Kafka Stream作为流式处理**类库**，直接提供具体的类给开发者调用，`整个应用的运行方式主要由开发者控制，方便使用和调试。`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012034870-605607318.png)

第二，虽然Cloudera与Hortonworks方便了Storm和Spark的部署，但是这些框架的`部署`仍然相对复杂。而Kafka Stream作为类库，可以非常方便的嵌入应用程序中，它对应用的打包和部署基本没有任何要求。更为重要的是，Kafka Stream充分利用了[Kafka的分区机制](http://www.jasongj.com/2015/03/10/KafkaColumn1/#Topic-amp-Partition)和[Consumer的Rebalance机制](http://www.jasongj.com/2015/08/09/KafkaColumn4/#High-Level-Consumer-Rebalance)，使得Kafka Stream可以`非常方便的水平扩展`，`并且各个实例可以使用不同的部署方式`。具体来说，每个运行Kafka Stream的应用程序实例都包含了Kafka Consumer实例，多个同一应用的实例之间并行处理数据集。而不同实例之间的部署方式并不要求一致，比如部分实例可以运行在Web容器中，部分实例可运行在Docker或Kubernetes中。

第三，`就流式处理系统而言，基本都支持Kafka作为数据源`。例如Storm具有专门的kafka-spout，而Spark也提供专门的spark-streaming-kafka模块。事实上，Kafka基本上是主流的流式处理系统的标准数据源。换言之，大部分流式系统中都已部署了Kafka，此时`使用Kafka Stream的成本非常低`。

第四，使用Storm或Spark Streaming时，需要为框架本身的进程预留资源，如Storm的supervisor和Spark on YARN的node manager。即使对于应用实例而言，`框架本身也会占用部分资源`，如Spark Streaming需要为shuffle和storage预留内存。

第五，由于`Kafka本身提供数据持久化`，因此Kafka Stream`提供滚动部署和滚动升级以及重新计算的能力。`

第六，由于Kafka Consumer Rebalance机制，`Kafka Stream可以在线动态调整并行度。`

# 二、Kafka Stream架构

## 1. Kafka Stream整体架构

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012221127-657291327.png)

```java
KStream<String, String> stream = builder.stream("words-stream");
KTable<String, String> table = builder.table("words-table", "words-store");
```

另外，上图中的Consumer和Producer并不需要开发者在应用中显示实例化，而是`由Kafka Stream根据参数隐式实例化和管理`，从而降低了使用门槛。`开发者只需要专注于开发核心业务逻辑，也即上图中Task内的部分`。

## 2. Processor Topology

基于Kafka Stream的流式应用的业务逻辑`全部通过一个被称为Processor Topology的地方执行`。它与Storm的Topology和Spark的DAG类似，都定义了数据在各个处理单元（在Kafka Stream中被称作Processor）间的流动方式，或者说定义了数据的处理逻辑。

```java
public class WordCountProcessor implements Processor<String, String> {
    private ProcessorContext context;
    private KeyValueStore<String, Integer> kvStore;
    @SuppressWarnings("unchecked")
    @Override
    //init()方法：可以获取ProcessorContext实例，用来维护当前上下文；通过上下文ProcessorContext得到状态仓库实例以及使用上下文用于基于时间推移周期性的执行；
    public void init(ProcessorContext context) {
        this.context = context;
        this.context.schedule(1000);
        this.kvStore = (KeyValueStore<String, Integer>) context.getStateStore("Counts");
    }
    //process方法：用于对收到的数据集执行对状态仓库的操作；
    @Override
    public void process(String key, String value) {
        Stream.of(value.toLowerCase().split(" ")).forEach((String word) -> {
            Optional<Integer> counts = Optional.ofNullable(kvStore.get(word));
            int count = counts.map(wordcount -> wordcount + 1).orElse(1);
            kvStore.put(word, count);
        });
    }
    @Override
    public void punctuate(long timestamp) {   //punctuate():用于基于时间推移周期性执行；
        KeyValueIterator<String, Integer> iterator = this.kvStore.all();
        iterator.forEachRemaining(entry -> {
            context.forward(entry.key, entry.value);
            this.kvStore.delete(entry.key);
        });
        context.commit();
    }
    @Override   //close:关闭相应资源操作
    public void close() {
        this.kvStore.close();
    }
}
```

- `process定义了对每条记录的处理逻辑`，也印证了Kafka可具有记录级的数据处理能力。
- context.scheduler定义了`punctuate被执行的周期`，从而提供了实现窗口操作的能力。
- context.getStateStore提供的`状态存储为有状态计算（如窗口，聚合）提供了可能。`

## 3. Kafka Stream并行模型

Kafka Stream的并行模型中，`最小粒度为Task`，而`每个Task包含一个特定子Topology的所有Processor`。因此每个Task所执行的代码完全一样，唯一的不同在于所处理的数据集互补。这一点跟Storm的Topology完全不一样。Storm的Topology的每一个Task只包含一个Spout或Bolt的实例。因此Storm的一个Topology内的不同Task之间需要通过网络通信传递数据，而Kafka Stream的Task包含了完整的子Topology，所以`Task之间不需要传递数据，也就不需要网络通信`。这一点降低了系统复杂度，也提高了处理效率。

如果某个Stream的输入Topic有多个(比如2个Topic，1个Partition数为4，另一个Partition数为3)，则总的Task数等于Partition数最多的那个Topic的Partition数（max(4,3)=4）。这是因为Kafka Stream使用了Consumer的Rebalance机制，每个Partition对应一个Task。

下图展示了在一个进程（Instance）中`以2个Topic（Partition数均为4）为数据源的Kafka Stream应用的并行模型。`从图中可以看到，由于Kafka Stream应用的默认线程数为1，所以4个Task全部在一个线程中运行。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012423201-1891805178.png)

为了充分利用多线程的优势，可以设置Kafka Stream的线程数。下图展示了线程数为2时的并行模型。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012446423-1214824884.png)

前文有提到，Kafka Stream可被嵌入任意Java应用（理论上基于JVM的应用都可以）中，下图展示了在`同一台机器的不同进程中同时启动同一Kafka Stream应用时的并行模型`。注意，这里要保证两个进程的StreamsConfig.`APPLICATION_ID_CONFIG完全一样`。因为`Kafka Stream将APPLICATION_ID_CONFIG作为隐式启动的Consumer的Group ID`。只有保证APPLICATION_ID_CONFIG相同，才能保证这两个进程的Consumer属于同一个Group，从而可以通过Consumer Rebalance机制拿到互补的数据集。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012616004-1473262041.png)

既然实现了多进程部署，可以以同样的方式实现多机器部署。该部署方式也要求所有进程的APPLICATION_ID_CONFIG完全一样。从图上也可以看到，每个实例中的线程数并不要求一样。但是无论如何部署，Task总数总会保证一致。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012703011-1321655264.png)

这里对比一下Kafka Stream的Processor Topology与Storm的Topology。

- `Storm的Topology由Spout和Bolt组成，Spout提供数据源，而Bolt提供计算和数据导出`。Kafka Stream的Processor Topology完全`由Processor组成，因为它的数据固定由Kafka的Topic提供`。
- Storm的不同Bolt运行在不同的Executor中，很可能位于不同的机器，需要通过网络通信传输数据。而Kafka Stream的Processor Topology的`不同Processor完全运行于同一个Task中`，也就完全处于同一个线程，无需网络通信。
- Storm的Topology可以同时包含Shuffle部分和非Shuffle部分，并且往往一个Topology就是一个完整的应用。而Kafka Stream的一个物理Topology只包含非Shuffle部分，而Shuffle部分需要通过through操作显示完成，该操作将一个大的Topology分成了2个子Topology。
- Storm的Topology内，不同Bolt/Spout的并行度可以不一样，而Kafka Stream的子Topology内，所有Processor的并行度完全一样。
- Storm的一个Task只包含一个Spout或者Bolt的实例，而`Kafka Stream的一个Task包含了一个子Topology的所有Processor。`

## 4. KTable vs. KStream

`KTable和KStream`是Kafka Stream中非常重要的两个概念，它们是Kafka实现各种语义的基础。因此这里有必要分析下二者的区别。

- `KStream是一个数据流`，可以认为`所有记录都通过Insert only的方式插入进这个数据流里`。

- `KTable代表一个完整的数据集`，可以`理解为数据库中的表`。由于每条记录都是Key-Value对，这里可以将Key理解为数据库中的Primary Key，而Value可以理解为一行记录。可以认为KTable中的数据都是`通过Update only的方式进入的`。也就意味着，`如果KTable对应的Topic中新进入的数据的Key已经存在，那么从KTable只会取出同一Key对应的最后一条数据，相当于新的数据更新了旧的数据。`

以下图为例，假设有一个KStream和KTable，基于同一个Topic创建，并且该Topic中包含如下图所示5条数据。此时遍历KStream将得到与Topic内数据完全一样的所有5条数据，且顺序不变。而此时遍历KTable时，因为这5条记录中有3个不同的Key，所以将得到3条记录，每个Key对应最新的值，并且这三条数据之间的顺序与原来在Topic中的顺序保持一致。这一点与Kafka的日志compact相同。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/963903-20180823012822162-142241598.png)

此时如果对该KStream和KTable分别基于key做Group，对Value进行Sum，得到的结果将会不同。对KStream的计算结果是<Jack，4>，<Lily，7>，<Mike，4>。而对Ktable的计算结果是<Mike，4>，<Jack，3>，<Lily，5>。

## 5. State store

流式处理中，部分操作是无状态的，例如过滤操作（Kafka Stream DSL中用`filer`方法实现）。而部分操作是有状态的，需要记录中间状态，如Window操作和聚合计算。`State store被用来存储中间状态。它可以是一个持久化的Key-Value存储，也可以是内存中的HashMap，或者是数据库。Kafka提供了基于Topic的状态存储。`

# 三、Kafka Stream如何解决流式系统中关键问题

## 1. 时间

在流式数据处理中，`时间是数据的一个非常重要的属性`。从`Kafka 0.10开始`，每条记录除了Key和Value外，还增加了timestamp属性。目前Kafka Stream支持三种时间

- **事件发生时间**。事件发生的时间，包含在数据记录中。发生时间`由Producer在构造ProducerRecord时指定`。并且`需要Broker或者Topic将message.timestamp.type设置为CreateTime（默认值）才能生效。`
- **消息接收时间**。也即`消息存入Broker的时间`。当`Broker或Topic将message.timestamp.type设置为LogAppendTime时生效`。此时Broker会在`接收到消息后，存入磁盘前`，将其timestamp属性值设置为当前机器时间。一般消息接收时间比较接近于事件发生时间，部分场景下可代替事件发生时间。
- **消息处理时间**。也即Kafka Stream`处理消息时的时间`。

> 注：Kafka Stream允许通过实现org.apache.kafka.streams.processor.TimestampExtractor接口自定义记录时间。

## 2. 窗口

流式数据是在时间上无界的数据。而聚合操作只能作用在特定的数据集，也即有界的数据集上。因此需要通过某种方式从无界的数据集上按特定的语义选取出有界的数据。窗口是一种非常常用的设定计算边界的方式。不同的流式处理系统支持的窗口类似，但不尽相同。

Kafka Stream支持的窗口如下。

（1）**Hopping Time Window** 该窗口定义如下图所示。它有两个属性，一个是`Window size，一个是Advance interval`。`Window size指定了窗口的大小`，也即每次计算的数据集的大小。而`Advance interval定义输出的时间间隔`。一个典型的应用场景是，每隔5秒钟输出一次过去1个小时内网站的PV或者UV。

 

![img](https://images2018.cnblogs.com/blog/963903/201808/963903-20180823013127357-1860834711.gif)

（2）**Tumbling Time Window**该窗口定义如下图所示。可以认为它是Hopping Time Window的一种特例，也即**Window size和Advance interval相等**。它的特点是各个Window之间完全不相交。

![img](https://images2018.cnblogs.com/blog/963903/201808/963903-20180823013225516-1830552448.gif)

（3）**Sliding Window** 该窗口`只用于2个KStream进行Join计算时`。该窗口的大小定义`了Join两侧KStream的数据记录被认为在同一个窗口的最大时间差`。假设该窗口的大小为5秒，则参与Join的2个KStream中，记录时间差小于5的记录被认为在同一个窗口中，可以进行Join计算。

（4）**Session Window**该窗口用于对Key做Group后的聚合操作中。它需要对Key做分组，然后对组内的数据根据业务需求定义一个窗口的起始点和结束点。一个典型的案例是，希望通过Session Window计算某个用户访问网站的时间。对于一个特定的用户（用Key表示）而言，当发生登录操作时，该用户（Key）的窗口即开始，当发生退出操作或者超时时，该用户（Key）的窗口即结束。窗口结束时，可计算该用户的访问时间或者点击次数等。

## 3. Join

Kafka Stream由于包含`KStream`和`Ktable`两种数据集，因此提供如下Join计算

- `KTable Join KTable 结果仍为KTable`。任意一边有更新，结果KTable都会更新。
- `KStream Join KStream 结果为KStream`。必须带窗口操作，否则会造成Join操作一直不结束。
- `KStream Join KTable / GlobalKTable 结果为KStream`。只有当KStream中有新数据时，才会触发Join计算并输出结果。KStream无新数据时，KTable的更新并不会触发Join计算，也不会输出数据。并且该更新只对下次Join生效。一个典型的使用场景是，KStream中的订单信息与KTable中的用户信息做关联计算。

`对于Join操作，如果要得到正确的计算结果，需要保证参与Join的KTable或KStream中Key相同的数据被分配到同一个Task`。具体方法是

- 参与Join的KTable或KStream的`Key类型相同`（实际上，业务含意也应该相同）
- 参与Join的KTable或KStream`对应的Topic的Partition数相同`
- Partitioner策略的最终结果等效（实现不需要完全一样，只要效果一样即可），也即Key相同的情况下，被分配到ID相同的Partition内

如果上述条件不满足，可通过调用如下方法使得它满足上述条件。

```java
KStream<K, V> through(Serde<K> keySerde, Serde<V> valSerde, StreamPartitioner<K, V> partitioner, String topic)
```

## 4. 聚合与乱序处理

聚合操作可应用于KStream和KTable。当`聚合发生在KStream上时必须指定窗口，从而限定计算的目标数据集。`

需要说明的是，聚合操作的`结果肯定是KTable`。因为KTable是可更新的，可以在晚到的数据到来时（也即发生数据乱序时）更新结果KTable。

但需要说明的是，`Kafka Stream并不会对所有晚到的数据都重新计算并更新结果集`，而是让用户设置一个`retention period`，将每个窗口的结果集在内存中保留一定时间，该窗口内的数据晚到时，直接合并计算，并更新结果KTable。超过`retention period`后，该窗口结果将从内存中删除，并且晚到的数据即使落入窗口，也会被直接丢弃。

## 5. 容错

Kafka Stream从如下几个方面进行容错

- `高可用的Partition保证无数据丢失`。每个Task计算一个Partition，而Kafka数据复制机制保证了Partition内数据的高可用性，故无数据丢失风险。同时由于数据是持久化的，即使任务失败，依然可以重新计算。
- `状态存储实现快速故障恢复和从故障点继续处理`。对于Join和聚合及窗口等有状态计算，状态存储可保存中间状态。即使发生Failover或Consumer Rebalance，仍然可以通过状态存储恢复中间状态，从而可以继续从Failover或Consumer Rebalance前的点继续计算。
- `KTable与retention period提供了对乱序数据的处理能力`。

# 四、Kafka Stream应用示例

- **创建KTable和KStream**

```java
StreamsBuilder builder = new StreamsBuilder();
//StreamsBuilder.table(final String topic)创建KTable实例的同时，内部会创建一个StateStore来跟踪流的状态，但它不可用于交互式查询。
// 创建KTable实例
KTable<String, StockTickerData> stockTickerTable = builder.table("stock-ticker-table");
// 创建KStream实例
KStream<String, StockTickerData> stockTickerStream = builder.stream("stock-ticker-stream");
// 打印结果到控制台
stockTickerTable.toStream().print(Printed.<String, StockTickerData>toSysOut().withLabel("Stocks-KTable"));
stockTickerStream.print(Printed.<String, StockTickerData>toSysOut().withLabel("Stocks-KStream"));
```

- **属性配置**

```java
//application id相当于group id，bootstrap servers配置kafka的brokers地址，并配置key与value的序列化、反序列化实现类。
Properties props = new Properties();
props.put(StreamsConfig.APPLICATION_ID_CONFIG, "streams-pipe");
props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
```

- **读取并处理输出**

```java
//最后通过StreamsBuilder来创建KStream，进行数据处理转换后输出到一个新的topic或者其他外部存储器中。
builder.stream("streams-plaintext-input").to("streams-pipe-output");
final Topology topology = builder.build();
final KafkaStreams streams = new KafkaStreams(topology, props);
```

- **推出时处理逻辑**

```java
// attach shutdown handler to catch control-c
Runtime.getRuntime().addShutdownHook(new Thread("streams-shutdown-hook") { @Override public void run() { streams.close(); latch.countDown(); }
});
```

```java
package cc.gmem.study.kafka;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.KTable;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class NameCountApplication {

    private static final Logger LOGGER = LogManager.getLogger( NameCountApplication.class );

    public static void main( final String[] args ) throws Exception {
        Properties config = new Properties();
        // 应用的标识符，不同的实例依据此标识符相互发现
        config.put( StreamsConfig.APPLICATION_ID_CONFIG, "names-counter-application" );
        // 启动时使用的Kafka服务器
        config.put( StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka-1.gmem.cc:9092" );
        // 键值串行化类
        config.put( StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass() );
        config.put( StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass() );
        // High Level DSL for building topologies
        StreamsBuilder builder = new StreamsBuilder();
        // KStream是记录（KeyValue）流的抽象
        KStream<String, String> nameBatches = builder.stream( "names" );
        KTable<String, Long> nameCounts = nameBatches
            // 一到多映射，分割字符串
            .flatMapValues( nameBatch -> Arrays.asList( nameBatch.split( "\\W+" ) ) )
            // 根据人名分组
            .groupBy( ( key, name ) -> name )
            // 进行聚合，结果存放到StateStore中
            .count( Materialized.as( "names-count-store" ) );
        // 输出到目标
        nameCounts.toStream().to( "names-count", Produced.with( Serdes.String(), Serdes.Long() ) );
        // 构建流处理程序并启动
        KafkaStreams streams = new KafkaStreams( builder.build(), config );
        LOGGER.trace( "Prepare to start stream processing." );
        streams.start();

        TimeUnit.DAYS.sleep( 1 );  // 阻塞主线程
    }
}
```

# 五、API

## 1. Topology

```java
Topology topology = new Topology();
// 指定拓扑的输入，也就是Kafka主题
topology.addSource( "SOURCE", "src-topic" )
        // 添加一个处理器PROCESS1，其上游为拓扑输入（通过名称引用）
        .addProcessor( "PROCESS1", () -> new Processor1(), "SOURCE" )
        // 添加另一个处理器PROCESS2，以PROCESS1为上游
        .addProcessor( "PROCESS2", () -> new Processor2(), "PROCESS1" )
        // 添加另一个处理器PROCESS3，仍然以PROCESS1为上游，注意拓扑分叉了
        .addProcessor( "PROCESS3", () -> new Processor3(), "PROCESS1" )
        // 添加一个输出处理器，输出到sink-topic1，以PROCESS1为上游
        .addSink( "SINK1", "sink-topic1", "PROCESS1" )
        // 添加一个输出处理器，输出到sink-topic2，以PROCESS2为上游
        .addSink( "SINK2", "sink-topic2", "PROCESS2" )
        // 添加一个输出处理器，输出到sink-topic3，以PROCESS3为上游
        .addSink( "SINK3", "sink-topic3", "PROCESS3" );
```

## 2.Processor

> 该接口用于定义一个流处理器，也就是`处理器拓扑中的节点`。流处理器以参数化类型的方式限定了`其键、值的类型`。你可以定义任意数量的流处理器，并且连同它们关联的状态存储一起，组装出拓扑。Processor.process()方法针对收到的每一个记录进行处理。`Processor.init()方法实例化了一个ProcessorContext`，流处理器可以调用上下文：
>
> 1. `context().schedule，调度一个Punctuation函数，周期性执行`
> 2. `context().forward，转发新的或者被修改的键值对给下游处理器`
> 3. `context().commit，提交当前处理进度`

```java
package cc.gmem.study.kafka.streams.low;
 
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.processor.Processor;
import org.apache.kafka.streams.processor.ProcessorContext;
import org.apache.kafka.streams.processor.PunctuationType;
import org.apache.kafka.streams.state.KeyValueIterator;
import org.apache.kafka.streams.state.KeyValueStore;
 
public class NameCounterProcessor implements Processor<String, String> {
    private ProcessorContext context;
    private KeyValueStore<String, Long> kvStore;
    @Override
    public void init( ProcessorContext context ) {
        // 保存引用，类似于Storm的TopologyContext
        this.context = context;
        // 从上下文中取回一个状态存储
        this.kvStore = (KeyValueStore<String, Long>) context.getStateStore( "NameCounts" );
        // 以墙上时间为准，每秒执行Punctuator逻辑
        this.context.schedule( 1000, PunctuationType.WALL_CLOCK_TIME, timestamp -> {
            NameCounterProcessor.this.punctuate( timestamp );
        } );
    }
    /**
     * 接收一个记录（人名列表）并处理
     *
     * @param dummy 记录的键，无用
     * @param line  记录的值
     */
    @Override
    public void process( String dummy, String line ) {
        String[] names = line.toLowerCase().split( " " );
        // 在键值存储中更新人名计数
        for ( String name : names ) {
            Long oldCount = this.kvStore.get( name );
            if ( oldCount == null ) {
                this.kvStore.put( name, 1L );
            } else {
                this.kvStore.put( name, oldCount + 1L );
            }
        }
    }
    @Override
    public void punctuate( long timestamp ) {
        // 获得键值存储的迭代器
        KeyValueIterator<String, Long> iter = this.kvStore.all();
        while ( iter.hasNext() ) {
            KeyValue<String, Long> entry = iter.next();
            // 转发记录给下游处理器
            context.forward( entry.key, entry.value.toString() );
        }
        /**
         * 调用者必须要负责关闭状态存储上的迭代器
         * 否则可能（取决于底层状态存储的实现）导致内存、文件句柄的泄漏
         */
        iter.close();
        // 请求提交当前流状态（消费进度）
        context.commit();
    }
    @Override
    public void close() {
        // 在此关闭所有持有的资源，但是状态存储不需要关闭，由Kafka Stream自己维护其生命周期
    }
}
```

## 3. **StateStore**

#### .1. 使用状态存储

> 使用状态存储: 要位拓扑中每个Processor提供状态存储，调用：

```java
Topology topology = new Topology();
topology.addSource("Source", "source-topic")
    .addProcessor("Process", () -> new WordCountProcessor(), "Source")
    // 为处理器Process提供一个状态存储 
    .addStateStore(countStoreSupplier, "Process");
```

#### 2. changelog

> 为了支持容错、支持`无数据丢失的状态迁移`， 状态存储可以`持续不断的、在后台备份到Kafka主题中`。上述用于主题被称为状态存储的changelog主题，或者直接叫changelog。你可以启用或者禁用状态存储的备份特性。持久性的KV存储是容错的，它备份在一个紧凑格式的changelog主题中。使用紧凑格式的原因是：
>
> 1. `防止主题无限增长`
> 2. `减少主题占用的存储空间`
> 3. `当状态存储需要通过Changelog恢复时，缩短需要的时间`
>
> 持久性的窗口化存储也是容错的，它基于紧凑格式、支持删除机制的主题备份。窗口化存储的changelog的键的一部分是窗口时间戳，过期的窗口对应的段会被Kafka的日志清理器清理。changelog的默认存留时间是Windows#maintainMs() + 1天。指定StreamsConfig.WINDOW_STORE_CHANGE_LOG_ADDITIONAL_RETENTION_MS_CONFIG可以覆盖之。

#### .3. 监控状态恢复

> `启动应用程序时，状态存储通常不需要根据changelog来恢复`，`直接加载磁盘上持久化的数据`就可以。但以下场景中：
>
> 1. 宕机导致本地状态丢失
> 2. 运行在无状态环境下的应用程序重启
>
> 状态存储需要基于changelog进行完整的恢复。如果changelog中的数据量很大，则恢复过程可能相当的耗时。在恢复完成之前，处理器拓扑不能处理新的数据。
>
> 要监控状态存储的恢复进度，你需要实现org.apache.kafka.streams.processor.`StateRestoreListener接口`，并调用KafkaStreams#`setGlobalStateRestoreListener注册之`

```java
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.streams.processor.StateRestoreListener;
// 监听器会被所有org.apache.kafka.streams.processor.internals.StreamThread实例共享，并必须线程安全
public class ConsoleGlobalRestoreListerner implements StateRestoreListener {
    // 在恢复过程开始时回调
    public void onRestoreStart(
            final TopicPartition topicPartition,  // 主题分区
            final String storeName,               // 状态存储名称
            final long startingOffset,            // 需要恢复的起点
            final long endingOffset               // 需要恢复的终点
    ) {}
 
    // 在恢复一批次数据后回调
    public void onBatchRestored( final TopicPartition topicPartition,
            final String storeName,
            final long batchEndOffset,
            final long numRestored 
    ) {}
 
    // 恢复完成后回调
    public void onRestoreEnd( final TopicPartition topicPartition,
            final String storeName,
            final long totalRestored ) {}
}
```

#### 4. **启/禁changelog**

```java
// 启用：StateStoreBuilder#withLoggingEnabled(Map<String, String>);
// 禁用：StateStoreBuilder#withLoggingDisabled();
KeyValueBytesStoreSupplier countStoreSupplier = Stores.inMemoryKeyValueStore("Counts");
StateStoreBuilder builder = Stores
    .keyValueStoreBuilder(countStoreSupplier,Serdes.String(),Serdes.Long())
    .withLoggingDisabled();
```

### 4. 流表二元性

- Stream as Table：`一个流可以看做是一个表的changelog`。流中的每条记录都捕获了表的一次状态变更，通过Replay changelog，流可以转变为表。流记录和表行不一定是1:1对应关系，流记录可能经过聚合，更新到表中的一行
- Table as Stream：`表可以看做是某个瞬间的、流中每个键的最终值构成的快照。`迭代表中的键值对很容易将其转换为流

```java
//通过读取Kafka主题，即可为Kafka Streams提供输入流。首先你需要实例化一个StreamsBuilder：
StreamsBuilder builder = new StreamsBuilder();
//创建 KStream 流
KStream<String, Long> nameCounts = builder.stream( 
    "names-counts-input-topic",  // 输入主题名称
    Consumed.with(Serdes.String(), Serdes.Long()) // 指定键值的串行化器
);

//创建 KTable
KTable<String, Long> nameCounts = builder.table(
    Serdes.String(), /* 键串行化器 */
    Serdes.Long(),   /* 值串行化器 */
    "name-counts-input-topic", /* 输入主题 */
    "name-counts-partitioned-store" /* 表名 */);

//创建GlobalKTable
GlobalKTable<String, Long> nameCounts = builder.globalTable(
    Serdes.String(), /* 键串行化器 */
    Serdes.Long(),   /* 值串行化器 */
    "name-counts-input-topic", /* 输入主题 */
    "name-counts-global-store" /* 表名 */);
```

`可以把任何主题看做是changelog，并将其读入到KTable`。当：

1. 记录的键`不存在时`，相当于执行`INSERT操作`
2. 记录的键存在，值不为null时，相当于执行`UPDATE操作`
3. 记录的键存在，`值为null时`，相当于执行`DELETE操作`

KTable对应了从输入主题读取的、分区化的记录的流。流处理程序的每个实例，都会消费输入主题的分区的子集，并且在整体上保证所有分区都被消费。

### 5. 流转换操作

#### .1. 无状态转化

- 不依赖于任何状态即可完成转换，不要求流处理器有关联的StateStore。
- **branch**

> IO：KStream → KStream，`基于给定的断言集分割KStream，将其分割为1-N个KStream实例`。断言按照声明的顺序依次估算，每个记录只被转发到第一个匹配的下游流中：

```java
KStream<String, Long> stream = ...;
KStream<String, Long>[] branches = stream.branch(
        (key, value) -> key.startsWith("A"), /* 以A开头的键  */
        (key, value) -> key.startsWith("B"), /* 以B开头的键 */
        (key, value) -> true                 /* 所有其它的记录均发往此流  */
);
```

- **Filter**   **filterNot**

> IO：KStream → KStream 或 KTable → KTable; 基于给定的断言，`针对每个记录进行估算。估算结果为true的记录进入下游流：`

```java
// 仅保留正数值
stream.filter((key, value) -> value > 0);
// 针对一个KTable进行过滤，结果物化到一个StageStore中
Materialized m = Materialized.<String, Long, KeyValueStore<Bytes, byte[]>>as("filtered")
table.filter((key, value) -> value != 0, m);
```

- **flatMap**

> IO：KStream → KStream; 基于一个记录，产生0-N个输出记录：

```java
KStream<String, Integer> transformed = stream.flatMap(
    (key, value) -> {
        // 键值对的列表
        List<KeyValue<String, Integer>> result = new LinkedList<>();
        result.add(KeyValue.pair(value.toUpperCase(), 1000));
        result.add(KeyValue.pair(value.toLowerCase(), 9000));
        return result;
    }
);
```

- **foreach**

> IO：KStream → void; 终结性操作，`针对每个记录执行无状态的操作`; 需要注意：操作的副作用（例如对外部系统的写）无法被Kafka跟踪，也就是说`无法获得Kafka的处理语义保证`
>
> 示例： stream.**foreach**((key, value) -> System.out.println(key + " => " + value)); 

- **groupByKey**

>  IO：KStream → KGroupedStream; `分组是进行流/表的聚合操作的前提`。分组保证了数据被正确的分区，保证后续操作的正常进行和分组相关的一个操作是窗口化。`利用窗口化，可以将分组后的记录二次分组，形成一个个窗口，然后以窗口为单位进行聚合、Join仅当流被标记用于重新分区`，则此操作才会导致重新分区。该操作不允许修改键或者键类型

```java
KGroupedStream<byte[], String> groupedStream = stream.groupByKey(
    // 如果键值的类型不匹配配置的默认串行化器，则需要明确指定：
    Serialized.with(
         Serdes.ByteArray(),
         Serdes.String())
);
```

- **groupBy**

> IO：KStream → KGroupedStream 或 KTable → KGroupedTable; 实际上是`selectKey+groupByKey的组合`; 基于`一个新的键来分组记录，新键的类型可能和记录旧的键类型不同。`当对表进行分组时，还可以指定新的值、值类型; 该操作总是会导致数据的重新分区，因此在可能的情况下你应该优选groupByKey，后者仅在必要的时候分区.

```java
KGroupedStream<String, String> groupedStream = stream.groupBy(
    (key, value) -> value,  // 产生键值对value:value并依此分组
    Serialize.with(
         Serdes.String(), /* 键的类型发生改变 */
         Serdes.String())  /* value */
); 
KGroupedTable<String, Integer> groupedTable = table.groupBy(
    // 产生键值对  value:length(value)，并依此分组
    (key, value) -> KeyValue.pair(value, value.length()),
    Serialized.with(
        Serdes.String(), /* 键的类发生改变 */
        Serdes.Integer()) /* 值的类型发生改变  */
);
```

- **map**

> IO：KStream → KStream;  `根据一个输入记录产生一个输出记录，你可以修改键值的类型`

```java
KStream<byte[], String> stream = ...;
KStream<String, Integer> transformed = stream.map(
    (key, value) -> KeyValue.pair(value.toLowerCase(), value.length()));
```

- **mapValues**: 类似上面，但是仅仅映射值，键不变 
- **print**: IO：KStream → void; `终结操作`，`打印记录到输出流中。`stream.print(Printed.toFile("stream.out"));
- **selectKey**

> IO：KStream → KStream;  对每个记录分配一个新的键，键类型可能改变。

```java
KStream<String, String> rekeyed = stream.selectKey((key, value) -> value.split(" ")[0])
```

- **toStream**

> IO：KTable → KStream;  将表转换为流： table.toStream();

- **WriteAsText**

>  IO：KStream → void; `终结性操作，将流写出到文件`

#### .2. 有状态转转化

> 这类转换操作`需要依赖于某些状态信息`。例如在聚合性操作中，会`使用窗口化状态存储来保存上一个窗口的聚合结果`。在Join操作中，会使用窗`口化状态存储到目前为止接收到的、窗口边界内部的所有记录`。状态存储默认支持容错，`如果出现失败，则Kafka Streams会首先恢复所有的状态存储，然后再进行后续的处理`。高级的有状态转换操作包括：聚合、Join，以及针对两者的窗口化支持。

- **aggregate**

> IO：KGroupedStream → KTable 或 KGroupedTable → KTable; 滚动聚合（Rolling Aggregation）操作，根据分组键对非窗口化的记录的值进行聚合
>
> 当对已分组流进行聚合时，你需要提供初始化器（确定聚合初值）、聚合器adder。当聚合已分组表时，你需要额外提供聚合器subtractor。

```java
KGroupedStream<Bytes, String> groupedStream = null;
KGroupedTable<Bytes, String> groupedTable = null;
// 聚合一个分组流，值类型从字符串变为整数
KTable<Bytes, Long> aggregatedStream = groupedStream.aggregate(
    () -> 0L, /* 初始化器 */
    ( aggKey, newValue, aggValue ) -> aggValue + newValue.length(), /* 累加器 */
    Serdes.Long(), /* 值的串行化器 */
    "aggregated-stream-store" /* 状态存储的名称 */ );

KTable<Bytes, Long> aggregatedTable = groupedTable.aggregate(
    () -> 0L, /* 初始化器 */
    ( aggKey, newValue, aggValue ) -> aggValue + newValue.length(), /* 累加器 */
    ( aggKey, oldValue, aggValue ) -> aggValue - oldValue.length(), /* 减法器 */
    Serdes.Long(), /* 值的串行化器 */
    "aggregated-table-store" /* 状态存储的名称 */ );
```

KGroupedStream的聚合操作的行为：

1. 值为null的记录被忽略
2. 当首次收到某个新的记录键时，初始化器被调用
3. 每当接收到非null值的记录时，累加器被调用

KGroupedTable的聚合操作的行为：

1. 值为null的记录被忽略
2. 当首次收到某个新的记录键时，初始化器被调用（在调用累加器/减法器之前）。与KGroupedStream不同，随着时间的推移，针对一个键，可能调用初始化器多次。只要接收到目标键的墓碑记录
3. 当首次收到某个键的非null值时（INSERT操作），调用累加器
4. 当非首次收到某个键的非null值时（UPDATE操作）：
   1. 调用减法器，传入存储在KTable表中的旧值
   2. 调用累加器，传入刚刚接收到的新值
   3. 上述两个聚合器的执行顺序未定义
5. 当接收到墓碑记录（DELETE操作）亦即null值的记录时，调用减法器
6. 不论何时，减法器返回null时都会导致相应的键从结果KTable表中删除。遇到相同键的下一个记录时，会执行第3步的行为

- **KGroupedStream → KTable**

> 窗口化聚合：`以窗口为单位，根据分组键`，对KGroupedStream中的记录进行聚合操作，并把结果存放到窗口化的KTable

```java
KGroupedStream<String, Long> groupedStream = null;

// 基于时间的窗口化（滚动窗口）
KTable<Windowed<String>, Long> timeWindowedAggregatedStream = groupedStream
    .windowedBy( TimeWindows.of( TimeUnit.MINUTES.toMillis( 5 ) ) )
    .aggregate(
    () -> 0L, /* 初始化器 */
    ( aggKey, newValue, aggValue ) -> aggValue + newValue, /* 累加器 */
    /* 状态存储 */
    Materialized.<String, Long, WindowStore<Bytes, byte[]>>as( "time-windowed-aggregated-stream-store" )
    .withValueSerde( Serdes.Long() ) );
// 基于会话的窗口化
KTable<Windowed<String>, Long> sessionizedAggregatedStream = groupedStream
    .windowedBy( SessionWindows.with( TimeUnit.MINUTES.toMillis( 5 ) ) ) /* 窗口定义 */
    .aggregate(
    () -> 0L, /* 初始化器 */
    ( aggKey, newValue, aggValue ) -> aggValue + newValue, /* 累加器 */
    ( aggKey, leftAggValue, rightAggValue ) -> leftAggValue + rightAggValue, /* 会话合并器 */
    Materialized.<String, Long, SessionStore<Bytes, byte[]>>as( "sessionized-aggregated-stream-store" ).withValueSerde( Serdes.Long() ) );
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kafka_stream/  

