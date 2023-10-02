# kafka-python


> Kafka属于Apache组织，是一个高性能跨语言分布式发布订阅消息队列系统[7]。它的主要特点有：
>
> - 以时间复杂度O(1)的方式提供消息持久化能力，并对大数据量能保证常数时间的访问性能；
> - 高吞吐率，单台服务器可以达到每秒几十万的吞吐速率；
> - 支持服务器间的消息分区，支持分布式消费，同时保证了`每个分区内的消息顺序`；
> - 轻量级，支持实时数据处理和离线数据处理两种方式。

### 1. FrameWork

> Producer 使用 push 模式将消息发布到 broker，consumer 通过监听使用 pull 模式从 broker 订阅并消费消息。 多个 broker 协同工作，producer 和 consumer 部署在各个业务逻辑中。三者通过 zookeeper管理协调请求和转发。这样就组成了一个高性能的分布式消息发布和订阅系统。
>
> - broker: 消息中间件处理节点，一个Kafka节点就是一个broker，一个或者多个Broker可以组成一个Kafka集群
> - topic: 主题，Kafka根据topic对消息进行归类，发布到Kafka集群的每条消息都需要指定一个topic
> - consumergroup: 每个Consumer属于一个特定的Consumer Group，`一条消息可以发送到多个不同的Consumer Group`，但是`一个Consumer Group中只能有一个Consumer能够消费该消息`
> - Partion: 一个topic可以分为多个partition，每个partition内部是有序的

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-aa36f2bbc1a6ff0d8f03aad80759bb01_720w.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/16642395-cb6fccc9314e4278)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721210716067.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-23dcdca68db7e5406e3f036297f68c4d_720w.jpg)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210719182618893.png)

- cleaner-offset-checkpoint: 存了`每个log的最后清理offset`
- meta.properties: [broker.id](http://broker.id/) 信息
- recovery-point-offset-checkpoint:表示`已经刷写到磁盘的记录`。recoveryPoint以下的数据都是已经刷 到磁盘上的了。
- replication-offset-checkpoint: 用来存储每个replica的HighWatermark的(high watermark (HW)，表示已经被commited的message，HW以下的数据都是各个replicas间同步的，一致的。)

#### 1. KafkaConsumer

> The consumer will` transparently handle the failure of servers in the Kafka cluster,` and `adapt as topic-partitions are created or migrate between brokers`. It also interacts with the assigned kafka Group Coordinator node to allow multiple consumers to load balance consumption of topics (requires kafka >= 0.9.0.0).

1. `最多一次`：客户端收到消息后，在处理消息前自动提交，这样kafka就认为consumer已经消费过了，偏移量增加。enable.auto.commit为ture; enable.auto.commit为ture;  client不要调用commitSync()，kafka在特定的时间间隔内自动提交。
2. `最少一次`：客户端`收到消息，处理消息，再提交反馈`。这样就可能出现消息处理完了，在提交反馈前，网络中断或者程序挂了，那么kafka认为这个消息还没有被consumer消费，产生重复消息推送。设置enable.auto.commit为false; client调用commitSync()，增加消息偏移;
3. `正好一次`：保证`消息处理和提交反馈在同一个事务中`，即有原子性。

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer('my_topic', group_id= 'group2', bootstrap_servers= ['localhost:9092'], consumer_timeout_ms=1000)
for msg in consumer:
    print(msg)
```

- 第1个参数为 topic的名称

- group_id : 指定此消费者实例属于的组名，可以不指定

- bootstrap_servers ： 指定kafka服务器

- 若不指定 consumer_timeout_ms，默认一直循环等待接收，若指定，则超时返回，不再等待

  consumer_timeout_ms ： 毫秒数

```python
from kafka import KafkaConsumer
from kafka import TopicPartition

consumer = KafkaConsumer(group_id= 'group2', bootstrap_servers= ['localhost:9092'])
consumer.assign([TopicPartition(topic= 'my_topic', partition= 0)])  #手动分配partition
for msg in consumer:
    print(msg)
```

- **订阅多个topic**

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(group_id= 'group2', bootstrap_servers= ['localhost:9092'])
consumer.subscribe(topics= ['my_topic', 'topic_1'])
for msg in consumer:
    print(msg)
```

- **正则订阅一类topic**

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(group_id= 'group2', bootstrap_servers= ['localhost:9092'], value_deserializer=lambda m: json.loads(m.decode('ascii')))
consumer.subscribe(pattern= '^my.*')
for msg in consumer:
    print(msg)
```

#### 2. [KafkaProducer](https://kafka-python.readthedocs.io/en/master/apidoc/KafkaProducer.html)

![《kafka实战教程(python操作kafka)，kafka配置文件详解》](https://gitee.com/github-25970295/blogpictureV2/raw/master/jUbM73.jpg)

> 创建`ProducerRecord必须包含Topic和Value`，key和partition可选。然后，`序列化key和value对象为ByteArray`，并发送到网络。消息`发送到partitioner`。如果创建ProducerRecord时指定了partition，此时partitioner啥也不用做，简单的返回指定的partition即可。如果未指定partition，`partitioner会基于ProducerRecord的key生成partition`。producer选择好partition后，增加record到对应topic和partition的batch record。最后，`专有线程负责发送batch record到合适的Kafka broker`。
>
> 当broker收到消息时，它会返回一个应答（response）。如果`消息成功写入Kafka，broker将返回RecordMetadata对象（包含topic，partition和offset）`；相反，broker将返回error。这时producer收到error会尝试重试发送消息几次，直到producer返回error。

- `立即发送`：只管`发送消息到server端`，不care消息是否成功发送。大部分情况下，这种发送方式会成功，因为Kafka自身具有高可用性，producer会自动重试；但有时也会丢失消息；
- `同步发送`：`通过send()方法发送消息，并返回Future对象。get()方法会等待Future对象，看send()方法是否成功`；
- `异步发送`：通过带有`回调函数的send()方法发送消息`，当producer收到Kafka broker的response会触发回调函数

> The producer consists of `a pool of buffer space` that holds records that haven’t yet been transmitted to the server as well as a background I/O thread that is responsible for turning these records into requests and transmitting them to the cluster.
>
> - **bootstrap_servers** – ‘host[:port]’ string (or list of ‘host[:port]’ strings) that the `producer should contact to bootstrap initial cluster metadata`. This does not have to be the full node list. It just needs to have at least one broker that will respond to a Metadata API Request. Default port is 9092. If no servers are specified, will default to localhost:9092.
> - `send`**(***topic***,** *value=None***,** *key=None***,** *headers=None***,** *partition=None***,** *timestamp_ms=None***)**
>   -  a key to associate with the message. Can be used to determine which partition to send the message to. If partition is None (and producer’s partitioner config is left as default), then messages with the same key will be delivered to the same partition (but if key is None, partition is chosen randomly). Must be type bytes, or be serializable to bytes via configured key_serializer.

- 发送字符串类型的key和value

```python
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
future = producer.send('my_topic' , key= b'my_key', value= b'my_value', partition= 0)
result = future.get(timeout= 10)  #等待单条消息发送完成或超时，经测试，必须有这个函数，不然发送不出去，或用time.sleep代替
print(result)
```

- 消费者收到的为字符串类型，就需要解码操作，key_deserializer= bytes.decode

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(group_id= 'group2', bootstrap_servers= ['localhost:9092'], key_deserializer= bytes.decode, value_deserializer= bytes.decode)
consumer.subscribe(pattern= '^my.*')
for msg in consumer:
    print(msg)
```

- 发送msgpack消息: msgpack为MessagePack的简称，是高效二进制序列化类库，比json高效

```python
producer = KafkaProducer(value_serializer=msgpack.dumps)
producer.send('msgpack-topic', {'key': 'value'})
```

- 可压缩消息发送, 若消息过大，还可压缩消息发送，可选值为 ‘gzip’, ‘snappy’, ‘lz4’, or None

```python
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], compression_type='gzip')
future = producer.send('my_topic' ,  key= b'key_3', value= b'value_3', partition= 0)
future.get(timeout= 10)
```

#### 3. KafkaAdminClient

> The KafkaAdminClient class will negotiate for the latest version of each message protocol format supported by both the kafka-python client library and the Kafka broker. 

#### 4. KafkaClient

> A network client for asynchronous request/response network I/O. This is an internal class used to implement the user-facing producer and consumer clients.

#### 5. BrokerConnection

> Initialize a Kafka broker connection

#### 6. ClusterMetadata

> A class to manage kafka cluster metadata. This class does not perform any IO. It simply updates internal state given API responses (MetadataResponse, GroupCoordinatorResponse).

```python
ConsumerRecord(topic='my_topic', partition=0, offset=4, timestamp=1529569531392, timestamp_type=0, key=b'my_value', value=None, checksum=None, serialized_key_size=8, serialized_value_size=-1)
```

- topic
- partition
- offset ： 这条消息的偏移量
- timestamp ： 时间戳
- timestamp_type ： 时间戳类型
- key ： key值，字节类型
- value ： value值，字节类型
- checksum ： 消息的校验和
- serialized_key_size ： 序列化key的大小
- serialized_value_size ： 序列化value的大小，可以看到value=None时，大小为-1

#### 7. Example

- producer

```python
from kafka import KafkaProducer
from kafka.errors import KafkaError

producer = KafkaProducer(bootstrap_servers=['broker1:1234'])

# Asynchronous by default
future = producer.send('my-topic', b'raw_bytes')

# Block for 'synchronous' sends
try:
    record_metadata = future.get(timeout=10)
except KafkaError:
    # Decide what to do if produce request failed...
    log.exception()
    pass

# Successful result returns assigned partition and offset
print (record_metadata.topic)
print (record_metadata.partition)
print (record_metadata.offset)

# produce keyed messages to enable hashed partitioning
producer.send('my-topic', key=b'foo', value=b'bar')

# encode objects via msgpack
producer = KafkaProducer(value_serializer=msgpack.dumps)
producer.send('msgpack-topic', {'key': 'value'})

# produce json messages
producer = KafkaProducer(value_serializer=lambda m: json.dumps(m).encode('ascii'))
producer.send('json-topic', {'key': 'value'})

# produce asynchronously
for _ in range(100):
    producer.send('my-topic', b'msg')

def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)

def on_send_error(excp):
    log.error('I am an errback', exc_info=excp)
    # handle exception

# produce asynchronously with callbacks
producer.send('my-topic', b'raw_bytes').add_callback(on_send_success).add_errback(on_send_error)

# block until all async messages are sent
producer.flush()

# configure multiple retries
producer = KafkaProducer(retries=5)
```

- consumer.py

```python
from kafka import KafkaConsumer

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer('my-topic',
                         group_id='my-group',
                         bootstrap_servers=['localhost:9092'])
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                          message.offset, message.key,
                                          message.value))

# consume earliest available messages, don't commit offsets
KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

# consume json messages
KafkaConsumer(value_deserializer=lambda m: json.loads(m.decode('ascii')))

# consume msgpack
KafkaConsumer(value_deserializer=msgpack.unpackb)

# StopIteration if no message after 1sec
KafkaConsumer(consumer_timeout_ms=1000)

# Subscribe to a regex topic pattern
consumer = KafkaConsumer()
consumer.subscribe(pattern='^awesome.*')

# Use multiple consumers in parallel w/ 0.9 kafka brokers
# typically you would run each on a different server / process / CPU
consumer1 = KafkaConsumer('my-topic',
                          group_id='my-group',
                          bootstrap_servers='my.server.com')
consumer2 = KafkaConsumer('my-topic',
                          group_id='my-group',
                          bootstrap_servers='my.server.com')
```

### 2. Application

#### .1. 日志分析平台

> Kafka 性能高效，采集日志时业务无感知以及Hadoop/ODPS 等离线仓库存储和 Storm/Spark 等实时在线分析对接的特性决定它非常适合作为"日志收集中心"。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722101827282.png)

#### .2. **网站活动跟踪**

> 通过消息队列 Kafka 版可以`实时收集网站活动数据（包括用户浏览页面、搜索及其他行为等）`。发布-订阅的模式可以根据不同的业务数据类型，将消息发布到不同的 Topic；还可通过订阅消息的实时投递，将`消息流用于实时监控与业务分析或加载到 Hadoop、ODPS 等离线数据仓库系统进行离线处理`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722101936221.png)

#### .3. 流计算

> `股市走向分析、气象数据测控、网站用户行为分析等领域`，由于数据产生快、实时性强、数据量大，所以很难统一采集并入库存储后再做处理，这便导致传统的数据处理架构不能满足需求。而`大数据消息中间件 Kafka 以及 Storm/Samza/Spark 等流计算引擎的出现，可以根据业务需求对数据进行计算分析，最终把结果保存或者分发给需要的组件。`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722102127668.png)

#### .4. 数据中转枢纽

> 近年来`KV存储（HBase）、搜索（ElasticSearch）、流式处理（Storm/Spark Streaming/Samza）、时序数据库（OpenTSDB）`等专用系统应运而生，产生了同一份数据集需要被注入到多个专用系统内的需求。利用大数据消息中间件 Kafka 作为数据中转枢纽，同份数据可以被导入到不同专用系统中。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210722102238409.png)

#### Resouce

- https://kafka-python.readthedocs.io/en/master/usage.html#kafkaconsumer

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kafka-python/  

