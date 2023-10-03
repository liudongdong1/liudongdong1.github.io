# kafkaInstall


#### 1. Java JDK

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210719190439890.png)

#### 2. Zookeeper

- 下载地址： [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/apache/zookeeper/) [官网](http://zookeeper.apache.org/releases.html#download) [本地地址](E:\javacloud\apache-zookeeper-3.6.3-bin)

- 修改配置文件： 进入conf目录（例中为：E:\javacloud\apache-zookeeper-3.6.3-bin\conf）， 复制“zoo_sample.cfg”为“zoo.cfg”文件，编辑zoo.cfg
  - 查找并设置dataDir，设置数据存储目录，如下：`dataDir=E:\\javacloud\\apache-zookeeper-3.6.3-bin\\datadir`
  - 查找并设置clientPort（有必要的话），设置客户端连接端口，默认端口2181，`clientPort=2181`
  - Zookeeper AdminServer，默认使用8080端口；`admin.serverPort=9988；`
- 配置系统环境变量
  - 添加系统环境变量：ZOOKEEPER_HOME，设置对应值（例中配置：ZOOKEEPER_HOME= E:\\javacloud\\apache-zookeeper-3.6.3-bin\
  - 编辑path系统变量，添加路径：%ZOOKEEPER_HOME%\bin
- 打开cmd控制台窗口，输入“zkServer“，运行Zookeeper

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210719190921885.png)

#### 3. Kafka

- 下载地址： http://kafka.apache.org/downloads.html
- 修改 server.properties文件
  - `log.dirs=E:\\javacloud\\kafka_2.13-2.8.0\\tempdata`
  - 查找并设置zookeeper.connect，配置zookeeper连接字符串，格式：ip1:端口1,ip2:端口2,……,ipN:端口N，比如127.0.0.1:3000,127.0.0.1:3001,127.0.0.1:3002,每对ip和端口分别代表一个zookeeper服务器，kafka会按这里的配置去连接zookeeper, `zookeeper.connect=localhost:2181`
  - 查找并设置listener，配置监听端口，格式：listeners = listener_name://host_name:port，供kafka客户端连接用的ip和端口： `listeners=PLAINTEXT://127.0.0.1:9092` ;  这个有报错，使用下面，注意这个需要填写ip，否则无法与zookeeper进行连接；`advertised.listeners=PLAINTEXT://127.0.0.1:9092`
  - 启动命令： `.\bin\windows\kafka-server-start.bat .\config\server.properties`

```python

# broker的全局唯一编号，不能重复
broker.id=0
 
# 用来监听链接的端口，producer或consumer将在此端口建立连接
port=9092
 
# 处理网络请求的线程数量
num.network.threads=3
 
# 用来处理磁盘IO的线程数量
num.io.threads=8
 
# 发送套接字的缓冲区大小
socket.send.buffer.bytes=102400
 
# 接受套接字的缓冲区大小
socket.receive.buffer.bytes=102400
 
# 请求套接字的缓冲区大小
socket.request.max.bytes=104857600
 
# kafka消息存放的路径
log.dirs=D:/Net_Program/Net_Kafka/kafka-data
 
# topic在当前broker上的分片个数
num.partitions=1
 
# 用来恢复和清理data下数据的线程数量
num.recovery.threads.per.data.dir=1
 
# 默认消息的最大持久化时间，168小时，7天
log.retention.hours=168
 
# 日志文件中每个segment的大小，默认为1G
log.segment.bytes=1073741824
 
# 周期性检查文件大小的时间
log.retention.check.interval.ms=300000
 
# 日志清理是否打开，一般不用启用，启用的话可以提高性能
log.cleaner.enable=false
 
# zookeeper集群的地址，可以是多个，多个之间用逗号分割hostname1:port1,hostname2:port2,hostname3:port3
zookeeper.connect=localhost:2181
 
# zookeeper链接超时时间
zookeeper.connection.timeout.ms=6000
 
# partion buffer中，消息的条数达到阈值，将触发flush到磁盘
#log.flush.interval.messages=10000
 
# 消息buffer的时间，达到阈值，将触发flush到磁盘
#log.flush.interval.ms=1000
 
############################# 其他 #############################
 
# 消息保存的最大值20M
message.max.byte=20971520
 
# kafka保存消息的副本数，如果一个副本失效了，另一个还可以继续提供服务
default.replication.factor=2
 
# 取消息的最大直接数
replica.fetch.max.bytes=5242880
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210719194659065.png)

> Zookeeper 通过cmd窗口启动之后，长时间没有动，需要在cmd窗口按下return或者其他键；
>
> 然后重新运行kafka指令

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210719194841921.png)

> 不要关了这个窗口，启用Kafka前请确保ZooKeeper实例已经准备好并开始运行，这个窗口作为Kafka server。

`启动kafka的时候，bin/kafka-server-start.sh config/server.properties， 出现Classpath is empty. Please build the project first e.g. by running './gradlew jar -PscalaVersion=2.12.8'的提示，启动失败。`

- 安装版本不对，重新安装另一个版本就可以了

#### 4. 案例测试

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210719200102679.png)

- **创建主题**： 新建cmd窗口,进入kafka的windows目录下,cd D:\Tools\kafka_2.13-2.8.0\bin\windows,输入以下命令,创建一个叫topic001的主题

```shell
kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 3 --topic DHTtest
```

- **创建生产者**： 新建cmd窗口,进入kafka的windows目录下,cd D:\Tools\kafka_2.13-2.8.0\bin\windows,输入以下命令

```shell
kafka-console-producer.bat --broker-list localhost:9092 --topic topic001
```

- **创建消费者**：新建cmd窗口,进入kafka的windows目录下,cd D:\Tools\kafka_2.13-2.8.0\bin\windows,输入以下命令

```shell
kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic topic001 --from-beginning
```

- **列出主题**：kafka-topics.bat –list –zookeeper localhost:2181 
- **描述主题**：kafka-topics.bat –describe –zookeeper localhost:2181 –topic [Topic Name] 

--`bootstrap-server` 指定需要连接的服务器

--`group` 指定消费者所属消费组

--`topic` 指定消费者要消费的主题

--`from-beginning` `从头开始接收数据`，可以理解为offset为0

#### .1. json 案例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time

class Kafka_producer():
    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort
        ))

    def sendjsondata(self, params):
        try:
            parmas_message = json.dumps(params)
            producer = self.producer
            producer.send(self.kafkatopic, parmas_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)


def main():
    producer = Kafka_producer("127.0.0.1", 9092, "topic001")
    for i in range(10):
        dht_record = {
            'times':time.strftime("%H:%M:%S", time.localtime()),
            'DHT':i
        }
        params = dht_record
        print(params)
        producer.sendjsondata(params)
        time.sleep(0.3)


if __name__ == '__main__':

    main()
```

- DHTproduceer

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import sys
import cv2
class Kafka_producer():
    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort
        ))

    def sendjsondata(self, params):
        try:
            parmas_message = json.dumps(params)
            producer = self.producer
            producer.send(self.kafkatopic, parmas_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)


def main():
    producer = Kafka_producer("127.0.0.1", 9092, "flexSensor")
    for i in range(100):
        dht_record = {
            'times':time.strftime("%H:%M:%S", time.localtime()),
            'DHT':i
        }
        params = dht_record
        print(params)
        producer.sendjsondata(params)
        time.sleep(0.3)



def camareProduce(path_to_video):
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    topic = 'my-topic'
    print('start')
    video = cv2.VideoCapture(path_to_video)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        # png might be too large to emit
        data = cv2.imencode('.jpeg', frame)[1].tobytes()
        future = producer.send(topic, data)
        try:
            future.get(timeout=10)
        except KafkaError as e:
            print(e)
            break

        print('.', end='', flush=True)    
        


if __name__ == '__main__':

    #main()
    camareProduce(0)
```

- consumer.py

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import time

class Kafka_consumer():
    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        #self.groupid = groupid
        self.consumer = KafkaConsumer(self.kafkatopic,
                                      bootstrap_servers='{kafka_host}:{kafka_port}'.format(
                                          kafka_host=self.kafkaHost,
                                          kafka_port=self.kafkaPort))

    def consume_data(self):
        try:
            for message in self.consumer:
                # print json.loads(message.value)
                yield message
        except KeyboardInterrupt as e:
            print(e)

def main():
    consumer = Kafka_consumer("localhost",9092,"topic001")
    message = consumer.consume_data()
    for i in message:
        print(i.value)



if __name__ == '__main__':
    main()
```

- cameraConsumer.py

```python
from flask import Flask, Response
from kafka import KafkaConsumer
consumer = KafkaConsumer('my-topic', bootstrap_servers='localhost:9092')
app = Flask(__name__)
def kafkastream():
    for message in consumer:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + message.value + b'\r\n\r\n')
@app.route('/')
def index():
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
```

- 问题： 另一个电脑访问时出现问题，没报错，也没有显示

#### 5. bat脚本

```shell
@echo off
e:
cd E:\javacloud\apache-zookeeper-3.6.3-bin\bin
start zkServer
cd E:\javacloud\kafka_2.13-2.8.0
start .\bin\windows\kafka-server-start.bat .\config\server.properties
exit
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kafkainstall/  

