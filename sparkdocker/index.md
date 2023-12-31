# sparkdocker


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210227155237722.png)

#### 1. docker 从0搭建hadoop，spark

```shell
# 拉取并进入ubuntu镜像
docker pull daocloud.io/ubuntu
docker run -it daocloud.io/ubuntu  /bin/bash

#安装jdk1.8
apt-get install software-properties-common python-software-properties
add-apt-repository ppa:webupd8team/java
apt-get update
apt-get install oracle-java8-installer
update-java-alternatives -s java-8-oracle
java –version
#添加jdk1.8 环境变量
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
source /etc/profile
#提交保存镜像
docker commit 75bea785a41e 902chenjie/ubuntu_java
#下载hadoop、spark、scala、hive、hbase、zookeeper
wget http://mirrors.shu.edu.cn/apache/hadoop/common/hadoop-2.7.6/hadoop-2.7.6.tar.gz
tar -zxvf hadoop-2.7.6.tar.gz
wget http://mirrors.shu.edu.cn/apache/spark/spark-2.0.2/spark-2.0.2-bin-hadoop2.7.tgz
tar -zxvf spark-2.0.2-bin-hadoop2.7.tgz
wget https://downloads.lightbend.com/scala/2.11.11/scala-2.11.11.tgz
tar -zxvf scala-2.11.11.tgz
#下载hive安装包并解压（如果下面的地址下载不了，可以到http://archive.apache.org/dist/hive找到所有版本的下载地址）
wget http://mirrors.shu.edu.cn/apache/hive/hive-2.3.3/apache-hive-2.3.3-bin.tar.gz
tar -zxvf  apache-hive-2.3.3-bin.tar.gz
#下载hbase安装包并解压（如果下面的地址下载不了，可以到http://archive.apache.org/dist/hbase找到所有版本的下载地址）
wget http://mirrors.shu.edu.cn/apache/hbase/1.2.6/hbase-1.2.6-bin.tar.gz
tar -zxvf hbase-1.2.6-bin.tar.gz
#下载zookeeper（如果下面的地址下载不了，可以到http://archive.apache.org/dist/zookeeper找到所有版本的下载地址）
wget http://mirrors.shu.edu.cn/apache/zookeeper/zookeeper-3.4.11/zookeeper-3.4.11.tar.gz
tar -zxvf zookeeper-3.4.11.tar.gz
#配置环境
#zookeeper
export ZOOKEEPER_HOME=/root/zookeeper-3.4.11
export PATH=$ZOOKEEPER_HOME/bin:$PATH
#hadoop
export HADOOP_HOME=/root/hadoop-2.7.6
export CLASSPATH=.:$HADOOP_HOME/lib:$CLASSPATH
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_ROOT_LOGGER=INFO,console
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib"
#scala
export SCALA_HOME=/root/scala-2.11.11
export PATH=${SCALA_HOME}/bin:$PATH
#spark
export SPARK_HOME=/root/spark-2.0.2-bin-hadoop2.7
export PATH=${SPARK_HOME}/bin:${SPARK_HOME}/sbin:$PATH
#hive
export HIVE_HOME=/root/apache-hive-2.3.3-bin
export PATH=$PATH:$HIVE_HOME/bin
#hbase
export HBASE_HOME=/root/hbase-1.2.6
export PATH=$PATH:$HBASE_HOME/bin
#验证 输入hadoop， spark
#网络环境配置
apt-get install net-tools
apt-get install inetutils-ping
apt-get update
apt-get install openssh-server
service ssh start
ssh-keygen -t rsa
cat /root/.ssh/id_rsa.pub >>/root/.ssh/authorized_keys
#本机登录测试  ssh localhost
```

##### 1.1. hadoop 初步配置

- hadoop #hadoop 初步配置   #修改core-site.xml如下：

| 参数                | 说明           |
| ------------------- | -------------- |
| fs.defaultFS        | 默认的文件系统 |
| hadoop.tmp.dir      | 临时文件目录   |
| ha.zookeeper.quorum | Zkserver信息   |

```xml
<configuration>
        <property>
                <name>fs.defaultFS</name>
                <value>hdfs://localhost:9000</value>
        </property>
        <property>
         <name>io.file.buffer.size</name>
         <value>4096</value>
       </property>
 
        <property>
            <name>hadoop.tmp.dir</name>
            <value>/tmp</value>
        </property>
 
         <property>
            <name>hadoop.proxyuser.root.hosts</name>
            <value>*</value>
        </property>
        <property>
            <name>hadoop.proxyuser.root.groups</name>
            <value>*</value>
        </property>
</configuration>
```

- hadoop-env.sh

```xml
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
export JAVA_HOME=${JAVA_HOME}
```

- hdfs-site.xml

| 参数                                               | 说明                                                         |
| -------------------------------------------------- | ------------------------------------------------------------ |
| dfs.nameservices                                   | 名称服务，在基于HA的HDFS中，用名称服务来表示当前活动的NameNode |
| dfs.ha.namenodes.<nameservie>                      | 配置名称服务下有哪些NameNode                                 |
| dfs.namenode.rpc-address.<nameservice>.<namenode>  | 配置NameNode远程调用地址                                     |
| dfs.namenode.http-address.<nameservice>.<namenode> | 配置NameNode浏览器访问地址                                   |
| dfs.namenode.shared.edits.dir                      | 配置名称服务对应的JournalNode                                |
| dfs.journalnode.edits.dir                          | JournalNode存储数据的路径                                    |

```xml
<configuration>
        <property>
                <name>dfs.replication</name>
         <value>2</value>
        </property>
 
        <property>
                 <name>dfs.namenode.name.dir</name>
                <value>/root/hadoop-2.7.6/hdfs/name</value>
        </property>
        <property>
                 <name>dfs.datanode.data.dir</name>
                <value>/root/hadoop-2.7.6/hdfs/data</value>
        </property>
        <property>
                <name>dfs.permissions</name>
                <value>false</value>
        </property>
</configuration>
```

- mapred-site.xml

```xml
<configuration>
         <property>
         <name>mapreduce.framework.name</name>
         <value>yarn</value>
    </property>
</configuration>
```

- 修改slave 文件

```shell
vim slaves
#localhost
```

- yarn-site.xml

| 参数                          | 说明                                                        |
| ----------------------------- | ----------------------------------------------------------- |
| yarn.resourcemanager.hostname | RescourceManager的地址，NodeManager的地址在slaves文件中定义 |

```xml
<configuration>
        <property>
                <description>The hostname of the RM.</description>
                <name>yarn.resourcemanager.hostname</name>
                <value>master</value>
         </property>
        <property>
                 <name>yarn.nodemanager.aux-services</name>
                <value>mapreduce_shuffle</value>
       </property>
</configuration>
```

##### 1.2. spark 初步配置

![image-20210227165057910](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210227165057910.png)

- slaves文件

```
cd ~/spark-2.0.2-bin-hadoop2.7/conf 
#ip
a15ae831e68b
207343b5d21d
```

- spark-defaults.conf

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210227164642058.png)

- spark-env.sh

| 参数            | 说明                                         |
| --------------- | -------------------------------------------- |
| SPARK_MASTER_IP | Master的地址，Worker的地址在slaves文件中定义 |

```shell
export SPARK_MASTER_IP=cloud1
export SPARK_WORKER_MEMORY=128m
export JAVA_HOME=/software/jdk1.8.0_201
export SCALA_HOME=/software/scala-2.12.8
export SPARK_HOME=/software/spark-2.4.0-bin-hadoop2.7
export HADOOP_CONF_DIR=/software/hadoop-2.8.5/etc/hadoop
export SPARK_LIBRARY_PATH=$SPARK_HOME/lib 
export SCALA_LIBRARY_PATH=$SPARK_LIBRARY_PATH
export SPARK_WORKER_CORES=1
export SPARK_WORKER_INSTANCES=1
export SPARK_MASTER_PORT=7077
```

- etc/profile

```shell
# JAVA and JRE
export JAVA_HOME=/software/jdk1.8.0_201
export JRE_HOME=/software/jdk1.8.0_201/jre
export CLASSPATH=.:$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
export PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin
# HADOOP
export HADOOP_HOME=/software/hadoop-2.8.5
export HADOOP_CONFIG_HOME=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
# ZOOKEEPER
export ZOOKEEPER_HOME=/software/zookeeper-3.4.13
export PATH=$PATH:$ZOOKEEPER_HOME/bin
# SCALA
export SCALA_HOME=/software/scala-2.12.8
export PATH=$PATH:$SCALA_HOME/bin
# SPARK
export SPARK_HOME=/software/spark-2.4.0-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
```

- 打包成一个镜像

```shell
$ docker commit c4018f6e20e9
# sha256:190c26ac384acc8b35d50d2a518bb7731a9b9c95ff0e7448df4bd2a32db6e849 #这个是一段返回的ID也可以直接一个语句提交。
$ docker tag 190c26ac384acc8b35d50d2a518bb7731a9b9c95ff0e7448df4bd2a32db6e849 spark:1.0
```

- 将配置发送到其它节点

```shell
scp -r  ~/spark-2.0.2-bin-hadoop2.7/conf root@a15ae831e68b:/root/spark-2.0.2-bin-hadoop2.7/
scp -r  ~/spark-2.0.2-bin-hadoop2.7/conf root@207343b5d21d:/root/spark-2.0.2-bin-hadoop2.7/
```

- 启动spark集群

```shell
cd ~/spark-2.0.2-bin-hadoop2.7/sbin
hadoop fs -mkdir /historyserverforSpark  #创建spark history目录
./start-all.sh
jps  #查看进程
./spark-shell  #进入spark-shell测试
./pyspark  #进入pyspark 测试
```

#### 2. 脚本快速安装

```shell
wget https://raw.githubusercontent.com/zq2599/blog_demos/master/sparkdockercomposefiles/docker-compose.yml \
&& wget https://raw.githubusercontent.com/zq2599/blog_demos/master/sparkdockercomposefiles/hadoop.env \
&& docker-compose up -d 
#http://192.168.1.101:50070    DataNode
#worker，地址是：http://192.168.1.101:8080
#在宿主机执行，上传宿主机文件到hdfs； 其中宿主机的input_files目录已经挂载到namenode容器上了
docker exec namenode hdfs dfs -mkdir /input \
&& docker exec namenode hdfs dfs -put /input_files/GoneWiththeWind.txt /input

#执行提交命令
docker exec -it master spark-submit \
--class com.bolingcavalry.sparkwordcount.WordCount \
--executor-memory 512m \
--total-executor-cores 2 \
/root/jars/sparkwordcount-1.0-SNAPSHOT.jar \
namenode \
8020 \
GoneWiththeWind.txt
```

#### 3. 通过使用安装好`Zookeeper`、 `Hadoop`、 `Spark`、`Scala` 的镜像

```shell
docker run --name cloud1 -h cloud1 -ti spark:1.0
docker run --name cloud2 -h cloud2 -ti spark:1.0
docker run --name cloud3 -h cloud3 -ti spark:1.0
docker run --name cloud4 -h cloud4 -ti spark:1.0
docker run --name cloud5 -h cloud5 -ti spark:1.0
docker run --name cloud6 -h cloud6 -ti spark:1.0
# 修改/etc/hosts 文件，并传到其它主机上面，scp /etc/hosts cloud6:/etc/hosts
#直接通过命令 在启动docker容器的时候，在etc/hosts文件中写入对应的 端口映射
docker --name cloud2 -h cloud2 --add-host cloud1:172.17.0.2 --add-host cloud2:172.17.0.3 --add-host cloud3:172.17.0.4 -it Spark
```

#### 4. 学习链接

- https://blog.csdn.net/Magic_Ninja/article/details/87892216
- [spark&jupyter&Anaconda](https://www.yuque.com/7125messi/ouk92x/yvguly#d5f57e7c)
- docker spark imgae:
  - [nhervieu/spark2.4](https://hub.docker.com/r/nhervieu/spark2.4/tags?page=1&ordering=last_updated)
  - [spark-hadoop](https://hub.docker.com/r/navin0107/spark2.4.5-hadoop2.9.2)
  - [mirajgodha/spark3.0](https://hub.docker.com/r/mirajgodha/spark/tags?page=1&ordering=last_updated)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/sparkdocker/  

