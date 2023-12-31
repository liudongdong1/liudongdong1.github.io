# Environment


### 1. Maven 安装

#### 1.1. [下载](http://maven.apache.org/download.cgi)
![](https://img-blog.csdnimg.cn/2019052215085881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0ODQ1Mzk0,size_16,color_FFFFFF,t_70)

#### 1.2. 解压

```shell
tar zxvf apache-maven-3.6.1-bin.tar.gz -C /softdowload
```

#### 1.3. 环境变量

```shell
vim ~/.bashrc
export M2_HOME=/opt/apache-maven-3.6.1
export PATH=$PATH:$M2_HOME/bin
source ~/.bashrc
mvn -version
```

### 2. Java

- [百度云盘下载JDK1.8安装包](https://pan.baidu.com/s/1mUR3M2U_lbdBzyV_p85eSA)

```shell
sudo tar -zxvf ./jdk-8u162-linux-x64.tar.gz -C /usr/java/jdk1.8.0_261
#set oracle jdk environment
export JAVA_HOME=/usr/java/jdk1.8.0_261
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${PATH}:${JAVA_HOME}/bin

#注意
java -version
```

- openjdk

```shell
sudo apt-get install openjdk-7-jre openjdk-7-jdk
dpkg -L openjdk-7-jdk | grep '/bin/javac'  #该命令会输出一个路径，除去路径末尾的 “/bin/javac”，剩下的就是正确的路径了。如输出路径为 /usr/lib/jvm/java-7-openjdk-amd64/bin/javac，则我们需要的路径为 /usr/lib/jvm/java-7-openjdk-amd64

```

- jdk,jre

```shell
sudo apt-get install default-jre default-jdk
export JAVA_HOME=/usr/lib/jvm/default-java
```

### 3. Hadoop

- [下载地址](http://mirrors.cnnic.cn/apache/hadoop/common/)

```shell
sudo tar -zxf ~/softdowload/hadoop-2.6.0.tar.gz -C /usr/local    # 解压到/usr/local中
cd /usr/local/
sudo mv ./hadoop-2.6.0/ ./hadoop            # 将文件夹名改为hadoop
sudo chown -R ldd ./hadoop       # 修改文件权限
cd /usr/local/hadoop
./bin/hadoop version
```

```shell
sudo vim ~/.bashrc
#hadoop
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$PATH:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin
source ~/.bashrc
```

### 4. Spark

- [下载地址](https://downloads.apache.org/spark/spark-3.0.1/)

```shell
sudo apt install scala
sudo tar -zxvf ~/software/spark-3.0.1-bin-hadoop3.2.tgz -C/usr/local

export SPARK_HOME=/usr/local/spark-3.0.1-bin-hadoop3.2
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_PYTHON=/home/ldd/anaconda3/bin/python

#python 通过which 查看，注意anaconda 配合使用

spark-shell
pysark

```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201205234240178.png)

- `不知道为什么，在source 后没有在最终的path中生成相应的环境变量；`

### 学习链接

- [Hadoop 安装](http://dblab.xmu.edu.cn/blog/install-hadoop/)
- [spark 安装](https://phoenixnap.com/kb/install-spark-on-ubuntu)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/environment/  

