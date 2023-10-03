# PySpark


> Apart from real-time and batch processing, Apache Spark supports interactive queries and iterative algorithms. Using PySpark, you can work with **RDDs** in Python programming language also. It is because of a library called **Py4j** that they are able to achieve this.

> - **Resilient Distributed Dataset (RDD)**: RDD is an immutable (read-only), fundamental `collection of elements or items` that can be operated on many devices at the same time (parallel processing). Each dataset in an RDD can be divided into logical portions, which are then executed on different nodes of a cluster.
> - **Directed Acyclic Graph (DAG)**: DAG is the `scheduling layer of the Apache Spark architecture` that implements **stage-oriented scheduling**. Compared to MapReduce that creates a graph in two stages, Map and Reduce, Apache Spark can create DAGs that contain many stages.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119211239377.png)

## 0. Architecture

![Basic](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119211608027.png)

![standalone](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119211639658.png)

![Yarn](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119211701746.png)

![component](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119211800146.png)

## 1. core class

[`pyspark.SparkContext`](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext)Main entry point for Spark functionality.

[`pyspark.RDD`](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD)A Resilient Distributed Dataset (RDD), the basic abstraction in Spark.

[`pyspark.streaming.StreamingContext`](https://spark.apache.org/docs/latest/api/python/pyspark.streaming.html#pyspark.streaming.StreamingContext)Main entry point for Spark Streaming functionality.

[`pyspark.streaming.DStream`](https://spark.apache.org/docs/latest/api/python/pyspark.streaming.html#pyspark.streaming.DStream)A Discretized Stream (DStream), the basic abstraction in Spark Streaming.

[`pyspark.sql.SparkSession`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession)Main entry point for DataFrame and SQL functionality.

[`pyspark.sql.DataFrame`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)A distributed collection of data grouped into named columns.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210225232741569.png)

```shell
#连接spark cluster
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("sparkAppExample")
sc = SparkContext(conf=conf)

#使用session
from pyspark.sql import SparkSession
spark = SparkSession.builder \
          .master("local") \
          .appName("Word Count") \
          .config("spark.some.config.option", "some-value") \
          .getOrCreate()
# 如果使用 hive table 则加上 .enableHiveSupport()
#spark.sparkContext._conf.getAll()  # check the config
```

```python
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


def create_sc():
    sc_conf = SparkConf()
    sc_conf.setMaster('spark://master:7077')
    sc_conf.setAppName('my-app')
    sc_conf.set('spark.executor.memory', '2g')  #executor memory是每个节点上占用的内存。每一个节点可使用内存
    sc_conf.set("spark.executor.cores", '4') #spark.executor.cores：顾名思义这个参数是用来指定executor的cpu内核个数，分配更多的内核意味着executor并发能力越强，能够同时执行更多的task
    sc_conf.set('spark.cores.max', 40)    #spark.cores.max：为一个application分配的最大cpu核心数，如果没有设置这个值默认为spark.deploy.defaultCores
    sc_conf.set('spark.logConf', True)    #当SparkContext启动时，将有效的SparkConf记录为INFO。
    print(sc_conf.getAll())

    sc = SparkContext(conf=sc_conf)

    return sc

from pyspark.conf import SparkConf
conf=SparkConf()
        conf.set('spark.sql.execute.arrow.enabled','true')
        if os.getenv("APP_MODE") == 'prod':
            """
            集群环境
            """
            url = 'spark://master:7077'
            conf.setAppName('prod-practice-info').setMaster(url).set("spark.driver.maxResultSize", "12g").set("spark.executor.memory", '4g')
        else:
            """
            本地环境
            """
            print("本地环境")
            url = 'local[*]'
            conf.setAppName('prod-practice-info').setMaster(url)
        spark = SparkSession.builder. \
            config(conf=conf).\
            getOrCreate()
```

### 1.0. Submit Model

- `--class`: The entry point for your application (e.g. `org.apache.spark.examples.SparkPi`)
- `--master`: The [master URL](https://spark.apache.org/docs/latest/submitting-applications.html#master-urls) for the cluster (e.g. `spark://23.195.26.187:7077`)
- `--deploy-mode`: Whether to deploy your driver on the worker nodes (`cluster`) or locally as an external client (`client`) (default: `client`) **†**
- `--conf`: Arbitrary Spark configuration property in key=value format. For values that contain spaces wrap “key=value” in quotes (as shown). Multiple configurations should be passed as separate arguments. (e.g. `--conf <key>=<value> --conf <key2>=<value2>`)
- `application-jar`: Path to a bundled jar including your application and all dependencies. The URL must be globally visible inside of your cluster, for instance, an `hdfs://` path or a `file://` path that is present on all nodes.
- `application-arguments`: Arguments passed to the main method of your main class.

```shell
# Run application locally on 8 cores
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master local[8] \
  /path/to/examples.jar \
  100

# Run on a Spark standalone cluster in client deploy mode
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master spark://207.184.161.138:7077 \
  --executor-memory 20G \
  --total-executor-cores 100 \
  /path/to/examples.jar \
  1000

# Run on a Spark standalone cluster in cluster deploy mode with supervise
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master spark://207.184.161.138:7077 \
  --deploy-mode cluster \
  --supervise \
  --executor-memory 20G \
  --total-executor-cores 100 \
  /path/to/examples.jar \
  1000

# Run on a YARN cluster
export HADOOP_CONF_DIR=XXX
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master yarn \
  --deploy-mode cluster \  # can be client for client mode
  --executor-memory 20G \
  --num-executors 50 \
  /path/to/examples.jar \
  1000

# Run a Python application on a Spark standalone cluster
./bin/spark-submit \
  --master spark://207.184.161.138:7077 \
  examples/src/main/python/pi.py \
  1000

# Run on a Mesos cluster in cluster deploy mode with supervise
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master mesos://207.184.161.138:7077 \
  --deploy-mode cluster \
  --supervise \
  --executor-memory 20G \
  --total-executor-cores 100 \
  http://path/to/examples.jar \
  1000

# Run on a Kubernetes cluster in cluster deploy mode
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master k8s://xx.yy.zz.ww:443 \
  --deploy-mode cluster \
  --executor-memory 20G \
  --num-executors 50 \
  http://path/to/examples.jar \
  1000
```

### 1.1. RDD

> 支持两种类型的操作： 转化操作（transformation） 和行动操作（action）。转化操作会由一个RDD 生成一个新的RDD。行动操作是对的RDD 内容进行操作，它们会把最终求得的结果返回到驱动器程序，或者写入外部存储系统中。由于行动操作需要生成实际的输出，它们会强制执行那些求值必须用到的RDD 的转化操作。RDD的转化操作与行动操作不同，是惰性求值的，也就是在被调用行动操作之前Spark 不会开始计算。同样创建操作也是一样，数据并没有被立刻读取到内存中，只是记录了读取操作需要的相关信息。我理解为这与tensorflow的网络构建类似，我们之前编写的代码只是记录了整个操作过程的计算流程图，只有当计算操作被激活时，数据才会沿着之前定义的计算图进行计算.

#### 1.1.1. RDD creation

```python
#---  Creating a RDD from a file
import urllib
f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")
data_file = "./kddcup.data_10_percent.gz"
raw_data = sc.textFile(data_file).cache()
#---  Creating and RDD using parallelize
a = range(100)
data = sc.parallelize(a)
#---  makeRDD 操作
val rdd02 = sc.makeRDD(Array(1,2,3,4,5,6))
```




```python
#----  create key-value data
key_value_data = csv_data.map(lambda x: (x[41], x)) # x[41] contains the network interaction tag
durations_by_key = key_value_duration.reduceByKey(lambda x, y: x + y)
counts_by_key = key_value_data.countByKey()

head_rows = raw_data.take(5)# 查看前面5个
count=raw_data.count() # 查看个数
#-------   sample
raw_data_sample = raw_data.sample(False, 0.1, 1234)  # # whether the sampling is done with replacement, sample size as a fraction, random seed;
raw_data_sample = raw_data.takeSample(False, 400000, 1234)#grab a sample of raw data from our RDD into local memory, number samples

#----     set operation
attack_raw_data = raw_data.subtract(normal_raw_data)

csv_data = raw_data.map(lambda x: x.split(","))

normal_raw_data = raw_data.filter(lambda x: 'normal.' in x)  #count how many normal

def parse_interaction(line):
    elems = line.split(",")
    tag = elems[41]
    return (tag, elems)
key_csv_data = raw_data.map(parse_interaction)

all_raw_data = raw_data.collect()#get all the elements in the RDD into memory for us to work with them.
```

#### 1.1.2.  TransformOp

> 皆产生新的 RDD,且直保存运算逻辑，依赖原始 rdd

##### 1. map & mapValues

`对于每个元素`都应用这个func

- 入参：
  - func表示需要应用到每个元素的方法
  - preservesPartitioning是否保持当前分区方式，默认重新分区
- 返回：
  - `返回的结果是一个RDD`

```python
rdd = sc.parallelize(["b", "a", "c"])
sorted(rdd.map(lambda x: (x, 1)).collect()) #[('a', 1), ('b', 1), ('c', 1)]
```

`对键值对中每个value都应用这个func`，并保持key不变

- 入参：
  - func表示需要应用到每个元素值上的方法
- 返回：
  - 返回的结果是一个RDD

```python
x = sc.parallelize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])])
def f(x): return len(x)
x.mapValues(f).collect() # [('a', 3), ('b', 1)]
```

##### 2. flatmap

遍历全部元素，将传入方法应用到每个元素上，并将`最后结果展平（压成一个List）`

- 入参：
  - func表示需要应用到每个元素的方法
  - preservesPartitioning是否保持当前分区方式，默认重新分区
- 返回：
  - 返回的结果是一个RDD

```python
rdd = sc.parallelize([2, 3, 4])
sorted(rdd.flatMap(lambda x: range(1, x)).collect()) #[1, 1, 1, 2, 2, 3]
sorted(rdd.flatMap(lambda x: [(x, x), (x, x)]).collect()) #[(2, 2), (2, 2), (3, 3), (3, 3), (4, 4), (4, 4)]
```

遍历某个元素的元素值，将传入方法应用到每个元素值上，并将最后结果展平（压成一个List）

- 入参：
  - func表示需要应用到每个元素值的方法
- 返回：
  - 返回的结果是一个RDD

```python
x = sc.parallelize([("a", ["x", "y", "z"]), ("b", ["p", "r"])])
x.flatMapValues(lambda val: val).collect() # [('a', 'x'), ('a', 'y'), ('a', 'z'), ('b', 'p'), ('b', 'r')]
```

##### 3. filter

遍历全部元素，筛选符合传入方法的元素

- 入参：
  - func表示需要应用到每个元素的筛选方法
- 返回：
  - 返回的结果是一个RDD

```python
rdd = sc.parallelize([1, 2, 3, 4, 5])
rdd.filter(lambda x: x % 2 == 0)
print(rdd.collect()) # [2, 4]
kvRDD1.fiter(lambda x:x[0]<5) 根据 key过滤
kvRDD1.filter(lambda x:x[1]<5) 根据值过滤
```

##### 4. distinct

遍历全部元素，`并返回包含的不同元素的总数`

- 入参：
  - numPartitions表示需要将此操作分割成多少个分区
- 返回：
  - 返回的结果是一个Int

```python
print(rddInt.distinct().collect())
```

##### 5. cartesian

返回自己与传入rdd的笛卡尔积

- 入参：
  - rdd表示一个rdd对象，可以存储不同数据类型 RDD
- 返回：
  - 返回的结果是一个RDD

```python
num_rdd = sc.parallelize([1, 2])
str_rdd = sc.parallelize(['a', 'y'])
result = num_rdd.cartesian(str_rdd)
print(result.collect()) # [(1, 'a'), (1, 'y'), (2, 'a'), (2, 'y')]
```

#### 1.1.3. ActionOp

##### 1. collect

将数据以List取回本地
[官网](https://spark.apache.org/docs/latest/api/python/pyspark.html)提示，建议只在任务结束时在调用collect方法.

- 返回：
  - 返回的结果是一个List

##### 2. count&&take(num)

##### 3.  countByValue& countByKey

返回`每个key对应的元素数量`

- 返回：
  - 返回的结果是一个Dict

```python
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
print(rdd.countByKey()) # defaultdict(<class 'int'>, {'a': 2, 'b': 1})
```

返回`每个value出现的次数`

- 返回：
  - 返回的结果是一个Dict

```python
rdd2 = sc.parallelize([1, 2, 1, 2, 2], 2)
print(rdd2.countByValue())  # defaultdict(<class 'int'>, {1: 2, 2: 3})
```

##### 4. reduce

对于每个元素值都应用这个func

- 入参：
  - func表示需要应用到每个元素的方法
- 返回：
  - 返回的结果是一个Python obj, 与元素值得数据类型一致

```python
x = sc.parallelize([1, 2, 3])
y = x.reduce(lambda a, b : a + b )
print(x.collect()) # [1, 2, 3]
print(y) # 6
```

##### 5. fold

```python
# 和reduce() 一样， 但是需要提供初始值
>>> rdd = sc.parallelize([2, 2, 3, 4, 5, 5])
>>> rdd.fold(0, lambda x, y: x + y)
21
```

##### 6. aggregate

```python
# 和reduce() 相似， 但是通常返回不同类型的函数
>>> seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
>>> combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
>>> sc.parallelize([1, 2, 3, 4]).aggregate((0, 0), seqOp, combOp)
(10, 4)
```

##### 7. foreach

用于遍历RDD中的元素,将函数func应用于每一个元素。

- 入参：
  - func表示需要应用到每个元素的方法, 但这个方法不会在客户端执行
- 返回：
  - 返回的结果是一个RDD

```python
def f(x): print(x)
sc.parallelize([1, 2, 3, 4, 5]).foreach(f)
```

### 1.2. SQL&DataFrames

> - **Use of Input Optimization Engine**: DataFrames `make use of the input optimization engines`, e.g., **Catalyst Optimizer**, to process data efficiently. We can use the same engine for all Python, Java, Scala, and R DataFrame APIs.
> - **Handling of Structured Data**: DataFrames provide a` schematic view of data`. Here, the data has some meaning to it when it is being stored.
> - **Custom Memory Management**: In `RDDs, the data is stored in memory`, whereas `DataFrames store data off-heap` (outside the main Java Heap space, but still inside RAM), which in turn reduces the garbage collection overload.
> - **Flexibility**: `DataFrames, like RDDs`, can support various formats of data, such as CSV, [Cassandra](https://intellipaat.com/blog/apache-cassandra-a-brief-intro/), etc.
> - **Scalability**: `DataFrames can be integrated with various other [Big Data tools](https://intellipaat.com/blog/big-data-analytics-tools-performance-testing/),` and they allow processing megabytes to petabytes of data at once.

- pyspark.sql.SQLContext： DataFrame和SQL方法的主入口
- pyspark.sql.DataFrame： 将分布式数据集分组到指定列名的数据框中
- pyspark.sql.Column ：DataFrame中的列`Row(name="Alice", age=11).asDict() == {'name': 'Alice', 'age': 11}`
- pyspark.sql.Row： DataFrame数据的行
- pyspark.sql.HiveContext： 访问Hive数据的主入口
- pyspark.sql.GroupedData： 由DataFrame.groupBy()创建的聚合方法集
- pyspark.sql.DataFrameNaFunctions： 处理丢失数据(空数据)的方法
- pyspark.sql.DataFrameStatFunctions： 统计功能的方法
   -pyspark.sql.functions DataFrame：可用的内置函数
- pyspark.sql.types： 可用的数据类型列表
- pyspark.sql.Window： 用于处理窗口函数

##### 1. creation

```python
#--- create from csv
fifa_df = spark.read.csv("path-of-file/fifa_players.csv", inferSchema = True, header = True)
#--- create from json
val jsondata=spark.read.json("file.json")
#--- create from exitRDD
df=spark.createDataFrame(rdd).toDF("key","cube")
#--- create from dict
df = spark.createDataFrame([{'name':'Alice','age':1},
    {'name':'Polo','age':1}]) 
#--- create from schema
schema = StructType([
    StructField("id", LongType(), True),   
    StructField("name", StringType(), True),
    StructField("age", LongType(), True),
    StructField("eyeColor", StringType(), True)
])
df = spark.createDataFrame(csvRDD, schema)
#--- create from pandas
colors = ['white','green','yellow','red','brown','pink']
color_df=pd.DataFrame(colors,columns=['color'])
color_df['length']=color_df['color'].apply(len)

color_df=spark.createDataFrame(color_df)
color_df.show()
```

##### 2. show

```python
df.printSchema()
df.columns()
df.count()
df.describe('column name').show() # look at the summary of particular column
df.dtypes #将所有列名称及其数据类型作为列表返回。
# 查找每列出现次数占总的30%以上频繁项目
df.stat.freqItems(["id", "gender"], 0.3).show()
# ----  缺失值
# 计算每列空值数目
for col in df.columns:
    print(col, "\t", "with null values: ", 
          df.filter(df[col].isNull()).count())
```

##### 3. column select

```python
df.select('column name','name2').show()
color_df.filter(color_df['length']>=4).show()   # filter方法
# 返回具有新指定列名的DataFrame
df.toDF('f1', 'f2')

first_row = df.head()
# Row(address=Row(city='Nanjing', country='China'), age=12, name='Li')

# 读取行内某一列的属性值
first_row['age']           # 12
first_row.age              # 12
getattr(first_row, 'age')  # 12
first_row.address
# Row(city='Nanjing', country='China')

# -------------- column -----------------------

first_col = df[0]
first_col = df['adress']
# Column<b'address'>

# copy column[s]
address_copy = first_col.alias('address_copy')

# rename column / create new column
df.withColumnRenamed('age', 'birth_age')
df.withColumn('age_copy', df['age']).show(1)
"""
+----------------+---+----+--------+
|         address|age|name|age_copy|
+----------------+---+----+--------+
|[Nanjing, China]| 12|  Li|      12|
+----------------+---+----+--------+
only showing top 1 row
"""

df.withColumn('age_over_18',df['age'] > 18).show(1)
"""
+----------------+---+----+-----------+
|         address|age|name|age_over_18|
+----------------+---+----+-----------+
|[Nanjing, China]| 12|  Li|      false|
+----------------+---+----+-----------+
only showing top 1 row
"""
```

##### 4.  filter

```python
df.filter(df.MathchID=='1111').show()
```

##### 5. sort

```python
df.orderBy(df.column)  # default ascending
color_df.sort('column',ascending=False).show()
```

##### 6. group

```python
df.goupby('columnname').count().show()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20201119164459904.png)

##### 7. SQL

```python
df.registerTempTable('tablename') # 注册表
sqlContext.sql('select * from tablename')
#  方式二：
df.createOrReplaceTempView(tablename)
result=spark.sql("sql sentences")
#方式三 sql function
from pyspark.sql import functions as F
import datetime as dt

# 装饰器使用
@F.udf()
def calculate_birth_year(age):
    this_year = dt.datetime.today().year
    birth_year = this_year - age
    return birth_year 

calculated_df = df.select("*", calculate_birth_year('age').alias('birth_year'))
calculated_df .show(2)
"""
+------------------+---+-------+----------+
|           address|age|   name|birth_year|
+------------------+---+-------+----------+
|  [Nanjing, China]| 12|     Li|      2008|
|[Los Angeles, USA]| 14|Richard|      2006|
+------------------+---+-------+----------+
only showing top 2 rows
"""
```

##### 8. drop

```python
df.drop('column name')#：cols - 要删除的列的字符串名称，要删除的列或要删除的列的字符串名称的列表。
#新增一列
df.withColumn(colName, col)
#通过为原数据框添加一个新列或替换已存在的同名列而返回一个新数据框。colName —— 是一个字符串, 为新列的名字。必须是已存在的列的名字
#col —— 为这个新列的 Column 表达式。必须是含有列的表达式。如果不是它会报错 AssertionError: col should be Column

# 重新命名聚合后结果的列名(需要修改多个列名就跟多个：withColumnRenamed)
# 聚合之后不修改列名则会显示：count(member_name)
df_res.agg({'member_name': 'count', 'income': 'sum', 'num': 'sum'})
      .withColumnRenamed("count(member_name)", "member_num").show()
    
#修改数据类型
df = df.withColumn("height", df["height"].cast(IntegerType()))
```

##### 9. collect

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119220742676.png)

```python
print(data1[0]["words"])

df.collect() #Row列表形式返回所有记录。
# 在只有一列的情况下可以用 [0] 来获取值
# 获取一列的所有值，或者多列的所有值
# collect()函数将分布式的dataframe转成local类型的 list-row格式
rows= df.select('col_1', 'col_2').collect()
value = [[ row.col_1, row.col_2 ] for row in rows ]
#获取第一行的多个值，返回普通python变量
# first() 返回的是 Row 类型，可以看做是dict类型，用 row.col_name 来获取值
row = df.select('col_1', 'col_2').first()
col_1_value = row.col_1
col_2_value = row.col_2
```

##### 10. RDD &DF&Pandas

```python
rdd_df = df.rdd	  # DF转RDD
df = rdd_df.toDF()  # RDD转DF
pandas_df = spark_df.toPandas()	
spark_df = sqlContext.createDataFrame(pandas_df)
```

##### 11. UDF

```python
#udf(f=None, returnType=StringType)
#Parameters：
#  f – python函数（如果用作独立函数）
#  returnType – 用户定义函数的返回类型。
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, ArrayType
stopwords = [k.strip() for k in open('./data/stopwords.txt', encoding='utf-8') if k.strip() != '']
def clearTxt(line):
    if line != '':
        line = line.strip()
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]","",line)
         
        #去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
        return line
    else:
        return 'Empyt Line'
cutWords_list=[]
def cutwordshandle(jobinfo):
    jobinfo=clearTxt(jobinfo)
    cutWords = [k for k in jieba.cut(jobinfo,cut_all=False) if k not in stopwords]
    #print(len(cutWords_list))
    cutWords_list.append(cutWords)
    return cutWords

cutwords = udf(lambda z: cutwordshandle(z), ArrayType(StringType()))
sqlcontext.udf.register("label", cutwords)
df3 = df2.withColumn( 'words',cutwords('待遇'))
```



```python
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from pyspark.sql import Row
csv_data = raw_data.map(lambda l: l.split(","))
row_data = csv_data.map(lambda p: Row(
    duration=int(p[0]), 
    protocol_type=p[1],
    service=p[2],
    flag=p[3],
    src_bytes=int(p[4]),
    dst_bytes=int(p[5])
    )
)

interactions_df = sqlContext.createDataFrame(row_data)
interactions_df.registerTempTable("interactions")
# Select tcp network interactions with more than 1 second duration and no transfer from destination
tcp_interactions = sqlContext.sql("""
    SELECT duration, dst_bytes FROM interactions WHERE protocol_type = 'tcp' AND duration > 1000 AND dst_bytes = 0
""")
tcp_interactions.show()
# Output duration together with dst_bytes
tcp_interactions_out = tcp_interactions.map(lambda p: "Duration: {}, Dest. bytes: {}".format(p.duration, p.dst_bytes))
for ti_out in tcp_interactions_out.collect():
  print ti_out
interactions_df.printSchema()  # printdata schema
#---   dataframe query
interactions_df.select("protocol_type", "duration", "dst_bytes").filter(interactions_df.duration>1000).filter(interactions_df.dst_bytes==0).groupBy("protocol_type").count().show()
```

##### 11.1. Pandas UDF

```python
from pyspark.sql.functions import udf
# 使用 udf 定义一个 row-at-a-time 的 udf
@udf('double')
# 输入/输出都是单个 double 类型的值
def plus_one(v):
    return v + 1
df.withColumn('v2', plus_one(df.v))

from pyspark.sql.functions import pandas_udf, PandasUDFType
# 使用 pandas_udf 定义一个 Pandas UDF
@pandas_udf('double', PandasUDFType.SCALAR)
# 输入/输出都是 double 类型的 pandas.Series
def pandas_plus_one(v):
    return v + 1
df.withColumn('v2', pandas_plus_one(df.v))

#最小二乘法举例
import statsmodels.api as sm
# df has four columns: id, y, x1, x2
group_column = 'id'
y_column = 'y'
x_columns = ['x1', 'x2']
schema = df.select(group_column, *x_columns).schema
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
      X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return pd.DataFrame([[group_key] + [model.params[i] for i in   x_columns]], columns=[group_column] + x_columns)
beta = df.groupby(group_column).apply(ols)
```



##### 12. schema

```python
from pyspark.sql.types import StructField, MapType, StringType, IntegerType, StructType
# 常用的还包括 DateType 等

people_schema= StructType([
    StructField('address', MapType(StringType(), StringType()), True),
    StructField('age', LongType(), True),
    StructField('name', StringType(), True),
])

df = spark.read.json('people.json', schema=people_schema)

df.show(1)
"""
+--------------------+---+----+
|             address|age|name|
+--------------------+---+----+
|[country -> China...| 12|  Li|
+--------------------+---+----+
only showing top 1 row
"""

df.dtypes
# [('address', 'map<string,string>'), ('age', 'bigint'), ('name', 'string')]
```

### 1.3.  RDD&DataFrame&Dataset

| **Basis of Difference**               | **Spark RDD**                                            | **Spark DataFrame**                                          | **Spark Dataset**                                            |
| ------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **What is it?**                       | A low-level API                                          | A high-level abstraction                                     | A `combination` of both RDDs and DataFrames                  |
| **Input Optimization Engine**         | `Cannot make use of input optimization engines`          | Uses input optimization engines to generate logical queries  | Uses Catalyst Optimizer for input optimization, as DataFrames do |
| **Data Representation**               | `Distributed across multiple nodes of a cluster`         | `A collection of rows and named columns`                     | An extension of DataFrames, providing the functionalities of both RDDs and DataFrames |
| **Benefit**                           | A simple API                                             | Gives a schema for the distributed data                      | `Improves memory usage`                                      |
| **Immutability and Interoperability** | Tracks data lineage information to recover the lost data | Once transformed into a DataFrame, not possible to get the domain object | Can regenerate RDDs                                          |
| **Performance Limitation**            | Java Serialization and Garbage Collection overheads      | Offers huge performance improvement over RDDs                | Operations are performed on serialized data to improve performance |

### 1.4. Summary statistics

```python
from pyspark.mllib.stat import Statistics 
from math import sqrt 

# Compute column summary statistics.
summary = Statistics.colStats(vector_data)

print "Duration Statistics:"
print " Mean: {}".format(round(summary.mean()[0],3))
print " St. deviation: {}".format(round(sqrt(summary.variance()[0]),3))
print " Max value: {}".format(round(summary.max()[0],3))
print " Min value: {}".format(round(summary.min()[0],3))
print " Total value count: {}".format(summary.count())
print " Number of non-zero values: {}".format(summary.numNonzeros()[0])
```

### 1. 4. ML

- **mllib.classification**: The spark.mllib package offers support for various methods to perform binary classification, regression analysis, and multiclass classification. Some of the most used algorithms in classifications are Naive Bayes, decision trees, etc.
- **mllib.clustering**: In clustering, you can perform the grouping of subsets of entities on the basis of some similarities in the elements or entities.
- **mllib.linalg**: This algorithm offers MLlib utilities to support linear algebra.
- **mllib.recommendation**: This algorithm is used for recommender systems to fill in the missing entries in any dataset.
- **spark.mllib**: This supports collaborative filtering, where Spark uses ALS (Alternating Least Squares) to predict the missing entries in the sets of descriptions of users and products.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119215234237.png)

- [ ] https://spark.apache.org/docs/latest/ml-features  后期学习使用这里面的api，但是ML基本算法必须熟练掌握；
- [ ] 后面遇到比较好的代码，案例可以多多积累；

```python
import urllib
f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "kddcup.data.gz")
data_file = "./kddcup.data.gz"
raw_data = sc.textFile(data_file)
print "Train data size is {}".format(raw_data.count())
ft = urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz", "corrected.gz")
test_data_file = "./corrected.gz"
test_raw_data = sc.textFile(test_data_file)
print "Test data size is {}".format(test_raw_data.count())

from pyspark.mllib.regression import LabeledPoint
from numpy import array
csv_data = raw_data.map(lambda x: x.split(","))
test_csv_data = test_raw_data.map(lambda x: x.split(","))
protocols = csv_data.map(lambda x: x[1]).distinct().collect()
services = csv_data.map(lambda x: x[2]).distinct().collect()
flags = csv_data.map(lambda x: x[3]).distinct().collect()
def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[0:41]
    # convert protocol to numeric categorical variable
    try: 
        clean_line_split[1] = protocols.index(clean_line_split[1])
    except:
        clean_line_split[1] = len(protocols)  
    # convert service to numeric categorical variable
    try:
        clean_line_split[2] = services.index(clean_line_split[2])
    except:
        clean_line_split[2] = len(services)
    # convert flag to numeric categorical variable
    try:
        clean_line_split[3] = flags.index(clean_line_split[3])
    except:
        clean_line_split[3] = len(flags)
    # convert label to binary label
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, array([float(x) for x in ]))
training_data = csv_data.map(create_labeled_point)
test_data = test_csv_data.map(create_labeled_point)
#--- training the classifier
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from time import time

# Build the model
t0 = time()
tree_model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)}, impurity='gini', maxDepth=4, maxBins=100)
tt = time() - t0
print "Classifier trained in {} seconds".format(round(tt,3))

#--- predict
predictions = tree_model.predict(test_data.map(lambda p: p.features))
labels_and_preds = test_data.map(lambda p: p.label).zip(predictions)

t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
tt = time() - t0
print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))

print "Learned classification tree model:"
print tree_model.toDebugString()
```

![](https://intellipaat.com/mediaFiles/2019/03/spark-and-rdd-cheat-sheet-1.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119212743127.png)

- https://github.com/jadianes/spark-py-notebooks
- https://www.cnblogs.com/sight-tech/p/12990579.html
- https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
- https://intellipaat.com/mediaFiles/2019/03/PySpark-SQL-cheat-sheet.pdf
- https://www.cnblogs.com/liaowuhen1314/p/12792202.html
- https://spark.apache.org/docs/2.2.0/sql-programming-guide.html



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pyspark/  

