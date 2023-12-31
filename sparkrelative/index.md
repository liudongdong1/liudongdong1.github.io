# sparkRelative


#### 1. 部署方式

##### 1.1. Local 模式

- **Local模式**就是运行在一台计算机上的模式，通常用于在本机上测试，当不设置master参数的值时，默认此模式，具体有以下几种设置master的方式。
  1. local：所有计算都运行在一个线程当中，没有任何并行计算。
  2. local[n]：指定使用n个线程来运行计算。
  3. local[*]：按照CPU的最多核数来设置线程数。

##### 1.2.  Standalone 模式

- **Standalone**：独立模式，Spark原生的简单集群管理器，自带完整的服务，可单独部署到一个集群中，无需依赖任何其他资源管理系统，使用Standalone可以很方便地搭建一个集群；

> Standalone集群有四个重要运行机制:
>
> - Master: 是一个进程，主要负责资源的调度和分配，并进行集群的监控等职责
> - Worker: 是一个进程，可以启动其他的进程和线程(Executor)；同时用自己的内存存储RDD的某些partition
> - Driver：是一个进程，我们编写的Spark应用程序就运行在Driver上
> - Executor: 是一个进程，一个Worker可以运行多个Executor,Executor通过启动多个线程(task)来执行对RDD的partition进行并行计算。

![standalone-client](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210219095555023.png)

```shell
# Run on a Spark standalone cluster in client deploy mode
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master spark://207.184.161.138:7077 \
  --deploy-mode client \
  --executor-memory 20G \
--total-executor-cores 100 \
/path/to/examples.jar \
1000
```

![standalone-cluster](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210219095555023.png)

```shell
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
```

##### 1.3. Apache Mesos模式

- **Apache Mesos**：一个强大的分布式资源管理框架，它允许多种不同的框架部署在其上，包括yarn；

##### 1.4. Hadoop Yarn模式

- **Hadoop YARN**：统一的资源管理机制，在上面可以运行多套计算框架，如map reduce、storm等，根据driver在集群中的位置不同，分为yarn client和yarn cluster。
- 俩种模式区别：

> 　①. 在于driver端启动在本地(client)，还是在Yarn集群内部的AM中(cluster)。
>
> 　②. client提交作业的进程是不能停止的，否则作业就挂了；cluster提交作业后就断开了，因为driver运行在AM中。
>
> 　③. client提交的作业，日志在客户端看不到，因为作业运行在yarn上，可以通过 yarn logs -applicationId <application_id> 查看。
>
> 　④. Cluster适合生产环境，Client适合交互和调试。

![Client](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210219101359930.png)

![cluster](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210219101430442.png)

| Master URL（主节点参数） | Meaning（含义）                                           |
| ------------------------ | --------------------------------------------------------- |
| local                    | 在本地运行，只有一个工作进程，无并行计算能力。            |
| local[K]                 | 在本地运行，有K个工作进程，通常设置K为机器的CPU核心数量。 |
| local[*]                 | 在本地运行，工作进程数量等于机器的CPU核心数量。           |

| Master URL（主节点参数） | Meaning（含义）                                              |
| ------------------------ | ------------------------------------------------------------ |
| spark://HOST:PORT        | 以Standalone模式运行，这是Spark自身提供的集群运行模式， 默认端口号: 7077。 |
| mesos://HOST:PORT        | 在Mesos集群上运行，Driver进程和Worker进程运行在Mesos集群上， 部署模式必须使用固定值:–deploy-mode cluster |

| Master URL（主节点参数） | Meaning（含义）                                              |
| ------------------------ | ------------------------------------------------------------ |
| yarn-client              | 在Yarn集群上运行，Driver进程在本地，Executor进程在Yarn集群上， 部署模式必须使用固定值:–deploy-mode client。 Yarn集群地址必须在HADOOP_CONF_DIR or YARN_CONF_DIR变量里定义。 |
| yarn-cluster             | 在Yarn集群上运行，Driver进程在Yarn集群上，Work进程也在Yarn集群上， 部署模式必须使用固定值:–deploy-mode cluster。 Yarn集群地址必须在HADOOP_CONF_DIR or YARN_CONF_DIR变量里定义。 |

#### 2. Demo

##### 2.1. 文本处理

- [将旧金山犯罪记录（San Francisco Crime Description）分类到33个类目中](https://cloud.tencent.com/developer/article/1096712)

```python
import time
from pyspark.sql import SQLContext
from pyspark import SparkContext
# 利用spark的csv库直接载入csv格式的数据
sc = SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true',                                                             inferschema='true').load('train.csv')
# 选10000条数据集，减少运行时间
data = data.sample(False, 0.01, 100)
print(data.count())
# 除去一些不要的列，并展示前五行
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
data = data.select([column for column in data.columns if column not in drop_list])
data.show(5)
data.printSchema()

# 包含数量最多的20类犯罪
from pyspark.sql.functions import col
data.groupBy('Category').count().orderBy(col('count').desc()).show()

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
# 正则切分单词
# inputCol:输入字段名
# outputCol:输出字段名
regexTokenizer = RegexTokenizer(inputCol='Descript', outputCol='words', pattern='\\W')
# 停用词
add_stopwords = ['http', 'https', 'amp', 'rt', 't', 'c', 'the']
stopwords_remover = StopWordsRemover(inputCol='words', outputCol='filtered').setStopWords(add_stopwords)
# 构建词频向量
count_vectors = CountVectorizer(inputCol='filtered', outputCol='features', vocabSize=10000, minDF=5)

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
label_stringIdx = StringIndexer(inputCol='Category', outputCol='label')
pipeline = Pipeline(stages=[regexTokenizer, stopwords_remover, count_vectors, label_stringIdx])
# fit the pipeline to training documents
pipeline_fit = pipeline.fit(data)
dataset = pipeline_fit.transform(data)
dataset.show(5)

# set seed for reproducibility
# 数据集划分训练集和测试集，比例7:3， 设置随机种子100
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print('Training Dataset Count:{}'.format(trainingData.count()))
print('Test Dataset Count:{}'.format(testData.count()))

start_time = time.time()
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
# 过滤prediction类别为0数据集
predictions.filter(predictions['prediction'] == 0).select('Descript', 'Category', 'probability', 'label', 'prediction').orderBy('probability', accending=False).show(n=10, truncate=30)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# predictionCol: 预测列的名称
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
# 预测准确率
print(evaluator.evaluate(predictions))
end_time = time.time()
print(end_time - start_time)

#以TF-ID作为特征，利用逻辑回归进行分类
from pyspark.ml.feature import HashingTF, IDF
start_time = time.time()
# numFeatures: 最大特征数
hashingTF = HashingTF(inputCol='filtered', outputCol='rawFeatures', numFeatures=10000)
# minDocFreq：过滤的最少文档数量
idf = IDF(inputCol='rawFeatures', outputCol='features', minDocFreq=5)
pipeline = Pipeline(stages=[regexTokenizer, stopwords_remover, hashingTF, idf, label_stringIdx])
pipeline_fit = pipeline.fit(data)
dataset = pipeline_fit.transform(data)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lr_model = lr.fit(trainingData)
predictions = lr_model.transform(testData)
predictions.filter(predictions['prediction'] == 0).select('Descript', 'Category', 'probability', 'label', 'prediction').\
orderBy('probability', ascending=False).show(n=10, truncate=30)

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
print(evaluator.evaluate(predictions))
end_time = time.time()
print(end_time - start_time)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
start_time = time.time()
pipeline = Pipeline(stages=[regexTokenizer, stopwords_remover, count_vectors, label_stringIdx])
pipeline_fit = pipeline.fit(data)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
# 为交叉验证创建参数
# ParamGridBuilder：用于基于网格搜索的模型选择的参数网格的生成器
# addGrid：将网格中给定参数设置为固定值
# parameter：正则化参数
# maxIter：迭代次数
# numFeatures：特征值
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5])
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2])
             .addGrid(lr.maxIter, [10, 20, 50])
#              .addGrid(idf.numFeatures, [10, 100, 1000])
             .build())

# 创建五折交叉验证
# estimator：要交叉验证的估计器
# estimatorParamMaps：网格搜索的最优参数
# evaluator：评估器
# numFolds：交叉次数
cv = CrossValidator(estimator=lr,\
                   estimatorParamMaps=paramGrid,\
                   evaluator=evaluator,\
                   numFolds=5)
cv_model = cv.fit(trainingData)
predictions = cv_model.transform(testData)

# 模型评估
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
print(evaluator.evaluate(predictions))
end_time = time.time()
print(end_time - start_time)

#朴素贝叶斯
from pyspark.ml.classification import NaiveBayes
start_time = time.time()
# smoothing：平滑参数
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)
predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select('Descript', 'Category', 'probability', 'label', 'prediction') \
    .orderBy('probability', ascending=False) \
    .show(n=10, truncate=30)
    
#随机森林
from pyspark.ml.classification import RandomForestClassifier
start_time = time.time()
# numTree：训练树的个数
# maxDepth：最大深度
# maxBins：连续特征离散化的最大分类数
rf = RandomForestClassifier(labelCol='label', \
                            featuresCol='features', \
                            numTrees=100, \
                            maxDepth=4, \
                            maxBins=32)
# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select('Descript','Category','probability','label','prediction') \
    .orderBy('probability', ascending=False) \
    .show(n = 10, truncate = 30)
    
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
print(evaluator.evaluate(predictions))
end_time = time.time()
print(end_time - start_time)
```

- 评语分类

```python
#coding:utf-8
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF,IDF,StandardScaler
from pyspark.mllib.classification import LogisticRegressionWithSGD,SVMWithSGD,NaiveBayes
from pyspark.mllib.tree import DecisionTree


def split2(line):
    '''自定义的字符串分割函数，本例已步长3分割字符串'''
    step = 3
    result = []
    length = len(line)
    for i in xrange(0,length,step):
        result.append(line[i:i+step])
    return result

def check(test,model):
    '''模型检测函数，输出模型的正确率、准确率和召回率，本例省略'''
    

if __name__=="__main__":
    sc = SparkContext(appName="test")
    #分别读取正文件和负文件的训练集，然后读取测试集
    spam = sc.textFile("hdfs://ubuntu:9000/xxx/bad.txt")
    normal = sc.textFile("hdfs://ubuntu:9000/xxx/good.txt")
    test = sc.textFile("hdfs://ubuntu:9000/xxx/test.txt")
    # 创建一个HashingTF实例来把文本映射为包含10000个特征的向量
    tf = HashingTF(numFeatures = 10000)
    # 各http请求都被切分为单词，每个单词被映射为一个特征
    spamFeatures = spam.map(lambda line: tf.transform(split2(line)))
    normalFeatures = normal.map(lambda line: tf.transform(split2(line)))

    # =========使用词频统计构建向量=========
    # positiveExamples = spamFeatures.map(lambda features: LabeledPoint(0, features))
    # negativeExamples = normalFeatures.map(lambda features: LabeledPoint(1, features))
    # trainingData = positiveExamples.union(negativeExamples)
    # trainingData.cache() # 因为逻辑回归是迭代算法，所以缓存训练数据RDD
    #print trainingData.take(1)
    
    # =========使用TF-IDF构建向量=========
    spamFeatures.cache() # 因为逻辑回归是迭代算法，所以缓存训练数据RDD
    idf = IDF()
    idfModel = idf.fit(spamFeatures)
    spamVectors = idfModel.transform(spamFeatures)
    normalFeatures.cache() # 因为逻辑回归是迭代算法，所以缓存训练数据RDD
    idfModel = idf.fit(normalFeatures)
    normalVectors = idfModel.transform(normalFeatures)
    positiveExamples = normalVectors.map(lambda features: LabeledPoint(1, features))
    negativeExamples = spamVectors.map(lambda features: LabeledPoint(0, features))
    dataAll = positiveExamples.union(negativeExamples)

    # =========特征向量压缩=========
    # scaler = StandardScaler(withMean=True, withStd=True)
    # normalScaler = scaler.fit(normalVectors)
    # normalResult = normalScaler.transform(normalVectors)
    # spamScaler = scaler.fit(spamVectors)
    # spamResult = spamScaler.transform(spamVectors)
    
    # 使用分类算法进行训练,iterations位迭代次数,step为迭代步长
    # LogisticRegressionWithSGD可以替换为SVMWithSGD和NaiveBayes
    # 其中train函数的参数可以根据模型效果自定义
    model = LogisticRegressionWithSGD.train(data=dataAll,iterations=10000,step=1) 
    # 决策树的分类类别为2，映射表为空，不纯净度测量为gini，树的深度为5，数据箱子为32
    # model = DecisionTree.trainClassifier(dataAll, numClasses=2, categoricalFeaturesInfo={},
 #                                     impurity='gini', maxDepth=5, maxBins=32) 
    check(test,model)
    sc.stop()
```

##### 2.2. [蘑菇分类](https://www.kaggle.com/uciml/mushroom-classification)

```python
#分类毒蘑菇和可食用蘑菇，共22个特征值，其中特征描述都是字符，用于机器学习的话，要将特征转换成数值。
#https://blog.csdn.net/m0_37442062/article/details/91357264
import findspark #pip install findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
spark = SparkSession.builder.master('local[1]').appName('classification').getOrCreate()

# 载入数据
df0 = spark.read.csv('mushrooms.csv', header=True, inferSchema=True, encoding='utf-8')
# 查看是否有缺失值
# df0.toPandas().isna().sum()
df0.toPandas().isna().values.any()

#先使用StringIndexer将字符转化为数值，然后将特征整合到一起
from pyspark.ml.feature import StringIndexer, VectorAssembler
old_columns_names = df0.columns
new_columns_names = [name+'-new' for name in old_columns_names]
for i in range(len(old_columns_names)):
    indexer = StringIndexer(inputCol=old_columns_names[i], outputCol=new_columns_names[i])
    df0 = indexer.fit(df0).transform(df0)
vecAss = VectorAssembler(inputCols=new_columns_names[1:], outputCol='features')
df0 = vecAss.transform(df0)
# 更换label列名
df0 = df0.withColumnRenamed(new_columns_names[0], 'label')
# df0.show()
# 创建新的只有label和features的表
dfi = df0.select(['label', 'features'])
# 数据概观
dfi.show(5, truncate=0)
#构建训练数据
train_data, test_data = dfi.randomSplit([4.0, 1.0], 100)

from pyspark.ml.classification import LogisticRegression
blor = LogisticRegression(regParam=0.01)#设置regParam为0.01
blorModel = blor.fit(train_data)
result = blorModel.transform(test_data)
# 计算准确率
a = result.filter(result.label == result.prediction).count()/result.count()
print("逻辑回归： "+str(a))

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=5) #树的最大深度
dtModel = dt.fit(train_data)
result = dtModel.transform(test_data)
# accuracy
b = result.filter(result.label == result.prediction).count()/result.count()
print("决策树： "+str(b))

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxDepth=5)
gbtModel = gbt.fit(train_data)
result = gbtModel.transform(test_data)
# accuracy
c = result.filter(result.label == result.prediction).count()/result.count()
print("梯度增强树： "+str(c))

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=10, maxDepth=5)
rfModel = rf.fit(train_data)
result = rfModel.transform(test_data)
# accuracy
d = result.filter(result.label == result.prediction).count()/result.count()
# 1.0
print("随机森林： "+str(d))

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()
nbModel = nb.fit(train_data)
result = nbModel.transform(test_data)
#accuracy
e = result.filter(result.label == result.prediction).count()/result.count()
#0.9231714812538414
print("朴素贝叶斯： "+str(e))

from pyspark.ml.classification import LinearSVC
svm = LinearSVC(maxIter=10, regParam=0.01)
svmModel = svm.fit(train_data)
result = svmModel.transform(test_data)
# accuracy
f = result.filter(result.label == result.prediction).count()/result.count()
# 0.9797172710510141
print("支持向量机： "+str(f))
```

##### 2.3. 商品推荐

- ItemSimilarity

```python
# Item-Item Similarity computation on pySpark with cosine similarity

import sys
from itertools import combinations
import numpy as np

from pyspark import SparkContext


def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user) 
    '''
    for item1,item2 in combinations(items_with_rating,2):
        return (item1[0],item2[0]),(item1[1],item2[1])

def calcSim(item_pair,rating_pairs):
    ''' 
    For each item-item pair, return the specified similarity measure,
    along with co_raters_count
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))

    return item_pair, (cos_sim,n)


def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonUserCF")
    lines = sc.textFile(sys.argv[2])

    ''' 
    Obtain the sparse user-item matrix
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''
    user_item_pairs = lines.map(parseVector).groupByKey().cache()

    '''
    Get all item-item pair combos
        (item1,item2) ->    [(item1_rating,item2_rating),
                             (item1_rating,item2_rating),
                             ...]
    '''

    pairwise_items = user_item_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findItemPairs(p[0],p[1])).groupByKey()

    '''
    Calculate the cosine similarity for each item pair
        (item1,item2) ->    (similarity,co_raters_count)
    '''

    item_sims = pairwise_items.map(
        lambda p: calcSim(p[0],p[1])).collect()
```

- UserSimilarity

```python
# User-User Similarity computation on pySpark

import sys
from itertools import combinations
import numpy as np
import pdb

from pyspark import SparkContext


def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split("|")
    return line[1],(line[0],float(line[2]))

def keyOnUserPair(item_id,user_and_rating_pair):
    ''' 
    Convert each item and co_rating user pairs to a new vector
    keyed on the user pair ids, with the co_ratings as their value. 
    '''
    (user1_with_rating,user2_with_rating) = user_and_rating_pair
    user1_id,user2_id = user1_with_rating[0],user2_with_rating[0]
    user1_rating,user2_rating = user1_with_rating[1],user2_with_rating[1]
    return (user1_id,user2_id),(user1_rating,user2_rating)

def calcSim(user_pair,rating_pairs):
    ''' 
    For each user-user pair, return the specified similarity measure,
    along with co_raters_count.
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return user_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonUserCF")
    lines = sc.textFile(sys.argv[2])

    ''' 
    Parse the vector with item_id as the key:
        item_id -> (user_id,rating)
    '''
    item_user = lines.map(parseVector).cache()

    '''
    Get co_rating users by joining on item_id:
        item_id -> ((user_1,rating),(user2,rating))
    '''
    item_user_pairs = item_user.join(item_user)

    '''
    Key each item_user_pair on the user_pair and get rid of non-unique 
    user pairs, then aggregate all co-rating pairs:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
    '''
    user_item_rating_pairs = item_user_pairs.map(
        lambda p: keyOnUserPair(p[0],p[1])).filter(
        lambda p: p[0][0] != p[0][1]).groupByKey()

    '''
    Calculate the cosine similarity for each user pair:
        (user1,user2) ->    (similarity,co_raters_count)
    '''
    user_pair_sims = user_item_rating_pairs.map(
        lambda p: calcSim(p[0],p[1]))

    for p in user_pair_sims.collect():
        print p
```

- ItemBasedRecommender

```python
# Item-based Collaborative Filtering on pySpark with cosine similarity and weighted sums

import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
import pdb

from pyspark import SparkContext
from recsys.evaluation.prediction import MAE

def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def sampleInteractions(user_id,items_with_rating,n):
    '''
    For users with # interactions > n, replace their interaction history
    with a sample of n items_with_rating
    '''
    if len(items_with_rating) > n:
        return user_id, random.sample(items_with_rating,n)
    else:
        return user_id, items_with_rating

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user) 
    '''
    for item1,item2 in combinations(items_with_rating,2):
        return (item1[0],item2[0]),(item1[1],item2[1])

def calcSim(item_pair,rating_pairs):
    ''' 
    For each item-item pair, return the specified similarity measure,
    along with co_raters_count
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return item_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared
    return (numerator / (float(denominator))) if denominator else 0.0

def correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared):
    '''
    The correlation between two vectors A, B is
      [n * dotProduct(A, B) - sum(A) * sum(B)] /
        sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }

    '''
    numerator = size * dot_product - rating_sum * rating2sum
    denominator = sqrt(size * rating_norm_squared - rating_sum * rating_sum) * \
                    sqrt(size * rating2_norm_squared - rating2sum * rating2sum)

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstItem(item_pair,item_sim_data):
    '''
    For each item-item pair, make the first item's id the key
    '''
    (item1_id,item2_id) = item_pair
    return item1_id,(item2_id,item_sim_data)

def nearestNeighbors(item_id,items_and_sims,n):
    '''
    Sort the predictions list by similarity and select the top-N neighbors
    '''
    items_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return item_id, items_and_sims[:n]

def topNRecommendations(user_id,items_with_rating,item_sims,n):
    '''
    Calculate the top-N item recommendations for each user using the 
    weighted sums method
    '''

    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (item,rating) in items_with_rating:

        # lookup the nearest neighbors for this item
        nearest_neighbors = item_sims.get(item,None)

        if nearest_neighbors:
            for (neighbor,(sim,count)) in nearest_neighbors:
                if neighbor != item:

                    # update totals and sim_sums with the rating data
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # create the normalized list of scored items 
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # sort the scored items in ascending order
    scored_items.sort(reverse=True)

    # take out the item score
    # ranked_items = [x[1] for x in scored_items]

    return user_id,scored_items[:n]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonUserCF")
    lines = sc.textFile(sys.argv[2])

    ''' 
    Obtain the sparse user-item matrix:
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''
    user_item_pairs = lines.map(parseVector).groupByKey().map(
        lambda p: sampleInteractions(p[0],p[1],500)).cache()

    '''
    Get all item-item pair combos:
        (item1,item2) ->    [(item1_rating,item2_rating),
                             (item1_rating,item2_rating),
                             ...]
    '''

    pairwise_items = user_item_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findItemPairs(p[0],p[1])).groupByKey()

    '''
    Calculate the cosine similarity for each item pair and select the top-N nearest neighbors:
        (item1,item2) ->    (similarity,co_raters_count)
    '''

    item_sims = pairwise_items.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstItem(p[0],p[1])).groupByKey().map(
        lambda p : (p[0], list(p[1]))).map(
        lambda p: nearestNeighbors(p[0],p[1],50)).collect()

    '''
    Preprocess the item similarity matrix into a dictionary and store it as a broadcast variable:
    '''

    item_sim_dict = {}
    for (item,data) in item_sims: 
        item_sim_dict[item] = data

    isb = sc.broadcast(item_sim_dict)

    '''
    Calculate the top-N item recommendations for each user
        user_id -> [item1,item2,item3,...]
    '''
    user_item_recs = user_item_pairs.map(
        lambda p: topNRecommendations(p[0],p[1],isb.value,500)).collect()

    '''
    Read in test data and calculate MAE
    '''

    test_ratings = defaultdict(list)

    # read in the test data
    f = open("tests/data/cftest.txt", 'rt')
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        user = row[0]
        item = row[1]
        rating = row[2]
        test_ratings[user] += [(item,rating)]

    # create train-test rating tuples
    preds = []
    for (user,items_with_rating) in user_item_recs:
        for (rating,item) in items_with_rating:
            for (test_item,test_rating) in test_ratings[user]:                
                if str(test_item) == str(item):
                    preds.append((rating,float(test_rating)))

    mae = MAE(preds)
    result = mae.compute()
    print "Mean Absolute Error: ",result

```

- userbasedRemcommender

```python
# User-based Collaborative Filtering on pySpark with cosine similarity and weighted sums

import sys
from collections import defaultdict
from itertools import combinations
import random
import numpy as np
import pdb

from pyspark import SparkContext


def parseVectorOnUser(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Key is user_id, converts each rating to a float.
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def parseVectorOnItem(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Key is item_id, converts each rating to a float.
    '''
    line = line.split("|")
    return line[1],(line[0],float(line[2]))

def sampleInteractions(item_id,users_with_rating,n):
    '''
    For items with # interactions > n, replace their interaction history
    with a sample of n users_with_rating
    '''
    if len(users_with_rating) > n:
        return item_id, random.sample(users_with_rating,n)
    else:
        return item_id, users_with_rating

def findUserPairs(item_id,users_with_rating):
    '''
    For each item, find all user-user pairs combos. (i.e. users with the same item) 
    '''
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

def calcSim(user_pair,rating_pairs):
    ''' 
    For each user-user pair, return the specified similarity measure,
    along with co_raters_count.
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return user_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstUser(user_pair,item_sim_data):
    '''
    For each user-user pair, make the first user's id the key
    '''
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,item_sim_data)

def nearestNeighbors(user,users_and_sims,n):
    '''
    Sort the predictions list by similarity and select the top-N neighbors
    '''
    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return user, users_and_sims[:n]

def topNRecommendations(user_id,user_sims,users_with_rating,n):
    '''
    Calculate the top-N item recommendations for each user using the 
    weighted sums method
    '''

    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (neighbor,(sim,count)) in user_sims:

        # lookup the item predictions for this neighbor
        unscored_items = users_with_rating.get(neighbor,None)

        if unscored_items:
            for (item,rating) in unscored_items:
                if neighbor != item:

                    # update totals and sim_sums with the rating data
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # create the normalized list of scored items 
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # sort the scored items in ascending order
    scored_items.sort(reverse=True)

    # take out the item score
    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonUserCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1],"PythonUserItemCF")
    lines = sc.textFile(sys.argv[2])

    '''
    Obtain the sparse item-user matrix:
        item_id -> ((user_1,rating),(user2,rating))
    '''
    item_user_pairs = lines.map(parseVectorOnItem).groupByKey().map(
        lambda p: sampleInteractions(p[0],p[1],500)).cache()

    '''
    Get all item-item pair combos:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
    '''
    pairwise_users = item_user_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findUserPairs(p[0],p[1])).groupByKey()

    '''
    Calculate the cosine similarity for each user pair and select the top-N nearest neighbors:
        (user1,user2) ->    (similarity,co_raters_count)
    '''
    user_sims = pairwise_users.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstUser(p[0],p[1])).groupByKey().map(
        lambda p: nearestNeighbors(p[0],p[1],50))

    ''' 
    Obtain the the item history for each user and store it as a broadcast variable
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''

    user_item_hist = lines.map(parseVectorOnUser).groupByKey().collect()

    ui_dict = {}
    for (user,items) in user_item_hist: 
        ui_dict[user] = items

    uib = sc.broadcast(ui_dict)

    '''
    Calculate the top-N item recommendations for each user
        user_id -> [item1,item2,item3,...]
    '''
    user_item_recs = user_sims.map(
        lambda p: topNRecommendations(p[0],p[1],uib.value,100)).collect()
```

##### 2.4. [图像分类](http://172.26.85.202:65501/notebooks/Transfer-Learning-PySpark-master/PySpark-Digit-Image-MultiClass.ipynb)

```python
#拼接展示图片数据
import IPython.display as dp
# collect all .png files in ssample dir
fs = !ls sample/*.png
# create list of image objects
images = []
for ea in fs:
    images.append(dp.Image(filename=ea, format='png'))
# display all images
for ea in images:
    dp.display_png(ea)

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

spark = SparkSession.builder.\
            config("spark.executor.memory", "1g").\
            config("spark.driver.memory", "4g").\
            config("spark.cores.max", "2").\
            appName('SparkImageClassifier').getOrCreate()
#加载图片数据
import sys, glob, os
sys.path.extend(glob.glob(os.path.join(os.path.expanduser("~"), ".ivy2/jars/*.jar")))

from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit

# loaded image
zero_df = ImageSchema.readImages("images/0").withColumn("label", lit(0))
one_df = ImageSchema.readImages("images/1").withColumn("label", lit(1))
two_df = ImageSchema.readImages("images/2").withColumn("label", lit(2))
three_df = ImageSchema.readImages("images/3").withColumn("label", lit(3))
four_df = ImageSchema.readImages("images/4").withColumn("label", lit(4))
five_df = ImageSchema.readImages("images/5").withColumn("label", lit(5))
six_df = ImageSchema.readImages("images/6").withColumn("label", lit(6))
seven_df = ImageSchema.readImages("images/7").withColumn("label", lit(7))
eight_df = ImageSchema.readImages("images/8").withColumn("label", lit(8))
nine_df = ImageSchema.readImages("images/9").withColumn("label", lit(9))


# merge data frame
from functools import reduce
dataframes = [zero_df, one_df, two_df, three_df, 
              four_df,five_df,six_df,seven_df,eight_df,nine_df]

df = reduce(lambda first, second: first.union(second), dataframes)
# repartition dataframe 
df = df.repartition(200)

# On hot encoding 
from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["label"],outputCols=["one_hot_label"])
model = encoder.fit(df)
df = model.transform(df)

# split the data-frame
train, test = df.randomSplit([0.8, 0.2], 42)
train.printSchema()  #image, label
test.printSchema()

%%time
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer 
# model: InceptionV3
# extracting feature from images
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features",
                                 modelName="InceptionV3")
# used as a multi class classifier
lr = LogisticRegression(maxIter=5, regParam=0.03, 
                        elasticNetParam=0.5, labelCol="label") 
# define a pipeline model
sparkdn = Pipeline(stages=[featurizer, lr])
spark_model = sparkdn.fit(train)

#Evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# evaluate the model with test set
evaluator = MulticlassClassificationEvaluator() 
transform_test = spark_model.transform(test)
print('F1-Score ', evaluator.evaluate(transform_test, 
                                      {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(transform_test,
                                       {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(transform_test, 
                                    {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(transform_test, 
                                      {evaluator.metricName: 'accuracy'}))

#Confusion Matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
- Convert Spark-DataFrame to Pnadas-DataFrame
- Call Confusion Matrix With 'True' and 'Predicted' Label
'''


from sklearn.metrics import confusion_matrix
y_true = transform_test.select("label")
y_true = y_true.toPandas() # convert to pandas dataframe from spark dataframe
y_pred = transform_test.select("prediction")
y_pred = y_pred.toPandas() # convert to pandas dataframe from spark dataframe
cnf_matrix = confusion_matrix(y_true, y_pred,labels=range(10))
'''
- Visualize the 'Confusion Matrix' 
'''
import seaborn as sns
%matplotlib inline

sns.set_style("darkgrid")
plt.figure(figsize=(7,7))
plt.grid(False)

# call pre defined function
plot_confusion_matrix(cnf_matrix, classes=range(10)) 

'''
- Classification Report of each class group
'''

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(y_true, y_pred, target_names = target_names))

'''
- A custom ROC AUC score function for multi-class classification problem
'''

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


print('ROC AUC score:', multiclass_roc_auc_score(y_true,y_pred))
```

#### 3. 文件读取

- 图片数据

```python
#方式一
from pyspark.ml.image import ImageSchema
image_df = ImageSchema.readImages("/data/myimages")
image_df.show()
#方式二
from keras.preprocessing import image
img = image.load_img("/data/myimages/daisy.jpg", target_size=(299, 299))
```

- TransferLearning

```python
#方式三
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

model = p.fit(train_images_df)    # train_images_df is a dataset of images and labels

# Inspect training error
df = model.transform(train_images_df.limit(10)).select("image", "probability",  "uri", "label")
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
```

- 分布式调参

```python
from keras.applications import InceptionV3
model = InceptionV3(weights="imagenet")
model.save('/tmp/model-full.h5')
import PIL.Image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from sparkdl.estimators.keras_image_file_estimator import KerasImageFileEstimator

def load_image_from_uri(local_uri):
  img = (PIL.Image.open(local_uri).convert('RGB').resize((299, 299), PIL.Image.ANTIALIAS))
  img_arr = np.array(img).astype(np.float32)
  img_tnsr = preprocess_input(img_arr[np.newaxis, :])
  return img_tnsr

estimator = KerasImageFileEstimator( inputCol="uri",
                                     outputCol="prediction",
                                     labelCol="one_hot_label",
                                     imageLoader=load_image_from_uri,
                                     kerasOptimizer='adam',
                                     kerasLoss='categorical_crossentropy',
                                     modelFile='/tmp/model-full-tmp.h5' # local file path for model
                                   )
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = (
  ParamGridBuilder()
  .addGrid(estimator.kerasFitParams, [{"batch_size": 32, "verbose": 0},
                                      {"batch_size": 64, "verbose": 0}])
  .build()
)
bc = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label" )
cv = CrossValidator(estimator=estimator, estimatorParamMaps=paramGrid, evaluator=bc, numFolds=2)

cvModel = cv.fit(train_df)
```

- deep learning models

```python
from pyspark.ml.image import ImageSchema
from sparkdl import DeepImagePredictor

image_df = ImageSchema.readImages(sample_img_dir)

predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels", modelName="InceptionV3", decodePredictions=True, topK=10)
predictions_df = predictor.transform(image_df)
```

- KerasImageFileTransformer

```python
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from pyspark.sql.types import StringType
from sparkdl import KerasImageFileTransformer

def loadAndPreprocessKerasInceptionV3(uri):
  # this is a typical way to load and prep images in keras
  image = img_to_array(load_img(uri, target_size=(299, 299)))  # image dimensions for InceptionV3
  image = np.expand_dims(image, axis=0)
  return preprocess_input(image)

transformer = KerasImageFileTransformer(inputCol="uri", outputCol="predictions",
                                        modelFile='/tmp/model-full-tmp.h5',  # local file path for model
                              imageLoader=loadAndPreprocessKerasInceptionV3,
                                        outputMode="vector")
files = [os.path.abspath(os.path.join(dirpath, f)) for f in os.listdir("/data/myimages") if f.endswith('.jpg')]
uri_df = sqlContext.createDataFrame(files, StringType()).toDF("uri")
keras_pred_df = transformer.transform(uri_df)
```

- kerasTransferm

```python
from sparkdl import KerasTransformer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Generate random input data
num_features = 10
num_examples = 100
input_data = [{"features" : np.random.randn(num_features).tolist()} for i in range(num_examples)]
input_df = sqlContext.createDataFrame(input_data)

# Create and save a single-hidden-layer Keras model for binary classification
# NOTE: In a typical workflow, we'd train the model before exporting it to disk,
# but we skip that step here for brevity
model = Sequential()
model.add(Dense(units=20, input_shape=[num_features], activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model_path = "/tmp/simple-binary-classification"
model.save(model_path)

# Create transformer and apply it to our input data
transformer = KerasTransformer(inputCol="features", outputCol="predictions", modelFile=model_path)
final_df = transformer.transform(input_df)
```

- keras model udf

```python
from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF
registerKerasImageUDF("inceptionV3_udf", InceptionV3(weights="imagenet"))
registerKerasImageUDF("my_custom_keras_model_udf", "/tmp/model-full-tmp.h5")

from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

def keras_load_img(fpath):
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    img = load_img(fpath, target_size=(299, 299))
    return img_to_array(img).astype(np.uint8)

registerKerasImageUDF("inceptionV3_udf_with_preprocessing", InceptionV3(weights="imagenet"), keras_load_img)

from pyspark.ml.image import ImageSchema
image_df = ImageSchema.readImages(sample_img_dir)
image_df.registerTempTable("sample_images")
SELECT my_custom_keras_model_udf(image) as predictions from sample_images
```

#### 4. DL Relative

##### 4.1. spark+pytorch

```python
cuda = False
 
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "2048")
import os
import shutil
import tarfile
import time
import zipfile
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader  # private API
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
input_local_dir = "/dbfs/ml/tmp/flower/"
output_file_path = "/tmp/predictions"
bc_model_state = sc.broadcast(models.resnet50(pretrained=True).state_dict())


def get_model_for_eval():
  """Gets the broadcasted model."""
  model = models.resnet50(pretrained=True)
  model.load_state_dict(bc_model_state.value)
  model.eval()
  return model



def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    print(file_path)
    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        file_path, _ = urlretrieve(url=url, filename=file_path)
        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")
maybe_download_and_extract(url=URL, download_dir=input_local_dir)
local_dir = input_local_dir + 'flower_photos/'
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(local_dir) for f in filenames if os.path.splitext(f)[1] == '.jpg']
len(files)

files_df = spark.createDataFrame(
  map(lambda path: (path,), files), ["path"]
).repartition(10)  # number of partitions should be a small multiple of total number of nodes
display(files_df.limit(10))

class ImageDataset(Dataset):
  def __init__(self, paths, transform=None):
    self.paths = paths
    self.transform = transform
  def __len__(self):
    return len(self.paths)
  def __getitem__(self, index):
    image = default_loader(self.paths[index])
    if self.transform is not None:
      image = self.transform(image)
    return image

def predict_batch(paths):
  transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
  ])
  images = ImageDataset(paths, transform=transform)
  loader = torch.utils.data.DataLoader(images, batch_size=500, num_workers=8)
  model = get_model_for_eval()
  model.to(device)
  all_predictions = []
  with torch.no_grad():
    for batch in loader:
      predictions = list(model(batch.to(device)).cpu().numpy())
      for prediction in predictions:
        all_predictions.append(prediction)
  return pd.Series(all_predictions)

predictions = predict_batch(pd.Series(files[:200]))  #本地测试

predict_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(predict_batch)


predictions_df = files_df.select(col('path'), predict_udf(col('path')).alias("prediction"))
predictions_df.write.mode("overwrite").parquet(output_file_path)
result_df = spark.read.load(output_file_path)
display(result_df)
```

##### 4.2. Submarine

> Submarine计算引擎通过命令行向YARN提交定制的深度学习应用程序（如 Tensorflow，Pytorch 等）。这些应用程序与YARN上的其他应用程序并行运行，例如Apache Spark，Hadoop Map / Reduce 等。
>
> - Submarine-Zeppelin integration：允许数据科学家在 Zeppelin 的notebook中编写算法和调参进行可视化输出，并直接从notebook提交和管理机器学习的训练工作。
> - Submarine-Azkaban integration：允许数据科学家从Zeppelin 的notebook中直接向Azkaban提交一组具有依赖关系的任务，组成工作流进行周期性调度。
> - Submarine-installer：在你的服务器环境中安装Submarine和 YARN，轻松解决Docker、Parallel network和nvidia驱动的安装部署难题，以便你更轻松地尝试强大的工具集。
> -  Apache Hadoop 3.1 的 YARN 可以完全无误的支持 Hadoop 2.7 + 以上的 HDFS 系统。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210227083551241.png)

##### 4.3.  SparkDL

- https://github.com/databricks/spark-deep-learning  sparkdl 支持spark版本如下：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210226212016988.png)

4.4. **[Distributed TensorFlow on Apache Spark 3.0](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-distributor.)**

```python
from spark_tensorflow_distributor import MirroredStrategyRunner

# Taken from https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-distributor#examples
def train():
    import tensorflow as tf
    import uuid

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def make_datasets():
        (mnist_images, mnist_labels), _ = \
            tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')

        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64))
        )
        dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return dataset

    def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'],
        )
        return model

    train_datasets = make_datasets()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)

MirroredStrategyRunner(num_slots=8).run(train)
```

##### 4.4. 

```python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from pyspark import SparkContext, SparkConf
import gzip
import cPickle
APP_NAME = "mnist"
MASTER_IP = 'local[24]'
# Define basic parameters
batch_size = 128
nb_classes = 10
nb_epoch = 5
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# Load data
f = gzip.open("./mnist.pkl.gz", "rb")
dd = cPickle.load(f)
(X_train, y_train), (X_test, y_test) = dd
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
## spark
conf = SparkConf().setAppName(APP_NAME).setMaster(MASTER_IP)
sc = SparkContext(conf=conf)
# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, Y_train)
# Initialize SparkModel from Keras model and Spark context
spark_model = SparkModel(sc,model)
# Train Spark model
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, validation_split=0.1, num_workers=24)
# Evaluate Spark model by evaluating the underlying model
score = spark_model.get_network().evaluate(X_test, Y_test, show_accuracy=True, verbose=2)
print('Test accuracy:', score[1])
```



#### 4. 相关学习资源

- https://github.com/jupyter-incubator/sparkmagic

##### 4.1. Infrastructure Projects

- [REST Job Server for Apache Spark](https://github.com/spark-jobserver/spark-jobserver) - REST interface for managing and submitting Spark jobs on the same cluster.
- [MLbase](http://mlbase.org/) - Machine Learning research project on top of Spark
- [Apache Mesos](https://mesos.apache.org/) - Cluster management system that supports running Spark
- [Alluxio](https://www.alluxio.org/) (née Tachyon) - Memory speed virtual distributed storage system that supports running Spark
- [FiloDB](https://github.com/filodb/FiloDB) - a Spark integrated analytical/columnar database, with in-memory option capable of sub-second concurrent queries
- [Zeppelin](http://zeppelin-project.org/) - Multi-purpose notebook which supports 20+ language backends, including Apache Spark
- [EclairJS](https://github.com/EclairJS/eclairjs-node) - enables Node.js developers to code against Spark, and data scientists to use Javascript in Jupyter notebooks.
- [Mist](https://github.com/Hydrospheredata/mist) - Serverless proxy for Spark cluster (spark middleware)
- [K8S Operator for Apache Spark](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator) - Kubernetes operator for specifying and managing the lifecycle of Apache Spark applications on Kubernetes.
- [IBM Spectrum Conductor](https://developer.ibm.com/storage/products/ibm-spectrum-conductor-spark/) - Cluster management software that integrates with Spark and modern computing frameworks.
- [Delta Lake](https://delta.io/) - Storage layer that provides ACID transactions and scalable metadata handling for Apache Spark workloads.
- [MLflow](https://mlflow.org/) - Open source platform to manage the machine learning lifecycle, including deploying models from diverse machine learning libraries on Apache Spark.
- [Koalas](https://github.com/databricks/koalas) - Data frame API on Apache Spark that more closely follows Python’s pandas.
- [Apache DataFu](https://datafu.apache.org/docs/spark/getting-started.html) - A collection of utils and user-defined-functions for working with large scale data in Apache Spark, as well as making Scala-Python interoperability easier.

##### 4.2. Applications Using Spark

- [Apache Mahout](https://mahout.apache.org/) - Previously on Hadoop MapReduce, Mahout has switched to using Spark as the backend
- [Apache MRQL](https://wiki.apache.org/mrql/) - A query processing and optimization system for large-scale, distributed data analysis, built on top of Apache Hadoop, Hama, and Spark
- [BlinkDB](http://blinkdb.org/) - a massively parallel, approximate query engine built on top of Shark and Spark
- [Spindle](https://github.com/adobe-research/spindle) - Spark/Parquet-based web analytics query engine
- [Thunderain](https://github.com/thunderain-project/thunderain) - a framework for combining stream processing with historical data, think Lambda architecture
- [DF](https://github.com/AyasdiOpenSource/df) from Ayasdi - a Pandas-like data frame implementation for Spark
- [Oryx](https://github.com/OryxProject/oryx) - Lambda architecture on Apache Spark, Apache Kafka for real-time large scale machine learning
- [ADAM](https://github.com/bigdatagenomics/adam) - A framework and CLI for loading, transforming, and analyzing genomic data using Apache Spark
- [TransmogrifAI](https://github.com/salesforce/TransmogrifAI) - AutoML library for building modular, reusable, strongly typed machine learning workflows on Spark with minimal hand tuning
- [Natural Language Processing for Apache Spark](https://github.com/JohnSnowLabs/spark-nlp) - A library to provide simple, performant, and accurate NLP annotations for machine learning pipelines
- [Rumble for Apache Spark](http://rumbledb.org/) - A JSONiq engine to query, with a functional language, large, nested, and heterogeneous JSON datasets that do not fit in dataframes.



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/sparkrelative/  

