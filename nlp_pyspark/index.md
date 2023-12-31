# NLP_pyspark


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201113143906394.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201113144728692.png)

## 1. Concept

### 1.1. Estimators

> The **Estimators** have a method called fit() which secures and trains a piece of data to such application.

### 1.2. Transformers

> The **Transformer** is generally the result of a fitting process and applies changes to the the target dataset.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201113183140301.png)

###  1.3. Pipelines

> **Pipelines** are a mechanism for combining multiple estimators and transformers in a single workflow. They allow multiple chained transformations along a Machine Learning task. 

```
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[4]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.3")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .getOrCreate()
```

```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("Sentence")

regexTokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setCleanAnnotations(False)
    
pipeline = Pipeline() \
    .setStages([
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        finisher
    ])
```

```python
#-----        LightPipeline
from sparknlp.base import LightPipeline
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
lightPipeline = LightPipeline(explain_document_pipeline.model)
lightPipeline.annotate("Hello world, please annotate my text")
```

| Pipelines                                                    | Name                                  |
| :----------------------------------------------------------- | :------------------------------------ |
| [Explain Document ML](https://nlp.johnsnowlabs.com/docs/en/pipelines#explain_document_ml) | `explain_document_ml`                 |
| [Explain Document DL](https://nlp.johnsnowlabs.com/docs/en/pipelines#explain_document_dl) | `explain_document_dl`                 |
| Explain Document DL Win                                      | `explain_document_dl_noncontrib`      |
| Explain Document DL Fast                                     | `explain_document_dl_fast`            |
| Explain Document DL Fast Win                                 | `explain_document_dl_fast_noncontrib` |
| [Recognize Entities DL](https://nlp.johnsnowlabs.com/docs/en/pipelines#recognize_entities_dl) | `recognize_entities_dl`               |
| Recognize Entities DL Win                                    | `recognize_entities_dl_noncontrib`    |
| [OntoNotes Entities Small](https://nlp.johnsnowlabs.com/docs/en/pipelines#onto_recognize_entities_sm) | `onto_recognize_entities_sm`          |
| [OntoNotes Entities Large](https://nlp.johnsnowlabs.com/docs/en/pipelines#onto_recognize_entities_lg) | `onto_recognize_entities_lg`          |
| [Match Datetime](https://nlp.johnsnowlabs.com/docs/en/pipelines#match_datetime) | `match_datetime`                      |
| [Match Pattern](https://nlp.johnsnowlabs.com/docs/en/pipelines#match_pattern) | `match_pattern`                       |
| [Match Chunk](https://nlp.johnsnowlabs.com/docs/en/pipelines#match_chunks) | `match_chunks`                        |
| Match Phrases                                                | `match_phrases`                       |
| Clean Stop                                                   | `clean_stop`                          |
| Clean Pattern                                                | `clean_pattern`                       |
| Clean Slang                                                  | `clean_slang`                         |
| Check Spelling                                               | `check_spelling`                      |
| Analyze Sentiment                                            | `analyze_sentiment`                   |
| Analyze Sentiment DL                                         | `analyze_sentimentdl_use_imdb`        |
| Analyze Sentiment DL                                         | `analyze_sentimentdl_use_twitter`     |
| Dependency Parse                                             | `dependency_parse`                    |

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
SparkNLP.version()
val testData = spark.createDataFrame(Seq(
(1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
(2, "The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")
)).toDF("id", "text")
val pipeline = PretrainedPipeline("explain_document_ml", lang="en")
val annotation = pipeline.transform(testData)
annotation.show()
```

### 2. Operation

#### 2.1. Annotation

> 分词，命名实体识别，词性标注并称汉语词法分析“三姐妹”。词性标注即在给定的句子中判定每个词最合适的词性标记。词性标注的正确与否将会直接影响到后续的句法分析、语义分析，是中文信息处理的基础性课题之一。常用的词性标注模型有 N 元模型、隐马尔科夫模型、最大熵模型、基于决策树的模型等。

- annotatorType, begin, end, result, metadata, embeddings;

**【pretrained pipeline】**

```python
import sparknlp
sparknlp.start()
from sparknlp.pretrained import PretrainedPipeline
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
annotations = explain_document_pipeline.annotate("We are very happy about SparkNLP")
print(annotations)
OUTPUT:
{
  'stem': ['we', 'ar', 'veri', 'happi', 'about', 'sparknlp'],
  'checked': ['We', 'are', 'very', 'happy', 'about', 'SparkNLP'],
  'lemma': ['We', 'be', 'very', 'happy', 'about', 'SparkNLP'],
  'document': ['We are very happy about SparkNLP'],
  'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP'],
  'token': ['We', 'are', 'very', 'happy', 'about', 'SparkNLP'],
  'sentence': ['We are very happy about SparkNLP']
}
```

**【spark dataframes】**

```python
import sparknlp
sparknlp.start()

sentences = [
  ['Hello, this is an example sentence'],
  ['And this is a second sentence.']
]

# spark is the Spark Session automatically started by pyspark.
data = spark.createDataFrame(sentences).toDF("text")

# Download the pretrained pipeline from Johnsnowlab's servers
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
# Transform 'data' and store output in a new 'annotations_df' dataframe
annotations_df = explain_document_pipeline.transform(data)

# Show the results
annotations_df.show()
#annotations_df.select("token").show(truncate=False)
```

**【deal with just the resulting annotations】**

```python
finisher = Finisher().setInputCols(["token", "lemma", "pos"])
explain_pipeline_model = PretrainedPipeline("explain_document_ml").model
pipeline = Pipeline() \
    .setStages([
        explain_pipeline_model,
        finisher
        ])
sentences = [
    ['Hello, this is an example sentence'],
    ['And this is a second sentence.']
]
data = spark.createDataFrame(sentences).toDF("text")
model = pipeline.fit(data)
annotations_finished_df = model.transform(data)
annotations_finished_df.select('finished_token').show(truncate=False)
OUTPUT:
+-------------------------------------------+
|finished_token                             |
+-------------------------------------------+
|[Hello, ,, this, is, an, example, sentence]|
|[And, this, is, a, second, sentence, .]    |
+-------------------------------------------+
```

### 1.3. Training Dataset

#### 1.3.1. Pos datasets

> train a Part of Speech Tagger annotator;

#### 1.3.2. CoNLL Dataset

>  train a Named Entity Recognition DL annotator;

### 1.4. Word Embeddings

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201113193405004.png)

### 1.5. Text Classification

- NER DL uses Char CNNs - BiLSTM - CRF Neural Network architecture.
- Relation Extraction
- Spell checking & correction
- entity recognition
  - 接下来nlp 学习路线：

![image-20201113183016424](C:/Users/dell/AppData/Roaming/Typora/typora-user-images/image-20201113183016424.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/nlp_pyspark/  

