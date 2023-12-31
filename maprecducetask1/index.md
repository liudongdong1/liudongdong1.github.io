# MapReduceTask1


#### 1. MapReduce

> The `key` and `value` classes have to be serializable by the framework and hence need to implement the [Writable](http://hadoop.apache.org/docs/current/api/org/apache/hadoop/io/Writable.html) interface. Additionally, the key classes have to implement the [WritableComparable](http://hadoop.apache.org/docs/current/api/org/apache/hadoop/io/WritableComparable.html) interface to facilitate sorting by the framework. Input and Output types of a MapReduce job.

```java
import java.io.IOException;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Iterator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
public class WordCountAndSort {
	public static class TokenizerMapper 
	extends Mapper<Object, Text, Text, IntWritable>{
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();
		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				//To Do	job1: filter and uppercase the input data
				String str = itr.nextToken();
				boolean allLetters = str.chars().allMatch(Character::isLetter);
				if(allLetters) {
				   String upper_str = str.toUpperCase();
				   word.set(upper_str);
				   context.write(word,one);
				} 		
			}
		}
	}
	public static class IntSumReducer 
	extends Reducer<Text,IntWritable,Text,IntWritable> {
		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values, 
				Context context
				) throws IOException, InterruptedException {
			int sum = 0;
			Iterator<IntWritable> itr = values.iterator();
			while(itr.hasNext()) {
				sum += itr.next().get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}
	private static class MyComparator extends IntWritable.Comparator {
		@Override
		public int compare(Object a, Object b) {
			return -super.compare(a,b);
		}
		
		@Override
		public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
			return -super.compare(b1,s1,l1,b2,s2,l2);
		} 
	}
    //客户端代码，写完交给ResourceManager框架去执行
	public static void main(String[] args) throws Exception {
        ////创建Hadoop conf对象，，其构造方法会默认加载hadoop中的两个配置文件，分别是hdfs-site.xml以及core-site.xml，这两个文件中会有访问hdfs所需的参数值，主要是fs.default.name，指定了hdfs的地址，有了这个地址客户端就可以通过这个地址访问hdfs了。即可理解为configuration就是hadoop中的配置信息。    
		Configuration conf = new Configuration();
        //GenericOptionsParser是hadoop框架中解析命令行参数的基本类。它能够辨别一些标准的命令行参数，能够使应用程序轻易地指定namenode，jobtracker，以及其他额外的配置资源。
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		for (String string : otherArgs) {
			System.out.println(string);
		}
		if (otherArgs.length < 2) {
			System.err.println("Usage: wordcount <in> [<in>...] <out>");
			System.exit(2);
		}
        //conf.set(" mapred.textoutputformat.separator", " ");
		//MapReduce默认的key-value的分隔符为tab，这样输出过程中会导致格式不规律，即key1 key2 tab value1 value2... 可以通过这个语句，设定最后输出时，key value之间的分隔符为空格 
        //获取job对象
		Job job = Job.getInstance(conf, "word count");
        //设置job方法入口的驱动类
		job.setJarByClass(WordCountAndSort.class);
        //设置Mapper组件类
		job.setMapperClass(TokenizerMapper.class);
		job.setCombinerClass(IntSumReducer.class);
        //设置reduce组件类
		job.setReducerClass(IntSumReducer.class);
        //设置reduce输出的key和value类型
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
        //设置输入路径
		for (int i = 0; i < otherArgs.length - 1; ++i) {
			FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
		}
		Path tempDir = new Path("wordcount-temp-" + Integer.toString(  
	            new Random().nextInt(Integer.MAX_VALUE)));
		System.out.println(tempDir.toString());
		FileOutputFormat.setOutputPath(job, tempDir);
		job.waitForCompletion(true);
        Job sortJob = Job.getInstance(conf, "sort");
        sortJob.setJarByClass(WordCountAndSort.class);  
        FileInputFormat.addInputPath(sortJob, tempDir);  
        sortJob.setInputFormatClass(SequenceFileInputFormat.class); 
        sortJob.setMapperClass(InverseMapper.class);
        sortJob.setNumReduceTasks(1);   
        //设置输出结果路径，要求结果路径事先不能存在
        FileOutputFormat.setOutputPath(sortJob, new Path(otherArgs[1]));
        sortJob.setOutputKeyClass(IntWritable.class);  
        sortJob.setOutputValueClass(Text.class);
        sortJob.setSortComparatorClass(MyComparator.class);
        System.out.println("The first job finished.");
        ////此句是对job进行提交，一般情况下我们提交一个job都是通过job.waitForCompletion方法提交，该方法内部会调用job.submit()方法
        System.exit(sortJob.waitForCompletion(true) ? 0 : 1); 
	}
}

```

```shell
javac WordCountAndSort.java
jar -cvf WordCountAndSort.jar ./WordCountAndSort*.class
/root/cloudz/hadoop-2.7.3/sbin/start-all.sh
bin/hadoop fs -ls /user/joe/wordcount/input/
hadoop jar WordCountAndSort.jar WordCountAndSort /dataset1G /sortResult
hadoop fs -cat /sortResult/part-r-00000 | head
```

#### 2. Configuration   

>  Hadoop的配置文件的操作主要分为三个部分：配置的加载、属性读取和设置.

```java
/** 
  * 保存了所有的资源配置的来源信息，资源文件主要有以下几种形式: URL、String、Path、InputStream和Properties。
  */
private ArrayList<Resource> resources = new ArrayList<Resource>();

/**
  * 记录了配置文件中配置的final类型的属性名称，标记为final之后如果另外有同名的属性，那么该属性将不会被替换
  */
private Set<String> finalParameters = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());

/**
  * 记录了配置初始化后更新过的属性，其键为更新的属性名，值为更新操作的来源
  */
private Map<String, String[]> updatingResource;

/**
  * 记录了所有的属性，包括系统初始化以及后续设置的属性
  */
private Properties properties;

/**
  * 记录了除了初始化之后手动调用set方法设置的属性
  */
private Properties overlay;

/**
  * 记录了过期的属性
  */
private static AtomicReference<DeprecationContext> deprecationContext = new AtomicReference<DeprecationContext>(new DeprecationContext(null, defaultDeprecations));
```

##### 2.1. Resource

```java
private static class Resource {
    private final Object resource;
    private final String name;
    // 构造方法，get和set方法略
  }
```

- URL: 通过一个URL链接来进行读取；
- String: 从当前classpath下该字符串所指定的文件读取；
- Path: 以绝对路径指定的配置文件或者是使用url指定的配置；
- InputStream: 以流的形式指定的配置文件；
- Properties: 以属性配置类保存的配置信息。
   除了可以通过上述方式进行配置的设置以外，Hadoop还设置了几个默认的配置文件：core-default.xml、core-site.xml和hadoop-site.xml

```java
 static {
    // Add default resources
    addDefaultResource("core-default.xml");
    addDefaultResource("core-site.xml");

    // print deprecation warning if hadoop-site.xml is found in classpath
    ClassLoader cL = Thread.currentThread().getContextClassLoader();
    if (cL == null) {
      cL = Configuration.class.getClassLoader();
    }
    if (cL.getResource("hadoop-site.xml") != null) {
      LOG.warn("DEPRECATED: hadoop-site.xml found in the classpath. " +
          "Usage of hadoop-site.xml is deprecated. Instead use core-site.xml, "
          + "mapred-site.xml and hdfs-site.xml to override properties of " +
          "core-default.xml, mapred-default.xml and hdfs-default.xml " +
          "respectively");
      addDefaultResource("hadoop-site.xml");
    }
  }
```

> 在加载每个配置文件的时候，首先通过instanceof判断wrapper对象中Object类型的属性resource是什么类型，然后根据具体不同的类型将其转换为一个XMLStreamReader2类型的reader。对于转换之后的reader，其会依次读取xml配置文件中的具体标签信息，如：name、value、final、source等等，并且将读取后的信息保存在properties中，这里需要注意的是，在代码中的case "include"处可以看出，配置文件中如果使用了<include></include>标签引入其他的配置文件，那么其会递归的调用loadResource()方法对其进行读取。

##### 2.2. 属性的获取

```java
//处理过期键和对属性值进行标签替换处理
public String get(String name, String defaultValue) {
    // 处理过期的属性
    String[] names = handleDeprecation(deprecationContext.get(), name);
    String result = null;
    // 对属性值中形如${foo.bar}引入的其他属性进行替换
    for(String n : names) {
      result = substituteVars(getProps().getProperty(n, defaultValue));
    }
    return result;
}
```

> 对于属性值的获取，其可以通过三个途径进行：系统变量、当前环境变量和配置文件及用户设置的属性值，并且其优先级是：系统变量 > 当前环境变量 > 配置文件及用户设置的属性值。

##### 2.3. 属性设置

```java
public void set(String name, String value, String source) {
    Preconditions.checkArgument(name != null, "Property name must not be null");
    Preconditions.checkArgument(value != null, "The value of property %s must not be null", name);
    name = name.trim();
    DeprecationContext deprecations = deprecationContext.get();
    if (deprecations.getDeprecatedKeyMap().isEmpty()) {
      // 初始化配置文件信息
      getProps();
    }
    getOverlay().setProperty(name, value);
    getProps().setProperty(name, value);
    String newSource = (source == null ? "programmatically" : source);

    if (!isDeprecated(name)) {
      // 这里该name不是过期键分为两种情况，一种是在存储过期键的map（即deprecatedKeyMap）中没有相应数据，
      // 而在更新的map（即reverseDeprecatedKeyMap）中有数据，第二种是在这两个map中都没有数据
      updatingResource.put(name, new String[] {newSource});
      // 判断该键是否为更新某一个过期键之后的键，如果是，则获取所有更新了该过期键的键
      String[] altNames = getAlternativeNames(name);
      if(altNames != null) {
        // 更新所有该键所对应的过期键更新之后的键值
        for(String n: altNames) {
          if(!n.equals(name)) {
            getOverlay().setProperty(n, value);
            getProps().setProperty(n, value);
            updatingResource.put(n, new String[] {newSource});
          }
        }
      }
    } else {
      // 如果该键为过期键，则将该过期键更新之后的键的值都设置为新的值
      String[] names = handleDeprecation(deprecationContext.get(), name);
      String altSource = "because " + name + " is deprecated";
      for(String n : names) {
        getOverlay().setProperty(n, value);
        getProps().setProperty(n, value);
        updatingResource.put(n, new String[] {altSource});
      }
    }
  }
```

#### 3. MapReduce Pipline

##### 3.1. MR1

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201216215846266.png)

##### 3.2  MR2

- RM(Resource Manager)
- AM(Application Master)
- NM(Node Manager)
- CLC(Container Launch Context)：CLC发给ResourceManager，提供了资源需求（内存/CPU）、作业文件、安全令牌以及在节点上启动ApplicationMaster需要的其他信息。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201216220947566.png)

1. client向RM提交申请，包括CLC所需的信息。

2. 位于RM中的Application Manager会协商一个容器并为应用程序初始化AM。

3. AM注册到RM，并请求容器。

4. AM与NM通信以启动已授予的容器，并为每个容器指定CLC。

5. 然后AM管理应用程序执行

   在执行期间，应用程序向AM提供进度和状态信息。Client可以通过查询RM或直接与AM通信来监视应用程序的状态。

6. AM向RM报告应用程序的完成情况

7. AM从RM上注销，RM清理AM所在容器。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201216221243225.png)

##### 3.3. InputFormat

1. FileInputFormat: 是所有基于文件的inputFormat的基类。 它指定了输入文件的路径，当提交job时，它提供了所读文件的路径。 它将会读取所有文件，并将它们分为一个或多个InputSplit。
2. **TextInputFormat**: 是MapReduce默认的InputFormat。 它把文件中的每一行当作一条分离的记录。 key为每一行的偏移量，value为每一行的内容。
3. KeyValueTextInputFormat: 和TextInputFormat很像，都是把每一行当作分离的记录。 只不过它输出的key和value是根据tab（/t）分割开的两段。
4. SequenceFileInputFormat:  是一个读取序列文件的InputFormat。 序列文件是存储二进制键值对序列的二进制文件。 序列文件是块压缩的，并提供几种任意数据类型（不仅仅是文本）的直接序列化和反序列化。 这里的键和值都是用户自定义的。
5. SequenceFileAsTextInputFormat:  是SequenceFileInputFormat的另一种形式，它将序列文件键值转换为Text对象。 通过调用’tostring（）’，对key和value执行转换。 这个InputFormat使序列文件适合输入流。
6. SequenceFileAsBinaryInputFormat:  是一个SequenceFileInputFormat， 我们可以使用它来提取序列文件的键和值作为不透明的二进制对象。
7. **NLineInputFormat**:  是TextInputFormat的另一种形式。只是，它接收一个变量N，代表每个InputSplit处理多少行数据。
8. DBInputFormat:  是一个可以通过JDBC来读取关系型数据库的InputFormat。 由于它没有分割功能，所以我们需要小心不要让太多的Mapper读取数据库。 所以最好用它加载较小的数据集，比如是用来和一个HDFS上的大数据集进行join，这是可使用MultipleInputs。 key是LongWritables，Value是DBWritables。

##### 3.4. InputSplits

> 从逻辑上来讲，它会被单个Mapper所处理。 对每个InputSplit，会创建一个map task处理它，所以这里有多少个inputSplit就有多少个map task。

##### 3.5. RecordReader

> 并把数据变为key-value形式，来发送给mappe做后续处理。

##### 3.6.  Mapper

> **Mapper的输出作为中间结果写入硬盘**。中间结果不写入HDFS的原因有两点：
>
> - 中间结果是临时文件，上传hdfs会产生不必要的备份文件;
> - HDFS是一个高延迟的系统。
>
> Mapper的数量：**No. of Mapper= {(total data size)/ (input split size)}**

##### 3.7. Partitioner

> Partitioner对Mapper的输出执行分区操作。具体步骤如下：
>
> 1. partitioner拿到combiner的输出键值对中的key
> 2. 对key的值进行hash函数转换，获取分区id
> 3. 根据分区id，再将键值对分入对应分区
> 4. 每个分区又会被发送给一个Reducer。
>
> 需要注意，分区数量和reduce task数量一致的，所以若要控制Partitioner的数量，可以通过*JobConf.setNumReduceTasks()*来设置。

##### 3.8. Combiner

> Combiner对partition后的output进行local的聚合，来减少mapper和reducer之间的网络传输。 尤其是在处理一个巨大的数据集时，会产生很多巨大的中间数据，这些巨大的中间数据，不仅会加大网络传输的压力，同时也会加大Reducer的处理压力。

##### 3.9. Shuffling&Sorting

> 在Reducer开始处理数据之前，所有中间结果键值对会被MapReduce框架按key值来排序，而不是按value值。 被reducer处理的value们是不被排序的，它们可以是任何顺序。 (注：不过MapReduce也提供了对value排序的机制，叫做[Secondary Sorting](https://waltyou.github.io/Hadoop-MapReduce-Workflow/MapReduce-Secondary-Sort)。)

##### 3.10 RecordWriter

```java
public abstract class RecordWriter<K, V> {

  public abstract void write(K key, V value
                             ) throws IOException, InterruptedException;

  public abstract void close(TaskAttemptContext context
                             ) throws IOException, InterruptedException;
}
```

##### 3.11 OutputFormat

#### 4. 学习链接

- [configuration introduce](https://www.jianshu.com/p/305dd19f08cc)
- [mapreduce](https://waltyou.github.io/Hadoop-MapReduce-Workflow/)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/maprecducetask1/  

