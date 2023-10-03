# 定时任务


### 1. 系统自带

#### .1. Linux crontab

> crontab [参数] [文件名];  运行`crontab -e`，可以编辑定时器;
>
> crontab 文件具体语法：
>
> [分] [小时] [日期] [月] [星期] 具体任务
>
> `0 2 * * * /usr/local/java/jdk1.8/bin/java -jar /data/app/tool.jar > /logs/tool.log &`

| 参数 |              功能               |
| :--- | :-----------------------------: |
| -u   |            指定用户             |
| -e   |  编辑某个用户的crontab文件内容  |
| -l   |  显示某个用户的crontab文件内容  |
| -r   |     删除某用户的crontab文件     |
| -i   | 删除某用户的crontab文件时需确认 |

- 分，表示多少分钟，范围：0-59
- 小时，表示多少小时，范围：0-23
- 日期，表示具体在哪一天，范围：1-31
- 月，表示多少月，范围：1-12
- 星期，表示多少周，范围：0-7，0和7都代表星期日
- `*`代表如何时间，比如：`*1***` 表示每天凌晨1点执行。
- `/`代表每隔多久执行一次，比如：`*/5 ****` 表示每隔5分钟执行一次。
- `,`代表支持多个，比如：`10 7,9,12 ***` 表示在每天的7、9、12点10分各执行一次。
- `-`代表支持一个范围，比如：`10 7-9 ***` 表示在每天的7、8、9点10分各执行一次。

### 2. JDK

#### .1. Thread

> - 优点：这种定时任务非常简单，学习成本低，容易入手，对于那些`简单的周期性任务`，是个不错的选择。
> - 缺点：`不支持指定某个时间点执行任务`，`不支持延迟执行等操作`，功能过于单一，无法应对一些较为复杂的场景。

```java
public static void init() {
    new Thread(() -> {
        while (true) {
            try {
                System.out.println("doSameThing");
                Thread.sleep(1000 * 60 * 5);
            } catch (Exception e) {
                log.error(e);
            }
        }
    }).start();
}
```

#### .2. Timer类

`Timer`类是jdk专门提供的定时器工具，用来在后台线程计划执行指定任务，在`java.util`包下，要跟`TimerTask`一起配合使用。`Timer`类其实是一个任务调度器，它里面包含了一个`TimerThread`线程，在这个线程中无限循环从`TaskQueue`中获取`TimerTask`（该类实现了Runnable接口），调用其`run`方法，就能异步执行定时任务。我们需要继承`TimerTask`类，实现它的`run`方法，在该方法中加上自己的业务逻辑。

- 优点：非常`方便实现多个周期性的定时任务`，并且支`持延迟执行`，还支持`在指定时间之后`支持，功能还算强大。
- 缺点：如果其中一个任务耗时非常长，会影响其他任务的执行。并且如果`TimerTask`抛出`RuntimeException`，`Timer`会停止所有任务的运行。

<img src="https://gitee.com/github-25970295/blogimgv2022/raw/master/Callable.png" style="zoom:67%;" />

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210831092852884.png)

```java
public class TimerTest {

    public static void main(String[] args) {
        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println("doSomething");
            }
        },2000,1000);  //schedule方法最后的两次参数分别表示：延迟时间 和 间隔时间，单位是毫秒。上面例子中，设置的定时任务是每隔1秒执行一次，延迟2秒执行。
    }
}
```

- `schedule(TimerTask task, Date time)`, 指定任务task在指定时间time执行
- `schedule(TimerTask task, long delay)`, 指定任务task在指定延迟delay后执行
- `schedule(TimerTask task, Date firstTime,long period)`,指定任务task在指定时间firstTime执行后，进行重复固定延迟频率peroid的执行
- `schedule(TimerTask task, long delay, long period)`, 指定任务task 在指定延迟delay 后，进行重复固定延迟频率peroid的执行
- `scheduleAtFixedRate(TimerTask task,Date firstTime,long period)`, 指定任务task在指定时间firstTime执行后，进行重复固定延迟频率peroid的执行
- `scheduleAtFixedRate(TimerTask task, long delay, long period)`, 指定任务task 在指定延迟delay 后，进行重复固定延迟频率peroid的执行.

#### .3. [ScheduledExecutorService](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247487551&idx=1&sn=18f64ba49f3f0f9d8be9d1fdef8857d9&scene=21#wechat_redirect)

- 优点：基于多线程的定时任务，多个任务之间不会相关影响，支持周期性的执行任务，并且带延迟功能。
- 缺点：不支持一些较复杂的定时规则。

![Executors](https://gitee.com/github-25970295/blogimgv2022/raw/master/Executors.png)

`ScheduledExecutorService`是基于多线程的，设计的初衷是为了解决`Timer`单线程执行，多个任务之间会互相影响的问题。它主要包含4个方法：

- `schedule(Runnable command,long delay,TimeUnit unit)`，带延迟时间的调度，只执行一次，调度之后可通过Future.get()阻塞直至任务执行完毕。
- `schedule(Callable<V> callable,long delay,TimeUnit unit)`，带延迟时间的调度，只执行一次，调度之后可通过Future.get()阻塞直至任务执行完毕，并且可以获取执行结果。
- `scheduleAtFixedRate`，表示以固定频率执行的任务，如果当前任务耗时较多，超过定时周期period，则当前任务结束后会立即执行。
- `scheduleWithFixedDelay`，表示以固定延时执行任务，延时是相对当前任务结束为起点计算开始时间。

```java
/*
 * 一、线程池：提供了一个线程队列，队列中保存着所有等待状态的线程。避免了创建与销毁额外开销，提高了响应的速度。
 * 
 * 二、线程池的体系结构：
 *     java.util.concurrent.Executor : 负责线程的使用与调度的根接口
 *         |--**ExecutorService 子接口: 线程池的主要接口
 *             |--ThreadPoolExecutor 线程池的实现类
 *             |--ScheduledExecutorService 子接口：负责线程的调度
 *                 |--ScheduledThreadPoolExecutor ：继承 ThreadPoolExecutor， 实现 ScheduledExecutorService
 * 
 * 三、工具类 : Executors 
 * ExecutorService newFixedThreadPool() : 创建固定大小的线程池
 * ExecutorService newCachedThreadPool() : 缓存线程池，线程池的数量不固定，可以根据需求自动的更改数量。
 * ExecutorService newSingleThreadExecutor() : 创建单个线程池。线程池中只有一个线程
 * 
 * ScheduledExecutorService newScheduledThreadPool() : 创建固定大小的线程，可以延迟或定时的执行任务。
 */
public class ScheduleExecutorTest {
    public static void main(String[] args) {
        ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(5);
        scheduledExecutorService.scheduleAtFixedRate(() -> {
            System.out.println("doSomething");
        },1000,1000, TimeUnit.MILLISECONDS);
    }
}
```

### 3. Spring

#### .1. spring task

> spring task先通过ScheduledAnnotationBeanPostProcessor类的processScheduled方法，解析和收集`Scheduled`注解中的参数，包含：cron表达式。
>
> 然后在ScheduledTaskRegistrar类的afterPropertiesSet方法中，默认初始化一个单线程的`ThreadPoolExecutor`执行任务。
>
> - 优点：spring框架自带的定时功能，springboot做了非常好的封装，`开启和定义定时任务非常容易`，支持复杂的`cron`表达式，可以满足绝大多数单机版的业务场景。单个任务时，当前次的调度完成后，再执行下一次任务调度。
> - 缺点：`默认单线程，如果前面的任务执行时间太长，对后面任务的执行有影响`。不支持集群方式部署，不能做数据存储型定时任务。

#### .2. spring quartz

- `作业调度`：调用各种框架的作业脚本，例如shell,hive等。
- `定时任务`：在某一预定的时刻，执行你想要执行的任务。
- 优点：默认是多线程异步执行，单个任务时，在上一个调度未完成时，下一个调度时间到时，会另起一个线程开始新的调度，多个任务之间互不影响。支持复杂的`cron`表达式，它能被集群实例化，支持分布式部署。
- 缺点：相对于spring task实现定时任务成本更高，需要手动配置`QuartzJobBean`、`JobDetail`和`Trigger`等。需要引入了第三方的`quartz`包，有一定的学习成本。不支持并行调度，不支持失败处理策略和动态分片的策略等。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210831094757042.png)

- `Scheduler` 代表调度容器，一个调度容器中可以注册多个JobDetail和Trigger。
- `Job` 代表工作，即要执行的具体内容。
- `JobDetail` 代表具体的可执行的调度程序，Job是这个可执行程调度程序所要执行的内容。
- `JobBuilder` 用于定义或构建JobDetail实例。
- `Trigger` 代表调度触发器，决定什么时候去调。
- `TriggerBuilder` 用于定义或构建触发器。
- `JobStore` 用于存储作业和任务调度期间的状态。

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-quartz</artifactId>
</dependency>
```

```java
//创建真正的定时任务执行类，该类继承QuartzJobBean。
ppublic class QuartzTestJob extends QuartzJobBean {
    @Override
    protected void executeInternal(JobExecutionContext context) throws JobExecutionException {
        String userName = (String) context.getJobDetail().getJobDataMap().get("userName");
        System.out.println("userName:" + userName);
    }
}
```

```java
//创建调度程序JobDetail和调度器Trigger。
@Configuration
public class QuartzConfig {
    @Value("${sue.spring.quartz.cron}")
    private String testCron;

    /**
     * 创建定时任务
     */
    @Bean
    public JobDetail quartzTestDetail() {
        JobDetail jobDetail = JobBuilder.newJob(QuartzTestJob.class)
                .withIdentity("quartzTestDetail", "QUARTZ_TEST")
                .usingJobData("userName", "susan")
                .storeDurably()
                .build();
        return jobDetail;
    }

    /**
     * 创建触发器
     */
    @Bean
    public Trigger quartzTestJobTrigger() {
        //每隔5秒执行一次
        CronScheduleBuilder cronScheduleBuilder = CronScheduleBuilder.cronSchedule(testCron);

        //创建触发器
        Trigger trigger = TriggerBuilder.newTrigger()
                .forJob(quartzTestDetail())
                .withIdentity("quartzTestJobTrigger", "QUARTZ_TEST_JOB_TRIGGER")
                .withSchedule(cronScheduleBuilder)
                .build();
        return trigger;
    }
}
```

```xml
sue.spring.quartz.cron=*/5 * * * * ?
```

### 4. [分布式定时任务](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247487551&idx=1&sn=18f64ba49f3f0f9d8be9d1fdef8857d9&scene=21#wechat_redirect)

#### .1. [xxl-job](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247487551&idx=1&sn=18f64ba49f3f0f9d8be9d1fdef8857d9&scene=21#wechat_redirect)

> `xxl-job`是大众点评（许雪里）开发的一个分布式任务调度平台，其核心设计目标是开发迅速、学习简单、轻量级、易扩展。现已开放源代码并接入多家公司线上产品线，开箱即用。
>
> `xxl-job`框架对`quartz`进行了扩展，使用`mysql`数据库存储数据，并且内置jetty作为`RPC`服务调用。
>
> - 优点：有界面管理定时任务，支持弹性扩容缩容、动态分片、故障转移、失败报警等功能。它的功能非常强大，很多大厂在用，可以满足绝大多数业务场景。
> - 缺点：和`quartz`一样，通过数据库分布式锁，来控制任务不能重复执行。在任务非常多的情况下，有一些性能问题。

![image-20210831095238867](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210831095238867.png)

#### .2. [elastic-job](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247487551&idx=1&sn=18f64ba49f3f0f9d8be9d1fdef8857d9&scene=21#wechat_redirect)

> `elastic-job`是当当网开发的弹性分布式任务调度系统，功能丰富强大，采用zookeeper实现分布式协调，实现任务高可用以及分片。它是专门为高并发和复杂业务场景开发。
>
> `elastic-job`目前是`apache`的`shardingsphere`项目下的一个子项目，官网地址：http://shardingsphere.apache.org/elasticjob/。
>
> - 优点：支持分布式调度协调，支持分片，`适合高并发`，和一些业务相对来说较复杂的场景。
> - 缺点：需要依赖于zookeeper，实现定时任务相对于`xxl-job`要复杂一些，要对分片规则非常熟悉。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210831095354308.png)

#### .3. [TBSchedule](https://github.com/taobao/TBSchedule)

> TBSchedule是阿里开发的一款分布式任务调度平台，旨在将调度作业从业务系统中分离出来，降低或者是消除和业务系统的耦合度，进行高效异步任务处理。

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%AE%9A%E6%97%B6%E4%BB%BB%E5%8A%A1/  

