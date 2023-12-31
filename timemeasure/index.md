# TimeMeasure


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1734b1eef8bd5f34tplv-t2oaga2asx-watermark.awebp)

#### 1. System.currentTimeMillis

```java
public class TimeIntervalTest {
    public static void main(String[] args) throws InterruptedException {
        // 开始时间
        long stime = System.currentTimeMillis();
        // 执行时间（1s）
        Thread.sleep(1000);
        // 结束时间
        long etime = System.currentTimeMillis();
        // 计算执行时间
        System.out.printf("执行时长：%d 毫秒.", (etime - stime));
    }
}
```

#### 2. System.nanoTime

```java
public class TimeIntervalTest {
    public static void main(String[] args) throws InterruptedException {
        // 开始时间
        long stime = System.nanoTime();
        // 执行时间（1s）
        Thread.sleep(1000);
        // 结束时间
        long etime = System.nanoTime();
        // 计算执行时间
        System.out.printf("执行时长：%d 纳秒.", (etime - stime));
    }
}
```

#### 3. new Date

##### .1. api 介绍

- **String toString( )**：把此 Date 对象转换为以下形式的 String： dow mon dd hh:mm:ss zzz yyyy 其中： dow 是一周中的某一天 (Sun, Mon, Tue, Wed, Thu, Fri, Sat)。
- **void setTime(long time)**： 用自1970年1月1日00:00:00 GMT以后time毫秒数设置时间和日期。
- **long getTime( )**：返回自 1970 年 1 月 1 日 00:00:00 GMT 以来此 Date 对象表示的毫秒数。
- **boolean before(Date date)** ：若当调用此方法的Date对象在指定日期之前返回true,否则返回false。

```java
Date dNow = new Date( );
SimpleDateFormat ft = new SimpleDateFormat ("yyyy-MM-dd hh:mm:ss");
System.out.println("当前时间为: " + ft.format(dNow));
```

##### .2. 测量执行时间demo

```java
import java.util.Date;

public class TimeIntervalTest {
    public static void main(String[] args) throws InterruptedException {
        // 开始时间
        Date sdate = new Date();
        // 执行时间（1s）
        Thread.sleep(1000);
        // 结束时间
        Date edate = new Date();
        //  统计执行时间（毫秒）
        System.out.printf("执行时长：%d 毫秒." , (edate.getTime() - sdate.getTime())); 
    }
}
```

#### 4. Spring StopWatch

```java
StopWatch stopWatch = new StopWatch();
// 开始时间
stopWatch.start();
// 执行时间（1s）
Thread.sleep(1000);
// 结束时间
stopWatch.stop();
// 统计执行时间（秒）
System.out.printf("执行时长：%d 秒.%n", stopWatch.getTotalTimeSeconds()); // %n 为换行
// 统计执行时间（毫秒）
System.out.printf("执行时长：%d 毫秒.%n", stopWatch.getTotalTimeMillis()); 
// 统计执行时间（纳秒）
System.out.printf("执行时长：%d 纳秒.%n", stopWatch.getTotalTimeNanos());
```

#### 5. commons-lang3 Stopwatch

```xml
<!-- https://mvnrepository.com/artifact/org.apache.commons/commons-lang3 -->
<dependency>
  <groupId>org.apache.commons</groupId>
  <artifactId>commons-lang3</artifactId>
  <version>3.10</version>
</dependency>
```

```java
import org.apache.commons.lang3.time.StopWatch;

import java.util.concurrent.TimeUnit;

public class TimeIntervalTest {
    public static void main(String[] args) throws InterruptedException {
        StopWatch stopWatch = new StopWatch();
        // 开始时间
        stopWatch.start();
        // 执行时间（1s）
        Thread.sleep(1000);
        // 结束时间
        stopWatch.stop();
        // 统计执行时间（秒）
        System.out.println("执行时长：" + stopWatch.getTime(TimeUnit.SECONDS) + " 秒.");
        // 统计执行时间（毫秒）
        System.out.println("执行时长：" + stopWatch.getTime(TimeUnit.MILLISECONDS) + " 毫秒.");
        // 统计执行时间（纳秒）
        System.out.println("执行时长：" + stopWatch.getTime(TimeUnit.NANOSECONDS) + " 纳秒.");
    }
}
```

#### 6. Guava Stopwatch

```xml
<!-- https://mvnrepository.com/artifact/com.google.guava/guava -->
<dependency>
  <groupId>com.google.guava</groupId>
  <artifactId>guava</artifactId>
  <version>29.0-jre</version>
</dependency>
```

```java
import com.google.common.base.Stopwatch;
import java.util.concurrent.TimeUnit;
public class TimeIntervalTest {
    public static void main(String[] args) throws InterruptedException {
        // 创建并启动计时器
        Stopwatch stopwatch = Stopwatch.createStarted();
        // 执行时间（1s）
        Thread.sleep(1000);
        // 停止计时器
        stopwatch.stop();
        // 执行时间（单位：秒）
        System.out.printf("执行时长：%d 秒. %n", stopwatch.elapsed().getSeconds()); // %n 为换行
        // 执行时间（单位：毫秒）
        System.out.printf("执行时长：%d 豪秒.", stopwatch.elapsed(TimeUnit.MILLISECONDS));
    }
}
```

#### Resource

- https://juejin.cn/post/6850037270444802055

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/timemeasure/  

