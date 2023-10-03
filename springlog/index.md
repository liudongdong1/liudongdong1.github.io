# SpringLog


> [Logback](https://logback.qos.ch/) is one of the most widely used logging frameworks in the Java Community. It's a [replacement for its predecessor, Log4j.](https://logback.qos.ch/reasonsToSwitch.html) Logback offers a faster implementation than Log4j, provides more options for configuration, and more flexibility in archiving old log files.
>
> - slf4j是一系列的日志接口，而log4j和logback是具体实现了的日志框架。
>
> -    logger:作为日志的记录器，把它关联到对应的context后，主要用于存放日志对象，也可以定义日志类型，级别。appender:主要`用于指定日志输出的目的地`。目的地可以是控制台，文件，远程套接字服务器，数据库mysql等。layout:负责`把时间转换成字符串，格式化日志信息的输出。`
>
> - logback-core:所有logback模块的基础 , logback-classic:是log4j的一个改良版本，同时完整的实现了slf4j api,  logback-access:访问模块和servlet容器集成，提供通过http来访问日志的功能。
>
> 

```xml
<dependency>
    <groupId>ch.qos.logback</groupId>
    <artifactId>logback-core</artifactId>
    <version>1.2.3</version>
</dependency>
<dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-api</artifactId>
    <version>1.7.30</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>ch.qos.logback</groupId>
    <artifactId>logback-classic</artifactId>
    <version>1.2.3</version>
</dependency>
```

### 1. basic

#### .1. loggers

```java
private static final Logger LOGGER = 
LoggerFactory.getLogger(SematextLogbackTutorial.class);
```

#### .2. levels

- **TRACE** – the lowest level of information, mostly used for very deep code debugging, usually not included in production logs.
- **DEBUG** – low level of information used for debugging purposes, usually not included in production logs.
- **INFO** – a log severity carrying information, like an operation that started or finished.
- **WARN** – a log level informing about an event that may require our attention, but is not critical and may be expected.
- **ERROR** – a log level telling that an error, expected or unexpected, usually meaning that part of the system is not working properly.

#### .3. Appenders

> **Logback appender** is the component that Logback uses to write log events. They have their name and a single method that can process the event.

- **ConsoleAppender** – appends the log events to the System.out or System.err
- **OutputStreamAppender** – appends the log events to java.io.Outputstream providing the basic services for other appenders
- **FileAppender** – appends the log events to a file
- **RollingFileAppender** – appends the log events to a file with the option of automatic file rollover
- **SocketAppender** – appends the log events to a socket
- **SSLSocketAppender** – appends the log events to a socket using secure connection
- **SMTPAppender** – accumulates data in batches and send the content of the batch to a user-defined email after a user-specified event occurs
- **DBAppender** – appends data into a database tables
- **SyslogAppender** – appends data into Syslog compatible destination
- **SiftingAppender** – appender that is able to separate logging according to a given runtime attribute
- **AsyncAppender** – appends the logs events asynchronously

#### .4. Logback Encoders

> Logback encoder is responsible for `transforming a log event into a byte array `and `writing that byte array to an OutputStream`.

- **PatternLayoutEncoder** – encoder that takes a pattern and encodes the log event based on that pattern
- **LayoutWrappingEncoder** – encoder that closes the gap between the current Logback version and the versions prior to 0.9.19 that used to use Layout instances instead of the patterns.

#### .5. Logback Filters

> **Filter** is a mechanism for accepting or rejecting a log event based on the criteria defined by the filter itself.

1. class define

```java
package com.sematext.blog.logging.logback;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.filter.Filter;
import ch.qos.logback.core.spi.FilterReply;

public class SampleFilter extends Filter<ILoggingEvent> {
  @Override
  public FilterReply decide(ILoggingEvent event) {
    if (event.getMessage().contains("ERROR")) {
      return FilterReply.ACCEPT;
    }
    return FilterReply.DENY;
  }
}
```

2.  configuration to use class

```xml
<configuration>
    <appender name="console" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} %-5level %logger{36} - %msg%n</pattern>
        </encoder>
        <filter class="com.sematext.blog.logging.logback.SampleFilter" />
    </appender>

    <root level="info">
        <appender-ref ref="console" />
    </root>
</configuration>
```

- `在一个类前加上一个@slf4j 注解，将会在代码中自动生成一个Logger对象。`

```java
package com.sematext.blog.logging;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Logback {
  private static final Logger LOGGER = 
LoggerFactory.getLogger(Logback.class);

  public static void main(String[] args) {
    LOGGER.info("This is an INFO level log message!");
    LOGGER.error("This is an ERROR level log message!");
  }
}
```

### 2. logback configuration

- log4j.properties

```json
### 设置###
log4j.rootLogger = info,stdout,D,E

### 输出信息到控制抬 ###
log4j.appender.stdout = org.apache.log4j.ConsoleAppender
log4j.appender.stdout.Target = System.out
log4j.appender.stdout.layout = org.apache.log4j.PatternLayout
log4j.appender.stdout.layout.ConversionPattern = [%-5p] %d{yyyy-MM-dd HH:mm:ss,SSS} method:%l%n%m%n

### 输出DEBUG 级别以上的日志到=logs/error.log ###
log4j.appender.D = org.apache.log4j.DailyRollingFileAppender
log4j.appender.D.File = logs/log.log
log4j.appender.D.Append = true
log4j.appender.D.Threshold = DEBUG 
log4j.appender.D.layout = org.apache.log4j.PatternLayout
log4j.appender.D.layout.ConversionPattern = %-d{yyyy-MM-dd HH:mm:ss}  [ %t:%r ] - [ %p ]  %m%n

### 输出ERROR 级别以上的日志到=logs/error.log ###
log4j.appender.E = org.apache.log4j.DailyRollingFileAppender
log4j.appender.E.File =logs/error.log 
log4j.appender.E.Append = true
log4j.appender.E.Threshold = ERROR 
log4j.appender.E.layout = org.apache.log4j.PatternLayout
log4j.appender.E.layout.ConversionPattern = %-d{yyyy-MM-dd HH:mm:ss}  [ %t:%r ] - [ %p ]  %m%n
```

- logback-spring.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
 
<!-- scan属性未true时，如果配置文档发生改变将会进行重新加载 -->
 
<!-- scanPeriod属性设置监测配置文件修改的时间间隔，默认单位为毫秒，在scan为true时才生效 -->
 
<!-- debug属性如果为true时，会打印出logback内部的日志信息 -->
 
<configuration scan="true" scanPeriod="60 seconds" debug="false">
	<!-- 定义参数常量 -->
	<!-- 日志级别：TRACE<DEBUG<INFO<WARN<ERROR，其中常用的是DEBUG、INFO和ERROR -->
	<property name="log.level" value="debug" />
	<!-- 文件保留时间   60天-->
	<property name="log.maxHistory" value="60" />
	<!-- 日志存储路径 -->
	<property name="log.filePath" value="/opt/logs" />
	<!-- 日志的显式格式 -->
	<property name="log.pattern" value="%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50}-%msg%n"></property>
	
	<!-- 用于说明输出介质，此处说明控制台输出 -->
	<appender name="consoleAppender" class="ch.qos.logback.core.ConsoleAppender">
		<!-- 类似于layout，除了将时间转化为数组，还会将转换后的数组输出到相应的文件中 -->
		<encoder>
			<!-- 定义日志的输出格式 -->
			<pattern>${log.pattern}</pattern>
		</encoder>
	</appender>
	
	<!-- DEBUG，表示文件随着时间的推移按时间生成日志文件 -->
	<appender name="debugAppender" class="ch.qos.logback.core.rolling.RollingFileAppender">
		<!-- 文件路径 -->
		<file>${log.filePath}/debug.log</file>
		<!-- 滚动策略 -->
		<rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
			<!-- 设置文件名称 -->
			<fileNamePattern>
				${log.filePath}/debug/debug.%d{yyyy-MM-dd}.log
			</fileNamePattern>
			<!-- 设置最大保存周期 -->
			<MaxHistory>${log.maxHistory}</MaxHistory>
		</rollingPolicy>
		
		<encoder>
			<pattern>${log.pattern}</pattern>
		</encoder>
		
		<!-- 过滤器，过滤掉不是指定日志水平的日志 -->
		<filter class="ch.qos.logback.classic.filter.LevelFilter">
			<!-- 设置日志级别 -->
			<level>DEBUG</level>
			<!-- 如果跟该日志水平相匹配，则接受 -->
			<onMatch>ACCEPT</onMatch>
			<!-- 如果跟该日志水平不匹配，则过滤掉 -->
			<onMismatch>DENY</onMismatch>
		</filter>
	</appender>
	
	<!-- INFO，表示文件随着时间的推移按时间生成日志文件 -->
	<appender name="infoAppender"
		class="ch.qos.logback.core.rolling.RollingFileAppender">
		<!-- 文件路径 -->
		<file>${log.filePath}/info.log</file>
		<!-- 滚动策略 -->
		<rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
			<!-- 设置文件名称 -->
			<fileNamePattern>
				${log.filePath}/info/info.%d{yyyy-MM-dd}.log.gz
			</fileNamePattern>
			<!-- 设置最大保存周期 -->
			<MaxHistory>${log.maxHistory}</MaxHistory>
		</rollingPolicy>
		<encoder>
			<pattern>${log.pattern}</pattern>
		</encoder>
		<!-- 过滤器，过滤掉不是指定日志水平的日志 -->
		<filter class="ch.qos.logback.classic.filter.LevelFilter">
			<!-- 设置日志级别 -->
			<level>INFO</level>
			<!-- 如果跟该日志水平相匹配，则接受 -->
			<onMatch>ACCEPT</onMatch>
			<!-- 如果跟该日志水平不匹配，则过滤掉 -->
			<onMismatch>DENY</onMismatch>
		</filter>
	</appender>
	
	<!-- ERROR，表示文件随着时间的推移按时间生成日志文件 -->
	<appender name="errorAppender"
		class="ch.qos.logback.core.rolling.RollingFileAppender">
		<!-- 文件路径 -->
		<file>${log.filePath}/error.log</file>
		<!-- 滚动策略 -->
		<rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
			<!-- 设置文件名称 -->
			<fileNamePattern>
				${log.filePath}/error/error.%d{yyyy-MM-dd}.log.gz
			</fileNamePattern>
			<!-- 设置最大保存周期 -->
			<MaxHistory>${log.maxHistory}</MaxHistory>
		</rollingPolicy>
		<encoder>
			<pattern>${log.pattern}</pattern>
		</encoder>
		<!-- 过滤器，过滤掉不是指定日志水平的日志 -->
		<filter class="ch.qos.logback.classic.filter.LevelFilter">
			<!-- 设置日志级别 -->
			<level>ERROR</level>
			<!-- 如果跟该日志水平相匹配，则接受 -->
			<onMatch>ACCEPT</onMatch>
			<!-- 如果跟该日志水平不匹配，则过滤掉 -->
			<onMismatch>DENY</onMismatch>
		</filter>
	</appender>
	
	<!-- 用于存放日志对象，同时指定关联的package位置 -->
	<!-- name指定关联的package -->
	<!-- level表明指记录哪个日志级别以上的日志 -->
	<!-- appender-ref指定logger向哪个文件输出日志信息 -->
	<!-- additivity为true时，logger会把根logger的日志输出地址加入进来，但logger水平不依赖于根logger -->
	<logger name="com.campus.o2o" level="${log.level}" additivity="true">
		<appender-ref ref="debugAppender" />
		<appender-ref ref="infoAppender" />
		<appender-ref ref="errorAppender" />
	</logger>
	
	<!-- 特殊的logger，根logger -->
	<root lever="info">
		<!-- 指定默认的日志输出 -->
		<appender-ref ref="consoleAppender" />
	</root>
 
</configuration>
```

### 3. Tools

- https://sematext.com/logsene/

### 4. Resource

- https://blog.csdn.net/u014209205/article/details/80830904

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/springlog/  

