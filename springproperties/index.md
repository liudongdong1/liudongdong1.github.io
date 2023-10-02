# SpringProperties


### 1. [application.properties](https://github.com/eirunye/SpringBoot_Property/blob/master/src/main/resources/application.properties)配置

```
|--src
   |--main
      |--resources
         |--application.properties
```

```json
#启用调试日志。
debug=false
#启用跟踪日志。
trace=false

#--------------------------------------
# LOGGING 日记
#--------------------------------------
# 日志配置文件的位置。 例如，Logback的classpath:logback.xml
logging.config=classpath:logback.xml
# 日志文件名（例如，`myapp.log`）。名称可以是精确位置或相对于当前目录。
logging.file=property.log
# 最大日志文件大小。 仅支持默认的logback设置
logging.file.max-size=10MB
# 日志文件的位置。 例如，`/ var / log`。
logging.path=/var/log

#---------------------------------
# AOP
#---------------------------------
# 使用AOP 切面编程
spring.aop.auto=true
#是否要创建基于子类的（CGLIB）代理（true），而不是基于标准Java接口的代理（false）
spring.aop.proxy-target-class=true

#--------------------------------
# Email
#--------------------------------
# 编码格式
spring.mail.default-encoding=UTF-8
# SMTP服务器主机
spring.mail.host=smtp.property.com
#SMTP服务器端口
spring.mail.port=7800
# 登录SMTP用户名
spring.mail.username=property
# 登录SMTP密码
spring.mail.password=123456

#--------------------------------
# WEB 属性配置
#--------------------------------
# 服务器应绑定的网络地址
server.address=127.0.0.1
# 是否启用了响应压缩
server.compression.enabled=false
# 连接器在关闭连接之前等待另一个HTTP请求的时间。 未设置时，将使用连接器的特定于容器的默认值。 使用值-1表示没有（即无限）超时
server.connection-timeout=2000
# 错误控制器的路径
server.error.path=/error
# 是否启用HTTP / 2支持，如果当前环境支持它。
server.http2.enabled=false
# 服务器端口默认为:8080
server.port=8084 
# SP servlet的类名。
server.servlet.jsp.class-name=org.apache.jasper.servlet.JspServlet
# 主调度程序servlet的路径。
server.servlet.path=/home 
# 会话cookie名称
server.servlet.session.cookie.name=propertydemo

#------------------------------
# HTTP encoding
#------------------------------
# HTTP请求和响应的字符集。 如果未明确设置，则添加到“Content-Type”标头。
spring.http.encoding.charset=UTF-8 
# 是否启用http编码支持。
spring.http.encoding.enabled=true
#--------------------
# MULTIPART (MultipartProperties)
#--------------------
# 是否启用分段上传支持
spring.servlet.multipart.enabled=true
# 上传文件的中间位置
spring.servlet.multipart.location=/log
# 最大文件的大小
spring.servlet.multipart.max-file-size=1MB
# 最大请求大小
spring.servlet.multipart.max-request-size=10MB
# 是否在文件或参数访问时懒惰地解析多部分请求。
spring.servlet.multipart.resolve-lazily=false
#--------------------------------------------
# SPRING SESSION JDBC (JdbcSessionProperties)
#--------------------------------------------
# cron表达式用于过期的会话清理作业
spring.session.jdbc.cleanup-cron=0 * * * * *
# 数据库模式初始化模式
spring.session.jdbc.initialize-schema=embedded
# 用于初始化数据库模式的SQL文件的路径
spring.session.jdbc.schema=classpath:org/springframework/session/jdbc/schema-@@platform@@.sql
# 用于存储会话的数据库表的名称
spring.session.jdbc.table-name=SPRING_SESSION

#----------------------------------
# MONGODB 数据库配置
#----------------------------------
# 数据库名称
spring.data.mongodb.database=demo
# host 配置
spring.data.mongodb.host=127.0.0.1
# 登录用户名
spring.data.mongodb.username=property
# 登录密码
spring.data.mongodb.password=123456
# 端口号，自己根据安装的mongodb端口配置
spring.data.mongodb.port=9008
# 要启用的Mongo存储库的类型
spring.data.mongodb.repositories.type=auto
# 连接数据uri
spring.data.mongodb.uri=mongodb://localhost/test

#---------------------------------------
# DATASOURCE 数据库配置
#---------------------------------------
# MySql jdbc Driver
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
# 连接数据库
# demo表示的是你创建的数据库;
spring.datasource.url=jdbc:mysql://127.0.0.1:3306/demo?useSSL=false&requireSSL=false&characterEncoding=UTF-8&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=UTC
# 数据库用户名
spring.datasource.username=root
# 数据库密码
spring.datasource.password=123456
#-----------------------------------
# Jpa使用
#-----------------------------------
# 目标数据库进行操作，默认情况下自动检测。可以使用“databasePlatform”属性设置。
#spring.jpa.database= demo1
# 要操作的目标数据库的名称，默认情况下自动检测。 也可以使用“Database”枚举来设置。
#spring.jpa.database-platform=DEMO
# DDL模式 一般有这几种方式,Spring Boot会根据是否认为您的数据库是嵌入式的，为您选择一个默认值
# update: 更新架构时，使用;
spring.jpa.hibernate.ddl-auto=update
# 是否启用SQL语句的日志记录
spring.jpa.show-sql=true

#----------------------------------------
# TESTING PROPERTIES
#----------------------------------------
# 要替换的现有DataSource的类型
spring.test.database.replace=any
# MVC打印选项
spring.test.mockmvc.print=default

# ---------------大家查看文档进行配置，不一一列举了----------------------

#  各个属性注解在查看常用配置文件application.properties中

# FREEMARKER

# DEVTOOLS配置

# SPRING HATEOAS

# HTTP message conversion

# GSON

# JDBC 

# JEST (Elasticsearch HTTP client) (JestProperties)

# CASSANDRA (CassandraProperties)
# --------------------------等等----------------------------------
```

#### .1. 方式一

- application.properties添加:

```json
#--------------------------------
# 自定义属性
#--------------------------------
com.eirunye.defproname="root"
com.eirunye.defpropass="123456"
```

- **DefPropertyController.class**引用

```java
@RestController
public class DefPropertyController {

    @Value("${com.eirunye.defproname}")
    private String defProName;

    @Value("${com.eirunye.defpropass}")
    private String defProPass;

    @RequestMapping(value = "/defproprety")
    public String defPropretyUser() {
        return "这个自定义属性名为: " + defProName + ", 密码为:" + defProPass;
    }
}
```

#### .2. 方式二

- 新建一个`Properties.class`
- 添加`@ConfigurationProperties(prefix = "com.eirunye")`//表示的是通过自定义属性查找
- 在项目的入口文件**Application**添加注解**@EnableConfigurationProperties**,最后加上包名不然可能找不到扫描文件如:`@EnableConfigurationProperties({com.eirunye.defpropertys.bean.Properties.class})`。

```java
@ConfigurationProperties(prefix = "com.eirunye")//添加该注解
public class Properties {
    private String defproname;
    private String defpropass;
//  get/set方法
    public String getDefproname() {
        return defproname;
    }
    public void setDefproname(String defproname) {
        this.defproname = defproname;
    }
    public String getDefpropass() {
        return defpropass;
    }
    public void setDefpropass(String defpropass) {
        this.defpropass = defpropass;
    }
}
```

```java
@RestController
public class DefBeanPropertyController {
   //通过 Autowired注解来获取到 Properties属性，注:Autowired是按类型进行装配，可获取它所装配类的属性
    @Autowired
    Properties properties;

    @RequestMapping(value = "/bean/defproperty")
    public String getDefBeanProperties() {
        return "这是通过Bean注解的方式获取属性: " + properties.getDefproname() + ",密码为: " + properties.getDefpropass();
    }
}
```

```java
@SpringBootApplication
@EnableConfigurationProperties({com.eirunye.defpropertys.bean.Properties.class})//添加注解bean的扫描文件
public class DefpropertysApplication {

    public static void main(String[] args) {
        SpringApplication.run(DefpropertysApplication.class, args);
    }
}
```

#### .3. 方式三

- 添加自定义`def.properties`配置如下

```
#--------------------------------
# 自定义属性
#--------------------------------
# 用户名
com.eirunye.defineuser="property"
# 年龄
com.eirunye.defineage=20
```

- 创建 `DefineProperties.class`

```java
@Configuration
@ConfigurationProperties(prefix = "com.eirunye")//添加注解 ConfigurationProperties "com.eirunye"表示的是自定义属性
@PropertySource("classpath:defines.properties")// 添加注解 PropertySource 该注解能根据路径扫描到我们的文件
public class DefineProperties {
//    这里可以通过@Value("${}")方式添加,我已经屏蔽掉了，直接通过ConfigurationProperties注解的方式
//    @Value("${com.eirunye.defineuser}")
    private String defineuser;
//    @Value("${com.eirunye.defineage}")
    private int defineage;
// get/set方法
    public String getDefineuser() {
        return defineuser;
    }
    public void setDefineuser(String defineuser) {
        this.defineuser = defineuser;
    }
    public int getDefineage() {
        return defineage;
    }
    public void setDefineage(int defineage) {
        this.defineage = defineage;
    }
}
```

- 在`DefinePropertiesController.class`引用

```java
@RestController
public class DefinePropertiesController {
    @Autowired
    DefineProperties defineProperties;
    @RequestMapping(value = "define/Properties")
    public String getDefinePropertiesData(){
        return "新建文件自定义属性姓名："+defineProperties.getDefineuser()+",新建文件自定义属性年龄："+defineProperties.getDefineage();
    }
}
```

- 在Application里面添加配置`@EnableConfigurationProperties`

### 2. [application.yml](https://github.com/eirunye/SpringBoot_Property/blob/master/src/main/resources/application.yml)配置

> 理解yml和properties方式 格式的差别，具体代码使用详见上述方式一二三。

```
|--src
   |--main
      |--resources
         |--application.yml
         |--application-dev.yml   #正式开发
         |--application-prod.yml   #测试配置
```

```yml
spring:
  profiles:
    active: dev   #引用 application-dev.yml文件,这里我们可以改为 prod,表示引用application-prod.yml文件
  datasource:
      driver-class-name: com.mysql.cj.jdbc.Driver
      url: jdbc:mysql://127.0.0.1:3306/demo?useSSL=false&requireSSL=false&characterEncoding=UTF-8&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=UTC
      username: root
      password: 12346
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: true
  data:
    mongodb:
      host: 127.0.0.1
      uri: mongodb://localhost/test
      username: root
      password: 123456
      database: test
  test:
    database:
      replace: any
    mockmvc:
      print: default
  servlet:   
    multipart:
      enabled: true   
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/springproperties/  

