# SpringActuator


> [Spring Boot Actuator](https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html#actuator.enabling) 模块提供了生产级别的功能，比如`健康检查，审计，指标收集，HTTP 跟踪等`，帮助我们监控和管理Spring Boot 应用。这个模块是一个采集应用内部信息暴露给外部的模块，上述的功能都可以通过HTTP 和 JMX 访问。

```xml
<dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

### 1. Endpoints

> Spring Boot 提供了所谓的 endpoints （下文翻译为端点）给外部来与应用程序进行访问和交互。

- 应用配置类：获取应用程序中加载的应用配置、环境变量、自动化配置报告等与Spring Boot应用密切相关的配置类信息。
- 度量指标类：获取应用程序运行过程中用于监控的度量指标，比如：内存信息、线程池信息、HTTP请求统计等。
- 操作控制类：提供了对应用的关闭等操作类功能。
- 每一个端点都可以通过`配置来单独禁用或者启动`
- 不同于Actuator 1.x，**Actuator 2.x 的大多数端点默认被禁掉**。 Actuator 2.x 中的默认端点增加了`/actuator`前缀。默认暴露的两个端点为`/actuator/health`和 `/actuator/info`

| ID                 | Description                                                  |
| :----------------- | :----------------------------------------------------------- |
| `auditevents`      | Exposes audit events information for the current application. Requires an `AuditEventRepository` bean. |
| `beans`            | Displays a complete list of all the Spring beans in your application. |
| `caches`           | Exposes available caches.                                    |
| `conditions`       | Shows the conditions that were evaluated on configuration and auto-configuration classes and the reasons why they did or did not match. |
| `configprops`      | Displays a collated list of all `@ConfigurationProperties`.  |
| `env`              | Exposes properties from Spring’s `ConfigurableEnvironment`.  |
| `flyway`           | Shows any Flyway database migrations that have been applied. Requires one or more `Flyway` beans. |
| `health`           | Shows application health information.                        |
| `httptrace`        | Displays HTTP trace information (by default, the last 100 HTTP request-response exchanges). Requires an `HttpTraceRepository` bean. |
| `info`             | Displays arbitrary application info.                         |
| `integrationgraph` | Shows the Spring Integration graph. Requires a dependency on `spring-integration-core`. |
| `loggers`          | Shows and modifies the configuration of loggers in the application. |
| `liquibase`        | Shows any Liquibase database migrations that have been applied. Requires one or more `Liquibase` beans. |
| `metrics`          | Shows ‘metrics’ information for the current application.     |
| `mappings`         | Displays a collated list of all `@RequestMapping` paths.     |
| `quartz`           | Shows information about Quartz Scheduler jobs.               |
| `scheduledtasks`   | Displays the scheduled tasks in your application.            |
| `sessions`         | Allows retrieval and deletion of user sessions from a Spring Session-backed session store. Requires a Servlet-based web application using Spring Session. |
| `shutdown`         | Lets the application be gracefully shutdown. Disabled by default. |
| `startup`          | Shows the [startup steps data](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.spring-application.startup-tracking) collected by the `ApplicationStartup`. Requires the `SpringApplication` to be configured with a `BufferingApplicationStartup`. |
| `threaddump`       | Performs a thread dump.                                      |

### 2. 端点暴露配置

```json
management.endpoints.enabled-by-default=false
management.endpoint.info.enabled=true
```

| Property                                    | Default       |
| ------------------------------------------- | ------------- |
| `management.endpoints.jmx.exposure.exclude` |               |
| `management.endpoints.jmx.exposure.include` | `*`           |
| `management.endpoints.web.exposure.exclude` |               |
| `management.endpoints.web.exposure.include` | `info, healt` |

```json
#可以打开所有的监控点
management.endpoints.web.exposure.include=*
#也可以选择打开部分，"*" 代表暴露所有的端点，如果指定多个端点，用","分开
management.endpoints.jmx.exposure.include=health,info
#Actuator 默认所有的监控点路径都在/actuator/*，当然如果有需要这个路径也支持定制。
management.endpoints.web.base-path=/minitor
```

> Actuator 默认所有的监控点路径都在`/actuator/*`，当然如果有需要这个路径也支持定制。

#### .1. health 端点

> `/health`端点会聚合你程序的健康指标，来检查程序的健康情况。端点公开的应用健康信息取决于：management.endpoint.health.show-details=always

| Name              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `never`           | 不展示详细信息，up或者down的状态，默认配置                   |
| `when-authorized` | 详细信息将会展示给通过认证的用户。授权的角色可以通过`management.endpoint.health.roles`配置 |
| `always`          | 对所有用户暴露详细信息                                       |

```json
management.health.mongo.enabled: false #配置禁用某个组件的健康监测。
```

| Key             | Name                                                         | Description                                               |
| :-------------- | :----------------------------------------------------------- | :-------------------------------------------------------- |
| `cassandra`     | [`CassandraDriverHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/cassandra/CassandraDriverHealthIndicator.java) | Checks that a Cassandra database is up.                   |
| `couchbase`     | [`CouchbaseHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/couchbase/CouchbaseHealthIndicator.java) | Checks that a Couchbase cluster is up.                    |
| `db`            | [`DataSourceHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/jdbc/DataSourceHealthIndicator.java) | Checks that a connection to `DataSource` can be obtained. |
| `diskspace`     | [`DiskSpaceHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/system/DiskSpaceHealthIndicator.java) | Checks for low disk space.                                |
| `elasticsearch` | [`ElasticsearchRestHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/elasticsearch/ElasticsearchRestHealthIndicator.java) | Checks that an Elasticsearch cluster is up.               |
| `hazelcast`     | [`HazelcastHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/hazelcast/HazelcastHealthIndicator.java) | Checks that a Hazelcast server is up.                     |
| `influxdb`      | [`InfluxDbHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/influx/InfluxDbHealthIndicator.java) | Checks that an InfluxDB server is up.                     |
| `jms`           | [`JmsHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/jms/JmsHealthIndicator.java) | Checks that a JMS broker is up.                           |
| `ldap`          | [`LdapHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/ldap/LdapHealthIndicator.java) | Checks that an LDAP server is up.                         |
| `mail`          | [`MailHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/mail/MailHealthIndicator.java) | Checks that a mail server is up.                          |
| `mongo`         | [`MongoHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/mongo/MongoHealthIndicator.java) | Checks that a Mongo database is up.                       |
| `neo4j`         | [`Neo4jHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/neo4j/Neo4jHealthIndicator.java) | Checks that a Neo4j database is up.                       |
| `ping`          | [`PingHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/health/PingHealthIndicator.java) | Always responds with `UP`.                                |
| `rabbit`        | [`RabbitHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/amqp/RabbitHealthIndicator.java) | Checks that a Rabbit server is up.                        |
| `redis`         | [`RedisHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/redis/RedisHealthIndicator.java) | Checks that a Redis server is up.                         |
| `solr`          | [`SolrHealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/solr/SolrHealthIndicator.java) | Checks that a Solr server is up.                          |

#### .2. 自定义 Health Indicator

> To provide custom health information, you can register Spring beans that implement the [`HealthIndicator`](https://github.com/spring-projects/spring-boot/tree/v2.5.2/spring-boot-project/spring-boot-actuator/src/main/java/org/springframework/boot/actuate/health/HealthIndicator.java) interface.

```java
@Component
public class MyHealthIndicator implements HealthIndicator {

    @Override
    public Health health() {
        int errorCode = check();
        if (errorCode != 0) {
            return Health.down().withDetail("Error Code", errorCode).build();
        }
        return Health.up().build();
    }

    private int check() {
        // perform some specific health check
        return ...
    }

}
```

- Reactive Health Indicators

```java
@Component
public class MyReactiveHealthIndicator implements ReactiveHealthIndicator {

    @Override
    public Mono<Health> health() {
        return doHealthCheck().onErrorResume((exception) ->
            Mono.just(new Health.Builder().down(exception).build()));
    }

    private Mono<Health> doHealthCheck() {
        // perform some specific health check
        return ...
    }

}
```

#### .3. metrics 端点

> `/metrics`端点用来返回当前应用的`各类重要度量指标，比如：内存信息、线程信息、垃圾回收信息、tomcat、数据库连接池等`。http://localhost:8080/actuator/metrics/{MetricName} 来进行访问。

```json
{
    "names": [
        "tomcat.threads.busy",
        "jvm.threads.states",
        "jdbc.connections.active",
        "jvm.gc.memory.promoted",
        "http.server.requests",
        "hikaricp.connections.max",
        "hikaricp.connections.min",
        "jvm.memory.used",
        "jvm.gc.max.data.size",
        "jdbc.connections.max",
         ....
    ]
}
```

#### .4. loggers 端点

> `/loggers` 端点暴露了我们程序内部配置的所有logger的信息。我们访问`/actuator/loggers`可以看到，http://localhost:8080/actuator/loggers/{name}

#### .5. `info`端点

> `/info`端点可以用来展示你程序的信息。访问`http://localhost:8080/actuator/info`，

#### .6. `beans`端点

> `/beans`端点会返回Spring 容器中所有bean的别名、类型、是否单例、依赖等信息。

#### .7. `heapdump` 端点

> 访问：`http://localhost:8080/actuator/heapdump`会自动生成一个 Jvm 的堆文件 heapdump。我们可以使用 JDK 自带的 Jvm 监控工具 VisualVM 打开此文件查看内存快照。

#### .8. `threaddump` 端点

> 主要展示了线程名、线程ID、线程的状态、是否等待锁资源、线程堆栈等信息。就是可能查看起来不太直观。访问`http://localhost:8080/actuator/threaddump`.

### 3. 安全校验

> 由于端点的信息和产生的交互都是非常敏感的，必须防止未经授权的外部访问。如果您的应用程序中存在**Spring Security**的依赖，则默认情况下使用**基于表单的HTTP身份验证**来保护端点。

```xml
<dependency>
   <groupId>org.springframework.boot</groupId>
   <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

```java
import org.springframework.boot.actuate.autoconfigure.security.servlet.EndpointRequest;
import org.springframework.boot.actuate.context.ShutdownEndpoint;
import org.springframework.boot.autoconfigure.security.servlet.PathRequest;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
 
/**
 * @author Richard_yyf
 */
@Configuration
public class ActuatorSecurityConfig extends WebSecurityConfigurerAdapter {
 
    /*
     * version1:
     * 1. 限制 '/shutdown'端点的访问，只允许ACTUATOR_ADMIN访问
     * 2. 允许外部访问其他的端点
     * 3. 允许外部访问静态资源
     * 4. 允许外部访问 '/'
     * 5. 其他的访问需要被校验
     * version2:
     * 1. 限制所有端点的访问，只允许ACTUATOR_ADMIN访问
     * 2. 允许外部访问静态资源
     * 3. 允许外部访问 '/'
     * 4. 其他的访问需要被校验
     */
 
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        // version1
//        http
//                .authorizeRequests()
//                    .requestMatchers(EndpointRequest.to(ShutdownEndpoint.class))
//                        .hasRole("ACTUATOR_ADMIN")
//                .requestMatchers(EndpointRequest.toAnyEndpoint())
//                    .permitAll()
//                .requestMatchers(PathRequest.toStaticResources().atCommonLocations())
//                    .permitAll()
//                .antMatchers("/")
//                    .permitAll()
//                .antMatchers("/**")
//                    .authenticated()
//                .and()
//                .httpBasic();
 
        // version2
        http
                .authorizeRequests()
                .requestMatchers(EndpointRequest.toAnyEndpoint())
                    .hasRole("ACTUATOR_ADMIN")
                .requestMatchers(PathRequest.toStaticResources().atCommonLocations())
                    .permitAll()
                .antMatchers("/")
                    .permitAll()
                .antMatchers("/**")
                    .authenticated()
                .and()
                .httpBasic();
    }
}
```

```json
# Spring Security Default user name and password
spring.security.user.name=actuator
spring.security.user.password=actuator
spring.security.user.roles=ACTUATOR_ADMIN
```

### 4. Example

```yml
server:
  port: 8089
  servlet:
    context-path: /demo
# 若要访问端点信息，需要配置用户名和密码
spring:
  security:
    user:
      name: xkcoding
      password: 123456
management:
  # 端点信息接口使用的端口，为了和主系统接口使用的端口进行分离
  server:
    port: 8090
    servlet:
      context-path: /sys
  # 端点健康情况，默认值"never"，设置为"always"可以显示硬盘使用情况和线程情况
  endpoint:
    health:
      show-details: always
  # 设置端点暴露的哪些内容，默认["health","info"]，设置"*"代表暴露所有可访问的端点
  endpoints:
    web:
      exposure:
        include: '*'

```

```json
spring.application.name=@project.name@
# /health端点 暴露详细信息
management.endpoint.health.show-details=always
# "*" 代表暴露所有的端点 如果指定多个端点，用","分开
management.endpoints.web.exposure.include=*
# 赋值规则同上
management.endpoints.web.exposure.exclude=
# 正式应用 慎重启用
management.endpoint.shutdown.enabled=true
# 为指标设置一个名为的Tag，Tag是Prometheus提供的一种能力，从而实现更加灵活的筛选。
management.metrics.tags.application=${spring.application.name}

info.app.encoding=@project.build.sourceEncoding@
info.app.java.source=@java.version@
info.app.java.target=@java.version@
info.app.name=@project.name@
info.app.description=@project.description@
info.app.version=@project.version@

# Spring Security 配置
spring.security.user.name=actuator
spring.security.user.password=actuator
spring.security.user.roles=ACTUATOR_ADMIN
```

```java
@Component
public class CustomHealthIndicator extends AbstractHealthIndicator {

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        // 使用builder 来创建健康状态信息
        // 如果你throw 了一个 exception，那么status 就会被置为DOWN，异常信息会被记录下来
        builder.up()
                .withDetail("app", "这个项目很健康")
                .withDetail("error", "Nothing, I'm very good");
    }
}
```

### Resource

- https://ricstudio.top/archives/spring_boot_actuator_learn

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/springactuator/  

