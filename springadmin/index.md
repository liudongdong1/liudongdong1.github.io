# SpringAdmin


> Spring Boot Actuator 的使用，Spring Boot Actuator 提供了对单个 Spring Boot 的监控，信息包含：应用状态、内存、线程、堆栈等等，比较全面的监控了 Spring Boot 应用的整个生命周期。
>
> Spring Boot Admin 是一个`管理和监控 Spring Boot 应用程序的开源软件`。每个应用都认为是一个客户端，通过 HTTP 或者使用 Eureka 注册到 admin server 中进行展示，Spring Boot Admin UI 部分使用 VueJs 将数据展示在前端。

### 1. Admin Server

```xml
<dependencies>
  <dependency>
    <groupId>de.codecentric</groupId>
    <artifactId>spring-boot-admin-starter-server</artifactId>
    <version>2.1.0</version>
  </dependency>
  <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
  </dependency>
</dependencies>
```

```
server.port=8000
```

```java
@Configuration
@EnableAutoConfiguration
@EnableAdminServer
public class AdminServerApplication {

  public static void main(String[] args) {
    SpringApplication.run(AdminServerApplication.class, args);
  }
}
```

### 2. Admin Client

```xml
<dependencies>
    <dependency>
      <groupId>de.codecentric</groupId>
      <artifactId>spring-boot-admin-starter-client</artifactId>
      <version>2.1.0</version>
    </dependency>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

```json
server.port=8001
spring.application.name=Admin Client
spring.boot.admin.client.url=http://localhost:8000    #配置 Admin Server 的地址
management.endpoints.web.exposure.include=*  # 打开客户端 Actuator 的监控。
```

```yaml
server:
  port: 8089
  servlet:
    context-path: /demo
spring:
  application:
    # Spring Boot Admin展示的客户端项目名，不设置，会使用自动生成的随机id
    name: spring-boot-demo-admin-client
  boot:
    admin:
      client:
        # Spring Boot Admin 服务端地址
        url: "http://localhost:8000/"
        instance:
          metadata:
            # 客户端端点信息的安全认证信息
            user.name: ${spring.security.user.name}
            user.password: ${spring.security.user.password}
  security:
    user:
      name: xkcoding
      password: 123456
management:
  endpoint:
    health:
      # 端点健康情况，默认值"never"，设置为"always"可以显示硬盘使用情况和线程情况
      show-details: always
  endpoints:
    web:
      exposure:
        # 设置端点暴露的哪些内容，默认["health","info"]，设置"*"代表暴露所有可访问的端点
        include: "*"

```



### 3.监控微服务

> 如果我们使用的是单个 Spring Boot 应用，就需要在每一个被监控的应用中配置 Admin Server 的地址信息；如果应用都注册在 Eureka 中就不需要再对每个应用进行配置，Spring Boot Admin 会自动从注册中心抓取应用的相关信息。

- 客户端&服务端依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

- 服务器段

```java
@Configuration
@EnableAutoConfiguration
@EnableDiscoveryClient
@EnableAdminServer
public class SpringBootAdminApplication {
    public static void main(String[] args) {
        SpringApplication.run(SpringBootAdminApplication.class, args);
    }

    @Configuration
    public static class SecurityPermitAllConfig extends WebSecurityConfigurerAdapter {
        @Override
        protected void configure(HttpSecurity http) throws Exception {
            http.authorizeRequests().anyRequest().permitAll()  
                .and().csrf().disable();
        }
    }
}
```

- 客户端配置

```yaml
eureka:   
  instance:
    leaseRenewalIntervalInSeconds: 10
    health-check-url-path: /actuator/health
    metadata-map:
      startup: ${random.int}    #needed to trigger info and endpoint update after restart
  client:
    registryFetchIntervalSeconds: 5
    serviceUrl:
      defaultZone: ${EUREKA_SERVICE_URL:http://localhost:8761}/eureka/

management:
  endpoints:
    web:
      exposure:
        include: "*"  
  endpoint:
    health:
      show-details: ALWAYS
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/springadmin/  

