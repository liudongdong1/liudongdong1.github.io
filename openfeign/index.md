# OpenFeign


> [Feign](https://github.com/OpenFeign/feign) is a Java to HTTP client binder inspired by [Retrofit](https://github.com/square/retrofit), [JAXRS-2.0](https://jax-rs-spec.java.net/nonav/2.0/apidocs/index.html), and [WebSocket](http://www.oracle.com/technetwork/articles/java/jsr356-1937161.html). Feign's first goal was reducing the complexity of binding [Denominator](https://github.com/Netflix/Denominator) uniformly to HTTP APIs regardless of [ReSTfulness](http://www.slideshare.net/adrianfcole/99problems). 
>
> 使用 OpenFeign 的 Spring 应用架构一般分为三个部分，分别为`服务注中心`、`服务提供者`和`服务消费者`。服务提供者向服务注册中心注册自己，然后服务消费者通过 OpenFeign 发送请求时， OpenFeign 会向服务注册中心获取关于服务提供者的信息，然后再向服务提供者发送网络请求。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210627212904996.png)

### 1. 注解

- **@EnableFeignClients** 注解的定义如下：

```java
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@Documented
@Import(FeignClientsRegistrar.class)
public @interface EnableFeignClients {

    //下面三个函数都是为了指定需要扫描的包
	String[] value() default {};
	String[] basePackages() default {};
	Class<?>[] basePackageClasses() default {};

    //指定自定义feignclient的自定义配置，可以配置Decoder、Encoder、Contract等组件，FeignClientsConfiguration是默认的配置
	Class<?>[] defaultConfiguration() default {};

	Class<?>[] clients() default {};

}
```

> OpenFeign 通过 FeignClientsRegistrar 来处理 @FeignClient 注解修饰的 FeignClient 接口类，将这些接口类的 BeanDefinition 注册到 Spring 容器中，这样就可以使用 @Autowired 等方式来自动装载这些 FeignClient 接口类的 Bean 实例。

### Resource

- https://blog.csdn.net/caychen/article/details/107717311
- https://github.com/OpenFeign/feign

---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/openfeign/  

