# frameSpring


> Spring 框架指的都是 Spring Framework，它是很多模块的集合，使用这些模块可以很方便地协助我们进行开发。这些模块是：核心容器、数据访问/集成,、Web、AOP（面向切面编程）、工具、消息和测试模块。比如：Core Container 中的 Core 组件是Spring 所有组件的核心，Beans 组件和 Context 组件是实现IOC和依赖注入的基础，AOP组件用来实现面向切面编程。
>
> Spring 官网列出的 Spring 的 6 个特征:
>
> - **核心技术** ：依赖注入(DI)，AOP，事件(events)，资源，i18n，验证，数据绑定，类型转换，SpEL。
> - **测试** ：模拟对象，TestContext框架，Spring MVC 测试，WebTestClient。
> - **数据访问** ：事务，DAO支持，JDBC，ORM，编组XML。
> - **Web支持** : Spring MVC和Spring WebFlux Web框架。
> - **集成** ：远程处理，JMS，JCA，JMX，电子邮件，任务，调度，缓存。
> - **语言** ：Kotlin，Groovy，动态语言。

### 1. Component

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210619223017545.png)

- **Spring Core：** 基础,可以说 Spring 其他所有的功能都需要依赖于该类库。主要提供 IoC 依赖注入功能。
- **Spring  Aspects** ： 该模块为与AspectJ的集成提供支持。
- **Spring AOP** ：提供了面向切面的编程实现。
- **Spring JDBC** : Java数据库连接。
- **Spring JMS** ：Java消息服务。
- **Spring ORM** : 用于支持Hibernate等ORM工具。
- **Spring Web** : 为创建Web应用程序提供支持。
- **Spring Test** : 提供了对 JUnit 和 TestNG 测试的支持。

### Resouce

- [Spring官网](https://spring.io/)、[Spring系列主要项目](https://spring.io/projects)、[Spring官网指南](https://spring.io/guides)、[官方文档](https://spring.io/docs/reference)
- [spring-framework-reference](https://docs.spring.io/spring/docs/5.0.14.RELEASE/spring-framework-reference/index.html)
- [Spring Framework 4.3.17.RELEASE API](https://docs.spring.io/spring/docs/4.3.17.RELEASE/javadoc-api/)

-  [极客学院Spring Wiki](http://wiki.jikexueyuan.com/project/spring/transaction-management.html)
-  [Spring W3Cschool教程 ](https://www.w3cschool.cn/wkspring/f6pk1ic8.html)

- [自己动手实现的 Spring IOC 和 AOP - 上篇](http://www.coolblog.xyz/2018/01/18/自己动手实现的-Spring-IOC-和-AOP-上篇/)

- [自己动手实现的 Spring IOC 和 AOP - 下篇](http://www.coolblog.xyz/2018/01/18/自己动手实现的-Spring-IOC-和-AOP-下篇/)

 - [spring-core](https://github.com/seaswalker/Spring/blob/master/note/Spring.md)
- [spring-aop](https://github.com/seaswalker/Spring/blob/master/note/spring-aop.md)
- [spring-context](https://github.com/seaswalker/Spring/blob/master/note/spring-context.md)
- [spring-task](https://github.com/seaswalker/Spring/blob/master/note/spring-task.md)
- [spring-transaction](https://github.com/seaswalker/Spring/blob/master/note/spring-transaction.md)
- [spring-mvc](https://github.com/seaswalker/Spring/blob/master/note/spring-mvc.md)
- [guava-cache](https://github.com/seaswalker/Spring/blob/master/note/guava-cache.md)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/framespring/  

