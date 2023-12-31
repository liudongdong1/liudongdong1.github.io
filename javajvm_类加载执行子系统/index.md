# JavaJVM_类加载执行子系统


> 在Class文件格式和执行引擎这部分中，用户的程序能直接影响的内容并不太多，`Class文件以何种格式存储，类型何时加载、如何连接，以及虚拟机如何执行字节码指令`等都是`由虚拟机直接控制的行为`，用户程序无法对其进行改变。能通过程序进行操作的，主要是`字节码生成`与`类加载器`这两部分的功能

#### 1. Tomcat 类加载器架构

> - 部署在`同一个服务器上的两个Web应用程序`所使用的Java类库可以`实现相互隔离`。
> - 部署在`同一个服务器上的两个Web应用程序`所使用的Java`类库可以互相共享`。
> - 服务器需要尽可能地保证自身的安全不受部署的Web应用程序的影响。一般来说，`基于安全考虑，服务器所使用的类库与应用程序的类库互相独立。`
> - 支持JSP应用的Web服务器，大多数都需要支持HotSwap功能。

在Tomcat目录结构中，有3组目录("/common/","/server/**","/shared/**")可以存放Java类库，另外还可以加上Web应用程序自身的目录“/WEB-INF/*”，一共4组，把Java类库放置在这些目录中的含义分别如下：

- 放置在“`/common/`”目录中：`类库可被Tomcat和所有的Web应用程序共同使用`。
- 放置在“`/server`”目录中：`类库可被Tomcat使用`，`对所有的Web应用程序都不可见`。
- 放置在“`/shared`”目录中，`类库可被所有的Web应用程序共同使用`，但`对Tomcat自己不可见`。
- 放置在“`/WebApp/WEB-INF`”目录中，`类库仅仅可以被此Web应用程序使用`，对Tomcat和其他Web应用程序都不可见。

在Tomcat中增加了四个自己的自定义类加载器：CommonClassLoader、CatalinaClassLoader、SharedClassLoader和WebappClassLoader。它们分别负责加载/common/**,**/server/,/shared/,/WebApp/WEB-INF/;

#### 2. OSGI: 类加载器架构

> `OSGI 里每个程序模块`（Bundle，就是普通的 jar 包, 只是加入了特殊的头信息，是最小的部署模块）`都会有自己的类加载器`，当需要更换程序时，就连同 Bundle 和类加载器一起替换，是一种网状的加载模型，Bundle 间互相委托加载，并不是层次化的。
>
> `Java 类加载机制的隔离`是通过`不同类加载器加载指定目录`来实现的，类`加载的共享机制`是通过`双亲委派模型来实现`，而` OSGI 实现隔离靠的是每个 Bundle 都自带一个独立的类加载器 ClassLoader。`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210729080551553.png)

1. 首先`检查包名`是否以 java.* 开头，或者`是否在一个特定的配置文件`（org.osgi.framework.bootdelegation）中定义。`如果是，则 bundle 类加载器立即委托给父类加载器（通常是 Application 类加载器）`，如果不是则进入 2
2. `检查是否在 Import-Package、Require-Bundle 委派列表里`，如果是委托给对应 Bundle 类加载器，如果不是，进入 3
3. 检查`是否在当前 Bundle 的 Classpath 里`，如果是使用自己的类加载器加载，如果不是，进入 4
4. 搜索`可能附加在当前 bundle 上的 fragment 中的内部类`，找到则委派给 Fragment bundle 类加载器加载，如果找不到，进入 5
5. 查找`动态导入列表里的 Bundle，委派给对应的类加载器加载`，否则类加载失败



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javajvm_%E7%B1%BB%E5%8A%A0%E8%BD%BD%E6%89%A7%E8%A1%8C%E5%AD%90%E7%B3%BB%E7%BB%9F/  

