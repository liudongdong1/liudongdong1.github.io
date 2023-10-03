# HutoolLearning


> [Hutool](https://hutool.cn/docs/#/core/%E8%AF%AD%E8%A8%80%E7%89%B9%E6%80%A7/HashMap%E6%89%A9%E5%B1%95-Dict)是一个小而全的`Java工具类库`，通过`静态方法封装`，降低相关API的学习成本，提高工作效率，使Java拥有函数式语言般的优雅，让Java语言也可以“甜甜的”。
>
> Hutool中的工具方法来自每个用户的精雕细琢，它`涵盖了Java开发底层代码中的方方面面`，它既是大型项目开发中解决小问题的利器，也是小型项目中的效率担当；
>
> Hutool`是项目中“util”包友好的替代`，它节省了开发人员对项目中公用类和公用工具方法的封装时间，使开发专注于业务，同时可以最大限度的避免封装不完善带来的bug。
>
> 一个Java基础工具类，对`文件、流、加密解密、转码、正则、线程、XML等JDK方法进行封装`，组成`各种Util工具类`，同时提供以下组件:   具体的使用example可以从对应包test文件中查看，或者[官网](https://www.hutool.cn/docs/#/extra/%E6%A8%A1%E6%9D%BF%E5%BC%95%E6%93%8E/%E6%A8%A1%E6%9D%BF%E5%BC%95%E6%93%8E%E5%B0%81%E8%A3%85-TemplateUtil)

| 模块               | 介绍                                                         |
| ------------------ | ------------------------------------------------------------ |
| hutool-aop         | `JDK动态代理封装`，提供`非IOC下的切面支持`                   |
| hutool-bloomFilter | `布隆过滤`，提供一些Hash算法的布隆过滤                       |
| hutool-cache       | 简单`缓存实现`                                               |
| hutool-core        | 核心，包括`Bean操作、日期、各种Util等`                       |
| hutool-cron        | 定时任务模块，提供`类Crontab表达式的定时任务`                |
| hutool-crypto      | `加密解密模块`，提供对称、非对称和摘要算法封装               |
| hutool-db          | JDBC`封装后的数据操作`，基于ActiveRecord思想                 |
| hutool-dfa         | `基于DFA模型的多关键字查找`                                  |
| hutool-extra       | 扩展模块，对第三方封装`（模板引擎、邮件、Servlet、二维码、Emoji、FTP、分词等`） |
| hutool-http        | 基于`HttpUrlConnection的Http客户端封装`                      |
| hutool-log         | 自动`识别日志实现`的日志门面                                 |
| hutool-script      | 脚本`执行封装`，例如Javascript                               |
| hutool-setting     | 功能更强大的`Setting配置文件和Properties封装`                |
| hutool-system      | `系统参数调用封装`（JVM信息等）                              |
| hutool-json        | `JSON实现`                                                   |
| hutool-captcha     | `图片验证码实现`                                             |
| hutool-poi         | 针对`POI中Excel和Word的封装`                                 |
| hutool-socket      | 基于Java的`NIO和AIO的Socket封装`                             |
| hutool-jwt         | `JSON Web Token (JWT)封装实现`                               |

### 1. 导入

```xml
<dependency>
    <groupId>cn.hutool</groupId>
    <artifactId>hutool-all</artifactId>
    <version>5.7.2</version>
</dependency>
```

### 2. BloomFilter

> 布隆过滤器（Bloom Filter）由 `Burton Howard Bloom` 在 1970 年提出，是一种空间效率高的概率型数据结构。它专门用来检测集合中是否存在特定的元素。其实对于判断集合中是否存在某个元素，我们平时都会直接使用比较算法，例如：
>
> - 如果集合用线性表存储，查找的时间复杂度为 O(n)；
> - 如果用平衡 BST（如 AVL树、红黑树）存储，时间复杂度为 O(logn)；
> - 如果用哈希表存储，并用链地址法与平衡 BST 解决哈希冲突（参考 JDK8 的 HashMap 实现方法），时间复杂度也要有O[log(n/m)]，m 为哈希分桶数。
>
> **优点：**
>
> - 不需要存储数据本身，只用比特表示，因此空间占用相对于传统方式有巨大的优势，并且能够保密数据；
> - 时间效率也较高，插入和查询的时间复杂度均为O(k)；
> - 哈希函数之间相互独立，可以在硬件指令层面并行计算。
>
> **缺点：**
>
> - 存在假阳性的概率，`不适用于任何要求 100% 准确率的场景`；
> - `只能插入和查询元素，不能删除元素`，这与产生假阳性的原因是相同的。我们可以简单地想到通过计数（即将一个比特扩展为计数值）来记录元素数，但仍然无法保证删除的元素一定在集合中。

![Package bloomfilter](https://gitee.com/github-25970295/blogpictureV2/raw/master/Package%20bloomfilter-16308271946801.png)

### 3. hutool-socket

> Hutool只针对NIO和AIO做了简单的封装，用于简化Socket异步开发。现阶段，Hutool的socket封装依旧不是一个完整框架或者高效的工具类，不能提供完整的高性能IO功能. [t-io](https://www.t-io.org/) [Voovan](http://www.voovan.org/) [Netty](https://netty.io/) [Mina](http://mina.apache.org/)

![Package aio](https://gitee.com/github-25970295/blogpictureV2/raw/master/Package%20aio.png)

> NIO为我们提供了更好的解决方案，采用选择器（Selector）找出已经准备好读写的socket，并按顺序处理，基于通道（Channel）和缓冲区（Buffer）来传输和保存数据。为了实现Selector管理多个SocketChannel，必须将多个具体的SocketChannel对象注册到Selector对象，并声明需要监听的事件

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210905163638622.png)

### 4. 加密算法

![Package crypto](https://gitee.com/github-25970295/blogpictureV2/raw/master/Package%20crypto.png)

### 5. Captcha

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/Package%20generator.png)

### 6. Cache 工具

- 冷热数据LRUcache这里并没有实现

![](https://gitee.com/github-25970295/picture2023/raw/master/image-20230314000848927.png)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/hutoollearning/  

