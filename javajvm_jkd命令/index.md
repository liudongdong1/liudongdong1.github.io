# JavaJVM_JDK命令


>给一个系统定位问题的时候，知识、经验是关键基础，数据是依据，工具是运用知识处理数据的手段。这里说的数据包括但不限于异常堆栈、虚拟机运行日志、垃圾收集器日志、线程快照（`threaddump/javacore`文件）、堆转储快照（`heapdump/hprof`文件）等。

| 名称   | 主要作用                                                     |
| ------ | ------------------------------------------------------------ |
| jps    | JVM Process Status Tool，显示指定系统内所有的HotSpot虚拟机进程 |
| jstat  | JVM Statistics Monitoring Tool，用于收集HotSpot虚拟机各方面的运行数据 |
| jinfo  | Configuration Info for Java，显示虚拟机配置信息              |
| jmap   | Memory Map for Java，生成虚拟机的内存转储快照（heapdump文件） |
| jhat   | JVM Heap Dump Browser，用于分析heapdump文件，它会建立一个HTTP/HTML服务器，让用户可以在浏览器上查看分析结果 |
| jstack | Stack Trace for Java，显示虚拟机的线程快照                   |

### 1. jps

> jps：虚拟机`进程状况工具`

```shell
$ jps [options] [hostid]

$ jps -l
2388 D:\Develop\glassfish\bin\..\modules\admin-cli.jar
2764 com.sun.enterprise.glassfish.bootstrap.ASMain
3788 sun.tools.jps.Jps
```

| 选项 | 作用                                                   |
| ---- | ------------------------------------------------------ |
| -q   | 只输出LVMID，省略主类的名称                            |
| -m   | 输出虚拟机进程启动时传递给main()函数的参数             |
| -l   | 输出`主类的全名`，如果进程`执行的是Jar包，输出Jar路径` |
| -v   | 输出虚拟机进程启动时的JVM参数                          |

### 2. jstat

> `jstat`（`JVM Statistics Monitoring Tool`）是用于**监视虚拟机各种运行状态信息的命令行工具**。它可以`显示本地或者远程虚拟机进程中的类加载、内存、垃圾收集、即时编译等运行时数据`，在没有GUI图形界面、只提供了纯文本控制台环境的服务器上，它将是运行期定位虚拟机性能问题的常用工具。

```shell
$ jstat [options vmid [interval[s|ms]] [count] ]
$ jstat -gc 2764 250 20
#每250毫秒查询一次进程2764垃圾收集状况，一共查询20次
```

| 选项              | 作用                                                         |
| ----------------- | ------------------------------------------------------------ |
| -class            | 监视`类装载，卸载数量`，`总空间`以及类`装载所耗费的时间`     |
| -gc               | 监视`Java堆状况`，包括Eden区、2个survivor区、老年代、永久代等的容量、已用空间、GC时间合计等信息 |
| -gccapacity       | 监视内容与-gc基本相同，但`输出主要关注Java堆各个区域的最大和最小空间` |
| -gcutil           | 监视内容与-gc基本相同，但输出主要关注已使用空间占总空间的百分比 |
| -gccause          | 与-gcutil功能一样，但是会额外输出导致上一次GC产生的原因      |
| -gcnew            | `监视新生代GC情况`                                           |
| -gcnewcapacity    | 监视内容与-gcnew基本相同，但输出主要关注使用到的最大和最小空间 |
| -gcold            | 监视老年代GC情况                                             |
| -gcoldcapacity    | 监视内容与-gcold基本相同，但输出主要关注使用到的最大和最小空间 |
| -gcpermcapacity   | 输出永久代使用到的最大和最小空间                             |
| -complier         | 输出JIT 编译器编译过的方法、耗时的信息                       |
| -printcompilation | 输出已经被JIT编译的方法                                      |

### 3. jinfo

> jinfo：Java配置信息工具`jinfo`（Configuration Info for Java）的作用是**实时查看和调整虚拟机各项参数**。使用`jps`命令的`-v参数`可以**查看虚拟机启动时显式指定的参数列表**，但**如果想知道未被显式指定的参数的系统默认值，除了去找资料外，就只能使用`jinfo`的`-flag`选项进行查询**了（如果只限于`JDK 6`或以上版本的话，使用`java-XX：+PrintFlagsFinal`查看参数默认值也是一个很好的选择）。`jinfo`还可以使用`-sysprops`选项把虚拟机进程的`System.getProperties()`的内容打印出来。这个命令在`JDK 5`时期已经随着Linux版的`JDK`发布，当时只提供了信息查询的功能，`JDK 6`之后，`jinfo`在Windows和Linux平台都有提供，并且加入了在运行期修改部分参数值的能力（可以使用-flag[+|-]name或者-flag name=value在运行期修改一部分运行期可写的虚拟机参数值）。在`JDK 6`中，`jinfo`对于Windows平台功能仍然有较大限制，只提供了最基本的`-flag`选项。

```shell
$jinfo [option] pid

$jinfo -flag CMSInitiatingOccupancyFraction 1444
#查询CMSInitiatingOccupancyFraction参数值:
-XX:CMSInitiatingOccupancyFraction=85
```

### 4. jmap

> `jmap`（Memory Map for Java）命令用于`生成堆转储快照`（一般称为`heapdump`或`dump`文件）。如果不使用`jmap`命令，要想获取Java堆转储快照也还有一些比较“暴力”的手段：譬如在第2章中用过的`-XX：+HeapDumpOnOutOfMemoryError`参数，可以让虚拟机在内存溢出异常出现之后自动生成堆转储快照文件，通过`-XX：+HeapDumpOnCtrlBreak`参数则可以使用`[Ctrl]+[Break]`键让虚拟机生成堆转储快照文件，又或者在Linux系统下通过Kill-3命令发送进程退出信号“恐吓”一下虚拟机，也能顺利拿到堆转储快照。
>
> `jmap`的作用并**不仅仅是为了获取堆转储快照，它还可以查询finalize执行队列、Java堆和方法区的详细信息，如空间使用率、当前用的是哪种收集器等**。
>
> 和`jinfo`命令一样，`jmap`有部分功能在Windows平台下是受限的，除了生成堆转储快照的-dump选项和用于查看每个类的实例、空间占用统计的`-histo`选项在所有操作系统中都可以使用之外，其余选项都只能在`Linux/Solaris`中使用;

```shell
$jmap [option] vmid

#使用jmap生成dump文件：
$jmap -dump:format=b,file=eclipse.bin 3500
Dumping heap to C:\Users\IcyFenix\eclipse.bin ...
Heap dump file created
```

| 选项           | 作用                                                         |
| -------------- | ------------------------------------------------------------ |
| -dump          | 生成堆转储快照，格式为：`-dump:[live,]format=b,file=<filename>`，其中`live子参数说明是否只dump出存活的对象` |
| -finalizerinfo | 显示`在F-Queue队列等待Finalizer线程执行finalizer方法的对象`  |
| -heap          | `显示Java堆详细信息`，如使用哪种回收器，参数配置，分代状况等 |
| -histo         | `显示堆中对象的统计信息`，`GC使用的算法`，`heap的配置及wise heap的使用情况`，可以用此来判断内存目前的使用情况以及垃圾回收情况 |
| -permstat      | 已`ClassLoader为统计口径显示永久代内存状态`                  |
| -F             | 当`-dump没有响应时，强制生成dump快照`                        |

### 5. jhat

> jhat：虚拟机堆转储快照分析工具`JDK`提供`jhat`（`JVM Heap Analysis Tool`）命令与`jmap`搭配使用，来分析`jmap`生成的堆转储快照。`jhat`内置了一个微型的`HTTP/Web`服务器，`生成堆转储快照的分析结果后，可以在浏览器中查看`。不过实事求是地说，**在实际工作中，除非手上真的没有别的工具可用，否则多数人是不会直接使用`jhat`命令来分析堆转储快照文件的**，主要原因有两个方面。一是一般`不会在部署应用程序的服务器上直接分析堆转储快照`，即使可以这样做，也会尽量将堆转储快照文件复制到其他机器上进行分析，因为`分析工作是一个耗时而且极为耗费硬件资源的过程`，既然都要在其他机器上进行，就没有必要再受命令行工具的限制了。另外一个原因是`jhat`的分析功能相对来说比较简陋，后文将会介绍到的`VisualVM`，以及专业用于分析堆转储快照文件的`Eclipse Memory Analyzer`、`IBM HeapAnalyzer`等工具，都能实现比`jhat`更强大专业的分析功能。

```shell
$ jhat [option] [dumpfile]

# 使用jhat分析dump文件
$ jhat eclipse.bin
Reading from eclipse.bin...
Dump file created Fri Nov 19 22:07:21 CST 2010
Snapshot read, resolving...
Resolving 1225951 objects...
Chasing references, expect 245 dots....
Eliminating duplicate references...
Snapshot resolved.
Started HTTP server on port 7000
Server is ready.
```

### 6. jstack

> `jstack`（`Stack Trace for Java`）命令用于**生成虚拟机当前时刻的线程快照（一般称为`threaddump`或者`javacore`文件）**。**线程快照就是当前虚拟机内每一条线程正在执行的方法堆栈的集合，生成线程快照的目的通常是定位线程出现长时间停顿的原因，如线程间死锁、死循环、请求外部资源导致的长时间挂起等，都是导致线程长时间停顿的常见原因**。**线程出现停顿时通过`jstack`来查看各个线程的调用堆栈，就可以获知没有响应的线程到底在后台做些什么事情，或者等待着什么资源**。

```shell
$ jstack [option] vmid

$ jstack -l 3500
2010-11-19 23:11:26
Full thread dump Java HotSpot(TM) 64-Bit Server VM (17.1-b03 mixed mode):
"[ThreadPool Manager] - Idle Thread" daemon prio=6 tid=0x0000000039dd4000 nid= 0xf50 in Object.wait() [0x000000003c96f000]
    java.lang.Thread.State: WAITING (on object monitor)
        at java.lang.Object.wait(Native Method)
        - waiting on <0x0000000016bdcc60> (a org.eclipse.equinox.internal.util.impl.tpt.threadpool.Executor)
        at java.lang.Object.wait(Object.java:485)
        at org.eclipse.equinox.internal.util.impl.tpt.threadpool.Executor.run (Executor. java:106)
        - locked <0x0000000016bdcc60> (a org.eclipse.equinox.internal.util.impl.tpt.threadpool.Executor)
    Locked ownable synchronizers:
    	- None
```

| 选项 | 作用                                        |
| ---- | ------------------------------------------- |
| -F   | 当正常输出请求不被响应时，强制输出线程堆栈  |
| -l   | 除堆栈外，显示关于锁的附加信息              |
| -m   | 如果调用到本地方法的话，可以显示C/C++的堆栈 |

### 7. HSDIS

>HSDIS是一个Sun官方推荐的`HotSpot虚拟机JIT编译代码的反汇编插件`，它包含在HotSpot虚拟机的源码之中，但没有提供编译后的程序。它的作用是让`HotSpot的-XX：+PrintAssembly指令调用它来把动态生成的本地代码还原为汇编代码输出`，同时还`生成了大量非常有价值的注释`，这样我们就可以通过输出的代码来分析问题。

### 8. 可视化JConsole

>`JConsole`（Java Monitoring and Management Console）是一款基于`JMX`（`Java Manage-ment Extensions`）的可视化监视、管理工具。它的**主要功能是通过`JMX`的`MBean`（Managed Bean）对系统进行信息收集和参数动态调整**。`JMX`是一种开放性的技术，不仅可以用在虚拟机本身的管理上，还可以运行于虚拟机之上的软件中，典型的如中间件大多也基于`JMX`来实现管理与监控。虚拟机对`JMXMBean`的访问也是完全开放的，可以使用代码调用`API`、支持`JMX`协议的管理控制台，或者其他符合`JMX`规范的软件进行访问。

通过`JDK/bin`目录下的`jconsole.exe`启动`JCon-sole`后，会自动搜索出本机运行的所有虚拟机进程，而不需要用户自己使用`jps`来查询，双击选择其中一个进程便可进入主界面开始监控。`JMX`支持跨服务器的管理，也可以使用下面的“远程进程”功能来连接远程服务器，对远程虚拟机进行监控。

主界面里共包括“概述”“内存”“线程”“类”“`VM`摘要”“`MBean`”六个页签。“概述”页签里显示的是整个虚拟机主要运行数据的概览信息，包括“堆内存使用情况”“线 程”“类”“CPU使用情况”四项信息的曲线图，这些曲线图是后面“内存”“线程”“类”页签的信息汇总。

- “内存”页签的作用**相当于可视化的`jstat`命令**，用于监视被收集器管理的虚拟机内存（被收集器直接管理的Java堆和被间接管理的方法区）的变化趋势。
- **“线程”页签的功能就相当于可视化的`jstack`命令**，遇到**线程停顿的时候可以使用这个页签的功能进行分析**。线程长时间停顿的主要原因有等待外部资源（数据库连接、网络资源、设备资源等）、死循环、锁等待等。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210713085611001.png)

### 9. 可视化VisualVM

>`VisualVM`（`All-in-One Java Troubleshooting Tool`）是功能最强大的运行监视和故障处理程序之一，曾经在很长一段时间内是Oracle官方主力发展的虚拟机故障处理工具。Oracle曾在`VisualVM`的软件说明中写上了“`All-in-One`”的字样，预示着它除了常规的运行监视、故障处理外，还将提供其他方面的能力，譬如性能分析（Profiling）。`VisualVM`的性能分析功能比起`JProfiler`、`YourKit`等专业且收费的`Profiling`工具都不遑多让。而且相比这些第三方工具，`VisualVM`还有一个很大的优点：不需要被监视的程序基于特殊Agent去运行，因此它的通用性很强，对应用程序实际性能的影响也较小，使得它可以直接应用在生产环境中。

> VisualVM可以做到以下：
>
> - 显示虚拟机进程以及进程的配置、环境信息、jps、jinfo。
> - 监视应用程序的cpu、GC、堆、方法区以及线程的信息（jstat、jstack）。
> - dump以及分析堆转存储快照（jmap、jhat）。
> - 还有很多其他的功能。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210713090117077.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210713085935854.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/javajvm_jkd%E5%91%BD%E4%BB%A4/  

