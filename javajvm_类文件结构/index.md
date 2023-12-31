# JavaJVM_类文件结构


>**各种不同平台的虚拟机与所有平台都统一使用的程序存储格式——字节码是构成平台无关性的基石。****实现语言无关性的基础仍然是虚拟机和字节码存储格式。** Java虚拟机不和包含Java在内的任何语言绑定，它只和“Class 文件”这种特定的二进制文件格式所关联，Class 文件中包含了 Java 虚拟机指令集和符号表以及若干其他辅助信息。基于安全方面的考虑，Java 虚拟机规范要求在 Class 文件中使用许多强制性的语法和结构化约束，但任何一门功能性语言都可以表示为一个能被 Java 虚拟机所接受的有效的 Class 文件。**作为一个通用的、机器无关的执行平台，任何其他语言的实现者都可以将 Java 虚拟机作为语言的产品交付媒介。**Java 语言中的各种变量、关键字和运算符号的语义最终都是由多条字节码命令组合而成的，因此字节码命令所能提供的语义描述能力肯定会比 Java 语言更加强大。因此，有一些 Java 语言本身无法有效支持的语言特性不代表字节码本身无法有效支持，这也为其他语言实现一些有别于 Java 的语言特性提供了基础。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210714092415783.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210714092548751.png)

### 1. class 字节码文件介绍

#### .1. 字节码文件

> Java所有的`指令大概有 200 个左右`，`一个字节（8位）可以存储 256 种不同的信息`，我们将一个这样的字节称为字节码（ByteCode）。
>
> 而` class 文件便是一组以 8 位字节为基础单位流的二进制流`，各个数据项目严格按照顺序紧凑地排列在 class 文件之中，`中间没有添加任何分隔符`，所以整个class 文件中存储的内容几乎都是程序运行的必要数据，没有任何冗余。当遇到需要占用 8 位字节以上空间的数据项时，则会`按照高位在前的方式分割成若干个 8 位字节进行存储`。

```java
public class ClassTest {
    private static int i = 0;

    public static void main(String[] args) {
        System.out.println(i);
    }
}
```

我们将生成的class 文件，通过十六进制编辑器打开（在IDEA中，可以下载HexView插件，安装完成后，选择这个class文件，右键 HexView）

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190903214808635-2060178663.png)

　　![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190903214850020-763386845.png)

#### .2. javap 命令

```shell
javap <options> <classes>　　#通过 javap -help 命令，可以查看相关参数作用：
```

　　![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190909223213546-1910047276.png)

-  我们将 ClassTest.class 文件，通过 javap -v ClassTest.class 命令，执行后如下：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190909223340407-810991171.png)

### 2. 无符号数和表

Class 文件采用一种类似于 C 语言结构体的伪结构来存储，这种伪结构只有两种数据类型：**无符号数和表**。

- **无符号数**: 这是一种基本数据类型，以 u1,u2,u4,u8 来分别代表 1个字节、2个字节、4个字节、8个字节的无符号数，无符号数可以用来描述数字、索引引用、数量值或按照 UTF-8 编码构成的字符串值。
- **表**: 表是由多个无符号数或其它表作为数据项所构成的复合数据类型，所有表都习惯行的以“_info”结尾。表用于描述有层次关系的复合结构数据。

整个 Class 文件本质上就是一张表，结构如下：

　　![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190905081737355-432037429.png)

### 3. 魔数

> 每个 class 文件的`头 4 个字节称为魔数（Magic Number）`，它的唯一作用是：`标识该文件是一个Java类文件`。如果没有识别到该标志，则说明该文件不是Java类文件或者文件已受损。由上图，我们可以看到前 4 个字节是 **cafe babe**。这是 Gosling 定义的一个魔法数，意思是 Coffee Baby。其实很多`文件存储标准中都使用魔数进行身份识别，比如图片gif或者jpeg`，使用魔数而不是使用扩展名来进行识别主要是基于安全考虑，因为文件扩展名可以任意的改动。

### 4. Class 文件的版本号

紧随魔数的 4 个字节存储的是 class 文件的版本号：`第 5 和第 6 个字节是次版本号（Minor Version），第 7 和第 8 个字节是主版本号（Major Version）`。

Java的版本号是从 45 开始的，JDK1.1 之后的每个 JDK 大版本发布主版本号向上加1（JDK1.0~JDK1.1使用了45.0~45.3的版本号），高版本的 JDK 能向下兼容以前版本的 Class 文件，但不能运行以后版本的 Class 文件，即使文件格式未发生变化。

### 5. 常量池

紧随主版本号的是常量池入口，`是class文件中第一个出现的表类型数据项目`，也是`占用Class文件空间最大的项目之一`，更是`Class文件结构中与其它项目关联最多的数据类型`。

#### .1. 常量池容量计数值

因为`常量池中常量的数量是不固定的`，所以`在常量池的入口要放置一项 u2 类型的数据`，代表常量池容量计数值（constant_pool_count）。

PS：注意，`常量池容量计数值是从 1 开始的`，而不是从 0 开始。将 0 空出来，是为了满足后面某些指向常量池的索引值的数据在特定情况下需要表达“不引用任何一个常量池项目”的意思。`Class 文件结构中，只有常量池的容量是从 1 开始的，其它的集合类型，都是从 0 开始的`。

看上图的十六进制文件，常量池容量计数值为：0x0025，即十进制 37。这就表示常量池中有 36 项常量，索引值分别为 1~36(通过上面javap命令生成字节码文件可以很明显看出来有36个)

　　![img](https://img2018.cnblogs.com/blog/1120165/201909/1120165-20190909221951117-1133475480.png)



#### .2. 常量池内容

常量池主要存放两大类常量：

- 字面量（Literal）：`字面量比较接近于 Java 语言层面的常量概念`，比如` 文本字符串、被声明为 final 的常量值`等。

- 符号引用（Symbolic References）：符号引用属于编译原理方面的概念，包括下面三类常量：
  - `类和接口的权限定名`（Fully Qualified Name）
  - `字段的名称和描述符`（Descriptor）
  - `方法的名称和描述符`

需要说明的是，Java代码在进行javac 编译的时候，并不像 C 和 C++ 那样有“连接”这一步骤，而是`在虚拟机加载 Class 文件的时候进行动态连接。`也就是说，在 Class 文件中`不会保存各个方法和字段的最终内存布局信息`，因此这些字段和方法的符号引用不经过转换的话是无法被虚拟机使用的。`当虚拟机运行时，需要从常量池获得对应的符号引用，再在类创建时或运行时解析并翻译到具体的内存地址之中`。`常量池中的每一项内容都是一个表`，在JDK1.8中共有 14 种结构各不相同的表结构数据，每个表结构第一位是一个 u1 类型的标志位（tag，取值为1 到 18，缺少标志为 2、13、14、17 的数据类型）。代表当前这个常量属于哪种常量类型。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190910074838462-1324015738.png)

 接着看十六进制文件，紧跟常量池数量的十六进制是0x0a，这是一个标志位，0x0a的十进制数是10，查看常量池的项目表接口，表示的类型是 CONSTANT_Methodref_info。

 　![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190924224345474-1115190622.png)

 也就是说，接下来的u2类型0x0006，其十进制值为6，紧跟后面的u2类型十六进制为0x0017，其十进制值为23，这都是两个索引值，分别指向第索引值为6的常量和索引值为23的常量。

　　![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190924225521565-1815207971.png)

 　![](https://img2018.cnblogs.com/blog/1120165/201909/1120165-20190924225537140-277540503.png)

### 6. 访问标志

`常量池结束后的两个字节表示访问标志（access_flags）`，这个标识用于`识别一些类或接口层次的访问信息。`

包括：`这个 Class 是类还是接口`；`是否定义为 public 类型`，`是否定义为 abstract 类型`；`如果是类的话，是否被声明为 final 等`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20190924225921233-549353168.png)

 上表定义了 8 个标志位，但是我们说访问标志是一个 u2 类型，一共有 32 个标志位可以使用，没有定义的标志位一律为 0 。

### 7. 类索引、父类索引和接口索引集合

类索引、父类索引和接口索引按顺序排列在访问标志之后。

- 类索引：用于`确定这个类的全限类名` ，是一个 u2 类型的数据。
- 父类索引：用于`确定这个类的父类全限类名`，也是一个 u2 类型的数据。因为Java是单继承的，除了 java.lang.Object 类以外，所有的类都有父类。所以，除了Object 类以外，所有Java类的父类索引都不为0.
- 接口索引：用于`描述这个类实现了哪些接口`，是一组 u2 类型的数据集合，第一项为 u2 类型的接口计数器，表示实现接口的个数。如果没有实现任何接口，则为0。

### 8. 字段表集合

字段表（field_info）：描述接口或类中声明的变量。（不包括方法内部声明的变量）描述的信息包括：

　　①、字段的作用域（public，protected，private修饰）

　　②、是类级变量还是实例级变量（static修饰）

　　③、是否可变（final修饰）

　　④、并发可见性（volatile修饰，是否强制从主从读写）

　　⑤、是否可序列化（transient修饰）

　　⑥、字段数据类型（8种基本数据类型，对象，数组等引用类型）

　　⑦、字段名称　　

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20191016204550433-1527259240.png)

　　![img](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20191016204712857-582810092.png)

### 9. 方法表集合

Class 文件存储格式中对方法的描述和字段的描述基本上是一致的。也是依次包括：`访问标志（access_flags）、名称索引（name_index）、描述符索引（descriptor_index）、属性表集合数量（attributes_count）、属性表集合（attributes）`

　　![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20191016205805437-1637256926.png)

- 方法访问标志如下（access_flags）：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20191016205843354-2103960452.png)

### 10. 属性表集合

在前面介绍的字段表集合、方法表集合中都包括了属性表集合（attributes），其实就是引用的这里。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20191016210231595-1119218249.png)

 对于每一个属性，它的名称要从常量池中引用一个 CONSTANT_Utf8_info 类型的常量来表示，其属性值的结构则是完全自定义的，只需要说明属性值所占用的位数长度即可。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1120165-20191016210437172-1302665450.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javajvm_%E7%B1%BB%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84/  

