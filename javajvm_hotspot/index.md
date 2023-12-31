# JavaJVM_HotSpot


#### .1. 对象创建

> HotSpot VM遇到一条`new 类型（）指令时`，①首先在`常量池中是否`有这个类的符号引用(`符号引用代表类是否已被加载、解析和初始化过`)；②如果没有检测到此类的符号引用就必须先执行相应的`类加载过程`；3 类加载通过后`从Java堆中分配确定大小的内存`(分配内存的过程是一个并发进行的过程)；④内存分配完成后，将`分配到的内存空间初始化为零值`(对象头不初始化--对象头在下面介绍)；5 虚拟机对对象进行必要的设置，并将这些`设置信息存放在对象头中`；⑥此时就得到了`从JVM视角看到的对象`；⑦`从java程序的角度来看对象才刚刚创建---<init>方法还没有执行，所有的字段都还为0，执行init方法后真正的可用对象产生`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210604095009684.png)

#### .2. 对象内存分布

> Java程序需要通过栈上的reference数据来操作堆上的具体对象，reference类型在Java虚拟机规范中之规定了一个指向对象的引用，而应用应该怎样去定位、访问堆中的对象的具体位置就需要用到对象访问定位方式。

> 目前主流访问方式有使用句柄和直接指针两种。句柄---从Java堆中划分一块内存作为句柄池，refernece中存储对象的句柄地址，句柄中包含了对象实例数据与类型数据各自的具体地址信息，修改代价小；直接指针---reference中直接存储堆中对象地址，如图二所示，其访问速度更快，节省指针定位的时间开销，HotSpot VM就是采用直接指针方式。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210604100851032.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210604101002942.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javajvm_hotspot/  

