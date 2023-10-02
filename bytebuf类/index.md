# ByteBuf类


> 网络数据的基本单位总是字节。`Java NIO 提供了ByteBuffer 作为它的字节容器`，但是这个类使用起来过于复杂，而且也有些繁琐。Netty 的ByteBuffer 替代品是`ByteBuf`，一个强大的实现，既解决了JDK API 的局限性，又为网络应用程序的开发者提供了更好的API。

> `ByteBuf 维护了两个不同的索引`，名称以read 或者write 开头的ByteBuf 方法，将会推进其对应的索引. 如果打算读取字节直到readerIndex 达到和writerIndex 同样的值时会发生什么。在那时，你将会到达“可以读取的”数据的末尾.试图读取超出该点的数据将会触发一个IndexOutOf-BoundsException。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009162909372.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009162936542.png)

### 1. 分配方式

#### .1. 堆缓冲区

> 将数据存储在`JVM 的堆空间`中。这种模式被称为支撑数组（backing array），它能在`没有使用池化的情况下提供快速的分配和释放`。可以由hasArray()来判断检查ByteBuf 是否由数组支撑。

#### .2. 直接缓冲区

> 直接缓冲区的主要缺点是，相对于基于堆的缓冲区，它们的分配和释放都较为昂贵。

#### .3. ByteBufAllocator

> 通过interface ByteBufAllocator分配我们所描述过的任意类型的ByteBuf 实例。

#### .4. Unpooled缓冲区

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009163415328.png)

### 2. ByteBuf操作

> - get()和set()操作，从给定的索引开始，并且保持索引不变；get+数据字长（bool.byte,int,short,long,bytes）
>
> - read()和write()操作，从给定的索引开始，并且会根据已经访问过的字节数对索引进行调整。
> - isReadable() 如果至少有一个字节可供读取，则返回true
> - isWritable() 如果至少有一个字节可被写入，则返回true
> - readableBytes() 返回可被读取的字节数
> - writableBytes() 返回可被写入的字节数
> - capacity() 返回ByteBuf 可容纳的字节数。在此之后，它会尝试再次扩展直到达到maxCapacity()
> - maxCapacity() 返回ByteBuf 可以容纳的最大字节数
> - hasArray() 如果ByteBuf 由一个字节数组支撑，则返回true
> - array() 如果 ByteBuf 由一个字节数组支撑则返回该数组；否则，它将抛出一个UnsupportedOperationException 异常

#### .1. 可丢弃字节

> 为可丢弃字节的分段包含了已经被读过的字节。通过调用discardRead-Bytes()方法，可以丢弃它们并回收空间。这个分段的初始大小为0，存储在readerIndex 中，会随着`read 操作的执行而增加`。

#### .2. 可读字节

> ByteBuf 的可读字节分段存储了实际数据。新分配的、包装的或者复制的缓冲区的默认的readerIndex 值为0。

#### .3. 可写字节

> 可写字节分段是指一个拥有未定义内容的、写入就绪的内存区域。新分配的缓冲区的writerIndex 的默认值为0。任何名称以write 开头的操作都将从当前的writerIndex 处开始写数据，并将它增加已经写入的字节数。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009163847209.png)

#### .4. 索引管理

> 调用markReaderIndex()、markWriterIndex()、resetWriterIndex()和resetReaderIndex()来标记和重置ByteBuf 的readerIndex 和writerIndex。
>
> 通过调用readerIndex(int)或者writerIndex(int)来将索引移动到指定位置。试图将任何一个索引设置到一个无效的位置都将导致一个IndexOutOfBoundsException。
>
> 通过调用clear()方法来将readerIndex 和writerIndex 都设置为0。注意，这并不会清除内存中的内容。

#### .5. 查找操作

> 在ByteBuf中有多种可以用来确定指定值的索引的方法。最简单的是使用indexOf()方法。

#### .6. 派生缓冲器

> 派生缓冲区为ByteBuf 提供了以专门的方式来呈现其内容的视图。duplicate(), slice(), Unpooled.unmodifiableBuffer(), order(ByteOrder), readSlice(int);

> 每个这些方法都将返回一个`新的ByteBuf 实例，它具有自己的读索引、写索引和标记索引`。其`内部存储和JDK 的ByteBuffer 一样也是共享的。`

#### .7.  引用计数

> 引用计数是一种通过在某个对象所持有的资源不再被其他对象引用时释放该对象所持有的资源来优化内存使用和性能的技术。

### Resource

- 学习来源： https://juejin.cn/post/6997999345270259726

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/bytebuf%E7%B1%BB/  

