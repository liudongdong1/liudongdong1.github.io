# 粘包拆包


> 发送端为了将多个发给接收端的数据包，更有效地发送到接收端，会使用**Nagle算法**。Nagle算法会**将多次时间间隔较小且数据量小的数据合并成一个大的数据块**进行发送。虽然这样的确提高了效率，但是**因为面向流通信，数据是无消息保护边界的**，就会**导致接收端难以分辨出完整的数据包**了。
>
> - 在`数据的末尾添加特殊的符号`标识数据包的边界。通常会加\n\r、\t或者其他的符号。
> - 在`数据的头部声明数据的长度`，按长度获取数据。
> - `规定报文的长度，不足则补空位`。读取时按规定好的长度来读取。

### 1. 使用LineBasedFrameDecoder

> 在数据`末尾加上特殊符号以标识边界。默认是使用换行符\n`。
>
> **数据末尾一定是分隔符，分隔符后面不要再加上数据**，否则会当做下一条数据的开始部分。

- 发送方编码器

```java
@Override
protected void initChannel(SocketChannel ch) throws Exception {
    //添加编码器，使用默认的符号\n，字符集是UTF-8
    ch.pipeline().addLast(new LineEncoder(LineSeparator.DEFAULT, CharsetUtil.UTF_8));
    ch.pipeline().addLast(new TcpClientHandler());
}
```

```java
@Override
public void channelActive(ChannelHandlerContext ctx) throws Exception {
    for (int i = 1; i <= 5; i++) {
        //在末尾加上默认的标识符\n
        ByteBuf byteBuf = Unpooled.copiedBuffer("msg No" + i + StringUtil.LINE_FEED, Charset.forName("utf-8"));
        ctx.writeAndFlush(byteBuf);
    }
}
```

- 接收方解码器

```java
@Override
protected void initChannel(SocketChannel ch) throws Exception {
    //解码器需要设置数据的最大长度，我这里设置成1024
    ch.pipeline().addLast(new LineBasedFrameDecoder(1024));
    //给pipeline管道设置业务处理器
    ch.pipeline().addLast(new TcpServerHandler());
}
```

### 2. 使用自定义长度编码器

> - maxFrameLength 发送数据包的最大长度
> - lengthFieldOffset 长度域的偏移量。长度域位于整个数据包字节数组中的开始下标。
> - lengthFieldLength 长度域的字节数长度。长度域的字节数长度。
> - lengthAdjustment 长度域的偏移量矫正。如果长度域的值，除了包含有效数据域的长度外，还包含了其他域（如长度域自身）长度，那么，就需要进行矫正。矫正的值为：包长 - 长度域的值 – 长度域偏移 – 长度域长。
> - initialBytesToStrip `丢弃的起始字节数`。丢弃处于此索引值前面的字节。

- 消息接收端解码器

```java
@Override
protected void initChannel(SocketChannel ch) throws Exception {
    //数据包最大长度是1024
    //长度域的起始索引是0
    //长度域的数据长度是4
    //矫正值为0，因为长度域只有 有效数据的长度的值
    //丢弃数据起始值是4，因为长度域长度为4，我要把长度域丢弃，才能得到有效数据
    ch.pipeline().addLast(new LengthFieldBasedFrameDecoder(1024, 0, 4, 0, 4));
    ch.pipeline().addLast(new TcpClientHandler());
}
```

- 发送端代码

```java
@Override
public void channelActive(ChannelHandlerContext ctx) throws Exception {
    for (int i = 1; i <= 5; i++) {
        String str = "msg No" + i;
        ByteBuf byteBuf = Unpooled.buffer(1024);
        byte[] bytes = str.getBytes(Charset.forName("utf-8"));
        //设置长度域的值，为有效数据的长度
        byteBuf.writeInt(bytes.length);
        //设置有效数据
        byteBuf.writeBytes(bytes);
        ctx.writeAndFlush(byteBuf);
    }
}
```

### 3. google Protobuf编解器

> Protocol buffers是Google公司的**与语言无关、平台无关、可扩展的序列化数据的机制**，类似XML，但是**更小、更快、更简单**。您只需**定义一次数据的结构化方式**，然后就可以使用**特殊生成的源代码**，轻松地**将结构化数据写入和读取到各种数据流中，并支持多种语言**。
>
> - 发送端加上编码器**ProtobufVarint32LengthFieldPrepender**  可以解决发送多条数据的时候出现粘包问题。

- 添加maven依赖

```xml
<dependency>
    <groupId>com.google.protobuf</groupId>
    <artifactId>protobuf-java</artifactId>
    <version>3.6.1</version>
</dependency>
```

- 编写proto文件的message.proto

```js
syntax = "proto3"; //版本
option java_outer_classname = "MessagePojo";//生成的外部类名，同时也是文件名
message Message {
    int32 id = 1;//Message类的一个属性，属性名称是id，序号为1
    string content = 2;//Message类的一个属性，属性名称是content，序号为2
}
```

- 使用编译器，通过.proto生成代码,protoc.exe –java_out=. Message.proto  生成MessageProto.java 文件
- 客户端接收端代码

```java
@Override
protected void initChannel(SocketChannel ch) throws Exception {
    ch.pipeline().addLast(new ProtobufVarint32LengthFieldPrepender());
    ch.pipeline().addLast(new ProtobufEncoder());
    ch.pipeline().addLast(new TcpClientHandler());
}
```

- 客户端发送消息

```java
@Override
public void channelActive(ChannelHandlerContext ctx) throws Exception {
    //使用的是构建者模式进行创建对象
    MessagePojo.Message message = MessagePojo
        .Message
        .newBuilder()
        .setId(1)
        .setContent("芜湖大司马，起飞~")
        .build();
    ctx.writeAndFlush(message);
}
```

- 服务端解码器（ 问题：为什么这里是decoder 而不是encoder? 还是都可以）

```java
@Override
protected void initChannel(SocketChannel ch) throws Exception {
    ch.pipeline().addLast(new ProtobufVarint32FrameDecoder());
    ch.pipeline().addLast(new ProtobufDecoder(MessagePojo.Message.getDefaultInstance()));
    //给pipeline管道设置处理器
    ch.pipeline().addLast(new TcpServerHandler());
}
```

- 服务端接收消息

```java
@Override
protected void channelRead0(ChannelHandlerContext ctx, MessagePojo.Message messagePojo) throws Exception {
    System.out.println("id:" + messagePojo.getId());
    System.out.println("content:" + messagePojo.getContent());
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E7%B2%98%E5%8C%85%E6%8B%86%E5%8C%85/  

