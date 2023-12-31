# EncoderDecoder


### 1. 解码器

> 解码器是负责将入站数据从一种格式转换到另一种格式的，所以Netty 的解码器实现了ChannelInboundHandler。

#### .1. ByteToMessageDecoder

> 由于你不可能知道远程节点`是否会一次性地发送一个完整的消息`，所以这个类会对入站数据进行缓冲，直到它准备好处理。
>
> decode()方法被调用时将会传入一个`包含了传入数据的ByteBuf`，以及一个`用来添加解码消息的List`。对这个方法的调用将会重复进行，直到确定没有新的元素被添加到该List，或者该ByteBuf 中没有更多可读取的字节时为止。然后，`如果该List 不为空，那么它的内容将会被传递给ChannelPipeline 中的下一个ChannelInboundHandler`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009160601207.png)

- 定长处理的消息解码器

```java
//扩展 ByteToMessageDecoder 以处理入站字节，并将它们解码为消息
public class FixedLengthFrameDecoder extends ByteToMessageDecoder {
    private final int frameLength;

    //指定要生成的帧的长度
    public FixedLengthFrameDecoder(int frameLength) {
        if (frameLength <= 0) {
            throw new IllegalArgumentException(
                "frameLength must be a positive integer: " + frameLength);
        }
        this.frameLength = frameLength;
    }
    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in,
                          List<Object> out) throws Exception {
        //检查是否有足够的字节可以被读取，以生成下一个帧
        while (in.readableBytes() >= frameLength) {
            //从 ByteBuf 中读取一个新帧
            ByteBuf buf = in.readBytes(frameLength);
            //将该帧添加到已被解码的消息列表中
            out.add(buf);
        }
    }
}
```

#### .2. MessageToMessageDecoder

> decode(ChannelHandlerContext ctx,I msg,List out): 对于每个需要被解码为另一种格式的入站消息来说，该方法都将会被调用。解码消息随后会被传递给ChannelPipeline中的下一个ChannelInboundHandler

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009160642434.png)

#### .3. TooLongFrameException

> TooLongFrameException 类，其将由解码器在帧超出指定的大小限制时抛出。为了避免这种情况，你可以设置一个最大字节数的阈值，如果超出该阈值，则会导致抛出一个TooLongFrameException（随后会被ChannelHandler.exceptionCaught()方法捕获）。然后，如何处理该异常则完全取决于该解码器的用户。

### 2. 编码器

#### .1. MessageToByteEncoder

> encode(ChannelHandlerContext ctx,I msg,ByteBuf out):encode()方法是你需要实现的唯一抽象方法。它被调用时将会传入`要被该类编码为ByteBuf 的（类型为I 的）出站消息。`该ByteBuf 随后将会被转发给ChannelPipeline中的下一个ChannelOutboundHandler

```java
public class MsgPackEncode extends MessageToByteEncoder<Object> {
    @Override
    protected void encode(ChannelHandlerContext ctx, Object msg,
                          ByteBuf out) throws Exception {
        MessagePack messagePack = new MessagePack();
        byte[] raw = messagePack.write(msg);
        out.writeBytes(raw);
    }
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009161533484.png)

#### .2. MessageToMessageEncoder

```java
//扩展 MessageToMessageEncoder 以将一个消息编码为另外一种格式
public class AbsIntegerEncoder extends
        MessageToMessageEncoder<ByteBuf> {
    @Override
    protected void encode(ChannelHandlerContext channelHandlerContext,
                          ByteBuf in, List<Object> out) throws Exception {
        //检查是否有足够的字节用来编码,int为4个字节
        while (in.readableBytes() >= 4) {
            //从输入的 ByteBuf中读取下一个整数，并且计算其绝对值
            int value = Math.abs(in.readInt());
            //将该整数写入到编码消息的 List 中
            out.add(value);
        }
    }
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009161558246.png)

### 3. 内置编解码器和ChannelHandler

#### .1. SSL/TLS 保护Netty 应用程序

> 为了支持SSL/TLS，Java 提供了javax.net.ssl 包，它的SSLContext 和SSLEngine类使得实现解密和加密相当简单直接。Netty 通过一个名为SslHandler 的ChannelHandler实现利用了这个API，其中SslHandler 在内部使用SSLEngine 来完成实际的工作。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009162115628.png)

#### .2. HTTP 系列

##### 1. HTTP 编码器

- HttpRequestEncoder 将HttpRequest、HttpContent 和LastHttpContent 消息编码为字节
- HttpResponseEncoder 将HttpResponse、HttpContent 和LastHttpContent 消息编码为字节
- HttpRequestDecoder 将字节解码为HttpRequest、HttpContent 和LastHttpContent 消息
- HttpResponseDecoder 将字节解码为HttpResponse、HttpContent 和LastHttpContent 消息


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009162252925.png)

##### 2. HTTP 聚合消息

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009162330631.png)

##### 3. HTTP压缩

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211009162350906.png)



### Resource

- 学习链接： https://juejin.cn/post/6997999252303675399

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/encoderdecoder/  

