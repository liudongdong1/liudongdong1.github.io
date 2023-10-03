# HTTP 服务


![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211010180947816.png)

### 1. NettyHTTP编解码器

```java
public class HttpHelloWorldServerInitializer extends ChannelInitializer<SocketChannel> {
      @Override
      public void initChannel(SocketChannel ch) {
          ChannelPipeline p = ch.pipeline();
          /**
           * 或者使用HttpRequestDecoder & HttpResponseEncoder
           *HttpRequestDecoder 即把 ByteBuf 解码到 HttpRequest 和 HttpContent。HttpResponseEncoder 即把 HttpResponse 或 HttpContent 编码到 ByteBuf。HttpServerCodec 即 HttpRequestDecoder 和 HttpResponseEncoder 的结合。
           */
          p.addLast(new HttpServerCodec());
          /**
           * 在处理POST消息体时需要加上
           *把 HttpMessage 和 HttpContent 聚合成一个 FullHttpRequest 或者 FullHttpResponse （取决于是处理请求还是响应
           */
          p.addLast(new HttpObjectAggregator(1024*1024));
          p.addLast(new HttpServerExpectContinueHandler());
          p.addLast(new HttpHelloWorldServerHandler());
      }
  }
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/http%E6%9C%8D%E5%8A%A1/  

