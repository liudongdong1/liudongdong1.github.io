# telnet helloworld


> Netty是 一个异步事件驱动的网络应用程序框架，用于快速开发可维护的高性能协议服务器和客户端。是一个NIO客户端服务器框架，可以快速轻松地开发协议服务器和客户端等网络应用程序。它极大地简化并简化了TCP和UDP套接字服务器等网络编程。Netty经过精心设计，具有丰富的协议，如FTP，SMTP，HTTP以及各种二进制和基于文本的传统协议。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211008201945363.png)

> 绿色的部分**Core**核心模块，包括零拷贝、API库、可扩展的事件模型。
>
> 橙色部分**Protocol Support**协议支持，包括Http协议、webSocket、SSL(安全套接字协议)、谷歌Protobuf协议、zlib/gzip压缩与解压缩、Large File Transfer大文件传输等等。
>
> 红色的部分**Transport Services**传输服务，包括Socket、Datagram、Http Tunnel等等。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/cc27d56addd74e82b6b6b349c7f3769b.png)

### 1. 代码编写

> 代码模块主要分为服务端和客户端。 主要实现的业务逻辑： 服务端启动成功之后，客户端也启动成功，这时服务端会发送一条信息给客户端。客户端或者**telnet**发送一条信息到服务端，服务端会根据逻辑回复客户端一条客户端，当客户端或者**telent**发送`bye`给服务端，服务端和客户端断开链接。

```
 netty-helloworld
    ├── client
      ├── Client.class -- 客户端启动类
      ├── ClientHandler.class -- 客户端逻辑处理类
      ├── ClientInitializer.class -- 客户端初始化类
    ├── server 
      ├── Server.class -- 服务端启动类
      ├── ServerHandler -- 服务端逻辑处理类
      ├── ServerInitializer -- 服务端初始化类
```
#### .1. 服务器端代码

##### .1. Server 入口

```java
public final class Server {
      public  static void main(String[] args) throws Exception {
          //Configure the server
          //创建两个EventLoopGroup对象
          //创建boss线程组 用于服务端接受客户端的连接
          EventLoopGroup bossGroup = new NioEventLoopGroup(1);
          // 创建 worker 线程组 用于进行 SocketChannel 的数据读写
          EventLoopGroup workerGroup = new NioEventLoopGroup();
          try {
              // 创建 ServerBootstrap 对象
              ServerBootstrap b = new ServerBootstrap();
              //设置使用的EventLoopGroup
              b.group(bossGroup,workerGroup)
                  //设置要被实例化的为 NioServerSocketChannel 类
                      .channel(NioServerSocketChannel.class)
                  // 设置 NioServerSocketChannel 的处理器
                      .handler(new LoggingHandler(LogLevel.INFO))
                   // 设置连入服务端的 Client 的 SocketChannel 的处理器
                      .childHandler(new ServerInitializer());
              // 绑定端口，并同步等待成功，即启动服务端
              ChannelFuture f = b.bind(8888);
              // 监听服务端关闭，并阻塞等待
              f.channel().closeFuture().sync();
          } finally {
              // 优雅关闭两个 EventLoopGroup 对象
              bossGroup.shutdownGracefully();
              workerGroup.shutdownGracefully();
          }
      }
}
```
##### .2. ServerInitializer

> 使用Netty编写业务层的代码，我们需要继承**ChannelInboundHandlerAdapter** 或**SimpleChannelInboundHandler**类
>
> - 继承**SimpleChannelInboundHandler**类之后，会在接收到数据后会自动**release**掉数据占用的**Bytebuffer**资源。并且继承该类需要指定数据格式。 
> - 继承**ChannelInboundHandlerAdapter**则不会自动释放，需要手动调用**ReferenceCountUtil.release()**等方法进行释放。继承该类不需要指定数据格式。 （可以防止数据未处理完就被释放了）

```java
public class ServerInitializer extends ChannelInitializer<SocketChannel> {
      private static final StringDecoder DECODER = new StringDecoder();
      private static final StringEncoder ENCODER = new StringEncoder();
  
      private static final ServerHandler SERVER_HANDLER = new ServerHandler();
  
  
      @Override
      public void initChannel(SocketChannel ch) throws Exception {
          ChannelPipeline pipeline = ch.pipeline();
  
          // 添加帧限定符来防止粘包现象
          pipeline.addLast(new DelimiterBasedFrameDecoder(8192, Delimiters.lineDelimiter()));
          // 解码和编码，应和客户端一致
          pipeline.addLast(DECODER);
          pipeline.addLast(ENCODER);
 
          // 业务逻辑实现类
          pipeline.addLast(SERVER_HANDLER);
      }
  }
```

##### .3. ServerHandler

```java
@Sharable
  public class ServerHandler extends SimpleChannelInboundHandler<String> {
      /**
       * 建立连接时，发送一条庆祝消息
       */
      @Override
      public void channelActive(ChannelHandlerContext ctx) throws Exception {
          // 为新连接发送庆祝
          ctx.write("Welcome to " + InetAddress.getLocalHost().getHostName() + "!\r\n");
          ctx.write("It is " + new Date() + " now.\r\n");
          ctx.flush();
      }
  
      //业务逻辑处理
      @Override
      public void channelRead0(ChannelHandlerContext ctx, String request) throws Exception {
          // Generate and write a response.
          String response;
          boolean close = false;
          if (request.isEmpty()) {
              response = "Please type something.\r\n";
          } else if ("bye".equals(request.toLowerCase())) {
              response = "Have a good day!\r\n";
              close = true;
          } else {
              response = "Did you say '" + request + "'?\r\n";
          }
  
          ChannelFuture future = ctx.write(response);
  
          if (close) {
              future.addListener(ChannelFutureListener.CLOSE);
          }
      }
  
      @Override
      public void channelReadComplete(ChannelHandlerContext ctx) {
          ctx.flush();
      }
 
      //异常处理
      @Override
      public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
          cause.printStackTrace();
          ctx.close();
      }
  }
```

#### .2. 客户端代码

> 客户端过滤其这块基本和服务端一致。不过需要注意的是，传输协议、编码和解码应该一致。

##### .1. ClientInitializer

```java
public class ClientInitializer extends ChannelInitializer<SocketChannel> {
      private static final StringDecoder DECODER = new StringDecoder();
      private static final StringEncoder ENCODER = new StringEncoder();
  
      private static final ClientHandler CLIENT_HANDLER = new ClientHandler();
  
  
      @Override
      public void initChannel(SocketChannel ch) {
          ChannelPipeline pipeline = ch.pipeline();
          pipeline.addLast(new DelimiterBasedFrameDecoder(8192, Delimiters.lineDelimiter()));
          pipeline.addLast(DECODER);
          pipeline.addLast(ENCODER);
  
          pipeline.addLast(CLIENT_HANDLER);
      }
  }
```

##### .2. ClientHandler 业务处理逻辑

```java
@Sharable
  public class ClientHandler extends SimpleChannelInboundHandler<String> {
      //打印读取到的数据
      @Override
      protected void channelRead0(ChannelHandlerContext ctx, String msg) throws Exception {
          System.err.println(msg);
      }
      //异常数据捕获
      @Override
      public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
          cause.printStackTrace();
          ctx.close();
      }
  }
```


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/telnet-helloworld/  

