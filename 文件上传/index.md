# 文件上传


> 构建`高性能、低时延的各种 Java 中间件`，例如 MQ、分布式服务框架、ESB 消息总线等，Netty 主要作为基础通信框架提供高性能、低时延的通信服务；
>
> `公有或者私有协议栈的基础通信框架`，例如可以基于 Netty 构建异步、高性能的 WebSocket 协议栈；
>
> 各领域应用，例如大数据、游戏等，Netty 作为高性能的通信框架用于内部各模块的数据分发、传输和汇总等，实现模块之间高性能通信。

### 1. FileServer

```java
package io.netty.example.file;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.LineBasedFrameDecoder;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.util.CharsetUtil;

/**
 * Server that accept the path of a file an echo back its content.
 */
public final class FileServer {

    static final boolean SSL = System.getProperty("ssl") != null;
    // Use the same default port with the telnet example so that we can use the telnet client example to access it.
    static final int PORT = Integer.parseInt(System.getProperty("port", SSL? "8992" : "8023"));

    public static void main(String[] args) throws Exception {
        // Configure SSL.
        final SslContext sslCtx;
        if (SSL) {
            SelfSignedCertificate ssc = new SelfSignedCertificate();
            sslCtx = SslContextBuilder.forServer(ssc.certificate(), ssc.privateKey()).build();
        } else {
            sslCtx = null;
        }

        // Configure the server.
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup();
        try {
            ServerBootstrap b = new ServerBootstrap();
            b.group(bossGroup, workerGroup)
             .channel(NioServerSocketChannel.class)
             .option(ChannelOption.SO_BACKLOG, 100)
             .handler(new LoggingHandler(LogLevel.INFO))
             .childHandler(new ChannelInitializer<SocketChannel>() {
                 @Override
                 public void initChannel(SocketChannel ch) throws Exception {
                     ChannelPipeline p = ch.pipeline();
                     if (sslCtx != null) {
                         p.addLast(sslCtx.newHandler(ch.alloc()));
                     }
                     p.addLast(
                             new StringEncoder(CharsetUtil.UTF_8),
                             new LineBasedFrameDecoder(8192),
                             new StringDecoder(CharsetUtil.UTF_8),
                             new ChunkedWriteHandler(),
                             new FileServerHandler());
                 }
             });

            // Start the server.
            ChannelFuture f = b.bind(PORT).sync();

            // Wait until the server socket is closed.
            f.channel().closeFuture().sync();
        } finally {
            // Shut down all event loops to terminate all threads.
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}
```

### 2. FileServerHandler

```java
package io.netty.example.file;

import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.DefaultFileRegion;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.ssl.SslHandler;
import io.netty.handler.stream.ChunkedFile;

import java.io.RandomAccessFile;

public class FileServerHandler extends SimpleChannelInboundHandler<String> {

    @Override
    public void channelActive(ChannelHandlerContext ctx) {
        ctx.writeAndFlush("HELLO: Type the path of the file to retrieve.\n");
    }

    @Override
    public void channelRead0(ChannelHandlerContext ctx, String msg) throws Exception {
        RandomAccessFile raf = null;
        long length = -1;
        try {
            raf = new RandomAccessFile(msg, "r");
            length = raf.length();
        } catch (Exception e) {
            ctx.writeAndFlush("ERR: " + e.getClass().getSimpleName() + ": " + e.getMessage() + '\n');
            return;
        } finally {
            if (length < 0 && raf != null) {
                raf.close();
            }
        }

        ctx.write("OK: " + raf.length() + '\n');
        if (ctx.pipeline().get(SslHandler.class) == null) {
            // SSL not enabled - can use zero-copy file transfer.
            ctx.write(new DefaultFileRegion(raf.getChannel(), 0, length));
        } else {
            // SSL enabled - cannot use zero-copy file transfer.
            ctx.write(new ChunkedFile(raf));
        }
        ctx.writeAndFlush("\n");
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();

        if (ctx.channel().isActive()) {
            ctx.writeAndFlush("ERR: " +
                    cause.getClass().getSimpleName() + ": " +
                    cause.getMessage() + '\n').addListener(ChannelFutureListener.CLOSE);
        }
    }
}
```

### Resource

- 完整项目代码：https://github.com/1433365571/HttpUploadNetty/blob/master/src/main/java/server/HttpUploadServerHandler.java
- 官方example代码：https://github.com/netty/netty/blob/4.1/example/src/main/java/io/netty/example/file/FileServerHandler.java


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E6%96%87%E4%BB%B6%E4%B8%8A%E4%BC%A0/  

