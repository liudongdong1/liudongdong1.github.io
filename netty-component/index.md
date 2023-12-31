# netty component


### 1. Netty 事件相应机制

```java
public class NettyServer  {

    private final int port;

    public NettyServer(int port) {
        this.port = port;
    }

    public static void main(String[] args) throws InterruptedException {
        int port = 9999;
        NettyServer echoServer = new NettyServer(port);
        System.out.println("服务器启动");
        echoServer.start();
        System.out.println("服务器关闭");
    }

    public void start() throws InterruptedException {
        final NettyServerHandler serverHandler = new NettyServerHandler();
        /*线程组*/
        EventLoopGroup group = new NioEventLoopGroup();
        try {
            /*服务端启动必须*/
            ServerBootstrap b = new ServerBootstrap();
            b.group(group)/*将线程组传入*/
                    .channel(NioServerSocketChannel.class)/*指定使用NIO进行网络传输*/
                    .localAddress(new InetSocketAddress(port))/*指定服务器监听端口*/
                    /*服务端每接收到一个连接请求，就会新启一个socket通信，也就是channel，
                    所以下面这段代码的作用就是为这个子channel增加handle*/
                    .childHandler(new ChannelInitializer<SocketChannel>() {
                        protected void initChannel(SocketChannel ch) throws Exception {
                            /*添加到该子channel的pipeline的尾部*/
                            ch.pipeline().addLast(serverHandler);
                        }
                    });
            ChannelFuture f = b.bind().sync();/*异步绑定到服务器，sync()会阻塞直到完成*/
            f.channel().closeFuture().sync();/*阻塞直到服务器的channel关闭*/

        } finally {
            group.shutdownGracefully().sync();/*优雅关闭线程组*/
        }
    }
}

```

> Netty 在内部使用了回调来处理事件；当一个回调被触发时，相关的事件可以被一个interface-ChannelHandler 的实现处理。
>
> Future 提供了另一种在操作完成时通知应用程序的方式。这个对象可以看作是`一个异步操作的结果的占位符`；它将在未来的某个时刻完成，并提供对其结果的访问。
>
> Netty事件是按照它们`与入站或出站数据流的相关性进行分类的`。可能由入站数据或者相关的状态更改而触发的事件包括：`连接已被激活或者连接失活；数据读取；用户事件；错误事件`。
>
> 每个事件都可以被分发给ChannelHandler 类中的某个用户实现的方法。可以认为`每个ChannelHandler 的实例都类似于一种为了响应特定事件而被执行的回调`。

### 2. Channel, EventLoop, ChannelFuture

> - `Channel` 类似于JAVA中的`Socket，用于客户端连接，数据交换；`
> - `EventLoop`用于`控制流、多线程处理以及并发处理；`
> - `ChannelFuture` `异步事件通知`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/11930f8cded8499694d24b607c5ff925tplv-k3u1fbpfcp-watermark.awebp)

#### .1. Channel

> 基本的I/O 操作`bind()、connect()、read()和write()`依赖于底层网络传输所提供的原语。在基于Java 的网络编程中，其基本的构造是类Socket。Netty 的Channel 接口所提供的API，被用于所有的I/O 操作。大大地降低了直接使用Socket 类的复杂性。此外，Channel 也是`拥有许多预定义的、专门化实现的广泛类层次结构的根`。
>
> 由于`Channel 是独一无二的`，所以为了保证顺序将Channel 声明为java.lang.Comparable 的一个子接口。因此，如果两个不同的Channel 实例都返回了相同的散列码，那么AbstractChannel 中的compareTo()方法的实现将会抛出一个Error。
>
> 当这些`状态发生改变时，将会生成对应的事件`。这些`事件将会被转发给ChannelPipeline 中的ChannelHandler`，其可以随后对它们做出响应。

1. 注册事件 fireChannelRegistered。
2. 连接建立事件 fireChannelActive。
3. 读事件和读完成事件 fireChannelRead、fireChannelReadComplete。
4. 异常通知事件 fireExceptionCaught。
5. 用户自定义事件 fireUserEventTriggered。
6. Channel 可写状态变化事件 fireChannelWritabilityChanged。
7. 连接关闭事件 fireChannelInactive。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008210632014.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008210824805.png)

##### 1. channelRead

> channelRead 中调用了 channelRead0，其会先做消息类型检查，判断当前message 是否需要传递到下一个handler。

```java
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
    boolean release = true;
    try {
        if (acceptInboundMessage(msg)) {
            @SuppressWarnings("unchecked")
            I imsg = (I) msg;
            channelRead0(ctx, imsg);
        } else {
            release = false;
            ctx.fireChannelRead(msg);
        }
    } finally {
        if (autoRelease && release) {
            ReferenceCountUtil.release(msg);
        }
    }
}
```

#### .2. EventLoop和EventLoopGroup

> 在一个`while循环中select出事件`，然后`依次处理每种事件`。我们可以把它称为事件循环，这就是EventLoop。
>
> io.netty.channel 包中的类，为了与Channel 的事件进行交互，扩展了这些接口/类。一个EventLoop 将由一个永远都不会改变的Thread 驱动，同时任务（Runnable 或者Callable）可以直接提交给EventLoop 实现，以立即执行或者调度执行。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008211125000.png)

```java
//NIO处理时间事件方式
@Override
public void run() {
    //循环遍历selector
    while (started) {
        try {
            //阻塞,只有当至少一个注册的事件发生的时候才会继续.
            selector.select();
            Set<SelectionKey> keys = selector.selectedKeys();
            Iterator<SelectionKey> it = keys.iterator();
            SelectionKey key = null;
            while (it.hasNext()) {
                key = it.next();
                it.remove();
                try {
                    handleInput(key);
                } catch (Exception e) {
                    if (key != null) {
                        key.cancel();
                        if (key.channel() != null) {
                            key.channel().close();
                        }
                    }
                }
            }
        } catch (Throwable t) {
            t.printStackTrace();
        }
    }
    //selector关闭后会自动释放里面管理的资源
    if (selector != null)
        try {
            selector.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
}
```

##### .1. 任务调度

> 调度一个任务以便稍后（延迟）执行或者周期性地执行。例如你可能想要注册一个在客户端已经连接了5 分钟之后触发的任务。
>
> - 如果Handler处理器有一些长时间的业务处理，可以交给**taskQueue异步处理**。

```java
public class MyServerHandler extends ChannelInboundHandlerAdapter {

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        //获取到线程池eventLoop，添加线程，执行
        ctx.channel().eventLoop().execute(new Runnable() {
            @Override
            public void run() {
                try {
                    //长时间操作，不至于长时间的业务操作导致Handler阻塞
                    Thread.sleep(1000);
                    System.out.println("长时间的业务处理");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
}
```

```java
//scheduleTaskQueue延时任务队列
ctx.channel().eventLoop().schedule(new Runnable() {
    @Override
    public void run() {
        try {
            //长时间操作，不至于长时间的业务操作导致Handler阻塞
            Thread.sleep(1000);
            System.out.println("长时间的业务处理");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
},5, TimeUnit.SECONDS);//5秒后执行
```

##### .2. 线程管理

> 在内部，当提交任务到如果 `（ 当前）调用线程正是支撑EventLoop 的线程，那么所提交的代码块将会被（直接）执行`。否则，EventLoop 将调度该任务以便稍后执行，并将它放入到内部队列中。当EventLoop下次处理它的事件时，它会执行队列中的那些任务/事件。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008211430464.png)

##### .3.   线程分配

> EventLoopGroup 负责为每个新创建的Channel 分配一个EventLoop。在当前实现中，使用顺序循环（round-robin）的方式进行分配以获取一个均衡的分布，并且相同的EventLoop可能会被分配给多个Channel。`一旦一个Channel 被分配给一个EventLoop，它将在它的整个生命周期中都使用这个EventLoop（以及相关联的Thread）。` `因为一个EventLoop 通常会被用于支撑多个Channel，所以对于所有相关联的Channel 来说，ThreadLocal 都将是一样的。`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008211659256.png)

#### .3. ChannelFuture

> Netty 提供了ChannelFuture 接口，其addListener()方法注册了一个ChannelFutureListener，以便在`某个操作完成时（无论是否成功）得到通知`。
>
> - ChannelFuture是采取类似`观察者模式`的形式进行获取结果

```java
//添加监听器
channelFuture.addListener(new ChannelFutureListener() {
    //使用匿名内部类，ChannelFutureListener接口
    //重写operationComplete方法
    @Override
    public void operationComplete(ChannelFuture future) throws Exception {
        //判断是否操作成功    
        if (future.isSuccess()) {
            System.out.println("连接成功");
        } else {
            System.out.println("连接失败");
        }
    }
});
```

### 3. **ChannelHandler、ChannelPipeline和ChannelHandlerContext**

#### .1. ChannelHandler

> ChannelHandler 可专门用于几乎任何类型的动作，例如将数据从一种格式转换为另外一种格式，例如各种编解码，或者处理转换过程中所抛出的异常。
>
> 接口 ChannelHandler 定义的生命周期操作，在ChannelHandler被添加到ChannelPipeline 中或者被从ChannelPipeline 中移除时会调用这些操作。这些方法中的`每一个都接受一个ChannelHandlerContext 参数`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008212138781.png)

>  ChannelInboundHandler 的生命周期方法。这些方法将会在`数据被接收时或者与其对应的Channel 状态发生改变时被调用`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008212255420.png)

> `出站操作和数据`将由ChannelOutboundHandler 处理。它的方法将被Channel、Channel-Pipeline 以及ChannelHandlerContext 调用。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008212350016.png)

#### .2. **ChannelPipeline**

> 当`Channel 被创建时`，它将会被`自动地分配一个新的ChannelPipeline`。这项`关联是永久性的`；Channel 既不能附加另外一个ChannelPipeline，也不能分离其当前的。
>
> 事件流经ChannelPipeline 是ChannelHandler 的工作，它们是在应用程序的`初始化或者引导阶段被安装的`。这些`对象接收事件、执行它们所实现的处理逻辑，并将数据传递给链中的下一个ChannelHandler`。它们的执行顺序是由它们被添加的顺序所决定的。
>
> 入站和出站ChannelHandler 可以被安装到同一个ChannelPipeline中。`如果一个消息或者任何其他的入站事件被读取，那么它会从ChannelPipeline 的头部开始流动，最终，数据将会到达ChannelPipeline 的尾端`，届时，所有处理就都结束了。 
>
> 入站指的是`数据从底层java NIO Channel到Netty的Channel`。
>
> 出站指的是`通过Netty的Channel来操作底层的java NIO Channel。`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008212732407.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008212834279.png)

#### .3. ChannelHandlerContext

> 通过使用作为参数传递到每个方法的**ChannelHandlerContext**，事件可以被传递给当前ChannelHandler 链中的下一个ChannelHandler。虽然这个对象可以被用于获取底层的Channel，但是它`主要还是被用于写出站数据`。
>
> ChannelHandlerContext 代表了ChannelHandler 和ChannelPipeline 之间的`关联`，每当有ChannelHandler 添加到ChannelPipeline 中时，都会创建ChannelHandlerContext。`ChannelHandlerContext 的主要功能是管理它所关联的ChannelHandler 和在同一个ChannelPipeline 中的其他ChannelHandler 之间的交互。`
>
> ChannelHandlerContext 有很多的方法，其中一些方法也存在于Channel 和Channel-Pipeline 本身上，**但是有一点重要的不同。** 如果`调用Channel 或者ChannelPipeline 上的这些方法，它们将沿着整个ChannelPipeline 进行传播`。而`调用位于ChannelHandlerContext上的相同方法，则将从当前所关联的ChannelHandler 开始，并且只会传播给位于该ChannelPipeline 中的下一个（入站下一个，出站上一个）能够处理该事件的ChannelHandler。`
>
> ChannelHandlerContext 和ChannelHandler 之间的`关联（绑定）是永远不会改变的`，所以缓存对它的引用是安全的；

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008213237221.png)

```java
//ChannelPipeline实现类DefaultChannelPipeline的构造器方法
protected DefaultChannelPipeline(Channel channel) {
    this.channel = ObjectUtil.checkNotNull(channel, "channel");
    succeededFuture = new SucceededChannelFuture(channel, null);
    voidPromise =  new VoidChannelPromise(channel, true);
    //设置头结点head，尾结点tail
    tail = new TailContext(this);
    head = new HeadContext(this);
    
    head.next = tail;
    tail.prev = head;
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008213340873.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008213405599.png)

### 4. Bootstrap

> 网络编程里，“服务器”和“客户端”实际上表示了不同的网络行为；换句话说，是`监听传入的连接`还是`建立到一个或者多个进程的连接`。
>
> 因此，有两种类型的引导：一种用于`客户端`（简单地称为`Bootstrap`），而另一种（ServerBootstrap）用于`服务器`。无论你的应用程序使用哪种协议或者处理哪种类型的数据，唯一决定它使用哪种引导类的是它是作为一个客户端还是作为一个服务器。
>
> `服务器需要两组不同的Channel`。第一组将`只包含一个ServerChannel，代表服务器自身的已绑定到某个本地端口的正在监听的套接字`。而第二组将`包含所有已创建的用来处理传入客户端连接（对于每个服务器已经接受的连接都有一个）的Channel。`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008213530159.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211008213650025.png)

> 向Bootstrap 或ServerBootstrap 的实例提供你的ChannelInitializer 实现即可，并且一旦Channel 被注册到了它的EventLoop 之后，就会调用你的initChannel()版本。在`该方法返回之后，ChannelInitializer 的实例将会从ChannelPipeline 中移除它自己。`

```java
//使用匿名内部类的形式初始化通道对象
bootstrap.childHandler(new ChannelInitializer<SocketChannel>() {
    @Override
    protected void initChannel(SocketChannel socketChannel) throws Exception {
        //给pipeline管道设置自定义的处理器
        socketChannel.pipeline().addLast(new MyServerHandler());
    }
});
```

![image.png](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/ae5c6ed3008d4323aaa817e9cb46437a.png)

### 5. 通信传输模式

- NIO: io.netty.channel.socket.nio 使用java.nio.channels 包作为基础——`基于选择器的方式`
- Epoll: io.netty.channel.epoll 由 JNI 驱动的 epoll()和非阻塞 IO。这个传输支持`只有在Linux 上可用的多种特性`，如SO_REUSEPORT，比NIO 传输更快，而且是完全非阻塞的。将NioEventLoopGroup替换为EpollEventLoopGroup ， 并且将NioServerSocketChannel.class 替换为EpollServerSocketChannel.class 即可。
- OIO: io.netty.channel.socket.oio 使用java.net 包作为基础——`使用阻塞流`
- Local: io.netty.channel.local 可以在`VM 内部通过管道进行通信的本地传输`
- Embeded: io.netty.channel.embedded Embedded 传输，允许使用ChannelHandler 而又不需要一个真正的基于网络的传输。在测试ChannelHandler 实现时非常有用

```java
//server端代码，跟上面几乎一样，只需改三个地方
//这个地方使用的是OioEventLoopGroup
EventLoopGroup bossGroup = new OioEventLoopGroup();
ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.group(bossGroup)//只需要设置一个线程组boosGroup
        .channel(OioServerSocketChannel.class)//设置服务端通道实现类型

//client端代码，只需改两个地方
//使用的是OioEventLoopGroup
EventLoopGroup eventExecutors = new OioEventLoopGroup();
//通道类型设置为OioSocketChannel
bootstrap.group(eventExecutors)//设置线程组
        .channel(OioSocketChannel.class)//设置客户端的通道实现类型
```

### 6. Option 参数

> 1.Netty中的option主要是设置的ServerChannel的一些选项，而childOption主要是设置的ServerChannel的子Channel的选项。
>
> 2.如果是在客户端，因为是Bootstrap，只会有option而没有childOption，所以设置的是客户端Channel的选项.
>
> option主要是针对boss线程组，child主要是针对worker线程组;
>
>  childHandler()和childOption()都是给workerGroup （也就是group方法中的childGroup参数）进行设置的，option()和handler()都是给bossGroup（也就是group方法中的parentGroup参数）设置的。
>
> bossGroup是在服务器一启动就开始工作，负责监听客户端的连接请求。当建立连接后就交给了workGroup进行事务处理，两种是从不同的角度解释的。

#### .1. 通用参数

- CONNECT_TIMEOUT_MILLIS : 连接超时毫秒数，默认值30000毫秒即30秒。
- MAX_MESSAGES_PER_READ：一次Loop读取的最大消息数，对于ServerChannel或者NioByteChannel，默认值为16，其他Channel默认值为1。默认值这样设置，是因为：ServerChannel需要接受足够多的连接，保证大吞吐量，NioByteChannel可以减少不必要的系统调用select。
- WRITE_SPIN_COUNT：一个Loop写操作执行的最大次数，默认值为16。也就是说，对于大数据量的写操作至多进行16次，如果16次仍没有全部写完数据，此时会提交一个新的写任务给EventLoop，任务将在下次调度继续执行。这样，其他的写请求才能被响应不会因为单个大数据量写请求而耽误。
- ALLOCATOR：Netty参数，ByteBuf的分配器，默认值为ByteBufAllocator.DEFAULT，4.0版本为UnpooledByteBufAllocator，4.1版本为PooledByteBufAllocator。该值也可以使用系统参数io.netty.allocator.type配置，使用字符串值："unpooled"，"pooled"。
- RCVBUF_ALLOCATOR：用于Channel分配接受Buffer的分配器，默认值为AdaptiveRecvByteBufAllocator.DEFAULT，是一个自适应的接受缓冲区分配器，能根据接受到的数据自动调节大小。可选值为FixedRecvByteBufAllocator，固定大小的接受缓冲区分配器。
- AUTO_READ：自动读取，默认值为True。Netty只在必要的时候才设置关心相应的I/O事件。对于读操作，需要调用channel.read()设置关心的I/O事件为OP_READ，这样若有数据到达才能读取以供用户处理。该值为True时，每次读操作完毕后会自动调用channel.read()，从而有数据到达便能读取；否则，需要用户手动调用channel.read()。需要注意的是：当调用config.setAutoRead(boolean)方法时，如果状态由false变为true，将会调用channel.read()方法读取数据；由true变为false，将调用config.autoReadCleared()方法终止数据读取。
- WRITE_BUFFER_HIGH_WATER_MARK：写高水位标记，默认值64KB。如果Netty的写缓冲区中的字节超过该值，Channel的isWritable()返回False。
- WRITE_BUFFER_LOW_WATER_MARK：写低水位标记，默认值32KB。当Netty的写缓冲区中的字节超过高水位之后若下降到低水位，则Channel的isWritable()返回True。写高低水位标记使用户可以控制写入数据速度，从而实现流量控制。推荐做法是：每次调用channl.write(msg)方法首先调用channel.isWritable()判断是否可写。
- MESSAGE_SIZE_ESTIMATOR：消息大小估算器，默认为DefaultMessageSizeEstimator.DEFAULT。估算ByteBuf、ByteBufHolder和FileRegion的大小，其中ByteBuf和ByteBufHolder为实际大小，FileRegion估算值为0。该值估算的字节数在计算水位时使用，FileRegion为0可知FileRegion不影响高低水位。
- SINGLE_EVENTEXECUTOR_PER_GROUP：单线程执行ChannelPipeline中的事件，默认值为True。该值控制执行ChannelPipeline中执行ChannelHandler的线程。如果为Trye，整个pipeline由一个线程执行，这样不需要进行线程切换以及线程同步，是Netty4的推荐做法；如果为False，ChannelHandler中的处理过程会由Group中的不同线程执行。

#### .2. socketChannel

- SO_RCVBUF: TCP数据接收缓冲区大小。该缓冲区即TCP接收滑动窗口，linux操作系统可使用命令：cat /proc/sys/net/ipv4/tcp_rmem查询其大小。一般情况下，该值可由用户在任意时刻设置，但当设置值超过64KB时，需要在连接到远端之前设置。
- SO_SNDBUF: TCP数据发送缓冲区大小。该缓冲区即TCP发送滑动窗口，linux操作系统可使用命令：cat /proc/sys/net/ipv4/tcp_smem查询其大小。
- TCP_NODELAY: `立即发送数据`，默认值为Ture（Netty默认为True而操作系统默认为False）。该值设置`Nagle算法的启用`，改算法`将小的碎片数据连接成更大的报文来最小化所发送的报文的数量`，如果需要发送一些较小的报文，则需要禁用该算法。Netty默认禁用该算法，从而最小化报文传输延时。
- SO_KEEPALIVE: `连接保活`，默认值为False。启用该功能时，TCP会主动探测空闲连接的有效性。可以将此功能视为TCP的心跳机制，需要注意的是：默认的心跳间隔是7200s即2小时。Netty默认关闭该功能。
- SO_REUSEADDR: `地址复用，默认值False`。有四种情况可以使用：(1).当有一个有相同本地地址和端口的socket1处于TIME_WAIT状态时，而你希望启动的程序的socket2要占用该地址和端口，比如重启服务且保持先前端口。(2).有多块网卡或用IP Alias技术的机器在同一端口启动多个进程，但每个进程绑定的本地IP地址不能相同。(3).`单个进程绑定相同的端口到多个socket上，但每个socket绑定的ip地址不同`。(4).完全相同的地址和端口的重复绑定。但这只用于UDP的多播，不用于TCP。
- SO_LINGER: 关闭Socket的延迟时间，默认值为-1，表示禁用该功能。-1表示socket.close()方法立即返回，但OS底层会将发送缓冲区全部发送到对端。0表示socket.close()方法立即返回，OS放弃发送缓冲区的数据直接向对端发送RST包，对端收到复位错误。非0整数值表示调用socket.close()方法的线程被阻塞直到延迟时间到或发送缓冲区中的数据发送完毕，若超时，则对端会收到复位错误。
- IP_TOS： 设置IP头部的Type-of-Service字段，用于描述IP包的优先级和QoS选项。
- ALLOW_HALF_CLOSURE：一个连接的远端关闭时本地端是否关闭，默认值为False。值为False时，连接自动关闭；为True时，触发ChannelInboundHandler的userEventTriggered()方法，事件为ChannelInputShutdownEvent。

#### .3. ServerSocketChannel

- SO_RCVBUF: 当设置值超过64KB时，需要在绑定到本地端口前设置。该值设置的是由ServerSocketChannel使用accept接受的SocketChannel的接收缓冲区。
- SO_REUSEADDR
- SO_BACKLOG: `服务端接受连接的队列长度`，`如果队列已满，客户端连接将被拒绝。`默认值，Windows为200，其他为128。

###  Resource

- https://juejin.cn/post/6997776571910062094

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/netty-component/  

