# IO多路复用


> From：https://www.pdai.tech/md/java/io/java-io-nio-select-epoll.html
>
> 多路复用IO技术最适用的是“高并发”场景，所谓高并发是指1毫秒内至少同时有上千个连接请求准备好。其他情况下多路复用IO技术发挥不出来它的优势。另一方面，使用JAVA NIO进行功能实现，相对于传统的Socket套接字实现要复杂一些，所以实际应用中，需要根据自己的业务需求进行技术选择。目前流程的多路复用IO实现主要包括四种: `select`、`poll`、`epoll`、`kqueue`。下表是他们的一些重要特性的比较:

| IO模型 | 相对性能 | 关键思路         | 操作系统      | JAVA支持情况                                                 |
| ------ | -------- | ---------------- | ------------- | ------------------------------------------------------------ |
| select | 较高     | Reactor          | windows/Linux | 支持,Reactor模式(反应器设计模式)。Linux操作系统的 kernels 2.4内核版本之前，默认使用select；而目前windows下对同步IO的支持，都是select模型 |
| poll   | 较高     | Reactor          | Linux         | Linux下的JAVA NIO框架，Linux kernels 2.6内核版本之前使用poll进行支持。也是使用的Reactor模式 |
| epoll  | 高       | Reactor/Proactor | Linux         | Linux kernels 2.6内核版本及以后使用epoll进行支持；Linux kernels 2.6内核版本之前使用poll进行支持；另外一定注意，由于Linux下没有Windows下的IOCP技术提供真正的 异步IO 支持，所以Linux下使用epoll模拟异步IO |
| kqueue | 高       | Proactor         | Linux         | 目前JAVA的版本不支持                                         |

### 1. 传统IO模型

- 每个客户端连接到达之后，服务端会分配一个线程给该客户端，该线程会处理包括读取数据，解码，业务计算，编码，以及发送数据整个过程；
- 同一时刻，服务端的吞吐量与服务器所提供的线程数量是呈线性关系的。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210803133056043.png)

- 服务器的并发量对服务端能够创建的线程数有很大的依赖关系，但是`服务器线程却是不能无限增长的`；
- 服务端`每个线程不仅要进行IO读写操作，而且还需要进行业务计算`；
- 服务端在`获取客户端连接，读取数据，以及写入数据的过程都是阻塞类型的，在网络状况不好的情况下，这将极大的降低服务器每个线程的利用率`，从而降低服务器吞吐量。

### 2. Reactor 模型

> 在Reactor模型中，主要有四个角色：`客户端连接，Reactor，Acceptor和Handler`。这里Acceptor会不断地接收客户端的连接，然后`将接收到的连接交由Reactor进行分发`，最后有具体的Handler进行处理。改进后的Reactor模型相对于传统的IO模型主要有如下优点：
>
> - 从模型上来讲，如果仅仅还是只使用一个线程池来处理客户端连接的网络读写，以及业务计算，那么Reactor模型与传统IO模型在效率上并没有什么提升。但是Reactor模型是`以事件进行驱动的，其能够将接收客户端连接，+ 网络读和网络写，以及业务计算进行拆分`，从而极大的提升处理效率；
> - Reactor模型是`异步非阻塞模型`，工作线程在没有网络事件时可以处理其他的任务，而不用像传统IO那样必须阻塞等待。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210803133613944.png)

```java
public class Reactor implements Runnable {
    private final Selector selector;
    private final ServerSocketChannel serverSocket;
    public Reactor(int port) throws IOException {
        serverSocket = ServerSocketChannel.open();  // 创建服务端的ServerSocketChannel
        serverSocket.configureBlocking(false);  // 设置为非阻塞模式
        selector = Selector.open();  // 创建一个Selector多路复用器
        SelectionKey key = serverSocket.register(selector, SelectionKey.OP_ACCEPT);
        serverSocket.bind(new InetSocketAddress(port));  // 绑定服务端端口
        key.attach(new Acceptor(serverSocket));  // 为服务端Channel绑定一个Acceptor
    }
    @Override
    public void run() {
        try {
            while (!Thread.interrupted()) {
                selector.select();  // 服务端使用一个线程不断等待客户端的连接到达
                Set<SelectionKey> keys = selector.selectedKeys();
                Iterator<SelectionKey> iterator = keys.iterator();
                while (iterator.hasNext()) {
                    dispatch(iterator.next());  // 监听到客户端连接事件后将其分发给Acceptor
                    iterator.remove();
                }

                selector.selectNow();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void dispatch(SelectionKey key) throws IOException {
        // 这里的attachement也即前面为服务端Channel绑定的Acceptor，调用其run()方法进行
        // 客户端连接的获取，并且进行分发
        Runnable attachment = (Runnable) key.attachment();
        attachment.run();
    }
}
```

```java
public class Acceptor implements Runnable {
  private final ExecutorService executor = Executors.newFixedThreadPool(20);

  private final ServerSocketChannel serverSocket;

  public Acceptor(ServerSocketChannel serverSocket) {
    this.serverSocket = serverSocket;
  }

  @Override
  public void run() {
    try {
      SocketChannel channel = serverSocket.accept();  // 获取客户端连接
      if (null != channel) {
        executor.execute(new Handler(channel));  // 将客户端连接交由线程池处理
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
```

```java
public class Handler implements Runnable {
    private volatile static Selector selector;
    private final SocketChannel channel;
    private SelectionKey key;
    private volatile ByteBuffer input = ByteBuffer.allocate(1024);
    private volatile ByteBuffer output = ByteBuffer.allocate(1024);

    public Handler(SocketChannel channel) throws IOException {
        this.channel = channel;
        channel.configureBlocking(false);  // 设置客户端连接为非阻塞模式
        selector = Selector.open();  // 为客户端创建一个新的多路复用器
        key = channel.register(selector, SelectionKey.OP_READ);  // 注册客户端Channel的读事件
    }

    @Override
    public void run() {
        try {
            while (selector.isOpen() && channel.isOpen()) {
                Set<SelectionKey> keys = select();  // 等待客户端事件发生
                Iterator<SelectionKey> iterator = keys.iterator();
                while (iterator.hasNext()) {
                    SelectionKey key = iterator.next();
                    iterator.remove();

                    // 如果当前是读事件，则读取数据
                    if (key.isReadable()) {
                        read(key);
                    } else if (key.isWritable()) {
                        // 如果当前是写事件，则写入数据
                        write(key);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 这里处理的主要目的是处理Jdk的一个bug，该bug会导致Selector被意外触发，但是实际上没有任何事件到达，
    // 此时的处理方式是新建一个Selector，然后重新将当前Channel注册到该Selector上
    private Set<SelectionKey> select() throws IOException {
        selector.select();
        Set<SelectionKey> keys = selector.selectedKeys();
        if (keys.isEmpty()) {
            int interestOps = key.interestOps();
            selector = Selector.open();
            key = channel.register(selector, interestOps);
            return select();
        }

        return keys;
    }

    // 读取客户端发送的数据
    private void read(SelectionKey key) throws IOException {
        channel.read(input);
        if (input.position() == 0) {
            return;
        }

        input.flip();
        process();  // 对读取的数据进行业务处理
        input.clear();
        key.interestOps(SelectionKey.OP_WRITE);  // 读取完成后监听写入事件
    }

    private void write(SelectionKey key) throws IOException {
        output.flip();
        if (channel.isOpen()) {
            channel.write(output);  // 当有写入事件时，将业务处理的结果写入到客户端Channel中
            key.channel();
            channel.close();
            output.clear();
        }
    }

    // 进行业务处理，并且获取处理结果。本质上，基于Reactor模型，如果这里成为处理瓶颈，
    // 则直接将其处理过程放入线程池即可，并且使用一个Future获取处理结果，最后写入客户端Channel
    private void process() {
        byte[] bytes = new byte[input.remaining()];
        input.get(bytes);
        String message = new String(bytes, CharsetUtil.UTF_8);
        System.out.println("receive message from client: \n" + message);

        output.put("hello client".getBytes());
    }
}
```

### 3. java对多路复用支持

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/picture/java-io-nio-1.png)

#### .1. channel

> 通道，`被建立的一个应用程序和操作系统交互事件、传递内容的渠道`(注意是连接到操作系统)。一个通道会有一个`专属的文件状态描述符`。那么既然是和操作系统进行内容的传递，那么说明应用程序可以通过通道读取数据，也可以通过通道向操作系统写数据。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/java-io-nio-2.png)

- ServerSocketChannel: 应用服务器程序的监听通道。只有通过这个通道，应用程序才能向操作系统注册支持“多路复用IO”的端口监听。同时支持UDP协议和TCP协议。
- ScoketChannel: `TCP Socket套接字的监听通道`，一个Socket套接字对应了一个客户端IP: 端口 到 服务器IP: 端口的通信连接。
- DatagramChannel: `UDP 数据报文的监听通道`。

#### .2. Buffer

> 数据缓存区: 在JAVA NIO 框架中，为了保证每个通道的数据读写速度JAVA NIO 框架为每一种需要支持数据读写的通道集成了Buffer的支持. 在读模式下，应用程序只能从Buffer中读取数据，不能进行写操作。但是在写模式下，应用程序是可以进行读操作的，这就表示可能会出现脏读的情况。所以一旦您决定要从Buffer中读取数据，一定要将Buffer的状态改为读模式。

### 4. demo-多路复用

```java
package testNSocket;

import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.URLDecoder;
import java.net.URLEncoder;

import java.nio.ByteBuffer;
import java.nio.channels.SelectableChannel;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.BasicConfigurator;

public class SocketServer2 {

    static {
        BasicConfigurator.configure();
    }

    /**
     * 日志
     */
    private static final Log LOGGER = LogFactory.getLog(SocketServer2.class);

    /**
     * 改进的java nio server的代码中，由于buffer的大小设置的比较小。
     * 我们不再把一个client通过socket channel多次传给服务器的信息保存在beff中了(因为根本存不下)<br>
     * 我们使用socketchanel的hashcode作为key(当然您也可以自己确定一个id)，信息的stringbuffer作为value，存储到服务器端的一个内存区域MESSAGEHASHCONTEXT。
     * 
     * 如果您不清楚ConcurrentHashMap的作用和工作原理，请自行百度/Google
     */
    private static final ConcurrentMap<Integer, StringBuffer> MESSAGEHASHCONTEXT = new ConcurrentHashMap<Integer , StringBuffer>();

    public static void main(String[] args) throws Exception {
        ServerSocketChannel serverChannel = ServerSocketChannel.open();
        serverChannel.configureBlocking(false);
        ServerSocket serverSocket = serverChannel.socket();
        serverSocket.setReuseAddress(true);
        serverSocket.bind(new InetSocketAddress(83));

        Selector selector = Selector.open();
        //注意、服务器通道只能注册SelectionKey.OP_ACCEPT事件
        serverChannel.register(selector, SelectionKey.OP_ACCEPT);

        try {
            while(true) {
                //如果条件成立，说明本次询问selector，并没有获取到任何准备好的、感兴趣的事件
                //java程序对多路复用IO的支持也包括了阻塞模式 和非阻塞模式两种。
                if(selector.select(100) == 0) {
                    //================================================
                    //      这里视业务情况，可以做一些然并卵的事情
                    //================================================
                    continue;
                }
                //这里就是本次询问操作系统，所获取到的“所关心的事件”的事件类型(每一个通道都是独立的)
                Iterator<SelectionKey> selecionKeys = selector.selectedKeys().iterator();

                while(selecionKeys.hasNext()) {
                    SelectionKey readyKey = selecionKeys.next();
                    //这个已经处理的readyKey一定要移除。如果不移除，就会一直存在在selector.selectedKeys集合中
                    //待到下一次selector.select() > 0时，这个readyKey又会被处理一次
                    selecionKeys.remove();

                    SelectableChannel selectableChannel = readyKey.channel();
                    if(readyKey.isValid() && readyKey.isAcceptable()) {
                        SocketServer2.LOGGER.info("======channel通道已经准备好=======");
                        /*
                         * 当server socket channel通道已经准备好，就可以从server socket channel中获取socketchannel了
                         * 拿到socket channel后，要做的事情就是马上到selector注册这个socket channel感兴趣的事情。
                         * 否则无法监听到这个socket channel到达的数据
                         * */
                        ServerSocketChannel serverSocketChannel = (ServerSocketChannel)selectableChannel;
                        SocketChannel socketChannel = serverSocketChannel.accept();
                        registerSocketChannel(socketChannel , selector);

                    } else if(readyKey.isValid() && readyKey.isConnectable()) {
                        SocketServer2.LOGGER.info("======socket channel 建立连接=======");
                    } else if(readyKey.isValid() && readyKey.isReadable()) {
                        SocketServer2.LOGGER.info("======socket channel 数据准备完成，可以去读==读取=======");
                        readSocketChannel(readyKey);
                    }
                }
            }
        } catch(Exception e) {
            SocketServer2.LOGGER.error(e.getMessage() , e);
        } finally {
            serverSocket.close();
        }
    }

    /**
     * 在server socket channel接收到/准备好 一个新的 TCP连接后。
     * 就会向程序返回一个新的socketChannel。<br>
     * 但是这个新的socket channel并没有在selector“选择器/代理器”中注册，
     * 所以程序还没法通过selector通知这个socket channel的事件。
     * 于是我们拿到新的socket channel后，要做的第一个事情就是到selector“选择器/代理器”中注册这个
     * socket channel感兴趣的事件
     * @param socketChannel 新的socket channel
     * @param selector selector“选择器/代理器”
     * @throws Exception
     */
    private static void registerSocketChannel(SocketChannel socketChannel , Selector selector) throws Exception {
        socketChannel.configureBlocking(false);
        //socket通道可以且只可以注册三种事件SelectionKey.OP_READ | SelectionKey.OP_WRITE | SelectionKey.OP_CONNECT
        //最后一个参数视为 为这个socketchanne分配的缓存区
        socketChannel.register(selector, SelectionKey.OP_READ , ByteBuffer.allocate(50));
    }

    /**
     * 这个方法用于读取从客户端传来的信息。
     * 并且观察从客户端过来的socket channel在经过多次传输后，是否完成传输。
     * 如果传输完成，则返回一个true的标记。
     * @param socketChannel
     * @throws Exception
     */
    private static void readSocketChannel(SelectionKey readyKey) throws Exception {
        SocketChannel clientSocketChannel = (SocketChannel)readyKey.channel();
        //获取客户端使用的端口
        InetSocketAddress sourceSocketAddress = (InetSocketAddress)clientSocketChannel.getRemoteAddress();
        Integer resoucePort = sourceSocketAddress.getPort();

        //拿到这个socket channel使用的缓存区，准备读取数据
        //在后文，将详细讲解缓存区的用法概念，实际上重要的就是三个元素capacity,position和limit。
        ByteBuffer contextBytes = (ByteBuffer)readyKey.attachment();
        //将通道的数据写入到缓存区，注意是写入到缓存区。
        //这次，为了演示buff的使用方式，我们故意缩小了buff的容量大小到50byte，
        //以便演示channel对buff的多次读写操作
        int realLen = 0;
        StringBuffer message = new StringBuffer();
        //这句话的意思是，将目前通道中的数据写入到缓存区
        //最大可写入的数据量就是buff的容量
        while((realLen = clientSocketChannel.read(contextBytes)) != 0) {

            //一定要把buffer切换成“读”模式，否则由于limit = capacity
            //在read没有写满的情况下，就会导致多读
            contextBytes.flip();
            int position = contextBytes.position();
            int capacity = contextBytes.capacity();
            byte[] messageBytes = new byte[capacity];
            contextBytes.get(messageBytes, position, realLen);

            //这种方式也是可以读取数据的，而且不用关心position的位置。
            //因为是目前contextBytes所有的数据全部转出为一个byte数组。
            //使用这种方式时，一定要自己控制好读取的最终位置(realLen很重要)
            //byte[] messageBytes = contextBytes.array();

            //注意中文乱码的问题，我个人喜好是使用URLDecoder/URLEncoder，进行解编码。
            //当然java nio框架本身也提供编解码方式，看个人咯
            String messageEncode = new String(messageBytes , 0 , realLen , "UTF-8");
            message.append(messageEncode);

            //再切换成“写”模式，直接情况缓存的方式，最快捷
            contextBytes.clear();
        }

        //如果发现本次接收的信息中有over关键字，说明信息接收完了
        if(URLDecoder.decode(message.toString(), "UTF-8").indexOf("over") != -1) {
            //则从messageHashContext中，取出之前已经收到的信息，组合成完整的信息
            Integer channelUUID = clientSocketChannel.hashCode();
            SocketServer2.LOGGER.info("端口:" + resoucePort + "客户端发来的信息======message : " + message);
            StringBuffer completeMessage;
            //清空MESSAGEHASHCONTEXT中的历史记录
            StringBuffer historyMessage = MESSAGEHASHCONTEXT.remove(channelUUID);
            if(historyMessage == null) {
                completeMessage = message;
            } else {
                completeMessage = historyMessage.append(message);
            }
            SocketServer2.LOGGER.info("端口:" + resoucePort + "客户端发来的完整信息======completeMessage : " + URLDecoder.decode(completeMessage.toString(), "UTF-8"));

            //======================================================
            //          当然接受完成后，可以在这里正式处理业务了        
            //======================================================

            //回发数据，并关闭channel
            ByteBuffer sendBuffer = ByteBuffer.wrap(URLEncoder.encode("回发处理结果", "UTF-8").getBytes());
            clientSocketChannel.write(sendBuffer);
            clientSocketChannel.close();
        } else {
            //如果没有发现有“over”关键字，说明还没有接受完，则将本次接受到的信息存入messageHashContext
            SocketServer2.LOGGER.info("端口:" + resoucePort + "客户端信息还未接受完，继续接受======message : " + URLDecoder.decode(message.toString(), "UTF-8"));
            //每一个channel对象都是独立的，所以可以使用对象的hash值，作为唯一标示
            Integer channelUUID = clientSocketChannel.hashCode();

            //然后获取这个channel下以前已经达到的message信息
            StringBuffer historyMessage = MESSAGEHASHCONTEXT.get(channelUUID);
            if(historyMessage == null) {
                historyMessage = new StringBuffer();
                MESSAGEHASHCONTEXT.put(channelUUID, historyMessage.append(message));
            }
        }
    }
}
```

### 4. demo-异步

```java
package testASocket;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousChannelGroup;
import java.nio.channels.AsynchronousServerSocketChannel;
import java.nio.channels.AsynchronousSocketChannel;
import java.nio.channels.CompletionHandler;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.BasicConfigurator;

/**
 * @author yinwenjie
 */
public class SocketServer {

    static {
        BasicConfigurator.configure();
    }

    private static final Object waitObject = new Object();

    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        /*
         * 对于使用的线程池技术，我一定要多说几句
         * 1、Executors是线程池生成工具，通过这个工具我们可以很轻松的生成“固定大小的线程池”、“调度池”、“可伸缩线程数量的池”。具体请看API Doc
         * 2、当然您也可以通过ThreadPoolExecutor直接生成池。
         * 3、这个线程池是用来得到操作系统的“IO事件通知”的，不是用来进行“得到IO数据后的业务处理的”。要进行后者的操作，您可以再使用一个池(最好不要混用)
         * 4、您也可以不使用线程池(不推荐)，如果决定不使用线程池，直接AsynchronousServerSocketChannel.open()就行了。
         * */
        ExecutorService threadPool = Executors.newFixedThreadPool(20);
        AsynchronousChannelGroup group = AsynchronousChannelGroup.withThreadPool(threadPool);
        final AsynchronousServerSocketChannel serverSocket = AsynchronousServerSocketChannel.open(group);

        //设置要监听的端口“0.0.0.0”代表本机所有IP设备
        serverSocket.bind(new InetSocketAddress("0.0.0.0", 83));
        //为AsynchronousServerSocketChannel注册监听，注意只是为AsynchronousServerSocketChannel通道注册监听
        //并不包括为 随后客户端和服务器 socketchannel通道注册的监听
        serverSocket.accept(null, new ServerSocketChannelHandle(serverSocket));

        //等待，以便观察现象(这个和要讲解的原理本身没有任何关系，只是为了保证守护线程不会退出)
        synchronized(waitObject) {
            waitObject.wait();
        }
    }
}

/**
 * 这个处理器类，专门用来响应 ServerSocketChannel 的事件。
 * @author yinwenjie
 */
class ServerSocketChannelHandle implements CompletionHandler<AsynchronousSocketChannel, Void> {
    /**
     * 日志
     */
    private static final Log LOGGER = LogFactory.getLog(ServerSocketChannelHandle.class);

    private AsynchronousServerSocketChannel serverSocketChannel;

    /**
     * @param serverSocketChannel
     */
    public ServerSocketChannelHandle(AsynchronousServerSocketChannel serverSocketChannel) {
        this.serverSocketChannel = serverSocketChannel;
    }

    /**
     * 注意，我们分别观察 this、socketChannel、attachment三个对象的id。
     * 来观察不同客户端连接到达时，这三个对象的变化，以说明ServerSocketChannelHandle的监听模式
     */
    @Override
    public void completed(AsynchronousSocketChannel socketChannel, Void attachment) {
        ServerSocketChannelHandle.LOGGER.info("completed(AsynchronousSocketChannel result, ByteBuffer attachment)");
        //每次都要重新注册监听(一次注册，一次响应)，但是由于“文件状态标示符”是独享的，所以不需要担心有“漏掉的”事件
        this.serverSocketChannel.accept(attachment, this);

        //为这个新的socketChannel注册“read”事件，以便操作系统在收到数据并准备好后，主动通知应用程序
        //在这里，由于我们要将这个客户端多次传输的数据累加起来一起处理，所以我们将一个stringbuffer对象作为一个“附件”依附在这个channel上
        //
        ByteBuffer readBuffer = ByteBuffer.allocate(50);
        socketChannel.read(readBuffer, new StringBuffer(), new SocketChannelReadHandle(socketChannel , readBuffer));
    }

    /* (non-Javadoc)
     * @see java.nio.channels.CompletionHandler#failed(java.lang.Throwable, java.lang.Object)
     */
    @Override
    public void failed(Throwable exc, Void attachment) {
        ServerSocketChannelHandle.LOGGER.info("failed(Throwable exc, ByteBuffer attachment)");
    }
}

/**
 * 负责对每一个socketChannel的数据获取事件进行监听。<p>
 * 
 * 重要的说明: 一个socketchannel都会有一个独立工作的SocketChannelReadHandle对象(CompletionHandler接口的实现)，
 * 其中又都将独享一个“文件状态标示”对象FileDescriptor、
 * 一个独立的由程序员定义的Buffer缓存(这里我们使用的是ByteBuffer)、
 * 所以不用担心在服务器端会出现“窜对象”这种情况，因为JAVA AIO框架已经帮您组织好了。<p>
 * 
 * 但是最重要的，用于生成channel的对象: AsynchronousChannelProvider是单例模式，无论在哪组socketchannel，
 * 对是一个对象引用(但这没关系，因为您不会直接操作这个AsynchronousChannelProvider对象)。
 * @author yinwenjie
 */
class SocketChannelReadHandle implements CompletionHandler<Integer, StringBuffer> {
    /**
     * 日志
     */
    private static final Log LOGGER = LogFactory.getLog(SocketChannelReadHandle.class);

    private AsynchronousSocketChannel socketChannel;

    /**
     * 专门用于进行这个通道数据缓存操作的ByteBuffer<br>
     * 当然，您也可以作为CompletionHandler的attachment形式传入。<br>
     * 这是，在这段示例代码中，attachment被我们用来记录所有传送过来的Stringbuffer了。
     */
    private ByteBuffer byteBuffer;

    public SocketChannelReadHandle(AsynchronousSocketChannel socketChannel , ByteBuffer byteBuffer) {
        this.socketChannel = socketChannel;
        this.byteBuffer = byteBuffer;
    }

    /* (non-Javadoc)
     * @see java.nio.channels.CompletionHandler#completed(java.lang.Object, java.lang.Object)
     */
    @Override
    public void completed(Integer result, StringBuffer historyContext) {
        //如果条件成立，说明客户端主动终止了TCP套接字，这时服务端终止就可以了
        if(result == -1) {
            try {
                this.socketChannel.close();
            } catch (IOException e) {
                SocketChannelReadHandle.LOGGER.error(e);
            }
            return;
        }

        SocketChannelReadHandle.LOGGER.info("completed(Integer result, Void attachment) : 然后我们来取出通道中准备好的值");
        /*
         * 实际上，由于我们从Integer result知道了本次channel从操作系统获取数据总长度
         * 所以实际上，我们不需要切换成“读模式”的，但是为了保证编码的规范性，还是建议进行切换。
         * 
         * 另外，无论是JAVA AIO框架还是JAVA NIO框架，都会出现“buffer的总容量”小于“当前从操作系统获取到的总数据量”，
         * 但区别是，JAVA AIO框架中，我们不需要专门考虑处理这样的情况，因为JAVA AIO框架已经帮我们做了处理(做成了多次通知)
         * */
        this.byteBuffer.flip();
        byte[] contexts = new byte[1024];
        this.byteBuffer.get(contexts, 0, result);
        this.byteBuffer.clear();
        try {
            String nowContent = new String(contexts , 0 , result , "UTF-8");
            historyContext.append(nowContent);
            SocketChannelReadHandle.LOGGER.info("================目前的传输结果: " + historyContext);
        } catch (UnsupportedEncodingException e) {
            SocketChannelReadHandle.LOGGER.error(e);
        }

        //如果条件成立，说明还没有接收到“结束标记”
        if(historyContext.indexOf("over") == -1) {
            return;
        }

        //=========================================================================
        //          和上篇文章的代码相同，我们以“over”符号作为客户端完整信息的标记
        //=========================================================================
        SocketChannelReadHandle.LOGGER.info("=======收到完整信息，开始处理业务=========");
        historyContext = new StringBuffer();

        //还要继续监听(一次监听一次通知)
        this.socketChannel.read(this.byteBuffer, historyContext, this);
    }

    /* (non-Javadoc)
     * @see java.nio.channels.CompletionHandler#failed(java.lang.Throwable, java.lang.Object)
     */
    @Override
    public void failed(Throwable exc, StringBuffer historyContext) {
        SocketChannelReadHandle.LOGGER.info("=====发现客户端异常关闭，服务器将关闭TCP通道");
        try {
            this.socketChannel.close();
        } catch (IOException e) {
            SocketChannelReadHandle.LOGGER.error(e);
        }
    }
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/io%E5%A4%9A%E8%B7%AF%E5%A4%8D%E7%94%A8/  

