# Reactor模型


> 1. 同步的等待多个事件源到达（采用select()实现）
> 2. 将事件多路分解以及分配相应的事件服务进行处理，这个分派采用server集中处理（dispatch）
> 3. 分解的事件以及对应的事件服务应用从分派服务中分离出去（handler）

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009164914588.png)

### 1. BIO模式

> 1. `同步阻塞IO，读写阻塞，线程等待时间过长`
> 2. 在制定线程策略的时候，只能`根据CPU的数目来限定可用线程资源，不能根据连接并发数目来制定`，也就是连接有限制。否则很难保证对客户端请求的高效和公平。
> 3. `多线程之间的上下文切换`，造成线程使用效率并不高，并且不易扩展
> 4. 状态数据以及其他需要保持一致的数据，需要采用并发同步控制

```java
// 主线程维护连接
public void run() {
    try {
        while (true) {
            Socket socket = serverSocket.accept();
            //提交线程池处理
            executorService.submit(new Handler(socket));
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
}
// 处理读写服务
class Handler implements Runnable {
    public void run() {
        try {
            //获取Socket的输入流，接收数据
            BufferedReader buf = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            String readData = buf.readLine();
            while (readData != null) {
                readData = buf.readLine();
                System.out.println(readData);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 2. NIO模式

> 1. 非阻塞的IO读写
> 2. 基于IO事件进行分发任务，同时支持对多个fd的监听

```java
public NIOServer(int port) throws Exception {
    selector = Selector.open();
    serverSocket = ServerSocketChannel.open();
    serverSocket.socket().bind(new InetSocketAddress(port));
    serverSocket.configureBlocking(false);
    serverSocket.register(selector, SelectionKey.OP_ACCEPT);
}
@Override
public void run() {
    while (!Thread.interrupted()) {
        try {
            //阻塞等待事件
            selector.select();
            // 事件列表
            Set selected = selector.selectedKeys();
            Iterator it = selected.iterator();
            while (it.hasNext()) {
                it.remove();
                //分发事件
                dispatch((SelectionKey) (it.next()));
            }
        } catch (Exception e) {

        }
    }
}
private void dispatch(SelectionKey key) throws Exception {
    if (key.isAcceptable()) {
        register(key);//新链接建立，注册
    } else if (key.isReadable()) {
        read(key);//读事件处理
    } else if (key.isWritable()) {
        wirete(key);//写事件处理
    }
}

private void register(SelectionKey key) throws Exception {
    ServerSocketChannel server = (ServerSocketChannel) key
        .channel();
    // 获得和客户端连接的通道
    SocketChannel channel = server.accept();
    channel.configureBlocking(false);
    //客户端通道注册到selector 上
    channel.register(this.selector, SelectionKey.OP_READ);
}
```

### 3. Reactor

> - **Reactor** 将I/O事件分派给对应的Handler
> - **Acceptor** 处理客户端新连接，并分派请求到处理器链中
> - **Handlers** 执行非阻塞读/写 任务

#### .1. 单Reactor单线程模型

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009192446053.png)

> 其中Reactor线程，负责多路分离套接字，有新连接到来触发connect 事件之后，交由Acceptor进行处理，有IO读写事件之后交给hanlder 处理。
>
> `Acceptor主要任务就是构建handler` ，在`获取到和client相关的SocketChannel之后 ，绑定到相应的hanlder上`，对应的SocketChannel有读写事件之后，基于racotor 分发,hanlder就可以处理了（所有的IO事件都绑定到selector上，有Reactor分发）。

```java
/**
    * 等待事件到来，分发事件处理
    */
class Reactor implements Runnable {
    private Reactor() throws Exception {
        SelectionKey sk =
            serverSocket.register(selector,
                                  SelectionKey.OP_ACCEPT);
        // attach Acceptor 处理新连接
        sk.attach(new Acceptor());
    }
    public void run() {
        try {
            while (!Thread.interrupted()) {
                selector.select();
                Set selected = selector.selectedKeys();
                Iterator it = selected.iterator();
                while (it.hasNext()) {
                    it.remove();
                    //分发事件处理
                    dispatch((SelectionKey) (it.next()));
                }
            }
        } catch (IOException ex) {
            //do something
        }
    }
    void dispatch(SelectionKey k) {
        // 若是连接事件获取是acceptor
        // 若是IO读写事件获取是handler
        Runnable runnable = (Runnable) (k.attachment());
        if (runnable != null) {
            runnable.run();
        }
    }
}
```

```java
 /**
    * 连接事件就绪,处理连接事件
    */
  class Acceptor implements Runnable {
      @Override
      public void run() {
          try {
              SocketChannel c = serverSocket.accept();
              if (c != null) {// 注册读写
                  new Handler(c, selector);
              }
          } catch (Exception e) {
          }
      }
  }
```

#### .2. 单Reactor多线程模型

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009193442676.png)

```java
/**
    * 多线程处理读写业务逻辑
    */
class MultiThreadHandler implements Runnable {
    public static final int READING = 0, WRITING = 1;
    int state;
    final SocketChannel socket;
    final SelectionKey sk;
    //多线程处理业务逻辑
    ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    public MultiThreadHandler(SocketChannel socket, Selector sl) throws Exception {
        this.state = READING;
        this.socket = socket;
        sk = socket.register(selector, SelectionKey.OP_READ);
        sk.attach(this);
        socket.configureBlocking(false);
    }
    @Override
    public void run() {
        if (state == READING) {
            read();
        } else if (state == WRITING) {
            write();
        }
    }
    private void read() {
        //任务异步处理
        executorService.submit(() -> process());
        //下一步处理写事件
        sk.interestOps(SelectionKey.OP_WRITE);
        this.state = WRITING;
    }
    private void write() {
        //任务异步处理
        executorService.submit(() -> process());
        //下一步处理读事件
        sk.interestOps(SelectionKey.OP_READ);
        this.state = READING;
    }
    /**
        * task 业务处理
        */
    public void process() {
        //do IO ,task,queue something
    }
}
```

#### .3. 多Reactor多线程模型

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009193654253.png)

> 1. mainReactor负责监听server socket，用来`处理新连接的建立`，将建立的socketChannel指定注册给subReactor。
> 2. subReactor维护自己的selector, 基于mainReactor 注册的socketChannel多路分离IO读写事件，读写网络数据，对业务处理的功能，另其扔给worker线程池来完成。

```java
/**
    * 多work 连接事件Acceptor,处理连接事件
    */
class MultiWorkThreadAcceptor implements Runnable {
    // cpu线程数相同多work线程
    int workCount =Runtime.getRuntime().availableProcessors();
    SubReactor[] workThreadHandlers = new SubReactor[workCount];
    volatile int nextHandler = 0;
    public MultiWorkThreadAcceptor() {
        this.init();
    }
    public void init() {
        nextHandler = 0;
        for (int i = 0; i < workThreadHandlers.length; i++) {
            try {
                workThreadHandlers[i] = new SubReactor();
            } catch (Exception e) {
            }
        }
    }
    @Override
    public void run() {
        try {
            SocketChannel c = serverSocket.accept();
            if (c != null) {// 注册读写
                synchronized (c) {
                    // 顺序获取SubReactor，然后注册channel 
                    SubReactor work = workThreadHandlers[nextHandler];
                    work.registerChannel(c);
                    nextHandler++;
                    if (nextHandler >= workThreadHandlers.length) {
                        nextHandler = 0;
                    }
                }
            }
        } catch (Exception e) {
        }
    }
}

/**
    * 多work线程处理读写业务逻辑
    */
class SubReactor implements Runnable {
    final Selector mySelector;
    //多线程处理业务逻辑
    int workCount =Runtime.getRuntime().availableProcessors();
    ExecutorService executorService = Executors.newFixedThreadPool(workCount);
    public SubReactor() throws Exception {
        // 每个SubReactor 一个selector 
        this.mySelector = SelectorProvider.provider().openSelector();
    }
    /**
        * 注册chanel
        *
        * @param sc
        * @throws Exception
        */
    public void registerChannel(SocketChannel sc) throws Exception {
        sc.register(mySelector, SelectionKey.OP_READ | SelectionKey.OP_CONNECT);
    }
    @Override
    public void run() {
        while (true) {
            try {
                //每个SubReactor 自己做事件分派处理读写事件
                selector.select();
                Set<SelectionKey> keys = selector.selectedKeys();
                Iterator<SelectionKey> iterator = keys.iterator();
                while (iterator.hasNext()) {
                    SelectionKey key = iterator.next();
                    iterator.remove();
                    if (key.isReadable()) {
                        read();
                    } else if (key.isWritable()) {
                        write();
                    }
                }
            } catch (Exception e) {
            }
        }
    }
    private void read() {
        //任务异步处理
        executorService.submit(() -> process());
    }
    private void write() {
        //任务异步处理
        executorService.submit(() -> process());
    }
    /**
        * task 业务处理
        */
    public void process() {
        //do IO ,task,queue something
    }
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/reactor%E6%A8%A1%E5%9E%8B/  

