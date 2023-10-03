# Concurrency


![](https://gitee.com/github-25970295/blogpictureV2/raw/master/java-concurrent-overview-1.png)

### 1. 并发问题的根源

- 可见性：一个线程对共享变量的修改，另外一个线程能够立刻看到。
- 原子性：即一个操作或者多个操作 要么全部执行并且执行的过程不会被任何因素打断，要么就都不执行。
- 有序性：即程序执行的顺序按照代码的先后顺序执行。但从jvm中可以看到，在执行程序时为了提高性能，编译器和处理器常常会对指令做重排序。

### 2. 线程安全

#### .1. 不可变性

> `不可变(Immutable)的对象一定是线程安全的`，不需要再采取任何的线程安全保障措施。只要一个不可变的对象被正确地构建出来，永远也不会看到它在多个线程之中处于不一致的状态。

- final 关键字修饰的基本数据类型
- String
- 枚举类型
- Number 部分子类，如 Long 和 Double 等数值包装类型，BigInteger 和 BigDecimal 等大数据类型。但同为 Number 的原子类 AtomicInteger 和 AtomicLong 则是可变的。
- 对于集合类型，可以使用 Collections.unmodifiableXXX() 方法来获取一个不可变的集合。

> 1）immutable对象的状态在创建之后就不能发生改变，`任何对它的改变都应该产生一个新的对象`。
>
> 2）`Immutable类的所有的成员都应该是private final的`。通过这种方式保证成员变量不可改变。但只做到这一步还不够，因为如果成员变量是对象，它保存的只是引用，有可能在外部改变其引用指向的值，所以第5点弥补这个不足
>
> 3）对象必须被正确的创建，比如：`对象引用在对象创建过程中不能泄露`。
>
> 4）只提供读取成员变量的get方法，不提供改变成员变量的set方法，避免通过其他接口改变成员变量的值，破坏不可变特性。
>
> 5）类应该是final的，`保证类不被继承`，如果类可以被继承会破坏类的不可变性机制，只要继承类覆盖父类的方法并且继承类可以改变成员变量值，那么一旦子类以父类的形式出现时，不能保证当前类是否可变。
>
> 6）如果`类中包含mutable类对象，那么返回给客户端的时候，返回该对象的一个深拷贝，而不是该对象本身`（该条可以归为第一条中的一个特例）

```java
//String对象在内存创建后就不可改变，不可变对象的创建一般满足以上原则,但可以通过反射机制修改其中的值。
public final class String
    implements java.io.Serializable, Comparable<String>, CharSequence
{
    private final char value[]; 	/** The value is used for character storage. */
    /** The offset is the first index of the storage that is used. */
    private final int offset;
    /** The count is the number of characters in the String. */
    private final int count;
    private int hash; // Default to 0
    public String(char value[]) {
        this.value = Arrays.copyOf(value, value.length); // deep copy操作
    }
    public char[] toCharArray() {
        char result[] = new char[value.length];
        System.arraycopy(value, 0, result, 0, value.length);
        return result;
    }
    ...
}
```

#### .2. 相对线程安全

> 相对线程安全需要保证对这个对象单独的操作是线程安全的，在调用的时候不需要做额外的保障措施。但是对于一些特定顺序的连续调用，就可能需要在调用端使用额外的同步手段来保证调用的正确性。在 Java 语言中，大部分的线程安全类都属于这种类型，例如` Vector、HashTable、Collections 的 synchronizedCollection() 方法包装的集合`等。

### 3. 线程安全实现

- 互斥同步： synchronized & ReentrantLock
- 非阻塞同步： 
  - CAS： 硬件支持的原子性操作最典型的是: 比较并交换(Compare-and-Swap，CAS)。CAS 指令需要有 3 个操作数，分别是内存地址 V、旧的预期值 A 和新值 B。当执行操作时，只有当 V 的值等于 A，才将 V 的值更新为 B。
  - AtomicInteger: J.U.C 包里面的整数原子类 AtomicInteger，其中的 compareAndSet() 和 getAndIncrement() 等方法都使用了 Unsafe 类的 CAS 操作。
  - ABA:  J.U.C 包提供了一个带有标记的原子引用类 AtomicStampedReference 来解决这个问题，它可以通过控制变量值的版本来保证 CAS 的正确性。大部分情况下 ABA 问题不会影响程序并发的正确性，如果需要解决 ABA 问题，改用传统的互斥同步可能会比原子类更高效。

- 无同步方案： 不涉及共享数据
  - Thread Local
  - Reentrant Code


### 4. 线程使用方式

- 实现Runnable接口
- 实现Callable接口

```java
public static void main(String[] args) {
    ExecutorService executorService = Executors.newCachedThreadPool();
    FutureTask<String> futureTask = new FutureTask<>(new MyCallable());
    executorService.submit(futureTask);
    executorService.shutdown();
    System.out.println("do something in main");
    try {
        System.out.println("得到异步任务返回结果：" + futureTask.get());
    } catch (Exception e) {
        System.out.println("捕获异常成功：" + e.getMessage());
    }
    System.out.println("主线程结束");
 
}
static class MyCallable implements Callable<String> {
 
    @Override
    public String call() throws Exception {
        System.out.println("Callable子线程开始");
        Thread.sleep(2000); // 模拟做点事情
        System.out.println("Callable子线程结束");
        throw new NullPointerException("抛异常测试");
    }
}
```

- 继承Thread类

### 5. 并发与模式

#### 1. Immutability 模式

- **将一个类所有的属性都设置成 final 的，并且只允许存在只读方法，那么这个类基本上就具备不可变性了**。更严格的做法是**这个类本身也是 final 的**，也就是不允许继承。因为子类可以覆盖父类的方法，有可能改变不可变性。
-  Java 中的 Long、Integer、Short、Byte 等基本数据类型的包装类的实现。

```java
//Foo 线程安全
final class Foo{
  final int age=0;
  final int name="abc";
}
//Bar 线程不安全
class Bar {
  Foo foo;
  void setFoo(Foo f){   // 这里可以修改 foo属性值
    this.foo=f;
  }
}
```

```java
public class SafeWM {
  class WMRange{
    final int upper;
    final int lower;
    WMRange(int upper,int lower){
    // 省略构造函数实现
    }
  }
  final AtomicReference<WMRange>
    rf = new AtomicReference<>(
      new WMRange(0,0)
    );
  // 设置库存上限
  void setUpper(int v){
    while(true){
      WMRange or = rf.get();
      // 检查参数合法性
      if(v < or.lower){
        throw new IllegalArgumentException();
      }
      WMRange nr = new
          WMRange(v, or.lower);
      if(rf.compareAndSet(or, nr)){  //CAS 模式
        return;
      }
    }
  }
}	
```

#### 2. CopyOnWrite 模式

- java juc提供了 CopyOnWriteArrayList、CopyOnWriteArraySet（底层使用CopyOnWriteArrayList保证去重实现）
- 对数据的一致性要求不是非常的高，读多写少

> transient 关键字作用：
>
> 1）一旦变量被transient修饰，`变量将不再是对象持久化的一部分`，该变量内容在序列化后无法获得访问。
>
> 2）`transient关键字只能修饰变量`，而不能修饰方法和类。注意，本地变量是不能被transient关键字修饰的。变量如果是用户自定义类变量，则该类需要实现Serializable接口。
>
> 3）被transient关键字修饰的变量不再能被序列化，一个静态变量不管是否被transient修饰，均不能被序列化。

```java
public class CopyOnWriteArrayList<E>
    implements List<E>, RandomAccess, Cloneable, java.io.Serializable {
    private static final long serialVersionUID = 8673264195747942595L;

    /**
     * The lock protecting all mutators.  (We have a mild preference
     * for builtin monitors over ReentrantLock when either will do.)
     */
    final transient Object lock = new Object();

    /** The array, accessed only via getArray/setArray. */
    private transient volatile Object[] array;
    
    public boolean add(E e) {
        synchronized (lock) {
            Object[] es = getArray();
            int len = es.length;
            es = Arrays.copyOf(es, len + 1);
            es[len] = e;
            setArray(es);
            return true;
        }
    }
}
```

```java
 //路由信息
public final class Router{
  private final String  ip;
  private final Integer port;
  private final String  iface;
  //构造函数
  public Router(String ip, 
      Integer port, String iface){
    this.ip = ip;
    this.port = port;
    this.iface = iface;
  }
  //重写equals方法
  public boolean equals(Object obj){
    if (obj instanceof Router) {
      Router r = (Router)obj;
      return iface.equals(r.iface) &&
             ip.equals(r.ip) &&
             port.equals(r.port);
    }
    return false;
  }
  public int hashCode() {
    //省略hashCode相关代码
  }
}
//路由表信息
public class RouterTable {
  //Key:接口名
  //Value:路由集合
  ConcurrentHashMap<String, CopyOnWriteArraySet<Router>> 
    rt = new ConcurrentHashMap<>();
  //根据接口名获取路由表
  public Set<Router> get(String iface){
    return rt.get(iface);
  }
  //删除路由
  public void remove(Router router) {
    Set<Router> set=rt.get(router.iface);
    if (set != null) {
      set.remove(router);
    }
  }
  //增加路由
  public void add(Router router) {
    Set<Router> set = rt.computeIfAbsent(
      route.iface, r -> 
        new CopyOnWriteArraySet<>());
    set.add(router);
  }
}
```

#### 3. 线程本地存储模式

- 方法里的局部变量，因为不会和其他线程共享，所以没有并发问题，叫做**线程封闭**
- **ThreadLocal**

每一个 Thread 实例对象中，都会有一个 ThreadLocalMap 实例对象；ThreadLocalMap 是一个 Map 类型，底层数据结构是 Entry 数组；一个 Entry 对象中又包含一个 key 和 一个 value

- key 是 ThreadLocal 实例对象的弱引用
- value 就是通过 ThreadLocal#set() 方法实际存储的值

```java
public T get() {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t);
    if (map != null) {
        ThreadLocalMap.Entry e = map.getEntry(this);
        if (e != null) {
            @SuppressWarnings("unchecked")
            T result = (T)e.value;
            return result;
        }
    }
    return setInitialValue();
}
ThreadLocalMap getMap(Thread t) {
    return t.threadLocals;
}
public void set(T value) {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t);
    if (map != null) {
        map.set(this, value);
    } else {
        createMap(t, value);
    }
}
static class ThreadLocalMap {

    /**
         * The entries in this hash map extend WeakReference, using
         * its main ref field as the key (which is always a
         * ThreadLocal object).  Note that null keys (i.e. entry.get()
         * == null) mean that the key is no longer referenced, so the
         * entry can be expunged from table.  Such entries are referred to
         * as "stale entries" in the code that follows.
         */
    static class Entry extends WeakReference<ThreadLocal<?>> {
        /** The value associated with this ThreadLocal. */
        Object value;

        Entry(ThreadLocal<?> k, Object v) {
            super(k);
            value = v;
        }
    }

    /**
         * The initial capacity -- MUST be a power of two.
         */
    private static final int INITIAL_CAPACITY = 16;

    /**
         * The table, resized as necessary.
         * table.length MUST always be a power of two.
         */
    private Entry[] table;
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221129184004085.png)

- [ThreadLocal 内存泄漏问题](https://blog.51cto.com/haofeiyu/5556056)

>  Thread 持有的 ThreadLocalMap 一直都不会被回收，再加上 ThreadLocalMap 中的 Entry 对 ThreadLocal 是弱引用（WeakReference），所以只要 ThreadLocal 结束了自己的生命周期是可以被回收掉的。但是 Entry 中的 Value 却是被 Entry 强引用的，所以即便 Value 的生命周期结束了，Value 也是无法被回收的，从而导致内存泄露。

```java
ExecutorService es;
ThreadLocal tl;
es.execute(()->{
  //ThreadLocal增加变量
  tl.set(obj);
  try {
    // 省略业务逻辑代码
  }finally {
    //手动清理ThreadLocal 
    tl.remove();
  }
});
```

#### 4. Guarded Suspension 模式

GuardedObject 参与者是一个拥有被防卫的方法（guardedMethod）的类。当线程执行guardedMethod时，只要满足警戒条件，就能继续执行，否则线程会进入wait set区等待。警戒条件是否成立随着GuardedObject的状态而变化。GuardedObject 参与者除了guardedMethod外，可能还有用来更改实例状态的的方法stateChangingMethod。在Java语言中，是使用while语句和wait方法来实现guardedMethod的；使用notify/notifyAll方法实现stateChangingMethod。

```java
//request类表示请求
public class Request {
    private final String name;
    public Request(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    }
    public String toString() {
        return "[ Request " + name + " ]";
    }
}

//客户端线程不断生成请求，插入请求队列
public class ClientThread extends Thread {
    private Random random;
    private RequestQueue requestQueue;
    public ClientThread(RequestQueue requestQueue, String name, long seed) {
        super(name);
        this.requestQueue = requestQueue;
        this.random = new Random(seed);
    }
    public void run() {
        for (int i = 0; i < 10000; i++) {
            Request request = new Request("No." + i);
            System.out.println(Thread.currentThread().getName() + " requests " + request);
            requestQueue.putRequest(request);
            try {
                Thread.sleep(random.nextInt(1000));
            } catch (InterruptedException e) {
            }
        }
    }
}

//客户端线程不断从请求队列中获取请求，然后处理请求
public class ServerThread extends Thread {
    private Random random;
    private RequestQueue requestQueue;
    public ServerThread(RequestQueue requestQueue, String name, long seed) {
        super(name);
        this.requestQueue = requestQueue;
        this.random = new Random(seed);
    }
    public void run() {
        for (int i = 0; i < 10000; i++) {
            Request request = requestQueue.getRequest();
            System.out.println(Thread.currentThread().getName() + " handles  " + request);
            try {
                Thread.sleep(random.nextInt(1000));
            } catch (InterruptedException e) {
            }
        }
    }
}

public class RequestQueue {
    private final LinkedList<Request> queue = new LinkedList<Request>();
    public synchronized Request getRequest() {
        while (queue.size() <= 0) {
            try {                                   
                wait();
            } catch (InterruptedException e) {      
            }                                       
        }                                           
        return (Request)queue.removeFirst();
    }
    public synchronized void putRequest(Request request) {
        queue.addLast(request);
        notifyAll();
    }
}

public class RequestQueue {
    private final LinkedList<Request> queue = new LinkedList<Request>();
    public synchronized Request getRequest() {
        while (queue.size() <= 0) {
            try {                                   
                wait();
            } catch (InterruptedException e) {      
            }                                       
        }                                           
        return (Request)queue.removeFirst();
    }
    public synchronized void putRequest(Request request) {
        queue.addLast(request);
        notifyAll();
    }
}
```

#### 5. ThreadPerMessage & 线程池模式

- 每一个message都会分配一个线程，由这个线程执行工作，使用Thread-Per-Message Pattern时，“委托消息的一端”与“执行消息的一端”回会是不同的线程。

- 简易模式：

```java
public class Host {
    private final Helper helper = new Helper();
    public void request(final int count, final char c) {
        System.out.println("    request(" + count + ", " + c + ") BEGIN");
        new Thread() {
            public void run() {
                helper.handle(count, c);
            }
        }.start();
        System.out.println("    request(" + count + ", " + c + ") END");
    }
}

public class Helper {
    public void handle(int count, char c) {
        System.out.println("        handle(" + count + ", " + c + ") BEGIN");
        for (int i = 0; i < count; i++) {
            slowly();
            System.out.print(c);
        }
        System.out.println("");
        System.out.println("        handle(" + count + ", " + c + ") END");
    }
    private void slowly() {
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
        }
    }
}
```

- 使用Executor线程池以及回调方式，借鉴与[chat 项目代码](https://github1s.com/liudongdong1/Chat/blob/HEAD/chat-server/src/main/java/cn/sinjinsong/chat/server/ChatServer.java#L43-L46)

```java
// 每一个task所包含的内容
public class Task {
    private SocketChannel receiver;
    private TaskType type;
    private String desc;
    private Message message;
}
```

```java
// 每一个消息的处理handle， 放入BlockingQueue<Task>阻塞队列中
public class TaskMessageHandler extends MessageHandler {

    @Override
    public void handle(Message message, Selector server, SelectionKey client, BlockingQueue<Task> queue, AtomicInteger onlineUsers) throws InterruptedException {
        TaskDescription taskDescription = ProtoStuffUtil.deserialize(message.getBody(), TaskDescription.class);
        Task task = new Task((SocketChannel) client.channel(), taskDescription.getType(), taskDescription.getDesc(), message);
        try {
            queue.put(task);
            log.info("{}已放入阻塞队列",task.getReceiver().getRemoteAddress());
        }catch (IOException e) {
            e.printStackTrace();
        }
    }
}

this.downloadTaskQueue = new ArrayBlockingQueue<>(20);
this.taskManagerThread = new TaskManagerThread(downloadTaskQueue);
try {
    messageHandler.handle(message, selector, key, downloadTaskQueue, onlineUsers);
} catch (InterruptedException e) {
    log.error("服务器线程被中断");
    exceptionHandler.handle(client, message);
    e.printStackTrace();
}
```

```java
public class TaskManagerThread extends Thread {
    private ExecutorService taskPool;
    private BlockingQueue<Task> taskBlockingQueue;
    private HttpConnectionManager httpConnectionManager;

    private ExecutorService crawlerPool;


    public TaskManagerThread(BlockingQueue<Task> taskBlockingQueue) {
        this.taskPool = new ThreadPoolExecutor(
                5, 10, 1000,
                TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10),
                new ExceptionHandlingThreadFactory(SpringContextUtil.getBean("taskExceptionHandler")),
                new ThreadPoolExecutor.CallerRunsPolicy());
        this.taskBlockingQueue = taskBlockingQueue;
        this.httpConnectionManager = SpringContextUtil.getBean("httpConnectionManager");
        this.crawlerPool = new ThreadPoolExecutor(
                5, 10, 1000,
                TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }

    public void shutdown() {
        taskPool.shutdown();
        crawlerPool.shutdown();
        Thread.currentThread().interrupt();
    }

    /**
     * 如果当前线程被中断，那么Future会抛出InterruptedException，
     * 此时可以通过future.cancel(true)来中断当前线程
     * <p>
     * 由submit方法提交的任务中如果抛出了异常，那么会在ExecutionException中重新抛出
     */
    @Override
    public void run() {
        Task task;
        try {
            while (!Thread.currentThread().isInterrupted()) {
                task = taskBlockingQueue.take();
                log.info("{}已从阻塞队列中取出",task.getReceiver().getRemoteAddress());
                BaseTaskHandler taskHandler = SpringContextUtil.getBean("BaseTaskHandler", task.getType().toString().toLowerCase());
                taskHandler.init(task,httpConnectionManager,this);
                System.out.println(taskHandler);
                taskPool.execute(taskHandler);
            }
        } catch (InterruptedException e) {
            //这里也无法得知发来消息的是谁，所以只能直接退出了
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public ExecutorService getCrawlerPool() {
        return crawlerPool;
    }
}
```

#### 6. Balking 模式

- Balking Pattern和Guarded Suspension Pattern 一样需要警戒条件。在Balking Pattern中，当警戒条件不成立时，会马上中断,停止退出，而Guarded Suspension Pattern 则是等待到可以执行时再去执行。

```java
public class Data {
    private String filename;     // 文件名
    private String content;      // 数据内容
    private boolean changed;     // 标识数据是否已修改
    public Data(String filename, String content) {
        this.filename = filename;
        this.content = content;
        this.changed = true;
    }
    // 修改数据
    public synchronized void change(String newContent) {
        content = newContent;
        changed = true;
    }
    // 若数据有修改，则保存，否则直接返回
    public synchronized void save() throws IOException {
        if (!changed) {
            System.out.println(Thread.currentThread().getName() + " balks");
            return;
        }
        doSave();
        changed = false;
    }
    private void doSave() throws IOException {
        System.out.println(Thread.currentThread().getName() + " calls doSave, content = " + content);
        Writer writer = new FileWriter(filename);
        writer.write(content);
        writer.close();
    }
}
```

```java
//修改线程模仿“一边修改文章，一边保存”
public class ChangerThread extends Thread {
    private Data data;
    private Random random = new Random();
    public ChangerThread(String name, Data data) {
        super(name);
        this.data = data;
    }
    public void run() {
        try {
            for (int i = 0; true; i++) {
                data.change("No." + i);
                Thread.sleep(random.nextInt(1000));
                data.save();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

```java
//存储线程每个1s，会对数据进行一次保存，就像文本处理软件的“自动保存”一样。
public class SaverThread extends Thread {
    private Data data;
    public SaverThread(String name, Data data) {
        super(name);
        this.data = data;
    }
    public void run() {
        try {
            while (true) {
                data.save(); // 存储资料
                Thread.sleep(1000); // 休息约1秒
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        Data data = new Data("data.txt", "(empty)");
        new ChangerThread("ChangerThread", data).start();
        new SaverThread("SaverThread", data).start();
    }
}
```

#### 7. 俩阶段模式

- 将线程的正常处理状态称为“作业中”，当希望结束这个线程时，则送出“终止请求”。接着，这个线程并不会立刻结束，而是进入“终止处理中”状态，此时线程还是运行着的，可能处理一些释放资源等操作。直到终止处理完毕，才会真正结束。

```java
public class CountupThread extends Thread {
    private long counter = 0;
    private volatile boolean shutdownRequested = false;
    public void shutdownRequest() {
        shutdownRequested = true;
        interrupt();
    }
    public boolean isShutdownRequested() {
        return shutdownRequested;
    }
    public final void run() {
        try {
            while (!shutdownRequested) {
                doWork();
            }
        } catch (InterruptedException e) {
        } finally {
            doShutdown();
        }
    }
    private void doWork() throws InterruptedException {
        counter++;
        System.out.println("doWork: counter = " + counter);
        Thread.sleep(500);
    }
    private void doShutdown() {
        System.out.println("doShutdown: counter = " + counter);
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        System.out.println("main: BEGIN");
        try {
            CountupThread t = new CountupThread();
            t.start();
            Thread.sleep(10000);
            System.out.println("main: shutdownRequest");
            t.shutdownRequest();
            System.out.println("main: join");
            t.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("main: END");
    }
}
```

#### 8. 生成消费模式

```java
public class Table {
    private final String[] buffer;
    private int tail;
    private int head;
    private int count;
 
    public Table(int count) {
        this.buffer = new String[count];
        this.head = 0;
        this.tail = 0;
        this.count = 0;
    }
    public synchronized void put(String cake) throws InterruptedException {
        System.out.println(Thread.currentThread().getName() + " puts " + cake);
        while (count >= buffer.length) {
            wait();
        }
        buffer[tail] = cake;
        tail = (tail + 1) % buffer.length;
        count++;
        notifyAll();
    }
    public synchronized String take() throws InterruptedException {
        while (count <= 0) {
            wait();
        }
        String cake = buffer[head];
        head = (head + 1) % buffer.length;
        count--;
        notifyAll();
        System.out.println(Thread.currentThread().getName() + " takes " + cake);
        return cake;
    }
}
```

```java
public class EaterThread extends Thread {
    private final Random random;
    private final Table table;
    public EaterThread(String name, Table table, long seed) {
        super(name);
        this.table = table;
        this.random = new Random(seed);
    }
    public void run() {
        try {
            while (true) {
                String cake = table.take();
                Thread.sleep(random.nextInt(1000));
            }
        } catch (InterruptedException e) {
        }
    }
}
```

```java
public class MakerThread extends Thread {
    private final Random random;
    private final Table table;
    private static int id = 0;     //蛋糕的流水号(所有厨师共通)
    public MakerThread(String name, Table table, long seed) {
        super(name);
        this.table = table;
        this.random = new Random(seed);
    }
    public void run() {
        try {
            while (true) {
                Thread.sleep(random.nextInt(1000));
                String cake = "[ Cake No." + nextId() + " by " + getName() + " ]";
                table.put(cake);
            }
        } catch (InterruptedException e) {
        }
    }
    private static synchronized int nextId() {
        return id++;
    }
}
```

#### 9. Read-Write Lock 模式

```java
public class Data {
    private final char[] buffer;
    private final ReadWriteLock lock = new ReadWriteLock();
    public Data(int size) {
        this.buffer = new char[size];
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = '*';
        }
    }
    public char[] read() throws InterruptedException {
        lock.readLock();
        try {
            return doRead();
        } finally {
            lock.readUnlock();
        }
    }
    public void write(char c) throws InterruptedException {
        lock.writeLock();
        try {
            doWrite(c);
        } finally {
            lock.writeUnlock();
        }
    }
    private char[] doRead() {
        char[] newbuf = new char[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            newbuf[i] = buffer[i];
        }
        slowly();
        return newbuf;
    }
    private void doWrite(char c) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = c;
            slowly();
        }
    }
    private void slowly() {
        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
        }
    }
}
```

```java
public final class ReadWriteLock {
    private int readingReaders = 0;        //正在读取线程的数量 
    private int writingWriters = 0;     //正在写入线程的数量
    public synchronized void readLock() throws InterruptedException {
        while (writingWriters > 0 ) {
            wait();
        }
        readingReaders++;                      
    }
    public synchronized void readUnlock() {
        readingReaders--;   
        notifyAll();
    }
    public synchronized void writeLock() throws InterruptedException {
        while (readingReaders > 0 || writingWriters > 0) {
            wait();
        }
        writingWriters++;                       
    }
    public synchronized void writeUnlock() {
        writingWriters--;     
        notifyAll();
    }
}
```

#### 10. Future-Callable 模式

- 调用一个线程异步执行任务，没有办法获取到返回值，使用future 方法

```java
public class ThreadDemo {
    public static void main(String[] args) {

        //创建线程池
        ExecutorService es = Executors.newSingleThreadExecutor();
        //创建Callable对象任务
        Callable calTask=new Callable() {
            public String call() throws Exception {
                String str = "返回值";
                return str;
            }
        };
        //创建FutureTask
        FutureTask<Integer> futureTask=new FutureTask(calTask);
        //执行任务
        es.submit(futureTask);
        //关闭线程池
        es.shutdown();
        try {
            Thread.sleep(2000);
            System.out.println("主线程在执行其他任务");

            if(futureTask.get()!=null){
                //输出获取到的结果
                System.out.println("futureTask.get()-->"+futureTask.get());
            }else{
                //输出获取到的结果
                System.out.println("futureTask.get()未获取到结果");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("主线程在执行完成");
    }
}
```

### 6. 原子类

#### 1. AtomicInteger

```java
private static final jdk.internal.misc.Unsafe U = jdk.internal.misc.Unsafe.getUnsafe();
private volatile int value;

public final int getAndSet(int newValue) {
    return U.getAndSetInt(this, VALUE, newValue);
}
public final boolean compareAndSet(int expectedValue, int newValue) {
    return U.compareAndSetInt(this, VALUE, expectedValue, newValue);
}
public final int getAndAdd(int delta) {
    return U.getAndAddInt(this, VALUE, delta);
}
public final int addAndGet(int delta) {
    return U.getAndAddInt(this, VALUE, delta) + delta;
}

public final void lazySet(int newValue) {
    U.putIntRelease(this, VALUE, newValue);  //通过该方法对共享变量值的改变，不一定能被其他线程立即看到。也就是说以普通变量的操作方式来写变量。
}

```

> 通过volatile修饰的变量，可以保证在多处理器环境下的“可见性”。也就是说当一个线程修改一个共享变量时，其它线程能立即读到这个修改的值。volatile的实现最终是加了内存屏障：
>
> 1. 保证写volatile变量会强制把CPU写缓存区的数据刷新到内存
> 2. 读volatile变量时，使缓存失效，强制从内存中读取最新的值
> 3. 由于内存屏障的存在，volatile变量还能阻止重排序

**lazySet: 在不需要让共享变量的修改立刻让其他线程可见的时候，以设置普通变量的方式来修改共享状态，可以减少不必要的内存屏障，从而提高程序执行的效率。**

#### 2. AtomicReference

- AtomicInteger是对整数的封装，而AtomicReference则对应普通的对象引用，是操控多个属性的原子性的并发类。
- https://zhuanlan.zhihu.com/p/345700118
- 出现冲突案例

```java
public class BankCardTest {

    private static volatile BankCard bankCard = new BankCard("cxuan",100);

    public static void main(String[] args) {

        for(int i = 0;i < 10;i++){
            new Thread(() -> {
                // 先读取全局的引用
                final BankCard card = bankCard;    
                // 构造一个新的账户，存入一定数量的钱
                BankCard newCard = new BankCard(card.getAccountName(),card.getMoney() + 100);
                System.out.println(newCard);
                // 最后把新的账户的引用赋给原账户
                bankCard = newCard;
                try {
                    TimeUnit.MICROSECONDS.sleep(1000);
                }catch (Exception e){
                    e.printStackTrace();
                }
            }).start();
        }
    }
}
```

```java
// 使用 synchronized 避免冲突
public class BankCardSyncTest {

    private static volatile BankCard bankCard = new BankCard("cxuan",100);

    public static void main(String[] args) {
        for(int i = 0;i < 10;i++){
            new Thread(() -> {
                synchronized (BankCardSyncTest.class) {
                    // 先读取全局的引用
                    final BankCard card = bankCard;
                    // 构造一个新的账户，存入一定数量的钱
                    BankCard newCard = new BankCard(card.getAccountName(), card.getMoney() + 100);
                    System.out.println(newCard);
                    // 最后把新的账户的引用赋给原账户
                    bankCard = newCard;
                    try {
                        TimeUnit.MICROSECONDS.sleep(1000);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }).start();
        }
    }
}
```

```java
// 使用 AtomicReference
public class BankCardARTest {

    private static AtomicReference<BankCard> bankCardRef = new AtomicReference<>(new BankCard("cxuan",100));

    public static void main(String[] args) {

        for(int i = 0;i < 10;i++){
            new Thread(() -> {
                while (true){
                    // 使用 AtomicReference.get 获取
                    final BankCard card = bankCardRef.get();
                    BankCard newCard = new BankCard(card.getAccountName(), card.getMoney() + 100);
                    // 使用 CAS 乐观锁进行非阻塞更新
                    if(bankCardRef.compareAndSet(card,newCard)){
                        System.out.println(newCard);
                    }
                    try {
                        TimeUnit.SECONDS.sleep(1);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }).start();
        }
    }
}
```

一般来讲这并不是什么问题，比如数值运算，线程其实根本不关心变量中途如何变化，只要最终的状态和预期值一样即可。但是，有些操作会依赖于对象的变化过程，此时的解决思路一般就是使用版本号。在变量前面追加上版本号，每次变量更新的时候把版本号加一，那么A－B－A 就会变成1A - 2B - 3A。

#### 3. AtomicStampedReference

```java
AtomicStampedReference<Foo>  asr = new AtomicStampedReference<>(null,0);  // 创建AtomicStampedReference对象，持有Foo对象的引用，初始为null，版本为0

int[] stamp=new  int[1];
Foo  oldRef = asr.get(stamp);   // 调用get方法获取引用对象和对应的版本号
int oldStamp=stamp[0];          // stamp[0]保存版本号

asr.compareAndSet(oldRef, null, oldStamp, oldStamp + 1)   //尝试以CAS方式更新引用对象，并将版本号+1
```

#### 4. Atomic 数组

- AtomicIntegerArray`、`AtomicLongArray`、`AtomicReferenceArray

```java
AtomicIntegerArray  array = new AtomicIntegerArray(10);
array.getAndIncrement(0);   // 将第0个元素原子地增加1

AtomicInteger[]  array = new AtomicInteger[10];
array[0].getAndIncrement();  // 将第0个元素原子地增加1
```



### Resource

- https://www.cnblogs.com/inspred/p/9385897.html
- https://segmentfault.com/a/1190000015558833
- https://github1s.com/liudongdong1/Chat/blob/HEAD/chat-server/src/main/java/cn/sinjinsong/chat/server/ChatServer.java#L43-L46
- http://www.51gjie.com/java/721.html#


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/concurrency/  

