# ThreadPoolQueue

> 当ThreadPoolExecutor线程池被创建的时候，里边是没有工作线程的，直到有任务进来（执行了execute方法）才开始创建线程去工作，工作原理如下（即execute方法运行原理）：调用线程池的execute方法的时候`如果当前的工作线程数 小于 核心线程数`，则创建新的线程执行任务；`否则将任务加入阻塞队列`。如果`队列满了则根据最大线程数去创建额外（核心线程数以外）的工作线程去执行任务`；如果`工作线程数达到了最大线程数，则根据拒绝策略去执行`。存活时间到期的话只是回收核心线程（maximumPoolSize - corePoolSize）以外的线程.
>


```java

ThreadPoolExecutor(int corePoolSize,// 核心线程数
                   int maximumPoolSize,//最大线程数
                   long keepAliveTime,//空闲线程存活时间
                   TimeUnit unit,//存活时间单位
                   BlockingQueue<Runnable> workQueue,//阻塞队列
                   RejectedExecutionHandler handler)//拒绝策略
```

```java
// 分3个步骤进行:
// 1. 如果运行的线程少于corePoolSize，请尝试使用给定的命令作为第一个线程启动一个新线程的任务。对addWorker的调用会自动检查runState和workerCount，这样可以防止虚假警报的增加当它不应该的时候，返回false。
// 2. 如果任务可以成功排队，那么我们仍然需要来再次检查我们是否应该添加线程(因为自从上次检查后，现有的已经死了)或者那样自进入此方法后池就关闭了。所以我们重新检查状态，并在必要时回滚队列停止，或启动一个新线程(如果没有线程)。
// 3.如果我们不能将任务放入队列，那么我们尝试添加一个新的线程。如果它失败了，我们知道我们被关闭或饱和了所以拒绝这个任务。
public void execute(Runnable command) {
        if (command == null)
            throw new NullPointerException();
        int c = ctl.get();
        // 第一步
        if (workerCountOf(c) < corePoolSize) {
            if (addWorker(command, true))
                return;
            c = ctl.get();
        }
        // 第二步骤
        if (isRunning(c) && workQueue.offer(command)) {
            int recheck = ctl.get();
            if (! isRunning(recheck) && remove(command))
                reject(command);
            else if (workerCountOf(recheck) == 0)
                addWorker(null, false);
        }
        // 第三步
        else if (!addWorker(command, false))
            reject(command);
    }

```

### 1. 五种线程池

```java
ExecutorService threadPool = null;
threadPool = Executors.newCachedThreadPool();
//有缓冲的线程池，线程数 JVM 控制
threadPool = Executors.newFixedThreadPool(3);
//固定大小的线程池
threadPool = Executors.newScheduledThreadPool(2);
threadPool = Executors.newSingleThreadExecutor();
//单线程的线程池，只有一个线程在工作
threadPool = new ThreadPoolExecutor();
//默认线程池，可控制参数比较多
```

### 2. 四种拒绝策略

```java
RejectedExecutionHandler rejected = null;
rejected = new ThreadPoolExecutor.AbortPolicy();
//默认，队列满了丢任务抛出异常
rejected = new ThreadPoolExecutor.DiscardPolicy();
//队列满了丢任务不抛异常
rejected = new ThreadPoolExecutor.DiscardOldestPolicy();
//将最早进入队列的任务删，之后再尝试加入队列
rejected = new ThreadPoolExecutor.CallerRunsPolicy();
//如果添加到线程池失败，那么主线程会自己去执行该任务；如果执行程序已关闭（主线程运行结束），则会丢弃该任务
```

### 3. 三种阻塞队列

```java
BlockingQueue<Runnable> workQueue = null;
workQueue = new ArrayBlockingQueue<>(5);
//基于数组的先进先出队列，有界
workQueue = new LinkedBlockingQueue<>();
//基于链表的先进先出队列，无界
workQueue = new SynchronousQueue<>();
//无缓冲的等待队列，无界
```

todo ThreadFactory 用法

 Collections.synchronizedMap


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/threadpoolqueue/  

