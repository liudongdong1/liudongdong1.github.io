# EentLoopGroup


> 学习链接： https://juejin.cn/post/6999225608341291039#heading-6
>
> 还是得通过实战，自己来读一下代码，了解背后的实现机制。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009215336308.png)

### 1. Unsafe

#### .1. Java JDK

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009194524091.png)

#### .2.netty

> Unsafe接口中定义了s**ocket相关操作，包括SocketAddress获取、selector注册、网卡端口绑定、socket建连与断连、socket写数据。这些操作都和jdk底层socket相关**。
>
> Unsafe用于处理Channel对应网络IO的底层操作。ChannelHandler处理回调事件时产生的相关网络IO操作最终也会委托给Unsafe执行。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009195257533.png)

### 2. 初始化EventLoopGroup

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211009195011148.png)

> 在服务器启动的常规代码里，首先是实例化NioEventLoopGroup和ServerBootstrap。

执行NioEventLoopGroup时，由NioEventLoopGroup开始，一路调用，到达MultithreadEventLoopGroup，如果没有指定创建的线程数量，则默认创建的线程个数为DEFAULT_EVENT_LOOP_THREADS，该数值为：处理器数量x2。

```java
protected MultithreadEventLoopGroup(int nThreads, ThreadFactory threadFactory, Object... args) {
    super(nThreads == 0 ? DEFAULT_EVENT_LOOP_THREADS : nThreads, threadFactory, args);
}
```

> 每个NioEventLoop的执行器为`ThreadPerTaskExecutor`，ThreadPerTaskExecutor实现了Executor接口，并会`在execute方法中启动真正的线程`，但是要和NioEventLoop的线程挂钩则在SingleThreadEventExecutor的doStartThread方法里。

```java
public final class ThreadPerTaskExecutor implements Executor {
    private final ThreadFactory threadFactory;

    public ThreadPerTaskExecutor(ThreadFactory threadFactory) {
        if (threadFactory == null) {
            throw new NullPointerException("threadFactory");
        }
        this.threadFactory = threadFactory;
    }

    @Override
    public void execute(Runnable command) {
        //使用真正的线程执行方法
        threadFactory.newThread(command).start();
    }
}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/7c99415ff57243f094c2095f4032fbactplv-k3u1fbpfcp-watermark.awebp)

> EventLoop本质上是一个线程池，EventLoop内部维护着一个线程Thread和几个阻塞队列，所以EventLoop可以看成只有一个线程的线程池（SingleThreadPool）

### 3. NioEventLoop运行

> 调用selectStrategy.calculateStrategy 判断是否有 Task任务，`如果没有则调用 selectSupplier.get() 方法`，该方法是非阻塞的，判断是否有需要处理的 Channel。如果没有则返回 SelectStrategy.SELECT，然后执行 select(wakenUp.getAndSet(false)) 方法，阻塞等待可处理的 IO 就绪事件。
>
> 如果有 Task 任务，则判断 ioRatio 的比率值，该值为 EventLoop 处理 IO 和 处理 Task 任务的时间的比率。默认比率为 50%。

```java
@Override
protected void run() {
    for (;;) {
        try {
            try {
                // 1、通过 hasTasks() 判断当前消息队列中是否还有未处理的消息
                switch (selectStrategy.calculateStrategy(selectNowSupplier, hasTasks())) {
                case SelectStrategy.CONTINUE:
                    continue;

                //hasTasks() 没有任务则执行 select() 处理网络IO
                case SelectStrategy.SELECT:
                //轮询事件，见第三小节
                    select(wakenUp.getAndSet(false));

                    if (wakenUp.get()) {
                        selector.wakeup();
                    }
                    // fall through
                default:
                }
            } catch (IOException e) {
                // If we receive an IOException here its because the Selector is messed up. Let's rebuild
                // the selector and retry. https://github.com/netty/netty/issues/8566
                rebuildSelector0();
                handleLoopException(e);
                continue;
            }

            cancelledKeys = 0;
            needsToSelectAgain = false;
            // 处理IO事件所需的时间和花费在处理 task 时间的比例，默认为 50%
            final int ioRatio = this.ioRatio;
            if (ioRatio == 100) {
                try {
                    // 如果 IO 的比例是100，表示每次都处理完IO事件后，才执行所有的task
                    processSelectedKeys();
                } finally {
                    // 执行 task 任务
                    runAllTasks();
                }
            } else {
                // 记录处理 IO 开始的执行时间
                final long ioStartTime = System.nanoTime();
                try {
                //IO任务处理，见第四小节
                    processSelectedKeys();
                } finally {
                    // 计算处理 IO 所花费的时间
                    final long ioTime = System.nanoTime() - ioStartTime;
                    // 执行 task 任务，判断执行 task 任务时间是否超过配置的比例，如果超过则停止执行 task 任务
                    runAllTasks(ioTime * (100 - ioRatio) / ioRatio);
                }
            }
        } catch (Throwable t) {
            handleLoopException(t);
        }
        // Always handle shutdown even if the loop processing threw an exception.
        try {
            if (isShuttingDown()) {
                closeAll();
                if (confirmShutdown()) {
                    return;
                }
            }
        } catch (Throwable t) {
            handleLoopException(t);
        }
    }
}
```

### 4. Select方法

```java
private void select(boolean oldWakenUp) throws IOException {
    Selector selector = this.selector;
    try {
        int selectCnt = 0;
        long currentTimeNanos = System.nanoTime();
        // 计算出 NioEventLoop 定时任务最近执行的时间（还有多少 ns 执行），单位 ns
        long selectDeadLineNanos = currentTimeNanos + delayNanos(currentTimeNanos);

        for (;;) {
            // 为定时任务中的时间加上0.5毫秒，将时间换算成毫秒
            long timeoutMillis = (selectDeadLineNanos - currentTimeNanos + 500000L) / 1000000L;
            // 对定时任务的超时时间判断，如果到时间或超时，则需要立即执行 selector.selectNow()
            if (timeoutMillis <= 0) {
                if (selectCnt == 0) {
                    selector.selectNow();
                    selectCnt = 1;
                }
                break;
            }
            // 轮询过程中发现有任务加入,中断本次轮询
            if (hasTasks() && wakenUp.compareAndSet(false, true)) {
                selector.selectNow();
                selectCnt = 1;
                break;
            }
            // Nio 的 阻塞式 select 操作
            int selectedKeys = selector.select(timeoutMillis);
            // select 次数 ++ , 通过该次数可以判断是否出发了 JDK Nio中的 Selector 空轮循 bug
            selectCnt ++;

             // 如果selectedKeys不为空、或者被用户唤醒、或者队列中有待处理任务、或者调度器中有任务，则break
            if (selectedKeys != 0 || oldWakenUp || wakenUp.get() || hasTasks() || hasScheduledTasks()) {
                break;
            }
            //如果线程被中断则重置selectedKeys，同时break出本次循环，所以不会陷入一个繁忙的循环。
            if (Thread.interrupted()) {
                selectCnt = 1;
                break;
            }

            long time = System.nanoTime();
            // 如果超时，把 selectCnt 置为 1，开始下一次的循环
            if (time - TimeUnit.MILLISECONDS.toNanos(timeoutMillis) >= currentTimeNanos) {
                // timeoutMillis elapsed without anything selected.
                selectCnt = 1;
            }
            //  如果 selectCnt++ 超过 默认的 512 次，说明触发了 Nio Selector 的空轮训 bug，则需要重新创建一个新的 Selector，并把注册的 Channel 迁移到新的 Selector 上
            else if (SELECTOR_AUTO_REBUILD_THRESHOLD > 0 &&
                    selectCnt >= SELECTOR_AUTO_REBUILD_THRESHOLD) {
                // 重新创建一个新的 Selector，并把注册的 Channel 迁移到新的 Selector 上，
                //解决NIO Selector空轮询bug，见第五小节
                selector = selectRebuildSelector(selectCnt);
                selectCnt = 1;
                break;
            }

            currentTimeNanos = time;
        }

        if (selectCnt > MIN_PREMATURE_SELECTOR_RETURNS) {
            if (logger.isDebugEnabled()) {
                logger.debug("Selector.select() returned prematurely {} times in a row for Selector {}.",
                        selectCnt - 1, selector);
            }
        }
    } catch (CancelledKeyException e) {
        if (logger.isDebugEnabled()) {
            logger.debug(CancelledKeyException.class.getSimpleName() + " raised by a Selector {} - JDK bug?",
                    selector, e);
        }
    }
}
```

1、通过 delayNanos(currentTimeNanos) 计算出 定时任务队列中第一个任务的执行时间。
 2、判断是否到期，如果到期则执行 selector.selectNow()，退出循环
 3、如果定时任务未到执行时间，则通过 hasTasks() 判断是否有可执行的任务，如果有则中断本次循环。
 4、既没有到期的定时任务、也没有可执行的Task，则调用 selector.select(timeoutMillis) 方法阻塞，等待注册到 Selector 上感兴趣的事件。
 5、每次 select() 后都会 selectCnt++。通过该次数可以判断是否出发了 JDK Nio中的 Selector 空轮询 bug
 6、如果selectedKeys不为空、或者被用户唤醒、或者队列中有待处理任务、或者调度器中有任务，则break。
 7、通过 selectCnt 判断是否触发了 JDK Selector 的空轮询 bug，SELECTOR_AUTO_REBUILD_THRESHOLD 默认为 512, 可修改。
 8、通过 selectRebuildSelector() 方法解决 Selector 空轮询 bug。

### 5. processSelectedKeys IO事件处理

```java
private void processSelectedKeys() {
    if (selectedKeys != null) {
        processSelectedKeysOptimized();
    } else {   
        //默认没有使用优化的 Set，所有调用 processSelectedKeysPlain() 方法进行处理 IO 任务
        processSelectedKeysPlain(selector.selectedKeys());
    }
}
```

```java
private void processSelectedKeysPlain(Set<SelectionKey> selectedKeys) {
    // check if the set is empty and if so just return to not create garbage by
    // creating a new Iterator every time even if there is nothing to process.
    // See https://github.com/netty/netty/issues/597
    if (selectedKeys.isEmpty()) {
        return;
    }

    Iterator<SelectionKey> i = selectedKeys.iterator();
    //循环处理每个 selectionKey，每个selectionKey的处理首先根据attachment的类型来进行分发处理；
    for (;;) {
        final SelectionKey k = i.next();
        final Object a = k.attachment();
        i.remove();

        if (a instanceof AbstractNioChannel) {
            processSelectedKey(k, (AbstractNioChannel) a);
        } else {
            @SuppressWarnings("unchecked")
            NioTask<SelectableChannel> task = (NioTask<SelectableChannel>) a;
            processSelectedKey(k, task);
        }

        if (!i.hasNext()) {
            break;
        }

        if (needsToSelectAgain) {
            selectAgain();
            selectedKeys = selector.selectedKeys();

            // Create the iterator again to avoid ConcurrentModificationException
            if (selectedKeys.isEmpty()) {
                break;
            } else {
                i = selectedKeys.iterator();
            }
        }
    }
}
```

```java
private void processSelectedKey(SelectionKey k, AbstractNioChannel ch) {
    //首先获取 Channel 的 NioUnsafe，所有的读写等操作都在 Channel 的 unsafe 类中操作。
    final AbstractNioChannel.NioUnsafe unsafe = ch.unsafe();
    if (!k.isValid()) {
        final EventLoop eventLoop;
        try {
            eventLoop = ch.eventLoop();
        } catch (Throwable ignored) {

            return;
        }

        if (eventLoop != this || eventLoop == null) {
            return;
        }

        unsafe.close(unsafe.voidPromise());
        return;
    }

    try {
        int readyOps = k.readyOps();
        //熟悉的获取 SelectionKey 就绪事件，如果是 OP_CONNECT，则说明已经连接成功，并把注册的 OP_CONNECT 事件取消
        if ((readyOps & SelectionKey.OP_CONNECT) != 0) {

            int ops = k.interestOps();
            ops &= ~SelectionKey.OP_CONNECT;
            k.interestOps(ops);

            unsafe.finishConnect();
        }

        //如果是 OP_WRITE 事件，说明可以继续向 Channel 中写入数据，当写完数据后用户自己吧 OP_WRITE 事件取消掉。
        if ((readyOps & SelectionKey.OP_WRITE) != 0) {

            ch.unsafe().forceFlush();
        }

        //如果是 OP_READ 或 OP_ACCEPT 事件，则调用 unsafe.read() 进行读取数据。unsafe.read() 中会调用到 ChannelPipeline 进行读取数据。
        if ((readyOps & (SelectionKey.OP_READ | SelectionKey.OP_ACCEPT)) != 0 || readyOps == 0) {
            unsafe.read();
        }
    } catch (CancelledKeyException ignored) {
        unsafe.close(unsafe.voidPromise());
    }
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/eventloopgroup/  

