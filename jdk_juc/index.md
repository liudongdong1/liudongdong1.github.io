# JDK_JUC


![Overview](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/java-thread-x-juc-overview-1.png)

![Lock框架和Tools类](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/java-thread-x-juc-overview-lock.png)

![Collection并发集合](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/java-thread-x-juc-overview-2.png)

![线程池](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/java-thread-x-juc-executors-1.png)

### 1. Condition 案例

```java
class BoundedBuffer {
    final Lock lock = new ReentrantLock();
    final Condition notFull = lock.newCondition();
    final Condition notEmpty = lock.newCondition();
 
    final Object[] items = new Object[100];
    int putptr, takeptr, count;
 
    public void put(Object x) throws InterruptedException {
        lock.lock();
        try {
            while (count == items.length)    //防止虚假唤醒，Condition的await调用一般会放在一个循环判断中
                notFull.await();
            items[putptr] = x;
            if (++putptr == items.length)
                putptr = 0;
            ++count;
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }
 
    public Object take() throws InterruptedException {
        lock.lock();
        try {
            while (count == 0)
                notEmpty.await();
            Object x = items[takeptr];
            if (++takeptr == items.length)
                takeptr = 0;
            --count;
            notFull.signal();
            return x;
        } finally {
            lock.unlock();
        }
    }
}
```

### 2. AQS 方法

#### 2.1 CAS操作

CAS，即CompareAndSet，在Java中CAS操作的实现都委托给一个名为UnSafe类，关于`Unsafe`类，以后会专门详细介绍该类，目前只要知道，通过该类可以实现对字段的原子操作。

| 方法名                  | 修饰符               | 描述                    |
| ----------------------- | -------------------- | ----------------------- |
| compareAndSetState      | protected final      | CAS修改同步状态值       |
| compareAndSetHead       | private final        | CAS修改等待队列的头指针 |
| compareAndSetTail       | private final        | CAS修改等待队列的尾指针 |
| compareAndSetWaitStatus | private static final | CAS修改结点的等待状态   |
| compareAndSetNext       | private static final | CAS修改结点的next指针   |

#### 2.2 等待队列的核心操作

| 方法名              | 修饰符  | 描述                 |
| ------------------- | ------- | -------------------- |
| enq                 | private | 入队操作             |
| addWaiter           | private | 入队操作             |
| setHead             | private | 设置头结点           |
| unparkSuccessor     | private | 唤醒后继结点         |
| doReleaseShared     | private | 释放共享结点         |
| setHeadAndPropagate | private | 设置头结点并传播唤醒 |

#### 2.3 资源的获取操作

| 方法名                       | 修饰符         | 描述                              |
| ---------------------------- | -------------- | --------------------------------- |
| cancelAcquire                | private        | 取消获取资源                      |
| shouldParkAfterFailedAcquire | private static | 判断是否阻塞当前调用线程          |
| acquireQueued                | final          | 尝试获取资源,获取失败尝试阻塞线程 |
| doAcquireInterruptibly       | private        | 独占地获取资源（响应中断）        |
| doAcquireNanos               | private        | 独占地获取资源（限时等待）        |
| doAcquireShared              | private        | 共享地获取资源                    |
| doAcquireSharedInterruptibly | private        | 共享地获取资源（响应中断）        |
| doAcquireSharedNanos         | private        | 共享地获取资源（限时等待）        |

| 方法名                     | 修饰符       | 描述                       |
| -------------------------- | ------------ | -------------------------- |
| acquire                    | public final | 独占地获取资源             |
| acquireInterruptibly       | public final | 独占地获取资源（响应中断） |
| acquireInterruptibly       | public final | 独占地获取资源（限时等待） |
| acquireShared              | public final | 共享地获取资源             |
| acquireSharedInterruptibly | public final | 共享地获取资源（响应中断） |
| tryAcquireSharedNanos      | public final | 共享地获取资源（限时等待） |

#### 2.4 资源的释放操作

| 方法名        | 修饰符       | 描述         |
| ------------- | ------------ | ------------ |
| release       | public final | 释放独占资源 |
| releaseShared | public final | 释放共享资源 |

#### 3. AQS 原理

#### 同步状态

```java
/**
 * 同步状态.
 */
private volatile int state;

protected final int getState() {
    return state;
}

protected final void setState(int newState) {
    state = newState;
}
/**
 * 以原子的方式更新同步状态.
 * 利用Unsafe类实现
 */
protected final boolean compareAndSetState(int expect, int update) {
    return unsafe.compareAndSwapInt(this, stateOffset, expect, update);
}
```

#### 阻塞&唤醒

- java.util.concurrent.locks包提供了[LockSupport](https://segmentfault.com/a/1190000015562456)类来作为线程阻塞和唤醒的工具。

#### 等待队列

```java
static final class Node {
    // 共享模式结点
    static final Node SHARED = new Node();
    // 独占模式结点
    static final Node EXCLUSIVE = null;
    static final int CANCELLED =  1;
    static final int SIGNAL    = -1;
    static final int CONDITION = -2;
    static final int PROPAGATE = -3;
    /**
    * INITAL：      0 - 默认，新结点会处于这种状态。
    * CANCELLED：   1 - 取消，表示后续结点被中断或超时，需要移出队列；
    * SIGNAL：      -1- 发信号，表示后续结点被阻塞了；（当前结点在入队后、阻塞前，应确保将其prev结点类型改为SIGNAL，以便prev结点取消或释放时将当前结点唤醒。）
    * CONDITION：   -2- Condition专用，表示当前结点在Condition队列中，因为等待某个条件而被阻塞了；
    * PROPAGATE：   -3- 传播，适用于共享模式。（比如连续的读操作结点可以依次进入临界区，设为PROPAGATE有助于实现这种迭代操作。）
    * 
    * waitStatus表示的是后续结点状态，这是因为AQS中使用CLH队列实现线程的结构管理，而CLH结构正是用前一结点某一属性表示当前结点的状态，这样更容易实现取消和超时功能。
    */
    volatile int waitStatus;
    // 前驱指针
    volatile Node prev;
    // 后驱指针
    volatile Node next;
    // 结点所包装的线程
    volatile Thread thread;
    // Condition队列使用，存储condition队列中的后继节点
    Node nextWaiter;
    Node() {
    }
    Node(Thread thread, Node mode) { 
        this.nextWaiter = mode;
        this.thread = thread;
    }
}
/**
 * 以自旋的方式不断尝试插入结点至队列尾部
 *
 * @return 当前结点的前驱结点
 */
private Node enq(final Node node) {
    for (; ; ) {
        Node t = tail;
        if (t == null) { // 如果队列为空，则创建一个空的head结点
            if (compareAndSetHead(new Node()))
                tail = head;
        } else {
            node.prev = t;
            if (compareAndSetTail(t, node)) {
                t.next = node;
                return t;
            }
        }
    }
}
```

### Resource

- https://www.pdai.tech/md/java/thread/java-thread-x-juc-executor-FutureTask.html
- https://segmentfault.com/a/1190000015804888 todo
- https://juejin.cn/post/6844903796485668872  阻塞队列使用



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/jdk_juc/  

