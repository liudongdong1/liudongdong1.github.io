# EffectiveJava_Parrallel


#### .1. 同步访问共享的可变数据

- 使用synchronized方法

```java
package thread;

import java.util.concurrent.TimeUnit;

public class StopThreadSynch {

    private static boolean stopRequested;


    public static synchronized boolean isStopRequested() {
        return stopRequested;
    }


    public static synchronized void setStopRequested(boolean stopRequested) {
        StopThreadSynch.stopRequested = stopRequested;
    }


    public static void main(String[] args) throws InterruptedException {
        long startDate = System.currentTimeMillis();
        Thread thread = new Thread(new Runnable() {

            @Override
            public void run() {
                int i = 0;
                while(!isStopRequested()) {
                    i++;
                    System.out.println(i);
                }

            }
        });
        thread.start();

        TimeUnit.SECONDS.sleep(1);

        setStopRequested(true);
        long endDate = System.currentTimeMillis();
        System.out.println(endDate - startDate);
        System.out.println("-----------");
    }
}
```

- 使用volatile 关键字

```java
// Cooperative thread termination with a volatile field
public class StopThread {
    private static volatile Boolean stopRequested;
    public static void main(String[] args)
        throws InterruptedException {
        Thread backgroundThread = new Thread(() -> {
            int i = 0;
            while (!stopRequested)
                i++;
        });
        backgroundThread.start();
        TimeUnit.SECONDS.sleep(1);
        stopRequested = true;
    }
}

//问题： volatile并不能保证原子性， 所有注意这个问题
private static volatile int nextSerialNumber = 0; 
public static int generateSerialNumber() {
    return nextSerialNumber++;
}
```

- 使用atomic 类

```java
private static final AtomicInteger nextNumber = new AtomicInteger();

public static int generateNumber() {
    return nextNumber.getAndIncrement();
}
```

#### .2. 避免过度同步

>  `StringBuffer` 实例几乎总是被用于单个线程之中，而它们执行的却是内部同步。为此， `StringBuffer` 基本上都由 `StringBuilder` 代替，它是一个非同步的 `StringBuffer` 。同样地，`java.util.Random` 中线程安全的伪随机数生成器，被 `java.util.concurrent.ThreadLocalRandom` 中非同步的实现取代，主要也是出于上述原因。当你不确定的时候，就不要同步类，而应该建立文档，注明它不是线程安全的。

- `这个代码有问题`

```java
// Broken - invokes alien method from synchronized block!
public class ObservableSet<E> extends ForwardingSet<E> {
    public ObservableSet(Set<E> set) { super(set); }
    private final List<SetObserver<E>> observers= new ArrayList<>();
    public void addObserver(SetObserver<E> observer) {
        synchronized(observers) {
            observers.add(observer);
        }
    }
    public Boolean removeObserver(SetObserver<E> observer) {
        synchronized(observers) {
            return observers.remove(observer);
        }
    }
    private void notifyElementAdded(E element) {
        synchronized(observers) {
            for (SetObserver<E> observer : observers)
            observer.added(this, element);
        }
    }
    @Override 
    public Boolean add(E element) {
        Boolean added = super.add(element);
        if (added)
        notifyElementAdded(element);
        return added;
    }
    @Override 
    public Boolean addAll(Collection<? extends E> c) {
        Boolean result = false;
        for (E element : c)
        result |= add(element);
        // Calls notifyElementAdded
        return result;
    }
}
```

#### .3. Executor/Task/Stream优于线程

#### .4. 并发工具优于wait和notify

> `java.util.concurrent` 中更高级的工具分成三类： Executor Framework 、并发集合（Concurrent Collection）以及同步器（Synchronizer）。
>
> 并发集合为标准的集合接口（如 `List` 、`Queue` 和 `Map` ）提供了高性能的并发实现。为了提供高并发性 。 **并发集合中不可能排除并发活动；将它锁定没有什么作用，只会使程序的速度变慢。**

#### .5. 不要依赖线程调度器

> **任何依赖线程调度器来保证正确性或性能的程序都可能是不可移植的**
>
> 不要依赖 `Thread.yield` 或线程优先级。这些工具只是对调度器的提示。线程优先级可以少量地用于提高已经工作的程序的服务质量，但绝不应该用于「修复」几乎不能工作的程序。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210603212220087.png)

```java
public class CountDownLatchDemo {  
    final static SimpleDateFormat sdf=new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");  
    public static void main(String[] args) throws InterruptedException {  
        CountDownLatch latch=new CountDownLatch(2);//两个工人的协作  
        Worker worker1=new Worker("zhang san", 5000, latch);  
        Worker worker2=new Worker("li si", 8000, latch);  
        worker1.start();//  
        worker2.start();//  
        latch.await();//等待所有工人完成工作  
        System.out.println("all work done at "+sdf.format(new Date()));  
    }  
      
      
    static class Worker extends Thread{  
        String workerName;   
        int workTime;  
        CountDownLatch latch;  
        public Worker(String workerName ,int workTime ,CountDownLatch latch){  
             this.workerName=workerName;  
             this.workTime=workTime;  
             this.latch=latch;  
        }  
        public void run(){  
            System.out.println("Worker "+workerName+" do work begin at "+sdf.format(new Date()));  
            doWork();//工作了  
            System.out.println("Worker "+workerName+" do work complete at "+sdf.format(new Date()));  
            latch.countDown();//工人完成工作，计数器减一  
  
        }  
          
        private void doWork(){  
            try {  
                Thread.sleep(workTime);  
            } catch (InterruptedException e) {  
                e.printStackTrace();  
            }  
        }  
    }  
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/effectivejava_parrallel/  

