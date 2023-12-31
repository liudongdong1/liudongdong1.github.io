# MediatorMode


> 提供了一个中介类，该类通常`处理不同类之间的通信`，并支持松耦合，使代码易于维护。
>
> - 中介者模式**简化了对象之间的交互**，它用中介者和同事的一对多交互代替了原来同事之间的多对多交互，一对多关系更容易理解、维护和扩展，将原本难以理解的网状结构转换成相对简单的星型结构。
> - 中介者模式可**将各同事对象解耦**。中介者有利于各同事之间的松耦合，我们可以独立的改变和复用每一个同事和中介者，增加新的中介者和新的同事类都比较方便，更好地符合 "开闭原则"。
> - 可以**减少子类生成**，中介者将原本分布于多个对象间的行为集中在一起，改变这些行为只需生成新的中介者子类即可，这使各个同事类可被重用，无须对同事类进行扩展。
> - 例如：`一个主板上多个设备，主板看作是中介者，每个设备看作具体同事类，可以实现多个设备之间的交互。`
> - 例如： android开发中， activity 中的很多空间注册的毁掉方法。

> 中介者模式包含以下主要角色。
>
> 1. `抽象中介者（Mediator）角色`：它是中介者的接口，提供了`同事对象注册与转发同事对象信息的抽象方法。`
> 2. 具体中介者（Concrete Mediator）角色：实现中介者接口，定义一个 `List 来管理同事对象`，协调各个同事角色之间的交互关系，因此它依赖于同事角色。
> 3. 抽象同事类（Colleague）角色：定义同事类的接口，`保存中介者对象，提供同事对象交互的抽象方法，`实现所有相互影响的同事类的公共功能。
> 4. 具体同事类（Concrete Colleague）角色：是抽象同事类的实现者，当需要与其他同事对象交互时，由中介者对象负责后续的交互。

> **中转作用（结构性）**：通过`中介者提供的中转作用`，各个同事对象就`不再需要显式引用其他同事`，当需要`和其他同事进行通信时，可通过中介者来实现间接调用`。该中转作用属于中介者在结构上的支持。
>
> **协调作用（行为性）**：中介者可以更进一步的对同事之间的关系进行封装，同事可以一致的和中介者进行交互，而不需要指明中介者需要具体怎么做，中介者根据封装在自身内部的协调逻辑，对同事的请求进行进一步处理，将同事成员之间的关系行为进行分离和封装。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210306124827527.png)

```java
package net.biancheng.c.mediator;

import java.util.*;

public class MediatorPattern {
    public static void main(String[] args) {
        Mediator md = new ConcreteMediator();
        Colleague c1, c2;
        c1 = new ConcreteColleague1();
        c2 = new ConcreteColleague2();
        md.register(c1);
        md.register(c2);
        c1.send();
        System.out.println("-------------");
        c2.send();
    }
}

//抽象中介者
abstract class Mediator {
    public abstract void register(Colleague colleague);

    public abstract void relay(Colleague cl); //转发
}

//具体中介者
class ConcreteMediator extends Mediator {
    private List<Colleague> colleagues = new ArrayList<Colleague>();

    public void register(Colleague colleague) {
        if (!colleagues.contains(colleague)) {
            colleagues.add(colleague);
            colleague.setMedium(this);
        }
    }

    public void relay(Colleague cl) {
        for (Colleague ob : colleagues) {
            if (!ob.equals(cl)) {
                ((Colleague) ob).receive();
            }
        }
    }
}

//抽象同事类
abstract class Colleague {
    protected Mediator mediator;

    public void setMedium(Mediator mediator) {
        this.mediator = mediator;
    }

    public abstract void receive();

    public abstract void send();
}

//具体同事类
class ConcreteColleague1 extends Colleague {
    public void receive() {
        System.out.println("具体同事类1收到请求。");
    }

    public void send() {
        System.out.println("具体同事类1发出请求。");
        mediator.relay(this); //请中介者转发
    }
}

//具体同事类
class ConcreteColleague2 extends Colleague {
    public void receive() {
        System.out.println("具体同事类2收到请求。");
    }

    public void send() {
        System.out.println("具体同事类2发出请求。");
        mediator.relay(this); //请中介者转发
    }
}
```

### 1. Java Demo

> 有三种数据库 Mysql、Redis、Elasticsearch，其中的 Mysql 作为主数据库，当增加一条数据时**需要**同步到另外两个数据库中；Redis 作为缓存数据库，当增加一条数据时**不需要**同步到另外另个数据库；而 Elasticsearch 作为大数据查询数据库，有一个统计功能，当增加一条数据时**只需要**同步到 Mysql.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210705164722840.png)

```java
public abstract class AbstractDatabase {
    public static final String MYSQL = "mysql";
    public static final String REDIS = "redis";
    public static final String ELASTICSEARCH = "elasticsearch";

    protected AbstractMediator mediator;    // 中介者

    public AbstractDatabase(AbstractMediator mediator) {
        this.mediator = mediator;
    }

    public abstract void addData(String data);

    public abstract void add(String data);
}

```

```java
public class MysqlDatabase extends AbstractDatabase {
    private List<String> dataset = new ArrayList<String>();

    public MysqlDatabase(AbstractMediator mediator) {
        super(mediator);
    }

    @Override
    public void addData(String data) {
        System.out.println("Mysql 添加数据：" + data);
        this.dataset.add(data);
    }

    @Override
    public void add(String data) {
        addData(data);
        this.mediator.sync(AbstractDatabase.MYSQL, data); // 数据同步作业交给中介者管理
    }

    public void select() {
        System.out.println("Mysql 查询，数据：" + this.dataset.toString());
    }
}

```

```java
public class RedisDatabase extends AbstractDatabase {
    private List<String> dataset = new LinkedList<String>();

    public RedisDatabase(AbstractMediator mediator) {
        super(mediator);
    }

    @Override
    public void addData(String data) {
        System.out.println("Redis 添加数据：" + data);
        this.dataset.add(data);
    }

    @Override
    public void add(String data) {
        addData(data);
        this.mediator.sync(AbstractDatabase.REDIS, data);    // 数据同步作业交给中介者管理
    }

    public void cache() {
        System.out.println("Redis 缓存的数据：" + this.dataset.toString());
    }
}
```

```java
public class RedisDatabase extends AbstractDatabase {
    private List<String> dataset = new LinkedList<String>();

    public RedisDatabase(AbstractMediator mediator) {
        super(mediator);
    }

    @Override
    public void addData(String data) {
        System.out.println("Redis 添加数据：" + data);
        this.dataset.add(data);
    }

    @Override
    public void add(String data) {
        addData(data);
        this.mediator.sync(AbstractDatabase.REDIS, data);    // 数据同步作业交给中介者管理
    }

    public void cache() {
        System.out.println("Redis 缓存的数据：" + this.dataset.toString());
    }
}
```

```java
public class EsDatabase extends AbstractDatabase {
    private List<String> dataset = new CopyOnWriteArrayList<String>();

    public EsDatabase(AbstractMediator mediator) {
        super(mediator);
    }

    @Override
    public void addData(String data) {
        System.out.println("ES 添加数据：" + data);
        this.dataset.add(data);
    }

    @Override
    public void add(String data) {
        addData(data);
        this.mediator.sync(AbstractDatabase.ELASTICSEARCH, data);    // 数据同步作业交给中介者管理
    }

    public void count() {
        int count = this.dataset.size();
        System.out.println("Elasticsearch 统计，目前有 " + count + " 条数据，数据：" + this.dataset.toString());
    }
}
```

```java
@Data
public abstract class AbstractMediator {
    protected MysqlDatabase mysqlDatabase;
    protected RedisDatabase redisDatabase;
    protected EsDatabase esDatabase;

    public abstract void sync(String databaseName, String data);
}
```

```java
public class SyncMediator extends AbstractMediator {
    @Override
    public void sync(String databaseName, String data) {
        if (AbstractDatabase.MYSQL.equals(databaseName)) {
            // mysql 同步到 redis 和 Elasticsearch
            this.redisDatabase.addData(data);
            this.esDatabase.addData(data);
        } else if (AbstractDatabase.REDIS.equals(databaseName)) {
            // redis 缓存同步，不需要同步到其他数据库
        } else if (AbstractDatabase.ELASTICSEARCH.equals(databaseName)) {
            // Elasticsearch 同步到 Mysql
            this.mysqlDatabase.addData(data);
        }
    }
}
```

```java
public class Client {
    public static void main(String[] args) {
        AbstractMediator syncMediator = new SyncMediator();
        MysqlDatabase mysqlDatabase = new MysqlDatabase(syncMediator);
        RedisDatabase redisDatabase = new RedisDatabase(syncMediator);
        EsDatabase esDatabase = new EsDatabase(syncMediator);

        syncMediator.setMysqlDatabase(mysqlDatabase);
        syncMediator.setRedisDatabase(redisDatabase);
        syncMediator.setEsDatabase(esDatabase);

        System.out.println("\n---------mysql 添加数据 1，将同步到Redis和ES中-----------");
        mysqlDatabase.add("1");
        mysqlDatabase.select();
        redisDatabase.cache();
        esDatabase.count();

        System.out.println("\n---------Redis添加数据 2，将不同步到其它数据库-----------");
        redisDatabase.add("2");
        mysqlDatabase.select();
        redisDatabase.cache();
        esDatabase.count();

        System.out.println("\n---------ES 添加数据 3，只同步到 Mysql-----------");
        esDatabase.add("3");
        mysqlDatabase.select();
        redisDatabase.cache();
        esDatabase.count();
    }
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210705165059858.png)

### 2. Timer

```java
public class MyOneTask extends TimerTask {
    private static int num = 0;
    @Override
    public void run() {
        System.out.println("I'm MyOneTask " + ++num);
    }
}

public class MyTwoTask extends TimerTask {
    private static int num = 1000;
    @Override
    public void run() {
        System.out.println("I'm MyTwoTask " + num--);
    }
}

public class TimerTest {
    public static void main(String[] args) {
        // 注意：多线程并行处理定时任务时，Timer运行多个TimeTask时，只要其中之一没有捕获抛出的异常，
        // 其它任务便会自动终止运行，使用ScheduledExecutorService则没有这个问题
        Timer timer = new Timer();
        timer.schedule(new MyOneTask(), 3000, 1000); // 3秒后开始运行，循环周期为 1秒
        timer.schedule(new MyTwoTask(), 3000, 1000);
    }
}
```

```java
public class Timer {

    private final TaskQueue queue = new TaskQueue();
    private final TimerThread thread = new TimerThread(queue);
    
    public void schedule(TimerTask task, long delay) {
        if (delay < 0)
            throw new IllegalArgumentException("Negative delay.");
        sched(task, System.currentTimeMillis()+delay, 0);
    }
    
    public void schedule(TimerTask task, Date time) {
        sched(task, time.getTime(), 0);
    }
    
    private void sched(TimerTask task, long time, long period) {
        if (time < 0)
            throw new IllegalArgumentException("Illegal execution time.");

        if (Math.abs(period) > (Long.MAX_VALUE >> 1))
            period >>= 1;

        // 获取任务队列的锁(同一个线程多次获取这个锁并不会被阻塞,不同线程获取时才可能被阻塞)
        synchronized(queue) {
            // 如果定时调度线程已经终止了,则抛出异常结束
            if (!thread.newTasksMayBeScheduled)
                throw new IllegalStateException("Timer already cancelled.");

            // 再获取定时任务对象的锁(为什么还要再加这个锁呢?想不清)
            synchronized(task.lock) {
                // 判断线程的状态,防止多线程同时调度到一个任务时多次被加入任务队列
                if (task.state != TimerTask.VIRGIN)
                    throw new IllegalStateException(
                        "Task already scheduled or cancelled");
                // 初始化定时任务的下次执行时间
                task.nextExecutionTime = time;
                // 重复执行的间隔时间
                task.period = period;
                // 将定时任务的状态由TimerTask.VIRGIN(一个定时任务的初始化状态)设置为TimerTask.SCHEDULED
                task.state = TimerTask.SCHEDULED;
            }
            
            // 将任务加入任务队列
            queue.add(task);
            // 如果当前加入的任务是需要第一个被执行的(也就是他的下一次执行时间离现在最近)
            // 则唤醒等待queue的线程(对应到上面提到的queue.wait())
            if (queue.getMin() == task)
                queue.notify();
        }
    }
    
    // cancel会等到所有定时任务执行完后立刻终止定时线程
    public void cancel() {
        synchronized(queue) {
            thread.newTasksMayBeScheduled = false;
            queue.clear();
            queue.notify();  // In case queue was already empty.
        }
    }
    // ...
}
```

### Resource

- https://juejin.cn/post/6844903699643383821

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mediatormode/  

