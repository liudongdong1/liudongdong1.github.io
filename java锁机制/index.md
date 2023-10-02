# synchronized


- 修饰静态方法。锁的是当前类的class对象。
- 修饰普通方法。锁的是当前对象。(this)
- 修饰代码块。则锁的是`指定的对象`。

### 1. synchronized

- 是一种**互斥锁**, 一次只能允许一个线程进入被锁住的代码块;
- Java中**每个对象**都有一个**内置锁(监视器,也可以理解成锁标记)**，而synchronized就是使用**对象的内置锁(监视器)**来将代码块(方法)锁定的
- synchronized保证了线程的**原子性**。(被保护的代码块是一次被执行的，没有任何线程会同时访问)
- synchronized还保证了**可见性**。(当执行完synchronized之后，修改后的变量对其他的线程是可见的)

#### .1. 修饰普通方法

> `每个类实例对应一把锁`，`每个 synchronized 方法都必须获得调用该方法的类实例的锁`方能执行，否则所属线程阻塞，方法一旦执行，就独占该锁，直到从该方法返回时才将锁释放，此后被阻塞的线程方能获得该锁，重新进入可执行状态。这种机制确保了`同一时刻对于每一个类实例`，其`所有声明为 synchronized 的成员函数中至多只有一个处于可执行状态`，从而有效避免了类成员变量的访问冲突。
>
> - 同一时刻，同一实例的多个synchronized方法最多只能有一个被访问。

```java
public class Java3y {
    // 修饰普通方法，此时用的锁是Java3y对象(内置锁)
    public synchronized void test() {
        // doSomething
    }
}
```

#### .2. 修饰代码块或方法（对象锁）

> 调用此对象的同步方法或进入其同步区域时，就必须先获得对象锁。如果此对象的对象锁已被其他调用者占用，则需要等待此锁被释放。（方法锁也是对象锁） 　
>
> `java的所有对象都含有1个互斥锁，这个锁由JVM自动获取和释放`。线程进入synchronized方法的时候获取该对象的锁，当然如果已经有线程获取了这个对象的锁，那么当前线程会等待；synchronized方法正常返回或者抛异常而终止，`JVM会自动释放对象锁`。这里也体现了用synchronized来加锁的1个好处，**方法抛异常的时候，锁仍然可以由JVM来自动释放。**　
>
> - this 指的是当前对象实例本身，所以，所有使用 `synchronized(this) `方式的方法都共享同一把锁。　 　　
> - **只有使用同一实例的线程才会受锁的影响，多个实例调用同一方法也不会受影响。**

```java
public class Test
{
    // 对象锁：形式1(方法锁)
    public synchronized void Method1()
    {
        System.out.println("我是对象锁也是方法锁");
        try
        {
            Thread.sleep(500);
        } catch (InterruptedException e)
        {
            e.printStackTrace();
        }
    }
    // 对象锁：形式2（代码块形式）
    public void Method2()
    {
        synchronized (this)
        {
            System.out.println("我是对象锁");
            try
            {
                Thread.sleep(500);
            } catch (InterruptedException e)
            {
                e.printStackTrace();
            }
        }
    }
 ｝
```

- 随便使用一个对象作为锁不建议使用,**客户端锁**

```java
public class Java3y {
    // 使用object作为锁(任何对象都有对应的锁标记，object也不例外)
    private Object object = new Object();
    public void test() {
        // 修饰代码块，此时用的锁是自己创建的锁Object
        synchronized (object){

            // doSomething
        }
    }
}
```

#### .3. 修饰静态方法

> - 一个class不论被实例化多少次，其中的静态方法和静态变量在内存中都**只有一份**
>
> - 一个静态的方法被申明为synchronized。此类所有的实例化对象在调用此方法，共用同一把锁，我们称之为类锁。
> - `类的不同实例之间共享该类的Class对象`

```java
public class Test
{
　　 // 类锁：形式1
    public static synchronized void Method1()
    {
        System.out.println(＂我是类锁一号＂);
        try
        {
            Thread.sleep(500);
        } catch (InterruptedException e)
        {
            e.printStackTrace();
        }
    }
    // 类锁：形式2
    public void Method２()
    {
        synchronized (Test.class)
        {
            System.out.println(＂我是类锁二号＂);
            try
            {
                Thread.sleep(500);
            } catch (InterruptedException e)
            {
                e.printStackTrace();
            }

        }
    }
｝
```

#### .4. 类锁和对象锁不冲突

```java
public class SynchoronizedDemo {

    //synchronized修饰非静态方法
    public synchronized void function() throws InterruptedException {
        for (int i = 0; i <3; i++) {
            Thread.sleep(1000);
            System.out.println("function running...");
        }
    }
    //synchronized修饰静态方法
    public static synchronized void staticFunction()
            throws InterruptedException {
        for (int i = 0; i < 3; i++) {
            Thread.sleep(1000);
            System.out.println("Static function running...");
        }
    }

    public static void main(String[] args) {
        final SynchoronizedDemo demo = new SynchoronizedDemo();

        // 创建线程执行静态方法
        Thread t1 = new Thread(() -> {
            try {
                staticFunction();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        // 创建线程执行实例方法
        Thread t2 = new Thread(() -> {
            try {
                demo.function();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        // 启动
        t1.start();
        t2.start();
    }
}
```

### 2. Lock

> 在多线程环境中，多个线程可能会同时访问同一个资源，为了避免访问发生冲突，可以根据访问的复杂程度采取不同的措施，原子操作适用于简单的单个操作，无锁算法适用于相对简单的一连串操作，而线程锁适用于复杂的一连串操作

#### 2.1. Lock用法

##### .1. 锁对象

- https://segmentfault.com/a/1190000015562196

> main方法中，创建了一个对象testlock对象，线程1执行该对象的DoWorkWithLock方法，因为死锁（5s后释放），造成lock(this)无法释放，则导致了方法MotherCallYouDinner，DoWorkWithLock在线程2中无法被调用，直到lock(this)释放，lock(testlock)才能继续执行，可以这么理解，由于锁定的同一个对象，线程1释放了锁定的对象，其它线程才能访问。

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LockTest
{
    class Program
    {
        static void Main(string[] args)
        {
            TestLock testlock = new TestLock();
            Thread th = new Thread(() =>
            {
                //模拟死锁：造成死锁，使lock无法释放，在i=5时，跳出死循环，释放lock
                testlock.DoWorkWithLock();
            });
            th.Start();
            Thread.Sleep(1000);
            Thread th2 = new Thread(() =>
            {
                //这个地方你可能会有疑惑，但存在这种情况，比如你封装的dll，对其它开发人员不是可见的
                //开发人员很有可能在他的逻辑中，加上一个lock保证方法同时被一个线程调用，但这时有其它的线程正在调用该方法，
                //但并没有释放，死锁了，那么在这里就不会被执行，除非上面的线程释放了lock锁定的对象。这里的lock也可以理解为一个标识，线程1被锁定的对象
                //是否已经被释放，
                //如果没有释放，则无法继续访问lock块中的代码。
                lock (testlock)
                {
                    // 如果该对象中lock(this)不释放（testlock与this指的是同一个对象），则其它线程如果调用该方法，则会出现直到lock(this)释放后才能继续调用。
                    testlock.MotherCallYouDinner();
                    testlock.DoWorkWithLock();
                }
            });
            th2.Start();
            Console.Read();
        }
    }

    class TestLock
    {
        public static readonly object objLock = new object();
        /// <summary>
        ///  该方法，希望某人在工作的时候，其它人不要打扰（希望只有一个线程在执行)
        /// </summary>
        /// <param name="methodIndex"></param>
        public void DoWorkWithLock()
        {
            //锁当前对象
            lock (this)
            {
                Console.WriteLine("lock this");
                int i = 0;
                while (true)
                {
                    Console.WriteLine("At work, do not disturb...,Thread id is " + Thread.CurrentThread.ManagedThreadId.ToString());
                    Thread.Sleep(1000);
                    if (i == 5)
                    {
                        break;
                    }
                    Console.WriteLine(i.ToString());
                    i++;
                }
            }
            Console.WriteLine("lock dispose");
        }
        public void MotherCallYouDinner()
        {
            Console.WriteLine("Your mother call you to home for dinner.");
        }
    }
}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308105950668.png)

##### .2. 锁静态私有变量

> 将lock(this)更换为锁定私有的静态对象，线程2执行了，首先输出了“Your mother call you to home for dinner.”，同时实现了DoWorkWithLock方法中lock的代码块当前只被一个线程执行，直到lcok（objlock）被释放。因为锁定的对象，外部不能访问，线程2不再关心lock（this）是不是已经释放，都会执行，当然也保证了方法DoWorkWithLock同时被一个线程访问。

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LockTest
{
    class Program
    {
        static void Main(string[] args)
        {
            TestLock testlock = new TestLock();
            Thread th = new Thread(() =>
            {
                //模拟死锁：造成死锁，使lock无法释放，在i=5时，跳出死循环，释放lock
                testlock.DoWorkWithLock();
            });
            th.Start();
            Thread.Sleep(1000);
            Thread th2 = new Thread(() =>
            {

                lock (testlock)
                {
                    testlock.MotherCallYouDinner();
                    testlock.DoWorkWithLock();
                }
            });
            th2.Start();
            Console.Read();
        }
    }

    class TestLock
    {
        private static readonly object objLock = new object();
        /// <summary>
        ///  该方法，希望某人在工作的时候，其它人不要打扰（希望只有一个线程在执行)
        /// </summary>
        /// <param name="methodIndex"></param>
        public void DoWorkWithLock()
        {
            //锁
            lock (objLock)
            {
                Console.WriteLine("lock this");
                int i = 0;
                while (true)
                {
                    Console.WriteLine("At work, do not disturb...,Thread id is " + Thread.CurrentThread.ManagedThreadId.ToString());
                    Thread.Sleep(1000);
                    if (i == 5)
                    {
                        break;
                    }
                    Console.WriteLine(i.ToString());
                    i++;
                }
            }
            Console.WriteLine("lock dispose");
        }
        public void MotherCallYouDinner()
        {
            Console.WriteLine("Your mother call you to home for dinner.");
        }
    }
}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210308110059148.png)

##### .3.总结

1、避免使用lock(this)，因为无法保证你提供的方法，在外部类中使用的时候，开发人员会不会锁定当前对象。

> 通常，应避免锁定 **public** 类型，否则实例将超出代码的控制范围。常见的结构 `lock (this)`、`lock (typeof (MyType))` 和 `lock ("myLock")` 违反此准则：
>
> - 如果实例可以被公共访问，将出现 `lock (this)` 问题。
> - 如果 `MyType` 可以被公共访问，将出现 `lock (typeof (MyType))` 问题。
> - 由于进程中使用同一字符串的任何其他代码将共享同一个锁，所以出现 `lock(“myLock”)` 问题。
>
> 最佳做法是定义 **private** 对象来锁定, 或 **private static** 对象变量来保护所有实例所共有的数据。

#### 2.2. 锁类型

##### .1. 自旋锁 （CAS）

> 当线程等待加锁时，不会阻塞，不会进入等待状态，而是保持运行状态。大致的思路是：让当前线程不停地的在循环体内执行，当循环的条件被其他线程改变时才能进入临界区。
>
> 一种实现方式是通过[CAS](https://so.csdn.net/so/search?q=CAS&spm=1001.2101.3001.7020)原子操作：设置一个CAS原子共享变量，为该变量设置一个初始化的值；加锁时获取该变量的值和初始化值比较，若相等则加锁成功，让后把该值设置成另外一个值；若不相等，则进入循环（自旋过程），不停的比较该值，直到和初始化值相加锁成功。

```c#
// 不可重入方式  若一个已经加锁成功的线程再次获取该锁时，会失败。
public class SpinLock {

  // 定义一个原子引用变量
  private AtomicReference<Thread> sign = new AtomicReference<>();

  public void lock(){
    Thread current = Thread.currentThread();
    // 加锁时：若sign为null，则设置为current；若sihn不为空，则进入循环，自旋等待；
    while(!sign.compareAndSet(null, current)){
      // 自旋：Do Nothing！！ 
    }
  }

  public void unlock (){
    Thread current = Thread.currentThread();
    // 解锁时：sign的值一定为current，所以直接把sign设置为null。
    // 这样其他线程就可以拿到锁了（跳出循环）。
    sign.compareAndSet(current, null);
  }
}
// 可重入锁： 同一个线程多次加锁可重入，解锁只需要调用一次。若多次调用解锁函数，只有第一次解锁成功，后续的解锁操作无效。
public class ReentrantSpinLock {
    private AtomicReference<Thread> sign = new AtomicReference<Thread>();
  
    public void lock() {
        Thread current = Thread.currentThread();
      	// 若尝试加锁的线程和已加的锁中的线程相同，加锁成功
        if (current == sign.get()) {
            return;
        }
        //If the lock is not acquired, it can be spun through CAS
        while (!sign.compareAndSet(null, current)) {
            // DO nothing
        }
    }
  
    public void unlock() {
        Thread cur = Thread.currentThread();
      	// 锁的线程和目前的线程相等时，才允许释放锁
        if (cur == sign.get()) {
                sign.compareAndSet(cur, null);
            }
        }
    }
}
```

```java
class Worker implements  Runnable {
    private ReentrantSpinLock slock = new ReentrantSpinLock();

    public void run() {
        slock.lock(); 
        slock.lock(); // 按以上实现，这一句什么也不做
        for (int i = 0; i < 10; i ++) {
            System.out.printf("%s,", Thread.currentThread().getName());
        }
        System.out.println("");
        slock.unlock();
        slock.unlock(); // 按以上实现，若解锁成功，这一句什么也不做
    }
}

public class SpinLockTest {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        Runnable worker = new Worker();

        for (int i = 0; i < 2; i++) {
            executor.submit(worker);
        }

        executor.shutdown();
        while (!executor.isTerminated()){   }
    }
}
```

##### .2. [互斥锁Monitor&Mutex](https://docs.microsoft.com/zh-cn/dotnet/api/system.threading.monitor.tryenter?view=net-5.0)

> [TryEnter(Object, TimeSpan, Boolean)](https://docs.microsoft.com/zh-cn/dotnet/api/system.threading.monitor.tryenter?view=net-5.0#System_Threading_Monitor_TryEnter_System_Object_System_TimeSpan_System_Boolean__)  Attempts, for the specified amount of time, to acquire an exclusive lock on the specified object, and atomically sets a value that indicates whether the lock was taken.
>
> - boolean: The result of the attempt to acquire the lock, passed by reference. The input must be `false`. The output is `true` if the lock is acquired; otherwise, the output is `false`. The output is set even if an exception occurs during the attempt to acquire the lock.

> 定义：private static readonly object Lock = new object();
>
> 使用：Monitor.Enter(Lock);  //todo Monitor.Exit(Lock);
>
> 作用：将会锁住代码块的内容，并阻止其他线程进入该代码块，直到该代码块运行完成，释放该锁。
>
> 注意：定义的锁对象应该是` 私有的，静态的，只读的，引用类型的对象`，这样可以防止外部改变锁对象
>
> Monitor有TryEnter的功能，可以防止出现死锁的问题，lock没有;
>
> 
>
> 定义：private static readonly Mutex mutex = new Mutex();
>
> 使用：mutex.WaitOne(); //todo mutex.ReleaseMutex();
>
> 作用：将会锁住代码块的内容，并阻止其他线程进入该代码块，直到该代码块运行完成，释放该锁。
>
> 注意：定义的锁对象应该是 私有的，静态的，只读的，引用类型的对象，这样可以防止外部改变锁对象

- Monitor使用

```c#
using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
namespace Sample5_3_monitor_lock_timeout
{
    class Program
    {
        private static int _TaskNum = 3;
        private static Task[] _Tasks;
        private static StringBuilder _StrBlder;
        private const int RUN_LOOP = 50;

        private static void Work1(int TaskID)
        {
            int i = 0;
            string log = "";
            bool lockToken = false;
            while (i < RUN_LOOP)
            {
                log = String.Format("Time: {0}  Task : #{1}  Value: {2}  =====\n",
                                  DateTime.Now.TimeOfDay, TaskID, i);
                i++;
                try
                {
                    lockToken = false;
                    Monitor.TryEnter(_StrBlder, 2000, ref lockToken);
                    if (!lockToken)
                    {
                        Console.WriteLine("Work1 TIMEOUT!! Will throw Exception");
                        throw new TimeoutException("Work1 TIMEOUT!!");
                    }
                    System.Threading.Thread.Sleep(5000);
                    _StrBlder.Append(log);
                }
                finally
                {
                    if (lockToken)
                        Monitor.Exit(_StrBlder);
                }
            }
        }

        private static void Work2(int TaskID)
        {
            int i = 0;
            string log = "";
            bool lockToken = false;

            while (i < RUN_LOOP)
            {
                log = String.Format("Time: {0}  Task : #{1}  Value: {2}  *****\n",
                                  DateTime.Now.TimeOfDay, TaskID, i);
                i++;
                try
                {
                    lockToken = false;
                    Monitor.TryEnter(_StrBlder, 2000, ref lockToken);
                    if (!lockToken)
                    {
                        Console.WriteLine("Work2 TIMEOUT!! Will throw Exception");
                        throw new TimeoutException("Work2 TIMEOUT!!");
                    }

                    _StrBlder.Append(log);
                }
                finally
                {
                    if (lockToken)
                        Monitor.Exit(_StrBlder);
                }
            }
        }

        private static void Work3(int TaskID)
        {
            int i = 0;
            string log = "";
            bool lockToken = false;

            while (i < RUN_LOOP)
            {
                log = String.Format("Time: {0}  Task : #{1}  Value: {2}  ~~~~~\n",
                                  DateTime.Now.TimeOfDay, TaskID, i);
                i++;
                try
                {
                    lockToken = false;
                    Monitor.TryEnter(_StrBlder, 2000, ref lockToken);
                    if (!lockToken)
                    {
                        Console.WriteLine("Work3 TIMEOUT!! Will throw Exception");
                        throw new TimeoutException("Work3 TIMEOUT!!");
                    }
                    _StrBlder.Append(log);
                }
                finally
                {
                    if (lockToken)
                        Monitor.Exit(_StrBlder);
                }
            }
        }

        static void Main(string[] args)
        {
            _Tasks = new Task[_TaskNum];
            _StrBlder = new StringBuilder();

            _Tasks[0] = Task.Factory.StartNew((num) =>
            {
                var taskid = (int)num;
                Work1(taskid);
            }, 0);

            _Tasks[1] = Task.Factory.StartNew((num) =>
            {
                var taskid = (int)num;
                Work2(taskid);
            }, 1);

            _Tasks[2] = Task.Factory.StartNew((num) =>
            {
                var taskid = (int)num;
                Work3(taskid);
            }, 2);

            var finalTask = Task.Factory.ContinueWhenAll(_Tasks, (tasks) =>
            {
                Task.WaitAll(_Tasks);
                Console.WriteLine("==========================================================");
                Console.WriteLine("All Phase is completed");
                Console.WriteLine("==========================================================");
                Console.WriteLine(_StrBlder);
            });

            try
            {
                finalTask.Wait();
            }
            catch (AggregateException aex)
            {
                Console.WriteLine("Task failed And Canceled" + aex.ToString());
            }
            finally
            {
            }
            Console.ReadLine();
        }
    }
}
```

##### .3. 混合锁

> 混合锁的特征是在获取锁失败后像自旋锁一样重试一定的次数，超过一定次数之后（.NET Core 2.1 是30次）再安排当前进程进入等待状态
>
> 混合锁的好处是，如果第一次获取锁失败，但其他线程马上释放了锁，当前线程在下一轮重试可以获取成功，不需要执行毫秒级的线程调度处理；而如果其他线程在短时间内没有释放锁，线程会在超过重试次数之后进入等待状态，以避免消耗 CPU 资源，因此混合锁适用于大部分场景

```c#
internal sealed class SimpleHybridLock : IDisposable
    {
        //基元用户模式构造使用
        private int m_waiters = 0;

        //基元内核模式构造
        private AutoResetEvent m_waiterLock = new AutoResetEvent(false);

        public void Enter()
        {
            //指出该线程想要获得锁
            if (Equals(Interlocked.Increment(ref m_waiters), 1))
            {
                //无竞争，直接返回
                return;
            }

            //另一个线程拥有锁（发生竞争），使这个线程等待
            //线程会阻塞，但不会在CPU上“自旋”，从而节省CPU
            //这里产生较大的性能影响（用户模式与内核模式之间转换）
            //待WaitOne返回后，这个线程拿到锁
            m_waiterLock.WaitOne();
        }

        public void Leave()
        {
            //该线程准备释放锁
            if (Equals(Interlocked.Decrement(ref m_waiters), 0))
            {
                //无线程等待，直接返回
                return;
            }

            //有线程等待则唤醒其中一个
            //这里产生较大的性能影响（用户模式与内核模式之间转换）
            m_waiterLock.Set();
        }

        public void Dispose()
        {
            m_waiterLock.Dispose();
        }
    }
```

##### .4. 读写锁(ReaderWriterLock )

> 支持单个写线程和多个读线程的锁。该锁的作用主要是解决并发读的性能问题，使用该锁，可以大大提高数据并发访问的性能，只有在写时，才会阻塞所有的读锁。

```c#
using System.Collections.Generic;
using System.Windows;
using System.Threading;


namespace FYSTest
{
    public partial class MainWindow : Window
    {
        List<int> list = new List<int>();
        private ReaderWriterLock _rwlock = new ReaderWriterLock();

        public MainWindow()
        {
            InitializeComponent();
            Thread ThRead = new Thread(new ThreadStart(Read));
            ThRead.IsBackground = true;
            Thread ThRead2 = new Thread(new ThreadStart(Read));
            ThRead2.IsBackground = true;
            Thread ThWrite = new Thread(new ThreadStart(Write));
            ThWrite.IsBackground = true;
            ThRead.Start();
            ThRead2.Start();
            ThWrite.Start();
        }

        private void Read()
        {
            while (true)
            {
                //使用一个 System.Int32 超时值获取读线程锁。
                _rwlock.AcquireReaderLock(100);
                try
                {
                    if (list.Count > 0)
                    {
                        int result = list[list.Count - 1];
                    }
                }
                finally
                {
                    //减少锁计数,释放锁
                    _rwlock.ReleaseReaderLock();
                }
            }
        }

        int WriteCount = 0;//写次数
        private void Write()
        {
            while (true)
            {
                //使用一个 System.Int32 超时值获取写线程锁。
                _rwlock.AcquireWriterLock(100);
                try
                {
                    list.Add(WriteCount++);
                }
                finally
                {
                    //减少写线程锁上的锁计数，释放写锁
                    _rwlock.ReleaseWriterLock();
                }
            }
        }
    }
}
```

### 3. SemaphoreSlim

```c#
using System;
using System.Threading;

namespace CallnernawbawceKairwemwhejeene
{
    class Program
    {
        static void Main(string[] args)
        {
            var semaphoreSlim = new SemaphoreSlim(10, 10);

            for (int i = 0; i < 1000; i++)
            {
                var n = i;
                _autoResetEvent.WaitOne();
                new Thread(() => { GeregelkunoNeawhikarcee(semaphoreSlim, n); }).Start();
            }

            Console.Read();
        }

        private static readonly AutoResetEvent _autoResetEvent = new AutoResetEvent(true);

        private static void GeregelkunoNeawhikarcee(SemaphoreSlim semaphoreSlim, int n)
        {
            Console.WriteLine($"{n} 进入");
            _autoResetEvent.Set();

            semaphoreSlim.Wait();
            Console.WriteLine(n);

            Thread.Sleep(TimeSpan.FromSeconds(1));
            semaphoreSlim.Release();
        }
    }
}
```

### 4. 学习资源

- https://docs.microsoft.com/zh-cn/dotnet/api/system.threading.monitor.tryenter?view=net-5.0

- https://www.cnblogs.com/zhao987/p/12551815.html

- https://juejin.cn/post/6844903598011187208

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/java%E9%94%81%E6%9C%BA%E5%88%B6/  

