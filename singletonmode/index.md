# SingletonMode


> 单例模式也就是保证一个类只有一个实例的一种实现方法罢了(设计模式其实就是帮助我们解决实际开发过程中的方法, 该方法是为了降低对象之间的耦合度,`确保一个类只有一个实例,并提供一个全局访问点`。
>
> - 有`频繁实例化然后销毁`的情况，也就是频繁的 new 对象，可以考虑单例模式；
> - `创建对象时耗时过多或者耗资源过多`，但又`经常用到的对象`；socket或者传感器对象
> - `频繁访问 IO 资源的对象`，例如数据库连接池或访问本地文件；
> - 有状态的工具类对象------> 无状态工具类
> - 网站的计数器，一般也是采用单例模式实现，否则难以同步。 
> - 应用程序的日志应用，一般都何用单例模式实现，这一般是由于共享的日志文件一直处于打开状态，因为只能有一个实例去操作，否则内容不好追加。 
> - 一些方法的封装里面都是用静态函数，但类不是静态类。

> 单例模式在`多线程的 应用场合下必须小心使用`。如果当唯一实例尚未创建时，有两个线程同时调用创建方法，那么它们同时没有检测到唯一实例的存在，从而同时各自创建了一个实例， 这样就有两个实例被构造出来，从而违反了单例模式中实例唯一的原则。 解决这个问题的办法是`为指示类是否已经实例化的变量提供一个互斥锁`.
>
>   1.`不适用于变化的对象`，如果同一类型的对象总是要在不同的用例场景发生变化，单例就会引起数据的错误，不能保存彼此的状态。 
>   2.由于单利模式中没有抽象层，因此`单例类的扩展`有很大的困难。 
>   3.单例类的职责过重，在一定程度上违背了“单一职责原则”。 
>   4.滥用单例将带来一些负面问题，如`为了节省资源将数据库连接池对象设计为的单例类，可能会导致共享连接池对象的程序过多而出现连接池溢出`；`如果实例化的对象长时间不被利用`，`系统会认为是垃圾而被回收`，这将导致对象状态的丢失。 

#### 1. 枚举模式

```java
//借助JDK1.5中添加的枚举来实现单例模式。不仅能避免多线程同步问题，而且还能防止反序列化重新创建新的对象
public class StudentSingleton {

    private enum Singleton{
        INSTANCE;

        private final StudentSingleton instance;

        Singleton(){
            instance = new StudentSingleton();
        }

        public StudentSingleton getInstance(){
            return instance;
        }
    }
    
    public static StudentSingleton getInstance(){
        return Singleton.INSTANCE.getInstance();
    }
}
```

#### 2. 饿汉模式

```java
public class UserSingleton{
    private UserSingleton(){}
    private static UserSingleton instance = new UserSingleton();
    public static UserSingleton getInstance(){
        return instance;
    }
}
```

#### 2. 内部类(优)

```java
/**
 * <p>
 * 这种方式跟饿汉式方式采用的机制类似，但又有不同。
 * 两者都是采用了类装载的机制来保证初始化实例时只有一个线程。
 * 不同的地方:
 * 在饿汉式方式是只要Singleton类被装载就会实例化,
 * 内部类是在需要实例化时，调用getInstance方法，才会装载SingletonHolder类
 * <p>
 * 优点：避免了线程不安全，延迟加载，效率高。
 */

public class SingletonIn {

    private SingletonIn() {
    }

    private static class SingletonInHodler {
        private static SingletonIn singletonIn = new SingletonIn();
    }

    public static SingletonIn getSingletonIn() {
        return SingletonInHodler.singletonIn;
    }
}
```

- 在类加载的时候，内部类是不会进行加载的。
- 由于获取到的是StudentSingletonHolder 的静态成员变量，也只会初始化一次。所以又满足单例的要求。

#### 3. 双重校验锁

```java
public class SingletonLanHan {
/**
     * 6.单例模式懒汉式双重校验锁[推荐用]
     * 懒汉式变种,属于懒汉式的最好写法,保证了:延迟加载和线程安全
     */
    private SingletonLanHan(){}
    private static volatile SingletonLanHan singletonLanHanFour;  //使用volatile关键字会强制将修改的值立即写入到主存。且禁止指令重排序

    public static SingletonLanHan getSingletonLanHanFour() {
        if (singletonLanHanFour == null) {
            synchronized (SingletonLanHan.class) {
                if (singletonLanHanFour == null) {
                    singletonLanHanFour = new SingletonLanHan();
                }
            }
        }
        return singletonLanHanFour;
    }
}
```

假设线程A执行到`sInstance = new Singleton()语句，这里看起来是一句代码，但实际上它并不是一个原子操作`，这句代码最终会被编译成多条汇编指令，它大致做了3件事情:
(1）给Singleton的实例分配内存;
(2）调用Singleton()的构造函数，初始化成员字段;
(3）将sInstance对象指向分配的内存空间（此时sInstance就不是null 了)。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221126095057580.png)

#### 4. 容器管理单例模式

```java
public class SingletonManager{
    private static Map<String, Object> objMap=new HashMap<String, Object>();
    private SingletonManager(){}
    public static void registerService(String key, Object instance){
        if(!objMap.containsKey(key)){
            objMap.put(key,instance);
        }
    }
    public static Object getService(String key){
        
    }
}
```



#### 4. 实现片段

##### 4.1. c#

```c#
//--------代码片段一：  同时创建多个线程去创建这个对象实例的时候, 会被多次创建
public class Singleton
{
    private Sington() { }
    private static Singleton _Singleton = null;
    public static Singleton CreateInstance()
    {
        if (_Singleton == null)
        {
            Console.WriteLine("被创建");
            _Singleton = new Singleton();
        }
        return _Singleton;
    }
}
//--------代码片段二：  同步锁为了达到预期的效果, 也是损耗了性能的
public class Singleton
{
    private Sington() { }
    private static Singleton _Singleton = null;
    private static object Singleton_Lock = new object(); //锁同步
    public static Singleton CreateInstance()
    {
        lock (Singleton_Lock)
        {
            Console.WriteLine("路过");
            if (_Singleton == null)
            {
                Console.WriteLine("被创建");
                _Singleton = new Singleton();
            }
        }
        return _Singleton;
    }
}
//测试代码
TaskFactory taskFactory = new TaskFactory();
List<Task> taskList = new List<Task>();

for (int i = 0; i < 5; i++)
{
    taskList.Add(taskFactory.StartNew(() =>
                                      {
                                          Singleton singleton = Singleton.CreateInstance(); 
                                      }));
} 
//--------优化代码片段三：
public class Singleton
{
    private static Singleton _Singleton = null;
    private static object Singleton_Lock = new object();
    public static Singleton CreateInstance()
    {
        if (_Singleton == null) //双if +lock
        {
            lock (Singleton_Lock)
            {
                Console.WriteLine("路过。");
                if (_Singleton == null)
                {
                    Console.WriteLine("被创建。");
                    _Singleton = new Singleton();
                }
            }
        }
        return _Singleton;
    }
}
//---------代码片段四： 使用静态变量
//由CLR保证，在程序第一次使用该类之前被调用，而且只调用一次
//PS: 但是他的缺点也很明显, 在程序初始化后, 静态对象就被CLR构造了, 哪怕你没用
public sealed class Singleton
{
    private Singleton() { }

    private static readonly Singleton singleInstance = new Singleton();

    public static Singleton GetInstance
    {
        get
        {
            return singleInstance;
        }
    }
}
//-------------代码片段四： 使用静态函数
//静态构造函数：只能有一个，无参数的，程序无法调用 。
public class SingletonSecond
{
    private static SingletonSecond _SingletonSecond = null;
    static SingletonSecond()
    {
        _SingletonSecond = new SingletonSecond();
    }
    public static SingletonSecond CreateInstance()
    {
        return _SingletonSecond;
    }
}
//------------代码片段五： 延迟加载 
//通常用于将对象的初始化延迟到需要时。因此，延迟加载的主要目标是按需加载对象，或者您可以根据需要说出对象。
public sealed class Singleton
{
    private Singleton(){}
    private static readonly Lazy<Singleton> Instancelock =
        new Lazy<Singleton>(() => new Singleton());
    public static Singleton GetInstance
    {
        get
        {
            return Instancelock.Value;
        }
    }
}
```

#### 5 单例懒加载模式

```java
// 使用静态内部类完成单例模式封装，避免线程安全问题，避免重复初始化成员属性  
@Slf4j  
public class FilterIpUtil {  
  
    private FilterIpUtil() {  
    }  
  
    private List<String> strings = new ArrayList<>();  
  
    // 代码块在FilterIpUtil实例初始化时才会执行  
    {  
    
        // 在代码块中完成文件的第一次读写操作，后续不再读这个文件
        System.out.println("FilterIpUtil init");  
        try (InputStream resourceAsStream = FilterIpUtil.class.getClassLoader().getResourceAsStream("filterIp.txt")) {  
            // 将文件内容放到string集合中  
            IoUtil.readUtf8Lines(resourceAsStream, strings);  
        } catch (IOException e) {  
            log.error(e.getMessage(), e);  
        }  
    }  
  
    public static FilterIpUtil getInstance() {  
        return InnerClassInstance.instance;  
    }  
    // 使用内部类完成单例模式，由jvm保证线程安全  
    private static class InnerClassInstance {  
        private static final FilterIpUtil instance = new FilterIpUtil();  
    }  
  
    // 判断集合中是否包含目标参数  
    public boolean isFilter(String arg) {  
        return strings.contains(arg);  
    }  
  
}
```



---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/singletonmode/  

