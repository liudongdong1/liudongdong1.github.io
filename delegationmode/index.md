# Delegation


> 委派模式(Delegate Pattern ) 又叫委托模式， 是一种面向对象的设计模式， 允许`对象组合实现`与 `继承相同的代码重用`。它的基本作用就是负责任务的调用和分配任务， 是一种特殊的静态代理， 可以理 解为全权代理， 但是代理模式注重过程，而委派模式注重结果。
>
> 优点：通过任务委派能够将—个大型的任务细化，然后通过统—管理这些子任务的完成情况实现任务的跟进，能够加快任务执行的效率。
>
> 缺点：任务委派方式需要根据任务的复杂程度进行不同的改变，在任务比较复杂的情况下可能需要进行多 重委派，容易造成索乱。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704163135767.png)

> 抽象任务角色( Task ) : 定义一个抽象接口， 它有若干实现类。
>
> 委派者角色( Delegate ) : 负责在各个具体角色实例之间做出决策并判断并调用具体实现的方法。
>
> 具体任务角色( Concrete ) 真正执行任务的角色。

#### 2. java

##### 2.1. 双亲委派

> - `全盘负责`：即是当一个classloader加载一个Class的时候，这个Class所依赖的和引用的其它Class`通常`也由这个classloader负责载入。
> - `委托机制`：先让parent（父）类加载器 寻找，只有在parent找不到的时候才从自己的类路径中去寻找。
> - 类加载还采用了`cache机制`：如果cache中保存了这个Class就直接返回它，如果没有才从文件中读取和转换成Class，并存入cache，`这就是为什么修改了Class但是必须重新启动JVM才能生效，并且类只加载一次的原因`。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210302215456394.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210302215901836.png)

```java
public abstract class ClassLoader {
     private final ClassLoader parent;
     
     protected Class<?> loadClass(String name, boolean resolve)
         throws ClassNotFoundException
     {
         synchronized (getClassLoadingLock(name)) {
             // First, check if the class has already been loaded
             Class<?> c = findLoadedClass(name);
             if (c == null) {
                 long t0 = System.nanoTime();
                 try {
                     if (parent != null) {
                         c = parent.loadClass(name, false);
                     } else {
                         c = findBootstrapClassOrNull(name);
                     }
                 } catch (ClassNotFoundException e) {
                     // ClassNotFoundException thrown if class not found
                     // from the non-null parent class loader
                 }
 ​
                 if (c == null) {
                     // If still not found, then invoke findClass in order
                     // to find the class.
                     long t1 = System.nanoTime();
                     c = findClass(name);
 ​
                     // this is the defining class loader; record the stats
                     sun.misc.PerfCounter.getParentDelegationTime().addTime(t1 - t0);
                     sun.misc.PerfCounter.getFindClassTime().addElapsedTimeFrom(t1);
                     sun.misc.PerfCounter.getFindClasses().increment();
                 }
             }
             if (resolve) {
                 resolveClass(c);
             }
             return c;
         }
     }
 }
```

##### .2. **DispatcherServlet**

- MemberController :

```java
 public class MemberController {
     public void getMemberById(String mid) {
     }
 }
```

- OrderController 类 ：

```java
 public class OrderController {
     public void getOrderById(String mid) {
     }
 }
```

- SystemController 类：

```java
 public class SystemController {
     public void logout() {
     }
 }
```

- 创建 DispatcherServlet 类：

```java
 public class DispatcherServlet extends HttpServlet {
 
     private Map<String, Method> handlerMapping = new HashMap<String, Method>();
 
     @Override
     protected void service(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
         doDispatch(req, resp);
     }
 
     private void doDispatch(HttpServletRequest req, HttpServletResponse resp) {
         String url = req.getRequestURI();
         Method method = handlerMapping.get(url);
         // method.invoke();
     }
 
     @Override
     public void init() throws ServletException {
         try {
             handlerMapping.put("/web/getMemeberById.json", MemberController.class.getMethod("getMemberById", String.class));
         } catch (Exception e) {
             e.printStackTrace();
         }
     }
 }
```

- 配置 web.xml 文件 ：

```xml
 <?xml version="1.0" encoding="UTF-8"?>
 <web-app xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xmlns="http://java.sun.com/xml/ns/j2ee"
          xsi:schemaLocation="http://java.sun.com/xml/ns/j2ee http://java.sun.com/xml/ns/j2ee/web-app_2_4.xsd"
          version="2.4">
     <display-name>Gupao Web Application</display-name>
 
     <servlet>
         <servlet-name>delegateServlet</servlet-name>
         <servlet-class>com.gupaoedu.vip.pattern.delegate.mvc.DispatcherServlet</servlet-class>
         <load-on-startup>1</load-on-startup>
     </servlet>
 
     <servlet-mapping>
         <servlet-name>delegateServlet</servlet-name>
         <url-pattern>/*</url-pattern>
     </servlet-mapping>
 </web-app>
```

#### 3. c#

##### 3.1. 委托声明

-  delegate

>  Delegate至少0个参数，至多32个参数，可以`无返回值，也可以指定返回值类型`。
>
>  例：`public delegate int MethodtDelegate(int x, int y);`表示有两个参数，并返回int型。

```c#c#
public delegate int MethodDelegate(int x, int y);
private static MethodDelegate method;
static void Main(string[] args)
{
    method = new MethodDelegate(Add);
    Console.WriteLine(method(10,20));
    Console.ReadKey();
}
private static int Add(int x, int y)
{
    return x + y;
}
```

-  Action

>  Action是无返回值的泛型委托。
>
> Action 表示无参，无返回值的委托
>
> Action<int,string> 表示有传入参数int,string无返回值的委托
>
> Action<int,string,bool> 表示有传入参数int,string,bool无返回值的委托
>
>  Action<int,int,int,int> 表示有传入4个int型参数，无返回值的委托
>
> Action至少0个参数，至多16个参数，无返回值。

```c#
static void Main(string[] args)
{
    Test<string>(Action,"Hello World!");
    Test<int>(Action, 1000);
    Test<string>(p => { Console.WriteLine("{0}", p); }, "Hello World");//使用Lambda表达式定义委托
    Console.ReadKey();
}
public static void Test<T>(Action<T> action, T p)
{
    action(p);
}
private static void Action(string s)
{
    Console.WriteLine(s);
}
private static void Action(int s)
{
    Console.WriteLine(s);
}
```

-  Func

> 　　 Func是有返回值的泛型委托
>
> 　　 Func<int> 表示无参，返回值为int的委托
>
> 　　 Func<object,string,int> 表示传入参数为object, string 返回值为int的委托
>
> 　　 Func<object,string,int> 表示传入参数为object, string 返回值为int的委托
>
> 　　 Func<T1,T2,,T3,int> 表示传入参数为T1,T2,,T3(泛型)返回值为int的委托
>
> 　　 Func至少0个参数，至多16个参数，根据返回值泛型返回。必须有返回值，不可void

```c#
static void Main(string[] args)
{
	Console.WriteLine(Test<int,int>(Fun,100,200));
	Console.ReadKey();
}
public static int Test<T1, T2>(Func<T1, T2, int> func, T1 a, T2 b)
{
	return func(a, b);
}
private static int Fun(int a, int b)
{
	creturn a + b;
}
```

-   predicate

> 　　 predicate 是返回bool型的泛型委托
>
> 　　 predicate<int> 表示传入参数为int 返回bool的委托
>
> 　　 Predicate有且只有一个参数，返回值固定为bool
>
> 　　 例：public delegate bool Predicate<T> (T obj)

```c#
 static void Main(string[] args)
 {
     Point[] points = { new Point(100, 200), 
                       new Point(150, 250), new Point(250, 375), 
                       new Point(275, 395), new Point(295, 450) };
     Point first = Array.Find(points, ProductGT10);
     Console.WriteLine("Found: X = {0}, Y = {1}", first.X, first.Y);
     Console.ReadKey();
 }
private static bool ProductGT10(Point p)
{
    if (p.X * p.Y > 100000)
    {
        return true;
    }
    else
    {
        return false;
    }
}
```

##### 3.2. customer observation

- ObservableCollection 源代码  c#看不了源代码，`以后不建议看c#源代码`

```c#
 public class ObservableCollection<T> : Collection<T>, INotifyCollectionChanged, INotifyPropertyChanged
    {
        //
        // 摘要:
        //     Initializes a new instance of the System.Collections.ObjectModel.ObservableCollection`1
        //     class.
        public ObservableCollection();
        //
        // 摘要:
        //     Initializes a new instance of the System.Collections.ObjectModel.ObservableCollection`1
        //     class that contains elements copied from the specified collection.
        //
        // 参数:
        //   collection:
        //     The collection from which the elements are copied.
        //
        // 异常:
        //   T:System.ArgumentNullException:
        //     The collection parameter cannot be null.
        public ObservableCollection(IEnumerable<T> collection);
        //
        // 摘要:
        //     Initializes a new instance of the System.Collections.ObjectModel.ObservableCollection`1
        //     class that contains elements copied from the specified list.
        //
        // 参数:
        //   list:
        //     The list from which the elements are copied.
        //
        // 异常:
        //   T:System.ArgumentNullException:
        //     The list parameter cannot be null.
        public ObservableCollection(List<T> list);

        //
        // 摘要:
        //     Occurs when an item is added, removed, changed, moved, or the entire list is
        //     refreshed.
        public event NotifyCollectionChangedEventHandler CollectionChanged;
        //
        // 摘要:
        //     Occurs when a property value changes.
        protected event PropertyChangedEventHandler PropertyChanged;

        //
        // 摘要:
        //     Moves the item at the specified index to a new location in the collection.
        //
        // 参数:
        //   oldIndex:
        //     The zero-based index specifying the location of the item to be moved.
        //
        //   newIndex:
        //     The zero-based index specifying the new location of the item.
        public void Move(int oldIndex, int newIndex);
        //
        // 摘要:
        //     Disallows reentrant attempts to change this collection.
        //
        // 返回结果:
        //     An System.IDisposable object that can be used to dispose of the object.
        protected IDisposable BlockReentrancy();
        //
        // 摘要:
        //     Checks for reentrant attempts to change this collection.
        //
        // 异常:
        //   T:System.InvalidOperationException:
        //     If there was a call to System.Collections.ObjectModel.ObservableCollection`1.BlockReentrancy
        //     of which the System.IDisposable return value has not yet been disposed of. Typically,
        //     this means when there are additional attempts to change this collection during
        //     a System.Collections.ObjectModel.ObservableCollection`1.CollectionChanged event.
        //     However, it depends on when derived classes choose to call System.Collections.ObjectModel.ObservableCollection`1.BlockReentrancy.
        protected void CheckReentrancy();
        //
        // 摘要:
        //     Removes all items from the collection.
        protected override void ClearItems();
        //
        // 摘要:
        //     Inserts an item into the collection at the specified index.
        //
        // 参数:
        //   index:
        //     The zero-based index at which item should be inserted.
        //
        //   item:
        //     The object to insert.
        protected override void InsertItem(int index, T item);
        //
        // 摘要:
        //     Moves the item at the specified index to a new location in the collection.
        //
        // 参数:
        //   oldIndex:
        //     The zero-based index specifying the location of the item to be moved.
        //
        //   newIndex:
        //     The zero-based index specifying the new location of the item.
        protected virtual void MoveItem(int oldIndex, int newIndex);
        //
        // 摘要:
        //     Raises the System.Collections.ObjectModel.ObservableCollection`1.CollectionChanged
        //     event with the provided arguments.
        //
        // 参数:
        //   e:
        //     Arguments of the event being raised.
        protected virtual void OnCollectionChanged(NotifyCollectionChangedEventArgs e);
        //
        // 摘要:
        //     Raises the System.Collections.ObjectModel.ObservableCollection`1.PropertyChanged
        //     event with the provided arguments.
        //
        // 参数:
        //   e:
        //     Arguments of the event being raised.
        protected virtual void OnPropertyChanged(PropertyChangedEventArgs e);
        //
        // 摘要:
        //     Removes the item at the specified index of the collection.
        //
        // 参数:
        //   index:
        //     The zero-based index of the element to remove.
        protected override void RemoveItem(int index);
        //
        // 摘要:
        //     Replaces the element at the specified index.
        //
        // 参数:
        //   index:
        //     The zero-based index of the element to replace.
        //
        //   item:
        //     The new value for the element at the specific#ed index.
        protected override void SetItem(int index, T item);
    }
```

- observableCollection使用

```c#
private static void RunObservableCollectionCode()
{
    ObservableCollection<string> names = new ObservableCollection<string>();
    names.CollectionChanged += names_CollectionChanged;
    names.Add("Adam");
    names.Add("Eve");
    names.Remove("Adam");
    names.Add("John");
    names.Add("Peter");
    names.Clear();
}
static void names_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
{
    Debug.WriteLine("Change type: " + e.Action);
    if (e.NewItems != null)
    {
        Debug.WriteLine("Items added: ");
        foreach (var item in e.NewItems)
        {
            Debug.WriteLine(item);
        }
    }
    if (e.OldItems != null)
    {
        Debug.WriteLine("Items removed: ");
        foreach (var item in e.OldItems)
        {
            Debug.WriteLine(item);
        }
    }
}
```

- 自定义ObservableDictionary

```c#
// 包含EventHandler
using System;
// 包含KeyValuePair
using System.Collections.Generic;

namespace VehicleSystem.tools
{
    // TK（Templete Key）,TV（Templete Value）分别是键和值的泛型
    public class ObservableDictionary<TK,TV>
    {
        // 实际上存值的字典
        private readonly Dictionary<TK, TV> _dic = new Dictionary<TK, TV>();

        // 添值前触发的事件
        public event EventHandler<KeyValuePair<TK, TV>> BeforeAdd;

        // 添值后触发的事件
        public event EventHandler<KeyValuePair<TK, TV>> AfterAdd;

        // 清空前触发的事件
        public event EventHandler<EventArgs> BeforeClear;

        // 清空后触发的事件
        public event EventHandler<EventArgs> AfterClear;

        // 获取元素数时触发的事件
        public event EventHandler<EventArgs> OnGetCount;

        // 获取元素键时触发的事件
        public event EventHandler<EventArgs> OnGetKeys;

        // 获取元素值时触发的事件
        public event EventHandler<EventArgs> OnGetValues;

        // 元素数发生变化时触发的事件
        public event EventHandler<EventArgs> CollectionChanged;

        public int Count
        {
            get
            {
                // ?运算符代表可为空类型，为空时则不执行函数
                OnGetCount?.Invoke(this,null);
                return _dic.Count;
            }
        }

        public Dictionary<TK, TV>.KeyCollection Keys
        {
            get
            {
                OnGetKeys?.Invoke(this, null);
                return _dic.Keys;
            }
        }

        public Dictionary<TK,TV>.ValueCollection Values
        {
            get
            {
                OnGetValues?.Invoke(this, null);
                return _dic.Values;
            }
        }

        // 索引器
        public TV this[TK index]
        {
            get
            {
                return _dic[index];
            }
            set
            {
                _dic[index] = value;
                CollectionChanged?.Invoke(this, null);
            }
        }

        public void Add(TK key, TV value)
        {
            BeforeAdd?.Invoke(this, new KeyValuePair<TK, TV>(key, value));
            _dic.Add(key, value);
            AfterAdd?.Invoke(this, new KeyValuePair<TK, TV>(key, value));
            CollectionChanged?.Invoke(this, null);
        }

        public void Clear()
        {
            BeforeClear?.Invoke(this, null);
            _dic.Clear();
            AfterClear?.Invoke(this, null);
            CollectionChanged?.Invoke(this, null);
        }
    }
}
```



---

> 作者: liudongdong  
> URL: liudongdong1.github.io/delegationmode/  

