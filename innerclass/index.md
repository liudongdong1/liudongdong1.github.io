# InnerClass


> 内部类是一个编译时的概念，一旦编译成功，就会成为完全不同的两类。对于一个名为outer的外部类和其内部定义的名为inner的内部类。编译完成后出现outer.class和outer$inner.class两类。

### 1. 类型

#### .1. 成员内部类

- 成员`内部类可以无条件访问外部类的所有成员属性和成员方法`（包括private成员和静态成员）。

> 在定义的内部类的构造器是无参构造器，编译器还是会默认添加一个参数，`该参数的类型为指向外部类对象的一个引用`，所以成员内部类中的Outter this&0 指针便指向了外部类对象，因此可以在成员内部类中随意访问外部类的成员。

```java
class Circle {
    private double radius = 0;
    public static int count =1;
    public Circle(double radius) {
        this.radius = radius;
    }
     
    class Draw {     //内部类
        public void drawSahpe() {
            System.out.println(radius);  //外部类的private成员
            System.out.println(count);   //外部类的静态成员
        }
    }
}
```

> 当成员内部类拥有和外部类同名的成员变量或者方法时，会发生隐藏现象，即默认情况下访问的是成员内部类的成员。如果要访问外部类的同名成员，需要以下面的形式进行访问：
>
> - 外部类.``this``.成员变量
> - 外部类.``this``.成员方法

- 在外部类中如果要访问成员内部类的成员，必须`先创建一个成员内部类的对象，再通过指向这个对象的引用来访问`：

```java
class Circle {
    private double radius = 0;
    public Circle(double radius) {
        this.radius = radius;
        getDrawInstance().drawSahpe();   //必须先创建成员内部类的对象，再进行访问
    }
    private Draw getDrawInstance() {
        return new Draw();
    }  
    class Draw {     //内部类
        public void drawSahpe() {
            System.out.println(radius);  //外部类的private成员
        }
    }
}
```

- 成员内部类是依附外部类而存在的，也就是说，如果要创建成员内部类的对象，`前提是必须存在一个外部类的对象`。

```java
public class Test {
    public static void main(String[] args)  {
        //第一种方式：
        Outter outter = new Outter();
        Outter.Inner inner = outter.new Inner();  //必须通过Outter对象来创建
         
        //第二种方式：
        Outter.Inner inner1 = outter.getInnerInstance();
    }
}
 
class Outter {
    private Inner inner = null;
    public Outter() {
         
    }
     
    public Inner getInnerInstance() {
        if(inner == null)
            inner = new Inner();
        return inner;
    }
      
    class Inner {
        public Inner() {
             
        }
    }
}
```

#### .2. 局部内部类

> 定义在一个方法或者一个作用域里面的类，它和成员内部类的区别在于局部内部类的访问仅限于方法内或者该作用域内。`局部内部类就像是方法里面的一个局部变量一样，是不能有public、protected、private以及static修饰符的`。

```java
People{
    public People() {
         
    }
}
 
class Man{
    public Man(){
         
    }
     
    public People getWoman(){
        class Woman extends People{   //局部内部类
            int age =0;
        }
        return new Woman();
    }
}
```

#### .3. 内部类

> 匿名内部类也是不能有访问修饰符和static修饰符的，匿名内部类是唯一一种没有构造器的类。正因为其没有构造器，所以匿名内部类的使用范围非常有限，大部分匿名内部类用于接口回调。

```java
scan_bt.setOnClickListener(new OnClickListener() {  
    @Override
    public void onClick(View v) {
        // TODO Auto-generated method stub

    }
});     
history_bt.setOnClickListener(new OnClickListener() {
    @Override
    public void onClick(View v) {
        // TODO Auto-generated method stub

    }
});
```

#### .4. 静态内部类

> 静态内部类是`不需要依赖于外部类的`，这点和类的静态成员属性有点类似，`并且它不能使用外部类的非static成员变量或者方法`，因为在没有外部类的对象的情况下，可以创建静态内部类的对象，如果允许访问外部类的非static成员就会产生矛盾，因为外部类的非static成员必须依附于具体的对象。

```java
public class ClassOuter {
    private int noStaticInt = 1;
    private static int STATIC_INT = 2;

    public void fun() {
        System.out.println("外部类方法");
    }

    public class InnerClass {
        //static int num = 1; 此时编辑器会报错 非静态内部类则不能有静态成员
        public void fun(){
            //非静态内部类的非静态成员可以访问外部类的非静态变量。
            System.out.println(STATIC_INT);
            System.out.println(noStaticInt);
        }
    }

    public static class StaticInnerClass {
        static int NUM = 1;//静态内部类可以有静态成员
        public void fun(){
            System.out.println(STATIC_INT);
            //System.out.println(noStaticInt); 此时编辑器会报 不可访问外部类的非静态变量错
        }
    }
}

public class TestInnerClass {
    public static void main(String[] args) {
        //非静态内部类 创建方式1
        ClassOuter.InnerClass innerClass = new ClassOuter().new InnerClass();
        //非静态内部类 创建方式2
        ClassOuter outer = new ClassOuter();
        ClassOuter.InnerClass inner = outer.new InnerClass();
        //静态内部类的创建方式
        ClassOuter.StaticInnerClass staticInnerClass = new ClassOuter.StaticInnerClass();
    }
}
```

### 2.用法

#### .1. final

- 局部内部类和匿名内部类只能访问局部final变量

> 如果局部变量的值在编译期间就可以确定，则直接在匿名内部里面创建一个拷贝。如果局部变量的值无法在编译期间确定，则通过构造器传参的方式来对拷贝进行初始化赋值。
>
> `匿名内部类是创建后是存储在堆中的`，而`方法中的局部变量是存储在Java栈中`，当方法执行完毕后，就进行退栈，同时局部变量也会消失。编译器为自动地帮我们在匿名内部类中创建了一个局部变量的备份，也就是说即使方法执结束，匿名内部类中还有一个备份,规定死这些局部域必须是常量，一旦赋值不能再发生变化了。

- 继承成员内部类的继承问题。一般来说，内部类是很少用来作为继承用的。但是当用来继承的话，要注意两点：

  　　1）成员内部类的引用方式必须为 Outter.Inner.

    2）构造器中必须有指向外部类对象的引用，并通过这个引用调用super()。

```java
class WithInner {
    class Inner{
         
    }
}
class InheritInner extends WithInner.Inner {
      
    // InheritInner() 是不能通过编译的，一定要加上形参
    InheritInner(WithInner wi) {
        wi.super(); //必须有这句调用
    }
  
    public static void main(String[] args) {
        WithInner wi = new WithInner();
        InheritInner obj = new InheritInner(wi);
    }
}
```

#### .2. 内存泄漏

- 如果一个`匿名内部类没有被任何引用持有`，那么匿名内部类对象用完就`有机会被回收`。
- 如果内部类`仅仅只是在外部类中被引用`，当`外部类的不再被引用时`，外部类和内部类就可以`都被GC回收`。
- 如果`当内部类的引用被外部类以外的其他类引用时`，就会造成`内部类和外部类无法被GC回收的情况`，即使外部类没有被引用，因为内部类持有指向外部类的引用）。

- 没有回收案例

```java
public class ClassOuter {

    Object object = new Object() {
        public void finalize() {
            System.out.println("inner Free the occupied memory...");
        }
    };

    public void finalize() {
        System.out.println("Outer Free the occupied memory...");
    }
}

public class TestInnerClass {
    public static void main(String[] args) {
        try {
            Test();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static void Test() throws InterruptedException {
        System.out.println("Start of program.");

        ClassOuter outer = new ClassOuter();
        Object object = outer.object;
        outer = null;

        System.out.println("Execute GC");
        System.gc();

        Thread.sleep(3000);
        System.out.println("End of program.");
    }
}
```

- 在关闭Activity/Fragment 的 onDestry，取消还在排队的Message:

```java
mHandler.removeCallbacksAndMessages(null);
```

- 将 Hanlder 创建为静态内部类并采用软引用方式

```java
   private static class MyHandler extends Handler {

        private final WeakReference<MainActivity> mActivity;

        public MyHandler(MainActivity activity) {
            mActivity = new WeakReference<MainActivity>(activity);
        }

        @Override
        public void handleMessage(Message msg) {
            MainActivity activity = mActivity.get();
            if (activity == null || activity.isFinishing()) {
               return;
            }
            // ...
        }
    }
```

### 3. 学习资源

- https://juejin.cn/post/6844903566293860366#heading-1

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/innerclass/  

