# C# Reference&&Valuepassing


> `Java中传参都是值传递，如果是基本类型，就是对值的拷贝，如果是对象，就是对引用地址的拷贝。`C# 中传递的参数是基元类型（int，float等）或结构体（struct），那么就是传值调用。传递的参数是类（class）那么就是传引用调用。`传递的参数前有ref或者out关键字，那么就是传引用调用`。
>
> 传值：   函数参数压栈的是参数的副本。   任何的修改是在副本上作用，没有作用在原来的变量上。 
>
> 传指针：   压栈的是指针变量的副本。   当你对指针解指针操作时，其值是指向原来的那个变量，所以对原来变量操作。 
>
> 传引用：  压栈的是引用的副本。由于引用是指向某个变量的，对引用的操作其实就是对他指向的变量的操作。

## 1. 基本类型传参

```java
public class Process{
    public void function3(int a){
        a=2;
    }
    public void function4(int a){
        int b=5;
        a=b;
    }
    public static void main(String[] args){
        Process process=new Process();
        int age=18;
        process.function3(age);       //age 值不改变
        process.function4(age);       //age 值不改变
    }
}
```

> <font color=red>结论： 基本类型的传参，对传参进行修改，不影响原本参数的值。</font>基本类型的传参，在方法内部是值拷贝，有一个新的局部变量得到这个值，对这个局部变量的修改不影响原来的参数。

## 2. 对象类型传参

```java
public class Process{
    public void function1(Car car){     //最终值修改了
        car.setColor("blue");
    }
    public void function2(Car car){     //最终值没有修改,这里修改传参指向地址
        Car car2=new Car("black");
        car=car2;
        car.setColor("Orange");
    }
    public static void main(String []args){
        Process process =new Process();
        Car car=new Car("red");
        process.function1(car);
        process.function2(car);
    }
}
```

> <font color=red>结论: 对象类型的传参，直接调用传参set方法，可以对原本参数进行修改。如果**修改传参的指向地址**，调用传参的set方法，无法对原本参数的值进行修改。</font>对象类型的传参，传递的是堆上的地址，在方法内部是有一个新的局部变量得到引用地址的拷贝，对该局部变量的操作，影响的是同一块地址，因此原本的参数也会受影响，反之，若修改局部变量的引用地址，则不会对原本的参数产生任何可能的影响。

## 3. 效率

从功能上。按值传递在传递的时候，实参被复制了一份，然后在函数体内使用，**函数体内修改参数变量时修改的是实参的一份拷贝，**而实参本身是没有改变的，所以如果想在调用的函数中修改实参的值，使用值传递是不能达到目的的，这时只能使用引用或指针传递。

从类型安全上讲。值传递与引用传递在参数传递过程中都执行强类型检查，而指针传递的类型检查较弱，特别地，如果参数被声明为 void ，那么它基本上没有类型检查，只要是指针，编译器就认为是合法的，所以这给bug的产生制造了机会，使程序的健壮性稍差，如果没有必要，就使用值传递和引用传递，最好不用指针传递，更好地利用编译器的类型检查，使得我们有更少的出错机会，以增加代码的健壮性。

## 4. JVM

![JVM](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200513220321548.png)

1. 程序计数器: 存储每个线程下一步将执行的JVM指令。
2. JVM栈(JVM Stack): JVM栈是线程私有的，每个线程创建的同时都会创建JVM栈，JVM栈中存放的为当前线程中局部基本类型的变量（java中定义的八种基本类型：boolean、char、byte、short、int、long、float、double）、部分的返回结果以及Stack Frame(每个方法都会开辟一个自己的栈帧)，<font color=red>非基本类型的对象在JVM栈上仅存放一个指向堆上的地址</font>
3. 堆(heap): J<font color=red>VM用来存储对象实例以及数组值的区域，可以认为Java中所有通过new创建的对象的内存都在此分配，Heap中的对象的内存需要等待GC进行回收。</font>
4. 方法区（Method Area): 方法区域存放了所加载的类的信息（名称、修饰符等）、类中的静态变量、类中定义为final类型的常量、类中的Field信息、类中的方法信息，当开发人员在程序中通过Class对象中的getName、isInterface等方法来获取信息时，这些数据都来源于方法区域。
5. 本地方法栈（Native Method Stacks): JVM采用本地方法栈来支持native方法的执行，此区域用于存储每个native方法调用的状态。
6. 运行时常量池（Runtime Constant Pool): 存放的为类中的固定的常量信息、方法和Field的引用信息等，其空间从方法区域中分配。JVM在加载类时会为每个class分配一个独立的常量池，但是运行时常量池中的字符串常量池是全局共享的。

JVM是基于栈来操作的，每一个线程有自己的操作栈，遇到方法调用时会开辟栈帧，它含有自己的返回值，局部变量表，操作栈，以及对常量池的符号引用。 如果是基本类型，则存放在栈里的是值，如果是对象，存放在栈上是对象在堆上存放的地址。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200513220653291.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/c-reference-valuepassing/  

