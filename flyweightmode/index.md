# FlyweightMode


> 运用`共享技术`有效地支持大量细粒度对象的复用。系统`只使用少量的对象`，而这些对象都很相似，状态变化很小，可以实现对象的多次复用。由于享元模式要求能够共享的对象必须是细粒度对象，因此它又称为轻量级模式，它是一种对象结构型模式。享元模式结构较为复杂，一般结合工厂模式一起使用。
>
> - **Flyweight（抽象享元类）**：通常是一个接口或抽象类，在抽象享元类中`声明了具体享元类公共的方法`，这些方法可以向外界提供享元对象的内部数据（内部状态），同时也可以通过这些方法来设置外部数据（外部状态）。
> - **ConcreteFlyweight（具体享元类）**：它实现了抽象享元类，其实例称为享元对象；在具体享元类中为内部状态提供了存储空间。通常我们可以结合单例模式来设计具体享元类，为每一个具体享元类提供唯一的享元对象。
> - **UnsharedConcreteFlyweight（非共享具体享元类）**：并不是所有的抽象享元类的子类都需要被共享，`不能被共享的子类可设计为非共享具体享元类`；当需要一个非共享具体享元类的对象时可以直接通过实例化创建。
> - **FlyweightFactory（享元工厂类）**：享元工厂类用于`创建并管理享元对象`，它针对抽象享元类编程，将各种类型的具体享元对象存储在一个享元池中，享元池一般设计为一个存储“键值对”的集合（也可以是其他类型的集合），可以结合工厂模式进行设计；当用户请求一个具体享元对象时，享元工厂提供一个存储在享元池中已创建的实例或者创建一个新的实例（如果不存在的话），返回新创建的实例并将其存储在享元池中。

> - 单纯享元模式：在单纯享元模式中，所有的具体享元类都是可以共享的，不存在非共享具体享元类。
> - 复合享元模式：将一些单纯享元对象使用组合模式加以组合，还可以形成复合享元对象，这样的复合享元对象本身不能共享，但是它们可以分解成单纯享元对象，而后者则可以共享

> **适用场景**：
>
> - 一个系统有大量相同或者相似的对象，造成内存的大量耗费。
> - 对象的大部分状态都可以外部化，可以将这些外部状态传入对象中。
> - 在使用享元模式时需要维护一个存储享元对象的享元池，而这需要耗费一定的系统资源，因此，应当在需要多次重复使用享元对象时才值得使用享元模式。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210306122355580.png)

```java
import java.util.HashMap;
 
public class ShapeFactory {
   private static final HashMap<String, Shape> circleMap = new HashMap<>();
 
   public static Shape getCircle(String color) {
      Circle circle = (Circle)circleMap.get(color);
 
      if(circle == null) {
         circle = new Circle(color);
         circleMap.put(color, circle);
         System.out.println("Creating circle of color : " + color);
      }
      return circle;
   }
}
```

### 1. String 享元模式

```java
public class Main {
    public static void main(String[] args) {
        String s1 = "hello";
        String s2 = "hello";  //jvm就返回这个字面量绑定的引用，所以s1==s2
        String s3 = "he" + "llo";  //s3中字面量的拼接其实就是hello，jvm在编译期间就已经对它进行优化
        String s4 = "hel" + new String("lo");  //lo存在字符串常量池，new String("lo")存在堆中，String s4 = "hel" + new String("lo")实质上是两个对象的相加，编译器不会进行优化，相加的结果存在堆中，而s1存在字符串常量池中，当然不相等。s1==s9的原理一样。
        String s5 = new String("hello");   //s4==s5两个相加的结果都在堆中，不用说，肯定不相等
        String s6 = s5.intern();  //s5.intern()方法能使一个位于堆中的字符串在运行期间动态地加入到字符串常量池中（字符串常量池的内容是程序启动的时候就已经加载好了），如果字符串常量池中有该对象对应的字面量，则返回该字面量在字符串常量池中的引用，否则，创建复制一份该字面量到字符串常量池并返回它的引用。因此s1==s6输出true。
        String s7 = "h";
        String s8 = "ello";
        String s9 = s7 + s8;
        System.out.println(s1==s2);//true
        System.out.println(s1==s3);//true
        System.out.println(s1==s4);//false
        System.out.println(s1==s9);//false
        System.out.println(s4==s5);//false
        System.out.println(s1==s6);//true
    }
}
```

> 以字面量的形式创建String变量时，jvm会在编译期间就把该字面量`hello`放到字符串常量池中，由Java程序启动的时候就已经加载到内存中了。这个字符串常量池的特点就是有且只有一份相同的字面量，如果有其它相同的字面量，jvm则返回这个字面量的引用，如果没有相同的字面量，则在字符串常量池创建这个字面量并返回它的引用。

### 2. Integer 享元模式

```java
public static void main(String[] args) {
    Integer i1 = 12 ;
    Integer i2 = 12 ;
    System.out.println(i1 == i2);  //true

    Integer b1 = 128 ;
    Integer b2 = 128 ;
    System.out.println(b1 == b2); //false
}
```

```java
public final class Integer extends Number implements Comparable<Integer> {
    public static Integer valueOf(int var0) {
        return var0 >= -128 && var0 <= Integer.IntegerCache.high ? Integer.IntegerCache.cache[var0 + 128] : new Integer(var0);
    }
    //...省略...
}
```

```java
//是Integer内部的私有静态类,里面的cache[]就是jdk事先缓存的Integer。
private static class IntegerCache {
    static final int low = -128;//区间的最低值
    static final int high;//区间的最高值，后面默认赋值为127，也可以用户手动设置虚拟机参数
    static final Integer cache[]; //缓存数组

    static {
        // high value may be configured by property
        int h = 127;
        //这里可以在运行时设置虚拟机参数来确定h  :-Djava.lang.Integer.IntegerCache.high=250
        String integerCacheHighPropValue =
            sun.misc.VM.getSavedProperty("java.lang.Integer.IntegerCache.high");
        if (integerCacheHighPropValue != null) {//用户设置了
            int i = parseInt(integerCacheHighPropValue);
            i = Math.max(i, 127);//虽然设置了但是还是不能小于127
            // 也不能超过最大值
            h = Math.min(i, Integer.MAX_VALUE - (-low) -1);
        }
        high = h;

        cache = new Integer[(high - low) + 1];
        int j = low;
        //循环将区间的数赋值给cache[]数组
        for(int k = 0; k < cache.length; k++)
            cache[k] = new Integer(j++);
    }

    private IntegerCache() {}
}
```

>  `Integer` 默认先创建并缓存 `-128 ~ 127` 之间数的 `Integer` 对象，当调用 `valueOf` 时如果参数在 `-128 ~ 127` 之间则计算下标并从缓存中返回，否则创建一个新的 `Integer` 对象

### 3. Long中的享元模式

```java
public final class Long extends Number implements Comparable<Long> {
    public static Long valueOf(long var0) {
        return var0 >= -128L && var0 <= 127L ? Long.LongCache.cache[(int)var0 + 128] : new Long(var0);
    }   
    private static class LongCache {
        private LongCache(){}

        static final Long cache[] = new Long[-(-128) + 127 + 1];

        static {
            for(int i = 0; i < cache.length; i++)
                cache[i] = new Long(i - 128);
        }
    }
    //...
}
```

### 4. Apache Commons Pool2中的享元模式

> 对象池化的基本思路是：将用过的对象保存起来，等下一次需要这种对象的时候，再拿出来重复使用，从而在一定程度上减少频繁创建对象所造成的开销。用于充当保存对象的“容器”的对象，被称为“对象池”（Object Pool，或简称Pool）
>
> - **PooledObject（池对象）**：用于封装对象（如：线程、数据库连接、TCP连接），将其包裹成可被池管理的对象。
> -  **PooledObjectFactory（池对象工厂）**：定义了操作PooledObject实例生命周期的一些方法，PooledObjectFactory必须实现线程安全。
> -  **Object Pool （对象池）**：Object Pool负责管理PooledObject，如：借出对象，返回对象，校验对象，有多少激活对象，有多少空闲对象。

### Resource

- [Java中String字符串常量池](https://www.cnblogs.com/tongkey/p/8587060.html)

- [Integer的享元模式解析](https://blog.csdn.net/LuoZheng4698729/article/details/53995925)
- [7种结构型模式之：享元模式（Flyweight）与数据库连接池的原理](https://blog.csdn.net/qq_22075041/article/details/69802378?locationNum=7&fps=1)
- [Apache commons-pool2-2.4.2源码学习笔记](https://blog.csdn.net/zilong_zilong/article/details/78556281)
- [Apache Commons Pool2 源码分析](https://blog.csdn.net/amon1991/article/details/77110657)
- https://juejin.cn/post/6844903683860217864

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/flyweightmode/  

