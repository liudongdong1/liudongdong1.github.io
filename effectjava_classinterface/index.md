# EffectJava_class&Interface


### 1. 使类和成员的可访问性最小化

- **让每个类或成员尽可能地不可访问。** 换句话说，使用尽可能低的访问级别，与你正在编写的软件的对应功能保持一致。

- **公共类的实例字段很少情况下采用 public 修饰（详见第 16 条）。** 如果一个实例字段是非 final 的，或者是对可变对象的引用，那么通过将其公开，你就放弃了限制可以存储在字段中的值的能力。这意味着你放弃了执行涉及该字段的不变量的能力。另外，当字段被修改时，就放弃了采取任何操作的能力，**因此带有公共可变字段的类通常不是线程安全的** 。即使一个字段是 final 的，并且引用了一个不可变的对象，通过将其公开，你放弃了切换到一个新的内部数据表示的灵活性，而该字段并不存在。  `不是很明白`
- **类具有公共静态 final 数组字段，或返回这样一个字段的访问器是错误的。**

```java
// Potential security hole!
public static final Thing[] VALUES = { ... };
// 修改后安全版本
private static final Thing[] PRIVATE_VALUES = { ... };
public static final Thing[] values() {
    return PRIVATE_VALUES.clone();
}
```

### 2. 在公共类中使用访问方法而不是公共属性

```java
// Encapsulation of data by accessor methods and mutators
class Point {
    private double x;
    private double y;
    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }
    public double getX() { return x; }
    public double getY() { return y; }
    public void setX(double x) { this.x = x; }
    public void setY(double y) { this.y = y; }
}
```

- 如果属性是不可变的，可以直接暴露属性

```java
// Public class with exposed immutable fields - questionable
public final class Time {
    private static final int HOURS_PER_DAY    = 24;
    private static final int MINUTES_PER_HOUR = 60;
    public final int hour;
    public final int minute;
    public Time(int hour, int minute) {
        if (hour < 0 || hour >= HOURS_PER_DAY)
           throw new IllegalArgumentException("Hour: " + hour);
        if (minute < 0 || minute >= MINUTES_PER_HOUR)
           throw new IllegalArgumentException("Min: " + minute);
        this.hour = hour;
        this.minute = minute;
    }
    ... // Remainder omitted
}
```

### 3. 使可变性最小化

> 不可变类简单来说是其实例不能被修改的类。 包含在每个实例中的所有信息在对象的生命周期中是固定的，因此不会观察到任何变化。 Java 平台类库包含许多不可变的类，包括 `String` 类、基本类型包装类以及 `BigInteger` 类和 `BigDecimal` 类。 

1. **不要提供修改对象状态的方法（也称为 mutators，设值方法）。**
2. **确保这个类不能被继承。** 这可以防止粗心或者恶意的子类假装对象的状态已经改变，从而破坏类的不可变行为。 防止子类化，通常是通过 `final` 修饰类。
3. **把所有字段设置为 final。** 通过系统强制执行的方式，清楚地表达了你的意图。 另外，如果一个新创建的实例的引用在缺乏同步机制的情况下从一个线程传递到另一个线程，就必须保证正确的行为，正如内存模型[JLS，17.5; Goetz06 16] 所述。
4. **把所有的字段设置为 private。** 这可以防止客户端获得对字段引用的可变对象的访问权限，并直接修改这些对象。 虽然技术上允许不可变类具有包含基本类型数值的公有的 `final` 字段或对不可变对象的引用，但不建议这样做，因为这样使得在以后的版本中无法再改变内部的表示状态（详见第 15 和 16 条）。
5. **确保对任何可变组件的互斥访问。** 如果你的类有任何引用可变对象的字段，请确保该类的客户端无法获得对这些对象的引用。 切勿将这样的属性初始化为客户端提供的对象引用，或从访问方法返回属性。 在构造方法，访问方法和 `readObject` 方法（详见第 88 条）中进行防御性拷贝（详见第 50 条）。

>  注意算术运算如何创建并返回一个新的 `Complex` 实例，而不是修改这个实例。 这种模式被称为函数式方法，因为方法返回将操作数应用于函数的结果，而不修改它们。 与其对应的过程式的（procedural）或命令式的（imperative）的方法相对比，在这种方法中，将一个过程作用在操作数上，导致其状态改变。

> **构造方法应该创建完全初始化的对象，并建立所有的不变性。** 除非有令人信服的理由，否则不要提供独立于构造方法或静态工厂的公共初始化方法。 同样，不要提供一个“reinitialize”方法，使对象可以被重用，就好像它是用不同的初始状态构建的。 这样的方法通常以增加的复杂度为代价，仅仅提供很少的性能优势。

`不可变的对象可以完全处于一种状态，也就是被创建时的状态` 

- **线程安全的**
- **不可变对象可以被自由地共享。** 因此，不可变类应鼓励客户端尽可能重用现有的实例。 为常用的值提供公共的静态 final 常量
- 坚决不要为每个属性编写一个 get 方法后再编写一个对应的 set 方法。 **除非有充分的理由使类成为可变类，否则类应该是不可变的**
- 示例代码一：

```java
public static final Complex ZERO = new Complex(0, 0);
public static final Complex ONE  = new Complex(1, 0);
public static final Complex I    = new Complex(0, 1);

// Immutable complex number class
public final class Complex {
    private final double re;
    private final double im;
    public Complex(double re, double im) {
        this.re = re;
        this.im = im;
    }
    public double realPart() {
        return re;
    }
    public double imaginaryPart() {
        return im;
    }
    public Complex plus(Complex c) {
        return new Complex(re + c.re, im + c.im);
    }
    public Complex minus(Complex c) {
        return new Complex(re - c.re, im - c.im);
    }
    public Complex times(Complex c) {
        return new Complex(re * c.re - im * c.im,
                re * c.im + im * c.re);
    }
    public Complex dividedBy(Complex c) {
        double tmp = c.re * c.re + c.im * c.im;
        return new Complex((re * c.re + im * c.im) / tmp,
                (im * c.re - re * c.im) / tmp);
    }
    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }
        if (!(o instanceof Complex)) {
            return false;
        }
        Complex c = (Complex) o;
        // See page 47 to find out why we use compare instead of ==
        return Double.compare(c.re, re) == 0
                && Double.compare(c.im, im) == 0;
    }
    @Override
    public int hashCode() {
        return 31 * Double.hashCode(re) + Double.hashCode(im);
    }
    @Override
    public String toString() {
        return "(" + re + " + " + im + "i)";
    }
}
```

- 实例代码二

```java
// Immutable class with static factories instead of constructors
public class Complex {
    private final double re;
    private final double im;
    private Complex(double re, double im) {
        this.re = re;
        this.im = im;
    }
    public static Complex valueOf(double re, double im) {
        return new Complex(re, im);
    }
    ... // Remainder unchanged
}
```

### 4.组合优于继承

> `包装类不适合在回调框架（callback frameworks）中使用`，其中对象将自我引用传递给其他对象以用于后续调用（「回调」）。 因为一个被包装的对象不知道它外面的包装对象，所以它传递一个指向自身的引用（this），回调时并不记得外面的包装对象。 这被称为 SELF 问题[Lieberman86]。 
>
> `只有在两个类之间存在「is-a」关系的情况下，B 类才能继承 A 类。`

- 错误案例

```java
// Broken - Inappropriate use of inheritance!
public class InstrumentedHashSet<E> extends HashSet<E> {
    // The number of attempted element insertions
    private int addCount = 0;
    public InstrumentedHashSet() {
    }
    public InstrumentedHashSet(int initCap, float loadFactor) {
        super(initCap, loadFactor);
    }
    @Override public boolean add(E e) {
        addCount++;
        return super.add(e);
    }
    @Override public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return super.addAll(c);
    }
    public int getAddCount() {
        return addCount;
    }
}

InstrumentedHashSet<String> s = new InstrumentedHashSet<>();
s.addAll(List.of("Snap", "Crackle", "Pop"));  #大小由3错误成6   继承并重写了add 方法
```

- 不要继承一个现有的类，而应该给你的新类增加一个私有属性，该属性是 现有类的实例引用，这种设计被称为组合（composition）

```java
// Reusable forwarding class
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;
public class ForwardingSet<E> implements Set<E> {
    private final Set<E> s;           //组合
    public ForwardingSet(Set<E> s) {
        this.s = s;
    }
    public void clear() {
        s.clear();
    }
    public boolean contains(Object o) {
        return s.contains(o);
    }
    public boolean isEmpty() {
        return s.isEmpty();
    }
    public int size() {
        return s.size();
    }
    public Iterator<E> iterator() {
        return s.iterator();
    }
    public boolean add(E e) {
        return s.add(e);
    }
    public boolean remove(Object o) {
        return s.remove(o);
    }
    public boolean containsAll(Collection<?> c) {
        return s.containsAll(c);
    }
    public boolean addAll(Collection<? extends E> c) {
        return s.addAll(c);
    }
    public boolean removeAll(Collection<?> c) {
        return s.removeAll(c);
    }
    public boolean retainAll(Collection<?> c) {
        return s.retainAll(c);
    }
    public Object[] toArray() {
        return s.toArray();
    }
    public <T> T[] toArray(T[] a) {
        return s.toArray(a);
    }
    @Override
    public boolean equals(Object o) {
        return s.equals(o);
    }
    @Override
    public int hashCode() {
        return s.hashCode();
    }
    @Override
    public String toString() {
        return s.toString();
    }
}
```

- 封装类

```java
// Wrapper class - uses composition in place of inheritance
import java.util.Collection;
import java.util.Set;
public class InstrumentedSet<E> extends ForwardingSet<E> {
    private int addCount = 0;
    public InstrumentedSet(Set<E> s) {
        super(s);
    }
    @Override public boolean add(E e) {
        addCount++;
        return super.add(e);
    }
    @Override public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return super.addAll(c);
    }
    public int getAddCount() {
        return addCount;
    }
}
```

### 5. 要么设计继承并提供文档说明，要么禁用继承

### 6. 接口优于抽象类

> 抽象的骨架实现类（abstract skeletal implementation class）来与接口一起使用，将接口和抽象类的优点结合起来。 接口定义了类型，可能提供了一些默认的方法，而骨架实现类在原始接口方法的顶层实现了剩余的非原始接口方法。 继承骨架实现需要大部分的工作来实现一个接口。 这就是模板方法设计模式。

- 静态工厂方法，在 `AbstractList` 的顶层包含一个完整的功能齐全的 `List` 实现

```java
// Concrete implementation built atop skeletal implementation
static List<Integer> intArrayAsList(int[] a) {
    Objects.requireNonNull(a);
    // The diamond operator is only legal here in Java 9 and later
    // If you're using an earlier release, specify <Integer>
    return new AbstractList<>() {
        @Override 
        public Integer get(int i) {
            return a[i];  // Autoboxing ([Item 6](https://www.safaribooksonline.com/library/view/effective-java-third/9780134686097/ch2.xhtml#lev6))
        }
        @Override 
        public Integer set(int i, Integer val) {
            int oldVal = a[i];
            a[i] = val;     // Auto-unboxing
            return oldVal;  // Autoboxing
        }
        @Override 
        public int size() {
            return a.length;
        }
    };
}
```

- AbstractMapEntry

```java
// Skeletal implementation class
public abstract class AbstractMapEntry<K,V>
        implements Map.Entry<K,V> {
    // Entries in a modifiable map must override this method
    @Override public V setValue(V value) {
        throw new UnsupportedOperationException();
    }
    // Implements the general contract of Map.Entry.equals
    @Override 
    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (!(o instanceof Map.Entry))
            return false;
        Map.Entry<?,?> e = (Map.Entry) o;
        return Objects.equals(e.getKey(),  getKey())
            && Objects.equals(e.getValue(), getValue());
    }
    // Implements the general contract of Map.Entry.hashCode
    @Override 
    public int hashCode() {
        return Objects.hashCode(getKey())
             ^ Objects.hashCode(getValue());
    }
    @Override 
    public String toString() {
        return getKey() + "=" + getValue();
    }
}
```

### 7. 接口仅用来定义类型

- **常量接口模式是对接口的糟糕使用** 如果使用常量，使用以下方式：

```java
// Constant utility class
package com.effectivejava.science;
public class PhysicalConstants {
  private PhysicalConstants() { }  // Prevents instantiation
  public static final double AVOGADROS_NUMBER = 6.022_140_857e23;
  public static final double BOLTZMANN_CONST  = 1.380_648_52e-23;
  public static final double ELECTRON_MASS    = 9.109_383_56e-31;
}
```

### 8. 类层次结构由于标签

```java
// Class hierarchy replacement for a tagged class
abstract class Figure {
    abstract double area();
}
class Circle extends Figure {
    final double radius;
    Circle(double radius) { this.radius = radius; }
    @Override double area() { return Math.PI * (radius * radius); }
}
class Rectangle extends Figure {
    final double length;
    final double width;
    Rectangle(double length, double width) {
        this.length = length;
        this.width  = width;
    }
    @Override double area() { return length * width; }
}
```

### 8. 支持使用静态成员类而非静态类

### 9. 使用函数对象表示策略

```java
//策略类
class StringLengthComparator {
  private StringLengthComparator() {}
  public static StringLengthComparator instance = new StringLengthComparator();
  public int compare(String s1, String s2) {
    return s1.length() - s2.length();
  }
}

// strategy interface， 继承多个比较方法
public interface Comparator<T> {
  public int compare(T t1, T t2);
}

class StringLengthComparator implements Comparator {
  ...
}
```

- 使用匿名类每次调用都会新建一个实例，需要将函数存储到一个私有静态final域中

```java
// Exporting a concrete strategy
class Host {
  private static class StrLenCmp implements Comparator<String>, Serializable {
    public int compare(String s1, String s2) {
      return s1.length() - s2.length();
    }
  }

  public static final Comparator<String> STRING_LENGTH_COMPARATOR = new StrLenCmp();
  ...
}
```

### 10. 将源文件限制为单个顶级类

```python
// Static member classes instead of multiple top-level classes
public class Test {
    public static void main(String[] args) {
        System.out.println(Utensil.NAME + [Dessert.NAME](http://Dessert.NAME));
    }
    private static class Utensil {
        static final String NAME = "pan";
    }
    private static class Dessert {
        static final String NAME = "cake";
    }
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/effectjava_classinterface/  

