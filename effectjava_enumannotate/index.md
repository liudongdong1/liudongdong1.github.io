# EffectJava_Enum&Annotate


### 1. 使用枚举类型代替整型常量

> 它们是通过公共静态 `final` 属性为每个枚举常量导出一个实例的类。 由于没有可访问的构造方法，枚举类型实际上是 `final` 的。 由于客户既不能创建枚举类型的实例也不能继承它，除了声明的枚举常量外，不能有任何实例。

```java
public enum Apple  { FUJI, PIPPIN, GRANNY_SMITH }
public enum Orange { NAVEL, TEMPLE, BLOOD }
```

```java
// Enum type with data and behavior
public enum Planet {
    MERCURY(3.302e+23, 2.439e6),
    VENUS  (4.869e+24, 6.052e6),
    EARTH  (5.975e+24, 6.378e6),
    MARS   (6.419e+23, 3.393e6),
    JUPITER(1.899e+27, 7.149e7),
    SATURN (5.685e+26, 6.027e7),
    URANUS (8.683e+25, 2.556e7),
    NEPTUNE(1.024e+26, 2.477e7);
    private final double mass;           // In kilograms
    private final double radius;         // In meters
    private final double surfaceGravity; // In m / s^2
    // Universal gravitational constant in m^3 / kg s^2
    private static final double G = 6.67300E-11;
    // Constructor
    Planet(double mass, double radius) {
        this.mass = mass;
        this.radius = radius;
        surfaceGravity = G * mass / (radius * radius);
    }
    public double mass()           { return mass; }
    public double radius()         { return radius; }
    public double surfaceGravity() { return surfaceGravity; }
    public double surfaceWeight(double mass) {
        return mass * surfaceGravity;  // F = ma
    }
}

public class WeightTable {
   public static void main(String[] args) {
      double earthWeight = Double.parseDouble(args[0]);
      double mass = earthWeight / Planet.EARTH.surfaceGravity();
      for (Planet p : Planet.values())
          System.out.printf("Weight on %s is %f%n",
                            p, p.surfaceWeight(mass));
      }
}
```

- 特定于常量（constant-specific）的方法, 广泛使用的

```java
// Enum type with constant-specific class bodies and data
public enum Operation {
    PLUS("+") {
        public double apply(double x, double y) { return x + y; }
    },
    MINUS("-") {
        public double apply(double x, double y) { return x - y; }
    },
    TIMES("*") {
        public double apply(double x, double y) { return x * y; }
    },
    DIVIDE("/") {
        public double apply(double x, double y) { return x / y; }
    };
    private final String symbol;
    Operation(String symbol) { this.symbol = symbol; }
    @Override public String toString() { return symbol; }
    public abstract double apply(double x, double y);
}

public static void main(String[] args) {
    double x = Double.parseDouble(args[0]);
    double y = Double.parseDouble(args[1]);
    for (Operation op : Operation.values())
        System.out.printf("%f %s %f = %f%n",
                          x, op, y, op.apply(x, y));
}
```

- 计算工资

```java
// The strategy enum pattern
enum PayrollDay {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY,
    SATURDAY(PayType.WEEKEND), SUNDAY(PayType.WEEKEND);
    private final PayType payType;
    PayrollDay(PayType payType) { this.payType = payType; }
    PayrollDay() { this(PayType.WEEKDAY); }  // Default
    int pay(int minutesWorked, int payRate) {
        return payType.pay(minutesWorked, payRate);
    }
    // The strategy enum type
    private enum PayType {
        WEEKDAY {
            int overtimePay(int minsWorked, int payRate) {
                return minsWorked <= MINS_PER_SHIFT ? 0 :
                  (minsWorked - MINS_PER_SHIFT) * payRate / 2;
            }
        },
        WEEKEND {
            int overtimePay(int minsWorked, int payRate) {
                return minsWorked * payRate / 2;
            }
        };
        abstract int overtimePay(int mins, int payRate);
        private static final int MINS_PER_SHIFT = 8 * 60;
        int pay(int minsWorked, int payRate) {
            int basePay = minsWorked * payRate;
            return basePay + overtimePay(minsWorked, payRate);
        }
    }
}
```

### 2. 使用实例属性替代序数

> 永远不要`从枚举的序号中得出与它相关的值`; 请将其保存在实例属性中：

```jaav
public enum Ensemble {
    SOLO(1), DUET(2), TRIO(3), QUARTET(4), QUINTET(5),
    SEXTET(6), SEPTET(7), OCTET(8), DOUBLE_QUARTET(8),
    NONET(9), DECTET(10), TRIPLE_QUARTET(12);
    private final int numberOfMusicians;
    Ensemble(int size) { this.numberOfMusicians = size; }
    public int numberOfMusicians() { return numberOfMusicians; }
}
```

### 3. 使用 EnumSet 替代位属性

```java
// EnumSet - a modern replacement for bit fields
public class Text {
    public enum Style { BOLD, ITALIC, UNDERLINE, STRIKETHROUGH }
    // Any Set could be passed in, but EnumSet is clearly best
    public void applyStyles(Set<Style> styles) { ... }
}
```

### 4. 使用 EnumMap 替代序数索引

```java
class Plant {
    enum LifeCycle { ANNUAL, PERENNIAL, BIENNIAL }
    final String name;
    final LifeCycle lifeCycle;
    Plant(String name, LifeCycle lifeCycle) {
        this.name = name;
        this.lifeCycle = lifeCycle;
    }
    @Override public String toString() {
        return name;
    }
}
```

```java
// Using an EnumMap to associate data with an enum
Map<Plant.LifeCycle, Set<Plant>>  plantsByLifeCycle =
    new EnumMap<>(Plant.LifeCycle.class);
for (Plant.LifeCycle lc : Plant.LifeCycle.values())
    plantsByLifeCycle.put(lc, new HashSet<>());
for (Plant p : garden)
    plantsByLifeCycle.get(p.lifeCycle).add(p);
System.out.println(plantsByLifeCycle);
```

```java
// Using a nested EnumMap to associate data with enum pairs
public enum Phase {
   SOLID, LIQUID, GAS;
   public enum Transition {
      MELT(SOLID, LIQUID), FREEZE(LIQUID, SOLID),
      BOIL(LIQUID, GAS),   CONDENSE(GAS, LIQUID),
      SUBLIME(SOLID, GAS), DEPOSIT(GAS, SOLID);
      private final Phase from;
      private final Phase to;
      Transition(Phase from, Phase to) {
         this.from = from;
         this.to = to;
      }
      // Initialize the phase transition map
      private static final Map<Phase, Map<Phase, Transition>>
        m = Stream.of(values()).collect(groupingBy(t -> t.from,
         () -> new EnumMap<>(Phase.class),
         toMap(t -> [t.to](http://t.to), t -> t,
            (x, y) -> y, () -> new EnumMap<>(Phase.class))));
      public static Transition from(Phase from, Phase to) {
         return m.get(from).get(to);
      }
   }
}
```

### 5. 使用接口模拟可扩展的枚举

```java
// Emulated extensible enum using an interface
public interface Operation {
    double apply(double x, double y);
}
public enum BasicOperation implements Operation {
    PLUS("+") {
        public double apply(double x, double y) { return x + y; }
    },
    MINUS("-") {
        public double apply(double x, double y) { return x - y; }
    },
    TIMES("*") {
        public double apply(double x, double y) { return x * y; }
    },
    DIVIDE("/") {
        public double apply(double x, double y) { return x / y; }
    };
    private final String symbol;
    BasicOperation(String symbol) {
        this.symbol = symbol;
    }
    @Override public String toString() {
        return symbol;
    }
}
```

```java
// Emulated extension enum
public enum ExtendedOperation implements Operation {
    EXP("^") {
        public double apply(double x, double y) {
            return Math.pow(x, y);
        }
    },
    REMAINDER("%") {
        public double apply(double x, double y) {
            return x % y;
        }
    };
    private final String symbol;
    ExtendedOperation(String symbol) {
        this.symbol = symbol;
    }
    @Override public String toString() {
        return symbol;
    }
}
```

### 6.  注解优于命名模式

```java
// Code containing a repeated annotation
@ExceptionTest(IndexOutOfBoundsException.class)
@ExceptionTest(NullPointerException.class)
public static void doublyBad() { ... }
```

### 7. 始终使用 Override 注解

> 在你认为要重写父类声明的每个方法声明上使用 `Override` 注解。 如果正在编写一个没有标记为抽象的类，并且确信它重写了其父类中的抽象方法，则无需将 `Override` 注解放在该方法上。

```java
// Can you spot the bug?
public class Bigram {
    private final char first;
    private final char second;
    public Bigram(char first, char second) {
        this.first  = first;
        this.second = second;
    }
    //这里并不会报错，但会导致程序出错
    public boolean equals(Bigram b) {
        return b.first == first && b.second == second;
    }
    @Override public boolean equals(Object o) {
        if (!(o instanceof Bigram))
            return false;
        Bigram b = (Bigram) o;
        return b.first == first && b.second == second;
    }
    public int hashCode() {
        return 31 * first + second;
    }
    public static void main(String[] args) {
        Set<Bigram> s = new HashSet<>();
        for (int i = 0; i < 10; i++)
            for (char ch = 'a'; ch <= 'z'; ch++)
                s.add(new Bigram(ch, ch));
        System.out.println(s.size());
    }
}
```

### 8. 使用标记接口定义类型

- 标记接口（marker interface）：没有包含方法声明的接口，而只是指明一个类实现了具有某种属性的接口。例如，Serializable接口。
  - **标记接口则没有定义的类型；**
  - **标记接口可以更加精确的进行锁定。**
     如果注解类型利用@Target（ElementType.TYPE）声明，它就可以被应用到任何类或者接口，假设有一个标记只是适用于特殊的接口实现，但它却可以被应用到类，如果定义成一个标记接口，就可以用它将唯一的接口扩展成适用的接口。

```java
public interface Set<E> extends Collection<E> {
    // Query Operations
/** @param e element to be added to this set
     * @return <tt>true</tt> if this set did not already contain the specified
     *         element
     * @throws UnsupportedOperationException if the <tt>add</tt> operation
     *         is not supported by this set
     * @throws ClassCastException if the class of the specified element
     *         prevents it from being added to this set
     * @throws NullPointerException if the specified element is null and this
     *         set does not permit null elements
     * @throws IllegalArgumentException if some property of the specified element
     *         prevents it from being added to this set
     */
    boolean add(E e);
}

public interface Collection<E> extends Iterable<E> {
/** @param e element whose presence in this collection is to be ensured
     * @return <tt>true</tt> if this collection changed as a result of the
     *         call
     * @throws UnsupportedOperationException if the <tt>add</tt> operation
     *         is not supported by this collection
     * @throws ClassCastException if the class of the specified element
     *         prevents it from being added to this collection
     * @throws NullPointerException if the specified element is null and this
     *         collection does not permit null elements
     * @throws IllegalArgumentException if some property of the element
     *         prevents it from being added to this collection
     * @throws IllegalStateException if the element cannot be added at this
     *         time due to insertion restrictions
     */
    boolean add(E e);
}
```

- 标记注解（marker annotation）：特殊类型的注解，其中不包含成员。标记注解的唯一目的就是标记声明。例如，@Override。



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/effectjava_enumannotate/  

