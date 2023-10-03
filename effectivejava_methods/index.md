# EffectiveJava_Methods


#### .1. 检查参数有效性

> 清楚地在文档中记载所有这些限制，并在方法主体的开头用检查来强制执行。 应该尝试在错误发生后尽快检测到错误，这是一般原则的特殊情况。 如果不这样做，则不太可能检测到错误，并且一旦检测到错误就更难确定错误的来源。

```java
/**
 * Returns a BigInteger whose value is (this mod m). This method
 * differs from the remainder method in that it always returns a
 * non-negative BigInteger.
 *
 * @param m the modulus, which must be positive
 * @return this mod m
 * @throws ArithmeticException if m is less than or equal to 0
 */
public BigInteger mod(BigInteger m) {
    if (m.signum() <= 0)
        throw new ArithmeticException("Modulus <= 0: " + m);
    ... // Do the computation
}
```

```java
// Private helper function for a recursive sort
private static void sort(long a[], int offset, int length) {
    assert a != null;
    assert offset >= 0 && offset <= a.length;
    assert length >= 0 && length <= a.length - offset;
    ... // Do the computation
}
```

#### .2. 进行保护性拷贝

> 将每个可变参数的防御性拷贝应用到构造方法中，并将拷贝用作 `Period` 实例的组件，以替代原始实例:

```java
// Broken "immutable" time period class
public final class Period {
    private final Date start;
    private final Date end;
    /**
     * @param  start the beginning of the period
     * @param  end the end of the period; must not precede start
     * @throws IllegalArgumentException if start is after end
     * @throws NullPointerException if start or end is null
     */
    public Period(Date start, Date end) {
        if (start.compareTo(end) > 0)
            throw new IllegalArgumentException(
                start + " after " + end);
        this.start = start;
        this.end   = end;
    }
    public Date start() {
        return start;
    }
    public Date end() {
        return end;
    }
    ...    // Remainder omitted
}

// Attack the internals of a Period instance
Date start = new Date();
Date end = new Date();
Period p = new Period(start, end);
end.setYear(78);  // Modifies internals of p!   <--   该类是可变更的
```

```java
// Repaired constructor - makes defensive copies of parameters
public Period(Date start, Date end) {
    this.start = new Date(start.getTime());
    this.end   = new Date(end.getTime());
    if (this.start.compareTo(this.end) > 0)
      throw new IllegalArgumentException(
          this.start + " after " + this.end);
}
```

#### .3. 明智谨慎使用可变参数

```java
// The right way to use varargs to pass one or more arguments
static int min(int firstArg, int... remainingArgs) {
    int min = firstArg;
    for (int arg : remainingArgs)
        if (arg < min)
            min = arg;
    return min;
}
```

#### .4. 返回空的数组或集合，不要返回null

```java
// Optimization - avoids allocating empty collections
public List<Cheese> getCheeses() {
    return cheesesInStock.isEmpty() ? Collections.emptyList()
        : new ArrayList<>(cheesesInStock);
}
```

```java
//The right way to return a possibly empty array
public Cheese[] getCheeses() {
    return cheesesInStock.toArray(new Cheese[0]);
}
```

#### .5. 谨慎返回option

> 在 Java 8 之前，编写在特定情况下无法返回任何值的方法时，可以采用两种方法。
>
> - 要么抛出异常：创建异常时捕获整个堆栈跟踪，开销大。
> - 返回 null（假设返回类型是对象是引用类型）：客户端必须包含特殊情况代码来处理 null 返回的可能性，除非程序员能够证明 null 返回是不可能的。
> - `Optional<T>`类表示一个不可变的容器，它可以包含一个非 null 的`T`引用，也可以什么都不包含。不包含任何内容的 Optional 被称为空（empty）

#### .6. 为所有已公开API编写文档注释

> **要正确地记录 API，必须在每个导出的类、接口、构造方法、方法和属性声明之前加上文档注释**。如果一个类是可序列化的，还应该记录它的序列化形式 （详见第 87 条）

```java
/**
 * Returns the element at the specified position in this list.
 *
 * <p>This method is <i>not</i> guaranteed to run in constant
 * time. In some implementations it may run in time proportional
 * to the element position.
 *
 * @param  index index of element to return; must be
 *         non-negative and less than the size of this list
 * @return the element at the specified position in this list
 * @throws IndexOutOfBoundsException if the index is out of range
 *         ({@code index < 0 || index >= this.size()})
 */
```

```java
/**
 * An object that maps keys to values.  A map cannot contain
 * duplicate keys; each key can map to at most one value.
 *
 * (Remainder omitted)
 *
 * @param <K> the type of keys maintained by this map
 * @param <V> the type of mapped values
 */
public interface Map<K, V> { ... }
```

```java
/**
 * An instrument section of a symphony orchestra.
 */
public enum OrchestraSection {
    /** Woodwinds, such as flute, clarinet, and oboe. */
    WOODWIND,
    /** Brass instruments, such as french horn and trumpet. */
    BRASS,
    /** Percussion instruments, such as timpani and cymbals. */
    PERCUSSION,
    /** Stringed instruments, such as violin and cello. */
    STRING;
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/effectivejava_methods/  

