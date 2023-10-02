# EffectiveJava_Exception


#### .1. 只对异常的情况下才使用异常

> - 因为异常设计的初衷适用于不正常的情形，所有几乎没有 JVM 实现试图对他们进行优化，使它们与显式的测试一样快。
> - `把代码放在 try-catch 块中反而阻止了现代 JVM 实现本可能执行的某些特定优化`。
> - 对`数据进行遍历的标准模式并不会导致冗余的检查`。有些 JVM 实现会将它们优化掉。

> - **异常应该只用于异常的情况下；他们永远不应该用于正常的程序控制流程。** 
> - **设计良好的 API 不应该强迫它的客户端为了正常的控制流程而使用异常。**

```java
for ( Iterator<Foo> i = collection.iterator(); i.hasNext(); ){
    Foo foo = i.next();
    ...
}

/* Do not use this hideous code for iteration over a collection! */
try {
    Iterator<Foo> i = collection.iterator();
    while ( true )
    {
        Foo foo = i.next();
        ...
    }
} catch ( NoSuchElementException e ) {
}
```

#### .2. 对可恢复的情况使用受检异常，对编程错误使用运行时异常

> Java 程序设计语言提供了三种 throwable：受检异常（checked exceptions）、运行时异常（runtime exceptions）和错误（errors）。
>
> - **如果期望调用者能够合理的恢复程序运行，对于这种情况就应该使用受检异常。** 通过抛出受检异常，强迫调用者在一个 catch 子句中处理该异常，或者把它传播出去。
> - 有两种非受检的 throwable：运行时异常和错误。在行为上两者是等同的：它们都是`不需要也不应该被捕获的 throwable`。如果程序抛出非受检异常或者错误，往往属于不可恢复的情形，程序继续执行下去有害无益。如果程序没有捕捉到这样的 throwable，将会导致当前线程中断（halt），并且出现适当的错误消息。

```java
/* Invocation with state-testing method and unchecked exception */
if ( obj.actionPermitted( args ) ) {
    obj.action( args );
} else {
    ... /* Handle exceptional condition */
}
```

#### .3. 优先使用标准异常

> **不要直接重用 Exception、RuntimeException、Throwable 或者 Error。** 对待这些类要像对待抽象类一样。你无法可靠地测试这些异常，因为它们是一个方法可能抛出的其他异常的超类。

|              异常               |                   使用场合                   |
| :-----------------------------: | :------------------------------------------: |
|    IllegalArgumentException     |            非 null 的参数值不正确            |
|      IllegalStateException      |           不适合方法调用的对象状态           |
|      NullPointerException       |    在禁止使用 null 的情况下参数值为 null     |
|    IndexOutOfBoundsExecption    |                下标参数值越界                |
| ConcurrentModificationException | 在禁止并发修改的情况下，检测到对象的并发修改 |
|  UnsupportedOperationException  |           对象不支持用户请求的方法           |

#### .4. 抛出与抽象对应的异常

> **更高层的实现应该捕获低层的异常，同时抛出可以按照高层抽象进行解释的异常。**

```java
/**
 * Returns the element at the specified position in this list.
 * @throws IndexOutOfBoundsException if the index is out of range
 * ({@code index < 0 || index >= size()}).
 */
public E get(int index) {
    ListIterator<E> i = listIterator(index);
    try {
        return(i.next() );
    } catch (NoSuchElementException e) {
        throw new IndexOutOfBoundsException("Index: " + index);
    }
}
```

#### .5.每个方法抛出的异常都需要创建文档

#### .6. 不能忽略捕获信息

> **为了捕获失败，异常的细节信息应该包含“对该异常有贡献”的所有参数和字段的值**
>
> **千万不要在细节消息中包含密码、密钥以及类似的信息！**
>
> **如果选择忽略异常， catch 块中应该包含一条注释，说明为什么可以这么做，并且变量应该命名为 ignored:**

```java
/**
 * Constructs an IndexOutOfBoundsException.
 *
 * @param lowerBound the lowest legal index value
 * @param upperBound the highest legal index value plus one
 * @param index the actual index value
 */
public IndexOutOfBoundsException( int lowerBound, int upperBound,
                  int index ) {
    // Generate a detail message that captures the failure
    super(String.format(
              "Lower bound: %d, Upper bound: %d, Index: %d",
              lowerBound, upperBound, index ) );
    // Save failure information for programmatic access
    this.lowerBound = lowerBound;
    this.upperBound = upperBound;
    this.index = index;
}
```

#### .7. 失败原子性

> **失败的方法调用应该使对象保持在被调用之前的状态。** 具有这种属性的方法被称为具有失败原子性 （failure atomic）。
>
> - 调整计算处理过程的顺序，使得任何可能会失败的计算部分都在对象状态被修改之前发生



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/effectivejava_exception/  

