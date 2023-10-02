# EffectiveJava_Serialise


#### .1. 非常谨慎地实现 Serializable

> **实现 `Serializable` 接口的一个主要代价是，一旦类的实现被发布，它就会降低更改该类实现的灵活性。** 当类实现 `Serializable` 时，其字节流编码（或序列化形式）成为其导出 API 的一部分。一旦广泛分发了一个类，通常就需要永远支持序列化的形式.
>
> **实现 `Serializable` 接口的第二个代价是，增加了出现 bug 和安全漏洞的可能性**: 对象是用构造函数创建的；序列化是一种用于创建对象的超语言机制。无论你接受默认行为还是无视它，反序列化都是一个「隐藏构造函数」，其他构造函数具有的所有问题它都有。由于没有与反序列化关联的显式构造函数，因此很容易忘记必须让它能够保证所有的不变量都是由构造函数建立的，并且不允许攻击者访问正在构造的对象内部。依赖于默认的反序列化机制，会让对象轻易地遭受不变性破坏和非法访问.
>
> **实现 `Serializable` 接口的第三个代价是，它增加了与发布类的新版本相关的测试负担。** 当一个可序列化的类被修改时，重要的是检查是否可以在新版本中序列化一个实例，并在旧版本中反序列化它，反之亦然。因此，所需的测试量与可序列化类的数量及版本的数量成正比，工作量可能很大.
>
> **实现 `Serializable` 接口并不是一个轻松的决定。** 如果一个类要参与一个框架，该框架依赖于 Java 序列化来进行对象传输或持久化，这对于类来说实现 `Serializable` 接口就是非常重要的。此外，如果类 A 要成为另一个类 B 的一个组件，类 B 必须实现 `Serializable` 接口，若类 A 可序列化，它就会更易于被使用。然而，与实现 `Serializable` 相关的代价很多。每次设计一个类时，都要权衡利弊。历史上，像 `BigInteger` 和 `Instant` 这样的值类实现了 `Serializable` 接口，集合类也实现了 `Serializable` 接口。表示活动实体（如线程池）的类很少情况适合实现 `Serializable` 接口。
>
> **为继承而设计的类（详见第 19 条）很少情况适合实现 `Serializable` 接口，接口也很少情况适合扩展它。**
>
> **内部类（详见第 24 条）不应该实现 `Serializable`**.

#### .2. 使用自定义的序列化形式

> **在没有考虑默认序列化形式是否合适之前，不要接受它。** 接受默认的序列化形式应该是一个三思而后行的决定，即从灵活性、性能和正确性的角度综合来看.
>
> **如果对象的物理表示与其逻辑内容相同，则默认的序列化形式可能是合适的。**

```java
// Good candidate for default serialized form
public class Name implements Serializable {
    /**
     * Last name. Must be non-null.
     * @serial
     */
    private final String lastName;
    /**
     * First name. Must be non-null.
     * @serial
     */
    private final String firstName;
    /**
     * Middle name, or null if there is none.
     * @serial
     */
    private final String middleName;
    ... // Remainder omitted
}
```

> **当对象的物理表示与其逻辑数据内容有很大差异时，使用默认的序列化形式有四个缺点：**
>
> - **它将导出的 API 永久地绑定到当前的内部实现。** 在上面的例子中，私有 `StringList.Entry` 类成为公共 API 的一部分。如果在将来的版本中更改了实现，`StringList` 类仍然需要接受链表形式的输出，并产生链表形式的输出。这个类永远也摆脱不掉处理链表项所需要的所有代码，即使不再使用链表作为内部数据结构。
> - **它会占用过多的空间。** 在上面的示例中，序列化的形式不必要地表示链表中的每个条目和所有链接关系。这些链表项以及链接只不过是实现细节，不值得记录在序列化形式中。因为这样的序列化形式过于庞大，将其写入磁盘或通过网络发送将非常慢。
> - **它会消耗过多的时间。** 序列化逻辑不知道对象图的拓扑结构，因此必须遍历开销很大的图。在上面的例子中，只要遵循 next 的引用就足够了。
> - **它可能导致堆栈溢出。** 默认的序列化过程执行对象图的递归遍历，即使对于中等规模的对象图，这也可能导致堆栈溢出。用 1000-1800 个元素序列化 `StringList` 实例会在我的机器上生成一个 `StackOverflowError`。令人惊讶的是，序列化导致堆栈溢出的最小列表大小因运行而异（在我的机器上）。显示此问题的最小列表大小可能取决于平台实现和命令行标志；有些实现可能根本没有这个问题。

#### .3. 用序列化代理代替序列化实例

```java
// EnumSet's serialization proxy
private static class SerializationProxy<E extends Enum<E>>
    implements Serializable {
    private static final long serialVersionUID = 362491234563181265L;
    // The element type of this enum set.
    private final Class<E> elementType;
    // The elements contained in this enum set.
    private final Enum<?>[] elements;
    SerializationProxy(EnumSet<E> set) {
        elementType = set.elementType;
        elements = set.toArray(new Enum<?>[0]);
    }
    private Object readResolve() {
        EnumSet<E> result = EnumSet.noneOf(elementType);
        for (Enum<?> e : elements)
            result.add((E) e);
        return result;
    }
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/effectivejava_serialise/  

