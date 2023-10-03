# EffectiveJava_Design


#### .1. 局部变量作用域最小

> - 如果循环终止后不需要循环变量的内容，那么**优先选择 for 循环而不是 while 循环**。
> -  **for-each** 循环优于传统 for 循环
>   - **有损过滤（Destructive filtering）**——如果`需要遍历集合，并删除指定选元素`，则需要使用显式迭代器，以便可以调用其 remove 方法。 通常可以使用在 Java 8 中添加的 Collection 类中的 removeIf 方法，来避免显式遍历。
>   - **转换**——如果需要`遍历一个列表或数组并替换其元素的部分或全部值`，那么需要列表迭代器或数组索引来替换元素的值。
>   - **并行迭代**——如果需要`并行地遍历多个集合`，那么需要显式地控制迭代器或索引变量，以便所有迭代器或索引变量都可以同步进行 (正如上面错误的 card 和 dice 示例中无意中演示的那样)。

```java
// Preferred idiom for iterating over a collection or array
for (Element e : c) {
    ... // Do Something with e
}
```

```java
// Idiom for iterating when you need the iterator
for (Iterator<Element> i = c.iterator(); i.hasNext(); ) {
    Element e = i.next();
    ... // Do something with e and i
}
```

#### .2. 了解使用标准库

> - **通过使用标准库，你可以利用编写它的专家的知识和以前使用它的人的经验。**
> - **在每个主要版本中，都会向库中添加许多特性，了解这些新增特性是值得的。**
> - **每个程序员都应该熟悉 java.lang、java.util 和 java.io 的基础知识及其子包**，不要白费力气重新发明轮子

#### .3. 精确答案就应避免使用 float 和 double 类型

> - float 和 double 类型主要用于科学计算和工程计算。它们执行二进制浮点运算，该算法经过精心设计，能够在很大范围内快速提供精确的近似值。但是，它们不能提供准确的结果
> - **使用 BigDecimal、int 或 long 进行货币计算。**
> - 自己处理十进制小数点，而且数值不是太大，可以使用 int 或 long。如果数值不超过 9 位小数，可以使用 int；如果不超过 18 位，可以使用 long。如果数量可能超过 18 位，则使用 BigDecimal。

#### .4. 基本数据类型优于包装类

> 自动装箱和自动拆箱模糊了基本类型和包装类型之间的区别，但不会消除它们;
>
> 1. 基本类型只有它们的值，而包装类型具有与其值不同的标识
> 2. 基本类型只有全功能值，而每个包装类型除了对应的基本类型的所有功能值外，还有一个非功能值，即 null。
> 3. 基本类型比包装类型更节省时间和空间。

#### .5. 当使用其他类型更合适时应避免使用字符串

#### .6. 字符串连接引起的性能问题

> - **字符串串联运算符重复串联 n 个字符串需要 n 的平方级时间**
> - **要获得能接受的性能，请使用 StringBuilder 代替 String** 来存储正在构建的语句

#### .7. 通过接口引用对象

> **如果存在合适的接口类型，那么应该使用接口类型声明参数、返回值、变量和字段**
>
> **如果没有合适的接口，就使用类层次结构中提供所需功能的最底层的类**

```java
// Good - uses interface as type
Set<Son> sonSet = new LinkedHashSet<>();

// Bad - uses class as type!
LinkedHashSet<Son> sonSet = new LinkedHashSet<>();
```

#### .8. 接口优先反射

> 反射允许一个类使用另一个类，即使在编译前者时后者并不存在。然而，这种能力是有代价的：
>
> - **你失去了编译时类型检查的所有好处，** 包括异常检查。如果一个程序试图反射性地调用一个不存在的或不可访问的方法，它将在运行时失败，除非你采取了特殊的预防措施。
> - **执行反射访问所需的代码既笨拙又冗长。** 写起来很乏味，读起来也很困难。
> - **性能降低。** 反射方法调用比普通方法调用慢得多。到底慢了多少还很难说，因为有很多因素在起作用。在我的机器上，调用一个没有输入参数和返回 int 类型的方法时，用反射执行要慢 11 倍。

#### .9. 谨慎使用本地方法

> Java 本地接口（JNI）允许 Java 程序调用本地方法，这些方法是用 C 或 C++ 等本地编程语言编写的。
>
> - 使用本地方法有严重的缺点。由于本地语言不安全（详见第 50 条），使用本地方法的应用程序不再能免受内存毁坏错误的影响。
> - 使用本地方法的程序的可移植性较差。它们也更难调试。

#### .10. 遵守被广泛认可的命名约定

| Identifier Type    | Example                                              |
| :----------------- | :--------------------------------------------------- |
| Package or module  | `org.junit.jupiter.api`, `com.google.common.collect` |
| Class or Interface | Stream, FutureTask, LinkedHashMap,HttpClient         |
| Method or Field    | remove, groupingBy, getCrc                           |
| Constant Field     | MIN_VALUE, NEGATIVE_INFINITY                         |
| Local Variable     | i, denom, houseNum                                   |
| Type Parameter     | T, E, K, V, X, R, U, V, T1, T2                       |



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/effectivejava_design/  

