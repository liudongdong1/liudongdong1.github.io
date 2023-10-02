# EffectJava_generic


### 1. 不要在新代码中使用原生类型

> - 原始类型 List 和参数化类型 `List<Object>` 之间有什么区别？ 松散地说，前者已经选择了泛型类型系统，而后者明确地告诉编译器，它能够保存任何类型的对象。
> - 泛型类型信息在运行时被擦除，所以在无限制通配符类型以外的参数化类型上使用 `instanceof` 运算符是非法的。 使用无限制通配符类型代替原始类型，不会对 `instanceof` 运算符的行为产生任何影响。
> - `Set<Object>` 是一个参数化类型，表示一个可以包含任何类型对象的集合，`Set<?>` 是一个通配符类型，表示一个只能包含某些未知类型对象的集合，`Set` 是一个原始类型，它不在泛型类型系统之列。 前两个类型是安全的，最后一个不是。 

```java
// Legitimate use of raw type - instanceof operator
if (o instanceof Set) {       // Raw type
    Set<?> s = (Set<?>) o;    // Wildcard type
    ...
}
//一旦确定 o 对象是一个 Set，则必须将其转换为通配符 Set<?>，而不是原始类型 Set
```

![image-20210520083453610](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210520083453610.png)

### 2. 消除非检查警告

> 使用泛型编程时，会看到许多编译器警告：`未经检查的强制转换警告`，`未经检查的方法调用警告`，`未经检查的参数化可变长度类型警告`以及`未经检查的转换警告`。

- **每当使用 `@SuppressWarnings(“unchecked”)` 注解时，请添加注释，说明为什么是安全的。** 

```java
// Adding local variable to reduce scope of @SuppressWarnings
public <T> T[] toArray(T[] a) {
    if (a.length < size) {
        // This cast is correct because the array we're creating
        // is of the same type as the one passed in, which is T[].
        @SuppressWarnings("unchecked") T[] result =
            (T[]) Arrays.copyOf(elements, size, a.getClass());
        return result;
    }
    System.arraycopy(elements, 0, a, 0, size);
    if (a.length > size)
        a[size] = null;
    return a;
}
```

### 3. 列表优于数组

> 数组是协变和具体化的; 泛型是不变的，类型擦除的。 因此，数组提供运行时类型的安全性，但不提供编译时类型的安全性，反之亦然。 一般来说，数组和泛型不能很好地混合工作。 如果你发现把它们混合在一起，得到编译时错误或者警告，你的第一个冲动应该是用列表来替换数组。

### 4. 优先考虑&使用泛型

```java
// Initial attempt to generify Stack - won't compile!
public class Stack<E> {
    private E[] elements;
    private int size = 0;
    private static final int DEFAULT_INITIAL_CAPACITY = 16;
    public Stack() {
        //elements = new E[DEFAULT_INITIAL_CAPACITY]; //你不能创建一个不可具体化类型的数组，例如类型 E。每当编写一个由数组支持的泛型时，就会出现此问题。 有两种合理的方法来解决它。 第一种解决方案直接规避了对泛型数组创建的禁用：创建一个 Object 数组并将其转换为泛型数组类型。
        elements = (E[]) new Object[DEFAULT_INITIAL_CAPACITY];
    }
    public void push(E e) {
        ensureCapacity();
        elements[size++] = e;
    }
    //
    public E pop() {
        if (size == 0)
            throw new EmptyStackException();
        E result = elements[--size]; //E 是不可具体化的类型，编译器无法在运行时检查强制转换
        elements[size] = null; // Eliminate obsolete reference
        return result;
    }
    
    // Appropriate suppression of unchecked warning
    public E pop() {
        if (size == 0)
            throw new EmptyStackException();
        // push requires elements to be of type E, so cast is correct
        @SuppressWarnings("unchecked") E result =
            (E) elements[--size];
        elements[size] = null; // Eliminate obsolete reference
        return result;
    }
    ... // no changes in isEmpty or ensureCapacity
}
```

- 需要创建一个不可改变但适用于许多不同类型的对象

> 使用单个对象进行所有必需的类型参数化，但是需要编写一个静态工厂方法来重复地为每个请求的类型参数化分配对象。 这种称为泛型单例工厂（generic singleton factory）的模式用于方法对象（function objects）

### 5. 使用限定通配符来增加 API 的灵活性

> 尽管 `List<String>` 不是 `List<Object>` 的子类型是违反直觉的，但它确实是有道理的。 可以将任何对象放入 `List<Object>` 中，但是只能将字符串放入 `List<String>` 中。 由于 `List<String>` 不能做 `List<Object>` 所能做的所有事情，所以它不是一个子类型.

```java
public class Stack<E> {
    public Stack();
    public void push(E e);
    public E pop();
    public boolean isEmpty();
}
//pushAll 的输入参数的类型不应该是「E 的 Iterable 接口」，而应该是「E 的某个子类型的 Iterable 接口」，并且有一个通配符类型，这意味着：Iterable<? extends E>。
// Wildcard type for a parameter that serves as an E producer
public void pushAll(Iterable<? extends E> src) {
    for (E e : src)
        push(e);
}
//Collection<Object> 不是 Collection<Number> 的子类型。 通配符类型再一次提供了一条出路。popAll 的输入参数的类型不应该是「E 的集合」，而应该是「E 的某个父类型的集合」（其中父类型被定义为 E 是它自己的父类型[JLS，4.10]）
// Wildcard type for parameter that serves as an E consumer
public void popAll(Collection<? super E> dst) {
    while (!isEmpty())
        dst.add(pop());
}
```

### 6. 泛型&可变参数

```java
// Safe method with a generic varargs parameter
@SafeVarargs
static <T> List<T> flatten(List<? extends T>... lists) {
    List<T> result = new ArrayList<>();
    for (List<? extends T> list : lists)
        result.addAll(list);
    return result;
}

//如果不使用注解
// List as a typesafe alternative to a generic varargs parameter
static <T> List<T> flatten(List<List<? extends T>> lists) {
    List<T> result = new ArrayList<>();
    for (List<? extends T> list : lists)
        result.addAll(list);
    return result;
}
```

### 7. 优先考虑类型安全的异构容器

> `Favorites` 类，它允许其客户端保存和检索任意多种类型的 favorite 实例。

```java
// Typesafe heterogeneous container pattern - implementation
public class Favorites {
    private Map<Class<?>, Object> favorites = new HashMap<>();
    public<T> void putFavorite(Class<T> type, T instance) {
        favorites.put(Objects.requireNonNull(type), instance);
    }
    public<T> T getFavorite(Class<T> type) {
        return type.cast(favorites.get(type));
    }
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/effectjava_generic/  

