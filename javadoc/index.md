# javaDoc


> - 第一段：概要描述，通常用一句或者一段话简要`描述该类的作用`，以英文句号作为结束
> - 第二段：详细描述，通常用一段或者多段话来`详细描述该类的作用`，一般每段话都以英文句号作为结束
> - 第三段：文档标注，用于`标注作者、创建时间、参阅类等`信息

### 1. 类注释

> @author、@version、@since、@see、@link、@code、@param、@return、@exception、@throws等。

```java
package org.springframework.util;
/**
 * Miscellaneous {@link String} utility methods.
 * 
 */
public abstract class StringUtils {
}
```

```java
package java.lang;

/**
 * Class {@code Object} is the root of the class hierarchy.
 * Every class has {@code Object} as a superclass. All objects,
 * including arrays, implement the methods of this class.
 */
public class Object {}
```

```java
package org.springframework.util;

/**
 * Miscellaneous {@link String} utility methods.
 *
 * <p>Mainly for internal use within the framework; consider
 * <a href="http://commons.apache.org/proper/commons-lang/">Apache's Commons Lang</a>
 * for a more comprehensive suite of {@code String} utilities.
 *
 * <p>This class delivers some simple functionality that should really be
 * provided by the core Java {@link String} and {@link StringBuilder}
 * classes. It also provides easy-to-use methods to convert between
 * delimited strings, such as CSV strings, and collections and arrays.
 *
 */
public abstract class StringUtils {
}
```

#### .1. @link

> @link的使用语法`{@link 包名.类名#方法名(参数类型)}`，其中当包名在当前类中已经导入了包名可以省略，可以只是一个类名，也可以是仅仅是一个方法名，也可以是类名.方法名，使用此文档标记的类或者方法，可用`通过按住Ctrl键+单击 可以快速跳到相应的类或者方法上`，解析成html其实就是使用< code> 包名.类名#方法名(参数类型)< /code>

```java
// 完全限定的类名
{@link java.lang.Character}

// 省略包名
{@link String}

// 省略类名，表示指向当前的某个方法
{@link #length()}

// 包名.类名.方法名(参数类型)
{@link java.lang.String#charAt(int)}

```

#### .2. @code

> {@code text} 会被解析成`<code> text </code>`
> 将文本标记为代码样式的文本，在code内部可以使用 < 、> 等不会被解释成html标签, code标签有自己的样式.

#### .3. @param

> 一般类中支持泛型时会通过@param来解释泛型的类型

```java
/**
* @param <E> the type of elements in this list
*/
public interface List<E> extends Collection<E> {}
```

#### .4. @author

```java
// 纯文本作者
@author Rod Johnson

// 纯文本作者，邮件
@author Igor Hersht, igorh@ca.ibm.com

// 超链接邮件 纯文本作者
@author <a href="mailto:ovidiu@cup.hp.com">Ovidiu Predescu</a>

// 纯文本邮件
@author shane_curcuru@us.ibm.com

// 纯文本 组织
@author Apache Software Foundation

// 超链接组织地址 纯文本组织
@author <a href="https://jakarta.apache.org/turbine"> Apache Jakarta Turbine</a>

```

#### .5. @see

> @see 一般用于标记该类相关联的类,@see即可以用在类上，也可以用在方法上。

```java
/**
 * @see IntStream
 * @see LongStream
 * @see DoubleStream
 * @see <a href="package-summary.html">java.util.stream</a>
 * /
public interface Stream<T> extends BaseStream<T, Stream<T>> {}
```

#### .6. @since

> @since 一般用于标记文件创建时项目当时对应的版本，一般后面跟版本号，也可以跟是一个时间，表示文件当前创建的时间

```java
package java.util.stream;
/**
* @since 1.8
*/
public interface Stream<T> extends BaseStream<T, Stream<T>> {}
```

#### .7. @version

```java
package com.sun.org.apache.xml.internal.resolver;
 /**
 * @version 1.0
 */
public class Resolver extends Catalog {}
```

> - 第一段：概要描述，通常用一句或者一段话简要描述`该方法的作用`，以英文句号作为结束
> - 第二段：详细描述，通常用一段或者多段话来详细描述该方法的作用，一般每段话都以英文句号作为结束
> - 第三段：文档标注，用于`标注参数、返回值、异常、参阅等`

### 2. 方法注释

#### .1. @param

```java
/**
* @param str the {@code CharSequence} to check (may be {@code null})
*/
public static boolean containsWhitespace(@Nullable CharSequence str) {}
```

#### .2. @reture

```java
/**
* @return {@code true} if the {@code String} is not {@code null}, its
*/
public static boolean hasText(@Nullable String str){}
```

#### .3. @throws

```java
/**
* @throws IllegalArgumentException when the given source contains invalid encoded sequences
*/
public static String uriDecode(String source, Charset charset){}
```

#### .4. @exceptioln

```java
/**
* @exception IllegalArgumentException if <code>key</code> is null.
*/
public static Object get(String key) throws IllegalArgumentException {}
```

#### .5. @value

```java
/** 默认数量 {@value} */
private static final Integer QUANTITY = 1;
```

#### .6. @inheritDoc

> @inheritDoc用于注解在`重写方法或者子类上`，用于继承父类中的Javadoc
>
> - `基类的文档注释被继承到了子类`
> - `子类可以再加入自己的注释（特殊化扩展）`
> - `@return @param @throws 也会被继承`

### 3. 案例示范

```java
package org.springframework.util;

/**
 * Miscellaneous {@link String} utility methods.
 *
 * <p>Mainly for internal use within the framework; consider
 * <a href="http://commons.apache.org/proper/commons-lang/">Apache's Commons Lang</a>
 * for a more comprehensive suite of {@code String} utilities.
 *
 * <p>This class delivers some simple functionality that should really be
 * provided by the core Java {@link String} and {@link StringBuilder}
 * classes. It also provides easy-to-use methods to convert between
 * delimited strings, such as CSV strings, and collections and arrays.
 *
 * @author Rod Johnson
 * @author Juergen Hoeller
 * @author Keith Donald
 * @author Rob Harrop
 * @author Rick Evans
 * @author Arjen Poutsma
 * @author Sam Brannen
 * @author Brian Clozel
 * @since 16 April 2001
 */
public abstract class StringUtils {

	/**
	 * Decode the given encoded URI component value. Based on the following rules:
	 * <ul>
	 * <li>Alphanumeric characters {@code "a"} through {@code "z"}, {@code "A"} through {@code "Z"},
	 * and {@code "0"} through {@code "9"} stay the same.</li>
	 * <li>Special characters {@code "-"}, {@code "_"}, {@code "."}, and {@code "*"} stay the same.</li>
	 * <li>A sequence "{@code %<i>xy</i>}" is interpreted as a hexadecimal representation of the character.</li>
	 * </ul>
	 * 
	 * @param source the encoded String
	 * @param charset the character set
	 * @return the decoded value
	 * @throws IllegalArgumentException when the given source contains invalid encoded sequences
	 * @since 5.0
	 * @see java.net.URLDecoder#decode(String, String)
	 */
	public static String uriDecode(String source, Charset charset) {}

```

```java
package com.example.demo;

/**
 * 类 {@code OrderService} 订单服务层.
 *
 * <p> 主要包括 创建订单、取消订单、查询订单等功能更
 *
 * @see Order
 * @author <a href="mailto:mengday.zhang@gmail.com">Mengday Zhang</a>
 * @since 2018/5/12
 */
public class OrderService {

	/** 默认数量 {@value} */
    private static final Integer QUANTITY = 1;

    /**
     * 创建订单.
     *
     * <p> 创建订单需要传用户id和商品列表(商品id和商品数量).
     *
     * <p><pre>{@code
     *  演示如何使用该方法
     *  List<Goods> items = new ArrayList<>();
     *  Goods goods = new Goods(1L, BigDecimal.ONE);
     *  Goods goods2 = new Goods(2L, BigDecimal.TEN);
     *  items.add(goods);
     *  items.add(goods2);
     *
     *  Order order1 = new Order();
     *  order.setUserId("1");
     *  order.setItems(items);
     *  OrderService#createOrder(order);
     * }
     * </pre>
     *
     * @param order 订单信息
     * @throws NullPointerException 参数信息为空
     * @exception IllegalArgumentException  数量不合法
     * @return 是否创建成功
     * @version 1.0
     * @see {@link Order}
     */
    public boolean createOrder(Order order) throws IllegalArgumentException{
        Objects.requireNonNull(order);

        List<Goods> items = order.getItems();
        items.forEach(goods -> {
            BigDecimal quantity = goods.getQuantity();
            if (quantity == null || BigDecimal.ZERO.compareTo(quantity) == 0) {
                throw new IllegalArgumentException();
            }
        });

        System.out.println("create order...");

        return true;
    }
}
```

> 通过IDEA生成Javadoc： Tools --> Generate JavaDoc -->注意要配置编码，如果不配送为生成乱码，还需要配置Output directory

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210619155227016.png)

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/javadoc/  

