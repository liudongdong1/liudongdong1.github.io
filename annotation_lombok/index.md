# Annotation_Lombok


#### 1. @Getter and @Setter

- 使用@Getter和/或@Setter注释任何字段，以使lombok自动生成默认的getter / setter。

- `默认的getter只是返回该字段`，如果`该字段被称为foo，则名为getFoo`（如果该字段的类型为`boolean，则为isFoo`）。

- 默认生成的 getter / setter方法是`公共的`，除非你明确指定一个AccessLevel。合法访问级别为PUBLIC，PROTECTED，PACKAGE和PRIVATE。


- 你还可以在类上添加@Getter和/或@Setter注释。在这种情况下，就好像你使用该注释来注释该类中的所有非静态字段一样。


- 你始终可以使用特殊的`AccessLevel.NONE访问级别来手动禁用任何字段的getter / setter生成`。这使你可以覆盖类上的@Getter，@Setter或@Data注释的行为。

```java
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

public class GetterSetterExample {
  
  @Getter 
  @Setter 
  private int age = 10;
  
  @Setter(AccessLevel.PROTECTED) 
  private String name;
  
  @Override 
  public String toString() {
    return String.format("%s (age: %d)", name, age);
  }
}
```

#### 2.  @ToString

- 任何类定义都可以使用@ToString注释，以使lombok生成`toString()方法的实现`。

- 默认情况下，将打印所有非静态字段。`如果要跳过某些字段，可以使用@ ToString.Exclude注释这些字段`。或者，可以通过使用@ToString（onlyExplicitlyIncluded = true），然后使用`@ToString.Include标记要包含的每个字段，来确切指定希望使用的字段。`


- 通过将callSuper设置为true，可以将toString的超类实现的输出包含到输出中。请注意，`java.lang.Object中toString() 的默认实现几乎毫无意义`。

```java
import lombok.ToString;

@ToString
public class ToStringExample {
  private static final int STATIC_VAR = 10;
  private String name;
  private Shape shape = new Square(5, 10);
  private String[] tags;
  @ToString.Exclude 
  private int id;
  
  public String getName() {
    return this.name;
  }
  
  @ToString(callSuper=true, includeFieldNames=true)
  public static class Square extends Shape {
    private final int width, height;
    
    public Square(int width, int height) {
      this.width = width;
      this.height = height;
    }
  }
}
```

#### 3.@EqualsAndHashCode

- 任何类定义都可以使用@EqualsAndHashCode进行注释，以使lombok生成`equals(Object other)和hashCode()方法的实现`。默认情况下，它将使用所有非静态，非瞬态字段，但是您可以通过使用`@EqualsAndHashCode.Include标记类型成员来修改使用哪些字段`（甚至指定要使用各种方法的输出）。 @EqualsAndHashCode.Exclude。或者，可以通过使用@ EqualsAndHashCode.Include标记并使用`@EqualsAndHashCode(onlyExplicitlyIncluded = true)`来精确指定要使用的字段或方法。

- 如果将@EqualsAndHashCode应用于扩展另一个类的类，则此功能会有些棘手。通常，为此类自动生成equals和hashCode方法是一个坏主意，`因为超类还定义了字段，该字段也需要equals / hashCode代码，但不会生成此代码`。通过将`callSuper设置为true，可以在生成的方法中包括超类的equals和hashCode方法`。

```java
import lombok.EqualsAndHashCode;

@EqualsAndHashCode
public class EqualsAndHashCodeExample {
  private transient int transientVar = 10;
  private String name;
  private double score;
  @EqualsAndHashCode.Exclude 
  private Shape shape = new Square(5, 10);
  private String[] tags;
  @EqualsAndHashCode.Exclude 
  private int id;
  
  public String getName() {
    return this.name;
  }
  
  @EqualsAndHashCode(callSuper=true)
  public static class Square extends Shape {
    private final int width, height;
    
    public Square(int width, int height) {
      this.width = width;
      this.height = height;
    }
  }
}
```

#### 4.  @AllArgsConstructor, @RequiredArgsConstructor and @NoArgsConstructor

- `@NoArgsConstructor`将生成没有参数的构造函数。如果`字段由final修饰，则将导致编译器错误`，除非使用@NoArgsConstructor(force = true)，否则所有final字段都将初始化为0 / false / null。对于`具有约束的字段(例如@NonNull字段)，不会生成任何检查。`

- `@RequiredArgsConstructor`为每个需要特殊处理的字段`生成一个带有1个参数的构造函数`。所有`未初始化的final字段都会获取一个参数，以及所有未声明其位置的未标记为@NonNull的字段。`

- @AllArgsConstructor为类中的`每个字段生成一个带有1个参数的构造函数`。标有@`NonNull的字段将对这些参数进行空检查。`

```java
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.NonNull;
@RequiredArgsConstructor(staticName = "of")
@AllArgsConstructor(access = AccessLevel.PROTECTED)
public class ConstructorExample<T> {
  private int x, y;
  @NonNull 
  private T description;
  @NoArgsConstructor
  public static class NoArgsExample {
    @NonNull 
    private String field;
  }
}
```

#### 5. @Data

- @Data是一个方便的快捷方式批注，它将`@ToString，@EqualsAndHashCode，@ Getter / @Setter和@RequiredArgsConstructor的功能捆绑在一起`：换句话说，@Data生成通常与简单POJO关联的所有样板（普通的旧Java对象）和bean：`所有字段的getter，所有非final字段的setter`，以及涉及类字段的适当的`toString，equals和hashCode实现`，以及`初始化所有final字段以及所有非final字段的构造函数没有使用@NonNull标记的初始化程序，以确保该字段永远不会为null。`

```java
import lombok.AccessLevel;
import lombok.Setter;
import lombok.Data;
import lombok.ToString;

@Data 
public class DataExample {
  private final String name;
  @Setter(AccessLevel.PACKAGE) 
  private int age;
  private double score;
  private String[] tags;
  
  @ToString(includeFieldNames=true)
  @Data(staticConstructor="of")
  public static class Exercise<T> {
    private final String name;
    private final T value;
  }
}
```

#### 6. @Value

- @Value注解和`@Data`类似，区别在于它会把所有成员变量默认定义为`private final`修饰，并且不会生成`set`方法。

#### 7. @Builder

- 构建者模式：只能标注到类上，将生成类的一个当前流程的一种链式构造工厂，如下：

```java
import lombok.Builder;
import lombok.Singular;
import java.util.Set;

@Builder
public class BuilderExample {
  @Builder.Default 
  private long created = System.currentTimeMillis();
  private String name;
  private int age;
  @Singular 
  private Set<String> occupations;
}
User buildUser = User.builder().username("riemann").password("123").build();
```

#### 8. @Slf4j and @Log4j

在需要打印日志的类中使用，项目中使用`slf4j`、`log4j`日志框架

#### 9. @NonNull

该注解快速判断是否为空,为空抛出`java.lang.NullPointerException`。

#### 10. @Synchronized

注解自动添加到同步机制，生成的代码并`不是直接锁方法,而是锁代码块`， 作用范围是方法上。

#### 11. @Cleanup

注解用于确保已分配的资源被释放（`IO`的连接关闭）。

#### 12. @Accessors

- @Accessors批注用于`配置lombok如何生成和查找getter和setter`。


- 默认情况下，lombok遵循针对getter和setter的bean规范：例如，名为Pepper的字段的getter是getPepper。 但是，有些人可能希望打破bean规范，以得到更好看的API。 @Accessors允许您执行此操作。
- 标注到类上，`chain`属性设置为`true`时，类的所有属性的`setter`方法返回值将为`this`，用来支持`setter`方法的链式写法

```java
@Accessors(prefix = "r")
@Getter
@Setter
private String rUsername = "riemann";

//编译之后
public String getUsername() {
    return rUsername;
}
public void setUsername(String rUsername) {
    this.rUsername = rUsername;
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/annotation_lombok/  

