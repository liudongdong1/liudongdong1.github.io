# Annotation_basic


> - java注解是JDK1.5引入的一种注释机制，java语言的类、方法、变量、参数和包都可以被注解标注。和Javadoc不同，java注解可以通过反射获取标注内容
> - 在编译器生成.class文件时，注解可以被嵌入字节码中，而jvm也可以保留注解的内容，在运行时获取注解标注的内容信息
> - 注解信息怎么和代码关联在一起，java所有事物都是类，注解也不例外，加入代码`System.setProperty("sum.misc.ProxyGenerator.saveGeneratedFiles","true");` 可生成注解相应的代理类，然后和被注释的代码（类，方法，属性等）关联起来

### 1. 语法

#### .1.  声明注解

> Annotation是所有注解类的共同接口，不用显示实现。注解类使用@interface定义（代表它实现Annotation接口），搭配元注解使用，如下：

```java
package java.lang.annotation;
public interface Annotation {
    boolean equals(Object obj);
    int hashCode();
    String toString();
    // 返回定义的注解类型，你在代码声明的@XXX,相当于该类型的一实例
    Class<? extends Annotation> annotationType();
}
-----自定义示例-----
@Retention( value = RetentionPolicy.RUNTIME)
@Target(value = ElementType.TYPE)
public @interface ATest {
    String hello() default  "siting";
}
```

> 使用了`@interface`声明了Test注解，并使用`@Target`注解传入`ElementType.METHOD`参数来标明@Test只能用于方法上，`@Retention(RetentionPolicy.RUNTIME)`则用来表示该注解生存期是运行时

```java
//声明Test注解
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Test {

} 
```

> @Retention用来`约束注解的生命周期`，分别有三个值，源码级别（source），类文件级别（class）或者运行时级别（runtime），其含有如下：
>
> - SOURCE：注解将`被编译器丢弃`（该类型的注解信息只会保留在源码里，源码经过编译后，注解信息会被丢弃，不会保留在编译好的class文件里）
>
>
> - CLASS：`注解在class文件中可用，但会被VM丢弃`（该类型的注解信息会保留在源码里和class文件里，在执行的时候，不会加载到虚拟机中），请注意，当注解未定义Retention值时，默认值是CLASS，如Java内置注解，@Override、@Deprecated、@SuppressWarnning等
>
>
> - RUNTIME：注解信息将`在运行期(JVM)也保留`，因此可以通过反射机制读取注解的信息（源码、class文件和执行的时候都有注解的信息），如SpringMvc中的`@Controller、@Autowired、@RequestMapping等`。

> @Target 用来`约束注解可以应用的地方（如方法、类或字段）`，其中ElementType是枚举类型，其定义如下，也代表可能的取值范围，当注解未指定Target值时，则此注解可以用于任何元素之上，多个值使用{}包含并用逗号隔开，@Target(value={CONSTRUCTOR, FIELD, LOCAL_VARIABLE, METHOD, PACKAGE, PARAMETER, TYPE})

```java
public enum ElementType {
    /**标明该注解可以用于类、接口（包括注解类型）或enum声明*/
    TYPE,

    /** 标明该注解可以用于字段(域)声明，包括enum实例 */
    FIELD,

    /** 标明该注解可以用于方法声明 */
    METHOD,

    /** 标明该注解可以用于参数声明 */
    PARAMETER,

    /** 标明注解可以用于构造函数声明 */
    CONSTRUCTOR,

    /** 标明注解可以用于局部变量声明 */
    LOCAL_VARIABLE,

    /** 标明注解可以用于注解声明(应用于另一个注解上)*/
    ANNOTATION_TYPE,

    /** 标明注解可以用于包声明 */
    PACKAGE,

    /**
     * 标明注解可以用于类型参数声明（1.8新加入）
     * @since 1.8
     */
    TYPE_PARAMETER,

    /**
     * 类型使用声明（1.8新加入)
     * @since 1.8
     */
    TYPE_USE
}
```

#### .2. 注解元素及其数据类型

```java
@Target(ElementType.TYPE)//只能应用于类上
@Retention(RetentionPolicy.RUNTIME)//保存到运行时
public @interface DBTable {
    String name() default "";
}
```

> 声明注解元素时可以使用基本类型但不允许使用任何包装类型，同时还应该注意到注解也可以作为元素的类型，也就是嵌套注解. 其中基本类型包括：所有基本类型（int, float, boolean, double, byte, char, long, short), String, Class, enum, Annotation, 以及上述类型数组。

```java
package com.zejian.annotationdemo;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface Reference{
    boolean next() default false;
}

public @interface AnnotationElementDemo {
    //枚举类型
    enum Status {FIXED,NORMAL};

    //声明枚举
    Status status() default Status.FIXED;

    //布尔类型
    boolean showSupport() default false;

    //String类型
    String name()default "";

    //class类型
    Class<?> testCase() default Void.class;

    //注解嵌套
    Reference reference() default @Reference(next=true);

    //数组类型
    long[] value();
}
```

> 注解是`不支持继承的`，因此不能使用关键字extends来继承某个@interface，但注解在编译后，编译器会自动继承java.lang.annotation.Annotation接口

```java
package com.zejian.annotationdemo;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

//定义注解
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
@interface IntegerVaule{
    int value() default 0;
    String name() default "";
}

//使用注解
public class QuicklyWay {

    //当只想给value赋值时,可以使用以下快捷方式
    @IntegerVaule(20)
    public int age;

    //当name也需要赋值时必须采用key=value的方式赋值
    @IntegerVaule(value = 10000,name = "MONEY")
    public int money;

}
```

#### .3. java 内置注解

- @Override：用于标明此方法覆盖了父类的方法，源码如下

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.SOURCE)
public @interface Override {
}
```

- @Deprecated：用于标明已经过时的方法或类，源码如下

```java
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(value={CONSTRUCTOR, FIELD, LOCAL_VARIABLE, METHOD, PACKAGE, PARAMETER, TYPE})
public @interface Deprecated {
}
```

- @SuppressWarnnings:用于有选择的关闭编译器对类、方法、成员变量、变量初始化的警告，其实现源码如下

```java
@Target({TYPE, FIELD, METHOD, PARAMETER, CONSTRUCTOR, LOCAL_VARIABLE})
@Retention(RetentionPolicy.SOURCE)
public @interface SuppressWarnings {
    String[] value();
}
```

- @Documented 被修饰的注解会生成到javadoc中

```java
@Documented
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface DocumentA {
}

//没有使用@Documented
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface DocumentB {
}

//使用注解
@DocumentA
@DocumentB
public class DocumentDemo {
    public void A(){
    }
}
```

- @Inherited 可以让注解被继承，但这并不是真的继承，`只是通过使用@Inherited，可以让子类Class对象使用getAnnotations()获取父类被@Inherited修饰的注解`，如下：

```java
@Inherited
@Documented
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface DocumentA {
}

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface DocumentB {
}

@DocumentA
class A{ }

class B extends A{ }

@DocumentB
class C{ }

class D extends C{ }

//测试
public class DocumentDemo {

    public static void main(String... args){
        A instanceA=new B();
        System.out.println("已使用的@Inherited注解:"+Arrays.toString(instanceA.getClass().getAnnotations()));

        C instanceC = new D();

        System.out.println("没有使用的@Inherited注解:"+Arrays.toString(instanceC.getClass().getAnnotations()));
    }

    /**
     * 运行结果:
     已使用的@Inherited注解:[@com.zejian.annotationdemo.DocumentA()]
     没有使用的@Inherited注解:[]
     */
}
```

### 2. 注解反射机制

> `Java所有注解都继承了Annotation接口`，也就是说　Java使用Annotation接口代表注解元素，该接口是所有Annotation类型的父接口。同时为了运行时能准确获取到注解的相关信息，`Java在java.lang.reflect 反射包下新增了AnnotatedElement接口`，它主要用于`表示目前正在 VM 中运行的程序中已使用注解的元素`，通过该接口提供的方法可以利用反射技术地读取注解的信息，如反射包的Constructor类、Field类、Method类、Package类和Class类都实现了AnnotatedElement接口

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210627162038525.png)

```java
package com.zejian.annotationdemo;

import java.lang.annotation.Annotation;
import java.util.Arrays;

/**
 * Created by zejian on 2017/5/20.
 * Blog : http://blog.csdn.net/javazejian [原文地址,请尊重原创]
 */
@DocumentA
class A{ }

//继承了A类
@DocumentB
public class DocumentDemo extends A{

    public static void main(String... args){

        Class<?> clazz = DocumentDemo.class;
        //根据指定注解类型获取该注解
        DocumentA documentA=clazz.getAnnotation(DocumentA.class);
        System.out.println("A:"+documentA);

        //获取该元素上的所有注解，包含从父类继承
        Annotation[] an= clazz.getAnnotations();
        System.out.println("an:"+ Arrays.toString(an));
        //获取该元素上的所有注解，但不包含继承！
        Annotation[] an2=clazz.getDeclaredAnnotations();
        System.out.println("an2:"+ Arrays.toString(an2));

        //判断注解DocumentA是否在该元素上
        boolean b=clazz.isAnnotationPresent(DocumentA.class);
        System.out.println("b:"+b);

        /**
         * 执行结果:
         A:@com.zejian.annotationdemo.DocumentA()
         an:[@com.zejian.annotationdemo.DocumentA(), @com.zejian.annotationdemo.DocumentB()]
         an2:@com.zejian.annotationdemo.DocumentB()
         b:true
         */
    }
}
```

### 3. 运行时注解处理器

```java
@Target(ElementType.TYPE)//只能应用于类上
@Retention(RetentionPolicy.RUNTIME)//保存到运行时
public @interface DBTable {
    String name() default "";
}

@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface SQLInteger {
    //该字段对应数据库表列名
    String name() default "";
    //嵌套注解
    Constraints constraint() default @Constraints;
}

@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface SQLString {

    //对应数据库表的列名
    String name() default "";

    //列类型分配的长度，如varchar(30)的30
    int value() default 0;

    Constraints constraint() default @Constraints;
}

@Target(ElementType.FIELD)//只能应用在字段上
@Retention(RetentionPolicy.RUNTIME)
public @interface Constraints {
    //判断是否作为主键约束
    boolean primaryKey() default false;
    //判断是否允许为null
    boolean allowNull() default false;
    //判断是否唯一
    boolean unique() default false;
}

/**
 * Created by wuzejian on 2017/5/18.
 * 数据库表Member对应实例类bean
 */
@DBTable(name = "MEMBER")
public class Member {
    //主键ID
    @SQLString(name = "ID",value = 50, constraint = @Constraints(primaryKey = true))
    private String id;

    @SQLString(name = "NAME" , value = 30)
    private String name;

    @SQLInteger(name = "AGE")
    private int age;

    @SQLString(name = "DESCRIPTION" ,value = 150 , constraint = @Constraints(allowNull = true))
    private String description;//个人描述

   //省略set get.....
}
```

```java
package com.zejian.annotationdemo;
import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

public class TableCreator {

  public static String createTableSql(String className) throws ClassNotFoundException {
    Class<?> cl = Class.forName(className);
    DBTable dbTable = cl.getAnnotation(DBTable.class);
    //如果没有表注解，直接返回
    if(dbTable == null) {
      System.out.println(
              "No DBTable annotations in class " + className);
      return null;
    }
    String tableName = dbTable.name();
    // If the name is empty, use the Class name:
    if(tableName.length() < 1)
      tableName = cl.getName().toUpperCase();
    List<String> columnDefs = new ArrayList<String>();
    //通过Class类API获取到所有成员字段
    for(Field field : cl.getDeclaredFields()) {
      String columnName = null;
      //获取字段上的注解
      Annotation[] anns = field.getDeclaredAnnotations();
      if(anns.length < 1)
        continue; // Not a db table column

      //判断注解类型
      if(anns[0] instanceof SQLInteger) {
        SQLInteger sInt = (SQLInteger) anns[0];
        //获取字段对应列名称，如果没有就是使用字段名称替代
        if(sInt.name().length() < 1)
          columnName = field.getName().toUpperCase();
        else
          columnName = sInt.name();
        //构建语句
        columnDefs.add(columnName + " INT" +
                getConstraints(sInt.constraint()));
      }
      //判断String类型
      if(anns[0] instanceof SQLString) {
        SQLString sString = (SQLString) anns[0];
        // Use field name if name not specified.
        if(sString.name().length() < 1)
          columnName = field.getName().toUpperCase();
        else
          columnName = sString.name();
        columnDefs.add(columnName + " VARCHAR(" +
                sString.value() + ")" +
                getConstraints(sString.constraint()));
      }


    }
    //数据库表构建语句
    StringBuilder createCommand = new StringBuilder(
            "CREATE TABLE " + tableName + "(");
    for(String columnDef : columnDefs)
      createCommand.append("\n    " + columnDef + ",");

    // Remove trailing comma
    String tableCreate = createCommand.substring(
            0, createCommand.length() - 1) + ");";
    return tableCreate;
  }


    /**
     * 判断该字段是否有其他约束
     * @param con
     * @return
     */
  private static String getConstraints(Constraints con) {
    String constraints = "";
    if(!con.allowNull())
      constraints += " NOT NULL";
    if(con.primaryKey())
      constraints += " PRIMARY KEY";
    if(con.unique())
      constraints += " UNIQUE";
    return constraints;
  }

  public static void main(String[] args) throws Exception {
    String[] arg={"com.zejian.annotationdemo.Member"};
    for(String className : arg) {
      System.out.println("Table Creation SQL for " +
              className + " is :\n" + createTableSql(className));
    }

    /**
     * 输出结果：
     Table Creation SQL for com.zejian.annotationdemo.Member is :
     CREATE TABLE MEMBER(
     ID VARCHAR(50) NOT NULL PRIMARY KEY,
     NAME VARCHAR(30) NOT NULL,
     AGE INT NOT NULL,
     DESCRIPTION VARCHAR(150)
     );
     */
  }
}
```

### 4. Spring AOP和注解机制

- C是面向过程编程的，java则是面向对象编程，C++则是两者兼备，它们都是一种规范和思想。面向切面编程也一样，可以简单理解为：**切面**编程专注的是局部代码，主要为某些点植入增强代码
- 考虑要局部加入增强代码，使用动态代理则是最好的实现。在被代理方法调用的前后，可以加入需要的增强功能；因此spring的切面编程是基于动态代理的

| 概念                | 描述                                                         |
| ------------------- | ------------------------------------------------------------ |
| 通知（Advice）      | 需要切入的增加代码逻辑被称为通知                             |
| 切点（Pointcut）    | 定义增强代码在何处执行                                       |
| 切面（Aspect）      | 切面是通知和切点的集合                                       |
| 连接点（JoinPoint） | 在切点基础上，指定增强代码在切点执行的时机(在切点前，切点后，抛出异常后等) |
| 目标（target）      | 被增强目标类                                                 |

| 切面编程相关注解 | 功能描述                                                     |
| ---------------- | ------------------------------------------------------------ |
| @Aspect          | 作用于类，声明当前方法类是增强代码的切面类                   |
| @Pointcut        | 作用于方法，指定需要被拦截的其他方法。当前方法则作为拦截集合名使用 |

| spring通知（Advice）注解 | 功能描述                                         |
| ------------------------ | ------------------------------------------------ |
| @After                   | 增强代码在@Pointcut指定的方法之后执行            |
| @Before                  | 增强代码在@Pointcut指定的方法之前执行            |
| @AfterReturning          | 增强代码在@Pointcut指定的方法 return返回之后执行 |
| @Around                  | 增强代码可以在被拦截方法前后执行                 |
| @AfterThrowing           | 增强代码在@Pointcut指定的方法抛出异常之后执行    |

- 在spring切面基础上，开发具有增强功能的自定义注解 **(对注解进行切面)**

```java
新建spring-web + aop 项目；新建如下class
------ 目标Controller ------
@RestController
public class TestController {
    @STAnnotation
    @RequestMapping("/hello")
    public String hello(){  return "hello@csc";  }
}
------ Controller注解 -------
@Retention( value = RetentionPolicy.RUNTIME)
@Target(value = ElementType.METHOD)
public @interface STAnnotation {
    String value() default "注解hello!";
}
------ Controller切面 ------
@Aspect
@Component
public class ControllerAspect {
    //切点:注解指定关联 (对注解进行切面)
    @Pointcut("@annotation(STAnnotation)")
    public void controllerX(){}
    //切点:路径指定关联
    @Pointcut("execution(public * com.example.demo.TestController.*(..))")
    public void controllerY(){}
    //在controllerY()切点执行之前的连接点加入通知
    @Before("controllerY()")
    public void yBefore(JoinPoint joinPoint) throws Throwable {
        //可以加入增强代码
        MethodSignature methodS = (MethodSignature)joinPoint.getSignature();
        Method method = methodS.getMethod();
        if (method.isAnnotationPresent(STAnnotation.class)) {
            STAnnotation annotation = method.getAnnotation(STAnnotation.class);
            System.out.println(annotation.value());
        }
        System.out.println("controllerY");
    }
    //controllerX()切点执行之后的连接点加入通知
    @After("controllerX()")
    public void xBefore(JoinPoint joinPoint) throws Throwable {
        //可以加入增强代码
        System.out.println("controllerX");
    }
}
```

### 5. Resource

- https://blog.csdn.net/javazejian/article/details/71860633
- https://segmentfault.com/a/1190000027073489

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/annotation_basic/  

