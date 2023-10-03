# springbootAOP


> `OOP引入封装、继承、多态等概念`来建立一种对象层次结构，用于模拟公共行为的一个集合。不过OOP只允许开发者定义纵向的关系，但`并不适合定义横向的关系`，例如`日志，事务，安全等`。这些功能都是横向应用在业务处理中，而与它们对应的方法与其他代码基本没有联系，如异常处理和透明的持续性也都是如此，不仅增加了大量的代码量，还为程序后期的维护增生很多困难。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210628092953842.png)

### 1. 基本概念

1. 切面（Aspect）：一个关注点的模块化，这个关注点可能会横切多个对象。事务管理是J2EE应用中一个关于横切关注点的很好的例子。在spring AOP中，切面可以使用`基于模式）`或者`基于Aspect注解`方式来实现。通俗点说就是我们加入的`切面类（比如log类）`，可以这么理解。

2. 连接点（Joinpoint）：在程序执行过程中某个特定的点，比如某方法调用的时候或者处理异常的时候。在Spring AOP中，一个`连接点总是表示一个方法的执行`。通俗的说就是加入切点的那个点

3. 通知（Advice）：在切面的某个特定的连接点上执行的动作。其中包括了`“around”、“before”和“after”等不同类型的通知`（通知的类型将在后面部分进行讨论）。许多AOP框架（包括Spring）都是以拦截器做通知模型，并维护一个以连接点为中心的拦截器链。

   1. 前置通知（Before advice）：`在某连接点之前执行的通知`，但这个通知`不能阻止连接点之前的执行流程`（除非它抛出一个异常）。
   2. 后置通知（After returning advice）：在某连接点正常完成后执行的通知：例如，一个方法没有抛出任何异常，正常返回。

   3. 异常通知（After throwing advice）：在`方法抛出异常退出时执行的通知`。

   4. 最终通知（After (finally) advice）`：当某连接点退出的时候执行的通知`（不论是正常返回还是异常退出）。

   5. 环绕通知（Around Advice）：包围一个连接点的通知，如方法调用。这是最强大的一种通知类型。环绕通知可以在方法调用前后完成自定义的行为。

4. 切入点（Pointcut）：匹配连接点的断言。通知和一个切入点表达式关联，并在满足这个切入点的连接点上运行（例如，当执行某个特定名称的方法时）。`切入点表达式如何和连接点匹配是AOP的核心：Spring缺省使用AspectJ切入点语法。`

5. 引入（Introduction）：用来给一个类型声明额外的方法或属性（也被称为连接类型声明（inter-type declaration））。Spring允许引入新的接口（以及一个对应的实现）到任何被代理的对象。例如，你可以使用引入来使一个bean实现IsModified接口，以便简化缓存机制。

6. 目标对象（Target Object）：` 被一个或者多个切面所通知的对象`。也被称做被通知（advised）对象。 既然Spring AOP是通过运行时代理实现的，这个对象永远是一个被代理（proxied）对象。

7. AOP代理（AOP Proxy）：AOP框架创建的对象，用来实现切面契约（例如通知方法执行等等）。在Spring中，`AOP代理可以是JDK动态代理或者CGLIB代理。`

8. 织入（Weaving）：`把切面连接到其它的应用程序类型或者对象上，并创建一个被通知的对象`。这些可以在编译时（例如使用AspectJ编译器），类加载时和运行时完成。Spring和其他纯Java AOP框架一样，在运行时完成织入。

### 2. Example

- 切面类

```java
package com.example.demo.aop;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class LogAspect {

  //这个方法定义了切入点
  @Pointcut("@annotation(com.example.demo.aop.anno.MyLog)")
  public void pointCut() {}

  //这个方法定义了具体的通知
  @After("pointCut()")
  public void recordRequestParam(JoinPoint joinPoint) {
    for (Object s : joinPoint.getArgs()) {
      //打印所有参数，实际中就是记录日志了
      System.out.println("after advice : " + s);
    }
  }

  //这个方法定义了具体的通知
  @Before("pointCut()")
  public void startRecord(JoinPoint joinPoint) {
    for (Object s : joinPoint.getArgs()) {
      //打印所有参数
      System.out.println("before advice : " + s);
    }
  }

  //这个方法定义了具体的通知
  @Around("pointCut()")
  public Object aroundRecord(ProceedingJoinPoint pjp) throws Throwable {
    for (Object s : pjp.getArgs()) {
      //打印所有参数
      System.out.println("around advice : " + s);
    }
    return pjp.proceed();
  }
}
```

- 注解

```java
package com.example.demo.aop.anno;
import java.lang.annotation.*;

@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.TYPE})
public @interface MyLog {
}
```

- 目标类

```java
package com.example.demo.aop.target;

import com.example.demo.aop.anno.MyLog;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MockController {

  @RequestMapping("/hello")
  @MyLog
  public String helloAop(@RequestParam String key) {
    System.out.println("do something...");
    return "hello world";
  }

}
```

- 测试类

```java
package com.example.demo.aop.target;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
/**
 * @author Fcb
 * @date 2020/6/20
 * @description
 */
@SpringBootTest
class MockControllerTest {
  @Autowired
  MockController mockController;

  @Test
  void helloAop() {
    mockController.helloAop("aop");
  }
}
```

### 3. 表达式

1. execution：一般`用于指定方法的执行`，用的最多。
2. within：`指定某些类型的全部方法执行`，也可用来指定一个包。
3. this：Spring Aop是基于动态代理的，生成的bean也是一个代理对象，this就是这个`代理对象`，当这个对象可以转换为指定的类型时，对应的切入点就是它了，Spring Aop将生效。
4. target：当`被代理的对象可以转换为指定的类型时`，对应的切入点就是它了，Spring Aop将生效。
5. args：当执行的方法的`参数是指定类型时生效`。
6. @target：`当代理的目标对象上拥有指定的注解时生效`。
7. @args：当`执行的方法参数类型上拥有指定的注解时生效`。
8. @within：与@target类似，看官方文档和网上的说法都是@within只需要目标对象的类或者父类上有指定的注解，则@within会生效，而@target则是必须是目标对象的类上有指定的注解。而根据笔者的测试这两者都是只要目标类或父类上有指定的注解即可。
9. @annotation：当`执行的方法上拥有指定的注解时生效`。
10. `reference pointcut`：(经常使用)表示引用其他命名切入点，只有@ApectJ风格支持，Schema风格不支持
11. `bean：当调用的方法是指定的bean的方法时生效。(Spring AOP自己扩展支持的)`

#### execution：

execution是使用的最多的一种Pointcut表达式，表示某个方法的执行，其标准语法如下。

```javascript
execution(modifiers-pattern? ret-type-pattern declaring-type-pattern? name-pattern(param-pattern) throws-pattern?)
```

- 修饰符匹配（modifier-pattern?）
- 返回值匹配（ret-type-pattern）可以为*表示任何返回值,全路径的类名等
- 类路径匹配（declaring-type-pattern?）
- 方法名匹配（name-pattern）可以指定方法名 或者 *代表所有, set* 代表以set开头的所有方法
- 参数匹配（(param-pattern)）可以指定具体的参数类型，多个参数间用“,”隔开，各个参数也可以用“*”来表示匹配任意类型的参数，如(String)表示匹配一个String参数的方法；(*,String) 表示匹配有两个参数的方法，第一个参数可以是任意类型，而第二个参数是String类型；可以用(…)表示零个或多个任意参数
- 异常类型匹配（throws-pattern?）
- 其中后面跟着“?”的是可选项

```javascript
//表示匹配所有方法  
1）execution(* *(..))  
//表示匹配com.fsx.run.UserService中所有的公有方法  
2）execution(public * com.fsx.run.UserService.*(..))  
//表示匹配com.fsx.run包及其子包下的所有方法
3）execution(* com.fsx.run..*.*(..))  
```

Pointcut定义时，还可以使用&&、||、! 这三个运算。进行逻辑运算

```java
// 签名：消息发送切面
@Pointcut("execution(* com.fsx.run.MessageSender.*(..))")
private void logSender(){}
// 签名：消息接收切面
@Pointcut("execution(* com.fsx.run.MessageReceiver.*(..))")
private void logReceiver(){}
// 只有满足发送  或者  接收  这个切面都会切进去
@Pointcut("logSender() || logReceiver()")
private void logMessage(){}
```

这个例子中，logMessage()将匹配任何MessageSender和MessageReceiver中的任何方法。

当我们的切面很多的时候，我们可以把所有的切面放到单独的一个类去，进行统一管理，比如下面：

```java
//集中管理所有的切入点表达式
public class Pointcuts {

@Pointcut("execution(* *Message(..))")
public void logMessage(){}

@Pointcut("execution(* *Attachment(..))")
public void logAttachment(){}

@Pointcut("execution(* *Service.*(..))")
public void auth(){}
}
```

这样别的使用时，采用全类名+方法名的方式

```java
@Before("com.fsx.run.Pointcuts.logMessage()")
public void before(JoinPoint joinPoint) {
	System.out.println("Logging before " + 		joinPoint.getSignature().getName());
}
```

#### within：

within是用来指定类型的，指定类型中的所有方法将被拦截。

```java
// AService下面所有外部调用方法，都会拦截。备注：只能是AService的方法，子类不会拦截的
@Pointcut("within(com.fsx.run.service.AService)")
	public void pointCut() {
}
```

所以此处需要注意：上面写的是AService接口，是达不到拦截效果的，只能写实现类：

```java
    //此处只能写实现类
@Pointcut("within(com.fsx.run.service.impl.AServiceImpl)")
	public void pointCut() {
}
```

匹配包以及子包内的所有类：

```java
@Pointcut("within(com.fsx.run.service..*)")
	public void pointCut() {
}
```

#### this：

Spring Aop是基于代理的，`this就表示代理对象`。this类型的Pointcut表达式的语法是this(type)，当生成的代理对象可以转换为type指定的类型时则表示匹配。基于JDK接口的代理和基于CGLIB的代理生成的代理对象是不一样的。（注意和上面within的区别）

```java
// 这样子，就可以拦截到AService所有的子类的所有外部调用方法
@Pointcut("this(com.fsx.run.service.AService*)")
     public void pointCut() {
}
```

#### target：

Spring Aop是基于代理的，target则表示被代理的目标对象。当被代理的目标对象可以被转换为指定的类型时则表示匹配。 注意：和上面不一样，这里是target，因此如果要切入，只能写实现类了

```javascript
@Pointcut("target(com.fsx.run.service.impl.AServiceImpl)")
	public void pointCut() {
}
```

#### args：

args用来匹配方法参数的。

1、“args()”匹配任何不带参数的方法。

2、“args(java.lang.String)”匹配任何只带一个参数，而且这个参数的类型是String的方法。

3、“args(…)”带任意参数的方法。

4、“args(java.lang.String,…)”匹配带任意个参数，但是第一个参数的类型是String的方法。

5、“args(…,java.lang.String)”匹配带任意个参数，但是最后一个参数的类型是String的方法。

```javascript
@Pointcut("args()")
	public void pointCut() {
}
```

这个匹配的范围非常广，所以一般和别的表达式结合起来使用

#### @target：

@target匹配当被代理的目标对象对应的类型及其父类型上拥有指定的注解时。

```javascript
//能够切入类上（非方法上）标准了MyAnno注解的所有外部调用方法
@Pointcut("@target(com.fsx.run.anno.MyAnno)")
	public void pointCut() {
}
```

#### @args：

@args匹配被调用的方法上含有参数，且对应的参数类型上拥有指定的注解的情况。 例如：

```javascript
// 匹配**方法参数类型上**拥有MyAnno注解的方法调用。如我们有一个方法add(MyParam param)接收一个MyParam类型的参数，而MyParam这个类是拥有注解MyAnno的，则它可以被Pointcut表达式匹配上
@Pointcut("@args(com.fsx.run.anno.MyAnno)")
	public void pointCut() {
}
```

#### @within：

@within用于匹配被代理的目标对象对应的类型或其父类型拥有指定的注解的情况，但只有在调用拥有指定注解的类上的方法时才匹配。

“@within(com.fsx.run.anno.MyAnno)”匹配被调用的方法声明的类上拥有MyAnno注解的情况。比如有一个ClassA上使用了注解MyAnno标注，并且定义了一个方法a()，那么在调用ClassA.a()方法时将匹配该Pointcut；如果有一个ClassB上没有MyAnno注解，但是它继承自ClassA，同时它上面定义了一个方法b()，那么在调用ClassB().b()方法时不会匹配该Pointcut，但是在调用ClassB().a()时将匹配该方法调用，因为a()是定义在父类型ClassA上的，且ClassA上使用了MyAnno注解。但是如果子类ClassB覆写了父类ClassA的a()方法，则调用ClassB.a()方法时也不匹配该Pointcut。

#### @annotation：

@annotation用于匹配**方法上**拥有指定注解的情况。

```javascript
// 可以匹配所有方法上标有此注解的方法
@Pointcut("@annotation(com.fsx.run.anno.MyAnno)")
	public void pointCut() {
}
```

```javascript
@Before("@annotation(myAnno)")
public void doBefore(JoinPoint joinPoint, MyAnno myAnno) {
    System.out.println(myAnno); //@com.fsx.run.anno.MyAnno()
    System.out.println("AOP Before Advice...");
}
```

#### reference pointcut：

```javascript
@Aspect
public class HelloAspect {
    @Pointcut("execution(* com.fsx.service.*.*(..)) ")
    public void point() {
    }
    // 这个就是一个`reference pointcut`  甚至还可以这样 @Before("point1() && point2()")
    @Before("point()")  
    public void before() {
        System.out.println("this is from HelloAspect#before...");
    }
}
```

#### bean： 

bean用于匹配当调用的是指定的Spring的某个bean的方法时。 1、“bean(abc)”匹配Spring Bean容器中id或name为abc的bean的方法调用。 2、“bean(user*)”匹配所有id或name为以user开头的bean的方法调用。

```javascript
// 这个就能切入到AServiceImpl类的所有的外部调用的方法里
@Pointcut("bean(AServiceImpl)")
	public void pointCut() {
}
```

#### 类型匹配语法

*：匹配任何数量字符； …：匹配任何数量字符的重复，如在类型模式中匹配任何数量子包；而在方法参数模式中匹配任何数量参数。 +：匹配指定类型的子类型；仅能作为后缀放在类型模式后边。

```javascript
java.lang.String    匹配String类型； 
java.*.String       匹配java包下的任何“一级子包”下的String类型； 如匹配java.lang.String，但不匹配java.lang.ss.String 
java..*            匹配java包及任何子包下的任何类型。如匹配java.lang.String、java.lang.annotation.Annotation 
java.lang.*ing      匹配任何java.lang包下的以ing结尾的类型；
java.lang.Number+  匹配java.lang包下的任何Number的子类型； 如匹配java.lang.Integer，也匹配java.math.BigInteger 
```

#### Resource

- https://juejin.cn/post/6844903896788271117
- from： https://www.tuicool.com/articles/36jiEnB

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/springaop/  

