# ProxyMode


> [代理模式](https://zh.wikipedia.org/wiki/代理模式)是一种设计模式，提供了对目标对象额外的访问方式，即`通过代理对象访问目标对象`，这样可以在不修改原目标对象的前提下，提供额外的功能操作，扩展目标对象的功能。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221127192433132.png)

### 1. 静态代理

> 需要代理对象和目标对象实现一样的接口。
>
> 优点：可以在不修改目标对象的前提下扩展目标对象的功能。
>
> 缺点：
>
> 1. 冗余。由于代理对象要实现与目标对象一致的接口，`会产生过多的代理类`。
> 2. 不易维护。一旦接口增加方法，目标对象与代理对象都要进行修改。

- 接口类： IUserDao

```java
package com.proxy;

public interface IUserDao {
    public void save();
}
```

- 目标对象：UserDao

```java
package com.proxy;

public class UserDao implements IUserDao{

    @Override
    public void save() {
        System.out.println("保存数据");
    }
}
```

- 静态代理对象：UserDapProxy ***需要实现IUserDao接口！\***

```java
package com.proxy;

public class UserDaoProxy implements IUserDao{

    private IUserDao target;
    public UserDaoProxy(IUserDao target) {
        this.target = target;
    }
    
    @Override
    public void save() {
        System.out.println("开启事务");//扩展了额外功能
        target.save();
        System.out.println("提交事务");
    }
}
```

- 测试类：TestProxy

```java
package com.proxy;

import org.junit.Test;

public class StaticUserProxy {
    @Test
    public void testStaticProxy(){
        //目标对象
        IUserDao target = new UserDao();
        //代理对象
        UserDaoProxy proxy = new UserDaoProxy(target);
        proxy.save();
    }
}
```

### 2. 动态代理

> 动态代理利用了[JDK API](http://tool.oschina.net/uploads/apidocs/jdk-zh/)，`动态地在内存中构建代理对象`，从而实现对目标对象的代理功能。动态代理又被称为`JDK代理或接口代理`。
>
> 静态代理与动态代理的区别主要在：
>
> - 静态代理`在编译时就已经实现`，编译完成后代理类是一个实际的class文件
> - 动态代理是`在运行时动态生成的`，即编译完成后没有实际的class文件，而是在运行时动态生成类字节码，并加载到JVM中
>
> **特点：**
>
> - `动态代理对象不需要实现接口，但是要求目标对象必须实现接口`，否则不能使用动态代理。
>
> - JDK中生成代理对象主要涉及的类有[java.lang.reflect Proxy](http://tool.oschina.net/uploads/apidocs/jdk-zh/java/lang/reflect/Proxy.html)

```java
static Object    newProxyInstance(ClassLoader loader,  //指定当前目标对象使用类加载器

 Class<?>[] interfaces,    //目标对象实现的接口的类型
 InvocationHandler h      //事件处理器
) 
//返回一个指定接口的代理类实例，该接口可以将方法调用指派到指定的调用处理程序。
```

- [java.lang.reflect InvocationHandler](http://tool.oschina.net/uploads/apidocs/jdk-zh/java/lang/reflect/InvocationHandler.html)，主要方法为

```java
 Object    invoke(Object proxy, Method method, Object[] args) 
// 在代理实例上处理方法调用并返回结果。
```

> 举例：保存用户功能的动态代理实现

- 接口类： IUserDao

```java
package com.proxy;

public interface IUserDao {
    public void save();
}
```

- 目标对象：UserDao

```java
package com.proxy;

public class UserDao implements IUserDao{

    @Override
    public void save() {
        System.out.println("保存数据");
    }
}
```

- 动态代理对象：UserProxyFactory

```java
package com.proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

public class ProxyFactory {

    private Object target;// 维护一个目标对象

    public ProxyFactory(Object target) {
        this.target = target;
    }

    // 为目标对象生成代理对象
    public Object getProxyInstance() {
        return Proxy.newProxyInstance(target.getClass().getClassLoader(), target.getClass().getInterfaces(),
                new InvocationHandler() {

                    @Override
                    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                        System.out.println("开启事务");

                        // 执行目标对象方法
                        Object returnValue = method.invoke(target, args);

                        System.out.println("提交事务");
                        return null;
                    }
                });
    }
}
```

> Proxy.newProxyInstance: 参数，`代理的是接口(Interfaces)，不是类(Class)，也不是抽象类`。
>
> - loader: 用哪个类加载器去加载代理对象
> - interfaces:动态代理类需要实现的接口
> - h:动态代理方法在执行时，会调用h里面的invoke方法去执行

```
Object invoke(Object proxy, Method method, Object[] args) throws Throwable

proxy:　　指代我们所代理的那个真实对象
method:　　指代的是我们所要调用真实对象的某个方法的Method对象
args:　　指代的是调用真实对象某个方法时接受的参数
```

- 测试类：TestProxy

```java
package com.proxy;

import org.junit.Test;

public class TestProxy {

    @Test
    public void testDynamicProxy (){
        IUserDao target = new UserDao();
        System.out.println(target.getClass());  //输出目标对象信息
        IUserDao proxy = (IUserDao) new ProxyFactory(target).getProxyInstance();
        System.out.println(proxy.getClass());  //输出代理对象信息
        proxy.save();  //执行代理方法
    }
}
```

### 3. cglib代

> [cglib](https://github.com/cglib/cglib) (Code Generation Library )是一个第三方代码生成类库，运行时在内存中动态生成一个子类对象从而实现对目标对象功能的扩展。
>
> **cglib特点**
>
> - `JDK的动态代理有一个限制，就是使用动态代理的对象必须实现一个或多个接口`。
>   如果想`代理没有实现接口的类`，就可以使用CGLIB实现。
> - CGLIB是一个强大的高性能的代码生成包，它可以在运行期扩展Java类与实现Java接口。
>   它广泛的被许多AOP的框架使用，例如Spring AOP和dynaop，为他们提供方法的interception（拦截）。
> - CGLIB包的底层是通过使用一个小而快的`字节码处理框架ASM`，来转换字节码并生成新的类。
>   不鼓励直接使用ASM，因为它需要你对JVM内部结构包括class文件的格式和指令集都很熟悉。
>
> cglib与动态代理最大的**区别**就是
>
> - 使用动态代理的对象必须实现一个或多个接口
> - 使用cglib代理的对象则无需实现接口，达到代理类无侵入。
>
> 使用cglib需要引入[cglib的jar包](https://repo1.maven.org/maven2/cglib/cglib/3.2.5/cglib-3.2.5.jar)，如果你已经有spring-core的jar包，则无需引入，因为spring中包含了cglib。

- cglib的Maven坐标

```xml
<dependency>
    <groupId>cglib</groupId>
    <artifactId>cglib</artifactId>
    <version>3.2.5</version>
</dependency>
```

> 举例：保存用户功能的动态代理实现

- 目标对象：UserDao

```java
package com.cglib;

public class UserDao{

    public void save() {
        System.out.println("保存数据");
    }
}
```

- 代理对象：ProxyFactory

```java
package com.cglib;

import java.lang.reflect.Method;
import net.sf.cglib.proxy.Enhancer;
import net.sf.cglib.proxy.MethodInterceptor;
import net.sf.cglib.proxy.MethodProxy;

public class ProxyFactory implements MethodInterceptor{

    private Object target;//维护一个目标对象
    public ProxyFactory(Object target) {
        this.target = target;
    }
    
    //为目标对象生成代理对象
    public Object getProxyInstance() {
        //工具类
        Enhancer en = new Enhancer();
        //设置父类
        en.setSuperclass(target.getClass());
        //设置回调函数
        en.setCallback(this);
        //创建子类对象代理
        return en.create();
    }

    @Override
    public Object intercept(Object obj, Method method, Object[] args, MethodProxy proxy) throws Throwable {
        System.out.println("开启事务");
        // 执行目标对象的方法
        Object returnValue = method.invoke(target, args);
        System.out.println("关闭事务");
        return null;
    }
}
```

- 测试类：TestProxy

```java
package com.cglib;

import org.junit.Test;

public class TestProxy {

    @Test
    public void testCglibProxy(){
        //目标对象
        UserDao target = new UserDao();
        System.out.println(target.getClass());
        //代理对象
        UserDao proxy = (UserDao) new ProxyFactory(target).getProxyInstance();
        System.out.println(proxy.getClass());
        //执行代理对象方法
        proxy.save();
    }
}
```

### Resource

- https://blog.csdn.net/carson_ho/article/details/54910472

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/proxymode/  

