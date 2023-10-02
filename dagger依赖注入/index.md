# Dagger2依赖注入


> 依赖注入可以实现解耦，达到高内聚低耦合的目的，保证代码的健壮性、灵活性和可维护性。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1240.png)

### 1. 依赖注入方式

**1、构造注入：通过构造函数传参给依赖的成员变量赋值，从而实现注入。**

```java
public class Car{

    private Engine engine;

    public Car(Engine engine){
        this.engine = engine;
    }
}
```

**2、接口注入：实现接口方法，同样以传参的方式实现注入。**

```java
public interface Injection<T>{

    void inject(T t);
}

public class Car implements Injection<Engine>{

    private Engine engine;

    public Car(){}

    public void inject(Engine engine){
        this.engine = engine;
    }

}
```

**3、注解注入：使用Java注解在编译阶段生成代码实现注入或者是在运行阶段通过反射实现注入。**

```java
public class Car{

    @Inject
    Engine engine;

    public Car(){}
}
```

### 2. Dagger2 注解

- https://juejin.cn/post/7015528907516411911#heading-0

#### @Inject

- 一是用来标记需要依赖的变量，以此告诉Dagger2为它提供依赖；

- 二是用来标记构造函数，Dagger2通过@Inject注解可以在需要这个类实例的时候来找到这个构造函数并把相关实例构造出来，以此来为被@Inject标记了的变量提供依赖；

```java
public final class A_MembersInjector implements MembersInjector<A> {
  private final Provider<B> bProvider;

  public A_MembersInjector(Provider<B> bProvider) {
    this.bProvider = bProvider;
  }

  public static MembersInjector<A> create(Provider<B> bProvider) {
    return new A_MembersInjector(bProvider);}

  @Override
  public void injectMembers(A instance) {
    injectB(instance, bProvider.get());
  }

  public static void injectB(A instance, B b) {
    instance.b = b;
  }
}
```

#### @Module

- 提供依赖的构造函数是第三方库的，我们没法给它加上@Inject注解；
- 需要被注入的依赖类提供的`构造函数是带参数的`

#### @Provides

- 用于标注Module所标注的类中的方法，该方法在需要提供依赖时被调用，从而把预先提供好的对象当做依赖给标注了@Inject的变量赋值；

#### @Component

- 用于标注接口，是依赖需求方和依赖提供方之间的桥梁。被Component标注的接口在编译时会生成该接口的实现类（如果@Component标注的接口为CarComponent，则编译期生成的实现类为DaggerCarComponent）

```java
public final class DaggerAComponent implements AComponent {
  private DaggerAComponent() {

  }

  public static Builder builder() {
    return new Builder();
  }

  public static AComponent create() {
    return new Builder().build();
  }

  @Override
  public void injectA(A a) {
    injectA2(a);
  }

  private A injectA2(A instance) {
    A_MembersInjector.injectB(instance, new B());
    return instance;
  }

  public static final class Builder {
    private Builder() {
    }

    public AComponent build() {
      return new DaggerAComponent();
    }
  }
}
```

#### @Qulifier

- 用于自定义注解，也就是说@Qulifier就如同Java提供的几种基本元注解一样用来标记注解类。
- 当类型不足以鉴别一个依赖的时候，我们就可以使用这个注解标示；

#### @Scope

- 自定义的注解来限定注解作用域，实现单例（分局部和全局）；
- **@Scope需要Component和依赖提供者配合才能起作用**，对于@Scope注解的依赖，Component会持有第一次创建的依赖，后面注入时都会复用这个依赖的实例，实质上@Scope的目的就是为了让生成的依赖实例的生命周期与 Component 绑定
- **如果Component重建了，持有的@Scope的依赖也会重建，所以为了维护局部单例需要自己维护Component的生命周期。**

#### @Singleton

- @Singleton其实就是一个通过@Scope定义的注解，我们一般通过它来实现全局单例。但实际上它并不能提供全局单例，是否能提供全局单例还要取决于对应的Component是否为一个全局对象。

### 3. Dagger2 案例

#### 1. Inject 简单注解

如一个User类：

```java
public class User {
    public String name;
    //用这个@Inject表示来表示我可以提供User类型的依赖
    @Inject
    public User() {
        name = "sososeen09";
    }
    public String getName() {
        return name;
    }
}
```

在需要依赖的的目标类中标记成员变量，在这里我们这个目标类是OnlyInjectTestActivity。

```java
@Inject //在目标类中@Inject标记表示我需要这个类型的依赖      
User mUser;
```

#### 2. Module模式

```java
public class Person {
    private String sex;

    public Person(String sex) {
        this.sex = sex;
    }

    public Person() {
        sex = "太监";
    }

    public String getSex() {
        return sex;
    }
}
```

我们用Module提供Person实例，Module代码如下：

```java
@Module
public class DataModule {

    @Provides
    Person providePerson() {
        return new Person();
    }
```

上面的代码也算是一个固定套路了，用@Module标记类，用@Provides标记方法。如果想用Module提供实例，还要有一个Component，如我们下面的PersonComponent 。这个PersonComponent 与纯粹用@Inject方式提供依赖不同，还需要有一个modules指向DataModule 。这是告诉Component我们用DataModule 提供你想要的类型的实例。其它的方式相同。

```java
@Component(modules = DataModule.class)
public interface PersonComponent {
    void inject(ModuleTestActivity moduleTestActivity);
}
```

ModuleTestActivity 中需要一个Person类型的依赖：

```java
@Inject
Person mPerson;
```

编译之后，我们就可以在目标类ModuleTestActivity 中进行初始化注入了。

```java
DaggerPersonComponent.builder().dataModule(new DataModule()).build().inject(this);
```

### 4. 初始化依赖顺序

步骤如下：

1. 查找Module中是否存在创建该类型的方法（前提是@Conponent标记的接口中包含了@Module标记的Module类，如果没有则直接找@Inject对应的构造方法）
2. 若存在方法，查看该方法是否有参数
   - 若不存在参数，直接初始化该类的实例，一次依赖注入到此结束。
   - 若存在参数，则从**步骤1**开始初始化每个参数
3. 若不存在创建类方法，则查找该类型的类中有@Inject标记的构造方法，查看构造方法中是否有参数
   - 若构造方法中无参数，则直接初始化该类实例，一次依赖注入到此结束。
   - 若构造方法中有参数，从**步骤1**依次开始初始化每个参数。

### 5. Component组织

- 依赖方式——一个Component可以依赖一个或多个Component，采用的是@Component的**dependencies**属性。
- 包含方式——这里就用到了我们**@SubComponent**注解，用@SubComponent标记接口或者抽象类，表示它可以被包含。一个Component可以包含一个或多个Component，而且被包含的Component还可以继续包含其他的Component。说起来跟Activity包含Fragment方式很像。
- 继承方式——用一个Component继承另外一个Component。

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/1240-16711803693944.png)

- https://toutiao.io/posts/x5ekcg/preview

### Resource

- https://juejin.cn/post/7015528907516411911#heading-0
- https://blog.csdn.net/xiaowu_zhu/article/details/93238725
- https://toutiao.io/posts/x5ekcg/preview  

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/dagger%E4%BE%9D%E8%B5%96%E6%B3%A8%E5%85%A5/  

