# EffectJava


> 不要努力写快的程序，要努力写好程序；速度自然会提高。但是在设计系统时一定要考虑性能，特别是在设计API、线路层协议和持久数据格式时。当你完成了系统的构建之后，请度量它的性能。如果足够快，就完成了。如果没有，利用分析器找到问题的根源，并对系统的相关部分进行优化。第一步是检查算法的选择：再多的底层优化也不能弥补算法选择的不足。

### 1. 对象销毁创建

> 何时以及如何创建对象，何时以及如何避免创建对象，如何确保它们能够适时的销毁。

#### .1. 使用静态工厂代替构造器

> 随时随地的 new 是有很大风险的，除了可能导致性能、内存方面的问题外，也经常会使得代码结构变得混乱。

- 有名称：一个类只能有一个带指定签名的构造器，（通过提供多个构造器，参数列表不一样），不建议使用；
- 不必每次调用它们都创建一个新的对象；
- 可以返回原返回类型的子类
- 在创建带泛型的实例时，能使代码变得简洁
- 可以有多个参数相同但名称不同的工厂方法

```java
class Child{
    int age = 10;
    int weight = 30;
    public static Child newChild(int age, int weight) {
        Child child = new Child();
        child.weight = weight;
        child.age = age;
        return child;
    }
    public static Child newChildWithWeight(int weight) {
        Child child = new Child();
        child.weight = weight;
        return child;
    }
    public static Child newChildWithAge(int age) {
        Child child = new Child();
        child.age = age;
        return child;
    }
}
```

- 减少对外暴露

```java

// Player : Version 2
class Player {
    public static final int TYPE_RUNNER = 1;
    public static final int TYPE_SWIMMER = 2;
    public static final int TYPE_RACER = 3;
    int type;

    private Player(int type) {
        this.type = type;
    }

    public static Player newRunner() {
        return new Player(TYPE_RUNNER);
    }
    public static Player newSwimmer() {
        return new Player(TYPE_SWIMMER);
    }
    public static Player newRacer() {
        return new Player(TYPE_RACER);
    }
}
```

- 多了一层控制，方便统一修改

```java
static class User{
    String name ;
    int age ;
    String description;
    public static User newTestInstance() {
        User tester = new User();
        tester.setName("隔壁老张");
        tester.setAge(16);
        tester.setDescription("我住隔壁我姓张！");
        return tester;
    }
}
```

#### .2. 遇到多个构造器参数时用构造器

##### .2.1. 重叠构造器

> - 重叠构造器模式可行，但是当有`许多参数的时候，客户端代码会很难编写，并且仍然较难以阅读`。

```java
public class Student {
    /*必填参数*/
    private String name;
    private int age;
    /*可选参数*/
    private String sex;
    private String grade;

    public Student(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public Student(String name, int age, String sex) {
        this.name = name;
        this.age = age;
        this.sex = sex;
    }

    public Student(String name, int age, String sex, String grade) {
        this.name = name;
        this.age = age;
        this.sex = sex;
        this.grade = grade;
    }
}
```

##### .2.2. JavaBean

> `构造过程被分到几个调用中`，在构造过程中，J`avaBean可能处于不一致的状态`，线程不安全；把类做成不可变的可能性不复存在

```java
public class Student {
    /*必填参数*/
    private String name;
    private int age;
    /*可选参数*/
    private String sex;
    private String grade;

    public Student(){}

    public void setName(String name) {
        this.name = name;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    public void setGrade(String grade) {
        this.grade = grade;
    }
}
```

##### .2.3. &Builder模式

> - 客户端代码很容易边写，并且便于阅读。Builder模式模拟了具名的可选参数。
> - Builder模式还比重叠构造器模式更加冗长，因此`只在有很多参数的时候才使用，比如4个或更多个参数。`

```java
public class Student {
    /*必填*/
    private String name;
    private int age;
    /*选填*/
    private String sex;
    private String grade;

    public static class Builder {
        private String name;
        private int age;
        private String sex = "";
        private String grade = "";

        public Builder(String name, int age) {
            this.name = name;
            this.age = age;
        }
        public Builder sex(String sex) {
            this.sex = sex;
            return this;
        }
        public Builder grade(String grade) {
            this.grade = grade;
            return this;
        }
        public Student build() {
            return new Student(this);
        }
    }
    private Student(Builder builder) {
        this.name = builder.name;
        this.age = builder.age;
        this.sex = builder.sex;
        this.grade = builder.grade;
    }
}
//实例化代码
Student student = new Student.Builder("jtzen9", 24).grade("1年级").build();
```

#### .3. 私有构造器或枚举强化SingleTon属性

> Singleton通常代表无状态的对象，例如函数(第24项)或者本质上唯一的系统组件。指仅仅被实例化一次的类。

```java
public class Singleton implements Serializable {
    //private的构造函数用于避免外界直接使用new来实例化对象  
    private Singleton() {     
    }  
    public static Singleton getInstance()
    {     
        return SingletonHolder.INSTANCE; 
    } 
    private static class SingletonHolder
    {
        static final Singleton INSTANCE = new Singleton();     
    }        
    //readResolve方法应对单例对象被序列化   
    private Object readResolve() {     
        return getInstance();     
    }     
}
```

- 序列化测试代码

```java
public class SerializableTest {
        @Test
        public void serializableTest() throws Exception{
            serializable(Singleton.getInstance(), "test");
            Singleton singleton = deserializable("test");
            Assert.assertEquals(singleton, Singleton.getInstance());
        }
        //序列化
        private void serializable(Singleton singleton, String filename) throws IOException {
            FileOutputStream fos = new FileOutputStream(filename);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(singleton);
            oos.flush();
        }
        //反序列化
        @SuppressWarnings("unchecked")
        private <T> T deserializable(String filename) throwsIOException,
                                                ClassNotFoundException {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);
            return (T) ois.readObject();
        }
}
```

##### .3.2. 枚举强化单例

```java
public enum SingletonEnum {
        INSTANCE;

        private String filed01;

        public String getFiled01() {
            return filed01;
        }
        public void setFiled01(String filed01) {
            this.filed01 = filed01;
        }
}

SingletonEnum.INSTANCE.setFiled01("123");
System.out.println(SingletonEnum.INSTANCE.getFiled01());
```

##### 3.3. 双重校验锁

```java
public class DoubleCheckSingleton {
    private volatile static DoubleCheckSingleton singleton;
    private DoubleCheckSingleton() {

    }
    public static DoubleCheckSingleton getSingleton() {
        if (singleton == null) {
            synchronized (DoubleCheckSingleton.class) {
                if (singleton == null) {
                    singleton = new DoubleCheckSingleton();
                }
            }
        }
        return singleton;
    }
}
```

#### .4. 私有构造器强化不可实例化能力

> `工具类的实例化是没有任何用处的`，但是需要注意到的是在不提供构造函数的时候，编译器会自动提供一个默认的构造器。这会造成意识的实例化。
>
> - 企图将类做成抽象类来阻止实例化，但是这种方法并不可靠，因为抽象类可以被继承，而它的子类是可以实例化的;  有问题；
> - 显示的提供一个私有的构造函数来阻止实例化；
> - `之后常用的工具类封装参考这个；`

```java
public class UtilityClass {
	// Suppress default constructor for noninstantiability
	private UtilityClass() {
		throw new AssertionError();
	}
    // 方法
}
```

#### .5. 避免创建不必要的对象

##### .5.1. 重用对象

> - `最好能重用对象而不是在每次需要的时候就创建一个相同功能的新对象`。重用方式既快速，又流行。如果对象是不可变的，它就始终可以被重用。

```java
String s = new String("hello world");　　//Don't do this!
String s = "hello world"; //对于所有在同一台虚拟机中运行的代码，只要它们包含相同的字符串字面常量，该对象就会被重用。
```

##### .5.2. 使用静态工厂方法

```java
public class Test {
    public static void main(String[] args) {
        // 使用带参构造器
        Integer a1 = new Integer("1");
        Integer a2 = new Integer("1");
        
        //使用valueOf()静态工厂方法
        Integer a3 = Integer.valueOf("1");
        Integer a4 = Integer.valueOf("1");
        
        //结果为false，因为创建了不同的对象
        System.out.println(a1 == a2);
        
        //结果为true，因为不会新建对象
        System.out.println(a3 == a4);
    }

}
```

##### .5.3. 重用那些已知不会修改的可变对象

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210511105326262.png)

##### .5.4. mapkeyset

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210511110539945.png)

##### .5.5. 优先使用基本数据类型而不是装箱基本类型，要当心无意识的自动装箱。

```java
public class TestLonglong {
    public static void main(String[] args) {
        Long sum = 0L;         // 这里每次添加一个long 构造实例，  升级： long sum=0L;
        for(long i = 0; i < Integer.MAX_VALUE; i++){
            sum += i;
        }
        System.out.println(sum);
    }
    
}
```

#### .6. 消除过期的对象引用

##### .6.1.  过期引用

> **元素出栈，忘记设置为Null：**;
>
> 如果一个栈先是**增长**，然后再**收缩**，从栈中弹出来的对象**不会被当作垃圾**回收，即使使用栈的程序不再引用这些对象，它们也不会被回收。因为栈内部维护着对这些对象的**过期引用（obsolete reference）**。指永远也**不会再被解除的引用**。
> 在本例中，在elements数组的“活动部分（active portion）”之外的任何引用都是过期的。活动部分指elements中下标小于size的那些元素。
> **解决方法**：**清空**引用。元素**被弹出栈**，**指向它的引用就过期**了。修改如下：
>
> - 如果它们以后又被错误地解除引用，程序抛出NullPointerException异常；
> - **只要类是自己管理内存，程序员就应该警惕内存泄漏问题**。一旦元素被释放掉，该元素中包含的任何对象引用都应该被清空。

```java
public class Stack {
    pprivate Object[] elements;
    private int size = 0;
    private static final int DEFAULT_INITAL_CAPACITY = 16;
    
    public Stack() {
        elements = new Object[DEFAULT_INITAL_CAPACITY];
    }
    
    public void push(Object e) {
        ensureCapacity();
        elements[size++] = e;
    }
    //修改前
    public Object pop() {
        if(size == 0) {
            throw new EmptyStackException();
        }
        return elements[--size];
    }
    //修改后
    public Object pop() {
        if(size == 0) {
            throw new EmptyStackException();
        }
        Object result = elements[--size];
        elements[size] = null;//显式地清空引用
        return result;
	}
    
    private void ensureCapacity() {
        if(elements.length == size)
            elements = Arrays.copyOf(elements, 2 * size + 1);
    }
}
```

##### .6.2. 缓存

> 缓存的清除工作可以由一个后台线程（可能是Timer或者是ScheduledThreadPoolExecutor）来完成，或者也可以再给缓存添加新条目的时候顺便进行清理。

##### .6.3. 监听器和回调

> 往往通过Heap剖析工具发现内存泄漏。

#### .7. 避免使用终结方法

![image-20210511112412381](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210511112412381.png)

### 2. 对所有对象通用方法

> equals是Obejct提供的通用方法，作用是`比较两个实例逻辑上值是否相等`，在自己设计的类中要考虑是否需要复写equlas方法，以及遵守equals方法的规范约定。

- **java.lang.Object实现的equals()方法如下：Object类实现的equals()方法仅仅在两个变量指向同一个对象的引用时才返回true。**

```java
//JDK1.8中的Object类的equals()方法
public boolean equals(Object obj) {
        return (this == obj);
    }
```

#### .1. 覆盖equals 遵守通用约定

- 满足下列条件不覆盖equals

>- 类的每个实例本质上是唯一的；
>- 不关心类是否提供了“逻辑相等”的测试功能；
>- 超类覆盖了equals，该方法对于子类也合适；
>- 类是私有或者包级别私有的；

当类具备自身特有的逻辑相等概念时，并且超类还没覆盖equals以实现期望方式时，我们应该重载equals；

```java
public class IntObject {
	private int i;
	public IntObject(int i) {
		super();
		this.i = i;
	} @Override
	public boolean equals(Object obj) {
        if(obj instanceof IntObject){
            IntObject p = (IntObject) obj; 
            if(this.i == p.i){ 
                return true; 
            }
        }
		return false;
	} 
    public static void main(String[] args) {
		IntObject p1 = new IntObject(1);
		IntObject p2 = new IntObject(1); 
        System.out.println(p1.equals(p2));
		System.out.println(p1 == p2);
	}
	
}
```

> 如果类具有自己特定的“逻辑相等”概念(不同于对象等同的概念)，而且超类还没有覆盖equals()方法以实现期望的行为，这时就需要我们覆盖equals()方法。

```java
//JDK1.8中的Boolean类的equals方法，实现根据值而不是对象引用判定两个Boolean
//对象相等的equals()方法
public boolean equals(Object obj) {
        if (obj instanceof Boolean) {
            return value == ((Boolean)obj).booleanValue();
        }
        return false;
    }
```



- 在覆盖equals时，应该满足以下约定：

> - 自反性：x.equals(x)为true；
> - 对称性：x.equals(y)为true时，y.equals(x)也为true；
> - 传递性：x.equals(y), y.equals(z)为true，x.equals(z)也为true；
> - 一致性：如果两个对象相等，那么它们就必需始终保持相等，除非对象被修改了；
> - 非空性：任何对象都不等于null；非null引用值x，则x.equals(null) 必须返回false。

##### .1.1. 实现高质量equals：

- 使用==操作符检查“参数是否为对象的引用”；

```java
//JDK1.8中的String类的equals()方法
    public boolean equals(Object anObject) {
        if (this == anObject) {//性能优化
            return true;
        }
        if (anObject instanceof String) {
            String anotherString = (String)anObject;
            int n = value.length;
            if (n == anotherString.value.length) {
                char v1[] = value;
                char v2[] = anotherString.value;
                int i = 0;
                while (n-- != 0) {
                    if (v1[i] != v2[i])
                        return false;
                    i++;
                }
                return true;
            }
        }
        return false;
    }
```

- 使用instanceof检查“参数是否为正确的类型”，类或者类实现的接口；
- 把参数转换成正确的类型，要使用instanceof测试；
- 对类中的关键域要，检查参数中的域是否与对应的域对应；
- 不要企图让equals()方法过于智能。
- 不要将equals()方法声明中的Object对象替换为其它的类型。

> 对于`基本数据类型`，使用`==`。
> 对于`对象引用类型`，调用`equals()`方法。
> 对于`float，double，可调用Float.compare(),Double.compare()方法，防止Float.NAN,-0.0f等特殊的常量。`
> 为了获取最佳的性能，优先比较最有可能不一样的域。冗余域(可由关键域得到的域)不需要比较。

#### .2. 覆盖equals()方法时总要覆盖hashCode()

`导致该类无法结合所有的给予散列的集合一起正常运作`。这类集合包括 HashSet、HashMap，下面是Object 的通用规范：

- 在应用程序的执行期间，只要``对象的 equals 方法的比较操作所用到的信息没有被修改``，那么同一个对象的多次调用，``hashCode 方法都必须返回同一个值``。在一个应用程序和另一个应用程序的执行过程中，执行 hashCode 方法返回的值可以不相同。
- 如果``两个对象根据 equals 方法比较出来是相等的``，那么调用这两个对象的 `hashCode 方法都必须产生同样的整数结果`
- 如果两个对象根据 equals 方法比较是不相等的，那么调用这两个对象的 hashCode 方法不一定要求其产生相同的结果，但是程序员应该知道，给不相等的对象产生截然不同的整数结果，有可能提高散列表的性能。

```java
public class PhoneNumber {

    int numbersOne;
    int numbersTwo;
    int numbersThree;

    public PhoneNumber(int numbersOne, int numbersTwo, int numbersThree) {
        this.numbersOne = numbersOne;
        this.numbersTwo = numbersTwo;
        this.numbersThree = numbersThree;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PhoneNumber)) return false;
        PhoneNumber that = (PhoneNumber) o;
        return Objects.equals(numbersOne, that.numbersOne) &&
                Objects.equals(numbersTwo, that.numbersTwo) &&
                Objects.equals(numbersThree, that.numbersThree);
    }
    
    private volatile static int hashcode;
    @Override
    public int hashCode() {
      int result = hashCode;
      if(result == 0){
        result = Integer.hashCode(numbersOne);
        result = 31 * result + Integer.hashCode(numbersTwo);
        result = 31 * result + Integer.hashCode(numbersThree);
        hashCode = result;
      }
      return result;
    }
    public static void main(String[] args) {
        Map numberMap = new HashMap();
        numberMap.put(new PhoneNumber(707,867,5309),"Jenny");
        System.out.println(numberMap.get(new PhoneNumber(707,867,5309)));
    }
}
```

#### .3. 始终覆盖toString

```java
@override public String toString(){
	return String.fromat("%03d xxx",value);
}
```

#### .4. 谨慎使用clone

>  它决定了 Object 的受保护的 clone 方法实现的行为：如果一个类实现了 Cloneable 接口，那么 Object 的 clone 方法将返回该对象的逐个属性（field-by-field）拷贝；否则会抛出 `CloneNotSupportedException` 异常。这是一个非常反常的接口使用，而不应该被效仿。 通常情况下，实现一个接口用来表示可以为客户做什么。但对于 Cloneable 接口，它会修改父类上受保护方法的行为。

- `不可变类永远不应该提供 clone 方法`

##### .1. 普通克隆

> 假设你希望在一个类中实现 Cloneable 接口，它的父类提供了一个行为良好的 clone 方法。``首先调用 super.clone, 得到的对象将是原始的完全功能的复制品``。 在你的类中声明的任何属性将具有与原始属性相同的值。 如果`每个属性包含原始值或对不可变对象的引用，则返回的对象可能正是你所需要的，在这种情况下，不需要进一步的处理`。

```java
package com.jueee.item13;
public class Item13Example01 {
    public static void main(String[] args) {
        PhoneNumber1 number1 = new PhoneNumber1(707, 867, 5309);
        System.out.println(number1);    // 707-867-5309
        
        try {
            PhoneNumber1 number2 = (PhoneNumber1)number1.clone();
            System.out.println(number2);    // 707-867-5309
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
    }
}
class PhoneNumber1 implements Cloneable{
    private final short areaCode, prefix, lineNum;
    public PhoneNumber1(int areaCode, int prefix, int lineNum) {
        this.areaCode = rangeCheck(areaCode, 999, "area code");
        this.prefix = rangeCheck(prefix, 999, "prefix");
        this.lineNum = rangeCheck(lineNum, 9999, "line num");
    }
    private static short rangeCheck(int val, int max, String arg) {
        if (val < 0 || val > max)
            throw new IllegalArgumentException(arg + ": " + val);

        return (short)val;
    }
    @Override
    public String toString() {
        return String.format("%03d-%03d-%04d", areaCode, prefix, lineNum);
    }
    @Override
    protected Object clone() throws CloneNotSupportedException {
        try {
            return (PhoneNumber1) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();  // Can't happen
        }
    }
}
```

##### .2. 包含对象引用可变对象属性

```java
package com.jueee.item13;

import java.util.Arrays;
import java.util.EmptyStackException;

public class Item13Example03 {

    public static void main(String[] args) {
        Stack2 stack1 = new Stack2();
        stack1.push(1);
        stack1.push(2);
        stack1.push(3);
        System.out.println(stack1.pop());    // 3
        
        Stack2 stack2 = null;
        try {
            stack2 = (Stack2)stack1.clone();  
            System.out.println(stack2.pop());    // 2
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        
        System.out.println(stack1.pop());   // 2
        System.out.println(stack2.pop());   // 1
    }
}

class Stack2 implements Cloneable{

    private Object[] elements;     #可变属性
    private int size = 0;
    private static final int DEFAULT_INITIAL_CAPACITY = 16;

    public Stack2() {
        this.elements = new Object[DEFAULT_INITIAL_CAPACITY];
    }

    public void push(Object e) {
        ensureCapacity();
        elements[size++] = e;
    }

    public Object pop() {
        if (size == 0)
            throw new EmptyStackException();
        Object result = elements[--size];

        elements[size] = null; // Eliminate obsolete reference
        return result;
    }
    
    public int size() {
        return elements.length;
    }

    // Ensure space for at least one more element.
    private void ensureCapacity() {
        if (elements.length == size)
            elements = Arrays.copyOf(elements, 2 * size + 1);
    }

    @Override
    protected Object clone() throws CloneNotSupportedException {
        try {
            Stack2 result = (Stack2) super.clone();   //生成的 Stack 实例在其 size 属性中具有正确的值，但 elements 属性引用与原始 Stack 实例相同的数组。
            result.elements = elements.clone();
            return result;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();  // Can't happen
        }
    }
}
```

##### .3. 复制构造方法或复制工厂

> 复制构造方法及其静态工厂变体与 Cloneable/clone 相比有许多优点：
>
> - 它们不依赖风险很大的语言外的对象创建机制；
> - 不要求遵守那些不太明确的惯例；
> - 不会与 final 属性的正确使用相冲突;
> - 不会抛出不必要的检查异常;
> - 不需要类型转换。

```java
// 复制构造方法
public Yum(Yum yum) { ... };
// 复制工厂
public static Yum newInstance(Yum yum) { ... };
```

#### 5. 考虑实现comparable接口

> 将这个对象与指定的对象进行比较。当该对象小于、等于或大于指定对象的时候， 分别返回一个负整数、零或者正整数。如果`指定对象的类型与此对象不能进行比较，则引发ClassCastException 异常`。

> 每当实现一个对排序敏感的类时，都应该让这个类实现Comparable接 口，以便其实例可以轻松地被分类、搜索，以及用在基于比较的集合中。每当`在compareTo方法的实现中比较域值时`，`都要避免使用 < 和 >  - 操作符`很容易造成整数溢出，同时违反 IEEE 754 浮点算术标准，而应该在`装箱基本类型的类中使用静态的compare方法，或者在Comparator接口中使用比较器构造方法`

```java
// Comparator based on static compare method
static Comparator<Object> hashCodeOrder = new Comparator<>() {
    public int compare(Object o1, Object o2) {
        return Integer.compare(o1.hashCode(), o2.hashCode());
    }
};

// Comparator based on Comparator construction method
static Comparator<Object> hashCodeOrder =
    Comparator.comparingInt(o -> o.hashCode());
```

```java
// Comparable with comparator construction methods
private static final Comparator<PhoneNumber> COMPARATOR =
        comparingInt((PhoneNumber pn) -> pn.areaCode)
        .thenComparingInt(pn -> pn.prefix)
        .thenComparingInt(pn -> pn.lineNum);
public int compareTo(PhoneNumber pn) {
    return COMPARATOR.compare(this, pn);
}
```

##### .1. 多属性比较

```java
// Multiple-field `Comparable` with primitive fields
public int compareTo(PhoneNumber pn) {
    int result = Short.compare(areaCode, pn.areaCode);
    if (result == 0) {
        result = Short.compare(prefix, pn.prefix);
        if (result == 0)
            result = Short.compare(lineNum, pn.lineNum);
    }
    return result;
}

//改进版本
// Comparable with comparator construction methods
private static final Comparator<PhoneNumber> COMPARATOR =
        comparingInt((PhoneNumber pn) -> pn.areaCode)
        .thenComparingInt(pn -> pn.prefix)
        .thenComparingInt(pn -> pn.lineNum);
public int compareTo(PhoneNumber pn) {
    return COMPARATOR.compare(this, pn);
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/effectjava/  

