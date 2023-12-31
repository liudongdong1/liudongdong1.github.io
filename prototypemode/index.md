# PrototypeMode


> 原型模式（Prototype Pattern）是`用于创建重复的对象，同时又能保证性能`。这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。
>
> - 保护性拷贝场景
>
> - 
>
> 与通过对一个类进行实例化来构造新对象不同的是，原型模式是`通过拷贝一个现有对象生成新对象的`。`浅拷贝实现 Cloneable重写`，`深拷贝是通过实现 Serializable 读取二进制流`。
>
> - 缺点：需要为每一个类配备一个克隆方法，这对全新的类来说不是很难，但对已有的类进行改造时，需要修改其源代码，违背了ocp原则。
> - 在实现深克隆的时候可能需要比较复杂的代码

### 1. 简单模式

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210705094513926.png)

> **客户(Client)角色**：**客户类**提出创建对象的请求；
>
> **抽象原型(Prototype)角色**：这是一个抽象角色，通常由一个`Java`**接口**或者`Java`**抽象类**实现。此角色定义了的具体原型类所需的实现的方法。
>
> **具体原型(Concrete Prototype)角色**：此角色需要实现**抽象原型角色**要求的**克隆相关**的**接口**。

```java
//prototype.java
/**
 * 抽象原型角色
 */
public abstract class Prototype {
    private String id;

    public Prototype(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    /**
     * 克隆自身的方法
     * @return 一个从自身克隆出来的对象。
     */
    public abstract Prototype clone();
}

//conreteprototype1.java
public class ConcretePrototype1 extends Prototype {
    public ConcretePrototype1(String id) {
        super(id);
    }

    public Prototype clone() {
        Prototype prototype = new ConcretePrototype1(this.getId());
        return prototype;
    }
}

//concreteprototype2.java
public class ConcretePrototype2 extends Prototype {
    public ConcretePrototype2(String id) {
        super(id);
    }

    public Prototype clone() {
        Prototype prototype = new ConcretePrototype2(this.getId());
        return prototype;
    }
}

```

### 2. 登记模式

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210705094659388.png)

> **客户(Client)角色**：**客户类**提出创建对象的请求；
>
> **抽象原型(Prototype)角色**：这是一个抽象角色，通常由一个`Java`**接口**或者`Java`**抽象类**实现。此角色定义了的具体原型类所需的实现的方法。
>
> **具体原型(Concrete Prototype)角色**：此角色需要实现**抽象原型角色**要求的**克隆相关**的**接口**。
>
> **原型管理器(Prototype Manager)角色**：提供各种**原型对象**的**创建**和**管理**。

```java
public class PrototypeManager {
    /**
     * 用来记录原型的编号同原型实例的对象关系
     */
    private static Map<String, Prototype> map = new HashMap<>();

    /**
     * 私有化构造方法，避免从外部创建实例
     */
    private PrototypeManager() {
    }

    /**
     * 向原型管理器里面添加或者修改原型实例
     *
     * @param prototypeId 原型编号
     * @param prototype   原型实例
     */
    public static void setProtoType(String prototypeId, Prototype prototype) {
        map.put(prototypeId, prototype);
    }

    /**
     * 根据原型编号从原型管理器里面移除原型实例
     *
     * @param prototypeId 原型编号
     */
    public static void removePrototype(String prototypeId) {
        map.remove(prototypeId);
    }

    /**
     * 根据原型编号获取原型实例
     *
     * @param prototypeId 原型编号
     * @return 原型实例对象
     * @throws Exception 如果根据原型编号无法获取对应实例，则提示异常“您希望获取的原型还没有注册或已被销毁”
     */
    public static Prototype getPrototype(String prototypeId) throws Exception {
        Prototype prototype = map.get(prototypeId);

        if (prototype == null) {
            throw new Exception("您希望获取的原型还没有注册或已被销毁");
        }

        return prototype;
    }

}
```

### 3. Clone 拷贝方法

- 新（拷贝产生）、旧（元对象）`对象不同`，但是`内部如果有引用类型的变量，新、旧对象引用的都是同一引用。`
- 重写的clone方法一个要实现Cloneable接口。虽然这个接口并没有什么方法，但是必须实现该标志接口。 如果不实现将会在运行期间抛出：CloneNotSupportedException异常 

```java
package cn.cupcat.java8;

/**
 * Created by xy on 2017/12/25.
 */
public class Person implements Cloneable{
    private String name;    //  引用类型，指向堆中地址不变，但堆中指向方法区地址变了
    private int age;      //非引用类型，拷贝前后地址不一样
    private int[] ints;   //引用类型，拷贝前后地址一样, 需要使用
    // ......
    /**
     *  默认实现
     * */
    @Override
    public Object clone() throws CloneNotSupportedException {


        return  super.clone();   //拷贝产生了新的对象
        
    }
    //深拷贝
    @Override
    public Object clone() throws CloneNotSupportedException {
        Person person = new Person(name,age);
        
        //对mImages对象也调用clone ( )函数，进行深拷贝
        doc .mImages -[ArrayList<String>) this .mImages.clone ( ) ;

               
        int[] ints = new int[this.ints.length];
        System.arraycopy(this.ints,0,ints,0,ints.length);
        person.setInts(ints);

        return  person;
    }
}

```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjI2MDk0MTMzNTY3.png)

### Resource

- https://juejin.cn/post/6844903638138093581#heading-14


---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/prototypemode/  

