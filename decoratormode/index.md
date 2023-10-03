# DecoratorMode


> **装饰者模式(Decorator Pattern)**：动态地给一个对象增加一些额外的职责，增加对象功能来说，装饰模式比生成子类实现更为灵活。装饰模式是一种对象结构型模式。通常会定义一个抽象装饰类，而将具体的装饰类作为它的子类. 装饰模式的**核心在于抽象装饰类的设计**。
>
> - 对于扩展一个对象的功能，装饰模式比继承更加灵活性，不会导致类的个数急剧增加。

- **Component（抽象构件）**：它是`具体构件和抽象装饰类的共同父类`，`声明了在具体构件中实现的业务方法`，它的引入可以使客户端`以一致的方式处理未被装饰的对象以及装饰之后的对象`，实现客户端的透明操作。
- **ConcreteComponent（具体构件）**：它是抽象构件类的子类，用于`定义具体的构件对象`，实现了在抽象构件中声明的方法，装饰器可以给它增加额外的职责（方法）。
- **Decorator（抽象装饰类）**：它也是`抽象构件类的子类`，用于`给具体构件增加职责`，但是具体职责在其子类中实现。它`维护一个指向抽象构件对象的引用`，通过该引用可以调用装饰之前构件对象的方法，并通过其子类扩展该方法，以达到装饰的目的。
- **ConcreteDecorator（具体装饰类）**：它是抽象装饰类的子类，负责向构件添加新的职责。`每一个具体装饰类都定义了一些新的行为`，它可以调用在抽象装饰类中定义的方法，并可以增加新的方法用以扩充对象的行为。

### 1. Demo1

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704223811529.png)

> 煎饼 加一个鸡蛋 加一个鸡蛋 加一根香肠, 销售价格: 12

```java
public abstract class ABattercake {
    protected abstract String getDesc();
    protected abstract int cost();
}
```

```java
public class Battercake extends ABattercake {
    @Override
    protected String getDesc() {
        return "煎饼";
    }
    @Override
    protected int cost() {
        return 8;
    }
}
```

- 抽象装饰类，需要注意的是，**抽象装饰类通过成员属性的方式将 煎饼抽象类组合进来，同时也继承了煎饼抽象类**

```java
public abstract class AbstractDecorator extends ABattercake {
    private ABattercake aBattercake;

    public AbstractDecorator(ABattercake aBattercake) {
        this.aBattercake = aBattercake;
    }
    
    protected abstract void doSomething();

    @Override
    protected String getDesc() {
        return this.aBattercake.getDesc();
    }
    @Override
    protected int cost() {
        return this.aBattercake.cost();
    }
}
```

```java
public class EggDecorator extends AbstractDecorator {
    public EggDecorator(ABattercake aBattercake) {
        super(aBattercake);
    }

    @Override
    protected void doSomething() {

    }

    @Override
    protected String getDesc() {
        return super.getDesc() + " 加一个鸡蛋";
    }

    @Override
    protected int cost() {
        return super.cost() + 1;
    }
    
    public void egg() {
        System.out.println("增加了一个鸡蛋");
    }
}
```

### 2. Demo2

下面需求开始：设计游戏的装备系统，基本要求，要可以计算出每种装备在镶嵌了各种宝石后的攻击力和描述：

具体需求：

1、武器（攻击力20） 、戒指（攻击力5）、护腕（攻击力5）、鞋子（攻击力5）

2、蓝宝石（攻击力5/颗）、黄宝石（攻击力10/颗）、红宝石（攻击力15/颗）

3、每个装备可以随意镶嵌3颗

```java
package com.zhy.pattern.decorator;
 
/**
 * 装备的接口
 * 
 * @author zhy
 * 
 */
public interface IEquip
{
 
	/**
	 * 计算攻击力
	 * 
	 * @return
	 */
	public int caculateAttack();
 
	/**
	 * 装备的描述
	 * 
	 * @return
	 */
	public String description();
}
//Equip 实例
package com.zhy.pattern.decorator;
 
/**
 * 武器
 * 攻击力20
 * @author zhy
 * 
 */
public class ArmEquip implements IEquip
{
 
	@Override
	public int caculateAttack()
	{
		return 20;
	}
 
	@Override
	public String description()
	{
		return "屠龙刀";
	}
 
}

//宝石超类

package com.zhy.pattern.decorator;
 
/**
 * 装饰品的接口
 * @author zhy
 *
 */
public interface IEquipDecorator extends IEquip
{
}
package com.zhy.pattern.decorator;
 
/**
 * 蓝宝石装饰品
 * 每颗攻击力+5
 * @author zhy
 * 
 */
public class BlueGemDecorator implements IEquipDecorator
{
	/**
	 * 每个装饰品维护一个装备
	 */
	private IEquip equip;
 
	public BlueGemDecorator(IEquip equip)
	{
		this.equip = equip;
	}
 
	@Override
	public int caculateAttack()
	{
		return 5 + equip.caculateAttack();
	}
 
	@Override
	public String description()
	{
		return equip.description() + "+ 蓝宝石";
	}
 
}
```

```java
package com.zhy.pattern.decorator;
 
public class Test
{
	public static void main(String[] args)
	{
		// 一个镶嵌2颗红宝石，1颗蓝宝石的靴子
		System.out.println(" 一个镶嵌2颗红宝石，1颗蓝宝石的靴子");
		IEquip equip = new RedGemDecorator(new RedGemDecorator(new BlueGemDecorator(new ShoeEquip())));
		System.out.println("攻击力  : " + equip.caculateAttack());
		System.out.println("描述 :" + equip.description());
		System.out.println("-------");
		// 一个镶嵌1颗红宝石，1颗蓝宝石的武器
		System.out.println(" 一个镶嵌1颗红宝石，1颗蓝宝石,1颗黄宝石的武器");
		equip = new RedGemDecorator(new BlueGemDecorator(new YellowGemDecorator(new ArmEquip())));
		System.out.println("攻击力  : " + equip.caculateAttack());
		System.out.println("描述 :" + equip.description());
		System.out.println("-------");
	}
}
```

### 3. Java I/O中的装饰者模式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704224507594.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704224617505.png)

如果要**实现一个自己的包装流**，根据上面的类图，需要继承抽象装饰类 FilterInputStream,譬如来实现这样一个操作的装饰者类：将输入流中的所有小写字母变成大写字母

```java
import java.io.FileInputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

public class UpperCaseInputStream extends FilterInputStream {
    protected UpperCaseInputStream(InputStream in) {
        super(in);
    }

    @Override
    public int read() throws IOException {
        int c = super.read();
        return (c == -1 ? c : Character.toUpperCase(c));
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        int result = super.read(b, off, len);
        for (int i = off; i < off + result; i++) {
            b[i] = (byte) Character.toUpperCase((char) b[i]);
        }
        return result;
    }

    public static void main(String[] args) throws IOException {
        int c;
        InputStream in = new UpperCaseInputStream(new FileInputStream("D:\\hello.txt"));
        try {
            while ((c = in.read()) >= 0) {
                System.out.print((char) c);
            }
        } finally {
            in.close();
        }
    }
}
```

### 4. spring cache 中的装饰者模式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704224901969.png)

```java
public class TransactionAwareCacheDecorator implements Cache {
    private final Cache targetCache;
    
    public TransactionAwareCacheDecorator(Cache targetCache) {
        Assert.notNull(targetCache, "Target Cache must not be null");
        this.targetCache = targetCache;
    }
    
    public <T> T get(Object key, Class<T> type) {
        return this.targetCache.get(key, type);
    }

    public void put(final Object key, final Object value) {
        // 判断是否开启了事务
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            // 将操作注册到 afterCommit 阶段
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronizationAdapter() {
                public void afterCommit() {
                    TransactionAwareCacheDecorator.this.targetCache.put(key, value);
                }
            });
        } else {
            this.targetCache.put(key, value);
        }
    }
    // ...省略...
}
```

### 5. spring session 中的装饰者模式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704225417971.png)

```java
public class ServletRequestWrapper implements ServletRequest {
    private ServletRequest request;
    
    public ServletRequestWrapper(ServletRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        this.request = request;
    }
    
    @Override
    public Object getAttribute(String name) {
        return this.request.getAttribute(name);
    }
    //...省略...
}    
```

### 6. Mybatis 缓存中的装饰者模式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704225309124.png)

### Resource

- https://juejin.cn/post/6844903681322647566
- [Java日志框架：slf4j作用及其实现原理](https://www.cnblogs.com/xrq730/p/8619156.html)
- [HankingHu：由装饰者模式来深入理解Java I/O整体框架](https://blog.csdn.net/u013309870/article/details/75735676)
- [HryReal：Java的io类的使用场景](https://blog.csdn.net/qq_33394088/article/details/78512407)
- Todo: 进一步补充



---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/decoratormode/  

