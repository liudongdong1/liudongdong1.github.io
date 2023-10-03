# BridgeMode


> 桥接模式(Bridge Pattern)：将抽象部分与它的实现部分分离，使它们都可以独立地变化。它是一种对象结构型模式，又称为柄体(Handle and Body)模式或接口(Interface)模式。
>
> - 桥接模式需要先能分出系统中那些**独立变化的维度**，然后我们再进行分离。桥接模式的思想就是**如何进行分离的过程**。
> - 所有设计模式的思想其实都希望我们更多的**去利用组合，而不是继承**。所以桥接模式的主要思想就是：**将变化的维度抽象为不同的继承体系**，每一个维度是自己的一个继承体系，然后通过组合将所需要的这些变化维度拼接为最后的对象。这维度和维度之间的联系组合我们把它称为桥。

> 抽象类（Abstraction）: 将原来的那个设计`多维度的变化的对象`。这个抽象类主要担当接口的作用！用来多态其实现类的。
>
> 抽象类的扩充（RefinedAbstraction: 也可以理解为`抽象类的具体实现类`。他相当于就完成各种变化的最终组合的实现类，组合方式就是`调用桥的另一个组合对象来动态完成组合的`。我们可以将其中一种变化放入到该抽象类实现中去。比如上面Pen的大小。这样Pen内部通过多态的Color来组合不同的情况。
>
> 实现类接口（Implementor): 定义`实现类的接口`，是桥的另一端的继承体系的祖宗。只封装一个变化。
>
> 实现类的具体实现类（ConcreteImplementor): 就是`该维度变化的各种情况的实现`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705144604350.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705145331862.png)

```java
//典型实现类接口代码：
public interface InterfaceBrand {
   public void showBrand();
}
//典型的抽象类代码：
public abstract class AbstractBike {
   InterfaceBrand brand;//自行车的品牌
   String color;//自行车的颜色
   public AbstractBike(InterfaceBrand brand, String color) {
      this.brand = brand;
      this.color = color;
   }
   public abstract void print();
}
//典型的扩充抽象类代码：
public class MountatinBike extends AbstractBike {
   public MountatinBike(InterfaceBrand brand, String color) {
      super( brand, color);
   }
   public void print(){
      System.out.println("属性：山地车");
      System.out.println("颜色："+color);
      brand.showBrand();
      System.out.println("---------------");
   }
}
//典型的接口的实现的代码：
public class GiantBrand implements InterfaceBrand {
   @Override
   public void showBrand() {
      
      System.out.println("-品牌：捷安特");
   }
}
```

### 1. JDBC 驱动程序

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705145100863.png)



### 2. 银行转账系统

> ① 转账分类：网上转账，柜台转账，AMT 转账（实现层）
>
> ② 转账用户类型：普通用户，银卡用户，金卡用户...（抽象层）

```
itstack-demo-design-7-02
└── src
    ├── main
    │   └── java
    │       └── org.itstack.demo.design.pay
    │           ├── channel
    │           │   ├── Pay.java
    │           │   ├── WxPay.java
    │           │   └── ZfbPay.java
    │           └── mode
    │               ├── IPayMode.java
    │               ├── PayCypher.java
    │               ├── PayFaceMode.java
    │               └── PayFingerprintMode.java
    └── test
         └── java
             └── org.itstack.demo.design.test
                 └── ApiTest.java
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705145704633.png)

> - 左侧`Pay`是一个抽象类，往下是它的两个支付类型实现；微信支付、支付宝支付。
> - 右侧`IPayMode`是一个接口，往下是它的两个支付模型；刷脸支付、指纹支付。
> - 那么，`支付类型` × `支付模型` = 就可以得到相应的组合。
> - **注意**，每种支付方式的不同，刷脸和指纹校验逻辑也有差异，可以使用适配器模式进行处理，这里不是本文重点不做介绍，可以看适配器模式章节。

```java
public abstract class Pay {

    protected Logger logger = LoggerFactory.getLogger(Pay.class);

    protected IPayMode payMode;

    public Pay(IPayMode payMode) {
        this.payMode = payMode;
    }

    public abstract String transfer(String uId, String tradeId, BigDecimal amount);

}

public class WxPay extends Pay {

    public WxPay(IPayMode payMode) {
        super(payMode);
    }

    public String transfer(String uId, String tradeId, BigDecimal amount) {
        logger.info("模拟微信渠道支付划账开始。uId：{} tradeId：{} amount：{}", uId, tradeId, amount);
        boolean security = payMode.security(uId);
        logger.info("模拟微信渠道支付风控校验。uId：{} tradeId：{} security：{}", uId, tradeId, security);
        if (!security) {
            logger.info("模拟微信渠道支付划账拦截。uId：{} tradeId：{} amount：{}", uId, tradeId, amount);
            return "0001";
        }
        logger.info("模拟微信渠道支付划账成功。uId：{} tradeId：{} amount：{}", uId, tradeId, amount);
        return "0000";
    }

}

public class ZfbPay extends Pay {

    public ZfbPay(IPayMode payMode) {
        super(payMode);
    }

    public String transfer(String uId, String tradeId, BigDecimal amount) {
        logger.info("模拟支付宝渠道支付划账开始。uId：{} tradeId：{} amount：{}", uId, tradeId, amount);
        boolean security = payMode.security(uId);
        logger.info("模拟支付宝渠道支付风控校验。uId：{} tradeId：{} security：{}", uId, tradeId, security);
        if (!security) {
            logger.info("模拟支付宝渠道支付划账拦截。uId：{} tradeId：{} amount：{}", uId, tradeId, amount);
            return "0001";
        }
        logger.info("模拟支付宝渠道支付划账成功。uId：{} tradeId：{} amount：{}", uId, tradeId, amount);
        return "0000";
    }

}

public interface IPayMode {

    boolean security(String uId);

}

public class PayFaceMode implements IPayMode{

    protected Logger logger = LoggerFactory.getLogger(PayCypher.class);

    public boolean security(String uId) {
        logger.info("人脸支付，风控校验脸部识别");
        return true;
    }

}

public class PayFingerprintMode implements IPayMode{

    protected Logger logger = LoggerFactory.getLogger(PayCypher.class);

    public boolean security(String uId) {
        logger.info("指纹支付，风控校验指纹信息");
        return true;
    }

}

public class PayCypher implements IPayMode{

    protected Logger logger = LoggerFactory.getLogger(PayCypher.class);

    public boolean security(String uId) {
        logger.info("密码支付，风控校验环境安全");
        return true;
    }

}

//Test
@Test
public void test_pay() {
    System.out.println("\r\n模拟测试场景；微信支付、人脸方式。");
    Pay wxPay = new WxPay(new PayFaceMode());
    wxPay.transfer("weixin_1092033111", "100000109893", new BigDecimal(100));

    System.out.println("\r\n模拟测试场景；支付宝支付、指纹方式。");
    Pay zfbPay = new ZfbPay(new PayFingerprintMode());
    zfbPay.transfer("jlu19dlxo111","100000109894",new BigDecimal(100));
}
```

3、消息管理

　　　　① 消息类型：即时消息、延时消息（实现层）

　　　　② 消息分类：手机短信、邮件消息，QQ消息...（抽象层）

### Resource

- https://juejin.cn/post/68621864711528775754
- https://segmentfault.com/a/1190000022845087

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/bridgemode/  

