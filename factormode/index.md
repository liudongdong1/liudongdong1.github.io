# FactorMode


### 1. 工厂方法

> 在工厂方法模式中，抽象产品类Product负责定义产品的共性，事项对事物最抽象的定义，Creator为抽象创建类，也就是抽象工厂，具体如何创建产品类是由具体的实现工厂ConcreteCreator完成的。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/6eb8364ad8984f1bb01fa0b2e551e64dtplv-k3u1fbpfcp-zoom-in-crop-mark4536000.png)

- ConcreteCreator 有静态实现，和反射方式实现

```java
/**
 * 抽象产品类
 */
public abstract class Product {
    //产品类的公共方法
    public void method1(){}

    //抽象方法
    public abstract void method2();
}
/**
 * 具体产品类1
 */
public class ConcreteProduct1 extends Product {
    @Override
    public void method2() {
       //业务逻辑处理
    }
}


/**
 * 抽象工厂类
 */
public abstract class Creator {
    // 创建一个产品对象，其输入参数类型可以自行设置
    public abstract <T extends Product> T createProduct(Class<T> tClass) ;
}

/**
 * 具体工厂类
 */
public class ConcreteCreator extends Creator {
    @Override
    public <T extends Product> T createProduct(Class<T> tClass) {
        Product product = null;
        try {
            product = (Product) Class.forName(tClass.getName()).newInstance();
        } catch (Exception e) {
            //异常处理
        }
        return (T)product;
    }
}

/**
 * 具体场景类
 */
public class FactoryClient {
    public static void main(String[] args){
        Creator creator = new ConcreteCreator();
        creator.createProduct(ConcreteProduct1.class);
    }
}
```

### 2. Android onCreate 方法实现逻辑

- todo？

### 3. 抽象工厂方法

- Factory 产生的对象有多个部件，每一个部件有不同的公司生产。

```java
//抽象产品类-- CPU
public abstract class CPU {
    public abstract void showCPU();
}
//抽象产品类-- 内存
public abstract class Memory {
    public abstract void showMemory();
}
//抽象产品类-- 硬盘
public abstract class HD {
    public abstract void showHD();
}


//抽象工厂类，电脑工厂类
public abstract class ComputerFactory {
    public abstract CPU createCPU();

    public abstract Memory createMemory();

    public abstract HD createHD();
}

//具体工厂类--联想电脑
public class LenovoComputerFactory extends ComputerFactory {

    @Override
    public CPU createCPU() {
        return new IntelCPU();
    }

    @Override
    public Memory createMemory() {
        return new SamsungMemory();
    }

    @Override
    public HD createHD() {
        return new SeagateHD();
    }
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/factormode/  

