# IteratorMode


> **迭代器模式：** 把在元素之间游走的责任交给迭代器，而不是聚合对象。`简化了聚合的接口和实现`，让聚合更专注在它所应该专注的事情上，这样做更加符合单一责任原则。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210705152510354.png)

> - **Iterator（抽象迭代器）：** `具体迭代器`需要实现的接口，提供了游走聚合对象元素之间的方法
> - **ConcreteIterator（具体迭代器）：** 对具体的聚合对象进行遍历，每一个聚合对象都应该对应一个具体的迭代器。
> - **Aggregate（抽象聚合类）：** 存储和管理元素对象，声明了`createIterator()`用于创建一个迭代器对象，充当抽象迭代器工厂角色
> - **ConcreteAggregate（具体聚合类）：** 实现了在抽象聚合类中声明的`createIterator()`，该方法返回一个与该具体聚合类对应的具体迭代器`ConcreteIterator`实例。

### 1. Java Demo

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210705152844513.png)

```java
interface Iterator<E> {

    //判断是否有还有下一个元素
    boolean hasNext();

    //取出下一个对象
    E next();
}

class MusicIterator<E> implements Iterator<E> {
    private E[] es;
    private int position = 0;
    public MusicIterator(E[] es) {
        this.es = es;
    }
    @Override
    public boolean hasNext() {
        return position != es.length;
    }
    @Override
    public E next() {
        E e = es[position];
        position += 1;
        return e;
    }
}

interface AbstractList<E> {

    void add(E e);

    Iterator<E> createIterator();
}

class MusicList implements AbstractList<String> {
    private String[] books = new String[5];
    private int position = 0;
    @Override
    public void add(String name) {
        books[position] = name;
        position += 1;
    }
    @Override
    public Iterator<String> createIterator() {
        return new MusicIterator<>(books);
    }
}
```

```java
public class Client {
    public static void main(String[] args) {
        AbstractList<String> list = new MusicList();
        list.add("凉凉");
        list.add("奇谈");
        list.add("红颜");
        list.add("伴虎");
        list.add("在人间");
        Iterator<String> iterator = list.createIterator();
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
    }
}
```

### 2. 责任链模式

- android中的有序广播消息处理
- 多级领导处理

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/6e0466ef3ba3458797f4a27d4476c957tplv-k3u1fbpfcp-zoom-in-crop-mark4536000.png)

```java
public abstract class AbstractOrderHandler {
    /**
     * 区分类型
     *
     * @return
     */
    protected abstract OrderTypeEnum getTypeEnum();
    /**
     * 核心处理
     *
     * @param context 上下文
     * @param args    拓展参数
     */
    public void doHandle(OrderHandleContext context,
                         OrderHandlerChain chain, Object... args) {
        // 我是否可以处理
        if (Objects.isNull(getTypeEnum()) || 
            Objects.equals(context.getTypeEnum(), getTypeEnum())) {
            // 让我来处理
            doHandle(context, args);
        }
        // 我处理完了，交给下家
        chain.handle(context, args);
    }

    /**
     * 具体业务处理
     *
     * @param context
     * @param args
     */
    protected abstract void doHandle(OrderHandleContext context, Object... args);

}
```

```java
@Slf4j
@Service
@Order(100)
public class CreateOrderHandler extends AbstractOrderHandler {

    @Override
    protected OrderTypeEnum getTypeEnum() {
        return null;
    }
    
    @Override
    protected void doHandle(OrderHandleContext context, Object... args) {
        log.info("default create order ... ");

        // 锁定库存
        lockSku(context, args);
        
        // 保存订单
        saveOrder(context);
        
        // 扣除库存
        deductSku(context, args)
    }

}

// 订单反现金
@Service
@Slf4j
@Order(200)
public class RebateOrderHandler extends AbstractOrderHandler {

    @Override
    protected OrderTypeEnum getTypeEnum() {
        return null;
    }

    @Override
    protected void doHandle(OrderHandleContext context, Object... args) {
        log.info("default rebate order ... ");
    }
}
```

```java
@Slf4j
@Component
public class OrderHandlerChain {
    @Autowired
    private List<AbstractOrderHandler> chain;
    @Autowired
    private ApplicationContext applicationContext;

    public void handle(OrderHandleContext context, Object... objects) {
        if (context.getPos() < chain.size()) {
            AbstractOrderHandler handler = chain.get(context.getPos());
            // 移动位于处理器链中的位置
            context.setPos(context.getPos() + 1);
            handler.doHandle(context, this, objects);
        }
    }
}
```

### Resource

- https://segmentfault.com/a/1190000012232509
- https://juejin.cn/post/7043054480681598983


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/iteratormode/  

