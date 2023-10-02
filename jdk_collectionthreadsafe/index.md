# JDK_CollectionThreadSafe


> - `CopyOnWriteArrayList`的写操作与`Vector`的遍历操作性能消耗尤其严重，不推荐使用。
> - `CopyOnWriteArrayList`适用于读操作远远多于写操作的场景。
> - `Vector`读写性能可以和`Collections.synchronizedList`比肩，但`Collections.synchronizedList`不仅可以包装`ArrayList`,也可以包装其他List,扩展性和兼容性更好。

### 1. Vector

> 从JDK1.0开始，`Vector`便存在JDK中，`Vector`是一个线程安全的列表，底层采用数组实现。其线程安全的实现方式非常粗暴：`Vector`大部分方法和`ArrayList`都是相同的，只是加上了`synchronized`关键字，这种方式严重影响效率，因此，不再推荐使用`Vector`了。如果`不需要线程安全性，推荐使用ArrayList替代Vector`; `Vector`通过在方法级别上加入了`synchronized`关键字实现线程安全性。

```java
public synchronized boolean add(E e) {
    modCount++;
    ensureCapacityHelper(elementCount + 1);
    elementData[elementCount++] = e;
    return true;
}

public synchronized boolean add(E e) {
    modCount++;
    ensureCapacityHelper(elementCount + 1);
    elementData[elementCount++] = e;
    return true;
}

public synchronized Iterator<E> iterator() {
    return new Itr();
}    
```

### 2. Collections.synchronizedList

> `Collections.synchronizedList`静态方法将一个非线程安全的List(并不仅限ArrayList)包装为线程安全的List； 转换包装后的list可以实现add，remove，get等操作的线程安全性，但是`对于迭代操`作，`Collections.synchronizedList`并没有提供相关机制，所以迭代时需要对包装后的list（敲黑板，必须对包装后的list进行加锁，锁其他的不行）进行手动加锁;  通过源码可知`Collections.synchronizedList`生成了特定同步的`SynchronizedCollection`，生成的集合每个同步操作都是持有`mutex`这个锁，所以再进行操作时就是线程安全的集合了。

```java
List list = Collections.synchronizedList(new ArrayList());
//必须对list进行加锁
synchronized (list) {
  Iterator i = list.iterator();
  while (i.hasNext())
      foo(i.next());
}
```

- 迭代操作必须加锁，可以使用`synchronized`关键字修饰;
- synchronized持有的监视器对象必须是`synchronized (list)`,即包装后的list,使用其他对象如`synchronized (new Object())`会使`add`,`remove`等方法与迭代方法使用的锁不一致，无法实现完全的线程安全性。

```java
public static <T> List<T> synchronizedList(List<T> list) {
    return (list instanceof RandomAccess ?
            //ArrayList使用了SynchronizedRandomAccessList类
            new SynchronizedRandomAccessList<>(list) :
            new SynchronizedList<>(list));
}
//SynchronizedRandomAccessList继承自SynchronizedList
static class SynchronizedRandomAccessList<E> extends SynchronizedList<E> implements RandomAccess {
}
//SynchronizedList对代码块进行了synchronized修饰来实现线程安全性
static class SynchronizedList<E> extends SynchronizedCollection<E> implements List<E> {
    public E get(int index) {
        synchronized (mutex) {return list.get(index);}
    }
    public E set(int index, E element) {
        synchronized (mutex) {return list.set(index, element);}
    }
    public void add(int index, E element) {
        synchronized (mutex) {list.add(index, element);}
    }
    public E remove(int index) {
        synchronized (mutex) {return list.remove(index);}
    }   
    //迭代操作并未加锁，所以需要手动同步
    public ListIterator<E> listIterator() {
        return list.listIterator(); 
    }
}
```

### 3. CopyOnWriteArrayList

> `CopyOnWriteArrayList`是`java.util.concurrent`包下面的一个实现线程安全的List,顾名思义， Copy~On~Write~ArrayList在进行写操作(add,remove,set等)时会进行Copy操作，可以推测出在进行写操作时`CopyOnWriteArrayList`性能应该不会很高。

```java
public class CopyOnWriteArrayList<E>
    implements List<E>, RandomAccess, Cloneable, java.io.Serializable {
    private static final long serialVersionUID = 8673264195747942595L;

    /** The lock protecting all mutators */
    final transient ReentrantLock lock = new ReentrantLock();

    /** The array, accessed only via getArray/setArray. */
    private transient volatile Object[] array;
    
    /**
     * Creates an empty list.
     */
    public CopyOnWriteArrayList() {
        setArray(new Object[0]);
    }
}
```

```java
public boolean add(E e) {
    final ReentrantLock lock = this.lock;
    lock.lock();
    try {
        Object[] elements = getArray();
        int len = elements.length;
        Object[] newElements = Arrays.copyOf(elements, len + 1);
        newElements[len] = e;
        setArray(newElements);
        return true;
    } finally {
        lock.unlock();
    }
}
```

- 读写分离： 我们读取`CopyOnWriteArrayList`的时候读取的是`CopyOnWriteArrayList`中的`Object[] array`，但是修改的时候，操作的是一个新的`Object[] array`，读和写操作的不是同一个对象，这就是读写分离。这种技术数据库用的非常多，在高并发下为了缓解数据库的压力，即使做了缓存也要对数据库做读写分离，读的时候使用读库，写的时候使用写库，然后读库、写库之间进行一定的同步，这样就避免同一个库上读、写的IO操作太多。
- 最终一致： 对`CopyOnWriteArrayList`来说，线程1读取集合里面的数据，未必是最新的数据。因为线程2、线程3、线程4四个线程都修改了`CopyOnWriteArrayList`里面的数据，但是线程1拿到的还是最老的那个`Object[] array`，新添加进去的数据并没有，所以线程1读取的内容未必准确。不过这些数据虽然对于线程1是不一致的，但是对于之后的线程一定是一致的，它们拿到的`Object[] array`一定是三个线程都操作完毕之后的`Object array[]`，这就是最终一致。最终一致对于分布式系统也非常重要，它通过容忍一定时间的数据不一致，提升整个分布式系统的可用性与分区容错性。当然，最终一致并不是任何场景都适用的，像火车站售票这种系统用户对于数据的实时性要求非常非常高，就必须做成强一致性的。

### Resource

- 
  https://juejin.cn/post/6844904054745743367

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/jdk_collectionthreadsafe/  

