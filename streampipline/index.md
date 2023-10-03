# StreamPipeline


> 如果出于性能考虑，1. 对于`简单操作推荐使用外部迭代手动实现`，2. 对于`复杂操作，推荐使用Stream API`， 3. 在`多核情况下，推荐使用并行Stream API来发挥多核优势`，4.`单核情况下不建议使用并行Stream API`。
>
> 1. 对于简单操作，比如最简单的遍历，Stream串行API性能明显差于显示迭代，但并行的Stream API能够发挥多核特性。
> 2. 对于复杂操作，Stream串行API性能可以和手动实现的效果匹敌，在并行执行时Stream API效果远超手动实现。
>

> 通过`Collection.stream()`方法得到*Head*也就是stage0，紧接着调用一系列的中间操作，不断产生新的Stream。**这些Stream对象以双向链表的形式组织在一起，构成整个流水线，由于每个Stage都记录了前一个Stage和本次的操作以及回调函数，依靠这种结构就能建立起对数据源的所有操作**。这就是Stream记录操作的方式。

### 1. Stream Pipeline

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/Stream_pipeline_example-163056665569226.png)

> 结束操作不会创建新的流水线阶段(Stage)，直观的说就是流水线的链表不会在往后延伸了。结束操作会创建一个包装了自己操作的Sink，这也是流水线中最后一个Sink，这个Sink只需要处理数据而不需要将结果传递给下游的Sink（因为没有下游）。
>
> 首先要说明的是不是所有的Stream结束操作都需要返回结果，有些操作只是为了使用其副作用(*Side-effects*)，比如使用`Stream.forEach()`方法将结果打印出来就是常见的使用副作用的场景:
>
> 1. 对于表中返回boolean或者Optional的操作（Optional是存放 一个 值的容器）的操作，由于值返回一个值，只需要在对应的Sink中记录这个值，等到执行结束时返回就可以了。
> 2. 对于归约操作，最终结果放在用户调用时指定的容器中（容器类型通过[收集器](./5-Streams%20API(II).md#收集器)指定）。collect(), reduce(), max(), min()都是归约操作，虽然max()和min()也是返回一个Optional，但事实上底层是通过调用[reduce()](./5-Streams%20API(II).md#多面手reduce)方法实现的。
> 3. 对于返回是数组的情况，毫无疑问的结果会放在数组当中。这么说当然是对的，但在最终返回数组之前，结果其实是存储在一种叫做*Node*的数据结构中的。Node是一种多叉树结构，元素存储在树的叶子当中，并且一个叶子节点可以存放多个元素。这样做是为了并行执行方便。关于Node的具体结构，我们会在下一节探究Stream如何并行执行时给出详细说明。

<table width="600px"><tr><td align="center">方法名</td><td align="center">作用</td></tr><tr><td>void begin(long size)</td><td>开始遍历元素之前调用该方法，通知Sink做好准备。</td></tr><tr><td>void end()</td><td>所有元素遍历完成之后调用，通知Sink没有更多的元素了。</td></tr><tr><td>boolean cancellationRequested()</td><td>是否可以结束操作，可以让短路操作尽早结束。</td></tr><tr><td>void accept(T t)</td><td>遍历元素时调用，接受一个待处理元素，并对元素进行处理。Stage把自己包含的操作和回调方法封装到该方法里，前一个Stage只需要调用当前Stage.accept(T t)方法就行了。</td></tr></table>

<table width="350px"><tr><td align="center">返回类型</td><td align="center">对应的结束操作</td></tr><tr><td>boolean</td><td>anyMatch() allMatch() noneMatch()</td></tr><tr><td>Optional</td><td>findFirst() findAny()</td></tr><tr><td>归约结果</td><td>reduce() collect()</td></tr><tr><td>数组</td><td>toArray()</td></tr></table>

```java
// Stream.sort()方法用到的Sink实现
class RefSortingSink<T> extends AbstractRefSortingSink<T> {
    private ArrayList<T> list;// 存放用于排序的元素
    RefSortingSink(Sink<? super T> downstream, Comparator<? super T> comparator) {
        super(downstream, comparator);
    }
    @Override
    public void begin(long size) {
        ...
        // 创建一个存放排序元素的列表
        list = (size >= 0) ? new ArrayList<T>((int) size) : new ArrayList<T>();
    }
    @Override
    public void end() {
        list.sort(comparator);// 只有元素全部接收之后才能开始排序
        downstream.begin(list.size());
        if (!cancellationWasRequested) {// 下游Sink不包含短路操作
            list.forEach(downstream::accept);// 2. 将处理结果传递给流水线下游的Sink
        }
        else {// 下游Sink包含短路操作
            for (T t : list) {// 每次都调用cancellationRequested()询问是否可以结束处理。
                if (downstream.cancellationRequested()) break;
                downstream.accept(t);// 2. 将处理结果传递给流水线下游的Sink
            }
        }
        downstream.end();
        list = null;
    }
    @Override
    public void accept(T t) {
        list.add(t);// 1. 使用当前Sink包装动作处理t，只是简单的将元素添加到中间列表当中
    }
}
```

1. 首先begin()方法告诉Sink参与排序的元素个数，方便确定中间结果容器的的大小；
2. 之后通过accept()方法将元素添加到中间结果当中，最终执行时调用者会不断调用该方法，直到遍历所有元素；
3. 最后end()方法告诉Sink所有元素遍历完毕，启动排序步骤，排序完成后将结果传递给下游的Sink；
4. 如果下游的Sink是短路操作，将结果传递给下游时不断询问下游cancellationRequested()是否可以结束处理。

### 2. ParallelStream

> parallelStream其实就是一个并行执行的流，它通过默认的`ForkJoinPool`，**可能**提高你的多线程任务的速度。The parallel streams use the default `ForkJoinPool.commonPool` which [by default has one less threads as you have processors](http://docs.oracle.com/javase/8/docs/api/java/util/concurrent/ForkJoinPool.html), as returned by `Runtime.getRuntime().availableProcessors()` (This means that parallel streams use all your processors because they also use the main thread)。

```java
public static void main(String[] args) {
    IntStream list = IntStream.range(0, 10);
    Set<Thread> threadSet = new HashSet<>();
    //开始并行执行
    list.parallel().forEach(i -> {
        Thread thread = Thread.currentThread();
        System.err.println("integer：" + i + "，" + "currentThread:" + thread.getName());
        threadSet.add(thread);
    });
    System.out.println("all threads：" + Joiner.on("，").join(threadSet.stream().map(Thread::getName).collect(Collectors.toList())));
}
```

#### .1. Fork/Join 框架

> Fork/Join最核心的地方就是利用了现代硬件设备多核,在一个操作时候会有空闲的CPU,那么如何利用好这个空闲的cpu就成了提高性能的关键,而这里我们要提到的工作窃取（work-stealing）算法就是整个Fork/Join框架的核心理念,工作窃取（work-stealing）算法是指某个线程从其他队列里窃取任务来执行。  

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/13932958-dbceae46ea7c15c3-163056705416727.png)

#### .2. 缺点

1. parallelStream是`线程不安全的`；加锁、使用线程安全的集合或者采用`collect()`或者`reduce()`操作就是满足线程安全的了。

```java
List<Integer> values = new ArrayList<>();
for (int i = 0; i < 10000; i++) {
    values.add(i);
}
List<Integer> collect = values.stream().parallel().collect(Collectors.toList());
System.out.println(collect.size());
```

2) parallelStream `适用的场景是CPU密集型的`，只是做到别浪费CPU，假如本身电脑CPU的负载很大，那还到处用并行流，那并不能起到作用；
   - I/O密集型 磁盘I/O、网络I/O都属于I/O操作，这部分操作是较少消耗CPU资源，一般并行流中不适用于I/O密集型的操作，就比如使用并流行进行大批量的消息推送，涉及到了大量I/O，使用并行流反而慢了很多
   - CPU密集型 计算类型就属于CPU密集型了，这种操作并行流就能提高运行效率。

3) `不要在多线程中使用parallelStream`，原因同上类似，大家都抢着CPU是没有提升效果，反而还会加大线程切换开销；
4) `会带来不确定性`，请确保每条处理无状态且没有关联；
5) `考虑NQ模型`：N可用的数据量，Q针对每个数据元素执行的计算量，乘积 N * Q 越大，就越有可能获得并行提速。N * Q>10000（大概是集合大小超过1000） 就会获得有效提升；
6) parallelStream是创建一个并行的Stream,而且它的并行操作是*不具备线程传播性*的,所以是无法获取ThreadLocal创建的线程变量的值；
7) **在使用并行流的时候是无法保证元素的顺序的，也就是即使你用了同步集合也只能保证元素都正确但无法保证其中的顺序**；
8) `lambda的执行并不是瞬间完成的，所有使用parallel stream的程序都有可能成为阻塞程序的源头`，并且在执行过程中程序中的其他部分将无法访问这些workers，这意味着任何依赖parallel streams的程序在什么别的东西占用着common ForkJoinPool时将会变得不可预知并且暗藏危机。

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/streampipline/  

