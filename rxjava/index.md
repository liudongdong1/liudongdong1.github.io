# RxJava


> Rxjava中最重要的其实是Rx思想，所谓的Rx思想，也就是**响应式编程思想**（下一步随上一步的变化而变化，点菜->下单->做菜）。它很好的将**链式编程风格和异步**结合在一起。（又称他为 **卡片式编程** ）

### 1. 思维导图

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/webp)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmd6aGlibzY2Ng==,size_16,color_FFFFFF,t_70)

### 2. 使用步骤

#### .1 基本观察者模式

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211110093309234.png)

##### .1. 创建被观察者（Observable)

```java
// 1. 创建被观察者 Observable 对象
Observable<Integer> observable = Observable.create(new ObservableOnSubscribe<Integer>() {
    // create() 是 RxJava 最基本的创造事件序列的方法
    // 此处传入了一个 OnSubscribe 对象参数
    // 当 Observable 被订阅时，OnSubscribe 的 call() 方法会自动被调用，即事件序列就会依照设定依次被触发
    // 即观察者会依次调用对应事件的复写方法从而响应事件
    // 从而实现被观察者调用了观察者的回调方法 & 由被观察者向观察者的事件传递，即观察者模式

    // 2. 在复写的subscribe（）里定义需要发送的事件
    @Override
    public void subscribe(ObservableEmitter<Integer> emitter) throws Exception {
        // 通过 ObservableEmitter类对象产生事件并通知观察者
        // ObservableEmitter类介绍
        // a. 定义：事件发射器
        // b. 作用：定义需要发送的事件 & 向观察者发送事件
        emitter.onNext(1);
        emitter.onNext(2);
        emitter.onNext(3);
        emitter.onComplete();
    }
});

<--扩展：RxJava 提供了其他方法用于 创建被观察者对象Observable -->
    // 方法1：just(T...)：直接将传入的参数依次发送出来
    Observable observable = Observable.just("A", "B", "C");
// 将会依次调用：
// onNext("A");
// onNext("B");
// onNext("C");
// onCompleted();

// 方法2：from(T[]) / from(Iterable<? extends T>) : 将传入的数组 / Iterable 拆分成具体对象后，依次发送出来
String[] words = {"A", "B", "C"};
Observable observable = Observable.from(words);
// 将会依次调用：
// onNext("A");
// onNext("B");
// onNext("C");
// onCompleted();
```

##### .2. 创建观察者 （`Observer` ）并 定义响应事件的行为

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211111153406035.png)

```java

<--方式1：采用Observer 接口 -->
    // 1. 创建观察者 （Observer ）对象
    Observer<Integer> observer = new Observer<Integer>() {
    // 2. 创建对象时通过对应复写对应事件方法 从而 响应对应事件

    // 观察者接收事件前，默认最先调用复写 onSubscribe（）
    @Override
    public void onSubscribe(Disposable d) {
        Log.d(TAG, "开始采用subscribe连接");
    }

    // 当被观察者生产Next事件 & 观察者接收到时，会调用该复写方法 进行响应
    @Override
    public void onNext(Integer value) {
        Log.d(TAG, "对Next事件作出响应" + value);
    }

    // 当被观察者生产Error事件& 观察者接收到时，会调用该复写方法 进行响应
    @Override
    public void onError(Throwable e) {
        Log.d(TAG, "对Error事件作出响应");
    }

    // 当被观察者生产Complete事件& 观察者接收到时，会调用该复写方法 进行响应
    @Override
    public void onComplete() {
        Log.d(TAG, "对Complete事件作出响应");
    }
};

<--方式2：采用Subscriber 抽象类 -->
    // 说明：Subscriber类 = RxJava 内置的一个实现了 Observer 的抽象类，对 Observer 接口进行了扩展

    // 1. 创建观察者 （Observer ）对象
    Subscriber<Integer> subscriber = new Subscriber<Integer>() {

    // 2. 创建对象时通过对应复写对应事件方法 从而 响应对应事件
    // 观察者接收事件前，默认最先调用复写 onSubscribe（）
    @Override
    public void onSubscribe(Subscription s) {
        Log.d(TAG, "开始采用subscribe连接");
    }

    // 当被观察者生产Next事件 & 观察者接收到时，会调用该复写方法 进行响应
    @Override
    public void onNext(Integer value) {
        Log.d(TAG, "对Next事件作出响应" + value);
    }

    // 当被观察者生产Error事件& 观察者接收到时，会调用该复写方法 进行响应
    @Override
    public void onError(Throwable e) {
        Log.d(TAG, "对Error事件作出响应");
    }

    // 当被观察者生产Complete事件& 观察者接收到时，会调用该复写方法 进行响应
    @Override
    public void onComplete() {
        Log.d(TAG, "对Complete事件作出响应");
    }
};


<--特别注意：2种方法的区别，即Subscriber 抽象类与Observer 接口的区别 -->
    // 相同点：二者基本使用方式完全一致（实质上，在RxJava的 subscribe 过程中，Observer总是会先被转换成Subscriber再使用）
    // 不同点：Subscriber抽象类对 Observer 接口进行了扩展，新增了两个方法：
    // 1. onStart()：在还未响应事件前调用，用于做一些初始化工作
    // 2. unsubscribe()：用于取消订阅。在该方法被调用后，观察者将不再接收 & 响应事件
    // 调用该方法前，先使用 isUnsubscribed() 判断状态，确定被观察者Observable是否还持有观察者Subscriber的引用，如果引用不能及时释放，就会出现内存泄露
```

##### .3. 通过订阅（`Subscribe`）连接观察者和被观察者

```java
observable.subscribe(observer);   // 注意这里是 被观察者.subscribe(观察者)
 // 或者 observable.subscribe(subscriber)；
<-- Observable.subscribe(Subscriber) 的内部实现 -->

public Subscription subscribe(Subscriber subscriber) {
    subscriber.onStart();
    // 步骤1中 观察者  subscriber抽象类复写的方法，用于初始化工作
    onSubscribe.call(subscriber);
    // 通过该调用，从而回调观察者中的对应方法从而响应被观察者生产的事件
    // 从而实现被观察者调用了观察者的回调方法 & 由被观察者向观察者的事件传递，即观察者模式
    // 同时也看出：Observable只是生产事件，真正的发送事件是在它被订阅的时候，即当 subscribe() 方法执行时
}
```

#### .2 RxJava基于事件流的链式调用

> 注：整体方法调用顺序：观察者.onSubscribe（）> 被观察者.subscribe（）> 观察者.onNext（）>观察者.onComplete() 

```java
// RxJava的链式操作
Observable.create(new ObservableOnSubscribe<Integer>() {
    // 1. 创建被观察者 & 生产事件
    @Override
    public void subscribe(ObservableEmitter<Integer> emitter) throws Exception {
        emitter.onNext(1);
        emitter.onNext(2);
        emitter.onNext(3);
        emitter.onComplete();
    }
}).subscribe(new Observer<Integer>() {
    // 2. 通过通过订阅（subscribe）连接观察者和被观察者
    // 3. 创建观察者 & 定义响应事件的行为
    @Override
    public void onSubscribe(Disposable d) {
        Log.d(TAG, "开始采用subscribe连接");
    }
    // 默认最先调用复写的 onSubscribe（）

    @Override
    public void onNext(Integer value) {
        Log.d(TAG, "对Next事件"+ value +"作出响应"  );
    }

    @Override
    public void onError(Throwable e) {
        Log.d(TAG, "对Error事件作出响应");
    }

    @Override
    public void onComplete() {
        Log.d(TAG, "对Complete事件作出响应");
    }

});
```

### 3. 五种观察者

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211111103134786.png)

#### .1. Observable 

> 1. 创建Observable
> 2. 创建Observer
> 3. 使用subscribe()进行订阅

```java
Observable.just("Hello World!")
    .subscribeBy(
    onComplete = {Log.e("TAG","onComplete")},
    onError = {Log.e("TAG",it.localizedMessage)},
    onNext = {Log.e("TAG",it)}
)
```

#### .2. Flowable

> Flowable是RxJava 2.X新增的被观察者,Flowable可以看成Observable新的实现,它支持背压,在这五种观察者模式中,也只有Flowable支持背压.同时实现Reactive Streams的Publisher接口,Flowable所有的操作符强制支持被压
>
> - Observable的使用场景
>
> 1. 一般处理最大不超过1000条数据,并且几乎不会出现内存溢出
> 2. GUI鼠标事件,基本不会背压
> 3. 处理同步流
>
> - Flowable的使用场景
>
> 1. 处理以某种方式产生超过10KB的元素
> 2. 文件读取与分析
> 3. 读取数据库记录,也是一个阻塞的和基于拉取模式
> 4. 网络I/O流
> 5. 创建一个响应式非阻塞接口

#### .3. Single

> Single 只有onSuccess和onError事件.其中onSuccess用于发射数据,而且只能发射一个数据,后面即使再发射数据也不会做任何处理Single 可以通过toXXX方法转换成Observable,Flowable,Completable,Maybe

#### .4. Completable

> Completable 在创建后,不会发射任何数据,只有onComplete和onError事件,同时Completable并没有map,flatMap等操作符.

#### .5. Maybe

> Maybe 是RxJava 2.X 之后才有的新类型,可以看成是Single和Completable的结合

### 4. Action

> Action是RxJava 的一个接口，常用的有Action0和Action1。
>
> - **Action0**： 它`只有一个方法 call()`，这个方法是无参无返回值的；由于 onCompleted() 方法也是无参无返回值的，因此 Action0 可以被当成一个包装对象，将 `onCompleted() 的内容打包起来将自己作为一个参数传入 subscribe() 以实现不完整定义的回调`。
> -  **Ation1**：它同样只有一个方法 call(T param)，这个方法也无返回值，但有一个参数；与 Action0 同理，由于 onNext(T obj) 和 onError(Throwable error) 也是单参数无返回值的，因此` Action1 可以将 onNext(obj)和 onError(error) 打包起来传入 subscribe() 以实现不完整定义的回调`

```java
Observable observable = Observable.just("Hello", "World");
//处理onNext()中的内容
Action1<String> onNextAction = new Action1<String>() {
    @Override
    public void call(String s) {
        Log.i(TAG, s);
    }
};
//处理onError()中的内容
Action1<Throwable> onErrorAction = new Action1<Throwable>() {
    @Override
    public void call(Throwable throwable) {
    }
};
//处理onCompleted()中的内容
Action0 onCompletedAction = new Action0() {
    @Override
    public void call() {
        Log.i(TAG, "Completed");
    }
};
```

```java
//使用 onNextAction 来定义 onNext()
Observable.just("Hello", "World").subscribe(onNextAction);
//使用 onNextAction 和 onErrorAction 来定义 onNext() 和 onError()
Observable.just("Hello", "World").subscribe(onNextAction, onErrorAction);
//使用 onNextAction、 onErrorAction 和 onCompletedAction 来定义 onNext()、 onError() 和 onCompleted()
Observable.just("Hello", "World").subscribe(onNextAction, onErrorAction, onCompletedAction);
```

### 5. do操作符

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211111104901827.png)

```java
import io.reactivex.Observable;
import org.junit.Test;
public class ExeOrder {
    @Test
    public void orderTester() {
        Observable.just("Hello Tester1")
                .doOnNext(s -> System.out.println("doOnNext：" + s))
                .doFinally(() -> System.out.println("doFinally1"))
                .doAfterTerminate(() -> System.out.println("doAfterTerminate1"))
                .doFinally(() -> System.out.println("doFinally2"))
                .doAfterTerminate(() -> System.out.println("doAfterTerminate2"))
                .subscribe(
                        s -> System.out.println("onNext：" + s),
                        throwable -> System.out.println("onError" + throwable),
                        () -> System.out.println("onComplete"));
        System.out.println("*******调整顺序**********");
        //调整顺序
        Observable.just("Hello Tester2")
                .doAfterTerminate(() -> System.out.println("doAfterTerminate1"))
                .doFinally(() -> System.out.println("doFinally1"))
                .doOnNext(s -> System.out.println("doOnNext：" + s))
                .doAfterTerminate(() -> System.out.println("doAfterTerminate2"))
                .doFinally(() -> System.out.println("doFinally2"))
                .subscribe(
                        s -> System.out.println("onNext：" + s),
                        throwable -> System.out.println("onError" + throwable),
                        () -> System.out.println("onComplete"));
    }
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211111105326052.png)

### 6. 线程控制

#### .1. 默认当前线程

> 因为是在主线程中发起的，所以不管中间map的处理还是Action1的执行都是在主线程中进行的。若是map中有耗时的操作，这样会导致主线程拥塞，这并不是我们想看到的。

```java
Observable.just(student1, student2, student2)
    //使用map进行转换，参数1：转换前的类型，参数2：转换后的类型
    .map(new Func1<Student, String>() {
        @Override
        public String call(Student i) {
            String name = i.getName();//获取Student对象中的name
            return name;//返回name
        }
    })
    .subscribe(new Action1<String>() {
        @Override
        public void call(String s) {
            nameList.add(s);
        }
    });
```

#### .2. Scheduler

> Scheduler：线程控制器，可以指定每一段代码在什么样的线程中执行。
> 模拟一个需求：新的线程发起事件，在主线程中消费

```java
private void rxJavaTest3() {
    Observable.just("Hello", "Word")
        .subscribeOn(Schedulers.newThread())//指定 subscribe() 发生在新的线程
        .observeOn(AndroidSchedulers.mainThread())// 指定 Subscriber 的回调发生在主线程
        .subscribe(new Action1<String>() {
            @Override
            public void call(String s) {
                Log.i(TAG, s);
            }
        });
}
```

- subscribeOn()：指定subscribe() 所发生的线程，即 Observable.OnSubscribe 被激活时所处的线程。或者叫做事件产生的线程。
- observeOn()：指定Subscriber 所运行在的线程。或者叫做事件消费的线程。
- Schedulers.immediate()：直接在当前线程运行，相当于不指定线程。这是默认的 Scheduler。
- Schedulers.newThread()：总是启用新线程，并在新线程执行操作。
- Schedulers.io()： I/O 操作（读写文件、读写数据库、网络信息交互等）所使用的 Scheduler。行为模式和 newThread() 差不多，区别在于 io() 的内部实现是是用一个无数量上限的线程池，可以重用空闲的线程，因此多数情况下 io() 比 newThread() 更有效率。不要把计算工作放在 io() 中，可以避免创建不必要的线程。
- Schedulers.computation()：计算所使用的 Scheduler。这个计算指的是 CPU 密集型计算，即不会被 I/O 等操作限制性能的操作，例如图形的计算。这个 Scheduler 使用的固定的线程池，大小为 CPU 核数。不要把 I/O 操作放在 computation() 中，否则 I/O 操作的等待时间会浪费 CPU。
- AndroidSchedulers.mainThread()：它指定的操作将在 `Android 主线程运行。`

#### .3. 多个线程切换

> - ObserveOn用于切换下游执行线程，可以多次调用，每调用一次会切换一次，`observeOn`方法生成的ObserveOnObserver实例并不会对`onSubscribe`事件做切换线程的操作，

```go
fun threadName(desc: String) {
    println("$desc ${Thread.currentThread().name}")
}
fun main() {
    Observable.create<Int> {
        threadName("subscribe")
        it.onNext(1)
        it.onNext(2)
        it.onComplete()
    }.observeOn(Schedulers.io())
        .subscribe(object : Observer<Int> {
            override fun onComplete() {
                threadName("onComplete")
            }
            override fun onSubscribe(d: Disposable) {
                threadName("onSubscribe")
            }
            override fun onError(e: Throwable) {
                threadName("onError")
            }
            override fun onNext(t: Int) {
                threadName("onNext")
            }
        })
}
```

- 输出

```
onSubscribe main
subscribe main
onNext RxCachedThreadScheduler-1
onNext RxCachedThreadScheduler-1
onComplete RxCachedThreadScheduler-1
```

> - subscribeOn用于上游执行线程，并且多次调用只有第一次会生效；
> - 该段代码中没有调用observeOn所以下游执行线程并没有发生改变，因此上游在子线程中发送一个`onNext`事件过来，下游的`onNext`方法自然也会在子线程中执行

```go
fun main() {
    Observable.create<Int> {
        threadName("subscribe")
        it.onNext(1)
        it.onNext(2)
        it.onComplete()
    }.subscribeOn(Schedulers.io())
    .subscribe(object : Observer<Int> {
        override fun onComplete() {
            threadName("onComplete")
        }
        override fun onSubscribe(d: Disposable) {
            threadName("onSubscribe")
        }
        override fun onError(e: Throwable) {
            threadName("onError")
        }
        override fun onNext(t: Int) {
            threadName("onNext")
        }
    })
}
```

```
onSubscribe main
subscribe RxCachedThreadScheduler-1
onNext RxCachedThreadScheduler-1
onNext RxCachedThreadScheduler-1
onComplete RxCachedThreadScheduler-1
```

### Resource

- https://www.jianshu.com/p/a406b94f3188
- https://cloud.tencent.com/developer/article/1408083

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/rxjava/  

