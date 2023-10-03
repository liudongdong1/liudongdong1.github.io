# Android_RxAndroid


### 1. RxJava

|          角色          | 作用                                    |
| :--------------------: | --------------------------------------- |
| 被观察者（Observable） | 产生事件                                |
|   观察者（Observer）   | 接收事件，并给出响应动作                |
|   订阅（Subscribe）    | 连接 被观察者 & 观察者， 相当于注册监听 |
|     事件（Event）      | 被观察者 & 观察者 沟通的载体            |

- 调用dispose()并不会导致上游不再继续发送事件, 上游会继续发送剩余的事件.

```java
public static void case1() {
      /**
   * Called for each Observer that subscribes. * @param e the safe emitter instance, never null
   * @throws Exception on error
   */
    Observable.create(new ObservableOnSubscribe<Integer>() {
        @Override   // 产生事件
        public void subscribe(ObservableEmitter<Integer> emitter) throws Exception {
            emitter.onNext(1);
            emitter.onNext(2);
            emitter.onNext(3);
            emitter.onComplete();
        }
    }).subscribe(new Observer<Integer>() {
        private Disposable mDisposable;
        private int mCount = 0;

        @Override
        public void onSubscribe(Disposable d) {
            Log.d(TAG, "onSubscribe");
            mDisposable = d;
        }

        @Override
        public void onNext(Integer value) {
            Log.d(TAG, "onNext: value = " + value);
            mCount++;
            if (mCount == 2) {
                Log.d(TAG, "dispose");
                mDisposable.dispose();
                Log.d(TAG, "isDisposed : " + mDisposable.isDisposed());
            }
        }

        @Override
        public void onError(Throwable e) {
            Log.d(TAG, "onError: " + e.toString());
        }

        @Override
        public void onComplete() {
            Log.d(TAG, "onComplete");
        }
    });
}
```

- subscribe 方法重载

```java
public final Disposable subscribe() {}
public final Disposable subscribe(Consumer<? super T> onNext) {}
public final Disposable subscribe(Consumer<? super T> onNext, Consumer<? super Throwable> onError) {} 
public final Disposable subscribe(Consumer<? super T> onNext, Consumer<? super Throwable> onError, Action onComplete) {}
public final Disposable subscribe(Consumer<? super T> onNext, Consumer<? super Throwable> onError, Action onComplete, Consumer<? super Disposable> onSubscribe) {}
public final void subscribe(Observer<? super T> observer) {}
```

### 2. 线程调度

| 类型                           | 含义                  | 应用场景                         |
| ------------------------------ | --------------------- | -------------------------------- |
| Schedulers.immediate()         | 当前线程 = 不指定线程 | 默认                             |
| AndroidSchedulers.mainThread() | Android主线程         | 操作UI                           |
| Schedulers.newThread()         | 常规新线程            | 耗时等操作                       |
| Schedulers.io()                | io操作线程            | 网络请求、读写文件等io密集型操作 |
| Schedulers.computation()       | CPU计算操作线程       | 大量计算操作                     |

采用 `RxJava`内置的线程调度器（ `Scheduler` ），即通过 功能性操作符`subscribeOn（）` & `observeOn（）`实现。

```java
public static void case4() {
    Observable.create(new ObservableOnSubscribe<Integer>() {
        @Override
        public void subscribe(ObservableEmitter<Integer> emitter) throws Exception {
            Log.d(TAG, "subscribe: thread = " + Thread.currentThread());
            emitter.onNext(1);
            emitter.onComplete();
        }
    })
        .subscribeOn(Schedulers.io())   //多次指定被观察者 生产事件的线程，则只有第一次指定有效，其余的指定线程无效。
        .observeOn(AndroidSchedulers.mainThread()) //多次指定观察者 接收 & 响应事件的线程，则每次指定均有效，即每指定一次，就会进行一次线程的切换。
        .subscribe(new Consumer<Integer>() {
            @Override
            public void accept(Integer integer) throws Exception {
                Log.d(TAG, "accept: thread = " + Thread.currentThread());
            }
        });
}
```

#### .1. 数据库读取

```java
public Observable<List<Record>> readAllRecords() {
    return Observable.create(new ObservableOnSubscribe<List<Record>>() {
        @Override
        public void subscribe(ObservableEmitter<List<Record>> emitter) throws Exception {
            Cursor cursor = null;
            try {
                cursor = getReadableDatabase().rawQuery("select * from " + TABLE_NAME, new String[]{});
                List<Record> result = new ArrayList<>();
                while (cursor.moveToNext()) {
                    result.add(Db.Record.read(cursor));
                }
                emitter.onNext(result);
                emitter.onComplete();
            } finally {
                if (cursor != null) {
                    cursor.close();
                }
            }
        }
    }).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
}


readAllRecords().subscribe(new Consumer<List<Record>>() {
    @Override
    public void accept(List<Record> recordList) throws Exception {

    }
})
```

### 3. Rx 操作符

#### 1. 创建操作符

- https://www.cnblogs.com/andy-loong/p/11307436.html

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190809194736255-190276033.png)

```java
// 下列方法一般用于测试使用

<-- empty()  -->
// 该方法创建的被观察者对象发送事件的特点：仅发送Complete事件，直接通知完成
Observable observable1=Observable.empty(); 
// 即观察者接收后会直接调用onCompleted（）

<-- error()  -->
// 该方法创建的被观察者对象发送事件的特点：仅发送Error事件，直接通知异常
// 可自定义异常
Observable observable2=Observable.error(new RuntimeException())
// 即观察者接收后会直接调用onError（）

<-- never()  -->
// 该方法创建的被观察者对象发送事件的特点：不发送任何事件
Observable observable3=Observable.never();
// 即观察者接收后什么都不调用
```

- defer: 直到有观察者（`Observer` ）订阅时，才动态创建被观察者对象（`Observable`） & 发送事件

##### 1. 轮询器

```java
Disposable mDisposable;
//开启轮询
public void autoLoop() {
    if (mDisposable == null || mDisposable.isDisposed()) {
        Observable.interval(0, 5, TimeUnit.SECONDS)
            .subscribeOn(Schedulers.computation())
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe(new Consumer<Long>() {
                @Override
                public void accept(Long aLong) throws Exception {

                }
            });
    }
}

//关闭轮询
public void stopLoop() {
    if (mDisposable != null && !mDisposable.isDisposed()) {
        mDisposable.dispose();
        mDisposable = null;
    }
}
```

##### 2. 定时器

```java
//一段时间之后再做一些事情
public void timer() {
    Observable.timer(5, TimeUnit.SECONDS)
        .observeOn(AndroidSchedulers.mainThread())
        .subscribe(new Consumer<Long>() {
            @Override
            public void accept(Long aLong) throws Exception {

            }
        });
}
```

#### 2. 变换操作符

- https://www.cnblogs.com/andy-loong/p/11308956.html

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190809194625132-1468988107.png)

```java
public static void map() {
    Observable.just("I", "am", "RxJava")
        .map(new Function<String, Integer>() {
            @Override
            public Integer apply(String s) throws Exception {
                return s.length();
            }
        }).subscribe(new Observer<Integer>() {
        @Override
        public void onSubscribe(Disposable d) {
            Log.d(TAG, "onSubscribe");
        }

        @Override
        public void onNext(Integer value) {
            Log.d(TAG, "onNext: value = " + value);
        }

        @Override
        public void onError(Throwable e) {
            Log.d(TAG, "onError: " + e.toString());
        }

        @Override
        public void onComplete() {
            Log.d(TAG, "onComplete");
        }
    });
}
```

flatMap:

- 为事件序列中每个事件都创建一个 `Observable` 对象；
- 将对每个 原始事件 转换后的 新事件 都放入到对应 `Observable`对象；
- 将新建的每个`Observable` 都合并到一个 新建的、总的`Observable` 对象；
- 新建的、总的`Observable` 对象 将 新合并的事件序列 发送给观察者（`Observer`）
- 新合并生成的事件序列顺序是无序的，即 与旧序列发送事件的顺序无关

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190806150642728-1362518972.png)

```java
public static void flatMap() {
    Observable.just("I", "am", "RxJava")
        .flatMap(new Function<String, ObservableSource<Integer>>() {
            @Override
            public ObservableSource<Integer> apply(String s) throws Exception {
                int length = s.length();
                ArrayList<Integer> num = new ArrayList<>();
                for (int i = 0; i < length; i++) {
                    num.add(i);
                }
                return Observable.fromIterable(num);
            }
        }).subscribe(new Observer<Integer>() {
        @Override
        public void onSubscribe(Disposable d) {
            Log.d(TAG, "onSubscribe");
        }

        @Override
        public void onNext(Integer value) {
            Log.d(TAG, "onNext: value = " + value);
        }

        @Override
        public void onError(Throwable e) {
            Log.d(TAG, "onError: " + e.toString());
        }

        @Override
        public void onComplete() {
            Log.d(TAG, "onComplete");
        }
    });
}
```

#### 3. 功能性操作符

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190809194255877-1255003866.png)

```java
public static void onErrorReturn() {
    Observable.create(new ObservableOnSubscribe<Integer>() {
        @Override
        public void subscribe(ObservableEmitter<Integer> emitter) throws Exception {
            emitter.onNext(1);
            emitter.onNext(2);
            emitter.onError(new NullPointerException("null point exception"));
            emitter.onNext(3);
        }
    }).onErrorReturn(new Function<Throwable, Integer>() {
        @Override
        public Integer apply(Throwable throwable) throws Exception {
            Log.d(TAG, "onErrorReturn");
            return 666;
        }
    }).subscribe(new Observer<Integer>() {
        @Override
        public void onSubscribe(Disposable d) {
            Log.d(TAG, "onSubscribe");
        }

        @Override
        public void onNext(Integer value) {
            Log.d(TAG, "onNext: value = " + value);
        }

        @Override
        public void onError(Throwable e) {
            Log.d(TAG, "onError: " + e.toString());
        }

        @Override
        public void onComplete() {
            Log.d(TAG, "onComplete");
        }
    });
}
```

#### 4. 过滤性操作符

- https://www.cnblogs.com/andy-loong/p/11328568.html

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190809193847317-29445171.png)

```java
public static void filter() {
    Observable.just(1, 2, 3, 7, 6, 9)
        .filter(new Predicate<Integer>() {
            @Override
            public boolean test(Integer integer) throws Exception {
                return integer > 5;
            }
        }).subscribe(new Observer<Integer>() {
        @Override
        public void onSubscribe(Disposable d) {
            Log.d(TAG, "onSubscribe");
        }

        @Override
        public void onNext(Integer value) {
            Log.d(TAG, "onNext: value = " + value);
        }

        @Override
        public void onError(Throwable e) {
            Log.d(TAG, "onError: " + e.toString());
        }

        @Override
        public void onComplete() {
            Log.d(TAG, "onComplete");
        }
    });
}
```

##### 1. 功能防抖

```java
public static void throttleFirst() {
    Button button = null;
    RxView.clicks(button)
        .throttleFirst(1, TimeUnit.SECONDS)
        .subscribe(new Consumer<Object>() {
            @Override
            public void accept(Object o) throws Exception {
                //跳转
            }
        });
}
```

##### 2. 联想搜索优化

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190812165558707-1208776469.png)

```java
public static void search() {
    EditText editText = null;
    RxTextView.textChanges(editText)
        .debounce(1, TimeUnit.SECONDS)
        .skip(1)
        .subscribe(new Consumer<CharSequence>() {
            @Override
            public void accept(CharSequence charSequence) throws Exception {
                //开始搜索
            }
        });
}
```

#### 5. 布尔操作符

- https://www.cnblogs.com/andy-loong/p/11329331.html

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190812115023836-2082195042.png)

```java
public static void all() {
    Observable.just(1, 2, 3, 4, 5, 10)
        .all(new Predicate<Integer>() {
            @Override
            public boolean test(Integer integer) throws Exception {
                return integer >= 10;
            }
        })
        .subscribe(new SingleObserver<Boolean>() {
            @Override
            public void onSubscribe(Disposable d) {
                Log.d(TAG, "onSubscribe");
            }

            @Override
            public void onSuccess(Boolean value) {
                Log.d(TAG, "onSuccess: value = " + value);
            }

            @Override
            public void onError(Throwable e) {
                Log.d(TAG, "onError: " + e.toString());
            }
        });
}
```

#### 6. 组合操作符

- https://www.cnblogs.com/andy-loong/p/11310218.html

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1113592-20190809194432313-1619474051.png)

##### 1. 联合判断

```java
Observable<CharSequence> nameObservable = RxTextView.textChanges(name).skip(1);
Observable<CharSequence> ageObservable = RxTextView.textChanges(age).skip(1);
Observable<CharSequence> jobObservable = RxTextView.textChanges(job).skip(1);

/*
         * 通过combineLatest（）合并事件 & 联合判断
         **/
Observable.combineLatest(nameObservable,ageObservable,jobObservable,new Function3<CharSequence, CharSequence, CharSequence,Boolean>() {
    @Override
    public Boolean apply(@NonNull CharSequence name, @NonNull CharSequence age, @NonNull CharSequence job) throws Exception {
        //1. 姓名
        boolean isUserNameValid = !TextUtils.isEmpty(name) 
            // 2. 年龄信息
            boolean isUserAgeValid = !TextUtils.isEmpty(age);
        // 3. 职业信息
        boolean isUserJobValid = !TextUtils.isEmpty(job) ;

        return isUserNameValid && isUserAgeValid && isUserJobValid;
    }

}).subscribe(new Consumer<Boolean>() {
    @Override
    public void accept(Boolean s) throws Exception { 
        Log.e(TAG, "提交按钮是否可点击： " + s);
        list.setEnabled(s);
    }
});
```

##### 2. 数据源合并

```java
/ 用于存放最终展示的数据
    String result = "数据源来自 = " ;

/*
         * 设置第1个Observable：通过网络获取数据
         * 此处仅作网络请求的模拟
         **/
Observable<String> network = Observable.just("网络");

/*
         * 设置第2个Observable：通过本地文件获取数据
         * 此处仅作本地文件请求的模拟
         **/
Observable<String> file = Observable.just("本地文件");


/*
         * 通过merge（）合并事件 & 同时发送事件
         **/
Observable.merge(network, file)
    .subscribe(new Observer<String>() {
        @Override
        public void onSubscribe(Disposable d) {

        }

        @Override
        public void onNext(String value) {
            Log.d(TAG, "数据源有： "+ value  );
            result += value + "+";
        }

        @Override
        public void onError(Throwable e) {
            Log.d(TAG, "对Error事件作出响应");
        }

        // 接收合并事件后，统一展示
        @Override
        public void onComplete() {
            Log.d(TAG, "获取数据完成");
            Log.d(TAG,  result  );
        }
    });
```

##### 3. 网络、缓存

```java
// 该2变量用于模拟内存缓存 & 磁盘缓存中的数据
String memoryCache = null;
String diskCache = "从磁盘缓存中获取数据";

/*
         * 设置第1个Observable：检查内存缓存是否有该数据的缓存
         **/
Observable<String> memory = Observable.create(new ObservableOnSubscribe<String>() {
    @Override
    public void subscribe(ObservableEmitter<String> emitter) throws Exception {

        // 先判断内存缓存有无数据
        if (memoryCache != null) {
            // 若有该数据，则发送
            emitter.onNext(memoryCache);
        } else {
            // 若无该数据，则直接发送结束事件
            emitter.onComplete();
        }

    }
});

/*
         * 设置第2个Observable：检查磁盘缓存是否有该数据的缓存
         **/
Observable<String> disk = Observable.create(new ObservableOnSubscribe<String>() {
    @Override
    public void subscribe(ObservableEmitter<String> emitter) throws Exception {

        // 先判断磁盘缓存有无数据
        if (diskCache != null) {
            // 若有该数据，则发送
            emitter.onNext(diskCache);
        } else {
            // 若无该数据，则直接发送结束事件
            emitter.onComplete();
        }

    }
});

/*
         * 设置第3个Observable：通过网络获取数据
         **/
Observable<String> network = Observable.just("从网络中获取数据");
// 此处仅作网络请求的模拟


/*
         * 通过concat（） 和 firstElement（）操作符实现缓存功能
         **/

// 1. 通过concat（）合并memory、disk、network 3个被观察者的事件（即检查内存缓存、磁盘缓存 & 发送网络请求）
//    并将它们按顺序串联成队列
Observable.concat(memory, disk, network)
    // 2. 通过firstElement()，从串联队列中取出并发送第1个有效事件（Next事件），即依次判断检查memory、disk、network
    .firstElement()
    // 即本例的逻辑为：
    // a. firstElement()取出第1个事件 = memory，即先判断内存缓存中有无数据缓存；由于memoryCache = null，即内存缓存中无数据，所以发送结束事件（视为无效事件）
    // b. firstElement()继续取出第2个事件 = disk，即判断磁盘缓存中有无数据缓存：由于diskCache ≠ null，即磁盘缓存中有数据，所以发送Next事件（有效事件）
    // c. 即firstElement()已发出第1个有效事件（disk事件），所以停止判断。

    // 3. 观察者订阅
    .subscribe(new Consumer<String>() {
        @Override
        public void accept( String s) throws Exception {
            Log.d(TAG,"最终获取的数据来源 =  "+ s);
        }
    });
```

### Resource

- https://www.cnblogs.com/andy-loong/p/11340248.html
- todo?  flowable, backpressure，single，completable等模式地具体处理策略
- https://blog.csdn.net/wdd1324/article/details/70761514  Rx 相关项目


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_rxandroid/  

