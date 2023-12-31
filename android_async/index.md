# Android_Async


> - 只能在 UI 线程操作 UI 视图，不能在子线程中操作，否则报错：Only the original thread that created a view hierarchy can touch its views
> - 不能在 UI 线程中进行耗时操作，否则会阻塞 UI 线程，引起 ANR、卡顿等问题。

### 1. Thread 方式

#### .1. 继承 THread 类

> Java 的单继承限制，想通过 Thread 实现多线程，就只能继承 Thread 类，不可继承其他类。不能跟新UI操作

```java
new Thread(new Runnable() {
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void run() {
        ArrayList<FlexData> flexDataArrayList= (ArrayList<FlexData>) flexWindow.getFlexData();
        sqLiteOperation.addBatch(flexDataArrayList);
        Log.i(TAG,"performSave operation: OK");
    }
}).start();
```

```java
class MyThread extends Thread {
    @Override
    public void run() {
        super.run();
    }
}

private void testThread(){
    Thread thread = new MyThread();
    thread.start();
}
```

#### .2. 实现Runnable接口

> 继承 Runnable 接口后，想要启动线程，需要把该类的对象作为参数，传递给 Thread 的构造函数，并使用 Thread 类的实例方法 start 来启动。

```java
public class TestThread extends A implements Runnable {
    public void run() {
        ……    
    }
}
//启动线程
TestThread testThread = new TestThread();
Thread thread = new Thread(testThread);
thread.start();
```

#### .3. 实现Callable接口

> 如果想要`执行的线程有返回`，怎么处理呢？这时应该使用 Callable 接口了，与 Runnable 相比，Callable 可以有返回值，返回值通过 FutureTask 进行封装。

```java
public class MyCallable implements Callable<Integer> {
    public Integer call() {
        return 111;
    }
}
public static void main(String[] args) throws ExecutionException, InterruptedException {
    MyCallable mc = new MyCallable();
    FutureTask<Integer> ft = new FutureTask<>(mc);
    Thread thread = new Thread(ft);
    thread.start();
    System.out.println(ft.get());
}
```

### 2. Thread+Looper+handler

> Handler是发送消息g给Looper，Looper是封装消息的载体，Looper中封装了一个MessageQueue，Looper.loop()是把消息MessageQueue中的消息返还给Handler.
>
> - Handler 负责消息的发送与接收处理。
> - Message 负责消息的封装，他本身可以看做消息的载体。
> - MessageQueue：是一个消息队列，所有需要发送的消息用类似于链表的形式进行存储，并且依据于消息消费的时间为标志确定存储位置。
> - Looper：进行消息循环与消息分发。

> - 代码规范性较差，不易维护。
> - 每次操作都会开启一个匿名线程，系统开销较大。

#### .1. 使用handler发送消息来处理

> **Message创建方式一：Message message = new Message()**
> 这种方法很常见，就是常见的创建对象的方式。每次需要Message对象的时候都创建一个新的对象，每次都要去堆内存开辟对象存储空间，对象使用完后，jvm又要去对这个废弃的对象进行垃圾回收。
>
> **Message创建方式二：Message message = handler.obtainMessage()**
> 上面说到，这种方式去获取Message，效率会更高！

```java
/**
     * @function: 进入蓝牙数据传输处理，通过handler进行数据的交互
     * */
private void setupChat() {
    Log.i(TAG, "setupChat()");
    receiveMessage.setMovementMethod(ScrollingMovementMethod
                                     .getInstance());// 使TextView接收区可以滚动
    recognizeResult.setText("None");
    // 初始化BluetoothChatService以执行app_incon_bluetooth连接
    mChatService = new BluetoothChatService(this, mHandler);
}
```

```java
Handler mHandler = new Handler(){
    @Override
    public void handleMessage(Message msg){
        if(msg.what == 1){
            textView.setText("Task Done!!");
        }
    }
};
mRunnable = new Runnable() {
    @Override
    publicvoid run() {
        SystemClock.sleep(1000);    // 耗时处理
        mHandler.sendEmptyMessage(1);    //mHandler.obtainMessage(BluetoothChat.MESSAGE_STATE_CHANGE, state, -1).sendToTarget();
    }
};
private void startTask(){
    new Thread(mRunnable).start();
}
```

#### .2. 使用post 方法延迟更新ui

```java
new Thread() {
    @Override
    public void run() {
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        handler1.post(new Runnable() {
            @Override
            public void run() {
                tv1.setText(String.format(mContext.getResources().getString(R.string.tv_string, 1)));
            }
        });
    }
}.start();
```

#### .3. 使用RunOnUiThread

```java
runOnUiThread(new Runnable() {
    @Override
    public void run() {
        tv3.setText(String.format(mContext.getResources().getString(R.string.tv_string, 3)));
    }
});
```

#### .4. 使用View.Post

```java
tv4.post(new Runnable() {
    @Override
    public void run() {
        tv4.setText(String.format(mContext.getResources().getString(R.string.tv_string, 4)));
    }
});
```

### 3. AsyncTask

> 较为轻量级的异步类，封装了 FutureTask 的线程池、ArrayDeque 和 Handler 进行调度。AsyncTask 主要用于后台与界面持续交互。AsyncTask 的几个主要方法中，`doInBackground 方法运行在子线程`，execute、onPreExecute、onProgressUpdate、onPostExecute 这几个方法都是在 UI 线程运行的。
>
> AsyncTask　<Params, Progress, Result>
>
> - Params: 这个泛型指定的是我们传递给异步任务执行时的`参数的类型`。
> - Progress: 这个泛型指定的是我们的异步任务在`执行的时候将执行的进度返回给UI线程的参数的类型`。
> - Result: 这个`泛型指定的异步任务执行完后返回给UI线程的结果的类型`。

> - AsyncTask 的实例必须在 UI Thread 中创建。
> - 只能在 UI 线程中调用 AsyncTask 的 execute 方法。
> - AsyncTask 被重写的四个方法是系统自动调用的,不应手动调用。
> - 每个 AsyncTask 只能被执行一次，多次执行会引发异常。
> - AsyncTask 的四个方法，`只有 doInBackground 方法是运行在其他线程中`,其他三个方法都运行在 UI 线程中，也就说其他三个方法都可以进行 UI 的更新操作。
> - AsyncTask `默认是串行执行`，如果需要并行执行，使用接口 executeOnExecutor 方法。
>   - 串行：`execute(Params... params)/execute(Runnable runnable)`
>   - 并行：`executeOnExecutor(Executor exec, Params... params)`

```java
private class DownloadFilesTask extends AsyncTask<URL, Integer, Long> {
     protected Long doInBackground(URL... urls) {
         int count = urls.length;
         long totalSize = 0;
         for (int i = 0; i < count; i++) {
             totalSize += Downloader.downloadFile(urls[i]);
             publishProgress((int) ((i / (float) count) * 100));
             // Escape early if cancel() is called
             if (isCancelled()) break;
         }
         return totalSize;
     }
     protected void onProgressUpdate(Integer... progress) {
         setProgressPercent(progress[0]);
     }
     protected void onPostExecute(Long result) {
         showDialog("Downloaded " + result + " bytes");
     }
}
new DownloadFilesTask().execute(url1, url2, url3);
```

```java
button1.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        /**
                 * 步骤3：手动调用execute(Params... params) 从而执行异步线程任务
                 * 注：
                 *    a. 必须在UI线程中调用
                 *    b. 同一个AsyncTask实例对象只能执行1次，若执行第2次将会抛出异常
                 *    c. 执行任务中，系统会自动调用AsyncTask的一系列方法：onPreExecute() 、doInBackground()、onProgressUpdate() 、onPostExecute()
                 *    d. 不能手动调用上述方法
                 */
        myTask = new MyTask(textView, progressBar);
        myTask.execute();   // 执行该方法

    }
});
button2.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // 取消一个正在执行的任务,onCancelled方法将会被调用
        myTask.cancel(true);
    }
});
```

```java
package com.vision.myapplicationtest;

import android.os.AsyncTask;
import android.util.Log;
import android.widget.ProgressBar;
import android.widget.TextView;

public class MyTask extends AsyncTask<String, Integer, String> {

    private static final String TAG = "MyTask";

    TextView textView;
    ProgressBar progressBar;

    public MyTask() {
    }

    public MyTask(TextView textView, ProgressBar progressBar) {
        this.textView = textView;
        this.progressBar = progressBar;
    }

    // 方法1：onPreExecute（）
    // 作用：执行 线程任务前的操作
    @Override
    protected void onPreExecute() {
        textView.setText("加载中...");
    }

    // 方法2：doInBackground（）
    // 作用：接收输入参数、执行任务中的耗时操作、返回 线程任务执行的结果
    // 此处通过计算从而模拟“加载进度”的情况
    @Override
    protected String doInBackground(String... strings) {
        try {
            int count = 0;
            int length = 1;
            while (count < 99) {
                count += length;
                // 可调用publishProgress（）显示进度, 之后将执行onProgressUpdate（）
                publishProgress(count);
                // 模拟耗时任务
                Thread.sleep(5);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    // 方法3：onProgressUpdate（）
    // 作用：在主线程 显示线程任务执行的进度
    @Override
    protected void onProgressUpdate(Integer... progresses) {
        progressBar.setProgress(progresses[0]);
        textView.setText("loading..." + progresses[0] + "%");
    }

    // 方法4：onPostExecute（）
    // 作用：接收线程任务执行结果、将执行结果显示到UI组件
    @Override
    protected void onPostExecute(String s) {
        // 执行完毕后，则更新UI
        textView.setText("加载完毕");
    }

    // 方法5：onCancelled()
    // 作用：将异步任务设置为：取消状态
    @Override
    protected void onCancelled() {
        textView.setText("已取消");
        progressBar.setProgress(0);
    }

}
```

#### 1. [源码](https://cs.android.com/android/platform/superproject/+/master:frameworks/base/core/java/android/os/AsyncTask.java;l=199?q=AsyncTask&sq=&ss=android%2Fplatform%2Fsuperproject:frameworks%2F)

```java
public abstract class AsyncTask<Params, Progress, Result> {
    private static final int CORE_POOL_SIZE = 1;
    private static final int MAXIMUM_POOL_SIZE = 20;
    private static final int BACKUP_POOL_SIZE = 5;
    private static final int KEEP_ALIVE_SECONDS = 3;

    private static final ThreadFactory sThreadFactory = new ThreadFactory() {
        private final AtomicInteger mCount = new AtomicInteger(1);

        public Thread newThread(Runnable r) {
            return new Thread(r, "AsyncTask #" + mCount.getAndIncrement());
        }
    };
    public static final Executor THREAD_POOL_EXECUTOR;

    static {
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
            CORE_POOL_SIZE, MAXIMUM_POOL_SIZE, KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
            new SynchronousQueue<Runnable>(), sThreadFactory);
        threadPoolExecutor.setRejectedExecutionHandler(sRunOnSerialPolicy);
        THREAD_POOL_EXECUTOR = threadPoolExecutor;
    }
    /**
     * Creates a new asynchronous task. This constructor must be invoked on the UI thread.
     *
     * @hide
     */
    public AsyncTask(@Nullable Looper callbackLooper) {
        mHandler = callbackLooper == null || callbackLooper == Looper.getMainLooper()
            ? getMainHandler()
            : new Handler(callbackLooper);

        mWorker = new WorkerRunnable<Params, Result>() {
            public Result call() throws Exception {
                mTaskInvoked.set(true);
                Result result = null;
                try {
                    Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
                    //noinspection unchecked
                    result = doInBackground(mParams);
                    Binder.flushPendingCommands();
                } catch (Throwable tr) {
                    mCancelled.set(true);
                    throw tr;
                } finally {
                    postResult(result);
                }
                return result;
            }
        };

        mFuture = new FutureTask<Result>(mWorker) {
            @Override
            protected void done() {
                try {
                    postResultIfNotInvoked(get());
                } catch (InterruptedException e) {
                    android.util.Log.w(LOG_TAG, e);
                } catch (ExecutionException e) {
                    throw new RuntimeException("An error occurred while executing doInBackground()",
                                               e.getCause());
                } catch (CancellationException e) {
                    postResultIfNotInvoked(null);
                }
            }
        };
    }
    private static class SerialExecutor implements Executor {
        final ArrayDeque<Runnable> mTasks = new ArrayDeque<Runnable>();
        Runnable mActive;

        public synchronized void execute(final Runnable r) {
            mTasks.offer(new Runnable() {
                public void run() {
                    try {
                        r.run();
                    } finally {
                        scheduleNext();
                    }
                }
            });
            if (mActive == null) {
                scheduleNext();
            }
        }

        protected synchronized void scheduleNext() {
            if ((mActive = mTasks.poll()) != null) {
                THREAD_POOL_EXECUTOR.execute(mActive);
            }
        }
    }
}
```

- https://redspider110.github.io/2017/11/27/0024-async-task/ todo

### 4. HandlerThread

> Handler有多个重载版本，但可供我们使用版本的async参数都是默认值false。
> 首先根据布尔值FIND_POTENTIAL_LEAKS判断是否需要进行泄漏的检测，默认是false。
> 将Looper.myLooper()赋值给mLooper对象。
> 将looper.mQueue赋值给mQueue对象。
> 将参数callback、async赋值给相应属性。

```java
public Handler() {//版本1
    this(null, false);
}
public Handler(@Nullable Callback callback) {//版本2
    this(callback, false);
}
public Handler(@NonNull Looper looper) {//版本3
    this(looper, null, false);
}
public Handler(@NonNull Looper looper, @Nullable Callback callback) {//版本4
    this(looper, callback, false);
}
@UnsupportedAppUsage
public Handler(boolean async) {//版本5。不支持App使用
    this(null, async);
}
public Handler(@Nullable Callback callback, boolean async) {//版本6。主要实现逻辑在这里
    if (FIND_POTENTIAL_LEAKS) {
        final Class<? extends Handler> klass = getClass();
        if ((klass.isAnonymousClass() || klass.isMemberClass() || klass.isLocalClass()) &&
            (klass.getModifiers() & Modifier.STATIC) == 0) {
            Log.w(TAG, "The following Handler class should be static or leaks might occur: " +
                  klass.getCanonicalName());
        }
    }

    mLooper = Looper.myLooper();
    if (mLooper == null) {
        throw new RuntimeException(
            "Can't create handler inside thread " + Thread.currentThread()
            + " that has not called Looper.prepare()");
    }
    mQueue = mLooper.mQueue;
    mCallback = callback;
    mAsynchronous = async;
}
@UnsupportedAppUsage
public Handler(@NonNull Looper looper, @Nullable Callback callback, boolean async) {//版本7。不支持App直接使用，系统内部使用
    mLooper = looper;
    mQueue = looper.mQueue;
    mCallback = callback;
    mAsynchronous = async;
}
```

### 5. IntentService

> IntentService 继承自 Service 类，用于启动一个异步服务任务，它的内部是通过 HandlerThread 来实现异步处理任务的。

#### .1. [demo 讲解](https://juejin.cn/post/6844903847027015693)

##### 1. 定义 `IntentService`的子类

> 传入线程名称、复写`onHandleIntent()`方法

```java
public class myIntentService extends IntentService {
  /** 
    * 在构造函数中传入线程名字
    **/  
    public myIntentService() {
        // 调用父类的构造函数
        // 参数 = 工作线程的名字
        super("myIntentService");
    }
   /** 
     * 复写onHandleIntent()方法
     * 根据 Intent实现 耗时任务 操作
     **/  
    @Override
    protected void onHandleIntent(Intent intent) {
        // 根据 Intent的不同，进行不同的事务处理
        String taskName = intent.getExtras().getString("taskName");
        switch (taskName) {
            case "task1":
                Log.i("myIntentService", "do task1");
                break;
            case "task2":
                Log.i("myIntentService", "do task2");
                break;
            default:
                break;
        }
    }
    @Override
    public void onCreate() {
        Log.i("myIntentService", "onCreate");
        super.onCreate();
    }
   /** 
     * 复写onStartCommand()方法
     * 默认实现 = 将请求的Intent添加到工作队列里
     **/  
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.i("myIntentService", "onStartCommand");
        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public void onDestroy() {
        Log.i("myIntentService", "onDestroy");
        super.onDestroy();
    }
}
```

##### 2. 在Manifest.xml中注册服务

```xml
<service android:name=".myIntentService">
    <intent-filter >
        <action android:name="cn.scu.finch"/>
    </intent-filter>
</service>
```

##### 3. 在Activity中开启Service服务

```java
public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 同一服务只会开启1个工作线程
        // 在onHandleIntent（）函数里，依次处理传入的Intent请求
        // 将请求通过Bundle对象传入到Intent，再传入到服务里
        // 请求1
        Intent i = new Intent("cn.scu.finch");
        Bundle bundle = new Bundle();
        bundle.putString("taskName", "task1");
        i.putExtras(bundle);
        startService(i);

        // 请求2
        Intent i2 = new Intent("cn.scu.finch");
        Bundle bundle2 = new Bundle();
        bundle2.putString("taskName", "task2");
        i2.putExtras(bundle2);
        startService(i2);

        startService(i);  //多次启动
    }
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211204150235461.png)

### 6. 使用线程池处理异步

利用 Executors 的静态方法 `newCachedThreadPool()`、`newFixedThreadPool()`、`newSingleThreadExecutor()` 及重载形式实例化 `ExecutorService` 接口即得到线程池对象。

- `动态线程池 newCachedThreadPool()`：根据需求创建新线程的，需求多时，创建的就多，需求少时，JVM 自己会慢慢的释放掉多余的线程。
- `固定数量的线程池 newFixedThreadPool()`：内部有个任务阻塞队列，假设线程池里有2个线程，提交了4个任务，那么后两个任务就放在任务阻塞队列了，即使前2个任务 sleep 或者堵塞了，也不会执行后两个任务，除非前2个任务有执行完的。
- `单线程 newSingleThreadExecutor()`：单线程的线程池，这个线程池可以在线程死后（或发生异常时）重新启动一个线程来替代原来的线程继续执行下去。

### 7. Toast&Notification 使用

### 8. Rx模式&Eventhub通信


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_async/  

