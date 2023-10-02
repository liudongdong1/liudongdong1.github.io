# EventBus


> 1. **Event：** 事件，可以是任意类型的对象
> 2. **Subscriber：** 事件订阅者，在 **EventBus 3.0** 之前我们必须定义以onEvent开头的那几个方法，分别是 `onEvent` 、`onEventMainThread` 、`onEventBackgroundThread` 和`onEventAsync`，而在3.0之后事件处理的方法名可以随意取，不过需要加上注解@subscribe，并且指定线程模型，默认是POSTING。`EventBus还支持发送黏性事件，就是在发送事件之后再订阅该事件也能收到该事件`，这跟黏性广播类似。
> 3. **Publisher：** 事件发布者，可以在任意线程任意位置发送事件， 直接调用。一般情况下，使用EventBus.getDefault()就可以得到一个EventBus对象，然后再调用post(Object)方法即可

**EventBus3.0有四种线程模型，分别是：**

- **POSTING(默认)：** 表示事件处理函数的线程跟发布事件的线程在同一个线程
- **MAIN：** 表示事件处理函数的线程在主线程(UI)线程，因此在这里不能进行耗时操作
- **BACKGROUND：** 表示事件处理函数的线程在后台线程，因此不能进行UI操作。如果发布事件的线程是主线程(UI线程)，那么事件处理函数将会开启一个后台线程，如果果发布事件的线程是在后台线程，那么事件处理函数就使用该线程
- **ASYNC：** 表示无论事件发布的线程是哪一个，事件处理函数始终会新建一个子线程运行，同样不能进行UI操作

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/78acb11c50ad4c8ebe156b33c11e16f6tplv-k3u1fbpfcp-zoom-in-crop-mark4536000.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/adfb9ac045484820b9c29bdacadb17betplv-k3u1fbpfcp-zoom-in-crop-mark4536000.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/17ad7672f43541f19b632490360200b1tplv-k3u1fbpfcp-zoom-in-crop-mark4536000.png)

### 0. 使用步骤

#### 1. define event

```java
public class MessageEvent {
 
    public final String message;
 
    public MessageEvent(String message) {
        this.message = message;
    }
}
```

#### 2. Prepare subscribers

- be called when an event is posted
- Subscribers also need to **register** themselves to **and unregister** from the bus. Only while subscribers are registered, they will receive events. 

```java
// This method will be called when a MessageEvent is posted (in the UI thread for Toast)
@Subscribe(threadMode = ThreadMode.MAIN)
public void onMessageEvent(MessageEvent event) {
    Toast.makeText(getActivity(), event.message, Toast.LENGTH_SHORT).show();
}
 
// This method will be called when a SomeOtherEvent is posted
@Subscribe
public void handleSomethingElse(SomeOtherEvent event) {
    doSomethingWith(event);
}
```

#### 3. Post events

```java
EventBus.getDefault().post(new MessageEvent("Hello everyone!"));
```

### 1. 案例代码

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setTitle("Subscriber");
    }

    @Override
    protected void onStart() {
        super.onStart();
        EventBus.getDefault().register(this);
    }

    @Override
    protected void onStop() {
        super.onStop();
        EventBus.getDefault().unregister(this);
    }

    // POSTING 模式
    @Subscribe(threadMode = ThreadMode.POSTING)
    public void onPostingEvent(final PostingEvent event){
        String threadInfo = Thread.currentThread().toString();
        runOnUiThread(()->{
            setPublisherThreadInfo(event.threadInfo);
            setSubscriberThreadInfo(threadInfo);
        });
    }

    // MAIN 模式
    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onMainEvent(final MainEvent event){
        String threadInfo = Thread.currentThread().toString();
        runOnUiThread(()->{
            setPublisherThreadInfo(event.threadInfo);
            setSubscriberThreadInfo(threadInfo);
        });
    }

    // MAIN_ORDER 模式
    @Subscribe(threadMode = ThreadMode.MAIN_ORDERED)
    public void onMainOrderEvent(final MainOrderEvent event){
        Log.i(TAG, "onMainOrderEvent: enter @" + SystemClock.uptimeMillis());
        setPublisherThreadInfo(event.threadInfo);
        setSubscriberThreadInfo(Thread.currentThread().toString());
        try {
            TimeUnit.SECONDS.sleep(1);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Log.i(TAG, "onMainOrderEvent: exit @" + SystemClock.uptimeMillis());
    }

    // BACKGROUND 模式
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onBackgroundEvent(final BackgroundEvent event){
        final String threadInfo = Thread.currentThread().toString();
        runOnUiThread(()->{
            setPublisherThreadInfo(event.threadInfo);
            setSubscriberThreadInfo(threadInfo);
        });
    }

    // ASYNC 模式
    @Subscribe(threadMode = ThreadMode.ASYNC)
    public void onAsyncEvent(final AsyncEvent event){
        final String threadInfo = Thread.currentThread().toString();
        runOnUiThread(()->{
            setPublisherThreadInfo(event.threadInfo);
            setSubscriberThreadInfo(threadInfo);
        });
    }

    private void setPublisherThreadInfo(String threadInfo){
        setTextView(R.id.tv_publisher_thread, threadInfo);
    }

    private void setSubscriberThreadInfo(String threadInfo){
        setTextView(R.id.tv_subscriber_thread, threadInfo);
    }

    // Run on UI Thread
    private void setTextView(int resId, String text){
        TextView textView = findViewById(resId);
        textView.setText(text);
        textView.setAlpha(0.5f);
        textView.animate().alpha(1).start();
    }

    public void showDialogFragment(View view) {
        // Display DialogFragment
        PublisherDialogFragment fragment = new PublisherDialogFragment();
        fragment.show(getSupportFragmentManager(), "publisher");
    }
}
```

```java
public class PublisherDialogFragment extends DialogFragment {
    private static final String TAG = "PublisherDialogFragment";
    @NonNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setTitle("Publisher");
        String[] items = new String[]{"Posting", "Main", "MainOrder", "Background", "Async"};
        builder.setItems(items, (dialog, which) -> {
            switch (which){
                case 0: // Posting Mode
                    if(Math.random() > 0.5f){
                        new Thread("002"){
                            @Override
                            public void run() {
                                EventBus.getDefault().post(new PostingEvent(Thread.currentThread().toString()));
                            }
                        }.start();
                    }else {
                        EventBus.getDefault().post(new PostingEvent(Thread.currentThread().toString()));
                    }
                    break;
                case 1: // Main Mode
                    if(Math.random() > 0.5f){
                        new Thread("002"){
                            @Override
                            public void run() {
                                EventBus.getDefault().post(new MainEvent(Thread.currentThread().toString()));
                            }
                        }.start();
                    }else {
                        EventBus.getDefault().post(new MainEvent(Thread.currentThread().toString()));
                    }
                    break;

                case 2: // Main_Order Mode
                    Log.i(TAG, "onCreateDialog: before @" + SystemClock.uptimeMillis());
                    EventBus.getDefault().post(new MainOrderEvent(Thread.currentThread().toString()));
                    Log.i(TAG, "onCreateDialog: after @" + SystemClock.uptimeMillis());
                    break;

                case 3: // BACKGROUND Mode
                    if(Math.random() > 0.5f){
                        new Thread("002"){
                            @Override
                            public void run() {
                                EventBus.getDefault().post(new BackgroundEvent(Thread.currentThread().toString()));
                            }
                        }.start();
                    }else {
                        EventBus.getDefault().post(new BackgroundEvent(Thread.currentThread().toString()));
                    }
                    break;

                case 4: // ASYNC Mode
                    if(Math.random() > 0.5f){
                        new Thread("002"){
                            @Override
                            public void run() {
                                EventBus.getDefault().post(new AsyncEvent(Thread.currentThread().toString()));
                            }
                        }.start();
                    }else {
                        EventBus.getDefault().post(new AsyncEvent(Thread.currentThread().toString()));
                    }
                    break;
                default:
                    break;
            }
        });
        return builder.create();
    }
}
```

### Resource

- https://juejin.cn/post/6930562700128813070
- https://juejin.cn/post/6921601752424939534#heading-3 eventbus 案例使用
- https://github.com/greenrobot/EventBus
- https://blog.csdn.net/weixin_42324979/article/details/112172736 使用RxJava 实现Eventbus
- todo？  EventBus 实现机制，背后的原理
- https://juejin.cn/post/6844903969517469709#heading-8  eventbus 代码解析

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/eventbus/  

