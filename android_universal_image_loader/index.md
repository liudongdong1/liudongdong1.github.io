# Android_Universal_Image_Loader


#### ImageLoader--对外接口

##### ImageLoader 单例建造者模式构建

```java
public static void initImageLoader(Context context) {
    // This configuration tuning is custom. You can tune every option, you may tune some of them,
    // or you can create default configuration by
    //  ImageLoaderConfiguration.createDefault(this);
    // method.
    ImageLoaderConfiguration.Builder config = new ImageLoaderConfiguration.Builder(context);
    config.threadPriority(Thread.NORM_PRIORITY - 2);
    config.denyCacheImageMultipleSizesInMemory();
    config.diskCacheFileNameGenerator(new Md5FileNameGenerator());
    config.diskCacheSize(50 * 1024 * 1024); // 50 MiB
    config.tasksProcessingOrder(QueueProcessingType.LIFO);
    config.writeDebugLogs(); // Remove for release app

    // Initialize ImageLoader with configuration.
    ImageLoader.getInstance().init(config.build());
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/ImageLoaderConfiguration.png)

##### 提供对外接口

###### loadImage

```java
public void loadImage(String uri, ImageSize targetImageSize, DisplayImageOptions options,
                      ImageLoadingListener listener, ImageLoadingProgressListener progressListener) {
    checkConfiguration();
    if (targetImageSize == null) {
        targetImageSize = configuration.getMaxImageSize();
    }
    if (options == null) {
        options = configuration.defaultDisplayImageOptions;
    }

    NonViewAware imageAware = new NonViewAware(uri, targetImageSize, ViewScaleType.CROP);
    displayImage(uri, imageAware, options, listener, progressListener);
}
```

###### loadImageSync

```java
public Bitmap loadImageSync(String uri, ImageSize targetImageSize, DisplayImageOptions options) {
    if (options == null) {
        options = configuration.defaultDisplayImageOptions;
    }
    options = new DisplayImageOptions.Builder().cloneFrom(options).syncLoading(true).build();

    SyncImageLoadingListener listener = new SyncImageLoadingListener();
    loadImage(uri, targetImageSize, options, listener);
    return listener.getLoadedBitmap();
}
```

###### displayImage 函数执行流程

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221201155317236.png)

- display 函数执行时序图

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221202094654317.png)

![Task Flow](https://gitee.com/github-25970295/blogimgv2022/raw/master/UIL_Flow.png)

##### Task 相关，采用ThreadPoolExecutor 线程池方式创建Executor

```java
new ThreadPoolExecutor(threadPoolSize, threadPoolSize, 0L, TimeUnit.MILLISECONDS, taskQueue,
      createThreadFactory(threadPriority, "uil-pool-"));
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221202100758377.png)

ProcessAndDisplayImageTask 流程



![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221201163345026.png)

#### Download 工具

![Package download](https://gitee.com/github-25970295/blogimgv2022/raw/master/Package%20download.png)

#### imageaware

- 一个view，用于显示 Drawable 或者Bitmap数据
- 了解里面 Reference 用法  todo？

![Packageimageaware](https://gitee.com/github-25970295/blogimgv2022/raw/master/Packageimageaware.png)

#### Bitmap绘制工具

![Package display](https://gitee.com/github-25970295/blogimgv2022/raw/master/Package%20display.png)

#### 监听类

- 例如PauseOnScrollListener, 

```java
public abstract class AbsListView extends AdapterView<ListAdapter> implements TextWatcher,
ViewTreeObserver.OnGlobalLayoutListener, Filter.FilterListener,
ViewTreeObserver.OnTouchModeChangeListener,
RemoteViewsAdapter.RemoteAdapterConnectionCallback {
    
    public void setOnScrollListener(OnScrollListener l) {
        mOnScrollListener = l;
        invokeOnItemScrollListener();
    }
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221202101137173.png)

#### 方法类的写法

```java
public final class MemoryCacheUtils {
    private static final String URI_AND_SIZE_SEPARATOR = "_";
    private MemoryCacheUtils() {
    }
    public static String generateKey(String imageUri, ImageSize targetSize) {
        return new StringBuilder(imageUri).toString();
    }
}
```

#### Resource

- https://github.com/nostra13/Android-Universal-Image-Loader

---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/android_universal_image_loader/  

