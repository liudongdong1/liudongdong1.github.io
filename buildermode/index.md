# BuilderMode


> 将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。`在构建的对象需要很多配置的时候可以考虑Builder模式，可以避免过多的set方法`，同时把配置过程从目标类里面隔离出来，代码结构更加清晰
>
> - `具体的set方法放在配置类的内部类Builder类中，并且每个set方法都返回自身`，以便进行链式调用
>
> - `当初始化一个对象特别复杂时，如参数多，且很多参数都具有默认值时`, 否则每增加一个设置选项，都需要修改ImageLoader代码，违背开闭原则
> - 相同的方法，不同的执行顺序，产生不同的事件结果时
> - 多个部件或零件，都可以装配到一个对象中，但是产生的运行效果又不相同时
> - 产品类非常复杂，或者产品类中的调用顺序不同产生了不同的作用，这个时候使用建造者模式非常合适

```java
public class Computer {
    private String cpu;//必须
    private String ram;//必须
    private int usbCount;//可选
    private String keyboard;//可选
    private String display;//可选
}
```

### 1. 折叠构造函数模式（telescoping constructor pattern ）

```java
public class Computer {
     ...
    public Computer(String cpu, String ram) {
        this(cpu, ram, 0);
    }
    public Computer(String cpu, String ram, int usbCount) {
        this(cpu, ram, usbCount, "罗技键盘");
    }
    public Computer(String cpu, String ram, int usbCount, String keyboard) {
        this(cpu, ram, usbCount, keyboard, "三星显示器");
    }
    public Computer(String cpu, String ram, int usbCount, String keyboard, String display) {
        this.cpu = cpu;
        this.ram = ram;
        this.usbCount = usbCount;
        this.keyboard = keyboard;
        this.display = display;
    }
}
```

### 2. Javabean 模式

```java
public class Computer {
        ...

    public String getCpu() {
        return cpu;
    }
    public void setCpu(String cpu) {
        this.cpu = cpu;
    }
    public String getRam() {
        return ram;
    }
    public void setRam(String ram) {
        this.ram = ram;
    }
    public int getUsbCount() {
        return usbCount;
    }
...
}
```

### 3. builder 模式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210306121424477.png)

- Director类使用者使用组合方式直接和builder 基类进行打交道，通过开闭原则传入具体的builder实例，通过多态的方式构建不同的实例的builder类型
- ConcreteBuilder与产品打交道， ConcreteBuilder里面通过与产品类组合方式
- 构建不同的实例。

> - Product: 最终要生成的对象，例如 Product实例。
> - Builder： 构建者的`抽象基类`（有时会使用接口代替）。其定义了构建Product的抽象步骤，其实体类需要实现这些步骤。其会包含一个用来返回最终产品的方法`Product getProduct()`。
> - ConcreteBuilder: `Builder的实现类`。
> - Director: 决定`如何构建最终产品的算法`. 其会包含一个负责组装的方法`void Construct(Builder builder)`， 在这个方法中通过调用builder的方法，就可以设置builder，等设置完成后，就可以通过builder的 `getProduct()` 方法获得最终的产品。

```java
public abstract class Product {
    private String name;
    private String type;
    public void showProduct(){
        System.out.println("名称："+name);
        System.out.println("型号："+type);
    }
    public void setName(String name) {
        this.name = name;
    }
    public void setType(String type) {
        this.type = type;
    }
}

abstract class Builder {
    public abstract void setPart(String arg1, String arg2);
    public abstract Product getProduct();
}
class ConcreteBuilder extends Builder {
    private Product product = new Product();

    public Product getProduct() {
        return product;
    }

    public void setPart(String arg1, String arg2) {
        product.setName(arg1);
        product.setType(arg2);
    }
}

public class Director {
    private Builder builder = new ConcreteBuilder();
    public Product getAProduct(){
        builder.setPart("宝马汽车","X7");
        return builder.getProduct();
    }
    public Product getBProduct(){
        builder.setPart("奥迪汽车","Q5");
        return builder.getProduct();
    }
}
public class Client {
    public static void main(String[] args){
        Director director = new Director();
        Product product1 = director.getAProduct();
        product1.showProduct();

        Product product2 = director.getBProduct();
        product2.showProduct();
    }
}
```

### 4.  Android 中相关实现

#### .1. [AlertDialog](https://juejin.cn/post/6844903503219916807)

1. AlertDialog内部使用Builder设计模式，其内部类Builder每次构建返回自身，有利于链式调用，代码结构清晰
2. AlertDialog的Builder掌握所有的Dialog参数配置工作，其具体配置由AlertController来实现
3. AlertController是AlertDialog的重要实现类，用户配置的参数信息由内部静态类AlertParams来保存，其内部具体完成Dialog的装配工作
4. Dialog的具体显示和消失逻辑由WindowManager来完成
5. AlertDialog.Builder 同时扮演着builder，ConcreteBuilder，Director角色

#### .2. WinowManagerService

- todo？ 学习完整的调用过程，以及背后的原理

![image-20221126134946193](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221126134946193.png)

#### .3. [ImageLoader](https://juejin.cn/post/6844903650783920135)

```java
public class ImageLoader {
    //图片加载配置
    ImageLoaderConfig mConfig;

    // 图片缓存，依赖接口
    ImageCache mImageCache = new MemoryCache();

    // 线程池，线程数量为CPU的数量
    ExecutorService mExecutorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    //省略单例模式实现
    
    //初始化ImageLoader
    public void init(ImageLoaderConfig config) {
        mConfig = config;
        mImageCache = mConfig.mImageCache;
    }
    
    /**
     * 显示图片
     * @param imageUrl
     * @param imageView
     */
    public void displayImage(String imageUrl, ImageView imageView) {
        Bitmap bitmap = mImageCache.get(imageUrl);
        if (bitmap != null) {
            imageView.setImageBitmap(bitmap);
            return;
        }
        // 图片没有缓存，提交到线程池下载
        submitLoadRequest(imageUrl, imageView);
    }

    /**
     * 下载图片
     * @param imageUrl
     * @param imageView
     */
    private void submitLoadRequest(final String imageUrl, final ImageView imageView) {
        imageView.setImageResource(mConfig.displayConfig.loadingImageId);
        imageView.setTag(imageUrl);
        mExecutorService.submit(new Runnable() {
            @Override
            public void run() {
                Bitmap bitmap = downloadImage(imageUrl);
                if (bitmap == null) {
                    imageView.setImageResource(mConfig.displayConfig.loadingFailImageId);
                    return;
                }
                if (imageUrl.equals(imageView.getTag())) {
                    imageView.setImageBitmap(bitmap);
                }
                mImageCache.put(imageUrl, bitmap);
            }
        });
    }

    /**
     * 下载图片
     * @param imageUrl
     * @return
     */
    private Bitmap downloadImage(String imageUrl) {
        Bitmap bitmap = null;
        //省略下载部分代码
        return bitmap;
    }
}
```

##### 1. ImageLoaderConfig

```java
public class ImageLoaderConfig {
    // 图片缓存，依赖接口
    public ImageCache mImageCache = new MemoryCache();

    //加载图片时的loading和加载失败的图片配置对象
    public DisplayConfig displayConfig = new DisplayConfig();

    //线程数量，默认为CPU数量+1；
    public int threadCount = Runtime.getRuntime().availableProcessors() + 1;

    private ImageLoaderConfig() {
    }


    /**
     * 配置类的Builder
     */
    public static class Builder {
        // 图片缓存，依赖接口
        ImageCache mImageCache = new MemoryCache();

        //加载图片时的loading和加载失败的图片配置对象
        DisplayConfig displayConfig = new DisplayConfig();

        //线程数量，默认为CPU数量+1；
        int threadCount = Runtime.getRuntime().availableProcessors() + 1;

        /**
         * 设置线程数量
         * @param count
         * @return
         */
        public Builder setThreadCount(int count) {
            threadCount = Math.max(1, count);
            return this;
        }

        /**
         * 设置图片缓存
         * @param cache
         * @return
         */
        public Builder setImageCache(ImageCache cache) {
            mImageCache = cache;
            return this;
        }

        /**
         * 设置图片加载中显示的图片
         * @param resId
         * @return
         */
        public Builder setLoadingPlaceholder(int resId) {
            displayConfig.loadingImageId = resId;
            return this;
        }

        /**
         * 设置加载失败显示的图片
         * @param resId
         * @return
         */
        public Builder setLoadingFailPlaceholder(int resId) {
            displayConfig.loadingFailImageId = resId;
            return this;
        }

        void applyConfig(ImageLoaderConfig config) {
            config.displayConfig = this.displayConfig;
            config.mImageCache = this.mImageCache;
            config.threadCount = this.threadCount;
        }

        /**
         * 根据已经设置好的属性创建配置对象
         * @return
         */
        public ImageLoaderConfig create() {
            ImageLoaderConfig config = new ImageLoaderConfig();
            applyConfig(config);
            return config;
        }
    }
}
```

```java
ImageLoaderConfig config = new ImageLoaderConfig.Builder()
        .setImageCache(new MemoryCache())
        .setThreadCount(2)
       .setLoadingFailPlaceholder(R.drawable.loading_fail)
        .setLoadingPlaceholder(R.drawable.loading)
        .create();
ImageLoader.getInstance().init(config);
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/buildermode/  

