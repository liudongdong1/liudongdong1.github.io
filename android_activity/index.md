# Android_activity


> **Activity:** 负责用户界面的展示和用户交互,学习Activity就要学习Fragment，虽然它不是四大组件之一，但是它在我们的开发工作中也是频频被使用到，且必须和Activity一块使用，常用于分模块开发，比如慕课首页的几个tab,每个tab都是对应着一个Fragment.
>
> **Service服务：**不需要和用户交互，负责`后台任务`，比如播放音乐，socket长连接
>
> **BroadcastReceiver广播接收者:** `负责页面间通信`，`系统和APP通信，APP和APP通信`，比如监听网络连接状态变化，就是通过`BroadcastReceiver广播接收者来实现的`
>
> **ContentProvider内容提供者:** 负责数据存取，常用于APP进数据共享，跨进程数据存取等....比如读取相册，读取联系人，都是ContentProvider来实现的

#### .1. 生命周期

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/81425fe178072c8d6b216275d3c223ca.png)

- onCreate(): 会在 Activity 第一次创建时进行调用，在这个方法中通常会做 Activity `初始化相关的操作`，例如：`加载布局、绑定事件等`。
- onStart(): 在 Activity 由`不可见变为可见的时候调用`，但是还不能和用户进行交互。
- onResume(): Activity已经启动完成，进入到了前台，可以同用户进行交互了。
- onPause(): 系统`准备去启动另一个 Activity 的时候调用`。可以在这里`释放系统资源，动画的停止，不宜在此做耗时操作`。
- onStop(): 需要在这里`释放全部用户使用不到的资源`。可以做`较重量级的工作`，如对注册广播的解注册，对一些状态数据的存储。此时Activity还不会被销毁掉，而是保持在内存中，但随时都会被回收。通常发生在启动另一   个Activity或切换到后台时
- onDestroy(): Activity即将被销毁。此时必须主动释放掉所有占用的资源。
- onRestart(): 在 Activity 由停止状态变为运行状态之前调用，也就是 Activity 被重新启动了（APP切到后台会进入onStop(), 再切换到前台时会触发onRestart()方法）

```
// 启动A
A:: onCreate -> A:: onStart -> A:: onResume
// 点击A中的button启动B
A:: onPause -> B:: onCreate -> B:: onStart -> B:: onResume -> A:: onStop
// 在B中点击Back退出
B:: onPause -> A:: onRestart -> A:: onStart -> A:: onResume -> B:: onStop -> B:: onDestroy
// 在A中点击Back退出
A:: onPause -> A:: onStop -> A:: onDestroy
```

#### .2. 组件注册

- 显示注册

> androidManifest.xml文件中添加一个标签，标签的一般格式如下：
>
> - android:name是对应Activity的类名称
> - android:label是Activity标题栏显示的内容. 现已不推荐使用
> - intent-filter: 是意图过滤器. 常用语隐式跳转
> - action: 是动作名称，是指`intent要执行的动作`
> - category: 是过滤器的类别 一般情况下，每个 中都要显示指定一个默认的类别名称，即`<category android:name="android.intent.category.DEFAULT" />`

```xml
<activity
          android:name=".MainActivity"
          android:label="@string/app_name">
    <intent-filter>
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />  #Activity是应用程序的入口点
    </intent-filter>
</activity>
```

.3. activity 启动与传递

> - 动作（action），数据（Data），分类（Category），类型（Type），组件（Component），扩展信息（Extra）
> - android.intent.action.MAIN：应用程序`最早启动的Activity`，能够给多个Activity设置。
> - android.intent.category.LAUNCHER：应用程序`是否显示在程序列表里`，能够给多个Activity设置。
> - android.intent.category.DEFAULT：经过Intent启动时`系统会自动给Intent添加该category`。
> - android.intent.action.VIEW：根据传入的data执行一些标准操做`如打开浏览器`
> - android.intent.action.MAIN+android.intent.category.LAUNCHER设置会在launcher显示一个应用图标，单独设置android.intent.category.LAUNCHER不会出现图标，且一个应用程序最少要有一对。
> - 设置了intent-filter会自动将android:exported设置为true

- 系统Intent匹配时会严格区分对待这些组件，时Activity（startActivity），Service（startService） 还是BroadcastReceiver（SendBroadcast）.
- 系统将Intent和对应组件类型中的所有intent-filter 进行匹配，寻找最佳效果。 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221130184333188.png)

##### 1. 显示启动

```java
startActivity(intent intent);
startActivityForResult(Intent,int);
ComponentName startService(Intent service);
boolean bindService(Intent, ServiceConnection conn, int flags);
sendBroadcast(Intent intent);
sendBroadcast(Intent intent, String receiverPermission);
sendOrderedBroadcast(Intent intent, String receiverPremission, BroadcastReceiver resultReceicer, Handler scheduler, int initialCode,Bundle initialExtras);
sendOrderedBroadcast(Intent intent, String receiverPermission);
sendStickyBroadcast(Intent intent);
```

```java
//Intent构造
Intent intent = new Intent(this, SecondActivity.class);  
startActivity(intent); 

//setComponent方法
ComponentName componentName = new ComponentName(this, SecondActivity.class);  
// 或者ComponentName componentName = new ComponentName(this, "com.example.app016.SecondActivity");  
// 或者ComponentName componentName = new ComponentName(this.getPackageName(), "com.example.app016.SecondActivity");  
  
Intent intent = new Intent();  
intent.setComponent(componentName);  
startActivity(intent); 
```

- 简单数据传递

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211107225251570.png)

- 传递数组

```java
//写入数组
bd.putStringArray("StringArray", new String[]{"呵呵","哈哈"});
//可把StringArray换成其他数据类型,比如int,float等等...

//读取数组
String[] str = bd.getStringArray("StringArray")
```

- 传递集合List<基本数据类型或String>

```java
//写入集合
intent.putStringArrayListExtra(name, value)
intent.putIntegerArrayListExtra(name, value)
//读取集合
intent.getStringArrayListExtra(name)
intent.getIntegerArrayListExtra(name)
```

- 传递集合List<Object>

```java
//将list强转成Serializable类型,然后传入(可用Bundle做媒介)
//写入数据
putExtras(key,(Serializable)list)
//读取数据
（List<Object>)getIntent().getSerializable(key)
```

- 传递集合Map<String, Object>,或更复杂的

```java
//传递复杂些的参数 
Map<String, Object> map1 = new HashMap<String, Object>();  
map1.put("key1", "value1");  
map1.put("key2", "value2");  
List<Map<String, Object>> list = new ArrayList<Map<String, Object>>();  
list.add(map1);  

Intent intent = new Intent();  
intent.setClass(MainActivity.this,ComplexActivity.class);  
Bundle bundle = new Bundle();  

//须定义一个list用于在budnle中传递需要传递的ArrayList<Object>,这个是必须要的  
ArrayList bundlelist = new ArrayList();   
bundlelist.add(list);   
bundle.putParcelableArrayList("list",bundlelist);  
intent.putExtras(bundle);                
startActivity(intent); 
```

- 传递对象

> 传递对象的方式有两种：将`对象转换为Json字符串`或者`通过Serializable,Parcelable序列化 `不建议使用Android内置的Json解析器，可使用`fastjson或者Gson第三方库`！
>
> - 方式一： 将对象转化为Json字符串
> - 方式二： 使用Serializable,Parcelable序列化对象
>   - 业务Bean`继承Parcelable接口`,`重写writeToParcel`方法,将你的对象序列化为一个Parcel对象;
>   - `重写describeContents方法`，内容接口描述，默认返回0就可以
>   - `实例化静态内部对象CREATOR实现接口Parcelable.Creator`
>   - 同样式通过Intent的putExtra()方法传入对象实例,当然多个对象的话,我们可以先 放到Bundle里`Bundle.putParcelable(x,x)`,再`Intent.putExtras()`即可

```java
//方式一： 将对象转化为Json字符串
//    写入数据
Book book=new Book();
book.setTitle("Java编程思想");
Author author=new Author();
author.setId(1);
author.setName("Bruce Eckel");
book.setAuthor(author);
Intent intent=new Intent(this,SecondActivity.class);
intent.putExtra("book",new Gson().toJson(book));
startActivity(intent);
//读取数据
String bookJson=getIntent().getStringExtra("book");
Book book=new Gson().fromJson(bookJson,Book.class);
Log.d(TAG,"book title->"+book.getTitle());
Log.d(TAG,"book author name->"+book.getAuthor().getName());
```

```java
//方式二： 使用Serializable,Parcelable序列化对象
//    写入数据
//Internal Description Interface,You do not need to manage  
@Override  
public int describeContents() {  
     return 0;  
}      
@Override  
public void writeToParcel(Parcel parcel, int flags){  
    parcel.writeString(bookName);  
    parcel.writeString(author);  
    parcel.writeInt(publishTime);  
}  
public static final Parcelable.Creator<Book> CREATOR = new Creator<Book>() {  
    @Override  
    public Book[] newArray(int size) {  
        return new Book[size];  
    }  
      
    @Override  
    public Book createFromParcel(Parcel source) {  
        Book mBook = new Book();    
        mBook.bookName = source.readString();   
        mBook.author = source.readString();    
        mBook.publishTime = source.readInt();   
        return mBook;  
    }  
};
```

- 传递bitmap

```java
Bitmap bitmap = null;
Intent intent = new Intent();
Bundle bundle = new Bundle();
bundle.putParcelable("bitmap", bitmap);
intent.putExtra("bundle", bundle);
```

##### 2. [隐式启动](https://www.jb51.net/article/220564.htm)

> 隐式 Intent 要比显示 Intent 含蓄的多，他并不明确指定要启动哪个 Activity，而是通过指定 `action` 和 `category` 的信息，让系统去分析这个 `Intent`，并找出合适的 Activity 去启动。隐式启动的关键部分intent-filter。ps：固然也能够经过隐式方式启动Service和BroadcastReceiver，不过Android5.0以上系统禁止使用隐式方式启动Service
>
> 隐式启动主要是`action、category、data的匹配`。只有`三个都匹配成功`，才能`启动相应的Activity`，不然匹配失败。一个Activity能够有多个intent-filter，只要有一个匹配成功，就能够启动相应的Activity。

```xml
<activity android:name=".MainActivity">
    <intent-filter>
        <action android:name="com.sun.action.test"/>
        <category android:name="com.sun.category"/>
        <data
              android:host="sunHost"
              android:mimeType="sun/mimeType"
              android:path="/sunPath"
              android:port="8888"
              android:scheme="sunScheme"/>
    </intent-filter>
</activity>
```

```java
Intent intent = new Intent("com.sun.action.test");
intent.addCategory("com.sun.category ");
intent.setDataAndType(Uri.parse("sunScheme://sunHost:8888/sunPath"), "sun/mimeType");
startActivity(intent);
```

- action 匹配规则：
  - `intent-filter`的action能够有`多条`，而`Intent`的action`最多只有一条`。
  - Intent的action和Intent-filter中的某条action彻底同样时`包括大小写`，才算匹配成功。
  - `Intent的action只要和intent-filter中的一条action匹配成功便可`。
- Category 匹配规则：
  - Intent中能够不主动指定category。
  - intent-filter的category能够有多条，`Intent的category也能够有多条`。
  - Intent中的`全部category`都能在Intent-filter中找到彻底同样的category包括大小写，才算匹配成功。
  - 经过`Intent启动Activity的时候`，`系统会自动给Intent添加android.intent.category.DEFAULT`（若是Intent本身没有添加任何category，记得在intent-filter中添加android.intent.category.DEFAULT，否则匹配不成功）
- Data 匹配规则：
  - scheme：`数据的协议部分`，如http
  - host：主机名部分，如blog.csdn.net
  - port：端口，如8080
  - path：指定一个`完整的路径`，如sunzhaojie613
  - pathPrefix：指定了`部分路径`，它会跟Intent对象中的路径初始部分匹配，如sunzhaojie613
  - pathPattern：指定的路径能够进行正则匹配，如sunzhaojie613
  - mimeType：处理的`数据类型`，如image/*.net

```xml
scheme://host:port/path|pathPrefix|pathPattern
```

1. intent-filter能够有多个data，Intent最多只有一个data。
2. 若是intent-filter中指定了data，那么Intent就要指定一个相应的data。
3. scheme匹配成功、host匹配成功、port匹配成功、path|pathPrefix|pathPattern正则匹配成功、mimeType正则匹配成功，才算data匹配成功。
4. intent-filter若是没有指定scheme，默认为content和file。
5. Intent经过setData来设置data，该方法会清空mimeType。经过`setDataAndType来设置data和mimeType`。

##### 3. 定义全局数据

- 方式一：` 自定义Application类`，在AndroidManifest.xml 中声明， 在需要的地方调用

```java
class MyApp extends Application {
    private String myState;
    public String getState(){
        return myState;
    }
    public void setState(String s){
        myState = s;
    }
}
```

```xml
<application android:name=".MyApp" android:icon="@drawable/icon" 
  android:label="@string/app_name">
```

```java
class Blah extends Activity {
    @Override
    public void onCreate(Bundle b){
        ...
    MyApp appState = ((MyApp)getApplicationContext());
    String state = appState.getState();
        ...
    }
}
```

- 方式二：Applicaiton是系统的一个组件，他也有自己的一个生命周期，我们可以`在onCraete里获得这个 Application对象`。

```java
class MyApp extends Application {
    private String myState;
    private static MyApp instance;
    
    public static MyApp getInstance(){
        return instance;
    }
    public String getState(){
        return myState;
    }
    public void setState(String s){
        myState = s;
    }
    @Override
    public void onCreate(){
        onCreate();
        instance = this;
    }
}
```

#### .4. [常见的一些activity](https://segmentfault.com/a/1190000014530151)

- 拨打电话

```java
Uri uri = Uri.parse("tel:10086");
Intent intent = new Intent(Intent.ACTION_DIAL, uri);
startActivity(intent);
```

- 发送短信

```java
Uri uri = Uri.parse("smsto:10086");
Intent intent = new Intent(Intent.ACTION_SENDTO, uri);
intent.putExtra("sms_body", "Hello");
startActivity(intent);
```

- 打开浏览器

```java
Uri uri = Uri.parse("http://www.baidu.com");
Intent intent  = new Intent(Intent.ACTION_VIEW, uri);
startActivity(intent);
```

- 多媒体播放

```java
Intent intent = new Intent(Intent.ACTION_VIEW);
Uri uri = Uri.parse("file:///sdcard/foo.mp3");
intent.setDataAndType(uri, "audio/mp3");
startActivity(intent);
```

- 打开摄像头

```java
Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE); 
startActivityForResult(intent, 0);

>>> 在Activity的onActivityResult方法回调中取出照片数据
Bundle extras = intent.getExtras(); 
Bitmap bitmap = (Bitmap) extras.get("data");
```

- 从图库中选图并剪切

```java
// 获取并剪切图片
Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
intent.setType("image/*");
intent.putExtra("crop", "true"); // 开启剪切
intent.putExtra("aspectX", 1); // 剪切的宽高比为1：2
intent.putExtra("aspectY", 2);
intent.putExtra("outputX", 20); // 保存图片的宽和高
intent.putExtra("outputY", 40); 
intent.putExtra("output", Uri.fromFile(new File("/mnt/sdcard/temp"))); // 保存路径
intent.putExtra("outputFormat", "JPEG");// 返回格式
startActivityForResult(intent, 0);

>>>>  在Activity的onActivityResult方法中去读取保存的文件
```

- 无线网络设置页面

```java
// 进入无线网络设置界面（其它可以举一反三）  
Intent intent = new Intent(android.provider.Settings.ACTION_WIRELESS_SETTINGS);  
startActivityForResult(intent, 0);
```

#### .5. 启动模式

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/8500d96e0dbf278df981b2f1310442e2-166883070091660.png)

##### 1. standard

> 每启动一次，都会创建一个新的Activity实例。启动的生命周期为：onCreate()->onStart()->onResume()

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108083016701.png)

##### 2. singleTop

- 如果任务栈顶已经存在需要启动的目标Activity，则直接启动，并会回调onNewIntent()方法，生命周期顺序为： onPause() ->onNewIntent()->onResume()
- 如果任务栈上顶没有需要启动的目标Activity，则创建新的实例，此时生命周期顺序为： onCreate()->onStart()->onResume()

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108083334735.png)

##### 3. SingleTask

> 栈内复用模式，一个任务栈只能有一个实例。

- 当启动的Activity目标任务栈不存在时，则以此启动Activity为根Activity创建目标任务栈，并切换到前面
- D为singleTask模式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108083436941.png)

- 当启动的Activity存在时，则会直接切换到Activity所在的任务栈，并且任务栈中在Activity上面的所有其他Activity都出栈（调用destroy()），此时启动的Activity位于任务栈顶，并且会回调onNewIntent()方法。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108083510183.png)

##### 4. singleInstance

> singleInstance名称是单例模式，即App运行时，该Activity只有一个实例。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108083558464.png)


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android_activity/  

