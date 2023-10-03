# AndroidAIDL


> **AIDL（Android Interface Definition Language）是一种 IDL 语言，用于生成可以在 Android 设备上两个进程之间进行进程间通信（IPC）的代码。** 通过 AIDL，可以在一个进程中获取另一个进程的数据和调用其暴露出来的方法，从而满足进程间通信的需求。

### 1. 创建AIDL

- AIDL 文件可以分为两类。
  - 一类用来声明实`现了 Parcelable 接口的数据类型`，以供其他 AIDL 文件使用那些非默认支持的数据类型。
  - 一类是用来`定义接口方法`，声明要暴露哪些接口给客户端调用。
- 在 AIDL 文件中需要`明确标明引用到的数据类型所在的包名`，即使两个文件处在同个包名下。

支持数据类型：

- 八种基本数据类型：byte、char、short、int、long、float、double、boolean
- String，CharSequence
- List类型。List承载的数据必须是AIDL支持的类型，或者是其它声明的AIDL对象
- Map类型。Map承载的数据必须是AIDL支持的类型，或者是其它声明的AIDL对象

创建方式：客户端和服务端都需要创建，我们先在服务端中创建，然后复制到客户端即可。在 Android Studio 中右键点击新建一个 AIDL 文件

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/f9ec079047ec4780929d9d5632a15ff9.png)

系统就会默认创建一个 aidl 文件夹，文件夹下的目录结构即是工程的包名，AIDL 文件就在其中

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/710c29e5360b48b6b97f8c3b631068a0.png)

### 2. 实现接口

- 创建或修改过 AIDL 文件后需要 build 下工程，Android SDK 工具会`生成以 .aidl 文件命名的 .java 接口文件`。
- 生成的接口包含一个名为 Stub 的[子类](https://so.csdn.net/so/search?q=子类&spm=1001.2101.3001.7020)（例如，IRemoteService.Stub），该子类是其父接口的抽象实现，并且会声明 AIDL 文件中的所有方法。
- `如要实现 AIDL 生成的接口，请实例化生成的 Binder 子类`（例如，IRemoteService.Stub），并实现继承自 AIDL 文件的方法。

```java
private final IRemoteService.Stub binder = new IRemoteService.Stub() {
    public int getPid(){
        return Process.myPid();
    }
    public void basicTypes(int anInt, long aLong, boolean aBoolean,
        float aFloat, double aDouble, String aString) {
        // Does nothing
    }
};
```

### 3. 服务端公开接口

在为服务端实现接口后，需要向客户端公开该接口，以便客户端进行绑定。创建 Service 并实现 onBind()，从而返回生成的 Stub 的类实例。

```java
public class RemoteService extends Service {
    private final String TAG = "RemoteService";

    @Override
    public void onCreate() {
        super.onCreate();
    }

    @Override
    public IBinder onBind(Intent intent) {
        // Return the interface
        Log.d(TAG, "onBind");
        return binder;
    }

    private final IRemoteService.Stub binder = new IRemoteService.Stub() {
        public int getPid() {
            return Process.myPid();
        }

        public void basicTypes(int anInt, long aLong, boolean aBoolean,
                               float aFloat, double aDouble, String aString) {
            Log.d(TAG, "basicTypes anInt:" + anInt + ";aLong:" + aLong + ";aBoolean:" + aBoolean + ";aFloat:" + aFloat + ";aDouble:" + aDouble + ";aString:" + aString);
        }
    };
}
```

```xml
<service
         android:name=".RemoteService"
         android:enabled="true"
         android:exported="true">
    <intent-filter>
        <action android:name="com.example.aidl"/>
        <category android:name="android.intent.category.DEFAULT"/>
    </intent-filter>
</service>
```

### 4. 客户端调用

当客户端（如 Activity）调用 bindService() 以连接此服务时，客户端的 onServiceConnected() 回调会接收服务端的 onBind() 方法所返回的 binder 实例。

客户端还必须拥有接口类的访问权限，因此如果客户端和服务端在不同应用内，则客户端应用的 src/ 目录内必须包含 .aidl 文件（该文件会生成 android.os.Binder 接口，进而为客户端提供 AIDL 方法的访问权限）的副本。所以我们需要把服务端的 aidl 文件夹整个复制到客户端的 java 文件夹同个层级下，不需要改动任何代码。

```java
public class MainActivity extends AppCompatActivity {
    private final String TAG = "ClientActivity";
    private IRemoteService iRemoteService;
    private Button mBindServiceButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mBindServiceButton = findViewById(R.id.btn_bind_service);
        mBindServiceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String text = mBindServiceButton.getText().toString();
                if ("Bind Service".equals(text)) {
                    Intent intent = new Intent();
                    intent.setAction("com.example.aidl");
                    intent.setPackage("com.example.aidl.server");
                    bindService(intent, mConnection, Context.BIND_AUTO_CREATE);
                } else {
                    unbindService(mConnection);
                    mBindServiceButton.setText("Bind Service");
                }
            }
        });
    }

    ServiceConnection mConnection = new ServiceConnection() {

        @Override
        public void onServiceDisconnected(ComponentName name) {
            Log.d(TAG, "onServiceDisconnected");
            iRemoteService = null;
        }

        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d(TAG, "onServiceConnected");
            iRemoteService = IRemoteService.Stub.asInterface(service);
            try {
                int pid = iRemoteService.getPid();
                int currentPid = Process.myPid();
                Log.d(TAG, "currentPID: " + currentPid + ", remotePID: " + pid);
                iRemoteService.basicTypes(12, 123, true, 123.4f, 123.45,
                        "服务端你好，我是客户端");
            } catch (RemoteException e) {
                e.printStackTrace();
            }
            mBindServiceButton.setText("Unbind Service");
        }
    };
}
```

### 5. IPC 传递对象

AIDL 还可以传递对象，但是该类必须实现 Parcelable 接口。而该类是两个应用间都需要使用到的，所以也需要在 AIDL 文件中声明该类，为了避免出现类名重复导致无法创建 AIDL 文件的错误，这里需要先创建 AIDL 文件，之后再创建类。

```java
// Rect.aidl
package com.example.aidl.server;

// Declare Rect so AIDL can find it and knows that it implements
// the parcelable protocol.
parcelable Rect;
```

```java
public class Rect implements Parcelable {
    private int left;
    private int top;
    private int right;
    private int bottom;

    public Rect(int left, int top, int right, int bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    public static final Parcelable.Creator<Rect> CREATOR = new Parcelable.Creator<Rect>() {
        public Rect createFromParcel(Parcel in) {
            return new Rect(in);
        }

        public Rect[] newArray(int size) {
            return new Rect[size];
        }
    };

    private Rect(Parcel in) {
        readFromParcel(in);
    }

    @Override
    public void writeToParcel(Parcel out, int flags) {
        out.writeInt(left);
        out.writeInt(top);
        out.writeInt(right);
        out.writeInt(bottom);
    }

    public void readFromParcel(Parcel in) {
        left = in.readInt();
        top = in.readInt();
        right = in.readInt();
        bottom = in.readInt();
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @NonNull
    @Override
    public String toString() {
        return "Rect[left:" + left + ",top:" + top + ",right:" + right + ",bottom:" + bottom + "]";
    }
}
```

```java
// IRemoteService.aidl
package com.example.aidl.server;
import com.example.aidl.server.Rect;

// Declare any non-default types here with import statements

interface IRemoteService {
    /**
     * Demonstrates some basic types that you can use as parameters
     * and return values in AIDL.
     */
    void basicTypes(int anInt, long aLong, boolean aBoolean, float aFloat,
            double aDouble, String aString);

    int getPid();

    void addRectInOut(inout Rect rect);
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221124143552538.png)

### 6. Parcel 序列化

Parcel提供了一套机制，可以将序列化之后的数据写入到一个共享内存中，其他进程通过Parcel可以从这块共享内存中读出字节流，并反序列化成对象首先写一个类实现Parcelable接口,会让我们实现两个方法:

- `describeContents 描述`: 其中`describeContents就是负责文件描述`.通过源码的描述可以看出,只针对一些特殊的需要描述信息的对象,需要返回1,其他情况返回0就可以

- `writeToParcel 序列化`: 我们通过`writeToParcel方法实现序列化`,writeToParcel返回了Parcel,所以我们可以直接调用Parcel中的write方法,基本的write方法都有,对象和集合比较特殊下面单独讲,基本的数据类型除了boolean其他都有,Boolean可以使用int或byte存储
- `反序列化需要定义一个CREATOR的变量`

#### 1. 对象集合处理

一种是写入类的相关信息,然后通过类加载器去读取, –> writeList | readList
二是不用类相关信息,创建时传入相关类的CREATOR来创建 –> writeTypeList | readTypeList | createTypedArrayList

```java
import android.os.Parcel;
import android.os.Parcelable;
 
import java.util.ArrayList;
 
 
public class ParcelDemo implements Parcelable {
 
    private int count;
    private String name;
    private ArrayList<String> tags;
    private Book book;
    // ***** 注意: 这里如果是集合 ,一定要初始化 *****
    private ArrayList<Book> books = new ArrayList<>();
 
 
    /**
     * 反序列化
     *
     * @param in
     */
    protected ParcelDemo(Parcel in) {
        count = in.readInt();
        name = in.readString();
        tags = in.createStringArrayList();
 
        // 读取对象需要提供一个类加载器去读取,因为写入的时候写入了类的相关信息
        book = in.readParcelable(Book.class.getClassLoader());
 
 
        //读取集合也分为两类,对应写入的两类
 
        //这一类需要用相应的类加载器去获取
        in.readList(books, Book.class.getClassLoader());// 对应writeList
 
 
        //这一类需要使用类的CREATOR去获取
        in.readTypedList(books, Book.CREATOR); //对应writeTypeList
 
        //books = in.createTypedArrayList(Book.CREATOR); //对应writeTypeList
 
 
        //这里获取类加载器主要有几种方式
        getClass().getClassLoader();
        Thread.currentThread().getContextClassLoader();
        Book.class.getClassLoader();
 
 
    }
 
    public static final Creator<ParcelDemo> CREATOR = new Creator<ParcelDemo>() {
        @Override
        public ParcelDemo createFromParcel(Parcel in) {
            return new ParcelDemo(in);
        }
 
        @Override
        public ParcelDemo[] newArray(int size) {
            return new ParcelDemo[size];
        }
    };
 
    /**
     * 描述
     *
     * @return
     */
    @Override
    public int describeContents() {
        return 0;
    }
 
    /**
     * 序列化
     *
     * @param dest
     * @param flags
     */
    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(count);
        dest.writeString(name);
        //序列化一个String的集合
        dest.writeStringList(tags);
        // 序列化对象的时候传入要序列化的对象和一个flag,
        // 这里的flag几乎都是0,除非标识当前对象需要作为返回值返回,不能立即释放资源
        dest.writeParcelable(book, 0);
 
        // 序列化一个对象的集合有两种方式,以下两种方式都可以
 
 
        //这些方法们把类的信息和数据都写入Parcel，以使将来能使用合适的类装载器重新构造类的实例.所以效率不高
        dest.writeList(books);
 
 
        //这些方法不会写入类的信息，取而代之的是：读取时必须能知道数据属于哪个类并传入正确的Parcelable.Creator来创建对象
        // 而不是直接构造新对象。（更加高效的读写单个Parcelable对象的方法是：
        // 直接调用Parcelable.writeToParcel()和Parcelable.Creator.createFromParcel()）
        dest.writeTypedList(books);
 
 
    }
}
```

### 6. [FastJson2 序列化   ](https://github.com/alibaba/fastjson2)

### 6. [googleProtobuf](https://github.com/protocolbuffers/protobuf)

- Protobuf 介绍: https://juejin.cn/post/7156420605737173022#heading-4
- [Protobuf 使用建议](https://www.modb.pro/db/407012)
- [官方教程：](https://developers.google.com/protocol-buffers/docs/javatutorial)  [java protobuf 使用](https://developers.google.com/protocol-buffers/docs/javatutorial)
  - Define message formats in a `.proto` file.
  - Use the protocol buffer compiler. (protoc -I=$SRC_DIR --java_out=$DST_DIR $SRC_DIR/addressbook.proto)
  - Use the Java protocol buffer API to write and read messages. 见对应教程使用

```protobuf
//"proto3"; 使用proto2 还是proto3，这里声明
syntax = "proto3";

package cn.bluemobi.dylan.step.protobuf;

//enables generating a separate .java file for each generated class (instead of the legacy behavior of generating a single .java file for the wrapper class, using the wrapper class as an outer class, and nesting all the other classes inside the wrapper class).
option java_multiple_files = true;
//specifies in what Java package name your generated classes should live
option java_package = "com.example.tutorial.protos";
//efines the class name of the wrapper class which will represent this file
option java_outer_classname = "AddressBookProtos";

message Person {
  string name = 1;
  int32 id = 2;  // Unique ID number for this person.
  string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    string number = 1;
    PhoneType type = 2;
  }

  repeated PhoneNumber phones = 4;

}

// Our address book file is just one of these.
message AddressBook {
  repeated Person people = 1;
}
```



### 7. Parcelable & Serializable区别

- `Serializable`序列化不保存静态变量，可以使用`Transient`关键字对部分字段不进行序列化，也可以覆盖`writeObject`、`readObject`方法以实现序列化过程自定义。
- `Serializable`的作用是为了保存对象的属性到`本地文件、数据库、网络流、rmi`以方便数据传输，当然这种传输可以是程序内的也可以是两个程序间的。而Android的`Parcelable`的设计初衷是因为`Serializable`效率过慢，为了在程序内不同组件间以及不同Android程序间(AIDL)高效的传输数据而设计，这些数据仅在内存中存在，`Parcelable`是通过`IBinder`通信的消息的载体。

### 8. [案例代码](https://github1s.com/SpikeKing/wcl-aidl-demo/blob/HEAD/app/src/main/java/org/wangchenlong/wcl_aidl_demo/BookManagerService.java)

- 案例代码解析： https://www.jianshu.com/p/69e5782dd3c3
- https://www.cnblogs.com/qingblog/archive/2012/07/25/2607754.html  handler
- RemoteCallbackList 使用： https://blog.csdn.net/Jason_Lee155/article/details/125385522
- [DeathRecipient理解和调用](https://blog.csdn.net/briblue/article/details/51035412)（解决Server端进程意外终止，Client端得到的Binder对象丢失问题，server进程意外终止时，通过注册DeathRecipient回调函数，通过回调通知客户端进行处理）  todo？ [结合源码分析](http://gityuan.com/2016/10/03/binder_linktodeath/)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/androidaidl/  

