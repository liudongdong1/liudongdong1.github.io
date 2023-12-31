# Android_Service


> Thread是线程，程序执行的最小单元，分配CPU的基本单位！ 而`Service则是Android提供一个允许长时间留驻后台的一个组件`，最常见的 用法就是做轮询操作！或者想在后台做一些事情，比如后台下载更新！

### 1. 生命周期

![线程生命周期](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108205619410.png)

![Service 生命周期](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108211727096.png)

- **onCreate()**：当Service第一次被创建后立即回调该方法，该方法在`整个生命周期中只会调用一次`,并且它的调用在`onStartCommand()`以及`onBind()`之前，我们可以在这个方法中进行一些一次性的初始化工作。
- **onDestory()**：当Service被关闭时会回调该方法，该方法只会回调一次！
- **onStartCommand(intent,flag,startId)**：早期版本是onStart(intent,startId), 当客户端调用startService(Intent)方法时会回调，可多次调用StartService方法， 但不会再创建新的Service对象，而是继续复用前面产生的Service对象，但会继续回调 onStartCommand()方法！
- **IBinder onOnbind(intent)**：该方法是Service都必须实现的方法，该方法会返回一个 IBinder对象，`app通过该对象与Service组件进行通信`！
- **onUnbind(intent)**：当`该Service上绑定的所有客户端都断开时会回调该方法`！

### 2. Service 启动

-  **StartService启动Service**
  - 首次启动会创建一个Service实例,依次调用onCreate()和onStartCommand()方法,此时Service 进入运行状态,如果再次调用StartService启动Service,将不会再创建新的Service对象, 系统会直接复用前面创建的Service对象,调用它的onStartCommand()方法！
  - 这样的Service与它的调用者无必然的联系,就是说当调用者结束了自己的生命周期, 但是`只要不调用stopService,那么Service还是会继续运行的`!
  - 无论启动了多少次Service,只需调用一次StopService即可停掉Service
  - 点击启动服务，服务新建线程，执行完耗时，点击停止服务，并没有影响后台线程继续执行, 所有后台线程执行完后，服务才销毁

```java
private class MyRunnable implements Runnable {
    // 服务的 Id
    private int mStartId;

    public MyRunnable(int startId){
        mStartId = startId;
    }

    @Override
    public void run() {
        try {
            Log.d(TAG, "run: ");
            Thread.sleep(3000);
            Log.d(TAG, "run: stopSelfResult.mStartId = " + mStartId);
            // 耗时操作之后，结束服务
            stopSelfResult(mStartId);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

@Override
public int onStartCommand(Intent intent, int flags, int startId) {
    Log.d(TAG, "onStartCommand: flags = " + flags + ", startId = " + startId);
    // 新建线程执行耗时操作，并传递该服务的 Id
    Thread thread = new Thread(new MyRunnable(startId));
    thread.start();
    return super.onStartCommand(intent, flags, startId);
}
```

-  **BindService 启动Service**
  - 当首次使用bindService绑定一个Service时,系统会`实例化一个Service实例`,并调用其onCreate()和onBind()方法,然后调用者就可以`通过IBinder和Service进行交互`了,此后如果再次使用bindService绑定Service,系统不会创建新的Sevice实例,也不会再调用onBind()方法,只会`直接把IBinder对象传递给其他后来增加的客户端`!
  - 如果我们解除与服务的绑定,只需调用`unbindService()`,此时onUnbind和onDestory方法将会被调用!这是一个客户端的情况,假如是`多个客户端绑定同一个Service的话`,情况如下当一个客户完成和service之间的互动后，它调用 unbindService() 方法来解除绑定。`当所有的客户端都和service解除绑定后，系统会销毁service`。（除非service也被startService()方法开启）
  - 另外,和上面那张情况不同,bindService模式下的Service是与调用者相互关联的,可以理解为 "一条绳子上的蚂蚱",要死一起死,在bindService后,一旦调用者销毁,那么Service也立即终止!
  - **bindService**(Intent Service,ServiceConnection conn,int flags)

> 如果Service已经由某个客户端通过StartService()启动,接下来由其他客户端 再调用bindService(）绑定到该Service后调用unbindService()解除绑定最后在 调用bindService()绑定到Service的话,此时所触发的生命周期方法如下:
> **onCreate( )->onStartCommand( )->onBind( )->onUnbind( )->onRebind( )**

|              | **启动service的方式**                                    | **停止service的方式**                                        | **service与启动它的组件之间的通信方式**                      | **service的生命周期**                                        |
| ------------ | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| startService | 在其他组件中调用startService()方法后，服务即处于启动状态 | service中调用stopSelf()方法，或者其他组件调用stopService()方法后，service将停止运行 | 没有提供默认的通信方式，启动service后该service就处于独立运行状态 | `一旦启动，service即可在后台无限期运行，即使启动service的组件已被销毁也不受其影响，直到其被停止` |
| bindService  | 在其他组件中调用bindService()方法后，服务即处于启动状态  | 所有与service绑定的组件都被销毁，或者它们都调用了unbindService()方法后，service将停止运行 | 可以通过 ServiceConnection进行通信，组件可以与service进行交互、发送请求、获取结果，甚至是利用IPC跨进程执行这些操作 | 当所有与其绑定的组件都取消绑定(可能是组件被销毁也有可能是其调用了unbindService()方法)后，service将停止 |

### 3. 创建Service Demo

- 创建一个类继承自`Service`(或它的子类，如`IntentService`)，重写里面的一些键的回调方法，如`onStartCommand()`，`onBind()`等
- 在Manifests文件里面为其声明，并根据需要配置一些其他属性。

> 每当用户调用bindService()，Android都将之视作是要建立一个新的“逻辑连接”。而当连接建立起来时，系统会回调ServiceConnection 接口的onServiceConnected()。

```java
public class ServiceMainActivity extends Activity {
	private Button startBtn;
	private Button stopBtn;
	private Button bindBtn;
	private Button unBindBtn;
	private static final String TAG = "MainActivity";
	private LocalService myService;
 
 
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.service);
		startBtn = (Button) findViewById(R.id.start);
		stopBtn = (Button) findViewById(R.id.stop);
		bindBtn = (Button) findViewById(R.id.bind);
		unBindBtn = (Button) findViewById(R.id.unbind);
		startBtn.setOnClickListener(new MyOnClickListener());
		stopBtn.setOnClickListener(new MyOnClickListener());
		bindBtn.setOnClickListener(new MyOnClickListener());
		unBindBtn.setOnClickListener(new MyOnClickListener());
	}
 
 
	class MyOnClickListener implements OnClickListener {
		@Override
		public void onClick(View v) {
			Intent intent = new Intent();
			intent.setClass(ServiceMainActivity.this, LocalService.class);
			switch (v.getId()) {
			case R.id.start:						
				// 启动Service
				startService(intent);
				toast("startService");
				break;
			case R.id.stop:
				// 停止Service
				stopService(intent);
				toast("stopService");
				break;
			case R.id.bind:
				// 绑定Service
				bindService(intent, conn, Service.BIND_AUTO_CREATE);
				toast("bindService");
				break;
			case R.id.unbind:
				// 解除Service
				unbindService(conn);
				toast("unbindService");
				break;
			}
		}
	}
 
 
	private void toast(final String tip){
		runOnUiThread(new Runnable() {					 
            @Override
            public void run() {
            	Toast.makeText(getApplicationContext(), tip, Toast.LENGTH_SHORT).show();                         
            }
        });
	}
	private ServiceConnection conn = new ServiceConnection() {
		@Override
		public void onServiceConnected(ComponentName name, IBinder service) {
			Log.e(TAG, "连接成功");
			// 当Service连接建立成功后，提供给客户端与Service交互的对象（根据Android Doc翻译的，不知道准确否。。。。）
			myService = ((LocalService.LocalBinder) service).getService();
		}

		@Override
		public void onServiceDisconnected(ComponentName name) {
			Log.e(TAG, "断开连接");
			myService = null;
		}
	};
}
```

```java
public class LocalService extends Service {
    private static final String TAG = "MyService";
    private final IBinder myBinder = new LocalBinder();


    @Override
    public IBinder onBind(Intent intent) {
        Log.e(TAG, "onBind()");
        //当其他组件调用bindService()方法时，此方法将会被调用
        //如果不想让这个service被绑定，在此返回null即可
        //Toast.makeText(this, "onBind()", Toast.LENGTH_SHORT).show();
        return myBinder;
    }


    // 调用startService方法或者bindService方法时创建Service时（当前Service未创建）调用该方法
    @Override
    public void onCreate() {
        Log.e(TAG, "onCreate()");
        //只在service创建的时候调用一次，可以在此进行一些一次性的初始化操作
        //Toast.makeText(this, "onCreate()", Toast.LENGTH_SHORT).show();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "onStartCommand");
        // 调用startService方法启动Service时调用该方法
        return super.onStartCommand(intent, flags, startId);
    }
    
    @Override
    public void onStart(Intent intent, int startId) {
        Log.e(TAG, "onStart()");		
        //Toast.makeText(this, "onStart()", Toast.LENGTH_SHORT).show();
    }
    // Service创建并启动后在调用stopService方法或unbindService方法时调用该方法
    @Override
    public void onDestroy() {
        Log.e(TAG, "onDestroy()");
        //service调用的最后一个方法
        //在此进行资源的回收
        //Toast.makeText(this, "onDestroy()", Toast.LENGTH_SHORT).show();
    }
    //提供给客户端访问
    public class LocalBinder extends Binder {
        LocalService getService() {
            return LocalService.this;
        }
    }
}
```

```xml
<!-- 注册Service -->  
<service android:name="LocalService">  
    <intent-filter>  
        <action android:name="cn.fansunion.service.LocalService" />  
    </intent-filter>  
</service> 
```

```java
//启动服务
Intent startIntent = new Intent(this, ServiceDemo.class);  
startService(startIntent);
//停止服务
Intent stopIntent = new Intent(this, ServiceDemo.class);
stopService(stopIntent);
```

### 3. IntentService

> 1. Service不是一个单独的进程,它和它的应用程序在同一个进程中
> 2. Service不是一个线程,这样就意味着我们应该避免在Service中进行耗时操作
>
> 客户端通过`startService(Intent)来启动IntentService`; 我们并不需要手动地区控制IntentService,`当任务执行完后,IntentService会自动停止;` `可以启动IntentService多次,每个耗时操作会以工作队列的方式在IntentService的 onHandleIntent回调方法中执行`,并且`每次只会执行一个工作线程,`执行完一，再到二这样, 如果要并行使用service

```java
public class TestService3 extends IntentService {  
    private final String TAG = "hehe";  
    //必须实现父类的构造方法  
    public TestService3()  
    {  
        super("TestService3");  
    }  
    //必须重写的核心方法  
    @Override  
    protected void onHandleIntent(Intent intent) {  
        //Intent是从Activity发过来的，携带识别参数，根据参数不同执行不同的任务  
        String action = intent.getExtras().getString("param");  
        if(action.equals("s1"))Log.i(TAG,"启动service1");  
        else if(action.equals("s2"))Log.i(TAG,"启动service2");  
        else if(action.equals("s3"))Log.i(TAG,"启动service3");  
          
        //让服务休眠2秒  
        try{  
            Thread.sleep(2000);  
        }catch(InterruptedException e){e.printStackTrace();}          
    }  
    //重写其他方法,用于查看方法的调用顺序  
    @Override  
    public IBinder onBind(Intent intent) {  
        Log.i(TAG,"onBind");  
        return super.onBind(intent);  
    }  
    @Override  
    public void onCreate() {  
        Log.i(TAG,"onCreate");  
        super.onCreate();  
    }  
    @Override  
    public int onStartCommand(Intent intent, int flags, int startId) {  
        Log.i(TAG,"onStartCommand");  
        return super.onStartCommand(intent, flags, startId);  
    }  
    @Override  
    public void setIntentRedelivery(boolean enabled) {  
        super.setIntentRedelivery(enabled);  
        Log.i(TAG,"setIntentRedelivery");  
    }  
    @Override  
    public void onDestroy() {  
        Log.i(TAG,"onDestroy");  
        super.onDestroy();  
    }      
} 
```

- **AndroidManifest.xml注册下Service**

```xml
<service android:name=".TestService3" android:exported="false">  
    <intent-filter >  
        <action android:name="com.test.intentservice"/>  
    </intent-filter>  
</service>  
```

- MainActivity.java测试

```java
public class MainActivity extends Activity {  
    @Override  
    protected void onCreate(Bundle savedInstanceState) {  
        super.onCreate(savedInstanceState);  
        setContentView(R.layout.activity_main);  
        Intent it1 = new Intent("com.test.intentservice");  
        Bundle b1 = new Bundle();  
        b1.putString("param", "s1");  
        it1.putExtras(b1);  
        Intent it2 = new Intent("com.test.intentservice");  
        Bundle b2 = new Bundle();  
        b2.putString("param", "s2");  
        it2.putExtras(b2);  
        Intent it3 = new Intent("com.test.intentservice");  
        Bundle b3 = new Bundle();  
        b3.putString("param", "s3");  
        it3.putExtras(b3);  
        //接着启动多次IntentService,每次启动,都会新建一个工作线程  
        //但始终只有一个IntentService实例  
        startService(it1);  
        startService(it2);  
        startService(it3);  
    }  
} 
```

### 4. Activity 与 Service 通信

> 交流的媒介就是Service中的onBind()方法！ 返回一个我们自定义的Binder对象！
>
> - `自定义Service中`，`自定义一个Binder类`，然后将需要暴露的方法都写到该类中！
> - Service类中`，实例化这个自定义Binder类`，然后`重写onBind()方法，将这个Binder对象返回`！
> - Activity类中`实例化一个ServiceConnection对象`，重写`onServiceConnected()方法，然后 获取Binder对象`，然后通过Binder对象调用相关方法即可！

### 5. 定时后台程序

> - **Step 1：获得Service:** AlarmManager manager = (AlarmManager) getSystemService(ALARM_SERVICE);
> - **Step 2：通过set方法设置定时任务** int anHour = 2 * 1000; long triggerAtTime = SystemClock.elapsedRealtime() + anHour; manager.set(AlarmManager.RTC_WAKEUP,triggerAtTime,pendingIntent);
> - **Step 3：定义一个Service** 在onStartCommand中开辟一条事务线程,用于处理一些定时逻辑
> - **Step 4：定义一个Broadcast(广播)，用于启动Service** 最后别忘了，在AndroidManifest.xml中对这Service与Boradcast进行注册！

```java
public class LongRunningService extends Service {
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        //这里开辟一条线程,用来执行具体的逻辑操作:
        new Thread(new Runnable() {
            @Override
            public void run() {
                Log.d("BackService", new Date().toString());
            }
        }).start();
        AlarmManager manager = (AlarmManager) getSystemService(ALARM_SERVICE);
        //这里是定时的,这里设置的是每隔两秒打印一次时间=-=,自己改
        int anHour = 2 * 1000;
        long triggerAtTime = SystemClock.elapsedRealtime() + anHour;
        Intent i = new Intent(this,AlarmReceiver.class);
        PendingIntent pi = PendingIntent.getBroadcast(this, 0, i, 0);
        manager.set(AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAtTime, pi);
        return super.onStartCommand(intent, flags, startId);
    }
}
```

```java
public class AlarmReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        Intent i = new Intent(context,LongRunningService.class);
        context.startService(i);
    }
}
```

> - 与绑定本地Service不同的是,`本地Service 的onBind()方法会直接把IBinder对象本身传给客户端的ServiceConnection 的 onServiceConnected方法的第二个参数`。但`远程Service 的onBind()方法只是将IBinder对象的代理传给客户端的ServiceConnection 的 onServiceConnected方法的第二个参数`。
> - 当客户端获取了远程Service的 IBinder对象的代理之后，接下来就可通过该IBinder对象去回调远程Service的属性或方法了。

### 6. Binder 机制

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108214916019.png)

#### .1. 定义基本类

- Pet.java

```java
public class Pet implements Parcelable
{
	private String name;
	private double weight;
	public Pet()
	{
	}
	public Pet(String name, double weight)
	{
		super();
		this.name = name;
		this.weight = weight;
	}

	public String getName()
	{
		return name;
	}

	public void setName(String name)
	{
		this.name = name;
	}

	public double getWeight()
	{
		return weight;
	}

	public void setWeight(double weight)
	{
		this.weight = weight;
	}

	@Override
	public int describeContents()
	{
		return 0;
	}

	@Override
	public void writeToParcel(Parcel dest, int flags)
	{
		// 把该对象所包含的数据写到Parcel
		dest.writeString(name);
		dest.writeDouble(weight);
	}
	// 添加一个静态成员,名为CREATOR,该对象实现了Parcelable.Creator接口
	public static final Creator<Pet> CREATOR = new Creator<Pet>()
	{
		@Override
		public Pet createFromParcel(Parcel in)
		{
			// 从Parcel中读取数据，返回Person对象
			return new Pet(in.readString()
					, in.readDouble());
		}
		@Override
		public Pet[] newArray(int size)
		{
			return new Pet[size];
		}
	};
	@Override
	public String toString()
	{
		return "Pet [name=" + name + ", weight=" + weight + "]";
	}
}
```

- Person.java

```java
ublic class Person implements Parcelable
{
	private int id;
	private String name;
	private String pass;

	public Person()
	{
	}
	public Person(Integer id, String name, String pass)
	{
		super();
		this.id = id;
		this.name = name;
		this.pass = pass;
	}
	@Override
	public boolean equals(Object o)
	{
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;
		Person person = (Person) o;
		return Objects.equals(name, person.name) &&
				Objects.equals(pass, person.pass);
	}
	@Override
	public int hashCode()
	{
		return Objects.hash(name, pass);
	}

	// 实现Parcelable接口必须实现的方法
	@Override
	public int describeContents()
	{
		return 0;
	}
	// 实现Parcelable接口必须实现的方法
	@Override
	public void writeToParcel(Parcel dest, int flags)
	{
		// 把该对象所包含的数据写到Parcel
		dest.writeInt(id);
		dest.writeString(name);
		dest.writeString(pass);
	}
	// 添加一个静态成员,名为CREATOR,该对象实现了Parcelable.Creator接口
	public static final Creator<Person> CREATOR = new Creator<Person>() //①
	{
		@Override
		public Person createFromParcel(Parcel in)
		{
			return new Person(in.readInt(), in.readString(), in.readString());
		}
		@Override
		public Person[] newArray(int size)
		{
			return new Person[size];
		}
	};
}
```

#### .2. 定义AIDL接口

```java
package org.crazyit.service;

parcelable Pet;

package org.crazyit.service;
parcelable Person;

interface IPet
{
	// 定义一个Person对象作为传入参数
	List<Pet> getPets(in Person owner);
}

```

#### .3. 定义serverce

```java
public class ParcelableService extends Service
{
	private PetBinder petBinder;
	private static Map<Person , List<Pet>> pets = new HashMap<>();
	static
	{
		// 初始化pets Map集合
		List<Pet> list1 = new ArrayList<>();
		list1.add(new Pet("旺财" , 4.3));
		list1.add(new Pet("来福" , 5.1));
		pets.put(new Person(1, "sun" , "sun") , list1);
		ArrayList<Pet> list2 = new ArrayList<>();
		list2.add(new Pet("kitty" , 2.3));
		list2.add(new Pet("garfield" , 3.1));
		pets.put(new Person(2, "bai" , "bai") , list2);
	}
	// 继承Stub，也就是实现了IPet接口，并实现了IBinder接口
	class PetBinder extends IPet.Stub
	{
		@Override public List<Pet> getPets(Person owner)
		{
			// 返回Service内部的数据
			return pets.get(owner);
		}
	}
	@Override public void onCreate()
	{
		super.onCreate();
		petBinder = new PetBinder();
	}
	public IBinder onBind(Intent intent)
	{
		/* 返回catBinder对象
		 * 在绑定本地Service的情况下，该catBinder对象会直接
		 * 传给客户端的ServiceConnection对象
		 * 的onServiceConnected方法的第二个参数
		 * 在绑定远程Service的情况下，只将catBinder对象的代理
		 * 传给客户端的ServiceConnection对象
		 * 的onServiceConnected方法的第二个参数
		 */
		return petBinder; // ①
	}
	@Override public void onDestroy()
	{
	}
}
```



- 手机上的其他服务：电话管理，音频管理，振动器管理，闹钟服务
- Binder  todo？ 具体原理，和RPC 区别是什么


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_service/  

