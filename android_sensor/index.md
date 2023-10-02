# Android_Sensor


> **传感器的定义**：一种物理设备或者生物器官，能够探测、感受外界的信号，物理条件(如光，热， 适度)或化学组成（如烟雾），并将探知的信息传递给其他的设备或者器官！
>
> **传感器的种类**：可以从不同的角度对传感器进行划分，转换原理(传感器工作的基本物理或化学 效应)；用途；输出信号以及制作材料和工艺等。一般是按工作原来来分：物理传感器与化学传感器 两类！

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211109085703629.png)

### 1. Sensor架构

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211109090518291.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211109090548692.png)

- Application Framework： Sensor应用程序通过Sensor应用框架来获取sensor数据，应用框架层的Sensor Manager通过JNI与C++层进行通信。
- Sensor Libraries: Sensor中间层主要由Sensor Manager、Sensor service和Sensor硬件抽象层组成。
- Input Subsystem: 通用的Linux输入框架专为与键盘、鼠标和触摸屏等输入设备而设计，并定义了一套标准事件集合。Sensor输入子系统采用采用了通用的Linux输入框架，它通过/sys/class/input节点和用户空间进行交互。
- Event Dev: Evdev提供了一种访问/dev/input/eventX输入设备事件的通用方法。
- AccelerometerDriver: 驱动通过SIRQ和I2C总线与MMA7660模组进行通信。SIRQ用来产生传感器事件中断。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211109091822424.png)

### 2. 查看手机支持的传感器

```xml
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    <ScrollView
        android:layout_width="match_parent"
        android:layout_height="match_parent">
        <TextView
            android:id="@+id/txt_show"
            android:layout_width="match_parent"
            android:layout_height="match_parent" />
    </ScrollView>
</RelativeLayout>
```

```java
public class MainActivity extends AppCompatActivity {

    private TextView txt_show;
    private SensorManager sm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        sm = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        txt_show = (TextView) findViewById(R.id.txt_show);

        List<Sensor> allSensors = sm.getSensorList(Sensor.TYPE_ALL);
        StringBuilder sb = new StringBuilder();

        sb.append("此手机有" + allSensors.size() + "个传感器，分别有：\n\n");
        for(Sensor s:allSensors){
            switch (s.getType()){
                case Sensor.TYPE_ACCELEROMETER:
                    sb.append(s.getType() + " 加速度传感器(Accelerometer sensor)" + "\n");
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    sb.append(s.getType() + " 陀螺仪传感器(Gyroscope sensor)" + "\n");
                    break;
                case Sensor.TYPE_LIGHT:
                    sb.append(s.getType() + " 光线传感器(Light sensor)" + "\n");
                    break;
                case Sensor.TYPE_MAGNETIC_FIELD:
                    sb.append(s.getType() + " 磁场传感器(Magnetic field sensor)" + "\n");
                    break;
                case Sensor.TYPE_ORIENTATION:
                    sb.append(s.getType() + " 方向传感器(Orientation sensor)" + "\n");
                    break;
                case Sensor.TYPE_PRESSURE:
                    sb.append(s.getType() + " 气压传感器(Pressure sensor)" + "\n");
                    break;
                case Sensor.TYPE_PROXIMITY:
                    sb.append(s.getType() + " 距离传感器(Proximity sensor)" + "\n");
                    break;
                case Sensor.TYPE_TEMPERATURE:
                    sb.append(s.getType() + " 温度传感器(Temperature sensor)" + "\n");
                    break;
                default:
                    sb.append(s.getType() + " 其他传感器" + "\n");
                    break;
            }
            sb.append("设备名称：" + s.getName() + "\n 设备版本：" + s.getVersion() + "\n 供应商："
                    + s.getVendor() + "\n\n");
        }
        txt_show.setText(sb.toString());
    }
}
```

```java
SensorManager sm = (SensorManager)getSystemService(SENSOR_SERVICE); 
List<Sensor> allSensors = sm.getSensorList(Sensor.TYPE_ALL);
for(Sensor s:allSensors){
    sensor.getName();   //获得传感器名称
    sensor.getType();     //获得传感器种类
    sensor.getVendor();    //获得传感器供应商
    sensor.getVersion();    //获得传感器版本
    sensor.getResolution();  //获得精度值
    sensor.getMaximumRange(); //获得最大范围
    sensor.getPower();        //传感器使用时的耗电量 
}
```

### 3. 传感器使用

#### .1. 使用步骤

- Step 1：获得传感器管理器

```java
SensorManager sm = (SensorManager)getSystemService(SENSOR_SERVICE); 
```

- //Step 2：调用特定方法获得需要的传感器：

```java
Sensor mSensorOrientation = sm.getDefaultSensor(Sensor.TYPE_ORIENTATION);
```

- Step 3：实现SensorEventListener接口，重写onSensorChanged和onAccuracyChanged的方法
  - **onSensorChanged**：当传感器的值变化时会回调
  - **onAccuracyChanged**：当传感器的精度发生改变时会回调

```java
@Override
public void onSensorChanged(SensorEvent event) {
    final float[] _Data = event.values;
    this.mService.onSensorChanged(_Data[0],_Data[1],_Data[2]);
}
@Override
public void onAccuracyChanged(Sensor sensor, int accuracy) {
}
```

- Step 4：SensorManager对象调用registerListener注册监听器：
  - **SENSOR_DELAY_FASTEST**——延时：**0ms**
  - **SENSOR_DELAY_GAME**——延时：**20ms**
  - **SENSOR_DELAY_UI**——延时：**60ms**    推荐
  - **SENSOR_DELAY_NORMAL**——延时：**200ms**

```java
ms.registerListener(mContext, mSensorOrientation, android.hardware.SensorManager.SENSOR_DELAY_UI);
```

- Step 5：监听器的取消注册

```java
ms.unregisterListener(mContext, mSensorOrientation, android.hardware.SensorManager.SENSOR_DELAY_UI);
```

```java
public class MainActivity extends Activity implements SensorEventListener
{
	// 定义Sensor管理器
	private SensorManager mSensorManager;
	private TextView etOrientation;
	private TextView etGyro;
	private TextView etMagnetic;
	private TextView etGravity;
	private TextView etLinearAcc;
	private TextView etTemerature;
	private TextView etHumidity;
	private TextView etLight;
	private TextView etPressure;

	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取界面上的TextView组件
		etOrientation = findViewById(R.id.etOrientation);
		etGyro = findViewById(R.id.etGyro);
		etMagnetic = findViewById(R.id.etMagnetic);
		etGravity = findViewById(R.id.etGravity);
		etLinearAcc = findViewById(R.id.etLinearAcc);
		etTemerature = findViewById(R.id.etTemerature);
		etHumidity = findViewById(R.id.etHumidity);
		etLight = findViewById(R.id.etLight);
		etPressure = findViewById(R.id.etPressure);
		// 获取传感器管理服务
		mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);  // ①
	}
	@Override
	public void onResume()
	{
		super.onResume();
		// 为系统的方向传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的陀螺仪传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的磁场传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的重力传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的线性加速度传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的温度传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的湿度传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_RELATIVE_HUMIDITY),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的光传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_LIGHT),
				SensorManager.SENSOR_DELAY_GAME);
		// 为系统的压力传感器注册监听器
		mSensorManager.registerListener(this,
				mSensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE),
				SensorManager.SENSOR_DELAY_GAME);
	}
	@Override
	public void onPause()
	{
		// 程序暂停时取消注册传感器监听器
		mSensorManager.unregisterListener(this);
		super.onPause();
	}
	// 以下是实现SensorEventListener接口必须实现的方法
	// 当传感器精度改变时回调该方法
	@Override
	public void onAccuracyChanged(Sensor sensor, int accuracy)
	{
	}
	@Override
	public void onSensorChanged(SensorEvent event)
	{
		float[] values = event.values;
		// 获取触发event的传感器类型
		int sensorType = event.sensor.getType();
		StringBuilder sb;
		// 判断是哪个传感器发生改变
		switch (sensorType)
		{
			// 方向传感器
			case Sensor.TYPE_ORIENTATION:
				sb = new StringBuilder();
				sb.append("绕Z轴转过的角度：");
				sb.append(values[0]);
				sb.append("\n绕X轴转过的角度：");
				sb.append(values[1]);
				sb.append("\n绕Y轴转过的角度：");
				sb.append(values[2]);
				etOrientation.setText(sb.toString());
				break;
			// 陀螺仪传感器
			case Sensor.TYPE_GYROSCOPE:
				sb = new StringBuilder();
				sb.append("绕X轴旋转的角速度：");
				sb.append(values[0]);
				sb.append("\n绕Y轴旋转的角速度：");
				sb.append(values[1]);
				sb.append("\n绕Z轴旋转的角速度：");
				sb.append(values[2]);
				etGyro.setText(sb.toString());
				break;
			// 磁场传感器
			case Sensor.TYPE_MAGNETIC_FIELD:
				sb = new StringBuilder();
				sb.append("X轴方向上的磁场强度：");
				sb.append(values[0]);
				sb.append("\nY轴方向上的磁场强度：");
				sb.append(values[1]);
				sb.append("\nZ轴方向上的磁场强度：");
				sb.append(values[2]);
				etMagnetic.setText(sb.toString());
				break;
			// 重力传感器
			case Sensor.TYPE_GRAVITY:
				sb = new StringBuilder();
				sb.append("X轴方向上的重力：");
				sb.append(values[0]);
				sb.append("\nY轴方向上的重力：");
				sb.append(values[1]);
				sb.append("\nZ轴方向上的重力：");
				sb.append(values[2]);
				etGravity.setText(sb.toString());
				break;
			// 线性加速度传感器
			case Sensor.TYPE_LINEAR_ACCELERATION:
				sb = new StringBuilder();
				sb.append("X轴方向上的线性加速度：");
				sb.append(values[0]);
				sb.append("\nY轴方向上的线性加速度：");
				sb.append(values[1]);
				sb.append("\nZ轴方向上的线性加速度：");
				sb.append(values[2]);
				etLinearAcc.setText(sb.toString());
				break;
			// 温度传感器
			case Sensor.TYPE_AMBIENT_TEMPERATURE:
				sb = new StringBuilder();
				sb.append("当前温度为：");
				sb.append(values[0]);
				etTemerature.setText(sb.toString());
				break;
			// 湿度传感器
			case Sensor.TYPE_RELATIVE_HUMIDITY:
				sb = new StringBuilder();
				sb.append("当前湿度为：");
				sb.append(values[0]);
				etHumidity.setText(sb.toString());
				break;
			// 光传感器
			case Sensor.TYPE_LIGHT:
				sb = new StringBuilder();
				sb.append("当前光的强度为：");
				sb.append(values[0]);
				etLight.setText(sb.toString());
				break;
			// 压力传感器
			case Sensor.TYPE_PRESSURE:
				sb = new StringBuilder();
				sb.append("当前压力为：");
				sb.append(values[0]);
				etPressure.setText(sb.toString());
				break;
		}
	}
}
```

#### .2. APK 的监听

> 1、Activity实现了SensorEventListener接口。
> 2、在onCreate方法中，获取SystemSensorManager，并获取到加速传感器的Sensor；
> 3、在onResume方法中调用SystemSensorManager，registerListenerImpl注册监听器；
> 4、当Sensor数据有改变的时候将会回调onSensorChanged方法。

```java
SensorManager  mSensorManager =
    (SensorManager)getSystemService(SENSOR_SERVICE);
Sensor   mAccelerometer =
    mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
protected void onResume() {
    super.onResume();
    mSensorManager.registerListenerImpl (this, mAccelerometer,
                                         SensorManager.SENSOR_DELAY_NORMAL);
}
protected void onPause() {
    super.onPause();
    mSensorManager.unregisterListener(this);
}
public interface SensorEventListener {
    public void onSensorChanged(SensorEvent event);
    public void onAccuracyChanged(Sensor sensor, int accuracy);   
}
```

#### .3. 初始化SystemSensorManager

> 系统开机启动的时候，会创建SystemSensorManager的实例，在其构造函数中，主要做了四件事情：
> 1、初始化JNI：`调用JNI函数nativeClassInit()进行初始化`
> 2、`初始化Sensor列表`：调用JNI函数sensors_module_init，对Sensor模块进行初始化。创建了native层SensorManager的实例。
> 3、`获取Sensor列表`：调用JNI函数sensors_module_get_next_sensor()获取Sensor，并存在sHandleToSensor列表中
> 4、`构造SensorThread类`：构造线程的类函数，并没有启动线程，`当有应用注册的时候才会启动线程`

```java
public SystemSensorManager(Context context,Looper mainLooper) {
    mMainLooper = mainLooper;       
    mContext = context;

    synchronized(sListeners) {
        if (!sSensorModuleInitialized) {
            sSensorModuleInitialized = true;

            nativeClassInit();

            // initialize the sensor list
            sensors_module_init();
            final ArrayList<Sensor> fullList = sFullSensorsList;
            int i = 0;
            do {
                Sensor sensor = new Sensor();
                i = sensors_module_get_next_sensor(sensor, i);

                if (i>=0) {
                    //Log.d(TAG, "found sensor: " + sensor.getName() +
                    //        ", handle=" + sensor.getHandle());
                    fullList.add(sensor);
                    sHandleToSensor.append(sensor.getHandle(), sensor);
                }
            } while (i>0);

            sPool = new SensorEventPool( sFullSensorsList.size()*2 );
            sSensorThread = new SensorThread();
        }
    }
}
```

#### .4.  启动SensorThread线程读取消息队列中数据

> 当`有应用程序调用registerListenerImpl()方法注册监听的时候，会调用SensorThread.startLoacked() 启动线程`。线程只会启动一次，并调用enableSensorLocked()接口对指定的sensor使能，并设置采样时间。SensorThreadRunnable实现了Runnable接口，在SensorThread类中被启动

> 在open函数中调用JNI函数`sensors_create_queue()来创建消息队列`,然后调用SensorManager. createEventQueue()创建。在startLocked函数中启动新的线程后，做了一个while的等待while (mSensorsReady == false)，`只有当mSensorsReady等于true的时候，才会执行enableSensorLocked()函数对sensor使能`。而`mSensorsReady变量，是在open()调用创建消息队列成功之后才会true`，所以认为，三个功能调用顺序是如下：
> 1、调用`open函数创建消息队列`
> 2、调用`enableSensorLocked()函数对sensor使能`
> 3、调用s`ensors_data_poll从消息队列中读取数据，而且是在while (true)循环里一直读取`

```java
protected boolean registerListenerImpl(SensorEventListener listener, Sensor sensor,
                                       int delay, Handler handler) {
    synchronized (sListeners) {
        ListenerDelegate l = null;
        for (ListenerDelegate i : sListeners) {
            if (i.getListener() == listener) {
                l = i;
            }
        }
        …….
            // if we don't find it, add it to the list
            if (l == null) {
                l = new ListenerDelegate(listener, sensor, handler);
                sListeners.add(l);
                ……
                    if (sSensorThread.startLocked()) {
                        if (!enableSensorLocked(sensor, delay)) {
                            …….
                        }
                        ……
                    } else if (!l.hasSensor(sensor)) {
                        l.addSensor(sensor);
                        if (!enableSensorLocked(sensor, delay)) {
                            ……
                        }
                    }
            }
        return result;
    }

    boolean startLocked() {
        try {
            if (mThread == null) {
                SensorThreadRunnable runnable = new SensorThreadRunnable();
                Thread thread = new Thread(runnable, SensorThread.class.getName());
                thread.start();
                synchronized (runnable) {  //队列创建成功,线程同步
                    while (mSensorsReady == false) {
                        runnable.wait();
                    }
                }

            }
            private class SensorThreadRunnable implements Runnable {
                SensorThreadRunnable() {
                }
                private boolean open() {
                    sQueue = sensors_create_queue();
                    return true;
                }
                public void run() {
                    …….
                        if (!open()) {
                            return;
                        }
                    synchronized (this) {
                        mSensorsReady = true;
                        this.notify();
                    }
                    while (true) {
                        final int sensor = sensors_data_poll(sQueue, values, status, timestamp);
                        …….
                    }
                }
            }

```



### 4. 通讯方式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211109093121753.png)

> - 客户端
>   - SensorManager.cpp
>     - 负责和服务端SensorService.cpp的通信
>     - SensorEventQueue.cpp 消息队列
>     - 接口
>       - getSensorList()
>       - getDefaultSensor()
> - 服务端
>   - SensorService.cpp
>     - 服务端数据处理中心
>     - SensorEventConnection 从BnSensorEventConnection继承来，实现接口ISensorEventConnection的一些方法，**ISensorEventConnection在SensorEventQueue会保存一个指针，指向调用服务接口创建的SensorEventConnection对象**
>     - SensorService创建完之后，将会调用SensorService::onFirstRef()方法，在该方法中完成初始化工作。
>     - 首先获取SensorDevice实例，在其构造函数中，完成了对Sensor模块HAL的初始化：
>   - Bittube.cpp
>     - 在这个类中创建管道，用于服务端与客户端读写数据
>   - SensorDevice
>     - 负责与HAL读取数据

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211109092103749.png)

### Resource

- https://blog.csdn.net/u010164190/article/details/51908477

- https://www.136.la/jingpin/show-181250.html  代码详解

- 官方文档：https://source.android.google.cn/devices/sensors/sensors-multihal?hl=zh-cn  

  

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android_sensor/  

