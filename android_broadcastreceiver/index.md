# Android_BroadcastReceiver


> 分为两个角色：广播发送者、广播接收者,使用了设计模式中的**观察者模式**：基于消息的发布 / 订阅事件模型
>
> 标准广播：发出广播后，该广播事件的接收者，`几乎会在同一时刻收到通知`，都可以响应或不响应该事件
>
> 有序广播：发出广播后，同一时刻，`只有一个广播接收者能收到、一个接收者处理完后之后，可以选择继续向下传递给其它接收者，也可以拦截掉广播`。[不常用、不推荐使用了]

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108224217036.png)

### 1. 自定义广播接收者

- 继承`BroadcastReceiver`基类；必须复写抽象方法`onReceive()`方法

> 1. 广播接收器接收到相应广播后，会自动回调 `onReceive()` 方法
> 2. `onReceive`方法会涉及 与 其他组件之间的交互，如发送`Notification`、启动`Service`等
> 3. 默认情况下，广播接收器运行在 `UI` 线程，因此，`onReceive()`方法不能执行耗时操作，否则将导致`ANR`

```java
// 继承BroadcastReceivre基类
public class mBroadcastReceiver extends BroadcastReceiver {
  // 复写onReceive()方法
  // 接收到广播后，则自动调用该方法
  @Override
  public void onReceive(Context context, Intent intent) {
   //写入接收广播后的操作
    }
}
```

### 2. 广播接收器注册

#### .1. 静态注册

> 注册方式：在AndroidManifest.xml里通过**<receive>**标签声明， 当此 `App`首次启动时，系统会**自动**实例化`mBroadcastReceiver`类，并注册到系统中。

```xml
<receiver 
          android:enabled=["true" | "false"]
          //此broadcastReceiver能否接收其他App的发出的广播
          //默认值是由receiver中有无intent-filter决定的：如果有intent-filter，默认值为true，否则为false
          android:exported=["true" | "false"]
          android:icon="drawable resource"
          android:label="string resource"
          //继承BroadcastReceiver子类的类名
          android:name=".mBroadcastReceiver"
          //具有相应权限的广播发送者发送的广播才能被此BroadcastReceiver所接收；
          android:permission="string"
          //BroadcastReceiver运行所处的进程
          //默认为app的进程，可以指定独立的进程
          //注：Android四大基本组件都可以通过此属性指定自己的独立进程
          android:process="string" >

    //用于指定此广播接收器将接收的广播类型
    //本示例中给出的是用于接收网络状态改变时发出的广播
    <intent-filter>
        <action android:name="android.net.conn.CONNECTIVITY_CHANGE" />
    </intent-filter>
</receiver>
```

#### .2. 动态注册

> 注册方式：在代码中调用`Context.registerReceiver（）`方法；动态广播最好在Activity 的onResume()注册，在onPause()注销，否则会导致内存泄漏。

```java
// 选择在Activity生命周期方法中的onResume()中注册
@Override
protected void onResume(){
    super.onResume();
    // 1. 实例化BroadcastReceiver子类 &  IntentFilter
    mBroadcastReceiver mBroadcastReceiver = new mBroadcastReceiver();
    IntentFilter intentFilter = new IntentFilter();
    // 2. 设置接收广播的类型
    intentFilter.addAction(android.net.conn.CONNECTIVITY_CHANGE);
    // 3. 动态注册：调用Context的registerReceiver（）方法
    registerReceiver(mBroadcastReceiver, intentFilter);
}
// 注册广播后，要在相应位置记得销毁广播
// 即在onPause() 中unregisterReceiver(mBroadcastReceiver)
// 当此Activity实例化时，会动态将MyBroadcastReceiver注册到系统中
// 当此Activity销毁时，动态注册的MyBroadcastReceiver将不再接收到相应的广播。
@Override
protected void onPause() {
    super.onPause();
    //销毁在onResume()方法中的广播
    unregisterReceiver(mBroadcastReceiver);
}
```

### 3. 广播发送者像AMS发送广播

- 广播 是 用”意图（`Intent`）“标识
- 定义广播的本质 = 定义广播所具备的“意图（`Intent`）”
- 广播发送 = 广播发送者 将此广播的“意图（`Intent`）”通过**sendBroadcast（）方法**发送出去

#### .1. 普通广播

> 若被注册了的广播接收者中注册时`intentFilter`的`action`与上述匹配，则会接收此广播（即进行回调`onReceive()`）。如下`mBroadcastReceiver`则会接收上述广播； 若发送广播有相应权限，则广播接收者也需要相应权限。

```java
Intent intent = new Intent();
//对应BroadcastReceiver中intentFilter的action
intent.setAction(BROADCAST_ACTION);
//发送广播
sendBroadcast(intent);
```

```xml
<receiver 
    //此广播接收者类是mBroadcastReceiver
    android:name=".mBroadcastReceiver" >
    //用于接收网络状态改变时发出的广播
    <intent-filter>
        <action android:name="BROADCAST_ACTION" />
    </intent-filter>
</receiver>
```

#### .2. 系统广播

> - Android中内置了多个系统广播：只要涉及到手机的基本操作（如开机、网络状态变化、拍照等等），都会发出相应的广播
> - 每个广播都有特定的Intent - Filter（包括具体的action）

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108225811137.png)

#### .3. 有序广播

> 发送出去的广播被广播接收者按照先后顺序接收；
>
> 1. 按照Priority属性值从大-小排序；
> 2. Priority属性相同者，动态注册的广播优先；

#### .4. App应用内广播

> - Android中的广播可以跨App直接通信（exported对于有intent-filter情况下默认值为true）
>   - 其他App针对性发出与当前App intent-filter相匹配的广播，由此导致当前App不断接收广播并处理；
>   - 其他App注册与当前App一致的intent-filter用于接收广播，获取广播具体信息；
>     即会出现安全性 & 效率性的问题。
> - App应用内广播可理解为一种局部广播，`广播的发送者和接收者都同属于一个App`。

- 将全局广播设置成局部广播
  - 注册广播时`将exported属性设置为false`，使得非本App内部发出的此广播不被接收；
  - 在广播发送和接收时，`增设相应权限permission`，用于权限验证；
  - 发送广播时`指定该广播接收器所在的包名`，此广播将只会发送到此包中的App内与之相匹配的有效广播接收器中。
- 使用封装好的LocalBroadcastManager类
  - 只是注册/取消注册广播接收器和发送广播时将`参数的context变成了LocalBroadcastManager的单一实例`

```java
//注册应用内广播接收器
//步骤1：实例化BroadcastReceiver子类 & IntentFilter mBroadcastReceiver 
mBroadcastReceiver = new mBroadcastReceiver(); 
IntentFilter intentFilter = new IntentFilter(); 

//步骤2：实例化LocalBroadcastManager的实例
localBroadcastManager = LocalBroadcastManager.getInstance(this);

//步骤3：设置接收广播的类型 
intentFilter.addAction(android.net.conn.CONNECTIVITY_CHANGE);

//步骤4：调用LocalBroadcastManager单一实例的registerReceiver（）方法进行动态注册 
localBroadcastManager.registerReceiver(mBroadcastReceiver, intentFilter);

//取消注册应用内广播接收器
localBroadcastManager.unregisterReceiver(mBroadcastReceiver);

//发送应用内广播
Intent intent = new Intent();
// 设置Intent的Action属性
intent.setAction("org.crazyit.action.CRAZY_BROADCAST");
intent.setPackage("org.crazyit.broadcast");
intent.putExtra("msg", "简单的消息");
// 发送广播
sendBroadcast(intent);
```

```xml
<receiver android:name=".MyReceiver" android:enabled="true"
          android:exported="false">
    <intent-filter>
        <!-- 指定该BroadcastReceiver所响应的Intent的Action -->
        <action android:name="org.crazyit.action.CRAZY_BROADCAST" />
    </intent-filter>
</receiver>
```

```java
public class MyReceiver extends BroadcastReceiver
{
	@Override
	public void onReceive(Context context, Intent intent)
	{
		Toast.makeText(context, "接收到的Intent的Action为：" + intent.getAction()
			+ "\n消息内容是：" + intent.getStringExtra("msg"),
			Toast.LENGTH_LONG).show();
	}
}
```

### 4. 音乐播放器案例

- MainActivity.java 启动播放音乐服务，并通过按钮注册监听事件，在处理事件中发送对应的广播到服务中
- MusicService.java 服务类定义广播接受类，接受UI传来的指令，并在播放完的时候，通过广播发送信息通知UI界面更新信息

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221118143538185.png)





---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_broadcastreceiver/  

