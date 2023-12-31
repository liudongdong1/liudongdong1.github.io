# Android_BlueTooth


> `Ble`开发中,存在着两个角色：中心设备角色和外围设备角色。
>
> - 外围设备：一般指非常小或者低功耗设备,更强大的中心设备可以连接外围设备为中心设备提供数据。外设会不停的向外广播，让中心设备知道它的存在。 例如小米手环。
> - 中心设备：可以扫描并连接多个外围设备,从外设中获取信息。
>
> ​		外围设备会设定一个广播间隔，每个广播间隔中，都会发送自己的广播数据。广播间隔越长，越省电。**一个没有被连接的`Ble`外设会不断发送广播数据**，这时可以被多个中心设备发现。**一旦外设被连接，则会马上停止广播。**

### 1. 打开蓝牙

```java
//初始化ble设配器
BluetoothManager manager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
BluetoothAdapter mBluetoothAdapter = manager.getAdapter();
//判断蓝牙是否开启，如果关闭则请求打开蓝牙
if (mBluetoothAdapter == null || !mBluetoothAdapter.isEnabled()) {
    //方式一：请求打开蓝牙
    Intent intent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
    startActivityForResult(intent, 1);
    //方式二：半静默打开蓝牙
    //低版本android会静默打开蓝牙，高版本android会请求打开蓝牙
    //mBluetoothAdapter.enable();  #判断当前蓝牙是否打开，如果蓝牙处于打开状态返回true。
}

```
- 在activity层通过广播监听蓝牙的关闭与开启，进行自己的逻辑处理：

```java
new BroadcastReceiver() {
    @Override
    public void onReceive(Context context, Intent intent) {
        //获取蓝牙广播  本地蓝牙适配器的状态改变时触发
        String action = intent.getAction();
        if (action.equals(BluetoothAdapter.ACTION_STATE_CHANGED)) {
            //获取蓝牙广播中的蓝牙新状态
            int blueNewState = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, 0);
            //获取蓝牙广播中的蓝牙旧状态
            int blueOldState = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, 0);
            switch (blueNewState) {
                //正在打开蓝牙
                case BluetoothAdapter.STATE_TURNING_ON:
                    break;
                    //蓝牙已打开
                case BluetoothAdapter.STATE_ON:
                    break;
                    //正在关闭蓝牙
                case BluetoothAdapter.STATE_TURNING_OFF:
                    break;
                    //蓝牙已关闭
                case BluetoothAdapter.STATE_OFF:
                    break;
            }
        }
    }
};
```

### 2. 扫描蓝牙

#### .1. 扫描设置

```java
//获取 5.0 的扫描类实例
mBLEScanner = mBluetoothAdapter.getBluetoothLeScanner();
//开始扫描
//可设置过滤条件，在第一个参数传入，但一般不设置过滤。
mBLEScanner.startScan(null,mScanSettings,mScanCallback);
//停止扫描
mBLEScanner.stopScan(mScanCallback);
```

```java
//如果没打开蓝牙，不进行扫描操作，或请求打开蓝牙。
if(!mBluetoothAdapter.isEnabled()) {
    return;
}
 //处于未扫描的状态  
if (!mScanning){
    //android 5.0后
    if(android.os.Build.VERSION.SDK_INT >= 21) {
        //标记当前的为扫描状态
        mScanning = true;
        //获取5.0新添的扫描类
        if (mBLEScanner == null){
            //mBLEScanner是5.0新添加的扫描类，通过BluetoothAdapter实例获取。
            mBLEScanner = mBluetoothAdapter.getBluetoothLeScanner();
        }
        //开始扫描 
        //mScanSettings是ScanSettings实例，mScanCallback是ScanCallback实例，后面进行讲解。
        mBLEScanner.startScan(null,mScanSettings,mScanCallback);
    } else {
        //标记当前的为扫描状态
        mScanning = true;
        //5.0以下  开始扫描
        //mLeScanCallback是BluetoothAdapter.LeScanCallback实例
        mBluetoothAdapter.startLeScan(mLeScanCallback);
    }
    //设置结束扫描
    mHandler.postDelayed(new Runnable() {
        @Override
        public void run() {
            //停止扫描设备
            if(android.os.Build.VERSION.SDK_INT >= 21) {
                //标记当前的为未扫描状态
                mScanning = false;
                mBLEScanner.stopScan(mScanCallback);
            } else {
                //标记当前的为未扫描状态
                mScanning = false;
                //5.0以下  停止扫描
                mBluetoothAdapter.stopLeScan(mLeScanCallback);
            }
        }
    },SCAN_TIME);
}
```

#### .2. 扫描回调

> - **回调函数中尽量不要做耗时操作！**
> - 一般蓝牙设备对象都是通过`onScanResult(int,ScanResult)`返回，而不会在`onBatchScanResults(List)`方法中返回，除非手机支持批量扫描模式并且开启了批量扫描模式。

```java
mScanCallback = new ScanCallback() {
    //当一个蓝牙ble广播被发现时回调
    @Override
    public void onScanResult(int callbackType, ScanResult result) {
        super.onScanResult(callbackType, result);
        //扫描类型有开始扫描时传入的ScanSettings相关
        //对扫描到的设备进行操作。如：获取设备信息。
        
    }
    //批量返回扫描结果
    //@param results 以前扫描到的扫描结果列表。
    @Override
    public void onBatchScanResults(List<ScanResult> results) {
        super.onBatchScanResults(results);
        
    }
    //当扫描不能开启时回调
    @Override
    public void onScanFailed(int errorCode) {
        super.onScanFailed(errorCode);
        //扫描太频繁会返回ScanCallback.SCAN_FAILED_APPLICATION_REGISTRATION_FAILED，表示app无法注册，无法开始扫描。
    
    }
};
```

#### .3. 扫描设置

> `ScanSettings`实例对象是通过`ScanSettings.Builder`构建的。通过`Builder`对象为`ScanSettings`实例设置扫描模式、回调类型、匹配模式等参数，用于配置`android 5.0` 的扫描参数。

```java
//创建ScanSettings的build对象用于设置参数
ScanSettings.Builder builder = new ScanSettings.Builder()
    //设置高功耗模式
    .setScanMode(SCAN_MODE_LOW_LATENCY);
    //android 6.0添加设置回调类型、匹配模式等
    if(android.os.Build.VERSION.SDK_INT >= 23) {
        //定义回调类型
        builder.setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
        //设置蓝牙LE扫描滤波器硬件匹配的匹配模式
        builder.setMatchMode(ScanSettings.MATCH_MODE_STICKY);
    }
//芯片组支持批处理芯片上的扫描
if (bluetoothadapter.isOffloadedScanBatchingSupported()) {
    //设置蓝牙LE扫描的报告延迟的时间（以毫秒为单位）
    //设置为0以立即通知结果
    builder.setReportDelay(0L);
}
builder.build();
```

### 3. 蓝牙通信

#### .1. 通信协议

##### 1. GAP

> GAP（Generic Access Profile）：使蓝牙设备对外界可见，并决定设备是否可以或者怎样与其他设备进行交互。
>
> - **中心设备**：可以扫描并连接多个外围设备,从外设中获取信息。
> - **外围设备**：小型，低功耗，资源有限的设备。可以连接到功能更强大的中心设备，并为其提供数据。
>
> GAP 中`外围设备`通过两种方式向外广播数据：`广播数据` 和 `扫描回复`( 每种**数据最长可以包含 31 byte**)。
>
> - 外设必需不停的向外广播，让中心设备知道它的存在。而扫描回复是可选的，中心设备可以向外设请求扫描回复，这里包含一些设备额外的信息。
> - 外围设备会设定一个广播间隔。每个广播间隔中，它会重新发送自己的广播数据。广播间隔越长，越省电，同时也不太容易扫描到。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211110095808015.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211110102735026.png)

`AD Type 类型代码`，Flags：TYPE = **0x01**。用来标识设备LE物理连接。

- bit 0: LE 有限发现模式
- bit 1: LE 普通发现模式
- bit 2: 不支持 BR/EDR
- bit 3: 对 Same Device Capable(Controller) 同时支持 BLE 和 BR/EDR
- bit 4: 对 Same Device Capable(Host) 同时支持 BLE 和 BR/EDR
- bit 5..7: 预留

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211110102917775.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211110102242036.png)

- Type = 0x01 表示设备LE物理连接。
- Type = 0x09 表示设备的全名
- Type = 0x03 表示完整的16bit `UUID`。其值为0xFFF7。
- Type = 0xFF 表示厂商数据。前两个字节表示厂商ID,即厂商ID为0x11。后面的为厂商数据，具体由用户自行定义。
- Type = 0x16 表示16 bit `UUID`的数据，所以前两个字节为`UUID`,即`UUID`为0xF117，后续为`UUID`对应的数据，具体由用户自行定义。

##### 2. GATT

>  GATT（Generic Attribute Profile): BLE设备通过叫做 **Service** 和 **Characteristic** 的东西进行通信, GATT使用了 ATT（Attribute Protocol）协议，ATT 协议把 Service, `Characteristic`对应的数据保存在一个查询表中，次查找表使用 16 bit ID 作为每一项的索引。
>
>  GATT 连接是**独占**的。也就是**一个 BLE 外设同时只能被一个中心设备连接。一旦外设被连接，它就会马上停止广播，这样它就对其他设备不可见了。当外设与中心设备断开，外设又开始广播，让其他中心设备感知该外设的存在。而中心设备可**同时与多个外设进行连接。

- **Profile**：并不是实际存在于 BLE 外设上的，它只是一个被 Bluetooth SIG 或者外设设计者预先定义的 `Service` 的集合。例如心率`Profile`（`Heart Rate Profile`）就是结合了 `Heart Rate Service` 和 `Device Information Service`。
- **Service**：包含一个或者多个 `Characteristic`。每个 `Service` 有一个 `UUID` 唯一标识。
- **Characteristic**： 是最小的逻辑数据单元。一个`Characteristic`包括一个单一value变量和0-n个用来描述`characteristic`变量的`Descriptor`。与 `Service` 类似，每个 `Characteristic` 用 16 bit 或者 128 bit 的 `UUID` 唯一标识。
  - 若16 bit UUID为xxxx，转换为128 bit UUID为`0000xxxx-0000-1000-8000-00805F9B34FB`
  - 若32 bit UUID为xxxxxxxx，转换为128 bit UUID为`xxxxxxxx-0000-1000-8000-00805F9B34FB`

### Resource

- https://juejin.cn/post/6844903731796901902#heading-12


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_bluetooth/  

