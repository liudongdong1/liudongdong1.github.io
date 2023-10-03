# Android_GPS&WiFi使用


### 1. GPS 使用

```java
public class MainActivity extends Activity
{
	// 定义LocationManager对象
	private LocationManager locManager;
	// 定义程序界面中的TextView组件
	private TextView show;
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取程序界面上的EditText组件
		show = findViewById(R.id.show);
		requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, 0x123);
	}
	@SuppressLint("MissingPermission")
	@Override
	public void onRequestPermissionsResult(int requestCode,
										   @NonNull String[] permissions, @NonNull int[] grantResults)
	{
		// 如果用户允许使用GPS定位信息
		if(requestCode == 0x123	&& grantResults.length == 1
				&& grantResults[0] == PackageManager.PERMISSION_GRANTED)
		{
			// 创建LocationManager对象
			locManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
			// 从GPS获取最近的定位信息
			Location location =
					locManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
			// 使用location来更新EditText的显示
			updateView(location);
			// 设置每3秒获取一次GPS定位信息
			locManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
					3000, 8f, new LocationListener() // ①
			{
				@Override
				public void onLocationChanged(Location location)
				{
					// 当GPS定位信息发生改变时，更新位置
					updateView(location);
				}

				@Override
				public void onStatusChanged(String provider, int status, Bundle extras)
				{}

				@Override
				public void onProviderEnabled(String provider)
				{
					// 当GPS LocationProvider可用时，更新位置
					updateView(locManager.getLastKnownLocation(provider));
				}

				@Override
				public void onProviderDisabled(String provider)
				{
					updateView(null);
				}
			});
		}
	}
	// 更新EditText中显示的内容
	public void updateView(Location newLocation)
	{
		if (newLocation != null)
		{
			String sb = "实时的位置信息：\n" +
					"经度：" +
					newLocation.getLongitude() +
					"\n纬度：" +
					newLocation.getLatitude() +
					"\n高度：" +
					newLocation.getAltitude() +
					"\n速度：" +
					newLocation.getSpeed() +
					"\n方向：" +
					newLocation.getBearing();
			show.setText(sb);
		}
		else
		{
			// 如果传入的Location对象为空，则清空EditText
			show.setText("");
		}
	}
}
```

### 2. WiFi使用

```java
public class MainActivity extends Activity
{
	WifiRttManager mWifiRttManager;
	// 定义监听Wi-Fi状态改变的BroadcastReceiver
	public class WifiChangeReceiver extends BroadcastReceiver
	{
		@Override
		public void onReceive(Context context, Intent intent)
		{
			if (WifiManager.SCAN_RESULTS_AVAILABLE_ACTION.equals(intent.getAction()))
			{
				// 开始执行Wi-Fi定位
				startWifiLoc();
			}
		}
	}
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 定义一个监听网络状态改变、Wi-Fi状态改变的IntentFilter
		IntentFilter wifiFilter = new IntentFilter();
		wifiFilter.addAction(WifiManager.NETWORK_STATE_CHANGED_ACTION);
		wifiFilter.addAction(WifiManager.WIFI_STATE_CHANGED_ACTION);
		wifiFilter.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);
		// 为IntentFilter注册BroadcastReceiver
		registerReceiver(new WifiChangeReceiver(), wifiFilter);
	}
	// 定义执行WIFI定位的方法
	@SuppressLint("MissingPermission")
	private void startWifiLoc()
	{
		// 获取WIFI管理器
		WifiManager wifiManager = (WifiManager) getSystemService(Context.WIFI_SERVICE);
		// 判断是否支持室内Wi-Fi定位功能
		boolean hasRtt = getPackageManager().hasSystemFeature(
				PackageManager.FEATURE_WIFI_RTT);
		System.out.println("是否具有室内WIFI定位功能：" + hasRtt);
		// 只有当版本大于Android 9时候才能使用室内WIFI定位功能
		if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P)
		{
			// 获取室内Wi-Fi定位管理器
			mWifiRttManager = (WifiRttManager)
					getSystemService(Context.WIFI_RTT_RANGING_SERVICE);  // ①
			RangingRequest request = new RangingRequest.Builder()
					// 添加Wi-Fi的扫描结果（即添加Wi-Fi访问点）
					.addAccessPoints(wifiManager.getScanResults())
					// 创建RangingRequest对象
					.build();  // ②
			// 开始请求执行WIFI室内定位
			mWifiRttManager.startRanging(request, Executors.newCachedThreadPool(),
					new RangingResultCallback()   // ③
					{
						// 如果Wi-Fi定位出错时触发该方法
						@Override
						public void onRangingFailure(int code)
						{ }
						// 室内Wi-Fi定位返回结果时触发该方法
						@Override
						public void onRangingResults(@NonNull List<RangingResult> results)
						{
							// 通过RangingResult集合可获取与特定WIFI接入点之间的距离
							for(RangingResult rr : results)
							{
								System.out.println("与" + rr.getMacAddress()
										+ "WIFI的距离是:" + rr.getDistanceMm());
							}
						}
					});
		}
	}
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_gpswifi%E5%AE%9A%E4%BD%8D/  

