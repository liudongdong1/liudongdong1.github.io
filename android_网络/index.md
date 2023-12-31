# Android_网络


> 学习的案例代码都采用了Handler机制进行UI和后台请求的消息传输.
>
> - Android 支持JDK网络编程 ServerSocket， Socket，DatagramSocket，Datagrampacket，MulticastSocket
> - 支持JDK内置的URL，URLConnection，HttpURLConnection 等工具
> - 内置Apache HttpClient  工具
> - 现在网络采用retrofit Rx 模式

### 1. 基于TCP协议网络

- ClientThread

```java
public class ClientThread implements Runnable
{
	// 定义向UI线程发送消息的Handler对象
	private Handler handler;
	// 该线程所处理的Socket所对应的输入流
	private BufferedReader br;
	private OutputStream os;
	// 定义接收UI线程的消息的Handler对象
	Handler revHandler;
	ClientThread(Handler handler)
	{
		this.handler = handler;
	}

	@Override
	public void run()
	{
		try {
			Socket s = new Socket("192.168.1.88", 30000);
			br = new BufferedReader(new InputStreamReader(s.getInputStream()));
			os = s.getOutputStream();
			// 启动一条子线程来读取服务器响应的数据
			new Thread(() ->
			{
				String content;
				// 不断读取Socket输入流中的内容
				try
				{
					while ((content = br.readLine()) != null) {
						// 每当读到来自服务器的数据之后，发送消息通知
						// 程序界面显示该数据
						Message msg = new Message();
						msg.what = 0x123;
						msg.obj = content;
						handler.sendMessage(msg);
					}
				}
				catch (IOException e)
				{
					e.printStackTrace();
				}
			}).start();
			// 为当前线程初始化Looper
			Looper.prepare();
			// 创建revHandler对象
			revHandler = new Handler()
			{
				@Override
				public void handleMessage(Message msg)
				{
					// 接收到UI线程中用户输入的数据
					if (msg.what == 0x345) {
						// 将用户在文本框内输入的内容写入网络
						try {
							os.write((msg.obj.toString() + "\r\n")
									.getBytes("utf-8"));
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			};
			// 启动Looper
			Looper.loop();
		}
		catch (SocketTimeoutException e1)
		{
			System.out.println("网络连接超时！！");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
}
```

```java
public class MainActivity extends Activity
{
	private TextView show;
	// 定义与服务器通信的子线程
	private ClientThread clientThread;
	static class MyHandler extends Handler // ②
	{
		private WeakReference<MainActivity> mainActivity;
		MyHandler(WeakReference<MainActivity> mainActivity)
		{
			this.mainActivity = mainActivity;
		}
		@Override
		public void handleMessage(Message msg)
		{
			// 如果消息来自子线程
			if (msg.what == 0x123)
			{
				// 将读取的内容追加显示在文本框中
				mainActivity.get().show.append("\n" + msg.obj.toString());
			}
		}
	}

	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 定义界面上的两个文本框
		EditText input = findViewById(R.id.input);
		show = findViewById(R.id.show);
		// 定义界面上的一个按钮
		Button send = findViewById(R.id.send);
		MyHandler handler = new MyHandler(new WeakReference<>(this));
		clientThread = new ClientThread(handler);
		// 客户端启动ClientThread线程创建网络连接，读取来自服务器的数据
		new Thread(clientThread).start();  // ①
		send.setOnClickListener(view -> {
			// 当用户单击“发送”按钮后，将用户输入的数据封装成Message
			// 然后发送给子线程的Handler
			Message msg = new Message();
			msg.what = 0x345;
			msg.obj = input.getText().toString();
			clientThread.revHandler.sendMessage(msg);
			// 清空input文本框
			input.setText("");
		});
	}
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221118205459399.png)

### 2. 基于URL

- String getFile():获取此URL的资源名。
- String getHost():获取此URL的主机名。
- String getPath():获取此URL的路径部分。
- int getPort():获取此URL的端口号。
- String getProtocol():获取此URL的协议名称。
- String getQuery):获取此 URL的查询字符串部分。
- URLConnection openConnection():返回一个URLConnection对象,它表示到URL所引用的远程对象的连接
- lnputStream openStream():打开与此URL的连接，并返回一个用于读取该URL资源的InputStream.

```java
public class MainActivity extends Activity
{
	private ImageView show;
	// 代表从网络下载得到的图片
	private Bitmap bitmap;
	static class MyHandler extends Handler
	{
		private WeakReference<MainActivity> mainActivity;
		MyHandler(WeakReference<MainActivity> mainActivity)
		{
			this.mainActivity = mainActivity;
		}
		@Override public void handleMessage(Message msg)
		{
			if (msg.what == 0x123)
			{
				// 使用ImageView显示该图片
				mainActivity.get().show.setImageBitmap(mainActivity.get().bitmap);
			}
		}
	}
	private MyHandler handler = new MyHandler(new WeakReference<>(this));
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		show = findViewById(R.id.show);
		new Thread()
		{
			@Override
			public void run()
			{
				try {
					// 定义一个URL对象
					URL url = new URL("http://img10.360buyimg.com/n0"
							+ "/jfs/t15760/240/1818365159/368378/350e622b/"
							+ "5a60cbaeN0ecb487a.jpg"
					);
					// 打开该URL对应的资源的输入流
					InputStream is = url.openStream();
					// 从InputStream中解析出图片
					bitmap = BitmapFactory.decodeStream(is);
					// 发送消息，通知UI组件显示该图片
					handler.sendEmptyMessage(0x123);
					is.close();
					// 再次打开URL对应的资源的输入流
					is = url.openStream();
					// 打开手机文件对应的输出流
					OutputStream os = openFileOutput("crazyit.png", Context.MODE_PRIVATE);
					byte[] buff = new byte[1024];
					int hasRead = -1;
					// 将URL对应的资源下载到本地
					while ((hasRead = is.read(buff)) > 0) {
						os.write(buff, 0, hasRead);
					}
					is.close();
					os.close();
				}
				catch(Exception e)
				{
					e.printStackTrace();
				}
			}
		}.start();
	}
}
```

#### .2. get&post 请求

```java
public class GetPostUtil
{
	/**
	 * 向指定URL发送GET方法的请求
	 * @param url 发送请求的URL
	 * @param params 请求参数，请求参数应该是name1=value1&name2=value2的形式。
	 * @return URL所代表远程资源的响应
	 */
	public static String sendGet(String url, String params)
	{
		StringBuilder result = new StringBuilder();
		BufferedReader in = null;
		try
		{
			String urlName = url + "?" + params;
			URL realUrl = new URL(urlName);
			// 打开和URL之间的连接
			URLConnection conn = realUrl.openConnection();
			// 设置通用的请求属性
			conn.setRequestProperty("accept", "*/*");
			conn.setRequestProperty("connection", "Keep-Alive");
			conn.setRequestProperty("user-agent",
					"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)");
			// 建立实际的连接
			conn.connect();  // ①
			// 获取所有响应头字段
			Map<String, List<String>> map = conn.getHeaderFields();
			// 遍历所有的响应头字段
			for (String key : map.keySet())
			{
				System.out.println(key + "--->" + map.get(key));
			}
			// 定义BufferedReader输入流来读取URL的响应
			in = new BufferedReader(
					new InputStreamReader(conn.getInputStream()));
			String line;
			while ((line = in.readLine()) != null)
			{
				result.append(line).append("\n");
			}
		}
		catch (Exception e)
		{
			System.out.println("发送GET请求出现异常！" + e);
			e.printStackTrace();
		}
		// 使用finally块来关闭输入流
		finally
		{
			try
			{
				if (in != null)
				{
					in.close();
				}
			}
			catch (IOException ex)
			{
				ex.printStackTrace();
			}
		}
		return result.toString();
	}
	/**
	 * 向指定URL发送POST方法的请求
	 * @param url 发送请求的URL
	 * @param params 请求参数，请求参数应该是name1=value1&name2=value2的形式。
	 * @return URL所代表远程资源的响应
	 */
	public static String sendPost(String url, String params)
	{
		PrintWriter out = null;
		BufferedReader in = null;
		StringBuilder result = new StringBuilder();
		try
		{
			URL realUrl = new URL(url);
			// 打开和URL之间的连接
			URLConnection conn = realUrl.openConnection();
			// 设置通用的请求属性
			conn.setRequestProperty("accept", "*/*");
			conn.setRequestProperty("connection", "Keep-Alive");
			conn.setRequestProperty("user-agent",
					"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)");
			// 发送POST请求必须设置如下两行
			conn.setDoOutput(true);
			conn.setDoInput(true);
			// 获取URLConnection对象对应的输出流
			out = new PrintWriter(conn.getOutputStream());
			// 发送请求参数
			out.print(params);  // ②
			// flush输出流的缓冲
			out.flush();
			// 定义BufferedReader输入流来读取URL的响应
			in = new BufferedReader(
					new InputStreamReader(conn.getInputStream()));
			String line;
			while ((line = in.readLine()) != null)
			{
				result.append(line).append("\n");
			}
		}
		catch (Exception e)
		{
			System.out.println("发送POST请求出现异常！" + e);
			e.printStackTrace();
		}
		// 使用finally块来关闭输出流、输入流
		finally
		{
			try
			{
				if (out != null)
				{
					out.close();
				}
				if (in != null)
				{
					in.close();
				}
			}
			catch (IOException ex)
			{
				ex.printStackTrace();
			}
		}
		return result.toString();
	}
}
```

#### .3. 多线程下载

- 使用Handler进行UI信息和后台线程之间通信
- 使用timer.scheduler() 进行任务的调度
- 使用随机数划分数据内容，并使用多个线程同时进行下载

```java
public class MainActivity extends Activity
{
    private EditText url;
    private EditText target;
    private Button downBn;
    private ProgressBar bar;
    private DownUtil downUtil;

    static class MyHandler extends Handler
    {
        private WeakReference<MainActivity> mainActivity;

        MyHandler(WeakReference<MainActivity> mainActivity)
        {
            this.mainActivity = mainActivity;
        }

        @Override
        public void handleMessage(Message msg)
        {
            if (msg.what == 0x123)
            {
                mainActivity.get().bar.setProgress(msg.arg1);
            }
        }
    }

    // 创建一个Handler对象
    private MyHandler handler = new MyHandler(new WeakReference<>(this));

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 获取程序界面中的三个界面控件
        url = findViewById(R.id.url);
        target = findViewById(R.id.target);
        downBn = findViewById(R.id.down);
        bar = findViewById(R.id.bar);
        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0x456);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        if (requestCode == 0x456 && grantResults.length == 1
            && grantResults[0] == PackageManager.PERMISSION_GRANTED)
        {
            downBn.setOnClickListener(view ->
                                      {
                                          // 初始化DownUtil对象（最后一个参数指定线程数）
                                          downUtil = new DownUtil(url.getText().toString(), target.getText().toString(), 6);
                                          new Thread(() ->
                                                     {
                                                         try
                                                         {
                                                             // 开始下载
                                                             downUtil.download();
                                                         }
                                                         catch (Exception e)
                                                         {
                                                             e.printStackTrace();
                                                         }
                                                         // 定义每秒调度获取一次系统的完成进度
                                                         final Timer timer = new Timer();
                                                         timer.schedule(new TimerTask()
                                                                        {
                                                                            @Override
                                                                            public void run()
                                                                            {
                                                                                // 获取下载任务的完成比例
                                                                                double completeRate = downUtil.getCompleteRate();
                                                                                System.out.println(completeRate);
                                                                                Message msg = new Message();
                                                                                msg.what = 0x123;
                                                                                msg.arg1 = (int) (completeRate * 100);
                                                                                // 发送消息通知界面更新进度条
                                                                                handler.sendMessage(msg);
                                                                                // 下载完全后取消任务调度
                                                                                if (completeRate >= 1)
                                                                                {
                                                                                    timer.cancel();
                                                                                }
                                                                            }
                                                                        }, 0, 100);
                                                     }).start();
                                      });
        }
    }
}
```

- DownloadUtil.java

```java
public class DownUtil
{
	// 定义下载资源的路径
	private String path;
	// 指定所下载的文件的保存位置
	private String targetFile;
	// 定义需要使用多少线程下载资源
	private int threadNum;
	// 定义下载的线程对象
	private DownThread[] threads;
	// 定义下载的文件的总大小
	private int fileSize;
	public DownUtil(String path, String targetFile, int threadNum)
	{
		this.path = path;
		this.threadNum = threadNum;
		// 初始化threads数组
		threads = new DownThread[threadNum];
		this.targetFile = targetFile;
	}
	public void download() throws Exception
	{
		URL url = new URL(path);
		HttpURLConnection conn = (HttpURLConnection) url.openConnection();
		conn.setConnectTimeout(5 * 1000);
		conn.setRequestMethod("GET");
		conn.setRequestProperty(
				"Accept",
				"image/gif, image/jpeg, image/pjpeg, image/pjpeg, "
						+ "application/x-shockwave-flash, application/xaml+xml, "
						+ "application/vnd.ms-xpsdocument, application/x-ms-xbap, "
						+ "application/x-ms-application, application/vnd.ms-excel, "
						+ "application/vnd.ms-powerpoint, application/msword, */*");
		conn.setRequestProperty("Accept-Language", "zh-CN");
		conn.setRequestProperty("Charset", "UTF-8");
		conn.setRequestProperty("Connection", "Keep-Alive");
		// 得到文件大小
		fileSize = conn.getContentLength();
		conn.disconnect();
		int currentPartSize = fileSize / threadNum + 1;
		RandomAccessFile file = new RandomAccessFile(targetFile, "rw");
		// 设置本地文件的大小
		file.setLength(fileSize);
		file.close();
		for (int i = 0; i < threadNum; i++)
		{
			// 计算每条线程的下载的开始位置
			int startPos = i * currentPartSize;
			// 每个线程使用一个RandomAccessFile进行下载
			RandomAccessFile currentPart = new RandomAccessFile(targetFile,
					"rw");
			// 定位该线程的下载位置
			currentPart.seek(startPos);
			// 创建下载线程
			threads[i] = new DownThread(startPos, currentPartSize,
					currentPart);
			// 启动下载线程
			threads[i].start();
		}
	}
	// 获取下载的完成百分比
	public double getCompleteRate()
	{
		// 统计多条线程已经下载的总大小
		int sumSize = 0;
		for (int i = 0; i < threadNum; i++)
		{
			sumSize += threads[i].length;
		}
		// 返回已经完成的百分比
		return sumSize * 1.0 / fileSize;
	}
	private class DownThread extends Thread
	{
		// 当前线程的下载位置
		private int startPos;
		// 定义当前线程负责下载的文件大小
		private int currentPartSize;
		// 当前线程需要下载的文件块
		private RandomAccessFile currentPart;
		// 定义已经该线程已下载的字节数
		int length;
		DownThread(int startPos, int currentPartSize,
				   RandomAccessFile currentPart)
		{
			this.startPos = startPos;
			this.currentPartSize = currentPartSize;
			this.currentPart = currentPart;
		}
		@Override
		public void run()
		{
			try
			{
				URL url = new URL(path);
				HttpURLConnection conn = (HttpURLConnection)url
						.openConnection();
				conn.setConnectTimeout(5 * 1000);
				conn.setRequestMethod("GET");
				conn.setRequestProperty(
						"Accept",
						"image/gif, image/jpeg, image/pjpeg, image/pjpeg, "
								+ "application/x-shockwave-flash, application/xaml+xml, "
								+ "application/vnd.ms-xpsdocument, application/x-ms-xbap, "
								+ "application/x-ms-application, application/vnd.ms-excel, "
								+ "application/vnd.ms-powerpoint, application/msword, */*");
				conn.setRequestProperty("Accept-Language", "zh-CN");
				conn.setRequestProperty("Charset", "UTF-8");
				InputStream inStream = conn.getInputStream();
				// 跳过startPos个字节，表明该线程只下载自己负责的那部分文件
				inStream.skip(this.startPos);
				byte[] buffer = new byte[1024];
				int hasRead = 0;
				// 读取网络数据，并写入本地文件
				while (length < currentPartSize
						&& (hasRead = inStream.read(buffer)) > 0)
				{
					currentPart.write(buffer, 0, hasRead);
					// 累计该线程下载的总大小
					length += hasRead;
				}
				currentPart.close();
				inStream.close();
			}
			catch (Exception e)
			{
				e.printStackTrace();
			}
		}
	}
	// 定义一个为InputStream跳过bytes字节的方法
	private static void skipFully(InputStream in, long bytes) throws IOException
	{
		long remainning = bytes;
		long len = 0;
		while (remainning > 0)
		{
			len = in.skip(remainning);
			remainning -= len;
		}
	}
}
```

### 3. HttpClient 登录

```java
public class MainActivity extends Activity
{
    private TextView response;
    private OkHttpClient okHttpClient;

    static class MyHandler extends Handler
    {
        private WeakReference<MainActivity> mainActivity;

        MyHandler(WeakReference<MainActivity> mainActivity)
        {
            this.mainActivity = mainActivity;
        }

        @Override
        public void handleMessage(Message msg)
        {
            if (msg.what == 0x123)
            {
                // 使用response文本框显示服务器响应信息
                mainActivity.get().response.setText(msg.obj.toString());
            }
        }
    }

    private Handler handler = new MyHandler(new WeakReference<>(this));

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        response = findViewById(R.id.response);
        // 创建默认的OkHttpClient对象
        //        okHttpClient = OkHttpClient()
        final Map<String, List<Cookie>> cookieStore = new HashMap<>();
        okHttpClient = new OkHttpClient.Builder()
            .cookieJar(new CookieJar()
                       {
                           @Override
                           public void saveFromResponse(@NonNull HttpUrl httpUrl, @NonNull List<Cookie> list)
                           {
                               cookieStore.put(httpUrl.host(), list);
                           }

                           @Override
                           public List<Cookie> loadForRequest(@NonNull HttpUrl httpUrl)
                           {
                               List<Cookie> cookies = cookieStore.get(httpUrl.host());
                               return cookies == null ? new ArrayList<>() : cookies;
                           }
                       }).build();
    }

    public void accessSecret(View source)
    {
        new Thread(() ->
                   {
                       String url = "http://192.168.1.88:8888/foo/secret.jsp";
                       // 创建请求
                       Request request = new Request.Builder().url(url).build();  // ①
                       Call call = okHttpClient.newCall(request);
                       try
                       {
                           Response response = call.execute();  // ②
                           Message msg = new Message();
                           msg.what = 0x123;
                           msg.obj = response.body().string().trim();
                           handler.sendMessage(msg);
                       }
                       catch (IOException e)
                       {
                           e.printStackTrace();
                       }
                   }).start();
    }

    public void showLogin(View source)
    {
        // 加载登录界面
        View loginDialog = getLayoutInflater().inflate(R.layout.login, null);
        // 使用对话框供用户登录系统
        new AlertDialog.Builder(MainActivity.this)
            .setTitle("登录系统").setView(loginDialog)
            .setPositiveButton("登录", (dialog, which) ->
                               {
                                   // 获取用户输入的用户名、密码
                                   String name = ((EditText) loginDialog.findViewById(R.id.name))
                                       .getText().toString();
                                   String pass = ((EditText) loginDialog.findViewById(R.id.pass))
                                       .getText().toString();
                                   String url = "http://192.168.1.88:8888/foo/login.jsp";
                                   FormBody body = new FormBody.Builder().add("name", name)
                                       .add("pass", pass).build();  //③
                                   Request request = new Request.Builder().url(url)
                                       .post(body).build();  //④
                                   Call call = okHttpClient.newCall(request);
                                   call.enqueue(new Callback()  // ⑤
                                                {
                                                    @Override
                                                    public void onFailure(@NonNull Call call, @NonNull IOException e)
                                                    {
                                                        e.printStackTrace();
                                                    }

                                                    @Override
                                                    public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException
                                                    {
                                                        Looper.prepare();
                                                        Toast.makeText(MainActivity.this,
                                                                       response.body().string().trim(), Toast.LENGTH_SHORT).show();
                                                        Looper.loop();
                                                    }
                                                });
                               }).setNegativeButton("取消", null).show();
    }
}
```

### 4. Webview控件

#### .1. 渲染html

```java
public class MainActivity extends Activity
{
	private WebView showWv;
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取程序中的WebView组件
		showWv = this.findViewById(R.id.show);
		StringBuilder sb = new StringBuilder();
		// 拼接一段HTML代码
		sb.append("<html>");
		sb.append("<head>");
		sb.append("<title> 欢迎您 </title>");
		sb.append("</head>");
		sb.append("<body>");
		sb.append("<h2> 欢迎您访问<a href=\"http://www.crazyit.org\">" + "疯狂Java联盟</a></h2>");
		sb.append("</body>");
		sb.append("</html>");
		// 下面两个方法都可正常加载、显示HTML代码
		showWv.loadData(sb.toString() , "text/html" , "utf-8");
		// 加载并显示HTML代码
//		showWv.loadDataWithBaseURL(null, sb.toString(), "text/html", "utf-8", null);
	}
}
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
	android:layout_width="match_parent"
	android:layout_height="match_parent"
	android:orientation="vertical">

	<WebView
		android:id="@+id/show"
		android:layout_width="match_parent"
		android:layout_height="match_parent" />
</LinearLayout>
```

#### .2. 显示网页

```java
public class MainActivity extends Activity
{
	private EditText urlEt;
	private WebView showWv;
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取页面中文本框、WebView组件
		urlEt = this.findViewById(R.id.url);
		showWv = findViewById(R.id.show);
		// 为键盘的软键盘绑定事件监听器
		urlEt.setOnEditorActionListener((view, actionId, event) -> {
			if (actionId == EditorInfo.IME_ACTION_GO)
			{
				String urlStr = urlEt.getText().toString();
				// 加载并显示urlStr对应的网页
				showWv.loadUrl(urlStr);
			}
			return true;
		});
	}
}
```

### 5. Web service 编程

- 类似于一些给我们提供 原始数据API服务的数据平台，比如聚合数据！而WebService则用到了XML和SOAP，通过HTTP协议 即可完成与远程机器的交互
- 如何去获取WebService提供的服务， 然后解析返回的XML数据，然后把相关数据显示到我们的Android设备上
- WebService服务站点
  - http://www.36wu.com/Service
  - http://www.webxml.com.cn/zh_cn/index.aspx

- 获取手机号码归属地案例：https://www.runoob.com/w3cnote/android-tutorial-webservice.html

```java
//定义一个获取某城市天气信息的方法：
public void getWether() {
    result = "";
    SoapObject soapObject = new SoapObject(AddressnameSpace, Weathermethod);
    soapObject.addProperty("theCityCode:", edit_param.getText().toString());
    soapObject.addProperty("theUserID", "dbdf1580476240458784992289892b87");
    SoapSerializationEnvelope envelope = new SoapSerializationEnvelope(SoapEnvelope.VER11);
    envelope.bodyOut = soapObject;
    envelope.dotNet = true;
    envelope.setOutputSoapObject(soapObject);
    HttpTransportSE httpTransportSE = new HttpTransportSE(Weatherurl);
    System.out.println("天气服务设置完毕,准备开启服务");
    try {
        httpTransportSE.call(WeathersoapAction, envelope);
        //            System.out.println("调用WebService服务成功");
    } catch (Exception e) {
        e.printStackTrace();
        //            System.out.println("调用WebService服务失败");
    }

    //获得服务返回的数据,并且开始解析
    SoapObject object = (SoapObject) envelope.bodyIn;
    System.out.println("获得服务数据");
    result = object.getProperty(1).toString();
    handler.sendEmptyMessage(0x001);
    System.out.println("发送完毕,textview显示天气信息");
}


//定义一个获取号码归属地的方法：
public void getland() {
    result = "";
    // web service的命名空间，web service 命名方法
    SoapObject soapObject = new SoapObject(AddressnameSpace, Addressmethod);
    // 传递给webservice 的方法参数
    soapObject.addProperty("mobileCode", edit_param.getText().toString());
    soapObject.addProperty("userid", "dbdf1580476240458784992289892b87");
    
    SoapSerializationEnvelope envelope = new SoapSerializationEnvelope(SoapEnvelope.VER11);
    envelope.bodyOut = soapObject;
    envelope.dotNet = true;
    envelope.setOutputSoapObject(soapObject);
    
    //调用webservice 服务操作
    HttpTransportSE httpTransportSE = new HttpTransportSE(Addressurl);
    //    System.out.println("号码信息设置完毕,准备开启服务");
    try {
        //String AddresssoapAction = "http://WebXml.com.cn/getMobileCodeInfo";
        httpTransportSE.call(AddresssoapAction, envelope);
        //System.out.println("调用WebService服务成功");
    } catch (Exception e) {
        e.printStackTrace();
        //System.out.println("调用WebService服务失败");
    }

    //获得服务返回的数据,并且开始解析
    SoapObject object = (SoapObject) envelope.bodyIn;//System.out.println("获得服务数据");
    result = object.getProperty(0).toString();//System.out.println("获取信息完毕,向主线程发信息");
    handler.sendEmptyMessage(0x001);
    //System.out.println("发送完毕,textview显示天气信息");
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_%E7%BD%91%E7%BB%9C/  

