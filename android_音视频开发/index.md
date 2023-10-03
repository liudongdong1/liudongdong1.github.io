# Android_音视频应用开发


### 1. 音视频播放

#### .1. [MediaPayler 播放音频](https://github1s.com/liudongdong1/crazy-android/blob/HEAD/11/11.1/MediaPlay/mediaplayertest/src/main/java/org/crazyit/media/MainActivity.java)

- setOnCompletionListener(MediaPlayer.OnCompletionListener listener):为 MediaPlayer的播放完成事件绑定事件监听器。
- setOnErrorListener(MediaPlayer.OnErrorListener listener):为 MediaPlayer 的播放错误事件绑定事件监听器。
- setOnPreparedListener(MediaPlayer.OnPreparedListener listener):当MediaPlayer 调用prepare()方法时触发该监听器。
- setOnSeekCompleteListener(MediaPlayer.OnSeekCompleteListener listener):当MediaPlayer调用seek()方法时触发该监听器。

```java
// 播放应用文件
MediaPlayer mPalyer=MediaPayer.create(this,R.raw.song);
mPlayer.start();

// 播放应用原始资源文件
AssetManager am=getAssets();
AssetFileDescriptor afg=am.openFd(music);
MediaPayer mPlayer=new MediaPayer();
mPlayer.setDataSource(afd.getFileDescriptor(),afd.getStartOffset(),afd.getLength());
mPlayer.prepare();
mPayer.start();

// 播放外部存储器上音频文件
MediaPayer mPlayer=new MediaPayer();
mPlayer.setDataSource("/mnt/sdcard/mysong.mp3");
mPlayer.prepare();
mPlayer.start();

// 播放来自网络文件
Uri uir=Uri.parse("http://www.crazyit.org/abc.mp3");
MediaPlayer mPlayer=new MediaPayer();
mPlayer.setDataSource(this,url);
mPlayer.prepare();
mPlayer.start();
```

- 案例代码

```java
public void onRequestPermissionsResult(int requestCode,
                                       @NonNull String[] permissions, @NonNull int[] grantResults)
{
    if(requestCode == 0x123	&& grantResults[0] == PackageManager.PERMISSION_GRANTED)
    {
        // 创建MediaPlayer对象
        mPlayer = MediaPlayer.create(this, R.raw.beautiful);
        // 初始化示波器
        setupVisualizer();
        // 初始化均衡控制器
        setupEqualizer();
        // 初始化重低音控制器
        setupBassBoost();
        // 初始化预设音场控制器
        setupPresetReverb();
        // 开始播放音乐
        mPlayer.start();
    }
}
```

#### .2. SoundPool 播放音效

- 播放密集，短促的音效

- SoundPool(int maxStreams, int streamType, int srcQuality):第一个参数指定支持多少个声音:第二个参数指定声音类型;第三个参数指定声音品质。
- 一旦得到了SoundPool对象之后，接下来就可调用SoundPool 的多个重载的 load方法来加载声音了，SoundPool提供了如下4个load方法。
- int load(Context context, int resld, int priority):从resld所对应的资源加载声音。int load(FileDescriptor fd, long offset, long length, int priority):加载f似所对应的文件的offset开始、长度为length的声音。
- int load(AssetFileDescriptor afd, int priority):从afd所对应的文件中加载声音。int load(String path, int priority):从path对应的文件去加载声音。

```java
AudioAttributes attr = new AudioAttributes.Builder().setUsage(
    AudioAttributes.USAGE_GAME) // 设置音效使用场景
    // 设置音效的类型
    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build();
soundPool = new SoundPool.Builder().setAudioAttributes(attr) // 设置音效池的属性
    .setMaxStreams(10) // 设置最多可容纳10个音频流
    .build();  // ①
// 使用load方法加载指定的音频文件，并返回所加载的音频ID
// 此处使用HashMap来管理这些音频流
soundMap.put(1, soundPool.load(this, R.raw.bomb, 1));  // ②
soundMap.put(2, soundPool.load(this, R.raw.shot, 1));
soundMap.put(3, soundPool.load(this, R.raw.arrow, 1));
// 定义一个按钮的单击监听器
View.OnClickListener listener = source -> {
    // 判断哪个按钮被单击
    switch (source.getId())
    {
        case R.id.bomb:
            soundPool.play(soundMap.get(1), 1f, 1f, 0, 0, 1f);  // ③
            break;
        case R.id.shot:
            soundPool.play(soundMap.get(2), 1f, 1f, 0, 0, 1f);
            break;
        case R.id.arrow:
            soundPool.play(soundMap.get(3), 1f, 1f, 0, 0, 1f);
            break;
    }
};
```

#### .3. 使用VideoView播放视频

```java
public class MainActivity extends Activity
{
	private VideoView videoView;
	private MediaController mController;
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取界面上的VideoView组件
		videoView = findViewById(R.id.video);
		// 创建MediaController对象
		mController = new MediaController(this);
		requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 0x123);
	}

	@Override public void onRequestPermissionsResult(int requestCode,
		@NonNull String[] permissions, @NonNull int[] grantResults)
	{
		if (requestCode == 0x123
				&& grantResults[0] == PackageManager.PERMISSION_GRANTED) {
			// 设为横屏
			setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
			File video = new File("/mnt/sdcard/movie.mp4");
			if (video.exists()) {
				videoView.setVideoPath(video.getAbsolutePath()); // ①
				// 设置videoView与mController建立关联
				videoView.setMediaController(mController);  // ②
				// 设置mController与videoView建立关联
				mController.setMediaPlayer(videoView);  // ③
				// 让VideoView获取焦点
				videoView.requestFocus();
				videoView.start(); // 开始播放
			}
		}
	}
}
```

#### .4. [使用MediaPlayer&SurfaceView 播放音频](https://github1s.com/liudongdong1/crazy-android/blob/HEAD/11/11.1/MediaPlay/surfaceview/src/main/java/org/crazyit/media/MainActivity.java#L33-L171)

- 创建MediaPlayer对象，并让它加载指定的视频文件。
- 在界面布局文件中定义SurfaceView组件，或在程序中创建SurfaceView组件。并为Surfaceview的SurfaceHolder添加Callback监听器。
- 调用MediaPlayer对象的setDisplay(SurfaceHolder sh)将所播放的视频图像输出到指定的SurfaceView组件。
- 调用MediaPlayer 对象的start(、stopO和pauseO方法控制视频的播放。

```java
public class MainActivity extends Activity
{
	private SurfaceView surfaceView;
	private MediaPlayer mPlayer;
	private ImageButton playBn, pauseBn, stopBn;
	// 记录当前视频的播放位置
	int position = 0;

	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 创建MediaPlayer
		mPlayer = new MediaPlayer();
		surfaceView = this.findViewById(R.id.surfaceView);
		// 设置播放时打开屏幕
		surfaceView.getHolder().setKeepScreenOn(true);
		surfaceView.getHolder().addCallback(new SurfaceListener());
		// 获取界面上的三个按钮
		playBn = findViewById(R.id.play);
		pauseBn = findViewById(R.id.pause);
		stopBn = findViewById(R.id.stop);
		requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 0x123);
	}

	@Override
	public void onRequestPermissionsResult(int requestCode,
										   @NonNull String[] permissions, @NonNull int[] grantResults)
	{
		if (requestCode == 0x123 && grantResults[0] ==
				PackageManager.PERMISSION_GRANTED) {
			setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
			View.OnClickListener listener = source -> {
				switch (source.getId())
				{
					// “播放”按钮被单击
					case R.id.play:
						play();
						break;
					// “暂停”按钮被单击
					case R.id.pause:
						if (mPlayer.isPlaying()) {
							mPlayer.pause();
						} else {
							mPlayer.start();
						}
						break;
					// “停止”按钮被单击
					case R.id.stop:
						if (mPlayer.isPlaying())
							mPlayer.stop();
				}
			};
			// 为三个按钮的单击事件绑定事件监听器
			playBn.setOnClickListener(listener);
			pauseBn.setOnClickListener(listener);
			stopBn.setOnClickListener(listener);
		}
	}

	private void play()
	{
		mPlayer.reset();
		AudioAttributes audioAttributes = new AudioAttributes.Builder()
				.setUsage(AudioAttributes.USAGE_MEDIA)
				.setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
				.build();
		mPlayer.setAudioAttributes(audioAttributes);
		try {
			// 设置需要播放的视频
			mPlayer.setDataSource(Environment.getExternalStorageDirectory().toString() + "/movie.3gp");
			// 把视频画面输出到SurfaceView
			mPlayer.setDisplay(surfaceView.getHolder());  // ①
			mPlayer.prepare();
		}
		catch(IOException e)
		{ e.printStackTrace(); }
		// 获取窗口管理器
		WindowManager wManager = getWindowManager();
		DisplayMetrics metrics = new DisplayMetrics();
		// 获取屏幕大小
		wManager.getDefaultDisplay().getMetrics(metrics);
		// 设置视频保持纵横比缩放到占满整个屏幕
		surfaceView.setLayoutParams(new RelativeLayout.LayoutParams(metrics.widthPixels,
				mPlayer.getVideoHeight() * metrics.widthPixels / mPlayer.getVideoWidth()));
		mPlayer.start();
	}

	private class SurfaceListener implements SurfaceHolder.Callback
	{
		@Override
		public void surfaceCreated(SurfaceHolder holder)
		{
			if (position > 0) {
				// 开始播放
				play();
				// 并直接从指定位置开始播放
				mPlayer.seekTo(position);
				position = 0;
			}
		}

		@Override
		public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
		{

		}

		@Override
		public void surfaceDestroyed(SurfaceHolder holder)
		{

		}
	}

	// 当其他Activity被打开时，暂停播放
	@Override
	public void onPause()
	{
		super.onPause();
		if (mPlayer.isPlaying()) {
			// 保存当前的播放位置
			position = mPlayer.getCurrentPosition();
			mPlayer.stop();
		}
	}

	@Override
	public void onDestroy()
	{
		super.onDestroy();
		// 停止播放
		if (mPlayer.isPlaying()) mPlayer.stop();
		// 释放资源
		mPlayer.release();
	}
}
```

### 2. 视频录制MediaRecorder

```java
public class MainActivity extends Activity
{
	// 定义界面上的两个按钮
	private ImageButton recordBn;
	private ImageButton stopBn;
	// 系统的音频文件
	private File soundFile;
	private MediaRecorder mRecorder;

	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取程序界面中的两个按钮
		recordBn = findViewById(R.id.record);
		stopBn = findViewById(R.id.stop);
		stopBn.setEnabled(false);
		requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO,
				Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0x123);
	}

	@Override
	public void onRequestPermissionsResult(int requestCode,
		@NonNull String[] permissions, @NonNull int[] grantResults)
	{
		if (requestCode == 0x123 && grantResults.length == 2
				&& grantResults[0] == PackageManager.PERMISSION_GRANTED
				&& grantResults[1] == PackageManager.PERMISSION_GRANTED) {
			View.OnClickListener listener = source ->
			{
				switch (source.getId()) {
					// 单击录音按钮
					case R.id.record:
						if (!Environment.getExternalStorageState()
								.equals(Environment.MEDIA_MOUNTED)) {
							Toast.makeText(MainActivity.this,
									"SD卡不存在，请插入SD卡！",
									Toast.LENGTH_SHORT).show();
							return;
						}
						// 创建保存录音的音频文件
						soundFile = new File(Environment.getExternalStorageDirectory()
								.toString() + "/sound.amr");
						mRecorder = new MediaRecorder();
						// 设置录音的声音来源
						mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
						// 设置录制的声音的输出格式（必须在设置声音编码格式之前设置）
						mRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
						// 设置声音编码格式
						mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
						mRecorder.setOutputFile(soundFile.getAbsolutePath());
						try {
							mRecorder.prepare();
						} catch (IOException e) {
							e.printStackTrace();
						}
						// 开始录音
						mRecorder.start();  // ①
						recordBn.setEnabled(false);
						stopBn.setEnabled(true);
						break;
					// 单击停止录制按钮
					case R.id.stop:
						if (soundFile != null && soundFile.exists()) {


							// 停止录音
							mRecorder.stop();  // ②
							// 释放资源
							mRecorder.release();  // ③
							mRecorder = null;
							recordBn.setEnabled(true);
							stopBn.setEnabled(false);
						}
						break;
				}
			};
			// 为两个按钮的单击事件绑定监听器
			recordBn.setOnClickListener(listener);
			stopBn.setOnClickListener(listener);
		}
	}

	@Override
	public void onDestroy()
	{
		if (soundFile != null && soundFile.exists()) {
			// 停止录音
			mRecorder.stop();
			// 释放资源
			mRecorder.release();
			mRecorder = null;
		}
		super.onDestroy();
	}
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_%E9%9F%B3%E8%A7%86%E9%A2%91%E5%BC%80%E5%8F%91/  

