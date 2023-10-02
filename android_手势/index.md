# Android_手势


> Android关于手势的操作提供两种形式：一种是针对用户手指在屏幕上划出的动作而进行移动的检测，这些手势的检测通过android提供的监听器来实现；另一种是用户手指在屏幕上滑动而形成一定的不规则的几何图形(即为多个持续触摸事件在屏幕形成特定的形状)；
>
> - frameworks/base/core/java/android/gesture   todo? 需要理解这个具体库

### 1. 手势图片缩放

```java
protected void onCreate(Bundle savedInstanceState)
{
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    // 创建手势检测器
    detector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {
        @Override
        public boolean onFling(MotionEvent event1, MotionEvent event2,
                               float velocityX, float velocityY)  // ②
        {
            float vx = velocityX > 4000 ? 4000f : velocityX;
            vx = velocityX < -4000 ? -4000f : velocityX;
            // 根据手势的速度来计算缩放比，如果vx>0，则放大图片；否则缩小图片
            currentScale += currentScale * vx / 4000.0f;
            // 保证currentScale不会等于0
            currentScale = currentScale > 0.01 ? currentScale : 0.01f;
            // 重置Matrix
            matrix.reset();
            // 缩放Matrix
            matrix.setScale(currentScale, currentScale, 160f, 200f);
            BitmapDrawable tmp = (BitmapDrawable) imageView.getDrawable();
            // 如果图片还未回收，先强制回收该图片
            if (!tmp.getBitmap().isRecycled()) // ①
            {
                tmp.getBitmap().recycle();
            }
            // 根据原始位图和Matrix创建新图片
            Bitmap bitmap2 = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
            // 显示新的位图
            imageView.setImageBitmap(bitmap2);
            return true;
        }
    });
    imageView = findViewById(R.id.show);
    matrix = new Matrix();
    // 获取被缩放的源图片
    bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.flower);
    // 获得位图宽
    width = bitmap.getWidth();
    // 获得位图高
    height = bitmap.getHeight();
    // 设置ImageView初始化时显示的图片
    imageView.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.flower));
}
```

### 2. 图片滑动

```java
public class MainActivity extends Activity
{
	//  ViewFlipper实例
	private ViewFlipper flipper;
	// 定义手势检测器变量
	private GestureDetector detector;
	// 定义一个动画数组，用于为ViewFlipper指定切换动画效果
	private Animation[] animations = new Animation[4];
	// 定义手势动作两点之间的最小距离
	private float flipDistance = 0f;

	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		flipDistance = getResources().getDimension(R.dimen.flip_distance);
		// 创建手势检测器
		detector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener()
		{
			@Override public boolean onFling(MotionEvent event1, MotionEvent event2,
				float velocityX, float velocityY)
			{
				// 如果第一个触点事件的X坐标大于第二个触点事件的X坐标超过flipDistance
				// 也就是手势从右向左滑
				if (event1.getX() - event2.getX() > flipDistance)
				{
					// 为flipper设置切换的动画效果
					flipper.setInAnimation(animations[0]);
					flipper.setOutAnimation(animations[1]);
					flipper.showPrevious();
					return true;
				}
				// 如果第二个触点事件的X坐标大于第一个触点事件的X坐标超过flipDistance
				// 也就是手势从左向右滑
				else if (event2.getX() - event1.getX() > flipDistance)
				{
					// 为flipper设置切换的动画效果
					flipper.setInAnimation(animations[2]);
					flipper.setOutAnimation(animations[3]);
					flipper.showNext();
					return true;
				}
				return false;
			}
		});
		// 获得ViewFlipper实例
		flipper = this.findViewById(R.id.flipper);
		// 为ViewFlipper添加6个ImageView组件
		flipper.addView(addImageView(R.drawable.java));
		flipper.addView(addImageView(R.drawable.javaee));
		flipper.addView(addImageView(R.drawable.ajax));
		flipper.addView(addImageView(R.drawable.android));
		flipper.addView(addImageView(R.drawable.html));
		flipper.addView(addImageView(R.drawable.swift));
		// 初始化Animation数组
		animations[0] = AnimationUtils.loadAnimation(this, R.anim.left_in);
		animations[1] = AnimationUtils.loadAnimation(this, R.anim.left_out);
		animations[2] = AnimationUtils.loadAnimation(this, R.anim.right_in);
		animations[3] = AnimationUtils.loadAnimation(this, R.anim.right_out);
	}
	// 定义添加ImageView的工具方法
	private View addImageView(int resId)
	{
		ImageView imageView = new ImageView(this);
		imageView.setImageResource(resId);
		imageView.setScaleType(ImageView.ScaleType.CENTER);
		return imageView;
	}
	@Override public boolean onTouchEvent(MotionEvent event)
	{
		// 将该Activity上的触碰事件交给GestureDetector处理
		return detector.onTouchEvent(event);
	}
}
```

### 3. 添加手势

```java
public class MainActivity extends Activity
{
	private GestureOverlayView gestureView;
	private Gesture gesture;
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 获取手势编辑视图
		gestureView = findViewById(R.id.gesture);
		// 设置手势的绘制颜色
		gestureView.setGestureColor(Color.RED);
		// 设置手势的绘制宽度
		gestureView.setGestureStrokeWidth(4f);
		// 为gesture的手势完成事件绑定事件监听器
		gestureView.addOnGesturePerformedListener((source, gesture) -> {
			this.gesture = gesture;
			// 请求访问写入SD卡的权限
			requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0x123);
		});
	}
	@Override
	public void onRequestPermissionsResult(int requestCode,
		String[] permissions, int[] grantResults)
	{
		// 如果确认允许访问
		if(requestCode == 0x123 && grantResults != null && grantResults[0] == 0)
		{
			// 加载dialog_save.xml界面布局代表的视图
			LinearLayout saveDialog = (LinearLayout) getLayoutInflater().inflate(R.layout.dialog_save, null);
			// 获取saveDialog里的show组件
			ImageView imageView = saveDialog.findViewById(R.id.show);
			// 获取saveDialog里的gesture_name组件
			EditText gestureName = saveDialog.findViewById(R.id.gesture_name);
			// 根据Gesture包含的手势创建一个位图
			Bitmap bitmap = gesture.toBitmap(128, 128, 10, -0x10000);
			imageView.setImageBitmap(bitmap);
			// 使用对话框显示saveDialog组件
			new AlertDialog.Builder(MainActivity.this).setView(saveDialog)
				.setPositiveButton(R.string.bn_save, (dialog, which) -> {
						// 获取指定文件对应的手势库
						GestureLibrary gestureLib = GestureLibraries.fromFile(
						Environment.getExternalStorageDirectory().getPath() + "/mygestures");
						// 添加手势
						gestureLib.addGesture(gestureName.getText().toString(), gesture);
						// 保存手势库
						gestureLib.save();
				}).setNegativeButton(R.string.bn_cancel, null).show();
		}
	}
}

```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android_%E6%89%8B%E5%8A%BF/  

