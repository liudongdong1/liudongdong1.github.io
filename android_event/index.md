# Android_Event


> 基于监听的事件处理就是在android的组件上绑定特定的监听器，而基于回调的事件处理就是重写UI组件或者Activity的回调方法。
>
> 基于回调的事件处理模型就是将EventSource和EventListener合二为一了，即事件源也是事件监听器（处理器）

### 1. 基于监听的时间处理机制模型

> 事件监听机制是一种`委派式的事件处理机制`,事件源(组件)事件处理委托给事件监听器,当事件源发生指定事件时,就通知指定事件监听器,执行相应的操作

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108131626359.png)

#### .1. 直接匿名内部类

```java
package com.jay.example.innerlisten;    
import android.os.Bundle;    
import android.view.View;    
import android.view.View.OnClickListener;    
import android.widget.Button;    
import android.widget.Toast;    
import android.app.Activity;    
public class MainActivity extends Activity {    
    private Button btnshow;    
        
    @Override    
    protected void onCreate(Bundle savedInstanceState) {    
        super.onCreate(savedInstanceState);    
        setContentView(R.layout.activity_main);    
        btnshow = (Button) findViewById(R.id.btnshow);    
        btnshow.setOnClickListener(new OnClickListener() {    
            //重写点击事件的处理方法onClick()    
            @Override    
            public void onClick(View v) {    
                //显示Toast信息    
                Toast.makeText(getApplicationContext(), "你点击了按钮", Toast.LENGTH_SHORT).show();    
            }    
        });    
    }        
} 
```

#### .2. 使用内部类

```java
import android.os.Bundle;    
import android.view.View;    
import android.view.View.OnClickListener;    
import android.widget.Button;    
import android.widget.Toast;    
import android.app.Activity;     
public class MainActivity extends Activity {    
    private Button btnshow;    
    @Override    
    protected void onCreate(Bundle savedInstanceState) {    
        super.onCreate(savedInstanceState);    
        setContentView(R.layout.activity_main);    
        btnshow = (Button) findViewById(R.id.btnshow);    
        //直接new一个内部类对象作为参数    
        btnshow.setOnClickListener(new BtnClickListener());    
    }     
    //定义一个内部类,实现View.OnClickListener接口,并重写onClick()方法    
    class BtnClickListener implements View.OnClickListener    
    {    
        @Override    
        public void onClick(View v) {    
            Toast.makeText(getApplicationContext(), "按钮被点击了", Toast.LENGTH_SHORT).show();   
        }    
    }    
} 
```

#### .3. 使用外部类

- 实现OnClicklistener的自定义监听类，可以将多个空间的监听出发函数放在一个类里面，通过View.type 进行分别处理。

```java
/**
     * @function click控件相关监听函数
     * */
    class MyClickListener implements View.OnClickListener {

        @Override
        public void onClick(View v) {
            // TODO Auto-generated method stub
            switch (v.getId()) {
                case R.id.search:
                    Log.i(TAG,"you click the search bluetooth button");
                    search();
                    //Toast.makeText(MainActivity.this, "This is Button 111", Toast.LENGTH_SHORT).show();
                    break;
                case R.id.discoverable1:
                    Log.i(TAG,"you click the search discoverable button to make your phone to be discovered");
                    Toast.makeText(sqlitetest.this, "该设备已设置为可在300秒内发现，且可连接", Toast.LENGTH_SHORT).show();
                    ensureDiscoverable();
                    break;
                case R.id.button3:
                    Log.i(TAG,"you click the button3 to clear data");
                    clearButtonHandler();
                    //Toast.makeText(MainActivity.this, "This is Button 111", Toast.LENGTH_SHORT).show();
                    break;
                case R.id.button4:
                    Log.i(TAG,"you click the button4 to recognize the gesture");
                    performRecognize();
                    break;
                case R.id.button5:
                    Log.i(TAG,"you click the button5 to save data");
                    performSave();
                    //Toast.makeText(MainActivity.this, "This is Button 111", Toast.LENGTH_SHORT).show();
                    break;
                default:
                    break;
            }
        }
    }
```

```java
import android.os.Bundle;    
import android.widget.Button;    
import android.widget.TextView;    
import android.app.Activity;    
public class MainActivity extends Activity {    
    private Button btnshow;    
    private TextView txtshow;    
    @Override    
    protected void onCreate(Bundle savedInstanceState) {    
        super.onCreate(savedInstanceState);    
        setContentView(R.layout.activity_main);    
        btnshow = (Button) findViewById(R.id.btnshow);    
        txtshow = (TextView) findViewById(R.id.textshow);    
        //直接new一个外部类，并把TextView作为参数传入    
        btnshow.setOnClickListener(new MyClick(txtshow));    
    }         
} 
```

#### .4. 直接使用Activity作为事件监听器

> 只需要`让Activity类实现XxxListener事件监听接口`,在Activity中定义重写对应的事件处理器方法 eg:Actitity实现了OnClickListener接口,`重写了onClick(view)方法`在为某些组建添加该事件监听对象时,直接`setXxx.Listener(this) `即可

```java
import android.os.Bundle;    
import android.view.View;    
import android.view.View.OnClickListener;    
import android.widget.Button;    
import android.widget.Toast;    
import android.app.Activity;    
    
//让Activity方法实现OnClickListener接口    
public class MainActivity extends Activity implements OnClickListener{    
    private Button btnshow;    
    @Override    
    protected void onCreate(Bundle savedInstanceState) {    
        super.onCreate(savedInstanceState);    
        setContentView(R.layout.activity_main);    
            
        btnshow = (Button) findViewById(R.id.btnshow);    
        //直接写个this    
        btnshow.setOnClickListener(this);    
    }    
    //重写接口中的抽象方法    
    @Override    
    public void onClick(View v) {    
        Toast.makeText(getApplicationContext(), "点击了按钮", Toast.LENGTH_SHORT).show();         
    }         
}   
```

#### .5. 直接绑定到标签

> 直接在`xml布局文件中对应得Activity中定义一个事件处理方法` eg:public void myClick(View source) source对应事件源(组件) 接着布局文件中对应要触发事件的组建,设置一个属性:`onclick = "myclick"即可`

```java
import android.app.Activity;    
import android.os.Bundle;    
import android.view.View;    
import android.widget.Toast;    
    
public class MainActivity extends Activity {    
    @Override    
    protected void onCreate(Bundle savedInstanceState) {    
        super.onCreate(savedInstanceState);    
        setContentView(R.layout.activity_main);     
    }    
    //自定义一个方法,传入一个view组件作为参数    
    public void myclick(View source)    
    {    
        Toast.makeText(getApplicationContext(), "按钮被点击了", Toast.LENGTH_SHORT).show();    
    }    
} 
```

```xml
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"    
              xmlns:tools="http://schemas.android.com/tools"    
              android:id="@+id/LinearLayout1"    
              android:layout_width="match_parent"    
              android:layout_height="match_parent"    
              android:orientation="vertical" >    
    <Button     
            android:layout_width="wrap_content"    
            android:layout_height="wrap_content"    
            android:text="按钮"    
            android:onClick="myclick"/>    
</LinearLayout>  
```

#### .6. 成员变量

```java
private OnClickListener listener = new OnClickListener() {

    @Override
    public void onClick(View v) {
        //点击事件
    }
};

@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Button back = (Button) findViewById(R.id.back);
    back.setOnClickListener(listener);
}
```

### 2. 基于回调的事件处理机制

#### .1. 自定义view

>- 如果处理事件的回调方法返回true，表明`该处理方法已完全处理该事件`，`该事件不会传播出去`。
>- 如果处理事件的回调方法返回false，表明该处理方法并未完全处理该事件，该事件会传播出去。

> 当用户在GUI组件上激发某个事件时,`组件有自己特定的方法会负责处理该事件`。通常用法:`继承基本的GUI组件`,`重写该组件的事件处理方法`,即自定义view 注意:在xml布局中使用自定义的view时,需要使用"`全限定类名`"
>
> 对于基于回调的事件传播而言，某组件上所发生的事情`不仅激发该组件上的回调方法`,`也会触发该组件所在Activity的回调用法`——只要事件能传播到该Activity。

> *①在该组件上触发屏幕事件: boolean* **onTouchEvent***(MotionEvent event);
> *②在该组件上按下某个按钮时: boolean* **onKeyDown***(int keyCode,KeyEvent event);
> *③松开组件上的某个按钮时: boolean* **onKeyUp***(int keyCode,KeyEvent event);
> *④长按组件某个按钮时: boolean* **onKeyLongPress***(int keyCode,KeyEvent event);
> *⑤键盘快捷键事件发生: boolean* **onKeyShortcut***(int keyCode,KeyEvent event);
> *⑥在组件上触发轨迹球屏事件: boolean* **onTrackballEvent***(MotionEvent event);
> **⑦当组件的焦点发生改变,和前面的6个不同,这个`方法只能够在View中重写`哦！ protected void onFocusChanged (boolean gainFocus, int direction, Rect previously FocusedRect)

```java
public class MyButton extends Button{  
    private static String TAG = "呵呵";  
    public MyButton(Context context, AttributeSet attrs) {  
        super(context, attrs);  
    }  
  
    //重写键盘按下触发的事件  
    @Override  
    public boolean onKeyDown(int keyCode, KeyEvent event) {  
        super.onKeyDown(keyCode,event);  
        Log.i(TAG, "onKeyDown方法被调用");  
        return true;  
    }  
  
    //重写弹起键盘触发的事件  
    @Override  
    public boolean onKeyUp(int keyCode, KeyEvent event) {  
        super.onKeyUp(keyCode,event);  
        Log.i(TAG,"onKeyUp方法被调用");  
        return true;  
    }  
  
    //组件被触摸了  
    @Override  
    public boolean onTouchEvent(MotionEvent event) {  
        super.onTouchEvent(event);  
        Log.i(TAG,"onTouchEvent方法被调用");  
        return true;  
    }  
} 
```

```xml
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"  
    xmlns:tools="http://schemas.android.com/tools"  
    android:layout_width="match_parent"  
    android:layout_height="match_parent"  
    tools:context=".MyActivity">  
    <example.jay.com.mybutton.MyButton  
        android:layout_width="wrap_content"  
        android:layout_height="wrap_content"  
        android:text="按钮"/> 
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108162001757.png)

```java
public class MyActivity extends ActionBarActivity {  
    @Override  
    protected void onCreate(Bundle savedInstanceState) {  
        super.onCreate(savedInstanceState);  
        setContentView(R.layout.activity_my);  
  
        Button btn = (Button)findViewById(R.id.btn_my);  
        btn.setOnKeyListener(new View.OnKeyListener() {  
            @Override  
            public boolean onKey(View v, int keyCode, KeyEvent event) {  
                if(event.getAction() == KeyEvent.ACTION_DOWN)  
                {  
                    Log.i("呵呵","监听器的onKeyDown方法被调用");  
                }  
                return false;  
            }  
        });  
    }  
  
    @Override  
    public boolean onKeyDown(int keyCode, KeyEvent event) {  
        super.onKeyDown(keyCode, event);  
        Log.i("呵呵","Activity的onKeyDown方法被调用");  
        return false;  
    }  
} 
```

> 传播的顺序是: **监听器**--->**view组件的回调方法**--->**Activity的回调方法**

### 3. 相应系统设置的事件

- **densityDpi**：屏幕密度
- **fontScale**：当前用户设置的字体的缩放因子
- **hardKeyboardHidden**：判断硬键盘是否可见，有两个可选值：HARDKEYBOARDHIDDEN_NO,HARDKEYBOARDHIDDEN_YES，分别是十六进制的0和1
- **keyboard**：获取当前关联额键盘类型：该属性的返回值：KEYBOARD_12KEY（只有12个键的小键盘）、KEYBOARD_NOKEYS、KEYBOARD_QWERTY（普通键盘）
- **keyboardHidden**：该属性返回一个boolean值用于标识当前键盘是否可用。该属性不仅会判断系统的硬件键盘，也会判断系统的软键盘（位于屏幕）。
- **locale**：获取用户当前的语言环境
- **mcc**：获取移动信号的国家码
- **mnc**：获取移动信号的网络码
  ps:国家代码和网络代码共同确定当前手机网络运营商
- **navigation**：判断系统上方向导航设备的类型。该属性的返回值：NAVIGATION_NONAV（无导航）、 NAVIGATION_DPAD(DPAD导航）NAVIGATION_TRACKBALL（轨迹球导航）、NAVIGATION_WHEEL（滚轮导航）
- **orientation**：获取系统屏幕的方向。该属性的返回值：ORIENTATION_LANDSCAPE（横向屏幕）、ORIENTATION_PORTRAIT（竖向屏幕）
- **screenHeightDp**，**screenWidthDp**：屏幕可用高和宽，用dp表示
- **touchscreen**：获取系统触摸屏的触摸方式。该属性的返回值：TOUCHSCREEN_NOTOUCH（无触摸屏）、TOUCHSCREEN_STYLUS（触摸笔式触摸屏）、TOUCHSCREEN_FINGER（接收手指的触摸屏）

```java
public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TextView txtResult = (TextView) findViewById(R.id.txtResult);
        StringBuffer status = new StringBuffer();
        //①获取系统的Configuration对象
        Configuration cfg = getResources().getConfiguration();
        //②想查什么查什么
        status.append("densityDpi:" + cfg.densityDpi + "\n");
        status.append("fontScale:" + cfg.fontScale + "\n");
        status.append("hardKeyboardHidden:" + cfg.hardKeyboardHidden + "\n");
        status.append("keyboard:" + cfg.keyboard + "\n");
        status.append("keyboardHidden:" + cfg.keyboardHidden + "\n");
        status.append("locale:" + cfg.locale + "\n");
        status.append("mcc:" + cfg.mcc + "\n");
        status.append("mnc:" + cfg.mnc + "\n");
        status.append("navigation:" + cfg.navigation + "\n");
        status.append("navigationHidden:" + cfg.navigationHidden + "\n");
        status.append("orientation:" + cfg.orientation + "\n");
        status.append("screenHeightDp:" + cfg.screenHeightDp + "\n");
        status.append("screenWidthDp:" + cfg.screenWidthDp + "\n");
        status.append("screenLayout:" + cfg.screenLayout + "\n");
        status.append("smallestScreenWidthDp:" + cfg.densityDpi + "\n");
        status.append("touchscreen:" + cfg.densityDpi + "\n");
        status.append("uiMode:" + cfg.densityDpi + "\n");
        txtResult.setText(status.toString());
    }
}
```

### 4. AsyncTask

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108163130253.png)

```java
public class MyAsyncTask extends AsyncTask<Integer,Integer,String>  
{  
    private TextView txt;  
    private ProgressBar pgbar;  
    public MyAsyncTask(TextView txt,ProgressBar pgbar)  
    {  
        super();  
        this.txt = txt;  
        this.pgbar = pgbar;  
    }  
    //该方法不运行在UI线程中,主要用于异步操作,通过调用publishProgress()方法  
    //触发onProgressUpdate对UI进行操作  
    @Override  
    protected String doInBackground(Integer... params) {  
        DelayOperator dop = new DelayOperator();  
        int i = 0;  
        for (i = 10;i <= 100;i+=10)  
        {  
            dop.delay();  
            publishProgress(i);  
        }  
        return  i + params[0].intValue() + "";  
    }  
    //该方法运行在UI线程中,可对UI控件进行设置  
    @Override  
    protected void onPreExecute() {  
        txt.setText("开始执行异步线程~");  
    }  
    //在doBackground方法中,每次调用publishProgress方法都会触发该方法  
    //运行在UI线程中,可对UI控件进行操作  
    @Override  
    protected void onProgressUpdate(Integer... values) {  
        int value = values[0];  
        pgbar.setProgress(value);  
    }  
}
```

```java
public class MyActivity extends ActionBarActivity {  
 
    private TextView txttitle;  
    private ProgressBar pgbar;  
    private Button btnupdate;  
    @Override  
    protected void onCreate(Bundle savedInstanceState) {  
        super.onCreate(savedInstanceState);  
        setContentView(R.layout.activity_main);  
        txttitle = (TextView)findViewById(R.id.txttitle);  
        pgbar = (ProgressBar)findViewById(R.id.pgbar);  
        btnupdate = (Button)findViewById(R.id.btnupdate);  
        btnupdate.setOnClickListener(new View.OnClickListener() {  
            @Override  
            public void onClick(View v) {  
                MyAsyncTask myTask = new MyAsyncTask(txttitle,pgbar);  
                myTask.execute(1000);  
            }  
        });  
    }  
} 
```

### 5. Resource

- todo？ 这种监听或者回调模式设计思想，举Button 案例

```java
//view 基类通过setOnClickListener() 注入OnClickListener 监听处理事件
public void setOnClickListener(OnClickListener l) {
    throw new RuntimeException("Stub!");
}

// Button 案件继承TextView
public class Button extends TextView {
    
}
// TextView 继承View 和 ViewTreeObserver.OnPreDrawListener，后面这个类负责事件监听
public class TextView extends View implements ViewTreeObserver.OnPreDrawListener {
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221224152519789.png)

- view， viewgroup，layout，ViewTreeObserver 之间关系： https://blog.csdn.net/HardWorkingAnt/article/details/77408329



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android_event/  

