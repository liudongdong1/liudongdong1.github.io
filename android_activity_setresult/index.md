# Android_activity_setResult


> 在一个主界面(主Activity)通过意图跳转至多个不同子Activity上去，当子模块的代码执行完毕后再次返回主页面，将`子activity中得到的数据显示在主界面/完成的数据交给主Activity处理`。这种带数据的意图跳转需要使用activity的onActivityResult()方法。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211204195533006.png)

### 1. 相关函数

**（1）startActivityForResult([Intent](http://blog.csdn.net/sjf0115/article/details/7385152) intent, int requestCode);**

　　 第一个参数：一个Intent对象，用于携带将跳转至下一个界面中使用的数据，使用putExtra(A,B)方法，此处存储的数据类型特别多，基本类型全部支持。

　　 第二个参数：如果> = 0,当Activity结束时requestCode将归还在onActivityResult()中。以便确定返回的数据是从哪个Activity中返回，`用来标识目标activity`。

　　与下面的resultCode功能一致，感觉Android就是为了保证数据的严格一致性特地设置了两把锁，来保证数据的发送，目的地的严格一致性。

**（2）onActivityResult(int requestCode, int resultCode, [Intent](http://blog.csdn.net/sjf0115/article/details/7385152) data)**

　　第一个参数：这个整数requestCode用于与startActivityForResult中的requestCode中值进行比较判断，是以便确认返回的数据是`从哪个Activity返回的`。

　　第二个参数：这整数resultCode是由子Activity通过其`setResult()方法返回`。适用于多个activity都返回数据时，来标识到底是哪一个activity返回的值。

　　第三个参数：一个Intent对象，`带有返回的数据`。可以通过data.getXxxExtra( );方法来获取指定数据类型的数据，

**（3）setResult(int resultCode, [Intent](http://blog.csdn.net/sjf0115/article/details/7385152) data)**

　　在意图跳转的目的地界面调用这个方`法把Activity想要返回的数据返回到主Activity`，

　　第一个参数：当Activity结束时resultCode将归还在onActivityResult()中，一般为RESULT_CANCELED , RESULT_OK该值默认为-1。

　　第二个参数：一个Intent对象，返回给主Activity的数据。在intent对象携带了要返回的数据，使用putExtra( )方法。

### 2. 实例Demo

```java
public class MainActivity extends Activity {

    private Button button;
    private final static int REQUESTCODE = 1; // 返回的结果码
    private EditText one, two, result;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        one = (EditText) findViewById(R.id.Text_one);
        two = (EditText) findViewById(R.id.Text_two);
        result = (EditText) findViewById(R.id.Text_result);
        button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                // 获取用户输入的两个值
                int a = Integer.parseInt(one.getText().toString());
                int b = Integer.parseInt(two.getText().toString());
                // 意图实现activity的跳转
                Intent intent = new Intent(MainActivity.this,
                        OtherActivity.class);
                intent.putExtra("a", a);
                intent.putExtra("b", b);
                // 这种启动方式：startActivity(intent);并不能返回结果
                startActivityForResult(intent, REQUESTCODE); //REQUESTCODE--->1
            }
        });
    }
    // 为了获取结果
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // RESULT_OK，判断另外一个activity已经结束数据输入功能，Standard activity result:
        // operation succeeded. 默认值是-1
        if (resultCode == 2) {
            if (requestCode == REQUESTCODE) {
                int three = data.getIntExtra("three", 0);
                //设置结果显示框的显示数值
                result.setText(String.valueOf(three));
            }
        }
    }
}
```

```java
public class OtherActivity extends Activity {
    private Button button;
    private TextView textView;
    private EditText editText;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // TODO Auto-generated method stub
        super.onCreate(savedInstanceState);
        setContentView(R.layout.other);
        button = (Button) findViewById(R.id.button2);
        textView = (TextView) findViewById(R.id.msg);
        editText = (EditText) findViewById(R.id.Text_three);
        // 去除传递过来的意图,并提取数据
        Intent intent = getIntent();此处并不是创建而是直接获取一个intent对象Return the intent that started this activity. 
        int a = intent.getIntExtra("a", 0); // 没有输入值默认为0
        int b = intent.getIntExtra("b", 0); // 没有输入值默认为0
        textView.setText(a + " + " + b + " = " + " ? ");
        button.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                Intent intent = new Intent();
                // 获取用户计算后的结果
                int three = Integer.parseInt(editText.getText().toString());
                intent.putExtra("three", three); //将计算的值回传回去
                //通过intent对象返回结果，必须要调用一个setResult方法，
                //setResult(resultCode, data);第一个参数表示结果返回码，一般只要大于1就可以，但是
                setResult(2, intent);
                finish(); //结束当前的activity的生命周期
            }
        });
    }
}
```

### Resource

- https://juejin.cn/post/6844903938790014990

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_activity_setresult/  

