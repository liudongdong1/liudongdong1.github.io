# Tools_ButterKnife


> Android`视图的字段和方法绑定`，使用注释处理生成样板代码。
>
> 1. 在字段上使用@BindView消除findViewById调用。
> 2. 将列表或数组中的多个视图分组。 使用操作，设置器或属性一次操作所有这些操作。
> 3. 通过使用`@OnClick和其他方法注释方法来消除侦听器的匿名内部类。`
> 4. 通过在`字段上使用资源注释来消除资源查找`。

```json
dependencies {
    implementation 'com.jakewharton:butterknife:8.8.1'
    annotationProcessor 'com.jakewharton:butterknife-compiler:8.8.1'
}
```

### 1. 注解说明

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211119154900163.png)

### 2. 事件注解说明

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211119154945321.png)

### 3. 具体使用

#### .1. Activity

```java
private Unbinder unbinder;
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    unbinder = ButterKnife.bind(this); // 返回一个Unbinder对象
}
@Override
protected void onDestroy() {
    super.onDestroy();
    if (unbinder != null) {
        unbinder.unbind();
    }
}
```

#### .2. Fragment

```java
public class HomeFragment extends Fragment {
    private Unbinder unbinder;
    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.activity_main, container, false);
        unbinder = ButterKnife.bind(this, view);
        return view;
    }
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (unbinder != null) {
            unbinder.unbind();
        }
    }
}
```

#### .3. ViewHolder

```java
final class ViewHolder {

    @BindView(R.id.nameView)
    TextView nameView;

    @BindView(R.id.ageView)
    TextView ageView;

    ViewHolder(View view) {
        ButterKnife.bind(this, view);
    }

}
```

#### .4. 绑定view

```java
// 绑定多view
@BindViews({R.id.textView1, R.id.textView2, R.id.textView3})
List<TextView> viewList;

// 单个绑定
@BindView(R.id.editText1)
EditText editText;

@BindView(R.id.listView1)
ListView listView;
```

#### .5. 绑定资源

```java
@BindString(R.string.app_name)
String appName;

@BindArray(R.array.list)
String[] array;

@BindBitmap(R.mipmap.ic_launcher)
Bitmap icon;

@BindColor(R.color.colorAccent)
int colorAccent;

@BindDimen(R.dimen.width)
int width;

@BindAnim(R.anim.anim_translate_1)
Animation translateAnimation;
```

#### .6. 绑定事件

```java
// 绑定单个按钮
@OnClick(R.id.button1)
public void onClick(View view) {
    showToast("点击了按钮：" + view.toString());
}
// 绑定多个按钮
@OnClick({R.id.button1, R.id.button2, R.id.button3})
public void onClickEx(View view) {
    switch (view.getId()) {
        case R.id.button1:
            break;

        case R.id.button2:
            break;

        case R.id.button3:
            break;
    }
}
@OnTouch(R.id.textView1)
public boolean onTouch(View view) {
    showToast("touch:" + view.toString());
    return true;
}
@OnFocusChange(R.id.editText1)
public void onFocusChange(View view, boolean flag) {
    showToast("焦点改变...");
}
@OnItemClick(R.id.listView1)
public void onItemClickListener(int position) {
    showToast("ListView点击位置：" + position);
}
@OnItemLongClick(R.id.listView1)
public boolean onItemLongClickListener(int position) {
    showToast("ListView长按位置：" + position);
    return true;
}
@OnItemSelected(R.id.spinner1)
public void onItemSelectedClickListener(int position) {
    showToast("Spinner选择位置：" + position);
}
@OnItemSelected(value = R.id.spinner1, callback = OnItemSelected.Callback.NOTHING_SELECTED)
public void onNothingSelectedClickListener() {
    showToast("Spinner未选择item");
}
```

### Resource

- https://juejin.cn/post/6844903638595272711#heading-2

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/tools_butterknite/  

