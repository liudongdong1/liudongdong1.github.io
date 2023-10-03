# Android_fragment


> 使用Fragment 我们可以把页面结构划分成几块，每块使用一个Fragment来管理。这样我们可以更加方便的在运行过程中动态地更新Activity中的用户界面，日后迭代更新、维护也是更加方便。 **Fragment并不能单独使用，他需要嵌套在Activity 中使用**，尽管他拥有自己的生命周期，但是还是会受到宿主Activity的生命周期的影响，比如Activity 被destory销毁了，他也会跟着销毁！一个Activity可以嵌套多个Fragment。

### 1. 基本概念

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/41442282.jpg)

### 2. Fragment 生命周期

①Activity加载Fragment的时候,依次调用下面的方法: **onAttach** -> **onCreate** -> **onCreateView** -> **onActivityCreated** -> **onStart** ->**onResume**

②当我们`启动一个新的页面, 此时Fragment所在的Activity不可见`，会执行 **onPause**

③当`新页面返回后，当前Activity和Fragment又可见了`，会再次执行**onStart**和 **onResume**

⑥`退出了Activity的话,那么Fragment将会被完全结束, Fragment会进入销毁状态` **onPause** -> **onStop** -> **onDestoryView** -> **onDestory** -> **onDetach**

![fragment_lifecycle](https://gitee.com/github-25970295/blogpictureV2/raw/master/fragment_lifecycle.jpeg)

### 3. 创建Fragment

#### .1. 静态加载

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108095652651.png)

- 自定义Fragment 类

```java
public class Fragmentone extends Fragment {
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment1, container,false);
        return view;
    }   
}
```

- 在需要加载Fragment Activity 对应的布局文件中添加对应的fragment的标签

```xml
<fragment
    android:id="@+id/fragment1"
    android:name="com.jay.example.fragmentdemo.Fragmentone"  #属性是全限定类名，就是要包含Fragment的包名
    android:layout_width="match_parent"
    android:layout_height="0dp"
    android:layout_weight="1" />
```

#### .2. 动态加载Fragment类

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108100538891.png)

```java
public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Display dis = getWindowManager().getDefaultDisplay();
        if(dis.getWidth() > dis.getHeight())
        {
            Fragment1 f1 = new Fragment1();
            getFragmentManager().beginTransaction().replace(R.id.LinearLayout1, f1).commit();
        }
        
        else
        {
            Fragment2 f2 = new Fragment2();
            getFragmentManager().beginTransaction().replace(R.id.LinearLayout1, f2).commit();
        }
    }   
}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108100707502.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211108100808883.png)

### 4. [数据传输](https://carsonho.blog.csdn.net/article/details/75453770)

#### .1. activity->fragment

```java
// 步骤1：获取FragmentManager
FragmentManager fragmentManager = getFragmentManager();

// 步骤2：获取FragmentTransaction
FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();

// 步骤3：创建需要添加的Fragment 
final mFragment fragment = new mFragment();

// 步骤4:创建Bundle对象
// 作用:存储数据，并传递到Fragment中
Bundle bundle = new Bundle();

// 步骤5:往bundle中添加数据
bundle.putString("message", "I love Google");

// 步骤6:把数据设置到Fragment中
fragment.setArguments(bundle);

// 步骤7：动态添加fragment
// 即将创建的fragment添加到Activity布局文件中定义的占位符中（FrameLayout）
fragmentTransaction.add(R.id.fragment_container, fragment);
fragmentTransaction.commit();
```

```java
public class mFragment extends Fragment {
    Button button;
    TextView text;
    Bundle bundle;
    String message;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View contentView = inflater.inflate(R.layout.fragment, container, false);
        // 设置布局文件

        button = (Button) contentView.findViewById(R.id.button);
        text = (TextView) contentView.findViewById(R.id.text);

        // 步骤1:通过getArgments()获取从Activity传过来的全部值
        bundle = this.getArguments();

        // 步骤2:获取某一值
        message = bundle.getString("message");

        // 步骤3:设置按钮,将设置的值显示出来
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                // 显示传递过来的值
                text.setText(message);

            }
        });

        return contentView;
    }
}
```

#### .2. fragment->activity

- 设置回调接口

```java
public interface ICallBack {
    void get_message_from_Fragment(String string);

}
```

- 设置fragment类

```java
public class mFragment extends Fragment {

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View contentView = inflater.inflate(R.layout.fragment, container, false);
        // 设置布局文件
        return contentView;
    }

    // 设置 接口回调 方法
    public void sendMessage(ICallBack callBack){

        callBack.get_message_from_Fragment("消息:我来自Fragment");

    }
}
```

```java
// 步骤1：获取FragmentManager
FragmentManager fragmentManager = getFragmentManager();

// 步骤2：获取FragmentTransaction
FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();

// 步骤3：创建需要添加的Fragment 
final mFragment fragment = new mFragment();

// 步骤4：动态添加fragment
// 即将创建的fragment添加到Activity布局文件中定义的占位符中（FrameLayout）
fragmentTransaction.add(R.id.fragment_container, fragment);
fragmentTransaction.commit();


button.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {

        // 通过接口回调将消息从fragment发送到Activity
        fragment.sendMessage(new ICallBack() {
            @Override
            public void get_message_from_Fragment(String string) {
                text.setText(string);
            }
        });

    }
});
```

#### .3. Fragment-Activity 菜单问题

- https://blog.csdn.net/fei20121106/article/details/54580385  具体解释
- 想让`Fragment`中的onCreateOptionsMenu生效必须先调用`setHasOptionsMenu`方法
- [Fragment](https://so.csdn.net/so/search?q=Fragment&spm=1001.2101.3001.7020)和Activity一样，可以重写onCreateOptionsMenu方法来设定自己的菜单，其实这两个地方使用onCreateOptionsMenu的目的和效果都是完全一样的
- 如果Fragment和Activity都同时`inflate`了一个menu资源文件，那么`menu资源`所包含的菜单会出现两次

### 5. Adapter 作用

![Package fragment](https://gitee.com/github-25970295/blogimgv2022/raw/master/Package%20fragment-166985985806695.png)

#### 1. BaseAdapter

##### 1. 源码

```java
public abstract class BaseAdapter implements ListAdapter, SpinnerAdapter {
    private final DataSetObservable mDataSetObservable = new DataSetObservable();
    private CharSequence[] mAutofillOptions;

    public boolean hasStableIds() {
        return false;
    }
    
    public void registerDataSetObserver(DataSetObserver observer) {
        mDataSetObservable.registerObserver(observer);
    }

    public void unregisterDataSetObserver(DataSetObserver observer) {
        mDataSetObservable.unregisterObserver(observer);
    }
    
    /**
     * Notifies the attached observers that the underlying data has been changed
     * and any View reflecting the data set should refresh itself.
     */
    public void notifyDataSetChanged() {
        mDataSetObservable.notifyChanged();
    }

    /**
     * Notifies the attached observers that the underlying data is no longer valid
     * or available. Once invoked this adapter is no longer valid and should
     * not report further data set changes.
     */
    public void notifyDataSetInvalidated() {
        mDataSetObservable.notifyInvalidated();
    }

    public boolean areAllItemsEnabled() {
        return true;
    }

    public boolean isEnabled(int position) {
        return true;
    }

    public View getDropDownView(int position, View convertView, ViewGroup parent) {
        return getView(position, convertView, parent);
    }

    public int getItemViewType(int position) {
        return 0;
    }

    public int getViewTypeCount() {
        return 1;
    }
    
    public boolean isEmpty() {
        return getCount() == 0;
    }

    @Override
    public CharSequence[] getAutofillOptions() {
        return mAutofillOptions;
    }

    /**
     * Sets the value returned by {@link #getAutofillOptions()}
     */
    public void setAutofillOptions(@Nullable CharSequence... options) {
        mAutofillOptions = options;
    }
}
```

```java
public class DataSetObservable extends Observable<DataSetObserver> {
    /**
     * Invokes {@link DataSetObserver#onChanged} on each observer.
     * Called when the contents of the data set have changed.  The recipient
     * will obtain the new contents the next time it queries the data set.
     */
    public void notifyChanged() {
        synchronized(mObservers) {
            // since onChanged() is implemented by the app, it could do anything, including
            // removing itself from {@link mObservers} - and that could cause problems if
            // an iterator is used on the ArrayList {@link mObservers}.
            // to avoid such problems, just march thru the list in the reverse order.
            for (int i = mObservers.size() - 1; i >= 0; i--) {
                mObservers.get(i).onChanged();
            }
        }
    }

    /**
     * Invokes {@link DataSetObserver#onInvalidated} on each observer.
     * Called when the data set is no longer valid and cannot be queried again,
     * such as when the data set has been closed.
     */
    public void notifyInvalidated() {
        synchronized (mObservers) {
            for (int i = mObservers.size() - 1; i >= 0; i--) {
                mObservers.get(i).onInvalidated();
            }
        }
    }
}
```

##### 2. 数据binding

- https://www.examplecode.cn/2018/07/26/android-databinding-05-binding-objects/
- https://blog.51cto.com/u_15252276/5026746

##### 2. adapter viewholder模式

- https://hellosure.github.io/android/2015/06/02/android-viewholder  直接使用，contentview 缓冲，viewholder优化
- https://blog.csdn.net/qq_26222859/article/details/46827511  关于viewholder 解释

```java
public class ViewHolder {  
    // I added a generic return type to reduce the casting noise in client code  
    @SuppressWarnings("unchecked")  
    public static <T extends View> T get(View view, int id) {  
        SparseArray<View> viewHolder = (SparseArray<View>) view.getTag();  
        if (viewHolder == null) {  
            viewHolder = new SparseArray<View>();  
            view.setTag(viewHolder);  
        }  
        View childView = viewHolder.get(id);  
        if (childView == null) {  
            childView = view.findViewById(id);  
            viewHolder.put(id, childView);  
        }  
        return (T) childView;  
    }  
}  
```

```java
public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
    View rootView = inflater.inflate(R.layout.fr_image_list, container, false);
    listView = (ListView) rootView.findViewById(android.R.id.list);  //通过id找到对应的fragment view
    ((ListView) listView).setAdapter(new ImageAdapter(getActivity()));
    listView.setOnItemClickListener(new OnItemClickListener() {
        @Override
        public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
            startImagePagerActivity(position);
        }
    });
    return rootView;
}

private static class ImageAdapter extends BaseAdapter {

    private static final String[] IMAGE_URLS = Constants.IMAGES;

    private LayoutInflater inflater;
    private ImageLoadingListener animateFirstListener = new AnimateFirstDisplayListener();

    private DisplayImageOptions options;

    ImageAdapter(Context context) {
        inflater = LayoutInflater.from(context);

        options = new DisplayImageOptions.Builder()
            .showImageOnLoading(R.drawable.ic_stub)
            .showImageForEmptyUri(R.drawable.ic_empty)
            .showImageOnFail(R.drawable.ic_error)
            .cacheInMemory(true)
            .cacheOnDisk(true)
            .considerExifParams(true)
            .displayer(new CircleBitmapDisplayer(Color.WHITE, 5))
            .build();
    }

    @Override
    public int getCount() {
        return IMAGE_URLS.length;
    }   // 类似DataLoader 函数

    @Override
    public Object getItem(int position) {
        return position;
    }

    @Override
    public long getItemId(int position) {
        return position;
    }
    /*当第一个view滑出屏幕时，这个view就会落到recycler中，这时convertview就不再等于null了，而是这个view了，只可惜，
		每一个落到recycler中的view都会剥离掉它的值，只剩下它的型。
		也就是说落到recycler中的view仍然绑定了ViewHolder对象，但是convertview里的组件的内容已经不复存在了，
		需要重新设置。*/
    @Override
    public View getView(final int position, View convertView, ViewGroup parent) {
        View view = convertView;
        final ViewHolder holder;
        if (convertView == null) {
            view = inflater.inflate(R.layout.item_list_image, parent, false);
            holder = new ViewHolder();
            holder.text = (TextView) view.findViewById(R.id.text);
            holder.image = (ImageView) view.findViewById(R.id.image);
            view.setTag(holder);  //setTag才是将这些缓存起来供下次调用
        } else {
            holder = (ViewHolder) view.getTag();
        }

        holder.text.setText("Item " + (position + 1));

        ImageLoader.getInstance().displayImage(IMAGE_URLS[position], holder.image, options, animateFirstListener);

        return view;
    }
}
//ViewHolder只是将需要缓存的那些view封装好
static class ViewHolder {
    TextView text;
    ImageView image;
}
```

#### 2. PagerAdapter

- PagerAdapter主要是viewpager的适配器，而viewPager则也是在android.support.v4扩展包中新添加的一个强大的控件，可以实现控件的滑动效果，比如咱们在软件中常见的广告栏的滑动效果，用viewPager就可以实现。

```java
private static class ImageAdapter extends PagerAdapter {

    private static final String[] IMAGE_URLS = Constants.IMAGES;

    private LayoutInflater inflater;
    private DisplayImageOptions options;

    ImageAdapter(Context context) {
        inflater = LayoutInflater.from(context);

        options = new DisplayImageOptions.Builder()
            .showImageForEmptyUri(R.drawable.ic_empty)
            .showImageOnFail(R.drawable.ic_error)
            .resetViewBeforeLoading(true)
            .cacheOnDisk(true)
            .imageScaleType(ImageScaleType.EXACTLY)
            .bitmapConfig(Bitmap.Config.RGB_565)
            .considerExifParams(true)
            .displayer(new FadeInBitmapDisplayer(300))
            .build();
    }
    // PagerAdapter只缓存三张要显示的图片，如果滑动的图片超出了缓存的范围，就会调用这个方法，将图片销毁
    @Override
    public void destroyItem(ViewGroup container, int position, Object object) {
        container.removeView((View) object);
    }

    @Override
    public int getCount() {
        return IMAGE_URLS.length;
    }
    // 当要显示的图片可以进行缓存的时候，会调用这个方法进行显示图片的初始化，我们将要显示的ImageView加入到ViewGroup中，然后作为返回值返回即可
    @Override
    public Object instantiateItem(ViewGroup view, int position) {
        View imageLayout = inflater.inflate(R.layout.item_pager_image, view, false);
        assert imageLayout != null;
        ImageView imageView = (ImageView) imageLayout.findViewById(R.id.image);
        final ProgressBar spinner = (ProgressBar) imageLayout.findViewById(R.id.loading);

        ImageLoader.getInstance().displayImage(IMAGE_URLS[position], imageView, options, new SimpleImageLoadingListener() {
            @Override
            public void onLoadingStarted(String imageUri, View view) {
                spinner.setVisibility(View.VISIBLE);
            }

            @Override
            public void onLoadingFailed(String imageUri, View view, FailReason failReason) {
                String message = null;
                switch (failReason.getType()) {
                    case IO_ERROR:
                        message = "Input/Output error";
                        break;
                    case DECODING_ERROR:
                        message = "Image can't be decoded";
                        break;
                    case NETWORK_DENIED:
                        message = "Downloads are denied";
                        break;
                    case OUT_OF_MEMORY:
                        message = "Out Of Memory error";
                        break;
                    case UNKNOWN:
                        message = "Unknown error";
                        break;
                }
                Toast.makeText(view.getContext(), message, Toast.LENGTH_SHORT).show();

                spinner.setVisibility(View.GONE);
            }

            @Override
            public void onLoadingComplete(String imageUri, View view, Bitmap loadedImage) {
                spinner.setVisibility(View.GONE);
            }
        });

        view.addView(imageLayout, 0);
        return imageLayout;
    }
    // 来判断显示的是否是同一张图片，这里我们将两个参数相比较返回即可
    @Override
    public boolean isViewFromObject(View view, Object object) {
        return view.equals(object);
    }

    @Override
    public void restoreState(Parcelable state, ClassLoader loader) {
    }

    @Override
    public Parcelable saveState() {
        return null;
    }
}
```

### Resource

- https://www.runoob.com/w3cnote/android-tutorial-fragment-base.html


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_fragment/  

