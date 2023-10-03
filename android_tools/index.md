# Android_tools


![](https://cdn.jiler.cn/techug/uploads/2016/05/15-android-framework.png)

### 1. 图表绘制类

#### .1.  MPAndroidChart

```json
implementation 'com.github.PhilJay:MPAndroidChart:v3.1.0'
```

```json
allprojects {
    repositories {
        google()
        jcenter()
        maven { url "https://jitpack.io" }
    }
}
```

- 绘制多个线条demo

```xml
xml布局， lineChart 部件
<com.github.mikephil.charting.charts.LineChart
            android:id="@+id/chart1"
            android:layout_width="match_parent"
            android:layout_height="match_parent"
            tools:ignore="MissingClass" />
```

```java
/**
* @function: 初始化chart相关设置和监听函数
* */
public void initializeChart(){
    lineChart.setOnChartValueSelectedListener(this);
    lineChart.getDescription().setEnabled(true);
    lineChart.setTouchEnabled(true);
    // enable scaling and dragging
    lineChart.setDragEnabled(true);
    lineChart.setScaleEnabled(true);
    lineChart.setDrawGridBackground(false);
    // if disabled, scaling can be done on x- and y-axis separately
    lineChart.setPinchZoom(true);
    // set an alternative background color
    lineChart.setBackgroundColor(Color.TRANSPARENT);
    LineData data = new LineData();
    // 添加五条数据
    for(int i=0;i<5;i++){
        data.addDataSet(createSet("Flex"+i,ColorTemplate.LIBERTY_COLORS[i]));
    }
    data.setValueTextColor(Color.WHITE);
    lineChart.setData(data);
    // get the legend (only possible after setting data)
    Legend l = lineChart.getLegend();
    // modify the legend ...
    l.setForm(Legend.LegendForm.LINE);
    //l.setTypeface(tfLight);
    l.setTextColor(Color.WHITE);

    XAxis xl = lineChart.getXAxis();
    //xl.setTypeface(tfLight);
    xl.setTextColor(Color.WHITE);
    xl.setDrawGridLines(false);
    xl.setAvoidFirstLastClipping(true);
    xl.setEnabled(true);

    YAxis leftAxis = lineChart.getAxisLeft();
    //leftAxis.setTypeface(tfLight);
    leftAxis.setTextColor(Color.WHITE);
    leftAxis.setAxisMaximum(180f);
    leftAxis.setAxisMinimum(0f);
    leftAxis.setDrawGridLines(true);

    YAxis rightAxis = lineChart.getAxisRight();
    rightAxis.setEnabled(false);
}

/**
* @function: 添加模拟数据集,测试函数绘制功能
* */
private void addEntry() {

    LineData data = lineChart.getData();
    if (data != null&&data.getDataSetCount()==5) {

        for(int i=0;i<5;i++){
            ILineDataSet set = data.getDataSetByIndex(i);
            data.addEntry(new Entry(set.getEntryCount(), (float) (Math.random() * 30) + 30f*i), i);
        }
        data.notifyDataChanged();
        // let the lineChart know it's data has changed
        lineChart.notifyDataSetChanged();
        // limit the number of visible entries
        lineChart.setVisibleXRangeMaximum(120);
        // lineChart.setVisibleYRange(30, AxisDependency.LEFT)
        // move to the latest entry
        lineChart.moveViewToX(data.getEntryCount());
        // this automatically refreshes the lineChart (calls invalidate())
        // lineChart.moveViewTo(data.getXValCount()-7, 55f,
        // AxisDependency.LEFT);
    }
}
/**
* @function: 添加传感器数据,用于绘制
* @param arrayList :五个传感器数值，分别 1，2，3，4，5
* */
private void addEntry(ArrayList<Double>arrayList){
    LineData data = lineChart.getData();
    if (data != null&&data.getDataSetCount()==5) {
        for(int i=0;i<5;i++){
            ILineDataSet set = data.getDataSetByIndex(i);
            data.addEntry(new Entry(set.getEntryCount(), arrayList.get(i).floatValue()), i);
        }
        data.notifyDataChanged();
        lineChart.notifyDataSetChanged();
        lineChart.setVisibleXRangeMaximum(120);
        lineChart.moveViewToX(data.getEntryCount());
    }
}
/**
     * @function: 创建linechart中一条折线
     * @param label: 折现legend
     * @Param color:  折线颜色设置
     * */
private LineDataSet createSet(String label,int color) {

    LineDataSet set = new LineDataSet(null, label);
    set.setAxisDependency(YAxis.AxisDependency.LEFT);
    set.setColor(color);
    set.setCircleColor(color);
    set.setLineWidth(2f);
    //set.setCircleRadius(2f);
    set.setFillAlpha(65);
    set.setFillColor(color);
    set.setHighLightColor(Color.rgb(244, 117, 117));
    set.setValueTextColor(color);
    set.setValueTextSize(9f);
    set.setDrawValues(false);
    return set;
}
```

### 2. 图表裁剪

#### .1. [uCrop](https://github.com/Yalantis/uCrop)

> Image Cropping Library for Android

#### .2. **[Android-Image-Cropper](https://github.com/ArthurHub/Android-Image-Cropper)**

> Image Cropping Library for Android, optimized for Camera / Gallery.

#### .3. [SmartCropper](https://github.com/pqpo/SmartCropper)

> 智能图片裁剪框架。自动识别边框，手动调节选区，使用透视变换裁剪并矫正选区；适用于身份证，名片，文档等照片的裁剪

### 3. 图片压缩

#### .1. [Luban](https://github.com/Curzibn/Luban)

> Luban(鲁班)—Android图片压缩工具，仿微信朋友圈压缩策略,可能是最接近微信朋友圈的图片压缩算法

#### .2.  [AdvancedLuban](https://github.com/shaohui10086/AdvancedLuban)

> An Advanced Compress Image Library for Android / 高效、简洁的图片压缩工具库

### 4. 汉字转拼音

#### .1. [**Android_HanziToPinyin_Demo**](https://github.com/AndroidAppCodeDemo/Android_HanziToPinyin_Demo)

#### .2. [**TinyPinyin**](https://github.com/promeG/TinyPinyin)

> 适用于Java和Android的快速、低内存占用的汉字转拼音库。

### 5. JS 和 Native交互

#### .1. [JsBridge](https://github.com/lzyzsd/JsBridge)

> android java and javascript bridge, inspired by wechat webview jsbridge

### 6. 视频播放器

#### .1. **[GSYVideoPlayer](https://github.com/CarGuo/GSYVideoPlayer)**

> 基于IJKplayer实现的丰富多功能播放器

### 7. 录音

#### .1. [AndroidMP3Recorder](https://github.com/Jay-Goo/AndroidMP3Recorder)

> 为Android提供MP3录音功能

#### .2. [AndroidAudioRecorder](https://github.com/adrielcafe/AndroidAudioRecorder)

> A fancy audio recorder lib for Android. Supports WAV format at 48kHz.

#### .3. [recordutil](https://github.com/qssq/recordutil)

> support free record mp3 amr wav aac format可以录制android ios兼容的aac mp3格式切换录制格式也支持体积极少的amr格式，只需要改变工厂方法改变一句话就能实现，和iOS不撕逼录音，这是一个通用解决方案

#### .4. [RecordWaveView](https://github.com/Jay-Goo/RecordWaveView)

> 一款漂亮的波浪录音动画，附带封装好的MP3录音控件

### 8. 录制视频

#### .1. [small-video-record](https://github.com/mabeijianxi/small-video-record)

- 利用FFmpeg视频录制微信小视频与其压缩处理

#### .2. [VideoRecorder](https://github.com/qdrzwd/VideoRecorder)

- android视频录制，模仿微视，支持按下录制、抬起暂停。进度条断点显示

### 9. 数据库

#### .1. [greenDAO](https://github.com/greenrobot/greenDAO)

- greenDAO is a light & fast ORM solution for Android that maps objects to SQLite databases.

### 10. 二维码

#### .1. [BGAQRCode-Android](https://github.com/bingoogolapple/BGAQRCode-Android)

- QRCode 扫描二维码、扫描条形码、相册获取图片后识别、生成带 Logo 二维码、支持微博微信 QQ 二维码扫描样式

#### .2. [QRCodeReaderView](https://github.com/dlazaro66/QRCodeReaderView)

- Modification of ZXING Barcode Scanner project for easy Android QR-Code detection and AR purposes 这个是国外大神写的

#### .3. [AwesomeQRCode](https://github.com/SumiMakito/AwesomeQRCode)

一个优雅的QR 二维码生成器

### 11. 网络请求库：

#### .1. [OkGo](https://github.com/jeasonlzy/okhttp-OkGo)

- 该库是基于 Http 协议，封装了 OkHttp 的网络请求框架，比 Retrofit 更简单易用，支持 RxJava，RxJava2，支持自定义缓存，支持批量断点下载管理和批量上传管理功能

#### .2. [RxHttp](https://github.com/liujingxing/RxHttp/tree/master)

- OkHttp+RxJava 一条链发送请求，自动关闭未完成的请求,RxJava无缝衔接

#### .2. [MultiThreadDownloader](https://github.com/moz1q1/MultiThreadDownloader)

- 基于HttpURLConnection实现的多线程下载器

### 12. Adapter

#### .1. [RecyclerViewAdapter](https://github.com/SheHuan/RecyclerViewAdapter)

- 一个支持RecyclerView加载更多、添加HeaderView的BaseAdapter

#### .2. [BaseRecyclerViewAdapterHelper](https://github.com/CymChad/BaseRecyclerViewAdapterHelper)

- 强大的RecyclerViewAdapter万能适配器

#### .3. [CommonAdapter](https://github.com/qyxxjd/CommonAdapter)

- 一个适用于ListView/GridView/RecyclerView的Adapter库,简化大量重复代码,支持多种布局,可自定义图片加载的实现。

### 13. 图片选择器

#### .1.[RxGalleryFinal](https://github.com/FinalTeam/RxGalleryFinal)

- RxGalleryFinal是一个android图片/视频文件选择器。其支持多选、单选、拍摄和裁剪，主题可自定义，无强制绑定第三方图片加载器。

#### .2.[TakePhoto](https://github.com/crazycodeboy/TakePhoto)

- 一款用于在Android设备上获取照片（拍照或从相册、文件中选择）、裁剪图片、压缩图片的开源工具库

#### .3.[ImageSelector](https://github.com/ioneday/ImageSelector)

- Photo picker library for Android. Support single choice、multi-choice、cropping image and preview image.

#### .4.[RxImagePicker](https://github.com/qingmei2/RxImagePicker)

- RxImagePicker是一个用于Android的响应式图片选择器，它将您的图片选择需求转换为一个接口进行配置，并在任何一个Activity或者Fragment中展示任何样式的图片选择UI。j基于RxJava和注解的方式，提供了知乎和微信主题的支持。

### 14. 文件选择器

#### .1. [MultiType-FilePicker](https://github.com/fishwjy/MultiType-FilePicker)

- 多类型的文件选择器

#### .2.[Android-FilePicker](https://github.com/DroidNinja/Android-FilePicker)

- Photopicker and document picker for android

#### 15. 日历/时间选择

#### .1. [SuperCalendar](https://github.com/MagicMashRoom/SuperCalendar)

- 自定义日历控件 支持左右无限滑动 周月切换 标记日期显示 自定义显示效果跳转到指定日期

### 16. 路由跳转

#### .1. [ARouter](https://github.com/alibaba/ARouter)

- 一个用于帮助 Android App 进行组件化改造的框架 —— 支持模块间的路由、通信、解耦。阿里出品

### 17. 事件总线

#### .1. [EventBus](https://github.com/greenrobot/EventBus)

- EventBus is a publish/subscribe event bus for Android and Java that simplifies communication between Activities, Fragments, Threads, Services, etc. Less code, better quality.

### 18. UI控件

#### .1. [awesome-github-android-ui](https://github.com/opendigg/awesome-github-android-ui)

- UI开源控件库大集合

### 19. Dialog弹窗

FlycoDialog
An Android Dialog Lib simplify customization.
https://github.com/H07000223/FlycoDialog_Master

#### .1. [material-dialogs](https://github.com/afollestad/material-dialogs)

- A beautiful, fluid, and customizable dialogs API

#### .2. [Android-AlertView](https://github.com/Bigkoo/Android-AlertView)

- 仿iOS的AlertViewController

### 20. 依赖注入框架

#### .1. [Butter Knife](https://github.com/JakeWharton/butterknife)

- Bind Android views and callbacks to fields and methods.

#### .2. [Dagger 2](https://github.com/google/dagger)

- A fast dependency injector for Android and Java.

### 21. Json解析

#### .1. [fastjson](https://github.com/alibaba/fastjson)

- A fast JSON parser/generator for Java 阿里出品

### 22. Camera

#### .1. [android-Camera2Basic](https://github.com/googlesamples/android-Camera2Basic)

- Google官方Camera2 API使用范例

#### .2.[CameraView](https://github.com/google/cameraview)

Google提供的CameraView方便简化使用Camera实现拍照录像功能
相关链接：Android相机开发——CameraView源码解析
轻松玩转Camera,修改CameraView 实现自定义拍照分辨率

### 23. 手势/多点触控

#### .1. [android-gesture-detectors](https://github.com/Almeros/android-gesture-detectors)

- Gesture detector framework for multitouch handling on Android, based on Android’s ScaleGestureDetector 老外写的手势监听库，方便监听处理缩放旋转手势等

### 24. Android开源库搜索神器：

#### .1. [看源社区](http://www.see-source.com/androidwidget/list.html)

- 这个主要是一些开源的组件效果等

#### .2.[Android Arsenal](https://android-arsenal.com/)

- 这个主要是搜GitHub上面的Android开源库
  

### Resource

- https://www.techug.com/post/15-android-framework.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_tools/  

