# OkHttp


> OkHttp是一个优秀的网络请求框架,可能一说到网络请求框架,可能很多人都会想到`volley`,`volley`是一个Google提供的网络请求框架,我的博客里也有一篇专门介绍`volley`的博客,博客地址在此**[Android网络请求 ------ Volley的使用](https://link.juejin.cn/?target=http%3A%2F%2Fblog.csdn.net%2Fbingjianit%2Farticle%2Fdetails%2F52387916)** 那么既然Google提供了网络请求的框架,我们为什么还要使用`OkHttp`呢,原来是`volley`是要依靠`HttpCient`的,而Google在`Android6.0`的SDK中去掉了`HttpCient`,所以`OkHttp`就开始越来越受大家的欢迎.

### 1. get 请求

1. 获取OkHttpClient对象
2. 设置请求request
3. 封装call
4. 异步调用,并设置回调函数

```java
public void get(String url){
    // 1 获取OkHttpClient对象
    OkHttpClient client = new OkHttpClient();
    // 2设置请求
    Request request = new Request.Builder()
        .get()
        .url(url)
        .build();
    // 3封装call
    Call call = client.newCall(request);
    // 4异步调用,并设置回调函数
    call.enqueue(new Callback() {
        @Override
        public void onFailure(Call call, IOException e) {
            Toast.makeText(OkHttpActivity.this, "get failed", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onResponse(Call call, final Response response) throws IOException {
            final String res = response.body().string();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    contentTv.setText(res);
                }
            });
        }
    });
    //同步调用,返回Response,会抛出IO异常
    //Response response = call.execute();
}
```

### 2.  Post 请求

```java
OkHttpClient client = new OkHttpClient();
//构建FormBody,传入参数
FormBody formBody = new FormBody.Builder()
                .add("username", "admin")
                .add("password", "admin")
                .build();

//post方法需要传入的是一个RequestBody对象,FormBody是RequestBody的子类
//RequestBody requestBody = RequestBody.create(MediaType.parse("text/plain;charset=utf-8"), "{username:admin;password:admin}");

//构建Request,将FormBody作为Post方法的参数传入
final Request request = new Request.Builder()
                .url("http://www.jianshu.com/")
                .post(formBody)
                .build();
//将Request封装为Call
Call call = client.newCall(request);
//调用请求,重写回调方法
call.enqueue(new Callback() {
    @Override
    public void onFailure(Call call, IOException e) {
        Toast.makeText(OkHttpActivity.this, "Post Failed", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onResponse(Call call, Response response) throws IOException {
        final String res = response.body().string();
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                contentTv.setText(res);
            }
        });
    }
});
```

### 3. Post 上传文件&提交表单

```java
File file = new File(Environment.getExternalStorageDirectory(), "1.png");
if (!file.exists()){
    Toast.makeText(this, "文件不存在", Toast.LENGTH_SHORT).show();
}else{
    RequestBody requestBody2 = RequestBody.create(MediaType.parse("application/octet-stream"), file);
}
```

```java
File file = new File(Environment.getExternalStorageDirectory(), "1.png");
if (!file.exists()){
    Toast.makeText(this, "文件不存在", Toast.LENGTH_SHORT).show();
    return;
}
RequestBody muiltipartBody = new MultipartBody.Builder()
        //一定要设置这句
        .setType(MultipartBody.FORM)
        .addFormDataPart("username", "admin")//
        .addFormDataPart("password", "admin")//
        .addFormDataPart("myfile", "1.png", RequestBody.create(MediaType.parse("application/octet-stream"), file))
        .build();
```

### 4. get 下载文件

```java
OkHttpClient client = new OkHttpClient();
    final Request request = new Request.Builder()
            .get()
            .url("https://www.baidu.com/img/bd_logo1.png")
            .build();
    Call call = client.newCall(request);
    call.enqueue(new Callback() {
        @Override
        public void onFailure(Call call, IOException e) {
            Log.e("moer", "onFailure: ");;
        }

        @Override
        public void onResponse(Call call, Response response) throws IOException {
            //拿到字节流
            InputStream is = response.body().byteStream();

            int len = 0;
            File file  = new File(Environment.getExternalStorageDirectory(), "n.png");
            FileOutputStream fos = new FileOutputStream(file);
            byte[] buf = new byte[128];

            while ((len = is.read(buf)) != -1){
                fos.write(buf, 0, len);
            }

            fos.flush();
            //关闭流
            fos.close();
            is.close();
        }
    });
}
```

```java
@Override
public void onResponse(Call call, Response response) throws IOException {
    InputStream is = response.body().byteStream();

    final Bitmap bitmap = BitmapFactory.decodeStream(is);
    runOnUiThread(new Runnable() {
        @Override
        public void run() {
            imageView.setImageBitmap(bitmap);
        }
    });

    is.close();
}
```

### Resource

- https://cloud.tencent.com/developer/article/1910477
- https://zhuanlan.zhihu.com/p/337000646

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/network-okhttp/  

