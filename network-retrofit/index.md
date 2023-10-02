# network-retrofit


> `Retrofit`是`Square`公司开发的一款针对`Android`和`java`的网络请求框架，遵循`Restful`设计风格，底层基于`OkHttp`。
>
> 1. 支持同步/异步网络请求
> 2. 支持多种数据的解析&序列化格式(`Gson`、`json`、`XML`等等)
> 3. 通过注解配置网络请求参数
> 4. 提供对`Rxjava`的支持高度解耦,使用方便

### 1. 使用案例

```java
public interface GetRequestInterface {
    /**
     * 通过get（）方法获取图片的请求接口
     * GET注解中的参数值"2019-06-29-121904.png"和Retrofit的base url拼接在一起就是本次请求的最终地址
     *
     * @return
     */
    @GET("2019-06-29-121904.png")
    Call<ResponseBody> getPictureCall();
}
```

```java
public void retrofitGet() {
    /*
         创建Retrofit对象，这里设置了baseUrl，注意我们在声明网络配置接口GetRequestInterface的时候在GET注解中也声明了一个Url，
         我们将会这里的baseUrl和GET注解中设置的Url拼接之后就可以形成最终网络请求实际访问的url
         */
    Retrofit retrofit = new Retrofit.Builder().baseUrl("http://picture-pool.oss-cn-beijing.aliyuncs.com/").build();
    //创建网络请求配置的接口实例
    GetRequestInterface getRequestInterface = retrofit.create(GetRequestInterface.class);
    //调用我们声明的getPictureCall（）方法创建Call对象
    Call<ResponseBody> requestBodyCall = getRequestInterface.getPictureCall();
    //使用requestBodyCall发起异步网络请求
    requestBodyCall.enqueue(new Callback<ResponseBody>() {
        @Override
        public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
            //将网络请求返回的数据解析为图片并展示到界面
            ResponseBody body = response.body();
            InputStream inputStream = body.byteStream();
            Drawable drawable = Drawable.createFromStream(inputStream, "pic.png");
            imageView.setBackground(drawable);
            Log.e(TAG, "网络请求成功");
        }
        @Override
        public void onFailure(Call<ResponseBody> call, Throwable t) {
            Log.e(TAG, "网络请求失败，失败原因：" + t.getMessage());
        }
    });
}
```

### 2. 注解介绍

#### 1. 请求参数

- 在`Retrofit`中可以使用五种请求方法：`GET`,`POST`,`PUT`,`DELETE`,`HEAD`，而相对`URL`直接在对应注解中填写即可

```java
@GET("users/list")//表示以GET请求方法发起网络请求，子URL为：users/list
@GET("users/list?sort=desc")  //在URL中直接指定查询的参数
```

- 占位符指的是`URL`中由`{}`包裹的`String`类型字符串，而参数必须使用`@Path`注解进行修饰，而且也只能是`String`，这样，`@Path`修饰的变量会动态对应到`URL`中

```java
@GET("group/{id}/users")
Call<List<User>> groupList(@Path("id") int groupId);

@GET("group/{id}/users")
Call<List<User>> groupList(@Path("id") int groupId, @Query("sort") String sort);

@GET("group/{id}/users")
Call<List<User>> groupList(@Path("id") int groupId, @QueryMap Map<String, String> options);
```

```java
//通过@Body注解将一个对象以 Post方式发送给服务器
@POST("users/new")
Call<User> createUser(@Body User user);  

```

当我们想发送`form-encoded`的数据时，可以使用`@FormUrlEncode`注解，将每个键值对中的键使用`@Filed`注解来进行说明，而值设置为随后跟随的对象：

```java
@FormUrlEncoded
@POST("user/edit")
Call<User> updateUser(@Field("first_name") String first, @Field("last_name") String last);
```

当我们想发送`Multipart`的数据时（适用于文件发送的场景），可以使用`@Multipart`注解，时使用，将每个键值对中的键使用`@Part`注解来进行说明，而值设置为随后跟随的对象：

```java
@Multipart
@PUT("user/photo")
Call<User> updateUser(@Part("photo") RequestBody photo, @Part("description") RequestB
```

```java
@Headers({
    "Accept: application/vnd.github.v3.full+json",
    "User-Agent: Retrofit-Sample-App"
})
@GET("users/{username}")
Call<User> getUser(@Path("username") String username);
```

#### 2. 转换器

`Retrofit`只能反序列化`OkHttp`中的`ResponseBody`类型的返回值，通过添加不同类型的转换器可以使`Retrofit`拥有反序列化其他主流序列的能力

- `Gson`：`com.squareup.retrofit2:converter-gson`
- `Jackson`：`com.squareup.retrofit2:converter-jackson`
- `Moshi`：`com.squareup.retrofit2:converter-moshi`
- `Protobuf`：`com.squareup.retrofit2:converter-protobuf`
- `Wire`：`com.squareup.retrofit2:converter-wire`
- `Simple XML`：`com.squareup.retrofit2:converter-simplexml`

```java
Retrofit retrofit = new Retrofit.Builder()
    .baseUrl("https://api.github.com")
    .addConverterFactory(GsonConverterFactory.create())
    .build();

GitHubService service =retrofit.create(GitHubService.class);
```

#### 3. 网络适配器

```
Retrofit`支持多种网络请求适配器方式：`guava`、`Java8`和`rxjava
```

使用时如使用的是 `Android` 默认的 `CallAdapter`，则不需要添加网络请求适配器的依赖，否则则需要按照需求进行添加 `Retrofit` 提供的 `CallAdapter`，对应依赖为：

1. `guava` ：`com.squareup.retrofit2:adapter-guava:x.x.x`
2. `Java8`：`com.squareup.retrofit2:adapter-java8:x.x.x`
3. `rxjava`：`com.squareup.retrofit2:adapter-rxjava:x.x.x`

```java
public interface GetRequestInterfaceWithRxJava {
    /**
     * 通过get（）方法获取图片的请求接口
     * GET注解中的参数值"2019-06-29-121904.png"和Retrofit的base url拼接在一起就是本次请求的最终地址
     *
     设置返回值类型为Observable的
     * @return
     */
    @GET("2019-06-29-121904.png")
    Observable<ResponseBody> getPictureCall();
}
```

```java
public void retrofitGetWithRxJava() {
    /*
         创建Retrofit对象，这里设置了baseUrl，注意我们在声明网络配置接口GetRequestInterface的时候在GET注解中也声明了一个Url，
         我们将会这里的baseUrl和GET注解中设置的Url拼接之后就可以形成最终网络请求实际访问的url
         */
    Retrofit retrofit = new Retrofit.Builder().baseUrl("http://picture-pool.oss-cn-beijing.aliyuncs.com/").addCallAdapterFactory(RxJava2CallAdapterFactory.create()).build();

    //创建网络请求配置的接口实例
    GetRequestInterfaceWithRxJava getRequestInterfaceWithRxJava = retrofit.create(GetRequestInterfaceWithRxJava.class);
    //调用我们声明的getPictureCall（）方法创建Call对象
    Observable<ResponseBody> requestObservable = getRequestInterfaceWithRxJava.getPictureCall();
    requestObservable.subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new Observer<ResponseBody>() {
        @Override
        public void onSubscribe(Disposable d) {
            Log.e(TAG, "onSubscribe");
        }

        @Override
        public void onNext(ResponseBody body) {
            //将网络请求返回的数据解析为图片并展示到界面
            InputStream inputStream = body.byteStream();
            Drawable drawable = Drawable.createFromStream(inputStream, "pic.png");
            imageView.setBackground(drawable);
            Log.e(TAG, "网络请求成功");

        }

        @Override
        public void onError(Throwable e) {
            Log.e(TAG, "onError:" + e.getMessage());
        }

        @Override
        public void onComplete() {
            Log.e(TAG, "onComplete");
        }
    });


}
```

### Resource

- **https://blog.csdn.net/weixin_36709064/article/details/82468549** 注解介绍
- https://blog.csdn.net/qq_36982160/article/details/94201257

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/network-retrofit/  

