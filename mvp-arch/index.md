# Mvp-arch


### Architecture Blueprint

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/mvp-arch.png)

![MvpApp](https://gitee.com/github-25970295/picture2023/raw/master/MvpApp.png)

### Project Structure

![Structure](https://gitee.com/github-25970295/blogimgv2022/raw/master/mvp-project-structure-diagram.png)

1. **data**: It contains all the data accessing and manipulating components.
2. **di**: Dependency providing classes using Dagger2.  
3. **ui**: View classes along with their corresponding Presenters.
4. **service**: Services for the application.
5. **utils**: Utility classes.

### 数据层面

- 通过AppDataManager 提供数据库，网络数据，本地Preference数据操作

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221215153425170.png)

#### sqlite数据库

- 使用greenDao 第三方插件实现基本数据操作

![Option](https://gitee.com/github-25970295/blogimgv2022/raw/master/Option.png)

#### 网络操作

- 使用Rx2AndroidNetworking第三方库 封装网络数据请求
- 基本数据包括 User信息，Blog信息，Repo信息，并且每一个信息都有对应地封装地Response类中，其中ApiHeader 存放网络请求相关的header信息

![Package model](https://gitee.com/github-25970295/blogimgv2022/raw/master/Package%20model.png)

#### 本地Preferences文件

![image-20221215153126837](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221215153126837.png)

### UI层面

1. **View**: It is the part of the application which renders the UI and receives interactions from the user. Activity, Fragment, and CustomView constitute this part.
2. **MvpView**: It is an` interface, that is implemented by the View`. It `contains methods that are exposed to its Presenter for the communication`.
3. **Presenter**: It is the decision-making counterpart of the View and is a pure java class, with no access to Android APIs. It `receives the user interactions passed on from its View and then takes the decision based on the business logic`, finally instructing the View to perform specific actions. It also `communicates with the DataManager` for any data it needs to perform business logic.
4. **MvpPresenter**: It is an interface, that is implemented by the Presenter. It contains `methods that are exposed to its View for the communication`.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/Package%20base.png)

#### 登录界面

LoginActivity 中通过 @Inject 注入LoginMvpPresenter<LoginMvpView> mPresenter; 实例，并在onCreate中进行attach，在onDestroy进行dettach

```java
public class LoginActivity extends BaseActivity implements LoginMvpView {

    @Inject
    LoginMvpPresenter<LoginMvpView> mPresenter;
     @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        getActivityComponent().inject(this);

        setUnBinder(ButterKnife.bind(this));

        mPresenter.onAttach(LoginActivity.this);  //  在程序启动的时候 attach
    }
    @Override
    protected void onDestroy() {
        mPresenter.onDetach();   // 在销毁的时候 进行presenter detach操作
        super.onDestroy();
    }
}
```

```java
public interface LoginMvpView extends MvpView {

    void openMainActivity();
}

@PerActivity
public interface LoginMvpPresenter<V extends LoginMvpView> extends MvpPresenter<V> {

    void onServerLoginClick(String email, String password);

    void onGoogleLoginClick();

    void onFacebookLoginClick();

}
public class LoginPresenter<V extends LoginMvpView> extends BasePresenter<V>
        implements LoginMvpPresenter<V> {

    private static final String TAG = "LoginPresenter";

    @Inject
    public LoginPresenter(DataManager dataManager,
                          SchedulerProvider schedulerProvider,
                          CompositeDisposable compositeDisposable) {
        super(dataManager, schedulerProvider, compositeDisposable);
    }
    //...
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221215155241715.png)



- https://github.com/liudongdong1/android-mvp-architecture



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mvp-arch/  

