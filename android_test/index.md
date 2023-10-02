# Android_Test


> 1. 本地单元测试(Junit Test)， 本地单元测试是`纯java代码的测试`，`只运行在本地电脑的JVM环境上`，不依赖于Android框架的任何api, 因此执行速度快，效率较高，但是`无法测试Android相关的代码`。
> 2. 仪器化测试(Android Test)，是`针对Android相关代码的测试`，需要运行在`真机设备或模拟器上`，运行速度较慢，但是可以`测试UI的交互以及对设备信息的访问`，得到接近真实的测试结果。

- 其中`testImplementation`添加的依赖就是本地化测试库， `androidTestImplementation` 添加的依赖则是Android环境下的测试库

```json
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    testImplementation 'junit:junit:4.12'
    androidTestImplementation 'com.android.support.test:runner:1.0.2'
    androidTestImplementation 'com.android.support.test.espresso:espresso-core:3.0.2'
}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211203135444006.png)

#### 1.  本地单元测试

##### .1.  相关注解

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211203135714960.png)

##### .2.  断言

- 可以使用断言，可以时用System.out.println() 进行输出查看

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211203135850363.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211203135916205.png)

##### .3. demo

```java
@RunWith(JUnit4.class)
public class SimpleClassTest {
    private SimpleClass simpleClass;

    @Before
    public void setUp() throws Exception {
        simpleClass = new SimpleClass();
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void isTeenager() {
        Assert.assertFalse(simpleClass.isTeenager(20));
        Assert.assertTrue(simpleClass.isTeenager(14));
    }

    @Test
    public void add() {
        Assert.assertEquals(simpleClass.add(3, 2), 5);
        Assert.assertNotEquals(simpleClass.add(3, 2), 4);
    }

    @Test
    public void getNameById() {
        Assert.assertEquals(simpleClass.getNameById(1), "小明");
        Assert.assertEquals(simpleClass.getNameById(2), "小红");
        Assert.assertEquals(simpleClass.getNameById(10), "");
    }
}
```

#### 2. Mockito 框架

```json
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    testImplementation 'org.mockito:mockito-core:2.19.0'
}
```



#### 3. PowerMockito 框架

#### 4. Robolectric 框架

#### 5. Espresso 框架

#### Resource

- https://blog.csdn.net/lyabc123456/article/details/89363721

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android_test/  

