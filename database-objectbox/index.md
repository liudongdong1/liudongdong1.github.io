# database-objectbox


> ObjectBox数据库是用于对象的超快速轻量级数据库，由greenrobot团队开发，与GreenDao，EventBus等师出同门。

### 1. 使用流程

#### 1. Entity

```
@Entity：对象持久化；
@Id：这个对象的主键,默认情况下，id是会被objectbox管理的，也就是自增id。手动管理id需要在注解的时候加上@Id(assignable = true)。当你在自己管理id的时候如果超过long的最大值，objectbox 会报错；id的值不能为负数；当id等于0时objectbox会认为这是一个新的实体对象,因此会新增到数据库表中；
@Index：这个对象中的索引。经常大量进行查询的字段创建索引，用于提高查询性能；
@Transient：某个字段不想被持久化，可以使用此注解，字段将不会保存到数据库；
@NameInDb：数据库中的字段自定义命名；
@ToOne：做一对一的关联注解 ，此外还有一对多，多对多的关联，例如Class的示例；
@ToMany：做一对多的关联注解；
@Backlink：表示反向关联。
```

```java
@Entity
public class User {
    @Id
    public Long id;

    public String userId;

    public String userName;

    public int age;
}
```

#### 2. 代码生成

- 点击AndroidStudio中的`Make Project`（小锤子的图标），objectbox为项目生成类 `MyObjectBox`类，用于初始化生成 `BoxStore` 对象，进行数据库管理。然后，通过BoxStore对象为实体类获得一个 Box 类，Box 对象提供对所有主要函数的访问，比如 put、 get、 remove 和 query。
- `MyObjectBox`: 基于您的实体类生成，MyObjectBox 提供一个构建器为您的应用程序设置一个 BoxStore。
- `BoxStore`: 使用 ObjectBox.BoxStore 的`入口点是到数据库的直接接口`，并管理 Boxes。
- Box: `保存一个盒子并查询实体`。对于每个实体，有一个 Box (由 BoxStore 提供)。

#### 3. 数据库操作

- 新建ObjectBox操作类，用于初始化及数据库管理：

```java
public class ObjectBox {
    private static BoxStore mBoxStore;

    public static void init(Context context) {
        mBoxStore = MyObjectBox.builder()
                .androidContext(context.getApplicationContext())
                .build();
    }

    public static BoxStore get() { return mBoxStore; }
}
```

- 调用ObjectBox类进行初始化，并为User实体类获得一个 Box 类，进行具体操作：

```java
private void initUserBox() {
    ObjectBox.init(this);

    mBoxStore = ObjectBox.get();

    Box<User> mUserBox = mBoxStore.boxFor(User.class);
}
```

#### 4. 增删查改

- 批量添加，调用实体的`Box`对象，调用`put()`方法

```java
private void initUser() {

    //用户ID生成器
    mIdWorker = new SnowflakeIdGenerator(0, 0);

    mUserBox.removeAll();

    mUserList = new ArrayList<>();
    Random random = new Random();
    for (int i = 0; i < 10; i++) {
        User user = new User();
        user.setUserId(String.valueOf(mIdWorker.nextId()));
        // 随机生成汉语名称
        user.setUserName(NameUtils.createRandomZHName(random.nextInt(4) + 1));
        user.setAge(18 + random.nextInt(10));
        mUserList.add(user);
    }

    mUserAdapter = new UserAdapter(mUserList);
    rvUser.setAdapter(mUserAdapter);

    mUserBox.put(mUserList);
}
```

- 查询数据，通过`Box`的`query()`方法可以得到一个`QueryBuilder`对象，该对象可以实现各种查询操作，里面包含`contains()`,`equal()`等方法：

```java
private void queryAllUser() {
    mUserList = mUserBox.query().build().find();
    mUserAdapter.setNewData(mUserList);
    rvUser.smoothScrollToPosition(mUserList.size() - 1);
}
```

- 添加数据，调用实体的`Box`对象，调用`put()`方法即可完成新增操作：

```java
User user = new User();
user.setUserId(String.valueOf(mIdWorker.nextId()));
user.setUserName(NameUtils.createRandomZHName(new Random().nextInt(4) + 1));
user.setAge(18 + new Random().nextInt(10));

// 插入新用户
mUserBox.put(user);
```

- 修改数据，得到要修改的实体类，修改数据，随后调用实体的`Box`对象`put()`方法：

```java
User user = mUserList.get(mUserList.size() - 1);
user.setUserName(NameUtils.createRandomZHName(new Random().nextInt(4) + 1));

//更新最末用户
mUserBox.put(user);
```

- 删除数据，调用实体的`Box`对象`remove()`方法：

```java
User user = mUserList.get(mUserList.size() - 1);
//删除最末用户
mUserBox.remove(user);
```

- 异步操作

```java
boxStore.callInTxAsync(() -> {
    Box<User> box = boxStore.boxFor(User.class);
    String name = box.get(userId).name;
    box.remove(userId);
    return text;
}, (result, error) -> {
    if (error != null) {
        System.out.println("Failed to remove user with id " + userId);
    } else {
        System.out.println("Removed user with name: " + result);
    }
});
```

### Resource

- https://docs.objectbox.io/transactions
- https://github.com/objectbox/objectbox-examples  
- todo?  为什么objectBox 比realm等第三方库库更加优势，为什么轻量快捷
- todo？ objectBox 怎么和Rx模式结合的，具体应用中是怎么进行使用的

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/database-objectbox/  

