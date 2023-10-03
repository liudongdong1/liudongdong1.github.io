# database-greeDaos


> for new apps we recommend [ObjectBox](https://objectbox.io/), a new object-oriented database that is much faster than SQLite and easier to use. For existing apps based on greenDAO we offer [DaoCompat](https://greenrobot.org/greendao/documentation/objectbox-compat/) for an easy switch (see also the [announcement](https://greenrobot.org/release/daocompat-greendao-on-objectbox/)).

### 1. GreeDAO

> 一个将对象映射到SQLite数据库中的轻量且快速的ORM（object / relational mapping）解决方案。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/webp.webp)

- **DaoMaster**：Dao中的管理者。它保存了`sqlitedatebase对象`以及`操作DAO classes（`注意：不是对象）。其提供了一些`创建和删除table的静态方法`，其`内部类OpenHelper和DevOpenHelper实现了SQLiteOpenHelper，并创建数据库的框架`。
- **DaoSession**：会话层。`操作具体的DAO对象`（注意：是对象），比如各种getter方法。
- **Daos**：实际生成的某某DAO类，通常对应具体的java类，比如NoteDao等。其有更多的权限和方法来`操作数据库元素`。
- **Entities**：持久的实体对象。通常代表了一个数据库row的标准java properties。

```java
DBHelper devOpenHelper = new DBHelper(this);  
DaoMaster daoMaster = new DaoMaster(devOpenHelper.getWritableDb());  
DaoSession daoSession = daoMaster.newSession();  
userDao = daoSession.getUserDao(); 
```

#### 1. 添加依赖

```gradle
// In your root build.gradle file:
buildscript {
    repositories {
        jcenter()
        mavenCentral() // add repository
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:3.0.0'
        classpath 'org.greenrobot:greendao-gradle-plugin:3.2.2' // add plugin
    }
}

// In your app projects build.gradle file:
apply plugin: 'com.android.application'
apply plugin: 'org.greenrobot.greendao' // apply plugin
 
dependencies {
    compile 'org.greenrobot:greendao:3.2.2' // add library
}
```

#### 2. 设置数据库版本信息

```gradle
//在module的build.gradle文件添加：
greendao {
    schemaVersion 1
}
```

#### 3. 编写实体类

```
@Entity 　表明这个实体类会在数据库中生成一个与之相对应的表。 
@Id 　对应数据表中的 Id 字段，有了解数据库的话，是一条数据的唯一标识。 
@Property(nameInDb = “STUDENTNUM”) 　表名这个属性对应数据表中的 STUDENTNUM 字段。 
@Property 　可以自定义字段名，注意外键不能使用该属性 
@NotNull 　该属性值不能为空 
@Transient 　该属性不会被存入数据库中 
@Unique 　表名该属性在数据库中只能有唯一值
@ToMany 一对多
@ToOne 一对一
@OrderBy 排序
```

```java
@Entity
public class Player {
    @Id
    private Long id;
    @Unique
    private String name;
    private Integer age;
    @Convert(converter = NoteTypeConverter.class, columnType = String.class)   //自定义类型转化
    private NoteType type;
}
```

```java
public class NoteTypeConverter implements PropertyConverter<NoteType, String> {
    @Override
    public NoteType convertToEntityProperty(String databaseValue) {
        return NoteType.valueOf(databaseValue);
    }

    @Override
    public String convertToDatabaseValue(NoteType entityProperty) {
        return entityProperty.name();
    }
}
```

- build之后会生成三个文件 
  - DaoMaster.java
  - DaoSession.java
  - PlayerDao.java

#### 4. 初始化数据库

```java
 /**
     * 数据库名称
     */
    private static final String DATABASE_NAME = "players.db";
    private DaoSession mDaoSession;  //全局保存一个DaoSession对象，用来对数据库进行操作。
    /**
     * 初始化DaoSession
     * 即获取一个全局的DaoSession实例
     *可以使用一个单例类单独管理这个对象
     */
    private void initDaoSession() {
        DaoMaster.OpenHelper openHelper = new DaoMaster.DevOpenHelper(
                mContext.getApplicationContext(), DATABASE_NAME, null);
        DaoMaster daoMaster = new DaoMaster(openHelper.getWritableDatabase());
        mDaoSession = daoMaster.newSession();
    }
```

##### 1. 单例管理&数据库升级

```java
/**
 * GreenDaoManager单例类，用来集中操作数据库
 */

public class GreenDaoManager {
    private static final String TAG = "GreenDaoManager";

    private static final String DATABASE_NAME = "players.db";
    /**
     * 全局保持一个DaoSession
     */
    private DaoSession daoSession;

    private boolean isInited;

    private static final class GreenDaoManagerHolder {
        private static final GreenDaoManager sInstance = new GreenDaoManager();
    }

    public static GreenDaoManager getInstance() {
        return GreenDaoManagerHolder.sInstance;
    }

    private GreenDaoManager() {

    }

    /**
     * 初始化DaoSession
     *
     * @param context
     */
    public void init(Context context) {
        if (!isInited) {
            /**
             * 使用自定义的OpenHelper对象
             */
            MigrationHelper.DEBUG = true;
            MyOpenHelper openHelper = new MyOpenHelper(
                context.getApplicationContext(), DATABASE_NAME, null);
            DaoMaster daoMaster = new DaoMaster(openHelper.getWritableDatabase());
            daoSession = daoMaster.newSession();
            isInited = true;
        }
    }

    public DaoSession getDaoSession() {
        return daoSession;
    }

    /**
     * 定义一个MySQLiteOpenHelper类，用来处理数据库升级
     */
    static class MyOpenHelper extends DaoMaster.OpenHelper {

        public MyOpenHelper(Context context, String name, SQLiteDatabase.CursorFactory factory) {
            super(context, name, factory);
        }

        @Override
        public void onUpgrade(Database db, int oldVersion, int newVersion) {
            Log.d(TAG, "onUpgrade: old: " + oldVersion + ", new: " + newVersion);
            if (oldVersion <= 1) {
                MigrationHelper.migrate(db, new MigrationHelper.ReCreateAllTableListener() {
                    @Override
                    public void onCreateAllTables(Database db, boolean ifNotExists) {
                        DaoMaster.createAllTables(db, ifNotExists);
                    }

                    @Override
                    public void onDropAllTables(Database db, boolean ifExists) {
                        DaoMaster.dropAllTables(db, ifExists);
                    }
                }, PlayerDao.class);

            }

        }
    }
}
```

#### 5. 数据库操作

```java
/**
     * 插入一条数据
     *
     * @param player
     */
private void insertData(Player player) {
    PlayerDao playerDao = mDaoSession.getPlayerDao();
    playerDao.insert(player);
}
```

```java
/**
     * 根据id删除一条数据
     *
     * @param id
     */
private void deleteData(long id) {
    PlayerDao playerDao = mDaoSession.getPlayerDao();
    playerDao.deleteByKey(id);
}
```

```java
/**
     * 更新一条数据
     * 更新年龄
     *
     * @param id
     * @param age
     */
private void updateData(long id, int age) {
    Log.d(TAG, "updateData: id: " + id + ", age: " + age);
    PlayerDao playerDao = mDaoSession.getPlayerDao();
    Player player = playerDao.queryBuilder()
        .where(PlayerDao.Properties.Id.eq(id))
        .build()
        .unique();
    player.setAge(age);
    playerDao.update(player);
}
```

```java
/**
     * 获取全部数据，按照Id升序排列
     *
     * @return 数据列表
     */
private List<Player> getAllData() {
    PlayerDao playerDao = mDaoSession.getPlayerDao();
    return playerDao.queryBuilder()
        .orderAsc(PlayerDao.Properties.Id)
        .build()
        .list();
}
```

- loadAll()：查询所有数据。
- queryRaw()：根据条件查询。
- queryBuilder() : 方便查询的创建。

### Resource

- https://greenrobot.org/greendao/
- https://github.com/YoungBear/GreenDAOLean
- https://blog.csdn.net/u012585964/article/details/52460456
- https://www.cnblogs.com/jqnl/p/13900433.html  使用泛型进行方法封装
- ormlite-android database： https://github.com/j256/ormlite-android

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/database-greedaos/  

