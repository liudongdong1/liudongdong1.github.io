# Android_存储


> SQLite是Android内置的一个小型、关系型、属于文本型的数据库。 Android提供了对 SQLite数据库的完全支持，应用程序中的任何类都可以通过名称来访问任何的数据库，但是应用程序之外的就不能访问。
>
> Android中，通过**SQLiteOpenHelper类**来实现对SQLite数据库的操作。

### 1. **SharedPreferences**

适用范围**：**保存少量的数据，且这些数据的格式非常简单：字符串型、基本类型的值。比如应用程序的各种配置信息（如是否打开音效、是否使用震动效果、小游戏的玩家积分等），解锁口 令密码等

 核心原理**：**保存基于XML文件存储的key-value键值对数据，通常用来存储一些简单的配置信息。通过DDMS的File Explorer面板，展开文件浏览树,很明显SharedPreferences数据总是存储在/data/data/<package name>/shared_prefs目录下。SharedPreferences对象本身只能获取数据而不支持存储和修改,存储修改是通过SharedPreferences.edit()获取的内部接口Editor对象实现。 SharedPreferences本身是一 个接口，程序无法直接创建SharedPreferences实例，只能通过Context提供的getSharedPreferences(String name, int mode)方法来获取SharedPreferences实例，该方法中name表示要操作的xml文件名，第二个参数具体如下：

​         ***Context.MODE_PRIVATE\***: 指定该SharedPreferences数据只能被本应用程序读、写。

​         ***Context.MODE_WORLD_READABLE\***: 指定该SharedPreferences数据能被其他应用程序读，但不能写。

​         ***Context.MODE_WORLD_WRITEABLE***: 指定该SharedPreferences数据能被其他应用程序读，写

Editor有如下主要重要方法：

​         ***SharedPreferences.Editor clear():\***清空SharedPreferences里所有数据

​         ***SharedPreferences.Editor putXxx(String key , xxx value):\*** 向SharedPreferences存入指定key对应的数据，其中xxx 可以是boolean,float,int等各种基本类型据

​         ***SharedPreferences.Editor remove():\*** 删除SharedPreferences中指定key对应的数据项

​         ***boolean commit():\*** 当Editor编辑完成后，使用该方法提交修改

```java
switch(v.getId()){
    case R.id.btnSet:
        //步骤1：获取输入值
        String code = txtCode.getText().toString().trim();
        //步骤2-1：创建一个SharedPreferences.Editor接口对象，lock表示要写入的XML文件名，MODE_WORLD_WRITEABLE写操作
        SharedPreferences.Editor editor = getSharedPreferences("lock", MODE_WORLD_WRITEABLE).edit();
        //步骤2-2：将获取过来的值放入文件
        editor.putString("code", code);
        //步骤3：提交
        editor.commit();
        Toast.makeText(getApplicationContext(), "口令设置成功", Toast.LENGTH_LONG).show();
        break;
    case R.id.btnGet:
        //-- 如果使用其他应用程序的context
        // Context pvCount = createPackageContext("com.tony.app", Context.CONTEXT_IGNORE_SECURITY);这里的com.tony.app就是其他程序的包名
        //  SharedPreferences read = pvCount.getSharedPreferences("lock", Context.MODE_WORLD_READABLE);
        //步骤1：创建一个SharedPreferences接口对象
        SharedPreferences read = getSharedPreferences("lock", MODE_WORLD_READABLE);
        //步骤2：获取文件中的值
        String value = read.getString("code", "");
        Toast.makeText(getApplicationContext(), "口令为："+value, Toast.LENGTH_LONG).show();

        break;

}
```

### 2. File 文件操作

 核心原理: Context提供了两个方法来打开数据文件里的文件IO流 FileInputStream `openFileInput`(String name); `FileOutputStream`(String name , int mode),这两个方法第一个参数 用于指定文件名，第二个参数指定打开文件的模式。具体有以下值可选：

​       ***MODE_PRIVATE***：为默认操作模式，代表该文件是私有数据，只能被应用本身访问，在该模式下，写入的内容会覆盖原文件的内容，如果想把新写入的内容追加到原文件中。可  以使用Context.MODE_APPEND

​       ***MODE_APPEND\***：模式会检查文件是否存在，存在就往文件追加内容，否则就创建新文件。

​       ***MODE_WORLD_READABLE***：表示当前文件可以被其他应用读取；

​       ***MODE_WORLD_WRITEABLE\***：表示当前文件可以被其他应用写入。

 除此之外，Context还提供了如下几个重要的方法：

​       **getDir(String name , int mode)**:在应用程序的数据文件夹下获取或者创建name对应的子目录

​       **File getFilesDir()**:获取该应用程序的数据文件夹得绝对路径

​       **String[] fileList()**:返回该应用数据文件夹的全部文件    

```java
public String read() {
    try {
        //文件目录： /data/data/cn.tony.app/files/message.txt
        FileInputStream inStream = this.openFileInput("message.txt"); 
        byte[] buffer = new byte[1024];
        int hasRead = 0;
        StringBuilder sb = new StringBuilder();
        while ((hasRead = inStream.read(buffer)) != -1) {
            sb.append(new String(buffer, 0, hasRead));
        }

        inStream.close();
        return sb.toString();
    } catch (Exception e) {
        e.printStackTrace();
    } 
    return null;
}

public void write(String msg){
    // 步骤1：获取输入值
    if(msg == null) return;
    try {
        // 步骤2:创建一个FileOutputStream对象,MODE_APPEND追加模式
        FileOutputStream fos = openFileOutput("message.txt",
                                              MODE_APPEND);
        // 步骤3：将获取过来的值放入文件
        fos.write(msg.getBytes());
        // 步骤4：关闭数据流
        fos.close();
    } catch (Exception e) {
        e.printStackTrace();
    }
}
```

#### .2. SD 卡

- 调用Environment的getExternalStorageState()方法判断手机上是否插了sd卡,且应用程序具有读写SD卡的权限，如下代码将返回true， Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)
- 调用Environment.getExternalStorageDirectory()方法来获取外部存储器，也就是SD卡的目录,或者使用 "/mnt/sdcard/" 目录
- 使用IO流操作SD卡上的文件 

```java
// 文件写操作函数
private void write(String content) {
    if (Environment.getExternalStorageState().equals(
        Environment.MEDIA_MOUNTED)) { // 如果sdcard存在
        File file = new File(Environment.getExternalStorageDirectory()
                             .toString()
                             + File.separator
                             + DIR
                             + File.separator
                             + FILENAME); // 定义File类对象
        if (!file.getParentFile().exists()) { // 父文件夹不存在
            file.getParentFile().mkdirs(); // 创建文件夹
        }
        PrintStream out = null; // 打印流对象用于输出
        try {
            out = new PrintStream(new FileOutputStream(file, true)); // 追加文件
            out.println(content);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (out != null) {
                out.close(); // 关闭打印流
            }
        }
    } else { // SDCard不存在，使用Toast提示用户
        Toast.makeText(this, "保存失败，SD卡不存在！", Toast.LENGTH_LONG).show();
    }
}

// 文件读操作函数
private String read() {
    if (Environment.getExternalStorageState().equals(
        Environment.MEDIA_MOUNTED)) { // 如果sdcard存在
        File file = new File(Environment.getExternalStorageDirectory()
                             .toString()
                             + File.separator
                             + DIR
                             + File.separator
                             + FILENAME); // 定义File类对象
        if (!file.getParentFile().exists()) { // 父文件夹不存在
            file.getParentFile().mkdirs(); // 创建文件夹
        }
        Scanner scan = null; // 扫描输入
        StringBuilder sb = new StringBuilder();
        try {
            scan = new Scanner(new FileInputStream(file)); // 实例化Scanner
            while (scan.hasNext()) { // 循环读取
                sb.append(scan.next() + "\n"); // 设置文本
            }
            return sb.toString();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (scan != null) {
                scan.close(); // 关闭打印流
            }
        }
    } else { // SDCard不存在，使用Toast提示用户
        Toast.makeText(this, "读取失败，SD卡不存在！", Toast.LENGTH_LONG).show();
    }
    return null;
}
```

### 3. SQLiteOpenHelper

#### .1. 基本概念

- 定义：SQLiteOpenHelper是一个辅助类
- 作用：管理数据库（`创建、增、修、删`） &` 版本的控制`。
- 使用过程：通过`创建子类继承SQLiteOpenHelper类`，实现它的一些方法来对数据库进行操作。 在实际开发中，为了能够更好的管理和维护数据库，我们会`封装一个继承自SQLiteOpenHelper类的数据库操作类`，然后以`这个类为基础，再封装我们的业务逻辑方法`。
- // 调用getReadableDatabase()或getWritableDatabase()才算`真正创建或打开数据库`

| 方法名                | 作用                               | 备注                                                         |
| --------------------- | ---------------------------------- | ------------------------------------------------------------ |
| onCreate()            | `创建数据库`                       | 创建数据库时自动调用                                         |
| onUpgrade()           | `升级数据库`                       | 数据库版本发生变化的时候回调（取决于数据库版本),只要这个版本高于之前的版本, 就会触发这个onUpgrade()方法 |
| close()               | `关闭所有打开的数据库对象`         |                                                              |
| execSQL()             | 可进行增删改操作, 不能进行查询操作 |                                                              |
| query()、rawQuery()   | 查询数据库                         |                                                              |
| insert()              | 插入数据                           |                                                              |
| delete()              | 删除数据                           |                                                              |
| getWritableDatabase() | 创建或打开可以`读/写`的数据库      | 通过返回的SQLiteDatabase对象对数据库进行操作                 |
| getReadableDatabase() | 创建或打开`可读`的数据库           | 同上                                                         |

#### .2. 实例代码

```java
public class SQLiteHelper extends SQLiteOpenHelper{
    private SQLiteDatabase myDataBase;
    private final Context myContext;
    private final String Tag="SQLiteHepler";
    //数据库版本号
    private static Integer Version = 1;
    /**
     * Constructor
     * Takes and keeps a reference of the passed context in order to access to the application assets and resources.
     * @param context
     */
    public SQLiteHelper(Context context) {
        super(context, Constant.DB_NAME, null, 1);
        this.myContext = context;
    }

    public void openDataBase()  {
        String dbpath = myContext.getDatabasePath(Constant.DB_NAME).getPath();
        if (myDataBase != null && myDataBase.isOpen())
        {
            return;
        }
        myDataBase = SQLiteDatabase.openDatabase(dbpath,null,SQLiteDatabase.OPEN_READWRITE);
    }

    public Cursor rawQuery(String sql) {
        return myDataBase.rawQuery(sql, null);
    }
    @Override
    public synchronized void close() {

        if(myDataBase != null)
            myDataBase.close();

        super.close();

    }
    /**
     * @function: 创建数据库
     * */
    @Override
    public void onCreate(SQLiteDatabase db) {
        boolean dbExist = checkDataBase();
        if(dbExist){
            Log.i(Tag, "onCreate: do nothing - database already exist");
        }else {
            db.execSQL(Constant.SQL);
            Log.i(Tag, "onCreate: 创建数据库完成");
        }
    }
    /**
     * @function: 升级数据库
     * */
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
		Log.i(Tag,"更新数据库版本为:"+newVersion);
    }

    /**
     * @function: Creates a empty database on the system and rewrites it with your own database.
     * */
    public void createDataBaseByCopy() throws IOException {
        boolean dbExist = checkDataBase();
        if(dbExist){
            //do nothing - database already exist
        }else{
            //By calling this method and empty database will be created into the default system path
            //of your application so we are gonna be able to overwrite that database with our database.
            this.getReadableDatabase();  //创建或打开可读的数据库
            try {
                Log.e("Check DB", "copied");
                copyDataBase();
            } catch (IOException e) {
                throw new Error("Error copying database");
            }
        }

    }

    /**
     * Check if the database already exist to avoid re-copying the file each time you open the application.
     * @return true if it exists, false if it doesn't
     */
    private boolean checkDataBase(){
        SQLiteDatabase checkDB = null;
        try{
            String myPath = Constant.DB_PATH + Constant.DB_NAME;
            File file = new File(myPath);
            if (file.exists() && !file.isDirectory())
                checkDB = SQLiteDatabase.openDatabase(myPath, null, SQLiteDatabase.OPEN_READONLY);
        }catch(SQLiteException e){
            Log.i(Tag,"checkDataBase: database does't exist yet.");
        }
        if(checkDB != null){
            checkDB.close();
        }
        return checkDB != null ? true : false;
    }

    /**
     * Copies your database from your local assets-folder to the just created empty database in the
     * system folder, from where it can be accessed and handled.
     * This is done by transfering bytestream.
     * */
    private void copyDataBase() throws IOException{
        //Open your local db as the input stream
        InputStream myInput = myContext.getAssets().open(Constant.DB_NAME);
        // Path to the just created empty db
        String outFileName = Constant.DB_PATH + Constant.DB_NAME;
        //Open the empty db as the output stream
        OutputStream myOutput = new FileOutputStream(outFileName);
        //transfer bytes from the inputfile to the outputfile
        byte[] buffer = new byte[1024];
        int length;
        while ((length = myInput.read(buffer))>0){
            myOutput.write(buffer, 0, length);
        }
        //Close the streams
        myOutput.flush();
        myOutput.close();
        myInput.close();
    }
}
```

#### 2. 增删查改批量操作

- 使用sql语句
- 使用Android提供的contentvalue方式

```java
c.getColumnIndex(String columnName);//返回某列名对应的列索引值  
c.getString(int columnIndex);   //返回当前行指定列的值 
```

```java
public class SQLiteOperation {
    private final String Tag="SQLiteOperation";
    private SQLiteHelper sqLiteHelper;
    private SQLiteOperation(){
    }
    private static class Inner {
        private static final SQLiteOperation instance = new SQLiteOperation();
    }
    public static SQLiteOperation getSingleton(){
        return SQLiteOperation.Inner.instance;
    }
    public void initSQLiteHepler(Context context){
        sqLiteHelper=new SQLiteHelper(context);    //todo? 这里并发是不是会存在问题
    }
    /**
     * @function: 增加单个数据到表中
     * */
    public Boolean add(FlexData flexData){
        SQLiteDatabase database=sqLiteHelper.getWritableDatabase();
        ContentValues contentValues=new ContentValues();
        contentValues.put("flexdata",flexData.getStringFlexData());
        contentValues.put("timestamp",flexData.getTimestamp());
        return database.insert("table_flex",null,contentValues)>0?true:false;
    }
    public void addBatch(ArrayList<FlexData> flexDataArrayList){
        SQLiteDatabase database=sqLiteHelper.getWritableDatabase();
        database.beginTransaction();
        try{
            ContentValues contentValues=new ContentValues();
            for(FlexData flexData: flexDataArrayList){
                contentValues.put("flexdata",flexData.getStringFlexData());
                contentValues.put("timestamp",flexData.getTimestamp());
                database.insert("table_flex",null,contentValues);
                contentValues.clear();
            }
            //调用该方法设置事务成功，否则endTransaction()方法将事务进行回滚操作
            database.setTransactionSuccessful();
        }finally {
            //由事务的标志决定是提交还是回滚事务
            database.endTransaction();
        }
    }
    /**
     * @function: 查询表中所有数据
     * */
    public ArrayList<FlexData> queryAll(){
        SQLiteDatabase sqLiteDatabase=sqLiteHelper.getReadableDatabase();
        ArrayList<FlexData> flexDataArrayList=new ArrayList<FlexData>();
        //Log.i(Tag,"queryAll: 操作");
        Cursor cursor=sqLiteDatabase.query("table_flex",new String[]{"flexdata","timestamp"},null,null,null,null,null);
        //Log.i(Tag,"queryAll: successful,the size="+cursor.getCount());
        while(cursor.moveToNext()){
            //Log.i(Tag,"queryAll: successful,the size="+cursor.getCount());
            @SuppressLint("Range") FlexData flexData=new FlexData(cursor.getString(cursor.getColumnIndex("flexdata")),cursor.getString(cursor.getColumnIndex("timestamp")));
            //Log.i(Tag,flexData.toString());
            flexDataArrayList.add(flexData);
        }
        //Log.i(Tag,"queryAll: 操作成功，获取数据大小="+flexDataArrayList.size());
        return flexDataArrayList;
    }
    /**
     * @function: 删除表操作
     * */
    public Boolean deleteAll(){
        try{
            SQLiteDatabase sqLiteDatabase=sqLiteHelper.getWritableDatabase();
            String sql="delete from table_flex";    //"delete from table_flex"; "DROP TABLE table_flex"
            sqLiteDatabase.execSQL(sql);
            Log.i(Tag,"deleteAll: 删除操作成功");
            return true;
        }catch (SQLException e){
            e.printStackTrace();
        }
        return false;
    }
}
```

##### .1. 选择性查询

- table：表名。
- columns：要查询出来的列名。
- selection：查询条件子句。
- selectionArgs：对应于selection语句中占位符的值。
- groupBy：分组。相当于select语句group by关键字后面的部分。
- having：分组后聚合的过滤条件。相当于select语句having关键字后面的部分。
- orderBy：排序。相当于select语句order by关键字后面的部分 ASC或DESC。
- limit：指定偏移量和获取的记录数。

```java
query( table, columns, selection, selectionArgs, groupBy, having, orderBy, limit );
```

```java
public Cursor queryAllCursor(String arg){
    SQLiteDatabase sqLiteDatabase=sqLiteHelper.getReadableDatabase();
    Cursor cursor=sqLiteDatabase.query("table_flex",new String[]{"label","flexdata"},"label like ?",new String[]{arg},null,null,null);
    while(cursor.moveToNext()){
        @SuppressLint("Range") FlexData flexData=new FlexData(cursor.getString(cursor.getColumnIndex("flexdata")),cursor.getString(cursor.getColumnIndex("timestamp")),cursor.getString(cursor.getColumnIndex("label")));
        flexDataArrayList.add(flexData);
    }
    Log.i(Tag,"queryAllCursor,查询个数为："+cursor.getCount());
    return cursor;
}
```

#### 3. 测试工具

- 通过DeviceFile Explorer插件可以在手机上找到对应的sqlite数据库文件，然后使用sqlitespy工具进行查看
- 通过编写测试的activity来检验数据库操作是不是有问题。
- `通过在测试文件中编写（这个目前不会） `  [示例demo，但是找不到对应的包](https://github.com/thedeveloperworldisyours/SQLiteExample/blob/master/app/src/androidTest/java/com/thedeveloperworldisyours/sqlite/SQLiteTest.java)  [url2](https://www.coder.work/article/266185)

#### 4. 插件工具

- [SQLiteSpy](E:\软件安装\数据库工具\SQLiteSpy_1.9.13)
- Android SDK的tools目录下提供了一个sqlite3.exe工具，程序运行生成的*.db文件一般位于"/data/data/项目名(包括所处包名)/databases/*.db"，因此要对数据库文件进行操作需要先找到数据库文件：

```shell
# adb shell  如果有多个设备
adb devices
adb -s 设备号 shell
#cd data/data
#ls                --列出所有项目
#cd project_name   --进入所需项目名
#cd databases    
#ls                --列出现寸的数据库文件
#sqlite3 test_db   --进入所需数据库
>.databases        --产看当前数据库
>.tables           --查看当前数据库中的表
>.help             --sqlite3帮助
>.schema            --各个表的生成语句
```





---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_%E5%AD%98%E5%82%A8/  

