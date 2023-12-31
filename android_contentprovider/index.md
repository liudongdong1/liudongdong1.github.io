# Android_ContentProvider


> `ContentProvider`属于 `Android`的四大组件之一，基于 `Android`中的`Binder`机制实现。主要应用于进程间数据传输。Content Provider组件是Android应用的重要组件之一，管理对数据的访问，主要用于不同的应用程序之间实现数据共享的功能。Content Provider的数据源不止包括SQLite数据库，还可以是文件数据。通过将数据储存层和应用层分离，Content Provider为各种数据源提供了一个通用的接口。
>
> 一旦某个应用程序通过ContentProvider 暴露了自己的数据操作接口，那么不管该应用程序是否启动，其他应用程序都可通过该接口来操作该应用程序的内部数据，包括增加数据、删除数据、修改数据、查询数据等。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/162046022fba9845tplv-t2oaga2asx-watermark.awebp)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108232739629.png)

### 0. 关键类

#### .1. Uri 工具类

- void addURI(String authority, String path, int code):该方法用于向UriMatcher对象注册Uri。其中 authority和 path组合成一个Uri，而 code则代表该Uri对应的标识码。
- int match(Uri uri):根据前面注册的Uri来判断指定Uri对应的标识码。如果找不到匹配的标识码，该方法将会返回-1。

### 1.android 提供的ContentProvider

- 调用 Activity 的ContentResolver()获取ContentResolver对象
- 根据需要调用ContentResolver 提供的insert，delete，query方法， 需要了解具体的Uri

#### .1. 读取手机联系人

```xml
<uses-permission android:name="android.permission.READ_CONTACTS"/>
```

```java
private void getContacts(){
    //①查询raw_contacts表获得联系人的id
    ContentResolver resolver = getContentResolver();
    Uri uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI;
    //查询联系人数据
    cursor = resolver.query(uri, null, null, null, null);
    while(cursor.moveToNext())
    {
        //获取联系人姓名,手机号码
        String cName = cursor.getString(cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME));
        String cNum = cursor.getString(cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER));
    }
    cursor.close();
}
```

#### .2. 添加联系人

```java
private void AddContact() throws RemoteException, OperationApplicationException {
    //使用事务添加联系人
    Uri uri = Uri.parse("content://com.android.contacts/raw_contacts");
    Uri dataUri =  Uri.parse("content://com.android.contacts/data");

    ContentResolver resolver = getContentResolver();
    ArrayList<ContentProviderOperation> operations = new ArrayList<ContentProviderOperation>();
    ContentProviderOperation op1 = ContentProviderOperation.newInsert(uri)
            .withValue("account_name", null)
            .build();
    operations.add(op1);

    //依次是姓名，号码，邮编
    ContentProviderOperation op2 = ContentProviderOperation.newInsert(dataUri)
            .withValueBackReference("raw_contact_id", 0)
            .withValue("mimetype", "vnd.android.cursor.item/name")
            .withValue("data2", "Coder-pig")
            .build();
    operations.add(op2);

    ContentProviderOperation op3 = ContentProviderOperation.newInsert(dataUri)
            .withValueBackReference("raw_contact_id", 0)
            .withValue("mimetype", "vnd.android.cursor.item/phone_v2")
            .withValue("data1", "13798988888")
            .withValue("data2", "2")
            .build();
    operations.add(op3);

    ContentProviderOperation op4 = ContentProviderOperation.newInsert(dataUri)
            .withValueBackReference("raw_contact_id", 0)
            .withValue("mimetype", "vnd.android.cursor.item/email_v2")
            .withValue("data1", "779878443@qq.com")
            .withValue("data2", "2")
            .build();
    operations.add(op4);
    //将上述内容添加到手机联系人中~
    resolver.applyBatch("com.android.contacts", operations);
    Toast.makeText(getApplicationContext(), "添加成功", Toast.LENGTH_SHORT).show();
}
```

#### .3. 管理多媒体内容

- MediaStore.Audio.Media. INTERNAL_CONTENT_URI:存储在手机内部存储器上的音频文件内容的ContentProvider 的 Uri。
- MediaStore.Audio.Images.EXTERNAL_CONTENT_URl:存储在外部存储器(SD卡）上的图片文件内容的ContentProvider的 Uri。
- MediaStore.Audio.Images.INTERNAL_CONTENT_URI:存储在手机内部存储器上的图片文件内容的ContentProvider的Uri。
- MediaStore.Audio.Video.EXTERNAL_CONTENT_URI:存储在外部存储器（SD卡）上的音频文件内容的ContentProvider的 Uri。
- MediaStore.Audio.Video.INTERNAL_CONTENT_URI:存储在手机内部存储器上的音频文件内容的ContentProvider的 Uri。

### 2. 自定义ContentProvider

- 开发一个ContentProvider的子类，该子类需要实现增、删、改、查等方法。
- 在AndroidManifest.xml文件中注册该ContentProvider.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211108232007139.png)

> ContentProvider组件需要在清单文件声明

```xml
<provider android:name=".contentprovider.MyProvider"
    android:authorities="com.example.myprovider"
/>
```

#### .1. 基本URI定义

```java
public final class Words
{
	// 定义该ContentProvider的Authorities
	public static final String AUTHORITY
			= "org.crazyit.providers.dictprovider";
	// 定义一个静态内部类，定义该ContentProvider所包含的数据列的列名
	public static final class Word implements BaseColumns
	{
		// 定义Content所允许操作的三个数据列
		public final static String _ID = "_id";
		public final static String WORD = "word";
		public final static String DETAIL = "detail";
		// 定义该Content提供服务的两个Uri
		public final static Uri DICT_CONTENT_URI = Uri
				.parse("content://" + AUTHORITY + "/words");
		public final static Uri WORD_CONTENT_URI = Uri
				.parse("content://"	+ AUTHORITY + "/word");
	}
}
```

#### .2. 创建MyProvider继承自ContentProvider

```java
public class MyDatabaseHelper extends SQLiteOpenHelper
{

	public static final String CREATE_TABLE_SQL = "create table dict(_id integer primary " +
			"key autoincrement , word , detail)";

	public MyDatabaseHelper(Context context, String name, int version)
	{
		super(context, name, null, version);
	}

	@Override
	public void onCreate(SQLiteDatabase db)
	{
		// 第一次使用数据库时自动建表
		db.execSQL(CREATE_TABLE_SQL);
	}

	@Override
	public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion)
	{
		System.out.println("--------onUpdate Called--------" + oldVersion + "--->" + newVersion);
	}
}
```

```java
public class DictProvider extends ContentProvider
{
	private static UriMatcher matcher
			= new UriMatcher(UriMatcher.NO_MATCH);
	private static final int WORDS = 1;
	private static final int WORD = 2;
	private MyDatabaseHelper dbOpenHelper;
	static
	{
		// 为UriMatcher注册两个Uri
		matcher.addURI(Words.AUTHORITY, "words", WORDS);
		matcher.addURI(Words.AUTHORITY, "word/#", WORD);
	}
	// 第一次调用该DictProvider时，系统先创建DictProvider对象，并回调该方法
	@Override
	public boolean onCreate()
	{
		dbOpenHelper = new MyDatabaseHelper(this.getContext(),
				"myDict.db3", 1);
		return true;
	}
	// 返回指定Uri参数对应的数据的MIME类型
	@Override
	public String getType(@NonNull  Uri uri)
	{
		switch (matcher.match(uri))
		{
			// 如果操作的数据是多项记录
			case WORDS:
				return "vnd.android.cursor.dir/org.crazyit.dict";
			// 如果操作的数据是单项记录
			case WORD:
				return "vnd.android.cursor.item/org.crazyit.dict";
			default:
				throw new IllegalArgumentException("未知Uri:" + uri);
		}
	}
	// 查询数据的方法
	@Override
	public Cursor query(@NonNull  Uri uri, String[] projection, String where,
		String[] whereArgs, String sortOrder)
	{
		SQLiteDatabase db = dbOpenHelper.getReadableDatabase();
		switch (matcher.match(uri))
		{
			// 如果Uri参数代表操作全部数据项
			case WORDS:
				// 执行查询
				return db.query("dict", projection, where,
						whereArgs, null, null, sortOrder);
			// 如果Uri参数代表操作指定数据项
			case WORD:
				// 解析出想查询的记录ID
				long id = ContentUris.parseId(uri);
				String whereClause = Words.Word._ID + "=" + id;
				// 如果原来的where子句存在，拼接where子句
				if (where != null && !"".equals(where))
				{
					whereClause = whereClause + " and " + where;
				}
				return db.query("dict", projection, whereClause, whereArgs,
						null, null, sortOrder);
			default:
				throw new IllegalArgumentException("未知Uri:" + uri);
		}
	}
	// 插入数据方法
	@Override
	public Uri insert(@NonNull Uri uri, ContentValues values)
	{
		// 获得数据库实例
		SQLiteDatabase db = dbOpenHelper.getReadableDatabase();
		switch (matcher.match(uri))
		{
			// 如果Uri参数代表操作全部数据项
			case WORDS:
				// 插入数据，返回插入记录的ID
				long rowId = db.insert("dict", Words.Word._ID, values);
				// 如果插入成功返回uri
				if (rowId > 0)
				{
					// 在已有的 Uri的后面追加ID
					Uri wordUri = ContentUris.withAppendedId(uri, rowId);
					// 通知数据已经改变
					getContext().getContentResolver()
							.notifyChange(wordUri, null);
					return wordUri;
				}
				break;
			default :
				throw new IllegalArgumentException("未知Uri:" + uri);
		}
		return null;
	}
	// 修改数据的方法
	@Override
	public int update(@NonNull Uri uri, ContentValues values, String where,
					  String[] whereArgs)
	{
		SQLiteDatabase db = dbOpenHelper.getWritableDatabase();
		// 记录所修改的记录数
		int num;
		switch (matcher.match(uri))
		{
			// 如果Uri参数代表操作全部数据项
			case WORDS:
				num = db.update("dict", values, where, whereArgs);
				break;
			// 如果Uri参数代表操作指定数据项
			case WORD:
				// 解析出想修改的记录ID
				long id = ContentUris.parseId(uri);
				String whereClause = Words.Word._ID + "=" + id;
				// 如果原来的where子句存在，拼接where子句
				if (where != null && !where.equals(""))
				{
					whereClause = whereClause + " and " + where;
				}
				num = db.update("dict", values, whereClause, whereArgs);
				break;
			default:
				throw new IllegalArgumentException("未知Uri:" + uri);
		}
		// 通知数据已经改变
		getContext().getContentResolver().notifyChange(uri, null);
		return num;
	}
	// 删除数据的方法
	@Override
	public int delete(@NonNull Uri uri, String where, String[] whereArgs)
	{
		SQLiteDatabase db = dbOpenHelper.getReadableDatabase();
		// 记录所删除的记录数
		int num;
		// 对uri进行匹配
		switch (matcher.match(uri))
		{
			// 如果Uri参数代表操作全部数据项
			case WORDS:
				num = db.delete("dict", where, whereArgs);
				break;
			// 如果Uri参数代表操作指定数据项
			case WORD:
				// 解析出所需要删除的记录ID
				long id = ContentUris.parseId(uri);
				String whereClause = Words.Word._ID + "=" + id;
				// 如果原来的where子句存在，拼接where子句
				if (where != null && !where.equals(""))
				{
					whereClause = whereClause + " and " + where;
				}
				num = db.delete("dict", whereClause, whereArgs);
				break;
			default:
				throw new IllegalArgumentException("未知Uri:" + uri);
		}
		// 通知数据已经改变
		getContext().getContentResolver().notifyChange(uri, null);
		return num;
	}
}
```

#### .3. Activity中使用

```java
public class MainActivity extends Activity
{
	private MyDatabaseHelper dbHelper;
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		// 创建MyDatabaseHelper对象，指定数据库版本为1，此处使用相对路径即可
		// 数据库文件自动会保存在程序的数据文件夹的databases目录下
		dbHelper = new MyDatabaseHelper(this, "myDict.db3", 1);
		Button insertBn = findViewById(R.id.insert);
		Button searchBn = findViewById(R.id.search);
		EditText wordEt = findViewById(R.id.word);
		EditText detailEt = findViewById(R.id.detail);
		EditText keyEt = findViewById(R.id.key);
		insertBn.setOnClickListener(view -> {
			// 获取用户输入
			String word = wordEt.getText().toString();
			String detail = detailEt.getText().toString();
			// 插入单词记录
			insertData(dbHelper.getReadableDatabase(), word, detail);
			// 显示提示信息
			Toast.makeText(MainActivity.this, "添加单词成功！",
					Toast.LENGTH_LONG).show();
		});
		searchBn.setOnClickListener(view -> {
			// 获取用户输入
			String key = keyEt.getText().toString();
			// 执行查询
			Cursor cursor = dbHelper.getReadableDatabase().rawQuery(
					"select * from dict where word like ? or detail like ?",
					new String[] { "%" + key + "%", "%" + key + "%" });
			// 创建一个Bundle对象
			Bundle data = new Bundle();
			data.putSerializable("data", converCursorToList(cursor));
			// 创建一个Intent
			Intent intent = new Intent(MainActivity.this, ResultActivity.class);
			intent.putExtras(data);
			// 启动Activity
			startActivity(intent);
		});
	}

	private ArrayList<Map<String, String>> converCursorToList(Cursor cursor)
	{
		ArrayList<Map<String, String>> result = new ArrayList<>();
		// 遍历Cursor结果集
		while (cursor.moveToNext())
		{
			// 将结果集中的数据存入ArrayList中
			Map<String, String> map = new HashMap<>();
			// 取出查询记录中第2列、第3列的值
			map.put("word", cursor.getString(1));
			map.put("detail", cursor.getString(2));
			result.add(map);
		}
		return result;
	}

	private void insertData(SQLiteDatabase db, String word, String detail)
	{
		// 执行插入语句
		db.execSQL("insert into dict values(null , ? , ?)", new String[]{word, detail});
	}

	@Override
	public void onDestroy()
	{
		super.onDestroy();
		// 退出程序时关闭MyDatabaseHelper里的SQLiteDatabase
		dbHelper.close();
	}
}
```

#### .4. 其他应用注册数据变化监听

- 使用系统自带的 getContentResolver().registerContentObserver

```java
public class MainActivity extends Activity
{
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 请求获取读取短信的权限
        requestPermissions(new String[]{Manifest.permission.READ_SMS}, 0x123);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,  @NonNull int[] grantResults)
    {
        // 如果用户授权访问短信内容
        if (grantResults[0] == 0 && requestCode == 0x123)
        {
            // 为Telephony.Sms.CONTENT_URI的数据改变注册监听器
            getContentResolver().registerContentObserver(Telephony.Sms.CONTENT_URI,
                                                         true, new SmsObserver(new Handler()));
        } else
        {
            Toast.makeText(this, "您必须授权访问短信内容才能测试该应用",
                           Toast.LENGTH_SHORT).show();
        }
    }
    // 提供自定义的ContentObserver监听器类
    private class SmsObserver extends ContentObserver
    {
        SmsObserver(Handler handler)
        {
            super(handler);
        }
        private String prevMsg = "";
        @Override
        public void onChange(boolean selfChange)
        {
            // 查询发件箱中的短信（所有已发送的短信都处于发件箱中）
            Cursor cursor = getContentResolver().query(Telephony.Sms.Sent.
                                                       CONTENT_URI, null, null, null, null);
            // 遍历查询得到的结果集，即可获取用户正在发送的短信
            while (cursor.moveToNext())
            {
                // 只显示最近5秒内发出的短信
                if (Math.abs(System.currentTimeMillis() - cursor.
                             getLong(cursor.getColumnIndex("date"))) < 5000){
                    StringBuilder sb = new StringBuilder();
                    // 获取短信的发送地址
                    sb.append("address=").append(cursor.getString(cursor.
                                                                  getColumnIndex("address")));
                    // 获取短信的标题
                    sb.append(";subject=").append(cursor.getString(cursor.
                                                                   getColumnIndex("subject")));
                    // 获取短信的内容
                    sb.append(";body=").append(cursor.getString(cursor.
                                                                getColumnIndex("body")));
                    // 获取短信的发送时间
                    sb.append(";time=").append(cursor.getLong(cursor.
                                                              getColumnIndex("date")));
                    if (!prevMsg.equals(sb.toString()))
                    {
                        prevMsg = sb.toString();
                        System.out.println("发送短信：" + prevMsg);
                    }
                }
            }
            cursor.close();
        }
    }
}
```

### 3. document provider

#### .1. SAF 框架

> - **Document provider**：一个特殊的ContentProvider，让一个存储服务(比如Google Drive)可以 对外展示自己所管理的文件。它是**DocumentsProvider**的子类，另外，document-provider的存储格式 和传统的文件存储格式一致，至于你的内容如何存储，则完全决定于你自己，Android系统已经内置了几个 这样的Document provider，比如关于下载，图片以及视频的Document provider！
> - **Client app**：一个普通的客户端软件，通过触发**ACTION_OPEN_DOCUMENT** 和/或 **ACTION_CREATE_DOCUMENT**就可以接收到来自于Document provider返回的内容，比如选择一个图片， 然后返回一个Uri。
> - **Picker**：`类似于文件管理器的界面`，而且是系统级的界面，提供额访问客户端过滤条件的 Document provider内容的通道，就是起说的那个DocumentsUI程序！

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211109084331128.png)

#### .2. 客户端调用，并获取返回的Uri

```java
public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static final int READ_REQUEST_CODE = 42;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btn_show = (Button) findViewById(R.id.btn_show);
        btn_show.setOnClickListener(this);
    }
    @Override
    public void onClick(View v) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, READ_REQUEST_CODE);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == READ_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            Uri uri;
            if (data != null) {
                uri = data.getData();
                Log.e("HeHe", "Uri: " + uri.toString());
            }
        }
    }
}
```

#### .3. 根据uri获取文件参数

```java
public void dumpImageMetaData(Uri uri) {
    Cursor cursor = getContentResolver()
        .query(uri, null, null, null, null, null);
    try {
        if (cursor != null && cursor.moveToFirst()) {
            String displayName = cursor.getString(
                cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME));
            Log.e("HeHe", "Display Name: " + displayName);
            int sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE);
            String size = null;
            if (!cursor.isNull(sizeIndex)) {
                size = cursor.getString(sizeIndex);
            }else {
                size = "Unknown";
            }
            Log.e("HeHe", "Size: " + size);
        }
    }finally {
        cursor.close();
    }
}
```

#### .4. 根据Uri获取Bitmap

```java
private Bitmap getBitmapFromUri(Uri uri) throws IOException {
        ParcelFileDescriptor parcelFileDescriptor =
        getContentResolver().openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();
        return image;
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_contentprovider/  

