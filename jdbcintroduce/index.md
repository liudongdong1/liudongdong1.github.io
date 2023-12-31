# JDBCIntroduce


> JDBC（Java DataBase Connectivity,java数据库连接）是一种用于执行SQL语句的Java API，可以为`多种关系数据库提供统一访问`，它由一组用Java语言编写的类和接口组成。JDBC提供了一种基准，据此可以构建更高级的工具和接口，使数据库开发人员能够编写数据库应用程序。
>
> - 核心类：DriverManager  Connection  Statement  PreparedStatement  ResultSet
> - 数据查询处理(INSERT、UPDATE、DELETE)： public int executeUpdate(String sql) throws SQLException
> - 数据更新处理(SLECT、COUNT…)： public ResultSet executeQuery(String sql) throws SQLException

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODU2ODI5Mg==,size_16,color_FFFFFF,t_70)

![ResultSet](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODU2ODI5Mg==,size_16,color_FFFFFF,t_70-16390126586162)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211209092238005.png)

![PreparedStatement](https://img-blog.csdnimg.cn/20200712143134653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODU2ODI5Mg==,size_16,color_FFFFFF,t_70)

> 1. Statement:用于执行SQL语句的对象
>
> * 通过Connection的createStatement()方法来获取
> * 通过excuteUpdate(sql)可以执行SQL语句
> * 传入的SQL可以是insert,update或者delete,但是不能是select
>
> 2. Connection、Statement都是应用程序和数据库服务器的连接  资源，使用后一定要关闭
>
> * 需要在finally``中关闭Connection和Statement对象
>    ``* 异常可以不处理，但是连接一定要关闭
>
> 3. 关闭的顺序：先关闭后获取的,即先关闭Statement，后关闭Connection

#### 0. 建表操作

```sql
CREATE TABLE `news` (
  `nid` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(30) CHARACTER SET utf8 NOT NULL,
  `read_count` int(11) DEFAULT NULL,
  `price` double DEFAULT NULL,
  `content` text CHARACTER SET utf8,
  `pubdate` date DEFAULT NULL,
  PRIMARY KEY (`nid`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8
```

```sql
CREATE TABLE `websites` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` char(20) NOT NULL DEFAULT '' COMMENT '站点名称',
  `url` varchar(255) NOT NULL DEFAULT '',
  `alexa` int(11) NOT NULL DEFAULT '0' COMMENT 'Alexa 排名',
  `country` char(10) NOT NULL DEFAULT '' COMMENT '国家',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8;
```

#### 1. 获取连接

```java
/**
 * 工具类，获取数据库的连接
 */
public class DBUtil {
     private static String URL="jdbc:mysql://127.0.0.1:3306/test";
     private static String USER="root";
     private static String PASSWROD ="123456";
     private static Connection connection=null;
     static{
         try {
             Class.forName("com.mysql.jdbc.Driver");
             // 获取数据库连接
             connection=DriverManager.getConnection(URL,USER,PASSWROD);
             System.out.println("连接成功");
         } catch (ClassNotFoundException e) {
             // TODO Auto-generated catch block
             e.printStackTrace();
         } catch (SQLException e) {
             // TODO Auto-generated catch block
             e.printStackTrace();
         }
     }
     // 返回数据库连接
     public static Connection getConnection(){
         return connection;
     }
 }
```

#### 2.  批量&添加操作

```java
/**
     * @function: 添加一天flexdata的数据记录
     * @param flexData: 一条传感器数据
     * */
public Boolean addFlexData(FlexData flexData){
    String sql="insert into table_flex(label,flexdata,timestamp)"+"values(?,?,?)";
    boolean result=false;
    try{
        conn=getConnection();
        pst=conn.prepareStatement(sql);
        pst.setString(1,flexData.getLabel());
        pst.setString(2,flexData.getStringFlexData());
        pst.setString(3,flexData.getTimestamp());
        result=pst.execute();
        return result;
    } catch (SQLException throwables) {
        throwables.printStackTrace();
    }finally {
        closeAll();
        return result;
    }
}
```

```java
/**
     * @function: 批量添加flexdata的数据记录
     * @param flexDataArrayList: 一系列传感器数据
     * */
public void addBatchFlexData(ArrayList<FlexData>flexDataArrayList){
    String sql="insert into table_flex(label,flexdata,timestamp)"+"values(?,?,?)";
    boolean result=false;
    try{
        conn=getConnection();
        pst=conn.prepareStatement(sql);
        int i=0;
        for(FlexData flexData :flexDataArrayList){
            i=i+1;
            pst.setString(1,flexData.getLabel());
            pst.setString(2,flexData.getStringFlexData());
            pst.setString(3,flexData.getTimestamp());
            pst.addBatch();
            if(i%100==0){
                pst.executeBatch();
                pst.clearBatch();
            }
        }
        pst.executeBatch();
    } catch (SQLException throwables) {
        throwables.printStackTrace();
    }finally {
        closeAll();
    }
}
```

#### 3. 更新操作

```java
String sql = "INSERT INTO news(title, read_count, price, content, pubdate) VALUES 
('哈喽', 1, 20.0, '内容1', curdate())";
String sql2 = "UPDATE news SET title='UPDATE标题1',read_count=9999 where nid=6;";
int count = stmt.executeUpdate(sql2);
System.out.println("受影响的行数" + count);
```

```java
// 更新学生信息
public void updateStudent(Student student){
    String sql = "update student set name = ? ,age =?,score = ? where id = ? ";
    boolean result = false;
    try {
        PreparedStatement preparedStatement = connection.prepareStatement(sql);
        preparedStatement.setString(1, student.getName());
        preparedStatement.setInt(2, student.getAge());
        preparedStatement.setDouble(3, student.getScore());
        preparedStatement.setInt(4, student.getId());
        preparedStatement.executeUpdate();
    } catch (SQLException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
        System.out.println("连接数据库失败");
    }
}
```

#### 4. 删除一条记录

```java
// 根据id删除一个学生
public void deleteStudent(int id){
    String sql = "delete from student where id = ?";
    boolean result = false;
    try {
        PreparedStatement preparedStatement = connection.prepareStatement(sql);
        preparedStatement.setInt(1, id);
        result=preparedStatement.execute();
    } catch (SQLException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
    }
}
```

#### 5. 查询操作&模糊查询

```java
// 根据id查询学生
public Student selectStudent(int id){
    String sql ="select * from student where id =?";
    try {
        PreparedStatement preparedStatement = connection.prepareStatement(sql);
        preparedStatement.setInt(1, id);
        ResultSet resultSet = preparedStatement.executeQuery();
        Student student = new Student();
        // 一条也只能使用resultset来接收
        while(resultSet.next()){
            student.setId(resultSet.getInt("id"));
            student.setName(resultSet.getString("name"));
            student.setAge(resultSet.getInt("age"));
            student.setScore(resultSet.getDouble("score"));
        }
        return student;
    } catch (SQLException e) {
        // TODO: handle exception
    }
    return null;
}
```

##### .1.  Like Demo

> conn.execute("SELECT * FROM article WHERE content LIKE ?", ('%' + value + '%',))

```java
/**
     * SQL 模糊查询，label 数据包含Label
     * @param Label : 关键词
     * @return 结果集
     */
public ArrayList<FlexData> executeQueryContain(String Label) {
    String sql="select * from table_flex where label like ?";
    ArrayList<FlexData>arrayList=new ArrayList<FlexData>();
    try {
        // 获得连接
        conn = getConnection();

        // 调用SQL
        pst = conn.prepareStatement(sql);
        //System.out.println("select * from table_flex where label like '%?%' ");
        pst.setString(1,"%"+Label+"%");   // 注意在这里除了若干问题
        //System.out.println("2select * from table_flex where label like '%?%' ");

        System.out.println(pst.getParameterMetaData());

        // 执行
        rst = pst.executeQuery();

        while(rst.next()) {
            arrayList.add(new FlexData(rst.getString("flexdata"),rst.getString("timestamp"),rst.getString("label")));
        }
        return arrayList;
    } catch (SQLException e) {
        System.out.println(e.getMessage());
        System.out.println("异常错误");
    } finally {
        closeAll();
        return arrayList;
    }
}
```

##### .2. 正则类

- charindex（）——charindex（字符，字符串）>0 –>包含
- %: 表示任意0个或多个字符，可匹配任意类型和长度的字符。有些情况下是中文，需用两个百分号（%%）表示: SELECT * FROM [user] WHERE u_name LIKE ‘%三%’ AND u_name LIKE ‘%猫%’
- _: 表示任意单个字符。匹配单个任意字符，它常用来限定表达式的字符长度语句：
- []: 表示括号内所列字符中的一个（类似正则表达式）。指定一个字符、字符串或范围，要求所匹配对象为它们中的任一个：SELECT * FROM [user] WHERE u_name LIKE ‘[张李王]三’
- [^ ]: 表示不在括号所列之内的单个字符。其取值和 [] 相同，但它要求所匹配对象为指定字符以外的任一个字符：`SELECT * FROM [user] WHERE u_name LIKE ‘[^张李王]三’`

#### 6. 查询所有

```java
// 查询所有学生，返回List
public List<Student> selectStudentList(){
    List<Student>students  = new ArrayList<Student>(); 
    String sql ="select * from student ";
    try {
        PreparedStatement preparedStatement = DBUtil.getConnection().prepareStatement(sql);
        ResultSet resultSet = preparedStatement.executeQuery();
        // 不能把student在循环外面创建，要不list里面六个对象都是一样的，都是最后一个的值，
        // 因为list add进去的都是引用
        // Student student = new Student();
        while(resultSet.next()){
            Student student = new Student();
            student.setId(resultSet.getInt(1));
            student.setName(resultSet.getString(2));
            student.setAge(resultSet.getInt(3));
            student.setScore(resultSet.getDouble(4));
            students.add(student);

        }
    } catch (SQLException e) {
        // TODO: handle exception
    }
    return students;
}

```

#### 7. 关闭所有资源

```java
/**
     * 关闭所有资源
     */
private void closeAll() {
    // 关闭结果集对象
    if (rst != null) {
        try {
            rst.close();
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }

    // 关闭PreparedStatement对象
    if (pst != null) {
        try {
            pst.close();
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }

    // 关闭CallableStatement 对象
    if (callableStatement != null) {
        try {
            callableStatement.close();
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }

    // 关闭Connection 对象
    if (conn != null) {
        try {
            conn.close();
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }
}
```

#### Resource

- https://blog.csdn.net/weixin_48568292/article/details/107299093

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/jdbcintroduce/  

