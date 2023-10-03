# Tools


### 1. mysql

#### .1. cmd 配置使用

> - MySQL Community Server 社区版本，开源免费，但不提供官方技术支持。
> - MySQL Enterprise Edition 企业版本，需付费，可以试用30天。
> - MySQL Cluster 集群版，开源免费。可将几个MySQL Server封装成一个Server。
> - MySQL Cluster CGE 高级集群版，需付费。
> - MySQL Workbench（GUITOOL）一款专为MySQL设计的ER/数据库建模工具。它是著名的数据库设计工DBDesigner4的继任者。MySQLWorkbench又分为两个版本，分别是社区版（MySQL Workbench OSS）、商用版（MySQL WorkbenchSE）。

> 问题描述：在命令行输入 mysql -u root -p 登录mysql，返回”Can't connect to MySQL server on localhost (10061)”错误;
>
> - 安装mysqld服务器，输入命令：mysqld --install
> - 启动服务器了，输入命令：net start mysql
> - mysql 仍无法启动，输入命令：mysqld --initialize-insecure
> - 再次输入：net start mysql
> - 又给我出了个问题:Access denied for user 'root'@'localhost' (using password: YES)， 此时不用密码登录，输入命令： mysql -u root
> - 改密码,进入mysql数据库：use mysql;
> - ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '你的密码';  
> - 刷新一下：flush privileges;

- [下载链接](https://dev.mysql.com/downloads/file/?id=505213)
- [默认安装路径](C:\Program Files\MySQL\MySQL Server 8.0\bin)，添加到环境变量，使用cmd命令
- 用户root， 密码：1234qwer
- 本地登录

```sql
shell> mysql -u user -p
#Mysql    -h 电脑名（IP地址） -u 用户名  -p 密码
Enter password:
mysql> QUIT   #断开推出登录
```

| `mysql>` | 准备好进行新查询                                  |
| -------- | ------------------------------------------------- |
| `->`     | 等待多行查询的下一行                              |
| `'>`     | 等待下一行，等待以单引号开头的字符串的完成（`'`） |
| `">`     | 等待下一行，等待以双引号开头的字符串的完成（`"`） |
| ``>`     | 等待下一行，等待以反引号（```）开头的标识符的完成 |
| `/*>`    | 等待下一行，等待以＃开头的评论完成 `/*`           |

```sql
mysql> show databases;
mysql> use database;
mysql> show tables;
mysql> DESCRIBE pet;

create database 库名;
drop database 库名;
create table 表名(字段列表);
drop table 表名;
delete from 表名;  #清空表中记录：

source D:\ceshi.sql  #导入数据库

#创建用户创建新用户：create user ‘username’@‘host’ identified by ‘password’; 其中username为自定义的用户名；host为登录域名，host为’%'时表示为 任意IP，为localhost时表示本机，或者填写指定的IP地址；paasword为密码
#为用户授权：grant all privileges on . to ‘username’@’%’ with grant option; 其中*.第一个表示所有数据库，第二个表示所有数据表，如果不想授权全部那就把对应的写成相应数据库或者数据表；username为指定的用户；%为该用户登录的域名
#授权之后刷新权限：flush privileges;

create user ‘ldd’@‘localhost’ identified by ‘1234qwer’;
grant all privileges on *.* to ‘ldd’@‘localhost’ with grant option;
```

```sql
CREATE TABLE shop (
    article INT(4) UNSIGNED ZEROFILL DEFAULT '0000' NOT NULL,
    dealer  CHAR(20)                 DEFAULT ''     NOT NULL,
    price   DOUBLE(16,2)             DEFAULT '0.00' NOT NULL,
    PRIMARY KEY(article, dealer));
INSERT INTO shop VALUES
    (1,'A',3.45),(1,'B',3.99),(2,'A',10.99),(3,'B',1.45),
    (3,'C',1.69),(3,'D',1.25),(4,'D',19.95);
```

- 配置文件

```ini
[mysqld]
设置3306端口
port=3306
设置mysql的安装目录
basedir=D:\Program Files\MySQL
设置mysql数据库的数据的存放目录
datadir=D:\Program Files\MySQL\Data
允许最大连接数
max_connections=200
允许连接失败的次数。这是为了防止有人从该主机试图攻击数据库系统
max_connect_errors=10
服务端使用的字符集默认为UTF8
character-set-server=utf8
创建新表时将使用的默认存储引擎
default-storage-engine=INNODB
默认使用“mysql_native_password”插件认证
default_authentication_plugin=mysql_native_password
[mysql]
设置mysql客户端默认字符集
default-character-set=utf8
[client]
设置mysql客户端连接服务端时默认使用的端口
port=3306
default-character-set=utf8
```

- basic usage

```java
 
 
/**
 * 官方api 文档：https://dev.mysql.com/doc/connector-j/8.0/en/connector-j-examples.html
 * 连接数据库初始化
 * @author ldd
 *
 */
public class UtilDb {
	private static final String USER = "root";//数据库用户名
	private static final String PASSWORD = "199611@liu";//数据库密码
	private static final String URL = "jdbc:mysql://localhost:3306/MyshoppingSystem?useUnicode=true&characterEncoding=UTF-8&characterSetResults=utf8&serverTimezone=GMT";//数据库url
	//private static final String DRIVER = "com.mysql.jdbc.Driver";
	private static final String DRIVER = "com.mysql.cj.jdbc.Driver";
/*Loading class com.mysql.jdbc.Driver. This is deprecated. The new driver class is com.mysql.cj.jdbc.Driver. The driver is automatically registered via the SPI and manual loading of the driver class is generally unnecessary.*/
	private static Connection conn = null; //连接数据库对象
 
	
	//获取connection连接对象
	public static Connection getConnection(){
		//通过单例模式实现
		if(conn==null){
			try {
				Class.forName(DRIVER);//加载驱动
				conn = DriverManager.getConnection(URL, USER, PASSWORD);
				return conn;
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
		}
		return conn;
	}
	//测试连接
	public static void main(String[] args) {
		Connection conn = UtilDb.getConnection();
		if(conn!=null){
			System.out.println("连接数据库成功");
		}else{
			System.out.println("连接失败");
		}
	}	
}
```

```java
@Override
public Ingredient findOne(String id) {
    Connection connection = null;
    PreparedStatement statement = null;
    ResultSet resultSet = null;
    try {
        connection = dataSource.getConnection();
        statement = connection.prepareStatement(
            "select id, name, type from Ingredient");
        statement.setString(1, id);
        resultSet = statement.executeQuery();
        Ingredient ingredient = null;
        if(resultSet.next()) {
            ingredient = new Ingredient(
                resultSet.getString("id"),
                resultSet.getString("name"),
                Ingredient.Type.valueOf(resultSet.getString("type")));
        }
        return ingredient;
    } catch (SQLException e) {
        // ??? What should be done here ???
    } finally {
        if (resultSet != null) {
            try {
                resultSet.close();
            } catch (SQLException e) {
            }
        }
        if (statement != null) {
            try {
                statement.close();
            } catch (SQLException e) {
            }
        }
        if (connection != null) {
            try {
                connection.close();
            } catch (SQLException e) {
            }
        }
    }
    return null;
}
```


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/tools/  

