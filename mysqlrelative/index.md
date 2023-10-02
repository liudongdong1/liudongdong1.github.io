# MySQLRelative


### 0. 执行过程

* 查询语句的执行流程如下：权限校验（如果命中缓存）--->查询缓存--->分析器--->优化器--->权限校验--->执行器--->引擎
* 更新语句执行流程如下：分析器---->权限校验---->执行器--->引擎---redo log(prepare 状态)--->binlog--->redo log(commit状态)
* **连接器：**身份认证和权限相关(登录 MySQL 的时候)。![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210709233330122.png)
* **查询缓存：**执行查询语句的时候，`会先查询缓存（MySQL 8.0 版本后移除，因为这个功能不太实用）`。
* **分析器：** 没有命中缓存的话，SQL 语句就会经过分析器，分析器说白了就是要先看你的 SQL 语句要干嘛，再检查你的 SQL 语句语法是否正确。
* **优化器：**按照 MySQL 认为最优的方案去执行。
* **执行器：**执行语句，然后从存储引擎返回数据。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210624183525647.png)

* **Server 层**：主要包括连接器、查询缓存、分析器、优化器、执行器等，所有跨存储引擎的功能都在这一层实现，比如存储过程、触发器、视图，函数等，还有一个通用的日志模块 binglog 日志模块。
* **存储引擎**： 主要负责数据的存储和读取，采用可以替换的插件式架构，支持 `InnoDB、MyISAM、Memory 等多个存储引擎`，其中 InnoDB 引擎有自有的日志模块 redolog 模块。**现在最常用的存储引擎是 InnoDB，它从 MySQL 5.5.5 版本开始就被当做默认存储引擎了。**

### 1. 数据类型

- `整型`： TINYINT, SMALLINT, MEDIUMINT, INT, BIGINT 分别使用 8, 16, 24, 32, 64 位存储空间，一般情况下越小的列越好。
- `浮点数`: FLOAT 和 DOUBLE 为浮点类型，DECIMAL 为高精度小数类型。CPU 原生支持浮点运算，但是不支持 DECIMAl 类型的计算，因此 DECIMAL 的计算比浮点类型需要更高的代价。
- `字符串：`主要有 CHAR 和 VARCHAR 两种类型，一种是定长的，一种是变长的。VARCHAR 这种变长类型能够节省空间，因为只需要存储必要的内容。但是在执行 UPDATE 时可能会使行变得比原来长，当超出一个页所能容纳的大小时，就要执行额外的操作。MyISAM 会将行拆成不同的片段存储，而 InnoDB 则需要分裂页来使行放进页内。`VARCHAR 会保留字符串末尾的空格`，而 `CHAR 会删除`。
- `时间日期`： 
  - **DATETIME:**从 1001 年到 9999 年的日期和时间，精度为秒，使用 8 字节的存储空间。与时区无关。
  - **TIMESTAMP:**从 1970 年 1 月 1 日午夜(格林威治时间)以来的秒数，使用 4 个字节，只能表示从 1970 年 到 2038 年。`和时区有关.`
  - MySQL 提供了` FROM_UNIXTIME() 函数把 UNIX 时间戳转换为日期`，并提供了 `UNIX_TIMESTAMP() 函数把日期转换为 UNIX 时间戳`。
- **BLOB**和**TEXT**都是为存储很大的数据而设计的数据类型，分别采用`二进制`和`字符方式`存储。

### 2. 存储引擎

#### .1. InnoDB

- MySQL 默认的事务型存储引擎，**只有在需要它不支持的特性时，才考虑使用其它存储引擎**。
- 实现了四个标准的隔离级别，`默认级别是可重复读(REPEATABLE READ)`。在可重复读隔离级别下，`通过多版本并发控制(MVCC)+ 间隙锁(Next-Key Locking)`防止幻影读。

#### .2. MyISAM

- 不支持事务, 不支持行级锁，

**查看 MySQL 提供的所有存储引擎**

```sql
mysql> show engines;
```

**查看 MySQL 当前默认的存储引擎**

我们也可以通过下面的命令查看默认的存储引擎。

```sql
mysql> show variables like '%storage_engine%';
```

**查看表的存储引擎**

```sql
show table status like "table_name" ;
```

### 3. 索引机制(B+树)

### 4. 锁机制&锁算法

**MyISAM 和 InnoDB 存储引擎使用的锁：**

- MyISAM 采用表级锁(table-level locking)。
- InnoDB 支持行级锁(row-level locking)和表级锁,默认为行级锁

**表级锁和行级锁对比：**

- **表级锁：** MySQL 中锁定 **粒度最大** 的一种锁，对当前操作的整张表加锁，实现简单，资源消耗也比较少，加锁快，不会出现死锁。其锁定粒度最大，触发锁冲突的概率最高，并发度最低，MyISAM 和 InnoDB 引擎都支持表级锁。
- **行级锁：** MySQL 中锁定 **粒度最小** 的一种锁，只针对当前操作的行进行加锁。 行级锁能大大减少数据库操作的冲突。其加锁粒度最小，并发度高，但加锁的开销也最大，加锁慢，会出现死锁。

**InnoDB 存储引擎的锁的算法有三种：**

- Record lock：记录锁，单个行记录上的锁
- Gap lock：间隙锁，锁定一个范围，不包括记录本身
- Next-key lock：record+gap临键锁，锁定一个范围，包含记录本身

### 6. 查询缓存

`my.cnf` 加入以下配置，重启 MySQL 开启查询缓存

```properties
query_cache_type=1
query_cache_size=600000
```

MySQL 执行以下命令也可以开启查询缓存

```properties
set global  query_cache_type=1;
set global  query_cache_size=600000;
```

如上，**开启查询缓存后在同样的查询条件以及数据情况下，会直接在缓存中返回结果**。这里的查询条件包括查询本身、当前要查询的数据库、客户端协议版本号等一些可能影响结果的信息。因此任何两个查询在任何字符上的不同都会导致缓存不命中。此外，如果查询中包含任何用户自定义函数、存储函数、用户变量、临时表、MySQL 库中的系统表，其查询结果也不会被缓存。

缓存建立之后，MySQL 的查询缓存系统会跟踪查询中涉及的每张表，如果这些表（数据或结构）发生变化，那么和这张表相关的所有缓存数据都将失效。

**缓存虽然能够提升数据库的查询性能，但是缓存同时也带来了额外的开销，每次查询后都要做一次缓存操作，失效后还要销毁。** 因此，开启查询缓存要谨慎，尤其对于写密集的应用来说更是如此。如果开启，要注意合理控制缓存空间大小，一般来说其大小设置为几十 MB 比较合适。此外，**还可以通过 sql_cache 和 sql_no_cache 来控制某个查询语句是否需要缓存：**

```sql
select sql_no_cache count(*) from usr;
```

### 6. 连接池

- [数据库：数据库连接池原理详解与自定义连接池实现](https://www.fangzhipeng.com/javainterview/2019/07/15/mysql-connector-pool.html)
- [基于JDBC的数据库连接池技术研究与应用](http://blog.itpub.net/9403012/viewspace-111794/)
- [数据库连接池技术详解](https://juejin.im/post/5b7944c6e51d4538c86cf195)

### 7. 日期存储

- 不能用字符串存储日期：字符串占用的空间更大！字符串存储的日期效率比较低（逐个字符进行比对），``无法用日期相关的 API 进行计算和比较``。

#### .1. Timestamp &Datetime

- **DateTime 类型是没有时区信息的（时区无关）** ，DateTime 类型保存的时间都是`当前会话所设置的时区对应的时间`。这样就会有什么问题呢？当你的时区更换之后，比如你的服务器更换地址或者更换客户端连接时区设置的话，就会导致你从数据库中读出的时间错误。不要小看这个问题，很多系统就是因为这个问题闹出了很多笑话。

- **Timestamp 和时区有关**。Timestamp 类型字段的值`会随着服务器时区的变化而变化，自动换算成相应的时间`，说简单点就是在不同时区，查询到同一个条记录此字段的值会不一样。

建表 SQL 语句：

```sql
CREATE TABLE `time_zone_test` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `date_time` datetime DEFAULT NULL,
  `time_stamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

插入数据：

```sql
INSERT INTO time_zone_test(date_time,time_stamp) VALUES(NOW(),NOW());
```

查看数据：

```sql
select date_time,time_stamp from time_zone_test;
```

结果：

```
+---------------------+---------------------+
| date_time           | time_stamp          |
+---------------------+---------------------+
| 2020-01-11 09:53:32 | 2020-01-11 09:53:32 |
+---------------------+---------------------+
```

修改当前会话的时区:

```sql
set time_zone='+8:00';
```

再次查看数据：

```
+---------------------+---------------------+
| date_time           | time_stamp          |
+---------------------+---------------------+
| 2020-01-11 09:53:32 | 2020-01-11 17:53:32 |
+---------------------+---------------------+
```

**扩展：一些关于 MySQL 时区设置的一个常用 sql 命令**

```sql
# 查看当前会话时区
SELECT @@session.time_zone;
# 设置当前会话时区
SET time_zone = 'Europe/Helsinki';
SET time_zone = "+00:00";
# 数据库全局时区设置
SELECT @@global.time_zone;
# 设置全局时区
SET GLOBAL time_zone = '+8:00';
SET GLOBAL time_zone = 'Europe/Helsinki';
```

- Timestamp 只需要使用 4 个字节的存储空间，但是 DateTime 需要耗费 8 个字节的存储空间。但是，这样同样造成了一个问题，Timestamp 表示的时间范围更小。

  - DateTime ：1000-01-01 00:00:00 ~ 9999-12-31 23:59:59
  - Timestamp： 1970-01-01 00:00:01 ~ 2037-12-31 23:59:59

  > Timestamp 在不同版本的 MySQL 中有细微差别。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210624183128485.png)

#### .2. 数值型时间戳

- 这种存储方式的具有 Timestamp 类型的所具有一些优点，并且使用它的进行日期排序以及对比等操作的效率会更高，跨系统也很方便，毕竟只是存放的数值。缺点也很明显，就是数据的可读性太差了，你无法直观的看到具体时间。
- 时间戳的定义是从一个基准时间开始算起，这个基准时间是「1970-1-1 00:00:00 +0:00」，从这个时间开始，用整数表示，以秒计时，随着时间的流逝这个时间整数不断增加。这样一来，我只需要一个数值，就可以完美地表示时间了，而且这个数值是一个绝对数值，即无论的身处地球的任何角落，这个表示时间的时间戳，都是一样的，生成的数值都是一样的，并且没有时区的概念，所以在系统的中时间的传输中，都不需要进行额外的转换了，只有在显示给用户的时候，才转换为字符串格式的本地时间。

```sql
mysql> select UNIX_TIMESTAMP('2020-01-11 09:53:32');
+---------------------------------------+
| UNIX_TIMESTAMP('2020-01-11 09:53:32') |
+---------------------------------------+
|                            1578707612 |
+---------------------------------------+
1 row in set (0.00 sec)

mysql> select FROM_UNIXTIME(1578707612);
+---------------------------+
| FROM_UNIXTIME(1578707612) |
+---------------------------+
| 2020-01-11 09:53:32       |
+---------------------------+
1 row in set (0.01 sec)
```

### 8. 查询性能优化

#### .1. 切分大查询

>一个大查询如果一次性执行的话，可能一次锁住很多数据、占满整个事务日志、耗尽系统资源、阻塞很多小的但重要的查询。

```sql
DELEFT FROM messages WHERE create < DATE_SUB(NOW(), INTERVAL 3 MONTH);

%修改后
rows_affected = 0
do {
    rows_affected = do_query(
    "DELETE FROM messages WHERE create  < DATE_SUB(NOW(), INTERVAL 3 MONTH) LIMIT 10000")
} while rows_affected > 0
```

#### .2. 分解大查询

> 将一个大连接查询分解成对每一个表进行一次单表查询，然后将结果在应用程序中进行关联，这样做的好处有:
>
> - `让缓存更高效`。对于连接查询，如果其中一个表发生变化，那么整个查询缓存就无法使用。而分解后的多个查询，即使其中一个表发生变化，对其它表的查询缓存依然可以使用。
> - `分解成多个单表查询`，这些单表查询的缓存结果更可能被其它查询使用到，从而减少冗余记录的查询。
> - `减少锁竞争`；
> - `在应用层进行连接`，可以更容易对数据库进行拆分，从而更容易做到高性能和可伸缩。
> - 查`询本身效率也可能会有所提升`。例如下面的例子中，使用 IN() 代替连接查询，可以让 MySQL 按照 ID 顺序进行查询，这可能比随机的连接要更高效。

```sql
SELECT * FROM tab
JOIN tag_post ON tag_post.tag_id=tag.id
JOIN post ON tag_post.post_id=post.id
WHERE tag.tag='mysql';

%修改后
SELECT * FROM tag WHERE tag='mysql';
SELECT * FROM tag_post WHERE tag_id=1234;
SELECT * FROM post WHERE post.id IN (123,456,567,9098,8904);
```

### 9. 主从复制&读写分离

#### .1. 主从复制

> 主要涉及三个线程: binlog 线程、I/O 线程和 SQL 线程。
>
> - **binlog 线程** : 负责将主服务器上的数据更改写入二进制日志中。
> - **I/O 线程** : 负责从主服务器上读取二进制日志，并写入从服务器的中继日志中。
> - **SQL 线程** : 负责读取中继日志并重放其中的 SQL 语句。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210709232849857.png)

#### .2. 读写分离

- 主从服务器负责各自的读和写，极大程度缓解了锁的争用；
- 从服务器可以使用 MyISAM，提升查询性能以及节约系统开销；
- 增加冗余，提高可用性。

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/mysqlrelative/  

