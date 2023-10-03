# sqlite


### 1. Sqlite

- [installation]( http://www.sqlite.org/download.html )
- 进入`D:/software/sqlite`目录并打开`sqlite3`命令。它将如下所示：

```shell
tar xvfz sqlite-autoconf-3071502.tar.gz
cd sqlite-autoconf-3071502
./configure --prefix=/usr/local
make
make install
```

```shell
#-----  usage
# 创建数据库
sqlite3 DatabaseName.db
#退出
sqlite> .quit 或者 sqlite> .exit
# 导出数据库
qlite3 testDB.db .dump > testDB.sql
#查看数据库 表
sqlite> .database
sqlite> .table
# 查看表定义
sqlite> .schema [表名]
# 创建表
create table student(
    id int not null primary key),
    name varchar(20) not null,
    age int not null,
    address varchar(20) not null
);
# 删除表
drop table student;
# 清空表
truncate table student;
```

```sql
#  表的增删查改
insert into student (id,name,age,address)
    values (1,"hzq",22,"china");
insert into student (id,name,age,address)
    select (id,name,age,address) from student_T;
//删除ID为1的学生信息
delete from student where id=1;
//显示来自中国的id最大的前4位
select name from student
    where address="china"
    order by id DESC
    limit 4;
update student set age=18 where id=1;
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/sqlite/  

