# Beetlsql


> BeetlSQL 不仅仅是简单的类似MyBatis或者是Hibernate，或者是俩着的综合，BeetlSQL目的是对标甚至超越Spring Data，是实现数据访问统一的框架，无论是传统数据库，还是大数据，还是查询引擎或者时序库，内存数据库。
>
> - 传统数据库：MySQL, MariaDB, Oralce, Postgres, DB2, SQL Server，H2, SQLite, Derby，神通，达梦，华为高斯，人大金仓，PolarDB等
> - 大数据：HBase，ClickHouse，Cassandar，Hive
> - 物联网时序数据库：Machbase，TD-Engine，IotDB
> - SQL查询引擎:Drill,Presto，Druid
> - 内存数据库:ignite，CouchBase

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210629112813645.png)

```xml
<modules>
<!--核心功能 -->
<module>sql-core</module>
<module>sql-mapper</module>
<module>sql-util</module>
<module>sql-fetech</module>
<!-- 打包到一起 -->
<module>beetlsql</module>
<module>sql-gen</module>
<module>sql-test</module>
<module>sql-samples</module>
<!-- 集成和扩展太多的数据库,可以被屏蔽，以加速项目下载jar -->
<!--		<module>sql-integration</module>-->
<!--    <module>sql-jmh</module>-->
<!--		<module>sql-db-support</module>-->
</modules>
```

以`sql-samples`为例子

sql-samples 又包含了三个模块大约100个例子

- quickstart: BeetlSQL基础使用例子，可以快速了解BeetlSQL3
- usuage: BeetlSQL所有API和功能
- plugin:BeetlSQL高级扩展实例

以usage模块为例子，包含如下代码

- S01MapperSelectSample 15个例子， mapper中的查询演示
- S02MapperUpdateSample 11个例子， mapper中更新操作
- S03MapperPageSample 3个例子，mapper中的翻页查询
- S04QuerySample 9个例子，Query查询
- S05QueryUpdateSample 3个例子，Query完成update操作
- S06SelectSample 14个例子，SQLManager 查询API
- S07InsertSample 8个例子，SQLManager 插入新数据API,主键生成
- S08UpdateSample 6个例子,更新数据
- S09JsonMappingSample 5个例子， json配置映射
- S10FetchSample 2个例子，关系映射
- S11BeetlFunctionSample 2个例子，自定义sql脚本的方法

### Resource

- https://gitee.com/xiandafu/beetlsql

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/beetlsql/  

