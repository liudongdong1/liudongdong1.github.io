# DatabaseTool


### 1.[dbeaver](https://github.com/dbeaver/dbeaver)

Free multi-platform database tool for developers, SQL programmers, database administrators and analysts.
Supports any database which has JDBC driver (which basically means - ANY database). EE version also supports non-JDBC datasources (MongoDB, Cassandra, Couchbase, Redis, BigTable, DynamoDB, etc).

- Has a lot of [features](https://github.com/dbeaver/dbeaver/wiki) including metadata editor, SQL editor, rich data editor, ERD, data export/import/migration, SQL execution plans, etc.
- Based on [Eclipse](https://wiki.eclipse.org/Rich_Client_Platform) platform.
- Uses plugins architecture and provides additional functionality for the following databases: MySQL/MariaDB, PostgreSQL, Greenplum, Oracle, DB2 LUW, Exasol, SQL Server, Sybase/SAP ASE, SQLite, Firebird, H2, HSQLDB, Derby, Teradata, Vertica, Netezza, Informix, etc.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210322090704510.png)

### 2. [sql2nosql](https://github.com/facundopadilla/sql2nosql)

```python
from sql2nosql import Migrator
host = "0.0.0.0"
sql_config = {
    "host": host,
    "port": 33060,
    "username": "root",
    "password": "1234",
    "database": "classicmodels",
}
nosql_config = {
    "host": host,
    "port": 27018,
    "username": "sql2nosql",
    "password": "1234",
}
migrator = Migrator(
    sql_config=sql_config,
    nosql_config=nosql_config,
    sql_client="mysql.connector",
    nosql_client="pymongo",
)
migrator.migrate_data(tables=["customers", "employees", "offices"])
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/databasetool/  

