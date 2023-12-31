# MongoDBRelative


> MongoDB是面向文档的NoSQL数据库，用于大量数据存储。MongoDB是一个在2000年代中期问世的数据库。属于NoSQL数据库的类别。

> From: https://www.pdai.tech/md/db/nosql-mongo/mongo-x-basic.html

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210710221942311.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210710222212524.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210710222422472.png)

#### 1. 概念介绍

#### .1.  特点

- **面向文档的**–由于MongoDB是NoSQL类型的数据库，它不是以关系类型的格式存储数据，而是将数据存储在文档中。这使得MongoDB非常灵活，可以适应实际的业务环境和需求。
- **临时查询**-MongoDB支持按字段，范围查询和正则表达式搜索。可以查询返回文档中的特定字段。
- **索引**-可以创建索引以提高MongoDB中的搜索性能。MongoDB文档中的任何字段都可以建立索引。
- **复制**-MongoDB可以提供副本集的高可用性。副本集由两个或多个mongo数据库实例组成。每个副本集成员可以随时充当主副本或辅助副本的角色。主副本是与客户端交互并执行所有读/写操作的主服务器。辅助副本使用内置复制维护主数据的副本。当主副本发生故障时，副本集将自动切换到辅助副本，然后它将成为主服务器。
- **负载平衡**-MongoDB使用分片的概念，通过在多个MongoDB实例之间拆分数据来水平扩展。MongoDB可以在多台服务器上运行，以平衡负载或复制数据，以便在硬件出现故障时保持系统正常运行。

#### .2. 常用术语

- **_id** – 这是每个MongoDB文档中必填的字段。_id字段表示MongoDB文档中的唯一值。_id字段类似于文档的主键。如果创建的新文档中没有_id字段，MongoDB将自动创建该字段。
- **集合** – 这是MongoDB文档的分组。`集合等效于在任何其他RDMS（例如Oracle或MS SQL）中创建的表`。集合存在于单个数据库中。从介绍中可以看出，集合不强制执行任何结构。
- **游标** – 这是`指向查询结果集的指针`。客户可以遍历游标以检索结果。
- **数据库** – 这是像RDMS中那样的集合容器，其中是表的容器。每个数据库在文件系统上都有其自己的文件集。MongoDB服务器可以存储多个数据库。
- **文档** - MongoDB集合中的记录基本上称为文档。文档包含`字段名称`和`值`。
- **字段** - 文档中的名称/值对。一个文档具有零个或多个字段。字段类似于关系数据库中的列。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210710223113335.png)

### 2. 安装使用

- [下载地址](https://www.mongodb.com/try/download/community)： https://www.mongodb.com/try/download/community
- 配制文件:  mongodb\mongod.cfg

```yaml
systemLog:
  destination: file
  path: "D:/software/mongodb/log/mongod.log"
  logAppend: true
storage:
  journal:
    enabled: true
  dbPath: "D:/software/mongodb/data/db"
net:
  bindIp: 0.0.0.0
  port: 27017
setParameter:
  enableLocalhostAuthBypass: false
```

- 制作系统服务：

```shell
mongod --config "D:\software\mongodb\mongod.cfg" --bind_ip 0.0.0.0 --install

#或者命令行方式启动
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mongodbrelative/  

