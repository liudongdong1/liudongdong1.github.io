# EmbeddedDB


> SQLite、Berkeley DB等属于嵌入式数据库。 嵌入式数据库跟数据库服务器最大的区别在于它们运行的地址空间不同。通常，`数据库服务器独立地运行一个守护进程（daemon）`，而`嵌入式数据库与应用程序运行在同一个进程`。
>
> - 数据库架构：数据库客户端通常`通过数据库驱动程序`如JDBC、ODBC等访问数据库服务器，数据库服务器再操作数据库文件。 数据库服务是一种客户端服务器模式，客户端和服务器是完全两个独立的进程。它们可以分别位于在不同的计算机甚至网络中。`客户端和服务器`通过TCP/IP进行通讯。
> - 嵌入式数据库架构：嵌入式数据库不需要数据库驱动程序，`直接将数据库的库文件链接到应用程序中`。应用程序通过API访问数据库，而不是TCP/IP。

### 1. [IoTDB](https://github.com/apache/iotdb)

> IoTDB (Internet of Things Database) is a data management system for `time series data`, which can provide `users specific services, such as, data collection, storage and analysis`. Due to its light weight structure, high performance and usable features together with its seamless integration with the Hadoop and Spark ecology, IoTDB meets the requirements of massive dataset storage, high throughput data input, and complex data analysis in the industrial IoT field.

1. Flexible deployment strategy. IoTDB provides users a one-click installation tool on either the `cloud platform or the terminal devices`, and a data synchronization tool bridging the data on cloud platform and terminals.
2. `Low cost on hardware`. IoTDB can reach a high compression ratio of disk storage.
3. Efficient directory structure. IoTDB supports efficient organization for complex time series data structure from intelligent networking devices, organization for time series data from devices of the same type, fuzzy searching strategy for massive and complex directory of time series data.
4. High-throughput read and write. IoTDB supports millions of low-power devices' strong connection data access, high-speed data read and write for intelligent networking devices and mixed devices mentioned above.
5. Rich query semantics. IoTDB supports time alignment for time series data across devices and measurements, computation in time series field (frequency domain transformation) and rich aggregation function support in time dimension.
6. Easy to get started. IoTDB `supports SQL-Like language, JDBC standard API and import/export tools` which is easy to use.
7. Seamless integration with state-of-the-practice Open Source Ecosystem. IoTDB supports analysis ecosystems such as, Hadoop, Spark, and visualization tool, such as, Grafana.

### 2. **Berkeley DB**

> Berkeley DB可以保存任意类型的键/值对（Key/Value Pair），而且可以为一个键保存多个数据。Berkeley DB支持让数千的并发线程同时操作数据库，支持最大256TB的数据，广泛用于各种操作系统，其中包括大多数类Unix操作系统、Windows操作系统以及实时操作系统。

1. Berkeley DB是一个开放源代码的内嵌式数据库管理系统，能够为应用程序提供高性能的数据管理服务。应用它程序员只需要调用一些简单的API就可以完成对数据的访问和管理。(不使用SQL语言)
2. Berkeley DB为许多编程语言提供了实用的API接口，包括C、C++、Java、Perl、Tcl、Python和PHP等。所有同数据库相关的操作都由Berkeley DB函数库负责统一完成。
3. Berkeley DB轻便灵活（Portable），可以运行于几乎所有的UNIX和Linux系统及其变种系统、Windows操作系统以及多种嵌入式实时操作系统之下。Berkeley DB被链接到应用程序中，终端用户一般根本感觉不到有一个数据库系统存在。
4. Berkeley DB是可伸缩（Scalable）的，这一点表现在很多方面。Database library本身是很精简的（少于300KB的文本空间），但它能够管理规模高达256TB的数据库。它支持高并发度，成千上万个用户可同时操纵同一个数据库。Berkeley DB能以足够小的空间占用量运行于有严格约束的嵌入式系统。
   Berkeley DB在嵌入式应用中比关系数据库和面向对象数据库要好，有以下两点原因： （1）因为数据库程序库同应用程序在相同的地址空间中运行，所以数据库操作不需要进程间的通讯。在一台机器的不同进程间或在网络中不同机器间进行进程通讯所花费的开销，要远远大于函数调用的开销； （2）因为Berkeley DB对所有操作都使用一组API接口，因此不需要对某种查询语言进行解析，也不用生成执行计划，大大提高了运行效率。

### 3. SQLite

1. 支持事件，不需要配置，不需要安装，也不需要管理员；
2. 一个完整的数据库保存在磁盘上面一个文件，同一个数据库文件可以在不同机器上面使用，最大支持数据库到2T，字符和BLOB的支持仅限制于可用内存；
3. 整个系统少于3万行代码，少于250KB的内存占用(gcc)，大部分应用比目前常见的客户端/服务端的数据库快，没有其它依赖
4. 源代码开放，代码95%有较好的注释，简单易用的API。官方带有TCL的编译版本。
5. 功能完善：支持ACID(Atomicity 、Consistency、Isolation、Durability）事务， Atomicity（原子性）、Consistency（一致性）、Isolation（隔离性）和Durability（持久性）是一个支持事务（Transaction）的数据库系统必需要具有的四种特性，否则在事务过程（Transaction processing）中无法保证数据的正确性，交易过程很有可能达不到交易方的要求。SQLite支持大多数的SQL92，即支持触发器、多表和索引、事务、视图，还支持嵌套的SQL。SQLite数据库存储在单一的磁盘文件中，可以使不同字节序的机器进行自由共享，支持数据库的大小可以达到2TB。

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/embeddeddb/  

