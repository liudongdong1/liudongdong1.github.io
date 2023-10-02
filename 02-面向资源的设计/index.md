# 面向资源设计


2000年，为了与HTTP1.1搭配使用，[REST](https://en.wikipedia.org/wiki/Representational_state_transfer)架构风格出现。它的核心原则是定义用少量方法就能操作的命名资源。资源和方法可视为API的*名词*和*动词*。在HTTP协议中，资源名称自然地对应于URL，方法则对应于HTTP的`POST`、`GET`、`PUT`、`PATCH`和`DELETE`方法。

在互联网领域，HTTP REST API最近获得了巨大的成功。截至2010年，大约74%的公网API都是HTTP REST API。

尽管HTTP REST API在互联网领域已经很流行了，但其承载的流量还是比传统的RPC API小。比如，在高峰期，美国大约一半的互联网流量都是视频内容，出于性能上的考虑，很少有人用REST API。在数据中心内部，更多的公司也使用`基于socket的RPC API来承载大多数网络流量`，这比REST API的数量高出几个量级。

## 什么是REST API
一组REST API被建模为*一组*可独立寻址的*资源*（API的*名词*）。资源被通过[资源名称](https://cloud.google.com/apis/design/resource_names)被引用，通过一小组*方法*（也被称为*动词*或者*操作*）被操作。

Google REST API（也被称为*REST方法*）的*标准方法*有`List`，`Get`，`Create`，`Update`和`Delete`。API的设计者也可以用自定义方法（也被称为*自定义动词*或者*自定义操作*），来完成那些不易对应到标准方法的功能，比如数据库事务。

**注意**：自定义动词不意味着创建自定义的HTTP动词以支持自定义方法。对于基于HTTP的API来说，自定义动词被对应到最合适的HTTP动词上。（译者注：不能自创HTTP动词，应该使用含义最接近的HTTP动词去设计API）

## 设计流程

设计指南建议使用以下步骤设计面向资源的API（更多细节请参考下面指定章节）：
- 确定一个API`提供什么类型的资源`
- 确定`资源之间的关系`
- 根据`资源类型和关系确定资源命名方案`
- 明确`资源schema`
- `给资源添加最少的方法集`

## 资源

面向资源的API通常被建模为一个资源层次结构。其中每一个`节点可以是一个*简单资源*也可以是*一组资源`*。为了方便，它们通常被称为资源或者资源组。

- 资源组包含**相同类型**的一系列资源。比如，一个用户有一组联系人。
- 资源有相同的状态，也有零或者多个子资源。每一个子资源可以是单个资源也可是一组资源。

例如，Gmail API有一组用户，每个用户有一组消息，一组主题，一组标签，单个用户配置资源或多个设置项资源。

尽管REST API和存储系统有概念上的一致性，但具有面向资源的API的服务不一定是数据库，并且在如何解释资源和方法上有巨大的灵活性。比如，创建一个日历事件（资源）可能会为参会者创建附加事件，向参会者发送邀请邮件，预约会议室，更新视频会议日程表。

## 方法
面向资源的API的关键特性是强调资源（数据模型）和（译者注：原文看起来是笔误，这里应该翻译为和）运行在资源上的方法（功能）。典型的面向资源的API通过少量方法的暴露大量资源。这些方法可以是标准的或者自定义的。在本指南中，标准方法是指：`List`,`Get`,`Create`,`Update`和`Delete`。

当API功能自然地映射到一个标准方法时，这个方法**应当**用于API设计。若当功能不能自然的映射到标准方法时，**可以**使用*自定义方法*。[自定义方法](05-自定义方法.md)能提供与传统RPC API一样的设计自由度，可以用来实现常见的编程模式，比如数据库事务或者数据分析。

## 例子
接下来，我们看一看现实中如何使用面向资源的API来设计大规模服务。

### Gmail API
Gmail API服务实现了Gmail API，暴露了绝大多数Gmail的功能。它的资源模型如下：

- Gmail API服务：`gmail.googleapis.com`
- 一组用户：`users/*`。每个用户有如下资源。
  - 一组消息：`users/*/messages/*`。
  - 一组主题：`users/*/threads/*`。
  - 一组标签：`users/*/labels/*`。
  - 一组变更历史：`users/*/history/*`。
  - 一个代表用户资料的资源：`users/*/profile`
  - 一个代表用户配置的资源：`users/*/settings`

### Google Cloud Pub/Sub API

`pubsub.googleapis.com`服务实现了Goole Cloud Pub/Sub API，它定义了以下资源模型：
- API服务：`pubsub.googleapis.com`
- 一组主题：`projects/*/topics/*`。
- 一组订阅：`projects/*/subscriptions/*`。

**注意**：PUB/SUB API的其他实现可能选用了与上面不同的命名规则。


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/02-%E9%9D%A2%E5%90%91%E8%B5%84%E6%BA%90%E7%9A%84%E8%AE%BE%E8%AE%A1/  

