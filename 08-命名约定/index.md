# 命名约定


为了在跨API开发中向开发者提供一致的开发体验，所有的命名**应该**保证：

* 简单
* 直观
* 一致

这适用于接口、资源、集合、方法以及消息的命名。

因为很多开发者并非以英语作为母语，所以命名约定的目标之一是确保大多数开发者可以更容易理解 API。对于方法和资源，我们鼓励使用简单、直观和一致的单词来命名。

* API 中的命名**应该**使用正确的美式英语。例如，使用美式英语的 license 而非英式英语的 licence；使用美式英语的 color 而非英式英语的 colour
* 为了简化起见，**可以**使用已被广泛接受的短语和（或）缩写。例如，使用 API 比Application Programming Interface更好
* 尽量使用直观、熟悉的术语。例如，描述移除（removing）或销毁（destroying）一个资源时，使用删除（delete）比擦除（erase）更好
* 为了避免歧义，对于同一种概念应当使用相同的名称或短语，这一点适用于跨越多个不同API的概念
* 为了避免歧义，对于不同的概念应当使用不同的名称或短语
* 为了避免在 API 的上下文以及更大的 Google API 生态系统中存在含糊不清和过于笼统的名称（它们可能导致对 API 概念的误解）。应当选择能准确描述 API 概念的名称，这对定义 API 中的资源名称特别重要。
* 应当仔细思考那些**可能**与常用编程语言中关键字冲突的名称。



## 产品名称


产品名称应当参考那些API的公开产品名称，例如 Google Calendar API。同时，在API、用户界面、文档、服务协议、收费声明以及商业合同中，都**应该**使用一致的产品名称。

Google 的 API **必须**使用以 *Google* 开头的产品名称，除非产品所属品牌不同。例如，Gmail、YouTube等。通常来说，产品名称**应该**由产品和市场团队来确定。

下面这个表格以我们的产品为例展示了命名的一致性。您也可以通过本页附录中的链接获取更多信息。

|    API 名称    |                  示例                  |
| -------------- | -------------------------------------- |
| **产品名称**   | Google **Calendar** API                |
| **服务名称 **  | **calendar.googleapis.com**            |
| **包名称**     | **google.calendar.v3**                 |
| **接口名称**   | **google.calendar.v3.CalendarService** |
| **源代码目录** | **//google/calendar/v3**               |
| **API 名称**   | **calendar**                           |


## 服务名称

服务名称**应当**符合 DNS 命名的语法规范（请参考 [RFC 1035 ](http://www.ietf.org/rfc/rfc1035.txt)规范），确保它可以被解析成一个或多个网络地址。公开的 Google API 的服务名称遵循这样的规则：```xxx.googleapis.com``` 。例如，Google Calendar 的服务名称是 ```calendar.googleapis.com``` 。

如果一个 API 是由多个服务组成，那么它们的命名**应当**更容易被发现。 一种方法是为所有服务名称共享一个公共前缀，例如，服务 ```build.googleapis.com``` 和 ```buildresults.googleapis.com``` 都是 Google Build API 的一部分。



## 包名称

在 API.proto 文件中声明包名称，它**应当**与产品名称和服务名称保持一致。同时 API 的包名称**必须**包含版本信息，例如：

```
// Google Calendar API
package google.calendar.v3;
```

抽象的 API 并不直接关联服务，例如 Google Watcher API，**应当**使用与产品名称一致的 proto 包名称：

```
// Google Watcher API
package google.watcher.v1;
```

在 API.proto 中定义的 Java 包名称**必须**与具有标准 Java 包名称前缀（```com.``` 、```edu.``` 、```net.``` 等）的 proto 包名称相匹配。例如：

```
package google.calendar.v3;

// Specifies Java package name, using the standard prefix "com."
option java_package = “com.google.calendar.v3";
```



## 集合标识符

[集合命名](03-资源命名.md#资源组ID)**应该**使用复数形式、首字母小写驼峰体，并使用标准的美式英语。例如：```events```，```children``` 或 ```deletedEvents```。



## 接口名称

为了避免与[服务名称](08-命名约定#服务名称)混淆，比如 ```pubsub.google.apis.com``` ，这里的接口名称是指在 .proto 文件中定义服务（```service```）时使用的名称：

```
// Library is the interface name.
service Library {
  rpc ListBooks(...) returns (...);
  rpc ...
}
```

你可以将服务名称看作是对一组 API 实现的引用，而接口名称则是 API 的抽象定义。

接口名称**应当**使用直观准确的名词，例如 Calendar 或 Blob。同时名称**不应当**与主流编程语言及其运行时库中的任何概念相冲突。

在极少的情况下，接口名称可能会与 API 中的其它名称冲突，这时**应当**使用后缀（```Api``` 或 ```Service```）来消除歧义。



## 方法名称

服务**可以**在其 IDL 规范中定义 RPC 方法，用来对应集合或资源上的方法。方法名称**应当**使用首字母小写驼峰体的动名词，并且通常这里的名词就是资源类型。

|    动词    |   名词   |    方法名称    |         请求          |           响应            |
| ---------- | -------- | -------------- | --------------------- | ------------------------- |
| **List**   | **Book** | **ListBooks**  | **ListBooksRequest**  | **ListBooksResponse**     |
| **Get**    | **Book** | **GetBook**    | **GetBookRequest**    | **Book**                  |
| **Create** | **Book** | **CreateBook** | **CreateBookRequest** | **Book**                  |
| **Update** | **Book** | **UpdateBook** | **UpdateBookRequest** | **Book**                  |
| **Rename** | **Book** | **RenameBook** | **RenameBookRequest** | **RenameBookResponse**    |
| **Delete** | **Book** | **DeleteBook** | **DeleteBookRequest** | **google.protobuf.Empty** |



## 消息名称

RPC 方法的请求和响应消息**应该**分别以带有后缀 ```Request``` 和 ```Response``` 的方法名来命名，除非请求和响应的类型为：

* 一个空消息（使用 ```google.protobuf.Empty```)
*  一种资源
* 一种表示操作的资源

这通常适用于使用标准 ```Get```、```Create```、```Update``` 或 ```Delete``` 方法的请求和响应。



## 枚举名称

枚举类型的名称**必须**使用首字母大写驼峰体。

枚举值**必须**使用“以下划线分隔的全大写”（CAPITALIZED_NAME_WITH_UNDERSCORES）方式命名，每个枚举值**必须**以分号（；）而非逗号（，）结尾。并且第一个枚举值**应该**以“枚举_类型_未指定”（ENUM_TYPE_UNSPECIFIED）的形式命名，该值用于枚举值没有显式指定时的默认值。

```
enum FooBar {
  // The first value represents the default and must be == 0.
  FOO_BAR_UNSPECIFIED = 0;
  FIRST_VALUE = 1;
  SECOND_VALUE = 2;
}
```



## 字段名称

在 .proto 文件中的定义字段时，**必须**使用“以下划线分隔的全小写方式”（lower_case_underscore_separated_names）命名。针对不同的编程语言，这些名称将根据命名规范/惯例被映射到自动生成代码中。



### 重复字段名称

API 中的重复字段**必须**使用正确的复数形式。 这符合现有的 Google API 惯例，以及外部开发人员的普遍认知。



### 瞬时时间和持续时间

要表示一个与任何时区或日历无关的瞬时时间，**应当**使用 ```google.protobuf.Timestamp``` 类型，并且字段名称**应当**以时间```time```结束，例如 ```start_time``` 和 ```end_time```。

如果瞬态时间代表一个活动或行为，则字段名称**应当**采用“动词_时间”（```verb_time```）的形式，例如 ```create_time```，```update_time```。 同时注意避免使用动词的过去时态，例如 ```created_time``` 或 ```last_updated_time```。

要表示两个与任何日历和“天”/“月”无关的瞬时时间点之间的时间跨度，**应当**使用 google.protobuf.Duration 类型。

```
message FlightRecord {
  google.protobuf.Timestamp takeoff_time = 1;
  google.protobuf.Duration flight_duration = 2;
}
```

如果由于历史遗留或兼容性原因，不得不使用整数类型表示与时间相关的字段，那么字段名称**必须**采用以下格式：

```
xxx_{time|duration|delay|latency}_{seconds|millis|micros|nanos}
```

```
message Email {
  int64 send_time_millis = 1;
  int64 receive_time_millis = 2;
}
```

如果由于历史遗留或兼容性原因，不得不使用字符串类型表示时间戳，那么字段名称**不应包**含任何时间单位后缀，同时字符串**应该**使用 RFC 3339 格式。例如，“2014-07-30T10:43:17Z”。



### 日期和时刻

对于与任何时区和时刻无关的日期，**应当**使用 ```google.type.Date``` 类型，并使用“\_日期”（```_date```）后缀。如果日期必须表示为字符串类型，则应该使用 ISO 8601日期格式 YYYY-MM-DD。例如，2014-07-30。

对于与任何时区和日期无关的时段，**应当**使用 ```google.type.TimeOfDay``` 类型，并使用“\_时间”（```_time```）后缀。 如果时刻必须表示为字符串类型，则应该使用 ISO 8601 24时格式 HH：MM：SS [.FFF]。例如，14：55：01.672。

```
message StoreOpening {
  google.type.Date opening_date = 1;
  google.type.TimeOfDay opening_time = 2;
}
```



### 数量

由整数类型表示的数量**必须**包含计量单位。

```
xxx_{bytes|width_pixels|meters}
```

如果该数量代表多个内容的计数，则字段名城**应该**包含“\_计数” （```_count```）后缀。例如，```node_count```。



### 列表过滤器字段

如果 API 支持对 ```List``` 方法返回的资源进行过滤，那么包含过滤器表达式的字段**应该**命名为 ```filter```。 例如：

```
message ListBooksRequest {
  // The parent resource name.
  string parent = 1;

  // The filter expression.
  string filter = 2;
}
```



### 列表响应

```List``` 方法的响应消息中包含资源列表的字段名称**必须**是资源名称本身的复数形式。例如，```CalendarApi.ListEvents()``` 方法**必须**为返回的资源列表定义一个响应消息 ```ListEventsResponse```，其中包含一个称为 ```events``` 的重复字段。

```
service CalendarApi {
  rpc ListEvents(ListEventsRequest) returns (ListEventsResponse) {
    option (google.api.http) = {
      get: "/v3/{parent=calendars/*}/events";
    };
  }
}

message ListEventsRequest {
  string parent = 1;
  int32 page_size = 2;
  string page_token = 3;
}

message ListEventsResponse {
  repeated Event events = 1;
  string next_page_token = 2;
}
```



## 驼峰体

除了字段名称和枚举值，```.proto``` 文件中的所有名称都**必须**使用 [Google Java Style](https://google.github.io/styleguide/javaguide.html#s5.3-camel-case) 定义的首字母大写驼峰体（UpperCamelCase）。



## 名称缩写

对于软件开发人员熟知的名称缩写，例如 ```config``` 和 ```spec```，建议在 API 定义中**应该**使用缩写，而不是完整的名称。 这将使源代码易于读写。而在正式文档中，**应当**使用完整拼写的名称。例如：

* 配置（config/configuration）
* 标识符（id/identifier）
* 规格（spec/specification）
* 统计（stats/statistics）


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/08-%E5%91%BD%E5%90%8D%E7%BA%A6%E5%AE%9A/  

