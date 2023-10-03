# 资源命名


在面向资源的API中，资源是命名实体，资源名称是资源的标识符。每个资源必须有其唯一的资源名称。资源名称由资源ID本身，父资源的ID和资源对应的API服务名称组成。在下文，我们将探讨资源ID和如何构建资源名称。

gRPC API应该使用无模式的[URIs](http://tools.ietf.org/html/rfc3986)作为资源名称。它们通常遵循REST URL的惯例并且表现得更像网络文件路径。它们可以轻松的映射到REST URL上.

资源组是一种特殊的资源，它包含一组相同类型的子资源。例如，一个目录是一组文件资源。组的资源ID被称为组ID。

资源名称使用组ID和资源ID分层组织，以斜杠（译者注：/，下同）分割。如果某个资源包含子资源，子资源的名称为：父资源名称，斜杠，子资源ID。

例1：一个存储服务有一组`bucket`，每一个bucket有一组`objects`：


| API资源名称              | 组ID     | 资源ID    | 组ID    | 资源ID    |
| ------------------------ | -------- | --------- | ------- | --------- |
| //storage.googleapis.com | /buckets | bucket-id | objects | object-id |

例2：一个邮件服务有一组`user`，每个user有子资源`settings`，settings还有子资源`customFrom`

| API资源名称           | 组ID   | 资源ID           | 组ID     | 资源ID     |
| --------------------- | ------ | ---------------- | -------- | ---------- |
| //mail.googleapis.com | /users | name@example.com | settings | customFrom |

API的提供者可以选择任何可接受的值作为资源和资源组的ID，只要它们在资源的层次结构中是唯一的即可。关于选择合适的资源ID和资源组ID，下文还有更多的参考。

只要资源名称的每段都不包含斜杠，通过分割资源名称可以获取独立的资源组ID和资源ID，比如`name.split("/")[n]`。

## 资源全名
规则松散的URI由DNS兼容的API服务名称和资源路径组成。资源路径也称为相对资源名称。

> "//library.googleapis.com/shelves/shelf1/books/book2"

API服务名称用于定位API服务端；它也**可以**是仅供内部的服务的伪DNS名称（译者注：API服务名可能用于外部调用，也可能只用于内部调用）。如果API服务名称在上下文中比较明显，相对资源名称则比较常用。

## 相对资源名称
没有前导“/”的URI路径（[路径-无模式](http://tools.ietf.org/html/rfc3986#appendix-A)），标识API服务中的一个资源。比如：
> "shelves/shelf1/books/book2"

## 资源ID
一个非空的URI片段（[segment-nz-nc](http://tools.ietf.org/html/rfc3986#appendix-A)）标识父资源中的资源。比如上例。

资源名称中的资源ID后缀**可以**有不止一个URI片段。比如：


| 资源组ID | 资源ID               |
| -------- | -------------------- |
| files    | /source/py/parser.py |

API服务**应当**尽可能的使用URL友好的资源ID。资源ID**必须**清楚的描述，不管它是由客户端、服务器端还是其他方指定。

## 资源组ID
一个非空的URI片段（[segment-nz-nc](http://tools.ietf.org/html/rfc3986#appendix-A)）标识父资源中的资源组，比如上例。

因为资源组ID经常出现在生成的客户端库里，它们**必须**符合以下要求：
- **必须**是有效的C/C++标识符。
- **必须**是驼峰命名的复数形式；首字母小写。
- **必须**使用清晰简明的英文词语。
- **应当**避免或者限定过于笼统的词语。比如，RowValue优于Value。没有限定的情况下**应当**避免以下词语：
   - Element
   - Entry
   - Instance
   - Item
   - Object
   - Resource
   - Type
   - Value

## 资源名称 vs URL
虽然完整的资源名称类似于普通URL，但他们不是一回事。单个资源可以由不同版本的API和不同API的协议暴露出来。完整资源名称没有指定API的协议和版本，在实际使用中,它**必须**被映射到特定的协议和API版本（译者注：完整资源名称）。

为了通过REST API使用资源的全名，**必须**在服务名称前添加HTTP协议，在资源路径前添加API主要版本号以及对资源路径进行URL转义，将其转换为REST URL。比如：

> //这是一个日历事件资源名称
>
>"//calendar.googleapis.com/users/john smith/events/123"
>
> 这是对应的HTTP URL
>
> "https://calendar.googleapis.com/v3/users/john%20smith/events/123"

## 作为字符串的资源名称
Google API**必须**使用字符串表示资源名称，除非向后兼容性有问题。资源名称**应当**像正常文件路径那样处理，并且他们不支持%编码。

对于资源定义来说，资源名称的第一个字段**应当**是字符串字段，并被命名为**Name**。

**注意**：其他名称相关的字段**应当**具备避免混淆的命名，比如`display\_name`, `first\_name`, `last\_name`, `full\_name`。

```java
service LibraryService {
    rpc GetBook(GetBookRequest) returns (Book) {
        option (google.api.http) = {
            get: "/v1/{name=shelves/*/books/*}"
        };
    };
    rpc CreateBook(CreateBookRequest) returns (Book) {
        option (google.api.http) = {
            post: "/v1/{parent=shelves/*}/books"
                body: "book"
        };
    };
}

message Book {
    //书的资源名称。格式必须是："shelves/*/books/"
    //比如："shelves/shelf1/books/book2"。
    string name = 1;

    // ... 其他属性
}

message GetBookRequest {
    //书的资源名称。"shelves/shelf1/books/book2"。
    string name = 1;
}

message CreateBookRequest {
    // 新建书的父资源的资源名称
    // 比如"shelves/shelf1".
    string parent = 1;
    // 要创建的书籍资源，客户端绝不能设置‘Book.name’属性
    Book book = 2;
}
```
**注意**：为了资源名称的统一，开头的斜杠**绝不能**让URL模板变量捕获。例如，**必须**使用URL模板"/v1/{name=shelves/\*/books/\*}"而不能使用"/v1{name=/shelves/\*/books/\*}".


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/03-%E8%B5%84%E6%BA%90%E5%91%BD%E5%90%8D/  

