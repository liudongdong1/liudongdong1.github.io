# 标准方法


本章阐述标准方法的概念，包括了`List`, `Get`, `Create`, `Update`, and `Delete`。很多不同类型的API都拥有非常类似的语义，把它们归纳为标准方法能够显著降低复杂度并提高一致性。在[谷歌API](https://github.com/googleapis/googleapis)仓库中，超过70%的API属于标准方法。标准方法更容易学习和使用。以下表格描述了如何将标准方法映射为REST方法，也就是所谓的CRUD方法：

|       方法        |       HTTP 方法映射        | HTTP 请求体 | HTTP 返回体 |
| ----------------- | -------------------------- | ----------- | ----------- |
| [List](#list)     | **GET <集合URL>**          | 空          | 资源\* 列表 |
| [Get](#get)       | **GET <资源URL>**          | 空          | 资源\*      |
| [Create](#create) | **POST <集合URL>**         | 资源        | 资源\*      |
| [Update](#update) | **PUT or PATCH <资源URL>** | 资源        | 资源\*      |
| [Delete](#delete) | **DELETE <资源URL>**       | 空          | 空\*\*      |

如果方法支持字段掩码并指定要返回字段的子集时，，从`List`，`Get`，`Create`和`Update`方法返回的资源**可能**包含部分数据。在一些情况下，API平台的所有方法都支持字段掩码。

Delete方法如果并没有立刻删除响应的资源（例如创建一个耗时删除操作或者更新标识），它的响应**应该**包括耗时操作（译者注：耗时操作可看做是对服务器端长时间运行过程的抽象。因为运行过程耗时长。为了不阻塞客户端同时给客户端跟踪运行状况的机会，可以先给调用方返回一个对象，这个对象对应服务器端的执行过程，可以用它获取远程操作的状态和结果）或更新后的资源。

如果请求无法在单个API调用时间段内完成时，标准方法**可以**返回一个[耗时操作](https://github.com/googleapis/googleapis/blob/master/google/longrunning/operations.proto)。

以下章节描述了各标准方法的细节。范例中使用 .proto 文件定义方法，HTTP映射则通过特殊注释表明。你会发现[Google APIs](https://github.com/googleapis/googleapis)仓库中有很多使用标准方法的案例。

## List

`List`方法接受一个集合名，零或者多个参数，根据输入返回相应的资源列表。它也经常被用作搜索资源。

`List`适用于量有限且无缓存的单一集合数据查询；若需要更广的应用，**应该**[用自定义方法](05-自定义方法.md)`Search`。

批量获取（如接受多个资源ID并返回每个ID对象的方法）应该使用自定义方法`BatchGet`实现，而不是`List`方法。但如果你已经有了提供相似功能的`List`方法，你也**可以**继续使用。如果你使用自定义的`BatchGet`方法，**应该**确保它映射为HTTP GET方法。

使用常见模式：[分页](09-通用设计模式.md#列表分页)，[结果排序](09-通用设计模式.md#排序顺序)。

适用命名约定：[过滤字段](08-命名约定.md#列表过滤器字段)，[结果字段](08-命名约定.md#列表响应)。

### HTTP 映射

* List方法**必须**使用HTTP `Get`方法。
* 请求消息字段接收资源名称集合，而相关资源**应该**映射为URL路径。如果集合名称映射到URL路径，URL模板中的最后一段（即集合ID）**必须**为文字。
* 其他所有请求消息字段**应当**映射为URL请求参数（译者注：大意是资源名字段映射到URL路径，其他映射到请求参数，参考[统一资源定位符](https://zh.wikipedia.org/wiki/%E7%BB%9F%E4%B8%80%E8%B5%84%E6%BA%90%E5%AE%9A%E4%BD%8D%E7%AC%A6）)中。
* 没有请求体，即API配置中**不应该**声明请求体。
* 返回体**应该**包含资源集合以及可选的元数据。

```go
// 列举给定书架上的所有书
rpc ListBooks(ListBooksRequest) returns (ListBooksResponse) {
  // List方法映射为HTTP GET。
  option (google.api.http) = {
    // 在`parent`指定父级资源名，如"shelves/shelf1"。
    get: "/v1/{parent=shelves/*}/books"
  };
}

message ListBooksRequest {
  // 父级资源名，如"shelves/shelf1"
  string parent = 1;

  // 返回的最大数据条数
  int32 page_size = 2;

  // 上一次List请求返回的next_page_token值，如果有的话
  string page_token = 3;
}

message ListBooksResponse {
  // 字段名必须跟方法名中的"books"一致
  // 数据返回的最大条数由请求中的page_size属性值制定
  repeated Book books = 1;

  // 获取下一页数据的令牌，如果没有更多数据则为空.
  string next_page_token = 2;
}
```

## Get

`Get`方法接受一个资源名，零到多个参数，返回指定的资源。

### HTTP 映射

* `Get`方法**必须**使用HTTP Get方法。
* 接收资源名称的请求消息字段（可多个）**应该**映射到URL路径中。
* 其他所有请求消息字段**应当**映射到URL查询参数中。
* 无请求体，即API配置中**绝对不可以**出现请求体声明。
* 返回资源**应当**映射到返回体中。

```go
// 获得指定书籍
rpc GetBook(GetBookRequest) returns (Book) {
  // Get映射为HTTP GET，资源名绑定到URL中，无请求体
  option (google.api.http) = {
    // 注意URL模板中有多个片段包括多个变量，以指定书籍相应的不同资源名，例如：
    // "shelves/shelf1/books/book2"
    get: "/v1/{name=shelves/*/books/*}"
  };
}

message GetBookRequest {
  // 请求的资源名，如：
  // "shelves/shelf1/books/book2"
  string name = 1;
}
```

## Create

`Create`方法接受一个集合名，一个资源，并且有零或多个参数；然后在相应集合中创建新的资源，最后返回新创建的资源。

如果API支持创建资源，那么它**应该**有`Create`方法用于创建各种类型的资源。

### HTTP 映射

* `Create`方法**必须**使用HTTP POST方法。
* 请求消息**应该**有一个名为`parent`的字段，以接受父级资源名，当前资源将在父资源下创建。
* 其他所有请求消息字段**应当**映射到URL查询参数中。
* 请求**可以**包括一个名为\<resource\>_id的字段，以允许调用方选择客户端分配的ID（译者注：Create方法创建的资源id可以是客户端生成的）；此字段必须映射为URL查询参数。
* 包含资源的请求消息字段**应该**映射到请求体中，如果HTTP子句用于Create方法，则必须使用：\<resource_field\>的表单。
* 返回的资源**应当**映射到整个返回体中。


如果`Create`方法支持客户端指定资源名，并且相应资源已经存在；那么它**应该**返回错误（**推荐**使用google.rpc.Code.ALREADY_EXISTS错误代码），或使用其它服务器指定的资源名：文档中需要清晰说明被创建的资源名可能跟传入的资源名不同。

```go
rpc CreateBook(CreateBookRequest) returns (Book) {
  // Create映射为HTTP POST，集合名映射到URL路径
  // HTTP请求体包含资源
  option (google.api.http) = {
    // 在`parent`指定父级资源名，如"shelves/shelf1"。
    post: "/v1/{parent=shelves/*}/books"
    body: "book"
  };
}

message CreateBookRequest {
  // 待创建book资源所属的父级资源名。
  string parent = 1;

  // 此书的ID
  string book_id = 3;

  // 待创建的book资源
  // 字段名必须跟方法名中名词一致
  Book book = 2;
}

rpc CreateShelf(CreateShelfRequest) returns (Shelf) {
  option (google.api.http) = {
    post: "/v1/shelves"
    body: "shelf"
  };
}

message CreateShelfRequest {
  Shelf shelf = 1;
}
```

## Update

`Update`方法接受包括一个资源的请求消息，并且有零或多个参数。它更新相应资源以及它的属性；返回更新后的资源。

可变的资源属性**应当**被`Update`方法修改，除非属性包含资源的名称或者父资源。所有的重命名或者移动资源操作**一定不能**用`Update`方法，这些**应当**用自定义方法处理。

### HTTP 映射

* 标准的`Update`方法**应该**支持部分资源更新，并使用 HTTP `PATCH`方法以及名为`update_mask`的`FieldMask`字段。
* 如果`Update`方法需要更高级的修复语义，比方说给重复字段增补新值，那么**应该**使用[自定义方法](05-自定义方法.md)。
* 如果`Update`方法仅支持完整的资源更新，它**必须**使用HTTP `PUT`；但是强烈不推荐这么做，因为这会造成添加新资源字段时的兼容性问题。
* 接受资源名的字段**必须**映射到URL路径中；字段也**可以**包含在资源消息中。
* 包含资源的请求消息中字段**必须**映射到请求体中。
* 其他所有请求消息字段**必须**映射到URL查询参数中。
* 返回的结果*必须*是更新后的资源。

如果API允许客户端指定资源名，服务器**可以**允许客户端指定一个不存在的资源名并创建新的资源。否则，使用不存在的资源名时`Update`方法应该报错。如果不存在资源是唯一的错误条件，那么错误码**应该**用`NOT_FOUND`。

API如果有`Update`方法，并且支持资源创建的话，就应该提供`Create`方法；以避免调用者误以为`Update`方法是创建资源的唯一方式。

```go
rpc UpdateBook(UpdateBookRequest) returns (Book) {
  // Update 映射为HTTP PATCH。资源名映射到URL路径。
  // HTTP请求提包含资源
  option (google.api.http) = {
    // 注意URL模板中的变量指定了待更新的book资源名
    patch: "/v1/{book.name=shelves/*/books/*}"
    body: "book"
  };
}

message UpdateBookRequest {
  // 用于更新服务器上资源的book数据
  Book book = 1;

  // 用于更新资源的掩码
  FieldMask update_mask = 2;
}
```

## Delete

`Delete`方法接受一个资源名，零或多个参数；然后删除，或者安排删除相应的资源。`Delete`方法**应该**返回`google.protobuf.Empty`。

注意API**不应该**依赖于`Delete`方法返回的任何信息，因为它**不能**被反复调用（译者注：因为资源可能已经被删除了）。

### HTTP 映射

* `Delete`方法**必须**使用HTTP `DELETE`方法。
* 对应于资源名称的请求消息字段（可多个）**应该**绑定到URL路径中。
* 其他所有请求消息字段**应当**映射到URL查询参数中。
* 无请求体，即API配置中**绝对不可以**出现请求体声明。
* 如果`Delete`方法立刻删除除资源，它**应该**返回空。
* 如果`Delete`方法开启了一个耗时操作，它**应该**返回这个耗时操作（译者注：本节开始的译者注中提到了耗时操作）。
* 如果`Delete`方法仅是把资源标记为删除，它需要返回更新后的资源

调用`Delete`方法必须是幂等的，但返回值可以不同。任意次数的`Delete`请求应当使得一个资源(最终)被删除，但只有第一次请求获得成功的返回值，后续的请求应当返回`google.rpc.Code.NOT_FOUND`.

```go
rpc DeleteBook(DeleteBookRequest) returns (google.protobuf.Empty) {
  // Delete 映射为HTTP DELETE方法，资源名绑定到URL路径中。
  // 没有请求体。
  option (google.api.http) = {
    // 注意URL模板中有多个片段包括多个变量，以指定待删除书籍相应的不同资源名，例如：
    // "shelves/shelf1/books/book2"
    delete: "/v1/{name=shelves/*/books/*}"
  };
}

message DeleteBookRequest {
  // 等待删除的book数据资源名，如：
  // "shelves/shelf1/books/book2"
  string name = 1;
}
```


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/04-%E6%A0%87%E5%87%86%E6%96%B9%E6%B3%95/  

