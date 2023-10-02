# 错误资源命名


## 错误模型

错误模型在逻辑上由[`google.rpc.Status`](https://github.com/googleapis/googleapis/blob/master/google/rpc/status.proto)定义，当API发生错误时，返回一个Status实例给客户端。 以下代码段显示了错误模型的总体设计：

```go
package google.rpc;

message Status {
  // A simple error code that can be easily handled by the client. The
  // actual error code is defined by `google.rpc.Code`.
  int32 code = 1;

  // A developer-facing human-readable error message in English. It should
  // both explain the error and offer an actionable resolution to it.
  string message = 2;

  // Additional error information that the client code can use to handle
  // the error, such as retry delay or a help link.
  repeated google.protobuf.Any details = 3;
}
```

由于大多数Google API都使用面向资源的API设计，因此遵循相同的错误处理设计原则（即用一小组标准错误处理有大量资源）。 例如，服务器使用一个标准的`google.rpc.Code.NOT_FOUND`错误代码，而不是定义不同类型的“not found”错误，并告诉客户端找不到特定资源。 较小的状态空间降低了文档的复杂性，也在客户端库中提供更好的惯用映射，并降低客户端逻辑复杂性，而且没有限制包含的操作信息。

## 错误代码

Google API**必须**使用[`google.rpc.Code`](https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto)定义的规范错误代码。 因为开发人员不太可能编写大量处理逻辑错误的代码，所以单独的API**应该**避免定义额外的错误代码。 作为参考，每个API调用平均处理3个错误就意味着大多数应用程序只是处理错误了，这对开发者来说体验不够友好。

## 错误消息

错误消息应该帮助用户轻松，快速地**理解和解决**API错误。 一般来说，在编写错误消息时，请考虑以下准则：

* 不要假定用户是API专家。 用户可以是客户端开发人员，操作人员，IT人员或应用程序的最终用户。
* 不要假定用户了解服务实现或熟悉错误的上下文（如日志分析）。
* 如果可能，应构造错误消息，以便技术用户（但不一定是您的API开发人员）可以响应错误并进行更正。
* 保持`错误消息简练`。 如果需要请提供链接，这样困惑的读者可以提出问题，提供反馈或获取更多信息（这些信息不一定适合在错误消息中展示）。如果不合适，就可以使用详细信息字段展开。

## 错误细节

Google API为错误详细信息定义了一组标准错误有效内容，您可以在[`google/rpc/error_details.proto`](https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto)中找到它。它涵盖了API错误的最常见需求，例如配额失败和无效参数。与错误代码一样，错误详细信息应尽可能使用这些标准有效内容。只有详细信息能够帮助应用程序代码处理错误的时候，才应该引入其他的错误详细信息类型。如果错误信息只能由人来处理，请依赖于错误消息内容，并让开发人员手动处理它，而不是引入新的错误详细信息类型。请注意，如果引入了其他错误详细信息类型，则必须显式注册它们。

下面是一些示例`error_details`有效内容：

* `RetryInfo`描述客户端何时可以重试失败的请求，可以返回`Code.UNAVAILABLE`或`Code.ABORTED`
* `QuotaFailure`描述配额检查如何失败，可以返回`Code.RESOURCE_EXHAUSTED`
* `BadRequest`描述客户端请求中的违例，可以返回`Code.INVALID_ARGUMENT`

## HTTP映射

虽然proto3消息支持原生JSON编码，Google API Platform使用以下JSON表示形式来处理直接HTTP-JSON错误响应，以保证向后兼容性：

```json
{
  "error": {
    "code": 401,
    "message": "Request had invalid credentials.",
    "status": "UNAUTHENTICATED",
    "details": [{
      "@type": "type.googleapis.com/google.rpc.RetryInfo",
    }]
  }
}
```


   字段     |                                  描述
----------- | -----------------------------------------------------------------------
**error**   | 用于向后兼容Google API客户端库的额外层。 它使用JSON来标示以便人类阅读。
**code**    | **Status.code**的HTTP状态代码映射
**message** | 这对应于**Status.message**
**status**  | 这对应于**Status.status**
**details** | 这对应于**Status.details**

## RPC 映射

不同的RPC协议以不同的方式映射错误模型。 对于[gRPC](http://www.grpc.io/)，错误模型由生成的代码和每种语言的运行时库原生支持。 您可以在gRPC的API文档中找到更多信息（例如，请参阅gRPC Java的[`io.grpc.Status`](https://github.com/grpc/grpc-java/blob/master/core/src/main/java/io/grpc/Status.java)）。

## 客户端库映射

Google客户端库可能会选择以不同的语言显示错误，以保持与已建立的习语一致。 例如，[google-cloud-go](https://github.com/GoogleCloudPlatform/google-cloud-go)库将返回实现[`google.rpc.Status`](https://github.com/googleapis/googleapis/blob/master/google/rpc/status.proto)相同接口的error，而[google-cloud-java](https://github.com/GoogleCloudPlatform/google-cloud-java)将抛出异常。

## 错误本地化

[`google.rpc.Status`](https://github.com/googleapis/googleapis/blob/master/google/rpc/status.proto)中的`message`字段面向开发人员，**必须**使用英语。

如果需要面向用户的错误消息，请使用[`google.rpc.LocalizedMessage`](https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto)作为您的详细信息字段。 虽然[`google.rpc.LocalizedMessage`](https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto)中的message字段可以本地化，但请确保[`google.rpc.Status`](https://github.com/googleapis/googleapis/blob/master/google/rpc/status.proto)中的消息字段为英语。

默认情况下，API服务应使用经过身份验证的用户语言环境或HTTP `Accept-Language`头来确定本地化的语言。

## 处理异常

下面是一个表格，其中包含[`google.rpc.Code`](https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto)中定义的所有gRPC错误代码及其原因的简短说明。 要处理错误，您可以检查返回状态码的描述，并相应地修改您的请求。

HTTP |           RPC           |                                                  描述
---- | ----------------------- | ------------------------------------------------------------------------------------------------------
200  | **OK**                  | 没有错误
400  | **INVALID_ARGUMENT**    | 客户端指定了无效的参数。 检查错误消息和错误详细信息以获取更多信息。
400  | **FAILED_PRECONDITION** | 请求不能在当前系统状态下执行，例如删除非空目录。
400  | **OUT_OF_RANGE**        | 客户端指定了无效的范围。
401  | **UNAUTHENTICATED**     | 由于遗失，无效或过期的OAuth令牌而导致请求未通过身份验证。
403  | **PERMISSION_DENIED**   | 客户端没有足够的权限。这可能是因为OAuth令牌没有正确的范围，客户端没有权限，或者客户端项目尚未启用API。
404  | **NOT_FOUND**           | 找不到指定的资源，或者该请求被未公开的原因（例如白名单）拒绝。
409  | **ABORTED**             | 并发冲突，例如读-修改-写冲突。
409  | **ALREADY_EXISTS**      | 客户端尝试创建的资源已存在。
429  | **RESOURCE_EXHAUSTED**  | 资源配额达到速率限制。 客户端应该查找google.rpc.QuotaFailure错误详细信息以获取更多信息。
499  | **CANCELLED**           | 客户端取消请求
500  | **DATA_LOSS**           | 不可恢复的数据丢失或数据损坏。 客户端应该向用户报告错误。
500  | **UNKNOWN**             | 未知的服务器错误。 通常是服务器错误。
500  | **INTERNAL**            | 内部服务错误。 通常是服务器错误。
501  | **NOT_IMPLEMENTED**     | 服务器未实现该API方法。
503  | **UNAVAILABLE**         | 暂停服务。通常是服务器已经关闭。
504  | **DEADLINE_EXCEEDED**   | 已超过请求期限。如果重复发生，请考虑降低请求的复杂性。

## 错误重试

客户端**应该**在遇到500，503和504错误的时候重试。 最小延迟应为1s，除非另有说明。 对于429错误，客户端可能会以最少30秒的延迟重试。对于所有其他错误，重试可能不适用-首先确保您的请求是幂等的，并检查错误消息指引。

## 错误传播

如果您的API服务依赖于其他服务，您不应盲目地将错误从服务端传播到您的客户端。翻译错误时，我们建议如下：
- 隐藏接口实现的详细信息和机密信息。
- 调整发生错误方。例如，从另一个服务接收`INVALID_ARGUMENT`错误的服务器应将`INTERNAL`传播给其自己的调用者。

## 生成错误

如果您是服务器开发人员，则应该使用足够的信息来生成错误，以帮助客户开发人员了解并解决问题。 同时，您必须了解用户数据的安全性和隐私性，并避免在错误消息和错误详细信息中暴露敏感信息（因为错误通常会被记录并可能被其他人访问）。 例如，“客户端IP地址不在白名单128.0.0.0/8”等错误消息公开了服务器端的策略信息，用户不应该访问该信息。

要生成正确的错误，您首先需要熟悉[`google.rpc.Code`](https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto)，为每个错误条件选择最合适的错误代码。服务器应用程序可以并行地检查多个错误条件，并返回第一个错误条件。

下表列出了每个错误代码和一个良好错误消息的示例。

HTTP |           RPC           |                            消息实例
---- | ----------------------- | ---------------------------------------------------------------
400  | **INVALID_ARGUMENT**    | Request field x.y.z is xxx, expected one of \[yyy, zzz\].
400  | **FAILED_PRECONDITION** | Resource xxx is a non-empty directory, so it cannot be deleted.
400  | **OUT_OF_RANGE**        | Parameter 'age' is out of range \[0, 125\].
401  | **UNAUTHENTICATED**     | Invalid authentication credentials.
403  | **PERMISSION_DENIED**   | Permission 'xxx' denied on file 'yyy'.
404  | **NOT_FOUND**           | Resource 'xxx' not found.
409  | **ABORTED**             | Couldn’t acquire lock on resource ‘xxx’.
409  | **ALREADY_EXISTS**      | Resource 'xxx' already exists.
429  | **RESOURCE_EXHAUSTED**  | Quota limit 'xxx' exceeded.
499  | **CANCELLED**           | Request cancelled by the client.
500  | **DATA_LOSS**           | See note.
500  | **UNKNOWN**             | See note.
500  | **INTERNAL**            | See note.
501  | **NOT_IMPLEMENTED**     | Method 'xxx' not implemented.
503  | **UNAVAILABLE**         | See note.
504  | **DEADLINE_EXCEEDED**   | See note.

**注意**：由于客户端无法修复服务器错误，因此生成其他错误详细信息是无用的。 为了避免在错误条件下泄露敏感信息，建议不要生成除`google.rpc.DebugInfo`之外的错误详细信息。 `DebugInfo`是专门为服务器端日志记录设计的，**禁止**发送到客户端。

`google.rpc`包定义了一组标准错误有效载荷，这是自定义错误有效载荷的首选。 下表列出了每个错误代码及其匹配的标准错误有效内容（如果适用）。

HTTP |           RPC           |           推荐的错误细节
---- | ----------------------- | ----------------------------------
400  | **INVALID_ARGUMENT**    | **google.rpc.BadRequest**
400  | **FAILED_PRECONDITION** | **google.rpc.PreconditionFailure**
400  | **OUT_OF_RANGE**        | **google.rpc.BadRequest**
401  | **UNAUTHENTICATED**     |
403  | **PERMISSION_DENIED**   |
404  | **NOT_FOUND**           |
409  | **ABORTED**             |
409  | **ALREADY_EXISTS**      |
429  | **RESOURCE_EXHAUSTED**  | **google.rpc.QuotaFailure**
499  | **CANCELLED**           |
500  | **DATA_LOSS**           |
500  | **UNKNOWN**             |
500  | **INTERNAL**            |
501  | **NOT_IMPLEMENTED**     |
503  | **UNAVAILABLE**         |
504  | **DEADLINE_EXCEEDED**   |


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/07-%E9%94%99%E8%AF%AF/  

