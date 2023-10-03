# Go规范


> 包中成员以名称首字母大小写决定访问权限。
>
> - `Public` : 首字母大写，可被包外访问；
> - `internal` : 首字母小写，仅包内成员可以访问；
>
> **该规则适用于全局变量、全局常量、类型、结构字段、函数、方法等**。

### 1. go项目目录结构

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220523233412693.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220523233543166.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220528202737022.png)

### 2. 命名规则

- 函数命名规则：`驼峰式命名`，名字可以长但是得把功能，必要的参数描述清楚，函数名名应当是动词或动词短语，如postPayment、deletePage、save。并依Javabean标准加上get、set、is前缀。例如：xxx + With + 需要的参数名 + And + 需要的参数名 + .....
- 结构体命名规则：`结构体名应该是名词或名词短语`，如Custome、WikiPage、Account、AddressParser，避免使用Manager、Processor、Data、Info、这样的类名，类名不应当是动词。
- 包名命名规则：`包名应该为小写单词`，不要使用下划线或者混合大小写。
- 接口命名规则：`单个函数的接口名以"er"作为后缀`，如Reader,Writer。接口的实现则去掉“er”。

```go
type Reader interface {
        Read(p []byte) (n int, err error)
}
//俩个函数的接口命名
type WriteFlusher interface {
    Write([]byte) (int, error)
    Flush() error
}
//三个以上函数的接口名，抽象这个接口的功能，类似于结构体名
type Car interface {
    Start([]byte)
    Stop() error
    Recover()
}
```

- 常量命名：均需使用全部大写字母组成，并使用下划线分词，const APP_VER = "1.0"

> * 如果变量为私有，且特有名词为首个单词，则使用小写，如 apiClient
> * 其它情况都应当使用该名词原有的写法，如 APIClient、repoID、UserID

- import 包管理：对import的包进行`分组管理，用换行符分割，而且标准库作为分组的第一组`。如果你的包引入了三种类型的包，`标准库包，程序内部包，第三方包`，建议采用如下方式进行组织你的包

```go
package main

import (
    "fmt"
    "os"

    "kmg/a"
    "kmg/b"

    "code.google.com/a"
    "github.com/b"
)

//在项目中不要使用相对路径引入包：
// 错误示例
import “../net”

// 正确的做法
import “github.com/repo/proj/src/net”

//如果程序包名称与导入路径的最后一个元素不匹配，则必须使用导入别名
import (
    client "example.com/client-go"
    trace "example.com/trace/v2"
)
//在所有其他情况下，除非导入之间有直接冲突，否则应避免导入别名
import (
    "net/http/pprof"
    gpprof "github.com/google/pprof"
)
```

### 3. 错误处理

- 错误处理的原则就是`不能丢弃任何有返回err的调用，不要使用 _ 丢弃，必须全部处理。接收到错误，要么返回err，或者使用log记录下来`
- `尽早return`：一旦有错误发生，马上返回
- `尽量不要使用panic`，除非你知道你在做什么
- 错误描述如果是英文必须为小写，不需要标点结尾
- 采用独立的错误流进行处理

```go
// 错误写法
if err != nil {
    // error handling
} else {
    // normal code
}

// 正确写法
if err != nil {
    // error handling
    return // or continue, etc.
}
// normal code

```

### 4. 常用工具

**gofmt** 大部分的格式问题可以通过gofmt解决， gofmt 自动格式化代码，保证所有的 go 代码与官方推荐的格式保持一致，于是所有格式有关问题，都以 gofmt 的结果为准。

**goimport** 我们强烈建议使用 goimport ，该工具在 gofmt 的基础上增加了自动删除和引入包.

```go
go get golang.org/x/tools/cmd/goimports
```

**go vet** vet工具可以帮我们`静态分析我们的源码存在的各种问题，例如多余的代码，提前return的逻辑，struct的tag是否符合标准等`。

```go
go get golang.org/x/tools/cmd/vet
```

使用如下：

```go
go vet .
```

### 5. 注意事项

- `package` 的名字和目录名一样，`main` 除外
- `string` 表示的是不可变的字符串变量，对 `string` 的修改是比较重的操作，基本上都需要重新申请内存，如果没有特殊需要，需要修改时多使用 `[]byte`
- `append` 要小心自动分配内存，`append` 返回的可能是新分配的地址
- 如果要直接修改 `map` 的 `value` 值，则 `value` 只能是指针，否则要覆盖原来的值
- 使用 `defer`，保证退出函数时释放资源
- 尽量少用全局变量，通过参数传递，使每个函数都是“无状态”
- 参数如果比较多，将相关参数定义成结构体传递
- 写完代码都必须格式化，保证代码优雅 `gofmt -w main.go`
- 编译前先执行代码静态分析：`go vet pathxxx/` `go vet`是 Go 自带的工具，用于检查程序是否正确。

### Resource

- https://blog.csdn.net/wohu1104/article/details/123209272#t0

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/go%E8%A7%84%E8%8C%83/  

