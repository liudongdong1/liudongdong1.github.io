# 安装目录&工作区&源码&命令.md


> - GOROOT: `Go 语言安装路径`。
> - GOPATH: `若干工作区目录的路径`。是我们自己定义的工作空间。在 Go Module 模式之前非常重要，现在基本上用来`存放使用 go get 命令获取的项目`。
> - GOBIN: `Go 编译生成的程序的安装目录`，比如通过 go install 命令，会把生成的 Go 程序放到该目录下，即可在终端使用。

### 1. GOROOT 安装目录说明

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220525221337761.png)

通过 `go install` 命令，`Go` 程序（这里是标准库中的程序）会被编译成平台相关的归档文件存放到其中。另外，`pkg/tool/linux_amd64` 文件夹存放了使用 `Go` 制作软件时用到的很多强大的命令和工具。

### 2. GOPATH 工作区间说明

> `GOPATH` 是作为`编译后二进制的存放目的地`和 `import` 包时的搜索路径 (其实也是你的工作目录，可以在 `src` 下创建你自己的 `Go` 源文件, 然后开始工作)，一般情况下，`Go` 源码文件必须放在工作区中。
>
> 我们需要将工作区的目录路径添加到环境变量 GOPATH 中。否则，即使处于同一工作区（事实上，未被加入 GOPATH 中的目录不用管成为工作区），代码之间也无法通过绝对代码包路径调用。
>
> 在实际开发环境中，工作区可以只有一个，也可以由多个，这些工作区的目录路径都需要添加到 GOPATH 中。与 GOROOT 一样，我们应确保 GOPATH 一直有效。
>
> go语言在多个工作区查找包依赖顺序：先 `GOROOT` 然后再 `GOPATH` 

**使用命令行或者使用集成开发环境编译 `Go` 源码时， 不要设置全局的 `GOPATH` ，而是随项目设置 `GOPATH`** 。

```shell
$ go env
GOARCH="amd64"
GOBIN="/home/wohu/gocode/bin"
GOCACHE="/home/wohu/.cache/go-build"
GOEXE=""
GOFLAGS=""
GOHOSTARCH="amd64"
GOHOSTOS="linux"
GOOS="linux"
GOPATH="/home/wohu/gocode"
GOPROXY=""
GORACE=""
GOROOT="/usr/local/go"
GOTMPDIR=""
GOTOOLDIR="/usr/local/go/pkg/tool/linux_amd64"
GCCGO="gccgo"
CC="gcc"
CXX="g++"
CGO_ENABLED="1"
GOMOD=""
CGO_CFLAGS="-g -O2"
CGO_CPPFLAGS=""
CGO_CXXFLAGS="-g -O2"
CGO_FFLAGS="-g -O2"
CGO_LDFLAGS="-g -O2"
PKG_CONFIG="pkg-config"
GOGCCFLAGS="-fPIC -m64 -pthread -fmessage-length=0 -fdebug-prefix-map=/tmp/go-build076028171=/tmp/go-build -gno-record-gcc-switches"
```

- `src` 目录: 主要存放 `Go` 的源码文件
- `pkg` 目录：存放编译好的库文件, 主要是 `.a` 文件
- `bin` 目录：主要存放可执行文件

> **`GOPATH` 中不要包含 `Go` 语言的安装目录，以便将 `Go` 语言本身的工作区同用户工作区严格分开**。通过 `Go` 工具中的代码获取命令 `go get` ，可以指定项目的代码下载到我们在 `GOPATH` 中设定的第一个工作区中，并在其中完成编译和安装。

### 3. 源码文件

> `Go` 源码文件有分 3 种，即命令源码文件，库源码文件和测试源码文件。**不管是命令源文件还是库源文件，在同一个目录下的所有源文件，其所属包的名称必须一致的。**
>
> 构建 = 编译 + 链接
>
> - 编译和链接是两个步骤。
>
> - 一个代码包在被安装（编译、链接并生成相应文件）完成之后，相应的静态链接库文件就会被 go 工具放到 pkg 目录里。
> - 在编译的时候，如果 pkg 目录里已经有了依赖代码包的静态链接库文件，那么在默认情况下 go 工具就不会再编译一遍了。
> - 在链接的时候，如果依赖包的静态链接库文件已经存在于 pkg 目录中了，那么 go 工具就会直接使用。这个链接的过程实际上就是把主包和依赖包链接在一起。

#### 1. 命令源码文件

- 声明为属于 main 代码包，并且包含无参数声明和结果声明的 main 函数的源码文件。这类源码是程序的入口，可以独立运行（使用 go run 命令），也可以被 go build 或 go install 命令转换为可执行文件。

- `同一个代码包中的所有源码文件，其所属代码包的名称必须一致`。如果命令源码文件和库源码文件处于同一代码包中，该包就无法正确执行 go build 和 go install 命令。换句话说，这些源码文件也就无法被编译和安装。因此，命令源码文件通常会单独放在一个代码包中。一般情况下，一个程序模块或软件的启动入口只有一个。
- 同一个代码包中可以有多个命令源码文件，可通过 go run 命令分别运行它们。但通过 go build 和 go install 命令无法编译和安装该代码包。所以一般情况下，`不建议把多个命令源码文件放在同一个代码包中`。

- 当代码包中有且仅有一个命令源码文件时，在文件所在目录中执行 go build 命令，即可在该目录下生成一个与目录同名的可执行文件；若使用 go install 命令，则可在当前工作区的 bin 目录下生成相应的可执行文件。

- 总结：如果`一个 Go 源文件被声明属于 main 包，并且该文件中包含 main 函数，则它就是命令源码文件`。命令源文件属于程序的入口，可以通过 Go 语言的go run 命令运行或者通过go build 命令生成可执行文件。

#### 2. 库源码文件

- 库源文件则是指存在于某个包中的普通源文件，并且库源文件中不包含 main 函数。`库源码文件声明的包名会与它实际所属的代码包（目录）名一致，且库源码文件中不包含无参数声明和无结果声明的 main 函数`。

- 如在 basic/set 目录下执行 go install 命令，成功地安装了 basic/set 包，并生成一个名为 set.a 的归档文件。归档文件的存放目录由以下规则产生：

  - 安装库源码文件时所生成的归档文件会被存放到`当前工作区的 pkg 目录中`。
  - 根据被编译的目标计算机架构，归档文件会被放置在 `pkg 目录下的平台相关目录中`。如上的 set.a 在 64 位 window 系统上就是 pkg/windows_amd64 目录中。
  - 存放归档文件的目录的`相对路径与被安装的代码包的上一级代码包的相对路径是一致的`。
- 第一个相对路径就是相对于工作区的 pkg 目录下的平台相关目录而言的，而第二个相对路径是相对于工作区的 src 目录而言的。如果被安装的代码包没有上一级代码包（也就是说它的父目录就是工作的 src 目录），那么它的归档文件就会被直接存放到当前工作区的 pkg 目录的平台相关目录下。
  - 如 basic 包的归档文件 basic.a 总会被直接存放到 pkg/windows_amd64 目录下，而 `basic/set 包`的归档文件 set.a 则会被存放到 pkg/windows_amd64/`basic` 目录下。

- 总结：假如一个程序 main.go 和 log.go 两个文件放到同一个 package main 中，那么它们都是命令源码文件，如果 log.go 放到 package log 包中，那么它就是库源码文件。

#### 3. 测试源码文件

- 可以通过执行 `go test 命令运行当前代码包下的所有测试源码文件`。成为测试源码文件的充分条件有两个：

  - 文件名需要以 ”_test.go” 结尾
  - 文件中需要`至少包含该一个名称为 Test 开头或 Benchmark 开头，拥有一个类型为 *testing.T 或 testing.B 的参数的函数`。类型 testing.T 或 testing.B 是两个结构体类型。
  - 当在某个代码包中执行 go test 命令，该代码包中的所有测试源码文件就会被找到并运行。
- 注意：存储 Go 代码的文本文件需要以 UTF-8 编码存储。如果源码文件中出现了非 UTF-8 编码的字符，则在运行、编译或安装时，Go 会抛出 “illegal UTF-8 sequence” 的错误。

### 4. 包引入

> 标准包的源码位于 `$GOROOT/src/` 下面，标准包可以直接引用。自定义的包和第三方包的源码必须放到 `$GOPATH/src/` 目录下才能被引用。

#### 1. 绝对路径

- 包的绝对路径就是 `$GOROOT/src` 或 `$GOPATH/src` 后面包的源码的全路径

```go
import "common/upload"
import "database/sql/driver"
import "database/sql"
```

- upload 包是`自定义的包`，其源码位于 `$GOPATH/src/common/upload 目录下`，代码包导入使用的路径就是`代码包在工作区的 src 目录下的相对路径`，比如 upload 的绝对路径为 /home/wohu/gocode/src/common/upload ，而 /home/wohu/gocode 是被包含在环境变量 `GOPATH 中的工作区目录路径`，则其代码包导入路径就是`common/upload`。

- sql 和 driver 包的源码分别位于 $GOROOT/src/`database/sql` 和 $GOROOT/src/`database/sql/driver` 下。

编译器会首先查找 Go 的安装目录，然后才会按顺序查找 GOPATH 变量里列出的目录。一旦编译器找到一个满足 import 语句的包，就停止进一步查找。

#### 2. 相对路径引用

相对路径只能用于引用 `$GOPATH` 下的包，`标准包的引用只能使用全路径引用`。比如下面两个包：

- 包 `a` 的路径是 `$GOPATH/src/lab/a` ，包 `b` 的源码路径为 `$GOPATH/src/lab/b` ，假设 `b` 引用了 `a` 包，则可以使用相对路径引用方式。

```shell
// 相对路径引用
import "../a" 

// 绝对路径引用
import "lab/a"
```

### 5.  包初始化

> - 一个包可以有多个 init 函数，`包加载会执行全部的 init 函数，但并不能保证执行顺序`，所以不建议在一个包中放入多个 init 函数，将需要初始化的逻辑放到一个 init 函数里面。
> - 包不能出现循环引用。比如包 a 引用了包 b ，包 b 引用了包 c，如果包 c 又引用了包 a，则编译不能通过。
> - `包的重复引用是允许的`。比如包 a 引用了包 b 和包 c ，包 b 和包 c 都引用了包 d 。这种场景相当于重复引用了d，这种情况是允许的， 并且 Go 编译器保证 d 的 init 函数只会执行一次。

### 6. 标准命令

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220526222110494.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220526222144721.png)

- https://blog.csdn.net/wohu1104/article/details/106295007


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%AE%89%E8%A3%85%E7%9B%AE%E5%BD%95%E5%B7%A5%E4%BD%9C%E5%8C%BA%E6%BA%90%E7%A0%81%E5%91%BD%E4%BB%A4/  

