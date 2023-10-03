# GVM


> `Go` 语言版本管理器（`GVM`）是管理 `Go` 语言环境的开源工具。`GVM 「pkgsets」` 支持安装多个版本的 `Go` 并管理每个项目的模块。它最初由 `Josh Bussdieker` 开发，`GVM` 与 `Ruby RVM` 类似，允许你为每个项目或一组项目创建一个开发环境，分离不同的 `Go` 版本和包依赖关系，来提供更大的灵活性，以防不同版本造成的问题。`GVM` 主要有以下几个特性：
>
> - 管理 `Go` 的多个版本，包括安装、卸载和指定使用 `Go` 的某个版本
> - 查看官方所有可用的 `Go` 版本，同时可以查看本地已安装和默认使用的 `Go` 版本
> - 管理多个 `GOPATH`，并可编辑 `Go` 的环境变量
> - 可将当前目录关联到 `GOPATH`
> - 可以查看 `GOROOT` 下的文件差异
> - 支持 `Go` 版本切换

## 1. 使用GVM进行包管理

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220525101000944.png)

### 1. install

- https://github.com/moovweb/gvm

如果是 `go 1.5-` 的版本，直接安装：

```
gvm install go1.4  
gvm use go1.4 [--default]
```

它背后做的事情是先把源码下载下来，再用 C 做编译。

所以如果是 `go 1.5+` 的版本，因为至此 Go 实现了自举 **（用 Go 编译 Go）**，就需要用到 Go 1.4 来做编译

```
# -B 表示只安装二进制包
gvm install go1.4 -B  
gvm use go1.4  
export GOROOT_BOOTSTRAP=$GOROOT  
gvm install go1.7.3
```

#### .1. connection error

> ```shell
> ssh -T git@github.com  # 测试是否可以ssh连接github
> ```
>
> 

> fatal: 无法连接到 github.com：
> github.com[0: 13.229.188.59]: errno=Connection timed out
>
> ```shell
> git config --global url."https://".insteadOf git://
> ```

### 2. 安装管理go版本

```shell
gvm listall  #显示可以下载和编译可用的 Go 版
gvm install go1.12.8  #安装特定的版本
gvm list  # 显示已经安装的
#  GVM 仍然默认使用系统的 Go 版本，通过它旁边的 => 符号来表示。你可以使用 gvm use 命令来切换到新安装的 go 1.12.8 版本。
gvm use go1.12.6
gvm use go1.12.8 --default   #--default 参数来指定默认使用这个版本
go version   #查看go版本
```

```
Usage: gvm [command]

Description:
  GVM is the Go Version Manager

Commands:
  version    - print the gvm version number
  get        - gets the latest code (for debugging)
  use        - select a go version to use (--default to set permanently)
  diff       - view changes to Go root
  help       - display this usage text
  implode    - completely remove gvm
  install    - install go versions
  uninstall  - uninstall go versions
  cross      - install go cross compilers
  linkthis   - link this directory into GOPATH
  list       - list installed go versions
  listall    - list available versions
  alias      - manage go version aliases
  pkgset     - manage go packages sets
  pkgenv     - edit the environment for a package set
```

### 3.  GVM pkgset

> 如果你通过 `go get` 获取一个包，它会被下载到 `$GOPATH` 目录中的 `src` 和 `pkg` 目录下。然后你可以使用 `import` 将其引入到你的 `Go` 程序中。
>
> `GVM` 通过使用「`pkgsets`」将项目的新目录附加到 `Go` 安装版本的默认 `$GOPATH`，类似 `Linux` 系统上的 `$PATH`，这样就可以很好地完成了项目之间包的管理和隔离。

```
GVM pkgset is used to manage various Go packages
== Usage
  gvm pkgset Command
== Command
  create     - create a new package set
  delete     - delete a package set
  use        - select where gb and goinstall target and link
  empty      - remove all code and compiled binaries from package set
  list       - list installed go packages
```

```shell
$ echo $GOPATH
/home/chris/.gvm/pkgsets/go1.12.8/global
# 当 GVM 被告知使用一个新版本时，它将会更换一个新的 $GOPATH，gloabl pkgset 将默认使用该版本。
```

```
[chris@marvin]$ go get github.com/gorilla/mux
[chris@marvin]$ tree
[chris@marvin introToGvm ]$ tree
.
├── overlay
│   ├── bin
│   └── lib
│       └── pkgconfig
├── pkg
│   └── linux_amd64
│       └── github.com
│           └── gorilla
│               └── mux.a
src/
└── github.com
    └── gorilla
        └── mux
            ├── AUTHORS
            ├── bench_test.go
            ├── context.go
            ├── context_test.go
            ├── doc.go
            ├── example_authentication_middleware_test.go
            ├── example_cors_method_middleware_test.go
            ├── example_route_test.go
            ├── go.mod
            ├── LICENSE
            ├── middleware.go
            ├── middleware_test.go
            ├── mux.go
            ├── mux_test.go
            ├── old_test.go
            ├── README.md
            ├── regexp.go
            ├── route.go
            └── test_helpers.go
```

### 4. 卸载 GVM 或指定版本 Go 语言

如果你只是想卸载某个安装好的 `Go` 版本，可以使用以下指令。

```shell
$ gvm uninstall go1.12.8
```

如果你想完全卸载掉 GVM 和 所有安装的 Go 版本，可以使用以下指令。

```shell
# 需谨慎操作
$ gvm implode
```

## 2. 下载对应安装包

> 国外官网 https://golang.org/dl/
>
> 国内官网 https://golang.google.cn/dl/

### 1. 安装 Go 1.12

```shell
#  1. 下载安装包
wget https://dl.google.com/go/go1.12.4.linux-amd64.tar.gz
#  2. 解压至 /usr/local目录
sudo tar -C /usr/local -xzf go1.12.4.linux-amd64.tar.gz
#  3. 修改环境变量
vim ~/.profile
export GOROOT="/usr/local/go"    // Go 的安装目录。也就是刚才解压缩指定的路径
export GOPATH=$HOME/gocode      // 本机配置的 Go 代码目录
export GOBIN=$GOPATH/bin    // Go 代码编译后的可执行文件存放目录
export PATH=$PATH:$GOPATH:$GOBIN:$GOROOT/bin    // 将 Go 安装目录添加进操作系统 PATH 路径

#  4. 使配置生效 
source ~/.profile
go version
```

## 3. vscode 工具&golong.org源代码

### 1. 下载golong.org工具

```shell
mkdir -p $GOPATH/src/golang.org/x/
cd $GOPATH/src/golang.org/x/
git clone https://github.com/golang/tools.git;
git clone https://github.com/golang/lint.git;
git clone https://github.com/golang/sys.git; 
git clone https://github.com/golang/net.git;
git clone https://github.com/golang/text.git; 
git clone https://github.com/golang/crypto.git;
git clone https://github.com/golang/mod.git;
git clone https://github.com/golang/xerrors.git
```

### 2. 下载golong.org代码

```shell
#!/bin/bash
cd $GOPATH/src;
mkdir github.com;
cd $GOPATH/src/github.com;
mkdir acroca cweill derekparker go-delve josharian karrick mdempsky pkg ramya-rao-a rogpeppe sqs uudashr ;
cd $GOPATH/src/github.com/acroca;
git clone https://github.com/acroca/go-symbols.git;
cd $GOPATH/src/github.com/cweill;
git clone https://github.com/cweill/gotests.git;
cd $GOPATH/src/github.com/derekparker;
git clone https://github.com/derekparker/delve.git;
cd $GOPATH/src/github.com/go-delve;
git clone https://github.com/go-delve/delve.git;
cd $GOPATH/src/github.com/josharian;
git clone https://github.com/josharian/impl.git;
cd $GOPATH/src/github.com/karrick;
git clone https://github.com/karrick/godirwalk.git;
cd $GOPATH/src/github.com/mdempsky;
git clone https://github.com/mdempsky/gocode.git;
cd $GOPATH/src/github.com/pkg;
git clone https://github.com/pkg/errors.git;
cd $GOPATH/src/github.com/ramya-rao-a;
git clone https://github.com/ramya-rao-a/go-outline.git;
cd $GOPATH/src/github.com/rogpeppe;
git clone https://github.com/rogpeppe/godef.git;
cd $GOPATH/src/github.com/sqs;
git clone https://github.com/sqs/goreturns.git;
cd $GOPATH/src/github.com/uudashr;
git clone https://github.com/uudashr/gopkgs.git;
```

### 3. 创建安装脚本

```shell
#!/bin/bash
cd $GOPATH/src
go install github.com/mdempsky/gocode
go install github.com/uudashr/gopkgs/cmd/gopkgs
go install github.com/ramya-rao-a/go-outline
go install github.com/acroca/go-symbols
go install github.com/rogpeppe/godef
go install github.com/sqs/goreturns
go install github.com/derekparker/delve/cmd/dlv
go install github.com/cweill/gotests
go install github.com/josharian/impl
go install golang.org/x/tools/cmd/guru
go install golang.org/x/tools/cmd/gorename
go install golang.org/x/lint/golint
```

- gocode 用于代码`功能提示`
- delve 用于`调试源码`
- gopkgs 用于`对当前文件实现智能的包导入`
- golint 用于在保存文件时`检查语法`
- godef 用于`跳转到定义包`
- gopls 是官方的语言服务器 Go Language Server，也可以实现 Go 语言的代码智能提示和跳转到定义包等功能

## Resource

- https://blog.csdn.net/wohu1104/article/details/97965998

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/gvm/  

