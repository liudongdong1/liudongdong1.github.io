---
title: "【Go官方文档】Get started with Go"
subtitle: ""
date: 2022-10-17T13:40:31+08:00
description: ""
keywords: ""
tags: ["Go-document"]
categories: ["Programming"]
---

{{< admonition quote >}}
本文内容来自：[Tutorial: Get started with Go](https://golang.google.cn/doc/tutorial/getting-started)
{{< /admonition >}}

**环境配置：**
- 系统：*Windows11*
- 编辑器：*vscode*

#### 🍇 1. 创建项目
1. 新建文件夹`learninggo`并用vscode打开
2. 在根目录`learninggo`下执行命令：
```shell
go mod init learninggo
```
*自动创建go.mod文件：*
```go
module learninggo

go 1.19

```

#### 🍉 2. 编写代码
1. 创建`hello.go`文件：
```go
package main

import "fmt"

func main() {
	fmt.Println("Hello wrold!")
}
```
2. 运行 `hello.go`：
```shell
go run .
```
3. 控制台输出：
```shell
Hello wrold!
```