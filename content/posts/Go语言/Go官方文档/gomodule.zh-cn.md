---
title: "【Go官方文档】Create a Go module"
subtitle: ""
date: 2022-10-17T14:56:01+08:00
description: ""
keywords: ""
tags: [”Go-document]
categories: ["Programming"]
---

{{< admonition quote >}}
本文内容来自：[Tutorial: Create a Go module](https://golang.google.cn/doc/tutorial/create-module)
{{< /admonition >}}

**环境配置：**
- 系统：*Windows11*
- 编辑器：*vscode*

#### 🍒 1. 创建项目
1. 创建项目文件夹`greetings`并用vscode打开该项目
2. 创建文件夹`mymodule`并创建`greeting.go`文件
```go
package greeting

import "fmt"

// Hello returns a greeting for the named person.
func Hello(name string) string {
    // Return a greeting that embeds the name in a message.
    message := fmt.Sprintf("Hi, %v. Welcome!", name)
    return message
}
```
3. 在根目录创建`hello.go`文件
```go
package main

import (
	"fmt"
	"learninggo/mymodule"
)
func main() {
	message := greeting.Hello("小明")
	fmt.Println(message)
}
```

#### 运行代码
1. 在根目录打开命令提示行：
```shell
go run .
```
2. 控制台输出：
```go
Hi, 小明. Welcome!
```