---
title: "【Go官方文档】Getting started with generics"
subtitle: ""
date: 2022-10-18T09:11:15+08:00
description: ""
keywords: ""
tags: [”Go-document]
categories: ["Programming"]
---

{{< admonition quote >}}
本文内容来自：[Tutorial: Getting started with generics](https://golang.google.cn/doc/tutorial/generics)
{{< /admonition >}}

**环境配置：**
- 系统：*Windows11*
- 编辑器：*vscode*

#### 🌶️ 创建项目
1. 创建项目文件夹`generics`并用vscode打开
2. 在终端执行命令：
```shell
go mod init example/generics
```
3. 创建文件`main.go`：
```go
package main

import "fmt"

// 定义约束
type Number interface {
	int64 | float64
}

// SumIntsOrFloats sums the values of map m. It supports both int64 and float64
// as types for map values.
func SumNumbers[K comparable, V Number](m map[K]V) V {
	var s V
	for _, v := range m {
		s += v
	}
	return s
}

func main() {
	// Initialize a map for the integer values
	ints := map[string]int64{
		"first":  34,
		"second": 12,
	}

	// Initialize a map for the float values
	floats := map[string]float64{
		"first":  35.98,
		"second": 26.99,
	}

	fmt.Printf("Generic Sums: %v and %v\n",
		SumNumbers(ints),
		SumNumbers(floats))
}
```

#### 🥕 运行代码
1. 在终端执行命令：
```shell
go run .
```
2. 执行结果：
```shell
Generic Sums: 46 and 62.97
```