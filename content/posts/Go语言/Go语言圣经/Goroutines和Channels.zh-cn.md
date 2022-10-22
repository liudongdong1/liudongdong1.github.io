---
title: "【Go语言圣经】Goroutines和Channels"
subtitle: ""
date: 2022-10-19T17:48:48+08:00
description: ""
keywords: ""
tags: ["Go语言圣经"]
categories: ["Programming"]
---

{{< admonition quote >}}
本文内容来自：[Go语言圣经（中文版）](https://golang-china.github.io/gopl-zh/)
{{< /admonition >}}

**环境配置：**
- 系统：*Windows11*
- 编辑器：*vscode*

{{< admonition  >}}
本文假设你已经安装了Go并配置好相关环境，如果你还没有安装Go，请前往Go官方网站进行[下载安装](https://golang.google.cn/dl/)
{{< /admonition >}}

#### 🌱 Goroutines
在Go语言中，每一个并发的执行单元叫作一个goroutine。
```go
package main

import (
	"fmt"
	"time"
)

func main() {
	go spinner(100 * time.Millisecond)
	const n = 45
	fibN := fib(n) // slow
	fmt.Printf("\rFibonacci(%d) = %d\n", n, fibN)
}

func spinner(delay time.Duration) {
	for {
			for _, r := range `-\|/` {
					fmt.Printf("\r%c", r)
					time.Sleep(delay)
			}
	}
}

func fib(x int) int {
	if x < 2 {
			return x
	}
	return fib(x-1) + fib(x-2)
}
```