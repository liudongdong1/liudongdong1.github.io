---
title: "【Go官方文档】Getting started with fuzzing"
subtitle: ""
date: 2022-10-18T10:23:28+08:00
description: ""
keywords: ""
tags: [”Go-document]
categories: ["Programming"]
---

{{< admonition quote >}}
本文内容来自：[Tutorial: Getting started with fuzzing](https://golang.google.cn/doc/tutorial/fuzz)
{{< /admonition >}}

**环境配置：**
- 系统：*Windows11*
- 编辑器：*vscode*

#### 🍄 创建项目
1. 新建`fuzz`文件夹并用vscode打开
2. 在根目录`fuzz`下执行命令
```shell
go mod init example/fuzz
```
3. 在根目录`fuzz`下新建文件`main.go`：
```go
package main

import (
    "errors"
    "fmt"
    "unicode/utf8"
)

func main() {
    input := "The quick brown fox jumped over the lazy dog"
    rev, revErr := Reverse(input)
    doubleRev, doubleRevErr := Reverse(rev)
    fmt.Printf("original: %q\n", input)
    fmt.Printf("reversed: %q, err: %v\n", rev, revErr)
    fmt.Printf("reversed again: %q, err: %v\n", doubleRev, doubleRevErr)
}

func Reverse(s string) (string, error) {
    if !utf8.ValidString(s) {
        return s, errors.New("input is not valid UTF-8")
    }
    r := []rune(s) // type rune = int32； Go 语言通过 rune 处理中文，支持国际化多语言。
    for i, j := 0, len(r)-1; i < len(r)/2; i, j = i+1, j-1 {
        r[i], r[j] = r[j], r[i]
    }
    return string(r), nil
}
```
4. 在根目录`fuzz`下新建文件`reverse_test.go`：
```go
package main

import (
	"testing"
	"unicode/utf8"
)

func FuzzReverse(f *testing.F) {
	testcases := []string{"Hello, world", " ", "!12345"}
	for _, tc := range testcases {
		f.Add(tc) // Use f.Add to provide a seed corpus
	}
	f.Fuzz(func(t *testing.T, orig string) {
		rev, err1 := Reverse(orig)
		if err1 != nil {
			return
		}
		doubleRev, err2 := Reverse(rev)
		if err2 != nil {
			return
		}
		if orig != doubleRev {
			t.Errorf("Before: %q, after: %q", orig, doubleRev)
		}
		if utf8.ValidString(orig) && !utf8.ValidString(rev) {
			t.Errorf("Reverse produced invalid UTF-8 string %q", rev)
		}
	})
}

```

#### 2. 运行代码
1. 在终端执行命令：
```shell
go test
```
2. 输出结果：
```shell
PASS
ok      example/fuzz    0.265s
```
3. 在终端执行命令：
```shell
go test -fuzz=Fuzz -fuzztime 30s
```
4. 输出结果：
```shell
fuzz: elapsed: 0s, gathering baseline coverage: 0/47 completed
fuzz: elapsed: 0s, gathering baseline coverage: 47/47 completed, now fuzzing with 8 workers
fuzz: elapsed: 3s, execs: 474042 (156924/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 6s, execs: 950684 (159465/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 9s, execs: 1404211 (151611/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 12s, execs: 1902552 (165197/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 15s, execs: 2389600 (162590/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 18s, execs: 2876083 (162669/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 21s, execs: 3355965 (159985/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 24s, execs: 3844976 (162641/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 27s, execs: 4344065 (166235/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 30s, execs: 4846000 (167419/sec), new interesting: 0 (total: 47)
fuzz: elapsed: 30s, execs: 4846000 (0/sec), new interesting: 0 (total: 47)
PASS
ok      example/fuzz    30.413s
```

#### 🥩 3. 关于rune
*rune 类型是 Go 语言的一种特殊数字类型。在 builtin/builtin.go 文件中，它的定义：type rune = int32；官方对它的解释是：rune 是类型 int32 的别名，在所有方面都等价于它，用来区分字符值跟整数值。使用单引号定义 ，返回采用 UTF-8 编码的 Unicode 码点。Go 语言通过 rune 处理中文，支持国际化多语言。*

[了解更多rune的知识](https://www.cnblogs.com/cheyunhua/p/16007219.html)