# string操作


> `Go` 语言中字符串的内部实现使用 `UTF-8` 编码，通过 `rune` 类型，可以方便地对每个 `UTF-8` 字符进行访问。当然， `Go` 语言也支持按照传统的 `ASCII` 码方式逐字符进行访问。
>
> - 字符串是常量，可以通过类似数组索引访问其字节单元，但是不能修改某个字节的值；
> - 内置的 `len` 函数可以返回一个字符串中的字节数目（不是 `rune` 字符数目）。
> - 字符串转换为切片 `[]byte(s)` 要慎用，每转换一次需要复制一份内存，尤其是字符串数据量较大时；

### 1. 字符串类型

```go
// runtime/string.go
type stringStruct struct {
    str unsafe . Pointer //指向底层字节数组的指针
    len int //字节数组长度
}

var str string	//string 类型变量在定义后默认的初始值是空，不是 nil。
if str == "" {
    // str 为空
}
var str string
if len(str) == 0 {
    // str 为空
}
//字符串拼接
func main() {
	var buffer bytes.Buffer

	for i := 0; i < 500; i++ {
		buffer.WriteString("hello,world")
	}
	fmt.Println(buffer.String())	 // 对缓冲区调用函数String() 以字符串的方式输出结果。
}

```

### 2. 字符rune类型

- 一种是 uint8 类型，或者叫 byte 型（ byte 是 uint8 的别名），代表了 ASCII 码的一个字符，占用 1 个字节。

- 一种是 rune 类型，代表一个 UTF-8 字符，当需要处理中文、日文或者其他复合字符时，则需要用到 rune 类型，占用字节不确定。

```go
package main

import "fmt"

func main() {
	var str string = "中国"
	rangeRune([]rune(str))
	rangeStr(str)
}

func rangeRune(arg []rune) {
	fmt.Println("rune type arg length is ", len(arg))
	for i := 0; i < len(arg); i++ {
		fmt.Printf("i is %d, value is %c\n", i, arg[i])
	}
}

func rangeStr(arg string) {
	fmt.Println("str type arg length is ", len(arg))
	for i := 0; i < len(arg); i++ {
		fmt.Printf("i is %d, value is %c\n", i, arg[i])
	}
}
```

### 3. 字节byte类型

`byte` 用来表示字节，一个字节是 8 位。定义一个字节类型变量的语法是：

```go
func main() {
    s := "hello 世界"
    runeSlice := []rune(s) // len = 8
    byteSlice := []byte(s) // len = 12
    // 打印每个rune切片元素
    for i:= 0; i < len(runeSlice); i++ {
        fmt.Println(runeSlice[i])
        // 输出104 101 108 108 111 32 19990 30028
    }
    fmt.Println()
    // 打印每个byte切片元素
    for i:= 0; i < len(byteSlice); i++ {
        fmt.Println(byteSlice[i])
        // 输出104 101 108 108 111 32 228 184 150 231 149 140
    }
}
```

```go
package main

import (
	"bytes"
	"fmt"
	"strings"
	"unicode/utf8"
)

func main() {
	s := "hello,您好"
	s_length := len(s)

	fmt.Println(s_length)       // 12
	fmt.Println(len([]byte(s))) // 12

	byte_length := f1(s)
	fmt.Println(byte_length) // 8

	string_length := f2(s)
	fmt.Println(string_length) // 8

	rune_length := f3(s)
	fmt.Println(rune_length) // 8

	utf_length := f4(s)
	fmt.Println(utf_length) // 8
}

func f1(s string) int {
	return bytes.Count([]byte(s), nil) - 1
}

func f2(s string) int {
	return strings.Count(s, "") - 1
}

func f3(s string) int {
	return len([]rune(s))
}

func f4(s string) int {
	return utf8.RuneCountInString(s)
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/string%E6%93%8D%E4%BD%9C/  

