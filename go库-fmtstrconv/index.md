# go库-fmt&strconv


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29odTExMDQ=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center.png)

```go
package main

import "fmt"

type Person struct {
	Name string
	Age  int
}

func main() {
	a := 10
	s := "hello world"
	p := Person{Name: "wohu", Age: 25}
	c := []int{1, 2, 3, 4}

	fmt.Printf("p %%v is %v\n", p)   // p %v is {wohu 25}
	fmt.Printf("p %%+v is %+v\n", p) // p %+v is {Name:wohu Age:25}
	fmt.Printf("p %%#v is %#v\n", p) // p %#v is main.Person{Name:"wohu", Age:25}
	fmt.Printf("p type is %T\n", p)  // p type is main.Person

	fmt.Printf("a %%#v is %#v, a type is %T\n", a, a) // a %#v is 10, a type is int
	fmt.Printf("s %%#v is %#v, s type is %T\n", s, s) // s %#v is "hello world", s type is string
	fmt.Printf("c %%v is %v, c type is %T\n", c, c)   // c %v is [1 2 3 4], c type is []int
	fmt.Printf("c %%#v is %#v, c type is %T\n", c, c) // c %#v is []int{1, 2, 3, 4}, c type is []int
}
```

### 1. 常用函数

#### 1. Fprintf

```
func Fprintf(w io.Writer, format string, a ...interface{}) (n int, err error)
```

`Fprintf` 根据 `format` 参数生成格式化的字符串并写入 `w` 。返回写入的字节数和遇到的任何错误。

#### 2. Printf

```go
func Printf(format string, a ...interface{}) (n int, err error)
```

根据 `format` 参数生成格式化的字符串并写入标准输出。

#### 3. Sprintf

`Sprintf` 根据 `format` 参数生成格式化的字符串并返回该字符串。

```go
func Sprintf(format string, a ...interface{}) string
```

#### 4. Scanf

`Scanf` 从标准输入扫描文本，根据 `format` 参数指定的格式将成功读取的空白分隔的值保存进成功传递给本函数的参数。返回成功扫描的条目个数和遇到的任何错误。

```go
func Scanf(format string, a ...interface{}) (n int, err error)
```

### 2. strconv

#### 1. string 与 int 类型之间的转换

```go
package main
import (
	"fmt"
	"strconv"
)
func main() {
	num := 100
	str := strconv.Itoa(num)
	fmt.Printf("type:%T ---- value:%#v\n", str, str)	// type:string ---- value:100
}
```

```go
package main

import (
	"fmt"
	"strconv"
)

func main() {
	str1 := "110"
	str2 := "s100"

	num1, err := strconv.Atoi(str1)
	if err != nil {
		fmt.Printf("%v 转换失败！", str1)
	} else {
		fmt.Printf("type:%T value:%#v\n", num1, num1)
	}

	num2, err := strconv.Atoi(str2)
	if err != nil {
		fmt.Printf("%v 转换失败！", str2)
	} else {
		fmt.Printf("type:%T value:%#v\n", num2, num2)
	}
}
```

#### 2. Parse系列函数

##### 1. ParseBool

```go
func main() {
	str1 := "110"

	boo1, err := strconv.ParseBool(str1)
	if err != nil {
		fmt.Printf("str1: %v\n", err)
	} else {
		fmt.Println(boo1)
	}

	str2 := "t"
	boo2, err := strconv.ParseBool(str2)
	if err != nil {
		fmt.Printf("str2: %v\n", err)
	} else {
		fmt.Println(boo2)
	}
}
```

##### 2. ParseInt

```go
func ParseInt(s string, base int, bitSize int) (i int64, err error)
```
- base 指定进制，取值范围是 2 到 36。如果 base 为 0，则会从字符串前置判断，“0x”是 16 进制，“0”是 8 进制，否则是 10 进制。

- bitSize 指定结果必须能无溢出赋值的整数类型，0、8、16、32、64 分别代表 int 、 int8 、 int16 、 int32 、 int64 。
- 返回的 err 是 *NumErr 类型的，如果语法有误， err.Error = ErrSyntax ，如果结果超出类型范围 err.Error = ErrRange 。

```go
func main() {
	str := "-11"
	num, err := strconv.ParseInt(str, 10, 0)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println(num)
	}
}
```

##### 3. ParseUnit

```go
func ParseUint(s string, base int, bitSize int) (n uint64, err error)  // 用于无符号整型
```

##### 4. ParseFloat

```go
func ParseFloat(s string, bitSize int) (f float64, err error)
```

```go
func main() {
	str := "3.1415926"
	num, err := strconv.ParseFloat(str, 64)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println(num)	// 3.1415926
	}
}
```

#### 3. Format函数

> `Format` 系列函数实现了将给定类型数据`格式化为字符串类型`的功能，其中包括 `FormatBool()` 、 `FormatInt()` 、 `FormatUint()` 、 `FormatFloat()` 。

#### 4. Append函数

> `Append` 系列函数用于将指定类型转换成字符串后`追加到一个切片中`，其中包含 `AppendBool()` 、 `AppendFloat()` 、 `AppendInt()` 、 `AppendUint()` 。


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/go%E5%BA%93-fmtstrconv/  

