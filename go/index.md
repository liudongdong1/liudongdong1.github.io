# Go语言


> Go 语言被设计成一门应用于搭载 Web 服务器，存储集群或类似用途的巨型中央服务器的系统编程语言。对于高性能分布式系统领域而言，Go 语言无疑比大多数其它语言有着更高的开发效率。它提供了海量并行的支持，这对于游戏服务端的开发而言是再好不过了。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/640.png)

### 1. 语法结构

> 包声明，引入包，函数，变量，语句&表达式，注释

```go
package main

import "fmt"

func main() {   // { 不能在单独的行上
   /* 这是我的第一个简单的程序 */
   fmt.Println("Hello, World!")   //行结束，没有分号
    
     // % d 表示整型数字，% s 表示字符串
    var stockcode=123
    var enddate="2020-12-31"
    var url="Code=%d&endDate=%s"
    var target_url=fmt.Sprintf(url,stockcode,enddate)
    fmt.Println(target_url)
}
```

### 2. 数据类型

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvaHUxMTA0,size_16,color_FFFFFF,t_70#pic_center.jpeg)

| 序号 | 类型和描述                                                   |
| :--- | :----------------------------------------------------------- |
| 1    | **布尔型** 布尔型的值`只可以是常量 true 或者 false`。一个简单的例子：var b bool = true。 |
| 2    | **数字类型** 整型 int 和浮点型 float32、float64，Go 语言支持整型和浮点型数字，并且支持复数，其中位的运算采用补码。 |
| 3    | **字符串类型:** 字符串就是一串固定长度的字符连接起来的字符序列。`Go 的字符串是由单个字节连接起来的`。Go 语言的字符串的字节使用 UTF-8 编码标识 Unicode 文本。其中汉字占用三个字节。 |
| 4    | **派生类型:** 包括：(a) 指针类型（Pointer）(b) 数组类型(c) 结构化类型 (struct)(d) Channel 类型(e) 函数类型(f) 切片类型(g) 接口类型（interface）(h) Map 类型 |

| 序号 | 类型和描述                                                   |
| :--- | :----------------------------------------------------------- |
| 1    | **uint8** 无符号 8 位整型 (0 到 255)                         |
| 2    | **uint16** 无符号 16 位整型 (0 到 65535)                     |
| 3    | **uint32** 无符号 32 位整型 (0 到 4294967295)                |
| 4    | **uint64** 无符号 64 位整型 (0 到 18446744073709551615)      |
| 5    | **int8** 有符号 8 位整型 (-128 到 127)                       |
| 6    | **int16** 有符号 16 位整型 (-32768 到 32767)                 |
| 7    | **int32** 有符号 32 位整型 (-2147483648 到 2147483647)       |
| 8    | **int64** 有符号 64 位整型 (-9223372036854775808 到 9223372036854775807) |

| 序号 | 类型和描述                         |
| :--- | :--------------------------------- |
| 1    | **float32** IEEE-754 32 位浮点型数 |
| 2    | **float64** IEEE-754 64 位浮点型数 |
| 3    | **complex64** 32 位实数和虚数      |
| 4    | **complex128** 64 位实数和虚数     |

| 序号 | 类型和描述                               |
| :--- | :--------------------------------------- |
| 1    | **byte** 类似 uint8                      |
| 2    | **rune** 类似 int32                      |
| 3    | **uint** 32 或 64 位                     |
| 4    | **int** 与 uint 一样大小                 |
| 5    | **uintptr** 无符号整型，用于存放一个指针 |

#### .1. 数据类型转化

```go
package main

import (
	"fmt"
	"strconv"
)

func main() {

	// 基本数据类型转换
	x := 3
	y := float64(x)
	fmt.Println(y)


	// 基本类型 --> string
	// 方式一
	a, b, c, d := 99, 3.14, true, 'h'
	var str string
	str = fmt.Sprintf("%d", a)
	fmt.Printf("str type %T, str = %q\n", str, str)
	str = fmt.Sprintf("%f", b)
	fmt.Printf("str type %T, str = %q\n", str, str)
	str = fmt.Sprintf("%t", c)
	fmt.Printf("str type %T, str = %q\n", str, str)
	str = fmt.Sprintf("%c", d)
	fmt.Printf("str type %T, str = %q\n", str, str)

	// 方式二
	str = strconv.FormatInt(int64(a), 10)  // 或者：str = strconv.Itoa(a)
	fmt.Printf("str type %T, str = %q\n", str, str)
	str = strconv.FormatFloat(b, 'f', 10, 64)  // 将float64的数据转为string, 保留10位小数
	fmt.Printf("str type %T, str = %q\n", str, str)
	str = strconv.FormatBool(c)
	fmt.Printf("str type %T, str = %q\n", str, str)


	// string -> 基本类型
	fmt.Println()
	s1, s2, s3 := "12", "3.14", "true"
	n1, _ := strconv.ParseInt(s1, 10, 64)  // 或者：n1, _ := strconv.Atoi(s1)
	fmt.Printf("n1 type %T n1 = %v\n", n1, n1)
	n2, _ := strconv.ParseFloat(s2, 64)  // 转为 float64
	fmt.Printf("n2 type %T n2 = %v\n", n2, n2)
	n3, _ := strconv.ParseBool(s3)
	fmt.Printf("n3 type %T n3 = %v\n", n3, n3)
}
/** 输出结果：
3
str type string, str = "99"
str type string, str = "3.140000"
str type string, str = "true"
str type string, str = "h"
str type string, str = "99"
str type string, str = "3.1400000000"
str type string, str = "true"

n1 type int64 n1 = 12
n2 type float64 n2 = 3.14
n3 type bool n3 = true
**/
```

#### .2. nil

> 在 `Go` 语言中，布尔类型的零值（初始值）为 `false` ，数值类型的零值为 0，字符串类型的零值为空字符串 `""`，而指针、切片、映射、通道、函数和接口的零值则是 `nil` 。

- nil 标识符是不能比较的
- nil 没有默认类型
- 不同类型 nil 的指针是一样的
- 不同类型的 nil 值占用的内存大小可能是不一样的

#### .3. type

```go
package tempconv

type Celsius float64    // 摄氏温度
type Fahrenheit float64 // 华氏温度

const (
	AbsoluteZeroC Celsius = -273.15 // 绝对零度
	FreezingC     Celsius = 0       // 结冰点温度
	BoilingC      Celsius = 100     // 沸水温度
)

func CToF(c Celsius) Fahrenheit {
	return Fahrenheit(c*9/5 + 32)
}

func FToC(f Fahrenheit) Celsius {
	return Celsius((f - 32) * 5 / 9)
}
```

```go
package main

import (
	"fmt"
)

// 将NewInt定义为int类型
// 通过 type 关键字的定义，NewInt 会形成一种新的类型，NewInt 本身依然具备 int 类型的特性。
type NewInt int

// 将int取一个别名叫IntAlias, 将 IntAlias 设置为 int 的一个别名，使 IntAlias 与 int 等效。
type IntAlias = int

func main() {

	// 将a声明为NewInt类型
	var a NewInt
	// 查看a的类型名
	fmt.Printf("a type: %T\n", a)	// a type: main.NewInt

	// 将 b 声明为IntAlias类型
	var b IntAlias
	// 查看b的类型名
	fmt.Printf("b type: %T\n", b)	// b type: int
}
```

> **不能在一个非本地的类型 `time.Duration` 上定义新方法，非本地类型指的就是 `time.Duration` 不是在 `main` 包中定义的，而是在 `time` 包中定义的，与 `main` 包不在同一个包中，因此不能为不在一个包中的类型定义方法。**

```go
package main

import (
	"time"
)

// 定义time.Duration的别名为MyDuration
type MyDuration = time.Duration

type MyDuration time.Duration   //将类型别名修改为类型定义


// 为 MyDuration 添加一个方法
func (m MyDuration) EasySet(a string) {

}

func main() {

}
```

##### 1. 定义结构体

```go
type name struct {
    Field1  dataType
    Field2  dataType
    Field3  dataType
}
```

##### 2. 定义接口

```go
type name interface{
    Read()
    Write()
}
```

##### 3. 定义类型

```go
type name string
```

##### 4. 类型别名

> 使用类型别名定义出来的类型与原类型一样，即可以与原类型变量互相赋值，又拥有了原类型的所有方法集。
>
> 使用类型别名定义的类型与原类型等价，而使用类型定义出来的类型是一种新的类型。
>
> 给类型别名新增方法后，原类型也能使用这个方法。

```go
type name = string
```

```go
package main
import (
    "fmt"
)
type a = string
type b string
func SayA(str a) {
    fmt.Println(str)
}
func SayB(str b) {
    fmt.Println(str)
}
func main() {
    var str = "test"
    SayA(str)
    //错误参数传递，str是字符串类型，不能赋值给b类型变量
    SayB(str)
}
```

##### 5. 类型查询

> `Goalng` 中有一个特殊的类型 `interface{}` ，这个类型可以被任何类型的变量赋值

```go
package main
import (
    "fmt"
)
func main() {
    // 定义一个interface{}类型变量，并使用string类型值”abc“初始化
    var a interface{} = "abc"
    // 在switch中使用 变量名.(type) 查询变量是由哪个类型数据赋值。
    switch v := a.(type) {
    case string:
        fmt.Println("字符串")
    case int:
        fmt.Println("整型")
    default:
        fmt.Println("其他类型", v)
    }
}
```



### 3. 变量&常量

- 指定变量类型，如果没有初始化，则变量默认为零值。

- 数值类型（包括 complex64/128）为 **0**

- 布尔类型为 **false**

- 字符串为 **""**（空字符串）

- 以下几种类型为 **nil**：

  ```go
  var a *int
  var a []int
  var a map[string] int
  var a chan int
  var a func(string) int
  var a error // error 是接口
  ```

**intVal := 1** 相等于：

```go
var intVal int 
intVal =1 
```

```go
//类型相同多个变量, 非全局变量
var vname1, vname2, vname3 type
vname1, vname2, vname3 = v1, v2, v3

var vname1, vname2, vname3 = v1, v2, v3 // 和 python 很像,不需要显示声明类型，自动推断

vname1, vname2, vname3 := v1, v2, v3 // 出现在 := 左侧的变量不应该是已经被声明过的，否则会导致编译错误


// 这种因式分解关键字的写法一般用于声明全局变量
var (
    vname1 v_type1
    vname2 v_type2
)
```

- 显式类型定义： `const b string = "abc"`
- 隐式类型定义： `const b = "abc"`
- `iota` 在 const 关键字出现时将被重置为 0 (const 内部的第一行之前)，const 中每新增一行常量声明将使 iota 计数一次 (iota 可理解为 const 语句块中的行索引)。

### 4. 条件语句

```go
/* 判断布尔表达式 */
if a < 20 {
    /* 如果条件为 true 则执行以下语句 */
    fmt.Printf("a 小于 20\n" );
} else {
    /* 如果条件为 false 则执行以下语句 */
    fmt.Printf("a 不小于 20\n" );
}

switch {
      case grade == "A" :
         fmt.Printf(" 优秀！\n" )    
      case grade == "B", grade == "C" :
         fmt.Printf(" 良好 \n" )      
      case grade == "D" :
         fmt.Printf(" 及格 \n" )      
      case grade == "F":
         fmt.Printf(" 不及格 \n" )
      default:
         fmt.Printf(" 差 \n" );
   }

select {
      case i1 = <-c1:
         fmt.Printf("received ", i1, " from c1\n")
      case c2 <- i2:
         fmt.Printf("sent ", i2, " to c2\n")
      case i3, ok := (<-c3):  // same as: i3, ok := <-c3
         if ok {
            fmt.Printf("received ", i3, " from c3\n")
         } else {
            fmt.Printf("c3 is closed\n")
         }
      default:
         fmt.Printf("no communication\n")
   }    
```

```go
//for init; condition; post { }
//for key, value := range oldMap {
//    newMap[key] = value
//}
for i := 0; i <= 10; i++ {
    sum += i
}

//numbers := [6]int{1, 2, 3, 5}  // 这里定义长度为6的数组
strings := []string{"google", "runoob"}
for i, s := range strings {
    fmt.Println(i, s)
}
```

### 5. 函数&作用域

> 执行的顺序：全局变量 -> init()函数 -> 主函数

```go
func function_name( [parameter list] ) [return_types] {
   函数体
}
```

```go
func swap(x, y string) (string, string) {
   return y, x
}
/* 函数返回两个数的最大值 */
func max(num1, num2 int) int {
    return num1
}
```

```go
package main

import "fmt"

func main() {
   /* 定义局部变量 */
   var a int = 100
   var b int= 200

   fmt.Printf("交换前，a 的值 : %d\n", a )
   fmt.Printf("交换前，b 的值 : %d\n", b )

   /* 调用 swap() 函数
   * &a 指向 a 指针，a 变量的地址
   * &b 指向 b 指针，b 变量的地址
   */
   swap(&a, &b)

   fmt.Printf("交换后，a 的值 : %d\n", a )
   fmt.Printf("交换后，b 的值 : %d\n", b )
}

func swap(x *int, y *int) {
   var temp int
   temp = *x    /* 保存 x 地址上的值 */
   *x = *y      /* 将 y 值赋给 x */
   *y = temp    /* 将 temp 值赋给 y */
}
```

| [Go 指针数组](https://www.runoob.com/go/go-array-of-pointers.html) | 你可以定义一个指针数组来存储地址             |
| ------------------------------------------------------------ | -------------------------------------------- |
| [Go 指向指针的指针](https://www.runoob.com/go/go-pointer-to-pointer.html) | Go 支持指向指针的指针                        |
| [Go 向函数传递指针参数](https://www.runoob.com/go/go-passing-pointers-to-functions.html) | 通过引用或地址传参，在函数调用时可以改变其值 |

```go
package main
import (
   "fmt"
   "math"
)
func main(){
   /* 声明函数变量 */
   getSquareRoot := func(x float64) float64 {
      return math.Sqrt(x)
   }
   /* 使用函数 */
   fmt.Println(getSquareRoot(9))
}
```

#### .1. fmt

- `print`：直接输出内容，不会换行，不能格式化输出。
- `printf`：能够通过占位符输出格式化的内容。
- `println`：能够在输出内容后面加上换行符。

```go
package main

import "fmt"

func main() {
    var name string
    fmt.Print("输入你的姓名:")
    fmt.Scan(&name)
    fmt.Printf("你输入的姓名是：%s", name)
}
```

#### .2. defer 函数

> `Go` 函数的关键字 `defer` 可以提供注册多个延迟调用，只能出现在函数内部，在 `defer` 归属的函数即将返回时，将延迟处理的语句按 `defer` 的[逆序](https://so.csdn.net/so/search?q=逆序&spm=1001.2101.3001.7020)进行执行，这些调用遵循先进后出的顺序在函数返回前被执行。

> 在函数中，程序员经常需要创建资源(比如：数据库连接、文件句柄、锁等) ，为了在函数执行完毕后，及时的释放资源，Go 的设计者提供 defer (延时机制)。

- `defer` 函数的实参在注册时使用值拷贝传递进去，即 `defer` 后面的函数参数会被实时解析；
- 当主动调用 `os.Exit(int)` 退出进程时， `defer` 即使已经注册，那么也不再被执行。
- `defer` 尽量不要放到循环语句内部；
- `defer` 后边调用的函数如果有返回值，则这个返回值将会被丢弃。

```go
package main

import "fmt"

func main() {
	a := 10
	defer func(i int) {
		fmt.Println("defer func i is ", i)	// defer func i is  10
	}(a)

	a += 10
	fmt.Println("after defer a is ", a)	// after defer a is  20	

}
```

```go
package main

func test() {
	x, y := 10, 20
	defer func(i int) {
		println("defer:", i, y) // y 闭包引用
	}(x) // x 被复制
	x += 10
	y += 100
	println("x =", x, "y =", y)
}

func main() {
	test()
}
```



##### 1. 使用延迟并发解锁

```go
var (
    // 一个演示用的映射
    valueByKey      = make(map[string]int)
    // 保证使用映射时的并发安全的互斥锁
    valueByKeyGuard sync.Mutex
)

// 根据键读取值
func readValue(key string) int {
    // 对共享资源加锁
    valueByKeyGuard.Lock()
    // 取值
    v := valueByKey[key]
    // 对共享资源解锁
    valueByKeyGuard.Unlock()
    // 返回值
    return v
}
```

```go
func readValue(key string) int {
    valueByKeyGuard.Lock()
   
    // defer后面的语句不会马上调用, 而是延迟到函数结束时调用
    defer valueByKeyGuard.Unlock()
    return valueByKey[key]
}
```

##### 2. 使用延迟释放文件句柄

```go
// 根据文件名查询其大小
func fileSize(filename string) int64 {
    // 根据文件名打开文件, 返回文件句柄和错误
    f, err := os.Open(filename)
    // 如果打开时发生错误, 返回文件大小为0
    if err != nil {
        return 0
    }
    // 取文件状态信息
    info, err := f.Stat()
   
    // 如果获取信息时发生错误, 关闭文件并返回文件大小为0
    if err != nil {
        f.Close()
        return 0
    }
    // 取文件大小
    size := info.Size()
    // 关闭文件
    f.Close()
   
    // 返回文件大小
    return size
}
```

```go
func fileSize(filename string) int64 {
    f, err := os.Open(filename)
    if err != nil {
        return 0
    }
    // 延迟调用Close, 此时Close不会被调用
    defer f.Close()
    info, err := f.Stat()
    if err != nil {
        // defer机制触发, 调用Close关闭文件
        return 0
    }
    size := info.Size()
    // defer机制触发, 调用Close关闭文件
    return size
}
```

#### .3. 字符串相关函数

```go
package main

import (
	"fmt"
	"strconv"
	"strings"
)

func main() {

	// (1) 统计字符串长度
	str := "hello北"
	fmt.Println("len(str) =", len(str))  // 8

	// (2) 含有中文的字符串遍历
	str2 := []rune(str)  // 使用切片
	for i := 0; i < len(str2); i++ {  // hello北
		fmt.Printf("%c", str2[i])
	}
	fmt.Println()

	// (3) 字符串转整数
	num, err := strconv.Atoi("123")
	if err != nil {
		fmt.Println("转化错误：", err)
	} else {
		fmt.Println("转化结果：", num)
	}

	// (4) 整数转字符串
	str = strconv.Itoa(12345)

	// (5) 字符串 转 []byte
	var bytes = []byte("hello go")
	fmt.Printf("str = %v\n", bytes)

	// (6) []bytes 转 字符串
	str = string([]byte{97, 98, 99})

	// (7) 10进制转为其他进制字符串
	str = strconv.FormatInt(123, 3)  // 转为3进制

	// (8) 判断是否包含某个子串
	b := strings.Contains("hello go!", "go")
	fmt.Printf("b = %v\n", b)  // true

	// (9) 统计一个字符串有几个指定的子串
	num = strings.Count("go hello go!", "go")
	fmt.Printf("num = %v\n", num)

	// (10) 判断字符串是否相等
	b = "abc" == "Abc"
	fmt.Printf("b = %v\n", b)  // false

	// (11) 判断字符串是否相等（不区分大小写）
	b = strings.EqualFold("abc", "Abc")
	fmt.Printf("b = %v\n", b)  // true

	// (12) 返回子串在字符串中第一次出现的位置，如果没有返回-1
	index := strings.Index("go hello go!", "go")
	fmt.Println(index)  // 0

	// (13) 返回子串在字符串中最后一次出现的位置，如果没有返回-1
	index = strings.LastIndex("go hello go!", "go")
	fmt.Println(index)  // 9

	// (14) 将字符串中的指定子串替换为另一个子串，最后一个参数为-1代表全部替换
	str = strings.Replace("go hello go!", "go", "北京", 1)  // 只替换一次，从前面开始找
	fmt.Println(str)  // 北京 hello go!

	// (15) 将字符串按照某个字符分割成字符串数组
	strArr := strings.Split("hello,world,go", ",")
	fmt.Println(strArr)  // [hello world go]

	// (16) 将字符串全部转化为大写/小写
	str = "goLang Hello 北京!"
	fmt.Println(strings.ToUpper(str))  // GOLANG HELLO 北京!
	fmt.Println(strings.ToLower(str))

	// (17) 将字符串左右两边的空格去掉
	str = strings.TrimSpace("   hello golang   ")  // golang hello 北京!
	fmt.Println(str)  // hello golang

	// (18) 将字符串左右两边指定的字符去掉
	str = strings.Trim("!    hello world! go!   ", " !")  // 去掉左右两侧的 空格 和 !
	fmt.Println(str)  // hello world! go

	// (19) 将字符串左/右两边指定的字符去掉
	fmt.Println(strings.TrimLeft("!!!hello world! go!!!", "!"))  // hello world! go!!!
	fmt.Println(strings.TrimRight("!!!hello world! go!!!", "!"))  // !!!hello world! go

	// (20) 判断字符串是否以某个前缀/后缀开头
	fmt.Println(strings.HasPrefix("https://www.baidu.com", "https"))  // true
	fmt.Println(strings.HasSuffix("https://www.baidu.com", "cn"))  // false
}
```

#### .4. 时间相关函数

```go
package main

import (
	"fmt"
	"time"
)

func main() {

	// (1) 获取当前时间
	now := time.Now()
	fmt.Printf("now = %v type = %T\n", now, now)

	// (2) 通过 now 可以获取到年月日，时分秒
	fmt.Printf("年 = %v\n", now.Year())
	fmt.Printf("月 = %v\n", now.Month())
	fmt.Printf("月 = %v\n", int(now.Month()))
	fmt.Printf("日 = %v\n", now.Day())
	fmt.Printf("时 = %v\n", now.Hour())
	fmt.Printf("分 = %v\n", now.Minute())
	fmt.Printf("秒 = %v\n", now.Second())

	// (3) 格式化日期时间
	// 方式一
	fmt.Printf("当前时间 = %d-%d-%d %d:%d:%d\n", now.Year(), now.Month(), now.Day(),
		now.Hour(), now.Minute(), now.Second())
	dataStr := fmt.Sprintf("%d-%d-%d %d:%d:%d\n", now.Year(), now.Month(), now.Day(),
		now.Hour(), now.Minute(), now.Second())
	fmt.Printf("dataStr = %v\n", dataStr)
	// 方式二
	fmt.Printf(now.Format("2006/01/02 15:04:05"))
	fmt.Println()
	fmt.Printf(now.Format("2006-01-02"))
	fmt.Println()
	fmt.Printf(now.Format("15:04:05"))
	fmt.Println()

	// (4) 休眠函数：每隔 0.1s 打印一个数字，从 1 打印到 5
	fmt.Println()
	for i := 1; i <= 5; i++ {
		fmt.Println(i)
		time.Sleep(time.Millisecond * 100) // 参数不能是浮点数
	}

	// (5) Unix(秒) 和 UnixNano(纳秒) 的使用
	fmt.Println()
	fmt.Printf("Unix时间戳 = %v Unixnano时间戳 = %v\n", now.Unix(), now.UnixNano())
}
```

#### 5. 内置函数

```go
package main

import "fmt"

func main() {

	num1 := 100
	fmt.Printf("type of num1 = %T, val of num1 = %v, address of num1 = %v\n", num1, num1, &num1)

	num2 := new(int)  // num2是一个指针，其内容存储的是一个地址，该地址对应内容默认为0
	*num2 = 100        // 
	fmt.Printf("type of num2 = %T, val of num2 = %v, address of num2 = %v, *num2 = %v\n",
		num2, num2, &num2, *num2)
}
```

#### .6. 错误处理

> Go语言追求简洁优雅，所以，Go语言不支持传统的try... catch... finally 这种处理。
>
> Go中引入的处理方式为: defer, panic, recover。
>
> Go 中可以`抛出一个panic的异常`，然后`在defer中通过recover捕获`这个异常，然后正常处理。

```
package main
 
import (
	"fmt"	
)
func test() {
	//捕获处理异常
	defer func () {
		err := recover() //可以捕获到异常
		if err != nil { 
			fmt.Println("err=",err)
		}
	}()
	num1 := 4
	num2 := 0
	res := num1 / num2
	fmt.Println("res=",res)
}
 
func main() {
	test()
	fmt.Println("hello")
}
```
> 使用`errors.New和panic内置函数`。
>
> 1、 errors New(错误说明")。会返回一一个error类型的值，表示一个错误 
>
> 2、 panic 内置函数接收一个interface{}类型的值(也就是任何值了)作为参数。可以`接收error类型的变量，输出错误信息，并退出程序`。

```go
package main
 
import (
	"fmt"
	"errors"	
)
func read(name string) (err error) {
	if name == "AA" {
		return nil
	} else {
        //返回自定义错误
		return errors.New("名字有误")
	}
}
func test() {
	err := read("aa")
	if err != nil {
		panic(err)
	}
	fmt.Println("继续执行")
}
func main() {
	test()
	
}
```

#### .7. 值传递&引用传递

> （1）值类型：int系列、float系列、bool、string、数组、结构体；
>
> （2）引用类型：指针、slice、map、chain、interface

```go
package main

import (
	"fmt"
)

type Person struct {
	Name string `json:"name"` // 反引号中间的为 tag
	Age  int    `json:"age"`
}

func (p Person) test01() { // 这里的 p 是值传递
	p.Name = "Jack" // 不会影响外部的p
	fmt.Printf("Name = %v, Age = %v\n", p.Name, p.Age)
}

func (p *Person) test02() { // 这里的 p 是值传递
	p.Name = "Jack" // 会影响外部的p
	fmt.Printf("Name = %v, Age = %v\n", p.Name, p.Age)
}

func (p *Person) String() string {
	return fmt.Sprintf("Name = [%v], Age = [%v]", p.Name, p.Age)
}

// int, float32等也可以有方法
type integer int

func (i integer) test03() {
	i++
	fmt.Println("i =", i)
}

func (i *integer) test04() {
	*i++
	fmt.Println("i =", *i)
}

// 结构体属于值类型
func main() {

	// (1) 使用结构体调用方法
	p1 := Person{"Tom", 18}

	p1.test01()
	fmt.Println(p1)
	fmt.Println(&p1) // 会输出 String() 函数返回的字符串

	p1.test02()
	fmt.Println(p1)
	fmt.Println(&p1) // 会输出 String() 函数返回的字符串

	// (2) 使用结构体指针调用方法
	p2 := &Person{
		Name: "Marry",
		Age:  16,
	}
	p2.test01() // 等价于：(*p2).test01()

	// 基本数据类型也可以有方法
	var i integer = 10

	i.test03()
	fmt.Println(i)

	i.test04()
	fmt.Println(i)
}
```

#### .8. range 函数

> [range函数](https://so.csdn.net/so/search?q=range函数&spm=1001.2101.3001.7020)是个神奇而有趣的内置函数，你可以使用它来遍历数组，切片和字典。
>
> - 当用于遍历`数组和切片`的时候，`range函数返回索引和元素`；
>
> - 当用于遍历`字典的`时候，range函数返回字典的`键和值`。

```go
package main
 
import "fmt"
 
func main() {
 
    // 这里我们使用range来计算一个切片的所有元素和
    // 这种方法对数组也适用
    nums := []int{2, 3, 4}
    sum := 0
    for _, num := range nums {
        sum += num
    }
    fmt.Println("sum:", sum)
 
    // range 用来遍历数组和切片的时候返回索引和元素值
    // 如果我们不要关心索引可以使用一个下划线(_)来忽略这个返回值
    // 当然我们有的时候也需要这个索引
    for i, num := range nums {
        if num == 3 {
            fmt.Println("index:", i)
        }
    }
 
    // 使用range来遍历字典的时候，返回键值对。
    kvs := map[string]string{"a": "apple", "b": "banana"}
    for k, v := range kvs {
        fmt.Printf("%s -> %s\n", k, v)
    }
 
    // range函数用来遍历字符串时，返回Unicode代码点。
    // 第一个返回值是每个字符的起始字节的索引，第二个是字符代码点，
    // 因为Go的字符串是由字节组成的，多个字节组成一个rune类型字符。
    for i, c := range "go" {
        fmt.Println(i, c)
    }
}
```

```go
func main() {
    var m = []int{1, 2, 3, 4, 5}  
             
    for i, v := range m {
        go func() {
            time.Sleep(time.Second * 3)  //变量i,v在主Goroutine和启动Gorouting之间共享
            fmt.Println(i, v)   //这里会输出4,5
        }()
    }

    time.Sleep(time.Second * 10)
}
// 正确迭代版本
func main() {
    var m = []int{1, 2, 3, 4, 5}

    for i, v := range m {
        go func(i, v int) {
            time.Sleep(time.Second * 3)
            fmt.Println(i, v)
        }(i, v)
    }

    time.Sleep(time.Second * 10)
}
```

> **`range` 会复制对象， `range` 返回的是每个元素的副本，而不是直接返回对该元素的引用**。

```go
// range 值类型，  range中修改的数据不影响元数据
package main

import "fmt"

func main() {
	a := [3]int{0, 1, 2}
	for i, v := range a { // index、value 都是从复制品中取出。
		if i == 0 { // 在修改前，我们先修改原数组。
			a[1], a[2] = 999, 999
			fmt.Println(a) // 确认修改有效，输出 [0, 999, 999]。
		}
		a[i] = v + 100 // 使用复制品中取出的 value 修改原数组。
	}
	fmt.Println(a) // 输出 [100, 101, 102]。

}  
for i, v := range a[:] 
# or
for i, v := range &a

// range 引用类型， range中修改的数据影响元数据
package main

func main() {
	s := []int{1, 2, 3, 4, 5}
	for i, v := range s { // 复制 struct slice { pointer, len, cap }。
		if i == 0 {
			s = s[:3]  // 对 slice 的修改，不会影响 range。
			s[2] = 100 // 对底层数据的修改。
		}
		println(i, v)
	}

}
```

#### .9.  闭包

> `Go` 语言闭包函数的定义：当匿名函数引用了外部作用域中的变量时就成了闭包函数，闭包函数是函数式编程语言的核心。
>
> 也就是`匿名函数可以会访问其所在的外层函数内的局部变量`。当`外层函数运行结束后，匿名函数会与其使用的外部函数的局部变量形成闭包`。

### 6. 结构体

```go
package main

import "fmt"

type Books struct {
   title string
   author string
   subject string
   book_id int
}

func main() {
   var Book1 Books        /* 声明 Book1 为 Books 类型 */
   var Book2 Books        /* 声明 Book2 为 Books 类型 */

   /* book 1 描述 */
   Book1.title = "Go 语言"
   Book1.author = "www.runoob.com"
   Book1.subject = "Go 语言教程"
   Book1.book_id = 6495407
    /* 打印 Book1 信息 */
   printBook(&Book1)
}
func printBook( book *Books ) {
   fmt.Printf( "Book title : %s\n", book.title)
   fmt.Printf( "Book author : %s\n", book.author)
   fmt.Printf( "Book subject : %s\n", book.subject)
   fmt.Printf( "Book book_id : %d\n", book.book_id)
}
```

```go
package main

import (
	"encoding/json"
	"fmt"
)

type Person struct {
	Name string `json:"name"` // 反引号中间的为 tag
	Age  int    `json:"age"`
}

// 结构体属于值类型
func main() {

	// (1) 声明
	// 方式一
	var p1 Person // 默认为0值、空值
	fmt.Println(p1)
	// 方式二
	p2 := Person{"Tom", 18}
	fmt.Println(p2)
	// 方式三
	var p3 *Person = new(Person)
	p3.Name = "Jack" // 等价于：(*p3).Name = "Jack"
	p3.Age = 20      // 等价于：(*p3).Age = 20
	fmt.Println(*p3)
	// 方式四
	var p4 *Person = &Person{
		Name: "scott",
		Age:  25,
	}
	fmt.Println(*p4)

	// (2) tag 的使用，用于在转化为 json 字符串后让字段首字母小写
	if jt, err := json.Marshal(p1); err != nil {
		fmt.Println("err...")
	} else {
		fmt.Println(string(jt))
	}
}
/** 输出结果：
{ 0}
{Tom 18}           
{Jack 20}          
{scott 25}         
{"name":"","age":0}
**/
```

```go
type Rect struct {
	x, y          float64
	width, height float64
}
//成员方法
func (r *Rect) Area() float64 {
	return r.width * r.height
}

rect1 := new(Rect)
rect2 := &Rect{}
rect3 := &Rect{0, 0, 100, 200}
rect4 := &Rect{width: 100, height: 200}

func NewRect(x, y, width, height float64) *Rect {
	return &Rect{x, y, width, height}
}
```



### 7. map 集合

> **内建函数 make 用来为 slice，map 或 chan 类型分配内存和初始化一个对象(注意：只能用在这三种类型上)**,第一个参数也是一个类型而不是一个值，跟 new 不同的是，make 返回类型的引用而不是指针，而返回值也依赖于具体传入的类型，
>
> 内建函数 new 用来分配内存，它的第一个参数是一个类型，不是一个值，`它的返回值是一个指向新分配类型零值的指针`, 并初始化内存，只是将其置为0.

```go
// 长度为5，容量为10的slice，slice中的元素是int
var slice_ []int = make([]int,5,10)
```

```go
package main

import (
	"fmt"
	"sort"
)

// map属于引用类型，默认无序
func main() {

	// (1) 定义
	// 方式一
	var a map[string]string
	a = make(map[string]string, 10)
	a["i"] = "you"
	a["you"] = "i"
	fmt.Println(a)
	// 方式二
	b := make(map[string]string)
	b["byte"] = "dance"
	b["dance"] = "byte"
	fmt.Println(b)
	// 方式三
	c := map[string]string{
		"Inspire": "Creativity",
		"Enrich":  "Life",
	}
	c["byte"] = "dance"
	fmt.Println(c)

	// (2) crud
	d := make(map[int]int)
	// 增加/更改: 如果key没有就是增加，否则就是更改
	d[1] = 2
	d[10] = 20
	d[5] = 21
	d[3] = 4
	fmt.Println(d)
	// 删除: 存在即删除，不存在也不报错
	delete(d, 10)
	fmt.Println(d)
	// 查找
	if val, ok := d[5]; ok {
		fmt.Println("val =", val)
	} else {
		fmt.Println("不存在该key")
	}

	// (3) 遍历
	for k, v := range d {
		fmt.Printf("k = %v, v = %v\n", k, v)
	}

	// (4) 长度(没有容量一说)
	fmt.Println(len(d))

	// (5) 清空map: 没有专门的方法，直接新建一个即可
	d = make(map[int]int)

	// (6) map切片
	g := make([]map[string]string, 2)
	if g[0] == nil {
		g[0] = make(map[string]string, 2)
		g[0]["name"] = "牛魔王"
		g[0]["age"] = "100"
	}
	if g[1] == nil {
		g[1] = make(map[string]string, 2)
		g[1]["name"] = "玉兔精"
		g[1]["age"] = "400"
	}
	el := map[string]string{
		"name": "火云邪神",
		"age":  "200",
	}
	g = append(g, el)
	fmt.Println(g)

	// (7) 排序：先将key排好序，然后遍历key即可
	t := map[int]int{
		10: 100,
		1:  13,
		8:  90,
		4:  6,
	}
	var keys []int
	for k, _ := range t {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		fmt.Printf("t[%v] = %v\n", k, t[k])
	}
}
```

### 8. 数组

```go
package main

import "fmt"

func test01(arr [3]int) { // 值传递，不会影响外界数组
	arr[0] = 10
}

func test02(arr *[3]int) { // 指针传递，会影响外界数组
	arr[0] = 10 // 等价于：(*arr)[0] = 10
}

// 数组属于值类型
func main() {

	/************* 一维数组 **************/
	// (1) 数组的定义
	var arr1 [3]int // 方式一
	fmt.Println(arr1)
	arr2 := [...]int{1, 2, 3} // 方式二
	fmt.Println(arr2)

	// (2) 数组的遍历
	// 方式一
	for i := 0; i < len(arr2); i++ {
		fmt.Println(arr2[i])
	}
	// 方式二
	for index, val := range arr2 {
		fmt.Printf("index = %v, val = %v\n", index, val)
	}

	// (3) 验证数组是值传递
	test01(arr2)
	fmt.Println(arr2)
	test02(&arr2)
	fmt.Println(arr2)

	/************* 二维数组 **************/
	fmt.Println()
	// (1) 定义
	var arr3 [2][3]int // 方式一
	fmt.Println(arr3)
	arr4 := [2][3]int{{1, 2, 3}, {4, 5, 6}} // 方式二
	fmt.Println(arr4)

	// (2) 遍历
	// 方式一
	for i := 0; i < len(arr4); i++ {
		for j := 0; j < len(arr4[0]); j++ {
			fmt.Printf("%v\t", arr4[i][j])
		}
		fmt.Println()
	}
	// 方式二
	for _, v := range arr4 {
		for _, v2 := range v {
			fmt.Printf("%v\t", v2)
		}
		fmt.Println()
	}
}
```

### 9. 切片

```go
package main

import "fmt"

// 切片属于引用类型
func main() {

	// (1) 切片的定义
	// 方式一
	arr1 := [...]int{1, 2, 3, 4, 5} // 定义一个数组
	s1 := arr1[1:3]                 // s1是一个slice
	fmt.Println(s1)
	// 方式二
	s2 := make([]float64, 2, 4) // 容量为4，实际存储2个元素
	fmt.Println(s2)
	// 方式三
	s3 := []int{1, 2, 3}
	fmt.Println(s3)
	// 方式四
	var s4 []int
	fmt.Println(s4)

	// (2) 切片的长度、容量
	fmt.Printf("len(s2) = %v\n", len(s2))
	fmt.Printf("cap(s2) = %v\n", cap(s2))

	// (3) 遍历切片
	// 方式一
	for i := 0; i < len(s3); i++ {
		fmt.Println(s3[i])
	}
	// 方式二
	for _, val := range s3 {
		fmt.Printf("val = %v\n", val)
	}

	// (4) 切片后面追加元素
	s3 = append(s3, 10, 20)
	fmt.Println(s3)
}
```

### 10. 封装

```go
package main

import (
	"fmt"
)

// 矩形结构体
type Rectangle struct {
	Length int
	Width  int
}

// 计算矩形面积
func (r *Rectangle) Area() int {
	return r.Length * r.Width
}

func main() {
	r := Rectangle{4, 2}
	// 调用 Area() 方法，计算面积
	fmt.Println(r.Area())
}
```

### 10. 继承

```go
package main

import "fmt"

type Student struct { // 父类
	Name  string
	Age   int
	Score int
}

func (stu *Student) ShowInfo() {
	fmt.Printf("student : Name = %v Age = %v Score = %v\n", stu.Name, stu.Age, stu.Score)
}
func (stu *Student) SetScore(score int) {
	stu.Score = score
}

type Graduate struct { // 子类, 可以使用父类的所有字段和方法(即使首字母是小写的)
	Student
}

func (p *Graduate) testing() {
	fmt.Println("大学生正在考试...")
}

type Pupil struct { // 子类
	Student
}

func (p *Pupil) testing() {
	fmt.Println("小学生正在考试...")
}

// 演示结构体继承赋值
type Goods struct {
	Name  string
	Price float64
}

type Brand struct {
	Name    string
	Address string
}

type TV struct {
	Goods
	Brand
}

type Phone struct {
	*Goods
	*Brand
}

// 继承：通过匿名结构体实现
// 这里演示：大学生和小学生 都继承 学生
func main() {

	pupil := &Pupil{}
	pupil.Name = "Tom" // 就近访问原则，如果Pupil中也有Name字段，则这里使用的是Pupil中的Name
	pupil.Student.Age = 8
	pupil.testing()
	pupil.SetScore(90)
	pupil.Student.ShowInfo()

	graduate := &Graduate{}
	graduate.Name = "Tom"
	graduate.Student.Age = 20
	graduate.testing()
	graduate.SetScore(92)
	graduate.Student.ShowInfo()

	// 演示继承结构体初始化
	tv1 := TV{Goods{"电视机01", 3000.1}, Brand{"海尔", "山东"}}
	tv2 := TV{
		Goods{
			Price: 3200.1,
			Name:  "电视机02",
		},
		Brand{
			Name:    "夏普",
			Address: "北京",
		},
	}
	fmt.Println(tv1)
	fmt.Println(tv2)

	phone1 := Phone{&Goods{"手机01", 3000.1}, &Brand{"华为", "山东"}}
	phone2 := Phone{
		&Goods{
			Price: 3200.1,
			Name:  "手机02",
		},
		&Brand{
			Name:    "小米",
			Address: "北京",
		},
	}
	fmt.Println(*phone1.Goods, *phone1.Brand)
	fmt.Println(*phone2.Goods, *phone2.Brand)
}
/** 输出结果：
小学生正在考试...
student : Name = Tom Age = 8 Score = 90 
大学生正在考试...                       
student : Name = Tom Age = 20 Score = 92
{{电视机01 3000.1} {海尔 山东}}         
{{电视机02 3200.1} {夏普 北京}}         
{手机01 3000.1} {华为 山东}             
{手机02 3200.1} {小米 北京}
**/
```

### 11. 多态

```go
package main

import (
	"fmt"
)

// 正方形
type Square struct {
	side float32
}

// 长方形
type Rectangle struct {
	length, width float32
}

// 接口 Shaper
type Shaper interface {
	Area() float32
}

// 计算正方形的面积
func (sq *Square) Area() float32 {
	return sq.side * sq.side
}

// 计算长方形的面积
func (r *Rectangle) Area() float32 {
	return r.length * r.width
}

func main() {
// 创建并初始化 Rectangle 和 Square 的实例，由于这两个实例都实现了接口中的方法，
//所以这两个实例，都可以赋值给接口 Shaper 
	r := &Rectangle{10, 2}
	q := &Square{10}

	// 创建一个 Shaper 类型的数组
	shapes := []Shaper{r, q}
	// 迭代数组上的每一个元素并调用 Area() 方法
	for n, _ := range shapes {
		fmt.Println("矩形数据: ", shapes[n])
		fmt.Println("它的面积是: ", shapes[n].Area())
	}
}

/*
矩形数据:  &{10 2}
它的面积是:  20
图形数据:  &{10}
它的面积是:  100
*/
```

### 11. [接口](https://blog.csdn.net/wohu1104/article/details/106202971)

```go
/* 定义接口 */
type interface_name interface {
    method_name1 [return_type]
    method_name2 [return_type]
    method_name3 [return_type]
    ...
    method_namen [return_type]
}

/* 定义结构体 */
type struct_name struct {
    /* variables */
}

/* 实现接口方法 */
func (struct_name_variable struct_name) method_name1() [return_t
ype] {
    /* 方法实现 */
}
...
func (struct_name_variable struct_name) method_namen() [return_t
ype] {
    /* 方法实现*/
}
```

```go
package main

import "fmt"

type Usb interface { // 接口
	Start()
	Stop()
}

type Phone struct {
}

func (p Phone) Start() {
	fmt.Println("手机开始工作...")
}
func (p Phone) Stop() {
	fmt.Println("手机停止工作...")
}

type Camera struct {
}

func (c Camera) Start() {
	fmt.Println("相机开始工作...")
}
func (c Camera) Stop() {
	fmt.Println("相机停止工作...")
}

type Computer struct {
}

func (c Computer) Working(usb Usb) {
	usb.Start()
	usb.Stop()
}

type integer int
func (i integer) Start() {
	fmt.Println("integer start...")
}
func (i integer) Stop() {
	fmt.Println("integer stop...")
}

// 接口属于引用类型
func main() {

	computer := Computer{}
	phone := Phone{}
	camera := Camera{}

	computer.Working(phone)
	computer.Working(camera)

	// 实现接口的实例可以赋值给接口变量
	fmt.Println()
	var a Usb = Phone{}
	a.Start()

	// 自定义类型实现接口
	fmt.Println()
	var i integer
	computer.Working(i)
}
/** 输出结果：
手机开始工作...
手机停止工作...
相机开始工作...
相机停止工作...

手机开始工作...

integer start...
integer stop...
**/
```

### 12. 断言

```go
package main

import "fmt"

func main() {

	var b float64 = 3.14
	var x interface{}
	x = b

	if y, ok := x.(float64); ok { // 类型断言
		fmt.Printf("type of y = %T, val of y = %v\n", y, y)
	} else {
		fmt.Println("convert fail...")
	}
}
/** 输出结果：
type of y = float64, val of y = 3.14
**/
```

### 13. 文件操作

```go
package main

import (
    "bufio"
    "fmt"
    "io"
    "io/ioutil"
    "os"
)

/**test.txt文件内容
Hello,World!
北京！abc!golang!
go, hello world!

 */

func main() {

    // (1) 打开文件
    file, err := os.Open("d:/test.txt")
    if err != nil {
        fmt.Println("open file err =", err)
        return
    }

    fmt.Printf("file = %v\n", file)

    // (2) 关闭文件
    defer func(file *os.File) {
        err := file.Close()
        if err != nil {
            fmt.Println("close file err =", err)
        }
    }(file)

    // (3) 读取文件内容并显示(带缓冲区)
    fmt.Println()
    reader := bufio.NewReader(file)
    for {
        if str, err := reader.ReadString('\n'); err != io.EOF {
            fmt.Printf(str)
        } else {
            break
        }
    }

    // (4) 读取文件内容并显示(一次性读取)
    fmt.Println()
    if content, err := ioutil.ReadFile("d:/test.txt"); err == nil {
        fmt.Println(string(content))
    } else {
        fmt.Println(err)
    }

    // (5) 打开文件，并向其中写入数据
    file2, err2 := os.OpenFile("d:/abc.txt", os.O_WRONLY | os.O_CREATE, 0666)
    if err2 != nil {
        fmt.Printf("open file err = %v\n", err)
        return
    }
    defer func(file2 *os.File) {
        err := file2.Close()
        if err != nil {
            fmt.Println(err)
        }
    }(file2)
    writer := bufio.NewWriter(file2)
    for i := 0; i < 2; i++ {
        writer.WriteString("hello go!\n")
    }
    writer.Flush()  // 没有这句话，无法将数据写入文件

    // (6) 判断文件是否存在
    _, err3 := os.Stat("d:/abc.txt")
    if err3 == nil {
        fmt.Println("文件存在...")
    } else if os.IsNotExist(err3) {
        fmt.Println("文件不存在...")
    } else {
        fmt.Println("其他错误，错误信息为:", err3)
    }
}
/** 输出结果：
file = &{0xc00007e780}

Hello,World!
北京！abc!golang!
go, hello world!

Hello,World!
北京！abc!golang!
go, hello world!

文件存在...
**/
```

```go
package main

import (
	"fmt"
	"os"
)

func main() {
	fileName := "/home/wohu/gocode/src/test.txt"
	writeFile(fileName)
	readFile(fileName)

}

func writeFile(fileName string) {
	file, err := os.Create(fileName)

	if err != nil {
		fmt.Println(err)
		return
	}

	for i := 0; i <= 5; i++ {
		outStr := fmt.Sprintf("%s:%d\n", "hello, world", i)

		file.WriteString(outStr)
		file.Write([]byte("abcd\n"))

	}

	file.Close()
}

func readFile(fileName string) {
	file, err := os.Open(fileName)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close()

	buf := make([]byte, 1024)

	for {
		n, _ := file.Read(buf)

		if n == 0 {
			//0	表示到达EOF
			break
		}
		os.Stdout.Write(buf)
	}
}
```

### 14. 命令行参数

```go
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("参数个数 =", len(os.Args))
	for i, v := range os.Args {
		fmt.Printf("args[%v] = %v\n", i, v)
	}
}
```

```go
package main

import (
	"flag"
	"fmt"
)

/** 命令行操作
go build -o main.exe main.go
.\main.exe -h 127.0.0.1 -u root -pwd 123456
*/
func main() {

	var user string
	var pwd string
	var host string
	var port int

	// 四个参数：
	// (1) 负责接收用户输入值得变量
	// (2) 例如"-u" 表示接收用户输入的 -u 后面的参数
	// (3) 默认值
	// (4) 说明字段
	flag.StringVar(&user, "u", "", "用户名，默认为空")
	flag.StringVar(&pwd, "pwd", "", "密码，默认为空")
	flag.StringVar(&host, "h", "localhost", "主机名，默认为localhost")
	flag.IntVar(&port, "port", 3306, "端口号，默认为3306")

	flag.Parse() // 必须调用该方法

	// 输出结果
	fmt.Printf("user = %v, pwd = %v, host = %v, port = %v\n",
		user, pwd, host, port)
}
/** 输出结果：
user = root, pwd = 123456, host = 127.0.0.1, port = 3306
**/
```

### 15. json

```go
package main

import (
	"encoding/json"
	"fmt"
)

type Monster struct {
	Name string `json:"name"`
	Age int `json:"age"`
	Birthday string `json:"birthday"`
	Sal float64 `json:"sal"`
	Skill string `json:"skill"`
}

func testStruct() string {

	monster := Monster{
		Name: "牛魔王",
		Age: 500,
		Birthday: "2011-11-11",
		Sal: 8000.0,
		Skill: "牛魔拳",
	}

	data, err := json.Marshal(&monster)
	if err != nil {
		fmt.Printf("序列化错误，err = %v\n", err)
	} else {
		fmt.Printf("monster序列化后 = %v \n", string(data))
	}

	return string(data)
}

func testMap() string {

	var a map[string]interface{}
	a = make(map[string]interface{})
	a["name"] = "红孩儿"
	a["age"] = 30
	a["address"] = "洪崖洞"

	data, err := json.Marshal(a)  // 默认指针传递
	if err != nil {
		fmt.Printf("序列化错误，err = %v\n", err)
	} else {
		fmt.Printf("a序列化后 = %v \n", string(data))
	}

	return string(data)
}

func testSlice() string {

	var s []map[string]interface{}

	var m1 map[string]interface{}
	m1 = make(map[string]interface{})
	m1["name"] = "jack"
	m1["age"] = 7
	m1["address"] = "北京"
	s = append(s, m1)

	var m2 map[string]interface{}
	m2 = make(map[string]interface{})
	m2["name"] = "tom"
	m2["age"] = 20
	m2["address"] = "墨西哥"
	s = append(s, m2)

	data, err := json.Marshal(s)  // 默认指针传递
	if err != nil {
		fmt.Printf("序列化错误，err = %v\n", err)
	} else {
		fmt.Printf("s序列化后 = %v \n", string(data))
	}

	return string(data)
}

func testFloat64() {

	var num float64 = 3.14

	data, err := json.Marshal(num)  // 默认指针传递
	if err != nil {
		fmt.Printf("序列化错误，err = %v\n", err)
	} else {
		fmt.Printf("num序列化后 = %v \n", string(data))
	}
}

func unmarshalStruct(s string) {

	var monster Monster

	err := json.Unmarshal([]byte(s), &monster)
	if err != nil {
		fmt.Printf("unmarshal err = %v\n", err)
	} else {
		fmt.Printf("反序列化后 monster = %v\n", monster)
	}
}

func unmarshalMap(s string) {

	var a map[string]interface{}

	err := json.Unmarshal([]byte(s), &a)  // 即使是map，也要加上 &，不需要make
	if err != nil {
		fmt.Printf("unmarshal err = %v\n", err)
	} else {
		fmt.Printf("反序列化后 a = %v\n", a)
	}
}

func unmarshalSlice(s string) {

	var res []map[string]interface{}

	err := json.Unmarshal([]byte(s), &res)
	if err != nil {
		fmt.Printf("unmarshal err = %v\n", err)
	} else {
		fmt.Printf("反序列化后 res = %v\n", res)
	}
}

func main() {

	/******************** 序列化 ********************/
	// 结构体 序列化
	a := testStruct()

	// map 序列化
	b := testMap()

	// slice 序列化
	c := testSlice()

	// 基本类型 序列化，意义不大
	testFloat64()

	/******************** 反序列化 ********************/
	// 反结构体 序列化
	unmarshalStruct(a)

	// map 反序列化
	unmarshalMap(b)
	
	// slice 反序列化
	unmarshalSlice(c)
}
/** 输出结果：
monster序列化后 = {"name":"牛魔王","age":500,"birthday":"2011-11-11","sal":8000,"skill":"牛魔拳"} 
a序列化后 = {"address":"洪崖洞","age":30,"name":"红孩儿"} 
s序列化后 = [{"address":"北京","age":7,"name":"jack"},{"address":"墨西哥","age":20,"name":"tom"}] 
num序列化后 = 3.14 
反序列化后 monster = {牛魔王 500 2011-11-11 8000 牛魔拳}
反序列化后 a = map[address:洪崖洞 age:30 name:红孩儿]
反序列化后 res = [map[address:北京 age:7 name:jack] map[address:墨西哥 age:20 name:tom]]
**/
```

### 16. 单元测试

```go
// cal.go
package main

// 返回 1+2+3+...+n的结果
func addUpper(n int) int {

	res := 0
	for i := 1; i <= n; i++ {
		res += i
	}
	return res - 1  // 故意写错，方便单源测试
}

// cal_test.go
package main

import "testing"

func TestAddUpper(t *testing.T) {

	res := addUpper(10)
	if res != 55 {
		t.Fatalf("addUpper(10) 结果错误，期望值 = 55, 实际值 = %v\n", res)
	} else {
		t.Logf("addUpper(10) 结果正确...\n")
	}
}
//当执行go test -v时，会找到当前文件夹中的所有形式为xxx_test.go的文件，然后执行所有形式为TestXxx的函数。
```

### 17. 协程

```go
package main

import (
	"fmt"
	"strconv"
	"time"
)

func test() {
	for i := 0; i < 5; i++ {
		fmt.Println("test() hello world " + strconv.Itoa(i))
		time.Sleep(time.Millisecond * 200)
	}
}

// 在主线程中，开启一个协程，改协程每隔0.2秒输出一次 "hello world"
// 在主线程中每隔0.2秒输出 "hello golang"
// 要求同时执行
func main() {

	go test()  // 开启一个协程， ？？ 这里go作用

	for i := 0; i < 5; i++ {
		fmt.Println("main() hello golang " + strconv.Itoa(i))
		time.Sleep(time.Millisecond * 200)
	}
}
/** 输出结果：
main() hello golang 0
test() hello world 0
test() hello world 1
main() hello golang 1
main() hello golang 2
test() hello world 2
test() hello world 3
main() hello golang 3
main() hello golang 4
test() hello world 4
**/
```

### 18. 管道

```go
package main

import "fmt"

// channel 是引用类型，只能存放一种指定的类型
func main() {

	// (1) 创建管道
	// 方式一
	var intChain chan int    // chan 作用？？
	intChain = make(chan int, 5)
	// 方式二
	t := make(chan int)
	fmt.Println(t)
	// 方式三：只读管道，  如果读的管道没有数据，会怎么样
	var a <-chan int
	a = make(<-chan int, 3)
	fmt.Println(a)
	// 方式四：只写管道
	var b chan<- int
	b = make(chan<- int, 3)
	fmt.Println(b)

	// (2) 向管道写入数据，不能超出其容量(超出则报错)
	intChain <- 985
	intChain <- 211
	intChain <- 1314
	intChain <- 521

	// (3) 长度、容量
	fmt.Printf("len(intChain) = %v, cap(intChain) = %v\n", len(intChain), cap(intChain))

	// (4) 从管道读出数据，管道空了就不能再取(否则报错)
	num := <-intChain
	fmt.Println(num)

	// (5) 删除管道里的一个元素
	<-intChain

	// (6) 关闭管道，之后只能读，不能写
	close(intChain)
	v, ok := <-intChain
	fmt.Println(v, ok) // 如果还能读出数据，则ok为true，否则ok为false

	// (7) 遍历，只有关闭channel之后才可以遍历，否则会报错
	fmt.Println("开始遍历...")
	for v := range intChain {
		fmt.Println(v)
	}
}

/** 输出结果：
0xc00008c060
0xc0000b8000
0xc0000b8080
len(intChain) = 4, cap(intChain) = 5
985
1314 true
开始遍历...
521
**/
```

#### .1. 协程&管道demo

```go
package main

import (
	"fmt"
	"time"
)

func writeData(intChan chan int) {
	for i := 1; i <= 5; i++ {
		intChan <- i
		fmt.Printf("writeData() %v\n", i)
		time.Sleep(time.Millisecond * 100)
	}
	close(intChan)
}

func readData(intChan chan int, exitChan chan bool) {

	for i := 1; i <= 5; i++ {
		v := <- intChan
		fmt.Printf("readData() %v\n", v)
		time.Sleep(time.Millisecond * 300)
	}
	close(exitChan)
}

func main() {

	intChan := make(chan int, 2)  // 这个管道负责读写数据
	exitChan := make(chan bool, 1)  // 这个管道负责控制主线程是否结束

	go writeData(intChan)
	go readData(intChan, exitChan)

	for {
		_, ok := <- exitChan
		if !ok {
			break
		}
	}
}
```

#### .2. 求质数

```go
package main

import "fmt"

func putNum(intChan chan int) {

	for i := 2; i <= 50; i++ {
		intChan <- i
	}
	close(intChan)
}

func primeNum(intChan chan int, primeChan chan int, exitChan chan bool) {

	var flag bool
	for {
		num, ok := <-intChan
		if !ok { // 说明管道中没有数据了
			break
		}

		// 判断 num 是否是素数
		flag = true
		for i := 2; i <= num/i; i++ {
			if num%i == 0 {
				flag = false
				break
			}
		}
		if flag { // 说明 num 是质数
			primeChan <- num
		}
	}
	fmt.Println("有一个 primeNum 协程执行完毕...")
	exitChan <- true
}

func main() {

	intChan := make(chan int, 10)   // 待判断的数存入的管道
	primeChan := make(chan int, 40) // 质数存放的位置
	exitChan := make(chan bool, 4)  // 用于控制并发流程

	// 开启协程，用于判断是否是素数
	go putNum(intChan)       // 开启一个协程，向intChan放入数据
	for i := 0; i < 4; i++ { // 开启四个用于判断质数的协程
		go primeNum(intChan, primeChan, exitChan)
	}

	// 该协程目的：使得在协程 primeNum 没有执行完毕前主线程会阻塞
	go func() {
		for i := 0; i < 4; i++ {
			<-exitChan // 管道中没有数据时会阻塞住，导致没法关闭 primeChan，从而使得主函数从primeChan读取数据也阻塞住
		}
		close(primeChan)
	}()

	// 从 primeChan 读出数据
	for {
		res, ok := <-primeChan
		if !ok { // 说明 primeChan 关闭，并且其中的数据全部被读出来了
			break
		}
		fmt.Println(res) // 输出素数
	}
	fmt.Println("主函数执行完毕...")
}
```

> 第二种写法会等待 primeNum 全部执行完毕，才会执行主函数；而第一种写法主函数和协程会同时执行。

```go
package main

import (
	"fmt"
	"sync"
)

var (
	g sync.WaitGroup
)

func putNum(intChan chan int) {

	for i := 2; i <= 50; i++ {
		intChan <- i
	}
	close(intChan)
}

func primeNum(intChan chan int, primeChan chan int) {

	var flag bool
	for {
		num, ok := <-intChan
		if !ok { // 说明管道中没有数据了
			break
		}

		// 判断 num 是否是素数
		flag = true
		for i := 2; i <= num/i; i++ {
			if num%i == 0 {
				flag = false
				break
			}
		}
		if flag { // 说明 num 是质数
			primeChan <- num
		}
	}
	fmt.Println("有一个 primeNum 协程执行完毕...")
	g.Done()
}

func main() {

	intChan := make(chan int, 10)   // 待判断的数存入的管道
	primeChan := make(chan int, 40) // 质数存放的位置
	g.Add(4)

	// 开启协程，用于判断是否是素数
	go putNum(intChan)       // 开启一个协程，向intChan放入数据
	for i := 0; i < 4; i++ { // 开启四个用于判断质数的协程
		go primeNum(intChan, primeChan)
	}

	// 等待前面四个 primeNum 协程执行完毕
	g.Wait()
	close(primeChan)

	// 从 primeChan 读出数据
	for {
		res, ok := <-primeChan
		if !ok { // 说明 primeChan 关闭，并且其中的数据全部被读出来了
			break
		}
		fmt.Println(res) // 输出素数
	}
	fmt.Println("主函数执行完毕...")
}
```

### 20. 反射

```go
package main

import (
	"fmt"
	"reflect"
)

func test01(b interface{}) { // 演示对基本类型 int 的反射基本操作

	// 获取 reflect.Type
	rType := reflect.TypeOf(b)
	fmt.Printf("reflect.Type of b = %v\n", rType)

	// 获取 reflect.Value
	rValue := reflect.ValueOf(b)
	fmt.Printf("rValue = %v, type of rValue = %T\n", rValue, rValue)

	// 获取真实存的值
	val := rValue.Int()
	fmt.Printf("val = %v, type of val = %T\n", val, val)

	// 将 b 转为原始类型: interface{} -> reflect.Value -> interface{} -> int
	num := reflect.ValueOf(b).Interface().(int)
	fmt.Printf("num = %v\n", num)
}

func test02(b interface{}) {

	rValue := reflect.ValueOf(b)
	// 修改 b 对应的值
	rValue.Elem().SetInt(rValue.Elem().Int() + 5) // 自增 5
}

type Student struct {
	Name  string `json:"name"`
	Age   int    `json:"age"`
	Score float64
	Sex   string
}

func (s Student) Print() {
	fmt.Println("---- start ----")
	fmt.Println(s)
	fmt.Println("---- end ----")
}

func (s Student) GetSum(a, b int) int {
	return a + b
}

func (s Student) Set(name string, age int, score float64, sex string) {
	s.Name = name
	s.Age = age
	s.Score = score
	s.Sex = sex
}

func test03(b interface{}) { // 演示对 struct 的反射操作

	// 对 struct 来说：Kind 是 struct, Type 是 main.Student, Value 是 对应实例

	// 获取 reflect.Type
	rType := reflect.TypeOf(b)
	fmt.Printf("reflect.Type of b = %v\n", rType) // main.Student

	// 获取 reflect.Value
	rValue := reflect.ValueOf(b)
	fmt.Printf("rValue = %v, reflect.Value = %v\n", rValue, rValue)

	iValue := rValue.Interface()   // 这句作用是什么
	fmt.Printf("iValue = %v, type of iValue = %T\n", iValue, iValue)

	// 获取 Kind
	kind := rType.Kind() // 等价于：kind := rValue.Kind()
	fmt.Printf("kind = %v\n", kind)
	if kind != reflect.Struct {
		fmt.Println("not struct, expect struct...")
		return
	}

	// 将 b 转为原始类型: interface{} -> reflect.Value -> interface{} -> struct
	if student, ok := reflect.ValueOf(b).Interface().(Student); ok {
		fmt.Printf("转化成功，student = %v\n", student)
	}

	/* 对字段进行操作 */
	fmt.Println("对字段进行操作")
	cnt := rValue.NumField() // 字段个数
	for i := 0; i < cnt; i++ {
		fmt.Printf("val of field %d = %v, ", i, rValue.Field(i))
		// 获取到 struct 标签，注意需要通过 reflect.Type 来获取 tag 标签的值
		tagVal := rType.Field(i).Tag.Get("json")
		if tagVal != "" {
			fmt.Printf("tag of field %d = %v\n", i, tagVal)
		} else {
			fmt.Printf("field %d doesn't have tag\n", i)
		}
	}

	/* 对方法进行操作 */
	fmt.Println("对方法进行操作")
	cnt = rValue.NumMethod()
	// 调用方法, 根据函数字典序进行排序：GetSum，Print，Set
	rValue.Method(1).Call(nil) // 调用 Print
	// 调用方法
	var parms []reflect.Value
	parms = append(parms, reflect.ValueOf(10))
	parms = append(parms, reflect.ValueOf(20))
	res := rValue.Method(0).Call(parms) // 调用 GetSum
	fmt.Println("res =", res[0].Int())
}

func test04(b interface{}) {

	rValue := reflect.ValueOf(b)
	rValue.Elem().Field(0).SetString("jack")
	rValue.Elem().FieldByName("Score").SetFloat(92.1)
}

func main() {

	// 1. 对基本类型的反射操作
	num := 10
	test01(num)

	// 2. 通过反射修改变量
	fmt.Println()
	test03(&num)
	fmt.Printf("执行过反射后，num = %v\n", num)

	// 3. 对 struct 的反射操作
	fmt.Println()
	student := Student{
		Name:  "tom",
		Age:   18,
		Score: 98.5,
	}
	test03(student)

	// 4. 通过反射修改结构体变量
	fmt.Println()
	test04(&student)
	fmt.Printf("执行过反射后，student = %v\n", student)
}
/** 输出结果：
reflect.Type of b = int
rValue = 10, type of rValue = reflect.Value
val = 10, type of val = int64
num = 10

reflect.Type of b = *int
rValue = 0xc0000aa058, reflect.Value = 0xc0000aa058
iValue = 0xc0000aa058, type of iValue = *int
kind = ptr
not struct, expect struct...
执行过反射后，num = 10

reflect.Type of b = main.Student
rValue = {tom 18 98.5 }, reflect.Value = {tom 18 98.5 }
iValue = {tom 18 98.5 }, type of iValue = main.Student
kind = struct
转化成功，student = {tom 18 98.5 }
对字段进行操作
val of field 0 = tom, tag of field 0 = name
val of field 1 = 18, tag of field 1 = age
val of field 2 = 98.5, field 2 doesn't have tag
val of field 3 = , field 3 doesn't have tag
对方法进行操作
---- start ----
{tom 18 98.5 }
---- end ----
res = 30

执行过反射后，student = {jack 18 92.1 }
**/
```

### 21. TCP 通信

```go
package main

import (
	"fmt"
	"net"
)

func process(conn net.Conn) {

	defer conn.Close()

	// 读取客户端发送过来的数据
	for {
		buf := make([]byte, 1024)
		n, err := conn.Read(buf) // 如果客户端没有Write，协程会阻塞
		if err != nil {
			fmt.Println("server Read err =", err)
			return
		}
		fmt.Print(string(buf[:n]))
	}
}

// 服务端开启监听
func main() {

	fmt.Println("服务器开始监听...")

	listen, err := net.Listen("tcp", "0.0.0.0:8888")
	if err != nil {
		fmt.Println("listen err =", err)
		return
	}
	defer listen.Close() // 延时关闭

	// 循环等待客户端的连接
	for {
		conn, err := listen.Accept()
		if err != nil {
			fmt.Println("Accept() err =", err)
		} else {
			fmt.Printf("Accept() success, conn = %v, ip = %v\n", conn, conn.RemoteAddr().String())
		}

		go process(conn) // 启动一个协程与当前客户端连接通信
	}
}
```

```go
package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"strings"
)

func main() {

	// 客户端去连接服务器
	conn, err := net.Dial("tcp", "127.0.0.1:8888")
	if err != nil {
		fmt.Println("client dail err =", err)
		return
	}

	// 客户端 和 服务端 通信
	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadString('\n') // 从终端中读取用户输入
		if err != nil {
			fmt.Println("ReadString err =", err)
		}

		if strings.Trim(line, " \r\n") == "exit" {
			fmt.Println("客户端退出...")
			break
		}

		_, err = conn.Write([]byte(line))
		if err != nil {
			fmt.Println("conn.Write err =", err)
		}
	}
}
```

### Resource

- [x] https://mp.weixin.qq.com/s/Dwf98JFUnRij0Ha7o3ZSHQ
- [ ] https://www.runoob.com/go/go-concurrent.html
- [ ] https://www.topgoer.com/Go%E9%AB%98%E7%BA%A7/
- [ ] https://golang3.eddycjy.com/
- [ ] https://draveness.me/golang/
- [ ] Effective Go ： 高效 Go 编程，由 Golang 官方编写，里面包含了编写 Go 代码的一些建议，也可以理解为最佳实践。
- [ ] Go Code Review Comments Golang 官方编写的 Go 最佳实践，作为 Effective Go 的补充。
- [ ] Style guideline for Go packages 包含了如何组织 Go 包、如何命名 Go 包、如何写 Go 包文档的一些建议。
- [ ] https://github.com/cristaloleg/go-advice/blob/master/README_ZH.md
- [ ] https://github.com/cristaloleg/go-advice
- [ ] Uber Go 语言编码规范

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/go/  

