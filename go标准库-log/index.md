# go标准库-log


```go
// 这个示例程序展示如何使用最基本的log包
package main

import (
	"log"
)

func init() {
	log.SetPrefix("TRACE: ")
	log.SetFlags(log.Ldate | log.Lmicroseconds | log.Llongfile)
}

func main() {
	// Println写到标准日志记录器
	log.Println("message")

	// Fatalln在调用Println()之后会接着调用os.Exit(1)
	log.Fatalln("fatal message")

	// Panicln在调用Println()之后会接着调用panic()
	log.Panicln("panic message")
}
```

### 1. log源码

```go
const (
　// 将下面的位使用或运算符连接在一起，可以控制要输出的信息。没有
　// 办法控制这些信息出现的顺序（下面会给出顺序）或者打印的格式
　// （格式在注释里描述）。这些项后面会有一个冒号：
　//　　2009/01/23 01:23:23.123123 /a/b/c/d.go:23: message

　// 日期: 2009/01/23
　Ldate = 1 << iota

　// 时间: 01:23:23
　Ltime

　// 毫秒级时间: 01:23:23.123123。该设置会覆盖Ltime标志
　Lmicroseconds

　// 完整路径的文件名和行号: /a/b/c/d.go:23
　Llongfile

　// 最终的文件名元素和行号: d.go:23
　// 覆盖 Llongfile
　Lshortfile

　// 标准日志记录器的初始值
　LstdFlags = Ldate | Ltime
)
```

### 2. 定制日志

```go
// 这个示例程序展示如何创建定制的日志记录器
package main

import (
	"io"
	"io/ioutil"
	"log"
	"os"
)

var ( // 为4个日志等级声明了4个Logger类型的指针变量
	Trace   *log.Logger // 记录所有日志
	Info    *log.Logger // 重要的信息
	Warning *log.Logger // 需要注意的信息
	Error   *log.Logger // 非常严重的问题
)

func init() {
	file, err := os.OpenFile("errors.txt",
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalln("Failed to open error log file:", err)
	}

	Trace = log.New(ioutil.Discard, // 当某个等级的日志不重要时，使用Discard变量可以禁用这个等级的日志。
		"TRACE: ",
		log.Ldate|log.Ltime|log.Lshortfile)

	Info = log.New(os.Stdout,
		"INFO: ",
		log.Ldate|log.Ltime|log.Lshortfile)

	Warning = log.New(io.MultiWriter(file, os.Stdout),
		"WARN: ",
		log.Ldate|log.Ltime|log.Lshortfile)

	// io.MultiWriter(file, os.Stderr)
	/*
		这个函数调用会返回一个io.Writer接口类型值，这个值包含之前打开的文件file，以及stderr。
		MultiWriter函数是一个变参函数，可以接受任意个实现了io.Writer接口的值。
		这个函数会返回一个io.Writer值，这个值会把所有传入的io.Writer的值绑在一起。
		当对这个返回值进行写入时，会向所有绑在一起的io.Writer值做写入。
		这让类似log.New这样的函数可以同时向多个Writer做输出。
		现在，当我们使用Error记录器记录日志时，输出会同时写到文件和stderr。
	*/
	Error = log.New(io.MultiWriter(file, os.Stderr),
		"ERROR: ",
		log.Ldate|log.Ltime|log.Lshortfile)
}

func main() {
	Trace.Println("I have something standard to say")
	Info.Println("Special Information")
	Warning.Println("There is something you need to know about")
	Error.Println("Something has failed")
}
```

### 3. log插件

- logrus ： https://github.com/sirupsen/logrus
- seelog：https://github.com/cihub/seelog

### Resource

- https://blog.csdn.net/wohu1104/article/details/106391835

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/go%E6%A0%87%E5%87%86%E5%BA%93-log/  

