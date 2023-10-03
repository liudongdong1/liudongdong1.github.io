# 通道


> 在 `Go` 语言里，你不仅可以使用原子函数和互斥锁来保证对共享资源的安全访问以及消除竞争状态，还可以使用通道，通过发送和接收需要共享的资源，在 `goroutine` 之间做同步。
>
> - 通道中的各个元素值都是严格地按照发送的顺序排列的，先被发送通道的元素值一定会先被接收。
> - 在任何时候，同时只能有一个 `goroutine`访问通道进行发送和获取数据。 `goroutine`间通过通道就可以通信。
> - 对于通道中的`同一个元素值`来说，`发送操作`和`接收操作`之间也是`互斥`的。例如，虽然会出现，正在被复制进通道但还未复制完成的元素值，但是这时它绝不会被想接收它的一方看到和取走。
> - 元素值从外界`进入通道`时会被`复制`。更具体地说，进入通道的并不是在接收操作符右边的那个元素值，而是`它的副本`。
> - 发送操作和接收操作中对元素值的处理都是不可分割的。
>   - 发送操作要么还没复制元素值，要么已经复制完毕，绝不会出现只复制了一部分的情况。
>   - 接收操作在准备好元素值的副本之后，一定会删除掉通道中的原值，绝不会出现通道中仍有残留的情况。
> - 发送操作在完全完成之前会被阻塞。接收操作也是如此。
>   - 发送操作 包括了 `复制元素值 `和 `放置副本到通道内部` 这两个步骤。在这两个步骤完全完成之前，发起这个发送操作的那句代码会一直阻塞在那里。
>     也就是说，在它之后的代码不会有执行的机会，直到这句代码的阻塞解除。
>     更细致地说，在通道完成发送操作之后，运行时系统会通知这句代码所在的 goroutine，以使它去争取继续运行代码的机会。
>   - 接收操作 通常包含了 复制通道内的元素值、 放置副本到接收方、 删掉原值 三个步骤，也就是说通常，值进入通道时会被复制一次，然后出通道的时候依照通道内的那个值再被复制一次并给到接收方。在所有这些步骤完全完成之前，发起该操作的代码也会一直阻塞，直到该代码所在的 goroutine 收到了运行时系统的通知并重新获得运行机会为止。

### 1.生产者&消费者

```go
// 整段代码中，没有线程创建，没有线程池也没有加锁，
// 仅仅通过关键字 go 实现 goroutine，和通道实现数据交换。
package main

import (
	"fmt"
	"math/rand"
	"time"
)

// 数据生产者
func producer(header string, channel chan<- string) {
	// 无限循环, 不停地生产数据
	for {
		// 将随机数和字符串格式化为字符串发送给通道
		channel <- fmt.Sprintf("%s: %v", header, rand.Int31())
		// 等待1秒
		time.Sleep(time.Second)
	}
}

// 数据消费者
func customer(channel <-chan string) {
	// 不停地获取数据
	for {
		// 从通道中取出数据, 此处会阻塞直到信道中返回数据
		message := <-channel
		// 打印数据
		fmt.Println(message)
	}
}
func main() {
	// 创建一个字符串类型的通道
	channel := make(chan string)
	// 创建producer()函数的并发goroutine
	go producer("cat", channel)
	go producer("dog", channel)
	// 数据消费函数
	customer(channel)
}
```

### 2. 通道操作

#### .1. 通道声明

```go
// 有缓冲的字符串通道，数据类型是字符串，包含一个 10 个值的缓冲区。
buffered := make(chan string, 10)

// 通过通道发送一个字符串
buffered <- "Gopher"
```

#### .2. 通道发送阻塞

```go
package main

func main() {
	// 创建一个整型通道  channelName <- value 
	ch := make(chan int)	// 无缓冲的通道

	// 尝试将0通过通道发送
	ch <- 0    //阻塞
}
```

```go
package main

func main() {
	ch := make(chan int, 10)	// 有缓冲的通道
	ch <- 0           // 不阻塞
}
```

#### .3. 通道接受数据

- **通道的收发操作在两个不同的 goroutine 间进行。**
- **接收将持续阻塞直到发送方发送数据。**
-  **每次接收一个元素。**

##### 1. 阻塞与非阻塞接受

```go
//阻塞接受数据
data := <-ch
<-ch	//阻塞，直到接收到数据，但接收到的数据会被忽略
//非阻塞接受数据
data, ok := <- ch
```

> 非阻塞的通道接收方法可能造成高的 CPU 占用，因此使用非常少。如果需要实现接收超时检测，可以配合 `select` 和计时器 `channel` 进行。

```go
package main

import (
	"errors"
	"fmt"
	"time"
)

// 模拟RPC客户端的请求和接收消息封装
func RPCClient(ch chan string, req string) (string, error) {
	// 向服务器发送请求
	ch <- req

	// 等待服务器返回
	select {
	// 下面两个通道操作同时开启，那个先返回就先执行哪个后面的语句 
	case ack := <-ch: // 接收到服务器返回数据
		return ack, nil
	case <-time.After(time.Second): // 超时
		return "", errors.New("Time out")
	}
}

// 模拟RPC服务器端接收客户端请求和回应
func RPCServer(ch chan string) {
	for {
		// 接收客户端请求
		data := <-ch
		// 打印接收到的数据
		fmt.Println("server received:", data)
		time.Sleep(2 * time.Second)
		// 反馈给客户端收到
		ch <- "roger"
	}
}

func main() {

	// 创建一个无缓冲字符串通道
	ch := make(chan string)
	// 并发执行服务器逻辑
	go RPCServer(ch)
	// 客户端请求数据和接收数据
	recv, err := RPCClient(ch, "hi")
	if err != nil {
		// 发生错误打印
		fmt.Println(err)
	} else {
		// 正常接收到数据
	fmt.Println("client received", recv)
	}
}
```

##### 2. 接收任意数据，忽略接收的数据

```go
package main

import "fmt"

func main() {
	// 构建一个通道
	ch := make(chan int)
	// 开启一个并发匿名函数
	go func() {
		fmt.Println("start goroutine")
		// 通过通道通知main的goroutine
		ch <- 0
		fmt.Println("exit goroutine")
	}()

	fmt.Println("wait goroutine")
	// 等待匿名goroutine
	<-ch
	fmt.Println("all done")
}
```

```
wait goroutine
start goroutine
exit goroutine
all done
```

##### 3. 循环接收

> `for ... range `循环遍历通道时，信道必须关闭，否则会引发 deadlock 错误。
>
> 迭代为 `nil` 的通道值会让当前流程永远阻塞在 `for` 语句上。

### 3. 通道关闭

> `channel` 支持 `close` 操作，用于关闭 `channel` ，随后对基于该 `channel` 的任何**发送**操作都将导致 `panic` 异常。
>
> 对一个已经被 `close` 过的 `channel` 进行**接收**操作依然可以接受到之前已经成功发送的数据；如果 `channel` 中已经没有数据的话将产生一个零值的数据。

```go
v,	ok	:=	<-ch  //ok返回值是 false 则表示 ch 已经被关闭。
```

```go
package main

import "fmt"

func main() {
	// 创建一个整型带两个缓冲的通道
	ch := make(chan int, 2)
	// 给通道放入两个数据
	ch <- 0
	ch <- 1
	// 关闭缓冲
	close(ch)

	// 遍历缓冲所有数据, 且多遍历1个
	for i := 0; i < cap(ch)+1; i++ {
		// 从通道中取出数据
		v, ok := <-ch
		// 打印取出数据的状态
		fmt.Println(v, ok)
	}
}
```

### 4. 单向通道

```go
var 通道实例 chan<- 元素类型    // 只能发送通道
var 通道实例 <-chan 元素类型    // 只能接收通道
```

```go
// 只能发不能收的通道。
var uselessChan = make(chan<- int, 1)
// 只能收不能发的通道。
var anotherUselessChan = make(<-chan int, 1)

var ch1 chan int // ch1是一个正常的channel，不是单向的
var ch2 chan<- float64// ch2是单向channel，只用于写float64数据
var ch3 <-chan int // ch3是单向channel，只用于读取int数据
```

```go
package main

import (
	"fmt"
	"math/rand"
	"time"
)

// 数据生产者
func producer(header string, channel chan<- string) {
	// 无限循环，不停的生产数据
	for {
		// 将随机数和字符串格式化为字符串发送到通道
		channel <- fmt.Sprintf("%s: %v", header, rand.Int31())
		// 等待1秒
		time.Sleep(time.Second)
	}
}

// 数据消费者
func consumer(channel <-chan string) {
	// 不停的获取数据
	for {
		// 从通道中取出数据，此处会阻塞直到信道中返回数据
		message := <-channel
		// 打印数据
		fmt.Println(message)
	}
}

func main() {
	// 创建一个字符串类型的通道
	channel := make(chan string)
	// 创建producer函数的并发goroutine
	go producer("cat", channel)
	go producer("dog", channel)
	// 数据消费函数
	consumer(channel)
}
```

```go
package main

import "fmt"

func printer(c chan int) {
	// 开始无限循环等待数据
	for {
		// 从channel中获取一个数据
		data := <-c
		// 将0视为数据结束
		if data == 0 {
			break
		}
		// 打印数据
		fmt.Println(data)
	}
	// 通知main已经结束循环(我搞定了!)
	c <- 0
}

func main() {
	// 创建一个channel
	c := make(chan int)
	// 并发执行printer, 传入channel
	go printer(c)
	for i := 1; i <= 10; i++ {
		// 将数据通过channel投送给printer
		c <- i
	}
	// 通知并发的printer结束循环(没数据啦!)
	c <- 0
	// 等待printer结束(搞定喊我!)
	<-c
}
```

### 5. waitgroup

`goroutine` 和 `chan` ， 一个用于并发，另一个用于通信。没有缓冲的通道具有同步的功能，除此之外， `sync` 包也提供了多个 `goroutine` 同步的机制，主要是通过 `WaitGroup` 实现的。

`WaitGroup` 值中计数器的值不能小于 0，是因为这样会引发一个 `panic` 。

> **不要把增加其计数器值的操作和调用其Wait方法的代码，放在不同的 `goroutine` 中执行。换句话说，要杜绝对同一个`WaitGroup` 值的两种操作的并发执行。**
>
>  **先统一 `Add` ，再并发 `Done` ，最后 `Wait`** 这种标准方式，来使用 `WaitGroup` 值。 尤其不要在调用 `Wait` 方法的同时，并发地通过调用 `Add` 方法去增加其计数器的值，因为这也有可能引发 `panic` 。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvaHUxMTA0,size_16,color_FFFFFF,t_70#pic_center-16537042929663.png)

```go
package main

import (
	"net/http"
	"sync"
)

var wg sync.WaitGroup
var urls = []string{
	"http://www.baidu.com",
	"http://www.sina.com",
	"http://www.qq.com",
}

func getURLStatus(url string) {
	// 当前go routine 结束后给wg 计数减1, wg.Done() 等价于wg.Add(-1)
	// defer wg.Add(-1)
	defer wg.Done()

	// 发送 http get 请求并打印 http 返回码
	resp, err := http.Get(url)
	if err == nil {
		println(resp.Status)
	}
}

func main() {
	for _, url := range urls {
		// 为每一个 url 启动一个 goroutine，同时给 wg 加 1
		wg.Add(1)

		go getURLStatus(url)
	}

	// 等待所有请求结束
	wg.Wait()
}
```

### 6. select

>  `Go` 语言借用多路复用的概念，提供了 `select` 关键字，用于多路监昕多个通道。
>
> 当监听的通道没有状态是可读或可写的， `select` 是阻塞的；只要监听的通道中有一个状态是可读或可写的，则 `select` 就不会阻塞，而是进入处理

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    fmt.Println("开始时间：", time.Now().Format("2006-01-02 15:04:05"))
    select {
    case <-time.After(time.Second * 2):
        fmt.Println("2秒后的时间：", time.Now().Format("2006-01-02 15:04:05"))
    }
}
```

```go
package main

import (
	"fmt"
	"os"
)

func main() {
	a, b := make(chan int, 3), make(chan int)
	go func() {
		v, ok, s := 0, false, ""
		for {
			select { // 随机选择可⽤用 channel，接收数据。
			case v, ok = <-a:
				s = "a"
			case v, ok = <-b:
				s = "b"
			}
			if ok {
				fmt.Println(s, v)
			} else {
				os.Exit(0)
			}
		}
	}()
	for i := 0; i < 5; i++ {
		select { // 随机选择可用 channel，发送数据。
		case a <- i:
		case b <- i:
		}
	}
	close(a)
	select {} // 没有可用 channel，阻塞 main goroutine。
}
```

```go
package main

import (
	"fmt"
	"time"
)

func main() {

	i := 0
	c := make(chan int, 2)
	c <- 1
	c <- 2
	close(c)
	for {
		select {
		case value, ok := <-c:
			if !ok {
				c = make(chan int)
				fmt.Println("ch is closed")
			} else {
				fmt.Printf("value is %#v\n", value)
			}
		default:
			time.Sleep(1e9) // 等待1秒钟
			fmt.Println("default, ", i)
			i = i + 1
			if i > 3 {
				return
			}
		}
	}
}
```

### 7. 用 channel 实现信号量 (semaphore)

```go
package main

import (
	"fmt"
	"sync"
)

func main() {
	wg := sync.WaitGroup{}
	wg.Add(3)

	sem := make(chan int, 1)
	for i := 0; i < 3; i++ {
		go func(id int) {
			defer wg.Done()
			sem <- 1 // 向 sem 发送数据，阻塞或者成功。
			for x := 0; x < 3; x++ {
				fmt.Println(id, x)
			}
			<-sem // 接收数据，使得其他阻塞 goroutine 可以发送数据。
		}(i)
	}
	wg.Wait()
}
```

### 8. 用 closed channel 发出退出通知

```go
package main
import (
    "fmt"
    "time"
)
func write(ch chan int) {
    for i := 0; i < 10; i++ {
        ch <- i * 10
        time.Sleep(time.Second * 1)
    }
    close(ch)
}
func read(ch chan int) {
    for {
        if val, ok := <-ch; ok {
            fmt.Println("从通道中读取值：", val)
        } else {
            // 通道被关闭
            fmt.Println("通道已关闭，退出读取程序")
            break
        }
    }
}
func main() {
    var ch = make(chan int, 10)
    go write(ch)
    read(ch)
}
```



### Resource

- https://blog.csdn.net/wohu1104/article/details/105846296#t0

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E9%80%9A%E9%81%93/  

