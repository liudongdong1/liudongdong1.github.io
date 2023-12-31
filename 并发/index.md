# 并发


> 虽然 Go 程序编译后生成的是本地可执行代码，但是这些可执行代码必须运行在Go 语言的运行时（Runtime ）中。Go 运行时类似 Java 和 .NET 语言所用到的虚拟机，主要负责管理包括内存分配、自动垃圾回收、栈处理、协程（Goroutine）、信道（Channel，也称为通道）、切片（slice）、字典（map）和反射（reflect）等。
>
> `Go` 运行时通过接口函数调用来管理协程（`Goroutine`）和信道（`Channel`）等功能。`Go` 用户代码对操作系统内核 `API` 的调用会被 `Go` 运行时拦截并处理。
>
> `Go` 运行时的重要组成部分是协程调度器（`Goroutine Scheduler`）。它负责追踪、调度每个协程运行，实际上是从应用程序的进程（`Process`）所属的线程池（`Thread Pool`）中分配一个线程执行这个协程。每个协程只有分配到一个操作系统线程才能运行。
>
> `CSP` 是一种消息传递模型，通过在 `goroutine` 之间传递数据来传递消息，而不是对数据进行加锁来实现同步访问。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29odTExMDQ=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center.jpeg)

### 1. goroutine启动

> `Go` 语言的并发执行体称为 `goroutine` , `Go` 语言通过 `go` 关键字来启动一个 `goroutine` 。 `goroutine` 是一种非常轻量级的实现，可在单个进程里执行成千上万的并发任务。
>
> - `Go` 程序从 `main` 包的 `main()` 函数开始，在程序启动时， `Go` 程序就会为 `main()` 函数创建一个默认的 `goroutine` 。
> - `Go` 程序中使用 `go` 关键字为一个函数创建一个 `goroutine` 。一个函数可以被创建多个 `goroutine` ，而一个 `goroutine` 必定对应一个函数。
> - **调度器不能保证多个 `goroutine` 执行次序，且进程退出时不会等待它们结束。**

#### 1. 匿名函数启动

```go
go func( 参数列表 ){
    函数体
}( 调用参数列表 )
```

```go
package main

import (
	"runtime"
	"time"
)

func main() {
	go func() {
		sum := 0
		for i := 0; i <= 10000; i++ {
			sum += i
		}

		println("sum is: ", sum)
		time.Sleep(1 * time.Second)
	}()
	//NumGoroutine 可以返回当前程序的 goroutine 数目
	println("NumGoroutine=", runtime.NumGoroutine())

	// main goroutine 故意“ sleep ” 5 秒, 防止 main 提前退出
	time.Sleep(5 * time.Second)
}
```

#### 2. 有名函数启动

```go
package main

import (
	"runtime"
	"time"
)

func sum() {
	sum := 0
	for i := 0; i <= 10000; i++ {
		sum += i
	}

	println("sum is: ", sum)
	time.Sleep(1 * time.Second)
}

func main() {
	go sum()
	//NumGoroutine 可以返回当前程序的 goroutine 数目
	println("NumGoroutine=", runtime.NumGoroutine())

	// main goroutine 故意“ sleep ” 5 秒, 防止 main 提前退出
	time.Sleep(5 * time.Second)
}
```

### 2. goroutine特点

1. **`go` 的执行是非阻塞的，不会等待**；
2. **`go` 后面的函数的返回值会被忽略**；
3. **调度器不能保证多个 `goroutine` 的执行次序**；
4. **没有父子 `goroutine` 的概念，所有的 `goroutine` 是平等地被调度和执行的**；
5. **`Go` 程序在执行时会单独为 `main` 函数创建一个 `goroutine` ，遇到其他 `go` 关键字时再去创建其他的 `goroutine`** ；
6. **主函数返回时，所有的 `goroutine` 都会被直接打断，程序退出；**
7. **`Go` 没有暴露 `goroutine id` 给用户，所以不能在一个 `goroutine` 里面显式地操作另一个 `goroutine` ， 不过 `runtime` 包提供了一些函数访问和设置 `goroutine` 的相关信息；**
8. **`runtime.NumGoroutine` 返回一个进程的所有 `goroutine` 数, `main()` 的 `goroutine` 也被算在里面。因此实际创建的 `goroutine` 数量为扣除 `main()` 的 `goroutine` 数。**

### 3. runtime 包函数

#### 1. func GOMAXPROCS

```go
runtime.GOMAXPROCS(逻辑CPU数量)
=0：查询当前的 GOMAXPROCS 值。
=1：设置单核心执行。
.> 1：设置多核并发执行。
```

```go
package main

import (
	"runtime"
)

func main() {

	// 获取当前的 GOMAXPROCS 值
	println("GOMAXPROCS=", runtime.GOMAXPROCS(0))

	// 设置当前的 GOMAXPROCS 值
	runtime.GOMAXPROCS(2)

	// 获取当前的 GOMAXPROCS 值
	println("GOMAXPROCS=", runtime.GOMAXPROCS(0))
}
```

#### 2. func Coexit

`func Goexit ()` 是结束当前 `goroutine` 的运行， `Goexit` 在结束当前 `goroutine` 运行之前会调用当前 `goroutine` 已经注册的 `defer` 。

`Goexit` 并不会产生 `panic` ，所以该 goroutine defer 里面的 recover 调用都返回 nil 。

调用 `runtime.Goexit` 将立即终止当前 `goroutine` 执行，调度器确保所有已注册 `defer` 延迟调用被执行。

```go
package main

import (
	"runtime"
	"sync"
)

func main() {
	wg := new(sync.WaitGroup)
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer println("A.defer")
		func() {
			defer println("B.defer")
			runtime.Goexit() // 终止当前 goroutine
			println("B")     // 不会执行
		}()
		println("A") // 不会执行
	}()
	wg.Wait()
}
```

#### 3. func Gosched

>  `Gosched` 是放弃当前调度执行机会，将当前 `goroutine` 暂停，放回队列等待下次被调度执行。

1. `I/O`、`select`
2. `channel`
3. 等待锁
4. 函数调用（有时）
5. `runetime.Gosched()`

```go
package main

import (
	"runtime"
	"sync"
)

func main() {
	wg := new(sync.WaitGroup)
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			println("Hello, World!")
		}

	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			println(i)
			if i == 2 {
				runtime.Gosched()
			}
		}
	}()

	println("NumGoroutine is ", runtime.GOMAXPROCS(0))
	wg.Wait()
}
```

```go
package main

import (
	"fmt"
	"runtime"
)

func say(s string) {
	for i := 0; i < 2; i++ {
		runtime.Gosched()
		fmt.Println(s)
	}
}
func main() {
	go say("world")
	say("hello")
}
//runtime.Gosched() 会在不同的 goroutine 之间切换，当 main goroutine 退出时，其它的 goroutine 都会直接退出
//输出： hello world hello
```

#### 4. 一个 逻辑处理器处理 goroutine 时间较短

```go
// 这个示例程序展示如何创建goroutine 以及调度器的行为
package main

import (
	"fmt"
	"runtime"
	"sync"
)

// main是所有Go程序的入口
func main() {
	// 分配一个逻辑处理器给调度器使用, 这个函数允许程序更改调度器可以使用的逻辑处理器的数量。
	runtime.GOMAXPROCS(1)

	// wg用来等待程序完成
	// WaitGroup是一个计数信号量，可以用来记录并维护运行的 goroutine。如果WaitGroup的值大于0，Wait方法就会阻塞。
	var wg sync.WaitGroup

	// 计数加2，表示要等待两个goroutine
	wg.Add(2)

	fmt.Println("Start Goroutines")

	// 声明一个匿名函数，并创建一个goroutine
	go func() {
		// 关键字defer会修改函数调用时机，在正在执行的函数返回时才真正调用defer声明的函数。
		// 关键字defer保证，每个 goroutine 一旦完成其工作就调用Done方法。
		// 在函数退出时调用Done来通知main函数工作已经完成
		defer wg.Done()

		// 显示字母表3次
		for count := 0; count < 3; count++ {
			for char := 'a'; char < 'a'+26; char++ {
				fmt.Printf("%c ", char)
			}
		}
	}()

	// 声明一个匿名函数，并创建一个goroutine
	go func() {
		// 在函数退出时调用Done来通知main函数工作已经完成
		defer wg.Done()

		// 显示字母表3次
		for count := 0; count < 3; count++ {
			for char := 'A'; char < 'A'+26; char++ {
				fmt.Printf("%c ", char)
			}
		}
	}()

	fmt.Println("Waiting To Finish")
	// 等待goroutine结束,否则 main 函数将直接继续往下走
	wg.Wait()

	fmt.Println("\nTerminating Program")
}
```

#### 5. 一个 逻辑处理器处理 goroutine 时间较长

> 当 `goroutine` 占用时间过长时，调度器会停止当前正运行的 `goroutine` ，并给其他可运行的 `goroutine` 运行的机会。

```go
// 这个示例程序展示如何创建goroutine 以及调度器的行为
package main

import (
	"fmt"
	"runtime"
	"sync"
)

var wg sync.WaitGroup

// main是所有Go程序的入口
func main() {
	// 分配一个逻辑处理器给调度器使用, 这个函数允许程序更改调度器可以使用的逻辑处理器的数量。
	runtime.GOMAXPROCS(1)

	// wg用来等待程序完成
	// WaitGroup是一个计数信号量，可以用来记录并维护运行的 goroutine。如果WaitGroup的值大于0，Wait方法就会阻塞。

	// 计数加2，表示要等待两个goroutine
	wg.Add(2)

	fmt.Println("Start Goroutines")

	// 创建两个goroutine
	go printPrime("A")
	go printPrime("B")
	fmt.Println("Waiting To Finish")
	// 等待goroutine结束,否则 main 函数将直接继续往下走
	wg.Wait()

	fmt.Println("\nTerminating Program")
}

func printPrime(prefix string) {
	defer wg.Done()
next:
	for outer := 2; outer < 50000; outer++ {
		for inner := 2; inner < outer; inner++ {
			if outer%inner == 0 {
				continue next
			}
		}
		fmt.Printf("%s:%d\n", prefix, outer)
	}
	fmt.Println("Completed", prefix)
}
```

### 4. Context

> `Context` 是一个接口，它具备手动、定时、超时发出取消信号、传值等功能，主要用于控制多个 `goroutine` 之间的协作，尤其是取消操作。一旦取消指令下达，那么被 `Context` 跟踪的这些 `goroutine` 都会收到取消信号，就可以做清理和退出操作。
>
> - 传递数据
> - 主动取消
> - 超时取消

#### 1. 结口方法

```go
type Context interface {
   Deadline() (deadline time.Time, ok bool)
   Done() <-chan struct{}
   Err() error
   Value(key interface{}) interface{}
}
```

- `Deadline` 方法可以获取设置的截止时间，第一个返回值 `deadline` 是截止时间，到了这个时间点，`Context` 会自动发起取消请求，第二个返回值 `ok` 代表是否设置了截止时间。
- `Done` 方法返回一个只读的 `channel`，类型为 `struct{}`。在 `goroutine` 中，如果该方法返回的 `chan` 可以读取，则意味着 `Context` 已经发起了取消信号。通过 `Done` 方法收到这个信号后，就可以做清理操作，然后退出 `goroutine` ，释放资源。
- `Err` 方法返回取消的错误原因，即因为什么原因 `Context` 被取消。
- `Value` 方法获取该 `Context` 上绑定的值，是一个键值对，所以要通过一个 `key` 才可以获取对应的值。

#### 2. context树

- 空 Context：不可取消，没有截止时间，主要用于 Context 树的根节点。

- 可取消的 Context：用于发出取消信号，当取消的时候，它的子 Context 也会取消。

- 可定时取消的 Context：多了一个定时的功能。

- 值 Context：用于存储一个 key-value 键值对。

```go
WithCancel(parent Context) (ctx Context, cancel CancelFunc)：生成一个可取消的 Context。

WithDeadline(parent Context, d time.Time) (Context, CancelFunc)：生成一个可定时取消的 Context，参数 d 为定时取消的具体时间。

WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc)：生成一个可超时取消的 Context，参数 timeout 用于设置多久后取消

WithValue(parent Context, key, val interface{}) Context：生成一个可携带 key-value 键值对的 Context。
```

##### 1. context.WithCancel 取消多个 goroutine

```go
package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	var wg sync.WaitGroup
	wg.Add(3)
	ctx, stop := context.WithCancel(context.Background())
	go func() {
		defer wg.Done()
		watchDog(ctx, "watchdog_1")
	}()

	go func() {
		defer wg.Done()
		watchDog(ctx, "watchdog_2")
	}()

	go func() {
		defer wg.Done()
		watchDog(ctx, "watchdog_3")
	}()

	time.Sleep(5 * time.Second)
	stop() //发停止指令
	wg.Wait()
}

func watchDog(ctx context.Context, name string) {
	//开启for select循环，一直后台监控
	for {
		select {
		case <-ctx.Done():
			fmt.Println(name, "receive stop cmd, will stop")
			return
		default:
			fmt.Println(name, "is running ……")
		}
		time.Sleep(1 * time.Second)
	}
}
```

##### 2. context.WithValue 传值

```go
package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	var wg sync.WaitGroup
	wg.Add(4)
	ctx, stop := context.WithCancel(context.Background())
	go func() {
		defer wg.Done()
		watchDog(ctx, "watchdog_1")
	}()

	go func() {
		defer wg.Done()
		watchDog(ctx, "watchdog_2")
	}()

	go func() {
		defer wg.Done()
		watchDog(ctx, "watchdog_3")
	}()

	valCtx := context.WithValue(ctx, "userId", 2)
	go func() {
		defer wg.Done()
		getUser(valCtx)
	}()

	time.Sleep(5 * time.Second)
	stop() //发停止指令
	wg.Wait()
}

func watchDog(ctx context.Context, name string) {
	//开启for select循环，一直后台监控
	for {
		select {
		case <-ctx.Done():
			fmt.Println(name, "receive stop cmd, will stop")
			return
		default:
			fmt.Println(name, "is running ……")
		}
		time.Sleep(1 * time.Second)
	}
}

func getUser(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			fmt.Println("user exit")
			return
		default:
			userId := ctx.Value("userId")   // 在这里获取value的取值
			fmt.Println("userId is", userId)
			time.Sleep(1 * time.Second)
		}
	}
}
```

##### 3. context.WithTimeout 超时取消

```go
package main

import (
    "fmt"
    "sync"
    "time"

    "golang.org/x/net/context"
)

var (
    wg sync.WaitGroup
)

func startTask(ctx context.Context) error {
    defer wg.Done()

    for i := 0; i < 30; i++ {
        select {
        case <-time.After(2 * time.Second):
            fmt.Printf("in goroutine do task %v\n", i)

        // we received the signal of cancelation in this channel
        case <-ctx.Done():
            fmt.Printf("cancel goroutine task %v\n", i)
            return ctx.Err()
        }
    }
    return nil
}

func main() {
    timeoutCtx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
    defer cancel()

    fmt.Println("startTask")

    wg.Add(1)
    go startTask(timeoutCtx)
    wg.Wait()

    fmt.Println("endTask")
}   
```

##### 4. 调用 Context cancel 和 Done 取消任务

```go
package main

import (
    "fmt"
    "time"

    "golang.org/x/net/context"
)

func startTask(ctx context.Context, task string) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("stop goroutine startTask")
            return
        default:
            fmt.Println(task, "in goroutine do task")
            time.Sleep(2 * time.Second)
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())

    go startTask(ctx, "start allen.wu task 1")
    go startTask(ctx, "start allen.wu task 2")

    time.Sleep(6 * time.Second)

    fmt.Println("Now, call func cancel to stop all goroutines")
    cancel()

    time.Sleep(5 * time.Second)
}
```

#### 3. 注意事项

- Context 不要放在结构体中，要以`参数的方式传递`。
- Context 作为函数的参数时，`要放在第一位，也就是第一个参数`。
- 要使用 `context.Background 函数生成根节点的 Context，也就是最顶层的 Context`。
- `Context 传值要传递必须的值，而且要尽可能地少，不要什么都传`。
- Context `多 goroutine 安全`，可以在多个 goroutine 中放心使用。
- `可以把一个 Context 对象传递给任意个数的 Gorotuine，对它执行 取消 操作时，所有 goroutine 都会接收到取消信号`。
- Context 一般是作为函数的参数进行传递，并且最优的做法是把 Context 作为第一个参数放到每个关键函数的参数中，并且变量名都建议统一命名，名为 ctx。
- 一般而言，把 `context.Background() 作为第一个 parent Context`
- Context 的 Value 中应该传递必须的核心元数据，不要什么数据都使用 Context 传递。
- 永远记住，`只要传递 Context，就不要把 Context 设置为 nil 来传递`。

### 5. sync.map

> `Go` 语言在 1.9 版本中提供了一种效率较高的并发安全的 `sync.Map` ， `sync.Map` 和 `map` 不同，不是以语言原生形态提供，而是在 `sync` 包下的特殊结构。
>
> - `sync.Map` 不能使用 `map` 的方式进行取值和设置等操作，而是使用 `sync.Map` 的方法进行调用， `Store` 表示存储， `Load` 表示获取， `Delete` 表示删除。

```go
package main

import (
      "fmt"
      "sync"
)

func main() {
    
	// 声明 scene，类型为 sync.Map，注意，sync.Map 不能使用 make 创建。
    var scene sync.Map

    // 将键值对保存到sync.Map
    // sync.Map 将键和值以 interface{} 类型进行保存。
    scene.Store("greece", 97)
    scene.Store("london", 100)
    scene.Store("egypt", 200)

    // 从sync.Map中根据键取值
    fmt.Println(scene.Load("london"))

    // 根据键删除对应的键值对
    scene.Delete("london")

    // 遍历所有sync.Map中的键值对
    // 遍历需要提供一个匿名函数，参数为 k、v，类型为 interface{}，
    // 每次 Range() 在遍历一个元素时，都会调用这个匿名函数把结果返回。
    scene.Range(func(k, v interface{}) bool {

        fmt.Println("iterate:", k, v)
        return true
    })

}
```

### Resource

- https://blog.csdn.net/wohu1104/article/details/105750154
- context： https://blog.csdn.net/wohu1104/article/details/110501629#t1
- sync.map: https://blog.csdn.net/wohu1104/article/details/110456244

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%B9%B6%E5%8F%91/  

