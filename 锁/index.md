# 锁


### 1. 竞争状态

```go
// 这个示例程序展示如何在程序里造成竞争状态
// 实际上不希望出现这种情况
package main

import (
	"fmt"
	"runtime"
	"sync"
)

var (
	// counter是所有goroutine都要增加其值的变量
	counter int

	// wg用来等待程序结束
	wg sync.WaitGroup
)

// main是所有Go程序的入口
func main() {
	// 计数加2，表示要等待两个goroutine
	wg.Add(2)

	// 创建两个goroutine
	go incCounter(1)
	go incCounter(2)

	// 等待goroutine结束
	wg.Wait()
	fmt.Println("Final Counter:", counter)
}

// incCounter增加包里counter变量的值
func incCounter(id int) {
	// 在函数退出时调用Done来通知main函数工作已经完成
	defer wg.Done()

	for count := 0; count < 2; count++ {
		// 捕获counter的值
		value := counter

		// 当前goroutine从线程退出，并放回到队列
		/*
			用于将 goroutine 从当前线程退出，给其他 goroutine 运行的机会。
			在两次操作中间这样做的目的是强制调度器切换两个 goroutine，以便让竞争状态的效果变得更明显。
		*/
		runtime.Gosched()

		// 增加本地value变量的值
		value++

		// 将该值保存回counter
		counter = value
	}
}
//Final Counter: 2
```

### 2. 锁住共享资源

#### .1. 原子函数

>  `atmoic` 包的 `AddInt64` 函数。这个函数会同步整型值的加法，方法是强制同一时刻只能有一个 `goroutine` 运行并完成这个加法操作。
>
> 当 `goroutine` 试图去调用任何原子函数时，这些 `goroutine` 都会自动根据所引用的变量做同步处理。

```go
package main

import (
	"fmt"
	"sync/atomic"
)

var (
	// 序列号
	seq int64
)

// 序列号生成器
func GenID() int64 {
	// 尝试原子的增加序列号
	return atomic.AddInt64(&seq, 1)
}


func main() {
	// 10个并发序列号生成
	for i := 0; i < 10; i++ {
		go GenID()
	}

	fmt.Println(GenID())
}
```

```go
// 这个示例程序展示如何使用atomic包来提供
// 对数值类型的安全访问
package main

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

var (
	// counter是所有goroutine都要增加其值的变量
	counter int32

	// wg用来等待程序结束
	wg sync.WaitGroup
)

// main是所有Go程序的入口
func main() {
	// 计数加2，表示要等待两个goroutine
	wg.Add(2)

	// 创建两个goroutine
	go incCounter(1)
	go incCounter(2)

	// 等待goroutine结束
	wg.Wait()
	fmt.Println("Final Counter:", counter)
}

// incCounter增加包里counter变量的值
func incCounter(id int) {
	// 在函数退出时调用Done来通知main函数工作已经完成
	defer wg.Done()

	for count := 0; count < 2; count++ {
		// 安全地对counter加1
		atomic.AddInt32(&counter, 1)

		// 当前goroutine从线程退出，并放回到队列
		/*
			用于将 goroutine 从当前线程退出，给其他 goroutine 运行的机会。
			在两次操作中间这样做的目的是强制调度器切换两个 goroutine，以便让竞争状态的效果变得更明显。
		*/
		runtime.Gosched()

	}
}
```

#### .2. 互斥锁

> 互斥锁用于在代码上创建一个临界区，保证同一时间只有一个 `goroutine` 可以执行这个临界区代码。

```go
// 这个示例程序展示如何使用互斥锁来
// 定义一段需要同步访问的代码临界区
// 资源的同步访问
package main

import (
	"fmt"
	"runtime"
	"sync"
)

var (
	// counter是所有goroutine都要增加其值的变量
	counter int

	// wg用来等待程序结束
	wg sync.WaitGroup

	// mutex 用来定义一段代码临界区
	mutex sync.Mutex
)

// main是所有Go程序的入口
func main() {
	// 计数加2，表示要等待两个goroutine
	wg.Add(2)

	// 创建两个goroutine
	go incCounter(1)
	go incCounter(2)

	// 等待goroutine结束
	wg.Wait()
	fmt.Printf("Final Counter: %d\n", counter)
}

// incCounter使用互斥锁来同步并保证安全访问，
// 增加包里counter变量的值
func incCounter(id int) {
	// 在函数退出时调用Done来通知main函数工作已经完成
	defer wg.Done()

	for count := 0; count < 2; count++ {
		// 同一时刻只允许一个goroutine进入
		// 这个临界区
		mutex.Lock()
		{	 // 使用大括号只是为了让临界区看起来更清晰，并不是必需的。
			// 捕获counter的值
			value := counter

			// 当前goroutine从线程退出，并放回到队列  
            //强制将当前 goroutine 退出当前线程后，调度器会再次分配这个 goroutine 继续运行。
			runtime.Gosched()

			// 增加本地value变量的值
			value++

			// 将该值保存回counter
			counter = value
		}
		mutex.Unlock()
		// 释放锁，允许其他正在等待的goroutine
		// 进入临界区
	}
}
```

#### .3. 读写互斥锁 sync.RWMutex

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

var (
	count int
	// 变量对应的读写互斥锁
	countGuard sync.RWMutex
)

func GetCount() int {
	countGuard.RLock()
	defer countGuard.RUnlock()
	return count
}

func SetCount(c int) {
	countGuard.Lock()
	{
		count += c
	}
	countGuard.Unlock()
}

func main() {
	// 可以进行并发安全的设置
	for i := 0; i < 10; i++ {
		go SetCount(2)
	}
	time.Sleep(2 * time.Second)
	// 可以进行并发安全的读取
	fmt.Println(GetCount())
}
```

### Resource

- https://blog.csdn.net/wohu1104/article/details/105750171#t0

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E9%94%81/  

