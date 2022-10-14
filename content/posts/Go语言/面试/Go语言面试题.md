---
title: "【摘抄】Go常见面试题【由浅入深】2022版"
subtitle: ""
date: 2022-10-13T20:36:11+08:00
description: ""
keywords: "Go语言，面试题"
tags: ["zhihu","immersive-exercise","Golang"]
categories: ["Programming"]
---

{{< admonition  >}}
本文内容来源于：[【知乎】Go常见面试题【由浅入深】2022版](https://zhuanlan.zhihu.com/p/471490292)
{{< /admonition >}}

#### 🍇 = 和 := 的区别
{{<hide-text hide="=是赋值变量，:=是定义变量">}}

#### 🍈 指针的作用
{{< admonition info "详情" false >}}
- 获取变量的值
```go
import fmt

func main(){
  a := 1
  p := &a // 获取地址
  fmt.Prinf(*p) // 获取值
}
```

- 改变变量的值
```go
// 交换函数
func swap(a,b *int){
  *a,*b = *b,*a;
}
```

- 用指针替代传入函数
```go
type A struct{}
func (a *A) fun(){}
```
{{< /admonition >}}

#### 🍉 Go有异常类型吗
{{< admonition info "详情" false >}}
有。 Go用error类型代替try...catch
```go
_,err := funcDemo()
if err != nil {
  fmt.Println(err)
  return
}
```

也可以用errors.New()来定义自己的异常。
```go
type errorString struct {
  s string
}
func (e *errorString) Error() string {
  return e.s
}

// 多一个函数当作构造函数
func New(text string) error {
  return &errorString{text}
}
```
{{< /admonition >}}

#### 🍊 什么是协程(Goroutine)
{{<hide-text hide="协程是*用户状态轻量级线程*，它是*线程调度的基本单位*">}}

{{< details "展开了解更多" >}}
*通常在函数前加上go关键字就能实现并发。一个Goroutine会以一个很小的栈启动2KB或4KB，当遇到栈空间不足时，栈会自动伸缩， 因此可以轻易实现成千上万个goroutine同时启动。*
{{< /details >}}

#### 🍋 如何高效地拼接字符串
{{<hide-text hide="strings.Join ≈ strings.Builder > bytes.Buffer > \"+\" > fmt.Sprintf">}}

{{< details "展开了解更多" >}}
**利用加号(+)**

使用+操作符进行拼接时，会对字符串进行遍历，计算并开辟一个新的空间来存储原来的两个字符串。

**fmt.Sprintf**

由于采用了接口参数，必须要用反射获取值，因此有性能损耗。

**strings.Builder**

用WriteString()进行拼接，内部实现是指针+切片，同时String()返回拼接后的字符串，它是直接把[]byte转换为string，从而避免变量拷贝。

**bytes.Buffer**

bytes.Buffer是一个一个缓冲byte类型的缓冲器，这个缓冲器里存放着都是byte，bytes.buffer底层也是一个[]byte切片。

**strings.join**

strings.join也是基于strings.builder来实现的,并且可以自定义分隔符，在join方法内调用了b.Grow(n)方法，这个是进行初步的容量分配，而前面计算的n的长度就是我们要拼接的slice的长度，因为我们传入切片长度固定，所以提前进行容量分配可以减少内存分配，很高效。
{{< /details >}}

#### 🍌 什么是 rune 类型
{{<hide-text hide="Unicode在 Go 语言中称之为 rune，是 int32 类型的别名。">}}

#### 🍍 如何判断 map 中是否包含某个 key ？
```go 
var sample map[int]int
if _, ok := sample[10]; ok {

} else {

}
```

#### 🥭 Go 支持默认参数或可选参数吗？
{{<hide-text hide="不支持。但是可以利用结构体参数，或者...传入参数切片数组。">}}
```go
// 这个函数可以传入任意数量的整型参数
func sum(nums ...int) {
    total := 0
    for _, num := range nums {
        total += num
    }
    fmt.Println(total)
}
```

#### 🍎 defer 的执行顺序
{{<hide-text hide="defer执行顺序和调用顺序相反，类似于栈后进先出(LIFO)。">}}

```go
func test() int {
	i := 0
	defer func() {
		fmt.Println("defer1")
	}()
	defer func() {
		i += 1
		fmt.Println("defer2")
	}()
	return i
}

func main() {
	fmt.Println("return", test())
}
// defer2
// defer1
// return 0

func test() (i int) {
	i = 0
	defer func() {
		i += 1
		fmt.Println("defer2")
	}()
	return i
}

func main() {
	fmt.Println("return", test())
}
// defer2
// return 1
```

#### 🍏 如何获取一个结构体的所有tag？
{{< details "展开查看" >}}
*利用反射*

```go
import reflect
type Author struct {
	Name         int      `json:Name`
	Publications []string `json:Publication,omitempty`
}

func main() {
	t := reflect.TypeOf(Author{})
	for i := 0; i < t.NumField(); i++ {
		name := t.Field(i).Name
		s, _ := t.FieldByName(name)
		fmt.Println(name, s.Tag)
	}
}
```
{{< /details >}}

#### 🍐 如何判断 2 个字符串切片（slice) 是相等的？
{{<hide-text hide="reflect.DeepEqual() ， 但反射非常影响性能。">}}

#### 🍑 结构体打印时，%v 和 %+v 的区别
{{< details "展开查看" >}}
%v输出结构体各成员的值；

%+v输出结构体各成员的名称和值；

%#v输出结构体名称和结构体各成员的名称和值
{{< /details >}}

#### 🍒 Go 语言中如何表示枚举值(enums)？
```go
const (
	B = 1 << (10 * iota)
	KiB 
	MiB
	GiB
	TiB
	PiB
	EiB
)
```

#### 🍓 空 struct{} 的用途
{{<hide-text hide="*用map模拟一个set，那么就要把值置为struct{}，struct{}本身不占任何空间，可以避免任何多余的内存分配。*">}}

```go
type Set map[string]struct{}

func main() {
	set := make(Set)

	for _, item := range []string{"A", "A", "B", "C"} {
		set[item] = struct{}{}
	}
	fmt.Println(len(set)) // 3
	if _, ok := set["A"]; ok {
		fmt.Println("A exists") // A exists
	}
}
```

#### 🥝 init() 函数是什么时候执行的？
{{<hide-text hide=" *在main函数之前执行。*">}}

{{< details "展开查看详情" >}}
init()函数是go初始化的一部分，由runtime初始化每个导入的包，初始化不是按照从上到下的导入顺序，而是按照解析的依赖关系，没有依赖的包最先初始化。

每个包首先初始化包作用域的常量和变量（常量优先于变量），然后执行包的init()函数。同一个包，甚至是同一个源文件可以有多个init()函数。init()函数没有入参和返回值，不能被其他函数调用，同一个包内多个init()函数的执行顺序不作保证。

执行顺序：import –> const –> var –>init()–>main()

一个文件可以有多个init()函数！
{{< /details >}}

#### 🍅 如何知道一个对象是分配在栈上还是堆上？
{{<hide-text hide="*Go和C++不同，Go局部变量会进行逃逸分析。如果变量离开作用域后没有被引用，则优先分配到栈上，否则分配到堆上。*">}}

#### 🥥 2个nil 可能不相等吗？
{{<hide-text hide="*可能不等。interface在运行时绑定值，只有值为nil接口值才为nil，但是与指针的nil不相等。*">}}

#### 🫐 简述 Go 语言GC(垃圾回收)的工作原理
{{<hide-text hide="*垃圾回收机制是Go一大特(nan)色(dian)。Go1.3采用标记清除法， Go1.5采用三色标记法，Go1.8采用三色标记法+混合写屏障。*">}}

#### 🫒 函数返回局部变量的指针是否安全？
{{<hide-text hide="*这一点和C++不同，在Go里面返回局部变量的指针是安全的。因为Go会进行逃逸分析，如果发现局部变量的作用域超过该函数则会把指针分配到堆区，避免内存泄漏。*">}}

#### 🥑 非接口的任意类型 T() 都能够调用 *T 的方法吗？反过来呢？
{{<hide-text hide="*一个T类型的值可以调用*T类型声明的方法，当且仅当T是可寻址的。反之：*T 可以调用T()的方法，因为指针可以解引用。*">}}

#### 🍆 go slice是怎么扩容的？
{{< details "展开查看" >}}
Go <= 1.17

如果当前容量小于1024，则判断所需容量是否大于原来容量2倍，如果大于，当前容量加上所需容量；否则当前容量乘2。

如果当前容量大于1024，则每次按照1.25倍速度递增容量，也就是每次加上cap/4。

Go1.18之后，引入了新的扩容规则：[浅谈 Go 1.18.1的切片扩容机制](https://www.lookcos.cn/archives/1204.html)
{{< /details >}}

#### 🥔 无缓冲的 channel 和有缓冲的 channel 的区别？
{{< details "展开查看" >}}
**对于无缓冲区channel：**

发送的数据如果没有被接收方接收，那么发送方阻塞；如果一直接收不到发送方的数据，接收方阻塞；

**有缓冲的channel：**

发送方在缓冲区满的时候阻塞，接收方不阻塞；接收方在缓冲区为空的时候阻塞，发送方不阻塞。
{{< /details >}}

#### 🥕 为什么有协程泄露(Goroutine Leak)？
{{< details "展开查看" >}}
*协程泄漏是指协程创建之后没有得到释放。主要原因有：*

1. 缺少接收器，导致发送阻塞
2. 缺少发送器，导致接收阻塞
3. 死锁。多个协程由于竞争资源导致死锁。
4. 创建协程的没有回收。
{{< /details >}}

#### 🌽 Go 可以限制运行时操作系统线程的数量吗？ 常见的goroutine操作函数有哪些？
{{<hide-text hide="*可以，使用runtime.GOMAXPROCS(num int)可以设置线程数目。该值默认为CPU逻辑核数，如果设的太大，会引起频繁的线程切换，降低性能。*">}}

#### 🌶️ 如何控制协程数目。
{{<hide-text hide="*GOMAXPROCS 限制的是同时执行用户态 Go 代码的操作系统线程的数量，但是对于被系统调用阻塞的线程数量是没有限制的。*">}}

#### 🥒 new和make的区别？
{{< details "展开查看详情" >}}
1. *new只用于分配内存，返回一个指向地址的指针。它为每个新类型分配一片内存，初始化为0且返回类型\*T的内存地址，它相当于&T{}*
2. *make只可用于slice,map,channel的初始化,返回的是引用。*
{{< /details >}}

#### 🥬 请你讲一下Go面向对象是如何实现的？
{{< details "展开查看详情" >}}
*Go实现面向对象的两个关键是struct和interface。*

*封装：对于同一个包，对象对包内的文件可见；对不同的包，需要将对象以大写开头才是可见的。*

*继承：继承是编译时特征，在struct内加入所需要继承的类即可：*
```go
type A struct{}
type B struct{
	A
}
```

*多态：多态是运行时特征，Go多态通过interface来实现。类型和接口是松耦合的，某个类型的实例可以赋给它所实现的任意接口类型的变量。*

**Go支持多重继承，就是在类型中嵌入所有必要的父类型。**
{{< /details >}}

#### 🥦 init 函数
{{< details "展开查看详情" >}}
*go的init函数在main函数之前执行*

**init函数的特点：**

1. 初始化不能采用初始化表达式初始化的变量；
2. 程序运行前执行注册
3. 实现sync.Once功能
4. 不能被其它函数调用
5. init函数没有入口参数和返回值：
6. 每个包可以有多个init函数，每个源文件也可以有多个init函数。
7. 同一个包的init执行顺序，golang没有明确定义，编程时要注意程序不要依赖这个执行顺序。
8. 不同包的init函数按照包导入的依赖关系决定执行顺序。
{{< /details >}}

#### 🍄 下面这句代码是什么作用，为什么要定义一个空值？
```go
type GobCodec struct{
	conn io.ReadWriteCloser
	buf *bufio.Writer
	dec *gob.Decoder
	enc *gob.Encoder
}

type Codec interface {
	io.Closer
	ReadHeader(*Header) error
	ReadBody(interface{})  error
	Write(*Header, interface{}) error
}

var _ Codec = (*GobCodec)(nil)
```

{{<hide-text hide="*答：将nil转换为*GobCodec类型，然后再转换为Codec接口，如果转换失败，说明*GobCodec没有实现Codec接口的所有方法。*">}}

#### 🌰 golang的内存管理的原理清楚吗？简述go内存管理机制。
{{< details "展开查看详情" >}}
*golang内存管理基本是参考tcmalloc来进行的。go内存管理本质上是一个内存池，只不过内部做了很多优化：自动伸缩内存池大小，合理的切割内存块。*

> 一些基本概念：页Page：一块8K大小的内存空间。Go向操作系统申请和释放内存都是以页为单位的。span : 内存块，一个或多个连续的 page 组成一个 span 。如果把 page 比喻成工人， span 可看成是小队，工人被分成若干个队伍，不同的队伍干不同的活。sizeclass : 空间规格，每个 span 都带有一个 sizeclass ，标记着该 span 中的 page 应该如何使用。使用上面的比喻，就是 sizeclass 标志着 span 是一个什么样的队伍。object : 对象，用来存储一个变量数据内存空间，一个 span 在初始化时，会被切割成一堆等大的 object 。假设 object 的大小是 16B ， span 大小是 8K ，那么就会把 span 中的 page 就会被初始化 8K / 16B = 512 个 object 。所谓内存分配，就是分配一个 object 出去。

**mheap**

*一开始go从操作系统索取一大块内存作为内存池，并放在一个叫mheap的内存池进行管理，mheap将一整块内存切割为不同的区域，并将一部分内存切割为合适的大小。*

![mhead内存池](/posts/Go语言/面试/mhead.jpg)

**mcentral**

*用途相同的span会以链表的形式组织在一起存放在mcentral中。这里用途用sizeclass来表示，就是该span存储哪种大小的对象。*

*找到合适的 span 后，会从中取一个 object 返回给上层使用。*

**mcache**

*为了提高内存并发申请效率，加入缓存层mcache。每一个mcache和处理器P对应。Go申请内存首先从P的mcache中分配，如果没有可用的span再从mcentral中获取。*

> 参考资料：[Go 语言内存管理（二）：Go 内存管理](https://cloud.tencent.com/developer/article/1422392)

{{< /details >}}

#### 🫑 mutex有几种模式？
{{< details "展开查看详情" >}}
*mutex有两种模式：normal 和 starvation*

**正常模式**

*所有goroutine按照FIFO的顺序进行锁获取，被唤醒的goroutine和新请求锁的goroutine同时进行锁获取，通常新请求锁的goroutine更容易获取锁(持续占有cpu)，被唤醒的goroutine则不容易获取到锁。公平性：否。*

**饥饿模式**

*所有尝试获取锁的goroutine进行等待排队，新请求锁的goroutine不会进行锁获取(禁用自旋)，而是加入队列尾部等待获取锁。公平性：是。*

> 参考链接：[Go Mutex 饥饿模式，GO 互斥锁（Mutex）原理](https://blog.csdn.net/baolingye/article/details/111357407#:~:text=%E6%AF%8F%E4%B8%AAMutex%E9%83%BD,tarving%E3%80%82)
{{< /details >}}

#### 🍞 go如何进行调度的。GMP中状态流转。
{{< details "展开查看详情" >}}
*Go里面GMP分别代表：G：goroutine，M：线程（真正在CPU上跑的），P：调度器。*

{{< figure src="/posts/Go语言/面试/GMP模型.jpg" title="GMP模型" >}}

*调度器是M和G之间桥梁。*

**go进行调度过程：**

1. 某个线程尝试创建一个新的G，那么这个G就会被安排到这个线程的G本地队列LRQ中，如果LRQ满了，就会分配到全局队列GRQ中；
2. 尝试获取当前线程的M，如果无法获取，就会从空闲的M列表中找一个，如果空闲列表也没有，那么就创建一个M，然后绑定G与P运行。
3. 进入调度循环：
   1. 找到一个合适的G
   2. 执行G，完成以后退出
{{< /details >}}

#### 🥖 Go什么时候发生阻塞？阻塞时，调度器会怎么做。
{{< details "展开查看详情" >}}
1. 用于原子、互斥量或通道操作导致goroutine阻塞，调度器将把当前阻塞的goroutine从本地运行队列LRQ换出，并重新调度其它goroutine；
2. 由于网络请求和IO导致的阻塞，Go提供了网络轮询器（Netpoller）来处理，后台用epoll等技术实现IO多路复用。

> 更多关于netpoller的内容可以参看：[https://strikefreedom.top/archives/go-netpoll-io-multiplexing-reactor](https://strikefreedom.top/archives/go-netpoll-io-multiplexing-reactor)
{{< /details >}}

#### 🥨 Go中GMP有哪些状态？
{{< details "展开查看详情" >}}
{{< figure src="/posts/Go语言/面试/GMP状态.jpg" title="GMP状态" >}}

**G的状态：**

_Gidle：刚刚被分配并且还没有被初始化，值为0，为创建goroutine后的默认值

_Grunnable： 没有执行代码，没有栈的所有权，存储在运行队列中，可能在某个P的本地队列或全局队列中(如上图)。

_Grunning： 正在执行代码的goroutine，拥有栈的所有权(如上图)。

_Gsyscall：正在执行系统调用，拥有栈的所有权，与P脱离，但是与某个M绑定，会在调用结束后被分配到运行队列(如上图)。

_Gwaiting：被阻塞的goroutine，阻塞在某个channel的发送或者接收队列(如上图)。

_Gdead： 当前goroutine未被使用，没有执行代码，可能有分配的栈，分布在空闲列表gFree，可能是一个刚刚初始化的goroutine，也可能是执行了goexit退出的goroutine(如上图)。

_Gcopystac：栈正在被拷贝，没有执行代码，不在运行队列上，执行权在

_Gscan ： GC 正在扫描栈空间，没有执行代码，可以与其他状态同时存在。

**P的状态：**

_Pidle ：处理器没有运行用户代码或者调度器，被空闲队列或者改变其状态的结构持有，运行队列为空

_Prunning ：被线程 M 持有，并且正在执行用户代码或者调度器(如上图)

_Psyscall：没有执行用户代码，当前线程陷入系统调用(如上图)

_Pgcstop ：被线程 M 持有，当前处理器由于垃圾回收被停止

_Pdead ：当前处理器已经不被使用

**M的状态：**

自旋线程：处于运行状态但是没有可执行goroutine的线程，数量最多为GOMAXPROC，若是数量大于GOMAXPROC就会进入休眠。

非自旋线程：处于运行状态有可执行goroutine的线程。
{{< /details >}}

#### 🥯 GMP能不能去掉P层？会怎么样？
{{< details "展开查看详情" >}}
**P层的作用**

1. 每个 P 有自己的本地队列，大幅度的减轻了对全局队列的直接依赖，所带来的效果就是锁竞争的减少。而 GM 模型的性能开销大头就是锁竞争。
2. 每个 P 相对的平衡上，在 GMP 模型中也实现了 Work Stealing 算法，如果 P 的本地队列为空，则会从全局队列或其他 P 的本地队列中窃取可运行的 G 来运行，减少空转，提高了资源利用率。

> 参考资料：[https://juejin.cn/post/6968311281220583454](https://juejin.cn/post/6968311281220583454)
{{< /details >}}

#### 🧀 如果有一个G一直占用资源怎么办？什么是work stealing算法？
{{< details "展开查看详情" >}}
*如果有个goroutine一直占用资源，那么GMP模型会从正常模式转变为饥饿模式（类似于mutex），允许其它goroutine使用work stealing抢占（禁用自旋锁）。*

*work stealing算法指，一个线程如果处于空闲状态，则帮其它正在忙的线程分担压力，从全局队列取一个G任务来执行，可以极大提高执行效率。*
{{< /details >}}

#### 🍖 goroutine什么情况会发生内存泄漏？如何避免。
{{< details "展开查看详情" >}}
*在Go中内存泄露分为暂时性内存泄露和永久性内存泄露。*

**暂时性内存泄露**

1. 获取长字符串中的一段导致长字符串未释放
2. 获取长slice中的一段导致长slice未释放
3. 在长slice新建slice导致泄漏

*string相比切片少了一个容量的cap字段，可以把string当成一个只读的切片类型。获取长string或者切片中的一段内容，由于新生成的对象和老的string或者切片共用一个内存空间，会导致老的string和切片资源暂时得不到释放，造成短暂的内存泄漏*

**永久性内存泄露**

1. goroutine永久阻塞而导致泄漏
2. time.Ticker未关闭导致泄漏
3. 不正确使用Finalizer（Go版本的析构函数）导致泄漏
{{< /details >}}

#### 🍗 Go GC有几个阶段
{{< details "展开查看详情" >}}
*目前的go GC采用三色标记法和混合写屏障技术。*

**Go GC有四个阶段:**

1. STW，开启混合写屏障，扫描栈对象；
2. 将所有对象加入白色集合，从根对象开始，将其放入灰色集合。每次从灰色集合取出一个对象标记为黑色，然后遍历其子对象，标记为灰色，放入灰色集合；
3. 如此循环直到灰色集合为空。剩余的白色对象就是需要清理的对象。
4. STW，关闭混合写屏障；
5. 在后台进行GC（并发）。
{{< /details >}}

#### 🥩 go竞态条件了解吗？
{{< details "展开查看详情" >}}
*所谓竞态竞争，就是当两个或以上的goroutine访问相同资源时候，对资源进行读/写。*

比如var a int = 0，有两个协程分别对a+=1，我们发现最后a不一定为2.这就是竞态竞争。

通常我们可以用go run -race xx.go来进行检测。

解决方法是，对临界区资源上锁，或者使用原子操作(atomics)，原子操作的开销小于上锁。
{{< /details >}}

#### 🥓 如果若干个goroutine，有一个panic会怎么做？
{{< details "展开查看详情" >}}
*有一个panic，那么剩余goroutine也会退出，程序退出。如果不想程序退出，那么必须通过调用 recover() 方法来捕获 panic 并恢复将要崩掉的程序*

> 参考理解：[goroutine配上panic会怎样](https://blog.csdn.net/huorongbj/article/details/123013273)。
{{< /details >}}

#### 🍔 defer可以捕获goroutine的子goroutine吗？
{{< details "展开查看详情" >}}
*不可以。它们处于不同的调度器P中。对于子goroutine，必须通过 recover() 机制来进行恢复，然后结合日志进行打印（或者通过channel传递error）*

```go
// 心跳函数
func Ping(ctx context.Context) error {
    ... code ...
 
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Errorc(ctx, "ping panic: %v, stack: %v", r, string(debug.Stack()))
			}
		}()
 
        ... code ...
	}()
 
    ... code ...
 
	return nil
}
```
{{< /details >}}

#### 🍟 gRPC是什么？
{{< details "展开查看详情" >}}
*基于go的远程过程调用。RPC 框架的目标就是让远程服务调用更加简单、透明，RPC 框架负责屏蔽底层的传输方式（TCP 或者 UDP）、序列化方式（XML/Json/ 二进制）和通信细节。服务调用者可以像调用本地接口一样调用远程的服务提供者，而不需要关心底层通信细节和调用过程。*

{{< figure src="/posts/Go语言/面试/gRPC框架图.jpg" title="gRPC框架图" >}}
{{< /details >}}

#### 🍕 微服务了解吗？
{{< details "展开查看详情" >}}
*微服务是一种开发软件的架构和组织方法，其中软件由通过明确定义的 API 进行通信的小型独立服务组成。微服务架构使应用程序更易于扩展和更快地开发，从而加速创新并缩短新功能的上市时间。*

{{< figure src="/posts/Go语言/面试/微服务示意图.jpg" title="微服务示意图" >}}

> 参考资料：[什么是微服务？| AWS](https://aws.amazon.com/cn/microservices/)
{{< /details >}}

#### 🌭 服务发现是怎么做的？
{{< details "展开查看详情" >}}
*主要有两种服务发现机制：客户端发现和服务端发现。*

**客户端发现模式：**当我们使用客户端发现的时候，客户端负责决定可用服务实例的网络地址并且在集群中对请求负载均衡, 客户端访问服务登记表，也就是一个可用服务的数据库，然后客户端使用一种负载均衡算法选择一个可用的服务实例然后发起请求

{{< figure src="/posts/Go语言/面试/客户端发现模式.jpg" title="客户端发现模式" >}}

**服务端发现模式：**客户端通过负载均衡器向某个服务提出请求，负载均衡器查询服务注册表，并将请求转发到可用的服务实例。如同客户端发现，服务实例在服务注册表中注册或注销。

{{< figure src="/posts/Go语言/面试/服务端发现模式.jpg" title="服务的发现模式" >}}

> 参考资料：[「Chris Richardson 微服务系列」服务发现的可行方案以及实践案例](http://blog.daocloud.io/3289.html)
{{< /details >}}

#### 🥪 ETCD用过吗？
{{< details "展开查看详情" >}}
etcd是一个高度一致的分布式键值存储，它提供了一种可靠的方式来存储需要由分布式系统或机器集群访问的数据。它可以优雅地处理网络分区期间的领导者选举，即使在领导者节点中也可以容忍机器故障。

etcd 是用Go语言编写的，它具有出色的跨平台支持，小的二进制文件和强大的社区。etcd机器之间的通信通过Raft共识算法处理。

> 关于文档可以参考：[v3.5 docs](https://etcd.io/docs/v3.5/)
{{< /details >}}

#### 🌮 GIN怎么做参数校验？
{{< details "展开查看详情" >}}
*go采用validator作参数校验。*

1. 使用验证tag或自定义validator进行跨字段Field和跨结构体验证。
2. 允许切片、数组和哈希表，多维字段的任何或所有级别进行校验。
3. 能够对哈希表key和value进行验证
4. 通过在验证之前确定它的基础类型来处理类型接口。
5. 别名验证标签，允许将多个验证映射到单个标签，以便更轻松地定义结构体上的验证
6. gin web 框架的默认验证器；

参考资料：[validator package - pkg.go.dev](https://pkg.go.dev/github.com/go-playground/validator#section-readme)
{{< /details >}}

#### 🌯 中间件用过吗？
{{<hide-text hide="*Middleware是Web的重要组成部分，中间件（通常）是一小段代码，它们接受一个请求，对其进行处理，每个中间件只处理一件事情，完成后将其传递给另一个中间件或最终处理程序，这样就做到了程序的解耦。*">}}

#### 🥙 Go解析Tag是怎么实现的？
{{< details "展开查看详情" >}}
*Go解析tag采用的是反射。*

具体来说使用reflect.ValueOf方法获取其反射值，然后获取其Type属性，之后再通过Field(i)获取第i+1个field，再.Tag获得Tag。

反射实现的原理在: `src/reflect/type.go`中
{{< /details >}}

#### 🍲 你项目有优雅的启停吗？
{{< details "展开查看详情" >}}
所谓「优雅」启停就是在启动退出服务时要满足以下几个条件：
1. 不可以关闭现有连接（进程）
2. 新的进程启动并「接管」旧进程
3. 连接要随时响应用户请求，不可以出现拒绝请求的情况
4. 停止的时候，必须处理完既有连接，并且停止接收新的连接。

***为此我们必须引用信号来完成这些目的：***

启动：
1. 监听SIGHUP（在用户终端连接(正常或非正常)结束时发出）；
2. 收到信号后将服务监听的文件描述符传递给新的子进程，此时新老进程同时接收请求；

退出
1. 监听SIGINT和SIGSTP和SIGQUIT等。
2. 父进程停止接收新请求，等待旧请求完成（或超时）；
3. 父进程退出。

*实现：go1.8采用Http.Server内置的Shutdown方法支持优雅关机。 然后fvbock/endless可以实现优雅重启。*

> 参考资料：[gin框架实践连载八 | 如何优雅重启和停止 - 掘金](https://juejin.cn/post/6867074626427502600#heading-3)，[优雅地关闭或重启 go web 项目](http://www.phpxs.com/post/7186/)
{{< /details >}}

#### 🥗 持久化怎么做的？
{{< details "展开查看详情" >}}
*所谓持久化就是将要保存的字符串写到硬盘等设备。*
1. 最简单的方式就是采用ioutil的WriteFile()方法将字符串写到磁盘上，这种方法面临格式化方面的问题。
2. 更好的做法是将数据按照固定协议进行组织再进行读写，比如JSON，XML，Gob，csv等。
3. 如果要考虑高并发和高可用，必须把数据放入到数据库中，比如MySQL，PostgreDB，MongoDB等。

> 参考链接：[Golang 持久化](https://www.jianshu.com/p/015aca3e11ae)
{{< /details >}}

#### 🍿 channel 死锁的场景
{{< details "展开查看详情" >}}
- *当一个channel中没有数据，而直接读取时，会发生死锁：*
```go
q := make(chan int,2)
<-q
```

*解决方案是采用select语句，再default放默认处理方式：*
```go
q := make(chan int,2)
select{
   case val:=<-q:
   default:
         ...

}
```

- *当channel数据满了，再尝试写数据会造成死锁：*
```go
q := make(chan int,2)
q<-1
q<-2
q<-3
```

*解决方法，采用select*
```go
func main() {
	q := make(chan int, 2)
	q <- 1
	q <- 2
	select {
	case q <- 3:
		fmt.Println("ok")
	default:
		fmt.Println("wrong")
	}

}
```

- *向一个关闭的channel写数据。*

注意：一个已经关闭的channel，只能读数据，不能写数据。

参考资料：[Golang关于channel死锁情况的汇总以及解决方案](https://blog.csdn.net/qq_35976351/article/details/81984117)
{{< /details >}}

#### 🥫 对已经关闭的chan进行读写会怎么样？
{{< details "展开查看详情" >}}
1. 读已经关闭的chan能一直读到东西，但是读到的内容根据通道内关闭前是否有元素而不同。
   1. 如果chan关闭前，buffer内有元素还未读,会正确读到chan内的值，且返回的第二个bool值（是否读成功）为true。
   2. 如果chan关闭前，buffer内有元素已经被读完，chan内无值，接下来所有接收的值都会非阻塞直接成功，返回 channel 元素的零值，但是第二个bool值一直为false。

写已经关闭的chan会panic。
{{< /details >}}

#### 🍱 说说 atomic底层怎么实现的.
{{< details "展开查看详情" >}}
*atomic源码位于`sync\atomic`。通过阅读源码可知，atomic采用CAS（CompareAndSwap）的方式实现的。所谓CAS就是使用了CPU中的原子性操作。在操作共享变量的时候，CAS不需要对其进行加锁，而是通过类似于乐观锁的方式进行检测，总是假设被操作的值未曾改变（即与旧值相等），并一旦确认这个假设的真实性就立即进行值替换。本质上是不断占用CPU资源来避免加锁的开销。*

> 参考资料：[Go语言的原子操作atomic - 编程猎人](https://www.programminghunter.com/article/37392193442/)
{{< /details >}}

#### 🍘 channel底层实现？是否线程安全。
{{< details "展开查看详情" >}}
*channel底层实现在src/runtime/chan.go中*

*channel内部是一个循环链表。内部包含buf, sendx, recvx, lock ,recvq, sendq几个部分；*

**buf是有缓冲的channel所特有的结构，用来存储缓存数据。是个循环链表；**
1. sendx和recvx用于记录buf这个循环链表中的发送或者接收的index；
2. lock是个互斥锁；
3. recvq和sendq分别是接收(<-channel)或者发送(channel <- xxx)的goroutine抽象出来的结构体(sudog)的队列。是个双向链表。

*channel是线程安全的。*

> 参考资料：[Kitou：Golang 深度剖析 -- channel的底层实现](https://zhuanlan.zhihu.com/p/264305133)
{{< /details >}}

#### 🍚 map的底层实现。
{{< details "展开查看详情" >}}
*源码位于src\runtime\map.go 中。*

*go的map和C++map不一样，底层实现是哈希表，包括两个部分：hmap和bucket。*

**里面最重要的是buckets（桶），buckets是一个指针，最终它指向的是一个结构体**
```go
// A bucket for a Go map.
type bmap struct {
    tophash [bucketCnt]uint8
}
```
每个bucket固定包含8个key和value(可以查看源码bucketCnt=8).实现上面是一个固定的大小连续内存块，分成四部分：每个条目的状态，8个key值，8个value值，指向下个bucket的指针。

创建哈希表使用的是makemap函数.map 的一个关键点在于，哈希函数的选择。在程序启动时，会检测 cpu 是否支持 aes，如果支持，则使用 aes hash，否则使用 memhash。这是在函数 alginit() 中完成，位于路径：src/runtime/alg.go 下。

map查找就是将key哈希后得到64位（64位机）用最后B个比特位计算在哪个桶。在 bucket 中，从前往后找到第一个空位。这样，在查找某个 key 时，先找到对应的桶，再去遍历 bucket 中的 key。

> 关于map的查找和扩容可以参考[map的用法到map底层实现分析。](https://blog.csdn.net/chenxun_2010/article/details/103768011?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_aa&spm=1001.2101.3001.4242.1&utm_relevant_index=3)
{{< /details >}}

#### 🍛 select的实现原理？
{{< details "展开查看详情" >}}
**select源码位于src\runtime\select.go，最重要的scase 数据结构为：**
```go
type scase struct {
	c    *hchan         // chan
	elem unsafe.Pointer // data element
}
```
*scase.c为当前case语句所操作的channel指针，这也说明了一个case语句只能操作一个channel。*

**scase.elem表示缓冲区地址：**
1. caseRecv ： scase.elem表示读出channel的数据存放地址；
2. caseSend ： scase.elem表示将要写入channel的数据存放地址；

**select的主要实现位于：select.go函数：其主要功能如下：**
1. 锁定scase语句中所有的channel
2. 按照随机顺序检测scase中的channel是否ready
   1. 如果case可读，则读取channel中数据，解锁所有的channel，然后返回(case index, true)
   2. 如果case可写，则将数据写入channel，解锁所有的channel，然后返回(case index, false)
   3. 如果case可写，则将数据写入channel，解锁所有的channel，然后返回(case index, false)
3. 所有case都未ready，且没有default语句
   1. 将当前协程加入到所有channel的等待队列
   2. 当将协程转入阻塞，等待被唤醒
4. 唤醒后返回channel对应的case index
   1. 如果是读操作，解锁所有的channel，然后返回(case index, true)
   2. 如果是写操作，解锁所有的channel，然后返回(case index, false)

> 参考资料：[Go select的使用和实现原理.](https://www.cnblogs.com/wuyepeng/p/13910678.html#:~:text=%E4%B8%80%E3%80%81select%E7%AE%80%E4%BB%8B.%201.Go%E7%9A%84select%E8%AF%AD%E5%8F%A5%E6%98%AF%E4%B8%80%E7%A7%8D%E4%BB%85%E8%83%BD%E7%94%A8%E4%BA%8Echannl%E5%8F%91%E9%80%81%E5%92%8C%E6%8E%A5%E6%94%B6%E6%B6%88%E6%81%AF%E7%9A%84%E4%B8%93%E7%94%A8%E8%AF%AD%E5%8F%A5%EF%BC%8C%E6%AD%A4%E8%AF%AD%E5%8F%A5%E8%BF%90%E8%A1%8C%E6%9C%9F%E9%97%B4%E6%98%AF%E9%98%BB%E5%A1%9E%E7%9A%84%EF%BC%9B%E5%BD%93select%E4%B8%AD%E6%B2%A1%E6%9C%89case%E8%AF%AD%E5%8F%A5%E7%9A%84%E6%97%B6%E5%80%99%EF%BC%8C%E4%BC%9A%E9%98%BB%E5%A1%9E%E5%BD%93%E5%89%8Dgroutine%E3%80%82.%202.select%E6%98%AFGolang%E5%9C%A8%E8%AF%AD%E8%A8%80%E5%B1%82%E9%9D%A2%E6%8F%90%E4%BE%9B%E7%9A%84I%2FO%E5%A4%9A%E8%B7%AF%E5%A4%8D%E7%94%A8%E7%9A%84%E6%9C%BA%E5%88%B6%EF%BC%8C%E5%85%B6%E4%B8%93%E9%97%A8%E7%94%A8%E6%9D%A5%E6%A3%80%E6%B5%8B%E5%A4%9A%E4%B8%AAchannel%E6%98%AF%E5%90%A6%E5%87%86%E5%A4%87%E5%AE%8C%E6%AF%95%EF%BC%9A%E5%8F%AF%E8%AF%BB%E6%88%96%E5%8F%AF%E5%86%99%E3%80%82.,3.select%E8%AF%AD%E5%8F%A5%E4%B8%AD%E9%99%A4default%E5%A4%96%EF%BC%8C%E6%AF%8F%E4%B8%AAcase%E6%93%8D%E4%BD%9C%E4%B8%80%E4%B8%AAchannel%EF%BC%8C%E8%A6%81%E4%B9%88%E8%AF%BB%E8%A6%81%E4%B9%88%E5%86%99.%204.select%E8%AF%AD%E5%8F%A5%E4%B8%AD%E9%99%A4default%E5%A4%96%EF%BC%8C%E5%90%84case%E6%89%A7%E8%A1%8C%E9%A1%BA%E5%BA%8F%E6%98%AF%E9%9A%8F%E6%9C%BA%E7%9A%84.%205.select%E8%AF%AD%E5%8F%A5%E4%B8%AD%E5%A6%82%E6%9E%9C%E6%B2%A1%E6%9C%89default%E8%AF%AD%E5%8F%A5%EF%BC%8C%E5%88%99%E4%BC%9A%E9%98%BB%E5%A1%9E%E7%AD%89%E5%BE%85%E4%BB%BB%E4%B8%80case.%206.select%E8%AF%AD%E5%8F%A5%E4%B8%AD%E8%AF%BB%E6%93%8D%E4%BD%9C%E8%A6%81%E5%88%A4%E6%96%AD%E6%98%AF%E5%90%A6%E6%88%90%E5%8A%9F%E8%AF%BB%E5%8F%96%EF%BC%8C%E5%85%B3%E9%97%AD%E7%9A%84channel%E4%B9%9F%E5%8F%AF%E4%BB%A5%E8%AF%BB%E5%8F%96)
{{< /details >}}

#### 🍜 go的interface怎么实现的？
{{< details "展开查看详情" >}}
*go interface源码在runtime\iface.go中。*

*go的接口由两种类型实现iface和eface。iface是包含方法的接口，而eface不包含方法。*

**iface-对应的数据结构是（位于src\runtime\runtime2.go）：**
```go
type iface struct {
	tab  *itab
	data unsafe.Pointer
}
```
*可以简单理解为，tab表示接口的具体结构类型，而data是接口的值。*

**itab:**
```go
type itab struct {
	inter *interfacetype //此属性用于定位到具体interface
	_type *_type //此属性用于定位到具体interface
	hash  uint32 // copy of _type.hash. Used for type switches.
	_     [4]byte
	fun   [1]uintptr // variable sized. fun[0]==0 means _type does not implement inter.
}
```

属性interfacetype类似于_type，其作用就是interface的公共描述，类似的还有maptype、arraytype、chantype…其都是各个结构的公共描述，可以理解为一种外在的表现信息。interfaetype和type唯一确定了接口类型，而hash用于查询和类型判断。fun表示方法集。

**eface-与iface基本一致，但是用_type直接表示类型，这样的话就无法使用方法。**
```go
type eface struct {
	_type *_type
	data  unsafe.Pointer
}
```

> 这里篇幅有限，深入讨论可以看：[深入研究 Go interface 底层实现](https://halfrost.com/go_interface/#toc-1)
{{< /details >}}

#### 🍝 go的reflect 底层实现
{{< details "展开查看详情" >}}
*go reflect源码位于src\reflect\下面，作为一个库独立存在。反射是基于接口实现的。*

**Go反射有三大法则：**
1. 反射从接口映射到反射对象；
{{< figure src="/posts/Go语言/面试/法则1.jpg" title="法则1" >}}
2. 反射从反射对象映射到接口值；
{{< figure src="/posts/Go语言/面试/法则2.jpg" title="法则2" >}}
3. 只有值可以修改(settable)，才可以修改反射对象。

*Go反射基于上述三点实现。我们先从最核心的两个源文件入手type.go和value.go.*

*type用于获取当前值的类型。value用于获取当前的值。*

> 参考资料：[The Laws of Reflection](https://go.dev/blog/laws-of-reflection)， [图解go反射实现原理](https://i6448038.github.io/2020/02/15/golang-reflection/)
{{< /details >}}

#### 🍠 go GC的原理知道吗？
{{<hide-text hide="*如果需要从源码角度解释GC，推荐阅读（非常详细，图文并茂）：[内存管理](https://draveness.me/golang/docs/part3-runtime/ch07-memory/golang-garbage-collector/)*">}}

#### 🍢 go里用过哪些设计模式 ?
{{< link "https://zhuanlan.zhihu.com/p/542596378" "Go设计模式常见面试题【2022版】" "" true >}}

#### 🍣 go的调试/分析工具用过哪些。
{{< details "展开查看详情" >}}
1. go cover : 测试代码覆盖率；
2. godoc: 用于生成go文档；
3. pprof：用于性能调优，针对cpu，内存和并发；
4. race：用于竞争检测；
{{< /details >}}

#### 🍤进程被kill，如何保证所有goroutine顺利退出
{{< details "展开查看详情" >}}
*goroutine监听SIGKILL信号，一旦接收到SIGKILL，则立刻退出。可采用select方法。*
```go
var wg = &sync.WaitGroup{}

func main() {
	wg.Add(1)

	go func() {
		c1 := make(chan os.Signal, 1)
		signal.Notify(c1, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
		fmt.Printf("goroutine 1 receive a signal : %v\n\n", <-c1)
		wg.Done()
	}()

	wg.Wait()
	fmt.Printf("all groutine done!\n")
}
```
{{< /details >}}

#### 🍥 说说context包的作用？你用过哪些，原理知道吗？
{{< details "展开查看详情" >}}
*context可以用来在goroutine之间传递上下文信息，相同的context可以传递给运行在不同goroutine中的函数，上下文对于多个goroutine同时使用是安全的，context包定义了上下文类型，可以使用background、TODO创建一个上下文，在函数调用链之间传播context，也可以使用WithDeadline、WithTimeout、WithCancel 或 WithValue 创建的修改副本替换它，听起来有点绕，其实总结起就是一句话：context的作用就是在不同的goroutine之间同步请求特定的数据、取消信号以及处理请求的截止日期。*

> 关于context原理，可以参看：[小白也能看懂的context包详解：从入门到精通](https://cloud.tencent.com/developer/article/1900658)
{{< /details >}}

#### 🥮 grpc为啥好，基本原理是什么，和http比呢
{{< details "展开查看详情" >}}
*官方介绍：gRPC 是一个现代开源的高性能远程过程调用 (RPC) 框架，可以在任何环境中运行。它可以通过对负载平衡、跟踪、健康检查和身份验证的可插拔支持有效地连接数据中心内和跨数据中心的服务。它也适用于分布式计算的最后一英里，将设备、移动应用程序和浏览器连接到后端服务。*

**区别：**
1. rpc是远程过程调用，就是本地去调用一个远程的函数，而http是通过 url和符合restful风格的数据包去发送和获取数据；
2. rpc的一般使用的编解码协议更加高效，比如grpc使用protobuf编解码。而http的一般使用json进行编解码，数据相比rpc更加直观，但是数据包也更大，效率低下；
3. rpc一般用在服务内部的相互调用，而http则用于和用户交互；

**相似点：**

都有类似的机制，例如grpc的metadata机制和http的头机制作用相似，而且web框架，和rpc框架中都有拦截器的概念。grpc使用的是http2.0协议。

官网：[gRPC](https://grpc.io/)
{{< /details >}}

#### 🍡实现使用字符串函数名，调用函数。
{{< details "展开查看详情" >}}
*思路：采用反射的Call方法实现。*
```go
package main
import (
	"fmt"
    "reflect"
)

type Animal struct{
    
}

func (a *Animal) Eat(){
    fmt.Println("Eat")
}

func main(){
    a := Animal{}
    reflect.ValueOf(&a).MethodByName("Eat").Call([]reflect.Value{})
    
}
```
{{< /details >}}

#### 🥟 （Goroutine）有三个函数，分别打印"cat", "fish","dog"要求每一个函数都用一个goroutine，按照顺序打印100次。
{{< details "展开查看详情" >}}
*此题目考察channel，用三个无缓冲channel，如果一个channel收到信号则通知下一个。*
```go
package main

import (
	"fmt"
	"time"
)

var dog = make(chan struct{})
var cat = make(chan struct{})
var fish = make(chan struct{})

func Dog() {
	<-fish
	fmt.Println("dog")
	dog <- struct{}{}
}

func Cat() {
	<-dog
	fmt.Println("cat")
	cat <- struct{}{}
}

func Fish() {
	<-cat
	fmt.Println("fish")
	fish <- struct{}{}
}

func main() {
	for i := 0; i < 100; i++ {
		go Dog()
		go Cat()
		go Fish()
	}
	fish <- struct{}{}

	time.Sleep(10 * time.Second)
}
```
{{< /details >}}

#### 🦀 两个协程交替打印10个字母和数字
{{< details "展开查看详情" >}}
*思路：采用channel来协调goroutine之间顺序。*

*主线程一般要waitGroup等待协程退出，这里简化了一下直接sleep。*
```go 
package main

import (
	"fmt"
	"time"
)

var word = make(chan struct{}, 1)
var num = make(chan struct{}, 1)

func printNums() {
	for i := 0; i < 10; i++ {
		<-word
		fmt.Println(1)
		num <- struct{}{}
	}
}
func printWords() {
	for i := 0; i < 10; i++ {
		<-num
		fmt.Println("a")
		word <- struct{}{}
	}
}

func main() {
	num <- struct{}{}
	go printNums()
	go printWords()
	time.Sleep(time.Second * 1)
}
```
{{< /details >}}

#### 🦞 启动 2个groutine 2秒后取消， 第一个协程1秒执行完，第二个协程3秒执行完。
{{< details "展开查看详情" >}}
*思路：采用ctx, _ := context.WithTimeout(context.Background(), time.Second\*2)实现2s取消。协程执行完后通过channel通知，是否超时。*
```go
package main

import (
	"context"
	"fmt"
	"time"
)

func f1(in chan struct{}) {

	time.Sleep(1 * time.Second)
	in <- struct{}{}

}

func f2(in chan struct{}) {
	time.Sleep(3 * time.Second)
	in <- struct{}{}
}

func main() {
	ch1 := make(chan struct{})
	ch2 := make(chan struct{})
	ctx, _ := context.WithTimeout(context.Background(), 2*time.Second)

	go func() {
		go f1(ch1)
		select {
		case <-ctx.Done():
			fmt.Println("f1 timeout")
			break
		case <-ch1:
			fmt.Println("f1 done")
		}
	}()

	go func() {
		go f2(ch2)
		select {
		case <-ctx.Done():
			fmt.Println("f2 timeout")
			break
		case <-ch2:
			fmt.Println("f2 done")
		}
	}()
	time.Sleep(time.Second * 5)
}
```
{{< /details >}}

#### 🦐 当select监控多个chan同时到达就绪态时，如何先执行某个任务？
{{< details "展开查看详情" >}}
*可以在子case再加一个for select语句。*
```go
func priority_select(ch1, ch2 <-chan string) {
	for {
		select {
		case val := <-ch1:
			fmt.Println(val)
		case val2 := <-ch2:
		priority:
			for {
				select {
				case val1 := <-ch1:
					fmt.Println(val1)

				default:
					break priority
				}
			}
			fmt.Println(val2)
		}
	}

}
```
{{< /details >}}