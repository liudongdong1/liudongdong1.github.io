# unsafe


> 可以绕过 `Go` 的内存安全机制，直接对内存进行读写。所以有时候出于性能需要，还是会冒险使用它来对内存进行操作。

### 1. 指针类型转换

> `unsafe.Pointer` 是一种特殊意义的指针，可以表示任意类型的地址，类似 `C` 语言里的 `void*` 指针，是全能型的。

```go
func main() {
	i := 10
	ip := &i
	var fp *float64 = (*float64)(unsafe.Pointer(ip))
	*fp = *fp * 3
	fmt.Println(*ip) // 30
}
```

### 2. uintptr 指针类型

> `uintptr` 也是一种指针类型，它足够大，可以表示任何指针。
>
>  `unsafe.Pointer` 不能进行运算，比如不支持 +（加号）运算符操作，但是 `uintptr` 可以。通过它，可以对指针偏移进行计算，这样就可以访问特定的内存，达到对特定内存读写的目的，这是真正内存级别的操作。
>
> `uintptr` 是用于指针运算的，`GC` 不把 `uintptr` 当指针，也就是说 `uintptr` 无法持有对象， `uintptr` 类型的目标会被回收；

```go
func main() {
	p := new(person)
	//Name是person的第一个字段不用偏移，即可通过指针修改
	pName := (*string)(unsafe.Pointer(p))
	*pName = "wohu"
	//Age并不是person的第一个字段，所以需要进行偏移，这样才能正确定位到Age字段这块内存，才可以正确的修改
	pAge := (*int)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + unsafe.Offsetof(p.Age)))
	*pAge = 20
	fmt.Printf("p is %#v", *p) // p is main.person{Name:"wohu", Age:20}
}

type person struct {
	Name string
	Age  int
}
```

### 3. 指针转换规则

- 任何类型的 `*T` 都可以转换为 `unsafe.Pointer` ；
- `unsafe.Pointer` 也可以转换为任何类型的 `*T` ；
- `unsafe.Pointer` 可以转换为 `uintptr` ；
- `uintptr` 也可以转换为 `unsafe.Pointer` ；

### 4. unsafe.Sizeof

> `Sizeof` 函数可以返回一个类型所占用的内存大小，这个`大小只与类型有关`，和类型对应的变量存储的内容大小无关，比如 `bool` 型占用一个字节、`int8` 也占用一个字节。
>
> - 一个 `struct` 结构体的内存占用大小，等于它包含的字段类型内存占用大小之和。

```go
func main() {
	fmt.Println(unsafe.Sizeof(true))                 // 1
	fmt.Println(unsafe.Sizeof(int8(0)))              // 1
	fmt.Println(unsafe.Sizeof(int16(0)))             // 2
	fmt.Println(unsafe.Sizeof(int32(0)))             // 4
	fmt.Println(unsafe.Sizeof(int64(0)))             // 8
	fmt.Println(unsafe.Sizeof(int(0)))               // 8
	fmt.Println(unsafe.Sizeof(string("张三")))         // 16
	fmt.Println(unsafe.Sizeof([]string{"李四", "张三"})) // 24
}
```

```go
package main

import (
	"fmt"
	"unsafe"
)

type W struct {
	b int32
	c int64
}

func main() {
	var w *W = new(W)
	//这时w的变量打印出来都是默认值0，0
	fmt.Println(w.b, w.c)

	//现在我们通过指针运算给b变量赋值为10
	b := unsafe.Pointer(uintptr(unsafe.Pointer(w)) + unsafe.Offsetof(w.b))
	*((*int)(b)) = 10
	//此时结果就变成了10，0
	fmt.Println(w.b, w.c)
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/go%E5%BA%93-unsafe/  

