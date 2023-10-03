# STL配接器


> 配接器（Adapter）在 STL 组件的灵活组合运用功能上，**扮演着轴承、转换器的角色**，即`将一个 class 的接口转换为另一个 class 的接口`，使原本因接口不兼容而不能合作的 classes，可以一起运作，它事实上**是一种设计模式**。
>
> STL 主要提供如下三种配接器：
>
> （1）改变`仿函数（functors）接口`，称之为 function adapter
>
> （2）改变`容器（containers）接口`，称之位 container adapter
>
> （3）改变`迭代器（iterators）接口者`，称之为 iterator adapter

### 1. container adapter

> STL 提供的两个容器 queue 和 stack，其实都不过是一种配接器，是对 deque （双端队列）接口的修饰而成就自己的容器风貌。queue 和 stack 底层都是由 deque 构成的，它们封住所有 deque 对外接口，只开发符合对应原则的几个函数，故它们是适配器，是一个作用于容器之上的适配器。

### 2. iterator adapter. 

> STL 提供了许多应用于迭代器身上的配接器，包括 **insert iterators，reverse iterators，iostream iterators**。
>
> - insert iterators 可以将一般`迭代器的赋值操作转变为插入操作`。此迭代器包括专门`从尾端插入操作 back_insert_iterator`，专门`从头端插入操作 front_insert_iterator`，以及可`从任何位置执行插入操作的 insert_iterator`。因 iterator adapters 使用接口不是十分直观，STL 提供三个相应的函数 `back_inserter ()、front_inserter ()、inserter ()`，从而提高使用时的便利性。
>
> - reverse iterators 可以将一般`迭代器的行进方向逆转`，`使原本应该前进的 operator++ 变成了后退操作`，使原本应该后退的 operator–变成了前进操作。此操作用在 “从尾端开始进行” 的算法上，有很大的方便性。
>
> - iostream iterators 可以将迭代器绑定到某个 iostream 对象身上。`绑定到 istream 对象身上，为 istream_iterator，拥有输入功能`；`绑定到 ostream 对象身上，成为 ostream_iterator，拥有输出功能`。此迭代器用在屏幕输出上，非常方便。

#### .1. reverse iterators

```c++
template <class Iterator>
class reverse_iterator
{
protected:
  Iterator current;
public:
  // 反向迭代器的5种相应型别和正向的正向迭代器相同
  typedef typename iterator_traits<Iterator>::iterator_category iterator_category;
  typedef typename iterator_traits<Iterator>::value_type value_type;
  typedef typename iterator_traits<Iterator>::difference_type difference_type;
  typedef typename iterator_traits<Iterator>::pointer pointer;
  typedef typename iterator_traits<Iterator>::reference reference;

  // 正向迭代器
  typedef Iterator iterator_type;
  // 反向迭代器
  typedef reverse_iterator<Iterator> self;

public:
  // 构造函数
  reverse_iterator() {}
  // 将反向迭代器与某种迭代器联系起来
  explicit reverse_iterator(iterator_type x) : current(x) {}
  reverse_iterator(const self& x) : current(x.current) {}

  // 返回正向迭代器      
  iterator_type base() const { return current; }
  // 反向迭代器取值时,先将正向迭代器后退一位,再取值
  reference operator*() const {
    Iterator tmp = current;
    return *--tmp;
  }
  // 反转后end变成了begin. begin变成了end
  // 注意一点就是end是指向最后一个元素的后一个位置
  self& operator++() {
    --current;
    return *this;
  }
  self operator++(int) {
    self tmp = *this;
    --current;
    return tmp;
  }
  self& operator--() {
    ++current;
    return *this;
  }
  self operator--(int) {
    self tmp = *this;
    ++current;
    return tmp;
  }

  // 前进和后退方向相反
  self operator+(difference_type n) const {
    return self(current - n);
  }
  self& operator+=(difference_type n) {
    current -= n;
    return *this;
  }
  self operator-(difference_type n) const {
    return self(current + n);
  }
  self& operator-=(difference_type n) {
    current += n;
    return *this;
  }
  reference operator[](difference_type n) const { return *(*this + n); }  
};
```

### 3.  functor adapter

> functor adapters 是所有配接器中数量最庞大的一个族群，其配接灵活度是后两者不能及的，可以配接、配接、再配接。其中`配接操作包括系结（bind）、否定（negate）、组合（compose）、以及对一般函数或成员函数的修饰（使其成为一个仿函数）`。它的价值在于，`通过它们之间的绑定、组合、修饰能力，几乎可以无限制地创造出各种可能的表达式（expression），搭配 STL 算法一起演出`。
>
> 由于仿函数就是 “将 function call 操作符重载” 的一种 class，而任何算法接受一个仿函数时，总是在其演算过程中调用该仿函数的 operator ()，这使得不具备仿函数之形、却有真函数之实的 “一般函数” 和 “成员函数（member functions）感到为难。**如果” 一般函数 “和 “成员函数” 不能纳入复用的体系中，则 STL 的规划将崩落了一角**。为此，STL 提供了为数众多的配接器，`使 “一般函数” 和 “成员函数” 得以无缝地与其他配接器或算法结合起来。`
>
> 所有期望获取配接能力的组件，本身都必须是可配接的，即一元仿函数必须继承自 unary_function，二元仿函数必须继承自 binary_function，成员函数必须以 mem_fun 处理过，一般函数必须以 ptr_fun 处理过。**一个未经 ptr_fun 处理过的一般函数，虽然也可以函数指针的形式传给 STL 算法使用，却无法拥有任何配接能力**。

```c++
// 以下配接器其实就是把一个一元函数指针包起来；
// 当仿函数被使用时，就调用该函数指针
template <class _Arg, class _Result>
class pointer_to_unary_function : public unary_function<_Arg, _Result> 
{
protected:
  _Result (*_M_ptr)(_Arg);     // 内部成员，一个函数指针
public:
  pointer_to_unary_function() {}
  // 以下constructor将函数指针记录于内部成员之中
  explicit pointer_to_unary_function(_Result (*__x)(_Arg)) : _M_ptr(__x) {}
  // 通过函数指针执行函数
  _Result operator()(_Arg __x) const { return _M_ptr(__x); }
};

// 辅助函数，使我们能够方便运用pointer_to_unary_function
template <class _Arg, class _Result>
inline pointer_to_unary_function<_Arg, _Result> ptr_fun(_Result (*__x)(_Arg))
{
  return pointer_to_unary_function<_Arg, _Result>(__x);
}
```

### Resource

- https://wendeng.github.io/2019/05/22/c++%E5%9F%BA%E7%A1%80/%E3%80%8ASTL%E6%BA%90%E7%A0%81%E5%89%96%E6%9E%90%E3%80%8B%E7%AC%AC7%E7%AB%A0%20%E4%BB%BF%E5%87%BD%E6%95%B0%EF%BC%88%E5%87%BD%E6%95%B0%E5%AF%B9%E8%B1%A1%EF%BC%89%E5%92%8C%E9%85%8D%E6%8E%A5%E5%99%A8/
- https://github.com/FunctionDou/STL

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E9%85%8D%E6%8E%A5%E5%99%A8/  

