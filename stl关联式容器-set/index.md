# STL关联式容器-set


> - 所有元素根据`键值自动排序，键值就是实值,实值就是键值，不允许重复键值`。不可通过迭代器改变 set 元素值，是一种 constant iterators
> - 与 list 相同，当客户端对它进行元素新增操作 insert 或删除 erase 时，操作之前所有迭代器仍然有效。
> - `以 RB-tree 为底层机制。另一种用 hash-table 为底层机制的 set 为 hash_set`
> - 企图通过迭代器来改变set中的元素时不允许的

### 1. 数据结构

```c++
#ifndef __SGI_STL_INTERNAL_SET_H
#define __SGI_STL_INTERNAL_SET_H

__STL_BEGIN_NAMESPACE

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1174
#endif

#ifndef __STL_LIMITED_DEFAULT_TEMPLATES
template <class Key, class Compare = less<Key>, class Alloc = alloc>//缺省情况下用递增排序
#else
template <class Key, class Compare, class Alloc = alloc>
#endif
class set {
public:
  // typedefs:

  typedef Key key_type;
  typedef Key value_type;
  typedef Compare key_compare;
  typedef Compare value_compare;//用了同一个比较函数！
private://红黑树表现set
  typedef rb_tree<key_type, value_type, 
                  identity<value_type>, key_compare, Alloc> rep_type;
  rep_type t;  // red-black tree representing set
public:
  typedef typename rep_type::const_pointer pointer;
  typedef typename rep_type::const_pointer const_pointer;
  typedef typename rep_type::const_reference reference;
  typedef typename rep_type::const_reference const_reference;
  typedef typename rep_type::const_iterator iterator;
  //注意是const_iterator，表示这里的迭代器没有写入操作，因为set有次序安排
  typedef typename rep_type::const_iterator const_iterator;
  typedef typename rep_type::const_reverse_iterator reverse_iterator;
  typedef typename rep_type::const_reverse_iterator const_reverse_iterator;
  typedef typename rep_type::size_type size_type;
  typedef typename rep_type::difference_type difference_type;

  // allocation/deallocation
  set() : t(Compare()) {}
  explicit set(const Compare& comp) : t(comp) {}

#ifdef __STL_MEMBER_TEMPLATES
   // insert_unique()!multiset才允许同键值
  template <class InputIterator>
  set(InputIterator first, InputIterator last)
    : t(Compare()) { t.insert_unique(first, last); }

  template <class InputIterator>
  set(InputIterator first, InputIterator last, const Compare& comp)
    : t(comp) { t.insert_unique(first, last); }
#else
  set(const value_type* first, const value_type* last) 
    : t(Compare()) { t.insert_unique(first, last); }
  set(const value_type* first, const value_type* last, const Compare& comp)
    : t(comp) { t.insert_unique(first, last); }

  set(const_iterator first, const_iterator last)
    : t(Compare()) { t.insert_unique(first, last); }
  set(const_iterator first, const_iterator last, const Compare& comp)
    : t(comp) { t.insert_unique(first, last); }
#endif /* __STL_MEMBER_TEMPLATES */

  set(const set<Key, Compare, Alloc>& x) : t(x.t) {}
  set<Key, Compare, Alloc>& operator=(const set<Key, Compare, Alloc>& x) { 
    t = x.t; 
    return *this;
  }

  // accessors:
 //全部直接调用红黑树已实现的
  key_compare key_comp() const { return t.key_comp(); }
  value_compare value_comp() const { return t.key_comp(); }
  iterator begin() const { return t.begin(); }
  iterator end() const { return t.end(); }
  reverse_iterator rbegin() const { return t.rbegin(); } 
  reverse_iterator rend() const { return t.rend(); }
  bool empty() const { return t.empty(); }
  size_type size() const { return t.size(); }
  size_type max_size() const { return t.max_size(); }
  void swap(set<Key, Compare, Alloc>& x) { t.swap(x.t); }

  // insert/erase
  typedef  pair<iterator, bool> pair_iterator_bool; 
  pair<iterator,bool> insert(const value_type& x) { 
    pair<typename rep_type::iterator, bool> p = t.insert_unique(x); 
    return pair<iterator, bool>(p.first, p.second);
  }
  iterator insert(iterator position, const value_type& x) {
    typedef typename rep_type::iterator rep_iterator;
    return t.insert_unique((rep_iterator&)position, x);
  }
#ifdef __STL_MEMBER_TEMPLATES
  template <class InputIterator>
  void insert(InputIterator first, InputIterator last) {
    t.insert_unique(first, last);
  }
#else
  void insert(const_iterator first, const_iterator last) {
    t.insert_unique(first, last);
  }
  void insert(const value_type* first, const value_type* last) {
    t.insert_unique(first, last);
  }
#endif /* __STL_MEMBER_TEMPLATES */
  void erase(iterator position) { 
    typedef typename rep_type::iterator rep_iterator;
    t.erase((rep_iterator&)position); 
  }
  size_type erase(const key_type& x) { 
    return t.erase(x); 
  }
  void erase(iterator first, iterator last) { 
    typedef typename rep_type::iterator rep_iterator;
    t.erase((rep_iterator&)first, (rep_iterator&)last); 
  }
  void clear() { t.clear(); }

  // set operations:

  iterator find(const key_type& x) const { return t.find(x); }
  size_type count(const key_type& x) const { return t.count(x); }
  iterator lower_bound(const key_type& x) const {
    return t.lower_bound(x);
  }
  iterator upper_bound(const key_type& x) const {
    return t.upper_bound(x); 
  }
  pair<iterator,iterator> equal_range(const key_type& x) const {
    return t.equal_range(x);
  }
  friend bool operator== __STL_NULL_TMPL_ARGS (const set&, const set&);
  friend bool operator< __STL_NULL_TMPL_ARGS (const set&, const set&);
};

template <class Key, class Compare, class Alloc>
inline bool operator==(const set<Key, Compare, Alloc>& x, 
                       const set<Key, Compare, Alloc>& y) {
  return x.t == y.t;
}

template <class Key, class Compare, class Alloc>
inline bool operator<(const set<Key, Compare, Alloc>& x, 
                      const set<Key, Compare, Alloc>& y) {
  return x.t < y.t;
}

#ifdef __STL_FUNCTION_TMPL_PARTIAL_ORDER

template <class Key, class Compare, class Alloc>
inline void swap(set<Key, Compare, Alloc>& x, 
                 set<Key, Compare, Alloc>& y) {
  x.swap(y);
}

#endif /* __STL_FUNCTION_TMPL_PARTIAL_ORDER */

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1174
#endif

__STL_END_NAMESPACE

#endif /* __SGI_STL_INTERNAL_SET_H */
```

### 2. 修改set中元素值

> 元素在 `std::set` 中构造后，如果需要查找，则调用 `find`[成员函数](https://so.csdn.net/so/search?q=成员函数&spm=1001.2101.3001.7020)，但是该方式有一个致命的缺陷，就是返回的是一个常指针，无法通过指针更改元素的值。
>
> 把结构中可能需要更改的元素使用智能指针进行保存，利用 find 函数找到结构的索引，再通过索引获取指针进行操作。

> 但是，当我在 set 里面存的是 shared_ptr 元素时，根本无所谓有没有序。我就是要通过迭代器获取元素的非 const 引用。 这个不理解·`？？todo`

```c++
#include <iostream>
#include <set>
template<class T>
inline T & GetStdSetElement(std::_Rb_tree_const_iterator<T>  std_set_iterator)
{
    return *(T *)&(*std_set_iterator);
}
int main()
{    
    using namespace std;
    set<int> iset;
    pair< set<int>::iterator, bool> res = iset.insert(4);
    int & i = GetStdSetElement(res.first);
    i++;
    cout << *( iset.begin() ) << endl;
    return 0;
}
```

```c++
#include <iostream>
#include <string>
#include <mutex>
#include <set>
#include <memory>
#include <string>
#include <utility>

struct Object {
    int fd;
    std::shared_ptr<std::mutex> mtx;
    std::shared_ptr<std::string> msg;
    Object(int _fd) {
        fd = _fd;
        mtx = std::make_shared<std::mutex>();
        msg = std::make_shared<std::string>();
    }
    bool operator<(const Object& obj)const {
        return fd < obj.fd;
    }
    bool operator==(const Object& obj)const {
        return fd == obj.fd;
    }
};
int main() {
    std::set<Object> objSet;
    objSet.emplace(Object(1));
    auto it = objSet.find(Object(1));
    auto p = it->msg;  // 这里获取指针，就可以直接操作了
    *p += "hello world !";
    std::cout << *(it->msg) << std::endl;
    return 0;
}
```

### 3. 常用函数

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/693265793dc17bbcf9ec15d8ce097ac4.png)

### 4. multiset

> - 与 set 基本相同，除 insert_unique 和 insert_equal 的使用不同外基本一致

### Resource

- https://www.programminghunter.com/article/86332276862/


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%85%B3%E8%81%94%E5%BC%8F%E5%AE%B9%E5%99%A8-set/  

