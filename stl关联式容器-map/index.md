# STL关联式容器-map


> - 会根据元素键值自动被排序，所有元素都是 pair，同时拥有实值 value 和键值 key，其 pair 是 <key,pair>。键值不可改变，正值可以。
> - 新增删除操作后，之前的迭代器依然有效。
> - 以 RB-tree 为底层机制，和 set 一样也是调用即可。

### 1. 数据结构

```c++
#ifndef __SGI_STL_INTERNAL_MAP_H
#define __SGI_STL_INTERNAL_MAP_H

__STL_BEGIN_NAMESPACE

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1174
#endif

#ifndef __STL_LIMITED_DEFAULT_TEMPLATES
template <class Key, class T, class Compare = less<Key>, class Alloc = alloc>
#else
template <class Key, class T, class Compare, class Alloc = alloc>
#endif
class map {
public:

// typedefs:

  typedef Key key_type;
  typedef T data_type;
  typedef T mapped_type;
  typedef pair<const Key, T> value_type;
  typedef Compare key_compare;
    
  class value_compare
    : public binary_function<value_type, value_type, bool> {
  friend class map<Key, T, Compare, Alloc>;
  protected :
    Compare comp;
    value_compare(Compare c) : comp(c) {}
  public:
    bool operator()(const value_type& x, const value_type& y) const {
      return comp(x.first, y.first);
    }
  };

private:
  typedef rb_tree<key_type, value_type, 
                  select1st<value_type>, key_compare, Alloc> rep_type;
  rep_type t;  // red-black tree representing map
public:
  typedef typename rep_type::pointer pointer;
  typedef typename rep_type::const_pointer const_pointer;
  typedef typename rep_type::reference reference;
  typedef typename rep_type::const_reference const_reference;
  typedef typename rep_type::iterator iterator;
  typedef typename rep_type::const_iterator const_iterator;
  typedef typename rep_type::reverse_iterator reverse_iterator;
  typedef typename rep_type::const_reverse_iterator const_reverse_iterator;
  typedef typename rep_type::size_type size_type;
  typedef typename rep_type::difference_type difference_type;

  // allocation/deallocation

  map() : t(Compare()) {}
  explicit map(const Compare& comp) : t(comp) {}

#ifdef __STL_MEMBER_TEMPLATES
  template <class InputIterator>
  map(InputIterator first, InputIterator last)
    : t(Compare()) { t.insert_unique(first, last); }

  template <class InputIterator>
  map(InputIterator first, InputIterator last, const Compare& comp)
    : t(comp) { t.insert_unique(first, last); }
#else
  map(const value_type* first, const value_type* last)
    : t(Compare()) { t.insert_unique(first, last); }
  map(const value_type* first, const value_type* last, const Compare& comp)
    : t(comp) { t.insert_unique(first, last); }

  map(const_iterator first, const_iterator last)
    : t(Compare()) { t.insert_unique(first, last); }
  map(const_iterator first, const_iterator last, const Compare& comp)
    : t(comp) { t.insert_unique(first, last); }
#endif /* __STL_MEMBER_TEMPLATES */

  map(const map<Key, T, Compare, Alloc>& x) : t(x.t) {}
  map<Key, T, Compare, Alloc>& operator=(const map<Key, T, Compare, Alloc>& x)
  {
    t = x.t;
    return *this; 
  }

  // accessors:

  key_compare key_comp() const { return t.key_comp(); }
  value_compare value_comp() const { return value_compare(t.key_comp()); }
  iterator begin() { return t.begin(); }
  const_iterator begin() const { return t.begin(); }
  iterator end() { return t.end(); }
  const_iterator end() const { return t.end(); }
  reverse_iterator rbegin() { return t.rbegin(); }
  const_reverse_iterator rbegin() const { return t.rbegin(); }
  reverse_iterator rend() { return t.rend(); }
  const_reverse_iterator rend() const { return t.rend(); }
  bool empty() const { return t.empty(); }
  size_type size() const { return t.size(); }
  size_type max_size() const { return t.max_size(); }
  T& operator[](const key_type& k) {
    return (*((insert(value_type(k, T()))).first)).second;
  }
  void swap(map<Key, T, Compare, Alloc>& x) { t.swap(x.t); }

  // insert/erase

  pair<iterator,bool> insert(const value_type& x) { return t.insert_unique(x); }
  iterator insert(iterator position, const value_type& x) {
    return t.insert_unique(position, x);
  }
#ifdef __STL_MEMBER_TEMPLATES
  template <class InputIterator>
  void insert(InputIterator first, InputIterator last) {
    t.insert_unique(first, last);
  }
#else
  void insert(const value_type* first, const value_type* last) {
    t.insert_unique(first, last);
  }
  void insert(const_iterator first, const_iterator last) {
    t.insert_unique(first, last);
  }
#endif /* __STL_MEMBER_TEMPLATES */

  void erase(iterator position) { t.erase(position); }
  size_type erase(const key_type& x) { return t.erase(x); }
  void erase(iterator first, iterator last) { t.erase(first, last); }
  void clear() { t.clear(); }

  // map operations:

  iterator find(const key_type& x) { return t.find(x); }
  const_iterator find(const key_type& x) const { return t.find(x); }
  size_type count(const key_type& x) const { return t.count(x); }
  iterator lower_bound(const key_type& x) {return t.lower_bound(x); }
  const_iterator lower_bound(const key_type& x) const {
    return t.lower_bound(x); 
  }
  iterator upper_bound(const key_type& x) {return t.upper_bound(x); }
  const_iterator upper_bound(const key_type& x) const {
    return t.upper_bound(x); 
  }
  
  pair<iterator,iterator> equal_range(const key_type& x) {
    return t.equal_range(x);
  }
  pair<const_iterator,const_iterator> equal_range(const key_type& x) const {
    return t.equal_range(x);
  }
  friend bool operator== __STL_NULL_TMPL_ARGS (const map&, const map&);
  friend bool operator< __STL_NULL_TMPL_ARGS (const map&, const map&);
};

template <class Key, class T, class Compare, class Alloc>
inline bool operator==(const map<Key, T, Compare, Alloc>& x, 
                       const map<Key, T, Compare, Alloc>& y) {
  return x.t == y.t;
}

template <class Key, class T, class Compare, class Alloc>
inline bool operator<(const map<Key, T, Compare, Alloc>& x, 
                      const map<Key, T, Compare, Alloc>& y) {
  return x.t < y.t;
}

#ifdef __STL_FUNCTION_TMPL_PARTIAL_ORDER

template <class Key, class T, class Compare, class Alloc>
inline void swap(map<Key, T, Compare, Alloc>& x, 
                 map<Key, T, Compare, Alloc>& y) {
  x.swap(y);
}

#endif /* __STL_FUNCTION_TMPL_PARTIAL_ORDER */

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1174
#endif

__STL_END_NAMESPACE

#endif /* __SGI_STL_INTERNAL_MAP_H */
```

### 2. 使用demo

#### .1. pair

> pair<string, int>p1; 和 make_pair<string, ing>(); 使用方法

```c++
#include <iostream>
#include <string>
#include <vector>
using namespace std;
//将pair放入容器&initpair
int main(int argc, const char *argv[])
{
    vector<pair<string, int> > vec;//初始化vector

    pair<string, int> p1;//num.1
    p1.first = "hello";
    p1.second = 12 ;
    vec.push_back(p1);

    pair<string, int> p2("world", 22);//num.2
    vec.push_back(p2);

    vec.push_back(make_pair<string, int>("foo", 44));//num.3

    for(vector<pair<string,int> >::iterator it = vec.begin(); //g++ main.cc -std=c++0x
        it != vec.end();++it)
    {
        cout << "key: " << it->first << "  val:"<< it->second << endl;    
    }

    return 0;
}
```

#### .2. map

> - count：仅仅能得出该元素是否存在。
> - find： 能够返回该元素的迭代器 。

```c++
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;
//Red_black Tree
int main(int argc, const char *argv[])
{
    map<string, int> m;
    m["beijing"] = 2000;
    m["hangzhou"] = 880;
    m["shanghai"] = 1500;
    // key --> value  采用二叉排序树对key排序
    //两种打印方式
    for(map<string, int>::const_iterator it = m.begin();
        it != m.end();it++)
    {
        cout << it->first <<":" << it->second << endl;
    }
    for(const pair<int, string> &p: m) // -std=c++0x(C++11) 注意要用pair 
        cout << p.first << ":" << p.second << endl;
    return 0;
}
```

### 3. multimap

> - 与 map 基本相同，除 insert_unique 和 insert_equal 的使用不同外基本一致

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%85%B3%E8%81%94%E5%BC%8F%E5%AE%B9%E5%99%A8-map/  

