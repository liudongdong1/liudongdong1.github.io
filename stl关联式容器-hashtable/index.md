# STL关联式容器-hashtable


### 1. 哈希冲突

#### .1. 线性探测

- 根据元素的值然后除以数组大小，然后插入指定的位置

#### .2. 二次探测

- F(i)=i*i; 如果新元素起始插入位置为H，但是H已经被占用，则会尝试H+i^2; i=[1,n];

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220412131138470.png)

#### .3. 开链（链地址法）

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220412131821725.png)

### 2. 数据结构

#### .1. hash_table节点

```c++
template <class _Val>
struct _Hashtable_node
{
  _Hashtable_node* _M_next;
  _Val _M_val;
};
```

#### .2. 迭代器

> - 迭代器没有后退操作（operator–），也没有定义所谓的你想迭代器（reverse iterator）
> - 其前进操作时首先尝试从目前所知的节点出发，前进一个位置（节点），由于节点被安置于 list 内，所以利用节点的 next 指针即可轻易达到进行的目的
> - 如果目前节点正巧是 list 的尾端，就跳到下一个 bucket 内，跳过之后指向下一个 list 的头节点

```c++
template <class _Val, class _Key, class _HashFcn,
class _ExtractKey, class _EqualKey, class _Alloc>
    struct _Hashtable_iterator {
        typedef hashtable<_Val,_Key,_HashFcn,_ExtractKey,_EqualKey,_Alloc>
            _Hashtable;
        typedef _Hashtable_iterator<_Val, _Key, _HashFcn, 
        _ExtractKey, _EqualKey, _Alloc>
            iterator;
        typedef _Hashtable_const_iterator<_Val, _Key, _HashFcn, 
        _ExtractKey, _EqualKey, _Alloc>
            const_iterator;
        typedef _Hashtable_node<_Val> _Node;

        typedef forward_iterator_tag iterator_category;
        typedef _Val value_type;
        typedef ptrdiff_t difference_type;
        typedef size_t size_type;
        typedef _Val& reference;
        typedef _Val* pointer;

        _Node* _M_cur; //迭代器目前所指节点
        _Hashtable* _M_ht;//保持对容器的连结关系（因为可能需要从bucket跳到bucket）

        _Hashtable_iterator(_Node* __n, _Hashtable* __tab) 
            : _M_cur(__n), _M_ht(__tab) {}
        _Hashtable_iterator() {}
        reference operator*() const { return _M_cur->_M_val; }
        #ifndef __SGI_STL_NO_ARROW_OPERATOR
        pointer operator->() const { return &(operator*()); }
        #endif /* __SGI_STL_NO_ARROW_OPERATOR */
        iterator& operator++();
        iterator operator++(int);
        bool operator==(const iterator& __it) const
        { return _M_cur == __it._M_cur; }
        bool operator!=(const iterator& __it) const
        { return _M_cur != __it._M_cur; }
    };
```

```c++
template <class _Val, class _Key, class _HF, class _ExK, class _EqK, 
class _All>
    _Hashtable_iterator<_Val,_Key,_HF,_ExK,_EqK,_All>&
    _Hashtable_iterator<_Val,_Key,_HF,_ExK,_EqK,_All>::operator++()
    {
        const _Node* __old = _M_cur;
        _M_cur = _M_cur->_M_next; //如果存在，就是它，否则进入if
        if (!_M_cur) {
            //根据元素值，定位出下一个bucket。其起头处就是我们的目的地
            size_type __bucket = _M_ht->_M_bkt_num(__old->_M_val);
            while (!_M_cur && ++__bucket < _M_ht->_M_buckets.size())//注意operator++
                _M_cur = _M_ht->_M_buckets[__bucket];
        }
        return *this;
    }

template <class _Val, class _Key, class _HF, class _ExK, class _EqK, 
class _All>
    inline _Hashtable_iterator<_Val,_Key,_HF,_ExK,_EqK,_All>
    _Hashtable_iterator<_Val,_Key,_HF,_ExK,_EqK,_All>::operator++(int)
    {
        iterator __tmp = *this;
        ++*this; //调用operator++()
        return __tmp;
    }
```

#### .3. hashtable结构

> - _Val：节点的实值类型
> - _Key：节点的键值类型
> - _HF：hash function 的函数类型
> - _Ex：从节点取出键值的方法（函数或仿函数）
> - _Eq：判断键值相同与否的方法（函数或仿函数）
> - _All：空间配置器。缺省使用 std::alloc

```c++
template <class Value, class Key, class HashFcn,
          class ExtractKey, class EqualKey,
          class Alloc>
class hashtable {
public:
  typedef Key key_type;
  typedef Value value_type;
  typedef HashFcn hasher;
  typedef EqualKey key_equal;

  typedef size_t            size_type;
  typedef ptrdiff_t         difference_type;
  typedef value_type*       pointer;
  typedef const value_type* const_pointer;
  typedef value_type&       reference;
  typedef const value_type& const_reference;

  hasher hash_funct() const { return hash; }
  key_equal key_eq() const { return equals; }

private:
  hasher hash;
  key_equal equals;
  ExtractKey get_key;

  typedef __hashtable_node<Value> node;
  typedef simple_alloc<node, Alloc> node_allocator;

  vector<node*,Alloc> buckets;
  size_type num_elements;

public:
  typedef __hashtable_iterator<Value, Key, HashFcn, ExtractKey, EqualKey, 
                               Alloc>
  iterator;

  typedef __hashtable_const_iterator<Value, Key, HashFcn, ExtractKey, EqualKey,
                                     Alloc>
  const_iterator;

  friend struct
  __hashtable_iterator<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc>;
  friend struct
  __hashtable_const_iterator<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc>;

public:
  hashtable(size_type n,
            const HashFcn&    hf,
            const EqualKey&   eql,
            const ExtractKey& ext)
    : hash(hf), equals(eql), get_key(ext), num_elements(0)
  {
    initialize_buckets(n);
  }

  hashtable(size_type n,
            const HashFcn&    hf,
            const EqualKey&   eql)
    : hash(hf), equals(eql), get_key(ExtractKey()), num_elements(0)
  {
    initialize_buckets(n);
  }

  hashtable(const hashtable& ht)
    : hash(ht.hash), equals(ht.equals), get_key(ht.get_key), num_elements(0)
  {
    copy_from(ht);
  }

  hashtable& operator= (const hashtable& ht)
  {
    if (&ht != this) {
      clear();
      hash = ht.hash;
      equals = ht.equals;
      get_key = ht.get_key;
      copy_from(ht);
    }
    return *this;
  }
    ...
}
```

### 3. 内存管理

```c++
_Node* _M_new_node(const value_type& __obj)
{
    _Node* __n = _M_get_node();
    __n->_M_next = 0;
    __STL_TRY {
      construct(&__n->_M_val, __obj);
      return __n;
    }
    __STL_UNWIND(_M_put_node(__n));
}
 
void _M_delete_node(_Node* __n)
{
    destroy(&__n->_M_val);
    _M_put_node(__n);
}
 
void _M_put_node(_Node* __p) { _M_node_allocator.deallocate(__p, 1); }
```

```c++
template <class _Val, class _Key, class _HashFcn,
          class _ExtractKey, class _EqualKey, class _Alloc>
class hashtable {
public:
  hashtable(size_type __n,
            const _HashFcn&    __hf,
            const _EqualKey&   __eql,
            const _ExtractKey& __ext,
            const allocator_type& __a = allocator_type())
    : __HASH_ALLOC_INIT(__a)
      _M_hash(__hf),
      _M_equals(__eql),
      _M_get_key(__ext),
      _M_buckets(__a),
      _M_num_elements(0)
  {
    _M_initialize_buckets(__n);
  }
private:
    void _M_initialize_buckets(size_type __n)
    {
      const size_type __n_buckets = _M_next_size(__n);//调用_M_next_size
      //举例：传入50，返回53.以下首先保留53个元素空间，然后将其全部填0
      _M_buckets.reserve(__n_buckets);
      _M_buckets.insert(_M_buckets.end(), __n_buckets, (_Node*) 0);
      _M_num_elements = 0;
   }
 
   //该函数返回最近接n并大于n的质数。其中调用了我们上面介绍的__stl_next_prime函数
   size_type _M_next_size(size_type __n) const
       { return __stl_next_prime(__n); }
};
```

### 4. 插入操作

#### .1. **insert_unique()**

```c++
pair<iterator, bool> insert_unique(const value_type& __obj)
{
    resize(_M_num_elements + 1);          //判断是否需要重建表格
    return insert_unique_noresize(__obj); //在不需要重建表格的情况下插入节点，键值不允许重复
}
```

```c++
template <class V, class K, class HF, class Ex, class Eq, class A>
void hashtable<V, K, HF, Ex, Eq, A>::resize(size_type num_elements_hint)
{
    const size_type old_n = buckets.size();
    if (num_elements_hint > old_n) {
        const size_type n = next_size(num_elements_hint);
        if (n > old_n) {
            vector<node*, A> tmp(n, (node*) 0);  //设立新的buckets
            __STL_TRY {
                //处理每一个旧的buckets   todo这里还是没有看懂
                for (size_type bucket = 0; bucket < old_n; ++bucket) {
                    node* first = buckets[bucket];   //指向节点所对应之串行的起始节点
                    //处理每一个就bucket中所含的每一个节点
                    while (first) {
                        //找出节点落在哪一个新bucket内
                        size_type new_bucket = bkt_num(first->val, n);
                        //令旧bucket指向其所对应串行的下一个节点
                        buckets[bucket] = first->next;
                        //将当前节点插入到新bucket内，使其成为对应串行的第一个节点
                        first->next = tmp[new_bucket];
                        tmp[new_bucket] = first;
                        //回到旧bucket所指的待处理行，准备处理下一个节点
                        first = buckets[bucket];          
                    }
                }
                buckets.swap(tmp);
            }
            #         ifdef __STL_USE_EXCEPTIONS
            catch(...) {
                for (size_type bucket = 0; bucket < tmp.size(); ++bucket) {
                    while (tmp[bucket]) {
                        node* next = tmp[bucket]->next;
                        delete_node(tmp[bucket]);
                        tmp[bucket] = next;
                    }
                }
                throw;
            }
            #         endif /* __STL_USE_EXCEPTIONS */
        }
    }
}
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220412135623759.png)

#### .2. insert_unique_noresize

```c++
template <class V, class K, class HF, class Ex, class Eq, class A>
pair<typename hashtable<V, K, HF, Ex, Eq, A>::iterator, bool> 
hashtable<V, K, HF, Ex, Eq, A>::insert_unique_noresize(const value_type& obj)
{
  const size_type n = bkt_num(obj); //决定obj应位于 #n bucket
  node* first = buckets[n];   //令first指向bucket对应的串行头部

  for (node* cur = first; cur; cur = cur->next) 
    if (equals(get_key(cur->val), get_key(obj)))
        //遍历bucket所对应的链表，如果发现与链表中某个键值相同，就不插入，立刻返回
      return pair<iterator, bool>(iterator(cur, this), false);

  node* tmp = new_node(obj);
  tmp->next = first;  //令新节点成为链表中的第一个节点
  buckets[n] = tmp;
  ++num_elements;      //节点数加一
  return pair<iterator, bool>(iterator(tmp, this), true);
}
```

#### .3. insert_equal

```c++
iterator insert_equal(const value_type& obj)
{
    resize(num_elements + 1);
    return insert_equal_noresize(obj);
}
```

#### .4. insert_equal_noresize

```c++
template <class V, class K, class HF, class Ex, class Eq, class A>
typename hashtable<V, K, HF, Ex, Eq, A>::iterator 
hashtable<V, K, HF, Ex, Eq, A>::insert_equal_noresize(const value_type& obj)
{
  const size_type n = bkt_num(obj);
  node* first = buckets[n];

  for (node* cur = first; cur; cur = cur->next) 
    if (equals(get_key(cur->val), get_key(obj))) {
        //遍历bucket所对应的整个链表，如果发现与链表中的某键值相同，将新节点插入当前位置之后，返回指向新节点的iterator
      node* tmp = new_node(obj);
      tmp->next = cur->next;
      cur->next = tmp;
      ++num_elements;
      return iterator(tmp, this);
    }

  node* tmp = new_node(obj);
  tmp->next = first;
  buckets[n] = tmp;
  ++num_elements;
  return iterator(tmp, this);
}
```

### 5. hash function

> - 插入元素之后需要知道某个元素落脚于哪一个 bucket 内。这本来是哈希函数的责任，但是 SGI 把这个任务包装了一层，先交给 bkt_num () 函数，再由此函数调用哈希函数，取得一个可以执行 modulus（取模）运算的数值为什么要这么做？因为有些函数类型无法直接拿来对哈表表的大小进行模运算，例如字符串，这时候我们需要做一些转换
> - hash functions 是计算元素位置的函数，SGI 将这项任务赋予了先前提到过的 bkt_num () 函数，再由 bkt_num () 函数调用这些 hash function，取得一个可以对 hashtable 进行模运算的值，针对 char、int、long 等整数类型，这里大部分的 hash function 什么都没有做，只是直接返回原值。具体见<stl_hash_fun.h>
> - hashtable **无法处理上述所列各项类型之外的元素**。例如 string、double、float 等，这些类型**用户必须自己定义 hash function。**

```c++
//版本1：接受实值（value）和buckets个数
size_type _M_bkt_num(const value_type& __obj, size_t __n) const
{
    return _M_bkt_num_key(_M_get_key(__obj), __n); //调用版本4
}
 
//版本2：只接受实值（value）
size_type _M_bkt_num(const value_type& __obj) const
{
    return _M_bkt_num_key(_M_get_key(__obj)); //调用版本3
}
 
//版本3，只接受键值
size_type _M_bkt_num_key(const key_type& __key) const
{
    return _M_bkt_num_key(__key, _M_buckets.size()); //调用版本4
}
 
//版本4：接受键值和buckets个数
size_type _M_bkt_num_key(const key_type& __key, size_t __n) const
{
    return _M_hash(__key) % __n; //SGI的所有内建的hash()，在后面的hash functions中介绍
}
```

### 6. copy_from&clear

```c++
template <class V, class K, class HF, class Ex, class Eq, class A>
void hashtable<V, K, HF, Ex, Eq, A>::clear()
{
  for (size_type i = 0; i < buckets.size(); ++i) {
    node* cur = buckets[i];
    //删除桶中每一个元素
    while (cur != 0) {
      node* next = cur->next;
      delete_node(cur);
      cur = next;
    }
    buckets[i] = 0;  //令桶的内容为null指针
  }
  num_elements = 0;   //总结点个数为0
}
```

```c++
template <class V, class K, class HF, class Ex, class Eq, class A>
void hashtable<V, K, HF, Ex, Eq, A>::copy_from(const hashtable& ht)
{
  buckets.clear();
  //如果大于ht的空间就不动，否则增大自己空间
  buckets.reserve(ht.buckets.size());
  //此时，buckets vector为空，所以此处的尾端及开头
  buckets.insert(buckets.end(), ht.buckets.size(), (node*) 0);
  __STL_TRY {
    for (size_type i = 0; i < ht.buckets.size(); ++i) {
        //复制vector的每一个元素
      if (const node* cur = ht.buckets[i]) {
        node* copy = new_node(cur->val);
        buckets[i] = copy;
		//复制bucket list中每一个节点
        for (node* next = cur->next; next; cur = next, next = cur->next) {
          copy->next = new_node(next->val);
          copy = copy->next;
        }
      }
    }
    num_elements = ht.num_elements;
  }
  __STL_UNWIND(clear());
}
```

### 7. find & count

```c++
const_iterator find(const key_type& key) const
{
    size_type n = bkt_num_key(key);
    const node* first;
    for ( first = buckets[n];
         first && !equals(get_key(first->val), key);
         first = first->next)
    {}
    return const_iterator(first, this);
} 

size_type count(const key_type& key) const
{
    const size_type n = bkt_num_key(key);
    size_type result = 0;

    for (const node* cur = buckets[n]; cur; cur = cur->next)
        if (equals(get_key(cur->val), key))
            ++result;
    return result;
}
```

### 8. demo

```c++
#include <hash_set> //会包含<stl_hashtable.h>
#include <iostream>
using namespace std;
 
int main()
{
	hashtable<int, int, hash<int>, identity<int>, equal_to<int>, alloc>
		iht(50, hash<int>(), equal_to<int>());
 
	std::cout << iht.size() << std::endl; //0
	std::cout << iht.bucket_count()() << std::endl; //53。这是STL供应的第一个质数
	std::cout << iht.max_bucket_count() << std::endl; //4294967291，这是STL供应的最后一个质数
 
	iht.insert_unique(59);
	iht.insert_unique(63);
	iht.insert_unique(108);
	iht.insert_unique(2);
	iht.insert_unique(53);
	iht.insert_unique(55);
	std::cout << iht.size() << std::endl; //6。这个就是hashtable<T>::num_element
	
	
	//下面声明一个hashtable迭代器
	hashtable<int, int, hash<int>, identity<int>, equal_to<int>, alloc>::iterator
		ite = iht.begin();
		
	//遍历hashtable
	for (int i = 0; i < iht.size(); ++i, ++ite)
		std::cout << *ite << std::endl; //53 55 2 108 59 53
	std::cout << std::endl;
 
	//遍历所有buckets。如果其节点个数不为0，就打印节点个数
	for (int i = 0; i < iht.bucket_count(); ++i) {
		int n = iht.elems_in_bucket(i);  //桶中元素个数
		if (n != 0)
			std::cout << "bucket[" << i << "]has"<<n<<"elems." << std::endl;
	}
	//会打印如下内容
	//bucket[0] has 1 elems
	//bucket[2] has 3 elems
	//bucket[6] has 1 elems
	//bucket[10] has 1 elems
 
	/*为了验证bucket(list)的容量就是buckets vector的大小，
	这里从hastable<T>::Resize()得到结果。此处刻意将元素加到54个，
	看看是否发生”表格重建“
	*/
	for (int i = 0; i <= 47; ++i)
		iht.insert_equal(i);
	std::cout << iht.size() << std::endl; //54。元素(节点)个数
	std::cout << iht.bucket_count() << std::endl; //97，buckets个数
 
	//遍历所有buckets，如果其节点个数不为0，就打印节点个数
	for (int i = 0; i < iht.bucket_count(); ++i) {
		int n = iht.elems_in_bucket(i);
		if (n != 0)
			std::cout << "bucket[" << i << "]has" << n << "elems." << std::endl;
	}
	//打印的结果为：bucket[2]和bucket[11]的节点个数为2
	//其余的bucket[0]~bucket[47]的节点个数为1
	//此外，bucket[53],[55],[59],[63]的节点个数均为1
 
	//以迭代器遍历hashtable，将所有节点的值打印出来
	ite = iht.begin();
	for (int i = 0; i < iht.size(); ++i, ++ite)
		std::cout << *ite << " ";
	std::cout << std::endl;
	//0 1 2 2 3 4 5 6 7 8 9 10 11 108 12 13 14 15 16 17 18 19 20 21
	//22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42
	//43 44 45 46 47 53 55 59 63
 
	std::cout << *(iht.find(2)) << std::endl;  //2
	std::cout << *(iht.count(2)) << std::endl; //2
 
	return 0;
}
```

### Resource

- https://github1s.com/TBLGSn/SGI-STL/blob/HEAD/g++/stl_hashtable.h#L162-L240

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%85%B3%E8%81%94%E5%BC%8F%E5%AE%B9%E5%99%A8-hashtable/  

