# STL序列容器-list


> list底层实现是使用双向链表，内部有个node节点指向链表尾部，实现循环链表，提供快速插入删除操作。

### 1. 数据结构

```c++
template <class T> 
    struct __list_node { 
        typedef void* void_pointer; 
        void_pointer prev; // 类别void*, 可以设置为 __list_node <T>*
        void_pointer next; 
        T data; 
    }; 
```

- 迭代器

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411163840421.png)

- list环形数据结构

```c++
template <class T, class Alloc = alloc> // 预䆒使用 alloc δ配置器
    class list { 
     protected: 
        typedef __list_node<T> list_node; 
     public: 
        typedef list_node* link_type; 
     protected: 
        link_type node; //node指向 可以置为尾端的一个空白节点
        ... 
    }; 
```

```c++
iterator begin() { return (link_type)((*node).next); } 
iterator end() { return node; } 
bool empty() const { return node->next == node; } 
size_type size() const { 
    size_type result = 0; 
    distance(begin(), end(), result); // 全域函式
    return result; 
} 
reference front() { return *begin(); } 
reference back() { return *(--end()); } 
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411164123827.png)

### 2. 创建

```c++
protected: 

link_type get_node() { return list_node_allocator::allocate(); } 

void put_node(link_type p) { list_node_allocator::deallocate(p); } 

link_type create_node(const T& x) { 
    link_type p = get_node(); 
    construct(&p->data, x);// 全域函式
    return p; 
} 
// 摧毁′解构并释放″垼个节点
void destroy_node(link_type p) { 
    destroy(&p->data); // 全域函式
    put_node(p); 
} 
```

#### .1. 空list

```c++
public: 
	list() { empty_initialize(); } // 创建一个空list，默认构造函数
protected: 
    void empty_initialize() 
        node = get_node(); 
        node->next = node; 
        node->prev = node; 
    } 
```

#### .2. insert操作

```c++
iterator insert(iterator position, const T& x) { 
    link_type tmp = create_node(x); 
    tmp->next = position.node; 
    tmp->prev = position.node->prev; 
    (link_type(position.node->prev))->next = tmp; 
    position.node->prev = tmp; 
    return tmp; 
} 
```

### 3. 元素操作

> push_front, push_back, erase, pop_front, pop_back,  clear, remove, unique, splice, merge, reverse, sort

```c++
void push_front(const T& x) { insert(begin(), x); } 

void push_back(const T& x) { insert(end(), x); } 

iterator erase(iterator position) { 
    link_type next_node = link_type(position.node->next); 
    link_type prev_node = link_type(position.node->prev); 
    prev_node->next = next_node; 
    next_node->prev = prev_node; 
    destroy_node(position.node); // 记得及时的清空申请空间
    return iterator(next_node); 
} 
// 移除头节点
void pop_front() { erase(begin()); } 
// 移除尾部节点
void pop_back() 
    iterator tmp = end(); 
	erase(--tmp); 
} 

template <class T, class Alloc> 
    void list<T, Alloc>::clear() 
    { 
        link_type cur = (link_type) node->next; // begin() 
        while (cur != node) { 
            link_type tmp = cur; 
            cur = (link_type) cur->next; 
            destroy_node(tmp); 
        } 
        // 恢复 node 原始状态
        node->next = node; 
        node->prev = node; 
    } 

template <class T, class Alloc> 
    void list<T, Alloc>::remove(const T& value) { 
        iterator first = begin(); 
        iterator last = end(); 
        while (first != last) { 
            iterator next = first; 
            ++next; 
            if (*first == value) erase(first); 
            first = next; 
        } 
    } 

template <class T, class Alloc> 
    void list<T, Alloc>::unique() {   //需要进行排序操作
        iterator first = begin(); 
        iterator last = end(); 
        if (first == last) return; 
        iterator next = first; 
        while (++next != last) { 
            if (*first == *next) {
                erase(next); 
            }
            else {
                first = next; 
            }
            next = first; 
        } 
    }
```

> transfer操作，将某范围内连续元素迁移到某个特定位置之前。

```c++
//将[first,last）内的所有元素搬移到position之前
void transfer(iterator position, iterator first, iterator last) 
    if (position != last) { 
        { 
            (*(link_type((*last.node).prev))).next = position.node; 
            (*(link_type((*first.node).prev))).next = last.node; 
            (*(link_type((*position.node).prev))).next = first.node; 
            link_type tmp = link_type((*position.node).prev); 
            (*position.node).prev = (*last.node).prev; 
            (*last.node).prev = (*first.node).prev; 
            (*first.node).prev = tmp; 
        } 
    }
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411175827820.png)

```c++
//将x结合于position 所指位置之前，x 必须不同于 *this
void splice(iterator position, list& x) { 
    if (!x.empty()) 
        transfer(position, x.begin(), x.end()); 
} 
// 将 i 所指元素接合于 position 之前，position 和 i 可能指向同一个 *this
void splice(iterator position, list&, iterator i) { 
    iterator j = i; 
    ++j; 
    if (position == i || position == j) return; 
    transfer(position, i, j); 
} 
// 将 [first,last) 内的所有元素接合于 position 所指位置之前
// position 和[first,last)可能指向同一个 list，
// 但 position 不能位于[first,last)之内
void splice(iterator position, list&, iterator first, iterator last) 
    if (first != last) 
        transfer(position, first, last); 
}
```

```c++
template <class T, class Alloc> 
    void list<T, Alloc>::merge(list<T, Alloc>& x) { 
        iterator first1 = begin(); 
        iterator last1 = end(); 
        iterator first2 = x.begin(); 
        iterator last2 = x.end(); 
        // 俩个链表都经过了排序处理，
        while (first1 != last1 && first2 != last2) 
            if (*first2 < *first1) { 
                iterator next = first2; 
                transfer(first1, first2, ++next); 
                first2 = next; 
            } 
        else 
            ++first1; 
        if (first2 != last2) transfer(last1, first2, last2); 
    }
```

```c++
// reverse() 将 *this 的内容逆向重置
template <class T, class Alloc> 
    void list<T, Alloc>::reverse() { 
        // 只有一个元素或者为空，不进行处理
        // 使用 size() == 0 || size() == 1 来判断，速度比较慢
        if (node->next == node || link_type(node->next)->next == node) 
            return; 
        iterator first = begin(); 
        ++first; 
        while (first != end()) { 
            iterator old = first; 
            ++first; 
            transfer(begin(), old, first); 
        } 
    } 
// list 不能使用 STL 算法 sort()，只能使用自己的 sort() member function，
// 因δ STL 算法 sort() 只接受 RamdonAccessIterator. 
// 本函式采用 quick sort. 
template <class T, class Alloc> 
    void list<T, Alloc>::sort() { 
        // 只有一个元素或者为空，不进行处理
        // 使用 size() == 0 || size() == 1 来判断，速度比较慢
        if (node->next == node || link_type(node->next)->next == node) 
            return; 

        list<T, Alloc> carry; 
        list<T, Alloc> counter[64]; 
        int fill = 0; 
        while (!empty()) { 
            carry.splice(carry.begin(), *this, begin()); 
            int i = 0; 
            while(i < fill && !counter[i].empty()) { 
                counter[i].merge(carry); 
                carry.swap(counter[i++]); 
            } 
            carry.swap(counter[i]); 
            if (i == fill) ++fill; 
        } 
        for (int i = 1; i < fill; ++i) 
            counter[i].merge(counter[i-1]); 
        swap(counter[fill-1]); 
    }
```

### 4. 常用函数xmind

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/4c6ed408d36fe0f5519875f0e1e73eb8.png)

### Resource

- STL源码剖析（侯捷译）
- https://www.programminghunter.com/article/56122274299/

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-list/  

