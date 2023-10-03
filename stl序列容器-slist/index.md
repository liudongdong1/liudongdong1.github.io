# STL序列容器-slist


> - slist：迭代器属于`单向的 Forward Iterator（可读写）`。
> - list ：迭代器属于双向的 Bidirectional Iterator（可以双向读写）。
> - 看起来 slist 的功能应该会不如 list，但由于其单向链表的实现，其消耗的空间更小，某些操作更快。

### 1. 数据结构

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411204715391.png)

```c++
struct __slist_node_base 
{ 
    __slist_node_base* next; 
}; 
template <class T> 
    struct __slist_node : public __slist_node_base 
    { 
        T data; 
    }; 
inline __slist_node_base* __slist_make_link( 
    __slist_node_base* prev_node, 
    __slist_node_base* new_node) 
{ 
    new_node->next = prev_node->next; 
    prev_node->next = new_node; 
    return new_node; 
} 
inline size_t __slist_size(__slist_node_base* node) 
{ 
    size_t result = 0; 
    for ( ; node != 0; node = node->next) 
    {
        ++result; 
    }
    return result;
} 
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411205919338.png)

```c++
template <class T, class Alloc = alloc> 
class slist 
{ 
    public: 
        typedef T value_type; 
        typedef value_type* pointer; 
        typedef const value_type* const_pointer; 
        typedef value_type& reference; 
        typedef const value_type& const_reference; 
        typedef size_t size_type; 
        typedef ptrdiff_t difference_type; 
        typedef __slist_iterator<T, T&, T*> iterator; 
        typedef __slist_iterator<T, const T&, const T*> const_iterator; 
    private: 
    	typedef simple_alloc<list_node, Alloc> list_node_allocator; 
        typedef __slist_node<T> list_node; 
        typedef __slist_node_base list_node_base; 
        typedef __slist_iterator_base iterator_base; 

        static list_node* create_node(const value_type& x) { 
            list_node* node = list_node_allocator::allocate(); 
            __STL_TRY { 
                // 配置空间
                construct(&node->data, x); // 建构元素
                node->next = 0; 
            } 
            __STL_UNWIND(list_node_allocator::deallocate(node)); 
            return node; 
        } 
        static void destroy_node(list_node* node) { 
            destroy(&node->data); // 将元素解构
            list_node_allocator::deallocate(node); // 释䖬空间
        } 
    private: 
    	list_node_base head; 
    public: 
        slist() { head.next = 0; } 
        ~slist() { clear(); } 
    public: 
        iterator begin() { return iterator((list_node*)head.next); } 
        iterator end() { return iterator(0); } 
        size_type size() const { return __slist_size(head.next); } 
        bool empty() const { return head.next == 0; } 
        void swap(slist& L) 
        { 
            list_node_base* tmp = head.next; 
            head.next = L.head.next; 
            L.head.next = tmp; 
        } 
    public: 
        reference front() { return ((list_node*) head.next)->data; } 
        void push_front(const value_type& x) { 
            __slist_make_link(&head, create_node(x)); 
        } 
        void pop_front() { 
            list_node* node = (list_node*) head.next; 
            head.next = node->next; 
            destroy_node(node);   //注意这里要释放空间
        } 
};
```

### 2. 案例demo

```c++
#include <slist> 
#include <iostream> 
#include <algorithm> 
using namespace std; 
int main() 
{ 
    int i; 
    slist<int> islist; 
    cout << "size=" << islist.size() << endl;
    islist.push_front(9); 
    islist.push_front(1); 
    islist.push_front(2); 
    islist.push_front(3); 
    islist.push_front(4); 
    cout << "size=" << islist.size() << endl; 
    slist<int>::iterator ite =islist.begin(); 
    slist<int>::iterator ite2=islist.end(); 
    for(; ite != ite2; ++ite) 
        cout << *ite << ' '; 
    // size=5 
    // 4 3 2 1 9 
    cout << endl; 
    ite = find(islist.begin(), islist.end(), 1); 
    if (ite!=0) 
        islist.insert(ite, 99); 
    cout << "size=" << islist.size() << endl; 
    cout << *ite << endl; 
    ite =islist.begin(); 
    ite2=islist.end(); 
    for(; ite != ite2; ++ite) 
        cout << *ite << ' '; 
    // size=6 
    // 1 
    // 4 3 2 99 1 9 
    cout << endl; 
    ite = find(islist.begin(), islist.end(), 3); 
    if (ite!=0) 
        cout << *(islist.erase(ite)) << endl; 
    ite =islist.begin(); 
    ite2=islist.end(); 
    for(; ite != ite2; ++ite) 
        cout << *ite << ' '; 
    // 2 
    // 4 2 99 1 9 
    cout << endl; 
} 
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-slist/  

