# STL序列容器-priority_queue


### 1. 数据结构

```c++
template <class T, class Sequence = vector<T> , class Compare = less<typename Sequence::value_type> > 
class priority_queue { 
    public: 
        typedef typename Sequence::value_type value_type; 
        typedef typename Sequence::size_type size_type; 
        typedef typename Sequence::reference reference; 
        typedef typename Sequence::const_reference const_reference; 
    protected: 
        Sequence c; 
        Compare comp; 
    public: 
        priority_queue() : c() {} 
        explicit priority_queue(const Compare& x) : c(), comp(x) {} 
       
        template <class InputIterator> 
            priority_queue(InputIterator first, InputIterator last, const Compare& x) 
            : c(first, last), comp(x) { make_heap(c.begin(), c.end(), comp); } 
        template <class InputIterator> 
            priority_queue(InputIterator first, InputIterator last) 
            : c(first, last) { make_heap(c.begin(), c.end(), comp); } 
        bool empty() const { return c.empty(); } 
        size_type size() const { return c.size(); } 
        const_reference top() const { return c.front(); } 
        void push(const value_type& x) { 
            __STL_TRY { 
                c.push_back(x); 
                push_heap(c.begin(), c.end(), comp); // push_heap 是泛型算法
            } 
            __STL_UNWIND(c.clear()); 
        } 
        void pop() { 
            __STL_TRY { 
                pop_heap(c.begin(), c.end(), comp); 
                c.pop_back(); 
            } 
            __STL_UNWIND(c.clear()); 
        } 
};
```

### 2. 测试demo

```c++
#include <queue> 
#include <iostream> 
#include <algorithm> 
using namespace std; 
int main() 
{ 
    // test priority queue... 
    int ia[9] = {0,1,2,3,4,8,9,3,5}; 
    priority_queue<int> ipq(ia, ia+9); 
    cout << "size=" << ipq.size() << endl; 
    for(int i=0; i<ipq.size(); ++i) 
        // size=9 
        cout << ipq.top() << ' '; 
    cout << endl; 
    while(!ipq.empty()) { 
        cout << ipq.top() << ' '; 
        // 9 9 9 9 9 9 9 9 9 
        // 9 8 5 4 3 3 2 1 0 
        ipq.pop(); 
    } 
    cout << endl; 
} 
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-priority_queue/  

