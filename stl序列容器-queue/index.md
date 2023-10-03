# STL序列容器-queue


> - 先进先出操作，由底部容器完成所有的工作

### 1. 数据结构

```c++
template <class T, class Sequence = deque<T> > 
class queue { 
    
    friend bool operator== __STL_NULL_TMPL_ARGS (const queue& x, const queue& y); 
    friend bool operator< __STL_NULL_TMPL_ARGS (const queue& x, const queue& y); 
    public: 
        typedef typename Sequence::value_type value_type; 
        typedef typename Sequence::size_type size_type; 
        typedef typename Sequence::reference reference; 
        typedef typename Sequence::const_reference const_reference; 
    protected: 
    	Sequence c; 
    public: 
        bool empty() const { return c.empty(); } 
        size_type size() const { return c.size(); } 
        reference front() { return c.front(); } 
        const_reference front() const { return c.front(); } 
        reference back() { return c.back(); } 
        const_reference back() const { return c.back(); } 

        void push(const value_type& x) { c.push_back(x); } 
        void pop() { c.pop_front(); } 
}; 
template <class T, class Sequence> 
    bool operator==(const queue<T, Sequence>& x, const queue<T, Sequence>& y) 
{ 
    return x.c == y.c; 
}
template <class T, class Sequence> 
    bool operator<(const queue<T, Sequence>& x, const queue<T, Sequence>& y) 
{ 
    return x.c < y.c; 
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-queue/  

