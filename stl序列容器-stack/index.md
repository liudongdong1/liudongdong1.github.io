# STL序列容器-stack


> - stack所有元素进出都符合先进后出，只有stack顶端的元素才可以被外界访问
> - stack不提供迭代器功能，以底部容器完成所有工作

### 1. 数据结构

```c++
template <class T, class Sequence = deque<T> > 
class stack { 
    friend bool operator== __STL_NULL_TMPL_ARGS (const stack&, const stack&); 
    friend bool operator< __STL_NULL_TMPL_ARGS (const stack&, const stack&); 
    public: 
        typedef typename Sequence::value_type value_type; 
        typedef typename Sequence::size_type size_type; 
        typedef typename Sequence::reference reference; 
        typedef typename Sequence::const_reference const_reference; 
    protected: 
    	Sequence c; // 底层容器
    public: 
        bool empty() const { return c.empty(); } 
        size_type size() const { return c.size(); } 
        reference top() { return c.back(); } 
        const_reference top() const { return c.back(); } 
        
        void push(const value_type& x) { c.push_back(x); } 
        void pop() { c.pop_back(); } 
}; 
template <class T, class Sequence> 
    bool operator==(const stack<T, Sequence>& x, const stack<T, Sequence>& y) 
{ 
    return x.c == y.c; 
} 
template <class T, class Sequence> 
    bool operator<(const stack<T, Sequence>& x, const stack<T, Sequence>& y) 
{ 
    return x.c < y.c; 
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/1fa7f105ae486bd748e447abe4cee65a.png)

### Resource

- https://www.programminghunter.com/article/71792272395/


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-stack/  

