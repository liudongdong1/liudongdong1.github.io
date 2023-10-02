# STL序列容器-heap


> heap并不归属于STL容器组件，但是作为priority queue的底层实现

### 1. push_heap

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411200733610.png)

```c++
template <class RandomAccessIterator> 
    inline void push_heap(RandomAccessIterator first, 
                          RandomAccessIterator last) { 
    //此函数被调用时，新元素已经位于底部容器的最低端
    __push_heap_aux(first, last, distance_type(first), 
                    value_type(first)); 
} 
template <class RandomAccessIterator, class Distance, class T> 
    inline void __push_heap_aux(RandomAccessIterator first, 
                                RandomAccessIterator last, Distance*, T*) { 
    __push_heap(first, Distance((last - first) - 1), Distance(0), 
                T(*(last - 1))); 
} 
template <class RandomAccessIterator, class Distance, class T> 
    void __push_heap(RandomAccessIterator first, Distance holeIndex, 
                     Distance topIndex, T value) { 
    Distance parent = (holeIndex - 1) / 2; // 找出父节点
    while (holeIndex > topIndex && *(first + parent) < value) { 
        // 尚未到达顶端，且父节点小于新值，则修改位置
        *(first + holeIndex) = *(first + parent); 
        holeIndex = parent; 
        parent = (holeIndex - 1) / 2; 
    } 
    *(first + holeIndex) = value; 
} 
```

### 2. pop_heap

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411201309194.png)

```c++
template <class RandomAccessIterator> 
    inline void pop_heap(RandomAccessIterator first, 
                         RandomAccessIterator last) { 
    __pop_heap_aux(first, last, value_type(first)); 
} 
template <class RandomAccessIterator, class T> 
    inline void __pop_heap_aux(RandomAccessIterator first, 
                               RandomAccessIterator last, T*) { 
    __pop_heap(first, last-1, last-1, T(*(last-1)), 
               distance_type(first)); 
} 

template <class RandomAccessIterator, class T, class Distance> 
    inline void __pop_heap(RandomAccessIterator first, 
                           RandomAccessIterator last, 
                           RandomAccessIterator result, 
                           T value, Distance*) { 
    *result = *first; 
    
    __adjust_heap(first, Distance(0), Distance(last - first), value); 
   
} 

template <class RandomAccessIterator, class Distance, class T> 
    void __adjust_heap(RandomAccessIterator first, Distance holeIndex, 
                       Distance len, T value) { 
    Distance topIndex = holeIndex; 
    Distance secondChild = 2 * holeIndex + 2; // 洞节点的右子节点
    while (secondChild < len) { 
       
        if (*(first + secondChild) < *(first + (secondChild - 1))) 
            secondChild--; 
        
        *(first + holeIndex) = *(first + secondChild); 
        holeIndex = secondChild; 
        secondChild = 2 * (secondChild + 1); //洞节点的右子节点
    } 
    if (secondChild == len) {
        
        *(first + holeIndex) = *(first + (secondChild - 1)); 
        holeIndex = secondChild - 1; 
    } 
   
    //__push_heap(first, holeIndex, topIndex, value);  这一步操作没有理解
    *(first+holeIndex)=value;
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411201853611.png)

### 3. sort_heap

```c++
template <class RandomAccessIterator> 
    void sort_heap(RandomAccessIterator first, 
                   RandomAccessIterator last) { 
    while (last - first > 1) 
        pop_heap(first, last--);
} 
```

### 4. make_heap

```c++
template <class RandomAccessIterator> 
    inline void make_heap(RandomAccessIterator first, 
                          RandomAccessIterator last) { 
    __make_heap(first, last, value_type(first), distance_type(first)); 
} 

template <class RandomAccessIterator, class T, class Distance> 
    void __make_heap(RandomAccessIterator first, 
                     RandomAccessIterator last, T*, 
                     Distance*) { 
    if (last - first < 2) return; 
    Distance len = last - first; 
    Distance parent = (len - 2)/2; 
    while (true) { 
        __adjust_heap(first, parent, len, T(*(first + parent))); 
        if (parent == 0) return; 
        parent--; 
    } 
} 
```

### 5. 测试demo

```c++
#include <vector> 
#include <iostream> 
#include <algorithm> // heap algorithms 
using namespace std; 
int main() 
{ 
    { 
        // test heap (底部以vector) 
        int ia[9] = {0,1,2,3,4,8,9,3,5}; 
        vector<int> ivec(ia, ia+9); 
        make_heap(ivec.begin(), ivec.end()); 
        for(int i=0; i<ivec.size(); ++i) 
            cout << ivec[i] << ' '; // 9 5 8 3 4 0 2 3 1 
        cout << endl; 
        ivec.push_back(7); 
        push_heap(ivec.begin(), ivec.end()); 
        for(int i=0; i<ivec.size(); ++i) 
            cout << ivec[i] << ' '; // 9 7 8 3 5 0 2 3 1 4 
        cout << endl; 
        pop_heap(ivec.begin(), ivec.end()); 
        cout << ivec.back() << endl; 
        ivec.pop_back(); 
        for(int i=0; i<ivec.size(); ++i) 
            cout << ivec[i] << ' '; 
        // 9. return but no remove. 
        // remove last elem and no return 
        // 8 7 4 3 5 0 2 3 1 
        cout << endl; 
        sort_heap(ivec.begin(), ivec.end()); 
        for(int i=0; i<ivec.size(); ++i) 
            cout << ivec[i] << ' '; // 0 1 2 3 3 4 5 7 8 
        cout << endl; 
    } 
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411203557894.png)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-heap/  

