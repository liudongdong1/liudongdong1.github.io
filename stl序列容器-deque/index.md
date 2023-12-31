# STL序列容器-deque


> - deque 是一种双向开口的连续线性空间，可以在头部和尾部进行元素的插入和删除操作
>
> - 对deque进行排序，为了提高效率，先将deque复制到一个vector上，将vector排序后的元素复制到deque中
> - deque是分段连续空间，维护其整体连续的假象任务，通过迭代器++，--

### 1. 数据结构

#### .1. 内存分配

- deque采用一块所谓的map（不是 STL的map容器），作为主控，这里的map是一小块连续空间，其中每个元素（node）都是一个指针，指向另一段较大的连续线性空间，成为缓冲区（存储空间主题）。

```c++
template <class T, class Alloc = alloc, size_t BufSiz = 0> 
class deque { 
    public: // Basic types 
   		typedef T value_type; 
    	typedef value_type* pointer; 
    ... 
    protected: // Internal typedefs 
    	// 元素的指针的指针 pointer of pointer of T″
    	typedef pointer* map_pointer; 
    protected: // Data members 
    	map_pointer map; // 指向 map，map 是块连续空间，其内的每个元素都是一个指针，指向另一段较大的连续线性空间，成为缓冲区
    	size_type map_size; 
    ... 
};
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411185045109.png)

#### .2. 迭代器

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411190400286.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411190547730.png)

```c++
void set_node(map_pointer new_node) { 
    node = new_node; 
    first = *new_node; 
    last = first + difference_type(buffer_size()); 
}
```

#### .3. 数据结构

```c++
template <class T, class Alloc = alloc, size_t BufSiz = 0> 
class deque { 
    public: // Basic types 
        typedef T value_type;
        typedef value_type* pointer; 
        typedef size_t size_type; 
    public: // Iterators 
    	typedef __deque_iterator<T, T&, T*, BufSiz> iterator; 
    protected: // Internal typedefs
        typedef pointer* map_pointer; 
    protected: 
        iterator start; 
        iterator finish; 
        map_pointer map; 

    size_type map_size; // map 内由多少个指针
    ... 
}; 
```

### 2. 元素操作

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220411191622049.png)

```c++
void pop_back() { 
    if (finish.cur != finish.first) { 
        // 最后一个缓冲区由一个或者更多元素
        --finish.cur; 
        destroy(finish.cur); 
    } 
    else 
    {
        pop_back_aux(); // 进行缓冲区释放工作
    } 
} 
template <class T, class Alloc, size_t BufSize> 
    void deque<T, Alloc, BufSize>::pop_back_aux() { 
        deallocate_node(finish.first); // 释放最后一个缓冲区
        finish.set_node(finish.node - 1); // 调整 finish 的状态，使指向上一个缓冲区的最后一个元素
        finish.cur = finish.last - 1; 
        destroy(finish.cur); 
    } 
void pop_front() { 
    if (start.cur != start.last - 1) { 
        destroy(start.cur); 
        ++start.cur; 
    } 
    else 
	{
         pop_front_aux(); 
    }
} 
template <class T, class Alloc, size_t BufSize> 
    void deque<T, Alloc, BufSize>::pop_front_aux() { 
        destroy(start.cur); 
        deallocate_node(start.first); 
        start.set_node(start.node + 1); 
        start.cur = start.first; 
    }
```

```c++
// 最终需要保留一个缓冲区，也就是deque的初始状态
template <class T, class Alloc, size_t BufSize> 
    void deque<T, Alloc, BufSize>::clear() { 
        // 针对头尾以外的每一个缓冲区
        for (map_pointer node = start.node + 1; node < finish.node; ++node) { 
            destroy(*node, *node + buffer_size()); 
            data_allocator::deallocate(*node, buffer_size()); 
        } 
        if (start.node != finish.node) { 
            destroy(start.cur, start.last); 
            destroy(finish.first, finish.cur); 
            data_allocator::deallocate(finish.first, buffer_size()); 
        } else 
        {
            destroy(start.cur, finish.cur); 
        }
        finish = start; // 调整状燠
    }
```

```c++
iterator erase(iterator pos) { 
    iterator next = pos; 
    ++next; 
    difference_type index = pos - start; // 清除点之前的元素个数
    if (index < (size() >> 1)) {        //如果清楚点之前的元素比较少，就移动清楚点之前的元素，否则移动删除点之后的元素
        copy_backward(start, pos, next); 
        pop_front(); 
    } 
    else { 
        copy(next, finish, pos); 
        pop_back(); 
    } 
    return start + index; 
}
```

```c++
iterator insert(iterator position, const value_type& x) { 
    if (position.cur == start.cur) { 
        push_front(x); // 交给 push_front 去做
        return start; 
    } 
    else if (position.cur == finish.cur) { 
        push_back(x); // 交给 push_back 去做
        iterator tmp = finish;     
        --tmp;                 //todo, 这里为什么要--
        return tmp; 
    } 
    else { 
            return insert_aux(position, x); // 交给 insert_aux 去做
    } 
} 
template <class T, class Alloc, size_t BufSize> 
    typename deque<T, Alloc, BufSize>::iterator 
        deque<T, Alloc, BufSize>::insert_aux(iterator pos, const value_type& x) { 
            difference_type index = pos - start;  // 插入点之前的元素个数
            value_type x_copy = x; 
            if (index < size() / 2) { 
                push_front(front());  //在最前端插入一个和第一个元素相同的值
                iterator front1 = start; 
                ++front1; 
                iterator front2 = front1; 
                ++front2; 
                pos = start + index; 
                iterator pos1 = pos; 
                ++pos1; 
                copy(front2, pos1, front1); 
            } 
            else { 
                push_back(back()); 
                iterator back1 = finish; 
                --back1; 
                iterator back2 = back1; 
                --back2; 
                pos = start + index; 
                copy_backward(pos, back2, back1); 
            } 
            *pos = x_copy;
            return pos; 
        }
```

### 3. 常用函数思维导图

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/a2c6bf9652194dae142914fb1d2d6653.png)

### Resource

- https://www.programminghunter.com/article/86052272570/

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-deque/  

