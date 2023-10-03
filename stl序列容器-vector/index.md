# STL序列容器-vector


> vector底层实现使用连续数组，提供动态数组功能

### 1. 数据结构

```c++
template <class T, class Alloc = alloc> 
class vector { 
protected: 
    iterator start; 
    iterator finish; 
    iterator end_of_storage; 
}; 
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411154131578.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411154251849.png)

### 2. 构造&添加

#### .1. construct 操作

```c++
// 建构式，指定 vector 大小 n 和初值 value 
vector(size_type n, const T& value) { fill_initialize(n, value); } 
// 充填并给予赋值
void fill_initialize(size_type n, const T& value) { 
    start = allocate_and_fill(n, value); 
    finish = start + n; 
    end_of_storage = finish; 
} 
// 配置而后填充
iterator allocate_and_fill(size_type n, const T& x) {
    iterator result = data_allocator::allocate(n); // 配置 n 个元素空间
    uninitialized_fill_n(result, n, x); // 全域函式，根据第二个参数的型类别特性，决定使用算法fill_n()或者反复调用construct()来完成任务
    return result; 
}
```

#### .2. push_back操作

```c++
void push_back(const T& x) { 
    if (finish != end_of_storage) { 
        construct(finish, x); 
        ++finish; 
    } 
    else 
        insert_aux(end(), x); // vector member function
} 
template <class T, class Alloc> 
void vector<T, Alloc>::insert_aux(iterator position, const T& x) { 
    if (finish != end_of_storage) {
        //在备用空间起始处构建第一个元素，并以vector最后一个元素为其初值
        construct(finish, *(finish - 1)); 
        ++finish; 
        T x_copy = x; 
        copy_backward(position, finish - 2, finish - 1); 
        *position = x_copy; 
    } 
    else { // 无备用空间
        const size_type old_size = size(); 
        const size_type len = old_size != 0 ? 2 * old_size : 1; 
   
        iterator new_start = data_allocator::allocate(len); // 申请配置
        iterator new_finish = new_start; 
        try { 
            // 将原 vector 的内容拷贝到vector
            new_finish = uninitialized_copy(start, position, new_start); 
            // 为新元素设定初值 x 
            construct(new_finish, x);
            ++new_finish; 
            //todo
            new_finish = uninitialized_copy(position, finish, new_finish); 
        } 
        catch(...) { 
            // "commit or rollback" semantics. 
            destroy(new_start, new_finish); 
            data_allocator::deallocate(new_start, len); 
            throw; 
        } 
        // 解构并释放原 vector 
        destroy(begin(), end()); 
        deallocate(); 
        // 调整迭鰯器，指向鑄 vector 
        start = new_start; 
        finish = new_finish; 
        end_of_storage = new_start + len; 
    } 
} 
```

#### .3. pop_back, erase, insert, clear

```c++
void pop_back() { 
    --finish; 
    destroy(finish); 
} 
// 清除 [first,last) 所有元素
iterator erase(iterator first, iterator last) { 
    iterator i = copy(last, finish, first); // copy 是全域函式
    destroy(i, finish); 
    finish = finish - (last - first); 
    return first; 
} 
// 清除某个位置上的元素
iterator erase(iterator position) { 
    if (position + 1 != end()) 
        copy(position + 1, finish, position); // copy 是全域函式，把position到finish之间元素拷贝到position位置上
    --finish; 
    destroy(finish); // destroy 是全域函式，2.2.3 节
    return position; 
} 
void clear() { erase(begin(), end()); } 
```

- **insert操作**

```c++
void vector<T, Alloc>::insert(iterator position, size_type n, const T& x) 
{ 
    if (n != 0) { 
        if (size_type(end_of_storage - finish) >= n) 
        {
            // 备用空间大于等于新增元素个数
            T x_copy = x; 
        }

        const size_type elems_after = finish - position; 
        iterator old_finish = finish; 
        if (elems_after > n) 
        {
            //安插点之后的现有元素个数大于 新增元素个数
            uninitialized_copy(finish - n, finish, finish); 
            finish += n; // 将 vector 尾端后移
            copy_backward(position, old_finish - n, old_finish); 
            fill(position, position + n, x_copy); // 从插入点开始填充新值
        }else { 
            // 安插点之后的现有元素个数 < 新增元素个数
            uninitialized_fill_n(finish, n - elems_after, x_copy); 
            finish += n - elems_after; 
            uninitialized_copy(position, old_finish, finish); 
            finish += elems_after; 
            fill(position, old_finish, x_copy); 
        } 
    } 
    else { 
        // 备用空间小于< 新增元素个数
        //首先判断新长度：旧长度两倍，或者旧长度+新长度
        const size_type old_size = size(); 
        const size_type len = old_size + max(old_size, n); 
        
        iterator new_start = data_allocator::allocate(len); 
        iterator new_finish = new_start; 
        __STL_TRY { 
            // 将就vector的插入点之前的元素拷贝到新空间
            new_finish = uninitialized_copy(start, position, new_start); 
            // 添加新元素
            new_finish = uninitialized_fill_n(new_finish, n, x); 
            // 赋值剩余的旧元素
            new_finish = uninitialized_copy(position, finish, new_finish); 
        } 
        # ifdef __STL_USE_EXCEPTIONS 
        catch(...) { 
            // 如异常发生，实现 "commit or rollback" semantics. 
            destroy(new_start, new_finish); 
            data_allocator::deallocate(new_start, len); 
            throw; 
        } 
        # endif /* __STL_USE_EXCEPTIONS */ 
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411161923466.png)

### 3. 常用函数

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/c6798c11f83f42defcca4dec775ec7d4.png)

### Resource

- https://www.programminghunter.com/article/13112271844/
- 《The Annotated STL Sources》

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E5%BA%8F%E5%88%97%E5%AE%B9%E5%99%A8-vector/  

