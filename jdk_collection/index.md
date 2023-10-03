# JDK_Collection


### 1. set

#### .1. TreeSet

> 基于红黑树实现，支持`有序性操作`，例如根据一个范围查找元素的操作。但是查找效率不如 HashSet，`HashSet 查找的时间复杂度为 O(1)，TreeSet 则为 O(logN)`。TreeSet里面有一个TreeMap（适配器模式）.

#### .2. HashSet

> 基于哈希表实现，支持快速查找，但`不支持有序性操作`。并且`失去了元素的插入顺序信息`，也就是说使用 Iterator 遍历 HashSet 得到的结果是不确定的。

1)` HashSet实现了Set接口, 仅存储对象`; `HashMap实现了 Map接口, 存储的是键值对`.

2) HashSet底层其实是用HashMap实现存储的, HashSet封装了一系列HashMap的方法. `依靠HashMap来存储元素值,(利用hashMap的key键进行存储), 而value值默认为Object对象`. 所以HashSet也不允许出现重复值, 判断标准和HashMap判断标准相同, 两个元素的hashCode相等并且通过equals()方法返回true。

| HashMap                                     | HashSet                                                      |
| ------------------------------------------- | ------------------------------------------------------------ |
| HashMap实现了Map接口                        | HashSet实现了Set接口                                         |
| HashMap储存键值对                           | HashSet仅仅存储对象                                          |
| 使用put()方法将元素放入map中                | 使用add()方法将元素放入set中                                 |
| HashMap中使用键对象来计算hashcode值         | HashSet使用成员对象来计算hashcode值，对于两个对象来说hashcode可能相同，所以equals()方法用来判断对象的相等性，如果两个对象不同的话，那么返回false |
| HashMap比较快，因为是使用唯一的键来获取对象 | HashSet较HashMap来说比较慢                                   |

#### .3. LinkedHashSet

> 具有` HashSet 的查找效率`，且`内部使用双向链表`维护元素的插入顺序。

### 2. List

#### .1. ArrayList

> 基于`动态数组实现`，支持随机访问。*ArrayList*实现了*List*接口，是顺序容器，即元素存放的数据与放进去的顺序相同，允许放入`null`元素，底层通过**数组实现**。除该类未实现同步外，其余跟*Vector*大致相同。每个*ArrayList*都有一个容量（capacity），表示底层数组的实际大小，容器内存储元素的个数不能多于当前容量。当向容器中添加元素时，如果容量不足，容器会自动增大底层数组的大小。前面已经提过，Java泛型只是编译器提供的语法糖，所以`这里的数组是一个Object数组`，以便能够容纳任何类型的对象。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/ArrayList_base.png)

`size(), isEmpty(), get(), set()方法均能在常数时间内完成`，`add()方法的时间开销跟插入位置有关，addAll()方法的时间开销跟添加元素的个数成正比`。其余方法大都是线性时间。

为追求效率，`ArrayList没有实现同步（synchronized）`，如果需要多个线程并发访问，用户可以手动同步，也可使用Vector替代。

##### 1. set()

既然底层是一个数组*ArrayList*的`set()`方法也就变得非常简单，直接对数组的指定位置赋值即可。

```java
public E set(int index, E element) {
    rangeCheck(index);//下标越界检查
    E oldValue = elementData(index);
    elementData[index] = element;//赋值到指定位置，复制的仅仅是引用
    return oldValue;
}
```

#####  2. get()

`get()`方法同样很简单，唯一要注意的是由于底层数组是Object[]，`得到元素后需要进行类型转换`。

```java
public E get(int index) {
    rangeCheck(index);
    return (E) elementData[index];//注意类型转换
}
```

##### 3.  add()

跟C++ 的*vector*不同，*ArrayList*没有`push_back()`方法，对应的方法是`add(E e)`，*ArrayList*也没有`insert()`方法，对应的方法是`add(int index, E e)`。这两个方法都是向容器中添加新元素，这可能会导致*capacity*不足，因此在添加元素之前，都需要进行剩余空间检查，如果需要则自动扩容。扩容操作最终是通过`grow()`方法完成的。

```java
private void grow(int minCapacity) {
    int oldCapacity = elementData.length;
    int newCapacity = oldCapacity + (oldCapacity >> 1);//原来的1.5倍
    if (newCapacity - minCapacity < 0)
        newCapacity = minCapacity;
    if (newCapacity - MAX_ARRAY_SIZE > 0)
        newCapacity = hugeCapacity(minCapacity);
    elementData = Arrays.copyOf(elementData, newCapacity);//扩展空间并复制
}
```

由于Java GC自动管理了内存，这里也就不需要考虑源数组释放的问题。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/ArrayList_grow.png)

空间的问题解决后，插入过程就显得非常简单。

![ArrayList_add](https://gitee.com/github-25970295/blogpictureV2/raw/master/ArrayList_add.png)

`add(int index, E e)`需要先对元素进行移动，然后完成插入操作，也就意味着该方法有着线性的时间复杂度。

##### 4. addAll()

`addAll()`方法能够一次添加多个元素，根据位置不同也有两个版本，一个是在末尾添加的`addAll(Collection<? extends E> c)`方法，一个是从指定位置开始插入的`addAll(int index, Collection<? extends E> c)`方法。跟`add()`方法类似，在插入之前也需要进行空间检查，如果需要则自动扩容；如果从指定位置插入，也会存在移动元素的情况。 `addAll()`的时间复杂度不仅跟插入元素的多少有关，也跟插入的位置相关。

##### 5. remove()

`remove()`方法也有两个版本，一个是`remove(int index)`删除指定位置的元素，另一个是`remove(Object o)`删除第一个满足`o.equals(elementData[index])`的元素。删除操作是`add()`操作的逆过程，需要`将删除点之后的元素向前移动一个位置`。需要注意的是为了让GC起作用，`必须显式`的为最后一个位置赋`null`值。

```java
public E remove(int index) {
    rangeCheck(index);
    modCount++;
    E oldValue = elementData(index);
    int numMoved = size - index - 1;
    if (numMoved > 0)
        System.arraycopy(elementData, index+1, elementData, index, numMoved);
    elementData[--size] = null; //清除该位置的引用，让GC起作用
    return oldValue;
}
```

关于Java GC这里需要特别说明一下，**有了垃圾收集器并不意味着一定不会有内存泄漏**。对象能否被GC的依据是是否还有引用指向它，上面代码中如果不手动赋`null`值，除非对应的位置被其他元素覆盖，否则原来的对象就一直不会被回收。

#### .2. Vector

> 和 ArrayList 类似，但它是线程安全的。

#### .3. LinkedList

> 基于`双向链表`实现，只能顺序访问，但是可以`快速地在链表中间插入和删除元素`。不仅如此，LinkedList 还可以用作`栈、队列和双向队列`。*LinkedList*`同时实现了List接口和Deque接口`，也就是说它既可以看作一个`顺序容器`，又可以看作一个`队列（*Queue*）`，同时又可以看作一个`栈（*Stack*）`。这样看来，*LinkedList*简直就是个全能冠军。当你需要使用栈或者队列时，可以考虑使用*LinkedList*，一方面是因为Java官方已经声明不建议使用*Stack*类，更遗憾的是，Java里根本没有一个叫做*Queue*的类（它是个接口名字）。关于`栈或队列`，现在的`首选是ArrayDeque`，它有着比*LinkedList*（当作栈或队列使用时）有着更好的性能。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/LinkedList_base.png)

*LinkedList*底层**通过双向链表实现**，双向链表的每个节点用内部类*Node*表示。*LinkedList*通过`first`和`last`引用分别指向链表的第一个和最后一个元素。注意这里没有所谓的哑元，当链表为空的时候`first`和`last`都指向`null`。

```java
//Node内部类
private static class Node<E> {
    E item;
    Node<E> next;
    Node<E> prev;
    Node(Node<E> prev, E element, Node<E> next) {
        this.item = element;
        this.next = next;
        this.prev = prev;
    }
}
```

*`LinkedList*的实现方式决定了所有跟下标相关的操作都是线性时间，而在首段或者末尾删除元素只需要常数时间`。为追求效率*LinkedList*没有实现同步（synchronized），如果需要多个线程并发访问，可以先采用`Collections.synchronizedList()`方法对其进行包装。

#####  1. add()

add()方法有两个版本，一个是`add(E e)`，该方法在*LinkedList*的末尾插入元素，因为有`last`指向链表末尾，在末尾插入元素的花费是常数时间。只需要简单修改几个相关引用即可；另一个是`add(int index, E element)`，该方法是在指定下表处插入元素，需要先通过线性查找找到具体位置，然后修改相关引用完成插入操作。

[![LinkedList_add](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/LinkedList_add.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/LinkedList_add.png)

结合上图，可以看出`add(E e)`的逻辑非常简单。

```java
//add(E e)
public boolean add(E e) {
    final Node<E> l = last;
    final Node<E> newNode = new Node<>(l, e, null);
    last = newNode;
    if (l == null)
        first = newNode;//原来链表为空，这是插入的第一个元素
    else
        l.next = newNode;
    size++;
    return true;
}
```

`add(int index, E element)`的逻辑稍显复杂，可以分成两部分，1.先根据index找到要插入的位置；2.修改引用，完成插入操作。

```java
//add(int index, E element)
public void add(int index, E element) {
    checkPositionIndex(index);//index >= 0 && index <= size;
    if (index == size)//插入位置是末尾，包括列表为空的情况
        add(element);
    else {
        Node<E> succ = node(index);//1.先根据index找到要插入的位置
        //2.修改引用，完成插入操作。
        final Node<E> pred = succ.prev;
        final Node<E> newNode = new Node<>(pred, e, succ);
        succ.prev = newNode;
        if (pred == null)//插入位置为0
            first = newNode;
        else
            pred.next = newNode;
        size++;
    }
}
```

上面代码中的`node(int index)`函数有一点小小的trick，因为链表双向的，可以从开始往后找，也可以从结尾往前找，具体朝那个方向找取决于条件`index < (size >> 1)`，也即是index是靠近前端还是后端。

##### 2. remove()

`remove()`方法也有两个版本，一个是删除跟指定元素相等的第一个元素`remove(Object o)`，另一个是删除指定下标处的元素`remove(int index)`。

[![LinkedList_remove.png](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/LinkedList_remove.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/LinkedList_remove.png)

两个删除操作都要1.先找到要删除元素的引用，2.修改相关引用，完成删除操作。在寻找被删元素引用的时候`remove(Object o)`调用的是元素的`equals`方法，而`remove(int index)`使用的是下标计数，两种方式都是线性时间复杂度。在步骤2中，两个`remove()`方法都是通过`unlink(Node<E> x)`方法完成的。这里需要考虑删除元素是第一个或者最后一个时的边界情况。

```java
//unlink(Node<E> x)，删除一个Node
E unlink(Node<E> x) {
    final E element = x.item;
    final Node<E> next = x.next;
    final Node<E> prev = x.prev;
    if (prev == null) {//删除的是第一个元素
        first = next;
    } else {
        prev.next = next;
        x.prev = null;
    }
    if (next == null) {//删除的是最后一个元素
        last = prev;
    } else {
        next.prev = prev;
        x.next = null;
    }
    x.item = null;//let GC work
    size--;
    return element;
}
```

##### 3.  get()

`get(int index)`得到指定下标处元素的引用，通过调用上文中提到的`node(int index)`方法实现。

```java
public E get(int index) {
    checkElementIndex(index);//index >= 0 && index < size;
    return node(index).item;
}
```

#####  4. set()

`set(int index, E element)`方法将指定下标处的元素修改成指定值，也是先通过`node(int index)`找到对应下表元素的引用，然后修改`Node`中`item`的值。

```java
public E set(int index, E element) {
    checkElementIndex(index);
    Node<E> x = node(index);
    E oldVal = x.item;
    x.item = element;//替换新值
    return oldVal;
}
```

### 3. Queue

#### .1. ArrayDeque

> *ArrayDeque*底层通过数组实现，为了满足可以`同时在数组两端插入或删除元素的需求`，该数组还必须是循环的，即**循环数组（circular array）**，也就是说数组的任何一点都可能被看作起点或者终点。*ArrayDeque*是非线程安全的（not thread-safe），当多个线程同时使用的时候，需要程序员手动同步；另外，该容器不允许放入`null`元素。**`head`指向首端第一个有效元素，`tail`指向尾端第一个可以插入元素的空位**。因为是循环数组，所以`head`不一定总等于0，`tail`也不一定总是比`head`大。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/ArrayDeque_base.png)

##### 1. addFirst

`addFirst(E e)`的作用是在*Deque*的首端插入元素，也就是在`head`的前面插入元素，在空间足够且下标没有越界的情况下，只需要将`elements[--head] = e`即可。

[![ArrayDeque_addFirst.png](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/ArrayDeque_addFirst.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/ArrayDeque_addFirst.png)

实际需要考虑：1.空间是否够用，以及2.下标是否越界的问题。上图中，如果`head`为`0`之后接着调用`addFirst()`，虽然空余空间还够用，但`head`为`-1`，下标越界了。下列代码很好的解决了这两个问题。

```java
//addFirst(E e)
public void addFirst(E e) {
    if (e == null)//不允许放入null
        throw new NullPointerException();
    elements[head = (head - 1) & (elements.length - 1)] = e;//2.下标是否越界
    if (head == tail)//1.空间是否够用
        doubleCapacity();//扩容
}
```

上述代码我们看到，**空间问题是在插入之后解决的**，因为`tail`总是指向下一个可插入的空位，也就意味着`elements`数组至少有一个空位，所以插入元素的时候不用考虑空间问题。

下标越界的处理解决起来非常简单，`head = (head - 1) & (elements.length - 1)`就可以了，**这段代码相当于取余，同时解决了`head`为负值的情况**。因为`elements.length`必需是`2`的指数倍，`elements - 1`就是二进制低位全`1`，跟`head - 1`相与之后就起到了取模的作用，如果`head - 1`为负数（其实只可能是-1），则相当于对其取相对于`elements.length`的补码。

下面再说说扩容函数`doubleCapacity()`，其逻辑是申请一个更大的数组（原数组的两倍），然后将原数组复制过去。过程如下图所示：

[![ArrayDeque_doubleCapacity.png](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/ArrayDeque_doubleCapacity.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/ArrayDeque_doubleCapacity.png)

图中我们看到，复制分两次进行，第一次复制`head`右边的元素，第二次复制`head`左边的元素。

```java
//doubleCapacity()
private void doubleCapacity() {
    assert head == tail;
    int p = head;
    int n = elements.length;
    int r = n - p; // head右边元素的个数
    int newCapacity = n << 1;//原空间的2倍
    if (newCapacity < 0)
        throw new IllegalStateException("Sorry, deque too big");
    Object[] a = new Object[newCapacity];
    System.arraycopy(elements, p, a, 0, r);//复制右半部分，对应上图中绿色部分
    System.arraycopy(elements, 0, a, r, p);//复制左半部分，对应上图中灰色部分
    elements = (E[])a;
    head = 0;
    tail = n;
}
```

##### 2. addLast()

`addLast(E e)`的作用是在*Deque*的尾端插入元素，也就是在`tail`的位置插入元素，由于`tail`总是指向下一个可以插入的空位，因此只需要`elements[tail] = e;`即可。插入完成后再检查空间，如果空间已经用光，则调用`doubleCapacity()`进行扩容。

[![ArrayDeque_addLast.png](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/ArrayDeque_addLast.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/ArrayDeque_addLast.png)

```java
public void addLast(E e) {
    if (e == null)//不允许放入null
        throw new NullPointerException();
    elements[tail] = e;//赋值
    if ( (tail = (tail + 1) & (elements.length - 1)) == head)//下标越界处理
        doubleCapacity();//扩容
}
```

##### 3.pollFirst()

`pollFirst()`的作用是删除并返回*Deque*`首端元素`，也即是`head`位置处的元素。如果容器不空，只需要直接返回`elements[head]`即可，当然还需要处理下标的问题。由于`ArrayDeque`中不允许放入`null`，当`elements[head] == null`时，意味着容器为空。

```java
public E pollFirst() {
    E result = elements[head];
    if (result == null)//null值意味着deque为空
        return null;
    elements[h] = null;//let GC work
    head = (head + 1) & (elements.length - 1);//下标越界处理
    return result;
}
```

##### 4. pollLast()

`pollLast()`的作用是删除并返回*Deque*尾端元素，也即是`tail`位置前面的那个元素。

```java
public E pollLast() {
    int t = (tail - 1) & (elements.length - 1);//tail的上一个位置是最后一个元素
    E result = elements[t];
    if (result == null)//null值意味着deque为空
        return null;
    elements[t] = null;//let GC work
    tail = t;
    return result;
}
```

##### 5. peekFirst()

`peekFirst()`的作用是返回但不删除*Deque*首端元素，也即是`head`位置处的元素，直接返回`elements[head]`即可。

```java
public E peekFirst() {
    return elements[head]; // elements[head] is null if deque empty
}
```

##### 6. peekLast()

`peekLast()`的作用是返回但不删除*Deque*尾端元素，也即是`tail`位置前面的那个元素。

```java
public E peekLast() {
    return elements[(tail - 1) & (elements.length - 1)];
}
```

#### .2. LinkedList

> 基于双向链表实现，只能顺序访问，但是可以快速地在链表中间插入和删除元素。不仅如此，LinkedList 还可以用作栈、队列和双向队列。

#### .3. PriorityQueue

> **优先队列的作用是能保证每次取出的元素都是队列中权值最小的**（Java的优先队列每次取最小元素，C++的优先队列每次取最大元素）。这里牵涉到了大小关系，**元素大小的评判可以通过元素本身的自然顺序（\*natural ordering\*），也可以通过构造时传入的比较器**（*Comparator*，类似于C++的仿函数）。Java中*PriorityQueue*实现了*Queue*接口，不允许放入`null`元素；其通过堆实现，具体说是通过完全二叉树（*complete binary tree*）实现的**小顶堆**（任意一个非叶子节点的权值，都不大于其左右子节点的权值），也就意味着可以通过数组来作为*PriorityQueue*的底层实现。*PriorityQueue*的`peek()`和`element`操作是常数时间，`add()`, `offer()`, 无参数的`remove()`以及`poll()`方法的时间复杂度都是*log(N)*。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/PriorityQueue_base.png)

##### 1. add()和offer()

`add(E e)`和`offer(E e)`的语义相同，都是向优先队列中插入元素，只是`Queue`接口规定二者对插入失败时的处理不同，`前者在插入失败时抛出异常`，后则则会返回`false`。对于*PriorityQueue*这两个方法其实没什么差别。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/PriorityQueue_offer.png)

新加入的元素可能会破坏小顶堆的性质，因此需要进行必要的调整。

```java
//offer(E e)
public boolean offer(E e) {
    if (e == null)//不允许放入null元素
        throw new NullPointerException();
    modCount++;
    int i = size;
    if (i >= queue.length)
        grow(i + 1);//自动扩容
    size = i + 1;
    if (i == 0)//队列原来为空，这是插入的第一个元素
        queue[0] = e;
    else
        siftUp(i, e);//调整
    return true;
}
```

上述代码中，扩容函数`grow()`类似于`ArrayList`里的`grow()`函数，就是再申请一个更大的数组，并将原数组的元素复制过去，这里不再赘述。需要注意的是`siftUp(int k, E x)`方法，该方法用于插入元素`x`并维持堆的特性。

```java
//siftUp()
private void siftUp(int k, E x) {
    while (k > 0) {
        int parent = (k - 1) >>> 1;//parentNo = (nodeNo-1)/2
        Object e = queue[parent];
        if (comparator.compare(x, (E) e) >= 0)//调用比较器的比较方法
            break;
        queue[k] = e;
        k = parent;
    }
    queue[k] = x;
}
```

新加入的元素`x`可能会破坏小顶堆的性质，因此需要进行调整。调整的过程为：**从`k`指定的位置开始，将`x`逐层与当前点的`parent`进行比较并交换，直到满足`x >= queue[parent]`为止**。注意这里的比较可以是元素的自然顺序，也可以是依靠比较器的顺序。

##### 2. element()和peek()

`element()`和`peek()`的语义完全相同，都是`获取但不删除队首元素`，也就是队列中权值最小的那个元素，二者唯一的区别是当方法失败时前者抛出异常，后者返回`null`。根据小顶堆的性质，堆顶那个元素就是全局最小的那个；由于堆用数组表示，根据下标关系，`0`下标处的那个元素既是堆顶元素。所以**直接返回数组`0`下标处的那个元素即可**。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/PriorityQueue_peek.png)

```java
//peek()
public E peek() {
    if (size == 0)
        return null;
    return (E) queue[0];//0下标处的那个元素就是最小的那个
}
```

##### 3. remove()和poll()

`remove()`和`poll()`方法的语义也完全相同，都是`获取并删除队首元素`，区别是当方法失败时`前者抛出异常`，后者返回`null`。由于删除操作会改变队列的结构，为维护小顶堆的性质，需要进行必要的调整。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/PriorityQueue_poll.png)

```java
public E poll() {
    if (size == 0)
        return null;
    int s = --size;
    modCount++;
    E result = (E) queue[0];//0下标处的那个元素就是最小的那个
    E x = (E) queue[s];
    queue[s] = null;
    if (s != 0)
        siftDown(0, x);//调整
    return result;
}
```

上述代码首先记录`0`下标处的元素，并用最后一个元素替换`0`下标位置的元素，之后调用`siftDown()`方法对堆进行调整，最后返回原来`0`下标处的那个元素（也就是最小的那个元素）。重点是`siftDown(int k, E x)`方法，该方法的作用是**从`k`指定的位置开始，将`x`逐层向下与当前点的左右孩子中较小的那个交换，直到`x`小于或等于左右孩子中的任何一个为止**。

```java
//siftDown()
private void siftDown(int k, E x) {
    int half = size >>> 1;
    while (k < half) {
    	//首先找到左右孩子中较小的那个，记录到c里，并用child记录其下标
        int child = (k << 1) + 1;//leftNo = parentNo*2+1
        Object c = queue[child];
        int right = child + 1;
        if (right < size &&
            comparator.compare((E) c, (E) queue[right]) > 0)
            c = queue[child = right];
        if (comparator.compare(x, (E) c) <= 0)
            break;
        queue[k] = c;//然后用c取代原来的值
        k = child;
    }
    queue[k] = x;
}
```

##### 4. remove(Object o)

`remove(Object o)`方法用于删除队列中跟`o`相等的某一个元素（如果有多个相等，只删除一个），该方法不是*Queue*接口内的方法，而是*Collection*接口的方法。由于删除操作会改变队列结构，所以要进行调整；又由于删除元素的位置可能是任意的，所以调整过程比其它函数稍加繁琐。具体来说，`remove(Object o)`可以分为2种情况：1. 删除的是最后一个元素。直接删除即可，不需要调整。2. 删除的不是最后一个元素，从删除点开始以最后一个元素为参照调用一次`siftDown()`即可。此处不再赘述。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/PriorityQueue_remove2.png)

```java
//remove(Object o)
public boolean remove(Object o) {
	//通过遍历数组的方式找到第一个满足o.equals(queue[i])元素的下标
    int i = indexOf(o);
    if (i == -1)
        return false;
    int s = --size;
    if (s == i) //情况1
        queue[i] = null;
    else {
        E moved = (E) queue[s];
        queue[s] = null;
        siftDown(i, moved);//情况2
        ......
    }
    return true;
}
```

### 4. Map

#### .1. TreeMap

> 基于红黑树实现。Java *TreeMap*实现了*SortedMap*接口，也就是说会按照`key`的大小顺序对*Map*中的元素进行排序，`key`大小的评判可以通过其本身的自然顺序（natural ordering），也可以通过构造时传入的比较器（Comparator）.TreeMap是非同步的（not synchronized），如果需要在多线程环境使用，需要程序员手动同步；或者通过如下方式将*TreeMap*包装成（wrapped）同步的：SortedMap m = Collections.synchronizedSortedMap(new TreeMap(...));

##### 1. 红黑树

> **红黑树是一种近似平衡的二叉查找树，它能够确保任何一个节点的左右子树的高度差不会超过二者中较低那个的一倍**。具体来说，红黑树是满足如下条件的二叉查找树（binary search tree）：
>
> 1. 每个节点要么是红色，要么是黑色。
> 2. 根节点必须是黑色
> 3. 红色节点不能连续（也即是，红色节点的孩子和父亲都不能是红色）。
> 4. 对于每个节点，从该点至`null`（树尾端）的任何路径，都含有相同个数的黑色节点。
>
> 在树的结构发生改变时（插入或者删除操作），往往会破坏上述条件3或条件4，需要通过调整使得查找树重新满足红黑树的约束条件。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_base.png)

调整可以分为两类：一类是颜色调整，即改变某个节点的颜色；另一类是结构调整，集改变检索树的结构关系。结构调整过程包含两个基本操作：**左旋（Rotate Left），右旋（RotateRight）**。

###### 左旋

左旋的过程是将`x`的右子树绕`x`逆时针旋转，使得`x`的右子树成为`x`的父亲，同时修改相关节点的引用。旋转之后，二叉查找树的属性仍然满足。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_rotateLeft.png)

```java
//Rotate Left
private void rotateLeft(Entry<K,V> p) {
    if (p != null) {
        Entry<K,V> r = p.right;
        p.right = r.left;
        if (r.left != null)
            r.left.parent = p;
        r.parent = p.parent;
        if (p.parent == null)
            root = r;
        else if (p.parent.left == p)
            p.parent.left = r;
        else
            p.parent.right = r;
        r.left = p;
        p.parent = r;
    }
}
```

###### 右旋

右旋的过程是将`x`的左子树绕`x`顺时针旋转，使得`x`的左子树成为`x`的父亲，同时修改相关节点的引用。旋转之后，二叉查找树的属性仍然满足。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_rotateRight.png)

```java
//Rotate Right
private void rotateRight(Entry<K,V> p) {
    if (p != null) {
        Entry<K,V> l = p.left;
        p.left = l.right;
        if (l.right != null) l.right.parent = p;
        l.parent = p.parent;
        if (p.parent == null)
            root = l;
        else if (p.parent.right == p)
            p.parent.right = l;
        else p.parent.left = l;
        l.right = p;
        p.parent = l;
    }
}
```

###### 寻找节点后继

> 对于一棵二叉查找树，给定节点t，其后继（树中比大于t的最小的那个元素）可以通过如下方式找到：
>
> 1. `t的右子树不空，则t的后继是其右子树中最小的那个元素`。
> 2. t的`右孩子为空，则t的后继是其第一个向左走的祖先`。

后继节点在红黑树的删除操作中将会用到。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_successor.png)

```java
// 寻找节点后继函数successor()
static <K,V> TreeMap.Entry<K,V> successor(Entry<K,V> t) {
    if (t == null)
        return null;
    else if (t.right != null) {// 1. t的右子树不空，则t的后继是其右子树中最小的那个元素
        Entry<K,V> p = t.right;
        while (p.left != null)
            p = p.left;
        return p;
    } else {// 2. t的右孩子为空，则t的后继是其第一个向左走的祖先
        Entry<K,V> p = t.parent;
        Entry<K,V> ch = t;
        while (p != null && ch == p.right) {
            ch = p;
            p = p.parent;
        }
        return p;
    }
}
```

##### 2. get()

`get(Object key)`方法根据指定的`key`值返回对应的`value`，该方法调用了`getEntry(Object key)`得到相应的`entry`，然后返回`entry.value`。因此`getEntry()`是算法的核心。算法思想是根据`key`的自然顺序（或者比较器顺序）对二叉查找树进行查找，直到找到满足`k.compareTo(p.key) == 0`的`entry`。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_getEntry.png)

```java
//getEntry()方法
final Entry<K,V> getEntry(Object key) {
    ......
    if (key == null)//不允许key值为null
        throw new NullPointerException();
    Comparable<? super K> k = (Comparable<? super K>) key;//使用元素的自然顺序
    Entry<K,V> p = root;
    while (p != null) {
        int cmp = k.compareTo(p.key);
        if (cmp < 0)//向左找
            p = p.left;
        else if (cmp > 0)//向右找
            p = p.right;
        else
            return p;
    }
    return null;
}
```

##### 3. put()

`put(K key, V value)`方法是将指定的`key`, `value`对添加到`map`里。该方法首先会对`map`做一次查找，看是否包含该元组，如果已经包含则直接返回，查找过程类似于`getEntry()`方法；如果没有找到则会在红黑树中插入新的`entry`，如果插入之后破坏了红黑树的约束条件，还需要进行调整（旋转，改变某些节点的颜色）。

```java
public V put(K key, V value) {
	......
    int cmp;
    Entry<K,V> parent;
    if (key == null)
        throw new NullPointerException();
    Comparable<? super K> k = (Comparable<? super K>) key;//使用元素的自然顺序
    do {
        parent = t;
        cmp = k.compareTo(t.key);
        if (cmp < 0) t = t.left;//向左找
        else if (cmp > 0) t = t.right;//向右找
        else return t.setValue(value);
    } while (t != null);
    Entry<K,V> e = new Entry<>(key, value, parent);//创建并插入新的entry
    if (cmp < 0) parent.left = e;
    else parent.right = e;
    fixAfterInsertion(e);//调整
    size++;
    return null;
}
```

上述代码的插入部分并不难理解：首先在红黑树上找到合适的位置，然后创建新的`entry`并插入（当然，新插入的节点一定是树的叶子）。难点是调整函数`fixAfterInsertion()`，前面已经说过，调整往往需要1.改变某些节点的颜色，2.对某些节点进行旋转。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_put.png)

调整函数`fixAfterInsertion()`的具体代码如下，其中用到了上文中提到的`rotateLeft()`和`rotateRight()`函数。通过代码我们能够看到，情况2其实是落在情况3内的。情况4～情况6跟前三种情况是对称的，因此图解中并没有画出后三种情况，读者可以参考代码自行理解。

```java
//红黑树调整函数fixAfterInsertion()
private void fixAfterInsertion(Entry<K,V> x) {
    x.color = RED;
    while (x != null && x != root && x.parent.color == RED) {
        if (parentOf(x) == leftOf(parentOf(parentOf(x)))) {
            Entry<K,V> y = rightOf(parentOf(parentOf(x)));
            if (colorOf(y) == RED) {
                setColor(parentOf(x), BLACK);              // 情况1
                setColor(y, BLACK);                        // 情况1
                setColor(parentOf(parentOf(x)), RED);      // 情况1
                x = parentOf(parentOf(x));                 // 情况1
            } else {
                if (x == rightOf(parentOf(x))) {
                    x = parentOf(x);                       // 情况2
                    rotateLeft(x);                         // 情况2
                }
                setColor(parentOf(x), BLACK);              // 情况3
                setColor(parentOf(parentOf(x)), RED);      // 情况3
                rotateRight(parentOf(parentOf(x)));        // 情况3
            }
        } else {
            Entry<K,V> y = leftOf(parentOf(parentOf(x)));
            if (colorOf(y) == RED) {
                setColor(parentOf(x), BLACK);              // 情况4
                setColor(y, BLACK);                        // 情况4
                setColor(parentOf(parentOf(x)), RED);      // 情况4
                x = parentOf(parentOf(x));                 // 情况4
            } else {
                if (x == leftOf(parentOf(x))) {
                    x = parentOf(x);                       // 情况5
                    rotateRight(x);                        // 情况5
                }
                setColor(parentOf(x), BLACK);              // 情况6
                setColor(parentOf(parentOf(x)), RED);      // 情况6
                rotateLeft(parentOf(parentOf(x)));         // 情况6
            }
        }
    }
    root.color = BLACK;
}
```

##### 4. remove()

`remove(Object key)`的作用是删除`key`值对应的`entry`，该方法首先通过上文中提到的`getEntry(Object key)`方法找到`key`值对应的`entry`，然后调用`deleteEntry(Entry<K,V> entry)`删除对应的`entry`。由于删除操作会改变红黑树的结构，有可能破坏红黑树的约束条件，因此有可能要进行调整。

`getEntry()`函数前面已经讲解过，这里重点放`deleteEntry()`上，该函数删除指定的`entry`并在红黑树的约束被破坏时进行调用`fixAfterDeletion(Entry<K,V> x)`进行调整。

**由于红黑树是一棵增强版的二叉查找树，红黑树的删除操作跟普通二叉查找树的删除操作也就非常相似，唯一的区别是红黑树在节点删除之后可能需要进行调整**。现在考虑一棵普通二叉查找树的删除过程，可以简单分为两种情况：

> 1. 删除点p的左右子树都为空，或者只有一棵子树非空。
> 2. 删除点p的左右子树都非空。

对于上述情况1，处理起来比较简单，直接将p删除（左右子树都为空时），或者用非空子树替代p（只有一棵子树非空时）；对于情况2，可以用p的后继s（树中大于x的最小的那个元素）代替p，然后使用情况1删除s。

基于以上逻辑，红黑树的节点删除函数`deleteEntry()`代码如下：

```java
// 红黑树entry删除函数deleteEntry()
private void deleteEntry(Entry<K,V> p) {
    modCount++;
    size--;
    if (p.left != null && p.right != null) {// 2. 删除点p的左右子树都非空。
        Entry<K,V> s = successor(p);// 后继
        p.key = s.key;
        p.value = s.value;
        p = s;
    }
    Entry<K,V> replacement = (p.left != null ? p.left : p.right);
    if (replacement != null) {// 1. 删除点p只有一棵子树非空。
        replacement.parent = p.parent;
        if (p.parent == null)
            root = replacement;
        else if (p == p.parent.left)
            p.parent.left  = replacement;
        else
            p.parent.right = replacement;
        p.left = p.right = p.parent = null;
        if (p.color == BLACK)
            fixAfterDeletion(replacement);// 调整
    } else if (p.parent == null) {
        root = null;
    } else { // 1. 删除点p的左右子树都为空
        if (p.color == BLACK)
            fixAfterDeletion(p);// 调整
        if (p.parent != null) {
            if (p == p.parent.left)
                p.parent.left = null;
            else if (p == p.parent.right)
                p.parent.right = null;
            p.parent = null;
        }
    }
}
```

上述代码中占据大量代码行的，是用来修改父子节点间引用关系的代码，其逻辑并不难理解。下面着重讲解删除后调整函数`fixAfterDeletion()`。首先请思考一下，删除了哪些点才会导致调整？**只有删除点是BLACK的时候，才会触发调整函数**，因为删除RED节点不会破坏红黑树的任何约束，而删除BLACK节点会破坏规则4。

跟上文中讲过的`fixAfterInsertion()`函数一样，这里也要分成若干种情况。记住，**无论有多少情况，具体的调整操作只有两种：1.改变某些节点的颜色，2.对某些节点进行旋转。**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/TreeMap_fixAfterDeletion.png)

上述图解的总体思想是：将情况1首先转换成情况2，或者转换成情况3和情况4。当然，该图解并不意味着调整过程一定是从情况1开始。通过后续代码我们还会发现几个有趣的规则：a).如果是由情况1之后紧接着进入的情况2，那么情况2之后一定会退出循环（因为x为红色）；b).一旦进入情况3和情况4，一定会退出循环（因为x为root）。

删除后调整函数`fixAfterDeletion()`的具体代码如下，其中用到了上文中提到的`rotateLeft()`和`rotateRight()`函数。通过代码我们能够看到，情况3其实是落在情况4内的。情况5～情况8跟前四种情况是对称的，因此图解中并没有画出后四种情况，读者可以参考代码自行理解。

```java
private void fixAfterDeletion(Entry<K,V> x) {
    while (x != root && colorOf(x) == BLACK) {
        if (x == leftOf(parentOf(x))) {
            Entry<K,V> sib = rightOf(parentOf(x));
            if (colorOf(sib) == RED) {
                setColor(sib, BLACK);                   // 情况1
                setColor(parentOf(x), RED);             // 情况1
                rotateLeft(parentOf(x));                // 情况1
                sib = rightOf(parentOf(x));             // 情况1
            }
            if (colorOf(leftOf(sib))  == BLACK &&
                colorOf(rightOf(sib)) == BLACK) {
                setColor(sib, RED);                     // 情况2
                x = parentOf(x);                        // 情况2
            } else {
                if (colorOf(rightOf(sib)) == BLACK) {
                    setColor(leftOf(sib), BLACK);       // 情况3
                    setColor(sib, RED);                 // 情况3
                    rotateRight(sib);                   // 情况3
                    sib = rightOf(parentOf(x));         // 情况3
                }
                setColor(sib, colorOf(parentOf(x)));    // 情况4
                setColor(parentOf(x), BLACK);           // 情况4
                setColor(rightOf(sib), BLACK);          // 情况4
                rotateLeft(parentOf(x));                // 情况4
                x = root;                               // 情况4
            }
        } else { // 跟前四种情况对称
            Entry<K,V> sib = leftOf(parentOf(x));
            if (colorOf(sib) == RED) {
                setColor(sib, BLACK);                   // 情况5
                setColor(parentOf(x), RED);             // 情况5
                rotateRight(parentOf(x));               // 情况5
                sib = leftOf(parentOf(x));              // 情况5
            }
            if (colorOf(rightOf(sib)) == BLACK &&
                colorOf(leftOf(sib)) == BLACK) {
                setColor(sib, RED);                     // 情况6
                x = parentOf(x);                        // 情况6
            } else {
                if (colorOf(leftOf(sib)) == BLACK) {
                    setColor(rightOf(sib), BLACK);      // 情况7
                    setColor(sib, RED);                 // 情况7
                    rotateLeft(sib);                    // 情况7
                    sib = leftOf(parentOf(x));          // 情况7
                }
                setColor(sib, colorOf(parentOf(x)));    // 情况8
                setColor(parentOf(x), BLACK);           // 情况8
                setColor(leftOf(sib), BLACK);           // 情况8
                rotateRight(parentOf(x));               // 情况8
                x = root;                               // 情况8
            }
        }
    }
    setColor(x, BLACK);
}
```

#### .2. HashMap

> *HashMap*实现了*Map*接口，即允许放入`key`为`null`的元素，也允许插入`value`为`null`的元素；除该类未实现同步外，其余跟`Hashtable`大致相同；跟*TreeMap*不同，`该容器不保证元素顺序`，根据需要该容器可能会对元素重新哈希，元素的顺序也会被重新打散，因此不同时间迭代同一个*HashMap*的顺序可能会不同。 根据对冲突的处理方式不同，哈希表有两种实现方式，一种`开放地址方式`（Open addressing），另一种是`冲突链表方式`（Separate chaining with linked lists）。**Java \*HashMap\*采用的是冲突链表方式**。 
>
> HashMap 的实例有两个参数影响其性能：“**初始容量**” 和 “**加载因子**”。容量 是哈希表中桶的数量，初始容量 只是哈希表在创建时的容量。加载因子 是哈希表在其容量自动增加之前可以达到多满的一种尺度。当哈希表中的条目数超出了加载因子与当前容量的乘积时，则要对该哈希表进行 rehash 操作（即重建内部数据结构），从而哈希表将具有大约两倍的桶数。

```java
void                 clear()
Object               clone()
boolean              containsKey(Object key)
boolean              containsValue(Object value)
Set<Entry<K, V>>     entrySet()
V                    get(Object key)
boolean              isEmpty()
Set<K>               keySet()
V                    put(K key, V value)
void                 putAll(Map<? extends K, ? extends V> map)
V                    remove(Object key)
int                  size()
Collection<V>        values()
```

```java
//1. 遍历HashMap的键值对
// 假设map是HashMap对象
// map中的key是String类型，value是Integer类型
Integer integ = null;
Iterator iter = map.entrySet().iterator();
while(iter.hasNext()) {
    Map.Entry entry = (Map.Entry)iter.next();
    // 获取key
    key = (String)entry.getKey();
        // 获取value
    integ = (Integer)entry.getValue();
}
//2. 遍历HashMap的键
// 假设map是HashMap对象
// map中的key是String类型，value是Integer类型
String key = null;
Integer integ = null;
Iterator iter = map.keySet().iterator();
while (iter.hasNext()) {
        // 获取key
    key = (String)iter.next();
        // 根据key，获取value
    integ = (Integer)map.get(key);
}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/HashMap_base.png)

从上图容易看出，如果选择合适的哈希函数，`put()`和`get()`方法可以在常数时间内完成。但在对*HashMap*进行迭代时，需要遍历整个table以及后面跟的冲突链表。因此对于迭代比较频繁的场景，不宜将*HashMap*的初始大小设的过大。

有两个参数可以影响*HashMap*的性能：初始容量（inital capacity）和负载系数（load factor）。初始容量指定了初始`table`的大小，负载系数用来指定自动扩容的临界值。当`entry`的数量超过`capacity*load_factor`时，容器将自动扩容并重新哈希。对于插入元素较多的场景，将初始容量设大可以减少重新哈希的次数。

将对象放入到*HashMap*或*HashSet*中时，有两个方法需要特别关心：`hashCode()`和`equals()`。**`hashCode()`方法决定了对象会被放到哪个`bucket`里，当多个对象的哈希值冲突时，`equals()`方法决定了这些对象是否是“同一个对象”**。所以，如果要将自定义的对象放入到`HashMap`或`HashSet`中，需要*@Override*`hashCode()`和`equals()`方法。

##### 1. get()

`get(Object key)`方法根据指定的`key`值返回对应的`value`，该方法调用了`getEntry(Object key)`得到相应的`entry`，然后返回`entry.getValue()`。因此`getEntry()`是算法的核心。 算法思想是首先通过`hash()`函数得到对应`bucket`的下标，然后依次遍历冲突链表，通过`key.equals(k)`方法来判断是否是要找的那个`entry`。 [![HashMap_getEntry](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/HashMap_getEntry.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/HashMap_getEntry.png) 上图中`hash(k)&(table.length-1)`等价于`hash(k)%table.length`，原因是*HashMap*要求`table.length`必须是2的指数，因此`table.length-1`就是二进制低位全是1，跟`hash(k)`相与会将哈希值的高位全抹掉，剩下的就是余数了。

```java
//getEntry()方法
final Entry<K,V> getEntry(Object key) {
	......
	int hash = (key == null) ? 0 : hash(key);
    for (Entry<K,V> e = table[hash&(table.length-1)];//得到冲突链表
         e != null; e = e.next) {//依次遍历冲突链表中的每个entry
        Object k;
        //依据equals()方法判断是否相等
        if (e.hash == hash &&
            ((k = e.key) == key || (key != null && key.equals(k))))
            return e;
    }
    return null;
}
```

##### 2. put()

`put(K key, V value)`方法是将指定的`key, value`对添加到`map`里。该方法首先会对`map`做一次查找，看是否包含该元组，如果已经包含则直接返回，查找过程类似于`getEntry()`方法；如果没有找到，则会通过`addEntry(int hash, K key, V value, int bucketIndex)`方法插入新的`entry`，插入方式为**头插法**。 [![HashMap_addEntry](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/HashMap_addEntry.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/HashMap_addEntry.png)

```java
//addEntry()
void addEntry(int hash, K key, V value, int bucketIndex) {
    if ((size >= threshold) && (null != table[bucketIndex])) {
        resize(2 * table.length);//自动扩容，并重新哈希
        hash = (null != key) ? hash(key) : 0;
        bucketIndex = hash & (table.length-1);//hash%table.length
    }
    //在冲突链表头部插入新的entry
    Entry<K,V> e = table[bucketIndex];
    table[bucketIndex] = new Entry<>(hash, key, value, e);
    size++;
}
```

##### 3. remove()

`remove(Object key)`的作用是删除`key`值对应的`entry`，该方法的具体逻辑是在`removeEntryForKey(Object key)`里实现的。`removeEntryForKey()`方法会首先找到`key`值对应的`entry`，然后删除该`entry`（修改链表的相应引用）。查找过程跟`getEntry()`过程类似。 [![HashMap_removeEntryForKey](https://github.com/CarpenterLee/JCFInternals/raw/master/PNGFigures/HashMap_removeEntryForKey.png)](https://github.com/CarpenterLee/JCFInternals/blob/master/PNGFigures/HashMap_removeEntryForKey.png)

```java
//removeEntryForKey()
final Entry<K,V> removeEntryForKey(Object key) {
	......
	int hash = (key == null) ? 0 : hash(key);
    int i = indexFor(hash, table.length);//hash&(table.length-1)
    Entry<K,V> prev = table[i];//得到冲突链表
    Entry<K,V> e = prev;
    while (e != null) {//遍历冲突链表
        Entry<K,V> next = e.next;
        Object k;
        if (e.hash == hash &&
            ((k = e.key) == key || (key != null && key.equals(k)))) {//找到要删除的entry
            modCount++; size--;
            if (prev == e) table[i] = next;//删除的是冲突链表的第一个entry
            else prev.next = next;
            return e;
        }
        prev = e; e = next;
    }
    return e;
}
```

#### .3. HashTable

> 和 HashMap 类似，但它是线程安全的，这意味着同一时刻多个线程可以同时写入 HashTable 并且不会导致数据不一致。它是遗留类，不应该去使用它。现在可以使用 ConcurrentHashMap 来支持线程安全，并且 ConcurrentHashMap 的效率会更高，因为 ConcurrentHashMap 引入了分段锁

#### .4. LinkedHashMap

> 使用双向链表来维护元素的顺序，顺序为插入顺序或者最近最少使用(LRU)顺序。*LinkedHashMap*实现了*Map*接口，即允许放入`key`为`null`的元素，也允许插入`value`为`null`的元素。从名字上可以看出该容器是*linked list*和*HashMap*的混合体，也就是说它同时满足*HashMap*和*linked list*的某些特性。可将LinkedHashMap看作采用linked list增强的HashMap。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/LinkedHashMap_base.png)

事实上*LinkedHashMap*是*HashMap*的直接子类，二者唯一的区别是`LinkedHashMap在HashMap的基础上，采用双向链表`（doubly-linked list）的形式将所有`entry`连接起来，这样是为`保证元素的迭代顺序跟插入顺序相同`。上图给出了*LinkedHashMap*的结构图，主体部分跟*HashMap*完全一样，多了`header`指向双向链表的头部（是一个哑元），**该双向链表的迭代顺序就是`entry`的插入顺序**。

除了可以保迭代历顺序，这种结构还有一个好处：**迭代LinkedHashMap时不需要像HashMap那样遍历整个`table`，而只需要直接遍历`header`指向的双向链表即可**，也就是说*LinkedHashMap*的`迭代时间`就只跟`entry`的个数相关，而跟`table`的大小无关。

有两个参数可以影响*LinkedHashMap*的性能：初始容量（inital capacity）和负载系数（load factor）。初始容量指定了初始`table`的大小，负载系数用来指定自动扩容的临界值。当`entry`的数量超过`capacity*load_factor`时，容器将自动扩容并重新哈希。对于插入元素较多的场景，将初始容量设大可以减少重新哈希的次数。

将对象放入到*LinkedHashMap*或*LinkedHashSet*中时，有两个方法需要特别关心：`hashCode()`和`equals()`。**`hashCode()`方法决定了对象会被放到哪个`bucket`里，当多个对象的哈希值冲突时，`equals()`方法决定了这些对象是否是“同一个对象”**。所以，如果要将自定义的对象放入到`LinkedHashMap`或`LinkedHashSet`中，需要*@Override*`hashCode()`和`equals()`方法。

##### 1. get()

`get(Object key)`方法根据指定的`key`值返回对应的`value`。该方法跟`HashMap.get()`方法的流程几乎完全一样.

##### 2. put()

`put(K key, V value)`方法是将指定的`key, value`对添加到`map`里。该方法首先会对`map`做一次查找，看是否包含该元组，如果已经包含则直接返回，查找过程类似于`get()`方法；如果没有找到，则会通过`addEntry(int hash, K key, V value, int bucketIndex)`方法插入新的`entry`。

> 1. 从`table`的角度看，新的`entry`需要插入到对应的`bucket`里，当有哈希冲突时，采用头插法将新的`entry`插入到冲突链表的头部。
> 2. 从`header`的角度看，新的`entry`需要插入到双向链表的尾部。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/LinkedHashMap_addEntry.png)

```java
// LinkedHashMap.addEntry()
void addEntry(int hash, K key, V value, int bucketIndex) {
    if ((size >= threshold) && (null != table[bucketIndex])) {
        resize(2 * table.length);// 自动扩容，并重新哈希
        hash = (null != key) ? hash(key) : 0;
        bucketIndex = hash & (table.length-1);// hash%table.length
    }
    // 1.在冲突链表头部插入新的entry
    HashMap.Entry<K,V> old = table[bucketIndex];
    Entry<K,V> e = new Entry<>(hash, key, value, old);
    table[bucketIndex] = e;
    // 2.在双向链表的尾部插入新的entry
    e.addBefore(header);
    size++;
}
```

上述代码中用到了`addBefore()`方法将新`entry e`插入到双向链表头引用`header`的前面，这样`e`就成为双向链表中的最后一个元素。`addBefore()`的代码如下：

```java
// LinkedHashMap.Entry.addBefor()，将this插入到existingEntry的前面
private void addBefore(Entry<K,V> existingEntry) {
    after  = existingEntry;
    before = existingEntry.before;
    before.after = this;
    after.before = this;
}
```

##### 3. remove()

`remove(Object key)`的作用是删除`key`值对应的`entry`，该方法的具体逻辑是在`removeEntryForKey(Object key)`里实现的。`removeEntryForKey()`方法会首先找到`key`值对应的`entry`，然后删除该`entry`（修改链表的相应引用）。查找过程跟`get()`方法类似。

> 1. 从`table`的角度看，需要将该`entry`从对应的`bucket`里删除，如果对应的冲突链表不空，需要修改冲突链表的相应引用。
> 2. 从`header`的角度来看，需要将该`entry`从双向链表中删除，同时修改链表中前面以及后面元素的相应引用。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/LinkedHashMap_removeEntryForKey.png)

```java
// LinkedHashMap.removeEntryForKey()，删除key值对应的entry
final Entry<K,V> removeEntryForKey(Object key) {
	......
	int hash = (key == null) ? 0 : hash(key);
    int i = indexFor(hash, table.length);// hash&(table.length-1)
    Entry<K,V> prev = table[i];// 得到冲突链表
    Entry<K,V> e = prev;
    while (e != null) {// 遍历冲突链表
        Entry<K,V> next = e.next;
        Object k;
        if (e.hash == hash &&
            ((k = e.key) == key || (key != null && key.equals(k)))) {// 找到要删除的entry
            modCount++; size--;
            // 1. 将e从对应bucket的冲突链表中删除
            if (prev == e) table[i] = next;
            else prev.next = next;
            // 2. 将e从双向链表中删除
            e.before.after = e.after;
            e.after.before = e.before;
            return e;
        }
        prev = e; e = next;
    }
    return e;
}
```

##### 4. LinkedHashMap经典用法

*LinkedHashMap*除了可以保证迭代顺序外，还有一个非常有用的用法：可以轻松实现一个`采用了FIFO替换策略的缓存`。具体说来，LinkedHashMap有一个子类方法`protected boolean removeEldestEntry(Map.Entry<K,V> eldest)`，该方法的作用是告诉Map是否要删除“最老”的Entry，所谓最老就是当前Map中最早插入的Entry，如果该方法返回`true`，最老的那个元素就会被删除。在每次插入新元素的之后LinkedHashMap会自动询问removeEldestEntry()是否要删除最老的元素。这样只需要在子类中重载该方法，当元素个数超过一定数量时让removeEldestEntry()返回true，就能够实现一个固定大小的FIFO策略的缓存。示例代码如下：

```java
/** 一个固定大小的FIFO替换策略的缓存 */
class FIFOCache<K, V> extends LinkedHashMap<K, V>{
    private final int cacheSize;
    public FIFOCache(int cacheSize){
        this.cacheSize = cacheSize;
    }

    // 当Entry个数超过cacheSize时，删除最老的Entry
    @Override
    protected boolean removeEldestEntry(Map.Entry<K,V> eldest) {
       return size() > cacheSize;
    }
}
```

###  Resource

- https://github.com/CarpenterLee/JCFInternals/blob/master/markdown/0-Introduction.md



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/jdk_collection/  

