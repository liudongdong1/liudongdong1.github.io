# Collection


> - **数组（可以存储基本数据类型）**是用来存现对象的一种容器，但是数组的长度固定，不适合在对象数量未知的情况下使用。
> - **集合**（只能存储对象，对象类型可以不一样）的长度可变，可在多数情况下使用。

### 1. 集合类

#### .1. Collection

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220314143226382.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210510223825.png)

- **Collection接口**: 是集合类的根接口，Java中没有提供这个接口的直接的实现类。但是却让其被继承产生了两个接口，就是`Set和List`。`Set中不能包含重复的元素`。`List是一个有序的集合`，可以包含重复的元素，提供了按索引访问的方式。

- **Map**: 是Java.util包中的另一个接口，它和Collection接口没有关系，是相互独立的，但是都属于集合类的一部分。`Map包含了key-value对`。Map不能包含重复的key，但是可以包含相同的value。
  - **HashMap**: 根据键的HashCode值存储数据，根据键可以直接获取它的值，具有`很快的访问速度`，遍历时，`取得数据的顺序是完全随机的`。HashMap最多只允许一条记录的键为Null，允许多条记录的值为Null，是非同步的.
  - **Hashtable**: 即`任一时刻只有一个线程能写Hashtable`，因此也导致了Hashtale在`写入时会比较慢`，它继承自Dictionary类，`不同的是它不允许记录的键或者值为null`，同时效率较低。

- **Iterator:** 所有的集合类，都实现了Iterator接口，这是一个用于遍历集合中元素的接口，主要包含以下三种方法：
  1. hasNext()是否还有下一个元素。
  2. next()返回下一个元素。
  3. remove()删除当前元素。

|            | 是否有序    | 是否允许元素重复   |                                                           |
| ---------- | ----------- | ------------------ | --------------------------------------------------------- |
| Collection |             |                    |                                                           |
| List       | 是          | 是                 |                                                           |
| Set        | AbstractSet | 否                 | 否                                                        |
|            | HashSet     |                    |                                                           |
|            | TreeSet     | 是（用二叉排序树） |                                                           |
| Map        | AbstractMap | 否                 | 使用key-value来映射和存储数据，key必须唯一，value可以重复 |
|            | HashMap     |                    |                                                           |
|            | TreeMap     | 是（用二叉排序树） |                                                           |

#### .2. 遍历

```java
//for的形式：
for（int i=0;i<arr.size();i++）{...}

//foreach的形式： 
for（int　i：arr）{...}

//iterator的形式：
Iterator it = arr.iterator();
while(it.hasNext()){ object o =it.next(); ...}

//  keySet
Map map = new HashMap();
map.put("key1","lisi1");
map.put("key2","lisi2");
map.put("key3","lisi3");
map.put("key4","lisi4"); 
//先获取map集合的所有键的set集合，keyset（）
Iterator it = map.keySet().iterator();
 //获取迭代器
while(it.hasNext()){
Object key = it.next();
System.out.println(map.get(key));
}
// using entrySet
Iterator it = map.entrySet().iterator();
while(it.hasNext()){
Entry e =(Entry) it.next();
System.out.println("键"+e.getKey () + "的值为" + e.getValue());
}
```

#### .3. 对比

- `LinkedList经常用在增删操作较多而查询操作很少的情况下`，ArrayList则相反。
- Vector
  - 线程同步的，所以它也是线程安全的
  - vector`增长率为目前数组长度的100%`，而arraylist增长率为目前数组长度的50%。如果在集合中使用数据量比较大的数据，用vector有一定的优势。
- ArrayList
  - 不考虑到线程的安全因素，一般用arraylist效率比较高。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220314143518319.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220314143634834.png)

### 2.  ArrayList基本操作

- ArrayList 中的元素实际上是`对象`, 要使用包装类
- add(value); add(index, value); size(); isEmpty(); remove(value); remove(index); get(index); set(index,value); clear(); indexOf(); lastIndexOf();

```java
//System.out.println(arraylist.add("x"));  【增】  尾部插入元素x
//System.out.println(arraylist.add(2,x));   下标为2的地方插入元素x  todo

//System.out.println(arraylist);    打印数组个各元素
//System.out.println(arraylist.size());   计算数组大小， 这个是size，不是length
//System.out.println(arraylist.isEmpty());  用于检查此Arraylist是"空"还是"非空"

//System.out.println(arraylist.remove(2))；【删】 按位置删除，只删除一个元素，从左往右
//System.out.println(arrayList.remove(x));     按值删除，需要遍历整个数组

//boolean ret=arraylist.contains("x");    【查】
//System.out.println("查找到的元素为"+ret);    从前往后遍历，找不到返回-1
//int index=arraylist.indexOf("x");
//System.out.println("查找到的元素的位置为"+index);  第一次出现元素的位置

//String num=arrayList.get(0);    【改】   // todo, 这里出现比较多问题，get和set 修改值
//System.out.println("0号元素为"+num);   获取0号元素
//arrayList.set(0,"php");
//System.out.println("修改后的元素为"+arrayList);   将0号元素修改为php

//for(int i=0;i<=arraylist.size;i++) {        【访问数组下标遍历】
//    System.out.println(arrayList.get(i));   通过下标进行遍历访问数组
//}

//Iterator<String> iterator=arrayList.iterator();  【迭代器遍历】
//while(iterator.hasNext()) {
//  String e =iterator.next();
//    System.out.println("通过迭代器遍历元素"+e);
//   }  由于数组有顺序，所以可以使用访问下标进行遍历，使用迭代器可以遍历图，树

```

### 3. Stack 基本操作

- Object` push`（Object element）：将元素推送到堆栈顶部。
- Object `pop`（）：移除并返回堆栈的顶部元素。如果我们在调用堆栈为空时调用 pop（）, 则抛出’EmptyStackException’异常。
- Object `peek`（）：返回堆栈顶部的元素，但不删除它。
- boolean` empty`（）：如果堆栈顶部没有任何内容，则返回 true。否则，返回 false。
- int `search`（Object element）：确定对象是否存在于堆栈中。如果找到该元素，它将从堆栈顶部返回元素的位置。否则，它返回 - 1。

### 4. LinkedList 基本操作

- LinkedList 实现了 Queue 接口，可作为队列使用。
- LinkedList 实现了 List 接口，可进行列表的相关操作。
- LinkedList 实现了 Deque 接口，可作为队列使用。

| 方法                                           | 描述                                                         |
| :--------------------------------------------- | :----------------------------------------------------------- |
| public boolean `add`(E e)                      | 链表末尾添加元素，返回是否成功，成功为 true，失败为 false。  |
| public void `add`(int index, E element)        | 向指定位置插入元素。                                         |
| public boolean addAll(Collection c)            | 将一个集合的所有元素添加到链表后面，返回是否成功，成功为 true，失败为 false。 |
| public boolean addAll(int index, Collection c) | 将一个集合的所有元素添加到链表的指定位置后面，返回是否成功，成功为 true，失败为 false。 |
| public void `addFirst`(E e)                    | 元素添加到头部。                                             |
| public void `addLast`(E e)                     | 元素添加到尾部。                                             |
| public boolean `offer`(E e)                    | 向`链表末尾添加元素，返回是否成功，成功为 true`，失败为 false。 |
| public boolean `offerFirst`(E e)               | 头部插入元素，返回是否成功，成功为 true，失败为 false。      |
| public boolean` offerLast`(E e)                | 尾部插入元素，返回是否成功，成功为 true，失败为 false。      |
| public void clear()                            | 清空链表。                                                   |
| public E `removeFirst`()                       | 删除并返回第一个元素。                                       |
| public E `removeLast`()                        | 删除并返回最后一个元素。                                     |
| public boolean `remove`(Object o)              | 删除某一元素，返回是否成功，成功为 true，失败为 false。      |
| public E `remove`(int index)                   | 删除指定位置的元素。                                         |
| public E `poll`()                              | 删除并返回第一个元素。                                       |
| public E remove()                              | 删除并返回第一个元素。                                       |
| public boolean `contains`(Object o)            | 判断是否含有某一元素。                                       |
| public E get(int index)                        | 返回指定位置的元素。                                         |
| public E` getFirst`()                          | 返回第一个元素。                                             |
| public E `getLast`()                           | 返回最后一个元素。                                           |
| public int indexOf(Object o)                   | 查找指定元素从前往后第一次出现的索引。                       |
| public int lastIndexOf(Object o)               | 查找指定元素最后一次出现的索引。                             |
| public E `peek`()                              | 返回第一个元素。                                             |
| public E element()                             | 返回第一个元素。                                             |
| public E peekFirst()                           | 返回头部元素。                                               |
| public E peekLast()                            | 返回尾部元素。                                               |
| public E `set`(int index, E element)           | 设置指定位置的元素。                                         |
| public Object clone()                          | 克隆该列表。                                                 |
| public Iterator descendingIterator()           | 返回倒序迭代器。                                             |
| public int size()                              | 返回链表元素个数。                                           |
| public ListIterator listIterator(int index)    | 返回从指定位置开始到末尾的迭代器。                           |
| public Object[] toArray()                      | 返回一个由链表元素组成的数组。                               |
| public T[] toArray(T[] a)                      | 返回一个由链表元素转换类型而成的数组。                       |

### 5. HashSet 基本操作

- HashSet 基于 HashMap 来实现的，是一个不允许有重复元素的集合。
- HashSet 允许有 null 值。
- HashSet 是无序的，即不会记录插入的顺序。
- HashSet 不是线程安全的， 如果多个线程尝试同时修改 HashSet，则最终结果是不确定的。 您必须在多线程访问时显式同步对 HashSet 的并发访问。

```java
 boolean    add(E e) //如果 set 中尚未存在指定的元素，则添加此元素（可选操作）。
 boolean    addAll(Collection<? extends E> c) //如果 set 中没有指定 collection 中的所有元素，则将其添加到此 set 中（可选操作）。
 void    　　clear() //移除此 set 中的所有元素（可选操作）。
 boolean    contains(Object o) //如果 set 包含指定的元素，则返回 true。
 boolean    containsAll(Collection<?> c) //如果此 set 包含指定 collection 的所有元素，则返回 true。
 boolean    equals(Object o) //比较指定对象与此 set 的相等性。
 int    　　 hashCode() //返回 set 的哈希码值。
 boolean    isEmpty() //如果 set 不包含元素，则返回 true。
 Iterator<E>    iterator() //返回在此 set 中的元素上进行迭代的迭代器。
 boolean    remove(Object o) //如果 set 中存在指定的元素，则将其移除（可选操作）。
 boolean    removeAll(Collection<?> c) //移除 set 中那些包含在指定 collection 中的元素（可选操作）。
 boolean    retainAll(Collection<?> c) //仅保留 set 中那些包含在指定 collection 中的元素（可选操作）。
 int    　　 size() //返回 set 中的元素数（其容量）。
 Object[]   toArray() //返回一个包含 set 中所有元素的数组。
<T> T[]    toArray(T[] a) //返回一个包含此 set 中所有元素的数组；返回数组的运行时类型是指定数组的类型。
```

### 6. HashMap 基本操作

- HashMap 是一个散列表，它存储的内容是键值对 (key-value) 映射。
- HashMap 实现了 Map 接口，根据键的 HashCode 值存储数据，具有很快的访问速度，最多允许一条记录的键为 null，不支持线程同步。
- HashMap 是无序的，即不会记录插入的顺序。
- clear(); put(); remove();get(); getOrDefault(); entrySet(); keySet(); valueSet();

| 方法                                                         | 描述                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [clear()](https://www.runoob.com/java/java-hashmap-clear.html) | 删除 hashMap 中的所有键 / 值对                               |
| [clone()](https://www.runoob.com/java/java-hashmap-clone.html) | 复制一份 hashMap                                             |
| [isEmpty()](https://www.runoob.com/java/java-hashmap-isempty.html) | 判断 hashMap 是否为空                                        |
| [size()](https://www.runoob.com/java/java-hashmap-size.html) | 计算 hashMap 中键 / 值对的数量                               |
| [put()](https://www.runoob.com/java/java-hashmap-put.html)   | 将键 / 值对添加到 hashMap 中                                 |
| [putAll()](https://www.runoob.com/java/java-hashmap-putall.html) | 将所有键 / 值对添加到 hashMap 中                             |
| [putIfAbsent()](https://www.runoob.com/java/java-hashmap-putifabsent.html) | 如果 hashMap 中不存在指定的键，则将指定的键 / 值对插入到 hashMap 中。 |
| [remove()](https://www.runoob.com/java/java-hashmap-remove.html) | 删除 hashMap 中指定键 key 的映射关系                         |
| [containsKey()](https://www.runoob.com/java/java-hashmap-containskey.html) | 检查 hashMap 中是否存在指定的 key 对应的映射关系。           |
| [containsValue()](https://www.runoob.com/java/java-hashmap-containsvalue.html) | 检查 hashMap 中是否存在指定的 value 对应的映射关系。         |
| [replace()](https://www.runoob.com/java/java-hashmap-replace.html) | 替换 hashMap 中是指定的 key 对应的 value。                   |
| [replaceAll()](https://www.runoob.com/java/java-hashmap-replaceall.html) | 将 hashMap 中的所有映射关系替换成给定的函数所执行的结果。    |
| [get()](https://www.runoob.com/java/java-hashmap-get.html)   | 获取指定 key 对应对 value                                    |
| [getOrDefault()](https://www.runoob.com/java/java-hashmap-getordefault.html) | 获取指定 key 对应对 value，如果找不到 key ，则返回设置的默认值 |
| [forEach()](https://www.runoob.com/java/java-hashmap-foreach.html) | 对 hashMap 中的每个映射执行指定的操作。                      |
| [entrySet()](https://www.runoob.com/java/java-hashmap-entryset.html) | 返回 hashMap 中所有映射项的集合集合视图。                    |
| [keySet](https://www.runoob.com/java/java-hashmap-keyset.html)() | 返回 hashMap 中所有 key 组成的集合视图。                     |
| [values()](https://www.runoob.com/java/java-hashmap-values.html) | 返回 hashMap 中存在的所有 value 值。                         |
| [merge()](https://www.runoob.com/java/java-hashmap-merge.html) | 添加键值对到 hashMap 中                                      |
| [compute()](https://www.runoob.com/java/java-hashmap-compute.html) | 对 hashMap 中指定 key 的值进行重新计算                       |
| [computeIfAbsent()](https://www.runoob.com/java/java-hashmap-computeifabsent.html) | 对 hashMap 中指定 key 的值进行重新计算，如果不存在这个 key，则添加到 hasMap 中 |
| [computeIfPresent()](https://www.runoob.com/java/java-hashmap-computeifpresent.html) | 对 hashMap 中指定 key 的值进行重新计算，前提是该 key 存在于 hashMap 中。 |

### 7. String 操作

- char[] toCharArray();
- char charAt(int index);
- public int length()
- isEmpty()
- toUpperCase(); toLowerCase(); trim(); 
- Concat();
- split(String regex);
- substring(int beginIndex); substring(int beginIndex, int endIndex);
- contains(CharSequence s);
- indexOf(String str); indexOf(String str,int fromIndex); lastIndexOf(String str);lastIndexOf(String str,int fromIndex);
- startsWith(String prefix); endsWith(String suffix)

### 8. 工具类Arrays

#### .1. 数组填充

-  全部填充 **fill(int[] a,int value)**：该方法可将指定的 int 值分配给 int 型数组 a 的每个元素。
- 局部填充 **fill(int[] a,int fromIndex,int toIndex,int value)**：该方法将指定的 int 值分配给指定的 int 型数组的指定范围中的每个元素。

#### .2. 数组排序

> Arrays.sort()可以直接对基本类型(int、char、double..)数组进行`从小到大的排序`，也可以对包装类类型(Integer、Character、Double..)进行从小到大排序. `只能重写包装类的，不能重写基本类型的 `

```java
char[] a= {'j','c','e','o'};   Arrays.sort(a); 
//自定义排序规则
Character[] a= {'j','c','e','o'};  
Arrays.sort(a, new Comparator<Character>() {  
    public int compare(Character n1,Character n2)  
    {  
        return n1-n2;   // 这里如果返回1， 则颠倒位置， 
    }  
}); 
```

```java
public int[][] reconstructQueue(int[][] people) {
    Arrays.sort(people, new Comparator<int[]>() {
        public int compare(int[] person1, int[] person2) {
            if (person1[0] != person2[0]) {
                return person2[0] - person1[0];  // 按照递减的顺序进行排列  #二维排序
            } else {
                return person1[1] - person2[1];   //按照递增顺序进行排列
            }
        }
    });
    List<int[]> ans = new ArrayList<int[]>();
    //进行插入操作，满足相应的顺序关系
    for (int[] person : people) {
        ans.add(person[1], person);
    }
    // 这一步需要会写
    return ans.toArray(new int[ans.size()][]);
}
```

#### .3. 数组复制

- **copyOf(arr，int newLength)** 是复制数组 arr 至指定长度 newLength，生成一个长度为 newLength 的新数组。如果 newLength 的长度大于 arr 的长度，那么则用 0 填充（根据数组的类型来决定填充值，整型用 0，char 型用 null 来填充）
- **copyOfRange(arr,int fromIndex,int toIndex)** 则是将指定数组的指定长度生成为一个新数组，超过长度同样会进行填充，范围取值同样也是左闭右开。

#### .4. 数组查询

- 全部搜索 **binarySearch(Objcet[] a,Object key)**
- 局部搜索 **binarySearch(Objcet[] a，int fromIndex,int toIndex,Object key)**

#### .5. 数组和ArrayList转化

```java
String[] array = { "1", "2", "3" };
List<String> list = null;
// ------ 方法一： for循环
list = Arrays.asList(array);
// List 转变为数组
array = list.toArray(new String[0]);
list.toArray(new int[0][]);
Arrays.toString(int[ ] array); //会以字符串的形式返回数组的全部值而无需手动遍历，一般用于打印数组.
```

### 9. 工具类Collections

#### .1. sort()

```java
ArrayList<Character> list=new ArrayList<Character>();  
for(int i=0;i<a.length;i++)  
{  
    list.add(a[i]);  
}  
Collections.sort(list,new Comparator<Character>() {  
    public int compare(Character c1,Character c2)  
    {  
        if(c1>c2)  
            return -1;  
        else if(c1<c2) return 1;  
        else return 0;  
    }  
});
```

#### .2. 自定义sort--compareTo

```java
public class Employee implements Comparable<Employee> {  
    private int id;  
    private String name;  
    private int age;  
    public Employee(int id, String name, int age) { //利用构造方法初始化各个域  
        this.id = id;  
        this.name = name;  
        this.age = age;  
    }  
    @Override  
    public int compareTo(Employee o) {              //利用编号实现对象间的比较  
        if (id > o.id) {  
            return 1;  
        } else if (id < o.id) {  
            return -1;  
        }  
        return 0;  
    }  
    @Override  
    public String toString() {                      //重写toString()方法  
        StringBuilder sb = new StringBuilder();  
        sb.append("员工的编号：" + id + "，");  
        sb.append("员工的姓名：" + name + "\t，");  
        sb.append("员工的年龄：" + age);  
        return sb.toString();  
    }  
}  
```

```java

package com.ldd.second.shuzu;
 
import java.util.TreeMap;
/**
 * @author liudongdong19
 * @description TreeMap 介绍
 *  Map接口派生了一个SortedMap子接口，SortedMap有一个TreeMap实现类。
 *  TreeMap是基于红黑树对TreeMap中所有key进行排序，从而保证TreeMap中所有key-value对处于有序状态。TreeMap也有两种排序方式：
 *  自然排序：TreeMap的所有key必须实现Comparable接口，而且所有key应该是同一个类的对象，否则将会抛出ClassCastException异常。
 *  定制排序：创建TreeMap时，传入一个Comparator对象，该对象负责对TreeMap中所有key进行排序。采用定制排序时不要求Map的key实现Comparable接口。
 * */
public class MyTreeMap implements Comparable  {
    int count;
    public MyTreeMap(int count)
    {
        this.count = count;
    }
    public String toString()
    {
        return "MytreeMap [count:" + count + "]";
    }
    // 根据count来判断两个对象是否相等。
    public boolean equals(Object obj)
    {
        if (this == obj)
            return true;
        if (obj != null && obj.getClass() ==MyTreeMap.class)
        {
            MyTreeMap myTreeMap = (MyTreeMap)obj;
            return myTreeMap.count == this.count;
        }
        return false;
    }
    // 根据count属性值来判断两个对象的大小。
    public int compareTo(Object obj)
    {
        MyTreeMap myTreeMap = (MyTreeMap) obj;
        return count > myTreeMap.count ? 1 :
                count < myTreeMap.count ? -1 : 0;
    }
}
class TreeMapTest
{
    public static void main(String[] args)
    {
        TreeMap tm = new TreeMap();
        tm.put(new MyTreeMap(3) , "兰花");
        tm.put(new MyTreeMap(-5) , "龟背竹");
        tm.put(new MyTreeMap(9) , "紫罗兰");
        System.out.println(tm);
        // 返回该TreeMap的第一个Entry对象
        System.out.println(tm.firstEntry());
 
    }
}
```

#### .3.shuffle()

- 将 List 随机打乱顺序，底层会根据传入的是 ArrayList 还是 LinkedList 进行不同的处理。

#### .4.binarySearch()

- 以二分的算法查找指定元素，注意`最好将 List 变为有序的`，否则可能会找不到，并且返回值也可能与预期值不一致。

#### .5.max ()与min ()

- 采用 Collections 内含的比较法获取最大与最小值，当然也可以传入自定义的实现 Comparator 接口比较

#### .6.indexOfSubList () 与lastIndexOfSubList ()

- 参数是两个 list，第二个是子 list，分别返回的是 subList 在 list 中第一次出现和最后一次出现位置的索引

#### .7. replaceAll()

- 替换指定元素为某元素，若要替换的值（旧元素）存在刚返回 true，反之返回 false

#### .8.reverse()

- 反转集合中元素的顺序

#### .9. fill(strings1,“object”)

- 用 object 替换 string1 中的所有元素

#### .10. swap(strings1,2,3)

- 交换第 2 和第 3 个元素的位置

#### .11. **rotate (List list,int m)**

- 集合中的元素向后移 m 个位置，在后面被遮盖的元素循环到前面来
- 移动列表中的元素，负数向左移动，正数向右移动

```java
public static void main(String[] args){
        ArrayList<Integer> intList = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        System.out.println(intList);
        Collections.rotate(intList, 1);
        System.out.println(intList);
}
运行结果:[1, 2, 3, 4, 5]
        [5, 1, 2, 3, 4]
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/jdk_collection0/  

