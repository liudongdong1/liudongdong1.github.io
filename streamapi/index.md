# StreamAPI


> **代码简洁**函数式编程写出的代码简洁且意图明确，使用*stream*接口让你从此告别*for*循环。
>
> **多核友好**，Java函数式编程使得编写并行程序从未如此简单，你需要的全部就是调用一下`parallel()`方法。
>
> stream通常不会手动创建，而是调用对应的工具方法，比如：
>
> - 调用`Collection.stream()`或者`Collection.parallelStream()`方法
> - 调用`Arrays.stream(T[] array)`方法

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/Java_stream_Interfaces-163056525279424.png)

> - **无存储**。stream不是一种数据结构，它只是`某种数据源的一个视图`，数据源可以是一个`数组，Java容器或I/O channel`等。
> - **为函数式编程而生**。对`stream的任何修改都不会修改背后的数据源`，比如对*stream*执行过滤操作并不会删除被过滤的元素，而是会`产生一个不包含被过滤元素的新stream`。
> - **惰式执行**。*stream*上的操作并不会立即执行，只有等到用户真正需要结果的时候才会执行。
> - **可消费性**。*stream*只能被“消费”一次，`一旦遍历过就会失效`，就像容器的迭代器那样，想要再次遍历必须重新生成。

| 操作类型   | 接口方法                                                     |
| ---------- | ------------------------------------------------------------ |
| `中间操作` | concat() distinct() filter() flatMap() limit() map() peek() <br> skip() sorted() parallel() sequential() unordered() |
| `结束操作` | allMatch() anyMatch() collect() count() findAny() findFirst() <br> forEach() forEachOrdered() max() min() noneMatch() reduce() toArray() |

> 1. __中间操作总是会惰式执行__，调用中间操作只会生成一个标记了该操作的新*stream*，仅此而已。
> 2. __结束操作会触发实际计算__，计算发生时会把所有中间操作积攒的操作以*pipeline*的方式执行，这样可以减少迭代次数。计算完成之后*stream*就会失效。

### 1. Stream 方法使用

#### .1. forEach

```java
// 使用Stream.forEach()迭代
Stream<String> stream = Stream.of("I", "love", "you", "too");
stream.forEach(str -> System.out.println(str));
```

#### .2. filter

```java
// 保留长度等于3的字符串
Stream<String> stream= Stream.of("I", "love", "you", "too");
stream.filter(str -> str.length()==3)
    .forEach(str -> System.out.println(str));
```

#### .3. distinct

```java
Stream<String> stream= Stream.of("I", "love", "you", "too", "too");
stream.distinct()
    .forEach(str -> System.out.println(str));
```

#### .4. sorted

```java
Stream<String> stream= Stream.of("I", "love", "you", "too");
//输出按照长度升序排序后的字符串，
stream.sorted((str1, str2) -> str1.length()-str2.length())   
    .forEach(str -> System.out.println(str)); 
```

#### .5. map

> 对每个元素按照某种操作进行转换，转换前后`Stream`中元素的个数不会改变，但元素的类型取决于转换之后的类型。

```java
Stream<String> stream　= Stream.of("I", "love", "you", "too");
stream.map(str -> str.toUpperCase())
    .forEach(str -> System.out.println(str));
```

#### .6. flatmap

> `flatMap()`的作用就相当于把原*stream*中的所有元素都"摊平"之后组成的`Stream`，转换前后元素的个数和类型都可能会改变。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/Stream.flatMap-163056568062925.png)

```java
Stream<List<Integer>> stream = Stream.of(Arrays.asList(1,2), Arrays.asList(3, 4, 5));
stream.flatMap(list -> list.stream())
    .forEach(i -> System.out.println(i));
```

### .2. Reduction operation

> 规约操作（*reduction operation*）又被称作折叠操作（*fold*），是通过`某个连接动作将所有元素汇总成一个汇总结果的过程`。元素求和、求最大值或最小值、求出元素总个数、将所有元素转换成一个列表或集合，都属于规约操作。*Stream*类库有两个通用的规约操作`reduce()`和`collect()`，也有一些为简化书写而设计的专用规约操作，比如`sum()`、`max()`、`min()`、`count()`等。

#### .1. reduce

> *reduce*操作可以实现`从一组元素中生成一个值`，`sum()`、`max()`、`min()`、`count()`等都是*reduce*操作，将他们单独设为函数只是因为常用。`reduce()`的方法定义有三种重写形式：
>
> - `Optional<T> reduce(BinaryOperator<T> accumulator)`
> - `T reduce(T identity, BinaryOperator<T> accumulator)`
> - `<U> U reduce(U identity, BiFunction<U,? super T,U> accumulator, BinaryOperator<U> combiner)`

```java
// 找出最长的单词
Stream<String> stream = Stream.of("I", "love", "you", "too");
Optional<String> longest = stream.reduce((s1, s2) -> s1.length()>=s2.length() ? s1 : s2);
//Optional<String> longest = stream.max((s1, s2) -> s1.length()-s2.length());
System.out.println(longest.get());
```

```java
// 求单词长度之和
Stream<String> stream = Stream.of("I", "love", "you", "too");
Integer lengthSum = stream.reduce(0,　// 初始值　// (1)
        (sum, str) -> sum+str.length(), // 累加器 // (2)
        (a, b) -> a+b);　// 部分和拼接器，并行执行时才会用到 // (3)
// int lengthSum = stream.mapToInt(str -> str.length()).sum();
System.out.println(lengthSum);
```

#### .2. collect

> 收集器（*Collector*）是为`Stream.collect()`方法量身打造的工具接口（类）。考虑一下将一个*Stream*转换成一个容器（或者*Map*）需要做哪些工作？我们至少需要两样东西：
>
> 1. 目标容器是什么？是*ArrayList*还是*HashSet*，或者是个*TreeMap*。
> 2. 新元素如何添加到容器中？是`List.add()`还是`Map.put()`。
>
> 如果并行的进行规约，还需要告诉*collect()* 3. 多个部分结果如何合并成一个。

##### .1. collect()生成Collection

```java
// 将Stream转换成List或Set
Stream<String> stream = Stream.of("I", "love", "you", "too");
List<String> list = stream.collect(Collectors.toList()); // (1)
Set<String> set = stream.collect(Collectors.toSet()); // (2)

// 使用toCollection()指定规约容器的类型
ArrayList<String> arrayList = stream.collect(Collectors.toCollection(ArrayList::new));// (3)
HashSet<String> hashSet = stream.collect(Collectors.toCollection(HashSet::new));// (4)
```

##### .2. collect()生成Map

> 1. 使用`Collectors.toMap()`生成的收集器，用户需要指定如何生成*Map*的*key*和*value*。
> 2. 使用`Collectors.partitioningBy()`生成的收集器，对元素进行二分区操作时用到。
> 3. 使用`Collectors.groupingBy()`生成的收集器，对元素做*group*操作时用到。

情况1：使用`toMap()`生成的收集器，这是和`Collectors.toCollection()`并列的方法。如下代码展示将学生列表转换成由<学生，GPA>组成的*Map*。

```Java
// 使用toMap()统计学生GPA
Map<Student, Double> studentToGPA =
     students.stream().collect(Collectors.toMap(Function.identity(),// 如何生成key
                                     student -> computeGPA(student)));// 如何生成value
```

情况2：使用`partitioningBy()`生成的收集器，这种情况适用于将`Stream`中的元素依据某个二值逻辑（满足条件，或不满足）分成互补相交的两部分，比如男女性别、成绩及格与否等。下列代码展示将学生分成成绩及格或不及格的两部分。

```Java
// Partition students into passing and failing
Map<Boolean, List<Student>> passingFailing = students.stream()
         .collect(Collectors.partitioningBy(s -> s.getGrade() >= PASS_THRESHOLD));
```

情况3：使用`groupingBy()`生成的收集器，跟SQL中的*group by*语句类似，这里的*groupingBy()*也是按照某个属性对数据进行分组，属性相同的元素会被对应到*Map*的同一个*key*上。下列代码展示将员工按照部门进行分组：

```Java
// Group employees by department
Map<Department, List<Employee>> byDept = employees.stream()
            .collect(Collectors.groupingBy(Employee::getDepartment));
```

比如求和、计数、平均值、类型转换等。这种先将元素分组的收集器叫做**上游收集器**，之后执行其他运算的收集器叫做**下游收集器**(*downstream Collector*)。

```Java
// 使用下游收集器统计每个部门的人数
Map<Department, Integer> totalByDept = employees.stream()
                    .collect(Collectors.groupingBy(Employee::getDepartment,
                                                   Collectors.counting()));// 下游收集器
```

考虑将员工按照部门分组的场景，如果*我们想得到每个员工的名字（字符串），而不是一个个*Employee*对象*，可通过如下方式做到：

```Java
// 按照部门对员工分布组，并只保留员工的名字
Map<Department, List<String>> byDept = employees.stream()
                .collect(Collectors.groupingBy(Employee::getDepartment,
                        Collectors.mapping(Employee::getName,// 下游收集器
                                Collectors.toList())));// 更下游的收集器
```

#### .3. collect()做字符串join

```java
// 使用Collectors.joining()拼接字符串
Stream<String> stream = Stream.of("I", "love", "you");
//String joined = stream.collect(Collectors.joining());// "Iloveyou"
//String joined = stream.collect(Collectors.joining(","));// "I,love,you"
String joined = stream.collect(Collectors.joining(",", "{", "}"));// "{I,love,you}"
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/streamapi/  

