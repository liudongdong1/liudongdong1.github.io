# Python Property


> Python内置的@property装饰器就是负责把一个方法变成属性调用的, **属性**是对事物某种特性的抽象，面向对象编程中一个重要概念；区别于字段，它通常表示为字段的扩展，加以访问与设置保护机制。

### 1. 什么是property属性

一种用起来像是使用的实例属性一样的特殊属性，可以对应于某个方法

```python
# ############### 定义 ###############
class Foo:
    def func(self):
        pass
    # 定义property属性
    @property
    def prop(self):
        pass
# ############### 调用 ###############
foo_obj = Foo()
foo_obj.func()  # 调用实例方法
foo_obj.prop  # 调用property属性
```



### 2.为什么使用property属性

在绑定属性时，如果我们直接把属性暴露出去，虽然写起来很简单，但是，没办法检查参数，导致可以把成绩随便改：

```python
s = Student()
s.score = 9999
```

但是，上面的调用方法又略显复杂，没有直接用属性这么直接简单。

有没有既能检查参数，又可以用类似属性这样简单的方式来访问类的变量呢？对于追求完美的Python程序员来说，这是必须要做到的！

```python
class Student(object):
    def get_score(self):
        return self._score
    def set_score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
调用：
s = Student()
s.set_score(60) # ok!
s.get_score()
60
s.set_score(9999)
Traceback (most recent call last):
  ...
ValueError: score must between 0 ~ 100!
```



还记得装饰器（decorator）可以给函数动态加上功能吗？对于类的方法，装饰器一样起作用。Python内置的`@property`装饰器就是负责把一个方法变成属性调用的：

```python
class Student(object):
    @property
    def score(self):
        return self._score
    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```

`@property`的实现比较复杂，我们先考察如何使用。把一个getter方法变成属性，只需要加上`@property`就可以了，此时，`@property`本身又创建了另一个装饰器`@score.setter`，负责把一个setter方法变成属性赋值，于是，我们就拥有一个可控的属性操作：

```python
s = Student()
s.score = 60 # OK，实际转化为s.set_score(60)
s.score # OK，实际转化为s.get_score()
60
s.score = 9999
Traceback (most recent call last):
  ...
ValueError: score must between 0 ~ 100!
```

### 3. property属性的有两种方式

- 装饰器 即：在方法上应用装饰器
- 类属性 即：在类中定义值为property对象的类属性

#### 3.1 装饰器方式

在类的实例方法上应用@property装饰器

```python
#coding=utf-8
# ############### 定义 ###############
class Goods:
    """python3中默认继承object类
        以python2、3执行此程序的结果不同，因为只有在python3中才有@xxx.setter  @xxx.deleter
    """
    @property
    def price(self):
        print('@property')
    @price.setter
    def price(self, value):
        print('@price.setter')
    @price.deleter
    def price(self):
        print('@price.deleter')
# ############### 调用 ###############
obj = Goods()
obj.price          # 自动执行 @property 修饰的 price 方法，并获取方法的返回值
obj.price = 123    # 自动执行 @price.setter 修饰的 price 方法，并将  123 赋值给方法的参数
del obj.price      # 自动执行 @price.deleter 修饰的 price 方法
```



#### 3.2 类属性方式，创建值为PROPERTY对象的类属性

property方法中有个四个参数

- 第一个参数是方法名，调用 对象.属性 时自动触发执行方法
- 第二个参数是方法名，调用 对象.属性 ＝ XXX 时自动触发执行方法
- 第三个参数是方法名，调用 del 对象.属性 时自动触发执行方法
- 第四个参数是字符串，调用 对象.属性.__doc__ ，此参数是该属性的描述信息

```python
#coding=utf-8
class Foo(object):
    def get_bar(self):
        print("getter...")
        return 'laowang'
    def set_bar(self, value): 
        """必须两个参数"""
        print("setter...")
        return 'set value' + value
    def del_bar(self):
        print("deleter...")
        return 'laowang'
    BAR = property(get_bar, set_bar, del_bar, "description...")
obj = Foo()
obj.BAR  # 自动调用第一个参数中定义的方法：get_bar
obj.BAR = "alex"  # 自动调用第二个参数中定义的方法：set_bar方法，并将“alex”当作参数传入
desc = Foo.BAR.__doc__  # 自动获取第四个参数中设置的值：description...
print(desc)
del obj.BAR  # 自动调用第三个参数中定义的方法：del_bar方法
```

**[参考]**

- https://docs.python.org/release/2.6/library/functions.html#property
- https://www.tianqiweiqi.com/python-property.html

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/python-%E5%86%85%E7%BD%AE%E5%B1%9E%E6%80%A7%E5%87%BD%E6%95%B0/  

