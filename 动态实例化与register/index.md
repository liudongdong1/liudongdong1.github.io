# 动态实例化与Register


> 1. 写**具体的网络结构**，它往往是一个Class，并且往往是一个单独的文件
>
> 2. **配置文件**中会指定我们使用哪一个网络结构，往往是Class name
>
> 3. 在训练过程的某一个地方，会根据配置文件指定的Class name，**自动实例化相关的类**
>
> 转自：https://zhuanlan.zhihu.com/p/414759881

### 1. if-else

> 在构建网络结构的过程中，调用一个判断，根据配置文件中的信息，自动做出选择。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211023162738614.png)

### 2. 自动化实例

> 动态加载模块用的模块importlib 里面的`import_module(“字符串模块路径”， 相对路径包名)`
>
> - `hasattr `判断对象是否有什么属性和方法 返回真或假
> - `getattr(对象，“字符串属性”，默认值)`
>   - 如果 获取的是实例对象,获取实例方法，不用传实例直接可以调用
>   - 如果 获取的是类的实例方法，需要传实例对象 传给函数。一般可以类（）当做实例化
>   - 如果是实例对象或者类,去调用静态方法， 都不用给函数传self cls
>   - 如果没有字符串属性 可以给个默认值，不会抛出AttributeError 异常
> - setattr(对象， “字符串属性”， value)： value 可以是任何数据类型，重点是`value可以是一个函数的内存地址`，`如果value是一个函数，调用的时候需要把self或者cls传进去`.
> - delattr(对象，“属性名”)
>   - 只有类可以删除类的实例方法或者静态方法
>   - 只有类可以删除类属性
>   - 实例只能删除实例属性

```python
# import importlib
# from common.settings import module_name
# from common.constants import CONFIG_FILE_PATH
# # name参数是一个字符串 package是个包 一般相对路径的时候有用
#
# # s = importlib.import_module(module_name)
# s = importlib.import_module("scripts.handle_config")
# print(s.HandleConfig) # 获取类
#
# print(s.HandleConfig(CONFIG_FILE_PATH).get_value("file path", "cases_path"))
# 反射
import pprint
a = [1,3]
b = 3
# pprint.pprint(dir(a))
# hasattr 这个是判断 对象是不是有什么属性 方法
if hasattr(a, "__iter__"):
    print(a.__iter__)
    for i in a:
        print(i)
# 获取 对象的 属性和方法

class Dog:
    age = 1
    name = "kafei"
    def eat(self, x):
        print(f"正在吃{x}")
    @classmethod
    def lasi(cls, x):
        print(f"狗拉的是{x}")
# pprint.pprint(dir(Dog))
print(getattr(Dog, "age"))
print(getattr(Dog, "name"))
e = getattr(Dog, "eat")
print(e) # 实例方法返回的是一个函数
e(Dog(),"西瓜") # 直接调用需要传一个实例对象
p = Dog()

e1 = getattr(p, "eat")
e1("面包")
getattr(Dog, "lasi")

getattr(Dog, "lasi")("黄金")
pass

# setattr(对象，字符串的属性名， value)  value:可以是任何类型   给他 对象

def shuijiao(self):
    print("111111111111")
    return  1

# setattr(p, "shuijiao", shuijiao) # 给类添加实例方法，那么调用外部添加在方法时候必须把实力传给外部函数
setattr(Dog, "shuijiao", shuijiao)  # 给类添加类方法，那么调用外部添加在方法时候必须把类传给外部函数
pass

# delattr(对象， 字符串属性)

delattr(Dog, "age")
# delattr(p, "name") # 实例对象匀删除 类属性 会抛出异常AttributeError

delattr(Dog, 'eat')  # 删除方法必须用类去删， 实例对象删除不了
delattr(Dog, 'lasi')
pass

"""
总结：
动态加载模块用的模块importlib 里面的import_module(“字符串模块路径”， 相对路径包名)
1. hasattr 判断对象是否有什么属性和方法 返回真或假
2.getattr(对象，“字符串属性”，默认值) 
* 如果 获取的是实例对象获取实例方法，不用传实例直接可以调用
* 如果 获取的是类的实例方法，需要传实例对象 传给函数。一般可以类（）当做实例化
* 如果是实例对象或者类 去调用静态方法， 都不用给函数传self cls

* 如果没有字符串属性 可以给个默认值，不会抛出AttributeError 异常

3. setattr(对象， “字符串属性”， value)
value 可以是任何数据类型，重点是value可以是一个 函数的内存地址
如果value是一个函数，调用的时候需要把self或者cls传进去

4. delattr(对象，“属性名”)

只有类可以删除 类的实例方法或者静态方法
只有类可以删除 类属性
实例只能删除 实例属性

"""
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211023163743462.png)

> 先扫描相关文件，把文件中所有的类、函数都import进来，然后调用getattr动态实例化，这个方法有两个问题：
>
> 1. 如果有重名怎么办？我就遇到了这个问题，不小心类有重名了，它load错了类，然后怎么改都没用，因为实例化了同名的但不是我想要的类嘛
>
> 2. 它会把文件中所有的类、函数都import进来，就很冗余，因为很多类和函数都是中间的量

### 3. Register

```python
class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """
    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}
    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj
    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco
        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)
    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret
    def __contains__(self, name):
        return name in self._obj_map
    def __iter__(self):
        return iter(self._obj_map.items())
    def keys(self):
        return self._obj_map.keys()

DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211023164336359.png)

> 1. 写**具体的网络结构 (以_arch.py结尾)**，它往往是一个Class (加上@ARCH_REGISTRY.register()的装饰器)，并且往往是一个单独的文件
>
> 2. **配置文件**中会指定我们使用了哪一个网络结构，往往是Class name

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%8A%A8%E6%80%81%E5%AE%9E%E4%BE%8B%E5%8C%96%E4%B8%8Eregister/  

