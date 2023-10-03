# weekmodel


### 1. Event

> threading.Event可以`使一个线程等待其他线程的通知`。其内置了一个`标志，初始值为False`。线程`通过wait()方法进入等待状态`，直到`另一个线程调用set()方法将内置标志设置为True`时，Event通知所有等待状态的线程恢复运行；调用`clear()时重置为 False`。还可以通过`isSet()方法查询Envent对象内置状态的当前值`。Event没有锁，无法使线程进入同步阻塞状态.
>
> - isSet(): 当内置标志为True时返回True。
> - set(): 将标志设为True，并通知所有处于等待阻塞状态的线程恢复运行状态。
> - clear(): 将标志设为False。
> - wait([timeout]): 如果标志为True将立即返回，否则阻塞线程至等待阻塞状态，等待其他线程调用set()。

```python
import threading
import time
event = threading.Event()
def light():
    print('红灯正亮着')
    time.sleep(3)
    event.set() # 模拟绿灯亮
def car(name):
    print('车%s正在等绿灯' % name)
    event.wait() # 模拟等绿灯的操作，此时event为False,直到event.set()将其值设置为True,才会继续运行
    print('车%s通行' % name)
if __name__ == '__main__':
    print("嗨客网(www.haicoder.net)")
    # 红绿灯
    t1 = threading.Thread(target=light)
    t1.start()
    # 车
    for i in range(3):
        t = threading.Thread(target=car, args=(i,))
        t.start()
```

### 2. Timer

> 函数：Timer(interval, function, args=[ ], kwargs={ })
>
> - interval: 指定的时间
> - function: 要执行的方法
> - args/kwargs: 方法的参数

```python
import threading


def func(num):
    print('hello {} timer!'.format(num))

# 如果t时候启动的函数是含有参数的，直接在后面传入参数元组
timer = threading.Timer(5, func,(1,))
time0 = time.time()
timer.start()
print(time.time()-time0)
#------------------------------------------------
#0.0
#hello 1 timer!
```

### 3. Local

> 让他们在每个线程中一直存在，相当于一个线程内的共享变量，线程之间又是隔离的。 python threading模块中就提供了这么一个类，叫做local。local是一个小写字母开头的类，用于管理 thread-local（线程局部的）数据。对于同一个local，线程无法访问其他线程设置的属性；线程设置的属性不会被其他线程设置的同名属性替换。

```python
import threading

# Threading.local对象
localManager = threading.local()
lock = threading.RLock()

class MyThead(threading.Thread):
    def __init__(self, threadName, name):
        super(MyThead, self).__init__(name=threadName)
        self.__name = name

    def run(self):
        global localManager
        localManager.ThreadName = self.name
        localManager.Name = self.__name
        MyThead.ThreadPoc()
    # 线程处理函数
    @staticmethod
    def ThreadPoc():
        lock.acquire()
        try:
            print('Thread={id}'.format(id=localManager.ThreadName))
            print('Name={name}'.format(name=localManager.Name))
        finally:
            lock.release()

if __name__ == '__main__':
    bb = {'Name': 'bb'}
    aa = {'Name': 'aa'}
    xx = (aa, bb)
    threads = [MyThead(threadName='id_{0}'.format(i), name=xx[i]['Name']) for i in range(len(xx))]
    for i in range(len(threads)):
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
--------------------------------------------------------------
#Thread=id_0
#Name=aa
#Thread=id_1
#Name=bb
```

### 4. thread

> `_thread` 模块提供了最基本的 **[线程](https://haicoder.net/python/python-thread-process.html)** 和互斥锁支持

| 方法                                                 | 说明                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| _thread.start_new_thread (function,args,kwargs=None) | 派生一个新的线程，给定agrs和kwargs来执行function             |
| _thread.allocate_lock()                              | 分配锁对象                                                   |
| _thread.exit()                                       | 线程退出                                                     |
| lock.acquire(waitflag=1, timeout=-1)                 | 获取锁对象                                                   |
| lock.locked()                                        | 如果获取了锁对象返回true，否则返回false                      |
| lock.release()                                       | 释放锁                                                       |
| _thread.LockType()                                   | 锁对象类型                                                   |
| _thread.get_ident()                                  | 获取线程标识符                                               |
| _thread.TIMEOUT_MAX                                  | lock.acquire的最大时间，超时将引发OverflowError              |
| _thread.interrupt_main()                             | 引发主线程KeyboardInterrupt错误，子线程可以用这个函数来终止主线程 |

> 使用 threading 模块创建多线程有两种方式，即直接使用线程处理函数创建与 **[继承](https://haicoder.net/python/python-slots-extends.html)** threading.Thread 类实现多线程。

```python
import threading
import time
class MyThread(threading.Thread):
    def __init__(self, name, n):
        super(MyThread, self).__init__()
        self.name = name
        self.n = n
    def run(self):
        while True:
            print("Thread ", self.name, "is running")
            time.sleep(self.n)
def main():
    t1 = MyThread("t1", 3)
    t2 = MyThread("t2", 2)
    t1.start()
    t2.start()
if __name__ == "__main__":
    print("嗨客网(www.haicoder.net)")
    main()
```

### 5. 守护进程

> 在 **[Python](https://haicoder.net/python/python-tutorial.html)** 中，**[线程](https://haicoder.net/python/python-thread.html)** 分为三种形式，即主线程、守护线程和非守护线程。主线程也叫 **[main](https://haicoder.net/python/python-function-main.html)** 线程，主线程不是守护线程。守护线程是指在程序运行的时候在后台提供一种通用服务的线程，非守护线程也叫用户线程，是由用户创建的线程。

```python
class MyThread(threading.Thread):
    def __init__(self, params):
        pass
    def run(self):
       pass
t1 = MyThread(params)
t.setDaemon(True)
t1.start()
```

### 6. Join

> Thread.join([timeout])； 如果当前线程运行时间未超过 timeout，那么就一直等待线程结束，如果当前线程运行时间已经超过 timeout，那么主线程就不再继续等待，timeout 参数如果不传递，则是永久等待，直到子线程退出。

```python
import threading
import time
class MyThread(threading.Thread):
    def __init__(self, name, n):
        super(MyThread, self).__init__()
        self.name = name
        self.n = n
    def run(self):
        print("Thread ", self.name, "is running")
        time.sleep(self.n)
        print("Thread ", self.name, "exit")
def main():
    t1 = MyThread("t1", 30)
    t1.setDaemon(True)
    t1.start()
    t2 = MyThread("t2", 2)
    t2.setDaemon(True)
    t2.start()
    t1.join(5)
    t2.join()
if __name__ == "__main__":
    print("嗨客网(www.haicoder.net)")
    main()
```

### 7. 全局变量

```python
import threading
g_num = 100
def handler_incry():
    global g_num
    for i in range(2):
        g_num += 1
        print("in handler_incry g_num is : %d" % g_num)
def handler_decry():
    global g_num
    for i in range(2):
        g_num -= 1
        print("in handler_decry g_num is : %d" % g_num)
def main():
    print("嗨客网(www.haicoder.net)")
    t1 = threading.Thread(target=handler_incry)
    t1.start()
    t2 = threading.Thread(target=handler_decry)
    t2.start()
if __name__ == '__main__':
    main()
```

### 8. 互斥锁

```python
from threading import Thread,Lock
# 创建互斥锁
lock = threading.Lock()
# 对需要访问的资源加锁
lock.acquire()
# 资源访问结束解锁
lock.release()
```

```python
import threading
num = 0
# 创建互斥锁
lock = threading.Lock()
def handler_incry():
    global num
    lock.acquire()
    for i in range(100000):
        num += 1
    print("handler_incry done, num =", num)
    lock.release()
def handler_decry():
    global num
    lock.acquire()
    for i in range(100000):
        num -= 1
    print("handler_decry done, num =", num)
    lock.release()
if __name__ == '__main__':
    print("嗨客网(www.haicoder.net)")
    # 创建线程
    t1 = threading.Thread(target=handler_incry)
    t2 = threading.Thread(target=handler_decry)
    # 启动线程
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```

### 9. 递归锁

> Python 的递归锁，即 threading.RLock。RLock 内部维护着一个 Lock 和一个 counter **[变量](https://haicoder.net/python/python-variable.html)**，counter 记录了 acquire 的次数，从而使得资源可以被多次 acquire。直到一个线程所有的 acquire 都被 release，其他的线程才能获得资源。

```python
import threading
num = 0
# 创建递归锁
lock = threading.RLock()
def handler_incry():
    global num
    lock.acquire()
    lock.acquire()
    for i in range(100000):
        num += 1
    print("handler_incry done, num =", num)
    lock.release()
    lock.release()
def handler_decry():
    global num
    lock.acquire()
    lock.acquire()
    for i in range(100000):
        num -= 1
    print("handler_decry done, num =", num)
    lock.release()
    lock.release()
if __name__ == '__main__':
    print("嗨客网(www.haicoder.net)")
    # 创建线程
    t1 = threading.Thread(target=handler_incry)
    t2 = threading.Thread(target=handler_decry)
    # 启动线程
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```

### 10. Condition

> 实现条件变量对象的类。一个条件变量对象允许一个或多个线程在被其它线程所通知之前进行等待。
>
> - `acquire`(**args*)
>
>   请求底层锁。此方法调用底层锁的相应方法，返回值是底层锁相应方法的返回值。
>
> - `release`()
>
>   释放底层锁。此方法调用底层锁的相应方法。没有返回值。
>
> - `wait`(*timeout=None*)
>
>   等待直到被通知或发生超时。如果线程在调用此方法时没有获得锁，将会引发 [`RuntimeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#RuntimeError) 异常。这个方法释放底层锁，然后阻塞，直到在另外一个线程中调用同一个条件变量的 [`notify()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.notify) 或 [`notify_all()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.notify_all) 唤醒它，或者直到可选的超时发生。一旦被唤醒或者超时，它重新获得锁并返回。当提供了 *timeout* 参数且不是 `None` 时，它应该是一个浮点数，代表操作的超时时间，以秒为单位（可以为小数）。当底层锁是个 [`RLock`](https://docs.python.org/zh-cn/3/library/threading.html#threading.RLock) ，不会使用它的 [`release()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.release) 方法释放锁，因为当它被递归多次获取时，实际上可能无法解锁。相反，使用了 [`RLock`](https://docs.python.org/zh-cn/3/library/threading.html#threading.RLock) 类的内部接口，即使多次递归获取它也能解锁它。 然后，在重新获取锁时，使用另一个内部接口来恢复递归级别。返回 `True` ，除非提供的 *timeout* 过期，这种情况下返回 `False`。*在 3.2 版更改:* 很明显，方法总是返回 `None`。
>
> - `wait_for`(*predicate*, *timeout=None*)
>
>   等待，直到条件计算为真。 *predicate* 应该是一个可调用对象而且它的返回值可被解释为一个布尔值。可以提供 *timeout* 参数给出最大等待时间。这个实用方法会重复地调用 [`wait()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.wait) 直到满足判断式或者发生超时。返回值是判断式最后一个返回值，而且如果方法发生超时会返回 `False` 
>
> - `notify`(*n=1*)
>
>   默认唤醒一个等待这个条件的线程。如果调用线程在没有获得锁的情况下调用这个方法，会引发 [`RuntimeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#RuntimeError) 异常。这个方法唤醒最多 *n* 个正在等待这个条件变量的线程；如果没有线程在等待，这是一个空操作。当前实现中，如果至少有 *n* 个线程正在等待，准确唤醒 *n* 个线程。但是依赖这个行为并不安全。未来，优化的实现有时会唤醒超过 *n* 个线程。注意：被唤醒的线程实际上不会返回它调用的 [`wait()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.wait) ，直到它可以重新获得锁。因为 [`notify()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.notify) 不会释放锁，只有它的调用者应该这样做。
>
> - `notify_all`()
>
>   唤醒所有正在等待这个条件的线程。这个方法行为与 [`notify()`](https://docs.python.org/zh-cn/3/library/threading.html#threading.Condition.notify) 相似，但并不只唤醒单一线程，而是唤醒所有等待线程。如果调用线程在调用这个方法时没有获得锁，会引发 [`RuntimeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#RuntimeError) 异常。

- [python 基础](https://haicoder.net/python/python-event.html)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/threadrelative/  

