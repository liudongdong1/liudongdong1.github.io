# globalVariable&Lock


#### 1. 本文件中

```python
# 定义全局变量
num = 1
# 使用全局变量，并对其赋值
def fun1():
    global num
    print("num= " + str(num))
    num = 10
    print("num= " + str(num))
```

#### 2. 使用其他文件中

- teest1.py

```python
# 定义全局变量
num = 1
 

# 设置变量值
def set_num(p):
    global num
    num = p
 
# 获取变量值
def get_num():
    global num
    return num
```

- test2.py

```python
import test1
# 借助方法来访问其它文件中的变量
def get_int():
    n = test1.get_num()
    print(n)
    test1.set_num(10)
    print(test1.get_num())
```

#### 3. 互斥锁

```python
# 导入线程threading模块
import threading
 
# 创建互斥锁
mutex = threading.Lock()

#注意 acquire 和 release 要成对出现
mutex.acquire()
g_num = g_num + 1
# 解锁资源
mutex.release()
```

> 1.单个互斥锁的死锁：**acquire()/release()** 是成对出现的，互斥锁对资源锁定之后就一定要解锁，否则资源会一直处于锁定状态，其他线程无法修改；就好比上面的代码，任何一个线程没有释放资源**release()**，程序就会一直处于阻塞状态(在等待资源被释放)
>
> 2.`多个互斥锁的死锁`：在同时操作多个互斥锁的时候一定要格外小心，因为一不小心就容易进入死循环。

#### 4. 死锁

```python
import threading
import time

class MyThread1(threading.Thread):
    def run(self):
        # 对mutexA上锁
        mutexA.acquire()

        # mutexA上锁后，延时1秒，等待另外那个线程 把mutexB上锁
        print(self.name+'----do1---up----')
        time.sleep(1)

        # 此时会堵塞，因为这个mutexB已经被另外的线程抢先上锁了
        mutexB.acquire()
        print(self.name+'----do1---down----')
        mutexB.release()

        # 对mutexA解锁
        mutexA.release()

class MyThread2(threading.Thread):
    def run(self):
        # 对mutexB上锁
        mutexB.acquire()

        # mutexB上锁后，延时1秒，等待另外那个线程 把mutexA上锁
        print(self.name+'----do2---up----')
        time.sleep(1)

        # 此时会堵塞，因为这个mutexA已经被另外的线程抢先上锁了
        mutexA.acquire()
        print(self.name+'----do2---down----')
        mutexA.release()

        # 对mutexB解锁
        mutexB.release()

mutexA = threading.Lock()
mutexB = threading.Lock()

if __name__ == '__main__':
    t1 = MyThread1()
    t2 = MyThread2()
    t1.start()
    t2.start()
```

#### 5. 死锁避免

##### .1. 递归锁lock

> 为了支持在同一线程中多次请求同一资源，python提供了“可重入锁”：threading.RLock。RLock内部维护着一个Lock和一个counter变量，counter记录了acquire的次数，从而使得资源可以被多次acquire。直到一个线程所有的acquire都被release，其他的线程才能获得资源。
>
> - 递归锁可以连续acquire多次，而互斥锁只能acquire一次

```python
import threading,time

class myThread(threading.Thread):
    def doA(self):
        lock.acquire()
        print(self.name,"gotlockA",time.ctime())
        time.sleep(3)
        lock.acquire()
        print(self.name,"gotlockB",time.ctime())
        lock.release()
        lock.release()

    def doB(self):
        lock.acquire()
        print(self.name,"gotlockB",time.ctime())
        time.sleep(2)
        lock.acquire()
        print(self.name,"gotlockA",time.ctime())
        lock.release()
        lock.release()

    def run(self):
        self.doA()
        self.doB()

if __name__=="__main__":

    # lockA=threading.Lock()
    # lockB=threading.Lock()
    lock = threading.RLock()
    threads=[]
    for i in range(1):
        threads.append(myThread())
    for t in threads:
        t.start()
    for t in threads:
        t.join()#等待线程结束，后面再讲。
```

```python
import time
import threading
class Account:
    def __init__(self, _id, balance):
        self.id = _id
        self.balance = balance
        self.lock = threading.RLock()   # 在类中加锁是最根本的。

    def withdraw(self, amount):  #取款
        with self.lock:          #锁应该加载原子操作中。
            self.balance -= amount

    def deposit(self, amount):   #存款
        with self.lock:             #锁应该加载原子操作中。
            self.balance += amount

    def drawcash(self, amount):#lock.acquire中嵌套lock.acquire的场景
        with self.lock:             #锁应该加载原子操作中。
            interest=0.05           # 计算利息
            count=amount+amount*interest
            self.withdraw(count)    # 这里就出现了锁的嵌套，所有用RLock。

def transfer(_from, to, amount):
    #锁不可以加在这里 因为其他的其它线程执行的其它方法在不加锁的情况下数据同样是不安全的
     _from.withdraw(amount)
     to.deposit(amount)

alex = Account('alex',1000)
yuan = Account('yuan',1000)

t1=threading.Thread(target = transfer, args = (alex,yuan, 100))
t1.start()

t2=threading.Thread(target = transfer, args = (yuan,alex, 200))
t2.start()

t1.join()
t2.join()

print('>>>',alex.balance)
print('>>>',yuan.balance)
```



#### Resource

- https://www.cnblogs.com/chenhaiming/p/9915287.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/globalvariablelock/  

