# 


> ​		无论我们选择用何种语言进行程序设计时，都不可能只有一个文件（除了“hello world”），通常情况下，我们都需要在一个文件中调用另外一个文件的函数呀数据等等，总之要操作其他文件中的代码，在java中，只要在同一个文件目录下，我们就不需要通过import导入，但是在Python中，我们就需要通过import来进行导入，这样我们才能应用其他文件中定义的函数和数据等代码。

## 1. Python 调用

### 1.1. 同一目录

**调用函数**

```python
# A.py
def add(x,y):
	print("xxx")
	
# B.py
import A
A.add(1,2)

#或者
from A import add
add(1,2)
```

**调用类**

```python
# A.py
class A:
    def __init__(self,xx,yy):
        self.x=xx
        self.y=yy
    def add(self):
        print("x和y的和为：%d"%(self.x+self.y))
        
# B.py
from A import A
a=A(2,3)
a.add()

#或者
import A
a=A.A(2,3)
a.add()
```

### 1.2. 不同目录

若不在同一目录，python查找不到，必须进行查找路径的设置，将模块所在的文件夹加入系统查找路径

<font color=red>注意： import 是从项目文件夹开始的</font>

> import sys
> sys.path.append(‘a.py所在的路径’)
> import a
> a.func()

## 2. 模块调用

> 而一个package跟一个普通文件夹的区别在于，package的文件夹中多了一个__init__.py文件。换句话说，如果你在某个文件夹中添加了一个__init__.py文件，则python就认为这个文件夹是一个python中的package。

```
# 假设 文件目录如下
- mod_a
    __init__.py     # 模块文件夹内必须有此文件
    aaa.py
- mod_b
    __init__.py     # 模块文件夹内必须有此文件
    bbb.py
- ccc.py
```

### 2.1. 调用同级模块

```python
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 扫除路径迷思的关键！
from mod_b.bbb import *
```

### 2.1. 调用上级模块

```python
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ccc import *
```

## 3. 功能模块化编写

![](https://img-blog.csdnimg.cn/20190923083442642.png)

```python
###################################################
#               智能小车1.0 -- 舵机模块
###################################################
 
import RPi.GPIO as GPIO
import time
class ServoModule:
    # 初始模块
    def __init__(self, PIN):
        print('Servo Module In Progress')
        GPIO.setmode(GPIO.BOARD)
        self.PIN = PIN
        #initial 是启动引脚设置初始 
    # 舵机左转
    def turnLeft(self):
        self.pwm = GPIO.PWM(self.PIN, 50)   
    # 舵机右转
    def turnRight(self):
        self.pwm = GPIO.PWM(self.PIN, 50)
        self.pwm.start(0)      
 
if __name__ == "__main__":
    try:
        # 19,21,23
        m = ServoModule(19)       
    except KeyboardInterrupt:
        pass
    GPIO.cleanup()
```

```python
###################################################
#               智能小车1.0
###################################################
# 光敏传感器           红 黑 任意
# 超声波传感器-发送     红 黑  任意 * 2
# 超声波传感器-接收
# 红外避障传感器-左     红 黑  任意
# 红外避障传感器-右     红 黑  任意
# 无源蜂鸣器           红 黑   任意
# 寻迹传感器           红 黑   任意
# 七彩大灯R-G-B         黑 任意 * 3
# 超声波云台舵机-左右转   任意
# 摄像头云台舵机-左右转   红   黑   任意
# 摄像头云台舵机-上下转   红   黑   任意
# 左轮in1-in2         任意 * 2
# 右轮in1-in2         任意 * 2
 
# 接电：红 8
# 接地：黑 9
# 其他：17
 
import threading
import os
 
from PyCode.Modules.RGBLightModule import *
from PyCode.Modules.ServoModule import *
 
 
##----------小车对外提供的功能
class QQCar:
 
    def __init__(self):
        # 初始化智能小车使用控制脚--------------
        self.PIN_LIGHT = 8              # 01：光敏
        self.PIN_ULTRASON_TRIG = 11     # 02：超声波-发射
  
    # 超声波云台，右转
    def servoUltrasonicTurnRight(self):
        self.servoModule_U.turnRight()
 
    # 摄像头水平云台，左转
    def servoCameraHTurnLeft(self):
        self.servoModule_CH.turnLeft()
```



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/python-module-invoking/  

