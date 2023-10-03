# PygameRecord


> pygame 是跨平台 Python 模块，专为电子游戏设计，包含图像、声音。创建在 SDL(Simple Direct Media Layer) 基础上，允许实时电子游戏研发而无需被低级语言、如 C 语言或是更低级的汇编语言束缚。基于这样一个设想，所有需要的游戏功能和理念都（主要是图像方便）完全简化为游戏逻辑本身，所有的资源就够都可以由 Python 提供。

#### 1. 模块

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210614153615072.png)

```python
pip install pygame
#检测安装
import pygame
pygame.ver
```

```python
# -*- coding:utf-8 -*-
import sys
import pygame

pygame.init()   # 初始化 pygame
size = (width, height) = (300, 240)         # 设置窗口
screen = pygame.display.set_mode(size)  # 显示窗口

# 执行死循环，确保窗口一直显示
while True:
    # 检查事件
    for event in pygame.event.get():    # 遍历所有事件
        if event.type == pygame.QUIT:   # 如果单击关闭窗口，则退出
            sys.exit()
pygame.quit()
```

#### 2. 学习资源

- http://code.py40.com/pythongame/15.html
- [2、Pygame中的IO、数据](http://code.py40.com/pythongame/17.html)
- [3、Pygame事件与设备轮询](http://code.py40.com/pythongame/19.html)
- [4、加载位图与常用的数学函数](http://code.py40.com/pythongame/21.html)
- [5、大喵爱吃鱼小游戏开发实例](http://code.py40.com/pythongame/23.html)
- [6、Sprite模块和加载动画](http://code.py40.com/pythongame/25.html)
- [7、Pygame碰撞检测](http://code.py40.com/pythongame/27.html)
- [8、Pygame常用数据结构](http://code.py40.com/pythongame/29.html)
- [9、嗷大喵快跑小游戏开发实例](http://code.py40.com/pythongame/31.html)
- [10、愤怒的小鸟](https://github.com/sourabhv/FlapPyBird/blob/master/flappy.py)
- [11、PyGame-Learning-Environment](https://github.com/ntasfi/PyGame-Learning-Environment)
- [12、-star-path-finding-algorithm-visualizer-python](https://github.com/sourhub226/A-star-path-finding-algorithm-visualizer-python)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pygamerecord/  

