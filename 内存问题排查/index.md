# 内存问题排查命令


### 1. top命令
Linux的top命令提供Linux资源使用情况的实时更新信息。不仅可以查看Linux内存，也可以查看CPU以及各个进程之间的对资源的占用情况。使用方式如下：
![img](https://img2020.cnblogs.com/blog/1823155/202106/1823155-20210629180409728-33091310.png)

### 2. htop命令
- htop命令，htop命令是top命令的增强版，功能和top类似。不过，linux发行版中不一定都内置了htop命令。如果没有可以使用如下方式安装htop命令。sudo apt install htop

![img](https://img2020.cnblogs.com/blog/1823155/202106/1823155-20210629180641570-1383412175.png)

### 3. free命令
![img](https://img2020.cnblogs.com/blog/1823155/202106/1823155-20210629181256974-1200237495.png)

### 4. cat /proc/meminfo 内存映射文件
- 在linux中一切皆为文件，linux内核中把系统信息都映射到/proc 目录中，我们通过查看/proc/meminfo 文件来获取内存信息

![img](https://img2020.cnblogs.com/blog/1823155/202106/1823155-20210629181307109-1367829254.png)

### 5. vmstat 命令
![img](https://img2020.cnblogs.com/blog/1823155/202106/1823155-20210629181353061-37916802.png)

可以使用 --help参数来查看帮助信息
![img](https://img2020.cnblogs.com/blog/1823155/202106/1823155-20210629181406050-99451376.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E5%86%85%E5%AD%98%E9%97%AE%E9%A2%98%E6%8E%92%E6%9F%A5/  

