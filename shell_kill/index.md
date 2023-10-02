# shell_kill


> Linux 中的 kill 命令用来终止指定的进程（terminate a process）的运行，是 Linux 下进程管理的常用命令。通常，终止一个前台进程可以使用 Ctrl+C 键，但是，对于一个后台进程就须用 kill 命令来终止，我们就需要先使用 ps/pidof/pstree/top 等工具获取进程 PID，然后使用 kill 命令来杀掉该进程。kill 命令是通过向进程发送指定的信号来结束相应进程的。在默认情况下，采用编号为 15 的 TERM 信号。TERM 信号将终止所有不能捕获该信号的进程。对于那些可以捕获该信号的进程就要用编号为 9 的 kill 信号，强行 “杀掉” 该进程。 

```shell
kill [参数][进程号]
```

#### 1. 信号量

```shell
HUP    1    终端断线
 
INT     2    中断（同 Ctrl + C）
 
QUIT    3    退出（同 Ctrl + \）
 
TERM   15    终止
 
KILL    9    强制终止
 
CONT   18    继续（与STOP相反， fg/bg命令）
 
STOP    19    暂停（同 Ctrl + Z）
```

#### 2. **先用 ps 查找进程，然后用 kill 杀掉**

```shell
ps -ef|grep vim
kill 进程号

kill -9 进程号   #彻底杀死进程
```

#### 3. **杀死指定用户所有进程**

```shell
kill -9 $(ps -ef | grep peidalinux)

kill -u peidalinux
```

#### 4. **init 进程是不可杀的**

> init 是 Linux 系统操作中不可缺少的程序之一。所谓的 init 进程，它是一个`由内核启动的用户级进程`。内核自行启动（已经被载入内存，开始运行，并已初始化所有的设备驱动程序和数据结构等）之后，就通过启动一个用户级程序 init 的方式，完成引导进程。所以，init 始终是第一个进程（其进程编号始终为 1）。 其它所有进程都是 init 进程的子孙。init 进程是不可杀的！

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shell_kill/  

