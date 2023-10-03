# shell_iotop


> iotop 命令 是一个用来监视磁盘 I/O 使用状况的 top 类工具。iotop 具有与 top 相似的 UI，其中包括 PID、用户、I/O、进程等相关信息。Linux 下的 IO 统计工具如 iostat，nmon 等大多数是只能统计到 per 设备的读写情况，如果你想知道每个进程是如何使用 IO 的就比较麻烦，使用 iotop 命令可以很方便的查看。

#### 1. 命令参数

| **参数** | **长参数**    | **参数描述**                                                 |
| -------- | ------------- | ------------------------------------------------------------ |
|          | --version     | 显示版本号                                                   |
| -h       | --help        | 显示帮助信息                                                 |
| -o       | --only        | `只显示正在产生 I/O 的进程或线程，运行过程中，可以通过按 o 随时切换` |
| -b       | --batch       | `非交互模式下运行`，一般用来记录日志。                       |
| -n NUM   | --iter=NUM    | 设置`监控（显示）NUM 次`，主要用于非交互模式。默认无限       |
| -d SEC   | --delay=SEC   | `设置显示的间隔秒数`，支持非整数                             |
| -p PID   | --pid=PID     | `只显示指定进程（PID）的信息`                                |
| -u USER  | --user=USER   | 显示指定的`用户的进程`的信息                                 |
| -P       | --processes   | `只显示进程，不显示所有线程`                                 |
| -a       | --accumulated | 累积的 I/O, 显示`从 iotop 启动后每个进程累积的 I/O 总数`，便于诊断问题 |
| -k       | --kilobytes   | `显示使用 KB 单位`                                           |
| -t       | --time        | `非交互模式下，加上时间戳`。                                 |
| -q       | --quiet       | 只在第一次监测时显示列名. 去除头部一些行：这个参数可以设置最多 3 次来移除头部行：-q 列头部只在最初交互显示一次；-qq 列头部不显示；-qqq，I/O 的总结不显示 |

####  2. 相关案例

##### .1. **只显示正在产生 I/O 的进程**

```shell
iotop -o
```

#####  .2. **使用非交互模式将 iotop 命令输出信息写入日志**

```shell
nohup iotop -b -o -n 10 -d 5 -t  > /tmp/iotop.log &
```

##### .3.**非交互式，输出 pid 为 8382 的进程信息**

```shell
iotop -botq -p 8382
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shell_iotop/  
