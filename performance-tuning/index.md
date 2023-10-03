# Performance Tuning


> 性能调优是个大而复杂的系统性问题，涉及 Linux 系统（进程管理，文件系统，磁盘系统，网络 IO 处理等），内核参数调优，常见检测及配置工具的使用等

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/13192585-406c6ead07aec166.png)

### 1. CPU

#### .1. 查看服务

```shell
#查看 sysvinit 运行的服务，其中＋代表开启，－代表关闭：
service --status-all

#查看启动的 systemd 服务
systemctl

#查看装载的服务类型单元（服务即名称为.service）
systemctl list-units --type service

# 查看已安装的服务类型单元文件（服务类别）
systemctl list-unit-files --type service

#systemd-cgtop 按照资源使用率 (CPU, 内存，磁盘吞吐率) 从高到低的顺序显示系统中的控制组 (Control Group)。类似 top 的一个命令，每秒刷新
systemd-cgtop 
```

### 2. 存储

1. host 上专有的物理网卡（如 10G）；

2. 隔离的网络；

3. 较大的 MTU；

4. 高性能磁盘（SSD，SAS）；

5. 大的内存；

### 3. 网络

1. 更新 DPDK 驱动（uio-pci-generic ----> vfio-pci）；

2. vSwitch 微调， 如 NIC 卡的 mbuf 分配；

```shell
#查看默认 TCP 分配内存，其值是根据当前系统内存大小动态生成的一个值，其中显示的三个值分别是最小值，初始值和最大值。
cat /proc/sys/net/ipv4/tcp_mem

# 查看用来 socket 所使用接收缓冲区大小，固定值为 208K
cat /proc/sys/net/core/rmem_default

# 查看用来 socket 所使用发送缓冲区大小，固定值为 208K
cat /proc/sys/net/core/wmem_default

# 每个 socket 可用的最大缓冲区大小。默认 20K
cat /proc/sys/net/core/optmem_max
```

### Resource

- https://www.jianshu.com/p/a3f0a7d86932

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/performance-tuning/  

