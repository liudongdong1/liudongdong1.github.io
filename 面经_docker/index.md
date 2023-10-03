# 面经_docker


> - Docker 容器是`轻量级的虚拟技术`，占用更少系统资源。
> - 使用 Docker 容器，不同团队（如开发、测试，运维）之间更容易合作。
> - 可以在`任何地方部署 Docker 容器`，比如在任何物理和虚拟机上，甚至在云上。
> - 由于 Docker 容器非常轻量级，因此可扩展性很强。
> - 一个完整的docker包括： dockerclient，docker daemon， image， container
> - Docker 是一个 Client-Server 结构的系统，Docker 守护进程运行在主机上， 然后通过 Socket 连接从客户端访问，守护进程从客户端接受命令并管理运行在主机上的容器。守护进程和客户端可以运行在同一台机器上。
> - 四种状态：`运行、已暂停、重新启动、已退出`。

#### 1. VM & Docker

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220324133949123-16481004040975.png)

- **虚拟机：**传统的虚拟机需要模拟整台机器包括硬件，每台虚拟机都需要有自己的操作系统，虚拟机一旦被开启，预分配给他的资源将全部被占用。每一个虚拟机包括应用，必要的二进制和库，以及一个完整的用户操作系统。
- **Docker容器：**容器技术是和我们的宿主机共享硬件资源及操作系统可以实现资源的动态分配。容器包含应用和其所有的依赖包，但是与其他容器**共享内核**。容器在宿主机操作系统中，在用户空间以分离的进程运行。

#### 2. K8s

**k8s** 的全称是 **kubernetes**，它是`基于容器的集群管理平台`，是`管理应用的全生命周期的一个工具`，从创建应用、应用的部署、应用提供服务、扩容缩容应用、应用更新、都非常的方便，而且可以做到故障自愈，例如一个服务器挂了，可以自动将这个服务器上的服务调度到另外一个主机上进行运行，无需进行人工干涉。

#### 3. Docker 命令

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/4b08513737d74479b14edf3ab5c72979tplv-k3u1fbpfcp-zoom-in-crop-mark1304000.awebp)4. Docker运行原理

> Docker 只提供一个运行环境，他跟 VM 不一样，是不需要运行一个独立的 OS，`容器中的系统内核跟宿主机的内核是公用的`。**docker 容器本质上是宿主机的进程**。
>
> 1. 启用 **Linux Namespace** 配置。
> 2. 设置指定的 **Cgroups** 参数。
> 3. 切换进程的根目录 (**Change Root**)，优先使用 **pivot_root** 系统调用，如果系统不支持，才会使用 **chroot**。

##### .1. namespace 进程隔离

> Linux Namespaces 机制提供一种`进程资源隔离`方案。PID、IPC、Network 等系统资源不再是全局性的，而是属于某个特定的 **Namespace**。每个 **namespace** 下的资源对于其他 **namespace** 下的资源都是透明，不可见的。系统中可以同时存在两个进程号为 0、1、2 的进程，由于属于不同的 namespace，所以它们之间并不冲突。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/5e90882472374f3b93858d5122d3ab16tplv-k3u1fbpfcp-zoom-in-crop-mark1304000.awebp)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/a794f094d2564510a9022470e336b6d4tplv-k3u1fbpfcp-zoom-in-crop-mark1304000.awebp)

##### .2. CGroup 分配资源

> Docker 通过 **Cgroup** 来控制容器使用的资源配额，一旦超过这个配额就发出 **OOM**。配额主要包括 `CPU、内存、磁盘`三大方面， 基本覆盖了常见的资源配额和使用量控制。
>
>  **Cgroup** 是 Linux `内核`提供的一种可以`限制、记录、隔离进程组所使用的物理资源 (如 CPU、内存、磁盘 IO 等等) 的机制`，被 LXC (Linux container)、Docker 等很多项目用于实现进程资源控制。Cgroup 本身是提供将进程进行分组化管理的功能和接口的基础结构，I/O 或内存的分配控制等具体的资源管理是通过该功能来实现的，这些具体的资源 管理功能称为 Cgroup 子系统。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/d2dc4aca163144918e25d217c91496b3tplv-k3u1fbpfcp-zoom-in-crop-mark1304000.awebp)

##### .3. chroot 跟 pivot_root 文件系统

> chroot (change root file system) 命令的功能是 **改变进程的根目录到指定的位置**。比如我们现在有一个 `$HOME/test` 目录，想要把它作为一个 `/bin/bash` 进程的根目录。
>
> - 首先，创建一个 HOME/test/{bin,lib64,lib}
> -  把 bash 命令拷贝到 test 目录对应的 bin 路径下 cp -v /bin/{bash,ls} $HOME/test/bin
> -  把 bash 命令需要的所有 so 文件，也拷贝到 test 目录对应的 lib 路径下
> - 执行 chroot 命令，告诉操作系统，我们将使用 HOME/test/bin/bash
> - 而挂载在容器根目录上、用来为容器进程提供隔离后执行环境的文件系统，就是所谓的`容器镜像`。

##### .4. 一致性

> 由于 **rootfs** 里打包的不只是应用，而是`整个操作系统的文件和目录`，也就意味着应用以及它运行所需要的所有依赖都被封装在了一起。有了容器镜像`打包操作系统`的能力，这个最基础的依赖环境也终于变成了**应用沙盒**的一部分。

##### .5. UnionFS 联合文件系统 (AUFS)

> Docker 的镜像实际上由一层一层的文件系统组成，这种层级的文件系统就是 UnionFS。UnionFS 是一种分层、轻量级并且高性能的文件系统。联合加载会把各层文件系统叠加起来，这样最终的文件系统会包含所有底层的文件和目录。

> AUFS 作为联合文件系统，它能够将不同文件夹中的层联合（Union）到了同一个文件夹中，这些文件夹在 AUFS 中称作分支，整个『联合』的过程被称为*联合挂载（Union Mount）

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/2017-11-30-docker-aufs.png)

#### 5. Docker网络

##### .1. Host 模式

> 当启动容器的时候用 host 模式，容器将`不会虚拟出自己的网卡，配置自己的 IP 等`，而是`使用宿主机的 IP 和端口`。但是容器的其他方面，如文件系统、进程列表等还是和宿主机隔离的。

##### .2. Container 模式

> Container 模式指定新创建的容器和已经存在的一个容器共享一个 Network Namespace，而不是和宿主机共享。新创建的容器不会创建自己的网卡，配置自己的 IP，而是`和一个指定的容器共享 IP、端口范围等`。同样，两个容器除了网络方面，其他的如文件系统、进程列表等还是隔离的。两个容器的进程可以通过 lo 网卡设备通信。

##### .3. None 模式

> None 模式将容器放置在它自己的网络栈中，并不进行任何配置。实际上，`该模式关闭了容器的网络功能`，该模式下容器并不需要网络（例如只需要写磁盘卷的批处理任务）。

##### .4. Bridge 模式

> Bridge 模式是 Docker 默认的网络设置，此模式会`为每一个容器分配 Network Namespace、设置 IP 等`。当 Docker Server 启动时，会在主机上创建一个名为 **docker0** 的虚拟网桥，此主机上启动的 Docker 容器会连接到这个虚拟网桥上。虚拟网桥的工作方式和物理交换机类似，主机上的所有容器就通过交换机连在了一个二层网络中。
>
> - 容器访问外部：首先 IP 包从容器发往自己的默认网关 docker0，包到达 docker0 后，会查询主机的路由表，发现包应该从主机的 eth0 发往主机的网关 10.10.105.254/24。接着包会转发给 eth0，并从 eth0 发出去。这时 Iptable 规则就会起作用，将源地址换为 eth0 的地址。
> - 外部访问容器：创建容器并将容器的 80 `端口映射`到主机的 80 端口。当我们对主机 eth0 收到的目的端口为 80 的访问时候，Iptable 规则会进行 `DNAT 转换`，将流量发往 172.17.0.2:80

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/2017-11-30-docker-network-topology.png)

#### Resource

- https://juejin.cn/post/6933080338134466568
- https://docs.docker.com/engine/reference/commandline/build/

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E9%9D%A2%E7%BB%8F_docker/  

