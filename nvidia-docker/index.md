# Nvidia-docker


>**Make sure you have installed the [NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and Docker engine for your Linux distribution** **Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed**

>NVIDIA 于 2016 年开始设计 NVIDIA-Docker 已便于容器使用 NVIDIA GPUs。 第一代 nvidia-docker1.0 实现了对 docker client 的封装，并在容器启动时，将必要的 GPU device 和 libraries 挂载到容器中。
>
>- 设计高度与 docker 耦合，不支持其它的容器运行时。如: LXC, CRI-O 及未来可能会增加的容器运行时。
>- `不能更好的利用 docker 生态的其它工具`。如: docker compose。
>- `不能将 GPU 作为调度系统的一种资源来进行灵活的调度`。
>- 完善容器运行时对 GPU 的支持。如：自动的获取用户层面的 NVIDIA Driver libraries, NVIDIA kernel modules, device ordering 等。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220511192654316.png)

### 1. nvidia-docker2.0

- `nvidia-docker2.0` 是一个简单的包，它主要通过修改 docker 的配置文件 `/etc/docker/daemon.json` 来让 docker 使用 NVIDIA Container runtime。
- `nvidia-container-runtime` 才是真正的核心部分，它在原有的 `docker` 容器运行时 `runc` 的基础上增加一个 `prestart hook`，用于调用 libnvidia-container 库。
- `libnvidia-container` 提供一个库和一个简单的 CLI 工具，使用这个库可以使 NVIDIA GPU 被 Linux 容器使用。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220511193017741.png)

```shell
# docker --> dockerd --> containerd --> containerd-shim -->runc --> container-process
# docker --> dockerd --> containerd --> containerd-shim--> nvidia-container-runtime --> nvidia-container-runtime-hook --> libnvidia-container --> runc -- > container-process
```

>当 nvidia-container-runtime 创建容器时，先执行 nvidia-container-runtime-hook 这个 hook 去检查容器是否需要使用 GPU (通过环境变 `NVIDIA_VISIBLE_DEVICES` 来判断)。如果需要则调用 libnvidia-container 来暴露 GPU 给容器使用。否则走默认的 runc 逻辑。

### 2. installation

> **Make sure you have installed the [NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and Docker engine for your Linux distribution** **Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed**

```shell
# 1. setting up docker
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  
# 2. setting up nvidia container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
# testing the installation
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

#### .1. 镜像加速

-  --runtime=nvidia 指定，可以通过设置 default-runtime 解决，即在 /etc/docker/daemon.sh 中增加 "default-runtime": "nvidia"

```shell
[root@lv218 ~]# cat /etc/docker/daemon.json 
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "registry-mirrors": [
        "https://1nj0zren.mirror.aliyuncs.com",
        "https://docker.mirrors.ustc.edu.cn",
        "http://f1361db2.m.daocloud.io",
        "https://registry.docker-cn.com"
    ]
}
```

#### .2. 建立docker组

建立 docker 组：
`sudo groupadd docker`
将当前用户加入 docker 组：
`sudo usermod -aG docker $USER`

### 3. usage

#### .1. GPU enumeration

```shell
# Starting a GPU enabled CUDA container; using --gpus
docker run --rm --gpus all nvidia/cuda nvidia-smi

#Using NVIDIA_VISIBLE_DEVICES and specify the nvidia runtime
docker run --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda nvidia-smi
    
#Start a GPU enabled container on two GPUs
docker run --rm --gpus 2 nvidia/cuda nvidia-smi

#Starting a GPU enabled container on specific GPUs
docker run --gpus '"device=1,2"' \
    nvidia/cuda nvidia-smi --query-gpu=uuid --format=csv
```

#### .2. pytorch-gpu

>nvidia-docker run 与 docker run --runtime=nvidia  相似

```shell
$ sudo docker pull nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
$ sudo docker run --runtime=nvidia -it --name test -v /path:/path nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 /bin/bash

# nvidia-docker2
$docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm nvidia/cuda:10.0-base nvidia-smi
or 
$ nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=all --rm nvidia/cuda:10.0-base nvidia-smi
```

#### .3. 从基础cuda镜像安装python和docker环境

```shell
# 宿主机：提前在宿主机上下载好安装pip3.7要用到的包
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# 宿主机与容器传输文件
docker cp a.txt containerid:/path

# 宿主机：运行ubuntu:18.04容器
docker run -it -d --name=lz-ubuntu -v /root/get-pip.py:/root/get-pip.py ubuntu:18.04

# 宿主机：进入到容器
docker exec -it lz-ubuntu bash

# 容器内：可选-安装vim
apt-get update
apt-get install vim -y

# 容器内：配置pip源，用以加速安装
sudo mkdir ~/.pip
sudo vim ~/.pip/pip.conf

添加以下内容：
[global]
index-url=https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com

国内源：
清华：
https://pypi.tuna.tsinghua.edu.cn/simple
阿里云：
http://mirrors.aliyun.com/pypi/simple/
中国科技大学 
https://pypi.mirrors.ustc.edu.cn/simple/
华中理工大学：
http://pypi.hustunique.com/
山东理工大学：
http://pypi.sdutlinux.org/
豆瓣：
http://pypi.douban.com/simple/

# 容器内：可选-配置apt源
mv /etc/apt/sources.list /etc/apt/sources.list.bak
vim /etc/apt/sources.list
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

# 容器内：更新软件包列表
apt-get update

# 容器内：可选-安装调试工具
apt-get install iputils-ping net-tools curl

# 容器内：安装最主要的python包
apt-get install python3.7 python3.7-dev

# 容器内：安装pip3.7
apt install python3-distutils
python3.7 get-pip.py

# 容器内：安装pytorch
# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# 安装其他python包
pip install transformers==2.10.0
pip install pytorch-crf==0.7.2
pip install sklearn
pip install seqeval==1.2.2
pip install pandas

# 时区设置
# 宿主机：从宿主机中拷贝时区文件到容器内，/usr/share/zoneinfo/UCT这个文件是通过软链追溯到的，时区是亚洲/上海
docker cp /usr/share/zoneinfo/UCT  lyz-ubuntu:/etc/
# 容器内：然后在容器内将其改名为/etc/localtime
mv /etc/UCT /etc/localtime

# 容器内：清理无用的包
apt-get clean
apt-get autoclean
du -sh /var/cache/apt/
rm -rf /var/cache/apt/archives

# 容器内：清理pip缓存
rm -rf ~/.cache/pip

# 容器内：清理命令日志
history -c

# 宿主机：打包镜像
docker commit -a '提交人' -m '描述' <容器名/ID> <镜像名称>
```

### Resource

- https://zhuanlan.zhihu.com/p/111235300
- https://developer.nvidia.com/blog/gpu-containers-runtime/
- https://github.com/NVIDIA/nvidia-docker
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker  nvidia container 文档

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/nvidia-docker/  

