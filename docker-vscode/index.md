# Docker-vscode


>安装dockers，安装nvidia-docker, 从dockerhub上拉取镜像后, 注意  `docker run 是在镜像上构建一个新的容器`(新的容器里面没有之前的环境配置)，``如果容器已经构建了，可以通过docker start 启动容器`

### 1.  容器内ssh配置

>run 一个容器，在容器中安装 ssh，修改 \etc\ssh\sshd_config 文件，将容器 commit 为一个镜像。 接着重新 run 一个刚保存镜像的容器，注意用 - p 设置端口号，用于后续连接，进入后再 passwd 设置密码。

```shell
sudo nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=all -d -p 10000:22  imagename  /bin/bash
sudo docker run -it imagename /bin/bash
```

- 设置root用户密码：passwd， 直至输入新的密码，例如：123456
- 通过uname -a 修改/etc/apt/so*  镜像文件：https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
- apt-get install openssh-server -y
- 修改sshd配置： vim /etc/[ssh](https://so.csdn.net/so/search?q=ssh&spm=1001.2101.3001.7020)/sshd_config
  - 修改 AddressFamily any 前面的 # 删除
  - 修改 PermitRootLogin yes 前面的 # 删除
  - 修改 PasswordAuthentication yes 前面的 # 删除
  - 重启 ssh 服务，service ssh restart
- 提交容器成为新的镜像，例如叫做 ubuntu-ssh，输入` docker commit` 容器 ID ubuntu-ssh
- 启动这个镜像的容器，并映射本地的一个闲置的端口（例如 10000）到容器的 22 端口，并启动容器的 sshd docker run -d -p 10000:22 ubuntu-ssh
- 现在打开新的终端，输入 ssh root@127.0.0.1 -p 10000，(docker里面的用户和对应密码，docker所在宿主机ip），如果能链接成功，会要求输入密码的，输入刚才的 123456 就可以进入容器的终端了
- vscode使用免密登录配置

### 2. 登录pytorch环境的服务

0. 所需要安装环境： 

   - openvpn： [下载地址](http://211.81.52.58:8888/VPN_Server/web/index.html)； 根据电脑类型下载相应的exe文件；
   - openvpn导入配置文件ovpn：  [下载地址](http://211.81.52.58:8888/VPN_Server/web/index.html)

1. ssh tank@****    ； 密码 tanklab

2. docker container ls   # 首先查看 container 容器是否运行，如果没有运行，运行以下命令

   ```shell
   sudo docker run -itd --restart=always -p 10024:22 -v /home/tank/zjw/:/root/rfid --name rfid --gpus all pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
   ```

3. 进入docker 容器 命令

   ```shell
   sudo docker exec -it 317348660bba /bin/bash
   ```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210116095339646.png)

> 之前docker container 已经在运行了，所以可以直接远程登录;
>
> ssh root@**** -p 10024    #密码： tanklab

项目文件夹所在云服务器目录： ~/zjw/;       项目所在docker 容器目录： /root/rfid/

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210116095637184.png)

存在一个问题ssh进入找的python环境没有torch，经过分析应该是环境变量丢失的问题。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210116132014806.png)

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/docker-vscode/  

