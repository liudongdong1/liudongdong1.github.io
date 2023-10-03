# Docker Operaction


### 1. Docker Introduce

| ![Container stack example](https://docs.docker.com/images/Container%402x.png) | ![Virtual machine stack example](https://docs.docker.com/images/VM%402x.png) |
| -----------------------------------------------------------: | ------------------------------------------------------------ |
|                                                              |                                                              |

#### .1.  [**Docker Platfrom:**](https://docs.docker.com/docker-for-windows/install/)

- Docker provides the ability to package and run an application in a loosely isolated environment called a container. 
- Develop your application and its supporting components using containers.

#### .2.  Docker Engine

- *Docker Engine* is a client-server application with these major components:

  - A server which is a type of long-running program called a daemon process (the `dockerd` command).<font color=red> cerates and manages docker objects, eg: images, containers, networks, volumes</font>
  - A REST API which specifies interfaces that programs can use to talk to the daemon and instruct it what to do.
  - A command line interface (CLI) client (the `docker` command).

  ![Docker Engine Components Flow](https://docs.docker.com/engine/images/engine-components-flow.png)

#### .3. Docker architecture:

- **docker daemon:**` listens for Docker API requests and manages Docker objects `such as images, containers, networks, and volumes. 
- **docker client:** communicate with more than one daemon
- **docker registries**: registry that stores docker images, like Docker Hub.
- **docker object**:
  - **images:** read-only template with instructions for `creating docker container`, by Dockerfile.
  - **containers**: a runnable instance of an image. `isolated from other containers or host machine`.
  - **service:** `scale containers across multiple docker daemons`

![Docker Architecture Diagram](https://docs.docker.com/engine/images/architecture.svg)



#### .4. dockerfiles

```shell
# Use the official image as a parent image.
FROM node:current-slim
#ARG ND4J_CLASSIFIER   #ARG 指令是定义参数名称，以及定义其默认值。
# Set the working directory.
WORKDIR /usr/src/app
# Copy the file from your host to your current location.
COPY package.json 
# Run the command inside your image filesystem.
RUN npm install
# Inform Docker that the container is listening on the specified port at runtime.
EXPOSE 8080
# Run the specified command within the container.
CMD [ "npm", "start" ]
# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .
```

- create container & run

  ```shell
  docker build --tag bulletinboard:1.0 .
  docker run --publish 8000:8080 --detach --name bb bulletinboard:1.0
  ```

- multi-stage

```dockerfile
FROM golang:1.7.3
WORKDIR /go/src/github.com/sparkdevo/href-counter/
RUN go get -d -v golang.org/x/net/html
COPY app.go    .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=0 /go/src/github.com/sparkdevo/href-counter/app .
CMD ["./app"]
```

>  Dockerfile 文件的特点是同时存在多个 FROM 指令，每个 FROM 指令代表一个 stage 的开始部分。我们可以把一个 stage 的产物拷贝到另一个 stage 中。本例中的第一个 stage 完成了应用程序的构建，内容和前面的 Dockerfile.build 是一样的。第二个 stage 中的 COPY 指令通过 --from=0 引用了第一个 stage ，并把应用程序拷贝到了当前 stage 中.

- multi-stage v2

```dockerfile
FROM golang:1.7.3 as builder
WORKDIR /go/src/github.com/sparkdevo/href-counter/
RUN go get -d -v golang.org/x/net/html
COPY app.go    .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /go/src/github.com/sparkdevo/href-counter/app .
CMD ["./app"]
```

> 第一个 stage 使用 as 语法命名为 builder，然后在后面的 stage 中通过名称 builder 进行引用 --from=builder。通过使用命名的 stage， Dockerfile 更容易阅读了。

#### .5. compose.yml

```shell
FROM       # 基础镜像，一切从这里开始构建
MAINTAINER #镜像是谁写的，姓名+邮箱
RUN        # 镜像构建的时候需要运行的命令
ADD        # 步添加内容
WORKDIR    # 镜像的工作目录 
VOLUME     # 挂载的目录
EXPOSE     # 保留端口配置
CMD        # 指定这个容器启动的时候要运行的命令
ENTRYPOINT # 指定这个容器启动的时候要运行的命令，可以追加命令
ONBUILD    # 当构建一个被继承 Dockerfile，这个时候就会运行onbuild
COPY       # 类似add，将我们文件拷贝到镜像中
ENV        # 构建的时候设置环境变量
```

```yml
#docker-Compse的版本
version: '3'
#建立2个service 一个wordpress 一个 mysql  一个service代表一个container
services:
  wordpress:
    image: wordpress
#端口映射80 映射到8080端口
    ports:
      - 8080:80
#环境变量2个
    environment:
      WORDPRESS_DB_HOST: mysql
      WORDPRESS_DB_PASSWORD: root
    networks:
      - my-bridge

  mysql:
    image: mysql
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: wordpress
    volumes:
      - mysql-data:/var/lib/mysql
    networks:
      - my-bridge
#建立一个volumes 
volumes:
  mysql-data:
#建立一个networks
networks:
  my-bridge:
    driver: bridge
```

```yml
version: '3' # 版本，有1.x,2.x,3.x 跟docker 版本有对应关系，配置也有些差异，用新版就好了
services:   # 定义一组服务
    web:    # 第一个服务
        hostname: webapp # 给容器起个名字
        build: # 指定镜像来源，这是其中一种，使用 dockerfile 构建
            context: ../ # docker run 运行的上下文路径  
            dockerfile: build/Dockerfile # dockerfile 文件位置，注意跟上一个配置对应，不指定默认是当前目录的 Dockerfile
        networks: # 指定网络
            - dev-local-network # 网络名称，需要先定义
        depends_on: # 指定依赖服务，服务会在依赖服务启动后再开启
            - mysql # 服务名称
        ports: # 端口映射
            - "80:80" # 宿主机端口到容器端口的映射
        volumes: # 宿主机的数据卷或文件挂载到容器里
            - ../:/var/www/html # 宿主机路径：容器里的路径
        environment: # 环境变量，有两种方式，直接键值对或者 env_file
            OMS_DB_HOST: ${OMS_DB_HOST} # ${} 表示取配置文件里的值，默认文件是当前默认的.env，也可以--env-file 指定路径
        command: ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf", "--nodaemon"] # 这是容器启动后的第一个命令，注意是要在前台的命令，不能执行完就结束了，不然容器启动就关闭了
    mysql: # 第二个服务了
        image: "mysql:5.7" # 指定镜像源的第二种方式，直接指定 image，这是是官方的 mysql 5.7版本
networks: # 定义网络
    dev-local-network: # 网络名称，上面用到的网络就是这里定义的
```

## 2. 调用摄像头画面到本地显示器

- Tools to manage, scale, and maintain containerized applications are called *orchestrators*, and the most common examples of these are *Kubernetes* and *Docker Swarm*.

在host中运行摄像头等画面显示到本地显示器

1. 方法一: 每次重开机需要在本机运行 xhost +

```shell
#首先主系统先运行
sudo apt install x11-xserver-utils
xhost +
#启动容器的时候添加 共享本地unix端
#-v /tmp/.X11-unix:/tmp/.X11-unix \           #共享本地unix端口
#-e DISPLAY=unix$DISPLAY \                    #修改环境变量DISPLAY
#-e GDK_SCALE \                               #我觉得这两个是与显示效果相关的环境变量，没有细究
#-e GDK_DPI_SCALE \
#例如运行openpose如下:
sudo docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE --gpus all cwaffles/openpose -it --rm /bin/bash
```

1. 已经启动的容器

```shell
#使用ifconfig 查看主机和docker ip 地址
#主机=XXX   dockerIp=YYY
#docker 中
export DISPLAY=XXX
#主机中
sudo gedit /etc/lightdm/lightdm.conf
# add a line xserver-allow-tcp=true   但是ubuntu18.04中没有这个文件
sudo systemctl restart lightdm
xhost +
```

## 3.  Image Repository

- DockerHub：https://hub.docker.com/
- DaoCloud：https://hub.daocloud.io/
- Aliyun：[https://dev.aliyun.com/search...](https://dev.aliyun.com/search.html)

> - Busybox ：是一个集成了一百多个最常用 Linux 命令和工具的软件。L inux 工具里的瑞士军刀
- Alpine ： Alpine 操作系统是一个面向安全的轻型 Linux 发行版经典最小镜像，基于 busybox ，功能比 Busybox 完善。
> - Slim ： docker hub 中有些镜像有 slim 标识，都是瘦身了的镜像。也要优先选择

## 4. docker & nivdia-docker 安装

- [./Nvidia-docker.md](./Nvidia-docker.md)

### .2. 设置源
```shell
sudo yum-config-manager \
    --add-repo \
    https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/centos/docker-ce.repo
```

### .3. 镜像加速

```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{ 
    "registry-mirrors": ["https://82m9ar63.mirror.aliyuncs.com"] 
}
EOF 
sudo systemctl daemon-reload 
sudo systemctl restart docker
```

## 5. docker ssh server

```shell
apt-get install openssh-server
vim  /etc/ssh/sshd_config
#PermitRootLogin yes  
#UsePAM no
service ssh start
#设置登录密码
passwd root
ssh name@ip -p 9000
```

```shell
#文件拷贝
sudo docker cp id_rsa.pub 317348660bba:/root
sudo docker cp 317348660bba:/root id_rsa.pub 
# 注意container 中文件路径书写
```

### 问题记录

- [Error response from daemon: Get https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io: no such host](https://www.cnblogs.com/hanfan/p/12403520.html)

```shell
vim /etc/resolv.conf
nameserver 8.8.8.8
```

- [pytorch relateve docker image](https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated)
- [docker container](https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated)

- docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].

```shell
apt-get install nvidia-container-runtime
systemctl restart docker
//或者
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
systemctl restart dockerd
```

- ```
  ERROR: for rabbitmq  Cannot start service rabbitmq: container "3b6c...0aba": already exists
  The container name "/my-postgres" is already in use by container
  ```

> moved the project in an other directory. When I tryed to run `docker-compose up` it failed because of some conflicts. `docker system prune`
>
> ```shell
> docker-compose down
> docker-composo up -d
> ```

### window10专业版安装

>  控制面板>程序和功能>启用或关闭Windows功能 下找到Hyper-V：开启 Hyper-V 选项；
>
> docker 下载链接：https://docs.docker.com/docker-for-windows/install/  具体安装说明。

### window10 家庭版 安装

- 下载地址：http://mirrors.aliyun.com/docker-toolbox/windows/docker-toolbox/

- docker document: https://docs.docker.com/engine/reference/commandline/save/

## [Deepo:](http://ufoym.com/deepo/)

​     a series of [*Docker*](http://www.docker.com/) images that

- allows you to quickly set up your deep learning research environment
- supports almost all [commonly used deep learning frameworks](http://ufoym.com/deepo/#Available-tags)
- supports [GPU acceleration](http://ufoym.com/deepo/#GPU) (CUDA and cuDNN included), also works in [CPU-only mode](http://ufoym.com/deepo/#CPU)
- works on Linux ([CPU version](http://ufoym.com/deepo/#CPU)/[GPU version](http://ufoym.com/deepo/#GPU)), Windows ([CPU version](http://ufoym.com/deepo/#CPU)) and OS X ([CPU version](http://ufoym.com/deepo/#CPU))

and their Dockerfile generator that

- allows you to [customize your own environment](http://ufoym.com/deepo/#Build) with Lego-like modules

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/docker-operaction/  

