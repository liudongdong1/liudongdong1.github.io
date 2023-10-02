# Docker-Command


>Compose 是用于定义和运行多容器 Docker 应用程序的工具。通过 Compose，您可以使用 YML 文件来配置应用程序需要的所有服务。然后，使用一个命令，就可以从 YML 文件配置中创建并启动所有服务。
>
>- 使用 Dockerfile 定义应用程序的环境。
>
>- 使用 docker-compose.yml 定义构成应用程序的服务，这样它们可以在隔离环境中一起运行。
>
>- 最后，执行 docker-compose up 命令来启动并运行整个应用程序。

### 1. download

### 2. 常见命令

```shell
docker-compose -v
docker-compose ps
docker-compose logs
docker-compose port eureka 8761 #打印绑定的公共端口，下面命令可以输出 eureka 服务 8761 端口所绑定的公共端口
docker-compose build
docker-compose start eureka  #start：启动指定服务已存在的容器
docker-compose stop eureka
docker-compose rm eureka  #删除指定服务的容器
docker-compose up       #构建、启动容器
docker-compose kill eureka  #通过发送 SIGKILL 信号来停止指定服务的容器
ocker-compose scale user=3 movie=3  #设置指定服务运行容器的个数，以 service=num 形式指定
docker-compose run web bash  #在一个服务上执行一个命令
```



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: liudongdong1.github.io/docker-compose/  

