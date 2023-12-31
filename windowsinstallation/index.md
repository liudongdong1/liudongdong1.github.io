# Window CommandLine


### 0. windows10 系统镜像&Office&Kms激活

- 目录F:\softdownload\windows\系统镜像
- 注意，得将windows杀毒软件关掉，否则kms那个软件会被自动删除；

> 安装时问题，windows10版本太低，导致网卡驱动，显卡驱动等都有问题；解决办法是用windows update升级工具微软易升，升级版本，可以成功解决，不需要安装其他驱动问题。

### 1.驱动精灵&Nvidia Cuda和cudadnn安装

- cudnn 未安装https://developer.nvidia.com/zh-cn/cudnn

### 2.  工具

- maven：F:\softdownload\windows\编程语言\apache-maven-3.6.3-bin

> MAVEN_HOME： C:\LddTool\apache-maven-3.6.3
>
> %MAVEN_HOME%\bin
>
> mvn -version

- apache tomcat: F:\softdownload\windows\编程语言\apache-tomcat-9.0.41-windows-x64

> CATALINA_HOME: C:\LddTool\apache-tomcat-9.0.41
>
> CATALINA_BASE: C:\LddTool\apache-tomcat-9.0.41
>
> %CATALINA_HOME%\lib;%CATALINA_HOME%\bin
>
> startup

- Git: F:\softdownload\windows\基本办公
- typora： 配置vue主题
- 远程办公类
  - 向日葵软件
  - 腾讯会议
  - Deskreen 远程桌面
  - smartroom 远程会议
  - openvpn 相关配置文件
  - v2rayN安装：https://github.com/233boy/v2ray/wiki/V2RayN%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B
- 娱乐
  - qq音乐
- 火绒安全软件
- Everything
- 7zip 文件工具
- EV录屏
- 百度网盘
- picgo图床上传， 使用gitee uploader1.1.2

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210306165732084.png)

### 3. 语言类

- ruby 安装： F:\softdownload\windows\编程语言\rubyinstaller-devkit
- java 语言： F:\softdownload\windows\编程语言\jdk-15.0.1_windows-x64_bin

> JAVA_HOME：C:\language\jdk-15.0.1
>
> %JAVA_HOME%\bin;

- nodejs  [Node.js 官网 ](https://nodejs.org/en/download/) [安装](https://yafine66.gitee.io/posts/4ab2.html)

> node -v  查看命令；设置npm 包安装目录，并在环境变量中添加这些目录；
>
> ```shell
> npm config set prefix "C:\Program Files\nodejs\node_global"
> npm config set cache "C:\Program Files\nodejs\node_cache"
> ```

### 4. 软件开发

- visual studio code
- visual studio 2019
- anaconda 

> C:\Users\liudongdong\anaconda3
>
> C:\Users\liudongdong\anaconda3\Script
>
> C:\Users\liudongdong\anaconda3\bin

> conda环境和缓存的默认路径（envs directories 和 package cache）不一定要默认存储在用户目录，我们可以将他们设置到盈余空间稍大的其他目录来缓解这种空间压力，只要保证不同用户之间的设置不同即可。

```xml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
envs_dirs:
  - D:\work_anaconda_download\envs
  - C:\Users\liudongdong\anaconda3\envs
  - C:\Users\liudongdong\.conda\envs
  - C:\Users\xxx\AppData\Local\conda\conda\envs
pkgs_dirs:
  - D:\work_anaconda_download\pkgs
  - C:\Users\liudongdong\anaconda3\pkgs
  - C:\Users\xxx\AppData\Local\conda\conda\pkgs
  - C:\Users\liudongdong\.conda\pkgs
```

- IDEA
- matlab 安装

### 5. Dism++

> - 管理 Windows 映像中包含的数据或信息，例如罗列映像中所包含的组件、更新、驱动程序或应用，捕获或拆分映像，在 .wim 文件中追加或删除映像，或装载映像。
>
> - 为Windows映像提供服务，例如添加或删除驱动程序修改语言设置、启用或禁用 Windows功能，以及升级到更高版本的 Windows等等。
> - 超好用系统优化垃圾清理工具



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/windowsinstallation/  

