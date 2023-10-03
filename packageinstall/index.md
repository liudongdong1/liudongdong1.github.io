# PackageInstall


# 1. apt install

> apt-get，是一条linux命令，适用于deb包管理式的操作系统，主要用于自动从互联网的软件仓库中搜索、安装、升级、卸载软件或操作系统。

```shell
apt-get check #检查是否有损坏的依赖
apt-cache depends packagename #了解使用依赖
apt-get remove packagename --purge 删除包，包括删除配置文件等
apt-cache search packagename #搜索包
apt-cache show packagename #获取包的相关信息，如说明、大小、版本等
apt-get install packagename #安装包
apt-get install packagename --reinstall #重新安装包
apt-get -f install #修复安装-f = –fix-missing
apt-get remove packagename #删除包

sudo apt-get install ppa-purge
#To purge a PPA, you must use the following command:
sudo ppa-purge ppa:someppa/ppa     删除ppa 及对应软件
sudo add-apt-repository ppa:someppa/ppa
sudo apt update
sudo add-apt-repository --remove ppa:someppa/ppa
#多线程下载源
sudo add-apt-repository ppa:apt-fast/stable
sudo apt-get update
```

> apt 工作原理：
>
> 1. `apt`命令会访问`/etc/apt/sources.list`源列表
> 2. 查询包：从`Packages.gz`中获取到所有包的信息，然后`apt-get`就可以通过它找到所有的包并且自动下载安装了。
> 3. 下载包依赖： 它会`首先检查依赖,如果不存在则下载依赖包`,这个依赖包或许还有依赖(递归下载),在完成了所有依赖包则可以进行下载,安装完成,中间任意一环没有完成则失败退出.这就是整个过程啦

> `DebType AddressType://Hostaddress/Ubuntu Distribution  Component1 Component2……`
>
> 其中各字段含义如下所示。
>
> ●  DebType表示Deb软件包类型，使用deb表示二进制软件包，使用deb-src表示源码包；
>
> ●  AddressType表示访问地址类型，常用类型有：http、ftp、file、cdrom、ssh等；
>
> ● ` Distribution`表示Ubuntu的各个发行版本，例如`dapper、feisty`；常用发行商：CentOS、Ubuntu、Redhat、Debian、Fedora、SUSE、Kali Linux 、Archlinux。Debian 系列，包括 Debian 和 Ubuntu 等, Ubuntu 是基于 Debian 的 unstable 版本加强而来，可以这么说，Ubuntu 就是 一个拥有 Debian 所有的优点，以及自己所加强的优点的近乎完美的 Linux 桌面系统。
>
> ●  Component表示软件包组件类别，是由技术支持程度不同而划分的类别，可选择main、restricted、universe和multiverse中的一种或多种。可以包含 [main | contrib | non-free] 中的一个或多个。`main 表示纯正的遵循 Debian 开源规范的软件`，`contrib 表示遵循 Debian 开源规范但依赖于其它不遵循 Debian 开源规范的软件的软件`，n`on-free 表示不遵循 Debian 开源规范的软件`。

| Linux 发行版本选择                                     |                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| Linux 桌面系统                                         | Ubuntu                                                       |
| 服务器端 Linux 系统                                    | 首选 Redhat（付费）或者 CentOS                               |
| 如果对安全要求很高                                     | Debian 或者 FreeBSD (银行)                                   |
| 使用数据库高级服务或者电子邮件网络用户                 | SUSE（德国，收费）、openSUSE（开源）                         |
| 想新技术、新功能，是 rhel 和 CentOS 的测试版或预发布版 | Fedoras（Fedora 稳定之后 -->Redhat--> 去 LOGO 除收费 -->CentOS） |
| 中文                                                   | 红旗 Linux、麒麟 Linux                                       |

**【设置源】**

```shell
#https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
#添加阿里源
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
##中科大源
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
##清华源
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
##163源
deb http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse
#deb cdrom:[Ubuntu 18.04.2 LTS _Bionic Beaver_ - Release amd64 (20190210)]/ bionic main restricted
```

### 2. 源代码编译安装

1. tar -zxvf ****.tar.gz
2. tar -jxvf ****.tar.bz(或bz2)
3. 输入编译文件命令：./configure（有的压缩包已经编译过，这一步可以省去）
4. 然后是命令：make ..
5. 再是安装文件命令：make install

===如何卸载：

1. 用CD 命令进入编译后的软件目录，即安装时的目录
2. 执行反安装命令：make uninstall

### 3. deb 包安装

```shell
#若用 Ubuntu 自带的软件中心安装 deb 格式的文件不仅经常会崩溃而且会遇到各种各样的依赖问题。通过deb文件安装软件优选
sudo apt-get install gdebi
#其次安装deb 安装包方法
dpkg -p package-name  #显示包的具体信息
dpkg -s package-name  #报告指定包的状态信息	
dpkg -l				#显示所有已经安装的Deb包，同时显示版本号以及简短说明
dpkg -P            #删除一个包（包括配置信息）	
dpkg -A package_file  #从软件包里面读取软件的信息	
dpkg -i <.deb file name>  #安装软件	
```

### 4. snap 安装

```shell
#snap是一种全新的软件包管理方式，它类似一个容器拥有一个应用程序所有的文件和库，各个应用程序之间完全独立。所以使用snap包的好处就是它解决了应用程序之间的依赖问题，使应用程序之间更容易管理。但是由此带来的问题就是它占用更多的磁盘空间.snap软件包一般安装在/snap目录下
snap list #罗列
snap find | install | refresh | remove package
snap changes # 查看正在进行的下载
snap abort id # 停止下载
```

```shell
sudo snap install snap-store-proxy
sudo snap install snap-store-proxy-client
```

### 5. 新立得软件管理

   ```bash
sudo apt-get install synaptic  #  全面高效地管理各种软件和依赖。
   ```


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/packageinstall/  

