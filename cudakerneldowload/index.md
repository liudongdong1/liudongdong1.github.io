# CudaKernelDownload


### 1. NVIDIA显卡驱动 cuda

#### .1.  apt 方式安装

```shell
 #驱动安装
 sudo ubuntu-drivers devices  查看系统支持的显卡设备并下载
#**系统设置** > **细节**窗口，你会发现Ubuntu正在使用Nvidia显卡。
lspci -k | grep -A 2 -i "VGA"
software-properties-gtk
nvidia-settings       #打开nvidia 设置软件页面
sudo ubuntu-drivers autoinstall  #显示推荐的驱动
sudo apt-get update   
```

#### .2. 使用 PPA 第三方软件仓库安装最新版本

> 添加 PPA 软件仓库：sudo add-apt-repository ppa:graphics-drivers/ppa，需要输入用户密码，按照提示还需要按下 Enter 键。
> 更新软件索引：sudo apt update
> 接下来的步骤同方法一，只是这样我们就可以选择安装最新版本的驱动程序了。

#### .3. 官网最新版本安装 推荐

```shell
#  1. 安装必要软件
sudo apt-get update   #更新软件列表
sudo apt-get install g++
sudo apt-get install gcc
sudo apt-get install make
```

```bash
lspci | grep -i nvidia  #verify you have a cuda-Capble GPU
# 2. 去官网下载对应的驱动https://www.nvidia.cn/Download/index.aspx?lang=cn
#查看当前NVIDIA驱动版本
sudo dpkg --list | grep nvidia-*
#查看本机GPU
uname -r #current running kernel
sudo apt-get install linux-headers-$(uname -r) # the kernel headers and development packages
#  3. 卸载原有驱动
sudo apt-get remove --purge nvidia*
#  4.禁用nouveau(nouveau是通用的驱动程序)（必须）
lsmod | grep nouveau   #如果有输出则需要关闭
#创建文件  vim /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0

#  5. regenerate the kernel
sudo update-initramfs -u
#  6. 重启后在终端输入如下，没有任何输出表示屏蔽成功
lsmod | grep nouveau

#  7. 安装lightdm，lightdm是显示管理器，主要管理登录界面，ubuntu20.04、22.04需要自行安装,然后上下键选择lightdm即可

#（这一步也可以不安装lightdm，使用ubuntu20.04、22.04自带的gdm3显示管理器，直观的区别就是gdm3的登陆窗口在显示器正中间，而lightdm登录窗口在偏左边，正常使用没有区别。其他的区别这里不做探究；）
#（亲测需要注意的是，如果你有控制多屏显示的需要，gdm3可能更适合你，亲测使用lightdm设置多屏，可能会出现卡屏，死机，无法动弹情况，仅供参考）
sudo apt-get install lightdm

#  8. 进入终端
sudo telinit 3
#  9. 禁用X-window服务,在终端输入
sudo /etc/init.d/lightdm stop或者（sudo service lightdm stop）
#  10.   安装驱动
sudo chmod 777 NVIDIA-Linux-x86_64-430.26.run   #给你下载的驱动赋予可执行权限，才可以安装
 
sudo ./NVIDIA-Linux-x86_64-430.26.run （–no-opengl-files）   #安装
#–no-opengl-files 只安装驱动文件，不安装OpenGL文件。这个参数我亲测台式机不加没问题，笔记本不加有可能出现循环登录，也就是loop login。 看你自己需要把。
#  11. 安装结束后输入sudo  service  lightdm  start 重启x-window服务，即可自动进入登陆界面，不行的话，输入sudo reboot重启，再看看。（重启后不行，尝试在bios中去掉安全启动设置，改成 secure boot：disable）
nvidia-smi 
nvidia-settings  #检查是否安装成功
```



```
#cuda 有俩中安装方式
	# 1: distribution-specific packages(RPM,Deb packages) recommended
	# 2: distribute-independent package(runfile package)  working across a wider set of linux distribution ,but doesn't update the native package management system

#download the nvidia toolkit
# http://develop.nvidia.com/cuda-downloads  包含 cuda 驱动和一些工具包括库,应用程序,示例程序等
#sample: 每一个toolkit 都有一个sample可以测试是够安装好
$ cuda-install-samples-10.2.sh ~
$ cd ~/NVIDIA_CUDA-10.2_Samples/5_Simulations/nbody
$ make
$ ./nbody
```



#### .4. 驱动删除

```python
#卸载分俩种情况
# 1: 卸载通过 runfile  下载
sudo /usr/local/cuda-x.y/bin/uninstall_cuda_x.y.pl
#     卸载通过 runfile 下载的驱动
sudo /usr/bin/nvidia-uninstall
# 2: 卸载通过deb/RPM 包下载的软件
sudo apt-get --purge remove <package_name> # Ubuntu
#或者To remove CUDA Toolkit:
sudo apt purge --auto-remove libcud*
sudo apt-get --purge remove "*cublas*" "cuda*"
#To remove NVIDIA Drivers:
$ sudo apt-get --purge remove "*nvidia*"
```

### 2. Linux 内核与驱动版本

#### .1.  发行版本

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201117203235695.png)

#### .2. Linux 启动过程

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201117204242635.png)

1. `内核引导`当计算机打开电源后，首先是`BIOS开机自检`，按照BIOS中设置的启动设备（通常是硬盘）来启动。`操作系统接管硬件`以后，首先`读入 /boot 目录下的内核文件`。

2. `运行init`：init 进程是系统所有进程的起点，init 程序首先是需要读取配置文件 `/etc/inittab`。

3. `运行级别`：ctrl+alt+F1–F6  切换到相应的终端

   运行级别0：系统停机状态，系统默认运行级别不能设为0，否则不能正常启动

   运行级别1：`单用户工作状态`，`root权限`，用于系统维护，禁止远程登陆`

   运行级别2：`多用户状态(没有NFS)`

   运行级别3：`完全的多用户状态(有NFS)`，登陆后进入`控制台命令行模式`

   运行级别4：系统未使用，保留

   运行级别5：`X11控制台`，登陆后进入`图形GUI模式`

   运行级别6：系统正常关闭并重启，默认运行级别不能设为6，否则不能正常启动

> `shutdown –h 10 ‘This server will shutdown after 10 mins’ `这个命令告诉大家，计算机将在10分钟后关机，并且会显示在登陆用户的当前屏幕中。

#### .3. Grub文件

-  /boot/grub/grub.cfg 文件
   - 官方文件只说/boot/grub/grub.cfg不要手工修改，这个文件是运行 update-grub自动生成的。要修改配置文件的只要打开/boot/grub/grub.cfg文件，找到想修改的地方，然后根据注释找到相应的 /etc/default/grub或/etc/grub.d/ (folder)进行修改。
   - grub.cfg文件中主要包含两个部分，一部分是 各个启动项的定义，第二部分是启动界面的设置。你可以直接用gedit打开该文件看其中的内容。
-  /etc/grub.d/ 文件夹
   - 定义各个启动项，其中的文件代表了一个或多个启动项，命名规范都是"两个数字_名称"，前面的两位数字确定这个或这多个启动项在启动界面的位置， 默认的 "00_"是预留给"00_header"的，"10_是预留给当前系统内核的，20_是预留给第三方程序的，除了这些你都可以使用，增加自己的，比如 05_ , 15_，数字越小越前面。
   - 执行前面说的"update-grub"或者update- grub2"命令之后，这个文件夹中的文件就是用于生成 grub.cfg 中启动项的定义的
-  /etc/default/grub 文件
   - 启动界面的配置，比如默认的启动项，等待用户选择启动项的时间等。当执行前面说的"update-grub"或者update-grub2"命令之后，这个文件的内容就 用于生成 grub.cfg 中启动界面的设置。

#### .4. 内核相关命令

1. **file ‘which update-initramfs’**  学会这个命令

   - 编译内核的最后一步执行make install时会调用update-initramfs，update-initramfs继而调用mkinitramfs生成initrd.img.  一个往临时initrd目录copy文件的繁琐过程，mkinitramfs则用脚本替代了手工操作
   - 1).在临时initrd目录下构建FHS规定的文件系统;2).按/etc/initramfs-tools/module和/etc/modules文件的配置，往lib/modules/目录拷贝模块，同时生成模块依赖文件modules.dep，以后内核启动后会从initramfs中(initrd.img被解压到内存中)按模块依赖关系modprobe模块;3).拷贝/etc/initramfs-tools/scripts和/usr/share/initramfs-tools/scripts下的配置文件到conf/目录下,以后内核启动，创建第一个进程init(initrd.img根目录下init.sh文件)会从conf/*读取配置，按一定的顺序加载模块/执行程序;4).模块的加载离不开modprobe工具集，因此需要拷贝modprobe工具集及其他工具到initrd目录结构下，同时解决这些工具的依赖关系(依赖的so文件的路径);5).所有步骤完成，调用cpio和gzip工具打包压缩临时initrd目录结构。

2. **nouveau**(英语：[/](https://baike.baidu.com/item/%2F)[n](https://baike.baidu.com/item/n)uːˈ[v](https://baike.baidu.com/item/v)oʊ[/](https://baike.baidu.com/item/%2F)) 是一个自由开放源代码CPU驱动程序，是为AMD的[CPU](https://baike.baidu.com/item/CPU)所编写，也可用于属于[系统芯片](https://baike.baidu.com/item/系统芯片)的[高通](https://baike.baidu.com/item/高通)系列.

   Nouveau的内核模块应该在系统启动时就已自动加载，如果没有的话：

   - 确保你的[内核参数](https://wiki.archlinux.org/index.php/Kernel_parameters)中没有`nomodeset` 或者 `vga=`， 因为Nouveau需要内核模式设置。
   - 另外，确保你没有在 modprobe 配置文件 `/etc/modprobe.d/` 或 `/usr/lib/modprobe.d/` 中屏蔽 Nouveau。
   - 检查 dmesg 中有没有 opcode 错误，如果有的话，将 `nouveau.config=NvBios=PRAMIN` 加入 [内核参数](https://wiki.archlinux.org/index.php/Kernel_parameters)禁止模块卸载
   - Nouveau 驱动依赖[Kernel mode setting](https://wiki.archlinux.org/index.php/Kernel_mode_setting) (KMS)。当系统启动时，KMS 模块会在其它模块之后启用，所以显示的分辨率发生改变。

3. **dmesg** 命令:用来显示开机信息, kernel会将开机信息存储在ring buffer中。开机时来不及查看信息，可利用dmesg来查看。开机信息亦保存在/var/log/dmesg

   1) dmesg 是一个显示内核缓冲区系统控制信息的工具;比如系统在启动时的信息会写到/var/log/

   2) dmesg 命令显示Linux内核的环形缓冲区信息，我们可以从中获得诸如系统架构、CPU、挂载的硬件，RAM等多个运行级别的大量的系统信息。当计算机启动时，系统内核（操作系统的核心部分）将会被加载到内存中。在加载的过程中会显示很多的信息，在这些信息中我们可以看到内核检测硬件设备

   3) dmesg 命令设备故障的诊断是非常重要的。在dmesg命令的帮助下进行硬件的连接或断开连接操作时，我们可以看到硬件的检测或者断开连接的信息

   ![image-20191213212239267](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191213212239267.png)

4. watch :  execute a program periodically, showing output fullscreen  <font color=red>watch “dmesg | tail -20” </font>

5. **rmmod**: 可删除不需要的模块。Linux操作系统的核心具有模块化的特性，因此在编译核心时，不需要把全部的功能都放入核心。

6. **lsmod**: 显示内核中的模块作用同 **cat /proc/devices** 

7. **modinfo** 能查看模块的信息，通过查看模块信息来判定这个模块的用途；

8. **insmod**: 向linux 内核中加载摸块  

9. **modprobe** :向Linux内核中加载摸块,能够处理 module 载入的相依问题.  <font color=red>modprobe会检查/lib/modules/`uname -r`下的所有模块，除了/etc/modprobe.conf配置文件和/etc/modprobe.d目录以外。所有/etc/modprobe.d/arch/目录下的文件将被忽略。</font>

10. <font color=red>unable to correct problems,you have held broken package</font>    

    ```shell
    sudo apt install -f
    sudo aptitude install <packagename>  #get the detail information
    sudo apt update  | sudo apt upgrade
    sudo dpkg --configure -a
    sudo dpkg --get-selection | grep hold #get actual held packages
    dpkg --get-selections | grep linux-image  #产看内核文件有哪些
    ```


#### .5. 内核降级

**linux-image-**: 内核镜像

**linux-image-extra-**: 额外的内核模块

**linux-headers-**: 内核头文件

官网: https://kernel.ubuntu.com/~kernel-ppa/mainline/  以及相应内核安装位置   [查看稳定的内核](https://www.kernel.org/  )

安装4.19

- wget -c http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.19/linux-headers-4.19.0-041900_4.19.0-041900.201810221809_all.deb

- wget -c http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.19/linux-headers-4.19.0-041900-generic_4.19.0-041900.201810221809_amd64.deb
- wget -c http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.19/linux-image-unsigned-4.19.0-041900-generic_4.19.0-041900.201810221809_amd64.deb
- wget -c http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.19/linux-modules-4.19.0-041900-generic_4.19.0-041900.201810221809_amd64.deb
- sudo dpkg -i *.deb

```shell
#无法进入图形页面时，可以查看系统启动日志
cat /var/log/syslog | grep dkms
#查看可用的内核
apt-cache search linux-image
#备份软件源
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
#添加一个源
sudo vim /etc/apt/sources.list
deb http://security.ubuntu.com/ubuntu trusty-security main
sudo apt update
#查看所有内核
dpkg --get-selections| grep linux
#安装指定版本内核
sudo apt install 内核名称<linux-image-4.4.0-75-generic>
dpkg -l | grep 内核名称<linux-image-extra-3.16.0-43-generic>  #查看是否安装成功
#编辑grub 文件
sudo vim /etc/default/grub
# 这一行进行修改GRUB_DEFAULT=0
GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 内核名称<5.0.0-36-generic>"
Ubuntu,with Linux 5.3.0-25-generic
#更新grub 引导
sudo update-grub
sudo reboot
uname -r #查看当前版本是否安装正确
#卸载内核
sudo apt remove --purge 内核名称<linux-image-extra-3.16.0-43-generic>
sudo dpkg --purge linux-image-4.19.0-041900-generic linux-image-unsigned-4.19.0-041900-generic
sudo dpkg -P 内核名称  #通过deb包暗装的
#关闭启动内核自动更新
方式一：
sudo apt-mark hold linux-image-generic linux-headers-generic
sudo apt-mark unhold linux-image-generic linux-headers-generic
方式二：
修改系统配置文件， /etc/apt/apt/conf.d
将10periodic，20auto-upgrades 配置中的1改为0

# 删除内核
dpkg -l | grep linux-image #列出所有内核文件
#删除指定内核
dpkg -r --force-all 内核名称
```

使用指定版本内核  /boot 文件是内核相关的信息

```shell
grep menuentry /boot/grub/grub.cfg
```

例如文件如下:

```shell
if [ x"${feature_menuentry_id}" = xy ]; then
  menuentry_id_option="--id"
  menuentry_id_option=""
export menuentry_id_option
menuentry 'Ubuntu' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-5bce3795-da96-4c6f-bed2-67d37185a77d' {
submenu 'Ubuntu 高级选项' $menuentry_id_option 'gnulinux-advanced-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu，Linux 4.8.0-26-lowlatency' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-45-lowlatency-advanced-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu, with Linux 4.8.0-26-lowlatency (upstart)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-45-lowlatency-init-upstart-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu, with Linux 4.8.0-26-lowlatency (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-45-lowlatency-recovery-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu，Linux 4.8.0-26-generic' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-45-generic-advanced-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu, with Linux 4.8.0-26-generic (upstart)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-45-generic-init-upstart-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu, with Linux 4.8.0-26-generic (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-45-generic-recovery-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu，Linux 4.4.0-21-generic' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-21-generic-advanced-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu, with Linux 4.4.0-21-generic (upstart)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-21-generic-init-upstart-5bce3795-da96-4c6f-bed2-67d37185a77d' {
    menuentry 'Ubuntu, with Linux 4.4.0-21-generic (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.4.0-21-generic-recovery-5bce3795-da96-4c6f-bed2-67d37185a77d' {
menuentry 'Memory test (memtest86+)' {
menuentry 'Memory test (memtest86+, serial console 115200)' {
```

menuentry 代表一个内核, 从0开始记数字: 例如如果使用**以4.4.0-21内核版本启动，则将文件/etc/default/grub中**

```shell
#GRUB_DEFAULT=0 
GRUB_DEFAULT=”Ubuntu，Linux 4.4.0-21-generic“
```

- **sudo update-grub** 然后重启执行uname -r  查看系统内核

```shell
#禁止内核更新
sudo apt-mark hold linux-image-5.3.0-42-generic
sudo apt-mark hold linux-image-extra-5.3.0-42-generic

#重启内核更新
sudo apt-mark unhold linux-image-5.3.0-42-generic
sudo apt-mark unhold linux-image-extra-5.3.0-42-generic
```

#### .6. cuda安装

```shell
#---------- cudnn 有俩种安装方式， deb 安装，或者下载源代码替换 安装  推荐deb安装
#ubuntu cudnn 安装教程 https://developer.nvidia.com/rdp/cudnn-download  cudnn其实是一些加速CUDA性能的库，首先按照解压放到CUDA的相应路径中
然后把其中的lib64关联到环境变量当中
#将三个deb文件都下载下同时安装，否则会报错
sudo dpkg -i libcudnn7*.deb
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191206194750114.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201107092240848.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191212224333748.png)

> 如果nvcc --version 没有显示，可能是环境变量没有弄好。

网上一个脚本

```shell
# WARNING: These steps seem to not work anymore!

#!/bin/bash

# Purge existign CUDA first
sudo apt --purge remove "cublas*" "cuda*"
sudo apt --purge remove "nvidia*"

# Install CUDA Toolkit 10
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && sudo apt update
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo apt update
sudo apt install -y cuda

# Install CuDNN 7 and NCCL 2
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt update
sudo apt install -y libcudnn7 libcudnn7-dev libnccl2 libc-ares-dev

sudo apt autoremove
sudo apt upgrade

# Link libraries to standard locations
sudo mkdir -p /usr/local/cuda-10.0/nccl/lib
sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/nccl/lib/
sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda-10.0/lib64/

echo 'If everything worked fine, reboot now.'
```

- **window上查看cuda版本** 
- https://developer.nvidia.com/rdp/cudnn-archive

```shell
nvcc --version   #使用命令
#进入相应的目录  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
```

### 3. 无线网卡驱动问题

- 方式一：直接运行该命令

```shell
sudo apt-get update #更新软件源
sudo apt install broadcom-sta-dkms
#会提示有未能满足的依赖关系执行下面命令
sudo apt --fix-broken install  #下载对应的模块
```

- 无线网卡驱动列表：https://www.intel.com/content/www/us/en/support/articles/000005511/wireless.html
- 教程: https://www.1024sou.com/article/470920.html

```shell
# 查看网卡类型
lspci | grep -i net
##驱动自动加载
sudo modprobe e1000e
##重启网卡（注意用ifconfig -a 确认自己网卡名字）
##停止
sudo ifconfig eth0 down
##启动
sudo ifconfig eth0 up
```

### Resource

- https://blog.csdn.net/hongyiWeng/article/details/121233439  cuda10.2+ubuntu 18.04+kernel 4.19

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/cudakerneldowload/  

