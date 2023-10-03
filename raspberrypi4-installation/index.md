# RaspberryPi4 Installation


> ​      Raspberry Pi is the most attractive SBC among the developers, programmers, and students. It helps to build any prototypes and develop applications or software. Nowadays, Raspberry Pi can generate output like a desktop computer and has the ability to serve individuals and small businesses. Low power draw, small form factor, no noise, and solid-state storage are the main reasons behind the widespread use of Raspberry Pi. 

## 1. 引脚介绍

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200508112423709.png)

![](https://img-blog.csdn.net/20171013145614407?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzE2MjAzNQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200508152243708.png)

> Raspberry Pi 4 采用了博通 BCM2711 SoC，包含四个 1.5GHz Cortex A72 CPU 核心，与 Raspberry Pi 3 的四核 Cortex A53 CPU 相比，Raspberry Pi 4 的内存**可选 1GB、2GB 和 4GB DDR4**，比 Pi 3 的 1GB RAM 不知高到哪里去了。Raspberry Pi 4 包含两个 USB 2 端口、两个 USB 3 端口，通过一个 USB-C 端口供电，此外还有千兆以太网接口和耳机接口，两个 micro HDMI 端口，支持两台 4K 显示器。

## 2. 操作系统

### 2.1. [Raspbian](https://www.raspberrypi.org/downloads/raspbian/)

> ​       the official OS, and it can be used on all models of Raspberry. This free operating system is known as the modified version of the popular OS Debian. It can serve all the general purposes for the Raspberry users

### 2.2. LIBREELEC

> LIBREELC is a really small and open-source JEOS. It is often compared with OpenELEC, although the boot time is much faster in LIBREELEC. Just like other Linux distros, it offers backbones for backdated hardware. It was launched on 4 April 2016 and intended to bring major creative improvements to generate better multimedia output than OpenELEC. 

### 2.3. [Windows IoT Core](https://developer.microsoft.com/en-us/windows/iot)

> ​	 This is a powerful Raspberry Pi OS. Specially designed for writing sophisticated programs and making prototypes. It was intended to serve the developers and programmers. It has enabled the coders to [make IoT projects](https://www.ubuntupit.com/best-internet-of-things-projects-iot-projects-that-you-can-make-right-now/) using Raspberry Pi and Windows 10. You can check a lot of Microsoft projects listed on their site. 

### 2.4. [Ubuntu Core](https://ubuntu.com/download/iot/raspberry-pi)

> Ubuntu is one of the widely used operating systems all over the world. This version of Ubuntu is designed for [building and managing Internet of Things applications](https://www.ubuntupit.com/most-remarkable-iot-applications-in-todays-world/). This project is open source and backed by so many developers that you can not even imagine. 

- Ubuntu has 20+ other derivatives. So that if you decide to use Core, you will be a member of an active and welcoming forum.
- Covers the basic sets of the platform, services, and technologies to work more efficiently with IoT projects. 
- This OS is lightweight and highly secure. Besides, you can restrict each application and its data from other applications. 
- Focuses on meeting the requirement of IoT devices and their distributors. 
- Public and Private key is generated while two steps validation and authentication at every step makes it more secure. 
- Several snaps like a core snap, a kernel snap, a gadget snap are used to build the Ubuntu Core system. 
- You can distribute the application using Snaps that makes it easy to distribute through Linux distribution from a play store. 

### 2.5. [Kali Linux](https://www.kali.org/)

> Kali is one of the best Linux distributions available to run in Raspberry Pi. You will not get too many changes than any other ARM. You can use this image in the desktop computers also by upgrading to the full package known as kali-Linux-full. You can use additional tools that are available on the website to extend the capabilities of certain features. 

- You will need a Class 10 SD card with at least 8GB data storage for installing a prebuilt Kali Linux image on your Raspberry. 
- This Debian based distribution offers a lot of [security and forensics tools](https://www.ubuntupit.com/an-ultimate-list-of-ethical-hacking-and-penetration-testing-tools-for-kali-linux/) to ensure the security of your project or applications. 
- You can ensure security through research, testing, forensic reports, or even reverse engineering to accomplish your goal. 
- If you are a developer and have the desire to accomplish high computing projects, then Kali is the best choice for you. 
- Besides, If you want to indulge in Ethical Hacking, Kali can help in Cracking Wi-Fi password, spoofing, and testing networks all can be done.

### 2.6. [Kano](https://kano.me/downloadable/row)

> It can be referred to as an educational project entirely planned and designed for the children. Kano manufactures computer kits to inspire children to learn how a computer works, how to write code, or how to work with basic projects. Not only children but also the individuals who have an interest in developing art, music, apps, games software can start by the starting kit distributed by Kano. 

- Kano offers an open-source OS to use in Raspberry Pi, and you will be guided through a setup wizard after completing the installation. 
- You will need to create an account and set a username to start the adventure. The OS comes with several story modes and a fresh set of features. 
- Other applications like Minecraft, Youtube, web browsers are also available. These most used applications are usually located on the menu. 
- You can start building small projects right after installing the OS with the dedicated apps. 
- This is a new kind of distribution. But to help you, they have provided a lot of books, resources, and instruction videos on their website. 

### 2.7. Chromium Os

> This open-source version of the Chromium OS offered by Google. That was intended to use on Chromebook computers, but it is also available for Raspberry Pi. It can single-handedly convert your Raspberry Pi into a desktop PC as it allows users to run powerful applications using [cloud computing](https://www.ubuntupit.com/best-cloud-computing-courses-and-certifications/) rather than depending on the hardware resources. 

- If you do not need anything else rather than web browsing, then a Raspberry Pi and Chromium OS are what you need. 
- It comes with all the applications offered by Google like Gmail, drive, access, docs, keeps, and so on. 
- Chromium can be used on Raspberry Pi 3 or 3B+ devices as there are no images available for Pi Zero or Raspberry 4. 
- After downloading the image, compress it into XZ format, which can be expanded in the Linux distribution system. 
- You will need a Gmail account to get booting for the first time. The environment of the OS is pretty much different than what you have seen in Chromebook. 

### 2.8. [OpenWrt](https://openwrt.org/)

> OpenWrt provides a fully writable filesystem with package management. This frees you from the application selection and configuration provided by the vendor and allows you to customize the device through the use of packages to suit any application. For developers, OpenWrt is the framework to build an application without having to build a complete firmware around it; for users this means the ability for full customization, to use the device in ways never envisioned.

## 3. Raspbian Installation

### 3.1. Device Lists

- 树莓派4b  单板2G  335元
- 电源，外壳，HDMI线，散热片，16GTF卡，读卡器，小风扇，网线，引脚尺，扩展板+铜柱，按键，点阵，LED，排线，点阵转接板          60元

- [树莓派镜像](https://www.raspberrypi.org/downloads/raspbian/)

- 烧录镜像软件[Etcher](https://etcher.io/)

### 3.2. 连接

> 注意： Raspbian 系统默认用户名：Pi    密码：  raspberry

#### 3.2.1.  无屏幕有线

1. 在SD卡根目录（boot中）新建 ”ssh“ 文件，无后缀
2. 硬件连接开机（ 按电源开关，红灯亮，然后绿灯闪烁，加载成功）
3. 找到树莓派ip地址，  路由器查看，手机电脑共享， IP scanner 查看
4. 远程连接

#### 3.2.2. 无线Wifi设置

查看电脑已存WiFi密码：https://xw.qq.com/cmsid/20180310A0227F00

```json
#在根目录下新建文件  wpa_supplicant.conf 
country=CN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
    ssid="liudongdong"
    psk="12345678"
    key_mgmt=WPA-PSK
    priority=1
}
```

#### 3.2.3.  设置静态ip

```shell
#查看网络配置文件
sudo cat /etc/network/interfaces
# 根据注释信息进行修改
sudo nano /etc/dhcpcd.conf
```

#### 3.2.4. VNC 远程登录

>  VNC是什么？远程登录树莓派，一般是两种方法，一种是使用SSH协议的PUTTY，一种是VNC的VNC Viewer。VNC具有图形界面，更受大家喜欢，你不需要给树莓派外置任何设备，就能在电脑上像运行一个普通程序一样，操作树莓派。

1. sudo raspi-config 
2. 进入 Interfacing Options
3. 选择VNC， Enable
4. 输入vncserver ,会在最后一行显示端口号
5. 使用VNC Viewer软件或其他登录

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200508151641901.png)

> 错误： VNC View 连接 Cannot currently show the desktop 
>
> 方案： changing the resolution to the highest（更改更高的分辨率即可），sudo raspi-config  ->advanced

### 3.3.  软件源更换

```shell
sudo nano /etc/apt/sources.list
# 编辑 `/etc/apt/sources.list` 文件，删除原文件所有内容，用以下内容取代：
deb http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main non-free contrib rpi
deb-src http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main non-free contrib rpi

# 编辑 `/etc/apt/sources.list.d/raspi.list` 文件，删除原文件所有内容，用以下内容取代：
deb http://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ buster main ui
```

[树莓派的所有软件源地址](https://www.raspbian.org/RaspbianMirrors)

架构：

-  armhf

版本：

-  wheezy。  jessie，   stretch，     buster
- [镜像站：](https://mirror.tuna.tsinghua.edu.cn/help/raspbian/)

注：Raspbian 系统由于从诞生开始就基于（为了armhf，也必须基于）当时还是 testing 版本的 7.0/wheezy，所以 Raspbian 不倾向于使用 stable/testing 表示版本。  需要选择正确的版本

> #deb http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ stretch main contrib non-free rpi
> #deb-src http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ stretch main contrib non-free rpi
>
> 第一个单词代表包的类型，**deb**表示二进制包，**deb-src**表示源码包。
>
> 第二个网址表示源的地址。
>
> 第三个单词表示`系统的版本`，既可以是`[ wheezy | jessie | stretch | sid ]`中的一种，也可以是`[ oldstable | stable | testing | unstable ]`中的一种。前一个系列表示系统的release code name，后一个系列表示系统的release class，前者按阶段发布，后者持续演进。
>
> 第四部分表示`接受哪种开源类型的软件`，可以包含`[ main | contrib | non-free ]`中的一个或多个。main表示纯正的遵循Debian开源规范的软件，contrib表示遵循Debian开源规范但依赖于其它不遵循Debian开源规范的软件的软件，non-free表示不遵循Debian开源规范的软件。Debian开源规范指[DFSG](https://link.jianshu.com?t=http://www.debian.org/social_contract#guidelines)（Debian 自由软件指导方针）。

> 问题记录： [Unable to correct problems, you have held broken packages](https://askubuntu.com/questions/223237/unable-to-correct-problems-you-have-held-broken-packages)
>
> 原因：  可能是下载安装包中途停止， run    sudo apt install -f
>
> ​              可能是软件源不对，是否为系统版本
>
> 问题： LC_ALL: cannot change locale  (en_US.UTF-8) 
>
> 解决：
>
> ```
> echo "LC_ALL=en_US.UTF-8" >> /etc/environment
> echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
> echo "LANG=en_US.UTF-8" > /etc/locale.conf
> locale-gen en_US.UTF-8
> ```

### 3.4. 屏幕自动关闭

在目前的新版本的树莓派镜像中2019-04-08-raspbian-stretch.img，需要使用下面的方法禁止屏幕休眠：

方式一：

sudo vi /etc/lightdm/lightdm.conf

取消其中的注释　#xserver-command=X

并修改为                 xserver-command=X -s 0 -dpms

-s # –设置屏幕保护不启用       dpms 关闭电源节能管理
重启树莓派

方式二：

```shell
sudo nano /etc/profile.d/Screen.sh
#文件写入
xset dpms 0 0 0 
xset s off

sudo chmod +x /etc/profile.d/Screen.sh
#临时唤醒
xset dpms force on
```

## 4. Module

### 4.1.  Camera

```shell
ls /dev    #查看是否有video0 设备
#使能摄像头
sudo raspi-config
# 进入camere  enable
sudo reboot
```

**Python `picamera`库允许您控制相机模块并创建出色的项目。**

1.打开Python 3编辑器，例如**Thonny Python IDE**：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200514094638461.png)

2. 打开一个新文件并将其另存为`camera.py`。

**注意：** **永远不要将文件保存为`picamera.py`**，这一点很重要。

```python
from picamera import PiCamera
from time import sleep
camera = PiCamera()
camera.start_preview()
sleep(5)
camera.stop_preview()

#拍摄五张照片
camera.start_preview()
for i in range(5):
    sleep(5)
    camera.capture('/home/pi/Desktop/image%s.jpg' % i)
camera.stop_preview()
#https://www.jianshu.com/p/6644d4932136
```

```
raspistill -o Desktop/image.jpg
raspistill -o Desktop/image-small.jpg -w 640 -h 480
raspivid -o Desktop/video.h264  #录制视频
```

### 4.2. 4.2. 语言输出

> 树莓派默认是自动选择音频输出口的，当你同时外接了 HDMI 显示器和 3.5MM 耳机或音箱时，有时候你希望手动指定其中的一个作为音频的输出设备，那么可以通过下面的方法配置实现。如果你外接了DAC扩展板，配置好驱动之后也可以用这种方法选择DAC输出。

命令行输入：alsamixer

确定声卡设备是否可以访问，而且没有静音（按m可以切换）。

通过键盘的上下箭头可以调整音量，确定没问题后按Esc退出。

下一步在命令行输入speaker-test -t sine

如果能听到蜂鸣声，那说明没问题，如果听不到，那返回上一个命令调整。

1. 输入```sudo raspi-config```，这样就进入了树莓派的设置面板。
2. 选择第7项Advanced Options并回车，然后选择第4项Audio再回车：
3. 一共有三个选项，一个是Auto，一个是HDMI，一个是3.5mm耳机。
4. 我显示屏待音频输出，选择HDMI

## 5. Develop Language

### 5.1. WiringPi

> ***[WiringPi](https://github.com/WiringPi/WiringPi-Python/)*** is a ***PIN*** based GPIO access library written in C for the BCM2835, BCM2836 and BCM2837 SoC devices used in all **Raspberry Pi.** versions. It’s released under the [GNU LGPLv3](http://www.gnu.org/copyleft/lesser.html) license and is usable from C, C++ and RTB (BASIC) as well as many other languages with suitable wrappers (See below) It’s designed to be familiar to people who have used the Arduino “*wiring*” system1 and is intended for use by experienced C/C++ programmers. It is not a newbie learning tool.  [Tutorial:](http://www.waveshare.net/study/portal.php?mod=view&aid=603)

```shell
#  安装  
pip install wiringpi

#General IO
import wiringpi

# One of the following MUST be called before using IO functions:
wiringpi.wiringPiSetup()      # For sequential pin numbering
# OR
wiringpi.wiringPiSetupSys()   # For /sys/class/gpio with GPIO pin numbering
# OR
wiringpi.wiringPiSetupGpio()  # For GPIO pin numbering

#G  IO operation
wiringpi.pinMode(6, 1)       # Set pin 6 to 1 ( OUTPUT )
wiringpi.digitalWrite(6, 1)  # Write 1 ( HIGH ) to pin 6
wiringpi.digitalRead(6)      # Read pin 6
#Hook a speaker up to your Pi and generate music with softTone. Also useful for generating frequencies for other uses such as modulating A/C.
wiringpi.softToneCreate(PIN)
wiringpi.softToneWrite(PIN, FREQUENCY)
```

### [5.2. RPIO](https://pythonhosted.org/RPIO/genindex.html)

> - PWM via DMA (up to 1µs resolution)
> - GPIO input and output (drop-in replacement for [RPi.GPIO](http://pypi.python.org/pypi/RPi.GPIO))
> - GPIO interrupts (callbacks when events occur on input gpios)
> - TCP socket interrupts (callbacks when tcp socket clients send data)
> - Command-line tools `rpio` and `rpio-curses`
> - Well documented, fast source code with minimal CPU usage
> - Open source (LGPLv3+)

```
#installation
pip install RPi.GPIO
#gpio readall   读取所有引脚
import PRi.GPIO as GPIO

#设置模式
GPIO.setmode(GPIO.BCM)
# GPIO.setmode(GPIO.BOARD)
# 设置输入输出
GPIO.setup(11, GPIO.OUT)
GPIO.setup(pin,GPIO.IN)
#
GPIO.output(11, True)
GPIO.output(11, False)
#读取引脚状态
GPIO.input(12)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/raspberrypi4-installation/  

