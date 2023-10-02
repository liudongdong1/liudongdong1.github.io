# RPi Function Plugin


> 记录linux，debain 系统上常用功能插件，方便以后使用。

## 1.[Mplayer 视频音乐播放](https://pypi.org/project/mplayer.py/)

> 1. **[Player](https://github.com/baudm/mplayer.py/wiki/Player)** provides a clean, Pythonic interface to MPlayer.
> 2. **[AsyncPlayer](https://github.com/baudm/mplayer.py/wiki/AsyncPlayer)** is a *Player* subclass with asyncore integration (POSIX only).
> 3. **[GPlayer](https://github.com/baudm/mplayer.py/wiki/GPlayer)** is a *Player* subclass with GTK/GObject integration.
> 4. **[QtPlayer](https://github.com/baudm/mplayer.py/wiki/QtPlayer)** is a *Player* subclass with Qt integration (same usage as AsyncPlayer)
> 5. **[GtkPlayerView](https://github.com/baudm/mplayer.py/wiki/GtkPlayerView)** provides a basic (as of now) PyGTK widget that embeds MPlayer.
> 6. **[QPlayerView](https://github.com/baudm/mplayer.py/wiki/QPlayerView)** provides a PyQt4 widget similar to *GtkPlayerView* in functionality.

```
# 安装
pip install mplayer.py

player = mplayer.Player()
player.pause() #暂停
player.filename #显示文件名
player.time_pos += 5 #快进5s
player.time_pos -= 5 #快退5s
player.stream_length #查看视频长度
player.stream_pos #查看视频现在的位置， 根据上面可以做出进度条
player.volume #显示音量
player.volume(+30.0) #升高音量
player.volume(-30.0) #降低音量
player.quit() #关闭视频
```

## 2. [人脸识别本地Reconition](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-pip \
    zip
sudo apt-get clean
# 下载picamera
sudo apt-get install python3-picamera
sudo pip3 install --upgrade picamera[array]
# enable a larger swap file size
sudo nano /etc/dphys-swapfile
< change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024 and save / exit nano >
sudo /etc/init.d/dphys-swapfile restart
# 下载dlib
mkdir -p dlib
git clone -b 'v19.6' --single-branch https://github.com/davisking/dlib.git dlib/
cd ./dlib
sudo python3 setup.py install --compiler-flags "-mfpu=neon"
# 下载face_recognition
sudo pip3 install face_recognition
# revert the swap file size change
sudo nano /etc/dphys-swapfile
< change CONF_SWAPSIZE=1024 to CONF_SWAPSIZE=100 and save / exit nano >
sudo /etc/init.d/dphys-swapfile restart
```

## 3. 聊天功能

Slack其实是一个方便进行团队协作和交流的聊天工具，官网在此：https://slack.com/。SlackBot是Slack提供的一个Python聊天机器人框架。毕竟是官方支持的机器人框架，所以就没有模拟网页端的微信机器人框架的那些限制了。在官网注册帐号，并创建一个SlackBot，会得到一个Api Token，然后就可以用这个Token基于SlackBot框架开发自己的聊天机器人了。手机端有Slack客户端，打开客户端和机器人对话，就能实现与树莓派上运行的这个SlackBot交流了。具体文档可以参考：https://github.com/lins05/slackbot

## 4. 网站的部署和内网穿透

室外天气

https://www.seniverse.com/widget/more

树莓派状态

http://shumeipai.nxez.com/2014/10/04/get-raspberry-the-current-status-and-data.html

## 5. [openmediavault](https://www.openmediavault.org/)

> openmediavault is the `next generation network attached storage (NAS) solution` based on Debian Linux. It contains services like `SSH, (S)FTP, SMB/CIFS, DAAP media server, RSync, BitTorrent client and many more`. Thanks to the modular design of the framework it can be enhanced via plugins. openmediavault is primarily designed to be used in `home environments or small home offices, but is not limited to those scenarios`. It is a simple and easy to use out-of-the-box solution that will allow everyone to install and administrate a Network Attached Storage without deeper knowledge.  [Source Code](https://github.com/openmediavault/openmediavault/)

- **[Img Installation](https://blog.csdn.net/founderznd/article/details/52325332?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control)**: download .ios file, and flash in sd card. I haved tried different file format like NTFS, Fat32, fat16, but meet error: `Non Fat Paritition, bios start failer, the question doesn't solved;`
- **Based on Raspbian Buster system:**  first flash the buster img to the sd card, and start the system, config the network as normal;
- https://www.howtoforge.com/tutorial/install-open-media-vault-nas/
- https://www.cnblogs.com/JiYF/p/12991483.html

```bash
#或取自动安装脚本
wget -O - https://github.com/OpenMediaVault-Plugin-Developers/installScript/raw/master/install | sudo bash
sudo chmod a+x install.sh
sudo bash install.sh
```

- **Usage**

> 打开浏览器，输入树莓派IP地址; 用户名：admin 密码：openmediavault

- 使用过一次，感觉使用不太好，不如直接开启samba服务或者ftp服务。

## 6. [GoogleAIY VoiceKit](https://www.youtube.com/watch?v=9BmUNA1LBTw)

## 7. [KDE](https://www.cnbeta.com/articles/soft/960673.htm)

> 不推荐使用；

## 9. 视频播放VLC

> [流服务地址：](https://blog.csdn.net/u014162133/article/details/109180771?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-3&spm=1001.2101.3001.4242)

## 10. 树莓派乐高分类器

- https://mp.weixin.qq.com/s/1aG_lQL9ewBR6wAifHKvaQ

## 11. [DNS 广告拦截器](https://github.com/pi-hole/pi-hole)

- https://github.com/pi-hole/AdminLTE

![image-20210321162013844](D:\work_personnal\typoraPicture\image-20210321162013844.png)

## 12. **[ Raspberry-Pi-Security-Camera](https://github.com/Cyebukayire/Raspberry-Pi-Security-Camera)**

> `Motion detector`, Full body detection, Upper body detection, `Cat face detection,`` Smile detection`, Face detection (haar cascade), Silverware detection, Face detection (lbp), and `Sending email notifications`

![image-20210616141650169](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210616141650169.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/rpi-function-plugin/  

