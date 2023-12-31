# KaliRpi


### 0. img下载地址

- 登陆用户密码： kali, kali;


### 1.  镜像源地址

```shell
kali@kali:~$ sudo vi /etc/apt/sources.list
#阿里云镜像
deb https://mirrors.aliyun.com/kali kali-rolling main non-free contrib
deb-src https://mirrors.aliyun.com/kali kali-rolling main non-free contrib
#中科大
deb http://mirrors.ustc.edu.cn/kali kali-rolling main non-free contrib
deb-src http://mirrors.ustc.edu.cn/kali kali-rolling main non-free contrib
#浙大
deb http://mirrors.zju.edu.cn/kali kali-rolling main contrib non-free
deb-src http://mirrors.zju.edu.cn/kali kali-rolling main contrib non-free
#东软大学
deb http://mirrors.neusoft.edu.cn/kali kali-rolling/main non-free contrib
deb-src http://mirrors.neusoft.edu.cn/kali kali-rolling/main non-free contrib
#重庆大学
deb http://http.kali.org/kali kali-rolling main non-free contrib
deb-src http://http.kali.org/kali kali-rolling main non-free contrib
#官方源
#deb http://http.kali.org/kali kali-rolling main non-free contrib
#deb-src http://http.kali.org/kali kali-rolling main non-free contrib
```

### 2. VPN 配置

```shell
apt install x11vnc -y
x11vnc -storepasswd 123456 /home/kali/.vnc/passwd
cd /lib/systemd/system
sudo vim x11vnc.service
###
[Unit]
Description=Start x11vnc at startup
After=multi-user.target
[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -auth guess -forever -loop -noxdamage -repeat -rfbauth /home/kali/.vnc/passwd -rfbport 5900 -shared
[Install]
WantedBy=multi-user.target
###
systemctl enable x11vnc.service
reboot
netstat -tunlp
```

> -storepasswd：保存一个新的密码
> 123456：这个是密码
> /etc/x11vnc.pass：保存在这个路径下的 “x11vnc.pass” 文件

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210118235103487.png)

### 4. 黑客工具

> - [Aircrack-ng](https://www.aircrack-ng.org/)：无线 [WEP/WPA 破解实用程序](https://null-byte.wonderhowto.com/how-to/hack-wi-fi-getting-started-with-aircrack-ng-suite-wi-fi-hacking-tools-0147893/)。
> - [BeEF](https://beefproject.com/)：通过 Web 应用程序的浏览器[漏洞利用框架](https://null-byte.wonderhowto.com/how-to/hack-like-pro-hack-web-browsers-with-beef-0159961/)。
> - [Burp Suite](https://portswigger.net/burp/)：专为 [Web 应用程序安全性](https://null-byte.wonderhowto.com/how-to/hack-like-pro-hack-web-apps-part-4-hacking-form-authentication-with-burp-suite-0163642/)而设计的图形应用。
> - [Hydra](https://github.com/vanhauser-thc/thc-hydra)：登录[密码暴力破解](https://null-byte.wonderhowto.com/how-to/hack-like-pro-crack-online-web-form-passwords-with-thc-hydra-burp-suite-0160643/)实用程序。
> - [Nikto](https://cirt.net/Nikto2)：Web [服务器安全扫描器](https://null-byte.wonderhowto.com/how-to/hack-like-pro-find-vulnerabilities-for-any-website-using-nikto-0151729/)。
> - [Maltego](https://www.paterva.com/web7/)：开源取证和情报收集。
> - [Nmap](https://nmap.org/)：端口扫描器和[网络映射器](https://null-byte.wonderhowto.com/how-to/hack-like-pro-advanced-nmap-for-reconnaissance-0151619/)。
> - [Wireshark](https://www.wireshark.org/download.html)：用于[网络流量分析](https://null-byte.wonderhowto.com/news/8-wireshark-filters-every-wiretapper-uses-spy-web-conversations-and-surfing-habits-0134508/)的图形应用程序。
> - [ieaseMusic](https://github.com/trazyn/ieaseMusic)： 音乐播放器
> - [spotify](https://www.spotify.com/us/download/linux/)
> - [CoCoMusic](https://github.com/xtuJSer/CoCoMusic/releases)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kalirpi/  

