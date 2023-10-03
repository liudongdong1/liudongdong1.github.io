# NetworkRelative


### 1. 网络连接

```python
#查看网卡信息
sudo lshw -class network
# error Networking disabled
sudo service network-manager stop
sudo rm /var/lib/NetworkManager/NetworkManager.state
sudo service network-manager start
# 方式二： 是否启用该网卡
sudo ifconfig ens33 up  #ens33 为网卡的logic name; ifconfig 查看
sudo dhclient ens33 # 动态获取ip一次
ifconfig -a #查看IP
# ----------网卡配置方式
sudo vi /etc/network/interfaces
sudo /etc/init.d/networking restart  #修完完成生效
# 动态DHCP
并用下面的行来替换有关eth0的行:
# The primary network interface - use DHCP to find our address
auto eth0
iface eth0 inet dhcp
#静态ip地址
auto eth0  
iface eth0 inet static  
address 192.168.3.90  
gateway 192.168.3.1  
netmask 255.255.255.0  
#network 192.168.3.0  
#broadcast 192.168.3.255 
#设置虚拟ip
auto eth0:1  
iface eth0:1 inet static  
address 192.168.1.60  
netmask 255.255.255.0  
network x.x.x.x  
broadcast x.x.x.x  
gateway x.x.x.x 
```

### 2. ssh 免密登录

```shell
#  生成公钥-私钥文件对
ssh-keygen  #如果生成过了，就可以不需要操作，在C:/Users/dell/.ssh/id_rsa， Linux待查
#将公钥拷贝到服务器端， 并存储到 authorized_keys; 如果没有该目录则创建；
cat id_rsa.pub >> ~/.ssh/authorized_keys  
#在服务器端做必要的ssh配置
vim /etc/ssh/sshd_config
##RSAAuthentication yes
##PubkeyAuthentication yes
##PermitRootLogin yes
service ssh restart
passwd root
```

> 通过设置~/.ssh/config文件实现不同服务器用不同私钥文件的需求
>
> `ssh root@host -p port -i path`

```text
Host alpine  #指定登录别名
    HostName 192.168.52.129   #指定IP或者HOST
    User root  #指定用户名
    IdentityFile C:/Users/liudongdong/.ssh/id_rsa   #指定私钥文件
```

### 3. VNC

1.开启ssh，并允许root密码登录

```python
apt install openssh-server ssh
vi /etc/ssh/sshd_config
UsePAM yes
PermitRootLogin yes
PasswordAuthentication yes
#修改以上配置
systemctl enable ssh && systemctl restart ssh
#此时可以用root 密码登录ssh
```

2.开启屏幕共享

打开Settings --> Screen Sharing -->激活并设置密码

\#如果没有Screen Sharing选项，可能是vino没有安装，尝试apt install vino 安装

\#激活后，用netstat -tulp | grep 59,查看端口是否监听590X

\#如果正常，可以尝试用vnc连接，可能会出现“no security type suitable for RFB3.3”的错误。第3步就是解决这个问题。

```shell
gsettings set org.gnome.Vino require-encryption false
```

> [Screen-Sharing “no network selected for sharing” problem in unity control cent	1er (18.04)](https://askubuntu.com/questions/1070520/screen-sharing-no-network-selected-for-sharing-problem-in-unity-control-center)

```shell
vim /etc/NetworkManager/NetworkManager.conf
[ifupdown]
managed=true
sudo touch /etc/NetworkManager/conf.d/10-globally-manage-devices.conf
```

### 4. 网络命令 netstat ,top

```bash
# net-tools   包括ifconfig,netstat 等网络工具
top: #查看电脑个进程占用资源情况  b 高亮显示当前进程.
netstat -a :Listing all ports (both TCP and UDP) using option.
netstat -l : active listening ports connections
netstat -s : displays statistics by protocol
netstat -i : show the network interface
netstat -r : show the routing
netstat -ie : like ifconfig
netstat -ap | grep http : find the listening program
#查找程序是否运行
#pgrep command – Looks through the currently running bash processes on Linux and lists the process IDs (PID) on screen.
pgrep nginx
#pidof command – Find the process ID of a running program on Linux or Unix-like system
pidof nginx
#ps command – Get information about the currently running Linux or Unix processes, including their process identification numbers (PIDs).
ps aux | grep nginx
```

### 5. 设置代理

```shell
set | grep -i all_proxy
# Unset socks proxy
unset all_proxy     #根据上个命令输出决定是否用大写还是小写
unset ALL_PROXY      #系统中的设置还在
# Install missing dependencies:
pip install pysocks
# Reset proxy
source ~/.bashrc
```

### 6. 文件下载

```bash
wget -O  #下载并以不同的文件名保存
wget -b #后台下载   tail -f wget-log  查看下载速度
wget –spider url #测试下载链接是否可用等等
```

### 7. 文件夹共享

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200828150147.png)

> 访问目录： /mnt/hgfs/winshare 

### 7. SCP文件互传

```shell
#本地复制到远程
scp local_file remote_username@remote_ip:remote_folder  #复制目录
scp -r local_folder remote_username@remote_ip:remote_folder  #复制文件
#scp 命令将服务器上文件拷贝至本地
# tar -cvf script.tar script 打包文件
scp remote_username@remote_ip:remote_folder local_file   #复制目录
scp -P port file_name user@ip:/dir_name
ssh -p xx  user@ip    #区别记忆一下
```

### 8. Fpt&**Samba**文件传输

```shell
apt install -y vsftpd
#ftpd 配置文件/etc/vsftpd/vsftpd.conf
service vsftpd start
service vsftpd status
#重启fpt
systemctl restart vsftpd
systemctl enable vsftpd
#配置文件内容
listen=[YES|NO]              是否以独立运行的方式监听服务
listen_address=IP地址        设置要监听的IP地址
listen_port=21               设置FTP服务的监听端口
download_enable＝[YES|NO]    是否允许下载文件
userlist_enable=[YES|NO]     设置用户列表为“允许”还是“禁止”操作
userlist_deny=[YES|NO]      设置用户列表为“允许”还是“禁止”操作
max_clients=0                最大客户端连接数，0为不限制
max_per_ip=0                 同一IP地址的最大连接数，0为不限制
anonymous_enable=[YES|NO]    是否允许匿名用户访问
anon_upload_enable=[YES|NO]  是否允许匿名用户上传文件
anon_umask=022               匿名用户上传文件的umask值
anon_root=/var/ftp           匿名用户的FTP根目录
anon_mkdir_write_enable=[YES|NO]    是否允许匿名用户创建目录
anon_other_write_enable=[YES|NO]    是否开放匿名用户的其他写入权限（包括重命名、删除等操作权限）
anon_max_rate=0                     匿名用户的最大传输速率（字节/秒），0为不限制
local_enable=[YES|NO]               是否允许本地用户登录FTP
local_umask=022                     本地用户上传文件的umask值
local_root=/var/ftp                 本地用户的FTP根目录
chroot_local_user=[YES|NO]          是否将用户权限禁锢在FTP目录，以确保安全
local_max_rate=0                    本地用户最大传输速率（字节/秒），0为不限制
```

> - 使用方式一：Mobaterm 使用sftp 连接，使用ftp的时候，上传有问题：**Error EElFTPSError: Data channel transfer error (error code is 10054)**
> - 使用方式二： 文件夹操作： ftp://192.168.2.133/

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210124165341332.png)

- **Samba**

```shell
sudo apt install samba samba-common-bin
sudo nano /etc/samba/smb.conf
sudo /etc/init.d/smbd restart
#添加用户和密码
sudo smbpasswd -a pi
#密码： 123456
#window 上输入 ：   \\192.168.2.104\pi
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200508170115935.png)

- samba 配置文件详解： https://blog.csdn.net/ma111000522/article/details/75949065
- 案例： https://www.cnblogs.com/kevingrace/p/5569993.html

### 9. 网速测量speedtest   

```shell
git clone https://github.com/sivel/speedtest-cli.git
cd speedtest-cli
python speedtest.py
#具体可以看下readme操作，可以通过pip 方式安装
```

### 10. V2Ray 安装

```shell
#然后编辑`/etc/v2ray/config.json`文件
service v2ray stop 
service v2ray start 
service v2ray status
#https://github.com/FelisCatus/SwitchyOmega/wiki/GFWList
```

- ### #v2ray/go.sh脚本阅读记录

```bash
$# 表示执行脚本传入参数的个数
$*  表示执行脚本传入参数列表
$$ 表示进程id
$@表示执行脚本传入所有参数
$0 表示执行脚本名称
$1 表示第一个参数
$2 表示第二个参数
$? 表示脚本执行状态0正常，其他表示有错误
#提取文件到某个位置函数  
#获取系统本版/检查版本更新   getVersion()/checkUpdate()
#检查系统架构  SysArch()
#获得系统 install update 指令   'command -v apt-get' 判断系统是否有apt-get 指令
#prompt 颜色设置  colorEcho()
#下载文件   downloadv2ray()
# echo $VER | head -n 1 | cut -d " " -f2`  
#关闭或启动软件  stopV2ray() startV2ray() 通过检查systemctl/service 命令
#copy 文件  copyFile()
#添加执行权限 makeExecutable()
# help() 帮助提示框
installInitScript(){
    if [[ -n "${SYSTEMCTL_CMD}" ]];then
        if [[ ! -f "/etc/systemd/system/v2ray.service" ]]; then
            if [[ ! -f "/lib/systemd/system/v2ray.service" ]]; then
                cp "${VSRC_ROOT}/systemd/v2ray.service" "/etc/systemd/system/"
                systemctl enable v2ray.service
            fi
        fi
        return
    elif [[ -n "${SERVICE_CMD}" ]] && [[ ! -f "/etc/init.d/v2ray" ]]; then
        installSoftware "daemon" || return $?
        cp "${VSRC_ROOT}/systemv/v2ray" "/etc/init.d/v2ray"
        chmod +x "/etc/init.d/v2ray"
        update-rc.d v2ray defaults
    fi
    return
}
sed -i "s/10086/${PORT}/g" "/etc/v2ray/config.json"  #学习这个指令
downloadV2Ray || return $?
installV2Ray()  #包括下载到那个目录,copy了那些文件,如何根据配置文件进行配置的
remove() {# 卸载停止服务,把安转时写入的文件全部删除  
	/etc/systemd/system/v2ray.service   
	/usr/bin/v2ray
	/lib/systemd/system/v2ray.service
	/etc/init.d/v2ray
	
}
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200828215528.png)

注意端口需要和json文件配置的一样

1.FTP代理

2.HTTP代理

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200828215922.png)

3.SSL/TLS代理

4.SOCKS代理

> Sock5代理服务器则是把你的网络数据请求通过一条连接你和代理服务器之间的通道，由服务器转发到目的地。你没有加入任何新的网络，只是http/socks数据经过代理服务器的转发送出，并从代理服务器接收回应。你与代理服务器通信过程不会被额外处理，如果你用https，那本身就是加密的。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200828220120.png)

- PAC 模式：

> PAC模式就是会在你连接网站的时候读取PAC文件里的规则，来确定你访问的网站有没有被墙，如果符合，那就会使用代理服务器连接网站，而PAC列表一般都是从GFWList更新的。GFWList定期会更新被墙的网站（不过一般挺慢的）。

### 11. 修改github DNS

```shell
#https://www.linuxidc.com/Linux/2019-05/158461.htm
#github
219.76.4.4 github-cloud.s3.amazonaws.com
192.30.253.112 github.com
151.101.185.194 github.global.ssl.fastly.net
ldd@ldd:~/v2ray$ sudo vim /etc/hosts
ldd@ldd:~/v2ray$ sudo /etc/init.d/networking restart 
```

### 13. 防火墙

> 查询防火墙状态   :   [root@localhost ~]# service  iptables status
> 停止防火墙  :       [root@localhost ~]# service  iptables stop
> 启动防火墙  :       [root@localhost ~]# service  iptables start
> 重启防火墙  :       [root@localhost ~]# service  iptables restart
> 永久关闭防火墙   :   [root@localhost ~]# chkconfig  iptables off
> 永久关闭后启用   :   [root@localhost ~]# chkconfig  iptables on

### 14. 电脑浏览器很多网页打不开

- 排查零： 是不是浏览器问题；

> 如果电脑ping没有问题，浏览器打不开，清空浏览器缓存。

- 排查一： 是不是代理的问题；
- 排查二： 设置dns服务器： 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210125100911944.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210125101700537.png)

- 排查四： 清除电脑dns缓存

> `ipconfig /?`找到dns 相关的命令，执行 `ipconfig /flushdns` 命令，当出现“successfully flushed the dns resolver cache”(已成功刷新 DNS 解析缓存)；

### 15. 网络基础

- [学习链接](https://mp.weixin.qq.com/s/BQknXo0wVDyLUYY3awZLaA)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/networkrelative/  

