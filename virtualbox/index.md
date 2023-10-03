# virtualBox


### 1. VirtualBox配置SSH远程登录

```shell
sudo apt-get install openssh-server
sudo apt-get install openssh-client
sudo apt-get install ssh
ssh localhost  
#ssh: connect to hostlocalhost port 22: Connection refused 表示没有安装成功
#关闭防火墙
sudo ufw disable
#-----桥接方式网络
ip：虚拟机IP  port：22
#----- net方式端口映射
sudo vi /etc/network/interfaces
auto eth0
iface eth0 inet dhcp

```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201216204330355.png)

> 选择【高级】-->【端口转发】，将虚拟机的端口22映射为任一端口，如2222其他的服务也可以做相同的设置，22是ssh的服务端口。`ssh -p2222 user@宿主机IP`, 可以使用例如vscode之类编写代码。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201216204425002.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/virtualbox/  

