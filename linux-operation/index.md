# Linux Operation


## Linux 命令

### 链接  ln

```shell
sudo ln -sf /usr/bin/g++-8 /usr/bin/g++
ln - make links between files
SYNOPSIS
       ln [OPTION]... [-T] TARGET LINK_NAME   (1st form)
       ln [OPTION]... TARGET                  (2nd form)
       ln [OPTION]... TARGET... DIRECTORY     (3rd form)
       ln [OPTION]... -t DIRECTORY TARGET...  (4th form)
```

### [update-alternatives](https://www.jianshu.com/p/4d27fa2dce86)

```python
#ls -l filename  可以查看文件的软链接文件  #
#--install <链接> <名称> <路径> <优先级>  其中是指向 /etc/alternatives/<名称> 的符号链接， 在自动模式下，这个数字越高的选项，其优先级也就越高。

$ update-alternatives --install /usr/bin/python python /usr/bin/python2.7 2
# 第一个参数: --install 表示向update-alternatives注册服务名。
# 第二个参数: 注册最终地址，成功后将会把命令在这个固定的目的地址做真实命令的软链，以后管理就是管理这个软链；
# 第三个参数: 服务名，以后管理时以它为关联依据。
# 第四个参数: 被管理的命令绝对路径。
# 第五个参数: 优先级，数字越大优先级越高。
update-alternatives –remove python /usr/bin/python2.7

#-----------usage
--install <链接> <名称> <路径> <优先级>
    [--slave <链接> <名称> <路径>] ...
                           在系统中加入一组候选项。
  --remove <名称> <路径>   从 <名称> 替换组中去除 <路径> 项。
  --remove-all <名称>      从替换系统中删除 <名称> 替换组。
  --auto <名称>            将 <名称> 的主链接切换到自动模式。
  --display <名称>         显示关于 <名称> 替换组的信息。
  --query <名称>           机器可读版的 --display <名称>.
  --list <名称>            列出 <名称> 替换组中所有的可用候选项。
  --get-selections         列出主要候选项名称以及它们的状态。
  --set-selections         从标准输入中读入候选项的状态。
  --config <名称>          列出 <名称> 替换组中的可选项，并就使用其中哪一个，征询用户的意见。
  --set <名称> <路径>      将 <路径> 设置为 <名称> 的候选项。
  --all                    对所有可选项一一调用 --config 命令。
```

### man 命令

```
man -b (向前翻一屏)  space (向后翻一屏)  /keyword 查找  n: 下一个
whatis command # 查询命令执行什么功能
```

### 快捷键

```bash
Ctrl+c #在命令行下起着终止当前执行程序的作用，
Ctrl+d  #相当于exit命令，退出当前shell
win    #搜索浏览程序文件音乐文件
ctrl+L #清除屏幕
ctrl+A  #光标移到行首
super+R # terminal
ctrl+shift+prtsc  #截屏到剪切板
super+h #隐藏窗口
super+up #窗口最大化
super+down #窗口最小话
```

### 压缩包操作

```bash
tar -zxvf 4.1.2.tar.gz
unzip -d /temp test.zip  #解压到指定的目录下，需要用到-d参数
```

### evince  pdf 文件查看

## 软件安装

> apt install tilix  #terminal 工具

### CPU 温度

```shell
sudo apt install lm-sensors hddtemp
sudo sensors-detect
sensors
#如果有虚拟温度显示
sudo apt install psensor  #设置开机自启,监控温度
```

### ubuntu VMWare worstation pro 15

```shell
#下载地址 https://www.vmware.com/products/workstation-pro/workstation-pro-evaluation.html
#VMware Workstation All Key：https://www.cnblogs.com/dunitian/p/8414055.html
sudo ./VMWare-*
sudo vmware-installer -u vmware-workstation  #卸载  
```

### pytorch

```shell
#方式一：  
#      通过官方网站（https://pytorch.org/）给的方法进行安装，根据自己的系统环境及相应python，CUDA版本运行相应的命令进行安装。如果电脑中只有python3，这里的pip3可以直接就用pip代替。
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#遇问题 有关proxy
#解决方案: 在 .bashrc 中添加: export all_proxy="socks5://127.0.0.1:1080"
#cudatoolkit        pkgs/main/linux-64::cudatoolkit-10.1.243-h6bb024c_0
#  ninja              pkgs/main/linux-64::ninja-1.9.0-py37hfd86e86_0
 # pytorch            pytorch/linux-64::pytorch-1.3.1-py3.7_cuda10.1.243_cudnn7.6.3_0
 # torchvision        pytorch/linux-64::torchvision-0.4.2-py37_cu101

#方式二：   https://download.pytorch.org/whl/torch_stable.html
#	直接下载torch的whl文件，通过pip install （路径+whl文件名）
#	可以下载到本地 anaconda\install\Lib\site-packages路径下，或者在线下载安装
```

### caffe 安装

```shell
sudo apt install caffe-cuda
sudo apt build-dep caffe-cuda       # dependencies for CUDA version
sudo vim /etc/apt/sources.list   #将deb-src 注释掉
#遇到问题 dpkg-deb: error: paste subprocess was killed by signal (Broken pipe)
#Errors were encountered while processing:
# /var/cache/apt/archives/nvidia-cuda-dev_9.1.85-3ubuntu1_amd64.deb
#sudo dpkg -i --force-overwrite /var/cache/apt/archives/nvidia-418_418.39-0ubuntu1_amd64.deb
#sudo apt --fix-broken install
```

### 服务管理

```bash
sudo systemctl start application.service   #同 systemctl start application  ,系统默认查找application.service    stop, restart,reload
sudo systemctl enable/disable application.service   #start a service at boot create a symbolic link from the system’s copy of the service file (usually in /lib/systemd/system or /etc/systemd/system) into the location on disk where systemd looks for autostart files (usually /etc/systemd/system/some_target.target.wants
systemctl status application.service  #查看服务状态
systemctl list-units  # list all of the units that systemd currently has active 
systemctl list-dependencies application.service  #查找关系依赖树
```

### 搜狗输入法

```bash
sudo apt-get remove ibus
sudo apt-get purge ibus     #purge  
sudo  apt-get remove indicator-keyboard
sudo apt install fcitx-table-wbpy fcitx-config-gtk
im-config -n fcitx
选择系统设置语言 https://pinyin.sogou.com/linux/  
sudo apt-get install -f
fcitx-config-gtk3
fcitx设置 >>附加组件>>勾选高级 >>取消经典界面
Configure>>  Addon  >>Advanced>>Classic,sogouyun
#重启 把sogoupinyin放在第二个
#只用sogou 输入法一种就行了
#搜狗云输入的锅，在fcitx配置里把搜狗云拼音这个选项去掉就可以很完美的解决这问题了  解决占cpu
#中文输入时没有汉字提示时下载一个 皮肤 ,用搜狗软件打开就行可
#https://pinyin.sogou.com/skins/detail/view/info/588600?rf=cate_31_sign&tf=p

# 第二种方法，安装使用fcitx5： 使用教程https://blog.csdn.net/Mr_Sudo/article/details/124874239
```

> 如果重启以后右上角有小键盘，说明 fcitx 生效，否则说明配置有问题，看下一步。

```shell
cd ~
cd  .config/
sudo rm -rf fcitx/

sudo apt install fcitx-googlepinyin
fcitx-config-gtk3
#搜索 So   (大写 S 小写 o) ，找到，加上，搞定。sogoupinyin, 调整顺序
```

### JDK

```bash
sudo apt install openjdk-11-jdk
echo $JAVA_HOME  #使用$JAVA_HOME的话能定位JDK的安装路径的前提是配置了环境变量$JAVA_HOME，否则如下所示，根本定位不到JDK的安装路径

which java #是定位不到安装路径的。which java定位到的是java程序的执行路径。
ls -lrt /usr/bin/java

```

### VSCODE

- 格式化代码

  ```
  vs code格式化代码的快捷键如下：（来源于这里）
  On Windows Shift + Alt + F.
  On Mac Shift + Option + F.
  On Ubuntu Ctrl + Shift + I.
  ```

- 常用插件

  - Beautify
  - TODO Highlight
  - Code Spell Checker
  - IntelliSense for CSS class names in HTML

- 删除多余空行  全局替换  ^\s*(?=\r?$)\n     Alt+R 正则表达式

```bash
sudo apt-get install ubuntu-make  # 像这种开发软件去官网下载安装包
#查看版本
code --version
code #运行vscode
#Ctrl+Shift+P打开命令面板
#c_cpp_properties.json  该文件用于指定一般的编译环境，包括头文件路径，编译器的路径等。通过 Ctrl + Shift + p 打开命令行，键入关键字 "C++"，在下拉菜单中选择 "C/C++ Edit configuration"，系统即自动在 .vscode 目录下创建 c_cpp_properties.json 文件，供用户进行编译方面的环境配置。
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "clang-x64"
        }
    ],
    "version": 4
}
#build.json  该文件用于指定程序的编译规则，即如何将源文件编译为可执行程序。通过 Ctrl + Shift + p 打开命令行，键入关键字 "task"，并在下拉菜单中选择 Tasks: Configure Default Build Task -> Create tassk.json file from template -> Others ，系统即自动在 .vscode 目录下创建 build.json 文件，供用户设置具体的编译规则
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "echo",
            "type": "shell",
            "command": "g++",                   //编译时执行的程序
            "args": ["-g", "-o", "test", "test1.c"],    //传递给 command 的参数
            "problemMatcher": [
                "$gcc"
            ]
        }
    ]
}
# Ctrl+Shift+p 打开命令行，选择 Tasks:Run Build Task 运行上述编译过程
#launch.json  该文件主要与程序的调试相关。用户可通过 Ctrl+Shift+p 打开命令行，键入关键字 "launch",选择 "Debug:Open launch.json" -> "C++(GDB/LLDB)"，即可打开调试的配置文件 launch.json。在 VSCode 中，用户按 F5 即可进入调试模式，上述 launch.json 文件即设置在调试时的基本内容和要求。

```

### indicator-sysmonitor

一款可以监视 CPU 占用率、 CPU 温度、内存占用率、网速等系统信息的小软件，在桌面最上方进行显示。Top 的图形化命令

```bash
# sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor  
sudo apt-get update
sudo apt-get install indicator-sysmonitor
```

### Marp

用 Markdown 语法来制作 PPT，高效快速简洁实用，尤其是支持 LaTeX 语法，非常方便编辑大量的数学公式，值得推荐，官网有 deb 文件，下载后直接安装即可。

### Tim

> Tim 安装   去官网 下载linux QQ  但qq上没有我的设备 
> https://im.qq.com/linuxqq/download.html 
> https://github.com/wszqkzqk/deepin-wine-ubuntu/releases   #wine的一个版本
> https://www.lulinux.com/archives/1319  #deepin-wine Tim安装教程
> **Winehq**:https://wiki.winehq.org/Ubuntu_zhcn   学习如何使用  回去学习下winehq使用教程[https://wiki.winehq.org/Wine_User%27s_Guide](https://wiki.winehq.org/Wine_User's_Guide)
> Usage: wine PROGRAM [ARGUMENTS...]   Run the specified program
>        wine --help                   Display this help and exit
>        wine --version                Output version information and exit
> 运行方式1:cd '.wine/drive_c/Games/Tron'
> 		 wine tron.exe 
> 运行方式2:wine start 'C:\Games\Tron\tron.exe'
> 		wine start "C:\\Games\\Tron\\tron.exe"
> 		wine start /unix "$HOME/installers/TronSetup.exe"
> 		wine quake.exe -map e1m1   #带参数
> 		wine start whatever.msi
> 		 wine control
> 		 wine uninstaller

### mega网盘安装

> https://mega.nz/sync   去官网安装 需要联网

### WPS 去官网下载

```bash
#http://www.wps.cn/product/wpslinux  
sudo dpkg -i wps-office_10.1.0.6757_amd64.deb
```

### [IDEA下载](  https://www.jetbrains.com/idea/download/#section=linux)

```shell
sudo ln -s /opt/idea-IU-212.4746.92/bin/idea.sh /usr/local/bin/idea  #建立软连接
idea &  #后台启动
```

### Teamview   deb 安装

### proxyee-down命令行安装  百度云下载神器

### docky 桌面工具

```bash
sudo apt-get install  docky   
sudo apt-get install gnome-tweak-tool 
sudo apt-get install gnome-shell-extensions 
sudo apt-get install gnome-shell-extension-dashtodock
sudo apt-get install gnome-shell-extension-autohidetopbar
#也可以在Ubuntu软件中直接搜索hide top bar
sudo apt-get remove gnome-shell-extension-autohidetopbar #卸载
#快捷键设置
gnome-screenshot -ac  # 也具有qq截图到快捷键功能
#在打开——系统设置——>键盘——快捷键——自定义快捷键，然后输入名字和上边工具的命令
```

### Opencv

```bash
#python 包
pip uninstall opencv-python
pip install opencv-contrib-python 
#opencv4. 源码编译安装， 也可以直接编译Android 依赖库
#https://www.pluvet.com/archives/223.html 安装教程
sudo add-apt-repository “deb http://security.ubuntu.com/ubuntu xenial-security main”
sudo apt update
sudo apt install libjasper1 libjasper-dev  
sudo apt-fast install build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
cmake ..
make -j4
sudo make install
```

### Python 命令转换

pip 切换镜像  最终写入文件 /home/ldd/.config/pip/pip.conf

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

方式一：系统默认一个版本，在另装一个版本，通过软连接

```bash
# 以后使用anaconda
#查看当前默认Python版本
python --version 
#查看Python所在
which is python
which is python3
#Python下载的库可以查看这里。/usr/local/lib/
#显示Python代替版本信息
update-alternatives --list python
#设置 /usr/bin/python3.5 设置的优先级为2 优先级越高越大
update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
update-alternatives --install /usr/bin/python python /usr/bin/python3.5 2
#再次显示Python代替版本信息
update-alternatives --remove python /usr/bin/python2.7
#切换版本
sudo update-alternatives --config python
sudo apt-get install python3-pip # #安装Python3对应的pip3
sudo pip3 install --upgrade pip   #推荐在管理员模式下更新
sudo apt-get install python-pip  #安装Python2对应的pip
#Pip  安装的库会放在这个目录下面：python2.7/site-packages；
#pip3 新安装的库会放在这个目录下面：python3.6/site-packages；
#参考https://www.cnblogs.com/carle-09/p/9907274.html
#errorPermission denied: '/usr/local/lib/python3.6/dist-packages/cycler.py' Consider using the `--user` option or check the permissions.
pip3 install --user matplotlib  
#The 'pip==9.0.3' distribution was not found and is required by the application
sudo easy_install pip==9.0.3  #解决
```

方式二：安装anaconda，然后建立基于不同python版本的conda环境

方式三：建立虚拟机virtualenv，然后建立基于不同python版本的虚拟环境

### MySQL 安装  

```shell
sudo apt-get install mysql-server
sudo mysql_secure_installation  #设置密码 liudongdong
sudo mysql   #可以直接登录
sudo systemctl start mysql
```

### .[Net core 安装](https://dotnet.microsoft.com/download/linux-package-manager/ubuntu18-04/sdk-current) 

### [mssql-server安装](https://docs.microsoft.com/zh-cn/sql/linux/quickstart-install-connect-ubuntu?view=sql-server-ver15#connect-locally)

```shell
sudo apt-fast install libodbc1 unixodbc msodbcsql mssql-tools unixodbc-dev
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/linux-operation/  

