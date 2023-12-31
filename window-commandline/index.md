# Window CommandLine


## 1. PowerShell

```powershell
#  查看powershell 版本
get-host
$host.version
#  新建目录
#当前目录新建文件
new-item FILENAME.xxx -type file
#当前目录新建文件夹
new-item DIRECTORYNAME -type directory
#在指定目录新建
new-item TARGETDIR FILENAME.xxx -type file
#  重命名
#把 C:/Scripts/Test.txt 重命名为 C:/Scripts/New_Name.txt:
Rename-Item c:/scripts/Test.txt new_name.txt
#  移动文件
Move-Item c:\scripts\test.zip c:\testX
#  删除目录/文件
remove-item file
#显示文本内容
get-content 1.txt
#罗列系统驱动器
get-psdriver
#下载文件
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://ts', './src/ts')"
#支持linux 文件  ls，dir，pwd，cat, more
# 中文输出乱码
打开控制面板 -> Change date,time,or number -> 打开 “Region” 对话框
选择 Administrative 选项卡，点击 change system locale
选择
```

## 2. Cmd

```shell
where cmd #类似Linux中where 命令
find /r 目录名 %变量名 in (匹配模式1,匹配模式2) do 命令
for /r 目录名 %i in (匹配模式1,匹配模式2) do @echo %i
for /r TestDir %i in (*) do @echo %i  #将TestDir目录及所有子目录中所有的文件列举出来
for /r TestDir %i in (*.txt) do @echo %i  #在TestDir目录所有子目录中找出所有的txt文件
for /r TestDir %i in (.txt,.jpg) do @echo %i #找出所有的txt及jpg文件
for /r TestDir %i in (test) do @echo %i  #找出所有文件名中包含test的文件
Tree   #罗列文件目录

%windir%\fonts    #打开电脑字体目录
services.msc   #查看电脑上的服务
shell:startup   #打开开机自启动服务
```

## 3. 哈希

```shell
certutil -hashfile 路径/file.exe MD5
certutil -hashfile 路径/file.exe SHA1
certutil -hashfile 路径/file.exe SHA256
ertutil -hashfile -?  #帮助说明
```

```
用法:
  CertUtil [选项] -hashfile InFile [HashAlgorithm]
  通过文件生成并显示加密哈希

选项:
  -Unicode          -- 以 Unicode 编写重定向输出
  -gmt              -- 将时间显示为 GMT
  -seconds          -- 用秒和毫秒显示时间
  -v                -- 详细操作
  -privatekey       -- 显示密码和私钥数据
  -pin PIN                  -- 智能卡 PIN
  -sid WELL_KNOWN_SID_TYPE  -- 数字 SID
            22 -- 本地系统
            23 -- 网络服务
            24 -- 本地服务

哈希算法: MD2 MD4 MD5 SHA1 SHA256 SHA384 SHA512
```

## 4. cmd 快捷命令

- Win + A：操作中心
- Win + E：资源管理器
- Win + S：搜索界面
- Win + 空格：切换输入法

### .1. 软件开机自启动

```
shell:startup
#或者进入这个目录
%programdata%\Microsoft\Windows\Start Menu\Programs\Startup
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210707170127650.png)



https://d.serctl.com/?uuid=9c83ed93-dd5c-478f-b148-25ed560e7d59 

https://blog.csdn.net/qq_43427482/article/details/112757029

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/window-commandline/  

