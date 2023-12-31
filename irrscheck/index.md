# VirusCheck


> 对于主机的入侵痕迹排查，主要从`网络连接、进程信息、后门账号、计划任务、登录日志、自启动项、文件等`方面进行排查。

### 1. 显示隐藏文件

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210126084818634.png)

### 2. 查看网络情况

- 主机存在对内网网段大量主机的某些端口（常见如22，445，3389，6379等端口）或者全端口发起网络连接尝试，这种情况一般是当前主机被攻击者当作跳板机对内网实施端口扫描或者口令暴力破解等攻击。
- 主机和外网IP已经建立连接（ESTABLISHED状态）或者尝试建立连接（SYN_SENT状态），可以先查询IP所属地，如果IP为国外IP或者归属各种云厂商，则需要重点关注。进一步可以通过威胁情报（https://x.threatbook.cn/等）查询IP是否已经被标注为恶意IP。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210126090124013.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210126090139140.png)

- 通过任务管理程序查看网络异常对应程序id， 找到对应程序文件，上传至virustotal（https://www.virustotal.com）进行检测。如上面截图中对内网扫描的进程ID是2144，在任务管理器中发现对应的文件是svchost.exe。

### 3. 敏感目录

- 各个盘符下的临时目录，如C:\TEMP、C:\Windows\Temp等。`按照修改日期排序筛选出比较临近时间有变更的文件`.
- 浏览器的下载目录
- 用户最近文件%UserProfile%\Recent.
- 回收站，如C盘下回收站C:$Recycle.Bin.对于脚本文件可直接查看内容判定是否为恶意，若是遇到exe可执行文件，可将对应文件上传至virustotal（https://www.virustotal.com）进行检测。
- 注册表： 检查注册表“HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options”下所有exe项中是否有debugger键，若有debugger键，将其键值对应的程序上传至VT检测。如下图，攻击者利用该映像劫持的攻击者方式，在sethc.exe项中新建debugger键值指向artifact.exe，攻击效果为当连续按5下shift键后，不会执行sethc.exe，而是转而执行劫持后的artifact.exe文件。于是在排查中发现有debugger键值，均可认为指定的文件为后门文件，待上传VT后确认其危害。

### 4. 自启动排查

- **Autoruns**

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/irrscheck/  

