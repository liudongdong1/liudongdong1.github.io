# 鸟哥私房菜-基础学习篇


### 1. 主机规划与磁盘分区

#### .1. 各设备在linux中的文件名

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408100703112.png)

#### .2. MSDOS（MBR）

- 早期的 Linux 系统为了相容于 Windows 的磁盘，因此使用的是`支持 Windows 的 MBR（Master Boot Record, 主要开机纪录区）` 的方式来处理`开机管理程序与分区表`！而开机管理程序纪录区与分区表则通放在磁盘的第一个扇区， 这个`扇区通常是 512Bytes 的大小` （旧的磁盘 扇区都是 512Bytes ），所以说，第一个扇区 512Bytes 会有这两个数据： 
  - 主要开机记录区（Master Boot Record, MBR）：可以`安装开机管理程序`的地方，有446 Bytes 
  - 分区表（partition table）：记录`整颗硬盘分区的状态`，有64 Bytes 由于分区表所在区块仅有64 Bytes容量，因此最多仅能有四组记录区，每组记录区记录了该区段的启始与结束的柱面号码。(可以采用延申分区来划分多个磁盘)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408101141509.png)

> 延伸分区的目的是使用额 外的扇区来记录分区信息，延伸分区本身并不能被拿来格式化。 然后我们可以通过延伸分区所指向的那个区块继续作分区的记录。这五个由延伸分区继续切出来的分区，就被称为逻辑分区（logical partition）。
>
> `主要分区与延伸分区最多可以有四笔（硬盘的限制） 延伸分区最多只能有一个（操作系统的限制）`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408101535803.png)

#### .3. GPT 磁盘分区表（（partition table）

- 与 MBR 仅使用第一个 512Bytes 区块来纪录不同， GPT 使用了 34 个 LBA 区块来纪录分区信息！
- GPT 除了前面 34 个 LBA 之外，整个磁盘的最后 33 个 LBA 也拿来作为另一个备份！

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408104340830.png)

- **LBA0(MBR相容区块)：** 与 MBR 模式相似的，这个相容区块也分为两个部份，一个就是跟之前 446 Bytes 相似的区块，储存了第一阶段的开机管理程序！ 而在原本的分区表的纪录区内，这个相容模式仅放入一个特殊标志的分区，用来表示此磁盘为 GPT 格式之意。
- **LBA1 （GPT 表头纪录）**： 纪录了`分区表本身的位置与大小`，`同时纪录了备份用的 GPT 分区` （就是前面谈到的在最后 34 个 LBA 区块）` 放置的位 置`， 同时放置了`分区表的检验机制码` （CRC32），操作系统可以根据这个检验码来判断 GPT 是否正确
- **LBA2-33 （实际纪录分区信息处）:**  LBA2 区块开始，每个 LBA 都可以纪录 4 笔分区纪录，所以在默认的情况下，总共可以有 4*32 = 128 笔分区纪录.GPT 在每笔纪录中分别 提供了 64bits 来记载开始/结束的扇区号码

#### .4. BIOS 搭配 MBR/GPT 的开机流程

- CMOS是记录各项硬件参数且嵌入在主板上面的储存器，BIOS则是 一个写入到主板上的一个固件

> 1. BIOS：开机主动执行的固件，会认识第一个可开机的设备； 
>
> 2. MBR：第一个可开机设备的第一个扇区内的主要开机记录区块，内含开机管理程序； 
> 3.  开机管理程序（boot loader）：一支可读取核心文件来执行的软件； Boot loader则是操作系统安装在MBR上面的一套软件了, `提供菜单`：使用者可以选择不同的开机项目，这也是多重开机的重要功能！ `载入核心文件`：直接指向可开机的程序区段来开始操作系统； `转交其他loader`：将开机管理功能转交给其他loader负责。
> 4. 核心文件：开始操作系统的功能

#### .5. UEFI BIOS 搭配 GPT 开机的流程

- UEFI主要是想要取代 BIOS 这个固件界面，因此我们也称 UEFI为 UEFIBIOS 就是了。UEFI使用 C 程序语言，比起使用组合语言的传 统 BIOS 要更容易开发！也因为使用 C 语言来撰写，因此如果开发者够厉害，甚至可以在 UEFI开机阶段就让该系统了解 TCP/IP 而直接上 网！

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408110430346.png)

### 2. Linux操作

#### .1. 关机

- 数据同步写入磁盘： sync

```bash
[dmtsai@study ~]$ su - # 这个指令在让你的身份变成 root ！下面请输入 root 的密码！
Password: # 就这里！请输入安装时你所设置的 root 密码！
Last login: Mon Jun 1 16:10:12 CST 2015 on pts/0
[root@study ~]# sync
```

- shutdown

```bash
[root@study ~]# /sbin/shutdown [-krhc] [时间] [警告讯息]
选项与参数：
-k ： 不要真的关机，只是发送警告讯息出去！
-r ： 在将系统的服务停掉之后就重新开机（常用）
-h ： 将系统的服务停掉后，立即关机。 （常用）
-c ： 取消已经在进行的 shutdown 指令内容。
时间 ： 指定系统关机的时间！时间的范例下面会说明。若没有这个项目，则默认 1 分钟后自动进行。
范例：
[root@study ~]# /sbin/shutdown -h 10 'I will shutdown after 10 mins'
Broadcast message from root@study.centos.vbird （Tue 2015-06-02 10:51:34 CST）:
I will shutdown after 10 mins The system is going down for power-off at Tue 2015-06-02 11:01:34 CST!
```

```bash
[root@study ~]# shutdown -h now
立刻关机，其中 now 相当于时间为 0 的状态
[root@study ~]# shutdown -h 20:25
系统在今天的 20:25 分会关机，若在21:25才下达此指令，则隔天才关机
[root@study ~]# shutdown -h +10
系统再过十分钟后自动关机
[root@study ~]# shutdown -r now
系统立刻重新开机
[root@study ~]# shutdown -r +30 'The system will reboot'
再过三十分钟系统会重新开机，并显示后面的讯息给所有在线上的使用者
[root@study ~]# shutdown -k now 'This system will reboot'
仅发出警告信件的参数！系统并不会关机啦！吓唬人！
```

#### .2. 文件目录配置依据-FHS

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408112433082.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408134247976.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408134307088.png)![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408134745707.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408134807939.png)

#### .3. /usr & /val

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408135042584.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408135209162.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408135253015.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408195806304.png)

### 3. 文件

#### .1. 非纯文本文件查看

```bash
[root@study ~]# od [-t TYPE] 文件
选项或参数：
-t ：后面可以接各种“类型 （TYPE）”的输出，例如：
a ：利用默认的字符来输出；
c ：使用 ASCII 字符来输出
d[size] ：利用十进制（decimal）来输出数据，每个整数占用 size Bytes ；
f[size] ：利用浮点数值（floating）来输出数据，每个数占用 size Bytes ；
o[size] ：利用八进位（octal）来输出数据，每个整数占用 size Bytes ；
x[size] ：利用十六进制（hexadecimal）来输出数据，每个整数占用 size Bytes ；
范例一：请将/usr/bin/passwd的内容使用ASCII方式来展现！
[root@study ~]# od -t c /usr/bin/passwd
0000000 177 E L F 002 001 001 \0 \0 \0 \0 \0 \0 \0 \0 \0
0000020 003 \0 > \0 001 \0 \0 \0 364 3 \0 \0 \0 \0 \0 \0
0000040 @ \0 \0 \0 \0 \0 \0 \0 x e \0 \0 \0 \0 \0 \0
0000060 \0 \0 \0 \0 @ \0 8 \0 \t \0 @ \0 035 \0 034 \0
0000100 006 \0 \0 \0 005 \0 \0 \0 @ \0 \0 \0 \0 \0 \0 \0
.....（后面省略）....
# 最左边第一栏是以 8 进位来表示Bytes数。以上面范例来说，第二栏0000020代表开头是
# 第 16 个 byes （2x8） 的内容之意。
范例二：请将/etc/issue这个文件的内容以8进位列出储存值与ASCII的对照表
[root@study ~]# od -t oCc /etc/issue
0000000 134 123 012 113 145 162 156 145 154 040 134 162 040 157 156 040
\ S \n K e r n e l \ r o n
0000020 141 156 040 134 155 012 012
a n \ m \n \n
0000027
# 如上所示，可以发现每个字符可以对应到的数值为何！要注意的是，该数值是 8 进位喔！
# 例如 S 对应的记录数值为 123 ，转成十进制：1x8^2+2x8+3=83。
```

#### .2. 文件相关命令

```bash
$file 查看文件类型
$which [-a] command  # 查看文件目录 根据“PATH”这个环境变量所规范的路径，去搜寻“可执行文件”的文件名～ 所以，重点是找出“可执行文件”而已！且 which 后面接的是“完整文件名”喔！若加上 -a 选项，则可以列出所有的可以找到的同名可执行文件，而非仅显示第一个而已！
$locate pwd
$find /etc -name '*httpd*'
```

- **由文件找出正在使用该文件的程序fuser**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408200040194.png)

- **列出被程序所打开的文件文件名 lsof**

#### .3. 文件系统特性

- `superblock`：记录此 filesystem 的整体信息，包括`inode/block的总量、使用量、剩余量， 以及文件系统的格式与相关信息等`；
- ` inode`：记录文件的属性，一个文件占用一个inode，同时记录此文件的数据所在的 block 号码；
- ` block`：实际`记录文件的内容，若文件太大时，会占用多个 block` 。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408160045309.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408160147950.png)

- 原则上，block 的大小与数量在格式化完就不能够再改变了（除非重新格式化）；
-  每个 block 内最多只能够放置一个文件的数据；
-  承上，如果文件大于 block 的大小，则一个文件会占用多个 block 数量；
-  承上，若文件小于 block ，则该 block 的剩余容量就不能够再被使用了（磁盘空间会浪费）。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408160522848.png)

- 将 inode 记录 block 号码的区域定义为12个直接，一个间接, 一个双间接与一个三间接记 录区。

#### .4. du & df

```bash
范例二：将容量结果以易读的容量格式显示出来
[root@study ~]# df -h Filesystem Size Used Avail Use% Mounted on
/dev/mapper/centos-root 10G 3.3G 6.8G 33% /
devtmpfs 613M 0 613M 0% /dev
tmpfs 623M 80K 623M 1% /dev/shm
tmpfs 623M 25M 599M 4% /run
tmpfs 623M 0 623M 0% /sys/fs/cgroup
/dev/mapper/centos-home 5.0G 67M 5.0G 2% /home
/dev/vda2 1014M 131M 884M 13% /boot
# 不同于范例一，这里会以 G/M 等容量格式显示出来，比较容易看啦
```

- 由于 df 主要读取的数据几乎都是针对一整个文件系统，因此读取的范围主要是在 Superblock 内的信息， 所以这个指令显示结果的速度 非常的快速！
- du 这个指令其实会直接到文件系统内去搜寻所有的文件数据.

#### .5. 磁盘分区

```shell
[root@study ~]# lsblk [-dfimpt] [device]
选项与参数：
-d ：仅列出磁盘本身，并不会列出该磁盘的分区数据
-f ：同时列出该磁盘内的文件系统名称
-i ：使用 ASCII 的线段输出，不要使用复杂的编码 （再某些环境下很有用）
-m ：同时输出该设备在 /dev 下面的权限数据 （rwx 的数据）
-p ：列出该设备的完整文件名！而不是仅列出最后的名字而已。
-t ：列出该磁盘设备的详细数据，包括磁盘伫列机制、预读写的数据量大小等
范例一：列出本系统下的所有磁盘与磁盘内的分区信息
[root@study ~]# lsblk
NAME MAJ:MIN RM SIZE RO TYPE MOUNTPOINT
sr0 11:0 1 1024M 0 rom
vda 252:0 0 40G 0 disk # 一整颗磁盘
|-vda1 252:1 0 2M 0 part
|-vda2 252:2 0 1G 0 part /boot
`-vda3 252:3 0 30G 0 part
|-centos-root 253:0 0 10G 0 lvm / # 在 vda3 内的其他文件系统
|-centos-swap 253:1 0 1G 0 lvm [SWAP]
`-centos-home 253:2 0 5G 0 lvm /home
```

- `UUID 是全域单一识别码 （universally unique identifier）`，Linux 会将系统内所有的设备都给予一个独一无二的识别码， 这个识别 码就可以拿来作为挂载或者是使用这个设备/文件系统之用了
- “MBR 分区表请使用 fdisk 分区， GPT 分区表请使用 gdisk 分区！”

##### 1. gdisk

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408162713811.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408162913242.png)

- **partprobe 更新 Linux 核心的分区表信息**

> 不要去处理一个正在使用中的分区！例如，我们的系统现在已经使用了 /dev/vda2 ，那如果你要删除 /dev/vda2 的话， 必须要先 将 /dev/vda2 卸载，否则直接删除该分区的话，虽然磁盘还是慧写入正确的分区信息，但是核心会无法更新分区表的信息的！

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408163124586.png)

##### 2. 格式化 & 挂载

- EXT4 文件系统 mkfs.ext4
- XFS 文件系统 mkfs.xfs

```bash
blkid /dev/vda4 #找出 /dev/vda4 的 UUID 后，用该 UUID 来挂载文件系统到 /data/xfs 内
mount UUID="e0a6af55-26e7-4cb7-a515-826a8bd29e90" /data/xfs
```

- 单一文件系统不应该被重复挂载在不同的挂载点（目录）中；
-  单一目录不应该重复挂载多个文件系统； 
- 要作为挂载点的目录，理论上应该都是空目录才是

#### 6. 使用实体分区创建 swap

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408164109135.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408164217554.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408164237875.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408164258020.png)

#### 7. LVM 相关介绍

- LVM 的作法是`将几个实体的 partitions （或 disk） 通过软件组合成为一块看起来是独立的大磁盘 （VG） ，然后将这块大 磁盘再经过分区成为可使用分区 （LV）`， 最终就能够挂载使用了。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408194314913.png)

### 4. 软件相关

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408171534087.png)

- 通常不同的 distribution 所释出的 RPM 文件，并不能用在其他的 distributions 上。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408171734001.png)

### 5. 服务systemctl

```bash
[root@study ~]# systemctl [command] [unit]
command 主要有：
start ：立刻启动后面接的 unit
stop ：立刻关闭后面接的 unit
restart ：立刻关闭后启动后面接的 unit，亦即执行 stop 再 start 的意思
reload ：不关闭后面接的 unit 的情况下，重新载入配置文件，让设置生效
enable ：设置下次开机时，后面接的 unit 会被启动
disable ：设置下次开机时，后面接的 unit 不会被启动
status ：目前后面接的这个 unit 的状态，会列出有没有正在执行、开机默认执行否、登录等信息等！
is-active ：目前有没有正在运行中
is-enable ：开机时有没有默认要启用这个 unit
```

### 6. 内核

#### .1. 内核稳定版本链接

-  核心官网：http://www.kernel.org/ 
- 交大资科：ftp://linux.cis.nctu.edu.tw/kernel/linux/kernel/ 
- 国高中心：ftp://ftp.twaren.net/pub/Unix/Kernel/linux/kernel/

#### .2. 核心代码目录

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408164733509.png)

#### .3. 单一模块编译

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220408165238691.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E9%B8%9F%E5%93%A5%E7%A7%81%E6%88%BF%E8%8F%9C-%E5%9F%BA%E7%A1%80%E5%AD%A6%E4%B9%A0%E7%AF%87/  

