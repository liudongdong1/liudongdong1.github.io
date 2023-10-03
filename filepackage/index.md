# FilePackage


## 文件打包和压缩

Linux 上的压缩包文件格式，除了 Windows 最常见的＊.zip、＊.rar、.7z 后缀的压缩文件，还有 .gz、.xz、.bz2、.tar、.tar.gz、.tar.xz、tar.bz2

| 文件后缀名 | 说明                           |
| :--------- | :----------------------------- |
| *.zip      | zip 程序打包压缩的文件         |
| *.rar      | rar 程序压缩的文件             |
| *.7z       | 7zip 程序压缩的文件            |
| *.tar      | tar 程序打包，未压缩的文件     |
| *.gz       | gzip 程序 (GNU zip) 压缩的文件 |
| *.xz       | xz 程序压缩的文件              |
| *.bz2      | tar 打包，gzip 程序压缩的文件  |
| *.tar.gz   | tar打包，gzip程序压缩的文件    |
| *.tar.xz   | tar打包，xz程序压缩的文件      |
| *.tar.bz2  | tar打包，bzip2程序压缩的文件   |
| *.tar.7z   | tar打包，7z程序压缩的文件      |

### 1 zip 压缩打包程序

- 使用 zip 打包文件

```shell
# 将 test 目录打包成一个文件，-r 表示递归打包包含子目录的全部内容，-q 表示安静模式，-o 表示输出文件，其后紧跟打包输出文件名
zip -r -q -o test.zip  /home/test
# 使用 du 命令查看打包后文件的大小
du -h test.zip
# 使用 file 命令查看文件大小和类型
file test.zip
```

- 设置压缩级别为9和1（9最大,1最小），重新打包

```shell
# 1表示最快压缩但体积大，9表示体积最小但耗时最久，-x 排除上一次我们创建的zip文件，路径必需为绝对路径
zip -r -9 -q -o test_9.zip /home/test -x ~/*.zip
zip -r -1 -q -o test_1.zip /home/test -x ~/*.zip
# 再用 du 命令分别查看默认压缩级别、最低、最高压缩级别及未压缩的文件的大小，-h 表示可读，-d 表示所查看文件的深度
du -h -d 0 *.zip ~ | sort
```

- 创建加密 zip 包

```shell
# 使用 -e 参数可以创建加密压缩包
zip -r -q -o test.zip  /home/test
```

注意: 关于 zip 命令，因为 Windows 系统与 Linux/Unix 在文本文件格式上的一些兼容问题，比如换行符（为不可见字符），在 Windows 为 CR+LF（Carriage-Return+Line-Feed：回车加换行），而在 Linux/Unix 上为 LF（换行），所以如果在不加处理的情况下，在 Linux 上编辑的文本，在 Windows 系统上打开可能看起来是没有换行的。如果你想让你在 Linux 创建的 zip 压缩文件在 Windows 上解压后没有任何问题，那么你还需要对命令做一些修改
shell 中的变量有不同类型，可参与运算，有作用域限定

```shell
# 使用 -l 参数将 LF 转换为 CR+LF
zip -r -l -o test.zip /home/test
```

### 2 使用 unzip 命令解压缩 zip 文件

- 使用 zip 打包文件

```shell
# 将 test.zip 解压到当前目录
unzip test.zip
# 使用安静模式，将文件解压到指定目录
unzip -q test.zip -d ziptest
# 不想解压，只想查看压缩包的内容可以使用 -l 参数
unzip -l test.zip
#  Linux 上面默认使用的是 UTF-8 编码,防止解压后出现中文乱码，要用参数 -O
unzip -O GBK 中文压缩文件.zip
```

### 3 rar打包压缩命令

在 Linux 上可以使用 rar 和 unrar 工具分别创建和解压 rar 压缩包。

- 安装rar和unrar工具

```shell
sudo apt-get update
sudo apt-get install rar unrar
```

- 从指定文件或目录创建压缩包或添加文件到压缩包

```shell
rm *.zip
# 使用a参数添加一个目录～到一个归档文件中，如果该文件不存在就会自动创建
rar a test.rar .
```

注意：rar 的命令参数没有-，如果加上会报错。

- 从指定压缩包文件中删除某个文件

```shell
rar d test.rar .bashrc
```

- 查看不解压文件

```shell
rar l test.rar
```

- 使用 unrar 解压 rar 文件

```shell
# 全路径解压
unrar x test.rar
# 去路径解压
mkdir tmp
unrar e test.rar tmp/
```

### 4 tar 打包工具

在 Linux 上面更常用的是 tar 工具，tar 原本只是一个打包工具，只是同时还是实现了对 7z，gzip，xz，bzip2 等工具的支持，这些压缩工具本身只能实现对文件或目录（单独压缩目录中的文件）的压缩，没有实现对文件的打包压缩，所以我们也无需再单独去学习其他几个工具，tar 的解压和压缩都是同一个命令，只需参数不同，使用比较方便。

- 创建一个 tar 包

```shell
# -c 表示创建一个 tar 包文件，-f 用于指定创建的文件名，注意文件名必须紧跟在 -f 参数之后
# 会自动去掉表示绝对路径的 /，你也可以使用 -P 保留绝对路径符
tar -cf test.tar ~
```

- 解包一个文件 (-x参数) 到指定路径的已存在目录 (-C参数)

```shell
mkdir tardir
tar -xf test.tar -C tardir
```

- 只查看不解包文件-t参数

```shell
tar -tf test.tar
```

- 保留文件属性和跟随链接（符号链接或软链接），有时候我们使用tar备份文件当你在其他主机还原时希望保留文件的属性(-p参数)和备份链接指向的源文件而不是链接本身(-h参数)

```shell
tar -cphf etc.tar /etc
```

- 以使用 gzip 工具创建 *.tar.gz 文件为例来说明，只需在创建 tar 文件的基础上添加 -z 参数，使用 gzip 来压缩文件

```shell
tar -czf etc.tar.gz ~
```

- 解压 *.tar.gz 文件

```shell
tar -xzf etc.tar.gz
```

### 5 文件大小查看

#### .1. stat

> stat命令一般用于查看文件的状态信息。stat命令的输出信息比ls命令的输出信息要更详细。

`stat ~/iso/CentOS-6.10-x86_64-minimal.iso `

- 文件：/home/oucanrong/iso/CentOS-6.10-x86_64-minimal.iso
- 大小：425721856 块：831504 IO 块：4096 普通文件
- 设备：802h/2050d Inode：4471899 硬链接：1
- 权限：(0664/-rw-rw-r--) Uid：( 1000/oucanrong) Gid：( 1000/oucanrong)
- 最近访问：2018-12-23 21:51:07.778850541 +0800
- 最近更改：2018-12-21 15:24:14.446276453 +0800
- 最近改动：2018-12-21 16:19:33.764230839 +0800
- 创建时间：-

#### 2. wc

> wc命令一般用于统计文件的信息，比如文本的行数，文件所占的字节数。

`wc -c ~/iso/CentOS-6.10-x86_64-minimal.iso` 

- 425721856 /home/oucanrong/iso/CentOS-6.10-x86_64-minimal.iso

#### 3. du

> du命令一般用于统计文件和目录所占用的空间大小。

`du -h ~/iso/CentOS-6.10-x86_64-minimal.iso `

- 407M /home/oucanrong/iso/CentOS-6.10-x86_64-minimal.iso

#### 4. ls

> ls 命令一般用于查看文件和目录的信息，包括文件和目录权限、拥有者、所对应的组、文件大小、修改时间、文件对应的路径等等信息。-lh

 `ls -lh ~/iso/ubuntu-18.04.1-live-server-amd64.iso `

- -rw-rw-r-- 1 oucanrong oucanrong 812M 12月 21 15:23 /home/oucanrong/iso/ubuntu-18.04.1-live-server-amd64.iso

- 可以看出ubuntu-18.04.1-live-server-amd64.iso的大小为812M

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/filepackage/  

