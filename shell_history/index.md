# shell_history


> Linux 命令的历史记录，会持久化存储，默认位置是当前用户家目录的 `.bash_history` 文件。
>
> 当 Linux 系统启动一个 Shell 时，Shell 会从 `.bash_history` 文件中，读取历史记录，存储在相应内存的缓冲区中。

#### 1. 基本用法

```shell
history (选项) (参数)
```

选项

- -c：清空当前历史命令； 
- -a：将历史命令缓冲区中命令写入历史命令文件中；
-  -r：将历史命令文件中的命令读入当前历史命令缓冲区； 
- -w：将当前历史命令缓冲区命令写入历史命令文件中。

参数

- n：打印最近的 n 条历史命令。

```shell
history 10  #显示最后的 10 条历史记录
history -w  #主动保存缓冲区的历史记录
history -c  #将缓冲区内容直接删除
```

#### 2. 重复执行命令

```shell
!1024  #重复执行第 1024 历史命令
!!     #重复执行上一条命令
!-6    #重复执行倒数第 6 条历史命令
```

#### 3. **搜索历史命令**

```shell
!curl:p   #打印出了搜索到的命令，如果要执行，请按 Up 键，然后回车即可
```

#### 4. **显示时间戳**

```shell
export HISTTIMEFORMAT='%F %T '
history 3
#用于审计操作
export HISTTIMEFORMAT="%F %T `who -u am i 2>/dev/null| awk '{print $NF}'|sed \-e 's/[()]//g'` `whoami` "
```

#### 5. **控制历史记录总数**

```shell
echo $HISTSIZE
export HISTSIZE=10000

#永久生效配置
$ echo "export HISTSIZE=10000" >> ~/.bash_profile
$ echo "export HISTFILESIZE=200000" >> ~/.bash_profile
$ source ~/.bash_profile
```

#### Resource

- https://www.cnblogs.com/liwei0526vip/p/14757774.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/shell_history/  

