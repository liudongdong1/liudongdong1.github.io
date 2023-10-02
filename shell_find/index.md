# shell_find


>查找系统中所有的大小超过 200M 文件，查看近 7 天系统中哪些文件被修改过，找出所有子目录中的可执行文件，这些任务需求 `find` 命令都可以轻松胜任。`find` 搜索文件时通过扫描磁盘来进行的，尽可能不要大范围的搜索文件，尤其是在 `/` 目录下搜索，会长时间消耗服务器的 cpu 资源。

#### 1. 命令格式

```shell
find path -option [-exec ...]
```

#### 2. 案例

##### .1. 按文件名查找

```shell
find . -name "*.go"
find /etc -name "[A-Z]*.txt" -print       #查找大写字母开头的 txt 文件
find . -name "out*" -prune -o -name "*.txt" -print  #在当前目录下查找不是 out 开头的 txt 文件
find . -path "./git" -prune -o -name "*.txt" -print #在当前目录除 git 子目录外查找 txt 文件
```

##### .2. **按文件类型查找**

```shell
find . -type l -print     #在当前目录下，查找软连接文件
find . -type f -name "*.log"  #在当前目录下，查找 log 结尾的普通文件，f 表示普通文件类型
```

##### .3. **按文件大小查找**

```shell
find . -size -64k -print  #查找小于 64k 的文件
find . -size +200M -type f -print  #查找大小超过 200M 的文件
```

##### .4. **按时间查找**

```shell
find . -mtime -2 -type f -print   #查找 2 天内被修改过的文件
find . -mtime +2 -type f -print   #查找 2 天前被更改过的文件，-mtime 表示内容修改时间
find . -atime -1 -type f -print   #查找一天内被访问的文件，-atime 表示访问时间
find . -ctime -1 -type f -print   #查找一天内状态被改变的文件，-ctime 表示元数据被变化时间
```

##### .5. **根据权限查找**

```shell
find . -type f -perm 644   #查找当前目录权限为 644 的文件
find /etc -type f -perm /222   #查找 etc 目录下至少有一个用户有写权限的文件
```

##### .6. 根据inode号

```shell
$ ls  -i
138957 a.txt  138959 T.txt  132395 ڹ��.txt

$ find . -inum 132395 -exec rm {} \;
```

#### Resource

- https://www.cnblogs.com/liwei0526vip/p/14354590.html

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shell_find/  

