# rm&cp&mv原理


> 在 linux 系统中，磁盘通常被格式化为 ext3 或 ext4 格式，这两种文件系统对文件的存储和访问是通过一种被称为 inode 即 i 节点的机制来实现的。
>
> 当我们读写文件时，通常是以流的形式，即认为文件的内容是连续的。但是在磁盘上，一个文件的内容通常是由多个固定大小的数据块即 block 构成的，并且这些数据块通常是不连续的。这时就需要一个额外的数据结构来保存各数据块的位置、数据块之间的顺序关系、文件大小、文件访问权限、文件的拥有者及修改时间等信息，即文件的元信息，而维护这些元信息的数据结构就被称为 i 结点。可以说，一个 i 节点中包含了进程访问文件时所需要的所有信息。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403145939452.png)

#### 1. 硬链接

> 硬链接文件和原文件就是同一个文件，只不过有两个名字，类似于 C++ 中的引用。`当一个文件有多个链接时，删除其中一个文件名并不会删除文件本身，而只是减少文件的链接数`。当链接数为 0 时，文件内容才会真正被删除。

#### 2. 符号链接

> 符号链接并不增加目标文件 i 节点的链接数。符号链接本身也是一个文件，其中存储了目标文件的完整路径，类似于 windows 系统中的快捷方式。

> 如果文件链接数为 1，但是仍然有进程打开这一文件，那么 unlink 后，虽然在原目录中已经没有了被删除文件的名字，但是实际上系统还是保留了这一文件，直到打开这一文件的所有进程全部关闭此文件后，系统才会真正删除磁盘上的文件内容。由此可见，用 unlink 直接删除打开的文件是安全的。删除已经打开的文件，对使用此文件的进程，不会有任何影响，也不会导致进程崩溃（注意这里讨论的是删除已被打开的文件，通常是数据文件，并未讨论删除正在运行的可执行文件）。

#### 3. unlink

- 对于符号链接，unlink 删除的是符号链接本身，而不是其指向的文件。
- 删除文件名，并不一定删除磁盘上文件的内容。只有在文件的链接数为 1，即当前文件名是文件的最后一个链接并且有没有进程打开此文件的时候，unlink () 才会真正删除文件内容。

#### 4. rm 命令

```shell
# strace rm data.txt 2>&1 | grep 'data.txt' 
execve("/bin/rm", ["rm", "data.txt"], [/* 13 vars */]) = 0
lstat("data.txt", {st_mode=S_IFREG|0644, st_size=10, ...}) = 0
stat("data.txt", {st_mode=S_IFREG|0644, st_size=10, ...}) = 0
access("data.txt", W_OK)                = 0
unlink("data.txt")                      = 0
```

```shell
# strace unlink data.txt 2>&1 | grep 'data.txt'
execve("/bin/unlink", ["unlink", "data.txt"], [/* 13 vars */]) = 0
unlink("data.txt")
```

> 在 linux 中，`rm 命令比 unlink 命令多了一些权限的检查`，之后也是调用了 unlink () 系统调用。在文件允许删除的情况下，rm 命令和 unlink 命令其实是没有区别的。

#### 5. rename 命令

```shell
# strace rename data.txt  dest_file data.txt 2>&1 | grep  'data.txt|dest_file'
execve("/usr/bin/rename", ["rename", "data.txt", "dest_file", "data.txt"], [/* 13 vars */]) = 0
rename("data.txt", "dest_file")         = 0
```

> 在`目标文件 dest_file 已经存在的情况下`，执行 rename 后，dest_file 的 i 节点号发生了变化，因而 rename () 系统调用的作用类似于上述第二种情形：`即删除文件后再新建一个同名文件`。

#### 6. mv 命令

- 当目标文件不存在时

```shell
# strace mv data.txt  dest_file 2>&1 | egrep  'data.txt|dest_file'
execve("/bin/mv", ["mv", "data.txt", "dest_file"], [/* 13 vars */]) = 0
stat("dest_file", 0x7ffe1b4aab50)       = -1 ENOENT (No such file or directory)
lstat("data.txt", {st_mode=S_IFREG|0644, st_size=726, ...}) = 0
lstat("dest_file", 0x7ffe1b4aa900)      = -1 ENOENT (No such file or directory)
rename("data.txt", "dest_file")         = 0
```

- 当目标文件存在时

```shell
# strace mv src_data data.txt 2>&1 | egrep 'src_data|data.txt'
execve("/bin/mv", ["mv", "src_data", "data.txt"], [/* 13 vars */]) = 0
stat("data.txt", {st_mode=S_IFREG|0644, st_size=726, ...}) = 0
lstat("src_data", {st_mode=S_IFREG|0644, st_size=726, ...}) = 0
lstat("data.txt", {st_mode=S_IFREG|0644, st_size=726, ...}) = 0
stat("data.txt", {st_mode=S_IFREG|0644, st_size=726, ...}) = 0
access("data.txt", W_OK)                = 0
rename("src_data", "data.txt")          = 0
```

> mv 的主要功能就是`检查初始文件和目标文件是否存在及是否有访问权限`，之后执行 rename 系统调用，因而，当目标文件存在时，mv 的行为由 rename () 系统调用决定，即类似于删除文件后再重建一个同名文件。

#### 7. **cp 命令**

- 当文件不存在时

```shell
# strace cp data.txt dest_data 2>&1 | egrep 'data.txt|dest_data'
execve("/bin/cp", ["cp", "data.txt", "dest_data"], [/* 13 vars */]) = 0
stat("dest_data", 0x7fff135827f0)       = -1 ENOENT (No such file or directory)
stat("data.txt", {st_mode=S_IFREG|0644, st_size=726, ...}) = 0
stat("dest_data", 0x7fff13582640)       = -1 ENOENT (No such file or directory)
open("data.txt", O_RDONLY)              = 3
open("dest_data", O_WRONLY|O_CREAT, 0100644) = 4
```

- 当目标文件存在

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403151202436.png)

> 如果`目标文件存在`，在执行 cp 命令之后，文件的` inode 号并没有改变`，并且可以看出，`cp 使用了 open 及 O_TRUNC 参数打开了目标文件`。因而当目标文件已经存在时，`cp 命令实际是清空了目标文件内容，之后把新的内容写入目标文件。`

#### 8.如果一个文件正在被使用，这时对这个文件执行删除或更改其内容的操作，会发生什么？

##### .1. 文件正在被打开读写

> 当进程打开一个文件后，如果我们在磁盘上删除这个文件，虽然表面上看在目录中已经成功删除了这个文件名，但是`实际上系统依然保留了文件内容，直至所有进程都关闭了这一文件`。

> 进程打开一个文件后，如果我们删除被打开的文件，之后再重建这一文件，操作系统依然把之前打开的文件描述符 3 指向原有旧文件的 i 节点，即使`再次建立同名文件，系统也认为文件描述符 3 指向的文件被删除了`。
>
> 新建的同名文件与原文件及原进程无任何关系，新建的同名文件未被原进程使用，并且新建的文件使用了新的 i 节点，与原文件即使同名也没有产生任何联系。
>
> 删除原文件，再新建同名文件后，程序的输出依然是删除前 data.txt 文件的内容 “hello world”，与新建文件的内容无关。

##### .2. 可执行文件正在运行

```c
#include <stdio.h>
#include <unistd.h>

int main(void) {
  int i = 0;
  while (1) {
    printf("ix:%d, pid:%d\n", i, getpid());
    ++i;
    sleep(1);
  }
  return 0;
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403152758774.png)

> `当一个进程开始运行时，操作系统会在 /proc/*pid* 目录下建立一个名为 exe 的符号链接`，`这一链接指向磁盘上的可执行文件`。通过这一符号链接我们就可以观察到当可执行文件被删除时，程序的表现。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403153001119.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403153103281.png)

- 如果一个可执行文件正在运行，当我们把它删除后，删除的也只是文件名，操作系统依然保留了可执行文件的内容，这一点从 ls -ilL /proc/20295/exe 的结果中可以看出，其 i 节点号与删除前 pr_pid 文件的 i 节点号完全相同，执行 md5sum /proc/20295/exe 的结果与之前 pr_pid 的 md5 值也完全相同。
- `删除原可执行文件后，如果我们重建一个同名文件，会发现新文件与被删除的 pr_pid 文件的 i 节点号不一样`。重建文件后，原程序的输出也没有变化，说明重建的文件与原 pr_pid 无关。这种机制与删除被打开文件的机制类似。
- 当我们`试图向正在运行的可执行文件中写入内容时`，会写入失败，系统提示 “Text file busy” 表示有进程正在执行这一文件。

> - 用 rm 删除时，只是删除了文件名，系统为运行的进程自动保留了可执行文件的内容。
> - 用 cp 命令覆盖时，会尝试向当前可执行文件中写入新内容，如果成功写入，必然会影响当前正在运行的进程，因而操作系统禁止了对可执行文件的写入操作。

##### .3. 动态链接库正在被使用

-  当动态链接库正在被使用时，rm 命令删除的也只是文件名，虽然在原目录下已经没有了对应的库文件，但是操作系统会为使用库的进程保留库文件内容，因而 rm 并未真正删除磁盘上的库文件。从这点上来看，当删除使用中的动态链接库时，操作系统的机制和删除可执行文件及删除被打开的文件是一样的。
- `操作系统是允许我们向使用中的动态库中写入内容的，并不会像写入可执行文件一样报告 “Text file busy”，因而在写入方面，操作系统只对可执行文件进行了保护。程序崩溃了，这是由于内存映射区与磁盘文件的自动同步造成的。`

#### Resource

- https://zhuanlan.zhihu.com/p/25650525

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/rmcpmv%E5%8E%9F%E7%90%86/  

