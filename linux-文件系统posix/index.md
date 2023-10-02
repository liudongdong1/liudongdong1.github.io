# Linux 文件系统


> POSIX表示可移植操作系统接口（Portable Operating System Interface of UNIX，缩写为 POSIX ），`POSIX标准定义了操作系统应该为应用程序提供的接口标准`，是IEEE为要在各种UNIX操作系统上运行的软件而定义的一系列API标准的总称，其正式称呼为IEEE 1003，而国际标准名称为ISO/IEC 9945
>
> POSIX标准意在期望获得源代码级别的软件可移植性。换句话说，`为一个POSIX兼容的操作系统编写的程序，应该可以在任何其它的POSIX操作系统（即使是来自另一个厂商）上编译执行`

## 1. 创建 / 打开 / 关闭文件

### 1). 打开文件 / open

> - **作用**：
>   打开一个文件，并返回文件描述符
>
> - **头文件**：
>
>   ```cpp
>       #include <fcntl.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>       int open(const char *file_name, int flags)
>       int open(const char *file_name, int flags, mode_t mode)
>   ```
>
> - **参数**：
>
> > - **file_name**: 欲打开的文件名(可包含路径)
> > - **mode**: 创建文件的权限( mode & ~umask)
> > - **flags**：

| flags       | description                                                  |
| ----------- | ------------------------------------------------------------ |
| O_RDONLY    | 只读                                                         |
| O_WRONLY    | 只写                                                         |
| O_RDWR      | 读写                                                         |
| O_CREAT     | 若欲打开的文件不存在，则创建该文件 (**需要用到第三个参数mode**) |
| O_APPEND    | 追加的方式打开(定位到文件尾部)                               |
| O_TRUNC     | 截取一定长度                                                 |
| O_EXCL      | 若O_CREAT也设置，此指令检查文件是否存在：若不存在则建立新文件；若存在则报错 |
| O_LARGEFILE | 在32bit系统中，支持大于2G的文件的打开                        |
| O_DIRECTORY | 若打开的文件不是目录，则报错                                 |

> - **返回值**：
>   成功：返回文件描述符fd
>   失败：-1

### 2). 关闭文件 / close

> - **作用**：
>   关闭一个文件
>
> - **头文件**：
>
>   ```cpp
>       #include <unistd.h>
>   ```
>
> - **函数原型**：
>
>   ```go
>       int close(int fd)
>   ```
>
> - **参数**：
>
> > **fd**: 欲关闭的文件描述符
>
> - **返回值**：
>   成功：返回文件描述符fd
>   失败：-1

### 3). 创建文件 / creat

> - **作用**：
>   创建一个文件，并返回文件描述符
>
> - **头文件**：
>
>   ```cpp
>       #include <fcntl.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>       int creat(const char *pathname, mode_t mode)
>   ```
>
> - **参数**：
>
> > - **pathname**: 欲打开的文件名(可包含路径)
> > - **mode**: 创建文件的权限( mode & ~umask)
>
> - **返回值**：
>   成功：返回文件描述符fd
>   失败：-1

## 2. 文件读/写

### 1). 读 / read

> - **作用**：
>   读取文件内容，并返回实际读取到的字节数
>
> - **头文件**：
>
>   ```cpp
>       #include <unistd.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>       ssize_t read(int fd, void *buf, size_t count)
>   ```
>
> - **参数**：
>
> > - **fd**: 读取的文件的文件描述符
> > - **buf**: 暂存区(暂时存储读取到的文件内容)
> > - **count**: 读取的字节数
>
> - **返回值**：
>   成功：读取的文件字节数(若返回值小于count,可能是读取到了EOF或者出错)
>   失败：-1
> - **说明**:
>   ssize_t其实为int

### 2). 写 / write

> - **作用**：
>   向文件中写入一定的内容，并返回实际写入文件的字节数
>
> - **头文件**：
>
>   ```cpp
>       #include <unistd.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>       ssize_t write(int fd, void *buf, size_t count)
>   ```
>
> - **参数**：
>
> > - **fd**: 写入的文件的文件描述符
> > - **buf**: 暂存区(将该暂存区的内容写入到文件)
> > - **count**: 写入的字节数
>
> - **返回值**：
>   成功：实际写入到文件的字节数
>   失败：-1

## 3. 文件定位

### 1). 修改文件读写位置 / lseek

> - **作用**：
>   修改文件的读写位置，并返回当前文件指针所在的位置
>
> - **头文件**：
>
>   ```cpp
>       #include <sys/types.h>
>       #include <unistd.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>       off_t lseek(int fd, off_t offset, int whence)
>   ```
>
> - **参数**：
>
> > - **fd**: 写入的文件的文件描述符
> > - **offset**: 相对于第三个参数whence的偏移量
> > - **whence**:

| whence   | description      |
| -------- | ---------------- |
| SEEK_SET | 文件开始位置     |
| SEEK_CUR | 文件当前指针位置 |
| SEEK_END | 文件结束位置     |

> - **返回值**：
>   成功：文件指针相对开始位置的偏移量(bytes)
>   失败：-1

## 4. 文件映射

### 1). 文件映射到内存 / mmap

> - **作用**：
>   将某特定文件映射到内存，使得进程可以像读写内存一样对文件进行读写操作
>
> - **头文件**：
>
>   ```cpp
>       #include <sys/mman.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>       void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)  
>       int munmap(void *addr, size_t length)
>   ```
>
> - **参数**：
>
> > - **addr**: 文件映射的内存地址，一般为`NULL`,由操作系统决定映射的起始地址
> > - **length**: 映射的文件内容的长度
> > - **fd**: 文件描述符
> > - **offset**: 文件指针的偏移量，即在何处开始文件的映射
> > - **prot**: 内存的读写状态

| prot       | description        |
| ---------- | ------------------ |
| PROT_READ  | 允许读该内存段     |
| PROT_WRITE | 允许写该内存段     |
| PROT_EXEC  | 允许执行该内存段   |
| PROT_NONE  | 该内存段不可被访问 |

> > - **flags**:

| flags       | descripion                                 |
| ----------- | ------------------------------------------ |
| MAP_PRIVATE | 内存私有，其它进程不可见                   |
| MAP_SHARED  | 共享内存，该进程的内存的更新对其它进程可见 |

> - **返回值**：
>   成功：文件指针相对开始位置的偏移量(bytes)
>   失败：-1

## 5. 锁定/解锁文件

### 1). 锁定解锁文件 / fcntl & flock

> - **作用**：
>   多进程对同一个文件进行操作时，可能对导致文件破坏；为了避免该问题，采用文件锁定的方式，即一个时间段内，只允许一个进程对文件进行操作
>
> - **头文件**：
>
>   ```cpp
>       #include <unistd.h>
>       #include <fcntl.h>
>       #include <sys/file.h>   //for flock
>   ```
>
> - **函数原型**：
>
>   ```java
>       int fcntl(int fd, int cmd, ... /* arg */ )  //可对文件进行局部锁定
>       int flock(int fd, int operation)    //锁定整个文件
>   ```
>
> - **参数**：
>
> > - **fd**: 文件描述符
> > - **operation**: 映射的文件内容的长度

| operation | description                                    |
| --------- | ---------------------------------------------- |
| LOCK_SH   | 共享锁，多个进程可在同一时间内对该文件进行操作 |
| LOCK_EX   | 互斥锁定，任何两个进程不可同时对该文件进行操作 |
| LOCK_UN   | 解除文件锁定状态                               |
| LOCK_NB   | 无法建立锁定时，不被阻塞，立即返回给进程       |

> > - **cmd**:

| cmd     | description                                        |
| ------- | -------------------------------------------------- |
| F_DUPFD | 复制文件描述符                                     |
| F_GETFD | 获取与文件描述符相关联的close-on-exec              |
| F_SETFD | 将与文件描述符相关联的close-on-exec设置为第3个参数 |
| F_GETFL | 获取文件的状态标志与访问模式                       |
| F_SETFL | 将文件状态标志设置为第3个参数                      |

> - **返回值**：
>   成功：依赖于操作
>   失败：-1

## 6. 文件信息 / stat

> - **作用**：
>   通过文件名获取文件的信息，包括文件类型、访问权限、节点、总大小等
>
> - **头文件**
>
>   ```cpp
>       #include <sys/stat.h>  
>       #include <sys/types.h>
>   ```
>
> - **函数原型**：
>
>   ```cpp
>     int stat(const char *file_name, struct stat *buf)
>   ```
>
> - **参数**：
>
> > **file_name**: 文件路径名，可为绝对路径和相对路径(相对程序执行的目录)
> > **buf**: 为`struct stat`指针
> >
> > ```cpp
> >       struct stat {
> >           dev_t     st_dev;     /* ID of device containing file */
> >           ino_t     st_ino;     /* inode number */
> >           mode_t    st_mode;    /* file type & mode */
> >           nlink_t   st_nlink;   /* number of hard links */
> >           uid_t     st_uid;     /* user ID of owner */
> >           gid_t     st_gid;     /* group ID of owner */
> >           dev_t     st_rdev;    /* device ID (if special file) */
> >           off_t     st_size;    /* total size, in bytes */
> >           blksize_t st_blksize; /* blocksize for file system I/O */
> >           blkcnt_t  st_blocks;  /* number of 512B blocks allocated */
> >           time_t    st_atime;   /* time of last access */
> >           time_t    st_mtime;   /* time of last modification */
> >           time_t    st_ctime;   /* time of last status change */
> >       };
> > 
> >  st_mode定义了以下几种情况：
> >      S_IFMT   0170000    //文件类型的位遮罩
> >      S_IFSOCK 0140000    //scoket
> >      S_IFLNK 0120000     //符号连接
> >      S_IFREG 0100000     //一般文件
> >      S_IFBLK 0060000     //区块装置
> >      S_IFDIR 0040000     //目录
> >      S_IFCHR 0020000     //字符装置
> >      S_IFIFO 0010000     //先进先出
> >    
> >      S_ISUID 04000     //文件的(set user-id on execution)位
> >      S_ISGID 02000     //文件的(set group-id on execution)位
> >      S_ISVTX 01000     //文件的sticky位
> >    
> >      S_IRUSR 00400     //文件所有者具可读取权限
> >      S_IWUSR 00200     //文件所有者具可写入权限
> >      S_IXUSR 00100     //文件所有者具可执行权限
> >    
> >      S_IRGRP 00040    //用户组具可读取权限
> >      S_IWGRP 00020    //用户组具可写入权限
> >      S_IXGRP 00010    //用户组具可执行权限
> >    
> >      S_IROTH 00004    //其它户具可读取权限
> >      S_IWOTH 00002    //其它用户具可写入权限
> >      S_IXOTH 00001    //其它用户具可执行权限
> > 
> >  Linux定义了以下几个宏，用于测试st_mode的类型：
> >      S_ISLNK (st_mode)    //判断是否为符号连接
> >      S_ISREG (st_mode)    //是否为一般文件(字符文件与二进制文件)
> >      S_ISDIR (st_mode)    //是否为目录
> >      S_ISCHR (st_mode)    //是否为字符装置文件
> >      S_ISBLK (s3e)        //是否为先进先出
> >      S_ISSOCK (st_mode)   //是否为socket
> >  Return True(1) or False(0)
> > 折叠 
> > ```

> - **返回值**：
>   成功：0
>   失败：-1
>   失败原因：见errno.
>
> - **相似函数**：
>
>   ```cpp
>       int fstat(int fd, struct stat *buf)             //同stat（第一个参数为文件描述符）  
>       int lstat(const char *file_name, struct stat *buf)  //同stat(),只是若为符号链接文件，是链接本间，而非实际的文件
>   ```

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/linux-%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9Fposix/  

