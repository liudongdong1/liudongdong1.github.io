# FileOp


### 1. 文件共享

#### .1. 内核文件数据结构

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220410104557377.png)

#### .2. 多进程打开同一文件

- 每一个进程有自己的对该文件的当前偏移量
- lseek 定位到文件当前尾端，则文件表项的当前文件偏移量被设置i节点表项的当前文件长度
- 使用O_APPEND标志打开了一个文件，则相应的标志也被设置到文件表项的文件状态标志中，每次对文件具有添写标志文件执行写操作，在文件表项中的当前文件偏移量首先被设置为i节点表项中的文件长度，使得每次都添加到最后

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220410105019213.png)

### 2. dup2 & dup 文件拷贝

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220410110824682.png)

### 3. sync，fsync，fdatasync 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220410111239348.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220410111250299.png)

### 4. stat, fstat, lstat函数

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220410112742999.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/unix%E7%8E%AF%E5%A2%83%E9%AB%98%E7%BA%A7%E7%BC%96%E7%A8%8B-%E6%96%87%E4%BB%B6/  

