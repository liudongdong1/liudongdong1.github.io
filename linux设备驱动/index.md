# Linux Operation


### 1. 内核划分

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221108144636379.png)

 .1. 字符设备

- 文本控制台( /dev/console )和串口( /dev/ttyS0 及其友 )是字符设备的例子

 .2.块设备

.3. 网络接口： 内核与网络设备驱动间的通讯与字符和块设备驱动所用的完全不同. 不用 read 和 write, 内核调用和报文传递相关的函数.

####  连接一个模块到内核

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221108151129740.png)

- 应用程序存在于虚拟内存中, 有一个非常大的堆栈区，堆栈。
-  内核, 相反, 有一个非常小的堆栈; 它可能小到一个, 4096 字节的页. 

#### 堆叠模块

- modprobe工具使用

![image-20221108154002563](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221108154002563.png)



```c
moudle.h 包含了大量加载模块需要的函数和符号的定义
init.h 来指定你的初始化和清理函数
```

```c
static int __init initialization_function(void) 
{ 
 /* Initialization code here */ 
} 
module_init(initialization_function);
// 给定的函数只是在初始化使用. 模块加载者在模块加载后会丢掉这个初始化函数, 使它的内存可做其他用途.
//moudle_init 是强制的. 这个宏定义增加了特别的段到模块目标代码中, 表明在哪里找到模块的初始化函数.

static void __exit cleanup_function(void) 
{ 
 /* Cleanup code here */ 
} 
module_exit(cleanup_function);
```

#### 内核初始化错误处理

```c
int __init my_init_function(void) 
{ 
    int err; 
    err = register_this(ptr1, "skull"); /* registration takes a pointer and a name */ 
    if (err) 
        goto fail_this; 
    err = register_that(ptr2, "skull"); 
    if (err) 
        goto fail_that; 
    err = register_those(ptr3, "skull"); 
    if (err) 
        goto fail_those; 
    return 0; /* success */ 
    fail_those: 
    unregister_that(ptr2, "skull"); 
    fail_that: 
    unregister_this(ptr1, "skull"); 
    fail_this: 
    return err; /* propagate the error */
}

void __exit my_cleanup_function(void) 
{ 
    unregister_those(ptr3, "skull"); 
    unregister_that(ptr2, "skull"); 
    unregister_this(ptr1, "skull"); 
    return; 
}
```

```c
struct something *item1; 
struct somethingelse *item2; 
int stuff_ok; 
void my_cleanup(void) 
{ 
    if (item1) 
        release_thing(item1); 
    if (item2) 
        release_thing2(item2); 
    if (stuff_ok) 
        unregister_stuff(); 
    return; 
}
int __init my_init(void)
{ 
    int err = -ENOMEM; 
    item1 = allocate_thing(arguments); 
    item2 = allocate_thing2(arguments2); 
    if (!item2 || !item2) 
        goto fail; 
    err = register_stuff(item1, item2); 
    if (!err) 
        stuff_ok = 1; 
    else 
        goto fail; 
    return 0; /* success */ 
    fail: 
    my_cleanup(); 
    return err; 
}
```

#### 模块加载竞争&参数

```c
static char *whom = "world"; 
static int howmany = 1; 
module_param(howmany, int, S_IRUGO); 
module_param(whom, charp, S_IRUGO);
module_param_array(name,type,num,perm);
```

### 2. 字符设备

#### 分配释放设备号

```c
int register_chrdev_region(dev_t first, unsigned int count, char *name);
//first 是你要分配的起始设备编号. first 的次编号部分常常是 0, name 是应当连接到这个编号范围的设备的名子; 它会出现在 /proc/devices 和 sysfs 中.

//动态分配编号
int alloc_chrdev_region(dev_t *dev, unsigned int firstminor, unsigned int count, 
char *name);
// dev 是一个只输出的参数, 它在函数成功完成时持有你的分配范围的第一个数. fisetminor 应当是请求的第一个要用的次编号; 它常常是 0. count 和 name 参数如同给 request_chrdev_region 的一样.

void unregister_chrdev_region(dev_t first, unsigned int count);
```

> file_operations 结构持有一个字符驱动的方法; struct file 代表一个打开的文件, struct inode 代表磁盘上的一个文件. 

#### 文件操作

- file_operation 结构是一个字符驱动如何建立这个连接

```c
struct file_operations scull_fops = { 
 .owner = THIS_MODULE, 
 .llseek = scull_llseek, 
 .read = scull_read, 
 .write = scull_write, 
 .ioctl = scull_ioctl, 
 .open = scull_open, 
 .release = scull_release, 
};
```

```c++
struct module *owner; //它是一个指向拥有这个结构的模块的指针
loff_t (*llseek) (struct file *, loff_t, int); //用作改变文件中的当前读/写位置, 并且新位置作为(正的)返回值.
ssize_t (*read) (struct file *, char __user *, size_t, loff_t *); //用来从设备中获取数据, 一个非负返回值代表了成功读取的字节数( 返回值是一个 "signed size" 类型, 常常是目标平台本地的整数类型).
ssize_t (*aio_read)(struct kiocb *, char __user *, size_t, loff_t); //初始化一个异步读 -- 可能在函数返回前不结束的读操作.     
ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *); //发送数据给设备. 如果 NULL, -EINVAL 返回给调用 write 系统调用的程序. 如果非负, 返回值代表成功写的字节数.
ssize_t (*aio_write)(struct kiocb *, const char __user *, size_t, loff_t *); //初始化设备上的一个异步写.
int (*readdir) (struct file *, void *, filldir_t);//来读取目录, 并且仅对文件系统有用
unsigned int (*poll) (struct file *, struct poll_table_struct *); //poll 方法是 3 个系统调用的后端: poll, epoll, 和 select, 都用作查询对一个或多个文件描述符的读或写是否会阻塞
int (*ioctl) (struct inode *, struct file *, unsigned int, unsigned long); // 系统调用提供了发出设备特定命令的方法
int (*mmap) (struct file *, struct vm_area_struct *); //用来请求将设备内存映射到进程的地址空间
int (*open) (struct inode *, struct file *); //
int (*flush) (struct file *); //在进程关闭它的设备文件描述符的拷贝时调用; 它应当执行(并且等待)设备的任何未完成的操作
int (*release) (struct inode *, struct file *);  //在文件结构被释放时引用这个操作
int (*fsync) (struct file *, struct dentry *, int); //用户调用来刷新任何挂着的数据
int (*aio_fsync)(struct kiocb *, int);
int (*fasync) (int, struct file *, int); //通知设备它的 FASYNC 标志的改变
int (*lock) (struct file *, int, struct file_lock *); //实现文件加锁;
ssize_t (*readv) (struct file *, const struct iovec *, unsigned long, loff_t *); 
ssize_t (*writev) (struct file *, const struct iovec *, unsigned long, loff_t *);
ssize_t (*sendfile)(struct file *, loff_t *, size_t, read_actor_t, void *); //最少的拷贝从一个文件描述符搬移数据到另一个. 例如, 它被一个需要发送文件内容到一个网络连接的 web 服务器使用.
ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int); //它由内核调用来发送数据, 一次一页, 到对应的文件
unsigned long (*get_unmapped_area)(struct file *, unsigned long, unsigned long, unsigned 
long, unsigned long);  //是在进程的地址空间找一个合适的位置来映射在底层设备上的内存段中
int (*check_flags)(int); //允许模块检查传递给 fnctl(F_SETFL...) 调用的标志
int (*dir_notify)(struct file *, unsigned long);   //在应用程序使用 fcntl 来请求目录改变通知时调用
```

#### 文件结构

文件结构代表一个打开的文件. (它不特定给设备驱动; 系统中每个打开的文件有一个关联的 struct file 在内核空间). 它由内核在 open 时创建, 并传递给在文件上操作的任何函数, 直到最后的关闭. 在文件的所有实例都关闭后, 内核释放这个数据结构。

```c++
mode_t f_mode; //文件模式
loff_t f_pos;  //当前读写位置.
unsigned int f_flags;
struct file_operations *f_op;  //和文件关联的操作
void *private_data;
struct dentry *f_dentry;  //关联到文件的目录入口( dentry )结构, filp->f_dentry->d_inode 存取 inode 结构
```

#### inode 结构

```c++
dev_t i_rdev;  //代表设备文件的节点, 这个成员包含实际的设备编号
struct cdev *i_cdev;  //是内核的内部结构, 代表字符设备; 这个成员包含一个指针, 指向这个结构, 当节点指的是一个字符设备文件时.
//可用来从一个 inode 中获取主次编号
unsigned int iminor(struct inode *inode); 
unsigned int imajor(struct inode *inode);
```

#### 字符设备注册

```c++
void cdev_init(struct cdev *cdev, struct file_operations *fops);
int cdev_add(struct cdev *dev, dev_t num, unsigned int count);
void cdev_del(struct cdev *dev);
```

##### scull 设备注册

```c++
struct scull_dev { 
    struct scull_qset *data; /* Pointer to first quantum set */ 
    int quantum; /* the current quantum size */ 
    int qset; /* the current array size */ 
    unsigned long size; /* amount of data stored here */ 
    unsigned int access_key; /* used by sculluid and scullpriv */ 
    struct semaphore sem; /* mutual exclusion semaphore */ 
    struct cdev cdev; /* Char device structure */ 
};
static void scull_setup_cdev(struct scull_dev *dev, int index) 
{ 
    int err, devno = MKDEV(scull_major, scull_minor + index); 
    cdev_init(&dev->cdev, &scull_fops); 
    dev->cdev.owner = THIS_MODULE; 
    dev->cdev.ops = &scull_fops; 
    err = cdev_add (&dev->cdev, devno, 1);
    /* Fail gracefully if need be */ 
    if (err) 
        printk(KERN_NOTICE "Error %d adding scull%d", err, index); 
}
```

##### Open函数

```c++
container_of(pointer, container_type, container_field); //这个宏使用一个指向 container_field 类型的成员的指针, 它在一个 container_type 类型的结构中, 并且返回一个指针指向包含结构.
int scull_open(struct inode *inode, struct file *filp) 
{ 
    struct scull_dev *dev; /* device information */ 
    dev = container_of(inode->i_cdev, struct scull_dev, cdev); 
    filp->private_data = dev; /* for other methods */ 
    /* now trim to 0 the length of the device if open was write-only */ 
    if ( (filp->f_flags & O_ACCMODE) == O_WRONLY) 
    { 
        scull_trim(dev); /* ignore errors */ 
    } 
    return 0; /* success */ 
}
```

#####    release 函数

- 释放 open 分配在 filp->private_data 中的任何东西  
- 在最后的 close 关闭设备

```c
int scull_release(struct inode *inode, struct file *filp) 
{ 
    return 0; 
}
//fork 和 dup 都不创建新文件(只有 open 这样); 它们只递增正存在的结构中的计数. 
//close 系统调用仅在文件结构计数掉到 0 时执行 release 方法
```

#####  scull 内存布局

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221109115252792.png)

```c
struct scull_qset { 
    void **data; 
    struct scull_qset *next; 
};
int scull_trim(struct scull_dev *dev) 
{ 
    struct scull_qset *next, *dptr; 
    int qset = dev->qset; /* "dev" is not-null */ 
    int i; 
    for (dptr = dev->data; dptr; dptr = next) 
    { /* all the list items */ 
        if (dptr->data) { 
            for (i = 0; i < qset; i++) 
                kfree(dptr->data[i]); 
            kfree(dptr->data); 
            dptr->data = NULL; 
        } 
        next = dptr->next; 
        kfree(dptr); 
    }
    dev->size = 0; 
    dev->quantum = scull_quantum; 
    dev->qset = scull_qset; 
    dev->data = NULL; 
    return 0; 
}
```

##### 读写read&write

```c++
ssize_t read(struct file *filp, char __user *buff, size_t count, loff_t *offp); 
ssize_t write(struct file *filp, const char __user *buff, size_t count, loff_t *offp);
// 用户和内核空间数据拷贝
unsigned long copy_from_user (void *to, const void *from, unsigned long count); 
unsigned long copy_to_user (void *to, const void *from, unsigned long count);
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221109120002014.png)

-  todo ？ 这里read代码没有完全看明白， quantum这个有什么作用

```c
ssize_t scull_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos)
{ 
    struct scull_dev *dev = filp->private_data; 
    struct scull_qset *dptr; /* the first listitem */ 
    int quantum = dev->quantum, qset = dev->qset; 
    int itemsize = quantum * qset; /* how many bytes in the listitem */ 
    int item, s_pos, q_pos, rest; 
    ssize_t retval = 0; 
    if (down_interruptible(&dev->sem))
        return -ERESTARTSYS; 
    if (*f_pos >= dev->size) 
        goto out; 
    if (*f_pos + count > dev->size) 
        count = dev->size - *f_pos; 
    /* find listitem, qset index, and offset in the quantum */ 
    item = (long)*f_pos / itemsize; 
    rest = (long)*f_pos % itemsize; 
    s_pos = rest / quantum; 
    q_pos = rest % quantum; 
    /* follow the list up to the right position (defined elsewhere) */ 
    dptr = scull_follow(dev, item); 
    if (dptr == NULL || !dptr->data || ! dptr->data[s_pos])
        goto out; /* don't fill holes */ 
    /* read only up to the end of this quantum */ 
    if (count > quantum - q_pos) 
        count = quantum - q_pos; 
    if (copy_to_user(buf, dptr->data[s_pos] + q_pos, count)) 
    { 
        retval = -EFAULT; 
        goto out; 
    } 
    *f_pos += count; 
    retval = count; 
    out: 
    up(&dev->sem); 
    return retval; 
}
```

```c
ssize_t scull_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos) 
{ 
    struct scull_dev *dev = filp->private_data; 
    struct scull_qset *dptr; 
    int quantum = dev->quantum, qset = dev->qset; 
    int itemsize = quantum * qset; 
    int item, s_pos, q_pos, rest; 
    ssize_t retval = -ENOMEM; /* value used in "goto out" statements */ 
    if (down_interruptible(&dev->sem)) 
        return -ERESTARTSYS; 
    /* find listitem, qset index and offset in the quantum */ 
    item = (long)*f_pos / itemsize; 
    rest = (long)*f_pos % itemsize; 
    s_pos = rest / quantum; 
    q_pos = rest % quantum; 
    /* follow the list up to the right position */ 
    dptr = scull_follow(dev, item); 
    if (dptr == NULL) 
        goto out; 
    if (!dptr->data) 
    { 
        dptr->data = kmalloc(qset * sizeof(char *), GFP_KERNEL); 
        if (!dptr->data) 
            goto out; 
        memset(dptr->data, 0, qset * sizeof(char *)); 
    } 
    if (!dptr->data[s_pos]) 
    { 
        dptr->data[s_pos] = kmalloc(quantum, GFP_KERNEL); 
        if (!dptr->data[s_pos]) 
            goto out; 
    } 
    /* write only up to the end of this quantum */ 
    if (count > quantum - q_pos) 
        count = quantum - q_pos; 
    if (copy_from_user(dptr->data[s_pos]+q_pos, buf, count)) 
    { 
        retval = -EFAULT; 
        goto out; 
    } 
    *f_pos += count; 
    retval = count; 
    /* update the size */ 
    if (dev->size < *f_pos) 
        dev->size = *f_pos; 
    out: 
    up(&dev->sem); 
    return retval; 
} 
```

##### 读写矢量 readv&writev

```c++
ssize_t (*readv) (struct file *filp, const struct iovec *iov, unsigned long count, loff_t 
*ppos); 
ssize_t (*writev) (struct file *filp, const struct iovec *iov, unsigned long count, loff_t 
*ppos);
```

```c
struct iovec 
{ 
    void __user *iov_base; __kernel_size_t iov_len; 
};
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/v2-e14e59d101071b1ff39d9854bd248d9b_720w.webp)

### 3. 字符设备驱动

- 在Linux内核中，使用cdev结构体来描述一个字符设备，cdev结构体在/include/linux/cdev.h中定义。
- dev_t定义了设备号，一共32位，主设备号高12位，次设备号低20位。主设备号用来区分设备类型，次设备号用来区分同类型的不同设备。

#### .1. 设备驱动结构

```c++
struct cdev {
        struct kobject kobj;  /* 内嵌的内核对象 */
        struct module *owner;  /* 模块所有者，一般为THIS OWNER */
        const struct file_operations *ops;  /* 文件操作结构体 */
        struct list_head list;  /* 把所有向内核注册的字符设备形成链表 */
        dev_t dev;  /* 设备号，由主设备号和次设备号构成 */
        unsigned int count;  /* 属于同主设备号的次设备号的个数 */
} __randomize_layout;
```

```c
MAJOR(dev)  /* 从dev_t中获取主设备号 */
MINOR(dev)  /* 从dev_t中获取次设备号 */
MKDEV(ma,mi)  /* 通过主设备号ma和次设备号mi生成dev_t */
```
#### .2. 模块加载函数和模块卸载函数

```c
/* 设备驱动模块加载函数 */
static int _ _init xxx_init(void)
{
    ...
    cdev_init(&xxx_dev.cdev, &xxx_fops);  /* 初始化 cdev */
    xxx_dev.cdev.owner = THIS_MODULE;
    /* 获取字符设备号 */
    if (xxx_major) {
      register_chrdev_region(xxx_dev_no, 1, DEV_NAME);
    } else {
      alloc_chrdev_region(&xxx_dev_no, 0, 1, DEV_NAME);
    }

    ret = cdev_add(&xxx_dev.cdev, xxx_dev_no, 1); /* 注册设备 */
    ...
}

/* 设备驱动模块卸载函数 */
static void _ _exit xxx_exit(void)
{
    unregister_chrdev_region(xxx_dev_no, 1);  /* 释放占用的设备号 */
    cdev_del(&xxx_dev.cdev);  /* 注销设备 */
    ...
}
```

#### .3. file_operations结构体的成员函数

```c
/* 打开设备 */
static int xxx_open(struct inode *inode, struct file *filp)
{
    ...
}

/* 释放设备 */
static int xxx_release(struct inode *inode, struct file *filp)
{
    ...
}

/* ioctl */
static long xxx_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    ...   
    switch (cmd) {
        case XXX_CMD1:
        ...
        break;
    case XXX_CMD2:
        ...
        break;
    default:
    /* 不能支持的命令 */
        return - ENOTTY;
    ...
}

/* 读 
 * filp:文件结构体指针
 * buf：读取的内存地址 
 * size:字节大小 
 * ppos:对文件操作的起始位置 
*/
static ssize_t xxx_read(struct file *filp, char __user * buf, size_t size, loff_t * ppos)
{
    ...
    copy_to_user(buf, ..., ...);
    ...
}

/* 写
* filp:文件结构体指针
 * buf：写入的内存地址 
 * size:字节大小 
 * ppos:对文件操作的起始位置 
*/
static ssize_t xxx_write(struct file *filp, const char __user *buf, size_t size, loff_t *ppos)
{
    ...
    copy_from_user(..., buf, ...);
    ...
}

/* 文件操作结构体 */
static const struct file_operations test_fops = {
    .owner = THIS_MODULE,
    .open = xxx_open,
    .release = xxx_release,
    .read = xxx_read,
    .write = xxx_write,
    .unlocked_ioctl = xxx_ioctl,
};
```

#### .4. ioctl 接口

大部分设备可进行超出简单的数据传输之外的操作; 用户空间必须常常能够请求, 例如, 设备锁上它的门, 弹出它的介质, 报告错误信息, 改变波特率, 或者自我销毁. 

```c
int (*ioctl) (struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg);
```

```c
switch(cmd) 
{ 
    case SCULL_IOCRESET: 
        scull_quantum = SCULL_QUANTUM; 
        scull_qset = SCULL_QSET; 
        break; 
    case SCULL_IOCSQUANTUM: /* Set: arg points to the value */ 
        if (! capable (CAP_SYS_ADMIN)) 
            return -EPERM; 
        retval = __get_user(scull_quantum, (int __user *)arg); 
        break; 
    case SCULL_IOCTQUANTUM: /* Tell: arg is the value */ 
        if (! capable (CAP_SYS_ADMIN)) 
            return -EPERM; 
        scull_quantum = arg; 
        break; 
    case SCULL_IOCGQUANTUM: /* Get: arg is pointer to result */ 
        retval = __put_user(scull_quantum, (int __user *)arg); 
        break;
    case SCULL_IOCQQUANTUM: /* Query: return it (it's positive) */ 
        return scull_quantum; 
    case SCULL_IOCXQUANTUM: /* eXchange: use arg as pointer */ 
        if (! capable (CAP_SYS_ADMIN)) 
            return -EPERM; 
        tmp = scull_quantum; 
        retval = __get_user(scull_quantum, (int __user *)arg); 
        if (retval == 0) 
            retval = __put_user(tmp, (int __user *)arg); 
        break; 
    case SCULL_IOCHQUANTUM: /* sHift: like Tell + Query */ 
        if (! capable (CAP_SYS_ADMIN)) 
            return -EPERM; 
        tmp = scull_quantum; 
        scull_quantum = arg; 
        return tmp; 
    default: /* redundant, as cmd was checked against MAXNR */ 
        return -ENOTTY; 
} 
return retval;
```

#### .5. 阻塞IO

-  一个对 read 的调用可能当没有数据时到来, 而以后会期待更多的数据. 或者一个进程可能试图写, 但是你的设备没有准备好接受数据, 因为你的输出缓冲满了。
-  当一个进程被置为睡眠, 它被标识为处于一个特殊的状态并且从调度器的运行队列中去除. 直到发生某些事情改变了那个状态, 这个进程将不被在任何 CPU 上调度, 并且, 因此, 将不会运行
- 当你运行在原子上下文时不能睡眠，对于睡眠, 是你的驱动在持有一个自旋锁, seqlock, 或者 RCU 锁时不能睡眠. 如果你已关闭中断你也不能睡眠

```c
wait_event(queue, condition) 
wait_event_interruptible(queue, condition); // 一个非零值意味着你的睡眠被某些信号打断, 并且你的驱动可能应当返回 -ERESTARTSYS
wait_event_timeout(queue, condition, timeout) ; 
wait_event_interruptible_timeout(queue, condition, timeout);

void wake_up(wait_queue_head_t *queue); 
void wake_up_interruptible(wait_queue_head_t *queue);
```

```c
static DECLARE_WAIT_QUEUE_HEAD(wq); 
static int flag = 0; 
ssize_t sleepy_read (struct file *filp, char __user *buf, size_t count, loff_t 
                     *pos) 
{ 
    printk(KERN_DEBUG "process %i (%s) going to sleep\n", 
           current->pid, current->comm); 
    wait_event_interruptible(wq, flag != 0); 
    flag = 0; 
    printk(KERN_DEBUG "awoken %i (%s)\n", current->pid, current->comm); 
    return 0; /* EOF */ 
}

ssize_t sleepy_write (struct file *filp, const char __user *buf, size_t count, 
                      loff_t *pos) 
{ 
    printk(KERN_DEBUG "process %i (%s) awakening the readers...\n", 
           current->pid, current->comm); 
    flag = 1; 
    wake_up_interruptible(&wq); 
    return count; /* succeed, to avoid retrial */ 
}
```

#### .6. poll 和 select 

```c++
unsigned int (*poll) (struct file *filp, poll_table *wait);
void poll_wait (struct file *, wait_queue_head_t *, poll_table *);
```

```c
static unsigned int scull_p_poll(struct file *filp, poll_table *wait) 
{ 
    struct scull_pipe *dev = filp->private_data; 
    unsigned int mask = 0; 
    /* 
 * The buffer is circular; it is considered full 
 * if "wp" is right behind "rp" and empty if the 
 * two are equal. 
 */ 
    down(&dev->sem); 
    poll_wait(filp, &dev->inq, wait); 
    poll_wait(filp, &dev->outq, wait); 
    if (dev->rp != dev->wp) 
        mask |= POLLIN | POLLRDNORM; /* readable */ 
    if (spacefree(dev)) 
        mask |= POLLOUT | POLLWRNORM; /* writable */ 
    up(&dev->sem); 
    return mask; 
}
```

#### .7. 异步通知

- 首先, 它们指定一个进程作为文件的拥有者. 当一个进程使用 fcntl 系统调用发出 F_SETOWN 命令, 这个拥有者进程的 ID 被保存在 filp->f_owner 给以后使用. 
- 用户程序必须设置 FASYNC 标志在设备中, 通过 F_SETFL fcntl 命令，输入文件可请求递交一个 SIGIO 信号, 无论何时新数据到达. 信号被发送给存储于 filp->f_owner 中的进程(或者进程组, 如果值为负值). 
- 当 F_SETFL 被执行来打开 FASYNC, 驱动的 fasync 方法被调用. 这个方法被调用无论何时 FASYNC 的值在 filp->f_flags 中被改变来通知驱动这个变化, 因此它可正确地响应. 这个标志在文件被打开时缺省地被清除. 
- 当数据到达, 所有的注册异步通知的进程必须被发出一个 SIGIO 信号

##### 1.设备驱动如何实现异步信号

>1.  当发出 F_SETOWN, 什么都没发生, 除了一个值被赋值给 filp->f_owner. 
>2.  当 F_SETFL 被执行来打开 FASYNC, 驱动的 fasync 方法被调用. 这个方法被调用无论何时 FASYNC 的值在 filp->f_flags 中被改变来通知驱动这个变化, 因此它可正确地响应. 
>3. 当数据到达, 所有的注册异步通知的进程必须被发出一个 SIGIO 信号.   todo? 

#### .8. 移位一个设备

```c
loff_t scull_llseek(struct file *filp, loff_t off, int whence) 
{ 
    struct scull_dev *dev = filp->private_data; 
    loff_t newpos; 
    switch(whence) 
    { 
        case 0: /* SEEK_SET */ 
            newpos = off; 
            break; 
        case 1: /* SEEK_CUR */ 
            newpos = filp->f_pos + off; 
            break; 
        case 2: /* SEEK_END */ 
            newpos = dev->size + off; 
            break; 
        default: /* can't happen */ 
            return -EINVAL; 
    } 
    if (newpos < 0) 
        return -EINVAL; 
    filp->f_pos = newpos; 
    return newpos; 
}
```



 [字符驱动程序案例](https://www.tiandeng.xyz/posts/Linux%E5%AD%97%E7%AC%A6%E8%AE%BE%E5%A4%87%E9%A9%B1%E5%8A%A8/#%E8%AE%BE%E5%A4%87%E5%8F%B7)

```c
#include "linux/export.h"
#include "linux/gfp.h"
#include "linux/kern_levels.h"
#include "linux/printk.h"
#include "linux/types.h"
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#define MEM_SIZE 1024

#define DEV_TYPE                'k'
#define DEV_SET_VALUE           _IOR(DEV_TYPE, 1, int32_t*)
#define DEV_GET_VALUE           _IOW(DEV_TYPE, 2, int32_t*)

int32_t val = 0;

dev_t dev = 0;
static struct cdev my_cdev;
static struct class* dev_class;
uint8_t* kernel_buffer;

static int __init chr_driver_init(void);
static void __exit chr_driver_exit(void);
static int my_open(struct inode *_inode, struct file *_file);
static int my_release(struct inode *_inode, struct file *_file);
static ssize_t my_read(struct file *filp, char __user *buf, size_t len, loff_t *off);
static ssize_t my_write(struct file *filp, const char *buf, size_t len, loff_t *off);
static long my_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = my_read,
        .write          = my_write,
        .open           = my_open,
        .unlocked_ioctl = my_ioctl,
        .release        = my_release,
};

static int my_open(struct inode *_inode, struct file *_file)
{
        /* Creating Physical Memory */
        if((kernel_buffer = kmalloc(MEM_SIZE, GFP_KERNEL)) == 0) {
                printk(KERN_INFO "Cannot allocate memory to the kernel.\n");
                return -1;
        }
        printk(KERN_INFO "Device File opened.\n");
        return 0;
}

static int my_release(struct inode *_inode, struct file *_file)
{
        kfree(kernel_buffer);
        printk(KERN_INFO "Device File closed.\n");
        return 0;
}

static ssize_t my_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        ssize_t re;
        re = copy_to_user(buf, kernel_buffer, MEM_SIZE);
        printk(KERN_INFO "Data read : Done.\n");
        return re;
}

static ssize_t my_write(struct file *filp, const char *buf, size_t len, loff_t *off)
{
        ssize_t re;
        re = copy_from_user(kernel_buffer, buf, len);
        printk(KERN_INFO "Data write : successful.\n");
        return re;
}

#if 0
static long my_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{

        long re;
        switch (cmd) {
        case DEV_SET_VALUE:
                re = copy_from_user(&val, (int32_t*)arg, sizeof(val));
                break;
        case DEV_GET_VALUE:
                re = copy_to_user((int32_t*)arg, &val, sizeof(val));
                break;
        default:
                return -EINVAL;
        }
        return re;
}
#endif

static long my_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
        void __user *argp = (void __user*)arg;
        int32_t __user *p = argp;
        long err;

        switch (cmd) {
        case DEV_SET_VALUE:
                re = get_user(val, p);
                break;
        case DEV_GET_VALUE:
                re = put_user(val, p);
                break;
        default:
                return -EINVAL;
        }
        return re;
}

static int __init chr_driver_init(void)
{
        /* Allocating Major number */
        if (alloc_chrdev_region(&dev, 0, 1 , "my_Dev") < 0) {
                printk(KERN_INFO "Cannot allocate major number.");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d.\n", MAJOR(dev), MINOR(dev));

        /* creating cdev structure */
        cdev_init(&my_cdev, &fops);

        /* Adding character device to the system */
        if (cdev_add(&my_cdev, dev, 1) < 0) {
                printk(KERN_INFO "Cannot add the device to the system.\n");
                goto r_class;
        }

        /* creating struct class */
        if ((dev_class = class_create(THIS_MODULE, "my_class")) == NULL) {
                printk(KERN_INFO "Cannot create the struct class.\n");
                goto r_class;
        }

        /* creating device */
        if(device_create(dev_class, NULL, dev, NULL, "my_device") == NULL) {
                printk(KERN_INFO "Cannot create the device.\n");
                goto r_device;
        }
        printk(KERN_INFO "Device driver insert done.\n");
        return 0;

r_device:
        class_destroy(dev_class);

r_class:
        unregister_chrdev_region(dev, 1);
        return -1;
}

static void __exit chr_driver_exit(void)
{
        device_destroy(dev_class, dev);
        class_destroy(dev_class);
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device driver is removed successfully.\n");
}

module_init(chr_driver_init);
module_exit(chr_driver_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("tiandeng <tiandengzbc@gmail.com>");
MODULE_DESCRIPTION("A simple character device driver example");
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


u_int8_t write_buf[1024];
u_int8_t read_buf[1024];

int main()
{
        int fd;
        char options;


        printf("Welcome to the demo of character device driver.\n");
        fd = open("/dev/my_device", O_RDWR);

        if (fd < 0) {
                perror("failed to open device: ");
                return 0;
        }

        while (1) {
                printf("**************please enter your option***************** \n");
                printf("                1. Write                                \n");
                printf("                2. Read                                 \n");
                printf("                3. Exit                                 \n");
                scanf(" %c", &options);
                printf("Your options = %c\n", options);

                switch (options) {
                case '1':
                        printf("Enter the string to write into the driver:\n");
                        scanf(" %[^\t\n]s", write_buf);
                        printf("Data written .....\n");
                        write(fd, write_buf, strlen((char*)write_buf) + 1);
                        printf("Done...\n");
                        break;
                case '2':
                        printf("Data is Reading...\n");
                        read(fd, read_buf, 1024);
                        printf("Done...\n");
                        printf("Data = %s\n", read_buf);
                        break;
                case '3':
                        close(fd);
                        exit(1);
                        break;
                default:
                        printf("Enter valid option = %c\n", options);
                        break;
                }
        }

        return 0;
}
```

```c
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define DEV_TYPE                'k'
#define DEV_SET_VALUE           _IOR(DEV_TYPE, 1, int32_t*)
#define DEV_GET_VALUE           _IOW(DEV_TYPE, 2, int32_t*)

int main()
{
        int fd;
        int32_t val, num;
        printf("\n IOCTL based Character device driver operation from user space.\n");
        fd = open("/dev/my_device", O_RDWR);

        if (fd < 0) {
                perror("Cannot open the device file.:");
                return 0;
        }

        printf("Enter the data to set.\n");
        scanf("%d", &num);
        printf("Writing value to the driver.\n");
        ioctl(fd, DEV_SET_VALUE, &num);

        printf("Reading value from driver.\n");
        ioctl(fd, DEV_GET_VALUE, &val);
        printf("get value is %d\n", val);

        printf("Closed device\n");
        close(fd);
        return 0;
}
```

### 4. 打印调试

- todo

```c
printk(KERN_DEBUG "Here I am: %s:%i\n", __FILE__, __LINE__); 
printk(KERN_CRIT "I'm trashed; giving up on %p\n", ptr);
```

### 5. 并发和竞争情况

#### .1. **旗标和互斥体** 

```c
void down_write(struct rw_semaphore *sem); 
int down_write_trylock(struct rw_semaphore *sem); 
void up_write(struct rw_semaphore *sem); 
void downgrade_write(struct rw_semaphore *sem);
```

#### .2. Completions 机制

```c
struct completion my_completion;
init_completion(&my_completion); 
void wait_for_completion(struct completion *c); //进行一个不可打断的等待

void complete(struct completion *c); 
void complete_all(struct completion *c);
void complete_and_exit(struct completion *c, long retval);
```

```c
DECLARE_COMPLETION(comp); 
ssize_t complete_read (struct file *filp, char __user *buf, size_t count, 
                       loff_t *pos) 
{ 
    printk(KERN_DEBUG "process %i (%s) going to sleep\n",current->pid, 
           current->comm); 
    wait_for_completion(&comp); 
    printk(KERN_DEBUG "awoken %i (%s)\n", current->pid, current->comm); 
    return 0; /* EOF */ 
} 
ssize_t complete_write (struct file *filp, const char __user *buf, size_t 
                        count, loff_t *pos) 
{
    printk(KERN_DEBUG "process %i (%s) awakening the readers...\n", current-
           >pid, current->comm); 
    complete(&comp); 
    return count; /* succeed, to avoid retrial */ 
}
```

#### .3. 自旋锁

- 自旋锁的核心规则是任何代码必须, 在持有自旋锁时, 是原子性的. 它不能睡眠; 事实上, 它不能因为任何原因放弃处理器, 除了服务中断(并且有时即便此时也不行) 

```c
void spin_lock_init(spinlock_t *lock); //初始化
void spin_lock(spinlock_t *lock); //在进入一个临界区前, 你的代码必须获得需要的 lock
void spin_unlock(spinlock_t *lock);  // 取消自旋

void spin_lock(spinlock_t *lock); 
void spin_lock_irqsave(spinlock_t *lock, unsigned long flags); 
void spin_lock_irq(spinlock_t *lock); 
void spin_lock_bh(spinlock_t *lock);

void spin_unlock(spinlock_t *lock); 
void spin_unlock_irqrestore(spinlock_t *lock, unsigned long flags); 
void spin_unlock_irq(spinlock_t *lock); 
void spin_unlock_bh(spinlock_t *lock);


rwlock_t my_rwlock = RW_LOCK_UNLOCKED; /* Static way */ 
rwlock_t my_rwlock; 
rwlock_init(&my_rwlock); /* Dynamic way */
void read_lock(rwlock_t *lock); 
void read_lock_irqsave(rwlock_t *lock, unsigned long flags); 
void read_lock_irq(rwlock_t *lock); 
void read_lock_bh(rwlock_t *lock); 
void read_unlock(rwlock_t *lock); 
void read_unlock_irqrestore(rwlock_t *lock, unsigned long flags); 
void read_unlock_irq(rwlock_t *lock); 
void read_unlock_bh(rwlock_t *lock);

void write_lock(rwlock_t *lock); 
void write_lock_irqsave(rwlock_t *lock, unsigned long flags); 
void write_lock_irq(rwlock_t *lock); 
void write_lock_bh(rwlock_t *lock); 
int write_trylock(rwlock_t *lock); 
void write_unlock(rwlock_t *lock); 
void write_unlock_irqrestore(rwlock_t *lock, unsigned long flags); 
void write_unlock_irq(rwlock_t *lock); 
void write_unlock_bh(rwlock_t *lock);
```

#### .4. 锁陷阱

- 如果一个函数需要一个锁并且接着调用另一个函数也试图请求这个锁, 你的代码死锁.
- 不论旗标还是自旋锁都不允许一个持锁者第 2 次请求锁;
- 当多个锁必须获得时, 它们应当一直以同样顺序获得；
-  你可以用一个锁来涵盖你做的所有东西, 或者你可以给你管理的每个设备创建一个锁. 

#### .5. 加锁的选择

##### 1. 不加锁算法

- todo 无锁环形缓冲区

##### 2. 原子变量

```c
void atomic_set(atomic_t *v, int i); 
atomic_t v = ATOMIC_INIT(0);

int atomic_read(atomic_t *v);
void atomic_add(int i, atomic_t *v);
void atomic_sub(int i, atomic_t *v);
void atomic_inc(atomic_t *v); 
void atomic_dec(atomic_t *v);
int atomic_inc_and_test(atomic_t *v); 
int atomic_dec_and_test(atomic_t *v); 
int atomic_sub_and_test(int i, atomic_t *v);
int atomic_add_negative(int i, atomic_t *v);
```

##### 3. 位操作

```c
void set_bit(nr, void *addr);  //设置第 nr 位在 addr 指向的数据项中.
void clear_bit(nr, void *addr);
void change_bit(nr, void *addr); //翻转这个位.
test_bit(nr, void *addr);

int test_and_set_bit(nr, void *addr); 
int test_and_clear_bit(nr, void *addr); 
int test_and_change_bit(nr, void *addr);
```

```c
/* try to set lock */ 
while (test_and_set_bit(nr, addr) != 0) 
 wait_for_a_while(); 
/* do your work */ 
/* release lock, and check... */ 
if (test_and_clear_bit(nr, addr) == 0) 
 something_went_wrong(); /* already released: error */
```

##### 4.  **seqlock 锁**

```c
unsigned int seq; 
do { 
    seq = read_seqbegin(&the_lock);
    /* Do what you need to do */ 
} while read_seqretry(&the_lock, seq);

void write_seqlock_irqsave(seqlock_t *lock, unsigned long flags); 
void write_seqlock_irq(seqlock_t *lock); 
void write_seqlock_bh(seqlock_t *lock); 
void write_sequnlock_irqrestore(seqlock_t *lock, unsigned long flags); 
void write_sequnlock_irq(seqlock_t *lock); 
void write_sequnlock_bh(seqlock_t *lock);
```

##### 5. **读取-拷贝-更新** 

-  当数据结构需要改变, 写线程做一个拷贝, 改变这个拷贝, 接着使相关的指针对准新的版本

```c
struct my_stuff *stuff; 
rcu_read_lock(); 
stuff = find_the_stuff(args...); 
do_something_with(stuff); 
rcu_read_unlock();
```

### 6. 时间&延迟&延后工作

#### .1. 获取当前时间

```c
#include <linux/time.h> 
unsigned long mktime (unsigned int year, unsigned int mon, 
                      unsigned int day, unsigned int hour, 
                      unsigned int min, unsigned int sec);
```

#### .2. 延后执行

##### 1. 长延时

```c
//忙等待严重地降低了系统性能. 如果你不配置你的内核为抢占操作, 这个循环在延时期间完全锁住了处理器; 调度器永远不会抢占一个在内核中运行的进程, 并且计算机看起来完全死掉直到时间 j1 到时.
//当你进入循环时如果中断碰巧被禁止, jiffies 将不会被更新, 并且 while 条件永远保持真
while (time_before(jiffies, j1)) 
    cpu_relax();  //忙等待强加了一个重负载给系统总体;

//当前进程除了释放 CPU 不作任何事情, 但是它保留在运行队列中
while (time_before(jiffies, j1)) { 
    schedule();    //随着系统变忙会变得越来越坏, 并且驱动可能结束于等待长于期望的时间
}

//------------------ 推荐方式 
//驱动使用一个等待队列来等待某些其他事件, 但是你也想确保它在一个确定时间段内运行
#include <linux/wait.h> 
long wait_event_timeout(wait_queue_head_t q, condition, long timeout); 
long wait_event_interruptible_timeout(wait_queue_head_t q, condition, long 
timeout);
//如果超时到, 这些函数返回 0; 如果这个进程被其他事件唤醒, 它返回以 jiffies 表示的剩余超时值. 返回值从不会是负值, 甚至如果延时由于系统负载而比期望的值大.
wait_queue_head_t wait; 
init_waitqueue_head (&wait); 
wait_event_interruptible_timeout(wait, 0, delay);

//第一行调用 set_current_state 来设定一些东西以便调度器不会再次运行当前进程, 直到超时将它置回 TASK_RUNNING 状态. 为获得一个不可中断的延时, 使用 TASK_UNINTERRUPTIBLE 代替
set_current_state(TASK_INTERRUPTIBLE); 
schedule_timeout (delay);
```

##### .2. 短延时

```c
//这 3 个延时函数是忙等待; 其他任务在时间流失时不能运行
#include <linux/delay.h> 
void ndelay(unsigned long nsecs); 
void udelay(unsigned long usecs); 
void mdelay(unsigned long msecs);
//获得毫秒(和更长)延时而不用涉及到忙等待
void msleep(unsigned int millisecs); 
unsigned long msleep_interruptible(unsigned int millisecs); 
void ssleep(unsigned int seconds)
```

#### .3. 内核定时器

- 一个内核定时器是一个数据结构, 它指导内核执行一个用户定义的函数使用一个用户定义的参数在一个用户定义的时间
- 被调度运行的函数几乎确定不会在注册它们的进程在运行时运行. 它们是, 相反, 异步运行. 

```c
#include <linux/timer.h> 
struct timer_list 
{ 
    /* ... */ 
    unsigned long expires; 
    void (*function)(unsigned long); 
    unsigned long data; 
}; 
void init_timer(struct timer_list *timer); 
struct timer_list TIMER_INITIALIZER(_function, _expires, _data); 
void add_timer(struct timer_list * timer); 
int del_timer(struct timer_list * timer);

unsigned long j = jiffies; 
/* fill the data for our timer function */ 
data->prevjiffies = j; 
data->buf = buf2; 
data->loops = JIT_ASYNC_LOOPS; 
/* register the timer */ 
data->timer.data = (unsigned long)data; 
data->timer.function = jit_timer_fn; 
data->timer.expires = j + tdelay; /* parameter */ 
add_timer(&data->timer); 
/* wait for the buffer to fill */ 
wait_event_interruptible(data->wait, !data->loops); 
//The actual timer function looks like this: 
void jit_timer_fn(unsigned long arg) 
{ 
    struct jit_data *data = (struct jit_data *)arg; 
    unsigned long j = jiffies; 
    data->buf += sprintf(data->buf, "%9li %3li %i %6i %i %s\n", 
                         j, j - data->prevjiffies, in_interrupt() ? 1 : 0, 
                         current->pid, smp_processor_id(), current->comm); 
    if (--data->loops) { 
        data->timer.expires += tdelay; 
        data->prevjiffies = j; 
        add_timer(&data->timer); 
    } else { 
        wake_up_interruptible(&data->wait); 
    } 
}
```

#### .4. Tasklets

- 一个 tasklet 能够被禁止并且之后被重新使能; 它不会执行直到它被使能与被禁止相同的的次数
- 如同定时器, 一个 tasklet 可以注册它自己. 
- 一个 tasklet 能被调度来执行以正常的优先级或者高优先级. 后一组一直是首先执行
- taslet 可能立刻运行, 如果系统不在重载下, 但是从不会晚于下一个时钟嘀哒. 
- 一个 tasklet 可能和其他 tasklet 并发, 但是对它自己是严格地串行的 -- 同样的 tasklet 从不同时运行在超过一个处理器上. 同样, 如已经提到的, 一个 tasklet 常常在调度它的同一个 CPU 上运行. 

```c
#include <linux/interrupt.h> 
DECLARE_TASKLET(name, func, data); 
DECLARE_TASKLET_DISABLED(name, func, data);
void tasklet_init(struct tasklet_struct *t, void (*func)(unsigned long), 
unsigned long data);

void tasklet_disable(struct tasklet_struct *t); 
void tasklet_disable_nosync(struct tasklet_struct *t); 
void tasklet_enable(struct tasklet_struct *t);

void tasklet_schedule(struct tasklet_struct *t); 
void tasklet_hi_schedule(struct tasklet_struct *t);

void tasklet_kill(struct tasklet_struct *t);
```

#### .5. 工作队列--共享队列

```c
static struct work_struct jiq_work; 
/* this line is in jiq_init() */ 
INIT_WORK(&jiq_work, jiq_print_wq, &jiq_data);
int schedule_work(struct work_struct *work);

prepare_to_wait(&jiq_wait, &wait, TASK_INTERRUPTIBLE); 
schedule_work(&jiq_work); 
schedule(); 
finish_wait(&jiq_wait, &wait);

static void jiq_print_wq(void *ptr) 
{ 
    struct clientdata *data = (struct clientdata *) ptr; 
    if (! jiq_print (ptr)) 
        return; 
    if (data->delay) 
        schedule_delayed_work(&jiq_work, data->delay);   //重新提交它自己在延后的模式
    else 
        schedule_work(&jiq_work); 
}

void flush_scheduled_work(void);
```

### 7. 分配内存

#### .1. kmalloc

- 不清零它获得的内存; 分配的区仍然持有它原来的内容, 分配的区也是在物理内存中连续

```c
#include <linux/slab.h> 
void *kmalloc(size_t size, int flags);
```

- GFP_KENRL: 内核内存的正常分配. 可能睡眠. 意味着 kmalloc 能够使当前进程在少内存的情况下睡眠来等待一页,是可重入的并且不能在原子上下文中运行. 当当前进程睡眠, 内核采取正确的动作来定位一些空闲内存, 或者通过刷新缓存到磁盘或者交换出去一个用户进程的内存
- GFP_ATOMIC : 用来从中断处理和进程上下文之外的其他代码中分配内存. 从不睡眠. 
- GFP_USER :用来为用户空间页来分配内存; 它可能睡眠. 
- Linux 内核知道最少 3 个内存区: DMA-能够 内存, 普通内存, 和高端内存
- Linux 处理内存分配通过创建一套固定大小的内存对象池. 分配请求被这样来处理, 进入   一个持有足够大的对象的池子并且将整个内存块递交给请求者.

#### .2. 后备缓存

```c++
kmem_cache_t *kmem_cache_create(const char *name, size_t size, 
                                size_t offset, 
                                unsigned long flags, 
                                void (*constructor)(void *, kmem_cache_t *, 
                                                    unsigned long flags), void (*destructor)(void *, kmem_cache_t *, unsigned 
                                                                                             long flags));

void *kmem_cache_alloc(kmem_cache_t *cache, int flags);
void kmem_cache_free(kmem_cache_t *cache, const void *obj);
int kmem_cache_destroy(kmem_cache_t *cache);

/* declare one cache pointer: use it for all devices */ 
kmem_cache_t *scullc_cache;
/* scullc_init: create a cache for our quanta */ 
scullc_cache = kmem_cache_create("scullc", scullc_quantum, 
                                 0, SLAB_HWCACHE_ALIGN, NULL, NULL); /* no 
ctor/dtor */ 
if (!scullc_cache) 
{ 
    scullc_cleanup(); 
    return -ENOMEM; 
} 
/* Allocate a quantum using the memory cache */ 
if (!dptr->data[s_pos]) 
{ 
    dptr->data[s_pos] = kmem_cache_alloc(scullc_cache, GFP_KERNEL); 
    if (!dptr->data[s_pos]) 
        goto nomem; 
    memset(dptr->data[s_pos], 0, scullc_quantum); 
}

for (i = 0; i < qset; i++) 
    if (dptr->data[i]) 
        kmem_cache_free(scullc_cache, dptr->data[i]);
/* scullc_cleanup: release the cache of our quanta */ 
if (scullc_cache) 
    kmem_cache_destroy(scullc_cache);
```

##### .2. 内存池

```c
// min_nr: 内存池应当保留的最小数量的分配的对象. 实际的分配和释放对象由 alloc_fn 和 free_fn 处理
mempool_t *mempool_create(int min_nr, 
                          mempool_alloc_t *alloc_fn, 
                          mempool_free_t *free_fn, 
                          void *pool_data);

cache = kmem_cache_create(. . .); 
pool = mempool_create(MY_POOL_MINIMUM,mempool_alloc_slab, mempool_free_slab, 
cache);

void *mempool_alloc(mempool_t *pool, int gfp_mask); 
void mempool_free(void *element, mempool_t *pool);
int mempool_resize(mempool_t *pool, int new_min_nr, int gfp_mask);
void mempool_destroy(mempool_t *pool);
```

#### .3. **get_free_page 和其友** 

- 如果一个模块需要分配大块的内存, 它常常最好是使用一个面向页的技术. 
- kmalloc 和 _get_free_pages 返回的内存地址也是虚拟地址.

```c
get_zeroed_page(unsigned int flags);  //返回一个指向新页的指针并且用零填充了该页
__get_free_page(unsigned int flags);  //类似于 get_zeroed_page, 但是没有清零该页.
__get_free_pages(unsigned int flags, unsigned int order); //分配并返回一个指向一个内存区第一个字节的指针, 内存区可能是几个(物理上连续)页长但是没有清零.
void free_page(unsigned long addr); 
void free_pages(unsigned long addr, unsigned long order);
```

```c
/* Here's the allocation of a single quantum */ 
if (!dptr->data[s_pos]) 
{ 
    dptr->data[s_pos] =(void *)__get_free_pages(GFP_KERNEL, dptr->order); 
    if (!dptr->data[s_pos]) 
        goto nomem; 
    memset(dptr->data[s_pos], 0, PAGE_SIZE << dptr->order);
}

/* This code frees a whole quantum-set */ 
for (i = 0; i < qset; i++) 
    if (dptr->data[i]) 
        free_pages((unsigned long)(dptr->data[i]), dptr->order);
```

- alloc_pages 接口

```c
// nid 是要分配内存的 NUMA 节点 ID
// flags 是通常的 GFP_ 分配标志, 以及 order 是分配的大小
struct page *alloc_pages_node(int nid, unsigned int flags, 
                              unsigned int order);

void __free_page(struct page *page); 
void __free_pages(struct page *page, unsigned int order); 
void free_hot_page(struct page *page); 
void free_cold_page(struct page *page);
```

- vmalloc： 

-  调用 vmalloc 的正确时机是当你在为一个大的只存在于软件中的顺序缓冲分配内存时；
- vamlloc 比 __get_free_pages 有更多开销, 因为它必须获取内存并且建立页表

```c
//虚拟内存空间分配一块连续的内存区. 尽管这些页在物理内存中不连续
#include <linux/vmalloc.h> 
void *vmalloc(unsigned long size); 
void vfree(void * addr); 
void *ioremap(unsigned long offset, unsigned long size); 
void iounmap(void * addr);
```

```c
/* Allocate a quantum using virtual addresses */ 
if (!dptr->data[s_pos]) 
{ 
    dptr->data[s_pos] = 
        (void *)vmalloc(PAGE_SIZE << dptr->order); 
    if (!dptr->data[s_pos]) 
        goto nomem; 
    memset(dptr->data[s_pos], 0, PAGE_SIZE << dptr->order);
}
/* Release the quantum-set */ 
for (i = 0; i < qset; i++) 
    if (dptr->data[i]) 
        vfree(dptr->data[i]);
```

#### .4. 每一CPU变量

- 当你创建一个每-CPU 变量, 系统中每个处理器获得它自己的这个变量拷贝. 
- 内核维护无结尾的计数器来跟踪有每种报文类型有多少被接收; 

```c
DEFINE_PER_CPU(type, name);
DEFINE_PER_CPU(int[3], my_percpu_array);

get_cpu_var(sockets_in_use)++;   //get_cpu_var 宏来存取当前处理器的给定变量拷贝
put_cpu_var(sockets_in_use);  //调用 put_cpu_var. 对 get_cpu_var 的调用返回一个 lvalue 给当前处理器的变量版本并且禁止抢占

per_cpu(variable, int cpu_id); //存取另一个处理器的变量拷贝

void *alloc_percpu(type);   //动态分配每-CPU 变量
void *__alloc_percpu(size_t size, size_t align);
per_cpu_ptr(void *per_cpu_var, int cpu_id);  //宏返回一个指针指向 per_cpu_var 对应于给定 cpu_id 的版本

int cpu; 
cpu = get_cpu() 
ptr = per_cpu_ptr(per_cpu_var, cpu); 
/* work with ptr */ 
put_cpu();
```

#### .5. 获取大量缓冲

```c
#include <linux/bootmem.h> 
void *alloc_bootmem(unsigned long size); 
void *alloc_bootmem_low(unsigned long size); 
void *alloc_bootmem_pages(unsigned long size); 
void *alloc_bootmem_low_pages(unsigned long size);
```

### 8. 与硬件设备通讯

#### .1. IO寄存器和常用内存（内存屏障）

- I/O 寄存器和 RAM 的主要不同是 I/O 操作有边际效果, 而内存操作没有
- 一个内存写的唯一效果是存储一个值到一个位置, 并且一个内存读返回最近写到那里的值. 
- 编译器能够缓存数据值到 CPU 寄存器而不写到内存, 并且即便它存储它们, 读和写操作都能够在缓冲内存中进行而不接触物理 RAM
- 一个驱动必须确保`没有进行缓冲并且在存取寄存器时没有发生读或写的重编排`. 

```c
#include <linux/kernel.h> 
void barrier(void) 
//这个函数告知编译器插入一个内存屏障但是对硬件没有影响. 编译的代码将所有的当前改变的并且驻留在 CPU 寄存器的值存储到内存, 并且后来重新读取它们当需要时. 对屏障的调用阻止编译器跨越屏障的优化, 而留给硬件自由做它的重编排. 
#include <asm/system.h> 
void rmb(void); 
void read_barrier_depends(void); 
void wmb(void); 
void mb(void); 
//这些函数插入硬件内存屏障在编译的指令流中; 它们的实际实例是平台相关的. 一个 rmb ( read memory barrier) 保证任何出现于屏障前的读在执行任何后续读之前完成. wmb 保证写操作中的顺序, 并且 mb 指令都保证. 每个这些指令是一个屏障的超集.read_barrier_depends 是读屏障的一个特殊的, 弱些的形式. 而 rmb 阻止所有跨越屏障的读的重编排, read_barrier_depends 只阻止依赖来自其他读的数据的读的重编排.

void smp_rmb(void); 
void smp_read_barrier_depends(void); 
void smp_wmb(void); 
void smp_mb(void);


writel(dev->registers.addr, io_destination_address); 
writel(dev->registers.size, io_size); 
writel(dev->registers.operation, DEV_READ); 
wmb(); 
writel(dev->registers.control, DEV_GO);
```

#### .2. 使用IO端口

```c
//IO端口分配
#include <linux/ioport.h> 
struct resource *request_region(unsigned long first, unsigned long n, const 
                                char *name);
void release_region(unsigned long start, unsigned long n);
int check_region(unsigned long first, unsigned long n);

//操作IO端口，大部分硬件区别 8-位, 16-位, 和 32-位端口
unsigned inb(unsigned port); 
void outb(unsigned char byte, unsigned port); 
//读或写字节端口( 8 位宽 ). port 参数定义为 unsigned long 在某些平台以及 unsigned short 在其他的上. inb 的返回类型也是跨体系而不同的. 
unsigned inw(unsigned port); 
void outw(unsigned short word, unsigned port); 
//这些函数存取 16-位 端口( 一个字宽 ); 在为 S390 平台编译时它们不可用, 它只支持字节 I/O. 
unsigned inl(unsigned port); 
void outl(unsigned longword, unsigned port); 
//这些函数存取 32-位 端口. longword 声明为或者 unsigned long 或者 unsigned int, 根据平台. 如同字 I/O, "Long" I/O 在 S390 上不可用.
```

```c
void insb(unsigned port, void *addr, unsigned long count); 
void outsb(unsigned port, void *addr, unsigned long count); 
//读或写从内存地址 addr 开始的 count 字节. 数据读自或者写入单个 port 端口. 
void insw(unsigned port, void *addr, unsigned long count); 
void outsw(unsigned port, void *addr, unsigned long count); 
//读或写 16-位 值到一个单个 16-位 端口. 
void insl(unsigned port, void *addr, unsigned long count); 
void outsl(unsigned port, void *addr, unsigned long count); 
//读或写 32-位 值到一个单个 32-位 端口.
```

- 一个数字 I/O 端口, 在它的大部分的普通的化身中, 是一个字节宽的 I/O 位置, 或者内存映射的或者端口映射的

#### .3. 使用IO内存

- 从 ioremap 返回的地址不应当直接解引用,这样的使用不是可移植的; 相反, 应当使用内核提供的存取函数.
- IO 内存分配

```c
//I/O 内存区必须在使用前分配. 分配内存区的接口是( 在 <linux/ioport.h> 定义)
struct resource *request_mem_region(unsigned long start, unsigned long len, char *name);
void release_mem_region(unsigned long start, unsigned long len);

#include <asm/io.h> 
void *ioremap(unsigned long phys_addr, unsigned long size); 
void *ioremap_nocache(unsigned long phys_addr, unsigned long size); 
void iounmap(void * addr);
```

- 存取IO内存

```c

//从 I/O 内存读
unsigned int ioread8(void *addr); 
unsigned int ioread16(void *addr); 
unsigned int ioread32(void *addr);
//来写 I/O 内存
void iowrite8(u8 value, void *addr); 
void iowrite16(u16 value, void *addr); 
void iowrite32(u32 value, void *addr);

void ioread8_rep(void *addr, void *buf, unsigned long count); 
void ioread16_rep(void *addr, void *buf, unsigned long count);
void ioread32_rep(void *addr, void *buf, unsigned long count); 
void iowrite8_rep(void *addr, const void *buf, unsigned long count); 
void iowrite16_rep(void *addr, const void *buf, unsigned long count); 
void iowrite32_rep(void *addr, const void *buf, unsigned long count);

void memset_io(void *addr, u8 value, unsigned int count); 
void memcpy_fromio(void *dest, void *source, unsigned int count); 
void memcpy_toio(void *dest, void *source, unsigned int count);
```

- 作为IO内存端口

```c
void *ioport_map(unsigned long port, unsigned int count); //重映射 count I/O 端口和使它们出现为 I/O 内存
void ioport_unmap(void *addr);
```

```c
while (count--) { 
    iowrite8(*ptr++, address); 
    wmb(); 
}
```

### 9. 中断处理

-  一个中断不过是一个硬件在它需要处理器的注意时能够发出的信号
-  一个驱动只需要为它的设备中断注册一个处理函数, 并且当它们到来时正确处理它们

```c
int request_irq(unsigned int irq, 
                irqreturn_t (*handler)(int, void *, struct pt_regs *), 
                unsigned long flags, 
                const char *dev_name, 
                void *dev_id); 
void free_irq(unsigned int irq, void *dev_id);

if (short_irq >= 0) 
{
    result = request_irq(short_irq, short_interrupt, 
                         SA_INTERRUPT, "short", NULL); 
    if (result) { 
        printk(KERN_INFO "short: can't get assigned irq %i\n", 
               short_irq); 
        short_irq = -1; 
    } else { /* actually enable it -- assume this *is* a parallel port */ 
        outb(0x10,short_base+2); 
    } 
}
```

```c
static inline void short_incr_bp(volatile unsigned long *index, int delta) 
{ 
    unsigned long new = *index + delta; 
    barrier(); /* Don't optimize these two together */ 
    *index = (new >= (short_buffer + PAGE_SIZE)) ? short_buffer : new; 
}
```

```c
void disable_irq(int irq); 
void disable_irq_nosync(int irq); 
void enable_irq(int irq);

void local_irq_save(unsigned long flags); 
void local_irq_disable(void);
```

#### .1. 前和后半部

- 常常大量的工作必须响应一个设备中断来完成, 但是中断处理需要很快完成并且不使中断阻塞太长.
- 前半部保存设备数据到一个设备特定的缓存, 调度它的后半部, 并且退出: 这个操作非常快. 后半部接着进行任何其他需要的工作, 例如唤醒进程, 启动另一个 I/O 操作, 等等. 这种设置允许前半部来服务一个新中断而同时后半部仍然在工作。

```c
DECLARE_TASKLET(name, function, data); //name 是给 tasklet 的名子, function 是调用来执行 tasklet (它带一个 unsigned long 参数并且返回 void )的函数, 以及 data 是一个 unsigned long 值来传递给 tasklet 函数
void short_do_tasklet(unsigned long); 
DECLARE_TASKLET(short_tasklet, short_do_tasklet, 0);

irqreturn_t short_tl_interrupt(int irq, void *dev_id, struct pt_regs *regs) 
{ 
    do_gettimeofday((struct timeval *) tv_head); /* cast to stop 'volatile' warning */ 
    short_incr_tv(&tv_head); 
    tasklet_schedule(&short_tasklet); 
    short_wq_count++; /* record that an interrupt arrived */ 
    return IRQ_HANDLED; 
}
void short_do_tasklet (unsigned long unused) 
{ 
    int savecount = short_wq_count, written; 
    short_wq_count = 0; /* we have already been removed from the queue */
    /* 
 * The bottom half reads the tv array, filled by the top half, 
 * and prints it to the circular text buffer, which is then consumed 
 * by reading processes */ 
    /* First write the number of interrupts that occurred before this bh */ 
    written = sprintf((char *)short_head,"bh after %6i\n",savecount); 
    short_incr_bp(&short_head, written); 
    /* 
 * Then, write the time values. Write exactly 16 bytes at a time, 
 * so it aligns with PAGE_SIZE */ 
    do { 
        written = sprintf((char *)short_head,"%08u.%06u\n", 
                          (int)(tv_tail->tv_sec % 100000000), 
                          (int)(tv_tail->tv_usec)); 
        short_incr_bp(&short_head, written); 
        short_incr_tv(&tv_tail); 
    } while (tv_tail != tv_head); 
    wake_up_interruptible(&short_queue); /* awake any reading process */ 
}
```

```c
irqreturn_t short_wq_interrupt(int irq, void *dev_id, struct pt_regs *regs) 
{ 
    /* Grab the current time information. */
    do_gettimeofday((struct timeval *) tv_head); 
    short_incr_tv(&tv_head); 
    /* Queue the bh. Don't worry about multiple enqueueing */ 
    schedule_work(&short_wq); 
    short_wq_count++; /* record that an interrupt arrived */ 
    return IRQ_HANDLED; 
}
```

- 运行处理者

```c
irqreturn_t short_sh_interrupt(int irq, void *dev_id, struct pt_regs *regs) 
{ 
    int value, written; 
    struct timeval tv; 
    /* If it wasn't short, return immediately */ 
    value = inb(short_base); 
    if (!(value & 0x80)) 
        return IRQ_NONE; 
    /* clear the interrupting bit */ 
    outb(value & 0x7F, short_base); 
    /* the rest is unchanged */ 
    do_gettimeofday(&tv); 
    written = sprintf((char *)short_head,"%08u.%06u\n", 
                      (int)(tv.tv_sec % 100000000), (int)(tv.tv_usec)); 
    short_incr_bp(&short_head, written); 
    wake_up_interruptible(&short_queue); /* awake any reading process */ 
    return IRQ_HANDLED; 
}
```

### 10. 标准C类型使用&移植问题

- 不同平台上C语言基本类型大小不一致

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221112094746900.png)

- 时间间隔：不要假定每秒有 1000 个嘀哒. 尽管当前对 i386 体系是真实的, 不是每个 Linux 平台都以这个速度运行.
- 页大小： 记住一个内存页是 PAGE_SIZE 字节, 不是 4KB. 被支持的平台显示页大小从 4 KB 到 64 KB, 并且有时它们在相同平台上的不同的实现上不同

```c
#include <asm/page.h> 
int order = get_order(16*1024); 
buf = get_free_pages(GFP_KERNEL, order);
```

- 字节序： 依赖处理器的字节序

```c
u32 cpu_to_le32 (u32); 
u32 le32_to_cpu (u32);
```

- 数据对齐： 编写可移植代码而值得考虑的最后一个问题是如何存取不对齐的数据

```c
#include <asm/unaligned.h> 
get_unaligned(ptr); 
put_unaligned(val, ptr);
```

- 指针&错误值

```c
void *ERR_PTR(long error);  //个返回指针类型的函数可以返回一个错误值
long IS_ERR(const void *ptr); //IS_ERR 来测试是否一个返回的指针是不是一个错误码
```

- dpdk 里面无锁队列&环形链表  todo？

```c
#include <linux/list.h> 
list_add(struct list_head *new, struct list_head *head); 
list_add_tail(struct list_head *new, struct list_head *head); 
list_del(struct list_head *entry); 
list_del_init(struct list_head *entry); 
list_empty(struct list_head *head); 
list_entry(entry, type, member); 
list_move(struct list_head *entry, struct list_head *head); 
list_move_tail(struct list_head *entry, struct list_head *head); 
list_splice(struct list_head *list, struct list_head *head);

list_for_each(struct list_head *cursor, struct list_head *list) 
list_for_each_prev(struct list_head *cursor, struct list_head *list) 
list_for_each_safe(struct list_head *cursor, struct list_head *next, struct 
list_head *list) 
list_for_each_entry(type *cursor, struct list_head *list, member) 
list_for_each_entry_safe(type *cursor, type *next struct list_head *list, 
member)
```

### 11. PCI 驱动 todo



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/linux%E8%AE%BE%E5%A4%87%E9%A9%B1%E5%8A%A8/  

