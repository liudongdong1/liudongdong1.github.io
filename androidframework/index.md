# AndroidFramework


> Android底层内核空间以Linux Kernel作为基石，上层用户空间由`Native系统库`、`虚拟机运行环境`、`框架层组成`，通过`系统调用(Syscall)连通系统的内核空间与用户空间`。对于用户空间主要采用C++和Java代码编写，通过`JNI技术打通用户空间的Java层和Native层(C++/C)`，从而连通整个系统。

### 1. Android 系统框架

![image-20221120085004076](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221120085004076.png)



![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211107141336894.png)

> Android系统启动过程由上图从下往上的一个过程是由Boot Loader引导开机，然后依次进入 -> Kernel -> Native -> Framework -> App，接来下简要说说每个过程：

#### .1. Loader 层

- Boot ROM: 当手机处于关机状态时，长按Power键开机，引导芯片开始从固化在 ROM里的预设代码开始执行，然后加载引导程序到 RAM；
- Boot Loader：这是启动Android系统之前的引导程序，主要是检查RAM，初始化硬件参数等功能。

#### .2. Linux 内核层

- 启动Kernel的swapper进程(pid=0)：该进程又称为idle进程, 系统初始化过程Kernel由无到有开创的第一个进程, 用于`初始化进程管理、内存管理，加载Display,Camera Driver，Binder Driver等相关工作`；
- 启动kthreadd进程（pid=2）：是Linux系统的内核进程，会`创建内核工作线程kworkder`，`软中断线程ksoftirqd`，`thermal等内核守护进程`。 kthreadd进程是所有内核进程的鼻祖。

#### .3. 硬件抽象层

> 硬件抽象层 (HAL) 提供标准接口，HAL包含多个库模块，其中每个模块都为特定类型的硬件组件实现一组接口，比如WIFI/蓝牙模块，当框架API请求访问设备硬件时，Android系统将为该硬件加载相应的库模块。

#### .4. Android Runtime&系统库

每个应用都在其自己的进程中运行，都有自己的虚拟机实例。`ART通过执行DEX文件可在设备运行多个虚拟机`，`DEX文件是一种专为Android设计的字节码格式文件，经过优化，使用内存很少`。ART主要功能包括：`预先(AOT)和即时(JIT)编译，优化的垃圾回收(GC)，以及调试相关的支持`。

DX 工具将.class文件编译成.dex文件，运行该.dex文件。

这里的`Native系统库`主要包括`init孵化来的用户空间的守护进程、HAL层以及开机动画等`。启动init进程(pid=1),是Linux系统的用户进程，` init进程是所有用户进程的鼻祖`。

- init进程会孵化出ueventd、logd、healthd、installd、adbd、lmkd等用户守护进程；
- init进程还启动 servicemanager(binder服务管家)、 bootanim(开机动画)等重要服务
- init进程孵化出Zygote进程，Zygote进程是Android系统的第一个Java进程(即虚拟机进程)， Zygote是所有Java进程的父进程，Zygote进程本身是由init进程孵化而来的。

#### .5. Framework层

- Zygote进程，是由init进程通过解析init.rc文件后fork生成的，Zygote进程主要包含：
  - 加载ZygoteInit类，注册Zygote Socket服务端套接字
  - 加载虚拟机
  - 提前加载类preloadClasses
  - 提前加载资源preloadResouces
- System Server进程，是由Zygote进程fork而来， SystemServer是Zygote孵化的第一个进程，System Server负责启动和管理整个Java framework，包含ActivityManager，WindowManager，PackageManager，PowerManager等服务。
- Media Server进程，是由init进程fork而来，负责启动和管理整个C++**framework，包含AudioFlinger，Camera Service等服务。**

#### .6. App 层

- Zygote进程孵化出的第一个App进程是`Launcher，这是用户看到的桌面App`；
- Zygote进程还会`创建Browser，Phone，Email等App进程，每个App至少运行在一个进程上`。
- 所有的`App进程都是由Zygote进程fork生成的`。

#### .7. Syscall && JNI

- Native与Kernel之间有一层系统调用(SysCall)层，见Linux系统调用(Syscall)原理;
- `Java层与Native(C/C++)层之间的纽带JNI`，见Android JNI原理分析。

### 2. android [镜像文件](https://www.cnblogs.com/schips/p/introduction_of_image_about_android.html)

- `cache.img（缓存镜像）`：用于存储系统或用户应用产生的临时数据。
- `vendor.img`：包含所有不可分发给 Android 开源项目 (AOSP) 的二进制文件。如果没有专有信息，则可以省略此分区。
- `misc.img`：misc 分区供恢复映像使用，存储空间不能小于 4KB。
- `userdata.img`：userdata 分区包含用户安装的应用和数据，包括自定义数据。
- `vbmeta.img`：用于安全验证，bootloader验证vbmeta的签名，再用vbmeta的key以及hash值验证dtbo/boot/system/vendor。
- `system.img（系统镜像）`：系统镜像是地址ROM最常使用的一个镜像，用于存储Android系统的核心文件，System.img就是设备中system目录的镜像，里面包含了Android系统主要的目录和文件。一般这些文件是不允许修改的。
- `userdata.img（用户数据镜像）`：将会被挂接到 /data 下，包含了所有应用相关的配置文件，以及用户相关的数据 。
- system.img、userdata.img、vendor.img、persist.img都是sparse压缩文件系统镜像，目的是方便传输/刷机/存储等。
- `recovery.img`： recovery分区的镜像，一般用作系统恢复（刷机）。
- boot.img（Linux内核镜像）： Android系统中，通常会把zImage （ 内核镜像uImage文件） 和ramdisk.img打包到一起，生成一个boot.img镜像文件，放到boot分区，由bootloader来引导启动，其启动过程本质也是和分开的uImage&ramdisk.img类似，只不过把两个镜像按照一定的格式合并为一个镜像而已。
- `amdisk.img（内存磁盘镜像）`是根文件系统：android启动时 首先加载ramdisk.img镜像，并挂载到/目录下，并进行了一系列的初始化动作，包括创建各种需要的目录，初始化console，开启服务等，尽管ramdisk.img需要放在Linux内核镜像（boot.img）中，但却属于Android源代码的一部分。

### 3. Android mk 文件

- https://www.jianshu.com/p/703ef39dff3f

### 4. 通信方式

> 对于IPC(Inter-Process Communication, 进程间通信)，Linux现有`管道`、[消息队列](https://cloud.tencent.com/product/cmq?from=10680)、`共享内存`、`套接字`、`信号量`、`信号`这些IPC机制，Android额外还有`Binder IPC机制`，Android OS中的Zygote进程的IPC采用的是Socket机制，在上层system server、media server以及上层App之间更多的是采用Binder IPC方式来完成跨进程间的通信。对于Android上层架构中，很多时候是在同一个进程的线程之间需要相互通信，例如`同一个进程的主线程与工作线程之间的通信，往往采用的Handler消息机制`。
>
> 1. **管道：**在创建时分配一个page大小的内存，缓存区大小比较有限；
>
> 2. **消息队列**：信息复制两次，额外的CPU消耗；不合适频繁或信息量大的通信；
>
> 3. **共享内存**：无须复制，共享缓冲区直接付附加到进程虚拟地址空间，速度快；但进程间的同步问题操作系统无法实现，必须各进程利用同步工具解决；
>
> 4. **套接字**：作为更通用的接口，传输效率低，主要用于不通机器或跨网络的通信；
>
> 5. **信号量**：常作为一种锁机制，防止某进程正在访问共享资源时，其他进程也访问该资源。因此，主要作为进程间以及同一进程内不同线程之间的同步手段。
>
> 6. **信号**: 不适用于信息交换，更适用于进程中断控制，比如非法内存访问，杀死某个进程等；

#### 1. [共享内存方式](https://blog.csdn.net/Aliven888/article/details/119248596)

> 1、mmap保存到实际硬盘，实际存储并没有反映到主存上。优点：储存量可以很大（多于主存）（这里一个问题，需要高手解答,会不会太多拷贝到主存里面？？？）；缺点：进程间读取和写入速度要比主存的要慢。
>
> 2、shm保存到物理存储器（主存），实际的储存量直接反映到主存上。优点，进程间访问速度（读写）比磁盘要快；缺点，储存量不能非常大（多于主存）
>
> 使用上看：如果分配的存储量不大，那么使用shm；如果存储量大，那么使用mmap。
>
> 2）使用特殊文件提供匿名内存映射：适用于具有亲缘关系的进程之间；由于父子进程特殊的亲缘关系，在父进程中先调用mmap()，然后调用fork()。那么在调用fork()之后，子进程继承父进程匿名映射后的地址空间，同样也继承mmap()返回的地址，这样，父子进程就可以通过映射区域进行通信了。

- `创建共享内存`，使用shmget函数。
- `映射共享内存`，将这段创建的共享内存映射到具体的进程空间去，使用shmat函数。

- 当一个进程想和另外一个进程通信的时候，它将按以下顺序运行：
  - 获取mutex对象，锁定共享区域。
  - 将要通信的数据写入共享区域。
  - 释放mutex对象。
- 当一个进程从从这个区域读数据时候，它将重复同样的步骤，只是将第二步变成读取。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221120110134019.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221120110146858.png)

```
usage : ipcs -asmq -tclup 
    ipcs [-s -m -q] -i id
    ipcs -h for help.
m      输出有关共享内存(shared memory)的信息
-q      输出有关信息队列(message queue)的信息
-s      输出有关“遮断器”(semaphore)的信息

usage: ipcrm [ [-q msqid] [-m shmid] [-s semid]
          [-Q msgkey] [-M shmkey] [-S semkey] ... ]
```

##### 1. 映射共享内存

```c
#include <stdio.h>
#include <stdlib.h>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>


#define IPCKEY 0x366378



typedef struct st_setting
{
    char agen[10];
    unsigned char file_no;
}st_setting;

int main(int argc, char** argv)
{
    int         shm_id;
    //key_t       key;
    st_setting  *p_setting;

    //  首先检查共享内存是否存在，存在则先删除
    shm_id = shmget(IPCKEY , 1028, 0640);
    if(shm_id != -1)
    {
        p_setting = (st_setting *)shmat(shm_id, NULL, 0);

        if (p_setting != (void *)-1)
        {
            shmdt(p_setting);

            shmctl(shm_id,IPC_RMID,0) ;
        }
    }

    //  创建共享内存
    shm_id = shmget(IPCKEY, 1028, 0640 | IPC_CREAT | IPC_EXCL);
    if(shm_id == -1)
    {
        printf("shmget error\n");
        return -1;
    }

    //  将这块共享内存区附加到自己的内存段
    p_setting = (st_setting *)shmat(shm_id, NULL, 0);

    strncpy(p_setting->agen, "gatieme", 10);
    printf("agen : %s\n", p_setting->agen);

    p_setting->file_no = 1;
    printf("file_no : %d\n",p_setting->file_no);

    system("ipcs -m");//  此时可看到有进程关联到共享内存的信息，nattch为1

    //  将这块共享内存区从自己的内存段删除出去
    if(shmdt(p_setting) == -1)
       perror(" detach error ");

    system("ipcs -m");//  此时可看到有进程关联到共享内存的信息，nattch为0

    //  删除共享内存
    if (shmctl( shm_id , IPC_RMID , NULL ) == -1)
    {
        perror(" delete error ");
    }

    system("ipcs -m");//  此时可看到有进程关联到共享内存的信息，nattch为0


    return EXIT_SUCCESS;
}
```

##### 2. [俩个进程读写案例](https://blog.csdn.net/Aliven888/article/details/119248596)

```c
//shmDatadef.h
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <iostream>
using namespace std;
#define SHARE_MEMORY_BUFFER_LEN 1024
struct stuShareMemory{
	int iSignal;
	char chBuffer[SHARE_MEMORY_BUFFER_LEN];
	
	stuShareMemory(){
		iSignal = 0;
		memset(chBuffer,0,SHARE_MEMORY_BUFFER_LEN);
	}
};

//writeShareMemory.cpp
#include "shmDatadef.h"
int main(int argc, char* argv[])
{
	void *shm = NULL;
	struct stuShareMemory *stu = NULL;
	int shmid = shmget((key_t)1234, sizeof(struct stuShareMemory), 0666|IPC_CREAT);
	if(shmid == -1)
	{
		printf("shmget err.\n");
		return 0;
	}
	shm = shmat(shmid, (void*)0, 0);
	if(shm == (void*)-1)
	{
		printf("shmat err.\n");
		return 0;
	}

	stu = (struct stuShareMemory*)shm;

	stu->iSignal = 0;

	//while(true) //如果需要多次 读取 可以启用 while
	{
		if(stu->iSignal != 1)
		{
			printf("write txt to shm.");
			memcpy(stu->chBuffer, "hello world 666 - 888.\n", 30);
			stu->iSignal = 1;
		}
		else
		{
			sleep(10);
		}
	}
	
	shmdt(shm);

	std::cout << "end progress." << endl;
	return 0;
}

//readShareMemory.cpp

#include "shmDatadef.h"

int main(int argc, char* argv[])
{
	void *shm = NULL;
	struct stuShareMemory *stu;
	int shmid = shmget((key_t)1234, sizeof(struct stuShareMemory), 0666|IPC_CREAT);
	if(shmid == -1)
	{
		printf("shmget err.\n");
		return 0;
	}

	shm = shmat(shmid, (void*)0, 0);
	if(shm == (void*)-1)
	{
		printf("shmat err.\n");
		return 0;
	}

	stu = (struct stuShareMemory*)shm;

	stu->iSignal = 1;

	//while(true)  //如果需要多次写入，可以启用while
	{
		if(stu->iSignal != 0)
		{
			printf("current txt : %s", stu->chBuffer);
			stu->iSignal = 0;
		}
		else
		{
			sleep(10);
		}
	}
	
	shmdt(shm);
	shmctl(shmid, IPC_RMID, 0);

	std::cout << "end progress." << endl;
	return 0;
}
```

#### 2. 管道

- 管道是单向的，管道有容量，当满了之后，读写会发送阻塞

```c
#include<iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include<string.h>
#include <signal.h>
#define FIFO  "/home/whb/projects/11.fifo" //有名管道的名字

using namespace std;
int main()
{
	int result , fd;//result 为接收mkfifo的返回值   fd为打开文件的文件描述符
	char buffer[20] = { 0 };//定义一个字符数据
	if (access(FIFO, F_OK)==-1)//判断是否已经创建了有名管道，如果已经创建，则返回0 否则返回非0的数
	{
		result = mkfifo(FIFO, 0777);//创建有名管道,成功返回0,失败返回-1
		if (result < 0)
		{
			perror("creat mkfifo error");
			return 0;
		}
	}
	cout << "请输入数据：" << endl;
	//以只写方式打开有名管道，不能同时以读写权限打开,成功返回文件描述符，失败返回 - 1
	fd = open(FIFO,O_WRONLY);
	if (fd < 0)
	{
		perror("open error");
		return 0;
	}
	while (1)
	{
		fgets(buffer, sizeof(buffer), stdin);//从终端输入数据到buffer中
		write(fd, buffer, strlen(buffer));//数据写到有名管道
		memset(buffer, 0x0, sizeof(buffer));//清空缓存区
	}
	close(fd);//关闭有名管道
	return 0;
}
```

```c
#include<iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include<string.h>
#include <signal.h>
#define FIFO  "/home/whb/projects/11.fifo"//有名管道的名字
using namespace std;

int main()
{
	int result ,fd;//result 接收mkfifo 的返回值  fd 文件描述符
	char buffer[20] = { 0 };
	if (access(FIFO, F_OK)==-1)//判断有名管道是否存在  不存在就创建  存在就不创建   已经创建返回0  否则非0
	{
		result = mkfifo(FIFO, 0777);//创建有名管道  0 成功   -1 失败
		if (result < 0)
		{
			perror("creat mkfifo error");
			return 0;
		}
	}
	fd = open(FIFO, O_RDONLY);//只读的方式打开  返回小于0的值为打开失败
	if (fd < 0)
	{
		perror("open error");
		return 0;
	}
	while (1)
	{
		read(fd, buffer, sizeof(buffer));
		cout << "buffer= " << buffer << endl;
		memset(buffer,0,sizeof(buffer));
	}
	close(fd);//关闭有名管道
	return 0;
}
```

#### 3. Unix Domain Socket 案例

```c
#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<sys/un.h>
#include<errno.h>

//define send and recv buf size
#define BUFSIZE 512*1024

//define unix domain socket path
#define pmmanager "/tmp/pmmanager"
#define pmapi "/tmp/pmapi"

int main(int argc, char** argv)
{
    char rx_buf[BUFSIZE];
    int pmmanager_fd, ret;
    socklen_t len;
    struct sockaddr_un pmmanager_addr, pmapi_addr;

    //create pmmanager socket fd
    pmmanager_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if(pmmanager_fd == -1)
    {
        perror("cannot create pmmanager fd.");
    }

    unlink(pmmanager);
    memset(&pmmanager_addr, 0, sizeof(pmmanager_addr));
    pmmanager_addr.sun_family = AF_UNIX;
    strncpy(pmmanager_addr.sun_path, pmmanager, sizeof(pmmanager_addr.sun_path)-1);

    //bind pmmanager_fd to pmmanager_addr
    ret = bind(pmmanager_fd, (struct sockaddr*)&pmmanager_addr, sizeof(pmmanager_addr));
    if(ret == -1)
    {
        perror("can not bind pmmanager_addr");
    }

    int recvBufSize;
    len = sizeof(recvBufSize);
    ret = getsockopt(pmmanager_fd, SOL_SOCKET, SO_RCVBUF, &recvBufSize, &len);
    if(ret ==-1)
    {
        perror("getsocket error.");
    }
    printf("Before setsockopt, SO_RCVBUF-%d\n",recvBufSize); 
    recvBufSize = 512*1024;
    ret = setsockopt(pmmanager_fd, SOL_SOCKET, SO_RCVBUF, &recvBufSize, len);
    if(ret == -1)
    {
        perror("setsockopt error.");
    }
    ret = getsockopt(pmmanager_fd, SOL_SOCKET, SO_RCVBUF, &recvBufSize, &len);
    if(ret ==-1)
    {
        perror("getsocket error.");
    }
    printf("Set recv buf successful, SO_RCVBUF-%d\n",recvBufSize); 

    int recvSize;
    memset(&pmapi_addr, 0, sizeof(pmapi_addr));
    len = sizeof(pmapi_addr);
    printf("==============wait for msg from pmapi====================\n");
    for(;;)
    {
        memset(rx_buf, 0, sizeof(rx_buf));
        recvSize = recvfrom(pmmanager_fd, rx_buf, sizeof(rx_buf), 0, (struct sockaddr*)&pmapi_addr, &len);
        if(recvSize == -1)
        {
            perror("recvfrom error.");
        }
        printf("Recved message from pmapi: %s\n", rx_buf);
    }
}
```

```c
#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<sys/un.h>
#include<errno.h>

//define send and recv buf size
#define BUFSIZE 250*1024

//define unix domain socket path
#define pmmanager "/tmp/pmmanager"
#define pmapi "/tmp/pmapi"

int main(int argc, char** argv)
{
    char tx_buf[BUFSIZE];
    int pmapi_fd, ret;
    socklen_t len;
    struct sockaddr_un pmmanager_addr, pmapi_addr;

    //create pmmanager socket fd
    pmapi_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if(pmapi_fd == -1)
    {
        perror("cannot create pmapi fd.");
    }

    unlink(pmapi);
    //configure pmapi's addr
    memset(&pmapi_addr, 0, sizeof(pmapi_addr));
    pmapi_addr.sun_family = AF_UNIX;
    strncpy(pmapi_addr.sun_path, pmapi, sizeof(pmapi_addr.sun_path)-1);
    //bind pmapi_fd to pmapi_addr
    ret = bind(pmapi_fd, (struct sockaddr*)&pmapi_addr, sizeof(pmapi_addr));
    if(ret == -1)
    {
        perror("bind error.");
    }

    int sendBufSize;
    len = sizeof(sendBufSize);
    ret = getsockopt(pmapi_fd, SOL_SOCKET, SO_SNDBUF, &sendBufSize, &len);
    if(ret ==-1)
    {
        perror("getsocket error.");
    }
    printf("Before setsockopt, SO_SNDBUF-%d\n",sendBufSize); 
    sendBufSize = 512*1024;
    ret = setsockopt(pmapi_fd, SOL_SOCKET, SO_SNDBUF, &sendBufSize, len);
    if(ret == -1)
    {
        perror("setsockopt error.");
    }
    ret = getsockopt(pmapi_fd, SOL_SOCKET, SO_SNDBUF, &sendBufSize, &len);
    if(ret ==-1)
    {
        perror("getsocket error.");
    }
    printf("Set send buf successful, SO_SNDBUF-%d\n\n\n", sendBufSize); 

    //configure pmmanager's addr
    memset(&pmmanager_addr, 0, sizeof(pmmanager_addr));
    pmmanager_addr.sun_family = AF_UNIX;
    strncpy(pmmanager_addr.sun_path, pmmanager, sizeof(pmmanager_addr)-1);
    len = sizeof(pmmanager_addr);

    int sendSize = 0;
    int i;
    for(i=1; i<=4; i++)
    {
        memset(tx_buf, '0', sizeof(tx_buf));
        sprintf(tx_buf, "send msg %d to pmmanager.", i);
        printf("%s, msg size - %d\n",tx_buf, sizeof(tx_buf));
        sendSize = sendto(pmapi_fd, tx_buf, sizeof(tx_buf), 0, (struct sockaddr*)&pmmanager_addr, len);
        if(sendSize == -1)
        {
            perror("sendto error.");
        }
        printf("Send message to pmmanager: %s\n\n\n", tx_buf);
    }
}
```

#### 4.mmap

```c
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>

void sys_err(char *str)
{
    perror(str);
    exit(1);
}

int main(void)
{
    char *mem = NULL;
    int len = 0;

    int fd = open("hello244", O_RDWR|O_CREAT|O_TRUNC, 0644);
    if (fd < 0){
		sys_err("open error");
	}
        
    
	/*
	注意：由于mmap的参2是开辟文件大小的映射区，完美open时是一个新文件，
	此时大小为0，所以必须保证文件大小充足才能成功调用mmap开辟映射区。
	这就是下面两个方法的原因。
	方法1
    lseek(fd, 19, SEEK_END);	  //将文件下标移至19个字节
    write(fd, "\0", 1);			  //往文件写0，补充结尾。19+1=20个字节
	len = lseek(fd, 0, SEEK_END); //利用lseek的返回值获取当前文件下标即文件大小
	printf("The length of file = %d\n", len);
    */
	
	//方法2
	ftruncate(fd, 20);//该函数和lseek+write作用一样
	len = lseek(fd, 0, SEEK_END);
	
    mem = mmap(NULL, len, PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED){
		sys_err("mmap err: ");
	} 
	//此时可以直接关闭描述符，因为mem已经代替其作用
    close(fd);
	
	//使用mem(内存)对磁盘文件进行读写操作。
    strcpy(mem, "hello mmap");//写
    printf("%s\n", mem);//读

    if (munmap(mem, mem) < 0){
		sys_err("munmap");
	}

    return 0;
}
```

#### 4. RPC 机制

- 学习具体框架MecuryRPC 框架深入学习

#### 5. 同步互斥

##### 1. 信号量（Semaphore) & Mutex

- **Mutex**是一把钥匙，一个人拿了就可进入一个房间，出来的时候把钥匙交给队列的第一个。一般的用法是用于串行化对critical section代码的访问，保证这段代码不会被并行的运行。
- **Semaphore**是一件可以容纳N人的房间，如果人不满就可以进去，如果人满了，就要等待有人出来。对于N=1的情况，称为binary semaphore。一般的用法是，用于限制对于某一资源的同时访问。

##### 2. Monitor

##### 3. Condition

```c++
class Condition{
public:
    enum{
        PRIVATE=0;
        SHARED=1;
    };
    
    status_t wait(Mutex& mutex);
    status_t waitRelative(Mutex& mutex, nsecs_t reltime);
    void signal(); //条件满足时通知相应等待者
    void broadcast(); //条件满足时通知所有等待者
private:
    #if defined {HAVE_PTHREADS}
    	pthread_cond_t mCond;
    #else
    	void* mState;
    #endif
};
```

##### 4. Barrier

```c++
class Barrier{
public:
    inline Barrier():state(CLOSED){}
    inline ~Barrier(){}
    void open(){
        Mutex:: Autolock _l(lock);
        state=OPENED;
        cv.broadcast();
    }
    void close(){
        Mutex:: Autolock _l(lock);
        state=CLOSED;
    }
    void wait() const{
        Mutex:: Autolock _l(lock);
        while(state==CLOSED){
            cv.wait(lock);
        }
    }
private:
    enum {OPENED,CLOSED};
    mutable Mutex lock;
    mutable Condition cv;
    volatile int state;
}
```



```c++
class Autolock{
public:
    inline Autolock(Mutex& mutex): mLock(mutex){mLock.lock();}
    inline AUtolock(Mutex* mutex): mLock(*mutex){mLock.lock();}
    inline ~Autolock(){mLock.unlock();}
private:
    Mutex& mLock;
};
```

### 5. Android 进程线程理解

- activity 和 service 的主线程是 ActivityThread.java, 

```java
public static void main(String[] args) {
        Trace.traceBegin(Trace.TRACE_TAG_ACTIVITY_MANAGER, "ActivityThreadMain");

        // Install selective syscall interception
        AndroidOs.install();

        // CloseGuard defaults to true and can be quite spammy.  We
        // disable it here, but selectively enable it later (via
        // StrictMode) on debug builds, but using DropBox, not logs.
        CloseGuard.setEnabled(false);

        Environment.initForCurrentUser();

        // Make sure TrustedCertificateStore looks in the right place for CA certificates
        final File configDir = Environment.getUserConfigDirectory(UserHandle.myUserId());
        TrustedCertificateStore.setDefaultUserDirectory(configDir);

        // Call per-process mainline module initialization.
        initializeMainlineModules();

        Process.setArgV0("<pre-initialized>");

        Looper.prepareMainLooper();

        // Find the value for {@link #PROC_START_SEQ_IDENT} if provided on the command line.
        // It will be in the format "seq=114"
        long startSeq = 0;
        if (args != null) {
            for (int i = args.length - 1; i >= 0; --i) {
                if (args[i] != null && args[i].startsWith(PROC_START_SEQ_IDENT)) {
                    startSeq = Long.parseLong(
                            args[i].substring(PROC_START_SEQ_IDENT.length()));
                }
            }
        }
        ActivityThread thread = new ActivityThread();
        thread.attach(false, startSeq);

        if (sMainThreadHandler == null) {
            sMainThreadHandler = thread.getHandler();
        }

        if (false) {
            Looper.myLooper().setMessageLogging(new
                    LogPrinter(Log.DEBUG, "ActivityThread"));
        }

        // End of event ActivityThreadMain.
        Trace.traceEnd(Trace.TRACE_TAG_ACTIVITY_MANAGER);
        Looper.loop();

        throw new RuntimeException("Main thread loop unexpectedly exited");
    }
```

- 同一个程序包中包含俩个 Activity， 他们关系：运行在同一个进程中，可以访问静态变量， 主线程只有一个，切换时Binder数量有变化
- 不同包的组件可以通过android:process 属性表明组件运行在哪一个进程空间中。
- 一个Activity启动后至少会有3个线程，及一个主线程&俩个Binder线程。
- 四大组件 activity，service，receiver，provider只是Application零件。

#### .0. Handler, Looper, Message, MessageQueue 类关系

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221128104919571.png)

#### .1. **Binder**

> Binder通信采用c/s架构，从组件视角来说，包含`Client、Server、ServiceManager以及binder驱动`，其中ServiceManager用于管理系统中的各种服务。Service Manager是指Native层的ServiceManager（C++），并非指framework层的ServiceManager(Java)。ServiceManager是整个Binder通信机制的大管家，是Android进程间通信机制Binder的守护进程，要掌握Binder机制，首先需要了解系统是如何首次[启动Service Manager](http://gityuan.com/2015/11/07/binder-start-sm/)。当Service Manager启动之后，Client端和Server端通信时都需要先[获取Service Manager](http://gityuan.com/2015/11/08/binder-get-sm/)接口，才能开始通信服务。
>
> - binder dirver会将自己注册成一个misc device，并向上层提供一个/dev/binder节点，该驱动运行与内核态，提供open，ioctl，mmap操作
>
> 1. **[注册服务(addService)](http://gityuan.com/2015/11/14/binder-add-service/)**：Server进程要先注册Service到ServiceManager。该过程：Server是客户端，ServiceManager是服务端。
> 2. **[获取服务(getService)](http://gityuan.com/2015/11/15/binder-get-service/)**：Client进程使用某个Service前，须先向ServiceManager中获取相应的Service。该过程：Client是客户端，ServiceManager是服务端。
> 3. **使用服务**：Client根据得到的Service信息建立与Service所在的Server进程通信的通路，然后就可以直接与Service交互。该过程：client是客户端，server是服务端。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221122110712185.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221128123857494.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221123154311314.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221122101702736.png)

- 以后自己理解代码调用过程的时候，借鉴这里的分析流程，做好说明
- Service Manager 不断读取消息的循环中处理客户端请求

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221128151112016.png)

> `每个Android的进程，只能运行在自己进程所拥有的虚拟地址空间`。对应一个4GB的虚拟地址空间，其中3GB是用户空间，1GB是内核空间，当然内核空间的大小是可以通过参数配置调整的。对于`用户空间，不同进程之间彼此是不能共享的，而内核空间却是可共享的`。Client进程向Server进程通信，恰恰是`利用进程间可共享的内核内存空间来完成底层通信工作的`，Client端与Server端进程往往采用ioctl等方法跟内核空间的驱动进行交互。

#### .2. Socket

Socket通信方式也是C/S架构，比Binder简单很多。在Android系统中采用Socket通信方式的主要有：

- zygote：用于孵化进程，`system_server创建进程是通过socket向zygote进程发起请求`；
- installd：用于`安装App的守护进程`，上层PackageManagerService很多实现最终都是交给它来完成；
- lmkd：lowmemorykiller的守护进程，Java层的LowMemoryKiller最终都是由lmkd来完成；
- adbd：这个也不用说，用于`服务adb`；
- logcatd:这个不用说，用于`服务logcat`；
- vold：即volume Daemon，是存储类的守护进程，用于负责如USB、Sdcard等存储设备的事件处理。

#### .3. Handler

> `Binder/Socket用于进程间通信`，而``Handler消息机制用于同进程的线程间通信`(`只能用于共享内存地址空间的两个线程间通信)，Handler消息机制是由一组`MessageQueue、Message、Looper、Handler共同组成的`，为了方便且称之为Handler消息机制。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211107143740784.png)

> 由于工作线程与主线程共享地址空间，即Handler实例对象mHandler位于线程间共享的内存堆上，工作线程与主线程都能直接使用该对象，只需要`注意多线程的同步问题`。工作线程通过mHandler向其成员变量MessageQueue中添加新Message，主线程一直处于loop()方法内，当收到新的Message时按照一定规则分发给相应的handleMessage()方法来处理。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211107144203557.png)

Low Memory Killer

- 进行资源释放回收

Anonymous Shared Memory（代码解析87页）

- 将指定的物理内存分别映射到各自的虚拟地址空间中，从而便捷的实现进程间内存共享
- 类似binder可以看作时一个共享设备
- 设备节点什么时候创建
- 提供了哪些函数操作，以及函数实现原理
- 与Linux 内存共享机制有什么区别
- 代码解析： https://redspider110.github.io/2018/01/17/0043-android-ashmem/

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221120153232402.png)

### 6. GUI显示

#### .1. SurfaceFlinger

- SurfaceFlinger 需要手机系统中所有的应用程序绘制的图像数据，然后集中显示到物理屏幕上
- OpenGL ES 通过ANativeWindow 来与本地窗口系统建立正确的连接。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/Center.png)

1） 这里的FrameBuffer指显示设备驱动和Gralloc帧缓冲区管理

2） 面向SurfaceFlinger的[Native](https://so.csdn.net/so/search?q=Native&spm=1001.2101.3001.7020) Window

3） 通过[OpenGl](https://so.csdn.net/so/search?q=OpenGl&spm=1001.2101.3001.7020) ES图形库来处理图形数据后绘制到NativeWindow

4） SurfaceFlinger，是一个binderservice，用于管理接收各个App传输过来的图形数据

5） 面向App应用窗口绘制的Native Window

6） 采用OpenGL ES或者SKIA将图形数据绘制到Native Window，对于普通应用开发人员来说，使用OpenGLES的门槛相对会比较高，所以SKIA第三方[图形库](https://so.csdn.net/so/search?q=图形库&spm=1001.2101.3001.7020)基于OpenGL ES做了封装，提供更加简单的GUI接口供开发人员使用，SKIA是Android应用默认的图形引擎

#### .2. WindowManageService(WMS)

![image-20221129092634957](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221129092634957.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221129093908589.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20221129094135012.png)

### Resource

- 学习链接： https://cloud.tencent.com/developer/article/1415759


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/androidframework/  

