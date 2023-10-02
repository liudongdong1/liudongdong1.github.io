# Android_启动流程


![](https://gitee.com/github-25970295/blogimgv2022/raw/master/webp-166919226600621.webp)

### 1. Bootloader

- U-boot启动流程
  - 第一阶段：汇编代码：U-boot的第一条指令从cpu/armXXX/start.S文件开始
  - 第二阶段：C代码：从文件/lib_arm/board.c的start_armboot()函数开始。

| Bootloader | Monitor？ |                                           描述 |  X86 |  ARM | PowerPC |
| ---------- | :-------: | ---------------------------------------------: | ---: | ---: | ------: |
| U-boot     |    是     |                                   通用引导程序 |   是 |   是 |      是 |
| ReBoot     |    是     |                           是基于eCos的引导程序 |   是 |   是 |      是 |
| BLOB       |    否     | (StrongARM架构) LART(主板)等硬件平台的引导程序 |   否 |   是 |      否 |
| LILO       |    否     |                              Linux磁盘引导程序 |   是 |   否 |      否 |
| GRUB       |    否     |                              GNU的LILO替代程序 |   是 |   否 |      否 |
| Loadlin    |    否     |                                 从DOS引导Linux |   是 |   否 |      否 |
| Vivi       |    是     |                   韩国mizi公司开发的bootloader |   否 |   是 |      否 |

### 2. Init.cpp

init进程是Android系统启动的第一个进程。它通过解析init.rc脚本来构建出系统的初始形态。

- http://androidxref.com/6.0.1_r10/xref/system/core/init/init.cpp  ， [解析](https://www.jianshu.com/p/464c3d1203b1)
- android6 代码， android 10 不太一样

```cpp
int main(int argc, char** argv) {

// ****************** 第一部分 ****************** 
// 检查启动程序的文件名

    if (!strcmp(basename(argv[0]), "ueventd")) {
        return ueventd_main(argc, argv);
    }

    if (!strcmp(basename(argv[0]), "watchdogd")) {
        return watchdogd_main(argc, argv);
    }

// ****************** 第二部分 ****************** 
// 设置文件属性为0777
    // Clear the umask.
    umask(0);

// ****************** 第三部分 ****************** 
// 设置环境变量
    add_environment("PATH", _PATH_DEFPATH);

// ****************** 第四部分 ****************** 
// 创建一些基本目录，并挂载

    //判断是否是第一次
    bool is_first_stage = (argc == 1) || (strcmp(argv[1], "--second-stage") != 0);

    // Get the basic filesystem setup we need put together in the initramdisk
    // on / and then we'll let the rc file figure out the rest
    //如果是第一次.
    if (is_first_stage) {
        mount("tmpfs", "/dev", "tmpfs", MS_NOSUID, "mode=0755");
        mkdir("/dev/pts", 0755);
        mkdir("/dev/socket", 0755);
        mount("devpts", "/dev/pts", "devpts", 0, NULL);
        mount("proc", "/proc", "proc", 0, NULL);
        mount("sysfs", "/sys", "sysfs", 0, NULL);
    }


// ****************** 第五部分 ****************** 
// 把标准输入、标准输出和标准错误重定向到空设备文件"/dev/_null_"

    // We must have some place other than / to create the device nodes for
    // kmsg and null, otherwise we won't be able to remount / read-only
    // later on. Now that tmpfs is mounted on /dev, we can actually talk
    // to the outside world.
    open_devnull_stdio();


// ****************** 第六部分 ****************** 
// 启动kernel log
    klog_init();
    klog_set_level(KLOG_NOTICE_LEVEL);

    // 输出init启动阶段的log      
    NOTICE("init%s started!\n", is_first_stage ? "" : " second stage");


// ****************** 第七部分 ****************** 
// 设置系统属性
    if (!is_first_stage) {
        // Indicate that booting is in progress to background fw loaders, etc.
// 7.1 创建初始化标志
        close(open("/dev/.booting", O_WRONLY | O_CREAT | O_CLOEXEC, 0000));

//7.2 初始化Android的属性系统
        property_init();

        // If arguments are passed both on the command line and in DT,
        // properties set in DT always have priority over the command-line ones.
//7.3  解析DT和命令行中的kernel启动参数
        process_kernel_dt();
        process_kernel_cmdline();

        // Propogate the kernel variables to internal variables
        // used by init as well as the current required properties.
//7.4  设置系统属性
        export_kernel_boot_props();
    }

// ****************** 第八部分 ****************** 

    // Set up SELinux, including loading the SELinux policy if we're in the kernel domain.
    // 调用selinux_initialize函数启动SELinux
    selinux_initialize(is_first_stage);

    // If we're in the kernel domain, re-exec init to transition to the init domain now
    // that the SELinux policy has been loaded.
    if (is_first_stage) {

        // 按照selinux policy要求，重新设置init文件属性
        if (restorecon("/init") == -1) {
            ERROR("restorecon failed: %s\n", strerror(errno));
            security_failure();
        }
        char* path = argv[0];

        // 设置参数  --second-stage
        char* args[] = { path, const_cast<char*>("--second-stage"), nullptr };

        // 执行init进程，重新进入main函数
        if (execv(path, args) == -1) {
            ERROR("execv(\"%s\") failed: %s\n", path, strerror(errno));
            security_failure();
        }
    }

    // These directories were necessarily created before initial policy load
    // and therefore need their security context restored to the proper value.
    // This must happen before /dev is populated by ueventd.
    INFO("Running restorecon...\n");
    restorecon("/dev");
    restorecon("/dev/socket");
    restorecon("/dev/__properties__");
    restorecon_recursive("/sys");

// ****************** 第九部分 ****************** 
    epoll_fd = epoll_create1(EPOLL_CLOEXEC);  //调用epoll_create1创建epoll句柄
    if (epoll_fd == -1) {
        ERROR("epoll_create1 failed: %s\n", strerror(errno));
        exit(1);
    }
    signal_handler_init();  //调用signal_handler_init()函数，主要是装载进程信号处理器
    //主要是当子进程被kill之后，会在父进程接受一个信号。处理这个信号的时候往sockpair一段写数据，而另一端的fd是加入epoll中

// ****************** 第十部分 ****************** 

    property_load_boot_defaults();
    start_property_service();

// ****************** 第十一部分 ****************** 
// 重点部分，我们后面用专门用一篇文章讲解
    init_parse_config_file("/init.rc");

// ****************** 第十二部分 ****************** 

    action_for_each_trigger("early-init", action_add_queue_tail);

    // Queue an action that waits for coldboot done so we know ueventd has set up all of /dev...
    queue_builtin_action(wait_for_coldboot_done_action, "wait_for_coldboot_done");
    // ... so that we can start queuing up actions that require stuff from /dev.
    queue_builtin_action(mix_hwrng_into_linux_rng_action, "mix_hwrng_into_linux_rng");
    queue_builtin_action(keychord_init_action, "keychord_init");
    queue_builtin_action(console_init_action, "console_init");

    // Trigger all the boot actions to get us started.
    action_for_each_trigger("init", action_add_queue_tail);

    // Repeat mix_hwrng_into_linux_rng in case /dev/hw_random or /dev/random
    // wasn't ready immediately after wait_for_coldboot_done
    queue_builtin_action(mix_hwrng_into_linux_rng_action, "mix_hwrng_into_linux_rng");

    // Don't mount filesystems or start core system services in charger mode.
    char bootmode[PROP_VALUE_MAX];
    if (property_get("ro.bootmode", bootmode) > 0 && strcmp(bootmode, "charger") == 0) {
        action_for_each_trigger("charger", action_add_queue_tail);
    } else {
        action_for_each_trigger("late-init", action_add_queue_tail);
    }

    // Run all property triggers based on current state of the properties.
    queue_builtin_action(queue_property_triggers_action, "queue_property_triggers");

// ****************** 第十三部分 ****************** 
    while (true) {
        if (!waiting_for_exec) {
            execute_one_command();
            restart_processes();
        }

        int timeout = -1;
        if (process_needs_restart) {
            timeout = (process_needs_restart - gettime()) * 1000;
            if (timeout < 0)
                timeout = 0;
        }

        if (!action_queue_empty() || cur_action) {
            timeout = 0;
        }

        bootchart_sample(&timeout);

        epoll_event ev;
        int nr = TEMP_FAILURE_RETRY(epoll_wait(epoll_fd, &ev, 1, timeout));
        if (nr == -1) {
            ERROR("epoll_wait failed: %s\n", strerror(errno));
        } else if (nr == 1) {
            ((void (*)()) ev.data.ptr)();
        }
    }
    return 0;
}
```

- signal_handler_init()  todo？  具体实现原理

> 每个进程在处理其他进程发送的signal信号时都需要先注册，当进程的运行状态改变或终止时会产生某种signal信号，init进程是所有用户空间进程的父进程，当其子进程终止时产生SIGCHLD信号，init进程调用信号安装函数sigaction()，传递参数给sigaction结构体，便完成信号处理的过程。
>
> 当init进程调用signal_handler_init后，一旦受到子进程终止带来的SIGCHLD消息后，将利用信号处理者SIGCHLD_handler向signal_write_fd写入信息；epoll句柄监听到signal_read_fd收到消息后，将调用handle_signal进行处理。

```cpp
void signal_handler_init() {

    // 在Linux中，父进程是通过捕捉SIGCHILD信号来得知子进程运行结束的情况
    // Create a signalling mechanism for SIGCHLD.
    int s[2];

    // 利用socketpair创建出已经连接的两个socket，分别作为信号的读、写端
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0, s) == -1) {
        ERROR("socketpair failed: %s\n", strerror(errno));
        exit(1);
    }

    signal_write_fd = s[0];
    signal_read_fd = s[1];

    // Write to signal_write_fd if we catch SIGCHLD.
    struct sigaction act;
    memset(&act, 0, sizeof(act));

    // 信号处理器为SIGCHLD_handler，其被存在sigaction结构体重，负责处理SIGCHLD消息
    
    // 信号处理器
    act.sa_handler = SIGCHLD_handler;

     // 仅当进程终止时才接受+
    act.sa_flags = SA_NOCLDSTOP;

     // 调用信号安装函数sigaction，将监听的信号及对应的信号处理器注册到内核中
    sigaction(SIGCHLD, &act, 0);

    reap_any_outstanding_children();

    // 定义在system/core/init/init.cpp中，注册epoll handler，当signal_read_fd 有数据可读时，调用handle_signal
    register_epoll_handler(signal_read_fd, handle_signal);
}
```

#### 2. init.rc 文件

- http://androidxref.com/6.0.1_r10/xref/system/core/rootdir/init.rc
- https://www.jianshu.com/p/cb73a88b0eed  解析

### 3. [zygote](https://www.jianshu.com/p/4e5909d24d65)

>Linux的进程是通过系统调用fork产生的，fork出的子进程除了内核中的一些核心的数据结构和父进程不同之外，其余的内存映像都是和父进程共享的。只有当子进程需要去改写这些共享的内存时，操作系统才会为子进程分配一个新的页面，并将老的页面上的数据复制一份到新的页面，这就是所谓的"写拷贝"。
>
>- Zygote创建应用程序时却只使用了fork，没有调用exec。Android应用中执行的是Java代码，Java代码的不同才造成了应用的区别，而对于运行Java的环境，要求却是一样的。
>- Zygote初始化时会创建创建虚拟机，同时把需要的系统类库和资源文件加载到内存里面。Zygote fork出子进程后，这个子进程也继承了能正常工作的虚拟机和各类系统资源，接下来子进程只需要装载APK文件的字节码文件就可以运行了。这样应用程序的启动时间就会大大缩短。

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/webp-166919953084124.webp)

- 具体代码解析  todo？ 没有看懂

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/5713484-27464477f2175cbb.png)

#### .2. AndroidRuntime

- 负责启动虚拟机以及Java线程。AndroidRuntime类是在一个进程中只有一个实例对象，并将其保存在全局变量gCurRuntime中。

```java
AndroidRuntime::AndroidRuntime(char* argBlockStart, const size_t argBlockLength) :
        mExitWithoutCleanup(false),
        mArgBlockStart(argBlockStart),
        mArgBlockLength(argBlockLength)
{
    SkGraphics::Init();
    // There is also a global font cache, but its budget is specified by
    // SK_DEFAULT_FONT_CACHE_COUNT_LIMIT and SK_DEFAULT_FONT_CACHE_LIMIT.

    // Pre-allocate enough space to hold a fair number of options.    
    mOptions.setCapacity(20);
    

    assert(gCurRuntime == NULL);        // one per process
    gCurRuntime = this;
}
```

- **SkGraphics::Init()**: 这里主要是初始化skia图形系统。skia是google的第一个底层的图形、图像、动画、SVG、文本等多方面的图形图，是Android图形系统的引擎。skia作为第三方软件放在external目录下： external/skia/。后面附了一个skia结构图
- **mOptions.setCapacity(20);**：预先分配空间来存放传入虚拟机的参数
- **gCurRuntime = this;**：首先通过的断言判断gCurRuntime是否为空，保证只能被初始化一次

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/webp-166920030983629.webp)

```cpp
/*
 * Start the Android runtime.  This involves starting the virtual machine
 * and calling the "static void main(String[] args)" method in the class
 * named by "className".
 *
 * Passes the main function two arguments, the class name and the specified
 * options string.
 */
void AndroidRuntime::start(const char* className, const Vector<String8>& options, bool zygote)
{
    //******************* 第一部分**********************
    ALOGD(">>>>>> START %s uid %d <<<<<<\n",
            className != NULL ? className : "(unknown)", getuid());

    static const String8 startSystemServer("start-system-server");

    /*
     * 'startSystemServer == true' means runtime is obsolete and not run from
     * init.rc anymore, so we print out the boot start event here.
     */
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i] == startSystemServer) {
           /* track our progress through the boot sequence */
           const int LOG_BOOT_PROGRESS_START = 3000;
           LOG_EVENT_LONG(LOG_BOOT_PROGRESS_START,  ns2ms(systemTime(SYSTEM_TIME_MONOTONIC)));
        }
    }

     //******************* 第二部分**********************
    const char* rootDir = getenv("ANDROID_ROOT");
    if (rootDir == NULL) {
        rootDir = "/system";
        if (!hasDir("/system")) {
            LOG_FATAL("No root directory specified, and /android does not exist.");
            return;
        }
        setenv("ANDROID_ROOT", rootDir, 1);
    }

    //const char* kernelHack = getenv("LD_ASSUME_KERNEL");
    //ALOGD("Found LD_ASSUME_KERNEL='%s'\n", kernelHack);


    //******************* 第三部分**********************
    /* start the virtual machine */
    JniInvocation jni_invocation;
    jni_invocation.Init(NULL);
    JNIEnv* env;
    if (startVm(&mJavaVM, &env, zygote) != 0) {
        return;
    }

    //******************* 第四部分**********************
    onVmCreated(env);


    //******************* 第五部分**********************
    /*
     * Register android functions.
     */
    if (startReg(env) < 0) {
        ALOGE("Unable to register all android natives\n");
        return;
    }

   //******************* 第六部分**********************
    /*
     * We want to call main() with a String array with arguments in it.
     * At present we have two arguments, the class name and an option string.
     * Create an array to hold them.
     */
    jclass stringClass;
    jobjectArray strArray;
    jstring classNameStr;

    stringClass = env->FindClass("java/lang/String");
    assert(stringClass != NULL);
    strArray = env->NewObjectArray(options.size() + 1, stringClass, NULL);
    assert(strArray != NULL);
    classNameStr = env->NewStringUTF(className);
    assert(classNameStr != NULL);
    env->SetObjectArrayElement(strArray, 0, classNameStr);

    for (size_t i = 0; i < options.size(); ++i) {
        jstring optionsStr = env->NewStringUTF(options.itemAt(i).string());
        assert(optionsStr != NULL);
        env->SetObjectArrayElement(strArray, i + 1, optionsStr);
    }


   //******************* 第七部分**********************
    /*
     * Start VM.  This thread becomes the main thread of the VM, and will
     * not return until the VM exits.
     */
    char* slashClassName = toSlashClassName(className);
    jclass startClass = env->FindClass(slashClassName);
    if (startClass == NULL) {
        ALOGE("JavaVM unable to locate class '%s'\n", slashClassName);
        /* keep going */
    } else {
        jmethodID startMeth = env->GetStaticMethodID(startClass, "main",
            "([Ljava/lang/String;)V");
        if (startMeth == NULL) {
            ALOGE("JavaVM unable to find main() in '%s'\n", className);
            /* keep going */
        } else {
            env->CallStaticVoidMethod(startClass, startMeth, strArray);

#if 0
            if (env->ExceptionCheck())
                threadExitUncaughtException(env);
#endif
        }
    }
    free(slashClassName);

    ALOGD("Shutting down VM\n");
    if (mJavaVM->DetachCurrentThread() != JNI_OK)
        ALOGW("Warning: unable to detach main thread\n");
    if (mJavaVM->DestroyJavaVM() != 0)
        ALOGW("Warning: VM did not shut down cleanly\n");
}

```

#### 3. Java 层 ZygoteInit

```java
public static void main(String argv[]) {
    try {

        //**************** 第一阶段 **********************

        // 启动DDMS
        RuntimeInit.enableDdms();
        // Start profiling the zygote initialization.

        // 启动性能统计 
        SamplingProfilerIntegration.start();

        boolean startSystemServer = false;
        String socketName = "zygote";
        String abiList = null;
        for (int i = 1; i < argv.length; i++) {
            if ("start-system-server".equals(argv[i])) {
                startSystemServer = true;
            } else if (argv[i].startsWith(ABI_LIST_ARG)) {
                abiList = argv[i].substring(ABI_LIST_ARG.length());
            } else if (argv[i].startsWith(SOCKET_NAME_ARG)) {
                socketName = argv[i].substring(SOCKET_NAME_ARG.length());
            } else {
                throw new RuntimeException("Unknown command line argument: " + argv[i]);
            }
        }

        if (abiList == null) {
            throw new RuntimeException("No ABI list supplied.");
        }

        //**************** 第二阶段 **********************
        registerZygoteSocket(socketName);
        EventLog.writeEvent(LOG_BOOT_PROGRESS_PRELOAD_START,
                            SystemClock.uptimeMillis());

        //**************** 第三阶段 **********************
        preload();  //系统预加载类、Framework资源和openGL的资源
        EventLog.writeEvent(LOG_BOOT_PROGRESS_PRELOAD_END,
                            SystemClock.uptimeMillis());

        // Finish profiling the zygote initialization.
        SamplingProfilerIntegration.writeZygoteSnapshot();

        // Do an initial gc to clean up after startup
        gcAndFinalize();

        // Disable tracing so that forked processes do not inherit stale tracing tags from
        // Zygote.
        Trace.setTracingEnabled(false);

        //**************** 第四阶段 **********************
        if (startSystemServer) {
            startSystemServer(abiList, socketName);
        }

        Log.i(TAG, "Accepting command socket connections");


        //**************** 第五阶段 **********************
        runSelectLoop(abiList);

        closeServerSocket();
    } catch (MethodAndArgsCaller caller) {
        caller.run();
    } catch (RuntimeException ex) {
        Log.e(TAG, "Zygote died with exception", ex);
        closeServerSocket();
        throw ex;
    }
}
```

##### runSelectLoop

```java
654    /**
655     * Runs the zygote process's select loop. Accepts new connections as
656     * they happen, and reads commands from connections one spawn-request's
657     * worth at a time.
658     *
659     * @throws MethodAndArgsCaller in a child process when a main() should
660     * be executed.
661     */
662    private static void runSelectLoop(String abiList) throws MethodAndArgsCaller {
663        ArrayList<FileDescriptor> fds = new ArrayList<FileDescriptor>();
664        ArrayList<ZygoteConnection> peers = new ArrayList<ZygoteConnection>();
665
          //fds[0]为sServerSocket，即sServerSocket为位于zygote进程中的socket服务端
666        fds.add(sServerSocket.getFileDescriptor());
667        peers.add(null);
668
669        while (true) {
//************************** 第1部分   ************************** 
670            StructPollfd[] pollFds = new StructPollfd[fds.size()];
671            for (int i = 0; i < pollFds.length; ++i) {
672                pollFds[i] = new StructPollfd();
                   // pollFds[0].fd即为sServerSocket，位于zygote进程中的socket服务端。
673                pollFds[i].fd = fds.get(i);
674                pollFds[i].events = (short) POLLIN;
675            }
676            try {
                   // 查询轮训状态，当pollFdd有事件到来则往下执行，否则阻塞在这里
677                Os.poll(pollFds, -1);
678            } catch (ErrnoException ex) {
679                throw new RuntimeException("poll failed", ex);
680            }
681            for (int i = pollFds.length - 1; i >= 0; --i) {
                 // 采用I/O 多路复用机制，当接受到客户端发出的连接请求，或者处理出具时，则往下执行
                 // 否则进入continue，跳出本次循环 
682                if ((pollFds[i].revents & POLLIN) == 0) {
683                    continue;
684                }
//************************** 第2部分   **************************
685                if (i == 0) {
                      // 客户端第一次请求服务端，服务端调用accept与客户端建立连接，客户端在zygote以ZygoteConnection对象表示
686                    ZygoteConnection newPeer = acceptCommandPeer(abiList);
687                    peers.add(newPeer);
688                    fds.add(newPeer.getFileDesciptor());
689                } else {
//*************************** 第3部分   **************************
                      // 经过上个if操作后，客户端与服务端已经建立连接，并开始发送数据
                      //peers.get(index)取得发送数据客户端的ZygoteConnection对象
                      // 然后调用runOnce()方法来出具具体请求
690                    boolean done = peers.get(i).runOnce();
691                    if (done) {
692                        peers.remove(i);
                           // 处理完则从fds中移除该文件描述符
693                        fds.remove(i);
694                    }
695                }
696            }
697        }
698    }
```

### 4. SystemService

```java
176    private void run() {
177        // If a device's clock is before 1970 (before 0), a lot of
178        // APIs crash dealing with negative numbers, notably
179        // java.io.File#setLastModified, so instead we fake it and
180        // hope that time from cell towers or NTP fixes it shortly.
           // 计算时间 如果当前系统时间比1970年更早，就设置当前系统时间为1970年
181        if (System.currentTimeMillis() < EARLIEST_SUPPORTED_TIME) {
182            Slog.w(TAG, "System clock is before 1970; setting to 1970.");
183            SystemClock.setCurrentTimeMillis(EARLIEST_SUPPORTED_TIME);
184        }
185
186        // If the system has "persist.sys.language" and friends set, replace them with
187        // "persist.sys.locale". Note that the default locale at this point is calculated
188        // using the "-Duser.locale" command line flag. That flag is usually populated by
189        // AndroidRuntime using the same set of system properties, but only the system_server
190        // and system apps are allowed to set them.
191        //
192        // NOTE: Most changes made here will need an equivalent change to
193        // core/jni/AndroidRuntime.cpp
           // 如果没有设置 语言，则设置当地的语言
194        if (!SystemProperties.get("persist.sys.language").isEmpty()) {
195            final String languageTag = Locale.getDefault().toLanguageTag();
196
197            SystemProperties.set("persist.sys.locale", languageTag);
198            SystemProperties.set("persist.sys.language", "");
199            SystemProperties.set("persist.sys.country", "");
200            SystemProperties.set("persist.sys.localevar", "");
201        }
202
203        // Here we go!
204        Slog.i(TAG, "Entered the Android system server!");
205        EventLog.writeEvent(EventLogTags.BOOT_PROGRESS_SYSTEM_RUN, SystemClock.uptimeMillis());
206
207        // In case the runtime switched since last boot (such as when
208        // the old runtime was removed in an OTA), set the system
209        // property so that it is in sync. We can't do this in
210        // libnativehelper's JniInvocation::Init code where we already
211        // had to fallback to a different runtime because it is
212        // running as root and we need to be the system user to set
213        // the property. http://b/11463182

            // 设置虚拟机库文件路径，5.0以后是libart.so
214        SystemProperties.set("persist.sys.dalvik.vm.lib.2", VMRuntime.getRuntime().vmLibrary());
215
216        // Enable the sampling profiler.
           // 如果开启了性能分析标志，则开启性能分析
217        if (SamplingProfilerIntegration.isEnabled()) {
218            SamplingProfilerIntegration.start();
219            mProfilerSnapshotTimer = new Timer();
220            mProfilerSnapshotTimer.schedule(new TimerTask() {
221                @Override
222                public void run() {
223                    SamplingProfilerIntegration.writeSnapshot("system_server", null);
224                }
225            }, SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL);
226        }
227
228        // Mmmmmm... more memory!
          // 清楚VM内存增长上线，由于启动过程需要较多的虚拟机内存空间
229        VMRuntime.getRuntime().clearGrowthLimit();
230
231        // The system server has to run all of the time, so it needs to be
232        // as efficient as possible with its memory usage.
           // 设置内存可能有效使用率为0.8
233        VMRuntime.getRuntime().setTargetHeapUtilization(0.8f);
234
235        // Some devices rely on runtime fingerprint generation, so make sure
236        // we've defined it before booting further.
           // 针对部分设备依赖运行时就产生指纹信息，因此需要在开机完成前已经定义
237        Build.ensureFingerprintProperty();
238
239        // Within the system server, it is an error to access Environment paths without
240        // explicitly specifying a user.

           // 设置访问环境变量的条件，即需要明确指定用户
241        Environment.setUserRequired(true);
242
243        // Ensure binder calls into the system always run at foreground priority.
           //确保当前系统进程的binder调用，总是运行在前台优先级(foreground)
244        BinderInternal.disableBackgroundScheduling(true);
245
246        // Prepare the main looper thread (this thread).
247        android.os.Process.setThreadPriority(
248                android.os.Process.THREAD_PRIORITY_FOREGROUND);
249        android.os.Process.setCanSelfBackground(false);
           // 主线程Looper就在当前线程运行
250        Looper.prepareMainLooper();
251
           // 加载“android_servers.so”库，该库包含源码在frameworks/base/services/目录下
252        // Initialize native services.
253        System.loadLibrary("android_servers");
254
255        // Check whether we failed to shut down last time we tried.
256        // This call may not return.
           //检查上次关键是否失败了，可能没有返回值
257        performPendingShutdown();
258
259        // Initialize the system context.
           // 初始化系统上下文
260        createSystemContext();
261
262        // Create the system service manager.

           // 创建SystemServiceManager 用于后面的binder机制
263        mSystemServiceManager = new SystemServiceManager(mSystemContext);
264        LocalServices.addService(SystemServiceManager.class, mSystemServiceManager);
265
266        // Start services.
           //启动各种系统服务
267        try {
268            startBootstrapServices();
269            startCoreServices();
270            startOtherServices();
271        } catch (Throwable ex) {
272            Slog.e("System", "******************************************");
273            Slog.e("System", "************ Failure starting system services", ex);
274            throw ex;
275        }
276
277        // For debug builds, log event loop stalls to dropbox for analysis.
           // 如果是debug版本，为了方便分析，将log事件不断循环地输出到dropbox
278        if (StrictMode.conditionallyEnableDebugLogging()) {
279            Slog.i(TAG, "Enabled StrictMode for system server main thread.");
280        }
281
282        // Loop forever.
           // 主进程的looper开启死循环
283        Looper.loop();
284        throw new RuntimeException("Main thread loop unexpectedly exited");
285    }
```

#### createSystemContext

```java
09    private void createSystemContext() {
              // 获取ActivityThread对象
310        ActivityThread activityThread = ActivityThread.systemMain();
              // 获取系统的Context
311        mSystemContext = activityThread.getSystemContext();
           // 设置主题
312        mSystemContext.setTheme(android.R.style.Theme_DeviceDefault_Light_DarkActionBar);
313    }
```

#### addService

```java
49    /**
50     * Adds a service instance of the specified interface to the global registry of local services.
51     */
52    public static <T> void addService(Class<T> type, T service) {
53        synchronized (sLocalServiceObjects) {
54            if (sLocalServiceObjects.containsKey(type)) {
55                throw new IllegalStateException("Overriding service registration");
56            }
57            sLocalServiceObjects.put(type, service);
58        }
59    }
```

#### startBootstrapServices

- ActivityManagerService, PowerManagerService, LightsService, DisplayManagerService, PackageManagerService, UserManagerService, sensor服务

```java
    /**
316     * Starts the small tangle of critical services that are needed to get
317     * the system off the ground.  These services have complex mutual dependencies
318     * which is why we initialize them all in one place here.  Unless your service
319     * is also entwined in these dependencies, it should be initialized in one of
320     * the other functions.
321     */
322    private void startBootstrapServices() {
323        // Wait for installd to finish starting up so that it has a chance to
324        // create critical directories such as /data/user with the appropriate
325        // permissions.  We need this to complete before we initialize other services.
           // 阻塞等待与installd建立socket通道
326        Installer installer = mSystemServiceManager.startService(Installer.class);
327
328        // Activity manager runs the show.
           //创建AMS(ActivityManagerService)，并启动
329        mActivityManagerService = mSystemServiceManager.startService(
330                ActivityManagerService.Lifecycle.class).getService();
331        mActivityManagerService.setSystemServiceManager(mSystemServiceManager);
332        mActivityManagerService.setInstaller(installer);
333
334        // Power manager needs to be started early because other services need it.
335        // Native daemons may be watching for it to be registered so it must be ready
336        // to handle incoming binder calls immediately (including being able to verify
337        // the permissions for those calls).
           // 启动电源管理服务，即PowerManagerService
338        mPowerManagerService = mSystemServiceManager.startService(PowerManagerService.class);
339
340        // Now that the power manager has been started, let the activity manager
341        // initialize power management features.
           // mActivityManagerService初始化，并在其中初始化PowerManager
342        mActivityManagerService.initPowerManagement();
343
344        // Manages LEDs and display backlight so we need it to bring up the display.
           // 开启服务LightsService，即灯光服务
345        mSystemServiceManager.startService(LightsService.class);
346
347        // Display manager is needed to provide display metrics before package manager
348        // starts up.
           // 开启服务DisplayManagerService，显示服务
349        mDisplayManagerService = mSystemServiceManager.startService(DisplayManagerService.class);
350
351        // We need the default display before we can initialize the package manager.
           // 在初始化package manager之前，需要默认的显示
352        mSystemServiceManager.startBootPhase(SystemService.PHASE_WAIT_FOR_DEFAULT_DISPLAY);
353
354        // Only run "core" apps if we're encrypting the device.
           // 根据加密设备状态，决定mOnlyCore的值
355        String cryptState = SystemProperties.get("vold.decrypt");
356        if (ENCRYPTING_STATE.equals(cryptState)) {
357            Slog.w(TAG, "Detected encryption in progress - only parsing core apps");
358            mOnlyCore = true;
359        } else if (ENCRYPTED_STATE.equals(cryptState)) {
360            Slog.w(TAG, "Device encrypted - only parsing core apps");
361            mOnlyCore = true;
362        }
363
364        // Start the package manager.
365        Slog.i(TAG, "Package Manager");
           // 启动服务PackageManagerService 即包管理
366        mPackageManagerService = PackageManagerService.main(mSystemContext, installer,
367                mFactoryTestMode != FactoryTest.FACTORY_TEST_OFF, mOnlyCore);
368        mFirstBoot = mPackageManagerService.isFirstBoot();
369        mPackageManager = mSystemContext.getPackageManager();
370
371        Slog.i(TAG, "User Service");
           // 启动UserManagerService，即用户服务，新建目录“/data/user/”
372        ServiceManager.addService(Context.USER_SERVICE, UserManagerService.getInstance());
373
374        // Initialize attribute cache used to cache resources from packages.
375        AttributeCache.init(mSystemContext);
376
377        // Set up the Application instance for the system process and get started.
           // 设置AMS，这样SystemServer进程可以加入到AMS中，冰杯它管理。
378        mActivityManagerService.setSystemProcess();
379
380        // The sensor service needs access to package manager service, app ops
381        // service, and permissions service, therefore we start it after them.
//启动传感器服务
             //开启传感器服务
382        startSensorService();
383    } 
```

#### startCoreServices

```java
385    /**
386     * Starts some essential services that are not tangled up in the bootstrap process.
387     */
388    private void startCoreServices() {
389        // Tracks the battery level.  Requires LightService.
          // 启动服务BatteryService，用于统计电池量量
390        mSystemServiceManager.startService(BatteryService.class);
391
392        // Tracks application usage stats.
           // 启动服务UsageStatsService，用于统计应用使用情况
393        mSystemServiceManager.startService(UsageStatsService.class);
394        mActivityManagerService.setUsageStatsManager(
395                LocalServices.getService(UsageStatsManagerInternal.class));
396        // Update after UsageStatsService is available, needed before performBootDexOpt.
397        mPackageManagerService.getUsageStatsIfNoPackageUsageInfo();
398
399        // Tracks whether the updatable WebView is in a ready state and watches for update installs.
           //启动服务WebViewUpdateService
400        mSystemServiceManager.startService(WebViewUpdateService.class);
401    }
```

#### [startOtherServices](https://www.jianshu.com/p/327f583f970b#54576e41-16ed-b257-c092-01ad9e36c010)

```java
403    /**
404     * Starts a miscellaneous grab bag of stuff that has yet to be refactored
405     * and organized.
406     */
407    private void startOtherServices() {
408        final Context context = mSystemContext;
409        AccountManagerService accountManager = null;
410        ContentService contentService = null;
411        VibratorService vibrator = null;
412        IAlarmManager alarm = null;
413        IMountService mountService = null;
414        NetworkManagementService networkManagement = null;
415        NetworkStatsService networkStats = null;
416        NetworkPolicyManagerService networkPolicy = null;
417        ConnectivityService connectivity = null;
418        NetworkScoreService networkScore = null;
419        NsdService serviceDiscovery= null;
420        WindowManagerService wm = null;
421        UsbService usb = null;
422        SerialService serial = null;
423        NetworkTimeUpdateService networkTimeUpdater = null;
424        CommonTimeManagementService commonTimeMgmtService = null;
425        InputManagerService inputManager = null;
426        TelephonyRegistry telephonyRegistry = null;
427        ConsumerIrService consumerIr = null;
428        AudioService audioService = null;
429        MmsServiceBroker mmsService = null;
430        EntropyMixer entropyMixer = null;
431        CameraService cameraService = null;
535        StatusBarManagerService statusBar = null;
536        INotificationManager notification = null;
537        InputMethodManagerService imm = null;
538        WallpaperManagerService wallpaper = null;
539        LocationManagerService location = null;
540        CountryDetectorService countryDetector = null;
541        TextServicesManagerService tsms = null;
542        LockSettingsService lockSettings = null;
543        AssetAtlasService atlas = null;
544        MediaRouterService mediaRouter = null;  
}
```



### Resource

- https://www.jianshu.com/p/4e5909d24d65
- https://www.jianshu.com/p/4e5909d24d65#66fcc52f-96f9-9ad9-d07b-e7721cdd08af  zyogte 介绍 todo？没太懂多少
- https://www.jianshu.com/p/327f583f970b#54576e41-16ed-b257-c092-01ad9e36c010 SystemService 服务相关todo

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android-%E5%90%AF%E5%8A%A8%E6%B5%81%E7%A8%8B/  

