# FlashMethods


> 单片机的烧录方式主要可以分为三种，分别为ICP(在电路编程)、IAP(在应用编程)以及ISP(在系统编程)。
>
> - ICP(In Circuit Programing)在电路编程
> - ISP(In System Programing)在系统编程
> - IAP(In applicating Programing)在应用编程

### 3. ICP(In Circuit Programing)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120193334284.png)

> **ICP编程就是以SWD接口**进行的。执行ICP功能，仅需要3个引脚RESET、ICPDA及 ICPCK。`RESET用于进入或退出ICP模式`，ICPDA为数据输入输出脚`，`ICPCK为编程时钟输入脚`。用户需要在系统板上预留VDD、GND以及这三个脚。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120193524020.png)

#### 烧录方式

> **ICP使用SWD接口进行烧录程序**。常用的烧录工具为J-Link、ST-Link、Nu-Link。与之配套的烧录软件为J-Flash、NuMicro_ICP_Programming_Tool、st-link utility。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120193910303.png)

### 2. **ISP（In System Programing）**

> ISP是指“在系统上编程”，目标芯片使用`USB/UART/SPI/I²C/RS-485/CAN`**周边接口的LDROM引导代码**去更新晶片内部APROM、数据闪存(DataFlash)和用户配置字(Config)。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120193730367.png)

#### 烧录方式

> ISP是使用**引导程序通过USB/UART等接口进行烧录**的，首先就是需要有BoodLoad程序。最常见的烧录方式就是学习8051单片机时使用的STC-ISP烧录工具了。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120193947504.png)

#### Arduino ICSP

**USBASP烧录软件**： [progisp 1.72](https://pan.baidu.com/s/1xbX7V6qABuUMg0nO3t07Ag)

**USBASP驱动安装软件**：[zadig 2.4](https://pan.baidu.com/s/1eFpTzZU7ERWW3_B7WalFXA)

选择编程器： USBasp；

打开“项目”-选择“编译”-**“导出已编译的二进制文件"**，编译完成后，就可以获得HEX文件。

![接线](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120195419448.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120195203228.png)

> error: 烧录报错；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210120201126976.png)

### 3. **IAP（In applicating Programing）**

> **通过软件实现在线电擦除和编程的方法**。IAP技术是从结构上将Flash存储器映射为两个存储体，当运行一个存储体上的用户程序时，可对另一个存储体重新编程，之后将程序从一个存储体转向另一个。

> **软件自身实现在线电擦除和编程的方法，不使用任何工具。程序通常分成两块，分别为引导程序和应用程序。**

### 4. 学习教程

- https://blog.csdn.net/zeevia/article/details/103033321
- https://blog.csdn.net/sysjtlwx/article/details/73824903

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/flashmethods/  

