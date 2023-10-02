# Arduino Learning Record


> Arduino 是一块基于开放原始代码的 Simple i/o 平台，并且具有开发语言和开发 环境都很简单、易理解的特点。让您可以快速使用 Arduino 做出有趣的东西。 它是一个能够用来感应和控制现实物理世界的一套工具。 它由一个基于单片机并且开 放源码的硬件平台，和一套为 Arduino 板编写程序 的开发环境组成。 Arduino 可以用来开发交互产品，比如它可以读取大量的开关和传感器信号，并且可以 控制各式各样的电灯、电机和其他物理设备。Arduino 项目可以是单独的，也可以在运行时 和你电脑中运行的程序（例如：Flash，Processing，MaxMSP）进行通讯。Arduino 开源的 IDE 可以免费下载得到。

## 1. Arduino 介绍

### 1.1. 性能描述

-  Digital I/O 数字输入/输出端口 0—13。
-  Analog I/O 模拟输入/输出端口 0-5。 `注释： 这里模拟输入的具体数值为电路中 从电压零开始计算的电压值，及默认模拟引脚的另一端直接接地`
-  支持 ISP 下载功能。 
-  输入电压：接上 USB 时无须外部供电或外部 5V~9V 直流电压输入。 
-  输出电压：5V 直流电压输出和 3.3V 直流电压输出和外部电源输入。 z
-  采用 Atmel Atmega328 微处理控制器。因其支持者众多，已有公司开发出来 32 位 的 MCU 平台支持 arduino。 z Arduino 大小尺寸：宽 70mm X 高 54mm

-  **VIN 端口**：VIN 是 input voltage 的缩写，表示有外部电源时的输入端口。如果不使用 USB 供电时，外接电源可以通过此引脚提供电压。（如电池供电，电池正构接 VIN 端口，负 构接 GND 端口） 。
-  **AREF:**   Reference voltage for the analog inputs (模拟输入的基准电压）。使用 analogReference() 命令调用。 

### 1.2. 语法

> Arduino 语法是建立在 C/C++基础上的，其实也就是基础的 C 语法，Arduino 语法只不 过把相关的一些参数设置都函数化，不用我们去了解他的底层，让我们去了解 AVR 单片机 （微控制器）的朋友也能轻松上手。

- **关键字**： if ,	for,	switch case,	while,	do ... while,	break,	continue,	return,
- **语法符号**：；，   {}，   //，    /* */
- **数据类型**： boolean,    char,    byte,   int,    unsigned int,    long,    unsigned long,  float,   double,   string,   array,    void
- **常量**： HIGH（1）  LOW（0）， INPUT（高阻态），OUTPUT（输出），注意电压和引脚

### 1.3. 结构函数

- **结构**： 
  -  void setup() 　 初始化发量，管脚模式，调用库函数等  
  -  void loop() 　 连续执行函数内的语句 
- **函数**：
  - Serial.begin(9600); 这个函数是为串口数据传输设置每秒数据传输速率，每秒多少位 数（波特率）。为了能与计算机进行通信，可选择使用以下这些波特率：“ 300，1200，2400， 4800，9600，14400，19200，28800，38400，57600 或 115200 
  - pinMode(pin, mode)     数字 IO 口输入输出模式定义函数，pin 表示为 0～13， mode 表示为 INPUT 或 OUTPUT
- **数字IO：**
  - digitalWrite(pin, value)   数字 IO 口输出电平定义函数，pin 表示为 0～13，value 表示为 HIGH 或 LOW。比如定义 HIGH 可以驱动 LED。 
  - int digitalRead(pin)      数字 IO 口读输入电平函数，pin 表示为 0～13，value 表示为 HIGH 或 LOW。比如可以读数字传感器。 
- **模拟IO：**
  -  int analogRead(pin)      模拟 IO 口读函数，pin 表示为 0～5（Arduino Diecimila 为 0～5，Arduino nano 为 0～7）。比如可以读模拟传感器（10 位 AD，0～5V 表示为 0～1023）。 
  -  analogWrite(pin, value)    PWM 数字 IO 口 PWM 输出函数，Arduino 数字 IO 口 标注了 PWM 的 IO 口可使用该函数，pin 表示 3, 5, 6, 9, 10, 11，value 表示为 0～255。 比如可用于电机 PWM 调速或音乐播放
- **时间操作：**
  - delay(ms)     延时函数（单位 ms）。 
  - delayMicroseconds(us)    延时函数（单位 us）
- **内置函数：**
  - min(x,y)
  - max(x,y)
  - abs(x)
  - constrain(x, a, b)     约束函数，下限 a，上限 b，x 必须在 ab 之间才能返回
  - map(value, fromLow, fromHigh, toLow, toHigh)   约束函数，value 必须在 fromLow 与 toLow 之间和 fromHigh 与 toHigh 之间。 
  - pow(base,exponent)  开方函数
  - sqrt(x)   开平方

### 1.4. 安装地址

- windows
- Mac OS X
- Linux 

![](https://img-blog.csdn.net/20180712231217134?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdWRvbmdkb25nMTk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 2. Arduino Mega2560

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200408111825624.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

- 基于ATmega2560的主控开发板，采用USB接口
- 具有54路数字输入/输出口，16路模拟输入，4路UART接口，一个16MHz晶体振荡器，一个USB口，一个电源插座，一个ICSP header和一个复位按钮。
- 工作电压：5V 推荐输入电压范围：7-12V； 输入电压范围：6-20V； 数字输入输出口：54 ；模拟输入输出口：16 ；每个I/O口的输出电流：40mA； 3.3V管脚的输出电流：50mA； 内存空间：256KB； SRAM:8KB； EEPROM:4KB； 时钟频率：16MHz
- 输入输出：
  - 54路接口都可作为输入输出，并使用[**pinMode(), digitalWrite()**和**digitalRead(**](https://link.jianshu.com/?t=http%3A%2F%2Fblog.csdn.net%2Fyibu_refresh%2Farticle%2Fdetails%2F40891307)），5V电压，每个IO口最大电流40mA，并且接口内置20-50千欧上拉电阻
  - Serial 串口：Serial 0：0 (RX) and 1 (TX);Serial 1: 19 (RX) and 18 (TX);Serial 2: 17 (RX) and 16 (TX);Serial 3: 15 (RX) and 14 (TX).**[SoftwareSerial library](https://link.jianshu.com/?t=http%3A%2F%2Fblog.csdn.net%2Fyibu_refresh%2Farticle%2Fdetails%2F40896745),**通信时数据灯会闪烁
  - External Interrupts（外部中断）：2 (interrupt 0),3 (interrupt 1),18 (interrupt 5),19 (interrupt 4),20 (interrupt 3),21 (interrupt 2)每个引脚都可配置成低电平触发，或者上升、下降沿触发。详见**[attachInterrupt()](https://link.jianshu.com/?t=http%3A%2F%2Fblog.csdn.net%2Fyibu_refresh%2Farticle%2Fdetails%2F40891415)**功能。
  - PWM脉冲调制：2-13， 44-46； 提供8位PWM输出，有**[analogWrite()](https://link.jianshu.com/?t=http%3A%2F%2Fblog.csdn.net%2Fyibu_refresh%2Farticle%2Fdetails%2F40891627)**功能函数实现。
  - SPI(串行外设接口): 50 (MISO), 51 (MOSI), 52 (SCK), 53 (SS)。使用**[SPI  library](https://link.jianshu.com/?t=http%3A%2F%2Fblog.csdn.net%2Fyibu_refresh%2Farticle%2Fdetails%2F40892331)**库实现。是一种高速的，全双工，同步的通信总线，并且在芯片的管脚上只占用四根线。SPI总线系统是一种同步串行外设接口，它可以使MCU与各种外围设备以串行方式进行通信以交换信息。外围设置FLASHRAM、网络控制器、LCD显示驱动器、A/D转换器和MCU等。SPI总线系统可直接与各个厂家生产的多种标准外围器件直接接口，该接口一般使用4条线：串行时钟线（SCLK）、主机输入/从机输出数据线MISO、主机输出/从机输入数据线MOSI和低电平有效的从机选择线CS（有的SPI接口芯片带有中断信号线INT、有的SPI接口芯片没有主机输出/从机输入数据线MOSI）。
  - 板载LED：13引脚。这是板上自带的LED灯，高电平亮，低电平灭。
  - TWI：20 (SDA) 和21 (SCL)。使用**Wire library**实现功能。对I^2C总线接口的继承和发展，完全兼容I^2C总线，具有硬件实现简单、软件设计方便、运行可靠和成本低廉的优点。TWI由一根时钟线和一根传输数据线组成，以字节为单位进行传输。TWI_SCL\TWI_SDA是TWI总线的信号线。SDA是双向数据线，SCL是时钟线SCL。在TWI总线上传送数据，首先送最高位，由主机发出启动信号，SDA在SCL 高电平期间由高电平跳变为低电平，然后由主机发送一个字节的数据。数据传送完毕，由主机发出停止信号，SDA在SCL 高电平期间由低电平跳变为高电平。
  - Analog输入：16个模拟输入,每个提供10位的分辨率(即2^10=1024个不同的值)。默认情况下他们测量0到5v值。可以通过改变**AREF****引脚**和**analogReference()**功能改变他们变化范围的上界。AREF：是AD转换的参考电压输入端（模拟口输入的电压是与此处的参考电压比较的）。使用**[analogReference()（点击查看详细介绍）](https://link.jianshu.com?t=http%3A%2F%2Fblog.csdn.net%2Fyibu_refresh%2Farticle%2Fdetails%2F40894423)**完成功能。

### 2.2. 供电方式

#### 2.2.1. **使用USB端口为Arduino供电**

> 用Arduino的USB数据线连接在手机充电器或者充电宝为Arduino供电。

#### 2.2.2. **使用Vin引脚为Arduino供电**

> Vin引脚可用于为Arduino开发板供电使用。但使用Vin引脚为Arduino开发板供电时，直流电源电压必须为`7V ~ 12V`。使用`低于7V的电源电压可能导致Arduino工作不稳定`。使用高于12V电源电压存在着毁坏Arduino开发板的风险。

#### 2.2.3. **使用5V引脚为Arduino供电**

> 使用5V引脚为Arduino开发板供电时，一定要确保电源电压为稳定的直流电源，且电源电压为+5V。。

#### 2.2.4. **使用电源接口为Arduino供电**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210120181401581.png)

## 3. IDE使用

1. get an arduino board and usb cable![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200408104715951.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)
2. download the arduino IDE :http://www.arduino.cc/en/Main/Software
3. connect the board,  the green power(labelled PWR) go on.
4. install the driver, if download, you will find the USB Serial Port to see your board
5. open arduino examples: file->examples                        ![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200408105057219.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)
6. select your board:                                     ![](https://gitee.com/github-25970295/blogImage/raw/master/img/20200408105255207.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)
7. select your serial port
8. upload your program

或者下载 arduino agent create 使用网页版进行编程。

## 4. 基础实验

### 4.1. 板载LED

> ​       发光二极管简称为 LED。由镓（Ga）与砷（AS）、磷（P）的化合物制成的二极 管，当电子与空穴复合时能辐射出可见光，因而可以用来制成发光二极管，在电路 及仪器中作为指示灯，或者组成文字或数字显示。磷砷化镓二极管发红光，磷化镓 二极管发绿光，碳化硅二极管发黄光。 
>
> ​       它是半导体二极管的一种，可以把电能转化成光能；常简写为 LED。发光二极 管与普通二极管一样是由一个 PN 结组成，也具有单向导电性。当给发光二极管加 上正向电压后，从 P 区注入到 N 区的空穴和由 N 区注入到 P 区的电子，在 PN 结附 近数微米内分别与 N 区的电子和 P 区的空穴复合，产生自发辐射的荧光。不同的半 导体材料中电子和空穴所处的能量状态不同。当电子和空穴复合时释放出的能量多 少不同，释放出的能量越多，则发出的光的波长越短。常用的是发红光、绿光或黄 光的二极管。 

```c
// Pin 13 has an LED connected on most Arduino boards.
// give it a name:
int led = 13;

// the setup routine runs once when you press reset:
void setup() {                
  // initialize the digital pin as an output.
  pinMode(led, OUTPUT);     
}

// the loop routine runs over and over again forever:
void loop() {
  digitalWrite(led, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);               // wait for a second
  digitalWrite(led, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);               // wait for a second
}
```

### 4.2.  PWM 占空比

> ​			输出电压=（接通时间/脉冲时间）最大电压值 Pulse Width Modulation 就是通常所说的PWM，译为脉冲宽度调制，简称脉宽调制。 脉冲宽度调制（PWM）是一种对模拟信号电平进行数字编码的方法，由于计算机不能 输出模拟电压，只能输出0 或5V 的的数字电压值，我们就通过使用高分辨率计数器 ，利用方波的占空比被调制的方法来对一个具体模拟信号的电平进行编码。PWM 信号 仍然是数字的，因为在给定的任何时刻，满幅值的直流供电要么是5V(ON)，要么是0V (OFF)。电压或电流源是以一种通(ON)或断(OFF)的重复脉冲序列被加到模拟负载上去 的。通的时候即是直流供电被加到负载上的时候，断的时候即是供电被断开的时候。 只要带宽足够，任何模拟值都可以使用PWM 进行编码。输出的电压值是通过通和断的 时间进行计算的。输出电压=（接通时间/脉冲时间）最大电压值

```c
int brightness = 0;
int fadeAmount = 5;
void setup()  { 
  pinMode(9, OUTPUT);
} 
void loop()  { 
  analogWrite(9, brightness);
  brightness = brightness + fadeAmount;
  if (brightness == 0 || brightness == 255) {
    fadeAmount = -fadeAmount ;
  }     
  delay(30);                     
}
```

### 4.3. 流水灯

```c
int Led1 = 1;
int Led2 = 2;
int Led3 = 3;
int Led4 = 4;
int Led5 = 5;
int Led6 = 6;

void style_1(void)
{
  unsigned char j;
  for(j=1;j<=6;j++)
  {
    digitalWrite(j,HIGH);
    delay(200);
  }
  for(j=6;j>=1;j--)
  {
    digitalWrite(j,LOW);
    delay(200);
  } 
}

void flash(void)
{   
  unsigned char j,k;
  for(k=0;k<=1;k++)
  {
    for(j=1;j<=6;j++)
      digitalWrite(j,HIGH);
    delay(200);
    for(j=1;j<=6;j++)
      digitalWrite(j,LOW);
    delay(200);
  }
}

void style_2(void)
{
  unsigned char j,k;
  k=1;
  for(j=3;j>=1;j--)
  {   
    digitalWrite(j,HIGH);
    digitalWrite(j+k,HIGH);
    delay(400);
    k +=2;
  }
  k=5;
  for(j=1;j<=3;j++)
  {
    digitalWrite(j,LOW);
    digitalWrite(j+k,LOW);
    delay(400);
    k -=2;
  }
}

void style_3(void)
{
  unsigned char j,k;
  k=5;
  for(j=1;j<=3;j++)
  {
    digitalWrite(j,HIGH);
    digitalWrite(j+k,HIGH);
    delay(400);
    digitalWrite(j,LOW);
    digitalWrite(j+k,LOW);
    k -=2;
  }
  k=3;
  for(j=2;j>=1;j--)
  {   
    digitalWrite(j,HIGH);
    digitalWrite(j+k,HIGH);
    delay(400);
    digitalWrite(j,LOW);
    digitalWrite(j+k,LOW);
    k +=2;
  } 
}
void setup()
{ 
  unsigned char i;
  for(i=1;i<=6;i++)
    pinMode(i,OUTPUT);
}
void loop()
{   
  style_1();
  flash();
  style_2();
  flash();
  style_3();
  flash();
}
```

### 4.4. 蜂鸣器

> ​		蜂鸣器发声原理是电流通过电磁线圈，使电磁线圈产生磁场来驱动振动膜发声的，因此 需要一定的电流才能驱动它，本实验用的蜂鸣器内部带有驱动电路，所以可以直接使用。当 不蜂鸣器连接的引脚为高电平时，内部驱动电路导通，蜂鸣器发出声音；当不蜂鸣器连接的 引脚为低电平，内部驱动电路截止，蜂鸣器不发出声音

```c
int buzzer=7;
void setup()
{
  pinMode(buzzer,OUTPUT);
}
void loop()
{
  unsigned char i,j;
  while(1)
  {
    for(i=0;i<80;i++)
    {
      digitalWrite(buzzer,HIGH);
      delay(1);
      digitalWrite(buzzer,LOW);
      delay(1);
    }
    for(i=0;i<100;i++)
    {
      digitalWrite(buzzer,HIGH);
      delay(2);
      digitalWrite(buzzer,LOW);
      delay(2);
    }
  }
}
```

### 4.5. 倾斜开关

> 滚珠开关：也叫碰珠开关、摇珠开关、钢珠开关、倾斜开关，倒顺开关、角度传感器。 它主要是利用滚珠在开关内随不同倾斜角度的变化，达到触发电路的目的。 目前滚珠开关 在市场上使用的常用型号有 SW-200D、SW-460、SW-300DA 等，使用的是 SW-200D 型 号的。这类开关不象传统的水银开关，它功效同水银开关，但没有水银开关的环保及安全等问题

```c
void setup()
{
    pinMode(13,OUTPUT);//设置数字8引脚为输出模式
}
void loop()
{
    int i;//定义变量i
    while(1)
    {
       i=analogRead(5);//读取模拟5口电压值
       if(i>200)//如果大于512（2.5V）
       {
          digitalWrite(13,HIGH);//点亮led灯
       }
       else//否则
       {
          digitalWrite(13,LOW);//熄灭led灯
       }
    }
}

```

### 4.6  加速度&&超声波测距等

- [ADXL3xx](http://www.ncnynl.com/archives/201607/361.html): 读取一个 ADXL3xx 加速计
- [Knock](http://www.ncnynl.com/archives/201607/362.html): 通过一个压电元件来侦察敲击
- [Memsic2125](http://www.ncnynl.com/archives/201607/363.html): 2轴加速计
- [Ping](http://www.ncnynl.com/archives/201607/364.html): 通过一个超声波测距仪来侦察物品

### 4.7. FlexSensor

### 4.8. BlueTooth

> 使用串口调试工具例如XCOM V2.0.exe，进行AT命令操作，设置主从模式后会自动进行配对。注意工作电压3.3v。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210205235604849.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210205235524626.png)

```c++
#include <SoftwareSerial.h>
SoftwareSerial BT(10, 9); 
const int flexPin = A0; //pin A0 to read analog input
String val; 
void setup() {
  pinMode(flexPin, INPUT);
  Serial.begin(9600);  
  Serial.println("BT is ready!");
 
  BT.begin(9600);
}
void loop() {
  if (Serial.available()) {
    val = Serial.readString();
    Serial.print("Send: ");
    Serial.println(val);
    BT.print(val);
  }
  if (BT.available()) {
    val = BT.readString();
    Serial.print("Recieved: ");
    Serial.println(val);
  }
}
```

## 5. Arduino 导入库

> ​		Arduino包含两种库：标准库和第三方库，当然也可以自己写类库。标准库安装Arduino IDE后就已经导入，只需要直接调用就行，第三方类库则需要导入，如果没有导入编译器就会报错

### 5.1. 标准库

> [EEPROM](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:eeprom)- 对“永久存储器”进行读和写
>
> [Ethernet](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:ethernet)- 用于通过 Arduino 以太网扩展板连接到互联网
>
> [Firmata](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:firmata)- 与电脑上应用程序通信的标准串行协议。
>
> [LiquidCrystal](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:liquidcrystal)- 控制液晶显示屏（LCD）
>
> [SD](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:sd)- 对 SD 卡进行读写操作
>
> [Servo](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:servo)- 控制伺服电机
>
> [SPI](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:spi)- 与使用的串行外设接口（SPI）总线的设备进行通信
>
> [SoftwareSerial](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:softwareserial)- 使用任何数字引脚进行串行通信
>
> [Stepper](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:stepper)- 控制步进电机
>
> [WiFi](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:wifi)- 用于通过 Aduino 的 WiFi 扩展板连接到互联网
>
> [Wire](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:wire)- 双总线接口（TWI/I2C）通过网络对设备或者传感器发送和接收数据。
>
> [PWM Frequency Library](https://link.jianshu.com?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:pwm_frequency)- 自定义PWM频率

### 5.2. 第三方类库

> [IRremote](https://link.jianshu.com/?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:irremote)-红外控制
>
> [DS3231](https://link.jianshu.com/?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:ds3231)-时钟芯片
>
> [Timer](https://link.jianshu.com/?t=http://wiki.geek-workshop.com/doku.php?id=arduino:libraries:timer)- 利用millis（）函数来模拟多线程等等

### 5.3. 导入库

IDE -》项目--》加载库，导入库，   导入成功后就可以在IDE上直接查看到与库相关的例子

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200519203955639.png)

> 直接将压缩包解压到IDE安装路径下的libraries文件夹，然后直接打开IDE就行了！
>
> 注意：如果IDE之前就已经打开要先关闭再打开，否则无法导入成功

- 本地库目录查看方式：   文件--》偏好设置--》库文件目录
- 一个库的目录格式：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200519204735688.png)

## 6. 串口操作

### 6.1. Mega2560

> Serial.begin(); //开启串口
>
> Serial.end();    //关闭串口
>
> Serial.available();//判断串口缓冲器是否有数据装入
>
> Serial.read();    //读取串口数据
>
> Serial.peek();    //返回下一字节(字符)输入数据，但不删除它
>
> Serial.flush();    //清空串口缓存
>
> Serial.print();    //写入字符串数据到串口
>
> Serial.println();   //写入字符串数据+换行到串口
>
> Serial.write();     //写入二进制数据到串口
>
> Serial.SerialEvent();//read时触发的事件函数
>
> Serial.readBytes(buffer,length);//读取固定长度的二进制流

- Mega2560 有三个额外的串口：Serial 1使用19(RX)和18(TX)，Serial 2使用17(RX)和16(TX)，Serial3使用15(RX)和14(TX)

> 串口、COM口是指的物理接口形式(硬件)。而TTL、RS-232、RS-485是指的电平标准(电信号)
>
> TTL串口设备 TTL标准是低电平为0，高电平为1（+5V电平）
>
> - 高电平3.6~5V，
> - 低电平0V~2.4V
>
> RS232串口; 标准是正电平为0，负电平为1
>
> - －15v ~ －3v 代表1
> - ＋3v ~ ＋15v 代表0

### 6.2. 多串口操作

```c
void setup() {
  // initialize both serial ports:
  Serial.begin(9600);
  Serial1.begin(9600);
}
void loop() {
  // read from port 1, send to port 0:
  if (Serial1.available()) {
    int inByte = Serial1.read();
    Serial.write(inByte);
  }
  // read from port 0, send to port 1:
  if (Serial.available()) {
    int inByte = Serial.read();
    Serial1.write(inByte);
  }
}
```

### 6.3. 串口中断触发

#### 6.3.1  serialEvent 串口数据发送

```c
String inputString = "";         // a string to hold incoming data
boolean stringComplete = false;  // whether the string is complete
void setup() {
  // initialize serial:
  Serial.begin(9600);
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
}
void loop() {
  // print the string when a newline arrives:
  if (stringComplete) {
    Serial.println(inputString);
    // clear the string:
    inputString = "";
    stringComplete = false;
  }
}
/*
  SerialEvent occurs whenever a new data comes in the
 hardware serial RX.  This routine is run between each
 time loop() runs, so using delay inside loop can delay
 response.  Multiple bytes of data may be available.
 */
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag
    // so the main loop can do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
```

### 6.4. Arduino&&Processing

```c
void setup() {
  // initialize the serial communication:
  Serial.begin(9600);
}

void loop() {
  // send the value of analog input 0:
  Serial.println(analogRead(A0));
  // wait a bit for the analog-to-digital converter
  // to stabilize after the last reading:
  delay(2);
}
import processing.serial.*;

Serial myPort;        // The serial port
int xPos = 1;         // horizontal position of the graph
float inByte = 0;

void setup () {
  // set the window size:
  size(400, 300);
  // List all the available serial ports
  // if using Processing 2.1 or later, use Serial.printArray()
  println(Serial.list());
  // I know that the first port in the serial list on my mac
  // is always my  Arduino, so I open Serial.list()[0].
  // Open whatever port is the one you're using.
  myPort = new Serial(this, Serial.list()[0], 9600);
  // don't generate a serialEvent() unless you get a newline character:
  myPort.bufferUntil('\n');
  // set inital background:
  background(0);
}
void draw () {
  // draw the line:
  stroke(127, 34, 255);
  line(xPos, height, xPos, height - inByte);

  // at the edge of the screen, go back to the beginning:
  if (xPos >= width) {
    xPos = 0;
    background(0);
  } else {
    // increment the horizontal position:
    xPos++;
  }
}
void serialEvent (Serial myPort) {
  // get the ASCII string:
  String inString = myPort.readStringUntil('\n');

  if (inString != null) {
    // trim off any whitespace:
    inString = trim(inString);
    // convert to an int and map to the screen height:
    inByte = float(inString);
    println(inByte);
    inByte = map(inByte, 0, 1023, 0, height);
  }
}
```

### 6.5. SoftwareSerial

> 使用软件的串口功能（因此称为“SoftwareSerial”，即“软串口”），现有的SoftwareSerial 库，以允许其他的Arduino数字引脚的串行通信，这可能有多个软件串口速度高达115200bps。
>
> - 如果使用多个串口软件，一次只有一个软件可以接收数据。
> - 在 Mega 和 Mega 2560 上，不是所有的引脚都支持中断，允许用于RX的引脚包括：10, 11, 12, 13, 50, 51, 52, 53, 62, 63, 64, 65, 66, 67, 68, 69

```c++
/*
  Software serial multple serial test
 
 从硬件串口接收，发送到软件的序列。
 软件串行接收，发送到硬件序列。
 
 * RX是数字引脚2（连接到其他设备的TX）
 * TX是数字引脚3（连接到其他设备的RX）
 
 */
#include <SoftwareSerial.h>
 
SoftwareSerial mySerial(2, 3); // RX, TX
 
void setup()  
{
  //打开串行通信，等待端口打开：
  Serial.begin(57600);
  while (!Serial) {
    ; // 等待串口连接。Needed for Leonardo only
  }
 
 
  Serial.println("Goodnight moon!");
 
  // 设置串口通讯的速率
  mySerial.begin(4800);
  mySerial.println("Hello, world?");
}
 
void loop() // 循环
{
  if (mySerial.available())
    Serial.write(mySerial.read());
  if (Serial.available())
    mySerial.write(Serial.read());
}
```

## 7. 字符串操作

> - isAlphaNumeric()    这是文字和数字
> - isAlpha()      这是字母
> - isAscii()      这是ASCII
> - isWhitespace()    这是空白符（包括空格，TAB和回车等）
> - isControl()     这是控制字符
> - isDigit()         这是数字位数
> - isGraph()        这是无空格可打印的字符
> - isLowerCase()    这是小写字母
> - isPunct()     这是标点符号
> - isSpace()      这是空格字符
> - isUpperCase()           这是大写字母
> - isHexadecimalDigit()      这是有效的十六进制位数(即是 0 - 9, a - F, or A - F)

- 字符串拼接  等同Java中字符串操作

```c
Serial.println("I want " + analogRead(A0) + " donuts"); 

//toUpperCase()把所有字符串改为大写字母，
//toLowerCase()把所有字符串改为小写字母。只有 A 到Z 或者 a 到 z的字符受到影响。
```

- 字符串中字符操作

```c
String re="SensorReading: 456")
int pos=re.indexOf(':',position=0)   //.lastIndexOf()
//re.length()   获取长度
re.setCharAt(pos,'=')
//charAt()    setCharAt()
    
stringOne = "HTTP/1.1 200 OK";
  if (stringOne.startsWith("200 OK", 9)) {
    Serial.println("Got an OK from the server"); 
} 
//等同下面代码
stringOne = "HTTP/1.1 200 OK";
  if (stringOne.substring(9) == "200 OK") {
    Serial.println("Got an OK from the server"); 
} 
```

- 字符串初始化

```c
// using a constant String:
  String stringOne = "Hello String";
  Serial.println(stringOne);      // prints "Hello String"

  // converting a constant char into a String:
  stringOne =  String('a');
  Serial.println(stringOne);       // prints "a"

  // converting a constant string into a String object:
  String stringTwo =  String("This is a string");
  Serial.println(stringTwo);      // prints "This is a string"

  // concatenating two strings:
  stringOne =  String(stringTwo + " with more");
  // prints "This is a string with more":
  Serial.println(stringOne);

  // using a constant integer:
  stringOne =  String(13);
  Serial.println(stringOne);      // prints "13"

  // using an int and a base:
  stringOne =  String(analogRead(A0), DEC);
  // prints "453" or whatever the value of analogRead(A0) is
  Serial.println(stringOne);

  // using an int and a base (hexadecimal):
  stringOne =  String(45, HEX);
  // prints "2d", which is the hexadecimal version of decimal 45:
  Serial.println(stringOne);

  // using an int and a base (binary)
  stringOne =  String(255, BIN);
  // prints "11111111" which is the binary value of 255
  Serial.println(stringOne);

  // using a long and a base:
  stringOne =  String(millis(), DEC);
  // prints "123456" or whatever the value of millis() is:
  Serial.println(stringOne);

  //using a float and the right decimal places:
  stringOne = String(5.698, 3);
  Serial.println(stringOne);

  //using a float and less decimal places to use rounding:
  stringOne = String(5.698, 2);
  Serial.println(stringOne);
```

- 类型转化

```c
string.toInt()   //转化为Int类型
Serial.parseInt() //来分离带有逗号的数据，将信息读到变量里
//char()   byte() int()  word()   long()  float()  
```

## 8. 喇叭

> tone（）命令是通过Atmega的内置定时器来工作的，设置你想要的频率，并且用定时器来产生一个输出脉冲。因为它只用一个定时器，所以你只能一个定时器弹奏一个音调。然而你可以按顺序地在不同的引脚上弹奏音调。为了达到这个目的，你需要在移到下一个引脚前关闭这个引脚的定时器。

```c
#include "pitches.h"
 
// notes in the melody:
int melody[] = {
NOTE_E4, NOTE_E4, NOTE_E4, NOTE_C4, NOTE_E4, NOTE_G4, NOTE_G3,
NOTE_C4, NOTE_G3, NOTE_E3, NOTE_A3, NOTE_B3, NOTE_AS3, NOTE_A3, NOTE_G3, NOTE_E4, NOTE_G4, NOTE_A4, NOTE_F4, NOTE_G4, NOTE_E4, NOTE_C4, NOTE_D4, NOTE_B3,
NOTE_C4, NOTE_G3, NOTE_E3, NOTE_A3, NOTE_B3, NOTE_AS3, NOTE_A3, NOTE_G3, NOTE_E4, NOTE_G4, NOTE_A4, NOTE_F4, NOTE_G4, NOTE_E4, NOTE_C4, NOTE_D4, NOTE_B3,
NOTE_G4, NOTE_FS4, NOTE_E4, NOTE_DS4, NOTE_E4, NOTE_GS3, NOTE_A3, NOTE_C4, NOTE_A3, NOTE_C4, NOTE_D4, NOTE_G4, NOTE_FS4, NOTE_E4, NOTE_DS4, NOTE_E4, NOTE_C5, NOTE_C5, NOTE_C5,
NOTE_G4, NOTE_FS4, NOTE_E4, NOTE_DS4, NOTE_E4, NOTE_GS3, NOTE_A3, NOTE_C4, NOTE_A3, NOTE_C4, NOTE_D4, NOTE_DS4, NOTE_D4, NOTE_C4,
NOTE_C4, NOTE_C4, NOTE_C4, NOTE_C4, NOTE_D4, NOTE_E4, NOTE_C4, NOTE_A3, NOTE_G3, NOTE_C4, NOTE_C4, NOTE_C4, NOTE_C4, NOTE_D4, NOTE_E4,
NOTE_C4, NOTE_C4, NOTE_C4, NOTE_C4, NOTE_D4, NOTE_E4, NOTE_C4, NOTE_A3, NOTE_G3
};
// note durations: 4 = quarter note, 8 = eighth note, etc.:
int noteDurations[] = {
8,4,4,8,4,2,2,
3,3,3,4,4,8,4,8,8,8,4,8,4,3,8,8,3,
3,3,3,4,4,8,4,8,8,8,4,8,4,3,8,8,2,
8,8,8,4,4,8,8,4,8,8,3,8,8,8,4,4,4,8,2,
8,8,8,4,4,8,8,4,8,8,3,3,3,1,
8,4,4,8,4,8,4,8,2,8,4,4,8,4,1,
8,4,4,8,4,8,4,8,2
};
void setup() {
  // iterate over the notes of the melody:
  for (int thisNote = 0; thisNote < 98; thisNote++) { 
    // to calculate the note duration, take one second
    // divided by the note type.
    //e.g. quarter note = 1000 / 4, eighth note = 1000/8, etc.
    int noteDuration = 1000/noteDurations[thisNote];
    tone(8, melody[thisNote],noteDuration);
    // to distinguish the notes, set a minimum time between them.
    // the note's duration + 30% seems to work well:
    int pauseBetweenNotes = noteDuration * 1.30;
    delay(pauseBetweenNotes);
    // stop the tone playing:
    noTone(8);
  }
}
void loop() {
// no need to repeat the melody.
}
```

## 9. 问题记录

Arduino 开发板没有接入任何设备，模拟口电压为0.02v左右，但是读取到的模拟量呈现一定波形，不知道什么原因？

![Phenomena1](https://gitee.com/github-25970295/blogImage/raw/master/img/Phenomena1.png)

```c
// These constants won't change:
const int sensorPin = A0;    // pin that the sensor is attached to
const int ledPin = 9;        // pin that the LED is attached to
// variables:
int sensorValue = 0;         // the sensor value
int sensorMin = 1023;        // minimum sensor value
int sensorMax = 0;           // maximum sensor value
void setup() {
  // turn on LED to signal the start of the calibration period:
  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH);
  // calibrate during the first five seconds
  while (millis() < 5000) {
    sensorValue = analogRead(sensorPin);
    // record the maximum sensor value
    if (sensorValue > sensorMax) {
      sensorMax = sensorValue;
    }
    // record the minimum sensor value
    if (sensorValue < sensorMin) {
      sensorMin = sensorValue;
    }
  }
  // signal the end of the calibration period
  digitalWrite(13, LOW);
}
void loop() {
  // read the sensor:
  sensorValue = analogRead(sensorPin);
  // apply the calibration to the sensor reading
  sensorValue = map(sensorValue, sensorMin, sensorMax, 0, 255);
  // in case the sensor value is outside the range seen during calibration
  sensorValue = constrain(sensorValue, 0, 255);
  // fade the LED using the calibrated value:
  analogWrite(ledPin, sensorValue);
}
```

## 10. 过滤算法

### 10.1. 滑动平均处理

```c
const int numReadings = 10;
int readings[numReadings];      // the readings from the analog input
int readIndex = 0;              // the index of the current reading
int total = 0;                  // the running total
int average = 0;                // the average
int inputPin = A0;
void setup() {
  // initialize serial communication with computer:
  Serial.begin(9600);
  // initialize all the readings to 0:
  for (int thisReading = 0; thisReading < numReadings; thisReading++) {
    readings[thisReading] = 0;
  }
}
void loop() {
  // subtract the last reading:
  total = total - readings[readIndex];
  // read from the sensor:
  readings[readIndex] = analogRead(inputPin);
  // add the reading to the total:
  total = total + readings[readIndex];
  // advance to the next position in the array:
  readIndex = readIndex + 1;

  // if we're at the end of the array...
  if (readIndex >= numReadings) {
    // ...wrap around to the beginning:
    readIndex = 0;
  }
  // calculate the average:
  average = total / numReadings;
  // send it to the computer as ASCII digits
  Serial.println(average);
  delay(1);        // delay in between reads for stability
}
```

## 11. 学习网站

Arduino 具体语法介绍： https://www.ncnynl.com/archives/201607/370.html

Arduino UNO的原理图:https://www.arduino.cc/en/Main/ArduinoBoardUno

原理图PDF：https://www.arduino.cc/en/uploads/Main/Arduino_Uno_Rev3-schematic.pdf

SoftwareSerial: https://wiki.nxez.com/arduino:libraries:softwareserial

## 12. 问题记录

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210131204902585.png)

- 板子对应的电压引脚是有电压的，但是hex文件无法烧录进去，尝试了不同的flash方法，认为板子坏了，重新买了一个。 




$$
\begin{align*}
V_{out} &=\frac{1}{1+R_1/R_2}* V_{in}\\
对R_1求导： L_{r_1}'&=-(1/R_2+{R_1}^2/R_2+2R_1)\\
对 R_1求二阶导： L_{R_1}''&=-(2*R_1/R_2+2)
\end{align*}
$$



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/arduino-learning-record/  

