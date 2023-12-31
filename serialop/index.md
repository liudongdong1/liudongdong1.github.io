# SerialOp


### 1. Serial Type

#### 1.1. 物理接口形式

- **UART接口：**`通用异步收发器`（Universal Asynchronous Receiver/Transmitter)，UART是串口收发的逻辑电路，这部分可以独立成芯片，也可以作为模块嵌入到其他芯片里，单片机、SOC、PC里都会有UART模块。
- **COM口：**特指台式计算机或一些电子设备上的`D-SUB外形`(一种连接器结构，VGA接口的连接器也是D-SUB)的串行通信口，`应用了串口通信时序和RS232的逻辑电平`。
- **USB口：**通用`串行总线`，和串口完全是两个概念。虽然也是串行方式通信，但由于USB的通信时序和信号电平都和串口完全不同，因此和串口没有任何关系。`USB是高速的通信接口，用于PC连接各种外设，U盘、键鼠、移动硬盘、当然也包括“USB转串口”的模块`。（USB转串口模块，就是USB接口的UART模块）

#### 1.2. 逻辑电平形式

- **TTL：**TTL指双极型三极管逻辑电路，市面上很多“USB转TTL”模块，实际上是“USB转TTL电平的串口”模块。这种`信号0对应0V，1对应3.3V或者5V`。与单片机、SOC的IO电平兼容。T

  TL电平：全双工（逻辑1: 2.4V--5V 逻辑0: 0V--0.5V）. `常见的USB转TTL芯片有CH340、PL2303、FT232、CP2102等`

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210125233112716.png)

- **RS232：**是电子工业协会(Electronic Industries Association，EIA) 制定的`异步传输标准接口`，同时对应着电平标准和通信协议（时序），其电平标准：`+3V～+15V对应0，-3V～-15V对应1`。rs232 的逻辑电平和TTL 不一样但是协议一样。

  RS-232电平：全双工（逻辑1：-15V--5V 逻辑0：+3V--+15V).  `232芯片用于5V单片机，3232芯片用于3.3V单片机。`

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210125232933706.png)

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210125233044458.png)

- **RS485：**RS485是一种串口接口标准，为了长距离传输采用差分方式传输，传输的是差分信号，抗干扰能力比RS232强很多。两线压差为-(2~6)V表示0，两线压差为+(2~6)V表示1

  RS-485：半双工、（逻辑1：+2V--+6V 逻辑0： -6V---2V）这里的电平指AB 两线间的电压差。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210125233352145.png)

### 2. Serial Class-pyserial

```python
ser=serial.Serial("/dev/ttyUSB0",9600,timeout=0.5) #使用USB连接串行口
ser=serial.Serial("/dev/ttyAMA0",9600,timeout=0.5) #使用树莓派的GPIO口连接串行口
ser=serial.Serial(1,9600,timeout=0.5)#winsows系统使用com1口连接串行口
ser=serial.Serial("com1",9600,timeout=0.5)#winsows系统使用com1口连接串行口
ser=serial.Serial("/dev/ttyS1",9600,timeout=0.5)#Linux系统使用com1口连接串行口
```

- **属性**

> name:设备名字
> port：读或者写端口
> baudrate：波特率
> bytesize：字节大小
> parity：校验位
> stopbits：停止位
> timeout：读超时设置
> writeTimeout：写超时
> xonxoff：软件流控
> rtscts：硬件流控
> dsrdtr：硬件流控
> interCharTimeout:字符间隔超时

- **方法**

> ser.isOpen()：查看端口是否被打开。
> ser.open() ：打开端口‘。
> ser.close()：关闭端口。
> ser.read()：从端口读字节数据。默认1个字节。
> ser.read_all():从端口接收全部数据。
> ser.write("hello")：向端口写数据。
> ser.readline()：读一行数据。
> ser.readlines()：读多行数据。
> in_waiting()：返回接收缓存中的字节数。
> flush()：等待所有数据写出。
> flushInput()：丢弃接收缓存中的所有数据。
> flushOutput()：终止当前写操作，并丢弃发送缓存中的数据。

- **函数封装**

```python
from PyQt5.QtCore import QTimer
import serial
import serial.tools.list_ports

class FlexSensor(object):
    def __init__(self,com,bps,timeout):
        self.port = com
        self.bps = bps
        self.timeout =timeout
        #self.timer=Qtimer()
        global Ret     # flag: 判断是否打开，如果打开，Ret=true
        try:
            # 打开串口，并得到串口对象
            self.main_engine= serial.Serial(self.port,self.bps,timeout=self.timeout)
            # 判断是否打开成功
            if (self.main_engine.is_open):
               Ret = True
        except Exception as e:
            print("---异常---：", e)
    # 打印设备基本信息
    def Print_Name(self):
        print(self.main_engine.name) #设备名字
        print(self.main_engine.port)#读或者写端口
        print(self.main_engine.baudrate)#波特率
        print(self.main_engine.bytesize)#字节大小
        print(self.main_engine.parity)#校验位
        print(self.main_engine.stopbits)#停止位
        print(self.main_engine.timeout)#读超时设置
        print(self.main_engine.writeTimeout)#写超时
        print(self.main_engine.xonxoff)#软件流控
        print(self.main_engine.rtscts)#软件流控
        print(self.main_engine.dsrdtr)#硬件流控
        print(self.main_engine.interCharTimeout)#字符间隔超时

    #打开串口
    def Open_Engine(self):
        self.main_engine.open()

    #关闭串口
    def Close_Engine(self):
        self.main_engine.close()
        print(self.main_engine.is_open)  # 检验串口是否打开

    # 打印可用串口列表
    @staticmethod
    def Print_Used_Com():
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)

    #接收指定大小的数据
    #从串口读size个字节。如果指定超时，则可能在超时后返回较少的字节；如果没有指定超时，则会一直等到收完指定的字节数。
    def Read_Size(self,size):
        return self.main_engine.read(size=size)

    #接收一行数据
    # 使用readline()时应该注意：打开串口时应该指定超时，否则如果串口没有收到新行，则会一直等待。
    # 如果没有超时，readline会报异常。
    def Read_Line(self):
        return self.main_engine.readline()

    #发数据
    def Send_data(self,data):
        self.main_engine.write(data)

    #接收数据
    #一个整型数据占两个字节
    #一个字符占一个字节
    def Recive_data(self,way):
        # 循环接收数据，此为死循环，可用线程实现
        print("开始接收数据：")
        while True:
            try:
                # 一个字节一个字节的接收
                if self.main_engine.in_waiting:
                    if(way == 0):
                        for i in range(self.main_engine.in_waiting):
                            print("接收ascii数据："+str(self.Read_Size(1)))
                            data1 = self.Read_Size(1).hex()#转为十六进制
                            data2 = int(data1,16)#转为十进制print("收到数据十六进制："+data1+"  收到数据十进制："+str(data2))
                    if(way == 1):
                        #整体接收
                        # data = self.main_engine.read(self.main_engine.in_waiting).decode("utf-8")#方式一
                        data = self.main_engine.read_all()#方式二print("接收ascii数据：", data)
            except Exception as e:
                print("异常报错：",e)

if __name__ == "__main__":
    FlexSensor.Print_Used_Com()
    Ret =False #是否创建成功标志
    Engine1 = FlexSensor("com5",115200,0.5)
```

### 3. 串口相关项目

#### 3.1. Qt-based QSerial

- **UI布局文件**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>Form</class>
 <widget class="QWidget" name="Form">
  <property name="windowModality">
   <enum>Qt::NonModal</enum>
  </property>
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>704</width>
    <height>562</height>
   </rect>
  </property>
  <property name="font">
   <font>
    <family>方正兰亭中黑_GBK</family>
    <pointsize>9</pointsize>
    <weight>50</weight>
    <italic>false</italic>
    <bold>false</bold>
   </font>
  </property>
  <property name="cursor">
   <cursorShape>PointingHandCursor</cursorShape>
  </property>
  <property name="mouseTracking">
   <bool>false</bool>
  </property>
  <property name="windowTitle">
   <string>Form</string>
  </property>
  <property name="styleSheet">
   <string notr="true">color: rgb(255, 255, 255);
background-color: rgb(25, 25, 25);
font: 9pt &quot;方正兰亭中黑_GBK&quot;;</string>
  </property>
  <widget class="QLabel" name="label">
   <property name="geometry">
    <rect>
     <x>40</x>
     <y>40</y>
     <width>71</width>
     <height>21</height>
    </rect>
   </property>
   <property name="text">
    <string>接收区</string>
   </property>
  </widget>
  <widget class="QLabel" name="label_2">
   <property name="geometry">
    <rect>
     <x>40</x>
     <y>355</y>
     <width>54</width>
     <height>12</height>
    </rect>
   </property>
   <property name="text">
    <string>发送区</string>
   </property>
   <property name="textFormat">
    <enum>Qt::AutoText</enum>
   </property>
  </widget>
  <widget class="QPushButton" name="Send_Button">
   <property name="geometry">
    <rect>
     <x>330</x>
     <y>350</y>
     <width>75</width>
     <height>23</height>
    </rect>
   </property>
   <property name="text">
    <string>发送</string>
   </property>
  </widget>
  <widget class="QPushButton" name="ClearButton">
   <property name="geometry">
    <rect>
     <x>330</x>
     <y>40</y>
     <width>75</width>
     <height>23</height>
    </rect>
   </property>
   <property name="text">
    <string>清除</string>
   </property>
  </widget>
  <widget class="QTextEdit" name="textEdit_Recive">
   <property name="geometry">
    <rect>
     <x>40</x>
     <y>70</y>
     <width>361</width>
     <height>251</height>
    </rect>
   </property>
   <property name="styleSheet">
    <string notr="true">background-color: rgb(255, 255, 255);
background-color: rgb(0, 0, 0);</string>
   </property>
   <property name="html">
    <string>&lt;!DOCTYPE HTML PUBLIC &quot;-//W3C//DTD HTML 4.0//EN&quot; &quot;http://www.w3.org/TR/REC-html40/strict.dtd&quot;&gt;
&lt;html&gt;&lt;head&gt;&lt;meta name=&quot;qrichtext&quot; content=&quot;1&quot; /&gt;&lt;style type=&quot;text/css&quot;&gt;
p, li { white-space: pre-wrap; }
&lt;/style&gt;&lt;/head&gt;&lt;body style=&quot; font-family:'方正兰亭中黑_GBK'; font-size:9pt; font-weight:400; font-style:normal;&quot;&gt;
&lt;p style=&quot;-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'SimSun';&quot;&gt;&lt;br /&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
   </property>
  </widget>
  <widget class="QTextEdit" name="textEdit_Send">
   <property name="geometry">
    <rect>
     <x>40</x>
     <y>385</y>
     <width>361</width>
     <height>151</height>
    </rect>
   </property>
   <property name="styleSheet">
    <string notr="true">background-color: rgb(0, 0, 0);</string>
   </property>
  </widget>
  <widget class="QWidget" name="gridLayoutWidget">
   <property name="geometry">
    <rect>
     <x>440</x>
     <y>100</y>
     <width>221</width>
     <height>171</height>
    </rect>
   </property>
   <layout class="QGridLayout" name="gridLayout">
    <item row="1" column="0">
     <widget class="QLabel" name="Com_Baud_Label">
      <property name="text">
       <string>波特率</string>
      </property>
      <property name="alignment">
       <set>Qt::AlignCenter</set>
      </property>
     </widget>
    </item>
    <item row="4" column="1">
     <widget class="QPushButton" name="Com_Close_Button">
      <property name="text">
       <string>Close</string>
      </property>
      <property name="default">
       <bool>false</bool>
      </property>
     </widget>
    </item>
    <item row="2" column="0">
     <widget class="QLabel" name="Com_Name_Label">
      <property name="text">
       <string>串口选择</string>
      </property>
      <property name="alignment">
       <set>Qt::AlignCenter</set>
      </property>
     </widget>
    </item>
    <item row="2" column="1">
     <widget class="QComboBox" name="Com_Name_Combo"/>
    </item>
    <item row="0" column="0">
     <widget class="QLabel" name="Com_Refresh_Label">
      <property name="text">
       <string>串口搜索</string>
      </property>
      <property name="alignment">
       <set>Qt::AlignCenter</set>
      </property>
     </widget>
    </item>
    <item row="0" column="1">
     <widget class="QPushButton" name="Com_Refresh_Button">
      <property name="text">
       <string>刷新</string>
      </property>
     </widget>
    </item>
    <item row="3" column="0">
     <widget class="QLabel" name="Com_State_Label">
      <property name="text">
       <string>串口操作</string>
      </property>
      <property name="alignment">
       <set>Qt::AlignCenter</set>
      </property>
     </widget>
    </item>
    <item row="1" column="1">
     <widget class="QComboBox" name="Com_Baud_Combo">
      <property name="editable">
       <bool>true</bool>
      </property>
      <property name="currentText">
       <string>1200</string>
      </property>
      <property name="duplicatesEnabled">
       <bool>false</bool>
      </property>
      <property name="modelColumn">
       <number>0</number>
      </property>
      <item>
       <property name="text">
        <string>1200</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>2400</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>4800</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>9600</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>14400</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>19200</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>38400</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>43000</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>57600</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>76800</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>115200</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>128000</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>230400</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>256000</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>460800</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>921600</string>
       </property>
      </item>
      <item>
       <property name="text">
        <string>1382400</string>
       </property>
      </item>
     </widget>
    </item>
    <item row="3" column="1">
     <widget class="QPushButton" name="Com_Open_Button">
      <property name="text">
       <string>Open</string>
      </property>
     </widget>
    </item>
    <item row="4" column="0">
     <widget class="QLabel" name="Com_isOpenOrNot_Label">
      <property name="text">
       <string/>
      </property>
      <property name="alignment">
       <set>Qt::AlignCenter</set>
      </property>
     </widget>
    </item>
   </layout>
  </widget>
  <widget class="QCheckBox" name="hexSending_checkBox">
   <property name="geometry">
    <rect>
     <x>240</x>
     <y>345</y>
     <width>91</width>
     <height>31</height>
    </rect>
   </property>
   <property name="text">
    <string>16进制发送</string>
   </property>
  </widget>
  <widget class="QCheckBox" name="hexShowing_checkBox">
   <property name="geometry">
    <rect>
     <x>240</x>
     <y>40</y>
     <width>81</width>
     <height>20</height>
    </rect>
   </property>
   <property name="text">
    <string>16进制显示</string>
   </property>
  </widget>
  <widget class="QCalendarWidget" name="calendarWidget">
   <property name="geometry">
    <rect>
     <x>430</x>
     <y>350</y>
     <width>251</width>
     <height>181</height>
    </rect>
   </property>
   <property name="styleSheet">
    <string notr="true">alternate-background-color: rgb(0, 0, 0);
background-color: rgb(0, 0, 0);</string>
   </property>
   <property name="firstDayOfWeek">
    <enum>Qt::Sunday</enum>
   </property>
   <property name="horizontalHeaderFormat">
    <enum>QCalendarWidget::ShortDayNames</enum>
   </property>
   <property name="verticalHeaderFormat">
    <enum>QCalendarWidget::ISOWeekNumbers</enum>
   </property>
  </widget>
  <widget class="QLabel" name="Time_Label">
   <property name="geometry">
    <rect>
     <x>460</x>
     <y>310</y>
     <width>181</width>
     <height>21</height>
    </rect>
   </property>
   <property name="font">
    <font>
     <family>方正兰亭中黑_GBK</family>
     <pointsize>9</pointsize>
     <weight>50</weight>
     <italic>false</italic>
     <bold>false</bold>
    </font>
   </property>
   <property name="text">
    <string>Time</string>
   </property>
   <property name="alignment">
    <set>Qt::AlignCenter</set>
   </property>
  </widget>
  <widget class="QPushButton" name="About_Button">
   <property name="geometry">
    <rect>
     <x>440</x>
     <y>40</y>
     <width>221</width>
     <height>31</height>
    </rect>
   </property>
   <property name="font">
    <font>
     <family>方正兰亭中黑_GBK</family>
     <pointsize>9</pointsize>
     <weight>50</weight>
     <italic>false</italic>
     <bold>false</bold>
    </font>
   </property>
   <property name="text">
    <string>Made by PyQt5 - 查看源代码</string>
   </property>
  </widget>
 </widget>
 <resources/>
 <connections>
  <connection>
   <sender>ClearButton</sender>
   <signal>clicked()</signal>
   <receiver>textEdit_Recive</receiver>
   <slot>clear()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>367</x>
     <y>51</y>
    </hint>
    <hint type="destinationlabel">
     <x>368</x>
     <y>79</y>
    </hint>
   </hints>
  </connection>
 </connections>
</ui>

```

- **主程序逻辑**

```python
import re
import sys
import binascii
import time
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtWidgets import *
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import *
from Ui_SerialPort import Ui_Form
from PyQt5.QtCore import QDate
class MyMainWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # 设置实例
        self.CreateItems()
        # 设置信号与槽
        self.CreateSignalSlot()
    # 设置实例 
    def CreateItems(self):
        # Qt 串口类
        self.com = QSerialPort()
        # Qt 定时器类
        self.timer = QTimer(self) #初始化一个定时器
        self.timer.timeout.connect(self.ShowTime) #计时结束调用operate()方法
        self.timer.start(100) #设置计时间隔 100ms 并启动
    # 设置信号与槽
    def CreateSignalSlot(self):
        self.Com_Open_Button.clicked.connect(self.Com_Open_Button_clicked) 
        self.Com_Close_Button.clicked.connect(self.Com_Close_Button_clicked) 
        self.Send_Button.clicked.connect(self.SendButton_clicked) 
        self.Com_Refresh_Button.clicked.connect(self.Com_Refresh_Button_Clicked) 
        self.com.readyRead.connect(self.Com_Receive_Data) # 接收数据
        self.hexSending_checkBox.stateChanged.connect(self.hexShowingClicked)
        self.hexSending_checkBox.stateChanged.connect(self.hexSendingClicked)
        self.About_Button.clicked.connect(self.Goto_GitHub) 
    # 跳转到 GitHub 查看源代码
    def Goto_GitHub(self):
        self.browser = QWebEngineView()
        self.browser.load(QUrl('https://github.com/Oslomayor/PyQt5-SerialPort-Stable'))
        self.setCentralWidget(self.browser)
    # 显示时间
    def ShowTime(self):
        self.Time_Label.setText(time.strftime("%B %d, %H:%M:%S", time.localtime())) 
    # 串口发送数据
    def Com_Send_Data(self):
        txData = self.textEdit_Send.toPlainText()
        if len(txData) == 0 :
            return
        if self.hexSending_checkBox.isChecked() == False:
            self.com.write(txData.encode('UTF-8'))
        else:
            Data = txData.replace(' ', '')
            # 如果16进制不是偶数个字符, 去掉最后一个, [ ]左闭右开
            if len(Data)%2 == 1:
                Data = Data[0:len(Data)-1]
            # 如果遇到非16进制字符
            if Data.isalnum() is False:
                QMessageBox.critical(self, '错误', '包含非十六进制数')
            try:
                hexData = binascii.a2b_hex(Data)
            except:
                QMessageBox.critical(self, '错误', '转换编码错误')
                return
            # 发送16进制数据, 发送格式如 ‘31 32 33 41 42 43’, 代表'123ABC'
            try:
                self.com.write(hexData) 
            except:
                QMessageBox.critical(self, '异常', '十六进制发送错误')
                return
    # 串口接收数据
    def Com_Receive_Data(self):
        
        try:
            rxData = bytes(self.com.readAll())
        except:
            QMessageBox.critical(self, '严重错误', '串口接收数据错误')
        if self.hexShowing_checkBox.isChecked() == False :
            try:
                self.textEdit_Recive.insertPlainText(rxData.decode('UTF-8'))
            except:
                pass
        else :
            Data = binascii.b2a_hex(rxData).decode('ascii')
            # re 正则表达式 (.{2}) 匹配两个字母
            hexStr = ' 0x'.join(re.findall('(.{2})', Data))
            # 补齐第一个 0x
            hexStr = '0x' + hexStr
            self.textEdit_Recive.insertPlainText(hexStr)
            self.textEdit_Recive.insertPlainText(' ')     
    # 串口刷新
    def Com_Refresh_Button_Clicked(self):
        self.Com_Name_Combo.clear()
        com = QSerialPort()
        com_list = QSerialPortInfo.availablePorts()
        for info in com_list:
            com.setPort(info)
            if com.open(QSerialPort.ReadWrite):
                self.Com_Name_Combo.addItem(info.portName())
                com.close()
    # 16进制显示按下   
    def hexShowingClicked(self):
        if self.hexShowing_checkBox.isChecked() == True:
            # 接收区换行
            self.textEdit_Recive.insertPlainText('\n')
    # 16进制发送按下   
    def hexSendingClicked(self):
        if self.hexSending_checkBox.isChecked() == True:
            pass
    # 发送按钮按下
    def SendButton_clicked(self):
        self.Com_Send_Data()
    # 串口刷新按钮按下
    def Com_Open_Button_clicked(self):
        #### com Open Code here ####
        comName = self.Com_Name_Combo.currentText()
        comBaud = int(self.Com_Baud_Combo.currentText())
        self.com.setPortName(comName)
        try:
            if self.com.open(QSerialPort.ReadWrite) == False:
                QMessageBox.critical(self, '严重错误', '串口打开失败')
                return
        except:
            QMessageBox.critical(self, '严重错误', '串口打开失败')
            return
        self.Com_Close_Button.setEnabled(True)
        self.Com_Open_Button.setEnabled(False)
        self.Com_Refresh_Button.setEnabled(False)
        self.Com_Name_Combo.setEnabled(False)
        self.Com_Baud_Combo.setEnabled(False)
        self.Com_isOpenOrNot_Label.setText('  已打开')
        self.com.setBaudRate(comBaud)
    def Com_Close_Button_clicked(self):
        self.com.close()
        self.Com_Close_Button.setEnabled(False)
        self.Com_Open_Button.setEnabled(True)
        self.Com_Refresh_Button.setEnabled(True)
        self.Com_Name_Combo.setEnabled(True)
        self.Com_Baud_Combo.setEnabled(True)
        self.Com_isOpenOrNot_Label.setText('  已关闭')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210125221227180.png)

#### 3.2. Qt-serial

- UI_file

```python
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo_1.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(707, 458)
        self.formGroupBox = QtWidgets.QGroupBox(Form)
        self.formGroupBox.setGeometry(QtCore.QRect(20, 20, 167, 301))
        self.formGroupBox.setObjectName("formGroupBox")
        self.formLayout = QtWidgets.QFormLayout(self.formGroupBox)
        self.formLayout.setContentsMargins(10, 10, 10, 10)
        self.formLayout.setSpacing(10)
        self.formLayout.setObjectName("formLayout")
        self.s1__lb_1 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_1.setObjectName("s1__lb_1")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.s1__lb_1)
        self.s1__box_1 = QtWidgets.QPushButton(self.formGroupBox)
        self.s1__box_1.setAutoRepeatInterval(100)
        self.s1__box_1.setDefault(True)
        self.s1__box_1.setObjectName("s1__box_1")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.s1__box_1)
        self.s1__lb_2 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_2.setObjectName("s1__lb_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.s1__lb_2)
        self.s1__box_2 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_2.setObjectName("s1__box_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.s1__box_2)
        self.s1__lb_3 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_3.setObjectName("s1__lb_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.s1__lb_3)
        self.s1__box_3 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_3.setObjectName("s1__box_3")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.s1__box_3)
        self.s1__lb_4 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_4.setObjectName("s1__lb_4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.s1__lb_4)
        self.s1__box_4 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_4.setObjectName("s1__box_4")
        self.s1__box_4.addItem("")
        self.s1__box_4.addItem("")
        self.s1__box_4.addItem("")
        self.s1__box_4.addItem("")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.s1__box_4)
        self.s1__lb_5 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_5.setObjectName("s1__lb_5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.s1__lb_5)
        self.s1__box_5 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_5.setObjectName("s1__box_5")
        self.s1__box_5.addItem("")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.s1__box_5)
        self.open_button = QtWidgets.QPushButton(self.formGroupBox)
        self.open_button.setObjectName("open_button")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.SpanningRole, self.open_button)
        self.close_button = QtWidgets.QPushButton(self.formGroupBox)
        self.close_button.setObjectName("close_button")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.SpanningRole, self.close_button)
        self.s1__lb_6 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_6.setObjectName("s1__lb_6")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.s1__lb_6)
        self.s1__box_6 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_6.setObjectName("s1__box_6")
        self.s1__box_6.addItem("")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.s1__box_6)
        self.state_label = QtWidgets.QLabel(self.formGroupBox)
        self.state_label.setText("")
        self.state_label.setTextFormat(QtCore.Qt.AutoText)
        self.state_label.setScaledContents(True)
        self.state_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.state_label.setObjectName("state_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.state_label)
        self.verticalGroupBox = QtWidgets.QGroupBox(Form)
        self.verticalGroupBox.setGeometry(QtCore.QRect(210, 20, 401, 241))
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.s2__receive_text = QtWidgets.QTextBrowser(self.verticalGroupBox)
        self.s2__receive_text.setObjectName("s2__receive_text")
        self.verticalLayout.addWidget(self.s2__receive_text)
        self.verticalGroupBox_2 = QtWidgets.QGroupBox(Form)
        self.verticalGroupBox_2.setGeometry(QtCore.QRect(210, 280, 401, 101))
        self.verticalGroupBox_2.setObjectName("verticalGroupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalGroupBox_2)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.s3__send_text = QtWidgets.QTextEdit(self.verticalGroupBox_2)
        self.s3__send_text.setObjectName("s3__send_text")
        self.verticalLayout_2.addWidget(self.s3__send_text)
        self.s3__send_button = QtWidgets.QPushButton(Form)
        self.s3__send_button.setGeometry(QtCore.QRect(620, 310, 61, 31))
        self.s3__send_button.setObjectName("s3__send_button")
        self.s3__clear_button = QtWidgets.QPushButton(Form)
        self.s3__clear_button.setGeometry(QtCore.QRect(620, 350, 61, 31))
        self.s3__clear_button.setObjectName("s3__clear_button")
        self.formGroupBox1 = QtWidgets.QGroupBox(Form)
        self.formGroupBox1.setGeometry(QtCore.QRect(20, 340, 171, 101))
        self.formGroupBox1.setObjectName("formGroupBox1")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formGroupBox1)
        self.formLayout_2.setContentsMargins(10, 10, 10, 10)
        self.formLayout_2.setSpacing(10)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(self.formGroupBox1)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.formGroupBox1)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.formGroupBox1)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formGroupBox1)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.hex_send = QtWidgets.QCheckBox(Form)
        self.hex_send.setGeometry(QtCore.QRect(620, 280, 71, 16))
        self.hex_send.setObjectName("hex_send")
        self.hex_receive = QtWidgets.QCheckBox(Form)
        self.hex_receive.setGeometry(QtCore.QRect(620, 40, 71, 16))
        self.hex_receive.setObjectName("hex_receive")
        self.s2__clear_button = QtWidgets.QPushButton(Form)
        self.s2__clear_button.setGeometry(QtCore.QRect(620, 80, 61, 31))
        self.s2__clear_button.setObjectName("s2__clear_button")
        self.timer_send_cb = QtWidgets.QCheckBox(Form)
        self.timer_send_cb.setGeometry(QtCore.QRect(260, 390, 71, 16))
        self.timer_send_cb.setObjectName("timer_send_cb")
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setGeometry(QtCore.QRect(350, 390, 61, 20))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.dw = QtWidgets.QLabel(Form)
        self.dw.setGeometry(QtCore.QRect(420, 390, 54, 20))
        self.dw.setObjectName("dw")
        self.verticalGroupBox.raise_()
        self.verticalGroupBox_2.raise_()
        self.formGroupBox.raise_()
        self.s3__send_button.raise_()
        self.s3__clear_button.raise_()
        self.formGroupBox.raise_()
        self.hex_send.raise_()
        self.hex_receive.raise_()
        self.s2__clear_button.raise_()
        self.timer_send_cb.raise_()
        self.lineEdit_3.raise_()
        self.dw.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.formGroupBox.setTitle(_translate("Form", "串口设置"))
        self.s1__lb_1.setText(_translate("Form", "串口检测："))
        self.s1__box_1.setText(_translate("Form", "检测串口"))
        self.s1__lb_2.setText(_translate("Form", "串口选择："))
        self.s1__lb_3.setText(_translate("Form", "波特率："))
        self.s1__box_3.setItemText(0, _translate("Form", "115200"))
        self.s1__box_3.setItemText(1, _translate("Form", "2400"))
        self.s1__box_3.setItemText(2, _translate("Form", "4800"))
        self.s1__box_3.setItemText(3, _translate("Form", "9600"))
        self.s1__box_3.setItemText(4, _translate("Form", "14400"))
        self.s1__box_3.setItemText(5, _translate("Form", "19200"))
        self.s1__box_3.setItemText(6, _translate("Form", "38400"))
        self.s1__box_3.setItemText(7, _translate("Form", "57600"))
        self.s1__box_3.setItemText(8, _translate("Form", "76800"))
        self.s1__box_3.setItemText(9, _translate("Form", "12800"))
        self.s1__box_3.setItemText(10, _translate("Form", "230400"))
        self.s1__box_3.setItemText(11, _translate("Form", "460800"))
        self.s1__lb_4.setText(_translate("Form", "数据位："))
        self.s1__box_4.setItemText(0, _translate("Form", "8"))
        self.s1__box_4.setItemText(1, _translate("Form", "7"))
        self.s1__box_4.setItemText(2, _translate("Form", "6"))
        self.s1__box_4.setItemText(3, _translate("Form", "5"))
        self.s1__lb_5.setText(_translate("Form", "校验位："))
        self.s1__box_5.setItemText(0, _translate("Form", "N"))
        self.open_button.setText(_translate("Form", "打开串口"))
        self.close_button.setText(_translate("Form", "关闭串口"))
        self.s1__lb_6.setText(_translate("Form", "停止位："))
        self.s1__box_6.setItemText(0, _translate("Form", "1"))
        self.verticalGroupBox.setTitle(_translate("Form", "接受区"))
        self.verticalGroupBox_2.setTitle(_translate("Form", "发送区"))
        self.s3__send_text.setHtml(_translate("Form",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">123456</p></body></html>"))
        self.s3__send_button.setText(_translate("Form", "发送"))
        self.s3__clear_button.setText(_translate("Form", "清除"))
        self.formGroupBox1.setTitle(_translate("Form", "串口状态"))
        self.label.setText(_translate("Form", "已接收："))
        self.label_2.setText(_translate("Form", "已发送："))
        self.hex_send.setText(_translate("Form", "Hex发送"))
        self.hex_receive.setText(_translate("Form", "Hex接收"))
        self.s2__clear_button.setText(_translate("Form", "清除"))
        self.timer_send_cb.setText(_translate("Form", "定时发送"))
        self.lineEdit_3.setText(_translate("Form", "1000"))
        self.dw.setText(_translate("Form", "ms/次"))
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo_1.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(707, 458)
        self.formGroupBox = QtWidgets.QGroupBox(Form)
        self.formGroupBox.setGeometry(QtCore.QRect(20, 20, 167, 301))
        self.formGroupBox.setObjectName("formGroupBox")
        self.formLayout = QtWidgets.QFormLayout(self.formGroupBox)
        self.formLayout.setContentsMargins(10, 10, 10, 10)
        self.formLayout.setSpacing(10)
        self.formLayout.setObjectName("formLayout")
        self.s1__lb_1 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_1.setObjectName("s1__lb_1")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.s1__lb_1)
        self.s1__box_1 = QtWidgets.QPushButton(self.formGroupBox)
        self.s1__box_1.setAutoRepeatInterval(100)
        self.s1__box_1.setDefault(True)
        self.s1__box_1.setObjectName("s1__box_1")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.s1__box_1)
        self.s1__lb_2 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_2.setObjectName("s1__lb_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.s1__lb_2)
        self.s1__box_2 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_2.setObjectName("s1__box_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.s1__box_2)
        self.s1__lb_3 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_3.setObjectName("s1__lb_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.s1__lb_3)
        self.s1__box_3 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_3.setObjectName("s1__box_3")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.s1__box_3.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.s1__box_3)
        self.s1__lb_4 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_4.setObjectName("s1__lb_4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.s1__lb_4)
        self.s1__box_4 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_4.setObjectName("s1__box_4")
        self.s1__box_4.addItem("")
        self.s1__box_4.addItem("")
        self.s1__box_4.addItem("")
        self.s1__box_4.addItem("")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.s1__box_4)
        self.s1__lb_5 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_5.setObjectName("s1__lb_5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.s1__lb_5)
        self.s1__box_5 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_5.setObjectName("s1__box_5")
        self.s1__box_5.addItem("")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.s1__box_5)
        self.open_button = QtWidgets.QPushButton(self.formGroupBox)
        self.open_button.setObjectName("open_button")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.SpanningRole, self.open_button)
        self.close_button = QtWidgets.QPushButton(self.formGroupBox)
        self.close_button.setObjectName("close_button")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.SpanningRole, self.close_button)
        self.s1__lb_6 = QtWidgets.QLabel(self.formGroupBox)
        self.s1__lb_6.setObjectName("s1__lb_6")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.s1__lb_6)
        self.s1__box_6 = QtWidgets.QComboBox(self.formGroupBox)
        self.s1__box_6.setObjectName("s1__box_6")
        self.s1__box_6.addItem("")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.s1__box_6)
        self.state_label = QtWidgets.QLabel(self.formGroupBox)
        self.state_label.setText("")
        self.state_label.setTextFormat(QtCore.Qt.AutoText)
        self.state_label.setScaledContents(True)
        self.state_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.state_label.setObjectName("state_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.state_label)
        self.verticalGroupBox = QtWidgets.QGroupBox(Form)
        self.verticalGroupBox.setGeometry(QtCore.QRect(210, 20, 401, 241))
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.s2__receive_text = QtWidgets.QTextBrowser(self.verticalGroupBox)
        self.s2__receive_text.setObjectName("s2__receive_text")
        self.verticalLayout.addWidget(self.s2__receive_text)
        self.verticalGroupBox_2 = QtWidgets.QGroupBox(Form)
        self.verticalGroupBox_2.setGeometry(QtCore.QRect(210, 280, 401, 101))
        self.verticalGroupBox_2.setObjectName("verticalGroupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalGroupBox_2)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.s3__send_text = QtWidgets.QTextEdit(self.verticalGroupBox_2)
        self.s3__send_text.setObjectName("s3__send_text")
        self.verticalLayout_2.addWidget(self.s3__send_text)
        self.s3__send_button = QtWidgets.QPushButton(Form)
        self.s3__send_button.setGeometry(QtCore.QRect(620, 310, 61, 31))
        self.s3__send_button.setObjectName("s3__send_button")
        self.s3__clear_button = QtWidgets.QPushButton(Form)
        self.s3__clear_button.setGeometry(QtCore.QRect(620, 350, 61, 31))
        self.s3__clear_button.setObjectName("s3__clear_button")
        self.formGroupBox1 = QtWidgets.QGroupBox(Form)
        self.formGroupBox1.setGeometry(QtCore.QRect(20, 340, 171, 101))
        self.formGroupBox1.setObjectName("formGroupBox1")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formGroupBox1)
        self.formLayout_2.setContentsMargins(10, 10, 10, 10)
        self.formLayout_2.setSpacing(10)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(self.formGroupBox1)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.formGroupBox1)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.formGroupBox1)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formGroupBox1)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.hex_send = QtWidgets.QCheckBox(Form)
        self.hex_send.setGeometry(QtCore.QRect(620, 280, 71, 16))
        self.hex_send.setObjectName("hex_send")
        self.hex_receive = QtWidgets.QCheckBox(Form)
        self.hex_receive.setGeometry(QtCore.QRect(620, 40, 71, 16))
        self.hex_receive.setObjectName("hex_receive")
        self.s2__clear_button = QtWidgets.QPushButton(Form)
        self.s2__clear_button.setGeometry(QtCore.QRect(620, 80, 61, 31))
        self.s2__clear_button.setObjectName("s2__clear_button")
        self.timer_send_cb = QtWidgets.QCheckBox(Form)
        self.timer_send_cb.setGeometry(QtCore.QRect(260, 390, 71, 16))
        self.timer_send_cb.setObjectName("timer_send_cb")
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setGeometry(QtCore.QRect(350, 390, 61, 20))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.dw = QtWidgets.QLabel(Form)
        self.dw.setGeometry(QtCore.QRect(420, 390, 54, 20))
        self.dw.setObjectName("dw")
        self.verticalGroupBox.raise_()
        self.verticalGroupBox_2.raise_()
        self.formGroupBox.raise_()
        self.s3__send_button.raise_()
        self.s3__clear_button.raise_()
        self.formGroupBox.raise_()
        self.hex_send.raise_()
        self.hex_receive.raise_()
        self.s2__clear_button.raise_()
        self.timer_send_cb.raise_()
        self.lineEdit_3.raise_()
        self.dw.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.formGroupBox.setTitle(_translate("Form", "串口设置"))
        self.s1__lb_1.setText(_translate("Form", "串口检测："))
        self.s1__box_1.setText(_translate("Form", "检测串口"))
        self.s1__lb_2.setText(_translate("Form", "串口选择："))
        self.s1__lb_3.setText(_translate("Form", "波特率："))
        self.s1__box_3.setItemText(0, _translate("Form", "115200"))
        self.s1__box_3.setItemText(1, _translate("Form", "2400"))
        self.s1__box_3.setItemText(2, _translate("Form", "4800"))
        self.s1__box_3.setItemText(3, _translate("Form", "9600"))
        self.s1__box_3.setItemText(4, _translate("Form", "14400"))
        self.s1__box_3.setItemText(5, _translate("Form", "19200"))
        self.s1__box_3.setItemText(6, _translate("Form", "38400"))
        self.s1__box_3.setItemText(7, _translate("Form", "57600"))
        self.s1__box_3.setItemText(8, _translate("Form", "76800"))
        self.s1__box_3.setItemText(9, _translate("Form", "12800"))
        self.s1__box_3.setItemText(10, _translate("Form", "230400"))
        self.s1__box_3.setItemText(11, _translate("Form", "460800"))
        self.s1__lb_4.setText(_translate("Form", "数据位："))
        self.s1__box_4.setItemText(0, _translate("Form", "8"))
        self.s1__box_4.setItemText(1, _translate("Form", "7"))
        self.s1__box_4.setItemText(2, _translate("Form", "6"))
        self.s1__box_4.setItemText(3, _translate("Form", "5"))
        self.s1__lb_5.setText(_translate("Form", "校验位："))
        self.s1__box_5.setItemText(0, _translate("Form", "N"))
        self.open_button.setText(_translate("Form", "打开串口"))
        self.close_button.setText(_translate("Form", "关闭串口"))
        self.s1__lb_6.setText(_translate("Form", "停止位："))
        self.s1__box_6.setItemText(0, _translate("Form", "1"))
        self.verticalGroupBox.setTitle(_translate("Form", "接受区"))
        self.verticalGroupBox_2.setTitle(_translate("Form", "发送区"))
        self.s3__send_text.setHtml(_translate("Form",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">123456</p></body></html>"))
        self.s3__send_button.setText(_translate("Form", "发送"))
        self.s3__clear_button.setText(_translate("Form", "清除"))
        self.formGroupBox1.setTitle(_translate("Form", "串口状态"))
        self.label.setText(_translate("Form", "已接收："))
        self.label_2.setText(_translate("Form", "已发送："))
        self.hex_send.setText(_translate("Form", "Hex发送"))
        self.hex_receive.setText(_translate("Form", "Hex接收"))
        self.s2__clear_button.setText(_translate("Form", "清除"))
        self.timer_send_cb.setText(_translate("Form", "定时发送"))
        self.lineEdit_3.setText(_translate("Form", "1000"))
        self.dw.setText(_translate("Form", "ms/次"))

```

- pyserial_logic

```python
import sys
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer
from ui_demo_1 import Ui_Form


class Pyqt5_Serial(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(Pyqt5_Serial, self).__init__()
        self.setupUi(self)
        self.init()
        self.setWindowTitle("串口小助手")
        self.ser = serial.Serial()
        self.port_check()

        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))

    def init(self):
        # 串口检测按钮
        self.s1__box_1.clicked.connect(self.port_check)

        # 串口信息显示
        self.s1__box_2.currentTextChanged.connect(self.port_imf)

        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)

        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)

        # 发送数据按钮
        self.s3__send_button.clicked.connect(self.data_send)

        # 定时发送数据
        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer)

        # 定时器接收数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)

        # 清除发送窗口
        self.s3__clear_button.clicked.connect(self.send_data_clear)

        # 清除接收窗口
        self.s2__clear_button.clicked.connect(self.receive_data_clear)

    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.s1__box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")

    # 串口信息
    def port_imf(self):
        # 显示选定的串口的详细信息
        imf_s = self.s1__box_2.currentText()
        if imf_s != "":
            self.state_label.setText(self.Com_Dict[self.s1__box_2.currentText()])

    # 打开串口
    def port_open(self):
        self.ser.port = self.s1__box_2.currentText()
        self.ser.baudrate = int(self.s1__box_3.currentText())
        self.ser.bytesize = int(self.s1__box_4.currentText())
        self.ser.stopbits = int(self.s1__box_6.currentText())
        self.ser.parity = self.s1__box_5.currentText()

        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None

        # 打开串口接收定时器，周期为2ms
        self.timer.start(2)

        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
            self.formGroupBox1.setTitle("串口状态（已开启）")

    # 关闭串口
    def port_close(self):
        self.timer.stop()
        self.timer_send.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        self.formGroupBox1.setTitle("串口状态（已关闭）")

    # 发送数据
    def data_send(self):
        if self.ser.isOpen():
            input_s = self.s3__send_text.toPlainText()
            if input_s != "":
                # 非空字符串
                if self.hex_send.isChecked():
                    # hex发送
                    input_s = input_s.strip()
                    send_list = []
                    while input_s != '':
                        try:
                            num = int(input_s[0:2], 16)
                        except ValueError:
                            QMessageBox.critical(self, 'wrong data', '请输入十六进制数据，以空格分开!')
                            return None
                        input_s = input_s[2:].strip()
                        send_list.append(num)
                    input_s = bytes(send_list)
                else:
                    # ascii发送
                    input_s = (input_s + '\r\n').encode('utf-8')
                
                num = self.ser.write(input_s)
                self.data_num_sended += num
                self.lineEdit_2.setText(str(self.data_num_sended))
        else:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            # hex显示
            if self.hex_receive.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                self.s2__receive_text.insertPlainText(out_s)
            else:
                # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
                self.s2__receive_text.insertPlainText(data.decode('iso-8859-1'))

            # 统计接收字符的数量
            self.data_num_received += num
            self.lineEdit.setText(str(self.data_num_received))

            # 获取到text光标
            textCursor = self.s2__receive_text.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去
            self.s2__receive_text.setTextCursor(textCursor)
        else:
            pass

    # 定时发送数据
    def data_send_timer(self):
        if self.timer_send_cb.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
        else:
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)

    # 清除显示
    def send_data_clear(self):
        self.s3__send_text.setText("")

    def receive_data_clear(self):
        self.s2__receive_text.setText("")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = Pyqt5_Serial()
    myshow.show()
    sys.exit(app.exec_())

```

### 4. 驱动下载

- 使用驱动精灵工具；
- 推荐使用[dirverIdentifier](https://www.driveridentifier.com/scan/usb20-serial-driver/download/1053480298/D8E58274B15844AB899DA2299AE9F790/USB%5CVID_1A86%26PID_7523)软件，会自动检测系统设备管理器上各种驱动问题，然后提供对应的驱动安装口；

### 5.  学习链接

- https://github.com/Oslomayor/PyQt5-SerialPort-Stable
- [几种串口工具](https://download.csdn.net/download/qq_38258510/10751762?utm_medium=distribute.pc_relevant.none-task-download-BlogCommendFromMachineLearnPai2-11.control&depth_1-utm_source=distribute.pc_relevant.none-task-download-BlogCommendFromMachineLearnPai2-11.control)

### 6. 问题记录

>serial.serialutil.SerialException: WriteFile failed (PermissionError(13, '拒绝访问。', None, 5))
>
>- `从电源关闭设备`；



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/serialop/  

