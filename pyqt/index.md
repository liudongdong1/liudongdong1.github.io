# pyQt


#### 1. 安装

```shell
pip install PyQt5
pip install PyQt5-tools
```

```python
#安装测试
import sys
from PyQt5 import QtWidgets, QtCore

app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QWidget()
widget.resize(400, 400)
widget.setWindowTitle('Hello World')
widget.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210122074434357.png)

![settings.json set QT folder](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210201072322427.png)

#### 2. 界面设计

使用qt designer 或者使用vscode集成的qt designer 进行ui控件等布局，正对具体的控件属性布局使用css样式， 右击控件，样式，添加css代码。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210124192752484.png)

##### 2.0. 窗口控件

- **QMainWindow**：  包含菜单栏，工具栏，状态栏，标题栏等，是GUI程序的主窗口。

| 方法               | 描述                               |
| :----------------- | :--------------------------------- |
| addToolBar()       | 添加工具栏                         |
| centralWidget()    | 返回窗口中心的控件，未设置返回NULL |
| menuBar()          | 返回主窗口的菜单栏                 |
| setCentralWidget() | 设置窗口中心的控件                 |
| setStatusBar()     | 设置状态栏                         |
| statusBar()        | 获取状态栏对象                     |

- **QDialog**：  对话框窗口的基类，对话框一般用来执行短期任务，或者与用户进行互动，它可以是模态的也可以是非模态的。QDialog没有菜单栏，工具栏，状态栏等。
- **QWidget**： 作为QMainWindow和QWidget的父类，并未细化到主窗口或者对话框，作为通用窗口类，如果不确定具体使用哪种窗口类，就可以使用该类。
- **自适应布局：** Qt中如果想实现`窗体内空间随着窗体大小调整`，必须使用布局管理，常用的布局管理有QHBoxLayout、QVBoxLayout、QGridLayout，空的地方使用spacer控件进行填充。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126111321167.png)

#### 3. python 使用ui布局文件

##### 3.1. 通过`uic.loadUi('ui_files/show.ui',self)`

```python
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QTimer,Qt 
from PyQt5.QtWidgets import QMessageBox
class Dashboard(QMainWindow):
    def __init__(self):
        super(Dashboard, self).__init__()
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.FramelessWindowHint)
        uic.loadUi('ui_files/show.ui',self)
    def quitApplication(self):
        """shutsdown the GUI window along with removal of files"""
        userReply = QMessageBox.question(self, 'Quit Application', "Are you sure you want to quit this app?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if userReply == QMessageBox.Yes:
            #removeFile()
            keyboard.press_and_release('alt+F4')
app = QtWidgets.QApplication([])
win = Dashboard()
win.show()
sys.exit(app.exec())
```

##### 3.2. 通过python类

```python
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 80, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(150, 140, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(150, 200, 113, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_3.setText(_translate("MainWindow", "PushButton"))
```

```python
# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from mainwindow import Ui_MainWindow
class MainUI(Ui_MainWindow, QMainWindow):
    def __init__(self, parent=None):
        super(MainUI, self).__init__(parent)
        self.registerEvent()
    def registerEvent(self):
        self.pushButton.clicked.connect(self.on_btn_click)
    def on_btn_click(self):
        print("点击按钮")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainUI()
    main.show()
    sys.exec(app.exec_())
```



#### 4. slot 触发事件

##### 4.1. QTimer

> QTimer类提供了重复和单次的定时器，要使用定时器，需要先创建一个QTimer实例，将其Timeout信号连接到槽函数，并调用start（），然后，定时器，会以恒定的间隔发出timeout信号。当窗口的控件收到Timeout信号后，他就会停止这个定时器，这是在图形用户界面中实现复杂工作的一个典型用法，随着技术的进步，多线程在越来越多的平台上被使用，QTimer对象会被替代掉。

| 方法                | 描述                                                         |
| :------------------ | :----------------------------------------------------------- |
| start(milliseconds) | 启动或重新启动定时器，时间间隔为毫秒，如果定时器已经运行，他将停止并重新启动，如果singleSlot信号为真，定时器仅被激活一次 |
| Stop()              | 停止定时器                                                   |

| 信号       | 描述                                         |
| :--------- | :------------------------------------------- |
| singleShot | 在给定的时间间隔后调用一个槽函数时发射此信号 |
| timeout    | 当定时器超时时发射此信号                     |

```python
#定时显示时间
self.timer = QTimer(self) #初始化一个定时器
self.timer.timeout.connect(self.ShowTime) #计时结束调用operate()方法
self.timer.start(100) #设置计时间隔 100ms 并启动

self.Time_Label.setText(time.strftime("%B %d, %H:%M:%S", time.localtime()))
```

#### 5. Qt class

> 注意： 在QDialog的派生类中，添加Layout，可在创建Layout对象的同时指定其父窗口；但这在QMainWindow中行不通。基于主窗口的程序，默认已经有了自己的布局管理器。QMainWindow的中心控件是一个QWidget，可以通过setCentralWidget设置。若想在QMainWindow中添加 layout,需要通过将该Layout添加到一个QWidget对象中，然后将该布局设置为该空间的布局，最后设置该控件为QMainWindow的中心控件

```python
#创建部件(widget).
#创建布局(layout), 并将部件依次添加到布局中。
#创建中心部件(central widget), 并为中心部件添加布局。
def initUI(self):    
    # create new buttons
    self.btn_left = QPushButton('left', self)
    self.btn_right = QPushButton('right', self)
    # setting up a layout
    main_layout = QHBoxLayout()
    main_layout.addWidget(self.btn_left)
    main_layout.addWidget(self.btn_right)
    # create the central widget
    main_widget = QWidget()
    main_widget.setLayout(main_layout)
    self.setCentralWidget(main_widget)
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126222423686.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126222221780.png)

> QWidget类是所有用户界面对象的基类。
>
> Widget是用户界面最基础的原子，它接收鼠标、键盘产生的事件，然后回应。
>
> 一个没有嵌入到其他Widget中的Widget称为window

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126212855842.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210126212916447.png)

##### 5.1. QMovie

```python
movie = QtGui.QMovie("icons/dashAnimation.gif")
self.label_2.setMovie(movie)
self.label_2.setGeometry(160,160,750,421)
movie.start()
```

##### 5.2. QCursor

```python
self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
self.exp2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
```

##### 5.3. PlainTextEdit

```python
self.plainTextEdit.setPlaceholderText("Enter Gesture label") 
self.plainTextEdit.insertPlainText(hexStr)
self.plainTextEdit.toPlainText().strip()
```

##### 5.4.  RadioButton

```python
self.radioButton.isChecked()
```

##### 5.5. QWebEngineView

> 一款基于chrome浏览器内核引擎，Qt webenginewidgets模块中提供了QWebEngineView这个视图控件来很方便的加载和显示网页。

```python
self.browser = QWebEngineView()
self.browser.load(QUrl('https://github.com/Oslomayor/PyQt5-SerialPort-Stable'))
self.setCentralWidget(self.browser)
```

##### 5.6. QMessageBox

```python
QMessageBox().information()   # 通知消息
QMessageBox().question()   # 询问消息
QMessageBox().warning()   # 警告消息

mb = QMessageBox(self)
mb.setText('<h2>文件已被修改</h2>'
           '<h3>是否保存</h3>')
mb.setCheckBox(QCheckBox('下次不再提示',mb))

mb.setInformativeText('点击是保存文件')
mb.setDetailedText('<h3>详情文本</h3>')
mb.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210125223028646.png)

> QMessageBox().question(None, "询问", "确认删除？", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
>
> 参数一，有self用self，没有的话用None；
> 参数二，标题；
> 参数三，内容；
> 参数四，按钮，可以有多个，用|括起来，已知的还有 QMessageBox.Canel，QMessageBox.Close 等；
> 参数四，如果关闭的话，返回的值。
> **返回值是对象**，形如QMessageBox.Ok。
> 可以用 if(a == QMessageBox.Ok) 语句判断。

##### 5.7.  QCalendarWidget

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210125221103165.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210718090324044.png)

```c++
# 左右两边的箭头
QToolButton *prevBtn = calendar->findChild<QToolButton*>(QLatin1String("qt_calendar_prevmonth"));

QToolButton *bextBtn = calendar->findChild<QToolButton*>(QLatin1String("qt_calendar_nextmonth"));

prevBtn->setIcon("你自己的图标");

bextBtn->setIcon("你自己的图标");
#中间白色部分
QCalendarWidget QTableView 
{
    alternate-background-color: rgb(128, 128, 128); //颜色自己可以改
}
#背景色
QCalendarWidget QTableView 
{
    alternate-background-color: rgb(128, 128, 128); //颜色自己可以改
    background-color: #2F2F3E;
}
//月份 和年份
QToolButton#qt_calendar_monthbutton,#qt_calendar_yearbutton{
    color: #9ea5a9; //修改字体颜色
    font: 9pt simHei; //也可以修改字体
}
//显示月份和年份所在的导航条
QCalendarWidget QWidget#qt_calendar_navigationbar{
    //可以自己添加一些其他设置，比如边框
    background-color: #2F2F3E;//这个一般设置渐变色比较多，可以自行修改

}
```

```css
QCalendarWidget QWidget#qt_calendar_navigationbar { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop: 0 #cccccc, stop: 1 #333333); }
QCalendarWidget QToolButton {
   
    color: white;
    font-size: 18px;
    icon-size: 30px, 30px;
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop: 0 #cccccc, stop: 1 #333333);
}
QCalendarWidget QMenu {
    width: 150px;
    left: 20px;
    color: white;
    font-size: 18px;
    background-color: rgb(100, 100, 100);
}
QCalendarWidget QSpinBox { 
    width: 150px; 
    font-size:24px; 
    color: white; 
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop: 0 #cccccc, stop: 1 #333333); 
    selection-background-color: rgb(136, 136, 136);
    selection-color: rgb(255, 255, 255);
}
QCalendarWidget QSpinBox::up-button { subcontrol-origin: border;  subcontrol-position: top right;  width:65px; }
QCalendarWidget QSpinBox::down-button {subcontrol-origin: border; subcontrol-position: bottom right;  width:65px;}
QCalendarWidget QSpinBox::up-arrow { width:56px;  height:56px; }
QCalendarWidget QSpinBox::down-arrow { width:56px;  height:56px; }
 
/* header row */
QCalendarWidget QWidget { alternate-background-color: rgb(128, 128, 128); }
 
/* normal days */
QCalendarWidget QAbstractItemView:enabled 
{
    font-size:24px;  
    color: rgb(180, 180, 180);  
    background-color: black;  
    selection-background-color: rgb(64, 64, 64); 
    selection-color: rgb(0, 255, 0); 
}
 
/* days in other months */
QCalendarWidget QAbstractItemView:disabled { color: rgb(64, 64, 64); }
```



##### 5.8.  QCheckBox

```python
if(self.checkBox.checkState() ==Qt.Checked):
    print("image clicked")
```

##### 5.9. pyqtgraph

```shell
pip install pyqtgraph
```

```python
import sys
import random
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout


class Demo(QWidget):
    def __init__(self):
        super(Demo, self).__init__()
        self.resize(600, 600)

        # 1
        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # 2
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3','d', '+', 'x', 'p', 'h', 'star'])
        r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])

        # 3
        self.pw = pg.PlotWidget(self)
        self.plot_data = self.pw.plot(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)

        # 4
        self.plot_btn = QPushButton('Replot', self)
        self.plot_btn.clicked.connect(self.plot_slot)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.pw)
        self.v_layout.addWidget(self.plot_btn)
        self.setLayout(self.v_layout)

    def plot_slot(self):
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
        r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
        self.plot_data.setData(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())
```

```python
def plotgp():
    #==========================使用g.PlotWidget进行绘图显示=======================================
    self.graphWidget = pg.PlotWidget(self.centralwidget)
    self.graphWidget.setGeometry(QtCore.QRect(480,160,611,361))
    # 设置图表标题、颜色、字体大小
    self.graphWidget.setTitle("FlexVoltage",color='008080',size='12pt')
    # 显示表格线
    self.graphWidget.showGrid(x=True, y=True)
    # 设置上下左右的label
    # 第一个参数 只能是 'left', 'bottom', 'right', or 'top'
    #self.graphWidget.setLabel("left", "voltage")
    self.graphWidget.setLabel("bottom", "timestamp")
    self.graphWidget.setBackground("#fefefe")  # 背景色
    self.curve1 = self.graphWidget.plot( pen=pg.mkPen(color='r', width=5),name="Sensor 1") # 线条颜色
    self.curve2 = self.graphWidget.plot(pen=pg.mkPen(color='b', width=5)) # 线条颜色
    self.curve3 = self.graphWidget.plot( pen=pg.mkPen(color='y', width=5)) # 线条颜色
    self.curve4 = self.graphWidget.plot( pen=pg.mkPen(color='k', width=5)) # 线条颜色
    self.curve5 = self.graphWidget.plot( pen=pg.mkPen(color='m', width=5)) # 线条颜色
    # print("set graphWidget ok")
def plotFlexData(self):
    '''
            更新curve中的数据，以折现方式显示
        '''
    index=range(0,self.handData.getLength())
    # plot data: x, y values
    self.curve1.setData(index,self.handData.A)
    self.curve2.setData(index,self.handData.B)
    self.curve3.setData(index,self.handData.C)
    self.curve4.setData(index,self.handData.D)
    self.curve5.setData(index,self.handData.E)

```



- http://www.python3.vip/tut/py/gui/pyqtgraph-1/#%E6%9B%B2%E7%BA%BF%E5%9B%BE-%E7%A4%BA%E4%BE%8B
- https://www.learnpyqt.com/tutorials/plotting-pyqtgraph/
- https://github.com/pyqtgraph/pyqtgraph
- https://github.com/learnpyqt/15-minute-apps/blob/master/currency/currency.py![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210317181134655.png)

##### 5.11  LCDNumber

```python
 def initshowSlot(self):
        # LCDNumber 时间设置
        self.__ShowColon = True #是否显示时间如[12:07]中的冒号，用于冒号的闪烁
        self.__timer = QTimer(self) #新建一个定时器
        #关联timeout信号和showTime函数，每当定时器过了指定时间间隔，就会调用showTime函数
        self.__timer.timeout.connect(self.showTime)
        
        self.__timer.start(1000) #设置定时间隔为1000ms即1s，并启动定时器
        self.lcdNumber.setNumDigits(8)
    
    def showTime(self):
        #更新时间的显示
        time = QTime.currentTime() #获取当前时间
        time_text = time.toString(Qt.DefaultLocaleLongDate) #获取HH:MM:SS格式的时间，在中国获取后是这个格式，其他国家我不知道，如果有土豪愿意送我去外国旅行的话我就可以试一试
        
        #冒号闪烁
        if self.__ShowColon == True:
            self.__ShowColon = False
        else:
            time_text = time_text.replace(':',' ')
            self.__ShowColon = True
        self.lcdNumber.display(time_text) #显示时间
```

##### 5.12. 串口操作

```c++
from PyQt5.QtCore import QTimer
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QTimer
class FlexSensor(object):
    def __init__(self,com,bps,timeout):
        self.port = com
        self.bps = bps
        self.timeout =timeout
        self.timer=QTimer()
        global Ret     # flag: 判断是否打开，如果打开，Ret=true
        try:
            # 打开串口，并得到串口对象
            self.main_engine= serial.Serial(self.port,self.bps,timeout=self.timeout)
            # 判断是否打开成功
            if (self.main_engine.is_open):
               Ret = True
               print("打开串口成功")
        except Exception as e:
            print("---异常---：", e)
        
    
    def stop(self):
        self.timer.stop()
        #self.Close_Engine()
        return True
    
    def pause(self):
        self.timer.stop()
    
    def begin(self):
        self.timer.start(self.timeout)

    def start(self):
        self.timer.start(self.timeout)
    
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
        while True:
            data=str(self.main_engine.readline())  # bytes 数据类型和str数据格式类型转化
            #print(data,type(data))
            if "A0" in data:
                return [float(i) for i in data.split('\\')[0].split(":")[1].split(",")]

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
    Engine1 = FlexSensor("com4",9600,20)
    Engine1.start()
    while True:
        print(Engine1.Read_Line())
    '''
        b''
        b' A0:699,366,370,700,699\r\n'
    '''
    #更多示例
    # self.main_engine.write(chr(0x06).encode("utf-8"))  # 十六制发送一个数据
    # print(self.main_engine.read().hex())  #  # 十六进制的读取读一个字节
    # print(self.main_engine.read())#读一个字节
    # print(self.main_engine.read(10).decode("gbk"))#读十个字节
    # print(self.main_engine.readline().decode("gbk"))#读一行
    # print(self.main_engine.readlines())#读取多行，返回列表，必须匹配超时（timeout)使用
    # print(self.main_engine.in_waiting)#获取输入缓冲区的剩余字节数
    # print(self.main_engine.out_waiting)#获取输出缓冲区的字节数
    # print(self.main_engine.readall())#读取全部字符。
```

##### 5.13. [pyecharts](http://book.itgank.com/docs/pyecharts/pyecharts-1b32bvlfskdqn)

##### 5.14. QLabel

> 写入文本字体，图片填充等

```python
self.label_2.setText("Result: {}  Predict words: {}".format(charclass,charpre))

#样式表中： `border-image:url(:/images/bd.png) 4 4 4 4 stretch stretch;`  图片填充满label

#显示视频帧
frame = self.camera.frame
if frame is None:
    return None
height2, width2, channel2 = frame.shape
step2 = channel2 * width2
# create QImage from image
qImg2 = QImage(frame.data, width2, height2, step2, QImage.Format_RGB888)
# show image in img_label
try:
    self.label_3.setPixmap(QPixmap.fromImage(qImg2))
    except:
        pass
```

```python
from PyQt4 import QtCore, QtGui
import sys

class Example(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.initUi()

        text = "this is a test"
        self.write(text, 50)

    def initUi(self):
        self.setGeometry(300, 300, 250, 150) 
        self.show()

        self.label = QtGui.QLabel(self)
        self.label.move(120, 60)

    def write(self, text, msec):
        base = ""
        for char in text:
            base += char
            self.label.setText(base)
            QtCore.QThread.msleep(msec)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())
```

##### 5.15. matplotlib

```python
#coding:utf-8

# 导入matplotlib模块并使用Qt5Agg
import matplotlib
matplotlib.use('Qt5Agg')
# 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import *

class App(QtWidgets.QDialog):
    def __init__(self,parent=None):
        # 父类初始化方法
        super(App,self).__init__(parent)
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('PyQt5结合Matplotlib绘制函数图像')
        # 几个QWidgets
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.button_plot = QtWidgets.QPushButton("绘制函数图像")
        self.line = QLineEdit() # 输入函数
        # 连接事件
        self.button_plot.clicked.connect(self.plot_)
        
        # 设置布局
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.line)
        layout.addWidget(self.button_plot)
        self.setLayout(layout)

    # 连接的绘制的方法
    def plot_(self):
        AgeList = ['10', '21', '12', '14', '25']
        NameList = ['Tom', 'Jon', 'Alice', 'Mike', 'Mary']

        #将AgeList中的数据转化为int类型
        AgeList = list(map(int, AgeList))

        # 将x,y轴转化为矩阵式
        self.x = np.arange(len(NameList)) + 1
        self.y = np.array(AgeList)

        #tick_label后边跟x轴上的值，（可选选项：color后面跟柱型的颜色，width后边跟柱体的宽度）
        plt.bar(range(len(NameList)), AgeList, tick_label=NameList, color='green', width=0.5)

        # 在柱体上显示数据
        for a, b in zip(self.x, self.y):
            plt.text(a-1, b, '%d' % b, ha='center', va='bottom')

        #设置标题
        plt.title("Demo")
		
		#画图
        self.canvas.draw()
        # 保存画出来的图片
        plt.savefig('1.jpg')

# 运行程序
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = App()
    main_window.show()
    app.exec()
```

##### 5.16. Qtchart

```python
import functools
import random

from PyQt5.QtChart import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Widget(QWidget):

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)

        self.m_donuts = []

        self.chartView = QChartView()
        self.chartView.setRenderHint(QPainter.Antialiasing)
        self.chart = self.chartView.chart()
        self.chart.legend().setVisible(False)
        self.chart.setTitle("Nested donuts demo")
        self.chart.setAnimationOptions(QChart.AllAnimations)

        minSize = 0.1
        maxSize = 0.9
        donutCount = 5

        for i in range(donutCount):
            donut = QPieSeries()
            sliceCount = random.randrange(3, 6)
            for j in range(sliceCount):
                value = random.randrange(100, 200)
                slice_ = QPieSlice(str(value), value)
                slice_.setLabelVisible(True)
                slice_.setLabelColor(Qt.white)
                slice_.setLabelPosition(QPieSlice.LabelInsideTangential)
                slice_.hovered[bool].connect(functools.partial(self.explodeSlice, slice_=slice_))
                donut.append(slice_)
                donut.setHoleSize(minSize + i * (maxSize - minSize) / donutCount)
                donut.setPieSize(minSize + (i + 1) * (maxSize - minSize) / donutCount)

            self.m_donuts.append(donut)
            self.chartView.chart().addSeries(donut)


        # create main layout
        self.mainLayout = QGridLayout(self)
        self.mainLayout.addWidget(self.chartView, 1, 1)
        self.chartView.show()
        self.setLayout(self.mainLayout)

        self.updateTimer = QTimer(self)
        self.updateTimer.timeout.connect(self.updateRotation)
        self.updateTimer.start(1250)


    def updateRotation(self):
        for donut in self.m_donuts:
            phaseShift =  random.randrange(-50, 100)
            donut.setPieStartAngle(donut.pieStartAngle() + phaseShift)
            donut.setPieEndAngle(donut.pieEndAngle() + phaseShift)


    def explodeSlice(self, exploded, slice_):
        if exploded:
            self.updateTimer.stop()
            sliceStartAngle = slice_.startAngle()
            sliceEndAngle = slice_.startAngle() + slice_.angleSpan()

            donut = slice_.series()
            seriesIndex = self.m_donuts.index(donut)
            for i in range(seriesIndex + 1, len(self.m_donuts)):
                self.m_donuts[i].setPieStartAngle(sliceEndAngle)
                self.m_donuts[i].setPieEndAngle(360 + sliceStartAngle)
        else:
            for donut in self.m_donuts:
                donut.setPieStartAngle(0)
                donut.setPieEndAngle(360)
            self.updateTimer.start()
        slice_.setExploded(exploded)


a = QApplication([])
w = Widget()
w.show()
a.exec_()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726203409320.png)

#### 6. 自定义组件

```python
# -*- coding: utf-8 -*-
# !/usr/bin/env python
import os, sys
from Qt import QtCore, QtGui, QtWidgets, QtCompat

currentDir = os.path.dirname(__file__)

#使用QMainWindow时一定要使用setCentralWidget属性
class test_01(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(test_01, self).__init__(parent)

        self.create_UI()
        self.setWindowTitle("Tool Name")  # 窗口名字

    def create_UI(self):
        self.ui = self
        self.ui.centralwidget = QtWidgets.QWidget(self)
        #整个窗口的布局控件
        self.ui.main_layout = QtWidgets.QVBoxLayout(self.ui.centralwidget)
        # self.ui.main_layout = QtWidgets.QVBoxLayout(self.ui)
        wid1 = myWid(self)
        wid2 = myWid(self)

        self.ui.main_layout.addWidget(wid1)
        self.ui.main_layout.addWidget(wid2)
        
        self.ui.setCentralWidget(self.ui.centralwidget)



class test_01(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(test_01, self).__init__(parent)

        self.create_UI()
        self.setWindowTitle("Tool Name")  # 窗口名字
        
    def create_UI(self):
        self.ui = self

        self.ui.main_layout = QtWidgets.QVBoxLayout(self.ui)
        wid1 = myWid(self)
        wid2 = myWid(self)

        self.ui.main_layout.addWidget(wid1)
        self.ui.main_layout.addWidget(wid2)


class myWid(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(myWid, self).__init__(parent)

        self.create_UI()

    def create_UI(self):

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_fill_light_RotateY = QtWidgets.QLabel(self)
        self.label_fill_light_RotateY.setMinimumSize(QtCore.QSize(200, 30))
        self.label_fill_light_RotateY.setObjectName("label_fill_light_RotateY")
        self.horizontalLayout.addWidget(self.label_fill_light_RotateY)
        self.lineEdit_fill_light_RotateY = QtWidgets.QLineEdit(self)
        self.lineEdit_fill_light_RotateY.setMinimumSize(QtCore.QSize(70, 30))
        self.lineEdit_fill_light_RotateY.setObjectName("lineEdit_fill_light_RotateY")
        self.horizontalLayout.addWidget(self.lineEdit_fill_light_RotateY)
        self.horizontalSlider_fill_light_RotateY = QtWidgets.QSlider(self)
        self.horizontalSlider_fill_light_RotateY.setMinimumSize(QtCore.QSize(300, 0))
        self.horizontalSlider_fill_light_RotateY.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_fill_light_RotateY.setObjectName("horizontalSlider_fill_light_RotateY")
        self.horizontalLayout.addWidget(self.horizontalSlider_fill_light_RotateY)
        self.verticalLayout.addLayout(self.horizontalLayout)
        
        self.label_fill_light_RotateY.setText("fill_light_RotateY")
        #给lineEdit添加自定义属性
        self.lineEdit_fill_light_RotateY.setProperty("objAttr", "fill_light_RotateY")


def main():
    app = QtWidgets.QApplication(sys.argv)
    try:
        handle.close()
        handle.deleteLater()
    except Exception:
        pass
    handle = test_01()
    handle.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

```

#### 7. UI快速入手

##### 1. 夸夸机器人

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210726202405139.png)

```python
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys, random


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # set the title of main window
        self.setWindowTitle('PyQt5 desktop application - www.luochang.ink')

        # set the size of window
        self.Width = 700
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)

        # create all widgets
        self.Label1 = QLabel("夸夸机器人 - Praise me please")
        self.Label1.setFont(QFont('bold', 14))

        self.Label2 = QLabel("created by luochang")
        self.Label2.setFont(QFont('bold', 7))

        self.nameBox = QLineEdit('你')

        self.genderBox = QComboBox()
        self.genderBox.addItem('all')
        self.genderBox.addItem('female')
        self.genderBox.addItem('male')
        
        self.advantageBox = QComboBox()
        self.advantageBox.addItem('all')
        self.advantageBox.addItem('character')
        self.advantageBox.addItem('intelligence')
        self.advantageBox.addItem('appearance')

        self.textBox = QTextEdit(self)

        self.btn = QPushButton('Praise me', self)
        self.btn.clicked.connect(self.praise_me)

        self.initUI()

    def initUI(self):
        # setting up layout of main window
        upper_widget = self.create_upper_widget()
        lower_widget = self.create_lower_widget()
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(upper_widget)
        main_layout.addWidget(lower_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 4)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_upper_widget(self):
        upper_layout = QVBoxLayout()
        upper_layout.addWidget(self.Label1)
        upper_layout.addStretch(5)
        upper_layout.addWidget(self.Label2)
        upper_layout.addStretch(5)
        upper_widget = QWidget()
        upper_widget.setLayout(upper_layout)
        return upper_widget

    def create_lower_widget(self):
        lower_left_widget = QGroupBox("Selections")
        lower_left_layout = QVBoxLayout()        
        lower_left_layout.addWidget(QLabel("Your name:"))
        lower_left_layout.addWidget(self.nameBox)
        lower_left_layout.addWidget(QLabel("Your gender:"))
        lower_left_layout.addWidget(self.genderBox)
        lower_left_layout.addWidget(QLabel("Your advantage:"))
        lower_left_layout.addWidget(self.advantageBox)
        lower_left_layout.addStretch(5)
        lower_left_layout.addWidget(self.btn)
        lower_left_widget.setLayout(lower_left_layout)
        
        lower_right_layout = QVBoxLayout()
        lower_right_layout.addWidget(self.textBox)
        lower_right_widget = QWidget()
        lower_right_widget.setLayout(lower_right_layout)

        lower_layout = QHBoxLayout()
        lower_layout.addWidget(lower_left_widget)
        lower_layout.addWidget(lower_right_widget)
        lower_layout.setStretch(0,1)
        lower_layout.setStretch(1,2)
        lower_widget = QWidget()
        lower_widget.setLayout(lower_layout)
        return lower_widget

    def praise_me(self):
        name = str(self.nameBox.text())
        gender = str(self.genderBox.currentText())
        advantage = str(self.advantageBox.currentText())

        sentence = [['怎么可以这么好！', '是要萌死我吗？', '举止端方，温文尔雅', '知书达理', '言谈可亲', '是我的小天使',\
                    '豁达开朗', '温柔体贴善解人意', '非常绅士', '为人大方，乐于助人', '重情重义', '是个值得信任的男人'], 
                    ['博闻强记', '才高八斗', '饱读诗书', '秀外慧中', '真是个小机灵鬼', '明明可以靠脸吃饭，非要靠才华',\
                    '品学兼优', '学富五车', '上知天文下知地理','是诸葛亮转世', '有颜又有才', '可以说是“上得厅堂，下得厨房”'],
                    ['好苗条哦！我好酸', '是我的梦中女神', '美丽大方', '刚一出来我还以为是刘亦菲', '好可爱，像洋娃娃', '的可爱值得我用一生来守护',\
                    '好帅！！我想给你生猴子', '可太帅了，我能爱一辈子', '帅气又迷人', '是酷酷男孩！', '有着大海般深邃的眼睛', '是个帅小伙']]

        if gender == 'all':
            column_start = 0
            column_stop = len(sentence[0])
        elif gender == 'female':
            column_start = 0
            column_stop = int(len(sentence[0])/2)
        elif gender == 'male':
            column_start = int(len(sentence[0])/2)
            column_stop = len(sentence[0])
        else:
            print('genderBox error')

        if advantage == 'all':
            row = random.randrange(0, len(sentence))
        elif advantage == 'character':
            row = 0
        elif advantage == 'intelligence':
            row = 1
        elif advantage == 'appearance':
            row = 2
        else:
            print('advantageBox error')

        praise_sentence = sentence[row][random.randrange(column_start, column_stop)]

        self.textBox.setText("{}{}".format(name, praise_sentence))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
```



#### 8. 学习资源

- Qt 常用函数功能封装：https://github.com/PyQt5/PyQt
- Qt教程： https://wizardforcel.gitbooks.io/wudi-qt4/content/34.html
- 窗口布局： https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126111321167.png
- QDarkStyleSheet:    运行python example.py 可以查看各种qt案件样式

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126112035071.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126112117998.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210126112140274.png)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/pyqt/  

