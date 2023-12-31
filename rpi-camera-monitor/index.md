# RPI Camera Monitor


> ​       由人盯着监视屏幕，人总有疲劳限度的。研究表明，监控操作人员盯着电视墙屏幕超过10分钟后将漏掉90%的视频信息。由于人工筛选数据的低效率和低可靠性，视频监控系统不能局限于被动地提供视频画面，要求集成智能算法，能够自动识别不同的物体，发现监控画面中的异常情况，实现不再要人去盯、用计算机代替人进行监控，即实现“自动监控”或“智能监控”。智能视频监控是基于机器视觉对视频信号进行处理、分析和理解，在不需要人工干预的情况下，通过对序列图像自动分析对监控场景中的变化进行定位、识别和跟踪，并在此基础上分析和判断目标的行为，能在异常情况发生时及时发出警报或提供有用信息，从而有效地协助安全监管人员处理危机，并最大限度地降低误报和漏报现象，成为应对突发事件的有力辅助工具。
>

## 0. 准备工作

- 树莓派4b  单板2G  335
- 电源，外壳，HDMI线，散热片，16GTF卡，读卡器，小风扇，网线，引脚尺，扩展板+铜柱，按键，点阵，LED，排线，点阵转接板          60
- 显示屏：  7寸 ultra-thin TFT LCD color monitor    180
- 摄像头： 鱼眼广角夜视500w  视角130度   咸鱼  89

## 1. Magic Mirrors

> **MagicMirror²** is an open source modular smart mirror platform. With a growing list of installable modules, the **MagicMirror²** allows you to convert your hallway or bathroom mirror into your personal assistant

### 1.1 Installation Manually

1. Download and install the latest *Node.js* version:

- `curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -`
- `sudo apt install -y nodejs`

2. Clone the repository and check out the master branch: `git clone https://github.com/MichMich/MagicMirror`
3. Enter the repository: `cd MagicMirror/`
4. Install the application: `npm install`
5. Make a copy of the config sample file: `cp config/config.js.sample config/config.js`
6. Start the application: `npm run start`
   For **Server Only** use: `npm run server` .

### 1.2 Alternative Installation

- [Docker Image](https://github.com/bastilimbach/docker-MagicMirror)
- [MagicMirrorOs](https://github.com/guysoft/MagicMirrorOS)

### 1.3 Configuration

- check your configuration running `npm run config:check` in `/home/pi/MagicMirror`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200507210040338.png)

Model 官方模块包括： Alert， Calendar， Clock， Current Weather， Hello world， News Feed， Update Notification， Weather Module, Weather Forecast

FileStructure： 

- **modulename/modulename.js** - This is your core module script.
- **modulename/node_helper.js** - This is an optional helper that will be loaded by the node script. The node helper and module script can communicate with each other using an integrated socket system.
- **modulename/public** - Any files in this folder can be accessed via the browser on `/modulename/filename.ext`.
- **modulename/anyfileorfolder** Any other file or folder in the module folder can be used by the core module script. For example: *modulename/css/modulename.css* would be a good path for your additional module styles.

## [2. 自美系统](http://docs.16302.com/1318686)

硬件清单：https://shop418091054.taobao.com/

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200509141258351.png)

自美驱动板：

> ​      基于树莓派的智能魔镜：音频、人体感应、屏幕开关、镜前灯控件和控温风扇控件等模块而设计，采用WM8960低功耗立体声编解码器，通过I2C接口控制，I2S接口传输音频。板载两个3P标准可录立体有源硅麦接口，板载一个4P可接双通道喇叭。

- 供电电压：5V
- 逻辑电压：3.3V
- 音频编解码芯片：WM8960
- 控制接口：I2C
- 音频接口：I2S
- DAC信噪比：98dB
- ADC信噪比：94dB
- 扬声器驱动：1W per channel (8Ω BTL)

```shell
#布置环境   一键安装全部环境
sudo curl -sSL http://a.16302.com/init | sh
#系统安装
sudo curl -sSL http://a.16302.com/install | sudo python3
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200509143513833.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200509143703233.png)

### **2.1. 配网方法**

1. 如下图所示，将触摸配网按钮插入“按键”插口，手指轻触按钮 （TOUCH） 指示位置5秒钟以上就即可进入配网模式（如果没有触摸按钮也可以使用一根导线将P31与3V3引脚连接5秒钟以上）
   ![](http://qiniucn.16302.com/a412177ed26ee314b6ecb5bf47271558)
2. 语音提示开始配网，跳转出下面的页面，按提示操作
   ![](http://qiniucn.16302.com/4e93ee1e6121e1f86da343352002dfdf)
3. 打开手机微信扫一扫功能，扫描屏幕上二维码直接进入配网界面，如果没有显示器可在微信中搜索“自美系统”小程序，打开小程序在右下角“我”的选项里找到“设备配网”栏目，点击进入
   <img src="http://qiniucn.16302.com/5fb174bf147863c39d07cc54d8de6b17" alt="img" style="zoom:50%;" />

### 2.2. 唤醒词

- 链接网址:  https://snowboy.kitt.ai/
- 设备要求:  电脑上需要麦克风才能完成
- 使用文档：http://docs.16302.com/1144911

### 2.3. 微信绑定

- 扫描二维码，绑定设备

### 2.4. 系统内置插件

| 触发词                                                       | 功能             | 对应插件 | 备注                                  |
| ------------------------------------------------------------ | ---------------- | -------- | ------------------------------------- |
| "用户绑定","绑定用户","绑定设备","设备绑定","我是谁"         | 通过微信绑定机器 | user     |                                       |
| "IP地址", "本机IP"                                           | 播报ip地址       | SayIp    | 基本参考格式                          |
| ["播放本地歌曲", "暂停歌曲", "暂停播放", "继续播放", "歌曲继续","\\b播放\\S{0,4}\\d","\\b切换\\S{0,4}\\d", | 歌曲播放         | Music    | 使用mplay开源软件                     |
| "打开灯", "关闭灯"                                           | 开关操作         | Light    | 设置一个引脚input/output              |
| "打开屏幕", "打开显示", "关闭屏幕", "关闭显示","天气地址","修改位置","声音\\S{2,}", "音量\\S{2,}"**修改位置为铜陵市** | 设备操作         | Device   | 调节音量和打开或关闭屏幕,修改天气地址 |
|                                                              |                  |          |                                       |

### 2.5. 系统手动升级

```shell
sudo python3 /keyicx/update.py
# 若失败手动恢复升级
sudo curl -sSL http://a.16302.com/update | sudo python3
```

**`开发者模式`（推荐使用）**

```shell
sudo curl -sSL http://a.16302.com/initdev | sh
```

自美智能系统桌面任务栏

```shell
sudo mv /home/pi/.config/lxpanel/LXDE-pi/panels/panel /home/pi/.config/lxpanel/LXDE-pi/ 
sudo reboot
```

启动正常模式

```shell
/keyicx/python/run.py
```

启动调试模式

```shell
/keyicx/python/run.py debug
```

停止关闭系统

```shell
/keyicx/python/run.py stop
```



### 2.6. 驱动安装和升级

1. 一键镜像安装自美系统，会自动将本驱动板相应的驱动安装并已经调试到最佳状态，本教程可跳过。
   一键镜像安装自美系统请[点击这里](http://docs.16302.com/1144905)

2. 在线手动安装驱动

   > 手动安装驱动，建议烧录树莓派官方系统，因为咱们基本上都是在这个系统进行开发测试+调试的，其他系统没有相应的测试过，不保证一定能安装成功。

第一步：首先将驱动板插到树莓派开发板上，启动树莓派；
第二步：启动树莓派系统，启动终端（或用SSH连接到树莓派设备）

```
#打开I2C 接口
sudo raspi-config
#选择 5 Interfacing  Options  \→ P5 I2C \→ 是 启动 i2C 内核驱动
# 关闭树莓派默认音频驱动
sudo nano /boot/config.txt
#注：   dtparam=audio=on  将这行注释
sudo reboot
#  在线安装驱动
git clone https://github.com/waveshare/WM8960-Audio-HAT
# 这里需要等待一定的时间
cd WM8960-Audio-HAT
sudo ./install.sh
sudo reboot
#测试驱动安装
sudo dkms status
#wm8960-soundcard, 1.0, 4.19.58+, armv7l: installed
#wm8960-soundcard, 1.0, 4.19.58-v7+, armv7l: installed
#wm8960-soundcard, 1.0, 4.19.58-v7l+, armv7l: installed
#	检测声卡
root@raspberrypi:~ # aplay -l
**** List of PLAYBACK Hardware Devices ****
card 0: wm8960soundcard [wm8960-soundcard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
#	检测录音功能
root@raspberrypi:~ # arecord -l
**** List of CAPTURE Hardware Devices ****
card 0: wm8960soundcard [wm8960-soundcard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 []
  Subdevices: 0/1
  Subdevice #0: subdevice #0
  
#录音并播放测试
sudo arecord -f cd -Dhw:0 | aplay -Dhw:0
```

- 单录音测试

```shell
sudo arecord -D hw:0,0 -f S32_LE -r 16000 -c 2 test.wav
```

test.wav是录制生成的文件名。

- 播放录音（播放刚刚录制的音频）

```shell
sudo aplay -Dhw:0 test.wav
```

- 安装mpg123播放器

```
sudo apt-get install mpg123 
```

- 播放MP3

```
sudo mpg123 ***.mp3
```

### 2.7. 插件开发

> 原理：自美系统采用多进程消息队列管理模式运行，各模块和插件均为独立进程运行互不干扰。进程间采用消息通知方式通信。

> **系统结构说明：** 每一个模块都可以独立运行和协作运行，插件也可以理解为一个特有功能的模块。除了一些特有功能模块，系统自带集合了几大内置功能，分别如下：
> 一、 语音唤醒模块；
> 二、语音录音模块；
> 三、语音识别模块；
> 四、语音合成模块；
> 五、屏幕显示模块；
> 六、微信小程序通信模块；
> 七、外设万能开关通信模块;

#### 2.7.1. 插件结构

一、**插件位置：**
自美插件位于当前系统目录：`./python/plugin/`目录下，如果您采用[镜像安装](http://docs.16302.com/1144905)方式安装的自美系统，那么插件目录就是：`/keyicx/python/plugin/`下，一个插件一个目录，如:

```
/keyicx/python/plugin/Chat（聊天机器人插件）
/keyicx/python/plugin/Music（音乐插件）
```

二、**插件组成：**
自美系统是由*.py（插件入口文件） + config.json（插件配置文件）组成，如音乐插件是由：

```
/keyicx/python/plugin/Music/config.json （配置文件）
/keyicx/python/plugin/Music/Music.py （入口文件）
```

三、**插件命名约定**

> 插件命名约定可简单说叫：四名一致

1、插件文件夹名称；
2、config.json配置文件名中的`name`插件名称键值；
3、插件入口文件`.py`（也可以叫插件基本文件）名；
4、插件入口文件中的起始类名：`class 插件名称`
每一个插件必须保持以上四个位置处名称一致并且在插件文夹中唯一，否则都会导致插件不能正常启动和运行。

四、**config.json 配置文件**

```json
{
    "name": "插件名",
    "triggerwords": [],         // 插件语音触发词数组
    "IsEnable": false,          // 是否启用
    "IsSystem": false,          // 是否为系统插件，系统插件会保持后台运行
    "AutoLoader": false,        // 是否随系统启动自动装载 
    "displayName": "",          // 插件名称（通常为中文简称）
    "description": "",          // 插件简介说明
    "icon":"0",                 // 插件默认图标(默认选0就行)
    "version": "0.0.1",         // 插件版本号
    "updateTime":"",            // 插件最近一次更新时间
    "control":[],               // 微信小程序控制端配置              
    "webAdminApi":"",           // 插件管理配置文件接口
    "initControl": 0,           // 插件被激活是否通知微信小程序控制端
    "repository": {             // 插件版本管理配置
        "type": "git",          // 插件通讯协议，分为：git 和 http 两种，目前只支持git模式
        "url": ""               // 插件版本维护地址
    }
}
```

#### 2.7.2. 前端开发

前端文件目录： /keyicx/python/webroot/desktop/mojing/

**前端显示与Python通讯方式**

自美系统前端与后端Python通讯采用两种方式，一种是被动显示和主动请求：

1. 被动显示采用`websocket`技术，以达到实时性;
2.  主动请求，采用直接调用python文件方式，得到显示结果（注：这种方式后期可能会废弃）

**被动显示方式介绍**

被动显示即前端显示一直处于等待显示数据接收状态，我们采用的是NodeJS技术将前端设置为`websocket`服务器，Python通过socket技术向前端发送信息，具体代码可参考：`/keyicx/app/resources/app/control.js`文件中：

```
start_websocket: function(){
...
}
```

project 1：https://www.youtube.com/watch?v=cVmDjJmcd2M

#### 自定义功能：

- 添加按键触摸打开监控摄像头
- 添加电子相册， 修改对应的js代码
- 添加Monitor监控模块

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200514192013919.png)

### 2.8. 代码阅读

- python/WebServer.py:   启动web服务器
- python/ControlCenter.py: 启动后端服务
- app/moJing:  前端任务

>  self.send(MsgType=MsgType.Start, Receiver=module)

#### 2.8.1.  语言识别

```python
# -*- coding: utf-8 -*-
# @Author: GuanghuiSun
# @Date: 2020-02-22 10:37:52
# @LastEditTime: 2020-03-04 19:23:01
# @Description:  录音服务

import webrtcvad
import os
import time
import logging
from threading import Thread
from MsgProcess import MsgProcess, MsgType
import package.VoiceRecognition as VoiceRecognition
import socket
import wave
from bin.pyAlsa import pyAlsa

# 录音，并转化为文字，发送给 message[sender]
class SocketRec:
    """ 直接通过在后台运行的awake唤醒程序发来的录音包来录音 """
    def __init__(self, buffSize=3200):        
        self.BindFilePath = '/tmp/Record.zimei'
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if os.path.exists(self.BindFilePath):
            os.unlink(self.BindFilePath)
        self.server.bind(self.BindFilePath)
        # 根据后台awake录音参数调节 buffSize = frame*loop*2
        self.buffSize = buffSize  # 取决于 awake.ini中的 frames
        logging.debug("BindFilePath=%s buffSize=%d" % (self.BindFilePath, buffSize))

    def read(self):   
        data, address = self. server.recvfrom(self.buffSize)
        return data
    
    def close(self):
        os.unlink(self.BindFilePath)       

#进行录音
class Pyaudio:
    ''' 调用Pyaudio录音 '''
    def __init__(self, buffSize=3200):
        import pyaudio
        self.CHUNK = buffSize / 2
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000  # 16k采样率      
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(rate=RATE, channels=CHANNELS, format=FORMAT, input=True, frames_per_buffer=self.CHUNK)

    def read(self):
        return self.stream.read(self.CHUNK)
     
    def close(self):
        self.stream.close()
        self.pa.terminate()
      

class Record(MsgProcess):
    def __init__(self, msgQueue):
        super().__init__(msgQueue=msgQueue)
        self.vad = webrtcvad.Vad(1)  # 语音检测库
        self.isReset = False
        self.currentRecThread = None  # 当前录音线程
        self.VoiceRecognition = VoiceRecognition.VoiceRecognition()

    def Start(self, message):
        logging.info('[%s] request Record' % message['Sender'])
        if self.currentRecThread and self.currentRecThread.is_alive():
            self.isReset = True
            self.send(MsgType.Text, Receiver='Screen', Data='请说，我正在聆听...')
            return
        os.popen('aplay -q data/audio/ding.wav')
        time.sleep(0.4) 
        self.currentRecThread = Thread(target=self.RecordThread, args=(message,))
        self.currentRecThread.start()         # 启动录音线程

    def is_speech(self, buffer):
        '''
        检测长度为size字节的buffer是否是语音
        webrtcvad 要求你的总CHUNK用时只能有三种 10ms 20ms 30ms
        在16000hz采样下，一个frame用时为 0.0625ms 所以只能选160 320 480
        对应字节数分别为:320,640,960
        '''
        size = len(buffer)
        RATE = 16000
        assert size >= 320  # 长度不能小于10ms
        # if size < 640:
        #    return self.vad.is_speech(buffer[0:320], RATE)
        setp = 320
        score = 0
        blocks = size // setp  # 将音频分割
        for i in range(blocks):
            score += self.vad.is_speech(buffer[i*setp:(i+1)*setp], RATE)
        # logging.debug("语音概率 {}/{} = {:.2f} buffer size = {}".format(score, blocks, score / blocks,size))
        return score / blocks
    # 启动录音，并将录音转化为文字，发送给 message[sender]
    def RecordThread(self, message):
        if self.config['RecSeclet'] == 'ScoketRec':
            stream = SocketRec(buffSize=4000)                                                    # 使用unix socket录音
        elif self.config['RecSeclet'] == 'pyAlsa':            
            stream = pyAlsa.pyAlsa()                                                             # 使用pyalsa.so录音       
        elif self.config['RecSeclet'] == 'Pyaudio':
            stream = Pyaudio(buffSize=4000)
        else:
            logging.error('未知录音配置 config.yaml')
            return
     
        NoSpeechCheck = 4           # 常量,参考frames大小而定
        MinSpeechCHUNK = 4          # 常量,参考frames大小而定
        MAXRECLAN = 5               # 最长录音时间,秒
        if message['Data']:
            MAXRECLAN = message['Data']
        NoSpeechCHUNK = 0
        Speech_CHUNK_Counter = 0
        lastData = None             # 前导音
        frames = list()
        record_T = time.time()

        # info = {'type':'mic',      类型：dev 设备
        #        'state': 'start'}  状态：start / stop / 1 / 0

        info = {'type': 'mic', 'state': 'start'} 
        self.send(MsgType=MsgType.Text, Receiver='Screen', Data=info)       # 显示mic
        info = {'type': 'mic', 'state': 1}
        # logging.info('开始录音...')
        while (time.time() - record_T < MAXRECLAN):
            info['state'] = ('1' if info['state'] == '0' else '0')            
            self.send(MsgType=MsgType.Text, Receiver='Screen', Data=info)   # 前端mic动画            
            if (self.isReset):
                logging.info('录音重置')
                frames.clear()                
                record_T = time.time()
                Speech_CHUNK_Counter = 0
                NoSpeechCHUNK = 0
                self.isReset = False                
            data = stream.read()
            frames.append(data)
            if self.is_speech(data) >= 0.6:
                if NoSpeechCHUNK >= NoSpeechCheck:
                    frames.insert(0, lastData)
                Speech_CHUNK_Counter += 1
                NoSpeechCHUNK = 0
            else:
                NoSpeechCHUNK += 1
            lastData = data
            if NoSpeechCHUNK >= NoSpeechCheck:
                if Speech_CHUNK_Counter > MinSpeechCHUNK:
                    break
                else:
                    Speech_CHUNK_Counter = 0
                    frames.clear()

        info = {'type': 'mic', 'state': 'stop'} 
        self.send(MsgType=MsgType.Text, Receiver='Screen', Data=info)  # 不显示mic
        stream.close()
        logging.info('录音结束')

        if Speech_CHUNK_Counter > MinSpeechCHUNK:
            os.popen('aplay -q data/audio/dong.wav')
            frames = frames[0: 2 - NoSpeechCheck]
            frames = b"".join(frames)
            text = self.VoiceRecognition.Start(frames)
            if text:
                self.send(MsgType.Text, Receiver='Screen', Data=text)
                self.send(MsgType=MsgType.Text, Receiver=message['Sender'], Data=text)
                self.saveRec(frames, text)
                return
        logging.info('无语音数据')
        self.send(MsgType=MsgType.JobFailed, Receiver=message['Sender'])
        self.send(MsgType.Text, Receiver='Screen', Data='无语音数据')
        self.send(MsgType.QuitGeekTalk, Receiver='ControlCenter')

    def saveRec(self, frames, text):
        ''' 录音分析 日志为DEBUG或INFO时启用 '''
        if self.config['Logging']['Level'] not in ['DEBUG', 'INFO']:
            return
        recpath = r"./runtime/record/"
        if not os.path.exists(recpath):
            os.makedirs(recpath)
        file = os.path.join(recpath, text + '.wav')
        w = wave.open(file, 'w')
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frames)
#--------------百度语音合成类---------
from package.BDaip.speech import AipSpeech
import logging
import uuid
from package.mylib import mylib
''' 
调用百度语音识别，需要联网
上传data， 返回对应识别文字，或者异常信息
'''
class VoiceRecognition:
    ''' 语音识别 '''
    CUID = hex(uuid.getnode())

    def Start(self, data):
        return self.BDVoicerecognition(data)

    def BDVoicerecognition(self, data):
        BDAip = mylib.getConfig()['BDAip']
        APP_ID = BDAip['APP_ID']
        API_KEY = BDAip['API_KEY']
        SECRET_KEY = BDAip['SECRET_KEY']
        client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        # client.setConnectionTimeoutInMillis = 5000  # 建立连接的超时毫秒
        # client.setSocketTimeoutInMillis = 5000  # 传输数据超时毫秒

        logging.info('语音识别...')
        try:       
            bdResult = client.asr(speech=data, options={'dev_pid': 1536, 'cuid': VoiceRecognition.CUID})
        except Exception as e:
            logging.error('网络故障! %s' % e)
            return False
        logging.debug('语音识别已返回')
        text = ''

        if bdResult['err_msg'] == 'success.':  # 成功识别
            for t in bdResult['result']:
                text += t
            logging.info(text)
            return text

        elif bdResult['err_no'] == 3301:  # 音频质量过差
            text = '我没有听清楚您说的话'
            logging.info(text)
            return

        elif bdResult['err_no'] == 3302:  # 鉴权失败
            text = '鉴权失败，请与开发人员联系。'
            logging.warning(text)
            return 

        elif bdResult['err_no'] == 3304 or bdResult['err_no'] == 3305:  # 请求超限
            text = '请求超限，请与开发人员联系。'
            logging.warning(text)
            return 

        text = '语音识别错误,代码{}'.format(bdResult['err_no'])
        logging.error(text)

```



## 3. Home Assistant 系统

> HomeAssistant是构建智慧空间的神器。是一个成熟完整的基于 Python 的智能家居系统，设备支持度高，支持自动化（Automation)、群组化（Group）、UI 客制化（Theme) 等等高度定制化设置。同样实现设备的 Siri 控制。基于HomeAssistant，可以方便地连接各种外部设备（智能设备、摄像头、邮件、短消息、云服务等，成熟的可连接组件有近千种），手动或按照自己的需求自动化地联动这些外部设备，构建随心所欲的智慧空间。HomeAssistant是开源的，它不属于任何商业公司，用户可以无偿使用。

### 3.1. Home Assistant系统

1. Home Control，收集组件的信息并对组件进行控制，同样也接收来自用户的控制并返回信息。
2. Home Automation，根据用户的配置，自动发送控制指令（通过用户指导来替代用户层的操作）。
3. Smart Home，根据各种之前的控制行为与结果，自学习到下一次发送的控制指令（无需用户指导而替代用户层的操作）。

### 3.2. Installation

#### 3.2.1 Hass.io 

> [Hass.io](https://www.home-assistant.io/hassio/)是一个完整的UI管理的家庭自动化生态系统，它运行Home Assistant，Hass.io Supervisor和附加组件。它预先安装在HassOS上，但可以安装在任何Linux系统上。它利用了由Hass.io Supervisor管理的Docker。
>
> - **Hass.io Supervisor**：Hass.io Supervisor是管理Hass.io安装的程序，负责安装和更新Home Assistant，附加组件本身以及更新（如果使用的话）HassOS操作系统。
> - **HassOS**：HassOS，家庭助理操作系统，是一种嵌入式，简约的操作系统，旨在在单板计算机（如Raspberry Pi）或虚拟机上运行Hass.io生态系统。Hass.io Supervisor可以使其保持最新状态，从而无需管理操作系统

Installing using a Docker managed environment (recommended method).

- Download and extract the Home Assistant image for [your device](https://www.home-assistant.io/hassio/installation/)
- Download [balenaEtcher](https://www.balena.io/etcher) to write the image to an SD card
- http://X.X.X.X:8123  to see the home page

Docker - Installing on Docker.

#### 3.2.2 Manually 

 Manual installation using a Python virtual environment.

```shell
pip3 install homeassistant
hass 
# open  localhost:8123, 注册账号进入主界面
```

<img src="https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200507125144328.png" alt="image-20200507125144328" style="zoom:50%;" />

#### 3.2.3 docker install

```shell
sudo nano /etc/apt/sources.list.d/raspi.list
sudo apt-get update
#docker 安装
#1. docker apt 源
sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     software-properties-common
# GPG密钥
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg | sudo apt-key add -
#2. 添加docker ce
echo "deb [arch=armhf] https://download.docker.com/linux/debian \
      $(lsb_release -cs) stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list
#3. 安装docker ce
sudo apt-get update
sudo apt-get install docker-ce
#4. 创建docker 仓库镜像
sudo nano /etc/docker/daemon.json
{
  "registry-mirrors": ["https://registry.docker-cn.com"]
}
sudo systemctl daemon-reload
sudo systemctl restart docker
docker pull hello-world
docker run hello-world
# 安装图形化界面工具
docker pull portainer/portainer:latest
docker run -d -p 9000:9000 --name portainer --restart=always -e TZ="Asia/Shanghai" -v /var/run/docker.sock:/var/run/docker.sock portainer/portainer
```

## 4. 系统镜像

- [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) - 来自树莓派官方的操作系统，基于Debian，同时官方也提供了一个Raspbian的精简版。
- [Kali Linux](https://www.offensive-security.com/kali-linux-arm-images/) - 为渗透测试和正义黑客准备的Linux发行版，运行在ARM设备上
- [chilipie-kiosk](https://github.com/futurice/chilipie-kiosk) - 可直接引导到全屏Chrome的镜像，非常适合用于仪表板和构建监视器。 
- [DietPi](https://github.com/Fourdee/DietPi) - 为2G SD卡准备的最小镜像， 带有许多可配置项和脚本.
- [Minibian](https://minibianpi.wordpress.com/) - 最小的 Raspbian (比 Jessie Lite还要轻量).
- [Fedora](https://fedoraproject.org/wiki/Raspberry_Pi#Preparing_the_SD_card)
- [motionEyeOS](https://github.com/ccrisan/motioneyeos/wiki) - 将微型计算机打造为视频监控系统的Linux发行版。

## 5. 工具

- [Etcher](https://www.etcher.io/) - 跨平台的SD卡烧录程序，使用简单，易于扩展.
- [OpenVPN-Setup](https://github.com/StarshipEngineer/OpenVPN-Setup) - 用于将树莓派设置为OpenVPN服务器的Shell脚本.
- [Network Presence Detector](https://github.com/initialstate/pi-sensor-free-presence-detector/wiki) - 配置Pi0，使其可以在wifi网络里扫描，发现谁是"home"
- [Pi Dashboard来监控树莓派状态](https://github.com/spoonysonny/pi-dashboard)

## 6. 项目

- [Mini OONTZ](https://cdn-learn.adafruit.com/downloads/pdf/mini-oontz-3d-printed-midi-controller.pdf) - 3D打印的迷你MIDI控制器
- [Power Sniffing Strip](https://gnurds.com/index.php/2012/10/02/raspberry-pi-power-strip/) - 藏在电源插座里的树莓派, 用于嗅探网络数据.
- [Raspberry Pi Erlang Cluster](https://medium.com/@pieterjan_m/erlang-pi2-arm-cluster-vs-xeon-vm-40871d35d356#.bpao66cm8) - 跑在树莓派2代上的Erlang集群
- [NTP driven Nixie Clock](http://www.mjoldfield.com/atelier/2012/08/ntp-nixie.html) - 由树莓派驱动的数码管时钟
- [40-node Raspberry Pi Cluster](http://hackaday.com/2014/02/17/40-node-raspi-cluster/) - 40个节点构成的树莓派集群
- [Raspberry PI Hadoop Cluster](http://www.widriksson.com/raspberry-pi-hadoop-cluster/) - 跑在树莓派上的大数据集群.
- [Multi-Datacenter Cassandra on 32 Raspberry Pi’s](http://www.datastax.com/dev/blog/32-node-raspberry-pi-cassandra-cluster) - 32个节点的树莓派cassandra数据库集群.
- [Building a Ceph Cluster on Raspberry Pi](http://bryanapperson.com/blog/the-definitive-guide-ceph-cluster-on-raspberry-pi/) - 基于分布式对象存储系统RADOS的高度冗、低功耗家庭存储解决方案。
- [Smart Mirror](https://github.com/evancohen/smart-mirror) - 带语音控制智能镜子，集成物联网. 
- [Magic Mirror](http://magicmirror.builders/) - 开源模块化智能镜子平台. 
- [Door bot](https://blog.haschek.at/post/f31aa) - 门卫机器人，感知到门被打开时将给你发送信息.
- [SecPi](https://github.com/SecPi/SecPi) - 基于Raspberry Pi的家庭报警系统.
- [PiClock](https://github.com/n0bel/PiClock) - 别致的时钟
- [Movel](https://github.com/stevelacy/movel) - 树莓派车载电脑
- [PiFanTuner](https://github.com/winkidney/PIFanTuner) - CPU风扇控制程序
- [Sonus](https://github.com/evancohen/sonus) - 开源、跨平台的语音识别框架（Google Cloud Speech）
- [Alexa AVS](https://github.com/alexa/alexa-avs-sample-app/wiki/Raspberry-Pi) - 基于Java客户端和Node.js服务端的 Alexa Voice Service 示例程序
- [Harry Potter and the real life Daily Prophet](https://www.raspberrypi.org/blog/harry-potter-and-the-real-life-daily-prophet/) - 通过RPi的7英寸显示模拟哈利波特中的魔法报纸（动态头条）
- [Raspberry Pi login with SSH keys](https://blog.thibmaekelbergh.be/2015/05/07/raspberry-pi-login-with-ssh-keys.html) - ssh免密码登录树莓派（使用SSH key）
- [Use a Raspberry Pi with multiple WiFi networks](https://www.mikestreety.co.uk/blog/use-a-raspberry-pi-with-multiple-wifi-networks) - 将树莓派接入多个无线网络的教程.
- [Raspberry Pi Media Server Guides](http://www.htpcguides.com/category/raspberry-pi/) - 用树莓派搭建媒体服务器的教程（HTPC：Home Theater Personal Computer，即家庭影院电脑）
- [OSMC](https://osmc.tv/wiki/development/getting-involved-with-osmc-development/): 家庭影院系统







---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/rpi-camera-monitor/  

