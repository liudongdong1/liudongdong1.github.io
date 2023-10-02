# Socket


> socket屏蔽了各个协议的通信细节，使得程序员无需关注协议本身，直接使用socket提供的接口来进行互联的不同主机间的进程的通信。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510111010.png)

### 1.API

#### .1.  socket 接口

> int socket(int protofamily, int so_type, int protocol);

- protofamily 指协议族，常见的值有：

  AF_INET，指定so_pcb中的地址要采用ipv4地址类型

  AF_INET6，指定so_pcb中的地址要采用ipv6的地址类型

  AF_LOCAL/AF_UNIX，指定so_pcb中的地址要使用绝对路径名

  当然也还有其他的协议族，用到再学习了

- so_type 指定socket的类型，也就是上面讲到的so_type字段，比较常用的类型有：

  SOCK_STREAM:对应tcp

  SOCK_DGRAM：对应udp

  SOCK_RAW：自定义协议或者直接对应ip层

- protocol 指定具体的协议，也就是指定本次通信能接受的数据包的类型和发送数据包的类型，常见的值有：

  IPPROTO_TCP，TCP协议

  IPPROTO_UDP，UPD协议

  0，如果指定为0，表示由内核根据so_type指定默认的通信协议

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510111221.png)

> int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

bind函数就是给图三种so_pcb结构中的地址赋值的接口

- sockfd 是调用socket()函数创建的socket描述符
- addr 是具体的地址
- addrlen 表示addr的长度

举struct sockaddr其实是void的typedef，其常见的结构如下图（图片来源传智播客邢文鹏linux系统编程的笔记），这也是为什么需要addrlen参数的原因，不同的地址类型，其地址长度不一样：

![](https://img2018.cnblogs.com/blog/988061/201809/988061-20180904170327305-518430882.png)

> int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

这三个参数和bind的三个参数类型一直，只不过此处strcut sockaddr表示对端公开的地址。三个参数都是传入参数。connect顾名思义就是拿来建立连接的函数，只有像tcp这样面向连接、提供可靠服务的协议才需要建立连接

> int listen(int sockfd, int backlog)

告知内核在sockfd这个描述符上监听是否有连接到来，并设置同时能完成的最大连接数为backlog当调用listen后，内核就会建立两个队列，一个SYN队列，表示接受到请求，但未完成三次握手的连接；另一个是ACCEPT队列，表示已经完成了三次握手的队列

- sockfd 是调用socket()函数创建的socket描述符
- backlog 已经完成三次握手而等待accept的连接数

关于backlog , man listen的描述如下：

- The behavior of the backlog argument on TCP sockets changed with Linux 2.2. Now it specifies the queue length for completely established sockets waiting to be accepted, instead of the number of incomplete connection requests. The maximum length of the queue for incomplete sockets can be set using /proc/sys/net/ipv4/tcp_max_syn_backlog. When syncookies are enabled there is no logical maximum length and this setting is ignored. See tcp(7) for more information.翻译：TCP套接字上的积压参数的行为随着Linux 2.2而改变。现在，它指定等待被接受的完全建立的套接字的队列长度，而不是不完整连接请求的数量。不完整套接字队列的最大长度可以使用/PRO/sys／NET/IPv4/TCPPMAX Syth-ByLoSQL来设置。当启用同步功能时，没有逻辑最大长度，并且忽略该设置。有关更多信息，请参见TCP（7）。
- If the backlog argument is greater than the value in /proc/sys/net/core/somaxconn, then it is silently truncated to that value; the default value in this file is 128. In kernels before 2.4.25, this limit was a hard coded value, SOMAXCONN, with the value 128.如果backlog参数大于/proc/sys/net/core/somaxconn中的值，那么它将被悄悄地截断为该值；该文件中的默认值为128。在2.4.25之前的内核中，这个限制是一个硬编码的值，SOMAXCONN，值为128。

> accept接口

int accept(int listen_sockfd, struct sockaddr *addr, socklen_t *addrlen)

这三个参数与bind的三个参数含义一致，不过，此处的后两个参数是传出参数。在使用listen函数告知内核监听的描述符后，内核就会建立两个队列，一个SYN队列，表示接受到请求，但未完成三次握手的连接；另一个是ACCEPT队列，表示已经完成了三次握手的队列。而accept函数就是从ACCEPT队列中拿一个连接，并生成一个新的描述符，新的描述符所指向的结构体so_pcb中的请求端ip地址、请求端端口将被初始化。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510111521.png)

> 1. 服务器端在调用listen之后，内核会建立两个队列，SYN队列和ACCEPT队列，其中ACCPET队列的长度由backlog指定。
> 2. 服务器端在调用accpet之后，将阻塞，等待ACCPT队列有元素。
> 3. 客户端在调用connect之后，将开始发起SYN请求，请求与服务器建立连接，此时称为第一次握手。
> 4. 服务器端在接受到SYN请求之后，把请求方放入SYN队列中，并给客户端回复一个确认帧ACK，此帧还会携带一个请求与客户端建立连接的请求标志，也就是SYN，这称为第二次握手
> 5. 客户端收到SYN+ACK帧后，connect返回，并发送确认建立连接帧ACK给服务器端。这称为第三次握手
> 6. 服务器端收到ACK帧后，会把请求方从SYN队列中移出，放至ACCEPT队列中，而accept函数也等到了自己的资源，从阻塞中唤醒，从ACCEPT队列中取出请求方，重新建立一个新的sockfd，并返回。

- TCP: 
  - send函数只负责将数据提交给协议层。 当调用该函数时，send先比较待发送数据的长度len和套接字s的发送缓冲区的长度，如果len大于s的发送缓冲区的长度，该函数返回SOCKET_ERROR； 如果len小于或者等于s的发送缓冲区的长度，那么send先检查协议是否正在发送s的发送缓冲中的数据； 如果是就等待协议把数据发送完，如果协议还没有开始发送s的发送缓冲中的数据或者s的发送缓冲中没有数据，那么send就比较s的发送缓冲区的剩余空间和len； 如果len大于剩余空间大小，send就一直等待协议把s的发送缓冲中的数据发送完，如果len小于剩余空间大小，send就仅仅把buf中的数据copy到剩余空间里（注意并不是send把s的发送缓冲中的数据传到连接的另一端的，而是协议传的，send仅仅是把buf中的数据copy到s的发送缓冲区的剩余空间里）。 如果send函数copy数据成功，就返回实际copy的字节数，如果send在copy数据时出现错误，那么send就返回SOCKET_ERROR； 如果send在等待协议传送数据时网络断开的话，那么send函数也返回SOCKET_ERROR。要注意send函数把buf中的数据成功copy到s的发送缓冲的剩余空间里后它就返回了，但是此时这些数据并不一定马上被传到连接的另一端。 如果协议在后续的传送过程中出现网络错误的话，那么下一个Socket函数就会返回SOCKET_ERROR。（每一个除send外的Socket函数在执行的最开始总要先等待套接字的发送缓冲中的数据被协议传送完毕才能继续，如果在等待时出现网络错误，那么该Socket函数就返回SOCKET_ERROR）
  - recv先检查套接字s的接收缓冲区，如果s接收缓冲区中没有数据或者协议正在接收数据，那么recv就一直等待，直到协议把数据接收完毕。当协议把数据接收完毕，recv函数就把s的接收缓冲中的数据copy到buf中（注意协议接收到的数据可能大于buf的长度，所以在这种情况下要调用几次recv函数才能把s的接收缓冲中的数据copy完。recv函数仅仅是copy数据，真正的接收数据是协议来完成的），recv函数返回其实际copy的字节数。如果recv在copy时出错，那么它返回SOCKET_ERROR；如果recv函数在等待协议接收数据时网络中断了，那么它返回0 。对方优雅的关闭socket并不影响本地recv的正常接收数据；如果协议缓冲区内没有数据，recv返回0，指示对方关闭；如果协议缓冲区有数据，则返回对应数据(可能需要多次recv)，在最后一次recv时，返回0，指示对方关闭。

### 2. python

#### 2.1. 多线程基本结构

- 服务器代码

```python
#!C:\Python3.6.5\python.exe
# -*- coding: gbk -*-

import socket
import threading

class WSGIServer(object):
    def __init__(self, port):
        """初始化对象"""
        # 创建套接字
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 解决程序端口占用问题
        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定本地ip地址
        self.tcp_server_socket.bind(("", port))
        # 将套接字变为监听套接字，最大连接数量为100
        self.tcp_server_socket.listen(100)

    def run_forever(self):
        """设备连接"""
        while True:
            # 1.等待设备连接(通过ip地址和端口建立tcp连接)
            #   如果有设备连接，则会生成用于设备和服务器通讯的套接字：new_socket
            #   会获取到设备的ip地址和端口
            new_socket, client_addr = self.tcp_server_socket.accept()
            print("设备{0}已连接".format(client_addr))

            # 2.创建线程处理设备的需求
            t1 = threading.Thread(target=self.service_machine, args=(new_socket, client_addr))
            t1.start()

    def service_machine(self, new_socket, client_addr):
        """业务处理"""
        while True:
            # 3.接收设备发送的数据，单次最大1024字节，按‘gbk’格式解码
            receive_data = new_socket.recv(1024).decode("gbk")
            # 4.如果设备发送的数据不为空
            if receive_data:
                # 4.1 打印接收的数据，这里可以将设备发送的数据写入到文件中
                # 获取设备的ID信息
                print(receive_data)
                if receive_data[0:6] == "report":
                    response = "SET OK:" + receive_data
                else:
                    receive_data = receive_data[6:].split(",")[0]
                    # 拼接响应数据
                    response = "alarm=" + receive_data + ",Switch:clear"
                print(response)
                # 4.2 返回原数据作为应答，按‘utf-8’格式编码
                new_socket.send(response.encode("utf-8"))
            # 5.当设备断开连接时，会收到空的字节数据，判断设备已断开连接
            else:
                print('设备{0}断开连接...'.format(client_addr))
                break
        # 关闭套接字
        new_socket.close()

def main(port):
    """创建一个WEB服务器"""
    wsgi_server = WSGIServer(port)
    print("服务器已开启")
    wsgi_server.run_forever()

if __name__ == '__main__':
    port = 8125     # 指定端口
    main(8125)
```

- 客户端

```python
import os
import time
import socket

def start_client(addr, port):
    PLC_ADDR = addr
    PLC_PORT = port
    s = socket.socket()
    s.connect((PLC_ADDR, PLC_PORT))
    count = 0
    while True:
        msg = '进程{pid}发送数据'.format(pid=os.getpid())
        msg = msg.encode(encoding='utf-8')
        s.send(msg)
        recv_data = s.recv(1024)
        print(recv_data.decode(encoding='utf-8'))
        time.sleep(3)
        count += 1
        if count > 20:
            break

    s.close()

if __name__ == '__main__':
    start_client('127.0.0.1', 8801)
```

#### 2.2. 图片传输python+c++

- python

```python
#!/usr/bin/python
#-*-coding:utf-8 -*-
import socket
import cv2
import numpy

# 接受图片大小的信息
def recv_size(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# 接收图片
def recv_all(sock, count):
    buf = ''
    while count:
        # 这里每次只接收一个字节的原因是增强python与C++的兼容性
        # python可以发送任意的字符串，包括乱码，但C++发送的字符中不能包含'\0'，也就是字符串结束标志位
        newbuf = sock.recv(1)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# socket.AF_INET用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM代表基于TCP的流式socket通信
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置地址与端口，如果是接收任意ip对本服务器的连接，地址栏可空，但端口必须设置
address = ('', 8010)
s.bind(address) # 将Socket（套接字）绑定到地址
s.listen(True) # 开始监听TCP传入连接
print ('Waiting for images...')
# 接受TCP链接并返回（conn, addr），其中conn是新的套接字对象，可以用来接收和发送数据，addr是链接客户端的地址。
conn, addr = s.accept()

while 1:
    length = recv_size(conn,16) #首先接收来自客户端发送的大小信息
    if isinstance (length,str): #若成功接收到大小信息，进一步再接收整张图片
        stringData = recv_all(conn, int(length))
        data = numpy.fromstring(stringData, dtype='uint8')
        decimg=cv2.imdecode(data,1) #解码处理，返回mat图片
        cv2.imshow('SERVER',decimg)
        if cv2.waitKey(10) == 27:
            break 
        print('Image recieved successfully!')
        conn.send("Server has recieved messages!")
    if cv2.waitKey(10) == 27:
        break 

s.close()
cv2.destroyAllWindows()

#client
#!/usr/bin/python
#-*-coding:utf-8 -*-
import socket
import cv2
import numpy

# socket.AF_INET用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM代表基于TCP的流式socket通信
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# 连接服务端
address_server = ('10.106.20.111', 8010)
sock.connect(address_server)

# 从摄像头采集图像
capture = cv2.VideoCapture(0)
ret, frame = capture.read()
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90] #设置编码参数

while ret: 
	# 首先对图片进行编码，因为socket不支持直接发送图片
    result, imgencode = cv2.imencode('.jpg', frame)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    # 首先发送图片编码后的长度
    sock.send(str(len(stringData)).ljust(16))
    # 然后一个字节一个字节发送编码的内容
    # 如果是python对python那么可以一次性发送，如果发给c++的server则必须分开发因为编码里面有字符串结束标志位，c++会截断
    for i in range (0,len(stringData)):
    	sock.send(stringData[i])
    ret, frame = capture.read()
    #cv2.imshow('CLIENT',frame)
    # if cv2.waitKey(10) == 27:
    #     break
    # 接收server发送的返回信息
    data_r = sock.recv(50)
    print (data_r)

sock.close()
cv2.destroyAllWindows()
```

- c++

```c
#include <stdio.h>
#include <Winsock2.h>
#include <opencv2/opencv.hpp>
#include <vector> 
#pragma comment(lib,"ws2_32.lib")

using namespace cv;
using namespace std;

void main()
{
	WSADATA wsaData;
	SOCKET sockServer;
	SOCKADDR_IN addrServer;
	SOCKET conn;
	SOCKADDR_IN addr;

	WSAStartup(MAKEWORD(2, 2), &wsaData);
	//����Socket  
	sockServer = socket(AF_INET, SOCK_STREAM, 0);
	//׼��ͨ�ŵ�ַ  
	addrServer.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(8010);
	//��  
	bind(sockServer, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
	//����  
	listen(sockServer, 5);
	printf("Waiting for images...\n");

	int len = sizeof(SOCKADDR);
	//��������  
	conn = accept(sockServer, (SOCKADDR*)&addr, &len);

	char recvBuf[16];
	char recvBuf_1[1];
	Mat img_decode;
	vector<uchar> data;

	while (1)
	{
		if (recv(conn, recvBuf, 16, 0))
		{
			for (int i = 0; i < 16; i++)
			{
				if (recvBuf[i]<'0' || recvBuf[i]>'9') recvBuf[i] = ' ';
			}
			data.resize(atoi(recvBuf));
			for (int i = 0; i < atoi(recvBuf); i++)
			{
				recv(conn, recvBuf_1, 1, 0);
				data[i] = recvBuf_1[0];
			}
			printf("Image recieved successfully!\n");
			send(conn, "Server has recieved messages!", 29, 0);
			img_decode = imdecode(data, CV_LOAD_IMAGE_COLOR);
			imshow("server", img_decode);
			if (waitKey(30) == 27) break;
		}
	}
	closesocket(conn);
	WSACleanup();
}

#client
#include <stdio.h>
#include <string>
#include <iostream>
#include <Winsock2.h>
#include <opencv2/opencv.hpp>
#include <vector> 
#pragma comment(lib,"ws2_32.lib")

using namespace cv;
using namespace std;

void main()
{
	WSADATA wsaData;
	SOCKET sockClient;//�ͻ���Socket
	SOCKADDR_IN addrServer;//����˵�ַ
	WSAStartup(MAKEWORD(2, 2), &wsaData);
	//�½��ͻ���socket
	sockClient = socket(AF_INET, SOCK_STREAM, 0);
	//����Ҫ���ӵķ���˵�ַ
	addrServer.sin_addr.S_un.S_addr = inet_addr("10.106.20.111");//Ŀ��IP(10.106.20.74�ǻ��͵�ַ)
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(8010);//���Ӷ˿�
	//���ӵ������
	connect(sockClient, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));

	Mat image;
	VideoCapture capture(0);
	vector<uchar> data_encode;

	while (1)
	{
		if (!capture.read(image)) 
			break;
		imencode(".jpg", image, data_encode);
		int len_encode = data_encode.size();
		string len = to_string(len_encode);
		int length = len.length();
		for (int i = 0; i < 16 - length; i++)
		{
			len = len + " ";
		}
		//��������
		send(sockClient, len.c_str(), strlen(len.c_str()), 0);
		char send_char[1];
		for (int i = 0; i < len_encode; i++)
		{
			send_char[0] = data_encode[i];
			send(sockClient, send_char, 1, 0);
		}
		//���շ�����Ϣ
		char recvBuf[32];
		if(recv(sockClient, recvBuf, 32, 0))
			printf("%s\n", recvBuf);
	}
	closesocket(sockClient);
	WSACleanup();
}
```

#### 2.3. [pyqt通信](https://github.com/Wangler2333/tcp_udp_web_tools-pyqt5)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210510155418975.png)

- [TCP_Logic](D:\projectBack\tcp_udp_web_tools-pyqt5)

```python
from PyQt5 import QtWidgets
import tcp_udp_web_ui
import socket
import threading
import sys
import stopThreading
import time


class TcpLogic(tcp_udp_web_ui.ToolsUi):
    def __init__(self, num):
        super(TcpLogic, self).__init__(num)
        self.tcp_socket = None
        self.sever_th = None
        self.client_th = None
        self.client_socket_list = list()

        self.link = False  # 用于标记是否开启了连接

    def tcp_server_start(self):
        """
        功能函数，TCP服务端开启的方法
        :return: None
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 取消主动断开连接四次握手后的TIME_WAIT状态
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 设定套接字为非阻塞式
        self.tcp_socket.setblocking(False)
        try:
            port = int(self.lineEdit_port.text())
            self.tcp_socket.bind(('', port))
        except Exception as ret:
            msg = '请检查端口号\n'
            self.signal_write_msg.emit(msg)
        else:
            self.tcp_socket.listen()
            self.sever_th = threading.Thread(target=self.tcp_server_concurrency)
            self.sever_th.start()
            msg = 'TCP服务端正在监听端口:%s\n' % str(port)
            self.signal_write_msg.emit(msg)

    def tcp_server_concurrency(self):
        """
        功能函数，供创建线程的方法；
        使用子线程用于监听并创建连接，使主线程可以继续运行，以免无响应
        使用非阻塞式并发用于接收客户端消息，减少系统资源浪费，使软件轻量化
        :return:None
        """
        while True:
            try:
                client_socket, client_address = self.tcp_socket.accept()
            except Exception as ret:
                time.sleep(0.001)
            else:
                client_socket.setblocking(False)
                # 将创建的客户端套接字存入列表,client_address为ip和端口的元组
                self.client_socket_list.append((client_socket, client_address))
                msg = 'TCP服务端已连接IP:%s端口:%s\n' % client_address
                self.signal_write_msg.emit(msg)
            # 轮询客户端套接字列表，接收数据
            for client, address in self.client_socket_list:
                try:
                    recv_msg = client.recv(1024)
                except Exception as ret:
                    pass
                else:
                    if recv_msg:
                        msg = recv_msg.decode('utf-8')
                        msg = '来自IP:{}端口:{}:\n{}\n'.format(address[0], address[1], msg)
                        self.signal_write_msg.emit(msg)
                    else:
                        client.close()
                        self.client_socket_list.remove((client, address))

    def tcp_client_start(self):
        """
        功能函数，TCP客户端连接其他服务端的方法
        :return:
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            address = (str(self.lineEdit_ip_send.text()), int(self.lineEdit_port.text()))
        except Exception as ret:
            msg = '请检查目标IP，目标端口\n'
            self.signal_write_msg.emit(msg)
        else:
            try:
                msg = '正在连接目标服务器\n'
                self.signal_write_msg.emit(msg)
                self.tcp_socket.connect(address)
            except Exception as ret:
                msg = '无法连接目标服务器\n'
                self.signal_write_msg.emit(msg)
            else:
                self.client_th = threading.Thread(target=self.tcp_client_concurrency, args=(address,))
                self.client_th.start()
                msg = 'TCP客户端已连接IP:%s端口:%s\n' % address
                self.signal_write_msg.emit(msg)

    def tcp_client_concurrency(self, address):
        """
        功能函数，用于TCP客户端创建子线程的方法，阻塞式接收
        :return:
        """
        while True:
            recv_msg = self.tcp_socket.recv(1024)
            if recv_msg:
                msg = recv_msg.decode('utf-8')
                msg = '来自IP:{}端口:{}:\n{}\n'.format(address[0], address[1], msg)
                self.signal_write_msg.emit(msg)
            else:
                self.tcp_socket.close()
                self.reset()
                msg = '从服务器断开连接\n'
                self.signal_write_msg.emit(msg)
                break

    def tcp_send(self):
        """
        功能函数，用于TCP服务端和TCP客户端发送消息
        :return: None
        """
        if self.link is False:
            msg = '请选择服务，并点击连接网络\n'
            self.signal_write_msg.emit(msg)
        else:
            try:
                send_msg = (str(self.textEdit_send.toPlainText())).encode('utf-8')
                if self.comboBox_tcp.currentIndex() == 0:
                    # 向所有连接的客户端发送消息
                    for client, address in self.client_socket_list:
                        client.send(send_msg)
                    msg = 'TCP服务端已发送\n'
                    self.signal_write_msg.emit(msg)
                if self.comboBox_tcp.currentIndex() == 1:
                    self.tcp_socket.send(send_msg)
                    msg = 'TCP客户端已发送\n'
                    self.signal_write_msg.emit(msg)
            except Exception as ret:
                msg = '发送失败\n'
                self.signal_write_msg.emit(msg)

    def tcp_close(self):
        """
        功能函数，关闭网络连接的方法
        :return:
        """
        if self.comboBox_tcp.currentIndex() == 0:
            try:
                for client, address in self.client_socket_list:
                    client.close()
                self.tcp_socket.close()
                if self.link is True:
                    msg = '已断开网络\n'
                    self.signal_write_msg.emit(msg)
            except Exception as ret:
                pass
        if self.comboBox_tcp.currentIndex() == 1:
            try:
                self.tcp_socket.close()
                if self.link is True:
                    msg = '已断开网络\n'
                    self.signal_write_msg.emit(msg)
            except Exception as ret:
                pass
        try:
            stopThreading.stop_thread(self.sever_th)
        except Exception:
            pass
        try:
            stopThreading.stop_thread(self.client_th)
        except Exception:
            pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = TcpLogic(1)
    ui.show()
    sys.exit(app.exec_())

```

#### 2.4. python&C# 通信

> 注意发送内容编码形式

- server.py

```python
import face_recognition
import os
import numpy as np
import cv2
import json   
from socket import *
from threading import Thread
from seetaface.api import *
import numpy as np
import pickle
import threading
class SeetaFaceHandle:
    def __init__(self,path):
        """
        使用到的函数:
            Extract：提取一张原始图像里的一个指定区域的人脸的特征值
            ExtractCroppedFace: 提取一张已经裁剪好，只包含人脸图的人脸的特征值
            CalculateSimilarity: 计算两个特征值之间的相似度
        要加载的功能 :
            人脸识别功能：FACERECOGNITION
        该功能依赖 ：
            FACE_DETECT :人脸检测
            LANDMARKER5 :5点关键点识别
        path: 人脸图片所在的目录
        """
        self.init_mask = FACE_DETECT|FACERECOGNITION|LANDMARKER5
        self.seetaFace = SeetaFace(self.init_mask)
        self.pic_path=path
        self.faces,self.faceNames=self.initFaceData()

    def initFaceData(self):
        known_faces=[]
        known_faceNames=[]
        known_face_sids=[]
        json_lists=[]
        if os.path.exists('known_faces'):
            # npz = np.load('all4.npz')
            # known_faceNames = npz['names']
            # known_faces=npz['encode']
            with open('known_faces', 'wb') as f:
                known_faces=pickle.load(known_faces, f)
            with open('known_faceNames', 'wb') as f:
                known_faceNames=pickle.load(known_faceNames, f)
   
            #print("type",type(known_faces))
            #known_faces=self.seetaFace.get_feature_numpy(known_faces)
            #print("type",type(known_faces))
            #print("\nfaceimg",known_faces,"\nfacename",known_faceNames)
        else:
            i=0
            files=os.listdir(self.pic_path)
            for file in files:
                img_path=os.path.join(self.pic_path,file)
                obj_img = self.load_image_file(img_path)
                #print(file,"obj_img:",obj_img.size)
               
                import time
                a1 = time.time()
                detect_result1 = self.seetaFace.Detect(obj_img)
                #print("detect_result1:",detect_result1)
                face1 = detect_result1.data[0].pos
                #print("face1:",face1)
                points1 = self.seetaFace.mark5(obj_img,face1)
                #print("points1:",points1)
                obj_face_encoding = self.seetaFace.Extract(obj_img,points1)
                #print("name:",file,"feature:",obj_face_encoding)
                #print("obj_face_encoding:",obj_face_encoding)
                #print('handlefile=',file,'time used:', time.time()-a1)
                if not len(obj_face_encoding):
                    print('eee:', file)
                    continue
                known_faces.append(obj_face_encoding)
                name = file.split('.')[0]
                known_faceNames.append(name)
                known_face_sids.append(i)
                json_lists.append({'img': file, 'name': name, 'sid': str(i)})
            # with open('known_faces', 'wb') as f:
            #     pickle.dump(known_faces, f)
            # with open('known_faceNames', 'wb') as f:
            #     pickle.dump(known_faceNames, f)
            #np.savez('all4.npz', encode=known_faces, sids=known_face_sids, names=known_faceNames)
            # with open('facename.json', 'w') as f:
            #     json_list_str = json.dumps(json_lists)
            #     f.write(json_list_str)
        return known_faces,known_faceNames

    def load_image_file(self, file, mode='RGB'):
        print("pic_file",file)
        im = cv2.imdecode(np.fromfile(file,dtype=np.uint8),-1)
        return im
    
    def getFaceNameFromFile(self,targetPath):
        name="None"
        image2=self.load_image_file(targetPath)
        try:
            detect_result2 = self.seetaFace.Detect(image2)
            face2 = detect_result2.data[0].pos
            points2 = self.seetaFace.mark5(image2,face2)
            feature2 = self.seetaFace.Extract(image2,points2)
            max=0
            index=-1
            for i in range(0,len(self.faces)):                
                #targetEncoding=self.seetaFace.get_feature_numpy(feature2)
                #print("similar1:",type(self.faces[i]),"  feature2:",type(targetEncoding))
                #similar1 = self.seetaFace.compare_feature_np(self.faces[i],targetEncoding) #0.688
                similar1 = self.seetaFace.CalculateSimilarity(self.faces[i],feature2) #0.999
                if similar1>max:
                    max=similar1
                    index=i
            
            print("最大相似度： ",max, "\t 人脸name:",self.faceNames[index],"\t人脸特征:",self.faces[i])
            return self.faceNames[index]
        except:
            print("file don't detect face")
            return name

    def getFaceNameFromEncoding(self,targetEncoding):
        name="None"
        max=0
        index=-1
        targetEncoding=self.seetaFace.get_feature_numpy(targetEncoding)
        print(len(self.faces),type(self.faces[0]),type(targetEncoding))
        for i in range(0,6):
            print("处理",i,type(self.face[i]),type(targetEncoding))
            #similar1 = self.seetaFace.compare_feature_np(self.faces[i],targetEncoding)
            similar1=i
            if similar1>max:
                max=similar1
                index=i
        
        print("最大相似度： ",max, "\t 人脸name:",self.faceNames[index],"\t人脸特征:",self.faces[i])
        return self.faceNames[index]


class FaceData:
    def __init__(self,path):
        self.pic_path=path
        #self.initFaceDatatoFile()
        self.faces,self.faceNames=self.initFaceData()

    def initFaceData(self):
        known_faces=[]
        known_faceNames=[]
        if os.path.exists("all4.npz"):
            npz=np.load('all4.npz')
            known_faceNames = npz['names']
            known_faces=npz['encode']
        else:
            for file in os.listdir(self.pic_path):
                filepath=os.path.join(self.pic_path,file)
                print(filepath)
                image=face_recognition.load_image_file(filepath)
                try:
                    imageEncoding=face_recognition.face_encodings(image)[0]
                    known_faceNames.append(file.split('.')[0])
                    known_faces.append(imageEncoding)
                    print(known_faceNames[-1])
                except:
                    print("file don't detect face")
            np.savez('all4.npz', encode=known_faces, names=known_faceNames)
        return known_faces,known_faceNames

    def loadFaceDataFromFile(self):
        fw =open('face_info.json','w',encoding='utf-8')   #打开一个名字为‘user_info.json’的空文件
        face_dic=json.loads(fw.read())
        return face_dic.keys(),face_dic.values()

    def initFaceDatatoFile(self):
        known_faces=[]
        known_faceNames=[]
        for file in os.listdir(self.pic_path):
            filepath=os.path.join(self.pic_path,file)
            print(filepath)
            image=face_recognition.load_image_file(filepath)
            try:
                imageEncoding=face_recognition.face_encodings(image)[0]
                known_faceNames.append(file.split('.')[0])
                known_faces.append(imageEncoding)
                print(known_faceNames[-1])
            except:
                print("file don't detect face")
        d3 =dict(zip(known_faceNames, np.array(known_faces)))
        #print(d3)
        with open("./face_info.json","w",encoding='utf-8') as f:
            f.write(json.dump(known_faces, f, indent=4))
        

    def getFaceNameFromFile(self,targetPath):
        name="None"
        image=face_recognition.load_image_file(targetPath)
        try:
            face_encoding=face_recognition.face_encodings(image)[0]
            matches = face_recognition.compare_faces(self.faces, face_encoding)
            face_distances = face_recognition.face_distance(self.faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            #print("最小距离： ",face_distances[best_match_index])
            print(face_distances)
            if matches[best_match_index]:
                name = self.faceNames[best_match_index]
            return name
        except:
            print("file don't detect face")
            return name   

    
    def getFaceNameFromEncoding(self,targetEncoding):
        name="None"
        matches = face_recognition.compare_faces(self.faces, targetEncoding)
        face_distances = face_recognition.face_distance(self.faces, targetEncoding)
        best_match_index = np.argmin(face_distances)
        print("最小距离： ",face_distances[best_match_index])
        print(face_distances)
        if matches[best_match_index]:
            name = self.faceNames[best_match_index]
        return name

    def faceRecAcc(self,folder):
        cnt=0
        for tempfile in os.listdir(folder):
            filename=os.path.join(folder,tempfile)
            name=self.getFaceNameFromFile(filename)
            print(name)
            if name in ["葛伟平","刘冬冬","徐小龙"]:
                cnt=cnt+1
        print(folder," 总个数为:",len(os.listdir(folder))," 正确个数为:",cnt)

        

 
class TcpServer(object):
    """Tcp服务器"""
    def __init__(self, Port):
        """初始化对象"""
        self.code_mode = "utf-8"    #收发数据编码/解码格式
        self.server_socket = socket(AF_INET, SOCK_STREAM)   #创建socket
        self.server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, True)   #设置端口复用
        self.server_socket.bind(("", Port))     #绑定IP和Port
        self.server_socket.listen(100)  #设置为被动socket
        self.mutex=threading.Lock()
        self.facedata=FaceData("./picture")
        print("Listen...")
 
    def run(self):
        """运行"""
        while True:
            client_socket, client_addr = self.server_socket.accept()    #等待客户端连接
            print("{} online".format(client_addr))
            try:
                tr = Thread(target=self.recv_data, args=(client_socket, client_addr))   #创建线程为客户端服务
                tr.start()  #开启线程
            except:
	            print('face_server error')

 
    def recv_data(self, client_socket, client_addr):
        """收发数据"""
        while True:
            data = client_socket.recv(1024).decode(self.code_mode)
            if data:
                data=data.split(";")[0]
                print("{}:{}".format(client_addr, data))
                # 锁定资源
                self.mutex.acquire()
                name=self.facedata.getFaceNameFromFile("./realtime/"+data+".png")
                # 解锁资源
                self.mutex.release()
                print("人脸姓名:"+name)
                client_socket.send(name.encode(self.code_mode))
            else: 	#客户端断开连接
                print("{} offline".format(client_addr))
                break
 
        client_socket.close()
def localcameratestFaceData():
    video_capture = cv2.VideoCapture(0)
    facedata=FaceData("./picture")
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                name=facedata.getFaceNameFromEncoding(face_encoding)
                # 需要先把输出的中文字符转换成Unicode编码形式
                print("name",name)
                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def localcameratestSeetaFace():
    video_capture = cv2.VideoCapture(0)
    facedata=SeetaFaceHandle("./picture")
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame=frame
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            try:
                detect_result2 = facedata.seetaFace.Detect(rgb_small_frame)
                #print(detect_result2)
                face2 = detect_result2.data[0].pos
                #print(face2)
                points2 = facedata.seetaFace.mark5(rgb_small_frame,face2)
                feature2 = facedata.seetaFace.Extract(rgb_small_frame,points2)
                print("feature2",feature2)

            
                face_names = []
                name=facedata.getFaceNameFromEncoding(feature2)
                # 需要先把输出的中文字符转换成Unicode编码形式
                print("name",name)
                face_names.append(name)
            except :
                print("error")
               

        process_this_frame = not process_this_frame


        # Display the results
        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4

        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        cv2.waitKey(0)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #print("\033c", end="") 	#清屏
    serverfl=TcpServer(13333)
    serverfl.run()
    #localcameratestSeetaFace()
```

- C# client   

```c#
using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace KinectRFID.Util
{
    public class SocketUtil
    {
        private object _lockSend = new object();
        private Socket socket= null;
        private string host;
        private int port ;
        public SocketUtil() { }
        public SocketUtil(string _host, int _port)
        {
            host = _host;
            port = _port;
            connectToServer(host, port);
        }

        public void connectToServer(string host, int port)
        {
            if (socket == null || !socket.Connected)
            {
                try
                {
                    socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                    IPAddress ip = IPAddress.Parse(host);
                    IPEndPoint ipe = new IPEndPoint(ip, port);
                    socket.SendTimeout = 2000;
                    socket.ReceiveTimeout = 2000;
                    socket.SendBufferSize = 1024;
                    socket.ReceiveBufferSize = 1024;
                    socket.Connect(ipe);
                }
                catch (Exception)
                {
                    Console.WriteLine("connectVisualiseServer error，服务器没有开启");
                }
            }
        }
        public Socket GetSocket()
        {
            connectToServer(host, port);
            return socket;
        }

        #region Send
            /// <summary>
            /// Send
            /// </summary>
            public string SendReceive(string data)
        {
            lock (_lockSend)
            {
                //Encoding.ASCII.GetBytes(gestureDatas[gestureId].Id+";");

                //System.Diagnostics.Debug.WriteLine("Send data=" + data);
                byte[] lenArr = Encoding.UTF8.GetBytes(data);
                int sendTotal = 0;
                while (sendTotal < lenArr.Length)
                {
                    int sendOnce = socket.Send(lenArr, sendTotal, lenArr.Length - sendTotal, SocketFlags.None);
                    sendTotal += sendOnce;
                    Thread.Sleep(1);
                }
                //System.Diagnostics.Debug.WriteLine("send data ok, data=" + Encoding.UTF8.GetString(lenArr));

                try
                {        // 发送方 发送字符串，文件名前缀
                    int block = 1024;
                    byte[] buffer = new byte[block];
                    int receiveCount = socket.Receive(buffer, 0, block, SocketFlags.None);
                    if (receiveCount == 0)
                    {
                        return null;
                    }
                    else
                    {
                        //System.Diagnostics.Debug.WriteLine("recieve data ok, data=" + buffer.ToString());
                        System.Diagnostics.Debug.WriteLine("SendReceive: recieve data= " + Encoding.UTF8.GetString(buffer));
                        return Encoding.UTF8.GetString(buffer);
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine("接收数据出错：" + ex.Message + "\r\n" + ex.StackTrace);
                    return null;
                }

            }

        }
        #endregion



            #region 断开服务器
            /// <summary>
            /// 断开服务器clientSocketFace
            /// </summary>
            public  void DisconnectServer(Socket socket)
        {
            try
            {
                if (socket != null)
                {
                    if (socket.Connected)
                        socket.Disconnect(false);
                    socket.Close();
                    socket.Dispose();
                }
                System.Diagnostics.Debug.WriteLine("已断开服务器");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("断开服务器失败：" + ex.Message);
            }
        }
        #endregion

    }
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/socket/  

