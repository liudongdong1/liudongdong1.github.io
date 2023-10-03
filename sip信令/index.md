# SIP信令


> 用于多方多媒体通信的框架协议。它是一个基于文本的应用层控制协议, 独立于底层传输协议, 用于建立、修改和终止IP网络上的双方或多方多媒体会话。

#### 1. 概念介绍

> 信令是控制电路的信号，是终端和终端、终端和网络之间传递的一种消息，专门用来`控制电路，建立、管理、删除连接`，以使用户能够正常通过这些连接进行通信。
> 允许程控交换、网络数据库、网络中其它“智能”节点交换下列有关信息：`呼叫建立、监控（Supervision）、拆除（Teardown）、分布式应用进程所需的信息（进程之间的询问/响应或用户到用户的数据）、网络管理信息。`
>
> 信令通常需要在通信网络的`不同环节(基站、移动台和移动控制交换中心等)之间传输`，各环节进行分析处理并通过交互作用而形成一系列的操作和控制，其作用是**保证用户信息的有效且可靠的传输**。信令网关就是两个不同的网之间信令互通设备。

- 呼叫： 由一个公共源端所邀请的在一个会议中的所有参加者组成，由一个全球唯一的Call-ID 进行标识。
- 事务：客户和服务器之间的操作从第 1 个请求至最终响应为止的所有消息构成一个SIP事务 。呼叫启动包含两个操作请求：邀请（ Invite）和证实（ ACK），前者需要回送响应，后者只是证实已收到最终响应，不需要回送响应。呼叫终结包含一个操作请求：再见（ Bye）。
- SIP URL(uniform resource locators): **SIP: 用户名：口令 @ 主机：端口；传送参数：用户参数；方法参数；生存期参数；服务器地址参数？头部名=头部值**
  - 寻址，即采用什么样的地址形式标识终端用户；
  - 用户定位。
- 定位服务： SIP重定位服务器或代理服务器用来获得被叫位置的一种服务， 可由定位服务器提供。
- 代理服务器： 路由、 认证鉴权、 计费监控、 呼叫控制、 业务提供等。

#### 2. 会话事务

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20201218110509154.png)

#### 3. SIP 协议栈

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201218110913164.png)

> RSVP（ Resource ReServation Protocol ）用于`预约网络资源`，
> RTP（ Real-time Transmit Protocol ）用于传输`实时数据并提供服务质量（ QoS ）反馈`，
> RTSP （ Real-Time Stream Protocol ）用于控制`实时媒体流的传输`，
> SAP（ Session Announcement Protocol ）用于通过组播发布多媒体会话，
> SDP（ Session Description Protocol ）用于描述多媒体会话。

#### 4. SIP 消息

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201218111036570.png)

请求消息和响应消息的格式，一般由起始行、若干个消息头和消息体构成。
SIP一般消息 = 起始行
           *消息头
           CELF（空行）
          [消息体]

- 起始行 = 请求行、状态行（SIP请求消息起始行是请求行（Request-Line），响应消息起始行是状态行（Status-Line））
- 请求消息头至少包括 From, To, CSeq, Call-ID, Max-Forwards, Via六个头字段，它们是构建SIP消息基本单元
- 消息体一般采用SDP（Session Description Protocol）协议，会话描述协议

#### 5. SIP 监控域互联结构

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201218111226661.png)

#### 6. 学习链接

- https://www.cnblogs.com/11sgXL/p/13553517.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/sip%E4%BF%A1%E4%BB%A4/  

