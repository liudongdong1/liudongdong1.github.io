# qBittorrent


> “磁力链接”的主要作用是识别【能够通过“点对点技术（即：P2P）”下载的文件】。这种链接是`通过不同文件内容的Hash结果生成一个纯文本的“数字指纹”，来识别文件的`。而不是基于文件的位置或者名称。这个“数字指纹”可以被任何人从任何文件上生成，这也就注定了“磁力链接”不需要任何“中心机构”的支持（例如：BT Tracker服务器），且识别准确度极高。因此任何人都可以生成一个Magnet链接并确保通过该链接下载的文件准确无误。

> Magnet URI全称为Magnet Uniform Resource Identifier即“`磁力统一资源定位名`”，其主要支持参数(即组成部分)如下：dn (显示名称)-文件名、xl (绝对长度)-文件字节数、xt(eXact Topic)-包含文件散列函数的URN、as(Acceptable Source)-Web link to the file online、xs (绝对资源)-P2P链接、kt(关键字)-用于搜索的关键字、mt(文件列表)-链接到一个包含magnet链接的元文件(MAGMA – MAGnet Manifest)、tr(Tracker 地址)-BT下载的Tracker URL。

> `DHT(Distributed Hash Table，分布式哈希表)`类似Tracker的根据种子特征码返回种子信息的网络.DHT全称叫分布式哈希表(Distributed Hash Table)，是`一种分布式存储方法`。在不需要服务器的情况下，每个客户端负责一个小范围的路由，并负责存储一小部分数据，从而实现整个DHT网络的寻址和存储。新版BitComet允许同行连接DHT网络和Tracker，也就是说在完全不连上[Tracker服务器的情况下，也可以很好的下载，因为它可以在DHT网络中寻找下载同一文件的其他用户。BitComet的DHT网络协议和BitTorrent今年5月测试版的协议完全兼容，也就是说可以连入一个同DHT网络分享数据。 

### 1. 下载安装

> **qBittorrent是一款安全、合法的下载软件**，可在国内各大软件平台搜索下载，也可以前往**qBittorrent官网**下载，网址： [https://www.qbittorrent.org](https://www.qbittorrent.org/)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210731115341128.png)

### 2. 使用方法

#### .1. 添加tracker

> 添加 **trackers** 可以帮助我们连接到更多的资源节点，**解决下载没速度的问题，给下载加速**，是必备操作！打开 qBittorrent -> 工具 -> 选项 -> BitTorrent，下拉到末尾，将 trackers 粘贴进输入框，并勾选 “自动添加以下 trackers 到新的 torrents”

- https://newtrackon.com/list
- [GitHub - ngosang/trackerslist: An updated list of public BitTorrent trackers](https://link.zhihu.com/?target=https%3A//github.com/ngosang/trackerslist)
- [TrackersListCollection: 每天更新！全网热门公共 BitTorrent Tracker 列表](https://link.zhihu.com/?target=https%3A//trackerslist.com/all.txt)
- Traker List: https://github.com/ngosang/trackerslist
- [ACG领域的开放式Tracker](https://link.zhihu.com/?target=https%3A//acgtracker.com)
- 动漫资源tracker:  https://github.com/DeSireFire/animeTrackerList
- https://trackerslist.com/#/zh                                       国人大佬在进行维护
- https://github.com/ngosang/trackerslist                     ngosang 老外用得多
- https://github.com/DeSireFire/animeTrackerList       偏重动漫资源
- https://opentrackers.org/
- https://newtrackon.com/list                                        这两个列表质量也挺高 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210731141119551.png)

#### .2. 尝试多个BT种子、磁力链接，选择最有人气的

> 因资源类型、种子发布时间不同，BT种子、磁力链接都会有冷热门之分，`越热门用户越多，下载速度就会越快`。如果BT种子、磁力链接使用的用户少，就会导致下载慢，或者成为死链。这时候，只能更换为其他的种子、链接进行下载。

#### .3. 开启路由器的UPnP / NAT-PMP 功能

> 在qBittorrent的“**选项**”-“**连接**”中，选中“**使用我的路由器的UPnP / HAT -PMP 端口转发**”，再进入自己的路由器管理页面，开启路由器的“**UPnP / NAT-PMP**”功能。

#### .4. 设置更高的下载优先级别

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210731141249675.png)

#### .5. 搜索引擎插件

- https://github.com/qbittorrent/search-plugins/wiki/Unofficial-search-plugins

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210731145736859.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210731151934185.png)

### 3. 种子网址

- [海盗湾](https://thepiratebay10.org/)——最佳综合性种子网站
- [RARBG](https://rarbg.to/index70.php)——最佳4K电影和新上线内容种子网站
- [1337x](https://1337x.to/)——最佳应用程序种子网站
-  [Torlock](https://www.torlock.com/movies/1/size/desc.html)——最佳动漫、电子书种子网站
- [Torrentz2](https://torrentz2eu.org/)——最佳音乐专辑种子网站
- [YTS](https://yts.mx/)——最佳电影资源种子网站
- [EZTV](https://eztv.re/)——最佳电视节目种子网站
- [TorrentDownloads](https://torrentdownloads.mrunblock.best/)——最佳软件种子网站
- https://yts.mx/  电影
- https://github.com/qbittorrent/search-plugins/wiki/Unofficial-search-plugins#plugins-for-public-sites 汇总

1、状态栏的 “插头” 是黄色的（不是绿色）：不影响下载，只有文件有上传了才会变绿。

2、添加磁力链接，qBittorrent 一直显示 “正在下载元数据”：这个最常见了，相当于在等待下载种子信息。**强烈建议使用种子文件进行下载**，就不会出现这个了。

3、`连不上 DHT 节点：先下一个热门种子试试，DHT 涨起来了再下载`。

4、理论上 trackers 和 DHT 网络有一种连上了在工作，就有速度。还没速度查看下用户、种子、trackers 等的连接情况，如果都为 0 可能是死链，建议换热门的资源。

5`、磁力种子的下载速度依赖于资源数量和广大用户的上传做种，上传分享的人越多，速度越快。`

6、`建议下载完后不要急着移除任务，上传做种，可以设置下上传速度或时间`，原因就是上一条。

### Resource

- https://baiyunju.cc/3352

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/qbittorrent/  

