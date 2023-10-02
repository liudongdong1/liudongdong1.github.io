# WSGI_Learning


> WSGI有两方：“服务器”或“网关”一方，以及“应用程序”或“应用框架”一方。服务方调用应用方，提供环境信息，以及一个回调函数（提供给应用程序用来将消息头传递给服务器方），并接收Web内容作为返回值。WSGI 的设计参考了 Java 的 servlet。

- 重写环境变量后，根据目标URL，将请求消息路由到不同的应用对象。
- 允许在一个进程中同时运行多个应用程序或应用框架。
- 负载均衡和远程处理，通过在网络上转发请求和响应消息。
- 进行内容后处理，例如应用XSLT样式表。
- uwsgi的启动可以把参数加载命令行中，也可以是配置文件 .ini, .xml, .yaml 配置文件中

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210626221123557.png)

```python
uwsgi --ini uwsgi.ini             # 启动
uwsgi --reload uwsgi.pid          # 重启
uwsgi --stop uwsgi.pid            # 关闭
uwsgi --connect-and-read uwsgi/uwsgi.status # 读取实时状态
```

- 如何理解Nginx, WSGI, Flask之间的关系： [http://blog.csdn.net/lihao21/article/details/52304119](https://link.jianshu.com?t=http://blog.csdn.net/lihao21/article/details/52304119)
- uWSGI的安装与配置： [http://blog.csdn.net/chenggong2dm/article/details/43937433](https://link.jianshu.com?t=http://blog.csdn.net/chenggong2dm/article/details/43937433)

- uWSGI实战之操作经验： [http://blog.csdn.net/orangleliu/article/details/48437319](https://link.jianshu.com?t=http://blog.csdn.net/orangleliu/article/details/48437319)

- nginx配置参考： [http://wiki.nginx.org/HttpUwsgiModule#uwsgi_param](https://link.jianshu.com?t=http://wiki.nginx.org/HttpUwsgiModule#uwsgi_param)

- uwsgi安装参考： [http://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html](https://link.jianshu.com?t=http://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html)

- uwsgi配置参考： [http://uwsgi-docs.readthedocs.io/en/latest/Options.html#vacuum](https://link.jianshu.com?t=http://uwsgi-docs.readthedocs.io/en/latest/Options.html#vacuum)

- Nginx+uWSGI： [https://my.oschina.net/guol/blog/121418](https://link.jianshu.com?t=https://my.oschina.net/guol/blog/121418)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/wsgi_learning/  

