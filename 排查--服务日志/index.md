# 排查--服务日志


### journalctl使用案例

1. 尝试启动 apache2 服务：

   ```shell
   # systemctl start apache2
   Job for apache2.service failed. See 'systemctl status apache2' and 'journalctl -xn' for details.
   ```

2. 我们来看看该服务的状态如何：

   ```shell
   # systemctl status apache2
   apache2.service - The Apache Webserver
      Loaded: loaded (/usr/lib/systemd/system/apache2.service; disabled)
      Active: failed (Result: exit-code) since Tue 2014-06-03 11:08:13 CEST; 7min ago
     Process: 11026 ExecStop=/usr/sbin/start_apache2 -D SYSTEMD -DFOREGROUND \
              -k graceful-stop (code=exited, status=1/FAILURE)
   ```

3. 显示与进程 ID 11026 相关的详细讯息：

   ```shell
   # journalctl -o verbose _PID=11026
   [...]
   MESSAGE=AH00526: Syntax error on line 6 of /etc/apache2/default-server.conf:
   [...]
   MESSAGE=Invalid command 'DocumenttRoot', perhaps misspelled or defined by a module
   [...]
   ```

4. 修复 `/etc/apache2/default-server.conf` 中的拼写错误，启动 apache2 服务，然后列显其状态：

   ```shell
   # systemctl start apache2 && systemctl status apache2
   apache2.service - The Apache Webserver
      Loaded: loaded (/usr/lib/systemd/system/apache2.service; disabled)
      Active: active (running) since Tue 2014-06-03 11:26:24 CEST; 4ms ago
     Process: 11026 ExecStop=/usr/sbin/start_apache2 -D SYSTEMD -DFOREGROUND
              -k graceful-stop (code=exited, status=1/FAILURE)
    Main PID: 11263 (httpd2-prefork)
      Status: "Processing requests..."
      CGroup: /system.slice/apache2.service
              ├─11263 /usr/sbin/httpd2-prefork -f /etc/apache2/httpd.conf -D [...]
              ├─11280 /usr/sbin/httpd2-prefork -f /etc/apache2/httpd.conf -D [...]
              ├─11281 /usr/sbin/httpd2-prefork -f /etc/apache2/httpd.conf -D [...]
              ├─11282 /usr/sbin/httpd2-prefork -f /etc/apache2/httpd.conf -D [...]
              ├─11283 /usr/sbin/httpd2-prefork -f /etc/apache2/httpd.conf -D [...]
              └─11285 /usr/sbin/httpd2-prefork -f /etc/apache2/httpd.conf -D [...]
   ```



### Resource

- https://documentation.suse.com/zh-cn/sles/12-SP4/html/SLES-all/cha-journalctl.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/%E6%8E%92%E6%9F%A5--%E6%9C%8D%E5%8A%A1%E6%97%A5%E5%BF%97/  

