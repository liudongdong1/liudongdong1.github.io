# KaliTutorial


#### 1. nmap

> 使用原始 IP 报文来发现`网络上有哪些主机`，那些 主机`提供什么服务(应用程序名和版 本)`，那 些服务运行在`什么操作系统(包括版本信息)`， 它们`使用什么类型的报文过滤器/防火 墙`，以及一堆其它功能。

```shell
# 快速扫描
nmap -A -T4 ip 
#扫描该ip的所有端口信息 
nmap -sS -p 1-65535 -v ip
#探测操作系统，默认也扫描了端口
nmap -O ip
#穿透防火墙全面扫描
nmap -Pn -A ip -Pn
#进行ping扫描，打印出对扫描做出响应的主机，不做进一步测试(如端口扫描或者操作系统探测)：
nmap -sP 192.168.1.0/24
#仅列出指定网络上的每台主机，不发送任何报文到目标主机：
nmap -sL 192.168.1.0/24
#探测目标主机开放的端口，可以指定一个以逗号分隔的端口列表(如-PS22，23，25，80)：
nmap -PS 192.168.1.234
#使用UDP ping探测主机：
nmap -PU 192.168.1.0/24
#扫描开放端口号
nmap -vv 10.1.1.254

#路由器跟踪扫描
nmap -traceroute www.baidu.com
```

- Function: `Host Discovery`;  `Scan Techniques`; `Target Specification`; `Service Version Detection`; `Script Scan`;`OS detection`;

- [高级用法](https://link.zhihu.com/?target=https%3A//nmap.org/nsedoc/)

```
nmap --script 具体的脚本 www.baidu.com
```

> - auth: 负责处理鉴权证书绕开鉴权的脚本  
> - broadcast: 在局域网内探查更多服务开启状况如dhcp/dns/sqlserver等服务  
> - brute: 提供暴力破解方式针对常见的应用如http/snmp等  
> - default: 使用-sC或-A选项扫描时候默认的脚本提供基本脚本扫描能力  
> - discovery: 对网络进行更多的信息如SMB枚举、SNMP查询等  
> - dos: 用于进行拒绝服务攻击  
> - exploit: 利用已知的漏洞入侵系统  
> - external: 利用第三方的数据库或资源例如进行whois解析  
> - fuzzer: 模糊测试的脚本发送异常的包到目标机探测出潜在漏洞 
> - intrusive: 入侵性的脚本此类脚本可能引发对方的IDS/IPS的记录或屏蔽- malware: 探测目标机是否感染了病毒、开启了后门等信息  
> - safe: 此类与intrusive相反属于安全性脚本  
> - version: 负责增强服务与版本扫描Version Detection功能的脚本  
> - vuln: 负责检查目标机是否有常见的漏洞Vulnerability如是否有MS08_067

#### 2.searchsploit 

```shell
searchploit filename  #用于检索文件名
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kalitutorial/  

