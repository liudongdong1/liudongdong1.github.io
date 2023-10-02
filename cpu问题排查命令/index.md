# CPU问题排查命令


**方法一**

1. `top`命令，然后按shift+p按照CPU排序 ，找到占用CPU过高的进程
2. `ps -mp pid -o THREAD,tid,time | sort -rn `，获取线程信息，并找到占用CPU高的线程
3. `echo 'obase=16;[线程id]' | bc` 或者`printf "%x " [线程id] `，将需要的线程ID转换为16进制格式
4. `jstack pid |grep tid -A 30 [线程id的16进制]  ` ，打印线程的堆栈信息

### ps命令

```shell
# 查看内存占用前10位：
ps aux | head -1;ps aux |grep -v PID |sort -rn -k +4 | head -10
ps aux --sort -rss | head -n 10

#查看CPU占用前10位：
ps aux | head -1;ps aux |grep -v PID |sort -rn -k +3 | head -10
ps aux --sort -pcpu | head -n 10
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/cpu%E9%97%AE%E9%A2%98%E6%8E%92%E6%9F%A5%E5%91%BD%E4%BB%A4/  

