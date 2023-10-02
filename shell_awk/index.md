# shell_AWK


>awk 是报表生成工具，同样是逐行取出文件，但是取出的目的是对内容进行二次加工，然后将有用的数据单独格式化输出、或进行归纳统计得到统计结果等。` awk -F: '{print $1,$3}' /etc/passwd`
>
>1. awk 使用一行作为输入，并将这一行赋给内部变量 $0，每一行也可称为一个记录，以换行符 (RS) 结束
>
>2. 每行被间隔符 **==:==**(默认为空格或制表符) 分解成字段 (或域)，每个字段存储在已编号的变量中，从 $1 开始
>
> 3. awk 使用 print 函数打印字段，打印出来的字段会以 == 空格分隔 ==，因为 \$1,\$3 之间有一个逗号。逗号比较特殊，它映射为另一个内部变量，称为 == 输出字段分隔符 ==OFS，OFS 默认为空格
>
> 4. awk 处理完一行后，将从文件中获取另一行，并将其存储在 $0 中，覆盖原来的内容，然后将新的字符串分隔成字段并进行处理。该过程将持续到所有行处理完毕

#### 1. 命令格式

```shell
awk 选项 "模式或条件 {操作}"  文件1 文件2
awk -f 脚本文件 文件1 文件2
# -v var=value : 外键变量定义,例如awk -v var1='v1'
awk [options] 'BEGIN{} Pattern{Action} END{}' file1,file2
awk [optioms] 'BEGIN{}' file1,file2
awk [options] 'Pattern{Action}' file1,file2
awk [options] 'Pattern{Action} END{}' file1,file2
```
##### .1. **模式定义**

- / 正则表达式 /：使用通配符的扩展集。
- 关系表达式：可以用下面运算符表中的关系运算符进行操作，可以是字符串或数字的比较，如 $2>%1 选择第二个字段比第一个字段长的行。
- 模式匹配表达式：用运算符～(匹配) 和～！(不匹配)。
- 模式，模式：指定一个行的范围。该语法不能包括 BEGIN 和 END 模式。
- BEGIN：让用户指定在第一条输入记录被处理之前所发生的动作，通常可在这里设置全局变量。
- END：让用户在最后一条输入记录被读取之后发生的动作。

| 内建变量 | 说明                                                         |
| :------- | :----------------------------------------------------------- |
| FS       | `列分割符`，制定和每行文本的字段分割符，默认为空格或者制表符 |
| NF       | 表示当前的列数；                                             |
| NR       | 表示当前的行数；                                             |
| $0       | `当前处理行的整行内容`                                       |
| $n       | 当前处理行的`第 n 个字段（第 n 列）`                         |
| FILENAME | 被处理的文件名                                               |
| RS       | 行分隔符，awk 从文本上读取资料时，将`根据 RS 的定义把资料切割成许多条记录`，而 awk 一次仅读入一条，以进行处理，预设值是 \n |
| OFS      | 输出列分隔符，用于打印时分割字段，默认为空格                 |
| ORS      | 输出行分隔符，用于打印时分割记录，默认为换行符               |

##### .2. 内置函数

- `index(s, t)` 返回子串 t 在 s 中的位置
- `length(s)` 返回字符串 s 的长度
- `split(s, a, sep)` 分割字符串，并将分割后的各字段存放在数组 a 中
- `substr(s, p, n)` 根据参数，返回子串, 从1开始计数
- `tolower(s)` 将字符串转换为小写
- `toupper(s)` 将字符串转换为大写

##### .3. 从脚本文件执行

```shell
#! /bin/awk -f
BEGIN {
	math=0
	englisth=0
	computer=0
	printf "NAME  NO.  MATH  ENGLISH  COMPUTER  TOTAL\n"
	printf "-------------------------------------------"
}
{
	math+=$3
	english+=$4
	computer+=$5
	printf "%-6s  %-6s  %4d  %8d  %8d  %8d \n", $1, $2, $3, $4, $5, $3+$4+$5
}
END{
	printf "--------------------------\n"
	printf " total : %10d  %8d  %8d  \n", math, english,  computer
	printf " AVERAGE: %10.3f  %8.2f  %8.2f \n", math/NR, english/NR, computer/NR
}
```

- awk -f cal.awk sore.txt
- 在 `BEGIN` 阶段，我们初始化了相关变量，并打印了表头的格式
- 在 `body` 阶段，我们读取每一行数据，计算该学科和该同学的总成绩
- 在 `END` 阶段，我们先打印了表尾的格式，并打印总成绩，以及计算了平均值

#### 2. 按行输出

##### .1. 使用NR

```shell
awk '(NR>=1)&&(NR<=4){print}' 11.txt#输出第一到第四行
awk '(NR==1)||(NR==4){print}' 11.txt#输出第一和第四行
awk 'NR==1,NR==2{print}' 11.txt#输出第一和第二行
awk '(NR%2)==1{print}' 11.txt  #输出奇数行
awk '(NR%2)==0{print}' 11.txt  #输出偶数行
awk '/^1/{print}' 11.txt       #输出以1为开头的行
awk '/2$/{print}' 11.txt       #输出以2为结尾的行
awk 'BEGIN{x=0};/^2/{x++};END {print x}' 11.txt #统计以2开头的行的行数 
#BEGIN 模式表示，在处理指定文本之前，需要先执行 BEGIN 模式中指定的动作，awk 再处理指定的文本，之后再执行 END 模式中指定的动作，END {} 语句块中，往往会放入打印结果等语句。
```

##### .2. getline

- 当 getline `左右无重定向符 “<” 或 “|” 时`，getline 作用于当前文件，读入当前文件的第一行给其后跟的变量 var 或 $0；应该注意到，由于` awk 在处理 getline 之前已经读入了一行，所以 getline 得到的返回结果是隔行的`。
- 当 getline` 左右有重定向符 “<” 或 “|” 时`，getline 则作用于定向输入文件，由于该文件是刚打开，并没有被 awk 读入一行，只是 getline 读入，那么 getline 返回的是该文件的第一行，而不是隔行。

```shell
awk '{getline; print $0}' l1.txt    # awk 先读取，然后getline 读取， 打印偶数行
awk '{print $0; getline}' l1.txt    # 打印                            奇数行
```

#### 3. 以字段输出文本

```shell
echo $PATH | awk -F ":" '{print $1}'  #以“：”为分隔符，并打印第一个字段
awk -F ":" '{max=($3>$4)?$3:$4;{print max}}'   
awk -F ":" '/bash$/{print | "wc -l"}' /etc/passwd   #查看passwd中以bash结尾有多少行
```

#### 4. 案例

##### .1. 取进程运行时间 + 进程程序取前五

```shell
ps axu | awk -v OFS=" " '{print $10,$11}' | sort -k1 -nr | head -n 10  # OFS 输出默认分隔符
```

##### .2. 统计 `nginx` 日志中出现的 ip 次数，取前 10 个 ip

```shell
awk '/^[0-9]/{ip[$1]++}END{for(i in ip){print i,ip[i]}}' www.i7dom.cn.log |sort -k2 -nr | head -n 10
```

##### .3.取出 `netstat` 的简略信息已报表方式显示

```shell
netstat -luntp | grep -E '^tcp' | awk 'BEGIN{printf "%-10s\t%-10s\n","PID/Program name","Proto"}{printf "%-10s\t\t%-10s\n",$NF,$1}'
```

##### .4. 取出 `/etc/passwd` 的用户已 'r' 开头的用户 

```shell
awk '/^r/{print $0}' /etc/passwd
```

##### .5. **分析访问日志（Nginx 为例）**

```shell
#日志格式：
'$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" "$http_x_forwarded_for"'

#统计访问IP次数：
awk '{a[$1]++}END{for(v in a)print v,a[v]}' access.log
#统计访问访问大于100次的IP：
awk '{a[$1]++}END{for(v in a){if(a[v]>100)print v,a[v]}}' access.log
# 统计访问IP次数并排序取前10：
awk '{a[$1]++}END{for(v in a)print v,a[v]|"sort -k2 -nr |head -10"}' access.log
# 统计时间段访问最多的IP：
awk'$4>="[02/Jan/2017:00:02:00" &&$4< ="[02/Jan/2017:00:03:00"{a[$1]++}END{for(v in a)print v,a[v]}'access.log
# 统计上一分钟访问量：
date=$(date -d '-1 minute'+%d/%d/%Y:%H:%M)
awk -vdate=$date '$4~date{c++}END{printc}' access.log
# 统计访问最多的10个页面：
awk '{a[$7]++}END{for(vin a)print v,a[v]|"sort -k1 -nr|head -n10"}' access.log
# 统计每个URL数量和返回内容总大小：
awk '{a[$7]++;size[$7]+=$10}END{for(v ina)print a[v],v,size[v]}' access.log
# 统计每个IP访问状态码数量：
awk '{a[$1" "$9]++}END{for(v ina)print v,a[v]}' access.log
# 统计访问IP是404状态次数：
awk '{if($9~/404/)a[$1" "$9]++}END{for(i in a)print v,a[v]}' access.log
```

##### .6. 统计

- 统计当前目录下，所有 `*.c`、`*.h` 文件所占用空间大小总和

```shell
ls -l *.c *.h | awk '{sum+=$5} END {print sum}'
```

##### .7. **去除文本第一行和最后一行**

```shell
seq 5 |awk'NR>2{print s}{s=$0}'
#读取第一行，NR=1，不执行print s，s=1
#读取第二行，NR=2，不执行print s，s=2 （大于为真）
#读取第三行，NR=3，执行print s，此时s是上一次p赋值内容2，s=3
#最后一行，执行print s，打印倒数第二行，s=最后一行
```

##### .8. **获取 Nginx upstream 块内后端 IP 和端口**  todo

```shell
cat a
# upstream example-servers1 {
#    server 127.0.0.1:80 weight=1 max_fails=2fail_timeout=30s;
# }
# upstream example-servers2 {
#    server 127.0.0.1:80 weight=1 max_fails=2fail_timeout=30s;
#    server 127.0.0.1:82 backup;
# }
awk '/example-servers1/,/}/{if(NR>2){print s}{s=$2}}' a  
# 127.0.0.1:80
awk '/example-servers1/,/}/{if(i>1)print s;s=$2;i++}' a
awk '/example-servers1/,/}/{if(i>1){print s}{s=$2;i++}}' a
# 127.0.0.1:80
```

##### .9. **获取某列数字最大数**

```shell
# 获取第三字段最大值：
awk 'BEGIN{max=0}{if($3>max)max=$3}END{print max}' a
# 打印第三字段最大行：
awk 'BEGIN{max=0}{a[$0]=$3;if($3>max)max=$3}END{for(v in a)if(a[v]==max)print v}'a
```

##### .10. **费用统计**

```shell
cat a
# zhangsan 8000 1
# zhangsan 5000 1
# lisi 1000 1
awk '{name[$1]++;cost[$1]+=$2;number[$1]+=$3}END{for(v in name)print v,cost[v],number[v]}' a
```

##### .11. 统计字符串中每个字母出现的次数：

```shell
echo "a.b.c,c.d.e" |awk -F'[.,]' '{for(i=1;i< =NF;i++)a[$i]++}END{for(v in a)print v,a[v]}'
```

##### .12. 字符串拆分：

```shell
echo "hello" |awk -F '' '{for(i=1;i< =NF;i++)print $i}'
```

##### .13. **将第一列合并到一行**  todo

```shell
cat file
# 1 2 3
# 4 5 6
# 7 8 9
awk '{for(i=1;i< =NF;i++)a[i]=a[i]$i" "}END{for(v in a)print a[v]}' file
```

>for循环是遍历每行的字段，NF等于3，循环3次。
>读取第一行时：
>第一个字段：a[1]=a[1]1" "  值a[1]还未定义数组，下标也获取不到对应的值，所以为空，因此a[1]=1 。
>第二个字段：a[2]=a[2]2" "  值a[2]数组a已经定义，但没有2这个下标，也获取不到对应的值，为空，因此a[2]=2 。
>第三个字段：a[3]=a[3]3" "  值a[2]与上面一样，为空,a[3]=3 。
>读取第二行时：
>第一个字段：a[1]=a[1]4" "  值a[2]获取数组a的2为下标对应的值，上面已经有这个下标了，对应的值是1，因此a[1]=1 4
>第二个字段：a[2]=a[2]5" "  同上，a[2]=2 5
>第三个字段：a[3]=a[3]6" "  同上，a[2]=3 6
>读取第三行时处理方式同上，数组最后还是三个下标，分别是1=1 4 7，2=2 5 8，3=36 9。最后for循环输出所有下标值。

#### Resource

- https://www.jb51.net/article/233625.htm
- https://www.cnblogs.com/mq0036/p/14541876.html

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shell_awk/  

