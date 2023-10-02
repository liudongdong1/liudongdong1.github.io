# shell_scriptRecord


#### 1. 内存监控脚本

```shell
#!/bin/bash
#memory use
mem_war_file=/tmp/mem_war.txt
mem_use=`free -m | grep Mem | awk '{print $3}'`
mem_total=`free -m | grep Mem | awk '{print $2}'`
mem_percent=$((mem_use*100/mem_total))
# echo "$mem_percent"%
if (($mem_percent > 80));then
   echo "`date +%F-%H-%M` mem: ${mem_percent}%" >$mem_war_file
   echo "`date +%F-%H-%M` mem: ${mem_percent}%" | mail -s "mem warning" root 
fi
```


```shell
#!/bin/bash
#######################################################
#检测网卡流量，并按规定格式记录在日志中
#规定一分钟记录一次
#日志格式如下所示:
#2021-07-08 18:55
#eth0 input: 1234bps
#eth0 output: 1235bps
######################################################3
while :
do
#设置语言为英文，保障输出结果是英文，否则会出现bug
LANG=en
logfile=/tmp/`date +%d`.log
#将下面执行的命令结果输出重定向到logfile日志中
exec >> $logfile
date +"%F %H:%M"
#sar命令统计的流量单位为kb/s，日志格式为bps，因此要*1000*8
sar -n DEV 1 59|grep Average|grep eth0|awk '{print $2,"\t","input:","\t",$5*1000*8,"bps","\n",$2,"\t","output:","\t",$6*1000*8,"bps"}'
echo "####################"
#因为执行sar命令需要59秒，因此不需要sleep
done
```


```shell
#场景：
#1.访问日志文件的路径：/data/log/access.log
#2.脚本死循环，每10秒检测一次，10秒的日志条数为300条，出现502的比例不低于10%（30条）则需要重启php-fpm服务
#3.重启命令为：/etc/init.d/php-fpm restart
#!/bin/bash
###########################################################
#监测Nginx访问日志502情况，并做相应操作
###########################################################
log=/data/log/access.log
N=30 #设定阈值
while :
do
 #查看访问日志的最新300条，并统计502的次数
    err=`tail -n 300 $log |grep -c '502" '`
 if [ $err -ge $N ]
 then
 /etc/init.d/php-fpm restart 2> /dev/null
 #设定60s延迟防止脚本bug导致无限重启php-fpm服务
     sleep 60
 fi
 sleep 10
done
```

#### 4. 扫描主机端

```shell
#!/bin/bash
HOST=$1
PORT="22 80 8080 3306"
for PORT in $PORT; do
    if echo &>/dev/null > /dev/tcp/$HOST/$PORT; then
        echo "$PORT open"
    else
        echo "$PORT close"
    fi
done
```

#### 5. 检测两台服务器某个目录下的文件一致性口状态

```shell
#!/bin/bash
#####################################
#检测两台服务器指定目录下的文件一致性
#####################################
#通过对比两台服务器上文件的md5值，达到检测一致性的目的
dir=/data/web
b_ip=192.168.88.10
#将指定目录下的文件全部遍历出来并作为md5sum命令的参数，进而得到所有文件的md5值，并写入到指定文件中
find $dir -type f|xargs md5sum > /tmp/md5_a.txt
ssh $b_ip "find $dir -type f|xargs md5sum > /tmp/md5_b.txt"
scp $b_ip:/tmp/md5_b.txt /tmp
#将文件名作为遍历对象进行一一比对
for f in `awk '{print 2} /tmp/md5_a.txt'`
do
#以a机器为标准，当b机器不存在遍历对象中的文件时直接输出不存在的结果
if grep -qw "$f" /tmp/md5_b.txt
then
md5_a=`grep -w "$f" /tmp/md5_a.txt|awk '{print 1}'`
md5_b=`grep -w "$f" /tmp/md5_b.txt|awk '{print 1}'`
#当文件存在时，如果md5值不一致则输出文件改变的结果
if [ $md5_a != $md5_b ]
then
echo "$f changed."
fi
else
echo "$f deleted."
fi
done
```


```shell
#!/bin/bash
################################################################
#每小时执行一次脚本（任务计划），当时间为0点或12点时，将目标目录下的所有文件内容清空，但不删除文件，其他时间则只统计各个文件的大小，一个文件一行，输出到以时间和日期命名的文件中，需要考虑目标目录下二级、三级等子目录的文件
################################################################
logfile=/tmp/`date +%H-%F`.log
n=`date +%H`
if [ $n -eq 00 ] || [ $n -eq 12 ]
then
#通过for循环，以find命令作为遍历条件，将目标目录下的所有文件进行遍历并做相应操作
for i in `find /data/log/ -type f`
do
true > $i
done
else
for i in `find /data/log/ -type f`
do
du -sh $i >> $logfile
done
fi
```

#### 7. 查看局域网内主机是否存活

```shell
#!/usr/bin/bash
# check host status
for i in {1..254}
do
        {
    ip=192.168.8.$i
    ping -c 2 -W 1 $ip &>/dev/null
    if [ $? -eq 0 ];then
        echo "$ip is online" | tee -a /tmp/host_online.txt
    else
       # echo "$ip is offline" | tee -a /tmp/host_offline.txt
       echo "$ip is offline" &>/dev/null
    fi
        }&
done
wait
```


```shell
#!/usr/bin/bash
# ssh keygen

>ip_ok.txt
>ip_false.txt
user=root
passwd=123456

rpm -qa | grep expect &>/dev/null
if [ $? -ne 0 ];then
  echo "expect is not install"
  yum -y install expect
fi

if [ ! -f ~/.ssh/id_rsa ];then
  ssh-keygen -P "" -f ~/.ssh/id_rsa
fi

for i in {15..30}
do
  {
  ip=192.168.1."$i"
  ping -c 1 -W1 "$ip"
  if [ $? -eq 0 ];then
     echo "$ip" >> ip_ok.txt
     /usr/bin/expect <<-EOF
     spawn ssh-copy-id $user@$ip
     expect {
        "yes/no" { send "yes\r"; exp_continue }
        "password:" { send "$passwd\r" };
     }
     expect eof
        EOF
  else
    echo "$ip" >>ip_false.txt
  fi
  }&
done
wait
echo "finish"
```

#### 9. 检测 MySQL 主从复制是否异常

```shell
#!/bin/bash
user="root"
password="123456"
mycmd="mysql -u$user -p$password -h 192.168.1.88"

function chkdb() {
list=($($mycmd -e "show slave status \G"|egrep "Running|Behind"|awk -F: '{print $2}'))
if [ ${list[0]} = "Yes" -a ${list[1]} = "Yes" -a ${list[2]} -lt 120 ]
then echo "Mysql slave is ok"
else echo "Mysql slave replation is filed"
fi
}

function main() {
while true
do chkdb
   sleep 3
done
}
main
```

#### 11. MySQL 数据库备份脚本 (mysqldump)

```shell
#!/bin/bash
#删除15天以前备份

source /etc/profile           #加载系统环境变量
source ~/.bash_profile    #加载用户环境变量
set -o nounset             #引用未初始化变量时退出
#set -o errexit             #执行shell命令遇到错误时退出

user="root"
password="123456"
host="localhost"
port="3306"
#需备份的数据库，数组
db=("test")
#备份时加锁方式，
#MyISAM为锁表--lock-all-tables，
#InnoDB为锁行--single-transaction
lock="--single-transaction"
mysql_path="/usr/local/mysql"
backup_path="${mysql_path}/backup"
date=$(date +%Y-%m-%d_%H-%M-%S)
day=15
backup_log="${mysql_path}/backup.log"

#建立备份目录
if [ ! -e $backup_path ];then
    mkdir -p $backup_path
fi

#删除以前备份
find $backup_path -type f -mtime +$day -exec rm -rf {} \; > /dev/null 2>&1

echo "开始备份数据库：${db[*]}"

#备份并压缩
backup_sql(){
    dbname=$1
    backup_name="${dbname}_${date}.sql"
    #-R备份存储过程，函数，触发器
    mysqldump -h $host -P $port -u $user -p$password $lock --default-character-set=utf8 --flush-logs -R $dbname > $backup_path/$backup_name    
    if [[ $? == 0 ]];then
        cd $backup_path
        tar zcpvf $backup_name.tar.gz $backup_name
        size=$(du $backup_name.tar.gz -sh | awk '{print $1}')
        rm -rf $backup_name
        echo "$date 备份 $dbname($size) 成功 "
    else
        cd $backup_path
        rm -rf $backup_name
        echo "$date 备份 $dbname 失败 "
    fi
}

#循环备份
length=${#db[@]}
for (( i = 0; i < $length; i++ )); do
        backup_sql ${db[$i]} >> $backup_log 2>&1
done

echo "备份结束,结果查看 $backup_log"
du $backup_path/*$date* -sh | awk '{print "文件:" $2 ",大小:" $1}'
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shell_scriptrecord/  

