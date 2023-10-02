# shell刷题


> grep、awk、sed ， 以及 cat、head、tail、less、more 命令

#### 1. 统计文件行数

```shell
#解法1:awk
awk '{print NR}'| tail -n1
#awk print NR 打印每一行的行号
#tail -n1 只打印最后一行
 
#解法2:awk
awk 'END{print NR}'
#awk END{print NR} 最后打印行号
#关于 awk 脚本，我们需要注意两个关键词 BEGIN 和 END。
#BEGIN{ 这里面放的是执行前的语句}
#{这里面放的是处理每一行时要执行的语句}
#END {这里面放的是处理完所有的行后要执行的语句 }
 
#解法3:wc
wc -l
# wc -l 统计文件有多少行
# wc -wcl  -w word多少单词 -c char 多少字符 -l line 多少行
 
#解法4:sed
sed -n '$='
# -n或--quiet或--silent 仅显示script处理后的结果。
# 正则表达式中   行尾定位符“$” 行首定位符“^”
 
#解法5:grep
grep -n "" |tail -n1| awk -F: '{print $1}'
# grep -n ""  -n显示行号, 匹配"" 即显示每一行的行号
# awk -F:    只取第一列,即行号,以:分割
# tail -n1   只取最后一行,即总行数
 
#解法6:cat
cat -n |tail -1|awk '{print $1}'
# cat -n   显示每一行,并添加上行号
# tail -1  只取最后一行,即总行数
# awk -F   只取第一列,即行号,默认空格分隔
```

#### 2. 打印文件最后5行

```shell
#解法1:
tail -5 
#tail 从后往前看文件, -5 只看5行
# -f 循环读取,实时读取文件  -n 读多少行 只用-n n可以省略
```

#### 3. 输出7的倍数

```shell
for i in {0..500..7}
do
	echo $i
done

seq 0 7 500  # 或：seq [选项]... 首数 增量 尾数
```

#### 4. 输出第5行内容

```shell
awk 'NR==5'

cat -n |grep 5|awk '{print $2}'  #cat -n 显示每一行,并添加上行号

head -n 5 | tail -n 1  #先显示前5行,再从后往前显示一行
```

#### 5. 打印空行的行号

```shell
awk '/^$/ {print NR}'

grep -n "^$" |awk -F : '{print $1}'

#删除空行
awk '!/^$/ {print $NF}'
sed '/^$/d'
```

#### 6. 打印字母小于8

```shell
awk 'BEGIN{FS="";RS=" ";ORS="\n"}{if(NF<8)print$0}'
```

#### 7. 统计所有进程占用内存大小的和

```shell
awk '{sum+=$6} END{print sum}'
```

#### 8. 统计每个单词出现的个数

```shell
awk '{for(i=1;i<=NF;i++)a[$i]++} END{for (i in a){print i,a[i]}}'

cat $1  | tr -s ' ' '\n' |sort |uniq -c|sort | awk '{print $2" "$1}'
# cat 查看全部
#tr -s  将' '转化为 换行
#sort 排序
#uniq -c 统计行出现次数(需要先排序)
#sort 再排序,按升序显示
#awk 改变显示方式
```

#### 9. 第二列是否有重复

```shell
awk '{print $2}' nowcoder.txt|sort|uniq -cd|sort -n
# awk 先筛选出第二列
# sort 排序
# uniq -cd    -c统计每行行数 -d只显示重复行
# 再次排序按升序显示信息
```

#### 10. 转置文件的内容

```shell
#解法1:
awk '{printf $1}' nowcoder.txt
awk '{printf $2}' nowcoder.txt
#先输出第一列,再输出第二列

#解法2:
awk '{
    for(i=1;i<=NF;i++){rows[i]=rows[i]" "$i}  # 这种数组的赋值学会使用
} END{
    for(line in rows){print rows[line]}
}' 
#第一个式子将每一列存到了一个行数组里
#第二个式子  END最终,输出行数组
```

#### 11. 去掉所有包含 this 的句子

```shell
grep -v "this"

sed '/this/d'
```

#### 12. 求平均值

```shell
#解法1:
awk 'BEGIN{sum=0}NR>1{sum+=$0}END{printf("%.3f",sum/(NR-1))}'
# 将第一行读为长度,将其他行作为数值求和
# 最后将和/长度,格式化小数点后三位输出

#解法2:(同1)
awk '{if(NR!=1)total+=$1} END{printf("%.3f\n",total/(NR-1))}'
# 除了第一行不读,将其他行作为数值求和
# 最后将  和/长度 (即:总行数-1),格式化小数点后三位输出
```

#### 13. 去掉不需要的单词

```shell
#解法1: grep
grep -iv "b"
#grep -i ignore 忽略大小写 -v invert 不匹配
#即显示不含B和b的

#解法2: sed
sed '/B\|b/d'
#等同于 
# sed '/b/d' | sed '/B/d' 
# d delete 删除,将包含b或B的行,删除

#解法3: awk
awk '$0!~/[bB]/ {print $0}'
# $0表示整行
# 不包含b或B,输出整行
```

#### 14. **打印每一行出现的数字个数**

```shell
awk -F "" '{count=0;for(i=1;i<=NF;i++) {if($i>=1&&$i<=5) count++}sum+=count; print "line"NR" number:"count}END{print "sum is "sum}' 
```

#### 15. 判断输入的是否为 IP 地址

```shell
read -p "请输入IP地址:"  IP
IP_CHECK=$(echo $IP|awk -F. '$1<255&&$2<255&&$3<255&&$4<255{print "R"}')
if echo $IP|grep -E "^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$" >/dev/null; then
	if [ $IP_CHECK == "R" ]; then
		echo "IP $IP  合法!"
	else
		echo "IP $IP 不合法!"
	fi
else
	echo "IP输错啦!"
fi
```

```shell
awk -F "." '{
    if (NF == 4) {
        for (i=1; i<5; i++) {
            if ($i > 255 || $i < 0) {
                print("no");break
            }
        }
        if (i==5){print("yes")}
    } else {
        print("error")
    }
}'
```

#### 16. **将字段逆序输出文件的每行**

```shell
awk -F ":" '{
    for (i=1; i<=NF; i++) {
        if (i==1) {
            str = $1; continue
        }
        str = sprintf("%s:%s", $i, str)
    }
    print(str)
}'
```

```shell
awk -F: '{printf("%s:%s:%s:%s:%s:%s:%s\n",$7,$6,$5,$4,$3,$2,$1)}' nowcoder.txt
```

#### 17. **域名进行计数排序处理**

```shell
awk -F '/' '{print $3}'|sort|uniq -c|sort -r|awk '{print $1" "$2}'
```

```shell
awk -F '/' '{
    countMap[$3]++;
}END{
    for(domin in countMap){
        print countMap[domin] " " domin
    }
}' nowcoder.txt | sort -nr -k1  # r: 降序；n: 以数值来排序；k: 指定按照某一列进行排序
```

#### 18.**打印只有一个数字的行**

```shell
awk -F '' '{
    count=0
    for(i=1;i<=NR;i++){
        if($i>=0 && $i<=9){
            count++
        }
    }
    if(count==1){
        print $0
    }
}'
```

#### 19. 格式化输出

```shell
awk -F "" '{
    k=0
    for (i=NF; i>0; i--) {
        k++
        str = sprintf("%s%s", $i, str)
        if (k%3 == 0 && i>=2 && NF > 3) {
            str = sprintf(",%s", str)
        }
    }
    print(str)
    str=""
}'
```

#### 20. [处理文本](https://www.nowcoder.com/practice/908d030e676a4fac997a127bfe63da64?tpId=195&tqId=39431&rp=1&ru=/ta/shell&qru=/ta/shell&difficulty=&judgeStatus=&tags=/question-ranking)

```shell
awk -F ":" '{
        a[$1] = a[$1] $2 "\n"   # 学会这种数组表示法，类似key-value
    } 
    END {for (i in a){
        printf("[%s]\n%s",i,a[i])
        }
    }' nowcoder.txt
```

#### 21. **nginx 日志分析 1-IP 统计**

- 统计出 2020 年 4 月 23 号的访问 ip 次数，并且按照次数降序排序

```shell
awk '{
    if(substr($4, 2, 11) == "23/Apr/2020") {  # substr 中字符 从1开始计数
        res[$1]++;
    }
}END {
    for(k in res) {
        print res[k] " " k
    }
}' | sort -nr -k 1 -t " "
```

#### 22.**统计某个时间段的 IP**

-  2020 年 04 月 23 日 20-23 点的去重 IP 访问量

```shell
awk '{
    if ($0 ~ /\[23\/Apr\/2020:2[0-2]/) {  #使用正则表达式方式
         a[$1]=1
     }
} END {
     print (length(a))   #length 函数使用
 }'
```

#### 23. 统计访问3次以上IP

```shell
 awk '{
        if ($1 in a) {
            a[$1]++;
        } else {
            a[$1]=1
        }
    } END {
        for (j in a) {
            if (a[j] > 3) {
                print a[j],j
            }
        }
    }' nowcoder.txt | sort -r
```

#### 24. **查询某个 IP 的详细访问情况**

```shell
awk '{
    if ($1 == "192.168.1.22") {
        a[$7]++
    }
} END {
    for (i in a){
        printf("%d %s\n",a[i], i)
    }
}' | sort -r
```

#### 25. **统计每分钟的请求数**

```shell
awk -F ":" '{
    res[$2 ":" $3]++;
}END{
    for(k in res){
        print res[k] " " k
    }
}'|sort -nr -k1
```

#### 26.**查看各个状态的连接数**

```shell
awk '{
    if ($1 == "tcp") {
        arr[$6]++
    }
}END{
    for(i in arr){
        print i" "arr[i]
    }
}' | sort -nr -k2
```

#### 27. **查看和 3306 端口建立的连接**

```shell
grep "3306" | grep  -i "established" | awk '{print $5}' | awk -F ":" '{print $1}' | sort | uniq -c | awk '{print $1" "$2}' | sort -nr
```

```shell
grep "3306" | grep "ESTABLISHED" | awk '{
    a[$5]++
}END{
    for(i in a){
        print a[i]" "i
    }
}' | sort -nr -k1 | awk -F ":" '{
    print $1
}'
```

#### 28.**输出每个 IP 的连接数**

```shell
awk '/^tcp/{print $5}' | awk -F ":" '{
    a[$1]++
}END{
    for(i in a){
        print i" "a[i]
    }
}' | sort -nr -k2
```

#### Resource

- https://www.nowcoder.com/ta/shell
- https://blog.nowcoder.net/n/1fde90cce80546b5b6b5d122cdf8be84



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shell%E5%88%B7%E9%A2%98/  

