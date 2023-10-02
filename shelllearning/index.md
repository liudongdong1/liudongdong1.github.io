# shellLearning


#### 1. 基本语法

##### .1. 基本脚本

```shell
#!/bin/bash
echo "Hello World !"
# 注释内容
chmod +x ./test.sh  #使脚本具有执行权限
./test.sh  #执行脚本
```

##### .2. 定义使用变量

```shell
your_name="qinjx"   # 定义变量
for file in `ls /etc`  # 定义变量2

echo $your_name
echo ${your_name}   #加括号为了方便识别

vairable=$(command)  #命令的执行结果赋值给变量

readonly your_name  #定义变量只读
unset your_name  # 删除变量
```

##### .3. 单引号和双引号区别

- 以单引号`' '` 包围变量的值时，`单引号里面是什么就输出什么，即使内容中有变量和命令`（命令需要反引起来）也会把它们原样输出。这种方式比较适合定义显示纯字符串的情况，即不希望解析变量、命令等的场景。
- 以双引号 `" "` 包围变量的值时，`输出时会先解析里面的变量和命令`，而不是把双引号中的变量名和命令原样输出。这种方式比较适合字符串中附带有变量和命令并且想将其解析后再输出的变量定义。

```shell
#!/bin/bash
url="http://c.xinbaoku.com"
website1='新宝库：${url}'
website2="新宝库：${url}"
echo $website1  #新宝库：${url}
echo $website2  #新宝库：http://c.xinbaoku.com
```

##### .4. 字符串操作

- 由单引号`' '` 包围的字符串：
  - 任何字符都会原样输出，在其中使用变量是无效的。
  - 字符串中不能出现单引号，即使对单引号进行转义也不行。

-  由双引号 `" "` 包围的字符串：

  - 如果其中包含了某个变量，那么该变量会被解析（得到该变量的值），而不是原样输出。

  - 字符串中可以出现双引号，只要它被转义了就行。

- 不被引号包围的字符串

  - 不被引号包围的字符串中出现变量时也会被解析，这一点和双引号 `" "` 包围的字符串一样。

  - 字符串中不能出现空格，否则`空格后边的字符串会作为其他变量或者命令解析`。

```shell
your_name="runoob"
greeting_1="hello, ${your_name} !"  #字符串拼接
echo ${#greeting_1} #输出 字符串长度
```

- 字符串截取 ${string: start :length}

```shell
url="c.xinbaoku.com"
echo ${url: 2: 9}   #biancheng
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403141030515.png)

##### .5. 数组操作

```shell
#!/bin/bash

nums=(29 100 13 8 91 44)
echo ${nums[@]}  #输出所有数组元素
nums[10]=66  #给第10个元素赋值（此时会增加数组长度）
echo ${nums[*]}  #输出所有数组元素
echo ${nums[4]}  #输出第4个元素

echo ${#nums[*]}   #获取数组长度
#向数组中添加元素
nums[10]="http://c.xinbaoku.com/shell/"
echo ${#nums[@]}
echo ${#nums[10]}
#删除数组元素
unset nums[1]
echo ${#nums[*]}
```

- 数组拼接

```shell
#!/bin/bash

array1=(23 56)
array2=(99 "http://c.xinbaoku.com/shell/")
array_new=(${array1[@]} ${array2[*]})   #数组拼接

echo ${array_new[@]}  #也可以写作 ${array_new[*]}
```

##### .6. 函数传参

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403140308952.png)

```shell
funWithParam(){
    echo "第一个参数为 $1 !"
    echo "第二个参数为 $2 !"
    echo "第十个参数为 $10 !"
    echo "第十个参数为 ${10} !"
    echo "第十一个参数为 ${11} !"
    echo "参数总数有 $# 个!"
    echo "作为一个字符串输出所有参数 $* !"
}
funWithParam 1 2 3 4 5 6 7 8 9 34 73
```

```shell
#!/bin/bash
#得到两个数相加的和
function add(){
    return `expr $1 + $2`
}
add 23 50  #调用函数
echo $?  #获取函数返回值
```

##### .7. 关系运算符

| 运算符 | 说明                                                  | 举例                       |
| :----- | :---------------------------------------------------- | :------------------------- |
| -eq    | 检测两个数是否相等，相等返回 true。                   | [ $a -eq $b ] 返回 false。 |
| -ne    | 检测两个数是否不相等，不相等返回 true。               | [ $a -ne $b ] 返回 true。  |
| -gt    | 检测左边的数是否大于右边的，如果是，则返回 true。     | [ $a -gt $b ] 返回 false。 |
| -lt    | 检测左边的数是否小于右边的，如果是，则返回 true。     | [ $a -lt $b ] 返回 true。  |
| -ge    | 检测左边的数是否大于等于右边的，如果是，则返回 true。 | [ $a -ge $b ] 返回 false。 |
| -le    | 检测左边的数是否小于等于右边的，如果是，则返回 true。 | [ $a -le $b ] 返回 true。  |

| !    | 非运算，表达式为 true 则返回 false，否则返回 true。 | [ ! false ] 返回 true。                  |
| ---- | --------------------------------------------------- | ---------------------------------------- |
| -o   | 或运算，有一个表达式为 true 则返回 true。           | [ $a -lt 20 -o $b -gt 100 ] 返回 true。  |
| -a   | 与运算，两个表达式都为 true 才返回 true。           | [ $a -lt 20 -a $b -gt 100 ] 返回 false。 |

| 运算符 | 说明       | 举例                                       |
| :----- | :--------- | :----------------------------------------- |
| &&     | 逻辑的 AND | [[ \$a -lt 100 && $b -gt 100 ]] 返回 false |
| \|\|   | 逻辑的 OR  | [[ $a -lt 100 || $b -gt 100 ]] 返回 true   |

| =    | 检测两个字符串是否相等，相等返回 true。      | [ $a = $b ] 返回 false。 |
| ---- | -------------------------------------------- | ------------------------ |
| !=   | 检测两个字符串是否不相等，不相等返回 true。  | [ $a != $b ] 返回 true。 |
| -z   | 检测字符串长度是否为0，为0返回 true。        | [ -z $a ] 返回 false。   |
| -n   | 检测字符串长度是否不为 0，不为 0 返回 true。 | [ -n "$a" ] 返回 true。  |
| $    | 检测字符串是否为空，不为空返回 true。        | [ $a ] 返回 true。       |||

| 操作符  | 说明                                                         | 举例                      |
| :------ | :----------------------------------------------------------- | :------------------------ |
| -b file | 检测文件是否是`块设备文件`，如果是，则返回 true。            | [ -b $file ] 返回 false。 |
| -c file | 检测文件是否是`字符设备文件`，如果是，则返回 true。          | [ -c $file ] 返回 false。 |
| -d file | 检测文件是`否是目录`，如果是，则返回 true。                  | [ -d $file ] 返回 false。 |
| -f file | 检测文件是否是`普通文件`（既不是目录，也不是设备文件），如果是，则返回 true。 | [ -f $file ] 返回 true。  |
| -g file | 检测文件是否设置了 SGID 位，如果是，则返回 true。            | [ -g $file ] 返回 false。 |
| -k file | 检测文件是否设置了粘着位(Sticky Bit)，如果是，则返回 true。  | [ -k $file ] 返回 false。 |
| -p file | 检测文件是否是有名管道，如果是，则返回 true。                | [ -p $file ] 返回 false。 |
| -u file | 检测文件是否设置了 SUID 位，如果是，则返回 true。            | [ -u $file ] 返回 false。 |
| -r file | 检测文件`是否可读`，如果是，则返回 true。                    | [ -r $file ] 返回 true。  |
| -w file | 检测文件`是否可写`，如果是，则返回 true。                    | [ -w $file ] 返回 true。  |
| -x file | 检测文件`是否可执行`，如果是，则返回 true。                  | [ -x $file ] 返回 true。  |
| -s file | 检测文件是否为空（文件大小是否大于0），不为空返回 true。     | [ -s $file ] 返回 true。  |
| -e file | 检测文件（包括目录）`是否存在`，如果是，则返回 true。        | [ -e $file ] 返回 true。  |

- 使用test命令时候：将变量用双引号 `""` 包围起来，这样能避免变量为空值时导致的很多奇葩问题。

```shell
#!/bin/bash

read age    # test 和 []  用法

if test $age -le 2; then
    echo "婴儿"
elif test $age -ge 3 && test $age -le 8; then
    echo "幼儿"
elif [ $age -ge 9 ] && [ $age -le 17 ]; then
    echo "少年"
elif [ $age -ge 18 ] && [ $age -le 25 ]; then
    echo "成年"
elif test $age -ge 26 && test $age -le 40; then
    echo "青年"
elif test $age -ge 41 && [ $age -le 60 ]; then
    echo "中年"
else
    echo "老年"
fi
```

##### .7. 数学计算命令

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403142046933.png)

```shell
echo $((++a))  #如果++在a的前面，输出整个表达式时，先进行自增或自减计算，因为a为9，且要自增1，所以输出10。

a=10 b=35
let a+=6 c=a+b  #多个表达式以空格为分隔
echo let a+b  #错误，echo会把 let a+b作为一个字符串输出
```

##### .8. 流程控制

```shell
# if-else 结构 
if test $[num1] -eq $[num2]
then
    echo '两个数字相等!'
else
    echo '两个数字不相等!'
fi
# for-循环结构
for loop in 1 2 3 4 5
do
    echo "The value is: $loop"
done
#while 循环结构
while(( $int<=5 ))
do
    echo $int
    let "int++"
done
# util 循环结构
until [ ! $a -lt 10 ]
do
   echo $a
   a=`expr $a + 1`
done
#case 结构
echo '你输入的数字为:'
read aNum
case $aNum in
    1)  echo '你选择了 1'
    ;;
    2)  echo '你选择了 2'
    ;;
    3)  echo '你选择了 3'
    ;;
    4)  echo '你选择了 4'
    ;;
    *)  echo '你没有输入 1 到 4 之间的数字'
    ;;
esac

#break-continue 结构
while :
do
    echo -n "输入 1 到 5 之间的数字: "
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的!"
            continue
            echo "游戏结束"
        ;;
    esac
done
```

##### .9. 重定向

| 命令            | 说明                              |
| :-------------- | :-------------------------------- |
| command > file  | 将输出重定向到 file。             |
| command < file  | 将输入重定向到 file。             |
| command >> file | 将输出以追加的方式重定向到 file。 |

> command > /dev/null  ； 使用这个可以减少不看程序输出。

#### 2. 常用片段

##### .1. 时间相关

```shell
today=`date +%y%m%d`
ls /usr/bin -al > log.$today
```

##### .2. 输入相关

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220403141725747.png)

```shell
#超时计数输入
if read -t 5 -p "Please enter your name:" name
then
	echo "Hello, $name, welcome to my script"
else
	#起到换行的作用
	echo
	#输入计数 -n1
	read -n1 -p "Do you want to continue [Y/N]?" answer
	case $answer in
	Y | y) echo
		   echo "Fine, continue on...";;
	N | n) echo
		   echo "OK，goodbye";;
	*) echo
	   echo "OK, wrong, goodbye"
	esac
	echo "Sorry, this is the end of the script"
fi
```

```shell
#从文件中读取数据
count=1
cat test | while read line
do
	echo "Line $count: $line"
	count=$[ $count + 1 ]
done
echo "Finished processing the file"
```

##### .3. 进程相关

- **ps**

> ps -a - 列出所有运行中/激活进程  
> ps -ef |grep - 列出需要进程  
> ps -aux - 显示进程信息，包括无终端的（x）和针对用户（u）的进程：如USER, PID, %CPU, %MEM等  

- pstree   

> linux中，每一个进程都是由其父进程创建的。pstree以可视化方式显示进程，通过显示进程的树状图来展示进程间关系。  
> 如果指定了pid了，那么树的根是该pid，不然将会是init（pid： 1）。  

- top  

> top 是一个更加有用的命令，可以监视系统中不同的进程所使用的资源。它提供实时的系统状态信息。  
> 显示进程的数据包括 PID、进程属主、优先级、%CPU、%memory等。可以使用这些显示指示出资源使用量。  

- htop  

> htop与top很类似，但是htop是交互式的文本模式的进程查看器。  
> 它通过文字图形化地显示每一个进程的CPU和内存使用量、swap使用量。  
> 使用上下光标键选择进程，F7和F8改变优先级，F9杀死进程。Htop不是系统默认安装的，所以需要额外安装  

- nice 

> 通过nice命令的帮助，用户可以设置和改变进程的优先级。提高一个进程的优先级，内核会分配更多CPU时间片给这个进程。  
> 默认情况下，进程以0的优先级启动。进程优先级可以通过top命令显示的NI（nice value）列查看。  
> 进程优先级值的范围从-20到19。值越低，优先级越高。  
> nice <优先值> <进程名> - 通过给定的优先值启动一个程序  

- renice 

> renice命令类似nice命令。使用这个命令可以改变正在运行的进程优先值。  
> 注意，用户只能改变属于他们自己的进程的优先值。  
> renice -n -p - 改变指定进程的优先值  
> renice -u -g - 通过指定用户和组来改变进程优先值  

- kill  

> 这个命令用于发送信号来结束进程。如果一个进程没有响应杀死命令，这也许就需要强制杀死，使用-9参数来执行。  
> 注意，使用强制杀死的时候一定要小心，因为进程没有时机清理现场，也许写入文件没有完成。  
> 如果我们不知道进程PID或者打算用名字杀死进程时候，killall就能派上用场。  
> kill <pid>  
> kill -9 <pid>  
> killall -9 - 杀死所有拥有同样名字的进程  
> 如果你使用kill，你需要知道进程ID号。pkill是类似的命令，但使用模式匹配，如进程名，进程拥有者等。  
> pkill <进程名> 
>
> ```shell
> ps -ef | grep java   #查看所有关于java的进程
> kill -9 XXXXX   #线程将某线程终止时用
> ```

##### .4. 查寻

1. grep最简单的用法，匹配一个词：`grep word filename`


2. 能够从多个文件里匹配：`grep word filename1 filenam2 filename3`

3. 能够使用正則表達式匹配：`grep -E pattern f1 f2 f3...`

4. 能够使用-o仅仅打印匹配的字符，例如以下所看到的：

```shell
echo this is a line. | grep -E -o "[a-z]*\."
```

5. 打印除匹配行之外的其它行，使用-v

```shell
echo -e "1\n2\n3\n4" | grep -v -E "[1-2]"
```

- 启动指定进程

```shell
#!/bin/sh
ps -fe|grep processString |grep -v grep   #管道命令
if [ $? -ne 0 ]
then
echo "start process....."
else
echo "runing....."
fi
```

- 获取进程ID

```shell
#!/bin/bash
if [ $# -eq 1 ]
then
PROC_NAME="$1"
echo "server name:"${PROC_NAME}
num=`ps -ef | grep "${PROC_NAME}" | grep -v "grep" | grep -v "bash" | wc -l `
pidvar=`ps x | grep "${PROC_NAME}" | grep -v "grep" | grep -v "bash" | awk '{print $1}'`
echo $num
if [ $num -gt 0 ]
then
echo $pidvar
fi
else
echo "语法错误，正确语法如下:"
echo "getpidparam.sh process_name"
fi
```

- 指定进程运行状态

```shell
#!/bin/sh
echo "`date`"
echo "Start $0---------"
echo ""
#每十秒监视一下
sec=10
#取得指定进程名为mainAPP，内存的使用率，进程运行状态，进程名称
eval $(ps | grep "mainApp" | grep -v grep | awk {'printf("memInfo=%s;myStatus=%s;pName=%s",$3,$4,$5)'})
echo $pName $myStatus $memInfo
testPrg=""
while [ -n "$pName" -a "$myStatus" != "Z" ]
do
    echo "----------`date`---------------------"
    echo $pName $myStatus $memInfo
    sleep $sec
    ####You must initialize them again!!!!!
    pName=""
    myStatus=""
    memInfo=""
    eval $(ps | grep "mainApp" | grep -v grep | awk {'printf("memInfo=%s;myStatus=%s;pName=%s",$3,$4,$5)'})
    testPrg=`ps | grep "MyTester" | grep -v grep | awk '{print $0}'`
    if [ -z "$testPrg" ]; then
        break
    fi
    ##注意一定要再次初始化为空
    testPrg=""
done
echo "End $0---($pName,$myStatus,$testPrg)-------------------"
if [ -z "$pName" ]; then
        ###发现测被测试程序异常退出后，停止测试程序
    killall MyTester
    echo "stop TestProgram MyTester"
fi
echo "`date`"
echo "---------------Current Status------------------"
ps | grep -E "mainApp|SubApp" | grep -v grep
echo ""
```

#### 3. 成绩管理系统

```shell
#!/bin/bash 
function information 
{ 
  echo "---------------------------"
  echo "图书馆管理系统（5.4版本）"
  echo 
  echo -n "| " ;echo "1:添加图书"
  echo -n "| " ;echo "2:删除图书"
  echo -n "| " ;echo "3:图书列表"
  echo -n "| " ;echo "4:查找图书"
  echo -n "| " ;echo "5|q:退出系统"
  echo 
  echo "---------------------------"
  read -p "请输入你的选择:" a 
    
  
  case "$a" in
  1) 
    add ;; 
  2) 
    delete ;; 
  3) 
    list ;; 
  4) 
    search;; 
  5|q|Q) 
    return -1 ;; 
  *) 
    information ;; 
  esac 
} 
  
  
function file_exist 
{ 
  if [ ! -f .book.txt ];then
    touch .book.txt 
  fi
} 
  
  
function add 
{ 
  read -p "请输入图书的编号：" number 
  read -p "请输入图书的书名：" book_name 
  read -p "请输入图书的作者：" author 
  read -p "请输入图书的价格：" price  
    echo -e "$number\t$book_name\t$author\t$price" >>.book.txt && { 
      echo "添加图书成功！"
      echo "-------------------"
    } 
  if [ $? -ne 0 ];then
    echo "添加图书失败"
  fi
  information 
  
} 
  
function delete 
{ 
  read -p "请输入要删除的图书的编号：" number 
  grep $number .book.txt &>/dev/null && { 
      sed -i '/\<'$number'\>/d' .book.txt &>/dev/null && 
      echo "删除图书成功" 
  echo "-------------------------"
  } 
    
  if [ $? -ne 0 ];then
    echo "删除图书失败"
    echo "你要删除的图书不存在"
  fi
  information 
} 
  
#列出所有图书的信息 
function list 
{ 
  echo -e "编号\t书名\t作者\t价格"
  cat .book.txt 
  echo "----------------------------"
  information 
    
} 
  
  
#下面的函数用到的查询菜单 
function search_menu 
{ 
  echo;echo "----------------------------" 
  echo -n "|";echo -e "1：\t按图书编号查询"
  echo -n "|";echo -e "2：\t按图书书名查询"
  echo -n "|";echo -e "3：\t按图书作者查询"
  echo -n "|";echo -e "4：\t按图书价格查询"
  echo -n "|";echo -e "5|q：\t退出查询系统"
  echo;echo "----------------------------" 
  
} 
function search 
{ 
  search_menu 
  read -p "请输出你的选择：" myselect 
  case "$myselect" in
  1) 
    read -p "请输入要查询的图书的编号：" mynumber 
    echo -e "编号\t书名\t作者\t价格\n"
    awk '$1=='$mynumber'{print $0}' .book.txt 2>/dev/null 
                
    if [ $? -ne 0 ];then
      echo "图书不存在"
    fi
    search 
    ;; 
  2) 
    read -p "请输入你要查询的书名：" mybook_name 
    echo -e "编号\t书名\t作者\t价格\n"
    awk '$2~/'$mybook_name'/{print $0}' .book.txt 2>/dev/null
    if [ $? -ne 0 ];then
      echo "图书不存在"
    fi
    search 
    ;; 
  3) 
    read -p "请输入图书的作者：" myauthor 
    echo -e "编号\t书名\t作者\t价格\n"
    awk '$3~/'$myauthor'/{;print $0}' .book.txt 2>/dev/null
    if [ $? -ne 0 ];then
      echo "图书不存在"
    fi
    search 
    ;; 
  4) 
    read -p "请输入图书的价格：" myprice 
    echo -e "编号\t书名\t作者\t价格\n"
    awk '$4=='$myprice'{print $0}' .book.txt 2>/dev/null
    if [ $? -ne 0 ];then
      echo "图书不存在"
    fi
    search 
    ;; 
  5) 
    information 
    ;; 
  *) 
    information 
    ;; 
  esac 
  
} 
 
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/shelllearning/  

