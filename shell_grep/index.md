# shell_grep


>grep 命令是一种强大的文本搜索工具，它能使用正则表达式，按照指定的模式去匹配，并把匹配的行打印出来。需要注意的是，`grep 只支持匹配而不能替换匹配的内容`，替换的功能可以由 sed 来完成。

#### 1. 使用模式

```shell
grep [options] pattern [file...]
```

[options] 表示选项，具体的命令选项见下表。pattern 表示要匹配的模式（包括目标字符串、变量或者正则表达式），file 表示要查询的文件名，可以是一个或者多个。pattern 后面所有的字符串参数都会被理解为文件名。

| 选项   | 说明                                                         |
| ------ | :----------------------------------------------------------- |
| -color | 以颜色突出显示匹配到的字符串                                 |
| -c     | `只打印匹配的文本行的行数`，不显示匹配的内容                 |
| -i     | 匹配时`忽略字母的大小写`                                     |
| -h     | 当搜索多个文件时，`不显示匹配文件名前缀`                     |
| -n     | 列出所有的`匹配的文本行，并显示行号`                         |
| -l     | ` 只列出含有匹配的文本行的文件的文件名`，而不显示具体的匹配内容 |
| -s     | 不显示关于不存在或者无法读取文件的错误信息                   |
| -v     | 只显示不匹配的文本行                                         |
| -w     | ` 匹配整个单词`                                              |
| -x     | `匹配整个文本行`                                             |
| -r     | ` 递归搜索，搜索当前目录和子目录`                            |
| -q     | 禁止输出任何匹配结果，而是以退出码的形式表示搜索是否成功，其中 0 表示找到了匹配的文本行 |
| -b     | 打印匹配的文本行到文件头的偏移量，以字节为单位               |
| -E     | 支持`扩展正则表达式`                                         |
| -P     | 支持` Perl 正则表达式`                                       |
| -F     | 不支持正则表达式，将模式按照字面意思匹配                     |
| -o     | `仅显示匹配到的字符串`                                       |

#### 2. 使用案例

```shell
grep -n "syslog" g.txt  #把包含 syslog 的行过滤出来
grep "^ntp" g.txt  #把以 ntp 开头的行过虑出来
grep -A2 "syslog" g.txt       #把匹配 ntp 的行以及下边的两行过滤出来
grep -B1 "syslog" g.txt       #包含 syslog 及上边的一行过滤出来
grep -C1 "syslog" g.txt       #把包含 syslog 以及上、下一行内容过滤出来
grep -e "root" -e "syslog" g.txt # 过滤包含 root 或 syslog 的行
grep -E "root|syslog" g.txt      # 过滤包含 root 或 syslog 的行
grep -r "font" .                 #查看当前目录中包含某关键词的所有文件
grep -rnw --exclude-dir={.git,svn} "font" .                 #查看当前目录中包含某关键词的所有文件,排除某些目录
```

#### Resource

- https://www.jianshu.com/p/652b4975b242
- https://www.cnblogs.com/mq0036/p/14541876.html
- https://www.jianshu.com/p/652b4975b242

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/shell_grep/  

