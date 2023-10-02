# LaTex 的基本介绍


- from https://github.com/qinnian/LaTeX

## 一、TEX

TEX 是高德纳 (Donald E.Knuth) 开发的、以排版文字和数学公式为目的的一个计算机软件。高德纳从 1977 年开始开发 TEX ，以发掘当时开始用于出版工业的数字印刷设备的潜力。正在编写著作《计算机程序设计艺术》的高德纳，意图扭转排版质量每况愈下的状况，以免影响他的出书。我们现在使用的 TEX 排版引擎发布于 1982 年，在 1989 年又稍加改进以更好地支持 8-bit 字符和多语言排版。

TEX 以其卓越的稳定性、跨平台、几乎没有 Bug 而著称。TEX 的版本号不断趋近于 π，当前为 3.141592653。 TEX 读作 “Tech” ，其中 “ch” 的发音类似于 “h” ，与汉字“泰赫”的发音类似。TEX 的拼写来自希腊词语 τεχνική (technique，技术) 的开头几个字母。在 ASCII 字符环境，TEX 写作 TeX。

## 二、LATEX

LATEX 为 TEX 基础上的一套格式，令作者能够使用预定义的专业格式以较高质量排版和印刷他们的作品。LATEX 的最初开发者为 Leslie Lamport 博士 。LATEX 使用 TEX 程序作为自己的排版引擎。当下 LATEX 主要的维护者为 Frank Mittelbach。 

LATEX 读作 “Lah-tech” 或者 “Lay-tech” ，近似于汉字“拉泰赫”或“雷泰赫”。LATEX 在 ASCII 字符环境写作 LaTeX。当前的 LATEX 版本为 LATEX 2ε，意思是超出了第二版，接近但没达到第三版，在 ASCII 字符环境写作 LaTeX2e。

## 三、LATEX 的优缺点

LATEX 总会拿来和一些“所见即所得”（What You See Is What You Get）的文字处理和排版工具比较优缺点。我认为这种比较并不值得提倡，毕竟所有工具都有自己值得使用的原因。

 LATEX 的优点：

- 专业的排版输出，产生的文档看上去就像“印刷品”一样。

- 方便而强大的数学公式排版能力。
  
- 绝大多数时候，用户`只需专注于一些组织文档结构的基础命令，无需（或很少）操心文档的版面设计`。

- `很容易生成复杂的专业排版元素，如脚注、交叉引用、参考文献、目录等`。

- 强大的扩展性。世界各地的人开发了`数以千计的 LATEX 宏包用于补充和扩展 LATEX 的功能`。
  
- LATEX 依赖的 TEX 排版引擎和其它软件是跨平台、免费、开源的。无论用户使用的是 Windows，macOS（OS X），GNU/Linux 还是 FreeBSD 等操作系统，都能轻松获得和使用这一强大的排版工具。

LATEX 的缺点：

- 入门门槛高。

- 排查错误困难。LATEX 作为一个依靠编写代码工作的排版工具，其使用的宏语言比 C++ 或 Python 等程序设计语言在错误排查方面困难得多。它虽然能够提示错误，但不提供调试的机制，有时错误提示还很难理解。

- 样式定制困难。LATEX 提供了一个基本上良好的样式，为了让用户不去关注样式而专注于
文档结构。但如果想要改进 LATEX 生成的文档样式则是十分困难。

## 四、推荐网站

推荐给⼤家3个下载LaTeX模版的优秀⽹站，在这⾥你可以找到各种类型的精美LaTeX模版。

### 1.LaTeX Templates

网址：http://www.latextemplates.com/

网站分类得⾮常详细，⽐如`学术期刊、书籍、⽇历、会议poster、 简历、实验报告、Presentations等等`，涵盖了各种⽇常需求。

![](https://raw.githubusercontent.com/qinnian/FigureBed/master/20200227125232.png)

### 2. Overleaf

网址：https://www.overleaf.com/

Overleaf是⼀个⾮常有名的`在线LaTeX编辑`、协作平台有了Overleaf，你⽆需在电脑上安装TeX环境和 TeX编辑器，在⽹站上即可以完成TeX代码的撰写、编译和PDF导出。`⽹站的 Templates 页面拥有丰富的LaTeX模版`可供选择。

![](https://raw.githubusercontent.com/qinnian/FigureBed/master/20200227125951.png)

##### .1. 快捷键

| 快捷键       | 对应操作                                                     |
| ------------ | ------------------------------------------------------------ |
| Ctrl + F     | 查找和替换。（类似words里的同名功能）                        |
| Ctrl + Enter | 编译。（可以理解为刷新右边的pdf预览）                        |
| Ctrl + Z     | 撤销。（后退）                                               |
| Ctrl + Y     | 重做。（前进，或者理解为撤销撤销操作）                       |
| Ctrl + Home  | 跳跃到Latex文档第一行。                                      |
| Ctrl + End   | 跳跃到Latex文档最后一行。                                    |
| Ctrl + L     | 跳转到特定的一行。（会弹出一个对话框让你搜索）               |
| `Ctrl + /`   | 将所选文本全部注释。（如果所选的文本已注释，则会变成非注释状态。） |
| Ctrl + D     | 删除当前这一行。                                             |
| Ctrl + A     | 全选。                                                       |
| Tab          | 往后缩进四格。                                               |
| Shift + Tab  | 往前缩进四格。                                               |
| Ctrl + U     | 将所选文本变成大写字母。                                     |
| Ctrl + Space | 将所选文本变成小写字母。                                     |
| Ctrl + B     | 将所选文本加粗。（\textbf{}）                                |
| Ctrl + I     | 将所选文本变成斜体。（\textit{}）                            |
| Ctrl + Space | 使用cite{}时，在refernce中搜索。                             |

### 3. LaTeX⼯作室

网址：https://www.latexstudio.net/

LaTeX⼯作室，国内最⼤最好⽤的LaTeX平台，它拥有`⼤量的中⽂ LaTeX模版`，以及各⼤⾼校的学位论⽂模版。

![](https://raw.githubusercontent.com/qinnian/FigureBed/master/20200227130245.png)





---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/01-latex-%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BB%8B%E7%BB%8D/  

