# LaTex 的图片插入


> LaTeX插入图片时，常用的图片格式有：png, pdf, jpg, eps。以上四种图片格式各有优劣，其中最为显著的差异是清晰度和图片文件大小。在清晰度方面：eps是清晰度最高的，其次是pdf和png，最后是jpg。
>
> - 图片命名中不要出现中文字符、不要空格和其他特殊符号，建议只用英文字母、下划线和简单符号。
> - 若图片格式不是以上四种，或者图片中空白边缘过多，可以用PS进行处理并转存为以上四种格式之一。
> - 注意需要裁剪图片中多余空白部分

### 0. 参数设置

> `htbp` 选项用来指定插图的理想位置，这几个字母分别代表 here, top, bottom, float page，也就是就这里、页顶、页尾、浮动页（专门放浮动体的单独页面或分栏）。`\centering` 用来使插图居中；`\caption` 命令设置插图标题，LaTeX 会自动给浮动体的标题加上编号。注意 `\label` 应该放在标题命令之后。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630150535933.png)

#### .1. 位置

- h 当前位置。 将图形放置在 正文文本中给出该图形环境的地方。如果本页所剩的页面不够， 这一参数将不起作用。
- t 顶部。 将图形放置在页面的顶部。
- b 底部。 将图形放置在页面的底部 16.1。
- p 浮动页。 将图形放置在一只允许 有浮动对象的页面上。

#### .2. 标题样式

```latex
\usepackage[选项]{caption2}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630150744380.png)

- normal 标题文本两边对齐，其中最后一行为左对齐。
- center 标题文本居中。
- flushleft 标题文本左对齐。
- flushright 标题文本右对齐。
- centerlast 标题文本两边对齐，其中最后一行居中。
- indent 与 normal 式样相似，只是标题文本从第二行开始， 每行行首缩进由命令 captionindent 给出的长度。因为 captionindent 的缺省值为零，通常用像 setlength{captionindent}{1cm} 这样的命令 来设置缩进值。
- hang 与 normal 式样相似，只是标题文本从第二行开始， 每行行首缩进与标题标记宽度相等的长度。

#### .3. 指令含义

- `\includegraphics`命令，使用方括号`[]`传入一个表示图片宽度的参数，使用`{}`传入图像文件位置
- `\linewidth`参数表示图片尺寸适应行宽，尺寸过小的图片将被拉伸，尺寸过大的图片将被压缩
- 文件位置参数，如与`.tex`文件在同一目录下，则直接写明文件名，如在`.tex`所在目录的子目录下则写`folder/file_name.png`。
- `\caption`是出现在图片下方的描述信息
- `\label`是不可见的，作为下次引用的标签

### 1. 单张图片

```latex
%导言区插入下面三行
\usepackage{graphicx} %插入图片的宏包
\usepackage{float} %设置图片浮动位置的宏包
\usepackage{subfigure} %插入多图时用子图显示的宏包

\begin{document}

\begin{figure}[H] %H为当前位置，!htb为忽略美学标准，htbp为浮动图形
\centering %图片居中
\includegraphics[width=0.7\textwidth]{DV_demand} %插入图片，[]中设置图片大小，{}中是图片文件名
\caption{Main name 2} %最终文档中希望显示的图片标题
\label{Fig.main2} %用于文内引用的标签
\end{figure}

\end{document}
```

### 2. 多图排版自定义编号

```latex
%导言区插入下面三行
\usepackage{graphicx}
\usepackage{float} 
\usepackage{subfigure}

\begin{document}
Figure \ref{Fig.main} has two sub figures, fig. \ref{Fig.sub.1} is the travel demand of driving auto, and fig. \ref{Fig.sub.2} is the travel demand of park-and-ride.

\begin{figure}[H]
\centering  %图片全局居中
\subfigure[name1]{
\label{Fig.sub.1}
\includegraphics[width=0.45\textwidth]{DV_demand}}
\subfigure[name2]{
\label{Fig.sub.2}
\includegraphics[width=0.45\textwidth]{P+R_demand}}
\caption{Main name}
\label{Fig.main}
\end{figure}
\end{document}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630145101730.png)

### 3. 多图横排+自定义编号

```latex
%导言区的此三行无变化
\usepackage{graphicx}
\usepackage{float} 
\usepackage{subfigure}
%以下是新增的自定义格式更改
\usepackage[]{caption2} %新增调用的宏包
\renewcommand{\figurename}{Fig.} %重定义编号前缀词
\renewcommand{\captionlabeldelim}{.~} %重定义分隔符
 %\roman是罗马数字编号，\alph是默认的字母编号，\arabic是阿拉伯数字编号，可按需替换下一行的相应位置
\renewcommand{\thesubfigure}{(\roman{subfigure})}%此外，还可设置图编号显示格式，加括号或者不加括号
\makeatletter \renewcommand{\@thesubfigure}{\thesubfigure \space}%子图编号与名称的间隔设置
\renewcommand{\p@subfigure}{} \makeatother

\begin{document}
%注意：此段中在引用中增加了主图编号的引用
Figure \ref{Fig.main} has two sub-figures, fig. \ref{Fig.main}\ref{Fig.sub.1} is the travel demand of driving auto, and fig. \ref{Fig.main}\ref{Fig.sub.2} is the travel demand of park-and-ride.
%以下code与上一小结的无变化
\begin{figure}[H]
\centering  %图片全局居中
\subfigure[name1]{
\label{Fig.sub.1}
\includegraphics[width=0.45\textwidth]{DV_demand}}
\subfigure[name2]{
\label{Fig.sub.2}
\includegraphics[width=0.45\textwidth]{P+R_demand}}
\caption{Main name}
\label{Fig.main}
\end{figure}

\end{document}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630145302450.png)

### 4. 多图并排显示非子图

```latex
%导言区的此三行无变化
\usepackage{graphicx}
\usepackage{float} 
%文章如果不涉及子图，以下代码可以删除，本文因需要一起示例排版，就保留了
\usepackage{subfigure}
\usepackage[]{caption2} %新增调用的宏包
\renewcommand{\figurename}{Fig.} %重定义编号前缀词
\renewcommand{\captionlabeldelim}{.~} %重定义分隔符
 %\roman是罗马数字编号，\alph是默认的字母编号，\arabic是阿拉伯数字编号，可按需替换下一行的相应位置
\renewcommand{\thesubfigure}{(\roman{subfigure})}%此外，还可设置图编号显示格式，加括号或者不加括号
\makeatletter \renewcommand{\@thesubfigure}{\thesubfigure \space}%子图编号与名称的间隔设置
\renewcommand{\p@subfigure}{} \makeatother

\begin{document}

\begin{figure}[H]
\centering %图片全局居中
%并排几个图，就要写几个minipage
\begin{minipage}[b]{0.45\textwidth} %所有minipage宽度之和要小于1，否则会自动变成竖排
\centering %图片局部居中
\includegraphics[width=0.8\textwidth]{DV_demand} %此时的图片宽度比例是相对于这个minipage的，不是全局
\caption{name 1}
\label{Fig.1}
\end{minipage}
\begin{minipage}[b]{0.45\textwidth} %所有minipage宽度之和要小于1，否则会自动变成竖排
\centering %图片局部居中
\includegraphics[width=0.8\textwidth]{P+R_demand}%此时的图片宽度比例是相对于这个minipage的，不是全局
\caption{name 2}
\label{Fig.2}
\end{minipage}
\end{figure}

\end{document}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630145439643.png)

### .5. 多排插入多张图片

> 只需在并排插入图片的代码中加入**换行**即可

```latex
\usepackage{graphicx}
\usepackage{subfigure} %需要使用的宏包

\begin{figure}
\centering
\subfigure[Jackson Yee]{\includegraphics[width=3.5cm]{Jackson.JPG}} 
\subfigure[Jackson Yee]{\includegraphics[width=3.5cm]{Jackson.JPG}}
\\ %换行
\centering
\subfigure[Jackson Yee]{\includegraphics[width=3.5cm]{Jackson.JPG}}
\subfigure[Jackson Yee]{\includegraphics[width=3.5cm]{Jackson.JPG}}
\caption{Jackson Yee} %图片标题
\end{figure}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630145812667.png)

### .6. 垂直（竖向）插入多张图片

```latex
\usepackage{graphicx}
\usepackage{subfigure} %需要使用的宏包

\begin{figure}
\centering
\subfigure[Jackson Yee]{
    \begin{minipage}[b]{0.23\linewidth} %0.23为minipage的宽度，可以调节子图间的距离
    \includegraphics[width=4cm]{Jackson.JPG}\vspace{1pt} %图片的宽度、路径和垂直间距
    \includegraphics[width=4cm]{Jackson.JPG}\vspace{1pt}
    %\vspace要紧跟在对应的includegraphics，不然得不到想要的结果
    \end{minipage}
}
\quad %退一格
\qquad %退两格,调节子图间的距离
\subfigure[Jackson Yee]{
    \begin{minipage}[b]{0.23\linewidth}
    \includegraphics[width=4cm]{Jackson.JPG}\vspace{1pt} 
    \includegraphics[width=4cm]{Jackson.JPG}\vspace{1pt}
    \end{minipage}
}
\caption{Jackson Yee}
\end{figure}
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630150112827.png)

### .7. 问题

#### .1. 标题图片不居中

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630151734030.png)

```latex
\documentclass[UTF8]{ctexart}
\usepackage[margin=2cm]{geometry}
\usepackage{mwe}
\begin{document}

\begin{figure}[htbp]
\begin{minipage}[t]{0.4\textwidth}
\centering
\includegraphics[width=\linewidth]{example-image-a}
\caption{时间间隔为12.5s时压强}
\end{minipage}\hfill
\begin{minipage}[t]{0.4\textwidth}
\centering
\includegraphics[width=\linewidth]{example-image-b}
\caption{时间间隔为25.0s时压强}
\end{minipage}
\end{figure}
\end{document}
```

#### .2. 两张图片顶对齐

> 可以使用 `graphbox` 宏包。它给 `\includegraphics[]{}` 增加了几个选项，其中就有控制纵向对齐的 `align=t|c|b` 选项。

```latex
\documentclass{article}
%\usepackage{graphicx}
\usepackage{graphbox} % loads graphicx
\usepackage{mwe}

\begin{document}
\newcommand{\test}[1]{%
  xx
  \includegraphics[height=3cm, #1]{example-image}
  \includegraphics[height=2cm, #1]{example-image}
  xx\par
}

\test{}
\test{align=t}
\test{align=c}
\end{document}
```

> 在`minipage`内容的最前面加上 `\vspace{0pt}` 这样，`\vspace{0pt}` 就是 `minipage`内容的第一行，他们是零高度，图片自然就顶部对齐了。

```latex
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\begin{minipage}[t]{0.45\linewidth}
  \vspace{0pt}
  \includegraphics[height=2cm]{example-image-a.pdf}
\end{minipage}
\begin{minipage}[t]{0.45\linewidth}
  \vspace{0pt}
  \includegraphics[height=4cm]{example-image-a.pdf}
\end{minipage}
\end{document}
```

#### .3. 图片过宽

这时，我们需要使用一个叫`adjustbox`的宏包。在导言区加上一句

```latex
\usepackage[export]{adjustbox} 
```

然后在正文中使用

```tex
\begin{figure}[H]
\centering
\includegraphics[center]{pic.png}
\end{figure}
```

### Resource

- https://zhuanlan.zhihu.com/p/32925549
- https://www.jianshu.com/p/d9df490e48b8

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/07latex%E7%9A%84%E5%9B%BE%E7%89%87%E6%8F%92%E5%85%A5/  

