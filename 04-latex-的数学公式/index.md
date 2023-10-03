# LaTex 的数学公式


> - https://mathpix.com/ 能够通过热键呼出截屏，而后将截屏中的公式转换成 LaTeX 数学公式的代码。
> - http://detexify.kirelabs.org/classify.html 允许用户用鼠标在输入区绘制单个数学符号的样式，系统会根据样式返回对应的 LaTeX 代码（和所需的宏包）。这在查询不熟悉的数学符号时特别有用。

#### 1. 数学公式

#### .1. base structure

> LaTeX 的数学模式有两种：行内模式 (inline) 和行间模式 (display)。前者在正文的行文中，插入数学公式；后者独立排列单独成行，并自动居中。
>
> 在行文中，使用 `$ ... $` 可以插入行内公式，使用 `\[ ... \]` 可以插入行间公式，如果需要对行间公式进行编号，则可以使用 `equation` 环境：

```latex
\usepackage{amsmath}
\begin{equation}
...
\end{equation}
```

#### .2. 上下标

```latex
\documentclass{article}
\usepackage{amsmath}
\begin{document}
Einstein 's $E=mc^2$.

\[ E=mc^2. \]

\begin{equation}
E=mc^2.
\end{equation}
\end{document}
```

#### .3. 根式&分式

```latex
\documentclass{article}
\usepackage{amsmath}
\begin{document}
$\sqrt{x}$, $\frac{1}{2}$.

\[ \sqrt{x}, \]

\[ \frac{1}{2}. \]
\end{document}
```

#### .4. 省略号

> 省略号用 `\dots`, `\cdots`, `\vdots`, `\ddots` 等命令表示。`\dots` 和 `\cdots` 的纵向位置不同，前者一般用于有下标的序列。

```latex
\[ x_1,x_2,\dots ,x_n\quad 1,2,\cdots ,n\quad
\vdots\quad \ddots \]
```

#### .5. 矩阵

> `amsmath` 的 `pmatrix`, `bmatrix`, `Bmatrix`, `vmatrix`, `Vmatrix` 等环境可以在矩阵两边加上各种分隔符。

```latex
\[ \begin{pmatrix} a&b\\c&d \end{pmatrix} \quad
\begin{bmatrix} a&b\\c&d \end{bmatrix} \quad
\begin{Bmatrix} a&b\\c&d \end{Bmatrix} \quad
\begin{vmatrix} a&b\\c&d \end{vmatrix} \quad
\begin{Vmatrix} a&b\\c&d \end{Vmatrix} \]
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630144044800.png)

> 使用 `smallmatrix` 环境，可以生成行内公式的小矩阵。

```latex
Marry has a little matrix $ ( \begin{smallmatrix} a&b\\c&d \end{smallmatrix} ) $.
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630144118024.png)

#### .6.公式对齐

> 需要对齐的公式，可以使用 `aligned` *次环境*来实现，它必须包含在数学环境之内。

```latex
\[\begin{aligned}
x ={}& a+b+c+{} \\
&d+e+f+g
\end{aligned}\]
```

#### .7. 分段函数

```latex
\[ y= \begin{cases}
-x,\quad x\leq 0 \\
x,\quad x>0
\end{cases} \]
```

#### 2. 特殊符号

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630142225077.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630142328962.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630142507570.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210630142514882.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/04-latex-%E7%9A%84%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F/  

