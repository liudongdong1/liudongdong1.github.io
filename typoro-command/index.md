# Typoro Command


> ​      Typora 是一个 Markdown 文本编辑器，它支持且仅支持 Markdown 语法的文本编辑。在 [Typora 官网](https://typora.io/) 上他们将 Typora 描述为 「A truly **minimal** markdown editor. 」

## 1. 安装

```shell
#1 安装依赖包
 sudo apt-get install libapt-pkg-dev  
#2 安装、更新 
sudo apt-get install apt-transport-https
sudo apt-get update
#3 安装Typora源
wget -qO - https://typora.io/linux/public-key.asc | sudo apt-key add -
sudo add-apt-repository ‘deb https://typora.io/linux ./‘
sudo apt-get update
#4 安装typora 
sudo apt-get install typora

#首行缩进
&emsp;&emsp;春天来了，又到了万物复苏的季节。
#任务列表
- [ ] 一次性水杯
- [x] 西瓜
#各种表情链接： https://www.webfx.com/tools/emoji-cheat-sheet/
```

## 2. 图片排版

**方法一：嵌入HTML代码**
使用img标签

```html
<img src="./xxx.png" width = "300" height = "200" alt="图片名称" align=center />
<img src=' ' style='float:right; width:300px;height:100 px'/>
#或者
<div align="center">
   <img src="图片地址" height="300px" alt="图片说明" >
</div>
```

**方法二：预定义类**

```html
#居中对齐，img间不要换行，否则识别不了
<center class="half">
    <img src="图片链接" width="200"/><img src="图片链接" width="200"/><img src="图片链接" width="200"/>
</center>
#左对齐并排
<figure class="third">
    <img src="" width="200"/><img src="" width="200"/><img src="" width="200"/>
</figure>
```

## 3. 数学公式

**开启行内公式**：文件→偏好设置→Markdown，勾选内联公式，重启typora    

**下划线**

```
~~中划线~~
$\underline{\text{下划线}}$
$\overline{\text{上划线}}$
```

**分数，平方**

| 算式                        | markdown                    |
| :-------------------------- | :-------------------------- |
| $\frac{7x+5}{1+y^2}$，$1/2$ | \frac{7x+5}{1+y^2} ,    1/2 |

**下标**

| 算式              | markdown                  |
| :---------------- | :------------------------ |
| $z=z_l$ , $z=z^1$ | 下标： z=z_l,  上标 z=z^1 |

**省略号**

| 省略号 | markdown |
| :----- | :------- |
| ⋯      | \cdots   |

**开根号**

| 算式                   | markdown             |
| :--------------------- | :------------------- |
| $\sqrt{2};\sqrt[n]{3}$ | \sqrt{2};\sqrt[n]{3} |

**花括号**

| 算式                                                         | markdown                                                     |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| $c(u)=\begin{cases} \sqrt\frac{1}{N}，u=0\\ \sqrt\frac{2}{N}， u\neq0\end{cases}$ | c(u)=\begin{cases} \sqrt\frac{1}{N}，u=0\\ \sqrt\frac{2}{N}， u\neq0\end{cases}     ,花括号 |
| $a \quad b$                                                  | a \quad b  ,空格                                             |

**矢量**

| 算式                      | markdown                |
| :------------------------ | :---------------------- |
| $\vec{a} \cdot \vec{b}=0$ | \vec{a} \cdot \vec{b}=0 |

**积分**

| 算式                     | markdown               |
| :----------------------- | :--------------------- |
| $\int ^2_3 x^2 {\rm d}x$ | \int ^2_3 x^2 {\rm d}x |

**极限**

| 算式                           | markdown                     |
| :----------------------------- | :--------------------------- |
| $\lim_{n\rightarrow+\infty} n$ | \lim_{n\rightarrow+\infty} n |

**累加**

| 算式                 | markdown           |
| :------------------- | :----------------- |
| $\sum \frac{1}{i^2}$ | \sum \frac{1}{i^2} |

**累乘**

| 算式                  | markdown            |
| :-------------------- | :------------------ |
| $\prod \frac{1}{i^2}$ | \prod \frac{1}{i^2} |

**希腊字母**

| 大写 | markdown | 小写 | markdown    |
| :--- | :------- | :--- | :---------- |
| A    | A        | α    | \alpha      |
| B    | B        | β    | \beta       |
| Γ    | \Gamma   | γ    | \gamma      |
| Δ    | \Delta   | δ    | \delta      |
| E    | E        | ϵ    | \epsilon    |
|      |          | ε    | \varepsilon |
| Z    | Z        | ζ    | \zeta       |
| H    | H        | η    | \eta        |
| Θ    | \Theta   | θ    | \theta      |
| I    | I        | ι    | \iota       |
| K    | K        | κ    | \kappa      |
| Λ    | \Lambda  | λ    | \lambda     |
| M    | M        | μ    | \mu         |
| N    | N        | ν    | \nu         |
| Ξ    | \Xi      | ξ    | \xi         |
| O    | O        | ο    | \omicron    |
| Π    | \Pi      | π    | \pi         |
| P    | P        | ρ    | \rho        |
| Σ    | \Sigma   | σ    | \sigma      |

| 大写 | markdown | 小写 | markdown |
| :--- | :------- | :--- | :------- |
| T    | T        | τ    | \tau     |
| Υ    | \Upsilon | υ    | \upsilon |
| Φ    | \Phi     | ϕ    | \phi     |
|      |          | φ    | \varphi  |
| X    | X        | χ    | \chi     |
| Ψ    | \Psi     | ψ    | \psi     |
| Ω    | \Omega   | ω    | \omega   |

**三角函数**

| 三角函数 | markdown |
| :------- | :------- |
| sin      | \sin     |

**对数函数**

| 算式   | markdown  |
| :----- | :-------- |
| ln15   | \ln15     |
| log210 | \log_2 10 |
| lg7    | \lg7      |

**关系运算符**

| 运算符 | markdown |
| :----- | :------- |
| ±      | \pm      |
| ×      | \times   |
| ÷      | \div     |
| ∑      | \sum     |
| ∏      | \prod    |
| ≠      | \neq     |
| ≤      | \leq     |
| ≥      | \geq     |

**其它特殊字符**

| 符号         | markdown   |
| :----------- | :--------- |
| $\forall$    | \forall    |
| $\infty$     | \infty     |
| $\emptyset$  | \emptyset  |
| $\exists$    | \exists    |
| $\nabla$     | \nabla     |
| $\bot$       | \bot       |
| $\angle$     | \angle     |
| $\because$   | \because   |
| $\therefore$ | \therefore |

\emptyset \in \notin \subset \supset \subseteq \supseteq \bigcap \bigcup \bigvee \bigwedge \biguplus \bigsqcup

$\emptyset \in \notin \subset \supset \subseteq \supseteq \bigcap \bigcup \bigvee \bigwedge \biguplus \bigsqcup$

**矩阵：**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201108164058875.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201108164558410.png)

## 4. 主体样式之类

- 添加水印： https://blog.csdn.net/qq_36188663/article/details/105376834

## 5. Latex

```latex
\documentclass{article}
\pagestyle{empty}
\setcounter{page}{6}
\setlength\textwidth{266.0pt}
\usepackage{CJK}
\usepackage{amsmath}

\begin{CJK}{GBK}{song}
\begin{document}

\begin{align}
  (a + b)^3  &= (a + b) (a + b)^2        \\
             &= (a + b)(a^2 + 2ab + b^2) \\
             &= a^3 + 3a^2b + 3ab^2 + b^3
\end{align}
\begin{align}
  x^2  + y^2 & = 1                       \\
  x          & = \sqrt{1-y^2}
\end{align}
This example has two column-pairs.
\begin{align}    \text{Compare }
  x^2 + y^2 &= 1               &
  x^3 + y^3 &= 1               \\
  x         &= \sqrt   {1-y^2} &
  x         &= \sqrt[3]{1-y^3}
\end{align}
This example has three column-pairs.
\begin{align}
    x    &= y      & X  &= Y  &
      a  &= b+c               \\
    x'   &= y'     & X' &= Y' &
      a' &= b                 \\
  x + x' &= y + y'            &
  X + X' &= Y + Y' & a'b &= c'b
\end{align}

This example has two column-pairs.
\begin{flalign}  \text{Compare }
  x^2 + y^2 &= 1               &
  x^3 + y^3 &= 1               \\
  x         &= \sqrt   {1-y^2} &
  x         &= \sqrt[3]{1-y^3}
\end{flalign}
This example has three column-pairs.
\begin{flalign}
    x    &= y      & X  &= Y  &
      a  &= b+c               \\
    x'   &= y'     & X' &= Y' &
      a' &= b                 \\
  x + x' &= y + y'            &
  X + X' &= Y + Y' & a'b &= c'b
\end{flalign}

This example has two column-pairs.
\renewcommand\minalignsep{0pt}
\begin{align}    \text{Compare }
  x^2 + y^2 &= 1               &
  x^3 + y^3 &= 1              \\
  x         &= \sqrt   {1-y^2} &
  x         &= \sqrt[3]{1-y^3}
\end{align}
This example has three column-pairs.
\renewcommand\minalignsep{15pt}
\begin{flalign}
    x    &= y      & X  &= Y  &
      a  &= b+c              \\
    x'   &= y'     & X' &= Y' &
      a' &= b                \\
  x + x' &= y + y'            &
  X + X' &= Y + Y' & a'b &= c'b
\end{flalign}

\renewcommand\minalignsep{2em}
\begin{align}
  x      &= y      && \text{by hypothesis} \\
      x' &= y'     && \text{by definition} \\
  x + x' &= y + y' && \text{by Axiom 1}
\end{align}

\begin{equation}
\begin{aligned}
  x^2 + y^2  &= 1               \\
  x          &= \sqrt{1-y^2}    \\
 \text{and also }y &= \sqrt{1-x^2}
\end{aligned}               \qquad
\begin{gathered}
 (a + b)^2 = a^2 + 2ab + b^2    \\
 (a + b) \cdot (a - b) = a^2 - b^2
\end{gathered}      \end{equation}

\begin{equation}
\begin{aligned}[b]
  x^2 + y^2  &= 1               \\
  x          &= \sqrt{1-y^2}    \\
 \text{and also }y &= \sqrt{1-x^2}
\end{aligned}               \qquad
\begin{gathered}[t]
 (a + b)^2 = a^2 + 2ab + b^2    \\
 (a + b) \cdot (a - b) = a^2 - b^2
\end{gathered}
\end{equation}
\newenvironment{rcase}
    {\left.\begin{aligned}}
    {\end{aligned}\right\rbrace}

\begin{equation*}
  \begin{rcase}
    B' &= -\partial\times E          \\
    E' &=  \partial\times B - 4\pi j \,
  \end{rcase}
  \quad \text {Maxwell's equations}
\end{equation*}

\begin{equation} \begin{aligned}
  V_j &= v_j                      &
  X_i &= x_i - q_i x_j            &
      &= u_j + \sum_{i\ne j} q_i \\
  V_i &= v_i - q_i v_j            &
  X_j &= x_j                      &
  U_i &= u_i
\end{aligned} \end{equation}

\begin{align}
  A_1 &= N_0 (\lambda ; \Omega')
         -  \phi ( \lambda ; \Omega')   \\
  A_2 &= \phi (\lambda ; \Omega')
            \phi (\lambda ; \Omega)     \\
\intertext{and finally}
  A_3 &= \mathcal{N} (\lambda ; \omega)
\end{align}
\end{CJK}
\end{document}
```


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/typoro-command/  

