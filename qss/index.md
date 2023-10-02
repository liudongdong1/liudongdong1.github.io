# QSS


> Styles sheets are textual specifications that can be set on the whole application using [QApplication::setStyleSheet](https://doc.qt.io/archives/qt-4.8/qapplication.html#styleSheet-prop)() or on a specific widget (and its children) using [QWidget::setStyleSheet](https://doc.qt.io/archives/qt-4.8/qwidget.html#styleSheet-prop)(). If several style sheets are set at different levels, Qt derives the effective style sheet from all of those that are set. This is called cascading.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210717091042787.png)

> 样式表使用方式：
>
> 1. 在Qt Designer设计窗体时，直接用样式表编辑器为窗体或窗体上的某个部件设计样式表，对应用程序是固定的，无法取得换肤的效果，而且需要为每个窗体都涉及样式表，重复性工作太大；方法：`右击窗体或某个部件→选择“Change styleSheet”`。
>
> 2.  setStyleSheet函数:
>
>    1. 使用qApp的setStyleSheet函数可以为应用程序全局设置样式。`qApp->setStyleSheet("QLineEdit{background-color: gray}");`
>    2. 使用QWidget::setStyleSheet函数可以为`一个窗口、一个对话框、一个界面组件设置样式`。例如下面为主窗口MainWindow内的QLineEdit组件设置样式 `MainWindow->setStyleSheet("QLineEdit {background-color: lime}");`
>    3. 单独设置一个Object对象的样式表。这种情况无需设置selector（选择器）的名称。例如下面是设置一个名为editName的QLineEdit组件的样式: `editName-setStyleSheet("color:blue;" background-color: yellow;")`
>
> 3. `使用.qss文件`，为了实现动态切换样式表，一般将样式定义保存为.qss后缀的纯文本文件，然后再程序中打开文件,读取文件内容，再调用`setStyleSheet函数应用样式表`
>
>    ```c++
>    QFile file(":/qss/mystyle.qss");
>    file.open(QFile::ReadOnly);
>    QString styleSheet=QString:: fromLatin1(file.readAll());
>    aQpp-setStyleSheet(styleSheet)
>    ```

### 0. 选择器

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210717093643291.png)

```css
QPushButton:hover { color: white }  /*当鼠标悬停在QPushButton上时*/
QRadioButton:!hover { color: red }  /*当鼠标（不）悬停在QRadioButton上时*/
QCheckBox:hover:checked { color: red } /*鼠标悬停在已检查的QCheckBox上的情况：*/
QPushButton:hover:!pressed { color: blue } /*悬停在未按下的QPushButton上时*/
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210717094054711.png)

### 1. 盒子模型

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210717091717631.png)

- `border-style`: dotted;(点线) dashed;(虚线) solid; double;(双线) groove;(槽线) ridge;(脊状) inset;(凹陷) outset;
- `border-width`:; 边框宽度
- `border-color`:#;
- **border-image**：`边框图片`
- `border-radius`：元素的外边框圆角
  - **border-top-left-radius**
  - **border-top-right-radius**
  - **border-bottom-right-radius**
  - **border-bottom-left-radius**
- **opacity**：控件的不透明度, 可以用来显示一些控件颜色透明,  background-color:rgb(14 , 135 , 228,0);  RGB 透明度

##### .1. button 按钮

```css
QPushButton#create{
  padding-left:-30px;
  color:black;
  border: 2px solid #067d9e; 
  font: 18pt "TImes New Roman";
  background-image:url("icons/create.png");
  border-radius:10px;
  background-repeat:none;
}
QPushButton#create:hover
{
   border-radius: 10px;
   background-color:#067d9e;
}

QPushButton#transfer{
border-radius: 10px;
border-image:url("icons/gesture2.png") 4 4 4 4 stretch stretch;
background-repeat:no-repeat;
background-size:100% 100%;
}

border: 2px solid white;
```

##### .2. 图片填充

```css
border-image:url(:/images/bd.png) 4 4 4 4 stretch stretch;  /*图片填充满label*/
```

```css
padding-top:10px; /*上边框留空白*/
padding-right:10px; /*右边框留空白*/
padding-bottom:10px; /*下边框留空白*/
padding-left:10px; /*左边框留空白*/
margin-top:10px; /*上边界*/
margin-right:10px; /*右边界值*/
margin-bottom:10px; /*下边界值*/
margin-left:10px; /*左边界值*/

border-top : 1px solid #6699cc; /*上框线*/   
border-bottom : 1px solid #6699cc; /*下框线*/
border-left : 1px solid #6699cc; /*左框线*/
border-right : 1px solid #6699cc; /*右框线*/
solid /*实线框*/  dotted /*虚线框*/  double /*双线框*/  groove /*立体内凸框*/  ridge /*立体浮雕框*/
inset /*凹框*/  outset /*凸框*/
border-top-color : #369 /*设置上框线top颜色*/
border-top-width :1px /*设置上框线top宽度*/
border-top-style : solid/*设置上框线top样式*/
```

### 2. 字体属性

- 大小 {`font-size:` x-large;}(特大) xx-small;(极小) 一般中文用不到，只要用数值就可以，单位：`PX、PD`
- 样式 {`font-style`: oblique;}(偏斜体) italic;(斜体) normal;(正常)
- 行高 {`line-height`: normal;}(正常) 单位：PX、PD、EM
- 粗细 {`font-weight`: bold;}(粗体) lighter;(细体) normal;(正常)
- 变体 {`font-variant`: small-caps;}(小型大写字母) normal;(正常)
- 大小写 {`text-transform`: capitalize;}(首字母大写) uppercase;(大写) lowercase;(小写) none;(无)
- 修饰 {`text-decoration`: underline;}(下划线) overline;(上划线) line-through;(删除线) blink;(闪烁)
- 常用字体： `font-family`："Courier New", Courier, monospace, "Times New Roman", Times, serif, Arial, Helvetica, sans-serif, Verdana
- 字体颜色 {`color`:数值;}
- 阴影颜色 {`text-shadow`:16位色值}
- 水平对齐 {`text-align`:left|right|center|justify};  垂直对齐 {`vertical-align`:inherit|top|bottom|text-top|text-bottom|baseline|middle|sub|super}

```css
/* {font:font-style font-variant font-weight font-size font-family}*/
font: 18pt "TImes New Roman";
color : #999999; /*文字颜色*/

font-family : 宋体,sans-serif; /*文字字体*/

font-size : 9pt; /*文字大小*/

font-style:itelic; /*文字斜体*/

font-variant:small-caps; /*小字体*/

letter-spacing : 1pt; /*字间距离*/

line-height : 200%; /*设置行高*/

font-weight:bold; /*文字粗体*/

vertical-align:sub; /*下标字*/

vertical-align:super; /*上标字*/

text-decoration:line-through; /*加删除线*/

text-decoration: overline; /*加顶线*/

text-decoration:underline; /*加下划线*/

text-decoration:none; /*删除链接下划线*/

text-transform : capitalize; /*首字大写*/

text-transform : uppercase; /*英文大写*/

text-transform : lowercase; /*英文小写*/

text-align:right; /*文字右对齐*/

text-align:left; /*文字左对齐*/

text-align:center; /*文字居中对齐*/

text-align:justify; /*文字分散对齐*/

vertical-align属性

vertical-align:top; /*垂直向上对齐*/

vertical-align:bottom; /*垂直向下对齐*/

vertical-align:middle; /*垂直居中对齐*/

vertical-align:text-top; /*文字垂直向上对齐*/

vertical-align:text-bottom; /*文字垂直向下对齐*/
```

### 3. 背景属性

- 色彩 {`background-color`: #FFFFFF;}
- 图片 {`background-image`: url();}
- 重复 {`background-repeat`: no-repeat;}
- 滚动 {`background-attachment`: fixed;}(固定) scroll;(滚动)
- 位置 {`background-position:` left;}(水平) top(垂直);
- 简写方法 {`background:#000 url(..) repeat fixed left top;`} 

```css
background-color:#F5E2EC; /*背景颜色*/
background:transparent; /*透视背景*/
background-image : url(/image/bg.gif); /*背景图片*/
background-attachment : fixed; /*浮水印固定背景*/
background-repeat : repeat; /*重复排列-网页默认*/
background-repeat : no-repeat; /*不重复排列*/
background-repeat : repeat-x; /*在x轴重复排列*/
background-repeat : repeat-y; /*在y轴重复排列*/
background-position : 90% 90%; /*背景图片x与y轴的位置*/
background-position : top; /*向上对齐*/
background-position : buttom; /*向下对齐*/
background-position : left; /*向左对齐*/
background-position : right; /*向右对齐*/
background-position : center; /*居中对齐*/
```

### 4. 区块属性

- 字间距 {`letter-spacing`: normal;} 数值
- 对齐 {`text-align`: justify;}(两端对齐) left;(左对齐) right;(右对齐) center;(居中)
- 缩进 {`text-indent`: 数值px;}
- 垂直对齐 {`vertical-align`: baseline;}(基线) sub;(下标) super;(下标) top; text-top; middle; bottom; text-bottom;
- 词间距`word-spacing:` normal; 数值
- 空格`white-space`: pre;(保留) nowrap;(不换行)
- 显示 {`display`:block;}(块) inline;(内嵌) list-item;(列表项) run-in;(追加部分) compact;(紧凑) marker;(标记) table; inline-table; table-raw-group; table-header-group; table-footer-group; table-raw; table-column-group; table-column; table-cell; table-caption;(表格标题)

### 5. 鼠标样式

```css
链接手指 CURSOR: hand
十字体 cursor:crosshair
箭头朝下 cursor:s-resize
十字箭头 cursor:move
箭头朝右 cursor:move
加一问号 cursor:help
箭头朝左 cursor:w-resize
箭头朝上 cursor:n-resize
箭头朝右上 cursor:ne-resize
箭头朝左上 cursor:nw-resize
文字I型 cursor:text
箭头斜右下 cursor:se-resize
箭头斜左下 cursor:sw-resize
漏斗 cursor:wait
```



### Resource

- https://doc.qt.io/archives/qt-4.8/stylesheet.html
- https://www.w3cschool.cn/css/css-border.html
- [style-sheet-usage:](https://doc.qt.io/archives/qt-4.8/stylesheet-examples.html#style-sheet-usage)
- https://github.com/892768447/PyQtUiLibrary

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210717101650815.png)

- https://github.com/daodaoliang/NBaseUiKit  pyqt 组件

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210717101400458.png)

- https://github.com/TheOpenDevProject/QssUI

- 本地目录： E:\work_天津大学课程相关\QT教程

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: liudongdong1.github.io/qss/  

