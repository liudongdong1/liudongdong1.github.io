# Understand


## 导入项目

导入项目有两种方法，一种是从菜单栏点击File–>New–>Project,另一种是点击下面界面中间的New Project

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/1acce5830802178d2cff701e25dd4895.png)](http://www.codemx.cn/images/understand/understand1/under01.png)

点击后，会进入到如下界面，你可以更改项目名称为你要导入的项目名称，以便于以后查找，你可以直接导入你正在开发的项目，你的代码更改后，这个项目也会自动更新，方便你快速开发，不需要每次导入。

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/433ef1749e5b30824e37dca6c566b932.png)](http://www.codemx.cn/images/understand/understand1/under02.png)

更改名称后点击Next进入如下界面，这个界面是让你选择你要导入项目包含了哪几种语言，注意，在C/C++后面有两种模式，下面有注释，其中Strict模式包含Object-C和Object—C++，还有Web的注释，自己看看就好了，在此就不再解释，

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/953bba343317ddc54b288af60a873dd2.png)](http://www.codemx.cn/images/understand/understand1/under03.png)

然后点击Next进入下面界面：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/273ecb42a1c8f218744086571b36e1d5.png)](http://www.codemx.cn/images/understand/understand1/under04.png)

在此界面点击上面的“Add a Directory”,也就是添加你要导入项目的路径，点击后会弹出如下界面,此时有个奇葩就是弹出的界面会被上图界面遮挡，此时你要移开该界面，然后会出现下面界面：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/96885db9a8dfc896615c3403f151ee34.png)](http://www.codemx.cn/images/understand/understand1/under05.png)

点击后面的带有三个点的按钮选择你要加入的项目文件夹，此处不用打开文件夹，只要点中文件夹点击open按钮：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/fd87c3710cbe985a921538f36abed049.png)](http://www.codemx.cn/images/understand/understand1/under06.png)

此时只需要点击OK即可，界面会跳转到如下界面：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/3ec97f93738a08b2d4e030b537b7ac86.png)](http://www.codemx.cn/images/understand/understand1/under07.png)

此时有两个选项，一个是立即分析代码，一个选择配置，对于我们来说只需要默认即可，然后点击OK按钮，此时软件开始分析代码，分析完成后会出现如下界面：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/466b3d679c26ed04ea2a11209f53fb00.png)](http://www.codemx.cn/images/understand/understand1/under08.png)

左侧会出你的项目结构，中间出现你项目的名称，此时你可以操作左面项目来查看相关代码，如下图所示：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/2fcf6727bd6b468530c2209fb6d10292.png)](http://www.codemx.cn/images/understand/understand1/under09.png)

这么多类和方法如何快速定位，那肯定是搜索，该软件针对不同位置，不同属性有不同的搜索方法，下面介绍搜索功能。

## 搜索功能

1.左侧项目结构中搜索：在这个搜索中你可以快速搜索你要查看的类，快捷键，鼠标点击左侧上面项目结构窗口，然后按command + F键会出现如下图所示的搜索框，在框中输入你想要的类回车即可

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/331930e557a6f66ca12185dd71b9ebb3.png)](http://www.codemx.cn/images/understand/understand1/under10.png)

2.类中方法搜索：将鼠标定位到右侧代码中，点击command + F，会弹出搜索框，输入方法回车即可：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/d7c7284bf052d48f00c29fa8585fa88e.png)](http://www.codemx.cn/images/understand/understand1/under11.png)

3.在文件中搜索：也就是全局搜索，快捷键F5或者去上面菜单栏中的search栏中查找，输入你想要的类或者方法，回车查找，下面会列出所有使用的地方：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/fce08caaea82fcbaf9823ad4656b2b8a.png)](http://www.codemx.cn/images/understand/understand1/under12.png)

4.实体类查找：软件菜单栏search中最后一项–Find Entity，点击输入你要查找的实体类，回车查找：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/cb87619d1a29e46f4c1914744302425a.png)](http://www.codemx.cn/images/understand/understand1/under13.png)

快速搜索是软件快速使用必备的技能，包括我们常用的idea一样，快速定位类，方法，常量等，可以快速帮助我们解决问题。

上面我介绍改软件时提到可以绘制流程图等功能，下面就针对这个功能介绍一些一些图形的绘制功能，帮助你快速分析代码。

## 项目视图

项目视图包含很多的功能，能够自动生成各种流程图结构图，帮助你快速理清代码逻辑、结构等，以便快速理解项目流程，快速开发，视图查看方式有两种，一种是鼠标点击你要查看的类或者方法等上面，然后右键弹出菜单，鼠标移动到Graphical Views，然后弹出二级菜单，如下图所示：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/2f5d1daa5cbb35bf66ec45d64930cb7e.png)](http://www.codemx.cn/images/understand/understand1/under14.png)

另一种方式是点击要查看的类或者方法，然后找到代码上面菜单栏中的如下图标：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/627184460e0808989bdf7298c038bb98.png)](http://www.codemx.cn/images/understand/understand1/under15.png)

然后点击图标右下角的下拉箭头，弹出如下菜单，即可选择查看相关视图：

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/5e431327bfd4aa44e6f8525960a1dea7.png)

### 层级关系视图分类：

#### Butterfly

- 如果两个实体间存在关系，就显示这两个实体间的调用和被调用关系；如下图为Activity中的一个方法的关系图：

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/b3173a3c00294ef7504f02cc233bdbbe.png)

#### Call

- 展示从你选择的这个方法开始的整个调用链条；

#### Called By

- 展示了这个实体被哪些代码调用，这个结构图是从底部向上看或者从右到左看；

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/74e13801e33bfe75caea5d8fe3e6ae5a.png)](http://www.codemx.cn/images/understand/understand1/under19.png)

Calls Relationship/Calledby Relationship

- 展示了两个实体之间的调用和被调用关系，操作方法：首先右键你要选择的第一个实体，然后点击另一个你要选择的实体，如果选择错误，可以再次点击其他正确即可，然后点击ok；

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/86583c6619d9a1bf23fb88586cf8c592.png)](http://www.codemx.cn/images/understand/understand1/under20.png)

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/1064b7cd0f59435a38012b7090562036.png)](http://www.codemx.cn/images/understand/understand1/under21.png)

#### Contains

- 展示一个实体中的层级图，也可以是一个文件，一条连接线读作”x includes y“；

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/1ab6c33ce460682065568654b37e7e23.png)](http://www.codemx.cn/images/understand/understand1/under22.png)

#### Extended By

- 展示这个类被哪些类所继承，

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/9fba79e781ec1d4d2abe5dc8cf89e640.png)](http://www.codemx.cn/images/understand/understand1/under23.png)

#### Extends

- 展示这个类继承自那个类：

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/0a7385fe4189dbc0f083d8be4cef1395.png)](http://www.codemx.cn/images/understand/understand1/under24.png)

### 结构关系视图分类：

#### Graph Architecture

- 展示一个框架节点的结构关系；

#### Declaration

- 展示一个实体的结构关系，例如：展示参数，则返回类型和被调用函数，对于类，则展示私有成员变量（谁继承这个类，谁基于这个类）

#### Parent Declaration

- 展示这个实体在哪里被声明了的结构关系；

#### Declaration File

- 展示所选的文件中所有被定义的实体（例如函数，类型，变量，常量等）；

#### Declaration Type

- 展示组成类型；

#### Control Flow

- 展示一个实体的控制流程图或者类似实体类型；

[![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/2556442e0fed6a8894cc7bbb90bf67da.png)](http://www.codemx.cn/images/understand/understand1/under25.png)

#### UML Class Diagram

- 展示整个项目或者目录类关系

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221130200659278.png)

#### UML Sequence Diagram

- 展示两个实体之间的时序关系图；

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20221130201335287.png)

12.Package:展示给定包名中声明的所有实体

13.Task:展示一个任务中的参数，调用，实体

14.Rename Declaration:展示实体中被重命名的所有实体

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQWxwaGFBQkNE,size_17,color_FFFFFF,t_70,g_se,x_16.png)



## 问题

- 中文乱码问题：https://segmentfault.com/q/1010000002385460

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/understand-%E4%BB%A3%E7%A0%81%E6%9F%A5%E7%9C%8B/  

