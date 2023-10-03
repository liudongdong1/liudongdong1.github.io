# MatplotDraw Instance Record


## 1.基本概念

> - canvas 类似画板
> - figure 类似画布（或理解为画图区域）
> - axes 子图（或理解为坐标系）
> - 各类图表信息，包括：xaxis（x轴），yaxis（y轴），title（标题），legend（图例），grid（网格线），spines（边框线）,data（数据）等等

```python
# 正常显示中文标签，包括：xaxis（x轴），yaxis（y轴），title（标题），legend（图例）
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
#宋体，仿宋，新宋体，黑体，楷体。windows路径C:\Windows\Fonts ，选中右键属性可以查看英文名字，只能windows,其他系统没试过
plt.rcParams['axes.unicode_minus']=False
#有时候x轴或者y轴刻度负号不能显示，默认参数plt.rcParams['axes.unicode_minus']=True
plt.rcParams['axes.labelsize']=20
#控制x和y轴的标签大小为20像素
plt.rcParams['lines.linewidth']=17.5  #控制图例大小为14像素
plt.rcParams['figure.figsize']=[12,8]
#把图设置为12*8 大小
```

```python
import matplotlib.pyplot as plt
import numpy as np
# 正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
# 刻度大小
plt.rcParams['axes.labelsize']=16
# 线的粗细
plt.rcParams['lines.linewidth']=17.5
# x轴标签大小
plt.rcParams['xtick.labelsize']=14
# y轴标签大小
plt.rcParams['ytick.labelsize']=14
#图例大小
plt.rcParams['legend.fontsize']=14
# 图大小
plt.rcParams['figure.figsize']=[12,8]
plt.xlabel('我是x轴')
plt.ylabel('我是y轴')
#设置绘图标题
plt.title('Matplotlib绘图基础常用设置')
#根据x和y画图
plt.plot(x,y,label='我是图例')
#显示图例
plt.legend()
# 显示图形   
plt.show()
```

```python
params={
    'axes.labelsize': '16',       
    'xtick.labelsize':'14',
    'ytick.labelsize':'14',
    'lines.linewidth':17.5 ,
    'legend.fontsize': '14',
    'figure.figsize'   : '12, 8'}
plt.rcParams.update(params)
plt.xlabel('我是x轴')
plt.ylabel('我是y轴')
#设置绘图标题
plt.title('Matplotlib绘图基础常用设置')
#根据x和y画图
plt.plot(x,y,label='我是图例')
#显示图例
plt.legend()
# 显示图形   
plt.show()
```

## 2. 柱状图

> bar(left, height, width=0.8, bottom=None, **kwargs)
>
> - left：为和分类数量一致的数值序列，序列里的数值数量决定了柱子的个数，数值大小决定了距离0点的位置
> - height：为分类变量的数值大小，决定了柱子的高度
> - width：决定了柱子的宽度，仅代表形状宽度而已
> - bottom：决定了柱子距离x轴的高度，默认为None，即表示与x轴距离为0
>
> - edgecolor 或 ec 描边颜色
> - linestyle 或 ls 描边样式
> - linewidth 或 lw 描边宽度
>
> barh(bottom, width, height=0.8, left=None, **kwargs):   旋转90度

```python
fig = plt.figure()
# 生成第一个子图在1行2列第一列位置
ax1 = fig.add_subplot(121)
# 生成第二子图在1行2列第二列位置
ax2 = fig.add_subplot(122)
ax1.bar(x, y, fc='c')
ax2.bar(x, y,color=['r', 'g', 'b']) # 或者color='rgb' , color='#FFE4C4'
plt.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603184902401.png)

### 2.1 并列柱状图

```python
import numpy as np
#显示中文字体为SimHei
plt.rcParams['font.sans-serif']=['SimHei']
sale8 = [5,20,15,25,10]
sale9 = [10,15,25,30,5]
# x轴的刻度为1-5号衣服
labels = ["{}号衣服".format(i) for i in range(1,6)]
fig,ax = plt.subplots(figsize=(8,5),dpi=80)
width_1 = 0.4
ax.bar(np.arange(len(sale8)),sale8,width=width_1,tick_label=labels,label = "8月")
ax.bar(np.arange(len(sale9))+width_1,sale9,width=width_1,tick_label=labels,label="9月")
ax.legend()
plt.show()
```

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603185015379.png" alt="image-20200603185015379" style="zoom:50%;" />

### 2.2 正负条形柱

```python
import matplotlib.pyplot as plt
import numpy as np
# 正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.labelsize']=16
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
plt.rcParams['legend.fontsize']=12
plt.rcParams['figure.figsize']=[15,15]
plt.style.use("ggplot")
price = [39.5,39.9,45.4,38.9,33.34]
fig,ax = plt.subplots(figsize=(12,5),dpi=80)
x = range(len(price))
# 添加刻度标签
labels = np.array(['亚马逊','当当网','中国图书网','京东','天猫'])
#在子图对象上画条形图，并添加x轴标签，图形的主标题
ax.barh(x,price,tick_label=labels,alpha = 0.8)
ax.set_xlabel('价格',color='k')
ax.set_title('不同平台书的最低价比较')
# 设置Y轴的刻度范围
ax.set_xlim([32,47])
# 为每个条形图添加数值标签
for x,y in enumerate(price):
    ax.text(y+0.2,x,y,va='center',fontsize=14)
# 显示图形   
plt.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603185235694.png)

```python
import matplotlib.pyplot as plt
import numpy as np
# 正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.labelsize']=16
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
plt.rcParams['legend.fontsize']=16
plt.rcParams['figure.figsize']=[12,10]
plt.style.use("ggplot")
# 构建数据
Y2016 = [15600,12700,11300,4270,3620]
Y2017 = [17400,14800,12000,5200,4020]
labels = ['北京','上海','香港','深圳','广州']
bar_width = 0.35
x = np.arange(len(Y2016))
fig = plt.figure()
ax = fig.add_subplot(111)
# 绘图
ax.bar(x,Y2016,label='Y2016',width=bar_width)
ax.bar(x+bar_width,Y2017,label='Y2017',width=bar_width)
# 添加轴标签
ax.set_xlabel('Top5城市')
ax.set_ylabel('家庭数量')
# 添加标题
ax.set_title('亿万财富家庭数Top5城市分布',fontsize=16)
# 添加刻度标签
plt.xticks(x+bar_width,labels)
# 设置Y轴的刻度范围
plt.ylim([2500, 19000])
# 为每个条形图添加数值标签
for x2016,y2016 in enumerate(Y2016):
    plt.text(x2016, y2016+200, y2016,ha='center',fontsize=16)
for x2017,y2017 in enumerate(Y2017):
  plt.text(x2017+0.35,y2017+200,y2017,ha='center',fontsize=16)
# 显示图例
ax.legend()
# 显示图形
plt.show()
```

## 3. 折线图

```python
# coding: utf-8
import matplotlib.pyplot as plt

# figsize = 11, 9
# figure, ax = plt.subplots(figsize = figsize)
x1 =[0,5000,10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
y1=[0, 223, 488, 673, 870, 1027, 1193, 1407, 1609, 1791, 2113, 2388]
x2 = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
y2 = [0, 214, 445, 627, 800, 956, 1090, 1281, 1489, 1625, 1896, 2151]

# 设置输出的图片大小
figsize = 9, 9
figure, ax = plt.subplots(figsize=figsize)

# 在同一幅图片上画两条折线
A, = plt.plot(x1, y1, '-r', label='A', linewidth=5.0)
B, = plt.plot(x2, y2, 'b-.', label='B', linewidth=5.0)

# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 23,
         }
legend = plt.legend(handles=[A, B], prop=font1)

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]
# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
plt.xlabel('round', font2)
plt.ylabel('value', font2)
plt.show()
plt.savefig("./pngresult/validation/{}.png".format(savename))
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20200729104606159.png)

```python
import matplotlib.pyplot as plt
import numpy as np
 
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
 
# 输入统计数据
waters = ('100', '150', '200', '250')
buy_number_male = [0.999, 0.995, 0.974, 0.905]
buy_number_female = [0.985, 0.959, 0.938, 0.903]
 
bar_width = 0.3 # 条形宽度
index_male = np.arange(len(waters)) # 男生条形图的横坐标
index_female = index_male + bar_width # 女生条形图的横坐标
 
# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=buy_number_male, width=bar_width, color='b', label='SinglePerson')
plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='MultiPerson')
 
legend = plt.legend() # 显示图例
plt.xticks(index_male + bar_width/2, waters,fontsize=20) # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Acc',fontsize=20) # 纵坐标轴标题
plt.xlabel('Distance(cm)',fontsize=20)
plt.title('Item Identification Accuracy',fontsize=20) # 图形标题

plt.yticks(fontsize=20)
#legend.get_frame().set_facecolor('#00FFCC')
#legend.get_title().set_fontsize(fontsize = 20)
# 不仅可以设置字体大小，还可以设置什么字体，因为legend.get_title()返回的是一个'Text'属性
# 的对像，时刻不要忘记Matplotlib面向对像的画图方式啊
plt.show()
```



```python
import numpy as np
import matplotlib.pyplot as plt
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False

# 刻度大小
plt.rcParams['axes.labelsize']=16
# 线的粗细
plt.rcParams['lines.linewidth']=2
# x轴标签大小
plt.rcParams['xtick.labelsize']=14
# y轴标签大小
plt.rcParams['ytick.labelsize']=14
#图例大小
plt.rcParams['legend.fontsize']=14
# 图大小
plt.rcParams['figure.figsize']=[12,8]
# 生成-5~5 的10个数 array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])
x = np.arange(-5,5)
y = [10,13,5,40,30,60,70,12,55,25] 
# 正常显示中文字体
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
# 绘图，设置(label)图例名字为'第一条线'，显示图例plt.legend()
plt.plot(x,y,label='第一条线')
# x轴标签
plt.xlabel('横坐标')
# y轴标签
plt.ylabel('纵坐标')
# 可视化图标题
plt.title('这是一个折线图')
# 显示图例
plt.legend()
# 显示图形
plt.show() 
```

### 3.2. 图例

legend 方法可接受一个 loc 关键字参数来设定图例的位置，可取值为数字或字符串：

- 0: 'best' (自适应方式)
- 1: 'upper right'
- 2: 'upper left'
- 3: 'lower left'
- 4: 'lower right'
- 5: 'right'
- 6: 'center left'
- 7: 'center right'
- 8: 'lower center'
- 9: 'upper center'
- 10: 'center'

plot 方法的关键字参数 color(或c) 用来设置线的颜色。可取值为：

颜色名称或简写

- 'b' blue(蓝色)
- 'g' green(绿色)
- 'r' red(红色)
- 'c' cyan(青色)
- 'm' magenta(品红)
- 'y' yellow(黄色)
- 'k' black(黑色)
- 'w' white(白色)
- rgb

### 3.3. 样式

plot 方法的关键字参数 linestyle(或ls) 用来设置线的样式。可取值为：

- '-', 'solid'
- '--', 'dashed'
- '-.', 'dashdot'
- ':', 'dotted'
- '', ' ', 'None'

### 3.4. 粗细

设置 plot 方法的关键字参数 linewidth(或lw) 可以改变线的粗细，其值为浮点数。

### 3.5. marker

以下关键字参数可以用来设置marker的样式：

- marker 标记类型
- markeredgecolor 或 mec 标记边界颜色
- markeredgewidth 或 mew 标记宽度
- markerfacecolor 或 mfc 标记填充色
- markersize 或 ms 标记大小

- '-' 实线
- '--' 虚线
- '-.' 点与线
- ':' 点
- '.' 点标记
- ',' 像素标记
- 'o' 圆圈标记
- 'v' 倒三角标记
- '^' 正三角标记
- '<' 左三角标记
- '>' 右三角标记
- '1' 向下Y标记
- '2' 向上Y标记
- '3' 向左Y标记
- '4' 向右Y标记
- 's' 正方形标记
- 'p' 五角星标记
- '*' *标记
- 'h' 六边形1 标记
- 'H' 六边形2 标记
- '+' +标记
- 'x' x标记
- 'D' 钻石标记
- 'd' 薄砖石标记
- '|' 垂直线标记
- '_' 水平线标记

## 4. 散点图

plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None)

- x,y 形如shape(n,)的数组，可选值
- s 代表点的大小
- c 代表点的填充颜色,点的颜色或颜色序列，默认蓝色。其它如c = 'r' (red); c = 'g' (green); c = 'k' (black) ; c = 'y'(yellow)
- marker 点的形状,可选值，默认是圆

> - '-' 实线
> - '--' 虚线
> - '-.' 点与线
> - ':' 点
> - '.' 点标记
> - ',' 像素标记
> - 'o' 圆圈标记
> - 'v' 倒三角标记
> - '^' 正三角标记
> - '<' 左三角标记
> - '>' 右三角标记
> - '1' 向下Y标记
> - '2' 向上Y标记
> - '3' 向左Y标记
> - '4' 向右Y标记
> - 's' 正方形标记
> - 'p' 五角星标记
> - '*' *标记
> - 'h' 六边形1 标记
> - 'H' 六边形2 标记
> - '+' +标记
> - 'x' x标记
> - 'D' 钻石标记
> - 'd' 薄砖石标记
> - '|' 垂直线标记
> - '_' 水平线标记

- alpha：标量，可选，默认值：无， 0（透明）和1（不透明）之间的alpha混合值
- edgecolors 点的边界颜色或颜色序列，可选值，默认值：None

## 5. 直方图

plt.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None)

> - x：指定要绘制直方图的数据；输入值，这需要一个数组或者一个序列，不需要长度相同的数组。
> - bins：指定直方图条形的个数；
> - range：指定直方图数据的上下界，默认包含绘图数据的最大值和最小值；
> - density：布尔,可选。如果"True"，返回元组的第一个元素将会将计数标准化以形成一个概率密度，也就是说，直方图下的面积（或积分）总和为1。这是通过将计数除以数字的数量来实现的观察乘以箱子的宽度而不是除以总数数量的观察。如果叠加也是“真实”的，那么柱状图被规范化为1。(替代normed)
> - weights：该参数可为每一个数据点设置权重；
> - cumulative：是否需要计算累计频数或频率；
> - bottom：可以为直方图的每个条形添加基准线，默认为0；
> - histtype：指定直方图的类型，默认为bar，除此还有’barstacked’, ‘step’, ‘stepfilled’；
> - align：设置条形边界值的对其方式，默认为mid，除此还有’left’和’right’；
> - orientation：设置直方图的摆放方向，默认为垂直方向；
> - rwidth：设置直方图条形宽度的百分比；
> - log：是否需要对绘图数据进行log变换；
> - color：设置直方图的填充色；
> - label：设置直方图的标签，可通过legend展示其图例；
> - stacked：当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放；
> - normed：是否将直方图的频数转换成频率；(弃用，被density替代)
> - alpha：透明度，浮点数。

```python
plt.rcParams['font.family']='SimHei'
plt.rcParams['font.size']=20
mu = 100
sigma = 20
x = np.random.normal(100,20,100) # 均值和标准差
# 指定分组个数
num_bins = 10
fig, ax = plt.subplots()
# 绘图并接受返回值
n, bins_limits, patches = ax.hist(x, num_bins, density=1)
# 添加分布曲线
ax.plot(bins_limits[:10],n,'--')
plt.title('直方图数据添加分布曲线')
plt.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603203349970.png)

```python
#多类型直方图
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
# 指定分组个数
n_bins=10
fig,ax=plt.subplots(figsize=(8,5))
# 分别生成10000 ， 5000 ， 2000 个值
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
# 实际绘图代码与单类型直方图差异不大，只是增加了一个图例项
# 在 ax.hist 函数中先指定图例 label 名称
ax.hist(x_multi, n_bins, histtype='bar',label=list("ABC"))
ax.set_title('多类型直方图')
# 通过 ax.legend 函数来添加图例
ax.legend()
plt.show()
```

## 6. 箱线图

plt.boxplot(x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None)

> - x：指定要绘制箱线图的数据；
> - notch：是否是凹口的形式展现箱线图，默认非凹口；
> - sym：指定异常点的形状，默认为+号显示；
> - vert：是否需要将箱线图垂直摆放，默认垂直摆放；
> - whis：指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
> - positions：指定箱线图的位置，默认为[0,1,2…]；
> - widths：指定箱线图的宽度，默认为0.5；
> - patch_artist：是否填充箱体的颜色；
> - meanline：是否用线的形式表示均值，默认用点来表示；
> - showmeans：是否显示均值，默认不显示；
> - showcaps：是否显示箱线图顶端和末端的两条线，默认显示；
> - showbox：是否显示箱线图的箱体，默认显示；
> - showfliers：是否显示异常值，默认显示；
> - boxprops：设置箱体的属性，如边框色，填充色等；
> - labels：为箱线图添加标签，类似于图例的作用；
> - filerprops：设置异常值的属性，如异常点的形状、大小、填充色等；
> - medianprops：设置中位数的属性，如线的类型、粗细等；
> - meanprops：设置均值的属性，如点的大小、颜色等；
> - capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等；
> - whiskerprops：设置须的属性，如颜色、粗细、线的类型等；

```python
import matplotlib.pyplot as plt
import numpy as np
all_data=[np.random.normal(0,std,100) for std in range(1,4)]
#首先有图（fig），然后有轴（ax）
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))
bplot1=axes[0].boxplot(all_data,
                       vert=True,
                       patch_artist=True)
bplot2 = axes[1].boxplot(all_data,
                         notch=True,
                         vert=True, 
                         patch_artist=True)
#颜色填充
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# 加水平网格线
for ax in axes:
    ax.yaxis.grid(True) #在y轴上添加网格线
    ax.set_xticks([y+1 for y in range(len(all_data))] ) #指定x轴的轴刻度个数
    ## [y+1 for y in range(len(all_data))]运行结果是[1,2,3]
    ax.set_xlabel('xlabel') #设置x轴名称
    ax.set_ylabel('ylabel') #设置y轴名称
# 添加刻度
# 添加刻度名称，我们需要使用 plt.setp() 函数：
# 加刻度名称
plt.setp(axes, xticks=[1,2,3],
         xticklabels=['x1', 'x2', 'x3'])
# 我们的刻度数是哪些，以及我们想要它添加的刻度标签是什么。    
plt.show()
```

```python
import matplotlib.pyplot as plt
# 在图上去除离群值
fig,axes=plt.subplots(1,2,figsize=(9,5))
axes[0].boxplot(data,labels=labels,patch_artist=True)
axes[0].set_title("默认 showfliers=True",fontsize=15)
axes[1].boxplot(data,labels=labels,patch_artist=True,showfliers=False)
axes[1].set_title("showfliers=False",fontsize=15)
plt.show()
# 默认 showfliers=True，那么图中会显示出离群值
# showfliers=False，那么图中会去除离群值
```

## 7. 饼图

plt.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None, radius=None, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False)

> - x：指定绘图的数据；
> - explode：指定饼图某些部分的突出显示，即呈现爆炸式；
> - labels：为饼图添加标签说明，类似于图例说明；
> - colors：指定饼图的填充色；
> - autopct：自动添加百分比显示，可以采用格式化的方法显示；
> - pctdistance：设置百分比标签与圆心的距离；
> - shadow：是否添加饼图的阴影效果；
> - labeldistance：设置各扇形标签（图例）与圆心的距离；
> - startangle：设置饼图的初始摆放角度；
> - radius：设置饼图的半径大小；
> - counterclock：是否让饼图按逆时针顺序呈现；
> - wedgeprops：设置饼图内外边界的属性，如边界线的粗细、颜色等；
> - textprops：设置饼图中文本的属性，如字体大小、颜色等；
> - center：指定饼图的中心点位置，默认为原点
> - frame：是否要显示饼图背后的图框，如果设置为True的话，需要同时控制图框x轴、y轴的范围和饼图的中心位置；

```python
#定义饼状图的标签，标签是列表
labels =[ 'A','B','C','D']
#每个标签占多大，会自动去算百分比
x = [15,30,45,10]
# 绘制饼图,autopct='%.0f%%' 显示百分比
# textprops = {'fontsize':30, 'color':'k'} 大小为30，颜色为黑色
plt.pie(x,labels=labels,autopct='%.0f%%', textprops = {'fontsize':30, 'color':'k'})
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603210904775.png)

```python
#定义饼状图的标签，标签是列表
labels =[ 'A','B','C','D']
#每个标签占多大，会自动去算百分比
x = [15,30,45,10]
#0.1表示将B那一块凸显出来
explode = (0,0.1,0,0) 
# 绘制饼图,autopct='%.0f%%' 显示百分比
# textprops = {'fontsize':30, 'color':'k'} 大小为30，颜色为黑色
# explode=explode 将B那一块凸显出来
# shadow=True 显示阴影
plt.pie(x,labels=labels,autopct='%.0f%%', textprops = {'fontsize':30, 'color':'k'},explode=explode,shadow=True)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.show()
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603210952182.png)

```python
import matplotlib.pyplot as plt
# 设置绘图的主题风格
plt.style.use('ggplot')
# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.figsize']=[12,12]
# 构造数据
x = [0.2515,0.3724,0.3336,0.0368,0.0057]
# 提示标签
labels = ['中专','大专','本科','硕士','其他']
# 用于突出显示大专学历人群
explode = [0,0.1,0,0,0]
# 自定义颜色
colors=['#9F79EE','#4876FF','#EE9A00','#EE4000','#FFD700']
# 将横、纵坐标轴标准化处理，保证饼图是一个正圆，否则为椭圆
plt.axes(aspect='equal')
# 控制x轴和y轴的范围
plt.xlim(0,4)
plt.ylim(0,4)
# 绘图数据
plt.pie(x, # 绘图数据
        explode=explode, # 突出显示大专人群
        autopct='%1.1f%%', # 设置百分比的格式，这里保留一位小数
        pctdistance=0.6, # 设置百分比标签与圆心的距离
        labeldistance=1.2, # 设置教育水平标签与圆心的距离
        startangle = 180, # 设置饼图的初始角度
        radius = 1.5, # 设置饼图的半径
        counterclock = False,
        wedgeprops = {'linewidth': 1.5, 'edgecolor':'green'}, # 设置饼图内外边界的属性值
        textprops = {'fontsize':16, 'color':'k'}, # 设置文本标签的属性值
        center = (2,2), # 设置饼图的原点
        frame = 1, # 是否显示饼图的图框，这里设置显示
        labels=labels, # 添加教育水平标签
        colors=colors # 设置饼图的自定义填充色
       )
# 删除x轴和y轴的刻度
plt.xticks(())
plt.yticks(())
# 添加图标题
plt.title('芝麻信用失信用户教育水平分布')
plt.show()
```

<img src="https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200603211114601.png" alt="image-20200603211114601" style="zoom:50%;" />

- 颜色表：https://blog.csdn.net/CD_Don/article/details/88070453
- 图例位置：

| 位置     | String       | Number |
| -------- | ------------ | ------ |
| 右上     | upper right  | 1      |
| 左上     | upper left   | 2      |
| 左下     | lower left   | 3      |
| 右下     | lower right  | 4      |
| 正右     | right        | 5      |
| 中央偏左 | center left  | 6      |
| 中央偏右 | center right | 7      |
| 中央偏下 | lower center | 8      |
| 中央偏上 | upper center | 9      |
| 正中央   | center       | 10     |

## 8. 动态绘制

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
style.use("ggplot")
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(i):
    pullData = open("twitter-out.txt","r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines[-200:]:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

        xar.append(x)
        yar.append(y)
        
    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
```

## 9. 多图绘制

> 通过以保存文件方式，在vscode查看图片可以实时显示；或者通过plt.pause(0.1)； plt.clf()

```python
 def plotscatterVertical(self,kinectdata,rfiddata):
        plt.clf()
        #plt.cla()
        fig = plt.figure(num=1, figsize=(360,360),dpi=80)
        #plt.on()
        rfdikeys=[key for key in rfiddata.keys()]
        maxvalue=0
        minvalue=0
        for key, value in kinectdata.items():
            #print(kinectdata["kinect"])
            ax1 = plt.subplot(2+len(rfdikeys),3,2)
            #在第一个子区域中绘图
             g=plt.scatter(value[:,0],value[:,1],marker="v",s=50,color="r")
            #选中第二个子区域，并绘图
            ax2 = plt.subplot(2+len(rfdikeys),1,2)
            ax2.set_title(key+"phase", fontsize=18, fontweight='bold', x=0.5, y=0.001, bbox=dict(facecolor='white', alpha=0.5)) # 设置标题
            time, phase=self.getkinectPhase(value)
            maxvalue=time[-1]
            minvalue=time[0]
            kp=plt.scatter(time,phase,marker="v",s=50,color="r")
        rfdikeys=[key for key in rfiddata.keys()]
        for i in range(0,len(rfdikeys)):
            ax=fig.add_subplot(2+len(rfdikeys),1,i+3)
            ax.set_title(rfdikeys[i], fontsize=18, fontweight='bold', x=0.5, y=0.001, bbox=dict(facecolor='white', alpha=0.5)) # 设置标题
            #plt.axis('off') # 去掉每个子图的坐标轴
            plt.scatter(rfiddata[rfdikeys[i]][:,1],rfiddata[rfdikeys[i]][:,0],marker="*",s=50,color="r")
            plt.xlim(minvalue,maxvalue)
        plt.savefig("./data/1.png")
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201210110138677.png)

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()



classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210529112951199.png)

## 10. 绘制混淆矩阵

```python

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size=None):
    """ (Copied from sklearn website)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Display normalized confusion matrix ...")
    else:
        print('Display confusion matrix without normalization ...')

    # print(cm)

    fig, ax = plt.subplots()
    if size is None:
        size = (12, 8)
    fig.set_size_inches(size[0], size[1])

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim([-0.5, len(classes)-0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm
```

###  Resource

- [25中matplotlib图片](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247493676&idx=7&sn=f98ffc14cc046a86c3580d33ff2709e9&chksm=ebb7d0f8dcc059ee1d44f6f989cc0e23ffb60ef7a2cbc5a7c935702c650420c00013e7d00d75&scene=0&xtrack=1#rd)


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/matplotdraw-instance-record/  

