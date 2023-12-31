# DescriptionFeature


> **统计学提供**的是一套有关**数据**收集、处理、**分析**、解释并从数据中得出结论的**方法**；**数据分析**则是选择适当的统计方法研究数据，并从数据中提取有用信息进而得出结论。描述性分析用于`描述定量数据的整体情况`，例如研究消费者对于某商品的购买意愿情况，可用到描述性分析对样本的年龄、收入、消费水平等各指标进行初步分析，以了解掌握消费者总体的特征情况。

### 1.概念

#### .1. 数据类型

> **分类数据和顺序数据**说明的是事物的品质特征，通常用文字来表述的结果表现的都是类别，都可称为**定性数据或者品质数据**（qualitative data）;
>
> - `分类数据`我们通常计算各组的`频数、众数、异众比率`，进行`列联表分析和卡方检验`等；
> - `顺序数据`可计算`中位数和四分位差以及等级相关系数`等
>
> **数值型数据**说明的是现象的数量特征，通常是用数值来表现的，因此称为**定量数据或数量数据(**quantitative data)。
>
> - `数值型数据`可计算更多`统计量、进行参数估计和假设检`验等

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526102351902.png)

### 2. 描述指标

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526101810039.png)

> - `最大值、最小值`可用来检验数据是否存在`异常情况`。
> - `平均值、中位数`是用于描述数据的`集中趋势指标`。
> - `标准差`是用于描述数据的离散趋势指标。如果比较单位不同（或`数值相差`太大）的两组数据时，采用`变异系数`比较`离散程度`。
> - `峰度和偏度`通常用于判断数据`正态性情况`，峰度的绝对值越大，说明数据越陡峭，峰度的绝对值大于3，意味着数据严重不正态。同时偏度的绝对值越大，说明数据偏斜程度越高，偏度的绝对值大于3，意味着严重不正态（可通过正态图查看数据正态性情况）。

#### .1. 位置度量

> 位置的度量是用来`描述数据的集中趋势的统计量`。常用的有：`均值、众数、中位数、百分位数`等。

#### 2. 分散程度度量

> 表示`数据分散（或变异）程度的数字特征`有`方差、标准差、极差、四分卫极差和标准误`等。

#### .3. 分布形状度量

- **峰度系数**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526103135923.png)

- **偏度系数**

> 当数据的总体分布为正态分布时，峰度系数近似为0；当分布较正态分布的尾部更分散时，峰度系数为正；否则为负。当峰度系数为正时，两侧极端数据较多；当峰度系数为负时，两侧极端数据较少。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526103156238.png)

### 3. 数据分布

#### .1. 分布函数

> 正态`分布函数`dnorm(x, mean = 0, sd = 1, log = FALSE)
>
> 正态`概率密度函数`pnorm(q, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
>
> 正态分布`下分位点函数`qnorm(p, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
>
> `生成正态分布随机数函数`rnorm(n, mean = 0, sd = 1)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526103453568.png)

#### .2. 直方图&经验分布&QQ图

- 直方图&核密度估计函数

> 对于数据分布，常用直方图（histgram）进行描述。将数据取值的范围分成若干区间（一般是等间隔的），在等间隔的情况下，每个区间长度称为组距。考察数据落入每一区间的频数与频率，在每个区间上画一个矩形，它的宽度是组距，高度可以是频数、频率或频率/组距.
>
> 核密度估计函数density()，其目的是用已知样本，估计其密度。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526103738012.png)

- **经验分布**

> 直方图的制作`适合于总体为连续型分布的场合`。若对于`一般的总体分布`，若`要它的总体分布函数F(x)，可以用经验分布函数`。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526103909986.png)

- **QQ图**

> QQ图可以帮助我们~`鉴别样本的分布是否近似于某种类型的分布`~。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526104026930.png)

#### .3. 箱型图&五数

- **箱型图**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526104053656.png)

> 中位数，下四分位数、上四分位数、最小值、最大值。这五个数称为样本数据的`五数总结`。

#### .4. 正态检验&分布拟合

> 其中所配曲线是否合适，是需要进行统计检验的。

- **正态W检验方法**
- **经验分布K-S检验**

### 4. numpy code

```python
from numpy import array
from numpy.random import normal, randint
#使用List来创造一组数据
data = [1, 2, 3]
#使用ndarray来创造一组数据
data = array([1, 2, 3])
#创造一组服从正态分布的定量数据
data = normal(0, 10, size=10)
#创造一组服从均匀分布的定性数据
data = randint(0, 10, size=10)

from numpy import mean, median
#计算均值
mean(data)
#计算中位数
median(data)
from scipy.stats import mode
#计算众数
mode(data)

from numpy import mean, ptp, var, std
#极差
ptp(data)
#方差
var(data)
#标准差
std(data)
#变异系数
mean(data) / std(data)

from numpy import mean, std
 #计算第一个值的z-分数
(data[0]-mean(data)) / std(data)

from numpy import array, cov, corrcoef
data = array([data1, data2])
#计算两组数的协方差
#参数bias=1表示结果需要除以N，否则只计算了分子部分
#返回结果为矩阵，第i行第j列的数据表示第i组数与第j组数的协方差。对角线为方差
cov(data, bias=1)
#计算两组数的相关系数
#返回结果为矩阵，第i行第j列的数据表示第i组数与第j组数的相关系数。对角线为1
corrcoef(data)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210526104851021.png)

### 5. matlab 图分析案例

> 使用图分析可以更加直观地展示数据的`分布（频数分析）和关系（关系分析）`。`柱状图和饼形图`是对`定性数据`进行`频数分析`的常用工具，使用前需将每一类的频数计算出来。`直方图和累积曲线`是对`定量数据`进行`频数分析`的常用工具，`直方图对应密度函数而累积曲线对应分布函数`。`散点图`可用来对两组数据的`关系进行描述`。在没有分析目标时，需要对数据进行探索性的分析，箱形图将帮助我们完成这一任务。

- 数据生成

```python
from numpy import array
from numpy.random import normal

def genData():
    heights = []
    weights = []
    grades = []
    N = 10000

    for i in range(N):
        while True:
            #身高服从均值172，标准差为6的正态分布
            height = normal(172, 6)
            if 0 < height: break
        while True:
            #体重由身高作为自变量的线性回归模型产生，误差服从标准正态分布
            weight = (height - 80) * 0.7 + normal(0, 1)
            if 0 < weight: break
        while True:
            #分数服从均值为70，标准差为15的正态分布
            score = normal(70, 15)
            if 0 <= score and score <= 100:
                grade = 'E' if score < 60 else ('D' if score < 70 else ('C' if score < 80 else ('B' if score < 90 else 'A')))
                break
        heights.append(height)
        weights.append(weight)
        grades.append(grade)
    return array(heights), array(weights), array(grades)

heights, weights, grades = genData()
```

#### .1. 频数分析

##### .1.  定性分析（柱状图、饼形图）

```python
from matplotlib import pyplot

#绘制柱状图
def drawBar(grades):
    xticks = ['A', 'B', 'C', 'D', 'E']
    gradeGroup = {}
    #对每一类成绩进行频数统计
    for grade in grades:
        gradeGroup[grade] = gradeGroup.get(grade, 0) + 1
    #创建柱状图
    #第一个参数为柱的横坐标
    #第二个参数为柱的高度
    #参数align为柱的对齐方式，以第一个参数为参考标准
    pyplot.bar(range(5), [gradeGroup.get(xtick, 0) for xtick in xticks], align='center')

    #设置柱的文字说明
    #第一个参数为文字说明的横坐标
    #第二个参数为文字说明的内容
    pyplot.xticks(range(5), xticks)

    #设置横坐标的文字说明
    pyplot.xlabel('Grade')
    #设置纵坐标的文字说明
    pyplot.ylabel('Frequency')
    #设置标题
    pyplot.title('Grades Of Male Students')
    #绘图
    pyplot.show()

drawBar(grades)
```

```python
from matplotlib import pyplot

#绘制饼形图
def drawPie(grades):
    labels = ['A', 'B', 'C', 'D', 'E']
    gradeGroup = {}
    for grade in grades:
        gradeGroup[grade] = gradeGroup.get(grade, 0) + 1
    #创建饼形图
    #第一个参数为扇形的面积
    #labels参数为扇形的说明文字
    #autopct参数为扇形占比的显示格式
    pyplot.pie([gradeGroup.get(label, 0) for label in labels], labels=labels, autopct='%1.1f%%')
    pyplot.title('Grades Of Male Students')
    pyplot.show()

drawPie(grades)
```

##### .2. 定量分析（直方图、累积曲线）

```python
from matplotlib import pyplot

#绘制直方图
def drawHist(heights):
    #创建直方图
    #第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    #第二个参数为划分的区间个数
    pyplot.hist(heights, 100)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Heights Of Male Students')
    pyplot.show()

drawHist(heights)
```

```python
from matplotlib import pyplot

#绘制累积曲线
def drawCumulativeHist(heights):
    #创建累积曲线
    #第一个参数为待绘制的定量数据
    #第二个参数为划分的区间个数
    #normed参数为是否无量纲化
    #histtype参数为'step'，绘制阶梯状的曲线
    #cumulative参数为是否累积
    pyplot.hist(heights, 20, normed=True, histtype='step', cumulative=True)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Heights Of Male Students')
    pyplot.show()

drawCumulativeHist(heights)
```

#### .3. 关系分析

- 散点图

```python
from matplotlib import pyplot

#绘制散点图
def drawScatter(heights, weights):
    #创建散点图
    #第一个参数为点的横坐标
    #第二个参数为点的纵坐标
    pyplot.scatter(heights, weights)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Weights')
    pyplot.title('Heights & Weights Of Male Students')
    pyplot.show()

drawScatter(heights, weights)
```

#### .4. 探索分析

> 在不明确数据分析的目标时，我们对数据进行一些探索性的分析，通过我们可以知道数据的`中心位置，发散程度以及偏差程度`。

```python
from matplotlib import pyplot

#绘制箱形图
def drawBox(heights):
    #创建箱形图
    #第一个参数为待绘制的定量数据
    #第二个参数为数据的文字说明
    pyplot.boxplot([heights], labels=['Heights'])
    pyplot.title('Heights Of Male Students')
    pyplot.show()

drawBox(heights)
```

### Resource

- https://www.cnblogs.com/jasonfreak/p/5441512.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/descriptionfeature/  

