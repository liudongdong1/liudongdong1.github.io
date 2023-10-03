# Mathbasic

> **每一个概念**，**被定义**就是为了去**解决一个实际问题（问Why&What），接着寻找解决问题的方法（问How）**，这个“方法”在计算机领域被称为“算法”（非常多的人在研究）。我们无法真正衡量到底是提出问题重要，还是解决问题重要，但我们可以从不同的**解决问题的角度**来思考问题。一方面，**重复**以加深印象。另一方面，具有**多角度的视野**，能让我们获得更多的灵感，真正做到**链接并健壮自己的知识图谱**。**学习知识的最快捷径是带有思考的重复**，但那是**带思考的重复**，有一些**直观的方法**在帮助你理解和记忆上比**做题**有效率的多。

### 1. 正则化

- 损失函数：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015233935401.png)

> E(w)是**损失函数（又称误差函数）**，`E`即Evaluate，有时候写成`L`即Loss
> tn 是测试集的真实输出，又称目标变量【对应第一幅图中的蓝色点】
> w 是权重（需要训练的部分，未知数）
> ϕ()是**基函数**，例如多项式函数，核函数
> 测试样本有`n`个数据
> 整个函数直观解释就是**误差方差和**，

- 损失函数+正则化

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015234054348.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015234157084.png)

> λ被称为正则化系数，**越大，这个限制越强**

- 一般正则化

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015234342609.png)

> 另 $q=2;\quad X=\{x_1,x_2\}; W=\{w_1,w_2\}; q=[0.5:4]$; 则横轴是w1, 纵坐标是w2; z轴代表 正则项的值；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015234817880.png)

![image-20201015234838468](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015234838468.png)

> 蓝色的圆圈表示没有经过限制的**损失函数在寻找最小值过程**中，w的不断迭代（随最小二乘法，最终目的还是使损失函数最小）变化情况，**表示的方法是等高线，z轴的值就是** E(w)；目标函数（误差函数）就是**求蓝圈+红圈的和的最小值**，而这个值通在很多情况下是**两个曲面相交的地方**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015235032537.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201015235529378.png)

### 2. 线性代数

- 仿射变换（Affine Transformation）![仿射变换（Affine Transformation）](https://gitee.com/github-25970295/blogImage/raw/master/img/仿射变换（Affine Transformation）.gif)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016003001811.png)

![36-Matrix1](https://gitee.com/github-25970295/blogImage/raw/master/img/36-Matrix1.gif)

> Step1：i绿色i（x轴）进行移动（变换）
> Step2：红色 j（y轴）进行移动（变换）
> Step3：目标向量x轴**坐标值**与i **变换后向量**进行**向量乘法**
> Step4：目标向量y轴**坐标值**与j**变换后向量**进行**向量乘法**
> Step5：两者进行向量j加法，得到**线性变换结果**

- 矩阵乘法

![42-Cal](https://gitee.com/github-25970295/blogImage/raw/master/img/42-Cal.gif)

- 计算行列式

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016003723928.png)

- 线性方程组：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016003850373.png)

![62-xv](https://gitee.com/github-25970295/blogImage/raw/master/img/62-xv.gif)

- 零空间：  变换后**落在原点的向量**的集合，称为**这个矩阵**（再次强调矩阵 = 变换的数字表达）的**零空间或核**

> 【图1】二维压缩到一个直线（一维），有一条直线（一维）的点被压缩到原点
> 【图2】三维压缩到一个面（二维），有一条直线（一维）的点被压缩到原点
> 【图3】三维压缩到一条线（一维），有一条直线（二维）的点被压缩到原点
>
> 【注意】压缩就是变换，变换就是矩阵，其实说的就是矩阵满秩 = 列空间 + 零空间。

![64-NullSpace](https://gitee.com/github-25970295/blogImage/raw/master/img/64-NullSpace.gif)

> `n*m` 的几何意义是将**m维空间（输入空间）映射到n维空间（输出空间）**上；  矩阵乘法从右向左读；

- 点积：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016004601739.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016004623382.png)

![74-DualityAll](https://gitee.com/github-25970295/blogImage/raw/master/img/74-DualityAll.gif)

- 叉积：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016005152810.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016005114480.png)

![84-Volum](https://gitee.com/github-25970295/blogImage/raw/master/img/84-Volum.gif)

- 求特征值

![image-20201016005903476](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016005903476.png)

![102-Lambda](https://gitee.com/github-25970295/blogImage/raw/master/img/102-Lambda.gif)

- 矩阵与函数

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016010052045.png)

![image-20201016010104592](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016010104592.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201016010128833.png)

### 3. 几种基本函数

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027144127893.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027144143984.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027144203831.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027144216739.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027144235857.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201027144254996.png)



### 4. 高斯函数

**正态分布**的[概率密度函数](http://zh.wikipedia.org/wiki/概率密度函数)均值为μ [方差](http://zh.wikipedia.org/wiki/方差)为σ2 (或[标准差](http://zh.wikipedia.org/wiki/標準差)σ)是[高斯函数](http://zh.wikipedia.org/wiki/高斯函數)的一个实例：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201106152113676.png)

**累积分布函数**： [累积分布函数](http://zh.wikipedia.org/wiki/累积分布函数)是指随机变量*X*小于或等于*x*的概率

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201106152245618.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201106153226099.png)

```python
def average(data):
    return np.sum(data)/len(data)
#标准差
def sigmaHandle(data,avg):
    sigma_squ=np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.5)
#高斯分布概率
def prob(data,avg,sig):
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((data-avg),2))
    return coef*(np.exp(mypow))
#高斯连续分布
def curricularProb(data,avg,sig):
    gauss = norm(loc=avg, scale=sig)  # loc: mean 均值， scale: standard deviation 标准差
    return gauss.cdf(data)
#数据归一化处理
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

#使用高斯连续分布函数 
def getColor(data):
    mean=average(data)
    sig=sigmaHandle(data,mean)
    gauss=curricularProb(data,mean,sig)
    print(np.where(gauss==np.min(gauss)),np.min(gauss),np.where(gauss==np.max(gauss)),np.max(gauss))
    scattor(data,gauss)
    return gauss*data
#绘制散点图
def scattor(data,data1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i in range(len(data)):
        ax.scatter(data[i], data1[i], s=100)
    plt.show()
 #pdf：连续随机分布的概率密度函数
 #pmf：离散随机分布的概率密度函数
 #cdf：累计分布函数
 #ppf: 百分位函数（累计分布函数的逆函数）
 #lsf: 生存函数的逆函数（1 - cdf 的逆函数）
```

### 5. numpy 数学函数

> `副本是一个数据的完整的拷贝`，如果我们对副本进行修改，它不会影响到原始数据，物理内存不在同一位置。`视图是数据的一个别称或引用`，通过该别称或引用亦便可访问、操作原有数据，但原有数据不会产生拷贝。如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置

视图一般发生在：

1. numpy 的切片操作返回原数据的视图。
2. 调用 ndarray 的 view() 函数产生一个视图。

副本一般发生在：

1. Python 序列的切片操作，调用deepCopy()函数。
2. 调用 ndarray 的 copy() 函数产生一个副本。

```python
a = np.array([[1,  2],  [3,  4]])  #  从数组初始化
numpy.arange(start, stop, step, dtype) # 从arrange 创建

numpy.zeros(shape, dtype = float, order = 'C')
numpy.ones(shape, dtype = None, order = 'C')
numpy.append(arr, values, axis=None)

#---- 数学计算
np.sqrt(num)
np.square(num)
np.exp(num)  #e的x 次方
np.log(num)  # log函数
np.sign(num)  #正负判断
numpy.ceil() #返回大于或者等于指定表达式的最小整数，即向上取整。
numpy.floor() #返回小于或者等于指定表达式的最大整数，即向下取整。
numpy.around() #函数返回指定数字的四舍五入值。
np.sin(a*np.pi/180)#  sin(), cos(), tan()
numpy.mod(a,b) #计算输入数组中相应元素的相除后的余数。
np.power(a,b)  #计算指数
numpy.reciprocal() #函数返回参数逐元素的倒数
numpy.absolute() # 计算绝对值
np.add(a,b)  #add(), subtract(), multiply(), divide()
#---  统计相关
numpy.amin() #用于计算数组中的元素沿指定轴的最小值。  1: 按行计算；  0： 按列计算；
numpy.amax() #用于计算数组中的元素沿指定轴的最大值。
numpy.ptp()  #函数计算数组中元素最大值与最小值的差（最大值 - 最小值）。
numpy.median() #函数用于计算数组 a 中元素的中位数（中值）
numpy.mean() #函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
numpy.average() #函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。
np.std([1,2,3,4]) #计算标准差 sqrt(mean((x - x.mean())**2))
np.var([1,2,3,4])  #计算方差
#---  排序相关
numpy.sort(a, axis, kind, order)#  kind=quicksort, mergesort, heapsort; axis=0: 按列排序，1 行；order: 排序字段
numpy.argsort() #函数返回的是数组值从小到大的索引值。
numpy.nonzero() #函数返回输入数组中非零元素的索引。
y = np.where(x >  3)   #函数返回输入数组中满足给定条件的元素的索引。
np.extract(condition = np.mod(x,2)  ==  0  , x) #函数根据某个条件从数组中抽取元素，返回满条件的元素。
#---  绘图相关
plt.hist(a, bins =  [0,20,40,60,80,100]) #  绘制数据分布条形图
```

| ndarray.ndim  | 秩，即轴的数量或维度的数量                  |
| ------------- | ------------------------------------------- |
| ndarray.shape | 数组的维度，对于矩阵，n 行 m 列             |
| ndarray.size  | 数组元素的总个数，相当于 .shape 中 n*m 的值 |
| ndarray.dtype | ndarray 对象的元素类型                      |

- 广播
  -  a 和 b 形状相同，即满足 **a.shape == b.shape**，那么 a*b 的结果就是 a 与 b 数组对应位相乘。
  - 运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201115192633403.png)

| `reshape` | 不改变数据的条件下修改形状                           |
| --------- | ---------------------------------------------------- |
| `flat`    | 数组`元素迭代器`                                     |
| `flatten` | 返回一份数组拷贝，对`拷贝所做的修改不会影响原始数组` |
| `ravel`   | 返回展开数组                                         |

| `oncatenate` | 连接沿现有轴的数组序列           |
| ------------ | -------------------------------- |
| `stack`      | 沿着`新的轴`加入一系`列`数组。   |
| `hstack`     | `水平堆叠序列中的数组（列方向）` |
| `vstack`     | `竖直堆叠序列中的数组（行方向）` |

| `dot`         | 两个数组的点积，即元素对应相乘。 |
| ------------- | -------------------------------- |
| `vdot`        | 两个向量的点积                   |
| `inner`       | 两个数组的内积                   |
| `matmul`      | 两个数组的矩阵积                 |
| `determinant` | 数组的行列式                     |
| `solve`       | 求解线性矩阵方程                 |
| `inv`         | 计算矩阵的乘法逆矩阵             |

- https://www.runoob.com/numpy/numpy-array-manipulation.html
- https://blog.csdn.net/claroja/article/details/71081369  numpy 基本计算

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mathbasic/  

