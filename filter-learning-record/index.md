# Filter Learning Record


### 1.常见滤波算法

#### 1.1. 限幅滤波法（又称程序判断滤波法）

　　A方法： 根据经验判断，确定两次采样允许的最大偏差值（设为A），每次检测到新值时判断： 如果本次值与上次值之差<=A，则本次值有效，如果本次值与上次值之差>A，则本次值无效，放弃本次值，用上次值代替本次值。

　　B优点： 能有效克服因偶然因素引起的脉冲干扰。

　　C缺点： 无法抑制那种周期性的干扰，平滑度差。

#### 1.2. 中位值滤波法

　　A方法： 连续采样N次（N取奇数），把N次采样值按大小排列，取中间值为本次有效值。

　　B优点： 能有效克服因偶然因素引起的波动干扰，<font color=red>对温度、液位的变化缓慢的被测参数有良好的滤波效果</font>。

　　C缺点： <font color=red>对流量、速度等快速变化的参数不宜。</font>

#### 1.3. 算术平均滤波法

　　A方法： 连续取N个采样值进行算术平均运算，N值较大时：信号平滑度较高，但灵敏度较低；N值较小时：信号平滑度较低，但灵敏度较高。N值的选取：一般流量，N=12；压力：N=4。

　　B优点： 适用于对一般具有随机干扰的信号进行滤波，<font color=red>这样信号的特点是有一个平均值，信号在某一数值范围附近上下波动。</font>

　　C缺点： <font color=red>对于测量速度较慢或要求数据计算速度较快的实时控制不适用，比较浪费RAM 。</font>

#### 1.4. 递推平均滤波法（又称滑动平均滤波法)

　　A方法： 把连续取N个采样值看成一个队列，队列的长度固定为N，每次采样到一个新数据放入队尾，并扔掉原来队首的一次数据(先进先出原则) 。把队列中的N个数据进行算术平均运算，就可获得新的滤波结果。N值的选取：流量，N=12；压力：N=4；液面，N=4~12；温度，N=1~4。

　　B优点： 对周期性干扰有良好的抑制作用，平滑度高，适用于`高频振荡`的系统。

　　C缺点： 灵敏度低，对偶然出现的脉冲性干扰的抑制作用较差，不易消除由于脉冲干扰所引起的采样值偏差，不适用于脉冲干扰比较严重的场合，比较浪费RAM。

```python
class MovAvg(object):
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.data_queue = []
        self.sum=0

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            self.sum=self.sum-self.data_queue[0]
            del self.data_queue[0]
        self.data_queue.append(data)
        self.sum=self.sum+data
        return self.sum/len(self.data_queue)
```

#### 1.5. 中位值平均滤波法（又称防脉冲干扰平均滤波法）

　　A方法： 相当于“中位值滤波法”+“算术平均滤波法”，连续采样N个数据，去掉一个最大值和一个最小值，然后计算N-2个数据的算术平均值。N值的选取：3~14。

　　B优点： 融合了两种滤波法的优点，对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差。

　　C缺点： 测量速度较慢，和算术平均滤波法一样，比较浪费RAM。

#### 1.6. 限幅平均滤波法

　　A方法： 相当于“限幅滤波法”+“递推平均滤波法”，每次采样到的新数据先进行限幅处理，再送入队列进行递推平均滤波处理。

　　B优点： 融合了两种滤波法的优点，对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差。

　　C缺点： 比较浪费RAM 。

#### 1.7. 加权递推平均滤波法

　　A方法： 是对递推平均滤波法的改进，即不同时刻的数据加以不同的权，通常是，越接近现时刻的资料，权取得越大，给予新采样值的权系数越大，则灵敏度越高，但信号平滑度越低。

　　B优点： 适用于有较大纯滞后时间常数的对象和采样周期较短的系统。

　　C缺点： 对于纯滞后时间常数较小，采样周期较长，变化缓慢的信号，不能迅速反应系统当前所受干扰的严重程度，滤波效果差。

#### 1.8. 消抖滤波法

　　A方法： 设置一个滤波计数器，将每次采样值与当前有效值比较： 如果采样值＝当前有效值，则计数器清零。如果采样值<>当前有效值，则计数器+1，并判断计数器是否>=上限N(溢出)，如果计数器溢出，则将本次值替换当前有效值，并清计数器。

　　B优点： 对于变化缓慢的被测参数有较好的滤波效果，可避免在临界值附近控制器的反复开/关跳动或显示器上数值抖动。

　　C缺点： 对于快速变化的参数不宜，如果在计数器溢出的那一次采样到的值恰好是干扰值，则会将干扰值当作有效值导入系统。

#### 1.9. [高低通滤波](https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py)

> **scipy.signal.filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)**
>
> 输入参数：
>
> b: 滤波器的分子系数向量
>
> a: 滤波器的分母系数向量
>
> x: 要过滤的数据数组。（array型）
>
> axis: 指定要过滤的数据数组x的轴
>
> padtype: 必须是“奇数”、“偶数”、“常数”或“无”。这决定了用于过滤器应用的填充信号的扩展类型。{‘odd', ‘even', ‘constant', None}
>
> padlen：在应用滤波器之前在轴两端延伸X的元素数目。此值必须小于要滤波元素个数- 1。（int型或None）
>
> method：确定处理信号边缘的方法。当method为“pad”时，填充信号；填充类型padtype和padlen决定，irlen被忽略。当method为“gust”时，使用古斯塔夫森方法，而忽略padtype和padlen。{“pad” ，“gust”}
>
> irlen：当method为“gust”时，irlen指定滤波器的脉冲响应的长度。如果irlen是None，则脉冲响应的任何部分都被忽略。对于长信号，指定irlen可以显著改善滤波器的性能。（int型或None）

> **scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')**
>
> 输入参数：
>
> N:滤波器的阶数
>
> Wn：归一化截止频率。计算公式`Wn=2*截止频率/采样频率`。（注意：根据采样定理，采样频率要大于两倍的信号本身最大的频率，才能还原信号。截止频率一定小于信号本身最大的频率，所以Wn一定在0和1之间）。当构造带通滤波器或者带阻滤波器时，Wn为长度为2的列表。
>
> btype : 滤波器类型{‘lowpass', ‘highpass', ‘bandpass', ‘bandstop'},
>
> output : 输出类型{‘ba', ‘zpk', ‘sos'},
>
> 输出参数：
>
> b，a: IIR滤波器的分子（b）和分母（a）多项式系数向量。output='ba'
>
> z,p,k: IIR滤波器传递函数的零点、极点和系统增益. output= 'zpk'
>
> sos: IIR滤波器的二阶截面表示。output= 'sos'

- 高通滤波

```python
#这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下频率成分，即截至频率为10hz，则wn=2*10/1000=0.02
from scipy import signal
b, a = signal.butter(8, 0.02, 'highpass')
filtedData = signal.filtfilt(b, a, data)#data为要过滤的信号
```

- 低通滤波

```python
#这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以上频率成分，即截至频率为10hz，则wn=2*10/1000=0.02
from scipy import signal
b, a = signal.butter(8, 0.02, 'lowpass') 
filtedData = signal.filtfilt(b, a, data)    #data为要过滤的信号
```

- 带通滤波

```python
#这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.8。Wn=[0.02,0.8]
from scipy import signal
b, a = signal.butter(8, [0.02,0.8], 'bandpass')
filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
```

```python
>>> b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
>>> np.random.seed(123456)
>>> n = 60
>>> sig = np.random.randn(n)**3 + 3*np.random.randn(n).cumsum()
>>> fgust = signal.filtfilt(b, a, sig, method="gust")
>>> fpad = signal.filtfilt(b, a, sig, padlen=50)
>>> plt.plot(sig, 'k-', label='input')
>>> plt.plot(fgust, 'b-', linewidth=4, label='gust')
>>> plt.plot(fpad, 'c-', linewidth=1.5, label='pad')
>>> plt.legend(loc='best')
>>> plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210411214829941.png)

### 2. 频谱\功率谱\倒频谱

> [时间作为参照来观察动态世界的方法我们称其为时域分析，如果另一种方法来观察世界的话，你会发现世界是永恒不变的。（假定是周期的）](https://zhuanlan.zhihu.com/p/19759362)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200610225217869.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200610225327148.png)

>  通常一个信号波形变换到频域后，有的看到有的频率式连续分布在频率轴上的（频率连续的），有的是在频率轴上只出现个别的的频率点（频率离散的）；频率若是离散的则时域是周期的（有规律的波形）；若频域是连续的则时域是非周期的（无规律的波形）；剩下的时域连续和离散只是信号从模拟信号（连续的）采样量化到被计算机处理（离散的）的过程，可以从自己具体的信号来源去选择。

#### 2.1. 能量谱 

> 能量谱也叫能量谱密度，能量谱密度描述了信号或时间序列的能量如何随频率分布。能量谱是原信号傅立叶变换的平方。

能量信号与功率信号：对于信号 $f(t)$, 其能量:
$$
E=lim_{T->\infty}\int_{-T}^{T}|f(t)|^2dt
$$
其功率为：
$$
P=lim_{T->\infty}1/{2T}\int_{-T}^T|f(t)|^2dt
$$
对于电阻R， 施加电压$f(t)$, 在区间$(-\infty,+\infty)$,其能量为：
$$
E=1/R\int_{-\infty}^{\infty}|f(t)|^2dt
$$
![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611083833017.png)

#### 2.2. 倒频谱

> “该分析方法方便提取、分析原频谱图上肉眼难以识别的周期性信号，能将原来频谱图上成族的边频带谱线简化为单根谱线，受传感器的测点位置及传输途径的影响小。”

> 倒频谱能较好地检测出功率谱上的周期成分，通常在功率谱上无法对边频的总体水平作出定量估计，而倒频谱对边频成分具有“概括”能力，**能较明显地显示出功率谱上的周期成分，将原来谱上成族的边频带谱线简化为单根谱线**，便于观察，而齿轮发生故障时的振动频谱具有的边频带一般都具有等间隔（故障频率）的结构，利用倒频谱这个优点，可以检测出功率谱中难以辨识的周期性信号。

**调制**：分为幅值调制和频率调制。下面以齿轮的*幅值调制*为例进行说明：齿轮的振动信号主要包括两部分，分别是齿轮啮合振动信号（高频）和齿轮轴的转频振动信号（低频），时域和频域曲线分别如下图所示：

![高频信号和低频信号时域波形](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611085332483.png)

![高频信号和低频信号的频域波形](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611085432069.png)

> 调制就是高低频率信号的混合。**幅值调制从数学上看，相当于两个信号在时域上相乘；而在频域上，相当于两个信号的卷积。**调制后的信号在时域和频域上分别变为：

![调制后的时域信号](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611085531551.png)

![调制后的频域信号](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611085543847.png)

> 调制后的信号中，除原来的啮合频率分量外，增加了一对分量，它们是以高频信号特征频率为中心，对称分布于两侧，所以称为**边频带**。

![边缘带形成](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611085735068.png)

> 傅里叶变换处理非平稳信号有天生缺陷。它只能获取一段信号总体上包含哪些频率的成分，但是对各成分出现的时刻并无所知。因此时域相差很大的两个信号，可能频谱图一样。

#### 2.3. 小波分析

> 小波直接把傅里叶变换的基给换了——将**无限长的三角函数基**换成了**有限长的会衰减的小波基**。这样**不仅能够获取频率**，还可以**定位到时间**了。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611090331061.png)

对于频率随着时间变化的非平稳信号：这三个时域上有巨大差异的信号，频谱（幅值谱）却非常一致

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611090529855.png)

> 对于这样的非平稳信号，只知道包含哪些频率成分是不够的，我们还想知道各个成分出现的时间。知道信号频率随时间变化的情况，各个时刻的瞬时频率及其幅值——这也就是时频分析。

短时傅里叶变换（Short-time Fourier Transform, STFT）

> 把整个时域过程分解成无数个等长的小过程，每个小过程近似平稳，再傅里叶变换，就知道在哪个时间点上出现了什么频率了。STFT存在一个问题，我们应该用多宽的窗函数？框窄：频率分辨率差， 宽了：时间分辨率差。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611090938200.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611090949400.png)



![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200611091502463.png)

> 不同于傅里叶变换，变量只有频率ω，小波变换有两个变量：尺度a（scale）和平移量τ（translation）。**尺度**a控制小波函数的**伸缩**，**平移量** τ控制小波函数的**平移**。**尺度**就对应于**频率**（反比），**平移量** τ就对应于**时间**。

### 3. 学习链接

- c#各种滤波算法：https://github.com/dtaylor-530/TimeSeriesFilterSharp
- https://zhuanlan.zhihu.com/p/34989414
- https://zhuanlan.zhihu.com/p/36163931



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/filter-learning-record/  

