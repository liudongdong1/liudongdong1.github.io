# DimsTransfor


### 1. 一维时间序列转化二维图片

#### 1.1. Gramian Angular Field (GAF)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/2019011313333637.gif)

> 使用一个限定在 [-1,1] 的最小-最大定标器（Min-Max scaler）来把时间序列缩放到 [-1,1] 里，这样做的原因是为了使内积不偏向于值最大的观测。然后计算Gram矩阵。

![Gram Matrix](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027100007996.png)

> **关键结论**：为什么要用 Gram 矩阵？
>
> Gram 矩阵保留了时间依赖性。由于时间随着位置从左上角到右下角的移动而增加，所以时间维度被编码到矩阵的几何结构中。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20190113133414188.gif)

- **编码过程：**
  - 用 Min-Max scaler 把序列缩放到 [-1,1] 上
  - 将缩放后的时间序列转换到「极坐标」![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027100804003.png)
  - 时间序列的内积：
    - 作者定义了内积计算方式：![image-20201027101126101](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027101126101.png)![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027101144972.png)
    - ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201027101237911.png)

> 1. 对角线由缩放后的时间序列的原始值构成（我们将根据深度神经网络学习到的高层特征来近似重构时间序列）；
> 2. 时间相关性是通过时间间隔 k k*k* 的方向叠加，用相对相关性来解释的。

code：https://github.com/devitrylouis/imaging_time_series

学习链接：https://blog.csdn.net/weixin_39679367/article/details/86416439

### 2. 二维图片转一维序列

> Grabocka, Josif, et al. "Learning time-series shapelets." *Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining*. 2014.

> - Shapelets可以提供可解释的结果，这可能有助于领域从业者更好地理解他们的数据。例如，在图3中，我们可以看到shapelet可以概括为：“荨麻有一个茎，它几乎以90度的角度与叶相连。”
> -  使用局部特征，因为全局特征可能因为噪声使特征失。
> - (3) 计算复杂度低，速度快，分类时间为O ( m l ) O(ml)*O*(*m**l*)，m是查询时间序列的长度，l是shapelet的长度。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210202094919891.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210202094952633.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210202095327559.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210202095454277.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/dimstransfor/  

