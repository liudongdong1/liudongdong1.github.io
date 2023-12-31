# TimeSeqence


### 1. 时序特征

> [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/quick_start.html)是开源的提取时序数据特征的python包，能够提取出超过[64种特征.](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)

1. 时间序列`统计特征`：最大值、最小值、值域、均值、中位数、方差、峰度、同比、环比、周期性、自相关系数、变异系数
2. 时间序列`拟合特征`：移动平均算法、带权重的移动平均算法、指数移动平均算法、二次指数移动平均算法、三次指数移动平均算法、奇异值分解算法、自回归算法、深度学习算法
3. 时间序列`分类特征`：熵特征、小波分析特征、值分布特征（直方图分布、分时段的数据量分布）

### 2. 曲线形态

- 平稳

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521111628100.png)

- 不平稳，无趋势，有周期差异

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521111646799.png)

- 不平稳，无趋势，无周期差异

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521111657428.png)

- 不平稳，有趋势，有周期差异

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521111709262.png)

- 不平稳，有趋势，无周期差异

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521111717749.png)

| NO.  | 类型                         | 波动值 | 趋势 | 同环比差异 | 模型选择      |
| :--- | :--------------------------- | :----- | :--- | :--------- | :------------ |
| 1    | 平稳                         | 小     | /    | /          | 3-sigma       |
| 2    | 不平稳、无趋势、无同环比差异 | 大     | 无   | 有         | EWMA          |
| 3    | 不平稳、无趋势、有同环比差异 | 大     | 无   | 无         | 动态阈值      |
| 4    | 不平稳、有趋势、有同环比差异 | 大     | 有   | 有         | EWMA+变点检测 |
| 5    | 不平稳、有趋势、无同环比差异 | 大     | 有   | 无         | xgboost       |

- 动态阈值：
  - 移动平均：pandas.Series().rolling().mean()
  - 上下边界：原始序列和移动平均序列的MAE，标准差，以此计算动态阈值上下边界
  - 判断是否超出了上下边界：如果超出了，则认为是异常；否则，认为是正常	

### 3. 时间序列平稳化

> 将非平稳时间序列转化成平稳时间序列，包含三种类型：结构变化、差分平稳、确定性去趋势。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521112505013.png)

- **结构变化**

> 取`对数处理一些非线性趋势序列或将序列的指数趋势转化成线性趋势`。除此之外，还可以采用`指数转换`等方法将原来时间序列映射成不同的曲线形态。

- **差分**

> 经过`足够阶数的差分之后任何时间序列都会变成稳定的`，但是高于二阶的差分较少使用：每次差分会丢失一个观测值，丢失数据中所包含的一部分信息。
>
> 1. 一阶差分得到增长率
> 2. 二阶差分得到增长率的增长率（速度-加速度）
> 3. 高阶差分没有明确的解释

- **确定性去趋势**

> 去趋势是为了消除数据中的线性趋势或高阶趋势的过程。可以进行一个关于常数、时间t的线性或多项式回归，从回归中得到的残差代表去趋势的时间序列，多项式的阶数可以用F检验确定.

#### .1. **分解定理**

> 1. Wold分解定理：对于平稳时间序列，`时间序列=完全由历史信息确定的线性组合的确定性趋势部分+零均值白噪声序列构成的非确定性随机序列`。
> 2. Cramer分解定理：对于任何时间序列，`时间序列=完全由历史信息确定的多项式的确定性趋势部分+零均值白噪声序列构成的非确定性随机序列`。

- **组成部分**

> 1. 长期趋势Tt：长期总的变化趋势，递增、递减、或水平变动
> 2. 季节变化St：有规律的周期性的重复变动
> 3. 随机波动It：受众多偶然、难以预知和控制的因素影响

- **作用模式**

> 1. 加法模型：季节变动随着时间的推移保持相对不变，即三种成分相加，Xt = Tt + St + It；
> 2. 乘法模型：季节变动随着时间的推移递增或递减，即三种成分相乘，Xt = Tt * St * It；
> 3. 混合模型：三种成分有些相加、有的相乘，Xt = St * ( Tt +I t)。

#### .2. **趋势拟合计算长期趋势**

- 移动平均法

| 方法             | 描述                                   | 优缺点                                                       |
| :--------------- | :------------------------------------- | :----------------------------------------------------------- |
| 中心化移动平均法 | 取前后若干项求平均值作为趋势估计值     | 为消除季节变化的影响，移动平均项数应等于季节周期的长度       |
| 简单移动平均法   | 往前取若干项求平均值                   | 适用于未含有明显趋势的序列；移动平均项数多，平滑效果强，但对变化反应慢；有季节变化时，项数等于周期长度 |
| 二次移动平均法   | 在简单移动平均法的基础上再移动平均一次 | 简单移动平均法的结果比实际值存在滞后，二次移动可以避免这个问题 |

- **指数平均法**

| 方法                   | 描述                                                         | 特点                                                         |
| :--------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 简单指数平滑法         | 平滑估计值Xt’=aXt+(1-a)Xt-1’，反复迭代，平滑系数a取[0.05,3]效果较好，X0’可取X1 | 适用无季节变化、无长期趋势变化的序列；最好只做1期预测        |
| Holt线性指数平滑法     | 每期线性递增或递减的部分也做一个平滑修匀                     | 适用无季节变化、有线性趋势的序列，不考虑季节波动；可向前多期预测 |
| Holt-Winters指数平滑法 | 加上了季节变动                                               | 考虑了季节波动、长期趋势                                     |

#### 3. 回归模型

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210521113216905.png)

### 4. **AutoEncoder模型**

>原始的时间序列往往面临维度高，存在噪声等情况，通过特征提取，对原始数据进行清洗和丰富等，会比直接将原始数据扔给神经网络要好。
>
>  AutoEncoder是一种典型的无监督方法，可以将其扩展为Variational AutoEncoder，或者引入情景信息，从而扩展为Conditional Variational AutoEncoder。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210518225641.png)

```python
def create_sequences(values, steps=n_steps):
    output = []
    for i in range(len(values) - steps):
        output.append(values[i : (i + steps)])
    return np.stack(output)

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(filters=16, kernel_size=4, padding="same", strides=2, activation="relu"),
        layers.Dropout(rate=0.2),
        layers.Conv1D(filters=8, kernel_size=4, padding="same", strides=2, activation="relu"),
        layers.Conv1DTranspose(filters=8, kernel_size=4, padding="same", strides=2, activation="relu"),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(filters=16, kernel_size=4, padding="same", strides=2, activation="relu"),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
model.summary()

dota_model = model.fit(
    x_train,
    x_train,
    epochs=2009,
    batch_size=512,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210518225825.png)

### Resouce

- https://cloud.tencent.com/developer/article/1670322?from=information.detail.%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B
- 时序分析：https://otexts.com/fppcn/backcasting.html
- https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650422492&idx=5&sn=eb19e9502a3a7698c834a3ef1adf2a9f&chksm=becdba8689ba339084b80a877273205b5187f1a894a6bd54ac7ed6f7078b425d8e8df6051b12&mpshare=1&scene=24&srcid=0513hAdiho4nS51dijwu5e5z&sharer_sharetime=1620884589891&sharer_shareid=ed830dd7498411a03cfe8154944facae#rd



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/timesequence/  

