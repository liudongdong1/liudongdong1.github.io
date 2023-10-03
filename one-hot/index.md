# One-hot


### 1. 离散特征的编码

1. `离散特征的取值之间没有大小的意义`，比如color：[red,blue],那么就使用one-hot编码
2. `离散特征的取值有大小的意义`，比如size:[X,XL,XXL],那么就使用`数值的映射`{X:1,XL:2,XXL:3}

```python
import pandas as pd
df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'], 
            ['red', 'L', 13.5, 'class2'], 
            ['blue', 'XL', 15.3, 'class1']])
 
df.columns = ['color', 'size', 'prize', 'class label']
 
size_mapping = {
           'XL': 3,
           'L': 2,
           'M': 1}
df['size'] = df['size'].map(size_mapping)
 
class_mapping = {label:idx for idx,label in enumerate(set(df['class label']))}
df['class label'] = df['class label'].map(class_mapping)
```

### 2. One-hot

> 独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
>
> - 优点：独热编码`解决了分类器不好处理属性数据的问题`，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。
> - 缺点：`当类别的数量很多时，特征空间会变得非常大`。在这种情况下，一般可以用PCA来减少维度。而且`one hot encoding+PCA这种组合`在实际中也非常有用。
> - 独热编码（哑变量 dummy variable）是因为`大部分算法是基于向量空间中的度量来进行计算的`，为了使非偏序关系的变量取值不具有偏序性，并且到圆点是等距的。`使用one-hot编码，将离散特征的取值扩展到了欧式空间`，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理。
>
> 不需要one-hot编码：
>
> - 但如果`特征是离散的`，并且`不用one-hot编码就可以很合理的计算出距离`，那么就没必要进行one-hot编码。有些基于树的算法在处理变量时，并不是基于向量空间度量，数值只是个类别符号，即没有偏序关系，所以不用进行独热编码。 Tree Model不太需要one-hot编码： 对于决策树来说，one-hot的本质是增加树的深度。
>
> 使用One-hot编码：
>
> - **one hot的形式无法比较大小。**用于比较的label之间没有大小关系
> - **计算top N准确度**

```python
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])    # fit来学习编码 数据矩阵是4*3，即4个数据，3个特征维度， 每一个特征维度都单独进行onehot编码
enc.transform([[0, 1, 3]]).toarray()    # 进行编码
#输出 array([[ 1., 0., 0., 1., 0., 0., 0., 0., 1.]])
```

- 数值类别

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

testdata = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish'],                         
'age': [4 , 6, 3, 3],                         
'salary':[4, 5, 1, 1]})
OneHotEncoder(sparse = False).fit_transform( testdata[['age']] )
```

- 字符串编码

> - 方法一 `先用 LabelEncoder() 转换成连续的数值型变量`，再用 `OneHotEncoder() 二值化`
> - 方法二` 直接用 LabelBinarizer() 进行二值化`

```python
# 方法一: LabelEncoder() + OneHotEncoder()
a = LabelEncoder().fit_transform(testdata['pet'])
OneHotEncoder( sparse=False ).fit_transform(a.reshape(-1,1)) # 注意: 这里把 a 用 reshape 转换成 2-D array

# 方法二: 直接用 LabelBinarizer()
LabelBinarizer().fit_transform(testdata['pet'])
```

### 3. LabelEncoding

> Label Encoding是使用字典的方式，将`每个类别标签与不断增加的整数相关联`，即生成一个名为class_的实例数组的索引。
>
> - fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
> - fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
> - inverse_transform(y)：根据索引值y获得原始数据。
> - transform(y) ：将y转变成索引值.

> `只是将文本转化为数值，并没有解决文本特征的问题`：所有的标签都变成了数字，算法模型直接将根据其距离来考虑相似的数字，而不考虑标签的具体含义。使用该方法处理后的数据适合支持类别性质的算法模型，如LightGBM.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
city_list = ["paris", "paris", "tokyo", "amsterdam"]
le.fit(city_list)
print(le.classes_)  # 输出为：['amsterdam' 'paris' 'tokyo']
city_list_le = le.transform(city_list)  # 进行Encode
print(city_list_le)  # 输出为：[1 1 2 0]
city_list_new = le.inverse_transform(city_list_le)  # 进行decode
print(city_list_new) # 输出为：['paris' 'paris' 'tokyo' 'amsterdam']
```

### 4. Ordinal Encoding

> 对于一个具有`m个category的Feature，我们将其对应地映射到 [0,m-1] 的整数`。当然 Ordinal Encoding 更`适用于 Ordinal Feature，即各个特征有内在的顺序`。例如对于`”学历”这样的类别，”学士”、”硕士”、”博士” `可以很自然地编码成 [0,2]，因为它们内在就含有这样的逻辑顺序。

```python
ord_map = {'Gen 1': 1, 'Gen 2': 2, 'Gen 3': 3, 'Gen 4': 4, 'Gen 5': 5, 'Gen 6': 6}
df['GenerationLabel'] = df['Generation'].map(gord_map)
```

### 5. Frequency Encoding/Count Encoding

> 将类别特征替换为训练集中的计数（一般是根据训练集来进行计数，属于统计编码的一种，统计编码，就是用类别的统计特征来代替原始类别，比如类别A在训练集中出现了100次则编码为100）。这个方法对离群值很敏感，所以结果可以归一化或者转换一下（例如使用对数变换）。未知类别可以替换为1。

### 6. 目标编码

> 目标编码（target encoding），亦称均值编码（mean encoding）、似然编码（likelihood encoding）、效应编码（impact encoding），是`一种能够对高基数（high cardinality）自变量进行编码的方法` (Micci-Barreca 2001) 。如果某一个特征是定性的（categorical），而这个特征的可能值非常多（高基数），那么目标编码（Target encoding）是一种高效的编码方式。
>
> - 高基数定性特征的例子：IP地址、电子邮件域名、城市名、家庭住址、街道、产品号码。

### 7. LabelEncoder&LabelBinarizer&OneHotEncoder

> - 使用`LabelEncoder会以为两个相近的数字比两个较远的数字更为相似一些`，为了解决这个问题，使用独热编码，即OneHot编码，得到的输出结果默认是稀疏矩阵，可以使用toarray()方法完成转换。

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
test_data = np.array(["a", "b", "c", "d", "a"])

#[0 1 2 3 0]  LabelEncoder 产生结果为连续性特征
print(LabelEncoder().fit_transform(test_data))

#LabelBinarizer 直接返回numpy数组，可以通过使用sparse_output=True给LabelBinarizer构造函数，可以得到稀疏矩阵。
#[[1 0 0 0]
#[0 1 0 0]
#[0 0 1 0]
#[0 0 0 1]
#[1 0 0 0]]
print(LabelBinarizer().fit_transform(test_data))

print(OneHotEncoder().fit_transform(test_data.reshape(-1, 1)))  # 输出是一个SciPy稀疏矩阵
print(OneHotEncoder().fit_transform(test_data.reshape(-1, 1)).toarray())    # 转换成一个密集的NumPy数组 结果如下
#(0, 0) 1.0
#(1, 1) 1.0
#(2, 2) 1.0
#(3, 3) 1.0
#(4, 0) 1.0
#[[1. 0. 0. 0.]
#[0. 1. 0. 0.]
#[0. 0. 1. 0.]
#[0. 0. 0. 1.]
#[1. 0. 0. 0.]]
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/one-hot/  

