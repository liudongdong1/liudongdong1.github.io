# ModeType


### 1. 回归问题

> `回归问题通常是用来预测一个值`，如预测房价、未来的天气情况等等，例如一个产品的实际价格为500元，通过回归分析预测值为499元，我们认为这是一个比较好的回归分析。一个比较常见的回归算法是线性回归算法（LR）。另外，回归分析用在神经网络上，`其最上层是不需要加上softmax函数的`，而是直接对前一层累加即可。回归是对真实值的一种逼近预测。

#### .1. softmax 

> - 当标签类别使用**独热编码**时，使用 `loss = 'categorical_crossentropy'`
> -  当标签类别使用**顺序编码**时，使用 `loss = 'sparse_categorical_crossentropy'`来计算softmax交叉熵

### 2. 分类问题

> 分类问题是用于`将事物打上一个标签`，通常`结果为离散值`。例如判断一幅图片上的动物是一只猫还是一只狗，分类通常是建立在回归之上，`分类的最后一层通常要使用softmax函数进行判断其所属类别`。分类并没有逼近的概念，最终正确结果只有一个，错误的就是错误的，不会有相近的概念。最常见的分类方法是逻辑回归，或者叫逻辑分类。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523232210127.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210523231912750.png)

### 3. 案例分析

```python
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# 训练集属性，标签
train_image.shape, train_label.shape
#((60000, 28, 28), (60000,))
# 测试集属性，标签
test_image.shape, test_label.shape
#((10000, 28, 28), (10000,))

#------------------顺序编码------------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 使每一条图片数据进行扁平化
# 上边的等价于 data.reshape((data.shape[0], -1))
model.add(tf.keras.layers.Dense(128, activation='relu'))  
# 输出的维度不能太小（因为输入的样本维度(28*28)较大，不能舍弃过多细节）
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 输出10个概率值（分别对应类别0～9）
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])
             
model.fit(train_image, train_label, epochs=5)

#------------------使用one-hot编码-----------------
train_label_onehot = tf.keras.utils.to_categorical(train_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)
model_onehot = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])

model_onehot.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['acc'])

model_onehot.fit(train_image, train_label_onehot, epochs=5)

#------------------俩种方式进行测试-----------------
#获得结果都是对模型概率
pred = model.predict(test_image)[0]  # 对测试集所有数据进行预测，再取出第一条的结果
pred_onehot = model_onehot.predict(train_image[:1,:,:])  # 只对测试集的第1条进行预测
```

![image-20210524154737349](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210524154737349.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/modeltype/  

