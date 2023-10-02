# Panda Tutorial


### 1. 文件读写

```python
pd.read_csv(filepath_or_buffer, sep=’,’, delimiter=None, header=’infer’, names=None, index_col=None, usecols=None, squeeze=False, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None)
```

> filepath_or_buffer：文件名、文件具体或相对路径、文件对象
>
> usecols： 保留指定列
>
> sep、delimiter：俩者均为文件分割符号，或为正则表达式
>
> header：当文件中无列名需将其设为None
>
> names：结合header=None，读取时传入列名
>
> skiprows：忽略特定的行数
>
> nrows：读取一定行数
>
> na_values：一组将其值转换为NaN的特定值
>
> sueeze：返回Series对象

```python
#导入
import pandas as pd
pd.__version__
#读入csv文件
ted=pd.read_csv('ted.csv')
#查看
ted.head(5)
ted.shape   # rows, colums
ted.dtypes  #每一列名称 及其类型
ted.columns
ted.index   #查看如何索引的
ted.describe()
#缺失值
column   #number of missing values each column
c = ted.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
#按comment 属性升序序
ted.sort_values('comments').tail()

#绘图
ted.comments.plot()  #
ted.comments.plot(kind='hist')
ted[ted.comments<1000].comments.plot()
ted.loc[ted.comments<1000,'comments'].plot(kind='hist')
```

```python
#txt文件转化未csv文件批量操作
import numpy as np
import pandas as pd
import os
files=os.listdir("E:\\RFIDKinecttestData\\2020_3_4O1\\result\\")
for file in files:
    filepath=os.path.join("E:\\RFIDKinecttestData\\2020_3_4O1\\result\\",file)
    print(filepath)
    data=np.loadtxt(filepath,dtype='str',delimiter=' ')
    data=data[:,:2]
    print(data)
    data=data.astype('float')
    data=pd.DataFrame(data)
    writer=pd.ExcelWriter(file.split(".")[0]+".xlsx")
    data.to_excel(writer)
    writer.save()
    writer.close()
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/panda-tutorial/  

