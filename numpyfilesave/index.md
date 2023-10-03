# numpy&FileSave


### 1. attributes

- **ndarray.ndim**： the number of axes (dimensions) of the array.
- **ndarray.shape**: For a matrix with *n* rows and *m* columns, `shape` will be `(n,m)`. The length of the `shape` tuple is therefore the number of axes, `ndim`.
- **ndarray.size**: the total number of elements of the array.
- **ndarray.dtype**: numpy.int32, numpy.int16, and numpy.float64 
- **create**: np.array(列表数据)

### 2. 数据存储

#### .0. 按行存储

```python
with open(filename,coding):
    data=f.read()
    
with open('result.txt',mode='a+') as f:
    f.write("preds:{},charpre:{},charclassify:{}".format(index,charclass,charpre))  # write 写入

#f.readline()#读取第一行
#f.readlines() #把内容按行读取至一个list
```

#### .1. 一/二维数据

> numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

- fname：表示要保存文件的地址，可以自己建文件名，如‘test.txt’
- X：表示要保存的文件
- fmt: 存储格式，如 %d, %.2f 
- delimiter ：分隔符，默认空格，也可以用逗号等
- newline:表示换行的时候用什么，默认\n，表示换一行，也可以用\t，则表示空四
- header：表示头文件，如“test_data"
- footer: 文件下的脚注
- comment：注释，默认是#,因为python的注释是#，也可以用其它符号

>`numpy.``loadtxt`**(***fname***,** *dtype=<class 'float'>***,** *comments='#'***,** *delimiter=None***,** *converters=None***,** *skiprows=0***,** *usecols=None***,** *unpack=False***,** *ndmin=0***,** *encoding='bytes'***,** *max_rows=None***,** *****,** *like=None***)

```python
np.savetxt(filename, list1,fmt='%d',delimiter=',')
dets= np.loadtxt(filename,delimiter=',')
```

#### .2. 高维数据

##### .1. pickle

- **write**

```python
import pickle
my_data = {'a': [1, 2.0, 3, 4+6j],
           'b': ('string', u'Unicode string'),
           'c': None}
output = open('data.pkl', 'wb')
pickle.dump(my_data, output)
output.close()

###Extract from file
with open("myfile.pkl","rb") as f:
    x_temp = pickle.load(f)
```

- **read**

```python
import pprint, pickle
pkl_file = open('data.pkl', 'rb')
data1 = pickle.load(pkl_file)
pprint.pprint(data1)
pkl_file.close()

###Load into file
with open("myfile.pkl","wb") as f:
    pickle.dump(x_train,f)
```

##### .2. scipy Mat

```python
import numpy as np
import scipy.io

# Some test data
x = np.arange(200).reshape((4,5,10))

# Specify the filename of the .mat file
matfile = 'test_mat.mat'

# Write the array to the mat file. For this to work, the array must be the value
# corresponding to a key name of your choice in a dictionary
scipy.io.savemat(matfile, mdict={'out': x}, oned_as='row')

# For the above line, I specified the kwarg oned_as since python (2.7 with 
# numpy 1.6.1) throws a FutureWarning.  Here, this isn't really necessary 
# since oned_as is a kwarg for dealing with 1-D arrays.

# Now load in the data from the .mat that was just saved
matdata = scipy.io.loadmat(matfile)

# And just to check if the data is the same:
assert np.all(x == matdata['out'])
```

##### .3. json

```python
import json
with open(filename, 'w') as f:
   json.dump(myndarray.tolist(), f)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/numpyfilesave/  

