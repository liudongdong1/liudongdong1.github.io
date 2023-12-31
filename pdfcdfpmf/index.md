# Pdf&Cdf&Pmf


### 1. PDF&CDF&PMF

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201203105337963.png)

### 2.  code

```python
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns
def readTxt(path):
    f = open(path,'r', encoding='UTF-8')
    dataList = []
    dataList=[line.split(":")[-1].split("\n")[0] for line in f.readlines()]
    dataList=[float(i) for i in dataList]
    return dataList
data=['52.20153254455275', '48.421227186748', '50.95434359918541']
data=[float(i) for i in data]
fs_xk = np.sort(data)

hist, bin_edges = np.histogram(fs_xk)
width = (bin_edges[1] - bin_edges[0]) * 0.95
plt.bar(bin_edges[1:], hist/sum(hist), width=width, color='#5B9BD5')
cdf = np.cumsum(hist/sum(hist))
plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
plt.xlabel("distance(cm)")
plt.ylabel("probability")
plt.grid()
plt.show()
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pdfcdfpmf/  

