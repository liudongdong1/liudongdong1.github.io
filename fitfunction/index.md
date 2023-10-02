# FitFunction


### 1. 多项式拟合

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟生成一组实验数据
x = np.arange(0,10,0.2)
y = -(x-3.5)**2+4.7
noise = np.random.uniform(-3,3,len(x))
y += noise
fig, ax = plt.subplots()
ax.plot(x, y, 'b--')
ax.set_xlabel('x')
ax.set_ylabel('y')

# 二次拟合
coef = np.polyfit(x, y, 2)
y_fit = np.polyval(coef, x)
ax.plot(x, y_fit, 'g')
# 找出其中的峰值/对称点
if coef[0] != 0:
    x0 = -0.5 * coef[1] / coef[0]            
    x0 = round(x0, 2)        
    ax.plot([x0]*5, np.linspace(min(y),max(y),5),'r--')
    print(x0)
else:
    raise ValueError('Fail to fit.')

plt.show()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210412232942812.png)

### 2. 自定义函数拟合

> 对于自定义函数拟合，不仅可以用于`直线、二次曲线、三次曲线的拟合`，它可以`适用于任意形式的曲线的拟合`，`只要定义好合适的曲线方程`即可。

```python
from scipy.optimize import curve_fit
x = [20,30,40,50,60,70]
x = np.array(x)
num = [453,482,503,508,498,479]
y = np.array(num)

# 这里的函数可以自定义任意形式。
def func(x, a, b,c):
    return a*np.sqrt(x)*(b*np.square(x)+c)

# popt返回的是给定模型的最优参数。我们可以使用pcov的值检测拟合的质量，其对角线元素值代表着每个参数的方差。
popt, pcov = curve_fit(func, x, y)
a = popt[0] 
b = popt[1]
c = popt[2]
yvals = func(x,a,b,c) #拟合y值
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
```

### 3. 逆函数计算pynverse库

- pip install pynverse

```python
from pynverse import inversefunc
#计算某些y_values 点的反函数：
cube = (lambda x: x**3)
invcube = inversefunc(cube, y_values=3)
# array(3.0000000063797567)

```

### 4. Scipy 数值计算库

- https://docs.huihoo.com/scipy/scipy-zh-cn/scipy_intro.html

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210412233301.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/fitfunction/  

