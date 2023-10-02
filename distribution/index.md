# Distribution


```python
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import math
#均值
def average(data):
    return np.sum(data)/len(data)
#标准差
def sigmaHandle(data,avg):
    sigma_squ=np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.6)
#高斯分布概率
def prob(data,avg,sig):
    sqrt_2pi=np.power(2*np.pi,0.6)
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
    return gauss
#绘制散点图
def scattor(data,data1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(data, data1)
    plt.show()
def threeDscattor(data,data1):
    fig = plt.figure()  
    ax = Axes3D(fig)
    ax.scatter(data, data1,[0 for i in range(0,len(data))])
    plt.show()

def softmax(x):
    softmax_x=x/np.sum(x)
    return softmax_x
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
'''
# 108399Ditemp.txt
[180.] [90.] [0.0206388]    #for square
4.000000000000000000e+01,2.148998427795590152e-02   #for line
(array([52], dtype=int64),) [0.03297474]        # for line truth

127499Ditemp.txt
(array([159], dtype=int64),) [0.00131725]
'''
#108399, 127499, 145199, 159033, 175633, 187833, 203133, 215166, 225833, 238166, 249699    # final 203133Ditemp.txt
BASE_DIR =os.path.join(r'E:\RFIDKinecttestData\12_1_w',"110832Ditemp.txt")                #175633   16cm       # 203133 49cm   #238166  35cm  #249699  42cm
print(BASE_DIR)  

# 生成2-D数据
data = np.loadtxt(BASE_DIR,dtype="str",delimiter=" ")
color=data[:,4].astype("float")
x = np.arange(0,len(color)).astype("int")


# color=getColor(color)   #获得概率高斯连续分布
# color=np.power(6,np.absolute(np.log(color)))/data[:,4].astype("float")
# color=softmax(color)
# plt.hist(color,bins=10)
color=1.0/color
index=np.where(color==np.max(color))

print(index,color[index])

datat=np.c_[x,color.T]
np.savetxt("new.csv", datat, delimiter=',')
# 正常显示中文字体
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
# 绘图，设置(label)图例名字为'第一条线'，显示图例plt.legend()
plt.plot(x,color,label='DistanceValue')
# x轴标签
plt.xlabel('D(cm)')
# y轴标签
plt.ylabel('Distance(cm^2)')
# 可视化图标题
plt.title('LenghChange')
# 显示图例
plt.legend()
# 显示图形
plt.show() 
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/图片1.png)

- 球体分布

```python
# -*- coding: UTF-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import math
import os
from mpl_toolkits.mplot3d import proj3d
#均值
def average(data):
    return np.sum(data)/len(data)
#标准差
def sigmaHandle(data,avg):
    sigma_squ=np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.6)
#高斯分布概率
def prob(data,avg,sig):
    sqrt_2pi=np.power(2*np.pi,0.6)
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
    return gauss
#绘制散点图
def scattor(data,data1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(data, data1)
    plt.show()
def threeDscattor(data,data1):
    fig = plt.figure()  
    ax = Axes3D(fig)
    ax.scatter(data, data1,[0 for i in range(0,len(data))])
    plt.show()
 # 找出当前路径
#108399, 127499, 145199, 159033, 175633, 187833, 203133, 215166, 225833, 238166, 249699
BASE_DIR =os.path.join(r'E:\RFIDKinecttestData\12_1_w',"110832Pitemp.txt")             
print(BASE_DIR)

# 生成2-D数据
data = np.loadtxt(BASE_DIR,dtype="str",delimiter=" ")
x_data=data[:,-1].astype("float")
y_data=data[:,-2].astype("float")
color=data[:,-3].astype("float")
color=1.0/color
index=np.where(color==np.max(color))

print(index,x_data[index],y_data[index],color[index])


# calculate hist distribution
#plt.hist(color,bins=100)

#color=getColor(color)   #获得概率高斯连续分布
#color=np.power(6,np.absolute(np.log(color)))/data[:,4].astype("float")
plt.hist(color,bins=100)


#color=np.exp(color)

# scattor(data[:,4].astype("float"),color)
# color=np.log(1-color)
# scattor(data[:,4].astype("float"),color)
# color=np.absolute(color)
# scattor(data[:,4].astype("float"),color)


#color=getColor(color)*data[:,4].astype("float")

fig = plt.figure()  
ax = fig.gca(projection='3d')
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False


# 图大小
# 正常显示中文字体
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
#RdBu        OrRd
p1=ax.scatter(x_data,y_data,color,marker='o', c=color, cmap='YlGnBu',s=20)  # 绘制数据点,颜色是红色
#p2=ax.scatter(-40,0,0,c="green",marker="+",s=60)  #x_data[index]-1,y_data[index]+1,z_data[index]+1
p2=ax.scatter(x_data[index]-1,y_data[index]-1,color[index],c="red",marker="o",s=100)
print(x_data[index],y_data[index],color[index])
p3=ax.scatter(180,90,color[-1],c="green",marker="+",s=150)
#p3=ax.scatter(0,90,color[16291],c="green",marker="+",s=100)
print(0,90,color[91])
zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
#ax.text(x_data[index],y_data[index],z_data[index], "GroundTruth")


ax.set_zlabel('similarity')  # 坐标轴
ax.set_ylabel('alpha(degree)')
ax.set_xlabel('theta(degree)')
ax.legend([p1,p2,p3],["Similarity","MaxPoint","GroudTruth"],loc = 'best', scatterpoints=1)
#plt.axis('off')

plt.show()

```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210314161451827.png)

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/distribution/  

