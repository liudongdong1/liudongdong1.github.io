# Step Counter


> 随着人们生活水平的提高,生活方式和生活环境的改变,人们对健康体育锻炼越来越重视。大量研究表明,
> 运动能够减少慢性疾病的发生。快速准确地监测人体运 动中的能量消耗对改善人们身体健康状况具有重要意义。计步器是一种操作简便实用的运动测量设备,它通过内置的三轴加速度传感器实时获取人体在运动过程中三个方向的加速度数据,提取特征信息,计算出各种运动状态下的步数、距离、速度和能量消耗等数据,从而方便人们制定出更加科学合理的健身计划。

## 1. 总体思路 

人在走路时大致分为下面几种场景：

​	 1、正常走路，手机拿在手上（边走边看、甩手、不甩手）

​	 2、慢步走，手机拿在手上（边走边看、甩手、不甩手）

​	 3、快步走，手机拿在手上（甩手、不甩手、走的很快一般不会看手机吧）

​	 4、手机放在裤袋里（慢走、快走、正常走） 

​	5、手机放在上衣口袋里（慢走、快走、正常走） 

​	6、上下楼梯（上面五中场景可以在这个场景中再次适用一遍）

 所有场景的原始数据通过分析，其实是正弦波，每一个波峰为一个步点，算法其 实就是找到这些步点，分析波形特点寻找特征值，找到如下三个原则：

​	 a、规定曲线连续上升的次数

​	 b、波峰波谷的差值需要大于阈值

​	 c、阈值是动态改变的

## 2. 参数调节

- **detectorNewStep 中：** 在检测到是波峰时，波峰和相邻前一个波谷差值大于 ThreadValue（过滤掉抖动 的情况，时间差很小），并且时间差为 TimeInterval 以上，则认为是有效步点 同时，阈值大于 InitialValue 以上的情况,都会被纳入动态计算 ThreadValue 值。
- **在 detectorPeak 中** 规定连续上升两次，并且波峰值大于 20 才认为是一个有效波峰（目的也是过滤 掉波形上的抖动）
- **在 peakValleyThread averageValue 中** 滚动记录四个有效波峰波谷差值并计算平均值，再梯度化。合适的 ThreadValue 会过滤抖动，且识别步点。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200521220341636.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200521223025624.png)

> 如上图所示，可以通过矢量加速度的 斜率正负进行判断。3轴加速度传感器采集数据->滤波->二次波峰监测，波峰数就是步数。另外得有一些防抖动方法：连续监测到3（或其它）个波峰才开始算。

## 3.代码

- 优秀代码：https://github.com/linglongxin24/DylanStepCount
- [https://github.com/finnfu/stepcount/tree/master/demo%E4%BB%A5%E5%8F%8A%E7%AE%97%E6%B3%95%E6%96%87%E6%A1%A3](https://github.com/finnfu/stepcount/tree/master/demo以及算法文档)
- 这里通过实现Listener回调函数的方式实现传感器算法处理和最上层UI获得算法计算数据，并通过实现EventListener 实现计步器自定义算法，有没有其他的方式？如果有好几种算法，或者说消息通知有没有其他选择的方案。

### 1. 代码类结构

![代码阅读](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/%E4%BB%A3%E7%A0%81%E9%98%85%E8%AF%BB.png)

### 2. StepService 执行时序图

![StepService_onCreate](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/StepService_onCreate.png)

**1.如果先bindService,再startService:**
在bind的Activity退出的时候,Service会执行unBind方法而不执行onDestory方法,因为有startService方法调用过,所以Activity与Service解除绑定后会有一个与调用者没有关连的Service存在
**2.如果先bindService,再startService,再调用Context.stopService**
Service的onDestory方法不会立刻执行,因为有一个与Service绑定的Activity,但是在Activity退出的时候,会执行onDestory,如果要立刻执行stopService,就得先解除绑定

**3 先\**startService，再bindService。\****

首先在主界面创建时，startService(intent)启动方式开启服务，保证服务长期后台运行；
然后调用服务时，bindService(intent, connection, BIND_AUTO_CREATE)绑定方式绑定服务，这样可以调用服务的方法；

调用服务功能结束后，unbindService(connection)解除绑定服务，置空中介对象；
最后不再需要服务时，stopService(intent)终止服务。

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/step-counter/  

