# Matlab Learning Record


> **MATLAB**是一种语法简单用途广泛的编程语言，既可以用于编写脚本，函数，也可以用于面向对象的程序开发或开发GUI界面。**MATLAB**被广泛应用于数值计算，图像处理，机器学习等领域。
>
> 1. 高效的数值计算及符号计算功能，能使用户从繁杂的数学运算分析中解脱出来；
> 2. 广阔的线性代数，统计，傅立叶分析，筛选，优化，数值积分，解常微分方程的数学函数库。
> 3. 具有完备的图形处理功能，实现计算结果和编程的可视化；
> 4. 友好的用户界面及接近数学表达式的自然化语言，使学者易于学习和掌握；
> 5. 功能丰富的应用工具箱(如信号处理工具箱、通信工具箱等) ，为用户提供了大量方便实用的处理工具。 
> 6. MATLAB的编程接口给开发工具，提高代码质量和可维护性和性能的最大化。
> 7. 它提供了基于MATLAB算法集成了[C](https://www.w3cschool.cn/c/)，[Java](https://www.w3cschool.cn/java/)，NET和[Microsoft Excel](https://www.w3cschool.cn/exceljc/)等与外部应用程序和语言功能。 

### 1. 变量与矩阵

> 一个三维数组由行、列和页三维组成，其中每一页包含一个由行和列构成的二维数组。A=zeros(4,3,2) 生成一个`4行3列2页`的三维全0数组

**MATLAB**在`变量声明是不需要指出变量的类型`。

~~~matlab
clear; %清空内存
clc; %清空命令行
r1=1; %为一个变量赋值
z1=1+sqrt(3)*i; %赋值一个复数 sqrt()开方运算
z_real=real(z1); %复数的实部
z_img=imag(z1); %复数的虚部
z_abs=abs(z1); %复数的模
z_ang=angle(z1); %复数的幅角
z2=z1^2; %平方运算
~~~

**MATLAB**的`数组索引从1开始，这点需要牢记`。

~~~matlab
arr1=rand(1,5); %arr1=[0.1418,0.4217,0.9157,0.7922,0.9594]
arr2=zeros(1,5); %arr2=[0,0,0,0,0]
arr3=ones(1,5); %arr3=[1,1,1,1,1]
arr4=linspace(1,2,5); %arr4=[1,1.25,1.5,1.75,2]
arr4=linspace(2,2,5); %arr4=[2,2,2,2,2]   
mat1=rand(3,3); %随机生成3*3矩阵
mat2=[1,2,3;4,5,6;7,8,9];
~~~

> 获取`一维数组的长度用length函数`；获取`多维函数的维数大小用size`;

```matlab
n = ndims(A)  # 获取数组维度
numberOfElements = length(array) #即一维数组的长度或者多维数组中最大的维数行数或列数中的较大值
[m,n] = size(X)    #获得矩阵的各个维数的大小
a=[1,2,3,4,5]   #用逗号或空格间隔
a=[1 2 3 4 5]  
x=初始值 :[步长]:终值
x=linspace(初始值 ,终值,个数n)
x=logspace(初始值 ,终值,个数n)#生成[10初值，10终值]之间等分的n个数 如果步长省略，默认步长为50
M = max(A)
C = max(A,B)
for i=1:1:r
    plot([time(w(i,1)),timev2(w(i,2))],[phaseznormal(w(i,1)),phasev2(w(i,2))],'--','Color',[0.5 0.5 0.5], 'LineWidth',0.5);
    hold on
end
```

### 2. 分支与循环

**MATLAB**常用的分支语句有__if-else__和__switch-case__

```matlab
limit = 0.75;
A = rand(10,1)
if any(A > limit)
    disp('There is at least one value above the limit.')
else
    disp('All values are below the limit.')
end
```

**MATLAB**常用的循环有__while__循环和__for__循环

```matlab
for v = 1.0:-0.2:0.0
   disp(v)
end

for v = [1 5 8 17]
   disp(v)
end
```

### 3. 函数及函数句柄

这里分别使用`函数`和`函数句柄`的方法来生成__Fibonacci__数列。

需要注意`函数名和文件名要保持一致`，以下先使用`函数`的方式：

~~~matlab
function y = fibonacci (x)
if x == 1 || x==2
    y = 1;
    return % return可以不写
else 
    y = fibonacci(x-1) + fibonacci(x-2);
    return
end
~~~

以下是使用`函数句柄`的方式：

~~~matlab
fibo=@(n) (((1+sqrt(5))/2)^n-((1-sqrt(5))/2)^n)/sqrt(5);
fn=zeros(1,100);
for i=1:1:100
    fn(i)=fibo(i);
end
~~~

### 4. 数值微积分

使用dx=0.000001为步长的向前差分`求sin(x)的导数`：

$$f^,(x)=\frac{f(x+dx)-f(x)}{dx}$$

~~~matlab
figure ('name','diff demo1');
x=linspace(0,10,100);
y=sin(x);
dx=0.000001;dydx=[];
for i=1:100
    dydx(i)=(sin(x(i)+dx)-y(i))/dx;
end
plot(x,y,'r',x,dydx,'b');
legend('sin(x)','cos(x)');
title('diff demo');
xlabel('x');ylabel('y')
~~~

使用**MATLAB**的差分工具`diff`计算导数

~~~matlab
h = 0.001;       % step size
X = -pi:h:pi;    % domain
f = sin(X);      % range
Y = diff(f)/h;   % first derivative
Z = diff(Y)/h;   % second derivative
plot(X(:,1:length(Y)),Y,'r',X,f,'b', X(:,1:length(Z)),Z,'k')
~~~

使用矩形法计算$\int_0^1x^2dx$：

$$\int_a^bf(x)dx=\frac{b-a}{n}\sum_{i=1}^nf(x_i)$$

~~~matlab
n=100000;a=0;b=1; %取步长为100000
x=a:1/n:b;
dx=(b-a)/n;x=x+dx/2;
s=x.^2; %采样
int=dx*sum(s);
~~~

调用**MATLAB**中的`quad`函数`使用__Simpson__法计算数值积分`：

~~~matlab
func=@(x)x.^2;
int=quad(func,0,1)
~~~

### 5. 常微分方程（组）的数值解

使用__Euler__法计算常微分方程（误差较大，不推荐）：

$$\frac{dy}{dx}=x^2+y^2+3x-2y$$

$$y|_{x=0}=1$$

取时间步长为h，则

$$y(x_{n+1})=y(x_n)+f(y(x_n),x_n)*h$$

~~~matlab
function matlab_demo
    func=@(x,y)x.^2+y.^2+3*x-2*y
    [x,y]=euler(func,[0,1],1,0.01)
    plot(x,y)
return

function [x,y]=euler(fun,xspan,y0,h)
    x=xspan(1):h:xspan(2)
    y(1)=y0;
    for n=1:length(x)-1
        y(n+1)=y(n)+h*feval(fun,x(n),y(n))
    end
return
~~~

使用45阶__Runge-Kutta__算法`ode45`计算常微分方程组：

$$\frac{dx}{dt}=2x-3y$$

$$\frac{dy}{dt}=x+2y$$

$$x|_{t=0}=1$$

$$y|_{t=0}=1$$

~~~matlab
function ode_demo
y0=[1,1];
tspan=0:0.01:5;
option = odeset('AbsTol',1e-4);
[t,x]=ode45(@dfunc,tspan,y0,option);
figure('name','ode45 demo');
plot(t,x(:,1),'r',t,x(:,2),'b');
return

function dx=dfunc(t,x)
dx=zeros(2,1);
dx(1)=2*x(1)-3*x(2); % x(1)=x
dx(2)=x(1)+2*x(2); % x(2)=y
return
~~~

### 6. 偏微分方程（组）的数值解

使用`pdepe`进行微分方程（组）的求解，需要先将微分方程（组），以及边界和初值条件化为如下形式：

$$c(x,t,\frac{\partial{u}}{\partial{x}})\frac{\partial{u}}{\partial{t}}=x^{-m}\frac{\partial}{\partial{t}}[x^mf(x,t,u,\frac{\partial{u}}{\partial{x}})]+s(x,t,u,\frac{\partial{u}}{\partial{x}})$$

$$p(x,t,u)+q(x,t,u)*f(x,t,u,\frac{\partial{u}}{\partial{x}})=0$$

$$u(x,t_0)=u_0$$

举一个例子：

$$\frac{\partial{u}}{\partial{t}}=\frac{\partial^2{u}}{\partial{x^2}}-u$$

$$u|_{x=0}=1$$

$$u|_{x=1}=0$$

$$u|_{t=0}=(x-1)^2$$

求解过程如下：

~~~matlab
function pde_demo
    x=0:0.05:1;
    t=0:0.05:1;
    m=0;
    sol=pdepe(m,@pdefun,@pdeic,@pdebc,x,t);
    figure('name','pde demo');
    surf(x,t,sol(:,:,1));
    title('pde demo');
    xlabel('x');ylabel('t');zlabel('u');
return

function [c,f,s]=pdefun(x,t,u,du) %方程描述函数
    c=1;
    f=1*du;
    s=-1*u;
return

function [pa,qa,pb,qb]=pdebc(xa,ua,xb,ub,t) %边界描述函数
    pa=ua-1;
    qa=0;
    pb=ub;
    qb=0;
return

function u0=pdeic(x) %初值描述函数
    u0=(x-1)^2;
return
~~~

### 7. 文件读取

#### .1. load&np

```python
#通过python 文件进行存储为txt格式，然后通过matlab代码直接读取
np.savetxt("xy1.txt", yvals,fmt='%d',delimiter=',')
```

```matlab
clear ;close all; clc
data=load('./kinect/314637_Guesture_segment.txt')
x_dtw=data(:,1)
y_dtw=data(:,2)
z_dtw=data(:,3)
time=data(:,4)
%转置操作  data=data.’
phase=4*pi*(x_dtw.^2+y_dtw.^2+z_dtw.^2)/0.33
phaseznormal=zscore(phase)
save kinect.txt -ascii phaseznormal
```

#### .2. fscanf

> `A = fscanf(fileID,formatSpec,sizeA)` 将文件数据读取到维度为 `sizeA` 的数组 `A` 中，并将文件指针定位到最后读取的值之后。`fscanf` 按列顺序填充 `A`。`sizeA` 必须为正整数或采用 `[m n]` 的形式，其中 `m` 和 `n` 为正整数。
>
> `A = fscanf(fileID,formatSpec)` 将打开的文本文件中的数据读取到列向量 `A` 中，并根据 `formatSpec` 指定的格式解释文件中的值。
>
> - A为存放读取的数据，一般为矩阵
> - count为成功读取元素的个数
> - format为读取的数据格式，如%d为十进制的读取格式
> - size为A的数据格式，有如下三种：
>   - inf        一直读到最后结束    
>   - N         读N个元素放入列向量  
>   - [M,N]  按列顺序读取至少一个M×N矩阵的M * N元素。N可以是inf，但M不能。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211004104009893.png)

#### .3. fprintf

```matlab
x = 100*rand(8,1);
fileID = fopen('nums1.txt','w');
fprintf(fileID,'%4.4f\n',x);
fclose(fileID);
#查看文件内容
type filename
```

### 8. PointCloud

#### .1. PointCloud

> `ptCloud = pointCloud(xyzPoints)` returns a point cloud object with coordinates specified by `xyzPoints`.
>
> `ptCloud = pointCloud(xyzPoints,Name,Value)` creates a `pointCloud` object with properties specified as one or more `Name,Value` pair arguments. For example, `pointCloud(xyzPoints,'Color',[0 0 0])` sets the `Color` property of the point `xyzPoints` as [0 0 0]. Enclose each property name in quotes. Any unspecified properties have default values.

```matlab
pcshow(ptCloud)   %显示点云信息

ptCloud = pointCloud(xyzPoints,'Color',cmatrix);
pcshow(ptCloud)
%Display the point cloud and plot the surface normals.
pcshow(ptCloud)
x = ptCloud.Location(:,1);
y = ptCloud.Location(:,2);
z = ptCloud.Location(:,3);
u = normals(:,1);
v = normals(:,2);
w = normals(:,3);
hold on
quiver3(x,y,z,u,v,w);
hold off
```

#### .2. pcnormals

> **normals = pcnormals(ptCloud)** returns a matrix that stores a normal for each point in the input `ptCloud`. The function uses` six neighboring points to fit a local plane to determine each normal vector.`
>
> **normals = pcnormals(ptCloud,k)** additionally specifies `k`, the number of points used for local plane fitting. The function uses this value rather than the six neighboring points as described in the first syntax.

```matlab
normals = pcnormals(ptCloud);

figure
pcshow(ptCloud)
title('Estimated Normals of Point Cloud')
hold on

x = ptCloud.Location(1:10:end,1:10:end,1);
y = ptCloud.Location(1:10:end,1:10:end,2);
z = ptCloud.Location(1:10:end,1:10:end,3);
u = normals(1:10:end,1:10:end,1);
v = normals(1:10:end,1:10:end,2);
w = normals(1:10:end,1:10:end,3);
%plot the normal vectors
quiver3(x,y,z,u,v,w);
hold off
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211004113412687.png)

```matlab
%翻转法线以指向传感器位置。此步骤仅用于确定曲面的向内或向外方向。传感器中心设置在x、y、z坐标中。
sensorCenter = [0,-0.3,0.3]; 
for k = 1 : numel(x)
   p1 = sensorCenter - [x(k),y(k),z(k)];
   p2 = [u(k),v(k),w(k)];
   % Flip the normal vector if it is not pointing towards the sensor.
   angle = atan2(norm(cross(p1,p2)),p1*p2');
   if angle > pi/2 || angle < -pi/2
       u(k) = -u(k);
       v(k) = -v(k);
       w(k) = -w(k);
   end
end

%plot the adjust normals
figure
pcshow(ptCloud)
title('Adjusted Normals of Point Cloud')
hold on
quiver3(x, y, z, u, v, w);
hold off
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211004113516216.png)

### 快捷键

### .1. 注释

`Ctrl+r`：选中要注释的多行文本，然后按`Ctrl+r`就可以实现多行注释。
`Ctrl+t`：选中已经注释了的多行文本，然后按`Ctrl+t`就可以取消多行注释。

### Resource

- [matlab-tutorial](https://github.com/alamgirh/matlab-tutorial)： Matlab Tutorial using Jupyter Notebook
- [Image Processing MATLAB Codes, Simulink, GUI, and Standalone Applications](https://github.com/Tes3awy/MATLAB-Tutorials )
- [awesome-matlab-robotics](https://github.com/mathworks-robotics/awesome-matlab-robotics): a list of awesome demos, tutorials, utilities and overall resources for the robotics community that use MATLAB and Simulink. 后期学习机器人可以好好学习使用



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/matlab-learning-record/  

