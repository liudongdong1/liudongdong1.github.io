# Matlabdraw


### 1. Matlab 绘图

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201206180344948.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210418201240885.png)

###  1.1. Plot 函数

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201206180821136.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210715224005116.png)

```matlab
x=-pi:pi/10:pi;                  %以pi/10为步长
y=tan(sin(x))-sin(tan(x));       %求出各点上的函数值
plot(x,y,'--rs',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','g',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
        
#用不同的线型和颜色在同一坐标内绘制曲线y=2e^{-0.5x}sin(2pi*x)及其包络线
x=[0:pi/100:2*pi]; %数据准备
y1=2*exp(-0.5*x);
y2=2*exp(-0.5*x).*sin(2*pi*x);
plot(x,y1,'k:',x,-y1,'k:',x,y2,'b--')   %y1和-y1为包络线
%将y1和－y1设置为黑色点线，y2设置为蓝色虚线
```

#### 1.2. 坐标轴轴刻度

> - `axis[xmin xmax ymin ymax] `: 对当前二维图形对象的x轴和y轴进行设置，其中x轴的刻度范围为[xmin xmax], y轴的刻度范围为[ymin ymax]。
> - `axis off `: 把坐标轴，刻度，标注和说明变为不显示状态。
> - `axis on` : 把坐标轴，刻度，标注和说明变为显示状态。
> - `axis manual `: `冻结当前的坐标轴比例`，以后添加绘图都在当前坐标轴范围内显示。
> - axis auto : 恢复`系统的自动定比例功能`。
> - axis equal : `等比例坐标轴`。
> - `axis nomal` : `自动调整纵横轴比例`，使当前坐标轴范围内的图形显示达到最佳效果。
> - axis square : 以当前坐标轴范围为基础，将坐标轴区域调整为方格形。
> - `set(gca,‘xtick’,标识向量)`：x坐标轴刻度数据点位置
> - set(gca,‘xticklabel’,‘字符串|字符串’)：x坐标轴刻度处显示的字符。
> - set(gca,‘FontName’,‘Times New Roman’,‘FontSize’,14)：设置坐标轴字体名称和大小。
>   注意：gca是坐标轴的handle,即标识辨识码

```matlab
clear ;close all; clc
t=[0:0.01:2*pi];            %定义时间范围
x=sin(t);
y=cos(t);
plot(x,y)
axis([-1.5 1.5 -1.5 1.5])    %限定X轴和Y轴的显示范围
grid on
axis('equal')

#set 函数使用
t=0:0.05:7;
plot(t,sin(t))
set(gca,'xtick',[0 1.4 3.14 5 6.28])
set(gca,'xticklabel','0|1.4|half|5|one')
```

#### 1.3. 标注图例

> - `title(‘String’)` ：在坐标系的顶部添加一个文本串String，作为图形的标题。
> - `xlabel(‘String’)`，`ylabel(‘String’)`：采用字符串给x轴，y轴标注。
> - `text(x,y,‘string’)`：在二维图形(x,y)位置处标注文本注释’string’。
> - `gtext(‘string’)`：拖动鼠标确定文字’string’的位置，再单击鼠标左键。
> - `figure`: 打开不同的图形窗，以绘制不同的图形。
> - `grid on`：对当前坐标轴添加网格。
> - `hold on`：保持当前图形窗口内容命令，防止图形被下个图形覆盖。
> - lenged([图例],‘String1’,‘String2’,…)：在当前图形中添加图例。
> - lenged(…,pos)：由pos确定图例标注位置。
>   pos = 0 表示放置图例在轴边界内；
>   pos = 1 表示放置图例在轴边界内右上角（为默认设置）；
>   pos = 2表示放置图例在轴边界内左上角；
>   pos = 3表示放置图例在轴边界内左下角；
>   pos = 4表示放置图例在轴边界内右下角。

```matlab
clc
close all
clear
x=0:0.05:10; %数据准备
y=zeros(1,length(x));%生成一个1行length(x)列的矩阵。
for n=1:length(x)
    if x(n)>=8
      y(n)=1;
    elseif x(n)>=6
      y(n)=5-x(n)/2
    elseif x(n)>=4
      y(n)=2
    else
      y(n)=sqrt(x(n))
    end
end
plot(x,y)%画出图形
axis([0 10 0 2.5])%设置坐标轴的范围
title('分段函数曲线') %添加图形标题
xlabel('x') %给x轴标注
ylabel('y')%给y轴标注
text(2,1.3, 'y=x^{1/2}')%在（2,1.3）处添加文本注释
text(4.5,1.9, 'y=2')%在（4.5,1.9）处添加文本注释
text(7.3,1.5, 'y=5-x/2');
text(8.5,0.9, 'y=1');
```

```matlab
close all
clc
clear                   
t=[0:pi/20:9*pi];       %定义时间范围
figure(1)               %建立图形窗口1
plot(t,sin(t),'r:*')    %以红色的点绘制正弦函数图形，图形的标识符为星
grid on                 %在所画出的图形坐标中添加栅格，注意用在plot之后
text(pi,0,' \leftarrow sin(\pi)','FontSize',18) %在（pi,0）处添加左箭头和sin(pi)文本标识,字体大小为18
title('添加栅格的正弦曲线')%添加图形标题
xlabel('x')             %添加x坐标轴标识
ylabel('sint')          %添加y坐标轴标识

figure(2)      %建立图形窗口2
plot(t,cos(t)) %绘制余弦函数图形
grid on        %打开网格
pause          %暂停
grid off       %删除栅格
text(pi,0,' \leftarrow cos(\pi)','FontSize',18)   %添加文本标识
title('去除栅格的余弦曲线')  %添加图形标题
xlabel('x')    %添加x坐标轴标识
ylabel('cost') %添加y坐标轴标识
```

#### 1.4. 函数绘图分割

> - `fplot(‘funtion’,limits)`：绘制函数曲线在一个指定范围内，limits可以为[xmin xmax]或者[xmin xmax ymin ymax]
> - `subplot(m,n,p)`：将一个图形窗口分成mxn个子窗口，从左往右，从上往下第p个子图形窗口。
> - `ezplot(f)`：f是关于x的函数（-2*pi<x<2*pi）
> - `ezplot(f(x,y))`：f(x,y) = 0,隐函数，（-2*pi<x<2*pi，-2*pi<y<2*pi）
> - `eaplot(f,[A B])`: f是关于x的函数（A<x<B）
> - `ezplot(f(x,y)`：[XMIN,XMAX,YMIN,YMAX]):f(x,y)=0,隐函数

```matlab
subplot(2,2,1),fplot('humps',[0 1])
subplot(2,2,2)
fplot('abs(exp(-j∗x∗(0:9))∗ones(10,1))',[0 2∗pi])
subplot(2,2,3)
fplot('[tan(x),sin(x),cos(x)]',2∗pi∗[-1 1 -1 1])
subplot(2,2,4)
fplot('sin(1./x)',[0.01 0.1],1e-3)

ezplot(‘sin(x)’)

syms x y
ezplot('x^3+y^3-3*x*y')
grid on 
```

#### 1.5.bar

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721113314785.png)

##### .1. twobars-3

```matlab
clear ;close all; clc;
index=1:3;
bar1=[83.65 98.82;92.50,99.13;93.33,99.72];

figure
b=bar(index, bar1);

% 设置条形图颜色#FFFFFF
set(b(1), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255);
set(b(2), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255);
hold on;    

% 左边y轴（柱状图）
%yyaxis left
set(gca,'YLim',[0 102.1]);%X轴的数bai据显示du范围
set(gca,'ytick',[0:10:102.1]);%设置要显示坐标刻度
ylabel('Measure Value','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
% 右边y轴（折线图）
%yyaxis right
%set(gca,'YLim',[0 102.1]);%X轴的数bai据显示du范围
%set(gca,'ytick',[0:10:102.1]);%设置要显示坐标刻度
%ylabel('Similarity Value','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注

legend({' -- AP ' ' -- PCK '},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])
xlabel('Different algorithm','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注

set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',index, 'xticklabel', { 'CPM' 'Hourglass' 'Ours'})
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210731195408107.png)

```matlab
figure
bar3(Z)
title('Detached Style')
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201219165024052.png)

##### .2. 3D bar

```matlab
clc %https://zhuanlan.zhihu.com/p/312069817
clear all
close all
m = 5; n = 3;
for i = 1:m
    for j = 1:n
        a(i,j) = rand(1) + (j-1)*0.7;
    end
end
b = bar3h(a);
b(1).FaceColor = [0.1 0.5 0.9];
b(2).FaceColor = [0.9 0.1 0.5];
b(3).FaceColor = [0.5 0.9 0.1];
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727151255478.png)

##### .3. twobars twoY

```matlab
clear all;clc;close all;


l2_bear =load('data2/3d/dataset_l2_backgournd.txt').';
l2_bottle =load('data2/3d/dataset_l2_Kinect.txt').';
l2_box =load('data2/3d/dataset_l2_occlusion.txt').';


sqrel_bear =load('data2/3d/dataset_sqrel_background.txt').';
sqrel_bottle =load('data2/3d/dataset_sqrel_kinect.txt').';
sqrel_box =load('data2/3d/dataset_sqrel_occlusion.txt').';


index = [1,2,3];
%y1 = [5.113, 5.354,5.736,5.923,0,0,0,0];

%y2 = [0,0,0,0,1.018,0.802,0.384,0.211];
y1 = [mean(l2_bear),-1,mean(l2_bottle),-1,mean(l2_box)];
y2 = [mean(sqrel_bear),-1,mean(sqrel_bottle),-1,mean(sqrel_box)];

figure;
% x1 = 0.1 :2 : 10
% x2 = 0.8 : 2 : 10

x1 = 0.1 :1 : 5
x2 = 0.8 : 1 : 5
[AX,H1,H2] = plotyy( x1 ,y1,x2,y2,'bar','bar');

set(H1, 'LineWidth',1.5,'BarWidth',0.7,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255)
set(H2,'LineWidth',1.5,'BarWidth',0.7,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255)
% set(H1,'Marker','*','Color','k','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',10);
% set(H2,'Marker','diamond','Color',[217,83,25]/255,'LineWidth',2,'MarkerEdgeColor',[217,83,25]/255,'MarkerFaceColor',[217,83,25]/255,'MarkerSize',10);

new_name={'Dataset1','Dataset2','Dataset3'}

set(AX(1), 'xtick',0.5:2:5, 'xticklabel', new_name);
set(AX(1),'ytick',0:1:7.1,'ylim',[0 7.1],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(AX(1),'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(1),'ylabel'),'string','RMSE','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

set(AX(2),'XTickLabel','')
set(AX(2),'FontName','Times New Roman','FontWeight','Bold','FontSize',18);
set(AX(2),'ytick',0:0.5:2.0,'ylim',[0 2.0],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(2),'ylabel'),'string','SqRel','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

legend(' RMSE',' SqRel');

xlabel('Different Algorithm','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
% set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On');
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210729160214117.png)

##### .4. twobars twoy-4

```matlab
clear all;clc;close all;


index = [1,2,3,4];
%y1 = [5.113, 5.354,5.736,5.923,0,0,0,0];

%y2 = [0,0,0,0,1.018,0.802,0.384,0.211];
x1 = 0.1 :1 :7
x2 = 0.8 : 1 : 7
y1 = [5.113,-1, 5.354,-1,5.736,-1,5.923];
y2 = [1.018,-1,0.802,-1,0.384,-1,0.211];

figure;
[AX,H1,H2] = plotyy(x1,y1,x2,y2,'bar','bar');
set(H1, 'LineWidth',1.5,'BarWidth',0.7,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255)
set(H2,'LineWidth',1.5,'BarWidth',0.7,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255)
% set(H1,'Marker','*','Color','k','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',10);
% set(H2,'Marker','diamond','Color',[217,83,25]/255,'LineWidth',2,'MarkerEdgeColor',[217,83,25]/255,'MarkerFaceColor',[217,83,25]/255,'MarkerSize',10);

new_name={'Baseline','FastDepth','NoHRnet','RC6D'}

set(AX(1), 'xtick',[0.45,2.45,4.45,6.45], 'xticklabel', new_name);
set(AX(1),'ytick',5:0.1:6.08,'ylim',[5 6],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(1),'ylabel'),'string','RMSE','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

set(AX(2),'XTickLabel','')
set(AX(2),'ytick',0:0.1:1.08,'ylim',[0 1.08],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(2),'ylabel'),'string','SqRel','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

legend(' RMSE',' SqRel','Position',[0.389880955999803,0.821031749579646,0.212499996381146,0.129761901214009]);

xlabel('Different Algorithm','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Position',[3.450004327297223,4.910204083978609,-0.999999999999986]) %给x轴标注
set(gca,'OuterPosition',[-0.02,0.052857142857143,1,1],'Position',[0.11,0.162857142857143,0.775,0.815]);
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210731195534119.png)

##### .5. twobars twoy-3

```matlab
clear ;close all; clc;
index=1:3;
bar1=[0.820,0.899;0.793,0.886;0.761,0.832];

figure
b=bar(index, bar1);

% 设置条形图颜色#FFFFFF
set(b(1), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255);
set(b(2), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255);
hold on;    

set(gca,'YLim',[0 1.1]);%X轴的数bai据显示du范围
set(gca,'ytick',[0:0.1:1.02]);%设置要显示坐标刻度
legend({' -- 5^o 5cm' ' -- 10^o 10cm '},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('RFID Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('Different datasets','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Average Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',index, 'xticklabel', { 'Dataset_1' 'Dataset_2' 'Dataset_3'})
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210731195612266.png)

##### .6. stacked bar

```
clear ;close all; clc
color=[
    [0.49,0.18,0.56]
    [0.15 0.15 0.15]
    [0.00,0.00,0.00]
    [76,57,107]/255
    [11 23 70]/255
    [135 51 36]/255
    [8 46 84]/255
    ];
Y =[46/50 4/50; 45/50 5/50; 47/50 3/50; 279/300 21/300; 0 50/50];
index=[1 2 3 4 5];
figure
b=bar(Y,'stacked', 'DisplayName', 'Fraction')
%'LineWidth',1.5,'BarWidth',0.6,'FaceColor',[128,100,162]/255,'EdgeColor',[0 0 0]/255,
set(b(1), 'LineWidth',1.5,'BarWidth',0.6,'FaceColor',[128,100,162]/255,'EdgeColor',[0 0 0]/255);
set(b(2), 'LineWidth',1.5,'BarWidth',0.6,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255);
set(gca,'YLim',[0. 1.02]);%X轴的数bai据显示du范围
set(gca,'ytick',[0.:0.1:1.02]);%设置要显示坐标刻度
% legend({'--Accuracy'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])
legend({' -- Correct Matching ' ' -- Error Matching '},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('RFID Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('Different Objects','FontName','Times New Roman','FontWeight','Bold','FontSize',15) %给x轴标注
ylabel('Matching Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',15)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',15,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick', 1:length(Y), 'xticklabel', {'Paper' 'Plastic' 'Glass' 'Tape' 'Iron'})
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220107103744216.png)

#### 1.6. color

##### 1. [imagesc 函数](https://ww2.mathworks.cn/help/matlab/ref/imagesc.html)

> 使用缩放颜色显示图像 ;使用场景 : 3D 图显示时 , 不是很直观 , 这里将色彩当做一个维度 , 使用颜色值作为 z zz 轴的深度 。

```matlab
% 生成 x , y 矩阵 , 
[x, y] = meshgrid(-3 : .2 : 3 , -3 : .2 : 3);
% 生成 z 矩阵
z = x .^ 2 + x .*y + y .^2;
% 第一个图形中绘制 x,y,z 组成的面
figure, surf(x, y, z);
% 第二个图形中绘制 z 值对应的颜色网格
% 对应的 z 的最大值对应颜色值 1 
% 对应的 z 的最小值对应颜色值 0
figure, imagesc(z);
% 查看 z 轴的颜色值
% 可以看到最小值 ~ 最大值 对应的颜色区间
colorbar;
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721152503303.png)

#### 2. [Colormaps 颜色图](https://ww2.mathworks.cn/help/matlab/ref/colormap.html)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721152647080.png)

```matlab
% 生成 x , y 矩阵 , 
[x, y] = meshgrid(-3 : .2 : 3 , -3 : .2 : 3);
% 生成 z 矩阵
z = x .^ 2 + x .*y + y .^2;
% 第一个图形中绘制 x,y,z 组成的面
figure, surf(x, y, z);
% 第二个图形中绘制 z 值对应的颜色网格
% 对应的 z 的最大值对应颜色值 1 
% 对应的 z 的最小值对应颜色值 0
figure, imagesc(z);
% 查看 z 轴的颜色值
% 可以看到最小值 ~ 最大值 对应的颜色区间
colorbar;
% 改变 z 值对应的颜色值
% 暖色系
colormap(hot);
% 第 3 个图形中绘制 z 值对应的颜色网格
figure, imagesc(z);
% 查看 z 轴的颜色值
% 可以看到最小值 ~ 最大值 对应的颜色区间
colorbar;
% 改变 z 值对应的颜色值
% 暖色系
colormap(cool);
% 第 4 个图形中绘制 z 值对应的颜色网格
figure, imagesc(z);
% 查看 z 轴的颜色值
% 可以看到最小值 ~ 最大值 对应的颜色区间
colorbar;
% 改变 z 值对应的颜色值
% 暖色系
colormap(cool);
% 改变 z 值对应的颜色值
% 灰度颜色
colormap(gray);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721152753532.png)

#### 1.7. 散点图

> - scatter(x,y,sz) 指定圆大小。要绘制大小相等的圆圈，请将 sz 指定为标量。要绘制大小不等的圆，请将 sz 指定为长度等于 x 和 y 的长度的向量。
> - scatter(x,y,sz,c) 指定圆颜色。要以相同的颜色绘制所有圆圈，请将 c 指定为颜色名称或 RGB 三元组。要使用不同的颜色，请将 c 指定为向量或由 RGB 三元组组成的三列矩阵。

```matlab
clear ;close all; clc;
%bend90 =load('data/filter/1617934269.1755493origin.txt');
%index=linspace(0,length(bend90(1,:)),length(bend90(1,:)));
kinectIndex=[0.202  0.201 0.203 0.208 0.200 0.198]
kinect =[5.736 5.792 5.834 5.9923 5.65 5.342];

backgroundIndex=[1.374 1.294 1.385  1.397 1.379 1.372]   % 1*length
background=[6.268 6.314 6.708 6.332 6.205 6.239]

occlusionIndex=[1.892 1.923 1.944 1.901 1.889 1.895]
occlusion=[7.204 7.703 7.225 7.625 7.199 7.213]

hold on;     
s1=scatter(kinectIndex,kinect,'Marker','o',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','m',...% 设置标记点内部填充颜色为绿色
        'SizeData',50);         %设置标记点大小为10      
hold on;
s2=scatter(backgroundIndex,background,'Marker','+',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','r',...% 设置标记点内部填充颜色为绿色
        'SizeData',50);         %设置标记点大小为10
hold on;    
s3=scatter(occlusionIndex,occlusion,'Marker','d',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','g',...% 设置标记点内部填充颜色为绿色
        'SizeData',50);         %设置标记点大小为10
hold on;
set(gca,'XLim',[0 2.5]);%X轴的数bai据显示du范围
set(gca,'YLim',[5 8.1]);%X轴的数bai据显示du范围
set(gca,'ytick',[5:0.5:8.1]);%设置要显示坐标刻度
legend({' Baseline' ' OursNoHRnet' ' Ours'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('RFID Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('SqRel Value','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('RMSE Value','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',index, 'xticklabel', index)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721201737720.png)

##### 1.  4plot

```matlab
clear ;close all; clc;
raw_time=[12.405807, 12.921454, 13.384166, 13.670475, 14.197112, 14.765131, 14.888588, 15.290715, 15.620094, 15.895999, 16.197627, 16.577454, 16.689153, 16.976529, 17.163652, 17.475315];
raw_phase=[5.408698912480956, 3.5863297364728144, 1.8437275614347257, 1.027649782279565, 5.875029071998191, 5.1264464475099984, 5.0834949854492, 5.034407600236859, 5.433242605087126, 5.770718378421967, 0.1931642336697763, 1.616698404827651, 1.99712564022329, 2.9175141129546747, 3.659960814291325, 5.304388218904732];
ref_time=[12.405807   12.50926635 12.61272569 12.71618504 12.81964439 12.92310373 13.02656308 13.13002243 13.23348178 13.33694112 13.44040047 13.54385982 13.64731916 13.75077851 13.85423786 13.9576972  14.06115655 14.1646159 14.26807524 14.37153459 14.47499394 14.57845329 14.68191263 14.78537198 14.88883133 14.99229067 15.09575002 15.19920937 15.30266871 15.40612806 15.50958741 15.61304676 15.7165061  15.81996545 15.9234248  16.02688414 16.13034349 16.23380284 16.33726218 16.44072153 16.54418088 16.64764022 16.75109957 16.85455892 16.95801827 17.06147761 17.16493696 17.26839631 17.37185565 17.475315  ];
ref_phase=[1.21075739 0.72152623 0.23846223 6.04522429 5.57595761 5.11440909 4.66118989 4.21696453 3.7824547  3.35844295 2.94577601 2.54536765 2.15820062 1.78532759 1.42787038 1.08701728 0.76401773 0.46017392 0.17682883 6.19853542 5.96029508 5.74664469 5.55888674 5.39824269 5.26581919 5.16257408 5.08928437 5.04651838 5.03461438 5.05366742 5.10352556 5.18379588 5.29385965 5.43289542 5.59990802 5.79376121 6.01321186 6.25694339 0.24041179 0.52861464 0.83700301 1.1642421 1.50904097 1.87016368 2.24643694 2.63675476 3.04008061 3.4554478 3.88195829 4.31878043];
ref_rssi=[-62.5, -59.0, -56.5, -56.0, -54.5, -54.0, -53.5, -55.0, -56.5, -58.0, -60.5, -63.0, -63.5, -64.5, -65.5, -71.0];


color=[
    [0.00,0.00,0.00]
    [70,105,0]/255
    [0 140 158]/255
    [202,0,0]/255
    [0.49,0.18,0.56]
    ];
    
figure
set(gcf,'Position',[-1823,413,650,406],'InnerPosition',[1823,413,650,406],'OuterPosition',[-1831,405,666,500]);
subplot(2,2,1);
s1=scatter(raw_time,raw_phase,'Marker','o',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(2,:),...% 设置标记点内部填充颜色为绿色
        'SizeData',50);         %设置标记点大小为10    
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',12:1:18, 'xticklabel', 12:1:18)
set(gca,'XLim',[12 18]);%X轴的数bai据显示du范围
set(gca,'YLim',[0 6.5]);%X轴的数bai据显示du范围
set(gca,'ytick',0:1:6.5);%设置要显示坐标刻度
xlabel('Time (s)','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Phase (rad)','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
set(gca,'Position',[0.088056268634168,0.630541871921182,0.388866808288909,0.341753653797864],'OuterPosition',[0.010415244781282,0.475862072929373,0.484969370053137,0.526957735376341]);
legend({' RawPhase'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

subplot(2,2,2)
hold on;  
p1=plot(ref_time,ref_phase,'--*',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(5,:),...% 设置标记点内部填充颜色为绿色
        'MarkerSize',6);         %设置标记点大小为10
s2=scatter(raw_time,raw_phase,'Marker','o',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(2,:),...% 设置标记点内部填充颜色为绿色
        'SizeData',50);         %设置标记点大小为10    
% s2=scatter(raw_time,raw_phase,'Marker','+',...              %绘制红色的虚线，且每个转折点上用正方形表示。
%         'LineWidth',1.5,...        % 设置线宽为2
%         'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
%         'MarkerFaceColor','r',...% 设置标记点内部填充颜色为绿色
%         'SizeData',50);         %设置标记点大小为10
set(gca,'Position',[0.560511942049691,0.630541871921182,0.388866808288909,0.341753653797864],'OuterPosition',[0.482870918196805,0.475862072929372,0.511728676727903,0.526957735376341]);
set(gca, 'xtick',12:1:18, 'xticklabel', 12:1:18)
set(gca,'XLim',[12 18]);%X轴的数bai据显示du范围
set(gca,'YLim',[0 6.5]);%X轴的数bai据显示du范围
set(gca,'ytick',0:1:6.5);%设置要显示坐标刻度
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
% set(gca, 'xtick',index, 'xticklabel', index)
xlabel('Time (s)','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Phase (rad)','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
legend({' RawPhase' ' RefPhase'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

subplot(2,2,3)
hold on;     
s3=scatter(raw_time,unwrap(raw_phase),'Marker','o',...              %绘制红色的虚线，且每个转折点上用正方形表示。
    'LineWidth',1.5,...        % 设置线宽为2
    'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
    'MarkerFaceColor',color(4,:),...% 设置标记点内部填充颜色为绿色
    'SizeData',50);         %设置标记点大小为10
set(gca,'Position',[0.088056268634168,0.140394088669951,0.388866808288909,0.341753653797865]);
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
% set(gca, 'xtick',index, 'xticklabel', index)
% set(gca,'YLim',[0 6.5]);%X轴的数bai据显示du范围
% set(gca,'ytick',0:1:6.5);%设置要显示坐标刻度
set(gca, 'xtick',12:1:18, 'xticklabel', 12:1:18)
set(gca,'XLim',[12 18]);%X轴的数bai据显示du范围
xlabel('Time (s)','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Phase (rad)','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
legend({' UnwrapPhase'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

subplot(2,2,4)
hold on;     

s4=scatter(raw_time,ref_rssi,'Marker','d',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(4,:),...% 设置标记点内部填充颜色为绿色
        'SizeData',50);         %设置标记点大小为10
    %0.560511942049691,0.630541871921182,
set(gca,'Position',[0.560511942049691,0.140394088669951,0.388866808288909,0.341753653797865]);
set(gca, 'xtick',12:1:18, 'xticklabel', 12:1:18)
set(gca,'XLim',[12 18]);%X轴的数bai据显示du范围
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
% set(gca, 'xtick',index, 'xticklabel', index)
xlabel('Time (s)','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('RSSI (dBm)','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
%set(gca,'XLim',[15 32.5]);%X轴的数bai据显示du范围
set(gca,'YLim',[-71 -50]);%X轴的数bai据显示du范围
set(gca,'ytick',-70:5:-50);%设置要显示坐标刻度
% legend({' Baseline' ' OursNoHRnet' ' Ours'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
% xlabel('Experiment Index','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
% ylabel('Inferenec Time (s)','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
% set(gca, 'xtick',index, 'xticklabel', index)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210731194829619.png)

##### 2. 3d 散点图

```matlab
clear ;close all; clc;
figure
set(gcf,'Position',[-1589,470,560,378]);
dataset1 =load('data2/6d/Dataset_background.txt').';
dataset2 =load('data2/6d/Dataset_kinect_rgb.txt').';
dataset3 =load('data2/6d/Dataset_occlusion.txt').';

h1=scatter3(dataset1(1,:),dataset1(3,:),dataset1(5,:));
hold on;
h2=scatter3(dataset2(1,:),dataset2(3,:),dataset2(5,:));
hold on;
h3=scatter3(dataset3(1,:),dataset3(3,:),dataset3(5,:));
hold on;


set(h1,'Marker','o','LineWidth',3,'MarkerEdgeColor','r', 'MarkerFaceColor','r','SizeData',20,'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0.6,'MarkerFaceAlpha',0.6);
set(h2,'Marker','o','LineWidth',3,'MarkerEdgeColor','k', 'MarkerFaceColor','k','SizeData',20,'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0.6,'MarkerFaceAlpha',0.6);
set(h3,'Marker','o','LineWidth',3,'MarkerEdgeColor','b', 'MarkerFaceColor','b','SizeData',20,'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0.6,'MarkerFaceAlpha',0.6);



%添加颜色 h4,h5
set(gca,'XLim',[0 100]);%X轴的数bai据显示du范围
set(gca,'YLim',[0 100]);%X轴的数bai据显示du范围
set(gca,'ZLim',[0 100]);%X轴的数bai据显示du范围
set(gca,'xtick',0:20:100);%设置要显示坐标刻度
set(gca,'ytick',0:30:100);%设置要显示坐标刻度
set(gca,'ztick',0:20:100);%设置要显示坐标刻度

legend({' Dataset1' ' Dataset2' ' Dataset3'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941, 0.941, 0.941],'EdgeColor',[0 0 0])
% 
% title('','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('X (degree)','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Position',[65.23307459303516,-2.550150916909843,-13.474845044505177]) %给x轴标注
ylabel('Y (degree)','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Position',[102.6537907595557,58.33406190540428,39.39523877580883])%给y轴标注
zlabel('Z (degree)','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Position',[-11.696611771018906,0.023584349678913,51.14747953877499]) %给x轴标注
set(gca,'OuterPosition',[-0.003655911139854,0.009889313396206,0.996127686089335,0.999399488552371],'Position',[0.156280981417408,0.189944254493367,0.741558663353586,0.804891295373477],'View',[1.06635071090048,11.87365126676602],'CameraPosition',[65.7729116460651,-797.3477256175736,228.1938471129981],'CameraViewAngle',7.848486072219547,'CameraTarget',[50,50,50],'CameraUpVector',[0,0,1],'CameraViewAngleMode','auto');

set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
axis xy
% set(gca, 'xtick',0:10:70, 'xticklabel', 0:10:70)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210731195901326.png)

#### 1.8. 3D图

##### 1. 3d 折线图

```matlab
clear ;close all; clc;
data=load('1622894838.6200619.txt');
data=data.';

index=linspace(0,length(data(:,1)), length(data(:,1))).';
y1=linspace(0,0,length(data(:,1))).';
y2=linspace(2,2,length(data(:,1))).';
y3=linspace(4,4,length(data(:,1))).';
y4=linspace(6,6,length(data(:,1))).';
y5=linspace(8,8,length(data(:,1))).';

figure
plot3(index,y1,data(:,1),'--rs',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','r',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','r',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
%bar(Y,'LineWidth',1.5,'BarWidth',0.6,'FaceColor',[128,100,162]/255,'EdgeColor',[0 0 0]/255);

hold on;  
plot3(index,y2,data(:,2),'--bd',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','b',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','b',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;     
plot3(index,y3,data(:,3),'--kp',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;   
plot3(index,y4,data(:,4),'--kp',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;   
plot3(index,y5,data(:,5),'--kp',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;   

%set(gca,'XLim',[15 32.5]);%X轴的数bai据显示du范围
%set(gca,'YLim',[0 1.1]);%X轴的数bai据显示du范围
%set(gca,'ytick',[0:0.1:1.02]);%设置要显示坐标刻度
legend({' Voltage id1' ' Voltage id2' ' Voltage id3' ' Voltage id4' ' Voltage id5'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('RFID Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('Index Number','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Flex Sensor Index','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
zlabel('Voltage Value','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
%set(gca, 'xtick',index, 'xticklabel', { '15' '17.5' '20' '22.5' '25' '27.5' '30' '32.5'})
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210714223054506.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211201183800038.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211201183828822.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211201183858395.png)

##### 2. 3d mesh网格

```matlab
% 生成 x 向量
x = -2 : 0.1 : 2;
% 生成 y 向量
y = -2 : 0.1 : 2;
% 生成 X Y 两个矩阵 
% 生成了 x-y 坐标轴上的网格
[X, Y] = meshgrid(x, y);
% 生成 Z 矩阵
Z = X .* exp (-X .^ 2 - Y .^ 2);
% 绘制网格
mesh(X, Y, Z);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721153026392.png)

##### 3. [3d surface](https://ww2.mathworks.cn/help/matlab/ref/surf.html)

> surf 函数作用是绘制平面 , 给网格填充颜色 ;

```matlab
% 生成 x 向量
x = -2 : 0.1 : 2;
% 生成 y 向量
y = -2 : 0.1 : 2;
% 生成 X Y 两个矩阵 
% 生成了 x-y 坐标轴上的网格
[X, Y] = meshgrid(x, y);
% 生成 Z 矩阵
Z = X .* exp (-X .^ 2 - Y .^ 2);
% 绘制平面
surf(X, Y, Z);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721153145424.png)

##### 4. 绘制网格 + 等高线

```matlab
% 生成 x 向量
x = -2 : 0.1 : 2;
% 生成 y 向量
y = -2 : 0.1 : 2;
% 生成 X Y 两个矩阵 
% 生成了 x-y 坐标轴上的网格
[X, Y] = meshgrid(x, y);
% 生成 Z 矩阵
Z = X .* exp (-X .^ 2 - Y .^ 2);
% 绘制网格 + 等高线
meshc(X, Y, Z);
grid on;
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721153255033.png)

##### 5. 绘制平面 + 等高线

```matlab
% 生成 x 向量
x = -2 : 0.1 : 2;
% 生成 y 向量
y = -2 : 0.1 : 2;
% 生成 X Y 两个矩阵 
% 生成了 x-y 坐标轴上的网格
[X, Y] = meshgrid(x, y);
% 生成 Z 矩阵
Z = X .* exp (-X .^ 2 - Y .^ 2);
% 绘制平面 + 等高线
surfc(X, Y, Z);
grid on;
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721153329077.png)

#### 1.9 星空图

##### 1. 极坐标图

```matlab
% 生成 1 ~ 100 之间的数 , 步长 1
x = 1 : 100;
% 绘制第 1 张极坐标图
subplot(2, 2, 1);
% 角度值向量
theta = x / 10;
% 半径值向量
r = log10(x);
% 绘制极坐标图
polar(theta, r);
% 绘制第 2 张极坐标图
subplot(2, 2, 2);
% 角度值向量
theta = linspace(0, 2 * pi);
% 半径值向量
r = cos(4 * theta);
% 绘制极坐标图
polar(theta, r);
% 绘制第 3 张极坐标图
subplot(2, 2, 3);
% 角度值向量
theta = linspace(0, 2 * pi, 6);
% 半径值向量
r = ones(1, length(theta));
% 绘制极坐标图
polar(theta, r);
% 绘制第 4 张极坐标图
subplot(2, 2, 4);
% 角度值向量
theta = linspace(0, 2 * pi);
% 半径值向量
r = 1 - sin(theta);
% 绘制极坐标图
polar(theta, r);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721150900686.png)

##### 2. 29图形图

```matlab
clear ;close all; clc

theta = linspace(0,2*pi);
rho = 2*theta;
figure
hange=polarplot(theta,rho);

set( hange, 'Color', 'magenta','LineStyle','-','LineWidth',1.5, 'Marker','+','MarkerSize',8,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',[0,0,0]);

title('My Polar Plot')

pax = gca;
% thetaticks(0:12:360)   % 设置 theta轴的刻度值
% thetaticklabels({'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3'})
pax.ThetaTick=linspace(0,360,30);
pax.TickLabelInterpreter='latex';
pax.ThetaTickLabel={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','$\heartsuit$','$\Delta$','$\square$'};

pax.ThetaAxisUnits = 'radians';     % 显示弧度，非角度
pax.ThetaDir = 'clockwise';        %使其沿顺时针方向增加
pax.ThetaZeroLocation = 'left';   %旋转theta轴，以使零参考角在左侧。

 rlim([-5 15])
 rticks([-2 3 9 15])
 rticklabels({'r = -2','r = 3','r = 9','r = 15'})
% pax.RLim([0 15])   %r轴的边界，使值的范围从-5到15
% pax.RTick([0 3 9 15])   %值-2、3、9和15处显示行
% pax.RTickLabel({'r = -2','r = 3','r = 9','r = 15'})  %更改每行旁边显示的标签
%设置ThetaColor``RColor的属性，为theta轴和r轴网格线以及关联的标签使用不同的颜色。通过
pax.ThetaColor = 'blue';
pax.RColor = [0 .5 0];
% pax.RLineStyle='-';
% pax.LineWidth=1.5;
% 设置box 颜色
pax.Box='on';
pax.LineWidth=1.5 ;      %设置LineWidth属性来更改网格线的宽度。
pax.GridColor = 'red';  %更改所有网格线的颜色，而不会影响标签。
%设置字体
pax.FontName='Times New Roman';
pax.FontWeight='Bold';
pax.TitleFontWeight='Bold';
pax.FontSize = 14;
```

##### 3. polar图例3

```matlab
% 角度值向量
theta = linspace(0, 2 * pi, 100000);
% 半径值向量
r = 0.03 * cos(60 * theta);
% 角度值向量
theta1 = linspace(0, 2 * pi, 100000);
% 半径值向量
r1 = 0.027+ 0.003 * cos(250 * theta1);
% 绘制极坐标图, 设置极坐标的最大值范围
polar(0, 0.04);
hold on;
% 绘制内层图像
polar(theta, r, 'b');
hold on;
% 绘制外层图像
polar(theta1, r1, 'b');
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721151642398.png)

#### 2.0 gif 动图绘制

```matlab
%{ 
简单框架：
clc
clear
pic_num = 1;
for ***************************
    plot(fig(i));
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);

    if pic_num == 1
    imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.2);

    else
    imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);

    end

    pic_num = pic_num + 1;
end
%}
clc;
clear;
pic_num = 1;
filename = strcat('.\img\gif_plot_test',datestr(now,30),'.gif'); % .\img 表示保存在Matlab当前打开的目录下的img文件夹下
for epsilon = 0.01:-0.001:0.005
    syms x;
    w = -tanh(x/(2*epsilon));
    figure(1);
    ezplot(w);
    axis([-0.05,0.05 -1.5 1.5])
    drawnow;
    
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if pic_num == 1
        imwrite(I,map,filename,'gif', 'Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,filename,'gif','WriteMode','append','DelayTime',0.2);
    end
    pic_num = pic_num + 1;
end
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/gif_plot_test20210715T221036.gif)

```matlab
%% 计算
clc;
clear;
% xa = -30:0.1:30;
% ya = 400:0.1:600;
% 不同区间范围图像显示不同
% xa = -5:0.1:5;
% ya = -20:0.1:20;

% xa = -5:0.01:5;
% ya = -50:0.1:80;

xa = -2:0.01:2;
ya = -5:0.1:8;

[x,y] = meshgrid(xa,ya);
z =100.*x.^4-250.*x.^2.*y+x.^2-2.*x+80.*y.^2+1;
%% 作图
mesh(x,y,z);
xlabel('X轴');  
ylabel('Y轴');  
zlabel('Z轴');  
title('函数三维图像');  
grid on %打开网格  
view(-100,30); % 不同角度看
% view(-30,30); 
% view(-90,90); %俯视图
%% 动图制作
pic_num = 1;
filename = strcat('.\img\gif_plot_example',datestr(now,30),'.gif'); % .\img 表示保存在Matlab当前打开的目录下的img文件夹下
% for i=-30:3:330
for i=-120:2:240
    %view(a,b):a是角度，b是仰视角  
    view(i,30);  
%     pause(0.1); 
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if pic_num == 1
        imwrite(I,map,filename,'gif', 'Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,filename,'gif','WriteMode','append','DelayTime',0.2);
    end
    pic_num = pic_num + 1;
end  
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/gif_plot_example20210715T220952.gif)

#### 2.1. 混合图

##### 1. OneFIgure

> 使用hold同时绘制多个图例；

```matlab
% 数据
Ncar=1:3;
% 柱状图数据
Norder1 = [800 1100 1250];
Norder2 = [420 550 625];
% 折线图数据
qcar1 = [1 2 3];
qcar2 = [2 4 6];

% 打开新图
figure;
hold on;

% 左边y轴（柱状图）
yyaxis left
b=bar(Ncar, [Norder1',Norder2']);
ylim([0 1500])
ylabel('Transfer speed','FontName','Times New Roman','FontWeight','Bold','FontSize',18)
set(b(1), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255);
set(b(2), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255);

% 标记数据到柱状图
offset_vertical = 40;   % 根据需要调整
offset_horizon = 0.15;  % 根据需要调整
for i = 1:length(Norder1)
    if Norder1(i)>=0
        text(i - offset_horizon,Norder1(i) + offset_vertical,num2str(Norder1(i)),'VerticalAlignment','middle','HorizontalAlignment','center','FontName','Times New Roman','FontWeight','Bold','FontSize',18);
    else
        text(i - offset_horizon,Norder1(i) - offset_vertical,num2str(Norder1(i)),'VerticalAlignment','middle','HorizontalAlignment','center','FontName','Times New Roman','FontWeight','Bold','FontSize',18);
    end
end
for i = 1:length(Norder2)
    if Norder1(i)>=0
        text(i + offset_horizon,Norder2(i) + offset_vertical,num2str(Norder2(i)),'VerticalAlignment','middle','HorizontalAlignment','center','FontName','Times New Roman','FontWeight','Bold','FontSize',18);
    else
        text(i + offset_horizon,Norder2(i) - offset_vertical,num2str(Norder2(i)),'VerticalAlignment','middle','HorizontalAlignment','center','FontName','Times New Roman','FontWeight','Bold','FontSize',18);
    end
end

% 右边y轴（折线图）
yyaxis right
plot(Ncar,qcar1,'--ks',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;  
plot(Ncar,qcar2,'--bd',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','b',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','b',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10

ylim([0 8])
ylabel('Energy consumption per hour','FontName','Times New Roman','FontWeight','Bold','FontSize',18)

% 图注
legend({'Low melting point crude oil transfer pipeline','High melting point crude oil transfer pipeline',...
    'Low melting point crude oil transfer pipeline','High melting point crude oil transfer pipeline'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0]);

% x轴

set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',index, 'xticklabel', [1 2 3 4])
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721143234513.png)

##### 2. OneFigure-Multi

> ```matlab
> subplot(m, n, 1);
> ```
>
> - m 参数 : 行数 ;
>
> - n 参数与 : 列数 ;
>
> - 第三个参数是 1 11 ~ m × n m \times nm×n 之间的数值 ; 在本示例中是 1 11 ~ 6 66 之间的数值 ;
> - `square 样式`表示的是`坐标轴的 x xx 轴和 y yy 轴长度相同` ;
> - `equal tight 样式`是在 equal 样式基础上 ,` 贴边切割有效曲线图形`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721145331183.png)

```matlab
% 生成 x 轴数据 , -10 ~ 10 , 步长 0.1
t = 0 : 0.1 : 2 * pi;

% x,y 轴变量
x = 3 * cos(t);
y = sin(t);
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');

% 在第一行第一列绘制图形, 坐标轴正常 normal
subplot(2,2,1);
plot(x,y,'--ks',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;  
axis normal 

% 在第一行第二列绘制图形, 坐标轴方形 square
subplot(2,2,2);
plot(x,y,'--ks',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;  
axis square

% 在第二行第一列绘制图形, 坐标轴 equal
subplot(2,2,3);
plot(x,y,'--ks',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;  
axis equal

% 在第二行第二列绘制图形, 坐标轴 equal tight
subplot(2,2,4);
plot(x,y,'--ks',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',2,...        % 设置线宽为2
        'MarkerEdgeColor','k',...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor','k',...% 设置标记点内部填充颜色为绿色
        'MarkerSize',10)         %设置标记点大小为10
hold on;  
axis equal tight




```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721145717302.png)

##### 3. Multi-figure

> ```matlab
> figure('Position', [left, bottom, width, height]);
> ```
>
> - **left 参数 :** 图形对话框在 Windows 界面中 , 距离屏幕左侧的距离 ;
> - **bottom 参数 :** 图形对话框在 Windows 界面中 , 距离屏幕底部的距离 ;
> - **width 参数 :** 图形对话框宽度 ;
> - **height 参数 :** 图形对话框高度 ;
> - 一个figure代表一个图；

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721145102350.png)

##### 4. 双y轴曲线绘制

```matlab
% x 轴取值
x = 0 : 0.01 : 20;

% 曲线 1 对应的 y 值
y1 = 200 * exp (-0.05 * x) .* sin(x);

% 曲线 2 对应的 y 值
y2 = 0.8 * exp (-0.5 * x) .* sin(10 * x);

% 使用 plotyy 绘制两条曲线
% AX 是坐标系 axis 句柄值
% 曲线 1 的句柄值是 H1
% 曲线 2 的句柄值是 H2
[AX, H1, H2] = plotyy(x, y1, x, y2);

% 修改坐标轴标注 , 通过 AX 句柄值设置左右两侧 y 轴标注

%set(H1, 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255)
%set(H2,'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255)
set(H1,'Marker','*','Color','k','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',10);
set(H2,'Marker','diamond','Color',[217,83,25]/255,'LineWidth',2,'MarkerEdgeColor',[217,83,25]/255,'MarkerFaceColor',[217,83,25]/255,'MarkerSize',10);

new_name={'Bear','Bottle','Box','Milk','Disinfectant'}

set(AX(1), 'xtick',index, 'xticklabel', new_name);
%set(AX(1),'ytick',5:0.1:6.08,'ylim',[5 6],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(1),'ylabel'),'string','RMSE','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

set(AX(2),'XTickLabel','')
%set(AX(2),'ytick',0:0.1:1.08,'ylim',[0 1.08],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor',[217,83,25]/255);
set(get(AX(2),'ylabel'),'string','SqRel','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color',[217,83,25]/255)

legend(' RMSE','SqRel');

xlabel('Different Algorithm','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721150320538.png)

#### 2.2. Latex

```matlab
% 使用 latex 语法
set(gca, 'XTickLabel', {'0', '\pi / 2', '\pi', '3\pi/2', '2\pi'});
```

#### 2.3. pie

```matlab
% 饼图的数值列表
x = [1, 2, 5, 4, 8];

% 绘制饼图 , 绘制时根据数值自动分配百分比
% 后面跟着 有 x 相同个数的向量 ,
% 0 元素代表默认 
% 1 元素代表分离出来 
pie(x, [0, 0, 1, 0, 1]);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721150734974.png)

```matlab
% 饼图的数值列表
x = [1, 2, 5, 4, 8];

% 绘制 3D 饼图 , 绘制时根据数值自动分配百分比
% 后面跟着 有 x 相同个数的向量 ,
% 0 元素代表默认 
% 1 元素代表分离出来 
pie3(x, [0, 0, 1, 0, 1]);
```

#### 2.4. [箱型图](https://ww2.mathworks.cn/help/stats/boxplot.html)

##### 1. boxchart

```matlab
clc %https://zhuanlan.zhihu.com/p/312069817
clear all
close all
a = randi([0 1000], 110, 1);
a(101:105,1) = randi([1000 2000],5,1);
a(106:110,1) = randi([-1000 0],5,1);
x = int16(rand(110,1));
x = categorical(x);
b = boxchart(x,a);
b.MarkerStyle = '+';
b.MarkerColor = 'r';
b.BoxFaceColor = [0.5 0.1 0.9];
b.WhiskerLineColor = [0.2 0.6 0.4];
b.Notch = 'on';
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727150558440.png)

```matlab
% 加载数据
% 不同国家中每加仑汽油能跑多少英里
load carsmall

% MPG 是箱线图数据 
% Origin 中包含多个分组变量
boxplot(MPG, Origin);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721151325681.png)

```matlab
set(0,'defaultfigurecolor','w')
 
%errorbar函数实例
figure;
subplot(2,2,1);
%横轴
x = 1:10:100;
%均值
y = [20 30 45 40 60 65 80 75 95 90];
%标准差
err = 8*ones(size(y));
%线型，颜色，线宽，标记大小
errorbar(x,y,err,'-*b','LineWidth',1','MarkerSize',8) 
xlabel('Time');ylabel('Production');
%设置坐标轴字体大小粗细，字体样式以及横纵轴范围
set(gca,'fontsize',10,'fontweight','bold','FontName','Times New Roman','XLim',[0,120],'YLim',[0,120]);
  
subplot(2,2,2);
x = 1:10:100;
y = [20 30 45 40 60 65 80 75 95 90];
err1 = 10*ones(size(y));
err2 = 10*rand(size(y));
errorbar(x,y,err1,err2,'*b','LineWidth',1','MarkerSize',8) 
xlabel('Time');ylabel('Production');
title('标题','fontsize',10,'fontweight','bold');
%设置坐标轴字体大小粗细，字体样式以及横纵轴范围
set(gca,'fontsize',10,'fontweight','bold','FontName','Times New Roman','XLim',[0,120],'YLim',[0,120]);
 
subplot(2,2,3)
Average1=[12,11,7,7,6,5];
Variance1=[0.5,0.4,0.3,1,0.3,0.5];     %A地的数据
Average2=[10,8,5,4,3,3];
Variance2=[0.4,0.3,0.4,0.6,0.3,0.5];    %B地的数据
Time=1:1:6;
errorbar(Time,Average1,Variance1,'r-o')    %A地误差棒图，用红色线表示
hold on
errorbar(Time,Average2,Variance2,'b-s')    %B地误差棒图，用蓝色线表示
xlabel('Time');ylabel('Production');
 
subplot(2,2,4);
Average2=[120,110,70,70,60,50];
Variance2=[15,14,8,10,9,9];     %A地的数据
Average3=[100,80,50,40,30,30];
Variance3=[14,8.3,9.4,10.6,13,15];    %B地的数据
Time=1:1:6;
errorbar(Time,Average2,Variance2,'ro')    %A地误差棒图，用红色线表示
hold on
errorbar(Time,Average3,Variance3,'bs','MarkerSize',10,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')    %B地误差棒图，用蓝色线表示
xlabel('Time');ylabel('Production');
set(gca,'fontsize',10,'fontweight','bold','FontName','Times New Roman','XLim',[0,8],'YLim',[0,140]);
grid on;
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727141008591.png)

```matlab
clear ;close all; clc;



twomethods =load('data/RFIDCV_1.txt').';


mycolors=[
    [59,98,145]/255
    [35, 46, 82]/255
    [59, 59, 59]/255
    [198, 1, 1]/255
    [64, 32, 32]/255
    [102, 51, 51]/255
    ];
mysyms=['o' '*' 'h' 'p' 's' 'd'];

figure
subplot(2,1,1);
axis tight
set(gcf,'position',[1435,242,462,709],'InnerPosition',[1435,242,462,709],'OuterPosition',[-1443,234,478,803])

hold on;
h1=cdfplot(twomethods(1,:));
hold on;
h2=cdfplot(twomethods(3,:));

set(h1,'LineStyle', '--', 'Color', mycolors(1,:),'LineWidth',2,'MarkerEdgeColor', mycolors(1,:), 'MarkerFaceColor', mycolors(1,:) );
set(h2,'LineStyle', '--', 'Color', mycolors(2,:),'LineWidth',2,'MarkerEdgeColor',mycolors(2,:), 'MarkerFaceColor',mycolors(2,:));

%绘制做差部分
hold on;
h4=cdfplot(twomethods(2,:));
hold on;
h5=cdfplot(twomethods(4,:));
hold on;


set(h4,'LineStyle', '-', 'Color', mycolors(4,:),'LineWidth',2,'MarkerEdgeColor', mycolors(4,:), 'MarkerFaceColor', mycolors(4,:) );
set(h5,'LineStyle', '-', 'Color', mycolors(5,:),'LineWidth',2,'MarkerEdgeColor',mycolors(5,:), 'MarkerFaceColor',mycolors(5,:));


set(gca,'XLim',[0 10]);%X轴的数bai据显示du范围
set(gca,'YLim',[0 1.02]);%X轴的数bai据显示du范围
set(gca,'ytick',0:0.1:1.02);%设置要显示坐标刻度
legend([h1,h2,h4,h5],{' Fusion method in metric1' ' RFID method in metric1'  ' Fusion method in metric2' ' RFID method in metric2'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])
% 
title('','FontName','Times New Roman','FontWeight','Bold','FontSize',23) %添加图形标题
xlabel('Order Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',20) %给x轴标注
ylabel('CDF','FontName','Times New Roman','FontWeight','Bold','FontSize',20)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',0:1:10, 'xticklabel', 0:1:10);
%set(gca,'Position',[0.150876161467803,0.590037706763018,0.751785069243823,0.330659992730043]);


subplot(2,1,2);

index=[1,2];
barvalue=[mean(twomethods(1,:)),mean(twomethods(2,:));mean(twomethods(3,:)),mean(twomethods(4,:))];
speed_1y=[mean(twomethods(1,:)),mean(twomethods(3,:))];
speed_2y=[mean(twomethods(2,:)),mean(twomethods(4,:))];

error_speed_1y=[var(twomethods(1,:)),var(twomethods(3,:))];
error_speed_2y=[var(twomethods(2,:)),var(twomethods(4,:))];
hold on;
b=bar(index, barvalue);

% 设置条形图颜色#FFFFFF
set(b(1), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255);
set(b(2), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255);
hr1=errorbar([0.845,1.845],speed_1y,error_speed_1y,'-*b','LineWidth',1','MarkerSize',8); 
hold on;
hr2=errorbar([1.155,2.155],speed_2y,error_speed_2y,'-*b','LineWidth',1','MarkerSize',8);

set(hr1,'Color',[0,0,0]/255,'LineStyle','--', 'LineWidth',2,'CapSize',20,'Marker','diamond','MarkerSize',8,'MarkerFaceColor',[59,98,145]/255,'MarkerEdgeColor',[0 0 0]/255);
set(hr2,'Color',[0,0,0]/255,'LineStyle','--', 'LineWidth',2,'CapSize',20,'Marker','diamond','MarkerSize',8,'MarkerFaceColor',[148,60,57]/255,'MarkerEdgeColor',[0 0 0]/255);
%set(gca,'YLim',[0 1.1]);%X轴的数bai据显示du范围
%set(gca,'ytick',0:0.1:1.02);%设置要显示坐标刻度
legend([hr1,hr2],{' metric1' ' metric2'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('RFID Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('Different Methods','FontName','Times New Roman','FontWeight','Bold','FontSize',20) %给x轴标注
ylabel('','FontName','Times New Roman','FontWeight','Bold','FontSize',20)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',[1,2], 'xticklabel', { 'Fusion Method' 'RFID Method' });
set(gca,'OuterPosition',[0.024618624088152,0.054709302325581,0.972797100455081,0.445494390484461],'Position',[0.151082247147313,0.145446962599467,0.753917752852688,0.333647548335875]);

%axis tight
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727133854878.png)

#### 2.5. Error Bar 误差条线图

```matlab
% 生成 x 向量, 0 ~ pi , 步长 pi / 10
x = 0 : pi / 10 : pi;
% 生成 y 轴的值对应向量
y = sin(x);
% 生成 e 向量 , 表示每个对应 x 位置的误差范围
e = std(y) * ones(size(x));
% 绘制含误差条的线图
% e 表示误差范围
errorbar(x, y, e);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727144745498.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727144801702.png)

#### 2.7. stairs阶梯图

> - plot 函数绘制图像时 , 是将两点之间使用线连接起来 ;
> - stairs 函数绘制图像时 , 是`将两点之间使用阶梯线连接起来` ;
> - plot 与 stairs 绘图的大致形状相同 , 只是 stairs 是阶梯型的线 ;

```matlab
% 生成 0 ~ 4 * pi 之间的 40 个点
x = linspace(0, 4 * pi, 40);
y = sin(x);

% 绘制阶梯图
stairs(y);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721151846058.png)

#### 2.8. Stem 离散序列数据图

```matlab
% 生成 0 ~ 4 * pi 之间的 40 个点
x = linspace(0, 4 * pi, 40);
y = sin(x);
% 绘制 Stem 离散序列数据图
stem(y);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721151922635.png)

##### .1. 采样图

```matlab
% 同时在一个坐标系中绘制多个图
hold on;
% 生成 0 ~ 10 之间的 500 个点
% 生成 500 个点 , 保证曲线平滑
t = linspace(0, 10, 500);
y = sin(pi * t.^2 / 4);
% 绘制函数曲线
plot(t, y);
% 生成 50 个采样点 , 500 个点中采 50 个样本
sample_t = linspace(0, 10, 50);
sample_y = sin(pi * sample_t.^2 / 4);
% 绘制 Stem 离散序列数据图
stem(sample_t, sample_y);
hold off;
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721152024992.png)

#### 2.9. ROC 曲线绘制

##### 1. ROC曲线

```matlab
clc; clear; close all;

FPR = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.050,0.15,0.15,0.15,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.25,0.25,0.30,0.35,0.40,0.45,0.45,0.55,0.60,0.65,0.65,0.70,0.70,0.70,0.70,0.70,0.70,0.75,0.75,0.80,0.85,0.90,0.90,0.90,0.95,0.95,0.95,1];
TPR = [0,0,0.050,0.050,0.050,0.050,0.10,0.15,0.15,0.15,0.15,0.15,0.15,0.20,0.25,0.30,0.35,0.35,0.40,0.45,0.45,0.50,0.55,0.60,0.60,0.60,0.60,0.60,0.60,0.65,0.70,0.80,0.85,0.85,0.85,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];

acurve=trapz(FPR,TPR);%计算AUC

x_dig=0:0.01:1;
y_dig=x_dig;

color1=[0.9300    0.4700    0.5000];%红色
color2=[86 180 232]./255;
color3=[229 159 1]./255;
color4=[107,105,102]./255;
h=figure;
set(h,'units','normalized','position',[0.1 0.1 0.48 0.8]);%设置绘图窗口的大小
set(h,'color','w');%设置绘图窗口的背景为白色
h1=plot(FPR, TPR,'Color',color2,'LineWidth',3,'MarkerSize',3);hold on;
plot(x_dig,y_dig,'--','Color',color4,'LineWidth',1.5);%画中间的虚线
xlabel('False Positive Ratio (1-specificity)','fontsize',2,'FontWeight','bold');%x轴
ylabel('True Positive Ratio (Sensitivity)','fontsize',2,'FontWeight','bold');%y轴
set(gca,'YLim',[0,1.02]);
set(gca,'XLim',[-0.01,1.01]);

set(gca,'FontSize',24,'LineWidth',1.6)
set(get(gca,'YLabel'),'FontSize',24);
set(get(gca,'XLabel'),'FontSize',24);
set(gca,'YTick',[0:0.2:1])

grid on

ROCtitle_1=['AI AUC = ',num2str(roundn(acurve,-3))];
hh=legend([h1],ROCtitle_1)%,'Location','southeast');
set(hh,'edgecolor','white')
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721164019932.png)

##### 2. xy轴颠倒的ROC曲线

```matlab
function [handleCDF,stats] = cdfFunction(x)
%CDFPLOT Display an empirical cumulative distribution function.
%   CDFPLOT(X) plots an empirical cumulative distribution function (CDF) 
%   of the observations in the data sample vector X. X may be a row or 
%   column vector, and represents a random sample of observations from 
%   some underlying distribution.
%
%   H = CDFPLOT(X) plots F(x), the empirical (or sample) CDF versus the
%   observations in X. The empirical CDF, F(x), is defined as follows:
%
%   F(x) = (Number of observations <= x)/(Total number of observations)
%
%   for all values in the sample vector X. If X contains missing data
%   indicated by NaN's (IEEE arithmetic representation for
%   Not-a-Number), the missing observations will be ignored.
%
%   H is the handle of the empirical CDF curve (a Handle Graphics 'line'
%   object). 
%
%   [H,STATS] = CDFPLOT(X) also returns a statistical summary structure
%   with the following fields:
%
%      STATS.min    = minimum value of the vector X.
%      STATS.max    = maximum value of the vector X.
%      STATS.mean   = sample mean of the vector X.
%      STATS.median = sample median (50th percentile) of the vector X.
%      STATS.std    = sample standard deviation of the vector X.
%
%   In addition to qualitative visual benefits, the empirical CDF is 
%   useful for general-purpose goodness-of-fit hypothesis testing, such 
%   as the Kolmogorov-Smirnov tests in which the test statistic is the 
%   largest deviation of the empirical CDF from a hypothesized theoretical 
%   CDF.
%
%   See also QQPLOT, KSTEST, KSTEST2, LILLIETEST.

% Copyright 1993-2011 The MathWorks, Inc.


% Get sample cdf, display error message if any
[yy,xx,~,~,eid] = cdfcalc(x);
if isequal(eid,'VectorRequired')
    error(message('stats:cdfplot:VectorRequired'));
elseif isequal(eid,'NotEnoughData')
    error(message('stats:cdfplot:NotEnoughData'));
end

% Create vectors for plotting
k = length(xx);
n = reshape(repmat(1:k, 2, 1), 2*k, 1);
xCDF    = [-Inf; xx(n); Inf];
yCDF    = [0; 0; yy(1+n)];

%
% Now plot the sample (empirical) CDF staircase.
%
%hCDF = plot(xCDF,yCDF);  %这里进行了修改
hCDF = plot(yCDF,xCDF);
if (nargout>0), handleCDF=hCDF; end
grid  ('on')
xlabel(getString(message('stats:cdfplot:LabelX')))
ylabel(getString(message('stats:cdfplot:LabelFx')))
title (getString(message('stats:cdfplot:Title')))

%
% Compute summary statistics if requested.
%

if nargout > 1
   stats.min    =  min(x);
   stats.max    =  max(x);
   stats.mean   =  mean(x);
   stats.median =  median(x);
   stats.std    =  std(x);
end
```

```matlab
clear ;close all; clc;



speed0_1 =load('data/RFscanner.txt').';
speed0_2 =load('data/STPP.txt').';
speed0_3 =load('data2/RFIDCV_1.txt').';  %bottle(1,:)  RCInv
mycolors=[
    [70,105,0]/255
    [0 140 158]/255
    [202,0,0]/255
    ];
mysyms=['o' '*' 'h' 'p' 's' 'd'];

figure
subplot(2,1,1);

set(gcf,'position',[1435,242,462,709],'InnerPosition',[1435,242,462,709],'OuterPosition',[-1443,234,478,803])

hold on;
h1=cdfFunction(speed0_1(1,:));
hold on;
h2=cdfFunction(speed0_2(1,:));
hold on;
h3=cdfFunction(speed0_3(1,:))
set(h1,'LineStyle', '-', 'Color', mycolors(1,:),'LineWidth',2,'MarkerEdgeColor', mycolors(1,:), 'MarkerFaceColor', mycolors(1,:) );
set(h2,'LineStyle', '-', 'Color', mycolors(2,:),'LineWidth',2,'MarkerEdgeColor',mycolors(2,:), 'MarkerFaceColor',mycolors(2,:));
set(h3,'LineStyle', '-', 'Color', mycolors(3,:),'LineWidth',2,'MarkerEdgeColor',mycolors(3,:), 'MarkerFaceColor',mycolors(3,:));
%绘制做差部分
hold on;
h4=cdfFunction(speed0_1(2,:));
hold on;
h5=cdfFunction(speed0_2(2,:));
hold on;
h6=cdfFunction(speed0_3(2,:))

set(h4,'LineStyle', '--', 'Color', mycolors(1,:),'LineWidth',2,'MarkerEdgeColor', mycolors(1,:), 'MarkerFaceColor', mycolors(1,:) );
set(h5,'LineStyle', '--', 'Color', mycolors(2,:),'LineWidth',2,'MarkerEdgeColor',mycolors(2,:), 'MarkerFaceColor',mycolors(2,:));
set(h6,'LineStyle', '--', 'Color', mycolors(3,:),'LineWidth',2,'MarkerEdgeColor',mycolors(3,:), 'MarkerFaceColor',mycolors(3,:));

set(gca,'XLim',[0 1]);%X轴的数bai据显示du范围
set(gca,'YLim',[0.4 1.01]);%X轴的数bai据显示du范围
set(gca,'ytick',0.4:0.1:1.01);%设置要显示坐标刻度
legend([h3,h6,h1,h4,h2,h5],{' RCInv in NKTD'   ' RCInv in NVD' ' RFscanner in NKTD' ' RFscanner in NVD' ' STPP in NKTD' ' STPP in NVD' },'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])
% 
title('','FontName','Times New Roman','FontWeight','Bold','FontSize',23) %添加图形标题
xlabel('Percentile','FontName','Times New Roman','FontWeight','Bold','FontSize',20) %给x轴标注
ylabel('Order Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',20)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',0:0.10:1, 'xticklabel', 0:10:100);
%set(gca,'Position',[0.150876161467803,0.590037706763018,0.751785069243823,0.330659992730043]);


subplot(2,1,2);

index=[1,2,3];
barvalue=[mean(speed0_2(1,:)),mean(speed0_2(2,:));mean(speed0_1(1,:)),mean(speed0_1(2,:));mean(speed0_3(1,:)),mean(speed0_3(2,:))];
speed_1y=[mean(speed0_2(1,:)),mean(speed0_1(1,:)),mean(speed0_3(1,:))];
speed_2y=[mean(speed0_2(2,:)),mean(speed0_1(2,:)),mean(speed0_3(2,:))];

error_speed_1y=[var(speed0_1(1,:)),var(speed0_2(1,:)),var(speed0_3(1,:))];
error_speed_2y=[var(speed0_1(2,:)),var(speed0_2(2,:)),var(speed0_3(2,:))];
hold on;
b=bar(index, barvalue);

% 设置条形图颜色#FFFFFF
set(b(1), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',mycolors(1,:),'EdgeColor',[0 0 0]/255);
set(b(2), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',mycolors(2,:),'EdgeColor',[0 0 0]/255);
hr1=errorbar([0.855,1.855,2.855],speed_1y,error_speed_1y,'-*b','LineWidth',1','MarkerSize',8); 
hold on;
hr2=errorbar([1.145,2.145,3.145],speed_2y,error_speed_2y,'-*b','LineWidth',1','MarkerSize',8);

set(hr1,'Color',mycolors(1,:),'LineStyle','-', 'LineWidth',2,'CapSize',20,'Marker','diamond','MarkerSize',8,'MarkerFaceColor',mycolors(1,:),'MarkerEdgeColor',[0 0 0]/255);
set(hr2,'Color',mycolors(2,:),'LineStyle','--', 'LineWidth',2,'CapSize',20,'Marker','diamond','MarkerSize',8,'MarkerFaceColor',mycolors(2,:),'MarkerEdgeColor',[0 0 0]/255);%set(gca,'YLim',[0 1.1]);%X轴的数bai据显示du范围
set(gca,'YLim',[0.8 1.01]);%X轴的数bai据显示du范围
set(gca,'ytick',0.8:0.05:1.01);%设置要显示坐标刻度
legend([hr1,hr2],{' NKTD' ' NVD'},'Location','best','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0])

% title('RFID Recognition Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
xlabel('Method','FontName','Times New Roman','FontWeight','Bold','FontSize',20) %给x轴标注
ylabel('Order Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',20)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',[1,2,3], 'xticklabel', { 'STPP' 'RFscanner'  'RCInv'});
set(gca,'OuterPosition',[7.085867e-9,0.05770920262021,0.994231804690922,0.445494490189832],'Position',[0.177056272347252,0.14910561274245,0.727943727652748,0.335057177955224]);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210731195217953.png)

#### 3.0. Area

```matlab
area(Y)
area(X,Y)
area(...,basevalue)
area(...,Name,Value)
area(ax,...)
ar = area(...)
```

> - `area(Y) `绘制向量 Y 或将矩阵 Y 中`每一列作为单独曲线绘制并堆叠显示`。x 轴自动缩放到 1:size(Y,1)
> - `area(X,Y)` 绘制 Y 对 X 的图，并填充 0 和 Y 之间的区域。X 的值可以是数值、日期时间、持续时间或分类值
>   - 如果 Y 是向量，则将 X 指定为由递增值组成的向量，其长度等于 Y。如果 X 的值不增加，则 area 将在绘制之前对值进行排序。
> - area(...,basevalue) `指定区域填充的基值`。默认 basevalue 为 0。将基值指定为数值。

```matlab
clc %https://zhuanlan.zhihu.com/p/312069817
clear all
close all
x = [0:0.01:pi];
y(:,1) = sin(x);
y(:,2) = abs(cos(x));
subplot(1,2,1)
h = area(x,y,-0.5,'linestyle','none');
h(1).FaceColor = [0.3 0.8 0.8];
h(2).FaceColor = [0.6 0.2 0.6];
xlim([0,pi])
subplot(1,2,2)
h = area(x,y,-0.5,'linestyle','none');
h(1).FaceColor = [1 1 1];
h(2).FaceColor = [0.6 0.2 0.6];
xlim([0,pi])
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727150010104.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727145959991.png)

#### 3.1. stackedplot

```matlab
clc %https://zhuanlan.zhihu.com/p/312069817
clear all
close all
x = [1:0.1:10];
y(:,1) = sin(x);
y(:,2) = cos(x);
y(:,3) = sin(x).*cos(x);
h = stackedplot(x,y,'r-');
h.DisplayLabels = {'y1','y2','y3'};
h.XLabel = {'x-axis'};
h.LineProperties(1).Color = 'b';
h.LineProperties(1).LineWidth = 1.5;
h.LineProperties(2).LineStyle = 'none';
h.LineProperties(2).Marker = 'o';
h.LineProperties(2).MarkerSize = 8;
h.LineProperties(3).PlotType = 'stairs';
h.LineProperties(3).Color = 'k';
h.LineProperties(3).LineWidth = 1.5;
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727150149662.png)

#### 3.2. histogram

```matlab
clc 
clear all
close all
a = randn(10000,1)-2;
h1 = histogram(a,21);
b = randn(10000,1);
hold on
h2 = histogram(b,21);
c = randn(10000,1)+2;
hold on
h3 = histogram(c,21);
xlim([-6,6])
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727150336565.png)

```matlab
clc %https://zhuanlan.zhihu.com/p/312069817
clear all
close all
a = randn(50000,1);
b = randn(50000,1);
subplot(2,2,1)
h1 = histogram2(a,b,[10,10],'FaceColor','flat');
xlim([-4,4])
ylim([-4,4])
subplot(2,2,2)
h2 = histogram2(a,b,[25,25],'FaceColor','flat');
xlim([-4,4])
ylim([-4,4])
subplot(2,2,3)
h3 = histogram2(a,b,[10,10],'FaceColor','flat');
axis([-4,4,-4,4])
subplot(2,2,4)
h4 = histogram2(a,b,[25,25],'FaceColor','flat');
axis([-4,4,-4,4])
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727150410027.png)

#### 3.3. binscatter

```matlab
clc %https://zhuanlan.zhihu.com/p/312069817
clear all
close all
a = randn(1e6,1);
b = randn(1e6,1);
subplot(2,2,1)
b1 = binscatter(a,b);
axes2 = subplot(2,2,2);
b2 = binscatter(a,b);
colormap(axes2,'parula');
axes3 = subplot(2,2,3);
b3 = binscatter(a,b);
colormap(axes3,'jet');
axes4 = subplot(2,2,4);
b4 = binscatter(a,b);
colormap(axes4,'hot');
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727150833248.png)

#### 3.4. wordcloud

```matlab
clc 
clear all
close all
a = string(fileread('1.txt'));
punctuationCharacters = ["." "?" "!" "," ";" ":"];
a = replace(a,punctuationCharacters," ");
a = split(join(a));
name = a(:,1);
m = length(name);
s = rand(m,1);
c = rand(m,3);
figure(1)
wordcloud(name,s);
figure(2)
wordcloud(name,s,'color',c);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727151155150.png)

#### 3.5 plot折线图

- 双折线图

```matlab
clear all;clc;close all;


l2_bear =load('data2/3d/dataset_l2_backgournd.txt').';
l2_bottle =load('data2/3d/dataset_l2_Kinect.txt').';
l2_box =load('data2/3d/dataset_l2_occlusion.txt').';


sqrel_bear =load('data2/3d/dataset_sqrel_background.txt').';
sqrel_bottle =load('data2/3d/dataset_sqrel_kinect.txt').';
sqrel_box =load('data2/3d/dataset_sqrel_occlusion.txt').';


index = [1,2,3];
%y1 = [5.113, 5.354,5.736,5.923,0,0,0,0];

%y2 = [0,0,0,0,1.018,0.802,0.384,0.211];
y1 = [mean(l2_bear),-1,mean(l2_bottle),-1,mean(l2_box)];
y2 = [mean(sqrel_bear),-1,mean(sqrel_bottle),-1,mean(sqrel_box)];

figure;
% x1 = 0.1 :2 : 10
% x2 = 0.8 : 2 : 10

x1 = 0.1 :1 : 5
x2 = 0.8 : 1 : 5
[AX,H1,H2] = plotyy( x1 ,y1,x2,y2,'bar','bar');

set(H1, 'LineWidth',1.5,'BarWidth',0.7,'FaceColor',[59,98,145]/255,'EdgeColor',[0 0 0]/255)
set(H2,'LineWidth',1.5,'BarWidth',0.7,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255)
% set(H1,'Marker','*','Color','k','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',10);
% set(H2,'Marker','diamond','Color',[217,83,25]/255,'LineWidth',2,'MarkerEdgeColor',[217,83,25]/255,'MarkerFaceColor',[217,83,25]/255,'MarkerSize',10);

new_name={'Dataset1','Dataset2','Dataset3'}

set(AX(1), 'xtick',0.5:2:5, 'xticklabel', new_name);
set(AX(1),'ytick',0:1:7.1,'ylim',[0 7.1],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(AX(1),'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(1),'ylabel'),'string','RMSE','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

set(AX(2),'XTickLabel','')
set(AX(2),'FontName','Times New Roman','FontWeight','Bold','FontSize',18);
set(AX(2),'ytick',0:0.5:2.0,'ylim',[0 2.0],'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'YColor','k');
set(get(AX(2),'ylabel'),'string','SqRel','FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Color','k')

legend(' RMSE',' SqRel');

xlabel('Different Algorithm','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
% set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On');
```

- 折线图

```matlab
clear all;clc;close all;

color=[
    [0.00,0.00,0.00]
    [76,57,107]/255
    [11 23 70]/255
    [135 51 36]/255
    [8 46 84]/255
    ];

bend90 =load('data/filter/1617934269.1755493origin.txt');
index=linspace(0,length(bend90(1,:)),length(bend90(1,:)));

figure;
p1=plot(index,bend90(1,:),'--*',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'Color',color(1,:),...% 设置线条颜色为黑色
        'MarkerEdgeColor',color(1,:),...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(1,:),...% 设置标记点内部填充颜色为绿色
        'MarkerSize',5);         %设置标记点大小为10   
hold on;
p2=plot(index,bend90(2,:),'--*',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'Color',color(2,:),...% 设置线条颜色为黑色
        'MarkerEdgeColor',color(2,:),...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(2,:),...% 设置标记点内部填充颜色为绿色
        'MarkerSize',5);         %设置标记点大小为10  
hold on;
p3=plot(index,bend90(3,:),'--*',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'Color',color(3,:),...% 设置线条颜色为黑色
        'MarkerEdgeColor',color(3,:),...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(3,:),...% 设置标记点内部填充颜色为绿色
        'MarkerSize',5);         %设置标记点大小为10
hold on;
p4=plot(index,bend90(4,:),'--*',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'Color',color(4,:),...% 设置线条颜色为黑色
        'MarkerEdgeColor',color(4,:),...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(4,:),...% 设置标记点内部填充颜色为绿色
        'MarkerSize',5);         %设置标记点大小为10   
hold on;
p5=plot(index,bend90(5,:),'--*',...              %绘制红色的虚线，且每个转折点上用正方形表示。
        'LineWidth',1.5,...        % 设置线宽为2
        'Color',color(5,:),...% 设置线条颜色为黑色
        'MarkerEdgeColor',color(5,:),...% 设置标记点边框线条颜色为黑色
        'MarkerFaceColor',color(5,:),...% 设置标记点内部填充颜色为绿色
        'MarkerSize',5);         %设置标记点大小为10   
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
%set(gca, 'xtick',12:1:18, 'xticklabel', 12:1:18)
set(gca,'XLim',[0 100]);%X轴的数bai据显示du范围
set(gca,'YLim',[320 430]);%X轴的数bai据显示du范围
%set(gca,'ytick',0:1:6.5);%设置要显示坐标刻度
xlabel('Index','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Voltage (V)','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
%set(gca,'Position',[0.088056268634168,0.630541871921182,0.388866808288909,0.341753653797864],'OuterPosition',[0.010415244781282,0.475862072929373,0.484969370053137,0.526957735376341]);
legend([p1,p2],{' Flex 1' ' Flex 2'},'Location','best','Orientation','horizontal','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0]);
axesNew = axes('position',get(gca,'position'),'visible','off');
% 绘制第二个图例时指定在新建的坐标系中
set(axesNew,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
legend(axesNew,[p3,p4,p5],{ ' Flex 3' ' Flex 4' ' Flex 5'},'Location','best','Orientation','horizontal','FontName','Times New Roman','FontWeight','Bold','FontSize',14,'Box','On','Color',[0.941 0.941 0.941],'EdgeColor',[0 0 0]);
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211210152734579.png)

### 2. 属性编辑器

> 更改编辑任何你所能想到的图片元素的属性， 包括曲线，坐标轴， 网格~等等之类。**所有API编程能实现的画图效果， 可视化工具都能实现。**

#### 0. set属性修改

- gcf: 这个figure属性
- gca: 一个坐标轴属性，可以理解为figure中的一个图例；
- `% set(gca,'looseInset',[0 0 0 0]) %去掉多余白边`
- **图形属性 :**字体，字体大小， 曲线粗细， 坐标范围限制， 坐标轴的刻度，  坐标轴刻度标签，
- **绘图对象** : 在`绘图结果 Figure 1 窗口`中 , `工具栏下面的区域中显示的任何可见组件都是绘图对象` ;
  - 图形对象 : `完整的画布` 了
  - 坐标轴对象 : `图像中的 x y xyxy 坐标轴` ;
  - 线对象 : `在坐标轴中绘制的曲线` ;
- **层次结构 :** `图形对象`中`包含坐标轴对象` , `坐标轴对象`中包含了` 线 , 文本 , 刻度 等对象 `;
  - 图形
    - 坐标轴
      - 线
      - 文本
      - 刻度
- `使用代码，查找是哪一个对象，并通过set函数进行属性设置`, 也可以`通过图形属性，坐标轴属性，更多属性进行GUI方式编辑；`

```matlab
%set(句柄值 , 属性名称1 , 属性值1 , 属性名称2, 属性值2, … 属性名称n, 属性值n) ;
% 使用 h 变量接受 plot 函数绘制的曲线图像句柄值
h = plot(x, y);

% 设置 h 变量对应的线对象
% 线的样式是 -.
% 线宽 5 像素
% 线颜色 红色
set(h, 'LineStyle', '-.', 'LineWidth', 5.0, 'Color', 'r');
```

```matlab
set(b(2), 'LineWidth',1.5,'BarWidth',0.8,'FaceColor',[148,60,57]/255,'EdgeColor',[0 0 0]/255);
set(gca,'YLim',[0 1.1]);%X轴的数bai据显示du范围
set(gca,'ytick',[0:0.1:1.02]);%设置要显示坐标刻度
xlabel('Different algorithm','FontName','Times New Roman','FontWeight','Bold','FontSize',18) %给x轴标注
ylabel('Average Accuracy','FontName','Times New Roman','FontWeight','Bold','FontSize',18)%给y轴标注
set(gca,'linewidth',1.5,'FontName','Times New Roman','FontWeight','Bold','FontSize',18,'Box','On','XGrid','on','YGrid','on');
set(gca, 'xtick',index, 'xticklabel', { 'Bottle' 'Milk' 'Disinfectant'})
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201229195809601.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210721112908346.png)

- **图形对象属性**

```shell
set(句柄，属性名1，属性值1，属性名2，属性值2，……）
%其中句柄指明要操作的对象
%set中的属性可以全部缺省
x=0:pi/10:2*pi;
h=plot(x,sin(x));
set(h,'Color','b','LineStyle',':','Marker','p');
```

```matlab
句柄变量=figure(属性名1，属性值1，属性名2，属性值2);
%属性名和属性值可以缺省，则命令如下
句柄变量=figure  或者  figure
figure(窗口句柄)       %设置为当前窗口。
      %如果窗口句柄写成一个整数，则可以使用这个句柄生成一个新的图形窗口，并定义为当前窗口。
close(窗口句柄)        %关闭图形窗口
close all;                     %关闭全部图形窗口
clf;                              %清除当前图形窗口的内容，但不关闭窗口。
# figure 对象设置
x=linspace(0.2*pi,60);
y=sin(x);
hf=figure('Color',[0,1,0],'Position',[1,1,450,250],'Name','Fuck','NumberTitle','off','MenuBar','none','KeyPressFcn','plot(x,y);axis([0,2*pi,-1,1]);');
```

- **坐标轴设置**

> 句柄变量=axes(属性名1，属性值1，属性名2，属性值2，……）;
> %调用axes函数用制动的属性在当前图形窗口创建坐标轴，并将句柄赋给句柄变量。
>
> - Box属性：取值是on或者off（缺省值）。它决定坐标轴是否带有边框。
> - GridLineStyle属性：取值是‘:’（缺省值）、‘-’、‘-.’、‘--’、‘none’。该属性定义网格线的类型
> - Position属性：该属性是由四个元素构成的向量，其形式为[n1,n2,n3,n4]。这个向量在图形窗口中决定一个矩形区域，坐标轴在其中。(n1,n2)是左下角的坐标，(n3,n4)是矩形的宽和高。单位由Units属性决定
> - Unit属性：取值是normalized（相对单位，为缺省值）、inches（英寸）、centimeters（厘米）和points（磅）。
> - Title属性：该属性的取值是坐标轴标题文字对象的句柄，可以通过该属性对坐标轴标题文字对象进行操作。
> - XLim、YLim、ZLim属性：取值都是具有2个元素的数值向量。3个属性分别定义个坐标轴的上下限。缺省为[0,1]。
> - XLabel、YLabel、ZLabel属性：
> - XScale、YScale、ZScale属性：取值都是’linear’（缺省值）或’log’，这些属性定义个坐标轴的刻度类型
> - View属性： 取值是两个元素的数值向量，定义视点方向。

- **曲线对象**

> - Color属性：该属性的取值是代表某颜色的字符或者RGB值。定义曲线的颜色。
>
>
> - LineStyle属性：定义线性
>
>
> - LineWidth属性：定义线宽，缺省值为0.5磅。
>
>
> - Marker属性：定义数据点标记符号，缺省值为none
>
>
> - MarkerSize属性：定义数据点标记符号的大小，缺省值为6磅。
>
>
> - XData,YData,Zdata属性：取值都是数值向量或矩阵，分别代表曲线对象的3个坐标轴数据。

- **文字对象**

> 句柄变量=text(x,y,z,'说明文字',属性名1，属性值1，属性名2，属性值2，……）;
> %说明文字可以使用LaTeX控制字符
>
> - Color属性：定义文字对象的颜色。
> - String属性：取值是字符串或者字符串矩阵，记录文字标注的内容。
> - Interpreter属性：取值是latex(缺省值)或none，该属性控制对文字标注内容的解释方式，即LaTeX方式或者ASCII方式;
> - FontSize属性：定义文字对象的大小，缺省值为10磅。
> - Rotation属性：取值是数值量，缺省值为0.定义文字对象的旋转角度。取正值是表示逆时针旋转。
> - 曲面对象: 句柄变量=surface(x,y,z,属性名1，属性值1，属性名2，属性2，……);
>
> - EdgeColor属性: 取值是代表某颜色的字符或RGB值，还可以是flat、interp或者none。缺省为黑色。定义曲面网格线的颜色或着色方式
>
> - FaceColor属性：取值是代表某颜色的字符或RGB值，还可以是flat（缺省值），interp或none。定义曲面网格片的颜色或着色方式
>
> - LineStyle属性：定义曲面网格线的线型
>
> - LineWidth属性：定义曲面网格线的线宽，缺省值为0.5磅。
>
> - Marker属性：曲面数据点标记符号，缺省值为none。
>
> - MarkerSize属性：曲面数据点标记符号的大小，缺省值为6磅。
>
> - XData,YData,ZData属性: 3种属性的取值都是数值向量或矩阵，分别代表曲面对象的3个坐标轴数据。
>
> - latex 代码支持

```matlab
text('Interpreter','latex','String','$$\sqrt{x^2+y^2}$$','Position',[.5.5],… 'FontSize',16);
set(gca,'FontName','Times New Roman','FontSize',18)%设置坐标轴刻度字体名称，大小
xlabel('往返相移', 'fontsize', 20)%x坐标，设置坐标轴文字大小
ylabel('透射率', 'fontsize', 20)%y坐标，设置坐标轴文字大小
```

#### 1. 坐标轴属性

- 开关网格 : grid on/off
- 开关 box : box on/off , 坐标轴的 下方是 x xx 轴 , 左侧是 y yy 轴 , 上方和右侧是 box ;
- 开关坐标轴 : axis on/off
- 普通坐标轴 : axis normal , 默认坐标轴样式 ;
- square 坐标轴 : axis square , 坐标轴的 x xx 轴范围与 y yy 轴范围相等 ;
- equal 坐标轴 : axis equal , x xx 轴单位刻度与 y yy 轴单位刻度长度相等 , 最能体现出实际的曲线 ;
- equal tight 坐标轴 : axis equal tight , 在 equal 坐标轴基础上 , 将曲线剪切出来 ;

### 3. matlab 特殊绘图包

#### 1. [PlotPub](https://github.com/masumhabib/PlotPub) [tutorial](https://github.com/masumhabib/PlotPub/wiki/Tutorial-for-v1.4)

```matlab
clear all;


% load previously generated fig file
figFile = 'multiple.fig';
open(figFile)


% change properties
opt.XLabel = 'Time, t (ms)'; % xlabel
opt.YLabel = 'Voltage, V (V)'; %ylabel
opt.YTick = [-10, 0, 10]; %[tick1, tick2, .. ]
opt.XLim = [0, 80]; % [min, max]
opt.YLim = [-11, 11]; % [min, max]

opt.Colors = [ % three colors for three data set
    1,      0,       0;
    0.25,   0.25,    0.25;
    0,      0,       1;
    ];

opt.LineWidth = [2, 2, 2]; % three line widths
opt.LineStyle = {'-', '-', '-'}; % three line styles
opt.Markers = {'o', '', 's'};
opt.MarkerSpacing = [15, 15, 15];
opt.Legend = {'\theta = 0^o', '\theta = 45^o', '\theta = 90^o'}; % legends

% Save? comment the following line if you do not want to save
opt.FileName = 'plotMarkers.png'; 

% create the plot
setPlotProp(opt);
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727142304340.png)

#### 2. box

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727142846425.png)

#### 3. applyhatch_plusC

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727144151627.png)

#### 4. [gramm](https://github.com/piermorel/gramm)

> Gramm is a` data visualization toolbox for Matlab` that allows to produce publication-quality plots from grouped data easily and flexibly, gramm improves `Matlab's plotting functionality`, allowing to generate complex figures using high-level object-oriented code. 

```matlab
load carbig.mat %Load example dataset about cars
origin_region=num2cell(org,2); %Convert origin data to a cellstr

% Create a gramm object, provide x (year of production) and y (fuel economy) data,
% color grouping data (number of cylinders) and select a subset of the data
g=gramm('x',Model_Year,'y',MPG,'color',Cylinders,'subset',Cylinders~=3 & Cylinders~=5)
% Subdivide the data in subplots horizontally by region of origin
g.facet_grid([],origin_region)
% Plot raw data as points
g.geom_point()
% Plot linear fits of the data with associated confidence intervals
g.stat_glm()
% Set appropriate names for legends
g.set_names('column','Origin','x','Year of production','y','Fuel economy (MPG)','color','# Cylinders')
%Set figure title
g.set_title('Fuel economy of new cars between 1970 and 1982')
% Do the actual drawing
g.draw()
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727154841148.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727154904113.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727154935209.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727155026814.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727155040033.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727155101749.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161134085.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161613294.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161636279.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727160903652.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161159054.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161010487.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161457723.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161511564.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161526981.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161035333.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161048835.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161102268.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161242377.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161258481.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161415158.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210727161431531.png)

### 4. RGB颜色

- [颜色选择器](http://tools.jb51.net/color/jPicker)
- [常用RGB颜色表](https://blog.csdn.net/kc58236582/article/details/50563958)
- [常用RGB颜色表2](https://blog.csdn.net/cll1224666878/article/details/106570306?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control)

### 5. Resource

- https://zhuanlan.zhihu.com/p/344457531
- [matlab工具箱](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fmath.cumt.edu.cn%2F_upload%2Farticle%2Ffiles%2F0c%2F51%2Ff02bf76b4f06ad4652ba3818e1c4%2F90c825c4-382c-473f-9be8-c54074793153.pdf)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/matlabdraw/  

