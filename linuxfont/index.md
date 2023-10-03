# linuxfont


### 1. 查看&安装&显示字体

```python
cat /proc/version
fc-list :lang=zh-cn   #显示中文字体
sudo apt install -y --force-yes --no-install-recommends fonts-wqy-microhei  #下载安装字体
sudo apt install -y --force-yes --no-install-recommends ttf-wqy-zenhei
```

### 2. 词云绘制

```python
# coding:utf-8
import jieba  # 分词
import matplotlib.pyplot as plt  # 数据可视化
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS  # 词云
import numpy as np  # 科学计算
from PIL import Image  # 处理图片
def draw_cloud(text, graph, save_name):
    textfile = open(text).read()  # 读取文本内容
    wordlist = jieba.cut(textfile, cut_all=False)  # 中文分词
    space_list = " ".join(wordlist)  # 连接词语
    backgroud = np.array(Image.open(graph))  # 背景轮廓图
    mywordcloud = WordCloud(background_color="white",  # 背景颜色
                            mask=backgroud,  # 写字用的背景图，从背景图取颜色
                            max_words=100,  # 最大词语数量
                            stopwords=STOPWORDS,  # 停用词
                            font_path="simkai.ttf",  # 字体
                            max_font_size=200,  # 最大字体尺寸
                            random_state=50,  # 随机角度
                            scale=2,
                            collocations=False,  # 避免重复单词
                            )
    mywordcloud = mywordcloud.generate(space_list)  # 生成词云
    ImageColorGenerator(backgroud)  # 生成词云的颜色
    plt.imsave(save_name, mywordcloud)  # 保存图片
    plt.imshow(mywordcloud)  # 显示词云
    plt.axis("off")  # 关闭保存
    plt.show()
if __name__ == '__main__':
    draw_cloud(text="government.txt", graph="china_map.jpg", save_name='2019政府工作报告词云.png')
```

```python
font_path : string
# 字体路径，需要展现什么字体就把该字体路径+后缀名写上，如：font_path = '黑体.ttf'
width : int (default=400)
# 输出的画布宽度，默认为400像素
height : int (default=200)
# 输出的画布高度，默认为200像素
prefer_horizontal : float (default=0.90)
# 词语水平方向排版出现的频率，默认 0.9 （所以词语垂直方向排版出现频率为 0.1 ）
mask : nd-array or None (default=None)
# 如果参数为空，则使用二维遮罩绘制词云。如果 mask 非空，设置的宽高值将被忽略，遮罩形状被 mask 取代。除全白（#FFFFFF）的部分将不会绘制，其余部分会用于绘制词云。
# 如：bg_pic = imread('读取一张图片.png')，背景图片的画布一定要设置为白色（#FFFFFF），然后显示的形状为不是白色的其他颜色。可以用ps工具将自己要显示的形状复制到一个纯白色的画布上再保存，就ok了。
scale : float (default=1) 
# 按照比例进行放大画布，如设置为1.5，则长和宽都是原来画布的1.5倍。
min_font_size : int (default=4) 
# 显示的最小的字体大小
font_step : int (default=1)
# 字体步长，如果步长大于1，会加快运算但是可能导致结果出现较大的误差。
max_words : number (default=200)
# 要显示的词的最大个数
stopwords : set of strings or None
# 设置需要屏蔽的词，如果为空，则使用内置的STOPWORDS
background_color : color value (default=”black”)
# 背景颜色，如background_color='white',背景颜色为白色。
max_font_size : int or None (default=None)
# 显示的最大的字体大小
mode : string (default=”RGB”)
# 当参数为“RGBA”并且background_color不为空时，背景为透明。
relative_scaling : float (default=.5)
# 词频和字体大小的关联性
color_func : callable, default=None
# 生成新颜色的函数，如果为空，则使用 self.color_func
regexp : string or None (optional)
# 使用正则表达式分隔输入的文本
collocations : bool, default=True
# 是否包括两个词的搭配
colormap : string or matplotlib colormap, default=”viridis”
# 给每个单词随机分配颜色，若指定color_func，则忽略该方法。
wordcloud参数详解
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/linuxfont/  

