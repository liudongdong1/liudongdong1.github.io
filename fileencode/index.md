# FileEncode


> Unicode只是对信源编码，对字符集数字化，解决了字符到数字化的映射。
>
> UTF-32、UTF-16、UTF-8是信道编码，为更好的存储和传输。

### 1. 编码方式

> **GB2312**:简体中文编码，`一个汉字占用2字节`，在大陆是主要编码方式。当文章/网页中包含`繁体中文、日文、韩文等等时，这些内容可能无法被正确编码`。
>
> **BIG5: ** `繁体中文`编码。主要在台湾地区采用。
>
> **GBK**:  支持`简体及繁体中文`，但对他国非拉丁字母语言还是有问题。
>
> **UTF-8**:   Unicode编码的一种。Unicode用一些基本的保留字符制定了三套编码方式，它们分别UTF-8,UTF-16和UTF-32。在UTF－8中，字符是`以8位序列`来编码的，用一个或几个字节来表示一个字符。这种方式的最大好处，是UTF－8保留了ASCII字符的编码做为它的一部分。UTF-8俗称“万国码”，可以同屏显示多语种，一个汉字占用3字节。为了做到国际化，网页应尽可能采用UTF-8编码。
>
> **ASCII码：**用来表示`英文`，它使用一个字节表示具体字符，其中第一位规定为0，其他7位存储数据，（2^7）一共可以表示128个字符。

### 2.  查看文件编码

- **chardet**

> 在处理字符串时，常常会遇到不知道字符串是何种编码，如果不知道字符串的编码就不能将字符串转换成需要的编码。面对多种不同编码的输入方式，是否会有一种有效的编码方式？chardet是一个非常优秀的编码识别模块。

- **codecs**

> 在Python中，codecs模块提供了实现这些规则的方法，通过模块公开的方法我们能够方便地获取某种编码方式的Encoder和 Decoder工厂函数(Factory function)，以及StreamReader、StreamWriter和StreamReaderWriter类。

- **检测文件编码**

```python
#!-*- coding:utf-8 -*-
import chardet

f3 = open(file="word2.txt",mode='rb') # 以二进制模式读取文件
data = f3.read() # 获取文件内容
print(data) 
f3.close() # 关闭文件

result = chardet.detect(data) # 检测文件内容
print(result) # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}
```

- **批量化文件编码转换**

```python
import os
import sys
import codecs
import chardet
from subFunc_tools import *

#将路径下面的所有文件，从原来的格式变为UTF-8的格式

if __name__ == "__main__":
    path = r'D:\Code_Sources\Python_PyCharm\convert_GBK_UTF-8\test_txt'
    (list_folders, list_files) = list_folders_files(path)

    print("Path: " + path)
    for fileName in list_files:
        filePath = path + '\\' + fileName
        with open(filePath, "rb") as f:
            data = f.read()
            codeType = chardet.detect(data)['encoding']
            convert(filePath, codeType, 'UTF-8')

def convert(file, in_enc="GBK", out_enc="UTF-8"):
    """
    该程序用于将目录下的文件从指定格式转换到指定格式，默认的是GBK转到utf-8
    :param file:    文件路径
    :param in_enc:  输入文件格式
    :param out_enc: 输出文件格式
    :return:
    """
    in_enc = in_enc.upper()
    out_enc = out_enc.upper()
    try:
        print("convert [ " + file.split('\\')[-1] + " ].....From " + in_enc + " --> " + out_enc )
        f = codecs.open(file, 'r', in_enc)
        new_content = f.read()
        codecs.open(file, 'w', out_enc).write(new_content)
    # print (f.read())
    except IOError as err:
        print("I/O error: {0}".format(err))
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/fileencode/  

