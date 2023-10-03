# Crawl_record


### 1. HTMLParse

#### 1.1. XML 方式

> lxml是python的一个解析库，支持HTML和XML的解析，支持XPath解析方式，而且解析效率非常高 XPath，全称XML Path Language，即XML路径语言，它是一门在XML文档中查找信息的语言，它最初是用来搜寻XML文档的，但是它同样适用于HTML文档的搜索 XPath的选择功能十分强大，它提供了非常简明的路径选择表达式，另外，它还提供了超过100个内建函数，用于字符串、数值、时间的匹配以及节点、序列的处理等，几乎所有我们想要定位的节点，都可以用XPath来选择

##### 1.1.1. 匹配规则

| 表达式            | 描述                                        |
| :---------------- | :------------------------------------------ |
| nodename          | 选取此节点的所有子节点                      |
| /                 | 从当前节点选取直接子节点                    |
| //                | 从当前节点选取子孙节点                      |
| .                 | 选取当前节点                                |
| ..                | 选取当前节点的父节点                        |
| @                 | 选取属性，[@id="main"]，[@class="job-list"] |
| *                 | 通配符，选择所有元素节点与元素名            |
| @*                | 选取所有属性                                |
| [@attrib]         | 选取具有给定属性的所有元素                  |
| [@attrib='value'] | 选取给定属性具有给定值的所有元素            |
| [tag]             | 选取所有具有指定元素的直接子节点            |
| [tag='text']      | 选取所有具有指定元素并且文本内容是text节点  |

##### 1.1.2. 文本html 解析输出

```python
#可以通过浏览器复制xml路径或取
html=etree.parse('test.html',etree.HTMLParser()) #指定解析器HTMLParser会根据文件修复HTML文件中缺失的如声明信息
#-----  或者
html=etree.HTML(text) #初始化生成一个XPath解析对象
result=etree.tostring(html,encoding='utf-8')   #解析对象输出代码
print(type(html))
print(type(result))
print(result.decode('utf-8'))
# -------  属性匹配
html=etree.HTML(text,etree.HTMLParser())
#选取class为item-1的li节点
result=html.xpath('//li[@class="item-1"]')
# -------  文本获取
result=html.xpath('//li[@class="item-1"]/a/text()') #获取a节点下的内容
#---------  属性获取
result=html.xpath('//li/a/@href')  #获取a的href属性
```

##### 1.1.3. 属性多值匹配

```python
from lxml import etree

text1='''
<div>
    <ul>
         <li class="aaa" name="item"><a href="link1.html">第一个</a></li>
         <li class="aaa" name="fore"><a href="link2.html">second item</a></li>
     </ul>
 </div>
'''
html=etree.HTML(text1,etree.HTMLParser())
result=html.xpath('//li[@class="aaa" and @name="fore"]/a/text()')
result1=html.xpath('//li[contains(@class,"aaa") and @name="fore"]/a/text()')
print(result)
print(result1)
#
['second item']
['second item']（11）XPath中的运算符
#---------示例二
from lxml import etree

text1='''
<div>
    <ul>
         <li class="aaa" name="item"><a href="link1.html">第一个</a></li>
         <li class="aaa" name="item"><a href="link1.html">第二个</a></li>
         <li class="aaa" name="item"><a href="link1.html">第三个</a></li>
         <li class="aaa" name="item"><a href="link1.html">第四个</a></li> 
     </ul>
 </div>
'''

html=etree.HTML(text1,etree.HTMLParser())

result=html.xpath('//li[contains(@class,"aaa")]/a/text()') #获取所有li节点下a节点的内容
result1=html.xpath('//li[1][contains(@class,"aaa")]/a/text()') #获取第一个
result2=html.xpath('//li[last()][contains(@class,"aaa")]/a/text()') #获取最后一个
result3=html.xpath('//li[position()>2 and position()<4][contains(@class,"aaa")]/a/text()') #获取第一个
result4=html.xpath('//li[last()-2][contains(@class,"aaa")]/a/text()') #获取倒数第三个

```

##### 1.1.4.  运算符

| 运算符 | 描述             | 实例              | 返回值                                         |
| :----- | :--------------- | :---------------- | :--------------------------------------------- |
| or     | 或               | age=19 or age=20  | 如果age等于19或者等于20则返回true反正返回false |
| and    | 与               | age>19 and age<21 | 如果age等于20则返回true，否则返回false         |
| mod    | 取余             | 5 mod 2           | 1                                              |
| \|     | 取两个节点的集合 | //book \| //cd    | 返回所有拥有book和cd元素的节点集合             |
| +      | 加               | 6+4               | 10                                             |
| -      | 减               | 6-4               | 2                                              |
| *      | 乘               | 6*4               | 24                                             |
| div    | 除法             | 8 div 4           | 2                                              |
| =      | 等于             | age=19            | true                                           |
| !=     | 不等于           | age!=19           | true                                           |
| <      | 小于             | age<19            | true                                           |
| <=     | 小于或等于       | age<=19           | true                                           |
| >      | 大于             | age>19            | true                                           |
| >=     | 大于或等于       | age>=19           | true                                           |

#### 1.2. Json 方式

- 使用工具https://www.sojson.com/ 查看json 格式，找到索引的key
- 或者直接把json 文件用vscode打开，注意后缀是.json;

> JSON 的全称是 JavaScript Object Notation，即 JavaScript 对象符号，它是一种轻量级、跨平台、跨语言的数据交换格式，其设计意图是把所有事情都用设计的字符串来表示，这样既方便在互联网上传递信息，也方便人进行阅读。

- json.dumps(): 对数据进行编码。
- json.loads(): 对数据进行解码。

- **编码类型转换**

  - **JSON 解码为 Python 类型转换对应表**

  | JSON          | Python |
  | :------------ | :----- |
  | object        | dict   |
  | array         | list   |
  | string        | str    |
  | number (int)  | int    |
  | number (real) | float  |
  | true          | True   |
  | false         | False  |
  | null          | None   |

  - **Python 编码为 JSON 类型转换对应表**：

| Python                                 | JSON   |
| :------------------------------------- | :----- |
| dict                                   | object |
| list, tuple                            | array  |
| str                                    | string |
| int, float, int- & float-derived Enums | number |
| True                                   | true   |
| False                                  | false  |
| None                                   | null   |

```python
import json
# 创建字典类型Person
person = {
    'name': '知秋小梦',
    'gender': 'male',
    'age': 18
}
# Python字典类型转换为JSON对象
json_person = json.dumps(person)
print(json_person)

# 写入 JSON 数据
with open('data.json', 'w') as f:
    json.dump(data, f)
 
# 读取数据
with open('data.json', 'r') as f:
    data = json.load(f)
```

#### 1.3. 正则表达式

https://www.runoob.com/regexp/regexp-syntax.html

- **仅匹配精确短语**![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029132849384.png)

- **匹配列表中的字词或短语**![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029132950911.png)
- **匹配包含不同拼写或特殊字符的字词**![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029133113269.png)
- **匹配某个特定网域的所有电子邮件地址**![image-20201029133313415](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029133313415.png)

#### 1.4. 存储

##### 1.4.1. [panda.DataFrame](https://pbpython.com/pandas-list-dict.html)

- 数据存储，加载，参考https://pbpython.com/pandas-list-dict.html

> 使用pandas的DataFrame to_csv方法实现csv文件输出，但是中文乱码，已验证的正确的方法是：
>
> df.to_csv("cnn_predict_result.csv",**encoding="utf_8_sig"**)
>
> 关于utf-8与**utf_8_sig的区别：**
>
> UTF-8以字节为编码单元，它的字节顺序在所有系统中都是一様的，没有字节序的问题，也因此它实际上并不需要BOM(“ByteOrder Mark”)。但是UTF-8 with BOM即utf-8-sig需要提供BOM。
>
> 1）程序输出中出现乱码的原因是因为python2中中文编码的问题，需要注意的是要将处理的中文文件的编码和python源文件的编码保持一致，这样不会出现中文乱码。可以参考这两篇文章[关于Python脚本开头两行的：#!/usr/bin/python和# -*- coding: utf-8 -*-的作用 – 指定](http://blog.csdn.net/xw_classmate/article/details/51933904)和[Python中用encoding声明的文件编码和文件的实际编码之间的关系](http://blog.csdn.net/xw_classmate/article/details/51933851)
>
> 2）在程序中能够正常输出中文，但是导出到文件后使用excel打开是出现中文乱码是因为excel能够正确识别用gb2312、**gbk**、gb18030或**utf_8 with BOM** 编码的中文，如果是**utf_8 no BOM**编码的中文文件，excel打开会乱码。

```
  def parse_html(self, html):
        '''
            html: [{jobid},{job_href},{job_name},{jobwelf},{attribute_text},{providesalary_text},{company_name},{companysize_text},{companyind_text},{company_href},{degreefrom},{companytype_text},{workarea_text},{updatedate}]
            解析一页中html文件，获得如上文件个数数据
        '''
        items = []   #python 列表的使用
        for one in html:
            item={}
            item['jobid']=one['jobid']
            item['job_href']=one['job_href']
            item['job_name']=one['job_name']
            item['job_welf']=one['jobwelf']
            print(item['job_href'])
            # 根据job介绍url地址 获取 职位要求，工作地点，公司信息
            a,b,c=self.get_jobdetail(item['job_href'])
            item['job_info']=a
            item['job_work']=b
            item['company_info']=c
            item['attribute_text']=one['attribute_text']
            item['providesalary_text']=one['providesalary_text']
            item['company_name']=one['company_name']
            item['companysize_text']=one['companysize_text']
            item['companyind_text']=one['companyind_text']
            item['company_href']=one['company_href']
            item['degreefrom']=one['degreefrom']
            item['companytype_text']=one['companytype_text']
            item['workarea_text']=one['workarea_text']
            item['updatedate']=one['updatedate']
            items.append(item)
        print("parse_one_page finished: size=",len(html),"开始存储")
        df=pd.DataFrame(items)
        df.to_csv("51job_data.csv",encoding="utf_8_sig",mode = 'a',columns=['jobid','job_href','job_name','job_welf','job_info','job_work','company_info',
            'attribute_text','providesalary_text','company_name','companysize_text','companyind_text','company_href',
            'degreefrom','companytype_text','workarea_text','updatedate'])

```



### 2. HTML 渲染模式

#### 2.1.  静态网页

静态网页是指存放在服务器文件系统中实实在在的HTML文件。当用户在浏览器中输入页面的URL，然后回车，浏览器就会将对应的html文件下载、渲染并呈现在窗口中。早期的网站通常都是由静态页面制作的。

**[特点]**

- 静态网页每个网页都有一个固定的URL，且网页URL以.htm、.html、.shtml等常见形式为后缀，而不含有“?”；（动态网页中的“？”对搜索引擎检索存在一定的问题，搜索引擎一般不可能从一个网站的数据库中访问全部网页，或者出于技术方面的考虑，搜索蜘蛛不去抓取网址中“？”后面的内容。）

- 网页内容一经发布到网站服务器上，无论是否有用户访问，每个静态网页的内容都是保存在网站服务器上的，也就是说，静态网页是实实在在保存在服务器上的文件，每个网页都是一个独立的文件；

- 静态网页的内容相对稳定，因此容易被搜索引擎检索；

- 静态网页没有数据库的支持，在网站制作和维护方面工作量较大，因此当网站信息量很大时完全依靠静态网页制作方式比较困难；

- 静态网页的交互性较差，在功能方面有较大的限制。

- 页面浏览速度迅速，过程无需连接数据库，开启页面速度快于动态页面。

- 减轻了服务器的负担，工作量减少，也就降低了数据库的成本。
- 没得网络服务器或应用服务器，比如直接从CD-ROM（激光唱片-只读存储器）或USB闪存驱动器读取内容，可以通过网络浏览器直接访问。
- 网站更安全，HTML页面不会受Asp相关漏洞的影响；而且可以减少攻击，防SQL注入。数据库出错时，不影响网站正常访问。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201028223624298.png)

> 所谓的动态网页，是指跟静态网页相对的一种网页编程技术。静态网页，随着html代码生成，页面的内容和显示效果就不会发生变化了。而动态网页则不然，其显示的页面则是经过Javascript处理数据后生成的结果，可以发生改变。**这些数据的来源有多种，可能是经过Javascript计算生成的，也可能是通过Ajax加载的。**
>
> 动态加载网页的方式，一来是可以实现web开发的前后端分离，减少服务器直接渲染页面的压力；**二来是可以作为反爬虫的一种手段。**

#### 2.2. 动态网页

​		动态网页是相对于静态网页而言的。当浏览器请求服务器的某个页面时，服务器根据当前时间、环境参数、数据库操作等动态的生成HTML页面，然后在发送给浏览器（后面的处理就跟静态网页一样了）。很明显，动态网页中的“动态”是指服务器端页面的动态生成，相反，“静态”则指页面是实实在在的、独立的文件。

- HTML+JavaScript(Node.js)
- HTML+PHP
- HTML+ASP.NET(或ASP)
- HTML+JSP
- HTML+CGI(早期的动态网页技术)

[**动态网页**]

- 动态网页一般以数据库技术为基础，可以大大降低网站维护的工作量；

- 采用动态网页技术的网站可以实现更多的功能，如用户注册、用户登录、在线调查、用户管理、订单管理等等；

- 动态网页实际上并不是独立存在于服务器上的网页文件，只有当用户请求时服务器才返回一个完整的网页；

- 动态网页地址中的“?”对搜索引擎检索存在一定的问题，搜索引擎一般不可能从一个网站的数据库中访问全部网页，或者出于技术方面的考虑，搜索蜘蛛不去抓取网址中“?”后面的内容，因此采用动态网页的网站在进行搜索引擎推广时需要做一定的技术处理才能适应搜索引擎的要求。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201028223654422.png)

#### 2.3. 伪静态网页

实时的显示一些信息。或者还想运用动态脚本解决一些问题。不能用静态的方式来展示网站内容。但是这就损失了对搜索引擎的友好面。展示出来的是以html一类的静态页面形式，但其实是用ASP一类的动态脚本来处理的。

- https://www.jianshu.com/p/649d2a0ebde5

### 3. 动态网页爬取

> 以 [新浪读书——书摘](http://book.sina.com.cn/excerpt/) 为例，介绍如何得到无法筛选出来的Ajax请求链接:在Chrome中打开网页，右键检查，会发现首页中书摘列表包含在一个id为subShowContent1_static的div中，而查看网页源代码会发现id为subShowContent1_static的div为空。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029092325019.png)

并且点击更多书摘或下一页时，网页URL并没有发生变化。这与我们最前面所说的两种情况相同，说明这个网页就是使用 JS 动态加载数据的。

> F12打开调试工具，打开NetWork窗口，F5刷新，可以看到浏览器发送以及接收到的数据记录(我们可以点击上面的 XHR 或者 JS 对这些请求进行过滤)：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029092647745.png)

#### 步骤一. 根据id进行查找

> js 操作页面的数据一定要进行定位，最常用的方法就是使用 id 定位，因为 id 在整个页面中是唯一的，那么我们第一步就是在所有的 js 文件中找和 subShowContent1_static 这个 id 相关的文件，于是我在 network 页面使用 ctrl+f 进行全局搜索。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029093044283.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029093403370.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029095656616.png)

#### 步骤二：断点动态捕获

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029100221196.png)

```
#设置断点后 F5 刷新
#根据断点信息构造 访问url
http://feed.mix.sina.com.cn/api/roll/get?callback=xxxxxxxx&pageid=96&lid=560&num=20&page=1
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029100755169.png)

### 4. 反爬虫

1、**Headers反爬虫** ：Cookie、Referer、User-Agent

 解决方案: 通过F12获取headers,传给requests.get()方法

2、**IP限制** ：网站根据IP地址访问频率进行反爬,短时间内进制IP访问

 解决方案:

​    1、构造自己IP代理池,每次访问随机选择代理,经常更新代理池

​    2、购买开放代理或私密代理IP

​    3、降低爬取的速度

3、**User-Agent限制** ：类似于IP限制

 解决方案: 构造自己的User-Agent池,每次访问随机选择

5、**对查询参数或Form表单数据认证(salt、sign)**

 解决方案: 找到JS文件,分析JS处理方法,用Python按同样方式处理

6、**对响应内容做处理**

 解决方案: 打印并查看响应内容,用xpath或正则做处理

### 5. 爬虫练习

> 查看网页编码：在窗口console标签下，键入 "document.charset"

#### 5.1.  伪链接静态网页

> 目标: 抓取最新中华人民共和国县以上行政区划代码
>
> URL: http://www.mca.gov.cn/article/sj/xzqh/2019/ - 民政数据 - 行政区划代码

```python
import requests
from lxml import etree
import re
import pymysql
class GovementSpider(object):
    def __init__(self):
        self.url = 'http://www.mca.gov.cn/article/sj/xzqh/2019/'
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        # 创建2个对象
        self.db = pymysql.connect('127.0.0.1', 'root', '123456', 'govdb', charset='utf8')
        self.cursor = self.db.cursor()

    # 获取假链接
    def get_false_link(self):
        html = requests.get(url=self.url, headers=self.headers).text
        # 此处隐藏了真实的二级页面的url链接，真实的在假的响应网页中，通过js脚本生成，
        # 假的链接在网页中可以访问，但是爬取到的内容却不是我们想要的
        parse_html = etree.HTML(html)
        a_list = parse_html.xpath('//a[@class="artitlelist"]')
        for a in a_list:
            # get()方法:获取某个属性的值
            title = a.get('title')
            if title.endswith('代码'):
                # 获取到第1个就停止即可，第1个永远是最新的链接
                false_link = 'http://www.mca.gov.cn' + a.get('href')
                print("二级“假”链接的网址为", false_link)
                break
        # 提取真链接
        self.incr_spider(false_link)
    # 增量爬取函数
    def incr_spider(self, false_link):   #false_link: http://www.mca.gov.cn/article/sj/xzqh/2019/202002/20200200024708.shtml
        self.cursor.execute('select url from version where url=%s', [false_link])
        # fetchall: (('http://xxxx.html',),)
        result = self.cursor.fetchall()
        # not result:代表数据库version表中无数据
        if not result:
            self.get_true_link(false_link)
            # 可选操作: 数据库version表中只保留最新1条数据
            self.cursor.execute("delete from version")
            # 把爬取后的url插入到version表中
            self.cursor.execute('insert into version values(%s)', [false_link])
            self.db.commit()
        else:
            print('数据已是最新,无须爬取')

    # 获取真链接
    def get_true_link(self, false_link):
        # 先获取假链接的响应,然后根据响应获取真链接
        html = requests.get(url=false_link, headers=self.headers).text
        # 从二级页面的响应中提取真实的链接（此处为JS动态加载跳转的地址）
        re_bds = r'window.location.href="(.*?)"'
        pattern = re.compile(re_bds, re.S)
        true_link = pattern.findall(html)[0]
        self.save_data(true_link)  # 提取真链接的数据
    # 用xpath直接提取数据
    def save_data(self, true_link):
        html = requests.get(url=true_link, headers=self.headers).text
        # 基准xpath,提取每个信息的节点列表对象
        parse_html = etree.HTML(html)
        tr_list = parse_html.xpath('//tr[@height="19"]')
        for tr in tr_list:
            code = tr.xpath('./td[2]/text()')[0].strip()  # 行政区划代码
            name = tr.xpath('./td[3]/text()')[0].strip()  # 单位名称
            print(name, code)
    # 主函数
    def main(self):
        self.get_false_link()
if __name__ == '__main__':
    spider = GovementSpider()
    spider.main()
```

#### 5.2. 动态数据爬取

**特点**

> 1. 右键 -> 查看网页源码中没有具体数据
> 2. 滚动鼠标滑轮或其他动作时加载

**抓取**

> 1. F12打开控制台，选择XHR异步加载数据包，找到页面动作抓取网络数据包
> 2. 通过XHR-->Header-->General-->Request URL，获取json文件URL地址
> 3. 通过XHR-->Header-->Query String Parameters(查询参数)

##### **任务一： 豆瓣电影数据抓取案例**

目标

1. 地址: 豆瓣电影 - 排行榜 - 剧情
   - [https://movie.douban.com/typerank?](https://movie.douban.com/typerank?type_name=剧情&type=11&interval_id=100:90&action=)
   - [type_name=%E5%89%A7%E6%83%85&type=11&interval_id=100:90&action=](https://movie.douban.com/typerank?type_name=剧情&type=11&interval_id=100:90&action=)
2. 目标: 爬取电影名称、电影评分

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029112020668.png)

> 通过源代码找不到的情况下，可以在network那刷新，然后重新搜索。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029111302563.png)

```python
import requests
import time
from fake_useragent import UserAgent


class DoubanSpider(object):
    def __init__(self):
        self.base_url = 'https://movie.douban.com/j/chart/top_list?'
        self.i = 0
    def get_html(self, params):
        headers = {'User-Agent': UserAgent().random}
        res = requests.get(url=self.base_url, params=params, headers=headers)
        res.encoding = 'utf-8'
        html = res.json()  # 将json格式的字符串转为python数据类型
        self.parse_html(html)  # 直接调用解析函数
    def parse_html(self, html):
        # html: [{电影1信息},{电影2信息},{}]
        item = {}   #python 列表的使用
        for one in html:
            item['name'] = one['title']  # 电影名
            item['score'] = one['score']  # 评分
            item['time'] = one['release_date']  # 打印测试
            # 打印显示
            print(item)
            self.i += 1
    # 获取电影总数
    def get_total(self, typ):
        # 异步动态加载的数据 都可以在XHR数据抓包
        url = 'https://movie.douban.com/j/chart/top_list_count?type={}&interval_id=100%3A90'.format(typ)
        ua = UserAgent()
        html = requests.get(url=url, headers={'User-Agent': ua.random}).json()
        total = html['total']
        return total
    def main(self):
        typ = input('请输入电影类型(剧情|喜剧|动作):')
        typ_dict = {'剧情': '11', '喜剧': '24', '动作': '5'}
        typ = typ_dict[typ]
        total = self.get_total(typ)  # 获取该类型电影总数量
        for page in range(0, int(total), 20):
            params = {
                'type': typ,
                'interval_id': '100:90',
                'action': '',
                'start': str(page),
                'limit': '20'}
            self.get_html(params)
            time.sleep(1)
        print('爬取的电影的数量:', self.i)
if __name__ == '__main__':
    spider = DoubanSpider()
    spider.main()
```

##### **任务二：腾讯招聘数据抓取(Ajax)**

确定URL地址及目标

- URL: 百度搜索腾讯招聘 - 查看工作岗位https://careers.tencent.com/search.html
- 目标: 职位名称、工作职责、岗位要求

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029112427977.png)

> 确定所要数据所在页面： url 地址可以直接在header里面进行查看
>
> 职位名称：https://careers.tencent.com/tencentcareer/api/post/Query?timestamp=1603941718816&countryId=&cityId=&bgIds=&productId=&categoryId=&parentCategoryId=&attrId=&keyword=&pageIndex=1&pageSize=10&language=zh-cn&area=cn
>
> 工作职责、岗位要求： https://careers.tencent.com/jobdesc.html?postId=1290651460547125248

```python
import time
import json
import random
import requests
from useragents import ua_list


class TencentSpider(object):
    def __init__(self):
        self.one_url = 'https://careers.tencent.com/tencentcareer/api/post/Query?timestamp=1563912271089&countryId=&cityId=&bgIds=&productId=&categoryId=&parentCategoryId=&attrId=&keyword=&pageIndex={}&pageSize=10&language=zh-cn&area=cn'
        self.two_url = 'https://careers.tencent.com/tencentcareer/api/post/ByPostId?timestamp=1563912374645&postId={}&language=zh-cn'
        self.f = open('tencent.json', 'a')  # 打开文件
        self.item_list = []  # 存放抓取的item字典数据

    # 获取响应内容函数
    def get_page(self, url):
        headers = {'User-Agent': random.choice(ua_list)}
        html = requests.get(url=url, headers=headers).text
        html = json.loads(html)  # json格式字符串转为Python数据类型

        return html

    # 主线函数: 获取所有数据
    def parse_page(self, one_url):
        html = self.get_page(one_url)
        item = {}
        for job in html['Data']['Posts']:
            item['name'] = job['RecruitPostName']  # 名称
            post_id = job['PostId']  # postId，拿postid为了拼接二级页面地址
            # 拼接二级地址,获取职责和要求
            two_url = self.two_url.format(post_id)
            item['duty'], item['require'] = self.parse_two_page(two_url)
            print(item)
            self.item_list.append(item)  # 添加到大列表中

    # 解析二级页面函数
    def parse_two_page(self, two_url):
        html = self.get_page(two_url)
        duty = html['Data']['Responsibility']  # 工作责任
        duty = duty.replace('\r\n', '').replace('\n', '')  # 去掉换行
        require = html['Data']['Requirement']  # 工作要求
        require = require.replace('\r\n', '').replace('\n', '')  # 去掉换行

        return duty, require

    # 获取总页数
    def get_numbers(self):
        url = self.one_url.format(1)
        html = self.get_page(url)
        numbers = int(html['Data']['Count']) // 10 + 1  # 每页有10个推荐

        return numbers

    def main(self):
        number = self.get_numbers()
        for page in range(1, 3):
            one_url = self.one_url.format(page)
            self.parse_page(one_url)

        # 保存到本地json文件:json.dump
        json.dump(self.item_list, self.f, ensure_ascii=False)
        self.f.close()


if __name__ == '__main__':
    start = time.time()
    spider = TencentSpider()
    spider.main()
    end = time.time()
    print('执行时间:%.2f' % (end - start))
```

##### 任务三：51job网站爬取

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201029205633909.png)

```python
import requests
import time
from fake_useragent import UserAgent
import re
import json
from urllib import parse
from lxml import etree
import xlwt     
import pandas as pd
import random
class JobSpider(object):
    def __init__(self):
        self.base_url = 'https://movie.douban.com/j/chart/top_list?'
        self.i = 0   #记录计数器
        #self.wirte_sheet=self.init_xkwt()  

    def init_xkwt(self):
        '''
            初始化csv文件表头
        '''
        #新建表格空间
        excel1 = xlwt.Workbook()
        #新建一个sheet,设置单元格格式,cell_overwrite_ok=True防止对一个单元格重复操作引发的错误
        sheet1 = excel1.add_sheet('Job', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'jobid')
        sheet1.write(0, 1, 'job_href')
        sheet1.write(0, 2, 'job_name')
        sheet1.write(0, 3, 'jobwelf')
        sheet1.write(0, 4, 'attribute_text')
        sheet1.write(0, 5, 'providesalary_text')
        sheet1.write(0, 6, 'company_name')
        sheet1.write(0, 7, 'companysize_text')
        sheet1.write(0, 8, 'companyind_text')
        sheet1.write(0, 9, 'company_href')
        sheet1.write(0, 10,'degreefrom')
        sheet1.write(0, 11,'companytype_text')
        sheet1.write(0, 12,'workarea_text')
        sheet1.write(0, 13,'companytype_text')
        sheet1.write(0, 14,'updatedate')
        sheet1.write(0, 15,'job_info')
        sheet1.write(0, 16,'job_work')
        sheet1.write(0, 17,'company_info')
        return sheet1
        #sheet1.write(number,0,number)
    def parse_html(self, html):
        '''
            html: [{jobid},{job_href},{job_name},{jobwelf},{attribute_text},{providesalary_text},{company_name},{companysize_text},{companyind_text},{company_href},{degreefrom},{companytype_text},{workarea_text},{updatedate}]
            解析一页中html文件，获得如上文件个数数据
        '''
        items = []   #python 列表的使用
        for one in html:
            item={}
            item['jobid']=one['jobid']
            item['job_href']=one['job_href']
            item['job_name']=one['job_name']
            item['job_welf']=one['jobwelf']
            #print(item['job_href'])
            # 根据job介绍url地址 获取 职位要求，工作地点，公司信息
            a,b,c=self.get_jobdetail(item['job_href'])
            item['job_info']=a
            item['job_work']=b
            item['company_info']=c
            item['attribute_text']=one['attribute_text']
            item['providesalary_text']=one['providesalary_text']
            item['company_name']=one['company_name']
            item['companysize_text']=one['companysize_text']
            item['companyind_text']=one['companyind_text']
            item['company_href']=one['company_href']
            item['degreefrom']=one['degreefrom']
            item['companytype_text']=one['companytype_text']
            item['workarea_text']=one['workarea_text']
            item['updatedate']=one['updatedate']
            items.append(item)
        print("parse_one_page finished: size=",len(html),"开始存储")
        df=pd.DataFrame(items)
        df.to_csv("51job_data.csv",encoding="utf_8_sig",mode = 'a',columns=['jobid','job_href','job_name','job_welf','job_info','job_work','company_info',
            'attribute_text','providesalary_text','company_name','companysize_text','companyind_text','company_href',
            'degreefrom','companytype_text','workarea_text','updatedate'])



    def get_html(self, key,page):
        '''
            根据key，page获取对应的html文件中对应的有效数据
        '''
        #伪装爬取头部，以防止被网站禁止
        headers={'Host':'search.51job.com',
                'Upgrade-Insecure-Requests':'1',
                'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko)\
                Chrome/63.0.3239.132 Safari/537.36'}
        headers = {'User-Agent': UserAgent().random}
        url='https://search.51job.com/list/050000%252c010000,000000,0100,00,9,99,{},2,{}.html'.format(key,page)
        html = requests.get(url=url, headers=headers).content.decode('gbk')
        pattern=re.compile(r'window.__SEARCH_RESULT__ =(.*?)</script>')
        data=pattern.findall(html)[0]
        data=json.loads(data)["engine_search_result"]
        #print("长度",len(data))
        self.parse_html(data)  # 直接调用解析函数

    def get_jobdetail(self,url):
        '''
            https://jobs.51job.com/beijing-hdq/126533792.html?s=01&t=0
        '''
        headers = {'User-Agent': UserAgent().random}
        html = requests.get(url=url, headers=headers).content.decode('gbk')
        parse_html = etree.HTML(html)
        job_info=parse_html.xpath('/html/body/div[3]/div[2]/div[3]/div[1]/div/p/text()')  #/html/body/div[3]/div[2]/div[3]/div[1]/div/p[1]
        job_work=parse_html.xpath('/html/body/div[3]/div[2]/div[3]/div[2]/div/p/text()')
        company_info=parse_html.xpath('/html/body/div[3]/div[2]/div[3]/div[3]/div/text()')
        #print(job_info,job_work,company_info)
        return job_info,job_work,company_info

    # 获取page 总数   #"total_page":"195"
    def get_total(self,key,page=1):
        url='https://search.51job.com/list/000000,000000,0100,00,9,99,{},2,{}.html'.format(key,page)
        ua = UserAgent()
        html = requests.get(url=url, headers={'User-Agent': ua.random}).content.decode('gbk')
        pattern=re.compile(r'window.__SEARCH_RESULT__ =(.*?)</script>')
        data=pattern.findall(html)[0]
        data=json.loads(data)
        total =data['total_page']
        #print("url:",url,"该职位总页数为：",total)
        return total
    def main(self):
        #key = input('请输入职业类型(数据挖掘|喜剧|动作):')
        key=['数据挖掘','人工智能','语音识别','物联网','嵌入式','java','python','c++','android','运维','销售','制造']
        #key=parse.quote(parse.quote(key))
        for item in key:
            key=parse.quote(parse.quote(item))
            total = self.get_total(key)  # 获取该类型电影总数量
            print("当前存储记录：",self.i,"\t 即将爬取关键词：",item,"\t 该关键词个数: ",total)
            for page in range(1, int(total)):
                self.get_html(key,page)
                time.sleep(random.random()*4)
        #print('爬取的电影的数量:', self.i)
if __name__ == '__main__':
    spider = JobSpider()
    spider.main()

#https://search.51job.com/list/050000,000000,0100,00,9,99,%25E6%2595%25B0%25E6%258D%25AE%25E6%258C%2596%25E6%258E%2598,2,1.html?lang=c&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&ord_field=0&dibiaoid=0&line=&welfare=
#https://search.51job.com/list/010000,000000,0100,00,9,99,%25E6%2595%25B0%25E6%258D%25AE%25E6%258C%2596%25E6%258E%2598,2,1.html?lang=c&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&ord_field=0&dibiaoid=0&line=&welfare=
#010000-330000: 代表不同的省份，其中000000： 代表全国  ；%25E6%2595%25B0%25E6%258D%25AE%25E6%258C%2596%25E6%258E%2598 对应关键词编码； ,2: 代表页数
```

##### 任务五：boss直聘爬虫

```python
# common imports
import requests
from lxml import etree
import time
import random
import pymongo
from retrying import retry
from scrapy import Selector 
from lxml import etree
import urllib.request
import urllib.parse
import xlwt

# ---------------------
# 连接到MongoDB
MONGO_URL = 'localhost'
MONGO_DB = 'Graduation_project'
MONGO_COLLECTION = 'shanghai_discovery'
client = pymongo.MongoClient(MONGO_URL, port=27017)
db = client[MONGO_DB]
# 页面获取函数
def get_page(page, keyword):
    header = {    
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36"
    }
    print('正在爬取第', page, '页')
    url = 'https://www.zhipin.com/c101020100/?query={k}&page={page}&ka=page-{page}'.format(page=page, k=keyword)
    response = requests.get(url, headers=header)
    return response.decode("utf-8")


# --------------
@retry(wait_fixed=8000)
def job_detail(link):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/69.0.3497.12 Safari/537.36 '
    }
    response = requests.get(link, headers=header)
    data = etree.HTML(response.text)

    # ---检验是否出现验证码
    tips = data.xpath('/html/head/title/text()')
    tips_title = 'BOSS直聘验证码'
    if tips[0] == tips_title:
        print('检查是否弹出验证码')
        # 弹出验证码则引发IOError来进行循环
        raise IOError
    # ----------------------
    job_desc = data.xpath('//*[@id="main"]/div[3]/div/div[2]/div[3]/div[@class="job-sec"][1]/div/text()')

    jd = "".join(job_desc).strip()
    return jd


def parse_page(html, keyword, page):
    # 观察数据结构可得
    data = etree.HTML(html)
    items = data.xpath('//*[@id="main"]/div/div[2]/ul/li')
    for item in items:
        district = item.xpath('./div/div[1]/p/text()[1]')[0]
        job_links = item.xpath('./div/div[1]/h3/a/@href')[0]
        job_title = item.xpath('./div/div[1]/h3/a/div[1]/text()')[0]
        job_salary = item.xpath('./div/div[1]/h3/a/span/text()')[0]
        job_company = item.xpath('./div/div[2]/div/h3/a/text()')[0]
        job_experience = item.xpath('./div/div[1]/p/text()[2]')[0]
        job_degree = item.xpath('./div/div[1]/p/text()[3]')[0]
        fin_status = item.xpath('./div/div[2]/div/p/text()[2]')[0]
        try:
            company_scale = item.xpath('./div/div[2]/div/p/text()[3]')[0]
        except Exception:
            company_scale = item.xpath('./div/div[2]/div/p/text()[2]')[0]
        job_link = host + job_links
        print(job_link)
        # 获取职位描述
        detail = job_detail(job_link)
        # ---------------
        job = {
            'Keyword': keyword,
            '地区': district,
            '职位名称': job_title,
            '职位薪资': job_salary,
            '公司名称': job_company,
            '工作经验': job_experience,
            '学历要求': job_degree,
            '公司规模': company_scale,
            '融资情况': fin_status,
            '职位描述': detail,
        }
        print(job)
        save_to_mongo(job)
        time.sleep(random.randint(6, 9))
        # ---------------------------------------





def save_to_mongo(data):
    # 保存到MongoDB中
    try:
        if db[MONGO_COLLECTION].insert(data):
            print('存储到 MongoDB 成功')
    except Exception:
        print('存储到 MongoDB 失败')

#  header 通过浏览器检查， 网络， 点击对应的name在headers 属性下面；  选择某一个属性后可以通过 鼠标右击copy 复制xpath 路径。
if __name__ == '__main__':
    #url = r'https://list.jd.com/list.html?cat=670%2C671%2C2694&ev=exbrand_%E5%BE%AE%E8%BD%AF%EF%BC%88Microsoft%EF%BC%89%5E1107_90246%5E244_116227%5E3753_76033%5E&cid3=2694'
    MAX_PAGE = 10
    host = 'https://www.zhipin.com'
    keywords = ['数据分析', '数据挖掘', '商业分析', '机器学习']
    for keyword in keywords:
        for i in range(1, MAX_PAGE + 1):
            html = get_page(i, keyword)
            # ------------ 解析数据 ---------------
            parse_page(html, keyword, i)
            print('-' * 100)
            # -----------------
            timewait = random.randint(15, 18)
            time.sleep(timewait)
            print('等待', timewait, '秒')

```

### 6. Scraw 框架

```python
pip install scrapy
# projectname为项目名称，可自定义
scrapy startproject projectname
#文件结构
projectname/
    scrapy.cfg            # 配置文件
    projectname/             # 爬虫模块文件夹
        __init__.py
        items.py          # items定义文件,设置数据存储模板，用于结构化数据，如：Django的Model
        middlewares.py    # 中间件middlewares文件
        pipelines.py      # 项目管道pipelines文件,数据处理行为，如：一般结构化的数据持久化
        settings.py       # 设置settings文件
        spiders/          # 爬虫文件夹 爬虫目录，如：创建文件，编写爬虫规则
            __init__.py
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201030094148678.png)

1. The [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine) gets the initial Requests to crawl from the [Spider](https://doc.scrapy.org/en/latest/topics/architecture.html#component-spiders).
2. The [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine) schedules the Requests in the [Scheduler](https://doc.scrapy.org/en/latest/topics/architecture.html#component-scheduler) and asks for the next Requests to crawl.
3. The [Scheduler](https://doc.scrapy.org/en/latest/topics/architecture.html#component-scheduler) returns the next Requests to the [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine).
4. The [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine) sends the Requests to the [Downloader](https://doc.scrapy.org/en/latest/topics/architecture.html#component-downloader), passing through the [Downloader Middlewares](https://doc.scrapy.org/en/latest/topics/architecture.html#component-downloader-middleware) (see [`process_request()`](https://doc.scrapy.org/en/latest/topics/downloader-middleware.html#scrapy.downloadermiddlewares.DownloaderMiddleware.process_request)).
5. Once the page finishes downloading the [Downloader](https://doc.scrapy.org/en/latest/topics/architecture.html#component-downloader) generates a Response (with that page) and sends it to the Engine, passing through the [Downloader Middlewares](https://doc.scrapy.org/en/latest/topics/architecture.html#component-downloader-middleware) (see [`process_response()`](https://doc.scrapy.org/en/latest/topics/downloader-middleware.html#scrapy.downloadermiddlewares.DownloaderMiddleware.process_response)).
6. The [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine) receives the Response from the [Downloader](https://doc.scrapy.org/en/latest/topics/architecture.html#component-downloader) and sends it to the [Spider](https://doc.scrapy.org/en/latest/topics/architecture.html#component-spiders) for processing, passing through the [Spider Middleware](https://doc.scrapy.org/en/latest/topics/architecture.html#component-spider-middleware) (see [`process_spider_input()`](https://doc.scrapy.org/en/latest/topics/spider-middleware.html#scrapy.spidermiddlewares.SpiderMiddleware.process_spider_input)).
7. The [Spider](https://doc.scrapy.org/en/latest/topics/architecture.html#component-spiders) processes the Response and returns scraped items and new Requests (to follow) to the [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine), passing through the [Spider Middleware](https://doc.scrapy.org/en/latest/topics/architecture.html#component-spider-middleware) (see [`process_spider_output()`](https://doc.scrapy.org/en/latest/topics/spider-middleware.html#scrapy.spidermiddlewares.SpiderMiddleware.process_spider_output)).
8. The [Engine](https://doc.scrapy.org/en/latest/topics/architecture.html#component-engine) sends processed items to [Item Pipelines](https://doc.scrapy.org/en/latest/topics/architecture.html#component-pipelines), then send processed Requests to the [Scheduler](https://doc.scrapy.org/en/latest/topics/architecture.html#component-scheduler) and asks for possible next Requests to crawl.
9. The process repeats (from step 1) until there are no more requests from the [Scheduler](https://doc.scrapy.org/en/latest/topics/architecture.html#component-scheduler).

**【几个特殊的类】**

- scrapy.Spider 类

> the place where you define the custom behaviour for crawling and parsing pages for a particular site (or, in some cases, a group of sites).
>
> 1. You start by generating the initial Requests to crawl the first URLs, and specify a callback function to be called with the response downloaded from those requests.
>
>    The first requests to perform are obtained by calling the [`start_requests()`](https://doc.scrapy.org/en/latest/topics/spiders.html#scrapy.spiders.Spider.start_requests) method which (by default) generates [`Request`](https://doc.scrapy.org/en/latest/topics/request-response.html#scrapy.http.Request) for the URLs specified in the [`start_urls`](https://doc.scrapy.org/en/latest/topics/spiders.html#scrapy.spiders.Spider.start_urls) and the [`parse`](https://doc.scrapy.org/en/latest/topics/spiders.html#scrapy.spiders.Spider.parse) method as callback function for the Requests.
>
> 2. In the callback function, you parse the response (web page) and return [item objects](https://doc.scrapy.org/en/latest/topics/items.html#topics-items), [`Request`](https://doc.scrapy.org/en/latest/topics/request-response.html#scrapy.http.Request) objects, or an iterable of these objects. Those Requests will also contain a callback (maybe the same) and will then be downloaded by Scrapy and then their response handled by the specified callback.
>
> 3. In callback functions, you parse the page contents, typically using [Selectors](https://doc.scrapy.org/en/latest/topics/selectors.html#topics-selectors) (but you can also use BeautifulSoup, lxml or whatever mechanism you prefer) and generate items with the parsed data.
>
> 4. Finally, the items returned from the spider will be typically persisted to a database (in some [Item Pipeline](https://doc.scrapy.org/en/latest/topics/item-pipeline.html#topics-item-pipeline)) or written to a file using [Feed exports](https://doc.scrapy.org/en/latest/topics/feed-exports.html#topics-feed-exports).

```python
# -*- coding: utf-8 -*-
import scrapy
'''
    Spiders are classes that you define and that Scrapy uses to scrape information from a website (or a group of websites). They must subclass Spider and define the initial requests to make, optionally how to follow links in the pages, and how to parse the downloaded page content to extract data.
    start_requests(): must return an iterable of Requests (you can return a list of requests or write a generator function) which the Spider will begin to crawl from. Subsequent requests will be generated successively from these initial requests.
    parse(): a method that will be called to handle the response downloaded for each of the requests made. The response parameter is an instance of TextResponse that holds the page content and has further helpful methods to handle it.
    The parse() method usually parses the response, extracting the scraped data as dicts and also finding new URLs to follow and creating new requests (Request) from them.
'''
# 注意一定要继承 scrapy.Spider 类

class ToScrapeSpiderXPath(scrapy.Spider):
    name = 'toscrape-xpath'
    start_urls = [
        'http://quotes.toscrape.com/',
    ]

    def parse(self, response):
        for quote in response.xpath('//div[@class="quote"]'):
            yield {
                'text': quote.xpath('./span[@class="text"]/text()').extract_first(),
                'author': quote.xpath('.//small[@class="author"]/text()').extract_first(),
                'tags': quote.xpath('.//div[@class="tags"]/a[@class="tag"]/text()').extract()
            }

        next_page_url = response.xpath('//li[@class="next"]/a/@href').extract_first()
        if next_page_url is not None:
            yield scrapy.Request(response.urljoin(next_page_url))

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'http://quotes.toscrape.com/page/1/',
            'http://quotes.toscrape.com/page/2/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f'quotes-{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
        
class AuthorSpider(scrapy.Spider):
    name = 'author'

    start_urls = ['http://quotes.toscrape.com/']

    def parse(self, response):
        author_page_links = response.css('.author + a')
        yield from response.follow_all(author_page_links, self.parse_author)

        pagination_links = response.css('li.next a')
        yield from response.follow_all(pagination_links, self.parse)

    def parse_author(self, response):
        def extract_with_css(query):
            return response.css(query).get(default='').strip()

        yield {
            'name': extract_with_css('h3.author-title::text'),
            'birthdate': extract_with_css('.author-born-date::text'),
            'bio': extract_with_css('.author-description::text'),
        }
```

- **Item**

```python
#item 类型
from scrapy.item import Item, Field
class CustomItem(Item):
    one_field = Field()
    another_field = Field()
# database 类型
from dataclasses import dataclass
@dataclass
class CustomItem:
    one_field: str
    another_field: int
#class 类型
import scrapy
class Product(scrapy.Item):
    name = scrapy.Field()
    price = scrapy.Field()
    stock = scrapy.Field()
    tags = scrapy.Field()
    last_updated = scrapy.Field(serializer=str)
```

- **Item Loaders**

> Item Loaders provide a convenient mechanism for populating scraped [items](https://doc.scrapy.org/en/latest/topics/items.html#topics-items). Even though items can be populated directly, Item Loaders provide a much more convenient API for populating them from a scraping process, by automating some common tasks like parsing the raw extracted data before assigning it.
>
> In other words, [items](https://doc.scrapy.org/en/latest/topics/items.html#topics-items) provide the *container* of scraped data, while Item Loaders provide the mechanism for *populating* that container.

```python
from scrapy.loader import ItemLoader
from myproject.items import Product

def parse(self, response):
    l = ItemLoader(item=Product(), response=response)
    l.add_xpath('name', '//div[@class="product_name"]')
    l.add_xpath('name', '//div[@class="product_title"]')
    l.add_xpath('price', '//p[@id="price"]')
    l.add_css('stock', 'p#stock]')
    l.add_value('last_updated', 'today') # you can also use literal values
    return l.load_item()
#when all data is collected, the ItemLoader.load_item() method is called which actually returns the item populated with the data previously extracted and collected with the add_xpath(), add_css(), and add_value() calls.
```

- **Pipeline**

> They receive an item and perform an action over it, also deciding if the item should continue through the pipeline or be dropped and no longer processed.Typical uses of item pipelines are:
>
> - cleansing HTML data
>
> - validating scraped data (checking that the items contain certain fields)
>
> - checking for duplicates (and dropping them)
>
> - storing the scraped item in a database
>
> `Process_item`(*self*, *item*, *spider*)
>
> This method is called for every item pipeline component.
>
> item is an [item object](https://doc.scrapy.org/en/latest/topics/items.html#item-types), see [Supporting All Item Types](https://doc.scrapy.org/en/latest/topics/items.html#supporting-item-types).
>
> [`process_item()`](https://doc.scrapy.org/en/latest/topics/item-pipeline.html#process_item) must either: return an [item object](https://doc.scrapy.org/en/latest/topics/items.html#item-types), return a [`Deferred`](https://twistedmatrix.com/documents/current/api/twisted.internet.defer.Deferred.html) or raise a [`DropItem`](https://doc.scrapy.org/en/latest/topics/exceptions.html#scrapy.exceptions.DropItem) exception.
>
> Dropped items are no longer processed by further pipeline components.
>
> - Parameters
>
>   **item** ([item object](https://doc.scrapy.org/en/latest/topics/items.html#item-types)) – the scraped item**spider** ([`Spider`](https://doc.scrapy.org/en/latest/topics/spiders.html#scrapy.spiders.Spider) object) – the spider which scraped the item
>
> - Activate the Pipeline
>
> ```python
> ITEM_PIPELINES = {
>     'myproject.pipelines.PricePipeline': 300,
>     'myproject.pipelines.JsonWriterPipeline': 800,
> }
> ```

```python
#vaidation
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
class PricePipeline:
    vat_factor = 1.15
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter.get('price'):
            if adapter.get('price_excludes_vat'):
                adapter['price'] = adapter['price'] * self.vat_factor
            return item
        else:
            raise DropItem(f"Missing price in {item}")
#write to json
import json
from itemadapter import ItemAdapter
class JsonWriterPipeline:
    def open_spider(self, spider):
        self.file = open('items.jl', 'w')
    def close_spider(self, spider):
        self.file.close()
    def process_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file.write(line)
        return item
#write to Mongodb
import pymongo
from itemadapter import ItemAdapter
class MongoPipeline:
    collection_name = 'scrapy_items'
    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DATABASE', 'items')
        )
    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]
    def close_spider(self, spider):
        self.client.close()
    def process_item(self, item, spider):
        self.db[self.collection_name].insert_one(ItemAdapter(item).asdict())
        return item
#duplicate
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
class DuplicatesPipeline:
    def __init__(self):
        self.ids_seen = set()
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter['id'] in self.ids_seen:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.ids_seen.add(adapter['id'])
            return item
```

### 7. 数据可视化

使用pyechart 

- https://www.cnblogs.com/-wenli/p/10646261.html
- https://juejin.im/post/6844903861245722631

使用BI软件

使用Original2018 软件

### . 学习链接

- https://www.k0rz3n.com/2019/03/05/%E7%88%AC%E8%99%AB%E7%88%AC%E5%8F%96%E5%8A%A8%E6%80%81%E7%BD%91%E9%A1%B5%E7%9A%84%E4%B8%89%E7%A7%8D%E6%96%B9%E5%BC%8F%E7%AE%80%E4%BB%8B/
- https://www.cnblogs.com/LXP-Never/p/11374795.html
- https://www.runoob.com/python3/python3-json.html
- crawl:https://doc.scrapy.org/en/latest/topics/architecture.html
- 后续学习：
  - Dataset 框架
  - 分布式反爬虫机制
  - Datasetlab 平台： https://github.com/crawlab-team/crawlab； https://demo-pro.crawlab.cn/#/spiders/5f9ae48321225400243ed393



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/crawl_record/  

