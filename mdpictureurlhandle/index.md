# MDPictureURLHanle

###   1.  图片url地址替换

```python
#!/usr/bin/env -S -P${HOME}/anaconda/bin python
# -*- coding:utf-8 -*-

import re, os, shutil, time, sys, argparse
from itertools import chain
# import oss2

# 需要替换url的MD文件
md_file = ''

# 操作类型, L2L (默认本地到本地)， L2W（本地到图床）， W2L（图床到本地）
action = ''

# 保存图片文件的根目录
dir_base = '/*******/_MD_Media'

# Markdown中图片语法 ![](url) 或者 <img src='' />
img_patten = r'!\[.*?\]\((.*?)\)|<img.*?src=[\'\"](.*?)[\'\"].*?>'


def get_img_local_path(md_file, path):
    """
    获取MD文件中嵌入图片的本地文件绝对地址
    :param md_file: MD文件
    :param path: 图片URL
    :return: 图片的本地文件绝对地址
    """

    result = None

    # /a/b/c
    if path.startswith('/'):
        result = path
    # ./a/b/c
    elif path.startswith('.'):
        result = '{0}/{1}'.format(os.path.dirname(md_file), path)
    # file:///a/b/c
    elif path.startswith('file:///'):
        result = path[8:]
        result = result.replace('%20',' ')
    else:
        result = '{0}/{1}'.format(os.path.dirname(md_file), path)

    return result

def local_2_local(md_file, dir_ts, match):
    """
    把MD中的本地图片移动到指定目录下，并返回URL。 这里并没有进行URL的替换
    :param md_file:
    :param dir_ts:
    :param match:
    :return: new_url，新本地文件地址。如果不需要替换，就返回空
    """
    dir_tgt = '{0}/{1}'.format(dir_base, dir_ts)
    new_url = None
    # 判断是不是已经是一个图片的网址，或者已经在指定目录下
    if not (re.match('((http(s?))|(ftp))://.*', match) or re.match('{}/.*'.format(dir_base), match)):
        # 如果图片url是本地文件，就替换到指定目录
        img_file = get_img_local_path(md_file, match)
        if os.path.isfile(img_file):
            new_url = '{0}/{1}'.format(dir_tgt, os.path.basename(match))
            os.makedirs(dir_tgt, exist_ok=True)
            # 移动物理文件
            shutil.move(img_file, dir_tgt)

    return new_url

def local_2_web(md_file, dir_ts, match):
    """
    把MD中的本地图片上传到OSS下，并返回URL。 这里并没有进行URL的替换
    :param md_file:
    :param dir_ts:
    :param match:
    :return: new_url，新本地文件地址。如果不需要替换，就返回空
    """

    # 阿里云OSS信息
    bucket_name = "b******ce"
    endpoint = "http://oss-cn-beijing.aliyuncs.com"
    access_key_id = "******"
    access_key_secret = "******"
    web_img_prfix = 'https://******.oss-cn-beijing.aliyuncs.com'
    # 创建Bucket对象，所有Object相关的接口都可以通过Bucket对象来进行
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)

    new_url = None
    # 判断是不是已经是一个图片的网址
    if not (re.match('((http(s?))|(ftp))://.*', match) ):
        # 如果图片url是本地文件，就上传
        img_file = get_img_local_path(md_file, match)
        if os.path.isfile(img_file):
            key_url = '{0}/{1}'.format(dir_ts, os.path.basename(match))
            bucket.put_object_from_file(key_url, img_file)
            new_url = '{}/{}'.format(web_img_prfix, key_url)

    return new_url

def replace_md_url(md_file):
    """
    把指定MD文件中引用的图片移动到指定地点（本地或者图床），并替换URL
    :param md_file: MD文件
    :return:
    """

    if os.path.splitext(md_file)[1] != '.md':
        print('{}不是Markdown文件，不做处理。'.format(md_file))
        return

    cnt_replace = 0
    # 本次操作时间戳
    dir_ts = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    with open(md_file, 'r',encoding='utf-8') as f: #使用utf-8 编码打开
        post = f.read()
        matches = re.compile(img_patten).findall(post)
        if matches and len(matches)>0 :
            # 多个group整合成一个列表
            for match in list(chain(*matches)) :
                if match and len(match)>0 :
                    new_url = None
                    remote_url="https://gitee.com/github-25970295/blogimgv2022/raw/master"
                    new_url=remote_url+"/"+match.split('/')[-1]
                    # 进行不同类型的URL转换操作
                    if action == 'L2L':
                        new_url = local_2_local(md_file, dir_ts, match)
                    elif action == 'L2W':
                        new_url = local_2_web(md_file, dir_ts, match)

                    # 更新MD中的URL
                    if new_url :
                        post = post.replace(match, new_url)
                        cnt_replace = cnt_replace + 1
                    else:
                        print("None")

        # 如果有内容的话，就直接覆盖写入当前的markdown文件
        if post and cnt_replace > 0:
            open(md_file, 'w', encoding='utf-8').write(post)
            print('{0}的{1}个URL被替换到<{2}>/{3}'.format(os.path.basename(md_file), cnt_replace, action, dir_ts))
        elif cnt_replace == 0:
            print('{}中没有需要替换的URL'.format(os.path.basename(md_file)))

#使用gitee图床，替换本地图片链接
def replace_md_url_v1(md_file):
    """
    把指定MD文件中引用的图片移动到指定地点（本地或者图床），并替换URL
    :param md_file: MD文件
    :return:
    """

    if os.path.splitext(md_file)[1] != '.md':
        print('{}不是Markdown文件，不做处理。'.format(md_file))
        return

    cnt_replace = 0
    # 本次操作时间戳
    dir_ts = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    with open(md_file, 'r',encoding='utf-8') as f: #使用utf-8 编码打开
        post = f.read()
        matches = re.compile(img_patten).findall(post)
        #print(matches)
        if matches and len(matches)>0 :
            # 多个group整合成一个列表
            for match in list(chain(*matches)) :
                if match and len(match)>0 and not "https://" in match:
                    remote_url="https://gitee.com/github-25970295/blogimgv2022/raw/master/"    # 这里使用的时候可能需要进行修改
                    new_url=remote_url+match.split('/')[-1]
                    print(new_url)
                    # 更新MD中的URL
                    post = post.replace(match, new_url)
                    cnt_replace = cnt_replace + 1

        # 如果有内容的话，就直接覆盖写入当前的markdown文件
        if post and cnt_replace > 0:
            open(md_file, 'w', encoding='utf-8').write(post)
            print('{0}的{1}个URL被替换到<{2}>/{3}'.format(os.path.basename(md_file), cnt_replace, action, dir_ts))
        elif cnt_replace == 0:
            print('{}中没有需要替换的URL'.format(os.path.basename(md_file)))

basefolder="E:\项目记录\liudongdongBlog\source\_posts"
def getallfile(path):
    allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath=os.path.join(path,file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            getallfile(filepath)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            print(filepath)
            replace_md_url_v1(filepath)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-f', '--file', help='文件Full file name ofMarkdown file.')
    # parser.add_argument('-a', '--action', help='操作类型： L2L, L2W, W2L .')
    # parser.add_argument('-d', '--dir', help='Base directory to store MD images.')

    # args = parser.parse_args()

    # if args.action:
    #     action = args.action
    # if args.dir:
    #     dir_base = args.dir
    # if args.file:
    #     replace_md_url(args.file)
    #file=r"E:\项目记录\liudongdongBlog\source\_posts\Android\AndroidFrameWork.md"
    #replace_md_url_v1(file)
    getallfile(basefolder)

```

### 2. 博客主题更改

![](../../../../picture2023/image-20230920235854269.png)



```python

def transfertoBufferfly(md_file):
    if os.path.splitext(md_file)[1] != '.md':
        print('{}不是Markdown文件，不做处理。'.format(md_file))
        return

    # 本次操作时间戳
    dir_ts = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    with open(md_file, 'r',encoding='utf-8') as f: #使用utf-8 编码打开
        postlines = f.read().split("\n")
        start = 0
        end = 0
        img_index = 0
        cover_index = 0
        top_index = 0
        url = "https://cdn.pixabay.com/photo/2016/11/04/21/34/beach-1799006_640.jpg"
        i = 1
        while postlines[i] != "---":
            if postlines[i].startswith("img:"):
                img_index = i
                temp = postlines[i].split(": ")
                if len(temp) == 2:
                    url = temp[1]
            if postlines[i].startswith("cover"):
                cover_index = i
            if postlines[i].startswith("top_img"):
                top_index = i
            i = i + 1
        if img_index == 0:
            img_index = 3
            postlines.insert(img_index, "img: "+url)
        if cover_index == 0:
            postlines.insert(img_index+1, "cover: "+url) 
        else:
            postlines[cover_index] = "cover: "+url
        if top_index == 0:
            postlines.insert(img_index+1, "top_img: "+url) 
        open(md_file, 'w', encoding='utf-8').write("\n".join(postlines))

def tranferThemeForAllFiles(path):
    allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath=os.path.join(path,file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            tranferThemeForAllFiles(filepath)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            print(filepath)
            transfertoBufferfly(filepath)    
    
```



---

> 作者: liudongdong  
> URL: liudongdong1.github.io/mdpictureurlhandle/  

