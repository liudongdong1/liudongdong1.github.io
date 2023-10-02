# Hexo Blog Introduce


> Welcome to [Hexo](https://hexo.io/)! This is my very first post. Check [documentation](https://hexo.io/docs/) for more info. If you get any problems when using Hexo, you can find the answer in [troubleshooting](https://hexo.io/docs/troubleshooting.html).
>

## Hexo Introduce

> - Hexo is a fast, simple and powerful blog framework. You write posts in [Markdown](http://daringfireball.net/projects/markdown/) (or other markup languages) and Hexo generates static files with a beautiful theme in seconds.
> - Hexo+github+gitee blog deployment [tutorial](https://yafine66.gitee.io/posts/4ab2.html#toc-heading-60)
>   - download Git&&Node.js
>   - Github Register && GithubPage Create
>   - Configure Git user&&mail
>   - Install Theme && Config
>   - Config Some Plugins
> - Good Github Page Recommand:
>   - https://mazhuang.org/
>   - http://www.liberxue.com/
>   - https://rickfang666.github.io/about/
>   - https://ahrilove.top/
>

## Hexo Command

```bash
$ npm install hexo -g #安装  
$ npm update hexo -g #升级  
$ hexo init #初始化
$ hexo new page "categories"  #新建页面
# 简写
$ hexo n "我的博客" == hexo new "我的博客" #新建文章
$ hexo p == hexo publish
$ hexo g == hexo generate#生成
$ hexo s == hexo server #启动服务预览  对跟配置文件修改需要重启
$ hexo d == hexo deploy#部署
# 服务器
$ hexo server #Hexo 会监视文件变动并自动更新，您无须重启服务器。
$ hexo server -s #静态模式
$ hexo server -p 5000 #更改端口
$ hexo server -i 192.168.1.1 #自定义 IP
$ hexo clean #清除缓存db.json 网页正常情况下可以忽略此条命令

#需要删掉用命令新建的文章或页面时，只需要进入 Hexo 根目录下的 source 文件夹，删除对应文件或文件夹即可

$ hexo g #生成静态页面至public目录
$ hexo s #开启预览访问端口（默认端口4000，'ctrl + c'关闭server）
$ hexo d #将.deploy目录部署到GitHub
#监视文件变动
hexo generate --watch #监视文件变动
```

> | 配置选项      | 默认值                         | 描述                                                         |
> | :------------ | :----------------------------- | :----------------------------------------------------------- |
> | title         | `Markdown` 的文件标题          | 文章标题，强烈建议填写此选项                                 |
> | date          | 文件创建时的日期时间           | 发布时间，强烈建议填写此选项，且最好保证全局唯一             |
> | author        | 根 `_config.yml` 中的 `author` | 文章作者                                                     |
> | img           | `featureImages` 中的某个值     | 文章特征图，推荐使用图床(腾讯云、七牛云、又拍云等)来做图片的路径。如: [http://xxx.com/xxx.jpg](https://yafine66.gitee.io/go.html?url=aHR0cDovL3h4eC5jb20veHh4LmpwZw==) |
> | top           | `true`                         | 推荐文章（文章是否置顶），如果 `top` 值为 `true`，则会作为首页推荐文章 |
> | cover         | `false`                        | `v1.0.2`版本新增，表示该文章是否需要加入到首页轮播封面中     |
> | coverImg      | 无                             | `v1.0.2`版本新增，表示该文章在首页轮播封面需要显示的图片路径，如果没有，则默认使用文章的特色图片 |
> | password      | 无                             | 文章阅读密码，如果要对文章设置阅读验证密码的话，就可以设置 `password` 的值，该值必须是用 `SHA256` 加密后的密码，防止被他人识破。前提是在主题的 `config.yml` 中激活了 verifyPassword选项 |
> | toc           | `true`                         | 是否开启 TOC，可以针对某篇文章单独关闭 TOC 的功能。前提是在主题的 `config.yml` 中激活了 `toc` 选项 |
> | mathjax       | `false`                        | 是否开启数学公式支持 ，本文章是否开启 `mathjax`，且需要在主题的 `_config.yml` 文件中也需要开启才行 |
> | summary       | 无                             | 文章摘要，自定义的文章摘要内容，如果这个属性有值，文章卡片摘要就显示这段文字，否则程序会自动截取文章的部分内容作为摘要 |
> | categories    | 无                             | 文章分类，本主题的分类表示宏观上大的分类，只建议一篇文章一个分类 |
> | tags          | 无                             | 文章标签，一篇文章可以多个标签                               |
> | reprintPolicy | cc_by                          | 文章转载规则， 可以是 cc_by, cc_by_nd, cc_by_sa, cc_by_nc, cc_by_nc_nd, cc_by_nc_sa, cc0, noreprint 或 pay 中的一个 |
>

## Error Record

- FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory in ionic 3

> export NODE_OPTIONS="--max-old-space-size=5120" #increase to 5gb
> export NODE_OPTIONS="--max-old-space-size=6144" #increase to 6gb
> export NODE_OPTIONS="--max-old-space-size=7168" #increase to 7gb
> export NODE_OPTIONS="--max-old-space-size=8192" #increase to 8gb



- ERROR Process failed: _posts/存储设计/虚拟化技术/Minikube-tutorial.md
  YAMLException: end of the stream or a document separator is expected at line 2, column 5:
    date: 2022-05-2 22:10:04
> 检查：后是否有空格



## Hugo

- https://yanweixin.github.io/post/hugo-tutorial/ hugo 部署教程
- hexo gitee 部署教程： https://juejin.cn/post/7079959523976282119
- hugo 远程部署教程：https://lewky.cn/posts/hugo-1.html/#%E7%94%9F%E6%88%90%E9%9D%99%E6%80%81%E9%A1%B5%E9%9D%A2
- hugo css 路径问题: https://blog.csdn.net/qq_38250687/article/details/119455302
```
chcp 65001
rem 定义变量延迟环境，关闭回显
@echo off&setlocal enabledelayedexpansion
rem 读取config.toml所有内容
for /f "eol=* tokens=*" %%i  in (config.toml) do (
rem 设置变量a为每行内容
set a=%%i
set "a=!a:http://tablerows.gitee.io/tablerow.github.io/=https://baiban114.github.io/tablerow.github.io/!"
rem 把修改后的全部行存入$
echo !a!>>$)
rem 用$的内容替换原来config.toml内容
move $ config.toml

hugo -D
hugo
cd ./public
git add -A
git commit -m "脚本提交"
git push -u origin master

cd ..

@echo off&setlocal enabledelayedexpansion
for /f "eol=* tokens=*" %%i  in (config.toml) do (
set a=%%i
set "a=!a:https://baiban114.github.io/tablerow.github.io/=http://tablerows.gitee.io/tablerow.github.io/!"
echo !a!>>$)
move $ config.toml

hugo -D
hugo
cd ./public
git add -A
git commit -m "脚本提交"
git push -u gitee master
# git remote set-url origin git@github.com:liudongdong1/liudongdong1.github.io.git
pause
```

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/hexo-blog-introduce/  

