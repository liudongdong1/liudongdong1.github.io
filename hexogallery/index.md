# HexoGallery


From: https://juejin.cn/post/6891086750484004877#heading-8

### 新建相册页面

- 在站点的`source`文件夹下面新建一个`gallery`相册页面。

```
hexo new page gallery
```

- 打开新建的gallery文件夹，里面会有一个index.md文件，设置`index.md`文件内容。

```
---
title: gallery
date: 2020-10-05 12:00:00
type: "gallery"
layout: "gallery"
---
```

### 新建相册图片展示页面

1. 在gallery文件夹（也就是刚才创建的那个文件），可以在里面新建一些文件夹，也就是相册文件夹。例如：

![](https:////p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b63036080c464afe9b504c8d3a09ec35~tplv-k3u1fbpfcp-zoom-1.image)

1. 然后在新建的相册文件夹里，分别在每个文件夹里新建文件`index.md`，例如：

![](https:////p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/da980a8a676447f1900c0f2ea5471041~tplv-k3u1fbpfcp-zoom-1.image) 并分别设置每个`index.md`文件内容：

```shell
title: 峨眉山之行
date: 2020-10-02 23:00:17
type: "galleries"
layout: "galleries"
password: 
```

#### 实现加密相册

注意：此处的galleries代表接下里要新增的galleries.ejs文件，而password是给相册设置访问密码。

在上面`index.md`文件中，可以实现加密功能，加密使用`SHA256`加密，所以在加密前需要先将你的密码转换成`SHA256格式`然后输入到最上面的创建相片页面的index.md的`password`里面。至于SHA256加密，请自行网上搜索在线生成，例如: [www.ttmd5.com/hash.php?ty…](https://link.juejin.cn/?target=http%3A%2F%2Fwww.ttmd5.com%2Fhash.php%3Ftype%3D9)

### 编辑主题导航栏加入相册按钮

在站点主题`_config.yml`文件下`menu`菜单下新增gallery页面。 这应是第一步做的事，新建页面后应该立即添加，因为很容易忘却。

```
# 配置菜单导航的名称、路径和图标icon.
menu:
  gallery
    url: /gallery
```

### 新建相册的json文件

- 同样在站点的`source`文件夹下面新建一个`gallery.json`文件。

设置文件内容：（以我的为例，使用时修改成自己的） 这里图片是放在图床上的，可自定义修改图片链接。

```json
[
    {
        "name": "峨眉山之行",
        "cover": "https://i.loli.net/2020/10/05/kBcvAf7INgMLaem.jpg",
        "date": "2017-10",
        "description": "峨眉山之行",
        "url_name": "峨眉山之行",
        "album": [
            {
                "img_url": "https://i.loli.net/2020/10/05/qtOevHpw5XImS1J.jpg",
                "title": "峨眉山之行",
                "describe": "峨眉山之行"
            },
            {
                "img_url": "https://i.loli.net/2020/10/05/4acvniMKTx8euqp.jpg",
                "title": "峨眉山之行",
                "describe": "峨眉山之行"
            },
            {
                "img_url": "https://i.loli.net/2020/10/05/4acvniMKTx8euqp.jpg",
                "title": "峨眉山之行",
                "describe": "峨眉山之行"
            }
        ]
    }
]
```

注意：设置该json文件里的url_name属性值时，url_name属性值必须和对应相册文件里的index.md文件的title属性值一样。
所以建议，除了describe值自定义，其他属性值一律采用和title值一样的。

### 编辑相册及图片展示页面

在站点主题文件夹下layout文件夹下新建文件 `gallery.ejs` 和 `galleries.ejs`

![](https:////p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e65fcc0552514a77a97635d6d1278422~tplv-k3u1fbpfcp-zoom-1.image)

- gallery.ejs： 相册页面
- galleries.ejs：相册图片展示页面

#### 编辑相册页面gallery.ejs

编辑`gallery.ejs`内容： 以下是我的源代码，若同样主题，即可直接复制，其他主题可做参考，根据需要修改即可使用。

```js
<%- partial('_partial/bg-cover') %>
<!-- 增加相册显示的特效样式 -->
<style>
    .photo {
        padding: 0 40px!important;
        display: inline-block;
        position: relative;
        transition: all .6s;
    }
    .biaotiss {
        padding: 8px 10px;
        color: #4c4c4e;
        text-align: center;
        font-weight: 400;
        font-size: 18px;
    }
    .img-item {
        padding: .2rem;
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 1px 2px 2px 0 #aaa;
        transition: all .4s cubic-bezier(.63,.15,.03,1.12);
        margin-bottom: 20px;
    }
    .photo:nth-child(odd) {
        transform: rotateZ(-5deg);
    }    
    .photo:nth-child(even) {
        transform: rotateZ(5deg);
    }
    .text_des{
        position: absolute;
        width: 92%;
        height: 100%;
        top: 0;
        color: #000;
        overflow: hidden;
    }
    .text_des h3{
        margin: 5px 0 8px 0px;
        right: 40px;
        font-size: 1.5rem;
        font-weight: bold;
        line-height: 1.7rem;
        position: absolute;
        top: 10%;
        font-style: italic;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        /*transform: translateX(200px);  */
    }
    .text_des p{
        left: 15px;
        position: absolute;
        top: 50%;
    /*    transform: translateX(-200px); 
        transition-delay: 0.2s;    */
    }
    .my-gallery .photo img {
        border-radius: 10px;
        height: 240px;
    }
    .card .card-content {
        padding: 24px 24px 24px 24px;
    }
    .miaoshus {
        padding: 20px;
        border: 1px dashed #e6e6e6;
        color: #969696;
        position: relative;
        display: inline-block;
        width: 95%;
        background: #fbfbfb50;
        border-radius: 10px;
        font-size: 16px;
        margin: 12px auto;
    }
    body.dark .miaoshus {
        background: 0 0;
        border: 1px dashed #888;
    }
    .img-item {
            transition: all 0.4s ease-out;
        }
        .img-item a img{
            opacity: 1;
            transition: all 0.4s ease-out;
        }
        .img-item a:hover img{
            opacity: 0.8;
        }   
        .biaotiss {
            font-family: MV Boli;
        }   
        .miaoshus .title {
        font-family: MV Boli;
    } 
    /*大屏幕下(桌面类)的样式*/
    @media only screen and (min-width: 993px) {
        .text_des h3{
            transform: translateX(200px);
        }
        .text_des p{
            transform: translateX(-200px); 
            transition-delay: 0.2s;
        }
        .animate-text {
            opacity: 0;
            transition: all 0.6s ease-in-out; 
        }
    }
    /*小屏幕下(桌面类)的样式*/
    @media only screen and (max-width: 500px) {
        .my-gallery .photo img {
            height: 186px;
        }
    }

</style>
<main class="content">
    <div class="container chip-container">
        <div class="card">
            <div class="card-content">
                <div class="tag-title center-align">
                    <div class="miaoshus">
                        <div class="title center-align">“ <%- theme.gallery.title %> ”</div>
                        “ 这里有光影流年，还有两朵穿衣裳的云，相拥在明天的河岸。”                      
                    </div>
                </div>
                <div class="my-gallery">
                    <div class="row">
                    <% if (site.data && site.data.gallery) { %>
                        <% var gallery = site.data.gallery; %>
                        <% for (var i = 0, len = gallery.length; i < len; i++) { %>
                            <% var photo = gallery[i]; %>
                            <% if (photo.cover){ %>
                            <div class="photo col s12 m6 l4">
                                <div class="img-item">
                                    <a href="./<%- photo.url_name %>">
                                        <img src="/medias/loading.svg" data-src="<%- photo.cover %>" class="responsive-img" alt="img"+<%- i %> />
                                        <div class="biaotiss">“ <%- photo.name %> ”<br>“ <%- photo.date %> ”</div>
                                    </a>
                                </div>                              
                            </div>
                            <% } %>
                        <% } %>
                    <% } %>
                </div>
            </div>
        </div>
    </div>
    <script>
        start()
            $(window).on('scroll', function(){
            start();
        })
        function start(){
              //.not('[data-isLoaded]')选中已加载的图片不需要重新加载
            $('.container img').not('[data-isLoaded]').each(function(){
                var $node = $(this)
                if( isShow($node) ){
                    loadImg($node);
                }
            });
        }
        //判断一个元素是不是出现在窗口(视野)
        function isShow($node){
            return $node.offset().top <= $(window).height() + $(window).scrollTop();
        }
        //加载图片
        function loadImg($img){
        //.attr(值)
        //.attr(属性名称,值)
            $img.attr('src', $img.attr('data-src')); //把data-src的值 赋值给src
            $img.attr('data-isLoaded', 1);//已加载的图片做标记
        }
    </script>
</main>
```

#### 编辑相册图片展示页面galleries.ejs

编辑`galleries.ejs`内容： 以下是我的源代码，若同样主题，即可直接复制，其他主题可做参考，根据需要修改即可使用。

```js
<!-- 加密功能 -->
<% if (theme.PhotoVerifyPassword.enable) { %>
    <script src="<%- theme.libs.js.crypto %>"></script>
    <script>
        (function() {
            let pwd = '<%- page.password %>';

            if (pwd && pwd.length > 0) {
                if (pwd !== CryptoJS.SHA256(prompt('<%- theme.PhotoVerifyPassword.promptMessage %>')).toString(CryptoJS.enc.Hex)) {
                    alert('<%- theme.PhotoVerifyPassword.errorMessage %>');
                    location.href = '<%- url_for("/")  %>';
                }
            }
        })();
    </script>
<% } %>

<!-- <%- partial('_partial/bg-cover') %> -->

<link rel="stylesheet" href="<%- theme.libs.css.baguetteBoxCss %>">
<!-- 该主题自带的lightGallery.js在图片多的时候会很卡，所以弃用，使用了一个我在网上找的baguetteBox，很轻量级，就是功能少了点 -->
<script src="<%- theme.libs.js.baguetteBoxJs %>"></script> 
<style>    
    .my-gallery .photo img {
        transition: all 0.6s ease-in-out; 
    }     
    .my-gallery .photo:hover img {
        opacity: 0.6;
        transform: scale(1.05);
    }      
    .my-gallery {
        margin: 20px auto;
    }   
    .miaoshus .title {
        font-family: MV Boli;
    }
    .miaoshus {
        padding: 20px;
        border: 1px dashed #e6e6e6;
        color: #969696;
        position: relative;
        display: inline-block;
        width: 75%;
        background: #fbfbfb50;
        border-radius: 10px;
        font-size: 16px;
        margin: 24px auto 12px;
        
    }
    body.dark .miaoshus {
        background: 0 0;
        border: 1px dashed #888;
    }
    body {
        overflow: visible!important;
    }
    .box {
            position: relative;
        }
    .box img {
        width: 350px;
        vertical-align: top;
        padding: 8px;
        border-radius: 10px;
        transition: all 0.5s;
    }
    .box img:hover {
        transform: scale(1.05);
    }
    .page-footer {
        display: none
    }
    body {
        overflow-y: visible!important;
    }
    header {
        background-color: #000;
    }
    .biaotiss {
            font-family: MV Boli;
        }   
    @media only screen and (max-width: 1058px) {
        .box {
            margin-left: 145px;
        }
    }
    @media only screen and (max-width: 770px) {
        .box {
            margin-left: 15px;
        }
    }
    @media only screen and (max-width: 500px) {
        #previous-button, #next-button {
            display: none;
        }
    }
    @media only screen and (max-width: 380px) {
        .box {
            margin-left: -5px;
        }
    }
    @media only screen and (max-width: 323px) {
        .box img {
            width: 296px;
            left: 0;
        }
    }
</style>

<div class="tag-title center-align">
    <div class="miaoshus">
        <div class="title center-align">
            “ <% if (is_home() && config.subtitle && config.subtitle.length > 0) { %>
                <%= config.subtitle %>
            <% } else { %>
                <%= page.title %>
            <% } %> ”
        </div>
        “ <%- theme.gallery.title %> ”
        “ 这里有光影流年，还有两朵穿衣裳的云，相拥在明天的河岸。”
    </div>
</div>

<!-- 相册 -->
<section class="gallery">
    <div id="myGallery" class="my-gallery">
        <div class="row">
            <div class="box">
            <% if (site.data && site.data.gallery) { %>
                <% var galleries = site.data.gallery; 
                    var pageTitle = page.title;
                    function getCurrentGallery(galleries, pageTitle) {
                        for (let i = 0; i < galleries.length; i++) {
                            if (galleries[i]['name'] == pageTitle) {
                                return galleries[i];
                            }
                        }
                    }
                    var currentGallery = getCurrentGallery(galleries, pageTitle);
                    var photos = currentGallery.album;
                %>
                <% for (var i = 0, len = photos.length; i < len; i++) { %>
                    <% var my_album = photos[i]; %>                   
                        <a href="<%- my_album.img_url %>" data-caption="<%- my_album.title %>">
                            <img class="mat" src="/medias/loading.svg" data-src="<%- my_album.img_url %>" alt="img"+<%- i %> >
                        </a>                   
                <% } %>
            <% } %>
        </div>
        </div>
    </div>
</section>
<script>
    $(function () {
        // 获取图片的宽度(200px)
        let imgWidth = $('.mat').outerWidth(); // 200
        waterfallHandler();
        // 瀑布流处理
        function waterfallHandler() {
            // 获取图片的列数
            let column = parseInt($(window).width() / imgWidth);
            // 高度数组
            let heightArr = [];
            for(let i=0; i<column; i++) {
                heightArr[i] = 0;
            }
            // 遍历所有图片进行定位处理
            $.each($('.mat'), function (index, item) {
                // 当前元素的高度
                let itemHeight = $(item).outerHeight();
                // 高度数组最小的高度
                let minHeight = Math.min(...heightArr);
                // 高度数组最小的高度的索引
                let minIndex = heightArr.indexOf(minHeight);
                
                $(item).css({
                    position: 'absolute',
                    top: minHeight + 'px',
                    left: minIndex * imgWidth + 'px'
                });
                heightArr[minIndex] += itemHeight;
            });
        }
        // 窗口大小改变
        $(window).resize(function () {
            waterfallHandler();
        });
    });
</script>
<script>
    baguetteBox.run('.gallery', {
        // 配置参数
        buttons:Boolean,//是否显示导航按钮。
        noScrollbars:true,//是否在显示时隐藏滚动条。
        titleTag:true,//是否使用图片上的title属性作为图片标题
        async:false,//是否异步加载文件。
    });

    start()
        $(window).on('scroll', function(){
        start();
    })

    function start(){
          //.not('[data-isLoaded]')选中已加载的图片不需要重新加载
        $('.gallery img').not('[data-isLoaded]').each(function(){
            var $node = $(this)
            if( isShow($node) ){
                loadImg($node);
            }
        });
    }

    //判断一个元素是不是出现在窗口(视野)
    function isShow($node){
        return $node.offset().top <= $(window).height() + $(window).scrollTop();
    }
    //加载图片
    function loadImg($img){
    //.attr(值)
    //.attr(属性名称,值)
        $img.attr('src', $img.attr('data-src')); //把data-src的值 赋值给src
        $img.attr('data-isLoaded', 1);//已加载的图片做标记
    }
</script>
```

- 为了实现懒加载，在这里需要在站点主题文件夹下的`medias`文件夹下面放入一个用于懒加载的图片`loading.svg` ，这里你可以直接使用我的加载svg，地址[nekodeng.gitee.io/medias/load…](https://link.juejin.cn/?target=https%3A%2F%2Fnekodeng.gitee.io%2Fmedias%2Floading.svg)

### 编辑主题_config.yml文件内容

在站点主题_config.yml文件下新增以下内容

```js
# 增加了图片页面
gallery: 
  title: 光影流年   #标题
  icon: fa         #这个显示相册页面的图标
  icon2: fa        #这个显示自己的具体相册里面的相册的图标

PhotoVerifyPassword:
  enable: true
  promptMessage: 该相册已加密，请输入密码访问
  errorMessage: 密码错误，将返回主页！
复制代码
```

- 最后，完成，引用我的相册页面dec来结束：
- “ 光影流年 ”
- “ 这里有光影流年，还有两朵穿衣裳的云，相拥在明天的河岸。”


![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210803221618233.png)

- 这里懒加载实现失败： 通过设置俩个url方式来进行处理
- 之后添加照片文件的时候，先在 source/gallery目录下面添加文件夹，新建index.md 文件，然后通过脚本在相册文件中添加对应的图片和缩略图，最后在_data目录下的gallary.json文件中添加对应的文件信息。

```
title: 延吉之行
date: 2021-08-03 19:36:23
type: galleries
layout: galleries
password: 
```



---

> 作者: blinkfox  
> URL: liudongdong1.github.io/hexogallery/  

