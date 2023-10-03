# picture_Glide


> glide是google官方推荐的图片加载框架，[github](https://so.csdn.net/so/search?q=github&spm=1001.2101.3001.7020)地址为bumptech/glide 。glide的强大在于它的生命周期管理(glide可以根据Activity的生命周期自动加载或者暂停图片任务)；glide使用了三级缓存(一级活跃缓存、二级内存缓存、三级磁盘缓存)；gilide使用了BitmapTool机制对图片内存进行复用，可以防止界面快速滑动时的内存不断申请、释放造成的内存抖动；glide可以使用Thumbnail预览图的方式提高加载速率和加载体验。 主流图片加载库： [Glide](https://github.com/bumptech/glide)、[Picasso](https://github.com/square/picasso)、[Fresco](https://www.fresco-cn.org/)
>
> - 与使用环境生命周期相绑定：RequestManagerFragment & SupportRequestManagerFragment
> - 内存的三级缓存池：LruMemoryResources, ActiveResources, BitmapPool
> - 内存复用机制：BitmapPool

todo？ 结合代码理解这个第三方库设计思想

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM2OTU1MzMy,size_16,color_FFFFFF,t_70%23pic_center.png)

### Resource

- https://blog.51cto.com/zhaoyanjun/3815829  glide 使用教程
- https://blog.csdn.net/sinat_36955332/article/details/109774239 glide 模块解析
- https://zhuanlan.zhihu.com/p/81414586  todo？ 优势解析
- https://github1s.com/bumptech/glide/blob/HEAD/samples/flickr/src/main/java/com/bumptech/glide/samples/flickr/FlickrPhotoGrid.java

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/picture_glide/  

