# PictureCompress


![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200814165610056.png)

## 1. 鲁班图片压缩算法

```python
from PIL import Image
import os
from shutil import copyfile
from math import ceil
class Luban(object):
    def __init__(self, ignoreBy=102400, quality=60):
        self.ignoreBy = ignoreBy
        self.quality = quality
    def setPath(self, path):
        self.path = path
    def setTargetDir(self, foldername="target"):
        self.dir, self.filename = os.path.split(self.path)
        self.targetDir = os.path.join(self.dir, foldername)
        if not os.path.exists(self.targetDir): 
            os.makedirs(self.targetDir)  
        self.targetPath = os.path.join(self.targetDir, "c_"+self.filename)   
    def load(self):
        self.img = Image.open(self.path)
        if self.img.mode == "RGB": 
            self.type = "JPEG"
        elif self.img.mode == "RGBA": 
            self.type = "PNG"
        else: # 其他的图片就转成JPEG
            self.img == img.convert("RGB")
            self.type = "JPEG"
    def computeScale(self):
        # 计算缩小的倍数
        srcWidth, srcHeight = self.img.size 
        srcWidth = srcWidth + 1 if srcWidth % 2 == 1 else srcWidth
        srcHeight = srcHeight + 1 if srcHeight % 2 == 1 else srcHeight
        longSide = max(srcWidth, srcHeight)
        shortSide = min(srcWidth, srcHeight)
        scale = shortSide / longSide
        if (scale <= 1 and scale > 0.5625):
            if (longSide < 1664): 
                return 1
            elif (longSide < 4990):
                return 2
            elif (longSide > 4990 and longSide < 10240):
                return 4
            else:
                return max(1, longSide // 1280)
        elif (scale <= 0.5625 and scale > 0.5):
            return max(1, longSide // 1280)
        else: 
            return ceil(longSide / (1280.0 / scale))
    def compress(self):
        self.setTargetDir()
        # 先调整大小，再调整品质
        if os.path.getsize(self.path) <= self.ignoreBy:
            copyfile(self.path, self.targetPath)
        else:
            self.load() 
            scale = self.computeScale()
            srcWidth, srcHeight = self.img.size
            cache = self.img.resize((srcWidth//scale, srcHeight//scale), Image.ANTIALIAS)  
            cache.save(self.targetPath, self.type, quality=self.quality)
if __name__ == '__main__':
    path = r"C:\Users\William Chen\Documents\GitHub\Luban-Py\test.jpg"
    compressor = Luban()
    compressor.setPath(path)
    compressor.compress()
```

```c++
+(void)zipNSDataWithImage:(UIImage *)sourceImage imageBlock:(ImageBlock)block{
    //进行图像尺寸的压缩
    CGSize imageSize = sourceImage.size;//取出要压缩的image尺寸
    CGFloat width = imageSize.width;    //图片宽度
    CGFloat height = imageSize.height;  //图片高度
    
    CGFloat scale = height/width;
    //0.宽高比例大于8
    if (scale > 8.0 || scale < 1/8.) {
        if (scale > 8.0) {
            if (width > 1080) {
                width = 1080;
                height = width * scale;
            }else {
                //不压缩
            }
        }else {
            if (height > 1080.) {
                height = 1080;
                width = height / scale;
            }else {
                //不压缩
            }
        }
        //1.宽高大于1080(宽高比不按照2来算，按照1来算)
    }else if (width>1080 && height>1080) {
        if (height > width) {
            CGFloat scale = height/width;
            width = 1080;
            height = width*scale;
        }else{
            CGFloat scale = width/height;
            height = 1080;
            width = height*scale;
        }
        //2.宽大于1080高小于1080
    }else if(width>1080 && height<1080){
        CGFloat scale = height/width;
        width = 1080;
        height = width*scale;
        //3.宽小于1080高大于1080
    }else if(width<1080 && height>1080){
        CGFloat scale = width/height;
        height = 1080;
        width = height*scale;
        //4.宽高都小于1080
    }else{
    }
    UIGraphicsBeginImageContext(CGSizeMake(width, height));
    [sourceImage drawInRect:CGRectMake(0,0,width,height)];
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    //进行图像的画面质量压缩,统一按0.5压缩
    NSData *data=UIImageJPEGRepresentation(newImage, 0.5);
//    if (data.length>100*1024) {
//        if (data.length>1024*1024) {//1M以及以上
//            data=UIImageJPEGRepresentation(newImage, 0.5);
//        }else if (data.length>512*1024) {//0.5M-1M
//            data=UIImageJPEGRepresentation(newImage, 0.8);
//        }else {
//            //0.25M-0.5M
//            data=UIImageJPEGRepresentation(newImage, 0.9);
//        }
//    }
    block(data);
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/picturecompress/  

