# JavaFloatConvert


> Double.floatValue(), Double.doubleValue() 是准确的，但是Float.doubleValue()是不准确的, 单精度转双精度的时候，双精度会对单精度进行补位。导致出现偏差。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211210092737731.png)

```java
Float f=new Float(14.1);
System.out.println(f.floatValue());
System.out.println(f.doubleValue());
System.out.println(Double.parseDouble(f.floatValue()+""));
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/doublefloatconvert/  

