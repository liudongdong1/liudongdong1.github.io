# ErrorRecord


### 1. 错误记录

- RuntimeError: Given groups=1, `weight of size [64, 1, 3, 3]`, `expected input[1, 3, 512, 512]` to have `1 channels`, but got `3 channels` instead

```python
#
img = Image.open('test.png')
if img.mode != 'L':
	img = img.convert('L')  #[1, 1, 512, 512]
```

- **TypeError: can’t multiply sequence by non-int of type 'float’**

> 不同数据类型相乘；



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: liudongdong1.github.io/errorrecord/  

