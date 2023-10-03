# Lambda


### 1. 普通函数嵌套

> 定义一个普通的python函数并嵌入Lambda，函数接收传入的一个参数x。然后将此参数添加到lambda函数提供的某个未知参数y中求和。只要我们使用new_func()，就会调用new_func中存在的lambda函数。每次，我们都可以将不同的值传递给参数。

```python
def new_func(x)：
    return(lambda y：x + y)     #y  是待处理的参数
t = new_func(3)
u = new_func(2)
print(t(3))
print(u(3))
```

### 2. Lambda+filter

```python
my_list = [2,3,4,5,6,7,8] 
new_list = list(filter(lambda a:(a / 3 == 2),my_list))
print(new_list)
```

### 3. lambda+map

```python
my_list = [2,3,4,5,6,7,8] 
new_list = list(map(lambda a:(a / 3!= 2),li))
print(new_list)
```

### 4. lambda+reduce

```python
from functools import reduce 
reduce(lambda a,b: a+b,[23,21,45,98])
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201119220126880.png)

### 5. Lambda 函数参数 

```python
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)

def extract_features(imgs, feature_fns, verbose=False):
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
```

- 学习于：https://cloud.tencent.com/developer/article/1453528

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/lambda/  

