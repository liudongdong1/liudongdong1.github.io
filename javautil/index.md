# javaUtil


> 学习记录javaUtil各种工具使用；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201216234129227.png)

### 1. StringRelative

#### 1.1. StringTokenizer

> **StringTokenizer(String str)：**构造一个用来解析str的StringTokenizer对象。java默认的分隔符是“空格”、“制表符(‘\t’)”、“换行符(‘\n’)”、“回车符(‘\r’)”
>
> **StringTokenizer(String str, String delim)：**构造一个用来解析str的StringTokenizer对象，并提供一个指定的分隔符。
>
> **StringTokenizer(String str, String delim, boolean returnDelims)：**构造一个用来解析str的StringTokenizer对象，并提供一个指定的分隔符，同时，指定是否返回分隔符。

1. **int countTokens()：**返回nextToken方法被调用的次数。如果采用构造函数1和2，返回的就是分隔符数量(例2)。

2. **boolean hasMoreTokens()**：`返回是否还有分隔符`。

3. **boolean hasMoreElements()** ：结果同2。

4. **String nextToken()**：`返回从当前位置到下一个分隔符的字符串`。

5. **Object nextElement()**：结果同4。

6. **String nextToken(String delim)**：与4类似，以指定的分隔符返回结果。

```java
public static void main(String[] args) {
    String s = new String("The Java platform is the ideal platform for network computing");
    StringTokenizer st = new StringTokenizer(s);
    System.out.println("Token Total: " + st.countTokens());
    while (st.hasMoreElements()) {
        System.out.println("-----------------------");
        // 返回从当前位置到下一个分隔符的字符串。
        System.out.println(st.nextToken());
        // System.out.println(st.nextToken("i")); //以指定的分隔符返回结果 默认为“ ”；
    }
    // 如果上面的循环执行完nextToken()或nextElement()方法，下面的标记将不执行，也就是说不进入while循环
    while (st.hasMoreTokens()) {
        System.out.println("=======");
        System.out.println(st.nextElement());
    }
    String s2 = new String("The=Java=platform=is=the=ideal=platform=for=network=computing");
    // 构造方法中第一个参数表示字符串，第二参数分隔符，第三参数表示是否返回字符串
    StringTokenizer st2 = new StringTokenizer(s2, "=", true);
    System.out.println("Token Total: " + st2.countTokens());
    while (st2.hasMoreElements()) {
        System.out.println(st2.nextToken());
    }
}
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201216235427883.png)

### 学习资源

- [java util 介绍](https://juejin.cn/post/6844903560216313869)
- java api 接口
- java常用类封装总结
- https://www.jishuchi.com/read/java-lang/3049
- [java 集合](https://liudongdong.blog.csdn.net/article/details/79729622)，[迭代器](https://www.jianshu.com/p/ad984becc984) [iterator](https://www.cnblogs.com/Andya/p/12555666.html)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javautil/  

