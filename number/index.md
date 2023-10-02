# Number


> 8种基本数据类型并不支持面向对象的特征，它们既不是类，也不能调用方法。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510095205.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510215738.png)

### 1. Number

> `BigDecimal 和 BigInteger 用于高精度计算`。AtomicInteger 和 AtomicLong 用于多线程应用程序。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510094040.png)

#### 1.1. 类型转化

```java
byte byteValue()
short shortValue()
int intValue()
long longValue()
float floatValue()
double doubleValue()
```

#### 1.2. 比较

```java
int compareTo(Byte anotherByte)
int compareTo(Double anotherDouble)
int compareTo(Float anotherFloat)
int compareTo(Integer anotherInteger)
int compareTo(Long anotherLong)
int compareTo(Short anotherShort)
boolean equals(Object obj)
```

| static Integer decode(String s)             | 将字符串解码为整数。可以接受十进制，八进制或十六进制数字的字符串表示作为输入。 |
| ------------------------------------------- | ------------------------------------------------------------ |
| static int parseInt(String s)               | 返回一个整数（仅限十进制）。                                 |
| static int parseInt(String s, int radix)    | 以给定的十进制，二进制，八进制或十六进制（radix 分别等于 10、2、8 或 16）数字的字符串表示形式返回一个整数作为输入。 |
| String toString()                           | 返回 Integer 的 String 表示形式。                            |
| static String toString(int i)               | 返回 String 表示指定整数的对象。                             |
| static Integer valueOf(int i)               | 返回一个 Integer 持有指定基元的值的对象。                    |
| static Integer valueOf(String s)            | 返回一个 Integer 持有指定字符串表示的值的对象。              |
| static Integer valueOf(String s, int radix) | 返回一个 Integer 持有指定字符串表示形式的整数值的对象，用 radix 的值进行解析。例如，如果 s=“333” 且 radix = 8，则该方法返回八进制数 333 的十进制整数等效值。 |

#### 1.3. 格式化输出

```java
System.out.format(Locale.FRANCE,
                  "The value of the float " + "variable is %f, while the " +
                          "value of the integer variable " + "is %d, and the string is %s%n",
                  floatVar, intVar, stringVar);
```

#### 1.4. 自带函数

```java
double min(double arg1, double arg2)
double max(double arg1, double arg2)
double abs(double d)
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/number/  

