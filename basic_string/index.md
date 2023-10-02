# Basic_String


> 字符串广泛应用 在 Java 编程中，在 Java 中字符串属于对象，Java 提供了 String 类来创建和操作字符串。

#### 1. 创建

```java
String s1 = "Runoob";              // String 直接创建
String s2 = "Runoob";              // String 直接创建
String s3 = s1;                    // 相同引用
String s4 = new String("Runoob");   // String 对象创建
String s5 = new String("Runoob");   // String 对象创建
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/java-string-1-2020-12-01.png)

#### 2. 格式化字符串

```java
String.format("浮点型变量的值为 " +
              "%f, 整型变量的值为 " +
              " %d, 字符串变量的值为 " +
              " %s", floatVar, intVar, stringVar);   // 这种方式之前没有使用过
```

#### 3. 常用函数

- [char charAt(int index)](https://www.runoob.com/java/java-string-charat.html): 返回指定索引处的 char 值。

- [int compareTo(Object o)](https://www.runoob.com/java/java-string-compareto.html)：把这个字符串和另一个对象比较。
- [int compareTo(String anotherString)](https://www.runoob.com/java/java-string-compareto.html): 按字典顺序比较两个字符串。

- [int compareToIgnoreCase(String str)](https://www.runoob.com/java/java-string-comparetoignorecase.html) :按字典顺序比较两个字符串，不考虑大小写。

- [String concat(String str)](https://www.runoob.com/java/java-string-concat.html): 将指定字符串连接到此字符串的结尾。
- [boolean contentEquals(StringBuffer sb)](https://www.runoob.com/java/java-string-contentequals.html): 当且仅当字符串与指定的StringBuffer有相同顺序的字符时候返回真。
- [static String copyValueOf(char[] data)](https://www.runoob.com/java/java-string-copyvalueof.html): 返回`指定数组中表示该字符序列的 String`。
- [ boolean endsWith(String suffix)](https://www.runoob.com/java/java-string-endswith.html) 测试此字符串是否以指定的后缀结束。
- [ boolean equals(Object anObject)](https://www.runoob.com/java/java-string-equals.html) 将此字符串与指定的对象比较。
- [boolean equalsIgnoreCase(String anotherString)](https://www.runoob.com/java/java-string-equalsignorecase.html): 将此 String 与另一个 String 比较，不考虑大小写。
- [ byte[] getBytes()](https://www.runoob.com/java/java-string-getbytes.html)  使用平台的默认字符集将此 String 编码为 byte 序列，并将结果存储到一个`新的 byte 数组`中。
- [byte[] getBytes(String charsetName)](https://www.runoob.com/java/java-string-getbytes.html): 使用指定的字符集将此 String 编码为 byte 序列，并将结果存储到一个新的 byte 数组中。
- [ int hashCode()](https://www.runoob.com/java/java-string-hashcode.html) 返回此字符串的哈希码。
- [ int indexOf(int ch)](https://www.runoob.com/java/java-string-indexof.html) 返回指定字符在此字符串中第一次出现处的索引。
- [ int indexOf(int ch, int fromIndex)](https://www.runoob.com/java/java-string-indexof.html) 返回在此字符串中第一次出现指定字符处的索引，从指定的索引开始搜索。
- [int indexOf(String str)](https://www.runoob.com/java/java-string-indexof.html): 返回指定子字符串在此字符串中第一次出现处的索引。
- [ int lastIndexOf(int ch)](https://www.runoob.com/java/java-string-lastindexof.html) : 返回指定字符在此字符串中最后一次出现处的索引。
- [int lastIndexOf(int ch, int fromIndex)](https://www.runoob.com/java/java-string-lastindexof.html): 返回指定字符在此字符串中最后一次出现处的索引，从指定的索引处开始进行反向搜索。
- [ int length()](https://www.runoob.com/java/java-string-length.html) 返回此字符串的长度。
- [boolean matches(String regex)](https://www.runoob.com/java/java-string-matches.html): 告知此字符串是否匹配给定的正则表达式。
- [ String replace(char oldChar, char newChar)](https://www.runoob.com/java/java-string-replace.html) :返回一个新的字符串，它是通过用 newChar 替换此字符串中出现的所有 oldChar 得到的。
- [ String replace(char oldChar, char newChar)](https://www.runoob.com/java/java-string-replace.html) 返回一个新的字符串，它是通过用 newChar 替换此字符串中出现的所有 oldChar 得到的。
- [ boolean startsWith(String prefix)](https://www.runoob.com/java/java-string-startswith.html) :测试此字符串是否以指定的前缀开始。
- [ String substring(int beginIndex, int endIndex)](https://www.runoob.com/java/java-string-substring.html) 返回一个新字符串，它是此字符串的一个子字符串。
- [Char[] toCharArray()](https://www.runoob.com/java/java-string-tochararray.html): 将此`字符串转换为一个新的字符数组`。
- [ String toLowerCase()](https://www.runoob.com/java/java-string-tolowercase.html) 使用默认语言环境的规则将此 String 中的所有字符都转换为小写。
- [String toString()](https://www.runoob.com/java/java-string-tostring.html): 返回此对象本身（它已经是一个字符串！）。
- [ String trim()](https://www.runoob.com/java/java-string-trim.html) 返回字符串的副本，`忽略前导空白和尾部空白`。
- [static String valueOf(primitive data type x)](https://www.runoob.com/java/java-string-valueof.html): 返回给定data type类型x参数的字符串表示形式。
- [contains(CharSequence chars)](https://www.runoob.com/java/java-string-contains.html): 判断`是否包含`指定的字符系列。
- [ isEmpty()](https://www.runoob.com/java/java-string-isempty.html) :判断字符串`是否为空`。

#### 4. 特殊字符分割

##### .1. splite 分割特殊符号

```java
- “.”和“|”，* ^ : | . \，都是转义字符，必须得加"\\";
- String[] listItem3=defect.getFileUrl().split("\\\\");   // 截取斜杠/
"1234567891^1234567890".split("\\^")[1]    //截取^
```

##### .2. 占位符

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211210091844780.png)

#### 5. StringBuffer & StringBuilder

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20211210092406229.png)

- public StringBuffer append(String s)： 将指定的字符串追加到此字符序列。
- public StringBuffer reverse()： 将此字符序列用其`反转`形式取代。
- public delete(int start, int end)：`移除此序列的子字符串中的字符`。
- int capacity()： 返回`当前容量`。
-  int lastIndexOf(String str, int fromIndex): 返回 String 对象中子字符串最后出现的位置。
-  int indexOf(String str) :返回第一次出现的指定子字符串在该字符串中的索引。
- int length():  返回`长度（字符数）`。
-  String substring(int start, int end) :返回一个新的 `String`，它包含此序列当前所包含的字符子序列。

#### 6. SimpleDateFormat

> SimpleDateFormat是Java提供的一个格式化和解析日期的工具类。它允许进行格式化（日期 -> 文本）、解析（文本 -> 日期）和规范化。SimpleDateFormat 使得可以选择任何用户定义的日期-时间格式的模式。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/v2-0bf024c6e22519d64744de579d68b309_720w.jpg)

```java
public static String toLongDateString(Date dt){
    SimpleDateFormat myFmt=new SimpleDateFormat("yyyy年MM月dd日 HH时mm分ss秒 E ");       
    return myFmt.format(dt);
}
public static String toShortDateString(Date dt){
    SimpleDateFormat myFmt=new SimpleDateFormat("yy年MM月dd日 HH时mm分");       
    return myFmt.format(dt);
}   
public static String toLongTimeString(Date dt){
    SimpleDateFormat myFmt=new SimpleDateFormat("HH mm ss SSSS");       
    return myFmt.format(dt);
}
public static String toShortTimeString(Date dt){
    SimpleDateFormat myFmt=new SimpleDateFormat("yy/MM/dd HH:mm");       
    return myFmt.format(dt);
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/basic_string/  

