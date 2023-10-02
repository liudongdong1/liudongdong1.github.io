# 保留俩位小数


#### 1. BigDecimal

```java
public static String format1(double value) {
    BigDecimal bd = new BigDecimal(value);
    bd = bd.setScale(2, RoundingMode.HALF_UP);
    return bd.toString();
}
```

#### 2. **DecimalFormat**

```java
public static String format2(double value) {
    DecimalFormat df = new DecimalFormat("0.00");
    df.setRoundingMode(RoundingMode.HALF_UP);
    return df.format(value);
}
```

#### 3. NumberFormat

```java
public static String format3(double value) {
    NumberFormat nf = NumberFormat.getNumberInstance();
    nf.setMaximumFractionDigits(2);
    /*
  * setMinimumFractionDigits设置成2
  * 
  * 如果不这么做，那么当value的值是100.00的时候返回100
  * 
  * 而不是100.00
  */
    nf.setMinimumFractionDigits(2);
    nf.setRoundingMode(RoundingMode.HALF_UP);
    /*
  * 如果想输出的格式用逗号隔开，可以设置成true
  */
    nf.setGroupingUsed(false);
    return nf.format(value);
}
```

#### 4. **java.util.Formatter**

```java
public static String format4(double value) {
    /*
  * %.2f % 表示 小数点前任意位数 2 表示两位小数 格式后的结果为 f 表示浮点型
  */
    return new Formatter().format("%.2f", value).toString();
}
```

#### 5. String.format

```java
public static String format5(double value) {
    return String.format("%.2f", value).toString();
}
double num = 123.4567899;
System.out.print(String.format("%f %n", num)); // 123.456790 
System.out.print(String.format("%a %n", num)); // 0x1.edd3c0bb46929p6 
System.out.print(String.format("%g %n", num)); // 123.457
```

可用标识：

      -，在最小宽度内左对齐,不可以与0标识一起使用。
    
      0，若内容长度不足最小宽度，则在左边用0来填充。
    
      #，对8进制和16进制，8进制前添加一个0,16进制前添加0x。
    
      +，结果总包含一个+或-号。
    
      空格，正数前加空格，负数前加-号。
    
      ,，只用与十进制，每3位数字间用,分隔。
    
      (，若结果为负数，则用括号括住，且不显示符号。

可用转换符：

      b，布尔类型，只要实参为非false的布尔类型，均格式化为字符串true，否则为字符串false。
    
      n，平台独立的换行符, 也可通过System.getProperty("line.separator")获取。
    
      f，浮点数型（十进制）。显示9位有效数字，且会进行四舍五入。如99.99。
    
      a，浮点数型（十六进制）。
    
      e，指数类型。如9.38e+5。
    
      g，浮点数型（比%f，%a长度短些，显示6位有效数字，且会进行四舍五入）


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E4%BF%9D%E7%95%99%E4%BF%A9%E4%BD%8D%E5%B0%8F%E6%95%B0/  

