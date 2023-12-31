# JavaJVM_字节码指令


### 1. 字节码指令

字节码指令是一个字节长度的，代表着某种特殊操作含义的数字，总数不超过256条（[全部字节码指令汇编](https://link.juejin.cn/?target=http%3A%2F%2Fwww.blogjava.net%2FDLevin%2Farchive%2F2011%2F09%2F13%2F358497.html)）。对于大部分与数据类型相关的字节码指令，它们的操作码助记符中都有特殊字符来表明专门为那种数据类型服务，如下表：

| 描述字符 | 含义                                    |
| -------- | --------------------------------------- |
| i        | 基本类型int                             |
| s        | 基本类型short                           |
| l        | 基本类型long,这里注意不是L，L是最后一个 |
| f        | 基本类型float                           |
| d        | 基本类型double                          |
| b        | 基本类型byte                            |
| c        | 基本类型char                            |
| b        | 基本类型boolean                         |
| a        | 对象类型引用reference                   |

这里有一个注意的点，这对于不是整数类型的byte、char、short、boolean。`编译器会在编译器或运行期将byte和short类型的数据带符号扩展(Sign-extend)为相应的int类型数据`，`将boolean和char类型数据零位扩展(Zero-extend)为相应的int数据`。同样在处理上诉类型的数组数据是，也会转换为使用int类型的字节码指令来处理。

#### .1. 加载和存储指令。

加载和存储指令用于将数据在`栈帧中的局部变量表和操作数栈`之间来回传输。

`<类型>load_<下标>`：将一个`局部变量加载到操作数栈`。例如iload_1，将一个int类型局部变量（下标为1，0一般为this）从局部变量表加载到操作栈，其他的也都类似，例如：dload_2,fload_3。 `<类型>store_<下标>`：将一个`数值从操作数栈栈顶存储到局部变量表`。例如istore_3，将一个int类型的数值从操作数栈栈顶存储到局部变量3中，后缀为3，证明局部变量表中已经存在了两个值。 `<类型>const_<具体的值>`：将一个`常量加载到操作数栈`。例如iconst_3,将常量3加载到操作数栈。 `wide扩展`：当上述的下标志超过3时，就不用下划线的方式了，而是使用`istore 6`，load的写法也是一样。 `bipush、sipush、ldc`  :当上述的const指令后面的值变得很大时，该指令也会改变。

- 当` int 取值 -1~5 时`，JVM 采用` iconst 指令将常量压入栈中`。
- 当 int 取值 `-128~127 `时，JVM 采用` bipush 指令将常量压入栈中`。
- 当 int 取值` -32768~32767` 时，JVM 采用` sipush 指令将常量压入栈中`。
- 当 int 取值` -2147483648~2147483647 `时，JVM 采用` ldc 指令将常量压入栈中`。

```java
public void save() {
    int a = 1;
    int b = 6;
    int c = 128;
    int d = 32768 ;
    float f = 2.0f;
}
```

```assembly
Code:
	stack=1, locals=6, args_size=1
        0: iconst_1               //将常量1入栈，
        1: istore_1               //将栈顶的1存入局部变量表,下标为1，因为0存储了整个类的this
        2: bipush        6       //将常量6入栈，同时也是以wide扩展的形式
        4: istore_2               //将栈顶的6存入局部变量表,下标为2
        5: sipush        128    //将常量128入栈，
        8: istore_3               //将栈顶的128存入局部变量表,下标为3 ，后面一样的意思
        9: ldc           #2        // int 32768
        11: istore        4
        13: fconst_2
        14: fstore        5
        16: return
```

#### .2 运算指令。

运算主要分为两种：对`整数数据进行运算的指令`和对`浮点型数据运算的指令`，和前面说的一样，对于byte、char、short、和 boolean类型的算数质量都使用int类型的指令替代。整数和浮点数的运算指令在移除和被领出的时候也有各自不同的表现行为。具体的指令也是在运算指令前加上对应的类型即可，例如加法指令：iadd,ladd,fadd,dadd。

- 加法指令：（i,l,f,d）add
- 减法指令：（i,l,f,d）sub
- 乘法法指令：（i,l,f,d）mul
- 除法指令：（i,l,f,d）div
- 求余指令：（i,l,f,d）rem
- 取反指令：（i,l,f,d）neg
- 位移指令：  ishl、ishr、iushr、lshl、lshr、lushr
- 按位或指令：ior、lor
- 按位与指令：iand、land
- 按位异或指令：  ixor、lxor
- 局部变量自增： iinc（例如for循环中i++）
- 比较指令： dcmpg、dcmpl、fcmpg、fcmpl、lcmp

#### 2.3 类型转换指令。

类型转换指令可以`将两种不同的数值类型进行相互转换`，这些转换一般用于`实现用户代码中的显示类型转换操作`。

Java虚拟机直接支持`宽化数据类型转换`（小范围数据转换为大数据类型）,不需要显示的转换指令，例如int转换long，float和double。举例：`int a=10;long b =a`

Java虚拟机转换`窄化数据类型转换时`，`必须显示的调用转化指令`。举例：`long b=10;int a = (long)b`。

类型转换的字节码指令其实就比较简单了，`<前类型>2<后类型>`，例如i2l,l2i,i2f,i2d。当然这里举的都是基本数据类型，如果是`对象`，当类似宽化数据类型时就直接使用，当类似`窄化数据类型`时，需要`checkcast`指令。

```assembly
public class Main {
    public static void main(String[] args) {
        int a = 1;
        long b = a;
        Parent Parent = new Parent();
        Son son = (Son) Parent;
    }
}
字节码：
  Code:
      stack=2, locals=6, args_size=1
         0: iconst_1
         1: istore_1
         2: iload_1
         3: i2l
         4: lstore_2
         5: new           #2                  // class com/verzqli/snake/Parent
         8: dup
         9: invokespecial #3                  // Method com/verzqli/snake/Parent."<init>":()V
        12: astore        4
        14: aload         4
        16: checkcast     #4                  // class com/verzqli/snake/Son
        19: astore        5
        21: return
```

注意上面这个转换时错误的，父类是不能转化为子类的，`编译期正常，但是运行是会报错的，这就是checkcast指令`的原因。

#### 2.4  对象创建和访问指令

虽然类实例和数组都是对象，但Java虚拟基对`类实例和数组的创建与操作使用了不同的字节码指令`。对象创建后，就可以通过对象访问指令获取对象实例或者数组实例中的字段或者数组元素，这些指令如下。

- `new`:创建类实例的指令
- `newarray、anewarray、multianewarray`：创建数组的指令
- `getfield、putfield、getstatic、putstatic`：访问类字段（static字段，被称为类变量）和实例字段（非static字段，）。
- `(b、c、s、i、l、f、d、a)aload`:很明显，就是`基础数据类型加上aload，将一个数组元素加载到操作数栈。`
- `(b、c、s、i、l、f、d、a)astore`:同上面一样的原理，将操作数栈栈顶的值存储到数组元素中。
- `arraylength`:取数组长度
- `instanceof、checkcast`：检查类实例类型的指令。

#### 2.4  操作数栈管理指令

如同操作一个普通数据结构中的堆栈那样，Java虚拟机提供了一些直接操作操作数栈的指令。

- `pop、pop2`：将操作数栈栈顶的一个或两个元素出栈。
- `dup、dup2、dup_x1、dup2_x1、dup_x2、dup2_x2`:出站栈顶一个或两个数值并将期值复制一份或两份后重新压入栈顶。
- `swap`:将栈顶两个数互换。

#### 2.5  方法调用和返回指令。

方法调用的指令只要包含下面这5条

- `invokespecial`:用于调用一些`需要特殊处理的实例方法`，包括实例`初始化方法`、`私有方法和父类方法`。
- `invokestatic`:用于`调用static方法`。
- `invokeinterface`:用于调用`接口方法`，他会在运行时搜索一个实现了这个接口方法的对象，找出合适的方法进行调用。
- `invokevirtual`:用于`调用对象的实例方法`，根据对象的实际类型进行分派。
- `invokedynamic`:用于在运行时`动态解析出调用点限定符所引用的方法`，并执行该方法。前面4条指令的分派逻辑都固话在Java虚拟机内部，而此条指令的分派逻辑是由用户设定的引导方法决定的。
- `(i,l,f,d, 空)return`:根据前面的类型来确定返回的数据类型，为空时表示void

#### 2.5  异常处理指令。

在Java程序中显示抛出异常的操作（throw语句）都由`athrow`指令来实现。但是处理异常（catch语句）不是由字节码指令来实现的，而是`采用异常表来完成`的，如下例子。

```assembly
public class Main {
   public static void main(String[] args) throws Exception{
       try {
           Main a=new Main();
       }catch (Exception e){
           e.printStackTrace();
       }
   }
}
字节码：
public static void main(java.lang.String[]) throws java.lang.Exception;
   descriptor: ([Ljava/lang/String;)V
   flags: ACC_PUBLIC, ACC_STATIC
   Code:
     stack=2, locals=2, args_size=1
        0: new           #2                  // class com/verzqli/snake/Main
        3: dup
        4: invokespecial #3                  // Method "<init>":()V
        7: astore_1
        8: goto          16
       11: astore_1
       12: aload_1
       13: invokevirtual #5                  // Method java/lang/Exception.printStackTrace:()V
       16: return
```

#### 2.6  同步指令

Java虚拟机可以支持`方法级的同步和方法内部一段指令序列的同步`，这两种同步结构都是使用[Monitor](https://juejin.im/post/6844903840815251469) 实现的。 正常情况下Java运行是同步的，无需使用字节码控制。虚拟机可以从`方法常量池的方法表结构`中的`ACC_SYNCHRONIZE`访问标志得知一个方法是否声明为同步方法。当方法调用时，调用指令将会检查方法的`ACC_SYNCHRONIZE`访问表示是否被设置，如果设置了，执行线程就要求先持有`Monitor`,然后才能执行方法，最后当方法完成时释放`Monitor`。在方法执行期间，执行线程持有了`Monitor`，其他任何一个线程都无法在获取到同一个`Monitor`。如果一个同步方法执行期间抛出了异常，并且在方法内部无法处理次异常，那么这个同步方法所持有的`Monitor`将在异常抛出到同步方法之外时自动释放。 同步一段指令集序列通常是由`synchronized`语句块来表示的，Java虚拟机指令集中有`monitorenter`和`monitorexit`两条指令来支持`synchronized`关键字。如下例子

```assembly
public class Main {
    public void main() {
        synchronized (Main.class) {
            System.out.println("synchronized");
        }
        function();
    }

    private void function() {
        System.out.printf("function");
    }
}

字节码：
 Code:
      stack=3, locals=3, args_size=1
         0: ldc           #2                  // class com/verzqli/snake/Main  将Main引用入栈
         2: dup                                // 复制栈顶引用 Main
         3: astore_1                        // 将栈顶应用存入到局部变量astore1中
         4: monitorenter                  // 将栈顶元素（Main）作为锁，开始同步
        5: getstatic     #3                 // Field java/lang/System.out:Ljava/io/PrintStream;
         8: ldc           #4                   // String synchronized ldc指令在运行时创建这个字符串
        10: invokevirtual #5                  // Method java/io/PrintStream.println:(Ljava/lang/String;)V
        13: aload_1                         // 将局部变量表的astore1入栈（Main）
        14: monitorexit                    //退出同步
        15: goto          23                  // 方法正常结束，跳转到23
        18: astore_2                        //这里是出现异常走的路径，将栈顶元素存入局部变量表
        19: aload_1                          // 将局部变量表的astore1入栈（Main）
        20: monitorexit                      //退出同步
        21: aload_2                          //将前面存入局部变量的异常astore2入栈
        22: athrow                            //  把异常对象长线抛出给main方法的调用者
        23: aload_0                          // 将类this入栈，以便下面调用类的方法
        24: invokespecial #6                  // Method function:()V
        27: return
```

编译器必须确保无论方法通过何种方式完成，方法中调用过的每条`monitorenter`指令都必须执行其对应的`monitorexit`指令，无论这个方法是正常结束还是异常结束。

### 3 实例

#### 例一：

相信面试过的人基本地看过这个面试题，然后还扯过值传递还是引用传递这个问题，下面从字节码的角度来分析这个问题。

```assembly
public class Main {
    String str="newStr";
    String[] array={"newArray1","newArray2"};
  public static void main(String[] args) {
      Main main=new Main();
      main.change(main.str, main.array);
      System.out.println(main.str);
      System.out.println(Arrays.toString(main.array));
  }
  private void change(String str, String[] array) {
      str="newStrEdit";
      array[0]="newArray1Edit";
  }
}
输出结果：
newStr
[newArray1Edit, newArray2]
字节码：

private void change(java.lang.String, java.lang.String[]);
  descriptor: (Ljava/lang/String;[Ljava/lang/String;)V
  flags: ACC_PRIVATE
  Code:
    stack=3, locals=3, args_size=3
       0: ldc           #14                 // String newStrEdit
       2: astore_1
       3: aload_2
       4: iconst_0
       5: ldc           #15                 // String newArray1Edit
       7: aastore
       8: return
}
```

这里main方法的字节码内容可以忽略，主要看这个change方法，下面用图来表示。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1)

这是刚进入这个方法的情况，这时候还没有执行方法的内容，局部变量表存了三个值，第一个是this指代这个类，在普通方法内之所以可以拿到外部的全局变量就是因为方法内部的局部变量表的第一个就是类的this，当获取外部变量时，先将这个this入栈`aload_0`，然后就可以获取到这个类所有的成员变量（也就是外部全局变量）了。 因为这个方法传进来了两个值，这里局部变量表存储的是这两个对象的引用，也就是在堆上的内存地址。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1-16274398099532)

上面执行了`str = "newStrEdit";`这条语句，先ldc指令创建了newStrEdit（0xaaa）字符串入栈，然后`astore_1`指令将栈顶的值保存再局部变量1中，覆盖了原来的地址，所以这里对局部变量表的修改完全没有影响外面的值。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1-16274399820714)

上面执行`array[0] = "newArrar1Edit";`这条语句，将array的地址入栈，再将要修改的数组下标0入栈，最后创建newArray1Edit字符串入栈。最后调用`aastore`指令将栈顶的引用型数值（newArray1Edit）、数组下标（0）、数组引用（0xfff）依次出栈，最后将数值存入对应的数组元素中，这里可以看到对这个数组的操作一直都是这个0xfff地址，这个地址和外面的array指向的是同一个数组对象，所以这里修改了，外界的那个array也就同样修改了内容。

#### 例二：

看过前面那个例子应该对局部变量表是什么有所了解，下面这个例子就不绘制上面那个图了。这个例子也是一个常见的面试题，判断`try-catch-finally-return`的执行顺序。finally是一个最终都会执行的代码块，finally里面的return会覆盖try和catch里面的return,同时在finally里面修改局部变量不会影响try和catch里面的局部变量值，除非trycatch里面返回的值是一个引用类型。

```assembly
 public static void main(String[] args) {
        Main a=new Main();
        System.out.println("args = [" + a.testFinally() + "]");;
    }

    public   int testFinally(){
        int i=0;
        try{
            i=2;
            return i;
        }catch(Exception e){
            i=4;
            return i;
        }finally{
            i=6;
        }
字节码：
public int testFinally();
    descriptor: ()I
    flags: ACC_PUBLIC
    Code:
      stack=1, locals=5, args_size=1
         0: iconst_0				// 常量0入栈
         1: istore_1				// 赋值给内存变量1（i） i=0
         2: iconst_2				// 常量2入栈
         3: istore_1				// 赋值给内存变量1（i） i=2
         4: iload_1				    // 内存变量1（i）入栈
         5: istore_2			    // 将数据存储在内存变量2 这里原因下面说明
         6: bipush        6    		// 常量6入栈
         8: istore_1				// 保存再内存变量1
         9: iload_2					// 加载内存变量2
        10: ireturn					// 返回上一句加载的内存变量2(i) i=2
        11: astore_2				// 看最下面的异常表，如果2-6发生异常，就从11开始，下面就是发生异常后进入catch的内容
        12: iconst_4				// 常量4入栈
        13: istore_1				// 保存在局部变量1
        14: iload_1					// 加载局部变量1
        15: istore_3				// 将局部变量1内容保存到局部变量3，原因和上面5一样
        16: bipush        6			// 常量6入栈 （进入了catch最后也会执行finally,所以这里会重新再执行一遍finally）
        18: istore_1				// 保存在局部变量1
        19: iload_3					// 加载局部变量3并返回
        20: ireturn					//上面类似的语句，不过是catch-finally的路径
        21: astore        4			// finally 生成的冗余代码，这里发生的异常会抛出去
        23: bipush        6
        25: istore_1
        26: aload         4
        28: athrow
      Exception table:
         from    to  target type
             2     6    11   Class java/lang/Exception  //如果2-6发生指定的Exception异常（try），就从11开始
             2     6    21   any 						//如果2-6发生任何其他异常（finally)，就从21开始
            11    16    21   any						//如果11-16发生任何其他异常（catch)，就从21开始
            21    23    21   any						//其实这里有点不太能理解为什么会循环，如果有知道的大佬可以解答一下
复制代码
```

在Java1.4之后 Javac编译器 已经不再为 finally 语句生成 jsr 和 ret 指令了， 当异常处理存在finally语句块时，编译器会自动在每一段可能的分支路径之后都将finally语句块的内容冗余生成一遍来实现finally语义。（21~28）。但我们Java代码中，finally语句块是在最后的，编译器在生成字节码时候，其实将finally语句块的执行指令移到了ireturn指令之前，指令重排序了。所以，从字节码层面，我们解释了，为什么finally语句总会执行！

如果`try`中有`return`,会在`return`之前执行finally中的代码，但是会保存一个副本变量（第五和第十五行）。`finally`修改原来的变量，但`try`中`return`返回的是副本变量，所以如果是赋值操作，即使执行了`finally`中的代码，变量也不一定会改变，需要看变量是基本类型还是引用类型。 但是如果在finally里面添加一个return，那么第9行和第19行加载的就是`finally`块里修改的值（iload_1）,再在最后添加一个`iload_1`和`ireturn`,感兴趣的可以自己去看一下字节码。

#### 例三：

还是上面那个类似的例子，这里做一下改变

```assembly
 public static void main(String[] args) {
        Main a = new Main();
        System.out.println("args = [" + a.testFinally1() + "]");
        System.out.println("args = [" + a.testFinally2() + "]");
    }

    public StringBuilder testFinally1() {
        StringBuilder a = new StringBuilder("start");
        try {
            a.append("try");
            return a;
        } catch (Exception e) {
            a.append("catch");
            return a;
        } finally {
            a.append("finally");
        }
    }

    public String testFinally2() {
        StringBuilder a = new StringBuilder("start");
        try {
            a.append("try");
            return a.toString();
        } catch (Exception e) {
            a.append("catch");
            return a.toString();
        } finally {
            a.append("finally");
        }
    }

输出结果:
args = [starttryfinally]
args = [starttry]
复制代码
```

这里就不列举全局字节码了，两个方法有点多，大家可以自己尝试去看一下。这里做一下说明为什么第一个返回的结果没有`finally`。 首先这个方法的局部变量表1里面存储了一个StringBuilder地址，执行到try~finally这一部分没什么区别，都是复制了一份变量1的地址到变量3，注意，这两个地址是一样的。 那为什么第二个返回方法少了`finally`呢，那是因为`s.toString()`方法这个看起来是在return后面，但其实这个方法属于这个try代码块，分为两步，先调用`toString()`生成了一个新的字符串`starttry`然后返回，所以这里的字节码逻辑就如下：

```
      17: aload_1
      18: invokevirtual #12                 // Method java/lang/StringBuilder.toString:()Ljava/lang/String;
      21: astore_2
      22: aload_1
      23: ldc           #18                 // String finally
      25: invokevirtual #8                  // Method java/lang/StringBuilder.append:(Ljava/lang/String;)Ljava/lang/StringBuilder;
      28: pop
      29: aload_2
      30: areturn
复制代码
```

可以很清楚的看到 调用append方法拼接“start”和“try”后，先调用了`toString()`方法然后将值存入局部变量2。这时候finally没有和上面那样复制一份变量，而是继续使用局部变量1的引用来继续append，最后的结果也存入了局部变量1中，最后返回的是局部变量2中的值`starttry`，但是要注意此时局部变量1中指向的StringBuilder的值却是`starttryfinally`，所以这也就是方法1中返回的值。

### 4.如何快捷查看字节码

如果是ide的话，应该都可以，通过``Setting->Tools->External Tools进入 然后创建一个自定义的tools。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1-16274400287786)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/1-16274400346418)

### Resource

- https://juejin.cn/post/6844903983002157070#heading-16

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javajvm_%E5%AD%97%E8%8A%82%E7%A0%81%E6%8C%87%E4%BB%A4/  

