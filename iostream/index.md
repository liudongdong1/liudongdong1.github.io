# IOStream


> 字节流可以处理任何类型的数据，如图片，视频等;
>
> 字符流只能处理字符类型的数据。
>
> - **读写单位不同：字节流以字节（8bit）为单位，字符流以字符为单位，根据码表映射字符，一次可能读多个字节。**
> - **处理对象不同：字节流能处理所有类型的数据（如图片、avi等），而字符流只能处理字符类型的数据。**
>
> **结论：只要是处理纯文本数据，就优先考虑使用字符流。 除此之外都使用字节流。**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510205614.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510210657.png)

### 1. 字节流&字符流

字符流的由来： `Java中字符是采用Unicode标准`，一个`字符是16位`，即`一个字符使用两个字节来表示`。为此，JAVA中引入了处理字符的流。因为数据编码的不同，从而有了对字符进行高效操作的流对象。本质其实就是`基于字节流读取时，去查了指定的码表`。

两者的区别：(`1字符 = 2字节 、 1字节(byte) = 8位(bit) `、` 一个汉字占两个字节长度`)

1. 读写单位不同：`字节流以字节为单位，字符流以字符为单位`，根据码表映射字符，一次可能读多个字节。
2. 处理对象不同：`字节流能处理所有类型的数据（如图片、avi等）`，而字符流只能处理字符类型的数据。
3. 缓存不同：字节流在操作的时候本身是不会用到缓冲区的，是文件本身的直接操作的；而字符流在操作的时候下后是会用到缓冲区的，是通过缓冲区来操作文件。

总结：优先选用字节流。因为硬盘上的所有文件都是以字节的形式进行传输或者保存的，包括图片等内容。字符只是在内存中才会形成的，所以在开发中，字节流使用广泛。除非处理纯文本数据(如TXT文件)，才优先考虑使用字符流，除此之外都使用字节流。

### 2. 节点流&处理流

- 节点流：可以从或向一个特定的地方（节点）读写数据。如FileInputStream，FileReader。节点流是直接作用在文件上的流，可以理解为一个管道，文件在管道中传输。

- 处理流：`是对一个已存在的流的连接和封装`，通过所封装的流的功能调用实现数据读写。如BufferedReader。处理流的构造方法总是要带一个其他的流对象做参数。一个流对象经过其他流的多次包装，称为流的链接。处理流是作用在已有的节点流基础上，是包在节点流的外面的管道(可多层)，其目的是让管道内的流可以更快的传输。

###  3. 常用接口

- **File（文件特征与管理）**：File类是对文件系统中文件以及文件夹进行封装的对象，可以通过对象的思想来`操作文件和文件夹`。 File类保存文件或目录的各种元数据信息，包括文件名、文件长度、最后修改时间、是否可读、获取当前文件的路径名，判断指定文件是否存在、获得当前目录中的文件列表，创建、删除文件和目录等方法。 

- **InputStream（二进制格式操作）**：抽象类，基于字节的输入操作，是所有输入流的父类。定义了所有输入流都具有的共同特征。
  - `FileInputSream`：文件输入流。它通常用于对文件进行读取操作。
  - `FilterInputStream `：过滤流。作用是`为基础流提供一些额外的功能`。装饰者模式中处于装饰者，具体的装饰者都要继承它，所以在该类的子类下都是用来装饰别的流的，也就是处理类。
  - `BufferedInputStream`：缓冲流。对处理流进行装饰，增强，内部会有一个缓存区，用来存放字节，每次`都是将缓存区存满然后发送，而不是一个字节或两个字节这样发送`。效率更高。
  - `DataInputStream`：数据输入流。它是用来装饰其它输入流，它“`允许应用程序以与机器无关方式从底层输入流中读取基本 Java 数据类型`”。
  - PushbakInputStream：回退输入流。java中读取数据的方式是顺序读取,如果某个数据不需要读取，需要程序处理。PushBackInputStream就可以将某些不需要的数据回退到缓冲中。
  - `ObjectInputStream`：对象输入流。用来提供对`“基本数据或对象”的持久存储`。通俗点讲，也就是能直接传输对象（反序列化中使用）。
  - `PipedInputStream`：管道字节输入流。它和PipedOutputStream一起使用，能实现`多线程间的管道通信`。
  - SequenceInputStream:合并输入流。依次将多个源合并成一个源。
  - `ByteArrayInputStream`：字节数组输入流，该类的功能就是`从字节数组(byte[])中进行以字节为单位的读取`，也就是`将资源文件都以字节的形式存入到该类中的字节数组`中去，我们拿也是从这个字节数组中拿。

-  **OutputStream（二进制格式操作）**：抽象类。基于字节的输出操作。是所有输出流的父类。定义了所有输出流都具有的共同特征。

- **Reader（文件格式操作）**：抽象类，基于字符的输入操作。

- **Writer（文件格式操作）**：抽象类，基于字符的输出操作。

- **RandomAccessFile（随机文件操作）**：一个独立的类，直接继承至Object.它的功能丰富，可以从文件的任意位置进行存取（输入输出）操作。

-  **Serializable（序列化操作）**：是一个空接口，为对象提供标准的序列化与反序列化操作。

### 4. 转换流

> - GBK 编码中，中文字符占 2 个字节，英文字符占 1 个字节；
> - UTF-8 编码中，中文字符占 3 个字节，英文字符占 1 个字节；
> - UTF-16be 编码中，中文字符和英文字符都占 2 个字节。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510211525.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210803120841012.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510211618.png)

> **字符流 = 字节流 + 编码表**InputStreamReader将**字节**输入流转为**字符**输入流，继承自Reader。OutputStreamWriter是将**字符**输出流转为**字节**输出流，继承自Writer。
>
> 转换流的特点：其是字符流和字节流之间的桥梁。
>
> ​    可对读取到的字节数据经过指定编码转换成字符
>
> ​    可对读取到的字符数据经过指定编码转换成字节

```java
public class InputStreamReaderTest {
    public static void main(String[] args) {
        //定义转换流
        InputStreamReader isr = null;
        InputStreamReader isr1 = null;

        try {
            //创建流对象,默认编码方式
            isr = new InputStreamReader(new FileInputStream("D:\\IO\\utf8.txt"));
            //创建流对象,指定GBK编码
            isr1 = new InputStreamReader(new FileInputStream("D:\\IO\\utf8.txt"),"GBK");

            //默认方式打印
            int len;
            char[] buffer = new char[1024];
            while ((len=isr.read(buffer))!=-1){
                System.out.println(new String(buffer,0,len));
            }
            //GBK编码方式打印
            int len1;
            char[] buffer1 = new char[1024];
            while ((len1=isr1.read(buffer1))!=-1){
                System.out.println(new String(buffer1,0,len1));
            }

            System.out.println("成功...");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            //释放资源，先使用的后关闭
            if (isr1!=null){
                try {
                    isr1.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (isr!=null){
                try {
                    isr.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
```

- 文件编码转化

```java
public class ConversionStreamTest {

    public static void main(String[] args) {
        //定义转换流
        InputStreamReader isr = null;
        OutputStreamWriter osw = null;

        try {
            //创建流对象,指定GBK编码
            isr = new InputStreamReader(new FileInputStream("D:\\IO\\utf8.txt"),"UTF-8");
            osw = new OutputStreamWriter(new FileOutputStream("gbk.txt"),"GBK");

            int len;
            char[] buffer = new char[1024];
            while ((len=isr.read(buffer))!=-1){
                osw.write(buffer,0,len);
            }
            System.out.println("成功...");
        }  catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            //释放资源
            if (osw!=null){
                try {
                    osw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (isr!=null){
                try {
                    isr.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
```

### 5. 序列化操作

> -  **序列化**：指把堆内存中的 Java 对象数据，通过某种方式把[对象存储](https://cloud.tencent.com/product/cos?from=10680)到磁盘文件中或者传递给其他网络节点（在网络上传输）。这个过程称为序列化。通俗来说就是将数据结构或对象转换成二进制串的过程
> - **反序列化**：把磁盘文件中的对象数据或者把网络节点上的对象数据，恢复成Java对象模型的过程。也就是将在序列化过程中所生成的二进制串转换成数据结构或者对象的过程

- 在分布式系统中，此时需要把对象在网络上传输，就得把对象数据转换为二进制形式，需要共享的数据的 JavaBean 对象，都得做序列化。
- 服务器就会把这些内存中的对象持久化在本地磁盘文件中（Java对象转换为二进制文件）；如果服务器发现某些对象需要活动时，先去内存中寻找，找不到再去磁盘文件中反序列化我们的对象数据，恢复成 Java 对象。这样能节省服务器内存。

#### .1. 流程

1. 必须实现序列化接口：Java.lang.Serializable 接口（这是一个标志接口，没有任何抽象方法），Java 中大多数类都实现了该接口，比如：String，Integer
2. 底层会判断，如果当前对象是 Serializable 的实例，才允许做序列化，Java对象 instanceof Serializable 来判断。
3. 在 Java 中使用对象流来完成序列化和反序列化
   - **ObjectOutputStream**:通过 writeObject()方法做序列化操作
   -  **ObjectInputStream**:通过 readObject() 方法做反序列化操作

- javaBean

```java
package mypackage2;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
//对象流 ObjectInput
public class Test {
	public static void main(String[] args) throws Exception {
		File file = new File("object.txt");
		writeObject(file);
		readObject(file);
	}
	private static void writeObject(File file) throws Exception {
		ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(file));
		objectOutputStream.writeObject(new Person("小明", 12));
		objectOutputStream.close();
	}
	private static void readObject(File file) throws Exception {
		ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(file));
		Person person = (Person) objectInputStream.readObject();
		System.out.println(person);
		objectInputStream.close();
	}
}
class Person implements Serializable {
	private static final long serialVersionUID = 1L;
	private String name;
	private int age;

	public Person(String name, int age) {
		this.name = name;
		this.age = age;
	}
	@Override
	public String toString() {
		return "Person [name=" + name + ", age=" + age + "]";
	}
}
```

> 在 JavaBean 对象中增加一个 serialVersionUID 字段，用来固定这个版本，无论我们怎么修改，版本都是一致的，就能进行反序列化了



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/iostream/  

