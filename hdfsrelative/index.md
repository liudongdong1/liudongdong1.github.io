# HDFSRelative


HDFS是hadoop实现的一个分布式文件系统。(Hadoop Distributed File System)来源于Google的GFS论文。它的设计目标有：非常巨大的分布式文件系统。运行在普通廉价的硬件上，及一般的 PC机(相比于小型机，单片机而言的)。易扩展，为用户提供性能不错的文件存储服务。

#### 1. 架构介绍

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210226001055769.png)

- HDFS采用了1个 Msater(NameNode) 和N个slaves(DataNode)的架构：

  - 一个HDFS集群包含一个NameNode，主要职责是管理文件系统的元数据信息，控制客户端对文件的访问。

  - 一个HDFS集群包含多个DataNode，通常一个节点就是一个DataNode，负责相应节点上文件的存储。
  - NameNode: 
    - 负责客户端请求的响应
    - 维护整个文件系统的目录树(例如记录文件的增删改查操作)和负责元数据(文件名称、副本系数，文件和block的映射，DataNode和block的映射等)的管理
  - DataNode:
    - 存储文件对应的数据块，存储数据是核心作用
    - 定期向NameNode发送心跳信息，汇报本身及其所有block信息和健康状况
    - 执行来自NameNode的指示，如block的创建，删除，复制，文件读写请求的支持等

![work flow](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210226001505297.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210226001631645.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210226001750571.png)

#### 2. 基本命令

```shell
1) 打印文件列表
标准写法：
hadoop fs -ls hdfs:/#hdfs: 明确说明是HDFS系统路径
简写：
hadoop fs -ls /#默认是HDFS系统下的根目录
打印指定子目录：
hadoop fs -ls /package/test/#HDFS系统下某个目录

2) 上传文件、目录（put、copyFromLocal）
put用法：
上传新文件：
hdfs fs -put file:/root/test.txt hdfs:/  #上传本地test.txt文件到HDFS根目录，HDFS根目录须无同名文件，否则“File exists”
hdfs fs -put test.txt /test2.txt  #上传并重命名文件。
hdfs fs -put test1.txt test2.txt hdfs:/  #一次上传多个文件到HDFS路径。
上传文件夹：
hdfs fs -put mypkg /newpkg  #上传并重命名了文件夹。
覆盖上传：
hdfs fs -put -f /root/test.txt /  #如果HDFS目录中有同名文件会被覆盖
copyFromLocal用法：
上传文件并重命名：
hadoop fs -copyFromLocal file:/test.txt hdfs:/test2.txt

3) 下载文件、目录（get、copyToLocal）
get用法：
拷贝文件到本地目录：
hadoop fs -get hdfs:/test.txt file:/root/
拷贝文件并重命名，可以简写：
hadoop fs -get /test.txt /root/test.txt
copyToLocal用法
拷贝文件到本地目录：
hadoop fs -copyToLocal hdfs:/test.txt file:/root/
拷贝文件并重命名，可以简写：
hadoop fs -copyToLocal /test.txt /root/test.txt

4) 拷贝文件、目录（cp）
从本地到HDFS，同put
hadoop fs -cp file:/test.txt hdfs:/test2.txt
从HDFS到HDFS
hadoop fs -cp hdfs:/test.txt hdfs:/test2.txt
hadoop fs -cp /test.txt /test2.txt

5) 移动文件（mv）
hadoop fs -mv hdfs:/test.txt hdfs:/dir/test.txt
hadoop fs -mv /test.txt /dir/test.txt

6) 删除文件、目录（rm）
删除指定文件
hadoop fs -rm /a.txt
删除全部txt文件
hadoop fs -rm /*.txt
递归删除全部文件和目录
hadoop fs -rm -R /dir/

7) 读取文件（cat、tail）
hadoop fs -cat /test.txt  #以字节码的形式读取
hadoop fs -tail /test.txt

8) 创建空文件（touchz）
hadoop fs - touchz /newfile.txt

9) 创建文件夹（mkdir）
hadoop fs -mkdir /newdir /newdir2#可以同时创建多个
hadoop fs -mkdir -p /newpkg/newpkg2/newpkg3 #同时创建父级目录

10) 获取逻辑空间文件、目录大小（du）
hadoop fs - du /  #显示HDFS根目录中各文件和文件夹大小
hadoop fs -du -h /  #以最大单位显示HDFS根目录中各文件和文件夹大小
hadoop fs -du -s /  #仅显示HDFS根目录大小。即各文件和文件夹大小之和
```

#### 3. HDFS java函数封装

- java api

```java

package HDFSDemo;
 
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
 
import java.util.Iterator;
import java.util.Map;
//hdfs操作api
public class HDFSBasic {
    /**
     * 客户端访问hdfs服务，需要nameno的uri
     * 这些信息会封装在configuration中
     * 如果要执行Configuration conf = new Configuration();就会初始化
     * conf中的数据究竟有哪些呢？
     * 1.首先将客户端依赖的hadoop-common-2.7.1.jar下的core-site.xml文件读取
     * <property>
     *   <name>fs.defaultFS</name>
     *   <value>file:///</value>
     * </property>
     * 2.如果classpath下存在core-site.xml,如果有的话就读取这个文件
     *  <property>
     *         <name>fs.defaultFS</name>
     *         <value>hdfs://min1:9000</value>
     *     </property>
     *     注意：本地hosts文件需要建立ip和min1的映射关系
     * 3.如果代码里存在conf.set("fs.defaultFs","hdfs://192.168.196.101:9000");
     * conf中封装的内容就是该语句设置内容
     *
     * conf中分装值的优先级是3>2>1
     */
 
    //查看conf参数
    public void testConfiguration(){
        Configuration conf = new Configuration();
        Iterator<Map.Entry<String, String>> it = conf.iterator();
        while (it.hasNext()){
            Map.Entry<String, String> map = it.next();
            System.out.println(map.getKey()+"===="+map.getValue());
        }
    }
 
 
 
    //将本地文件上传到hdfs上
    public  void  uploadFile2HDFS(String s,String d) throws  Exception{
        Configuration conf = new Configuration();
        //FileSystem fs = FileSystem.get(new URI("hdfs://192.168.196.101:9000"), conf, "hadoop");
        FileSystem fs = FileSystem.get(conf);
        Path src =new Path(s);
        Path dst =new Path(d);
        fs.copyFromLocalFile(src,dst);
        fs.close();
    }
    //将hdfs文件下载到本地
    public void  downLoad(String s,String d) throws  Exception {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path src =new Path(s);
        Path dst =new Path(d);
        fs.copyToLocalFile(src,dst);
        fs.close();
    }
    //在hdfs上创建文件夹
    public  void  hdfsMkdir(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path dir= new Path(path);
        fs.mkdirs(dir);
        fs.close();
    }
 
    //在hdfs上删除文件
    public  void  deleteDir(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path dir= new Path(path);
        //true代表是否递归删除文件
        fs.delete(dir,true);
        fs.close();
    }
    //浏览hdfs上一个指定文件下的所有文件
    public  void  ListFile(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path dir= new Path(path);
        //如果文件下存在文件 第二个参数为false 时不会显示这个文件
        //如果文件下存在文件 第二个参数为true 时会以递归显示这个文件夹下的文件
        RemoteIterator<LocatedFileStatus> lfss = fs.listFiles(dir, true);
        while (lfss.hasNext()){
            LocatedFileStatus life = lfss.next();
            //打印所有文件路径
            System.out.println(life.getPath().toString());
            //打印文件名
            System.out.println(life.getPath().getName());
            //打印权限信息r--wrx--
            System.out.println("文件的权限信息： "+life.getPermission());
            //所属用户
            System.out.println("文件所属用户： "+life.getOwner());
            //所属组
            System.out.println("文件的所属组： "+life.getGroup());
            //文件的访问时间
            System.out.println("文件的访问时间： "+life.getAccessTime());
            //文件的修改时间
            System.out.println("文件的修改时间： "+life.getModificationTime());
            //文件的大小
            System.out.println("文件大小： "+life.getLen());
            //文件的副本数
            System.out.println("文件副本数量： "+life.getReplication());
            //block块的大小
            System.out.println("文件副本数量： "+life.getBlockSize());
            //block块的详细信息
            BlockLocation[] blockLocations = life.getBlockLocations();
            for (BlockLocation b1:blockLocations){
                //块所在主机名称
                System.out.println(b1.getHosts());
                //块对应便宜量
                System.out.println(b1.getOffset());
                //块的大小
                System.out.println(b1.getLength());
                //块的名称
                System.out.println(b1.getNames());
            }
        }
        fs.close();
    }
 
    //列出指定路径下所有的文件和文件夹，上面方法只列出了文件没有列出文件夹
    public  void  listFileAndDir(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path dir= new Path(path);
        FileStatus[] fsArray = fs.listStatus(dir);
        for (FileStatus fss:fsArray){
            System.out.println(fss.getPath().getName());
        }
        fs.close();
    }
    public static void main(String[] args) throws Exception{
        HDFSBasic hb =new HDFSBasic();
        String src="d:/workspace/EasyReport/easyreport-web/target/easyreport-web.jar";
        String dst="/temp";
        //将windos本地文件上传到hdfs
        hb.uploadFile2HDFS(src,dst);
        //查看conf配置信息
        hb.testConfiguration();
        //将hdfs文件下载到本地电脑中
        hb.downLoad(dst,"d:/");
        //删除hdfs文件路径
        hb.deleteDir(dst);
        //在hdfs上创建文件
        String mkdir="/temp/aa/bb/mkdir";
        hb.hdfsMkdir(mkdir);
        //文件信息
        hb.ListFile("/wordcount/output");
        //文件及文件夹信息
        hb.listFileAndDir("/");
 
    }
}
```

- hdfs 流操作java API

```java
package HDFSDemo;
 
 
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.Test;
 
import java.io.FileInputStream;
import java.io.FileOutputStream;
 
//流的基本操作
public class HDFSStream {
    @Test
    //流的方式上传文件
    public  void testUpload(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path file = new Path(path);
        FSDataOutputStream fdos = fs.create(file);
        FileInputStream fin = new FileInputStream("d:/aa.txt");
        IOUtils.copy(fin,fdos);
        System.out.println("流的方式上传文件：执行完成---------------");
    }
    @Test
    //流的方式下载
    public  void testDownload(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path file = new Path(path);
        FSDataInputStream fsdIn  = fs.open(file);
        FileOutputStream fos = new FileOutputStream("d:/bb.txt");
        IOUtils.copy(fsdIn,fos);
 
    }
 
    //流的随机读取方式
    public  void testRandomRead(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path file = new Path(path);
        FSDataInputStream in  = fs.open(file);
        in.seek(50);
       FileOutputStream fos = new FileOutputStream("d:/cc.txt");
       IOUtils.copy(in,fos);
        System.out.println("流的随机读取方式：执行成功------------");
 
    }
    //按照流的方式下载文件
    public  void testDownload2(String path) throws  Exception{
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path file = new Path(path);
        FSDataInputStream fsdIn  = fs.open(file);
        FileOutputStream fos = new FileOutputStream("d:/dd.txt");
 
        byte[] buffer = new byte[20];
        int len = 0;
        int total = 0;
        while ((len = fsdIn.read(buffer)) !=-1){
            fos.write(buffer,0,len);
            total += len;
            System.out.println("read byte total========"+total);
            Thread.sleep(100);
        }
        fsdIn.close();
        fos.close();
    }
    public static void main(String[] args) throws Exception{
        HDFSStream hs = new HDFSStream();
        //流的方式上传文件
        //hs.testUpload("/wordcount/output/aa.txt");
        //流的方式下载文件
        //hs.testDownload("/wordcount/output/part-r-00000");
        //流的随机方式读取
        //hs.testRandomRead("/wordcount/output/part-r-00000");
        // 流的方式下载文件
       //hs.testDownload2("/wordcount/output/part-r-00000");
    }
}
```

#### 4. 学习链接

- https://zhuanlan.zhihu.com/p/70602377



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/hdfsrelative/  

