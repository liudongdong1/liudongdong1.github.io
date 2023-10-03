# JavaJVM_OutofMemory


> - OutOfMemory异常原因：`根据报错信息确定出是那个区域发生OutOfMemory异常`，然后分析是`内存泄露还是内存溢出`。
>   - `内存泄漏`：与GC Roots相关联并导致GC无法自动回收。
>   - `内存溢出`：众多对象确实还必须活着，导致大量内存被占用而无法GC，当超出限制内存最大值时就抛出OutOfMemory异常。

#### .1. 堆溢出

```java
import java.util.ArrayList;
import java.util.List;
public class HeapOut{
    static class OOMObjet{}
    public static void main(String[] args){
        List<OOMObjet> list=new ArrayList<OOMObjet>();
        while(true){
            list.add(new OOMObjet());
        }
    }
}
```

> Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
> 	at HeapOut.main(HeapOut.java:9)
>
> - CPU 和内存 飙满了。

#### .2. 虚拟栈&本地方法溢出

```java
import java.util.ArrayList;
import java.util.List;

public class HeapOut{
    private int stackLength=1;
    public void stackLeak(){
        stackLength++;
        stackLeak();
    }
    public static void main(String[] args) throws Throwable{
        HeapOut oom=new HeapOut();
        try{
            oom.stackLeak();
        }catch (Throwable e){
            System.out.println("stack Length:"+oom.stackLength);
            throw e;
        }
    }
}
```

> - 操作系统给内个进程内存是有限的。
>
> Exception in thread "main" java.lang.StackOverflowError
> 	at HeapOut.stackLeak(HeapOut.java:8)
> 	at HeapOut.stackLeak(HeapOut.java:8)
> 	at HeapOut.stackLeak(HeapOut.java:8)
> 	at HeapOut.stackLeak(HeapOut.java:8)

#### .3. 创建线程导致内存溢出

```java
public class JavaVMStackOOM{
    private vodi dontStop(){
        while(true){ 
        }
    }
    public void stackLeakByThread(){
        while(true){
            Thread thread=new Thread(new Runnable(){
                @Override 
                public void run(){
                    dontStop();
                }
            });
            thread.start()
        }
    }
    public static void main(String[] args){
        JavaVMStackOOM oom=new JavaVMStackOOM();
        oom.stackLeakByThread();
    }
}
```

> 会导致电脑死机，Unable to create new native thread.

#### .4. 方法区&运行常量池溢出

```java
public class RuntimeConstantPoolOOM{
    public static void main(String[] args){
        List<String>list=new ArrayList<String>();
        int i=0;
        while(true){
            list.add(String.valueOf(i++).intern());
        }
    }
}
```

> java.lang.OutOfMemoryError: PerGen space;

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javajvm_outofmemory/  

