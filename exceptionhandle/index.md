# ExceptionHandle


> 异常本质上是程序上的错误，包括程序逻辑错误和系统错误。比如使用空的引用、数组下标越界、内存溢出错误等. 错误在我们编写程序的过程中会经常发生，包括`编译期间和运行期间的错误`，在编译期间出现的错误有编译器帮助我们一起修正，然而运行期间的错误便不是编译器力所能及了，并且运行期间的错误往往是难以预料的。假若程序在运行期间出现了错误，如果置之不理，程序便会终止或直接导致系统崩溃

### 1. 异常分类

- **检查性异常(checked exceptions)**: `必须在在方法的 throws 子句中声明的异常`。它们扩展了异常，旨在成为一种“在你面前”的异常类型。JAVA希望你能够处理它们，因为它们以某种方式依赖于程序之外的外部因素。检查的异常表示在正常系统操作期间可能发生的预期问题。当你尝试通过网络或文件系统使用外部系统时，通常会发生这些异常。大多数情况下，对检查性异常的正确响应应该是稍后重试，或者提示用户修改其输入。

- **非检查性异常(unchecked Exceptions)** 是不需要在throws子句中声明的异常。由于`程序错误，JVM并不会强制你处理它们，因为它们大多数是在运行时生成的。它们扩展了 RuntimeException`。最常见的例子是 NullPointerException， 未经检查的异常可能不应该重试，正确的操作通常`应该是什么都不做，并让它从你的方法和执行堆栈中出来。`
- **错误(errors)** 是严重的运行时环境问题，肯定无法恢复。例如 `OutOfMemoryError，LinkageError 和 StackOverflowError，通常会让程序崩溃。`

![1](https://gitee.com/github-25970295/blogpictureV2/raw/master/1.jpg)

### 2. 实践

#### 2.1. 异常声明

```java
//在 methodD 方法中若出现某些不正常的情况可能会触发 XxxException 或 YyyException 异常。
public void methodD() throws XxxException, YyyException {
  // 方法体抛出XxxException和YyyException异常
}
```

#### 2.2. 抛出异常

```java
//注意， 方法体内使用 throw 进行抛出， 方法外使用throws
public void methodD() throws XxxException, YyyException {   // 方法签名
   // 方法体
   ...
   ...
   // 出现XxxException异常
   if ( ... )
      throw new XxxException(...);   // 构造一个XxxException对象并抛给JVM
   ...
   // 出现YyyException异常
   if ( ... )
      throw new YyyException(...);   // 构造一个YyyException对象并抛给JVM
   ...
}
```

#### 2.3. 异常捕获

##### 2.3.1. try-catch

```java
try{
    //可能会抛出异常的代码
}
catch(Type1 id1){
    //处理Type1类型异常的代码
}
catch(Type2 id2){
    //处理Type2类型异常的代码
}
finally{
    //总是会执行的代码
}
```

##### 2.3.2. throws高层

> 如果 methodD 方法抛出 XxxException 或 YyyException，则 JVM 将终止 methodD 方法和methodC 方法并将异常对象沿调用堆栈传递给 methodC 方法的调用者。

```java
public void methodC() throws XxxException, YyyException { // 让更高层级的方法来处理
   ...
   // 调用声明XxxException和YyyException异常的methodD方法
   methodD();   // 无需使用try-catch
   ...
}
```

- **不要忽略捕获异常**

```java
//破坏了检查性异常的目的
catch (NoSuchMethodException e) { return null;
}
```

- **捕获具体的子类而不是捕获 Exception 类**

```java
try { someMethod();
} catch (Exception e) { //错误方式 LOGGER.error("method has failed", e);
}
```

- **尽量不要打印堆栈后再抛出异常**

```java
public static void main(String[] args) throws IOException {
	try (InputStream is = new FileInputStream("沉默王二.txt")) {
	}catch (IOException e) {
		e.printStackTrace();   //错误案例
		throw e;
	} 
}
```

- **千万不要用异常处理机制代替判断**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210510092135.png)

- **不要盲目地过早捕获异常**

> 延迟捕获异常，让程序在第一个异常捕获后就终止执行。



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/exceptionhandle/  

