# TemplateMode


> 模板方法模式适用于以下应用场景：
>
> -  —次性实现一个算法的不变的部分 ， 并将可变的行为留给子类来实现。利用模板方法将相同处理逻编的代码放到抽象父类中 ， 可以提高代码的复用性。
> -  各子类中公共的行为被提取出来并集中到一个公共的父类中 ，从而避免代码重复。

### 1. 基本案例

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704164447171.png)

```java
public abstract class AbastractCourse {
     
     public final void createCourse(){     //钩子方法
         //1、发布预习资料
         postPreResoucse();
         
         //2、制作课件
         createPPT();
         
         //3、直播授课
         liveVideo();
         
         //4、上传课后资料
         postResource();
         
         //5、布置作业
         postHomework();
         
         if(needCheckHomework()){
             checkHomework();
         }
     }
 
     protected abstract void checkHomework();
 
     //钩子方法
     protected boolean needCheckHomework(){return  false;}
 
     protected void postHomework(){
         System.out.println("布置作业");
     }
 
     protected void postResource(){
         System.out.println("上传课后资料");
     }
 
     protected void liveVideo(){
         System.out.println("直播授课");
     }
 
     protected void createPPT(){
         System.out.println("制作课件");
     }
 
     protected void postPreResoucse(){
         System.out.println("发布预习资料");
     }
 
 }
```

- 创建 JavaCourse 类：

```java
 public class JavaCourse extends AbastractCourse {
     private boolean needCheckHomework = false;
 
     public void setNeedCheckHomework(boolean needCheckHomework) {
         this.needCheckHomework = needCheckHomework;
     }
 
     @Override
     protected boolean needCheckHomework() {
         return this.needCheckHomework;
     }
 
     protected void checkHomework() {
         System.out.println("检查Java作业");
     }
 }
```

- 创建 PythonCourse 类：

```java
 public class PythonCourse extends AbastractCourse {
     protected void checkHomework() {
         System.out.println("检查Python作业");
     }
 }
```

### 2. Android 

- AsyncTask： onPreExecute, doInBackground, onPostExecute 执行顺序
- Activity： onCreate，onStart，onResume 执行顺序



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/templatemode/  

