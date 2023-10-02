# ObserveMode


> 观察者模式又被称为**发布订阅（Publish/Subscribe）模式**，属于对象行为型模式。它定义对象间的一种一对多的依赖关系，`让多个观察者对象同时监听某一个主题对象`，当`主题对象状态发生改变时`，它的所有观察者都会收到通知并自动更新相关内容。主题是通知的发布者，它发出通知时并不需要知道谁是它的观察者，可以有任意数目的观察者订阅并接收通知.

**核心类说明：**

- Subject：抽象主题，即被观察对象，本身维护一个观察者集合。
- Observer：抽象观察者，根据主题状态变化做出相应反应，本身维护一个主题的引用。

**注意事项：**

- Java 中已经有了对观察者模式的支持类。
- 避免循环引用。
- 如果顺序执行，某一观察者错误会导致系统卡壳，一般采用异步方式。

##### .1. 自定义观察者

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210304000518763.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210303232630378.png)

```java
//定义主题
public interface Subject {
    void registerObserver(Observer observer);
    void removeObserver(Observer observer);
    void notifyObserver();
}
public class Blog implements Subject{
 	//存储已定阅的读者
	private ArrayList observerList;
	//这是更新的文章
	private String news;
	
	public Blog(){
		observerList = new ArrayList();
	}
	@Override
	public void registerObserver(Observer observer){
		observerList.add(observer);
	}
	@Override
    public void removeObserver(Observer observer){
    	int index = observerList.indexOf(observer);
    	//不要忘记如果observer没有订阅的话会返回-1
    	if(index>=0){
    		observerList.remove(index);
    	}
    }
    @Override
    public void notifyObserver(Object arg){
    	 for(int i=0;i<observerList.size();i++){
            Observer observer = (Observer)observerList.get(i);
            observer.update(this,arg);
        }
    }
    //文章更新时进行
    public void newsChanged(){
    	notifyObserver(this.news);
    }
	//模拟博客更新文章
	public void setNews(){
		news = "观察者模式";
		notifyObserver();
	}
    
}
//观察者
public interface Observer {
	//因为一个观察者可能订阅了多个主题，所以需要确定是哪个主题
    void update(Subject subject,Object arg);
}
public class Reader implements Observer{
	private String news;
    //存储主题，以方便之后的退订操作
    private Subject subject;
    public Reader(Subject subject){
        this.subject = subject;
        //读者订阅
        subject.registerObserver(this);
    }

    @Override
    public void update(Subject subject, Object arg) {
    //确定主题身份
        if(subject instanceof Blog){
            news = (String)arg;
        }
        //读者阅读博客
        read(this.news);
    }
	//该方法模拟读者阅读
    public void read(String news){
        System.out.println(news);
    }
    //退订
    public removeSubscript(){
    	this.subject.removeObserver(this);
    }
}
public static void main(String[] args) {
        Blog blog = new Blog();
        //读者订阅博客
        Reader reader = new Reader(blog);
        //博客更新
        blog.setNews();
}
```

##### .2. java自带类

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210304000650643.png)

```java
public class NewspaperOffice extends Observable {  //主题
    private String news; // 报社新闻内容

    //注意与自己实现的不同处，添加观察者放在超类中处理
    public NewspaperOffice() {}

    public void setNews(String news) {
        this.news = news;
        newsChanged();
    }
    public String getNews() {
        return news;
    }
    // 内容改变时，通知观察者
    public void newsChanged() {
        setChanged();
        notifyObservers(); 
    }
}
public class Eric implements Observer {
    private Observable observable;  // 主题的引用
    private String news;
    
    public Eric(Observable observable){
        this.observable = observable;
        this.observable.addObserver(this); // 把此对象添加为观察者
    }

    @Override
    public void update(Observable o, Object arg) {
        if (o instanceof NewspaperOffice) {
            NewspaperOffice newspaperOffice = (NewspaperOffice) o;
            this.news = newspaperOffice.getNews();
            showNews(); //更新信息后显示信息
        }
    }

    public void showNews(){
        System.out.println("Eric: "+news);
    }
}
public class Shealtiel implements Observer {
    private Observable observable; //主题的引用
    private String news;

    public Shealtiel(Observable observable){
        this.observable = observable;
        this.observable.addObserver(this); //把此对象添加为观察者
    }

    @Override
    public void update(Observable o, Object arg) {
        if (o instanceof NewspaperOffice) {
            NewspaperOffice newspaperOffice = (NewspaperOffice) o;
            this.news = newspaperOffice.getNews();
            showNews(); //更新信息后显示信息
        }
    }

    public void showNews(){
        System.out.println("Shealtiel: "+news);
    }
}
public class Demo {
    public static void main(String[] args) {
        //实例化报社类（被观察者）
        NewspaperOffice newspaperOffice = new NewspaperOffice();
        //实例化订阅者类（观察者）
        Eric eric = new Eric(newspaperOffice);
        Shealtiel shealtiel = new Shealtiel(newspaperOffice);

        //报社更新信息，自动会通知相应的观察者
        System.out.println("报社更新第一条信息，注意观察者接受的信息：");
        newspaperOffice.setNews("This is the first news!");
        System.out.println("\n报社更新第二条信息，注意观察者接受的信息：");
        newspaperOffice.setNews("This is the second news!");

        //eric对象取消订阅报纸新闻，则此后的新闻信息将不会通知eric
        System.out.println("\nEric取消订阅新闻信息");
        newspaperOffice.deleteObserver(eric);

        System.out.println("报社更新第三条信息，注意此时观察者数量：");
        newspaperOffice.setNews("This is the third news!");
    }
}
```


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/observemode/  

