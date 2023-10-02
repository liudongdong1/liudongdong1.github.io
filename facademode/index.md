# FacadeMode


> 定义了一个高层、统一的接口，外部与通过这个统一的接口对子系统中的一群接口进行访问。通过创建一个统一的类，用来包装子系统中一个或多个复杂的类，客户端可以通过调用外观类的方法来调用内部子系统中所有方法
>
> - 实现客户类与子系统类的松耦合
> - 降低原有系统的复杂度
> - 提高了客户端使用的便捷性，使得客户端无须关心子系统的工作细节，通过外观角色即可调用相关功能。
> - **适配器模式是将一个对象包装起来以改变其接口，而外观是`将一群对象 ”包装“起来以简化其接口`。它们的意图是不一样的，适配器是将接口转换为不同接口，而外观模式是提供一个统一的接口来简化接口**。

> - 在设计初期阶段，应该要`有意识的将不同的两个层分离`，比如经典的三层架构，就需要考虑在三层的层与层之间建立外观Facade(外观)，这样可以为复杂的子系统提供应该简单的接口，使得耦合大大降低
> - 其次，在开发阶段，子系统往往因为不断的重构演化而变得越来越复杂，大多数的模式使用时也都会产生很多很小的类，这本是好事，但也给外部调用它们的用户程序带来使用上的困难，增加外观Facade可以提供一个简单接口，减少它们之间的依赖
> - 在维护一个遗留的大型系统时，可能这个系统已经非常难以维护和扩展了，但因为它包含非常重要的功能，新的需求开发必须依赖它，此时用外观模式Facade也是非常合适的，你可以为新系统开发一个外观Facade类，给设计粗糙或高度复杂的遗留代码提供比较清晰简单的接口，让新系统与Facade对象进行交互，Facade完成与遗留代码交互所有复杂的工作

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705082237371.png)

> - **外观（Facade）角色**：为多个子系统对外提供一个共同的接口。
> - **子系统（Sub System）角色**：实现系统的部分功能，客户可以通过外观角色访问它。
> - **客户（Client）角色**：通过一个外观角色访问各个子系统的功能。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705082636479.png)

### 1. java Demo

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705082733354.png)



```java
//播放器
public class Player {
    //由于都是设备，所以我们返回全局唯一对象
    //使用饿汉式单例模式
    private static Player player = new Player();
    private Player(){}
    public static Player getInstance(){
        return player;
    }
    public void on(){
        System.out.println("打开播放器");
    }
    public void off(){
        System.out.println("关闭播放器");
    }
    public void play(){
        System.out.println("使用播放器播放电影");
    }
    public void pause(){
        System.out.println("暂停播放器");
    }
    public void select(){
        System.out.println("选择喜欢的电影");
    }
}
//投影仪
public class Projector {
    private static Projector projector = new Projector();
    private Projector(){}
    public static Projector getInstance(){
        return projector;
    }
    public void on(){
        System.out.println("打开投影仪");
    }
    public void off(){
        System.out.println("关闭投影仪");
    }
    public void focus(){
        System.out.println("调节投影仪焦距");
    }
    public void zoom(){
        System.out.println("投影仪变焦");
    }
}
//音响
public class Stereo {
    private static Stereo stereo = new Stereo();
    private Stereo(){}
    public static Stereo getInstance(){
        return stereo;
    }
    public void on(){
        System.out.println("打开音响");
    }
    public void off(){
        System.out.println("关闭音响");
    }
    public void setVolume(){
        System.out.println("调节音响音量");
    }
}
//爆米花机
public class Popcorn {
    private static Popcorn popcorn = new Popcorn();
    private Popcorn(){}
    public static Popcorn getInstance(){
        return popcorn;
    }
    public void on(){
        System.out.println("打开爆米花机");
    }
    public void off(){
        System.out.println("关闭爆米花机");
    }
    public void pop(){
        System.out.println("制作爆米花");
    }
}
//外观类
public class HomeTheaterFacade {
    //聚合各个子系统
    private Player player;
    private Popcorn popcorn;
    private Projector projector;
    private Stereo stereo;
    //构造器里初始化各个子系统
    public HomeTheaterFacade() {
        this.player = Player.getInstance();
        this.popcorn = Popcorn.getInstance();
        this.projector = Projector.getInstance();
        this.stereo = Stereo.getInstance();
    }
    //把操作分成四步 准备
    public void ready(){
        popcorn.on();
        popcorn.pop();
        player.on();
        projector.on();
        stereo.on();
    }
    public void play(){
        player.select();
        projector.focus();
        player.play();
        stereo.setVolume();
    }
    public void pause(){
        player.pause();
    }
    public void end(){
        player.off();
        projector.off();
        stereo.off();
        popcorn.off();
    }
}
//客户端
public class Client {
    public static void main(String[] args) {
        HomeTheaterFacade homeTheaterFacade = new HomeTheaterFacade();
        homeTheaterFacade.ready();
        homeTheaterFacade.play();
        homeTheaterFacade.pause();
        homeTheaterFacade.end();
    }
}
```

### 2. Mybatis

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705083055289.png)

### Resource

- https://blog.csdn.net/carson_ho/article/details/54910625

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/facademode/  

