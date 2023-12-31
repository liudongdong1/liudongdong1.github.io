# CommandMode


> 将一个请求封装成一个对象，从而使你可用不同的请求把客户端参数化，对请求排队或者记录请求日志，以及支持可撤销和恢复操作。
>
> - 系统需要将请求调用者和请求接收者解耦，使得`调用者和接收者不直接交互`。
> - 在需要`支持撤销和恢复撤销的地方，如GUI、文本编辑器等`。
> - 需要用到`日志请求、队列请求的地方`。
> - 在需要事务的系统中。命令模式提供了对事物进行建模的方法，`命令模式有一个别名就是Transaction`。
> -  Progress bars（状态条） 假如系统需要按顺序执行一系列的命令操作，如果每个command对象都提供一个 getEstimatedDuration()方法，那么系统可以简单地评估执行状态并显示出合适的状态条。 
> - Thread pools（线程池） 通常一个典型的线程池实现类可能有一个名为addTask()的public方法，用来添加一项工作任务到任务  队列中。该任务队列中的所有任务可以用command对象来封装，通常这些command对象会实现一个通用的  接口比如java.lang.Runnable。 
> -  Networking 通过网络发送command命令到其他机器上运行。 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210704201001137.png)

- `Command 命令接口`，`声明执行方法`。
- `ConcreteCommand 命令接口`的实现对象，是“虚”的实现，`通常会持有接收者，通过调用接收者的方法，来完成命令要执行的操作。`
- `Receiver 接收者`，真正执行命令的对象。
- `Invoker 调用者`，通常会持有命令，相当于使用命令模式的入口。

### 1.  命令模式示例

- **Command 命令接口**

```java
/**
 * 命令接口，声明执行的操作
 */
public interface Command {
    /**
     * 执行命令对应的操作
     */
    void execute();
}
```

```java
/**
 * 做面的命令
 */
public class NoodleCommand implements Command{

    /**
     * 持有真正实现命令的接收者--后厨对象
     */
    private Kitchen kitchen;

    public NoodleCommand(Kitchen kitchen) {
        this.kitchen = kitchen;
    }

    @Override
    public void execute() {
        kitchen.noodle();
    }
}

/**
 * 做馅饼的命令
 */
public class PieCommand implements Command{

    /**
     * 持有真正实现命令的接收者--后厨对象
     */
    private Kitchen kitchen;

    public PieCommand(Kitchen kitchen) {
        this.kitchen = kitchen;
    }

    @Override
    public void execute() {
        kitchen.pie();
    }
}
```
- **Waiter 调用者**

```java
**
 * 餐厅服务人员对象，持有下单操作命令
 */
@Data
public class Waiter {

    /**
     * 做面条命令对象
     */
    private Command noodleCommand;

    /**
     * 做馅饼命令对象
     */
    private Command pieCommand;

    /**
     * 下达做面条的命令
     */
    public void noodleCommandExecute(){
        noodleCommand.execute();
    }

    /**
     * 下达做馅饼的命令
     */
    public void pieCommandExecute(){
        pieCommand.execute();
    }

}
```

- Kitchen 接收者

```java
/**
 * 后厨类，接收做菜命令，真正的实现做菜功能，在Command模式中充当Receiver
 */
public class Kitchen {

    /**
     * 做面条
     */
    public void noodle(){
        System.out.println("正在做一碗美味的拉面。");
    }

    /**
     * 做馅饼
     */
    public void pie(){
        System.out.println("正在做一个香喷喷的馅饼。");
    }

}
```

- main 调用

```java
public static void main(String[] args) {
    // 创建后厨对象
    Kitchen kitchen = new Kitchen();

	// 创建做面和做馅饼的命令对象
    NoodleCommand noodleCommand = new NoodleCommand(kitchen);
    PieCommand pieCommand = new PieCommand(kitchen);
    
    // 创建一个服务员对象
    Waiter waiter = new Waiter();
    // 服务员手拿无线点菜机，设置有做面条和做馅饼命令触发按钮
    waiter.setNoodleCommand(noodleCommand);
    waiter.setPieCommand(pieCommand);

    // 客人A：服务员您好，我想来碗面
    waiter.noodleCommandExecute();
    // 客人B：服务员您好，我要来个馅饼
    waiter.pieCommandExecute();
    // 客人C：服务员您好，给我来碗面
    waiter.noodleCommandExecute();

}
```

### 2. 宏命令

> 宏命令指的是：包含多个命令的命令，是一个命令的组合。假设餐馆里来了一伙4个人，坐在了1号桌上，服务员过来招待，4个人都点完单后，服务员才将1号桌的菜单发送给后厨。

- **MenuCommand 宏命令对象**

```java
/**
 * 菜单对象，是个宏命令对象
 */
public class MenuCommand implements Command{

    /**
     * 记录多个命令对象
     */
    private List<Command> list = new ArrayList();

    /**
     * 点餐，将下单餐品加入到菜单中
     * @param command
     */
    public void addCommand(Command command){
        list.add(command);
    }

    @Override
    public void execute() {
        for(Command command : list){
            command.execute();
        }
    }
}
```

- **Waiter 调用者**

```java
public static void main(String[] args) {
	// 创建后厨对象
	Kitchen kitchen = new Kitchen();

	// 创建一个服务员对象
	Waiter waiter = new Waiter();

	// 创建做面和做馅饼的命令对象
	NoodleCommand noodleCommand = new NoodleCommand(kitchen);
	PieCommand pieCommand = new PieCommand(kitchen);

	// 服务员来到1号餐桌：我们有面条和馅饼，请问4位顾客想吃什么
	// 客人A：我要一碗面条
	waiter.orderDish(noodleCommand);
	// 客人B：给我也来一碗面条
	waiter.orderDish(noodleCommand);
	// 客人C：我要一个馅饼加一碗面条
	waiter.orderDish(pieCommand);
	waiter.orderDish(noodleCommand);
	// 客人D：我要一碗面条
	waiter.orderDish(noodleCommand);

	// 服务员：总共4碗面条，一个馅饼，4位请稍等。
	waiter.orderOver();

}
```

### 3. 可撤销和恢复操作示例

> 可撤销指的是回到未执行该命令之前的状态（类似于 Ctrl+z），可恢复指的是取消上次的撤销动作（类似于 Ctrl+Shift+z）。有两种方式来实现可撤销的操作：1、反操作式，也叫补偿式，就是撤销时，执行相反的动作；2、存储恢复式，就是把操作前的状态记录下来，撤销时，直接恢复到上一状态。

#### .1. 反操作式（补偿式）

- Command 命令接口，声明执行、撤销操作

```java
/**
 * 命令接口，声明执行、撤销操作
 */
public interface Command {

    /**
     * 执行命令对应的操作
     */
    void execute();
    /**
     * 执行撤销命令对应的操作
     */
    void undo();

}
```

- AddCommand（加法命令）和 SubtractCommand（减法命令）（命令接口的实现对象）

```java
/**
 * 加法命令
 */
public class AddCommand implements Command{

    /**
     * 持有真正进行运算操作的对象
     */
    private Processor processor;

    /**
     * 要加上的数值
     */
    private int number;

    public AddCommand(Processor processor, int number) {
        this.processor = processor;
        this.number = number;
    }

    @Override
    public void execute() {
        this.processor.add(number);
    }

    @Override
    public void undo() {
        this.processor.substract(number);
    }

}


/**
 * 减法命令
 */
public class SubtractCommand implements Command{

    /**
     * 持有真正进行运算操作的对象
     */
    private Processor processor;

    /**
     * 要减去的数值
     */
    private int number;

    public SubtractCommand(Processor processor, int number) {
        this.processor = processor;
        this.number = number;
    }

    @Override
    public void execute() {
        this.processor.substract(number);
    }

    @Override
    public void undo() {
        this.processor.add(number);
    }

}

```

- Processor 计算器的处理器（接收者）

```java
/**
 * 计算器的处理器，真正实现运算操作
 */
@Data
public class Processor {

    /**
     * 记录运算的结果
     */
    private int result;

    public void add(int num){
        //实现加法功能
        result += num;
    }
    public void substract(int num){
        //实现减法功能
        result -= num;
    }
}
```

- Calculator 计算器（调用者）

```java
/**
 * 计算器类，上边有加法、减法、撤销和恢复按钮
 */
@Data
public class Calculator {

    /**
     * 操作的命令的记录，撤销时用
     */
    private List<Command> undoCmds = new ArrayList();
    /**
     * 撤销的命令的记录，恢复时用
     */
    private List<Command> redoCmds = new ArrayList();

    private Command addCommand = null;
    private Command substractCommand = null;

    /**
     * 执行加法操作
     */
    public void addPressed(){
        this.addCommand.execute();
        //把操作记录到历史记录里面
        undoCmds.add(this.addCommand);
    }

    /**
     * 执行减法操作
     */
    public void substractPressed(){
        this.substractCommand.execute();
        //把操作记录到历史记录里面
        undoCmds.add(substractCommand);
    }

    /**
     * 撤销一步操作
     */
    public void undoPressed(){
        if(this.undoCmds.size()>0){
            //取出最后一个命令来撤销
            Command cmd = this.undoCmds.get(this.undoCmds.size()-1);
            cmd.undo();
            //把这个命令记录到恢复的历史记录里面
            this.redoCmds.add(cmd );
            //把最后一个命令删除掉，
            this.undoCmds.remove(cmd);
        }else{
            System.out.println("很抱歉，没有可撤销的命令");
        }
    }

    /**
     * 恢复一步操作
     */
    public void redoPressed(){
        if(this.redoCmds.size()>0){
            //取出最后一个命令来恢复
            Command cmd = this.redoCmds.get(this.redoCmds.size()-1);
            cmd.execute();
            //把命令记录到可撤销的历史记录里面
            this.undoCmds.add(cmd);
            //把最后一个命令删除掉
            this.redoCmds.remove(cmd);
        }else{
            System.out.println("很抱歉，没有可恢复的命令");
        }
    }
}
```

```java
public static void main(String[] args) {

	// 创建接收者，就是我们的计算机的处理器
	Processor processor = new Processor();

	// 创建计算器对象
	Calculator calculator = new Calculator();

	System.out.println("爸爸：小明，过来帮爸爸算算今天赚了多少钱");
	System.out.println("小明：来了，爸爸你说吧");
	System.out.println("爸爸：白菜卖了20块");
	AddCommand addCommand = new AddCommand(processor,20);
	calculator.setAddCommand(addCommand);
	calculator.addPressed();
	System.out.println("小明：卖白菜的钱："+processor.getResult());

	System.out.println("爸爸：萝卜卖了15块");
	addCommand = new AddCommand(processor,15);
	calculator.setAddCommand(addCommand);
	calculator.addPressed();
	System.out.println("小明：加上卖萝卜的钱："+processor.getResult());

	System.out.println("买了一包烟，花了5块");
	SubtractCommand subtractCommand = new SubtractCommand(processor, 5);
	calculator.setSubstractCommand(subtractCommand);
	calculator.substractPressed();
	System.out.println("小明：减去买烟的钱："+processor.getResult());

	System.out.println("爸爸：不对好像算错了，重来");
	calculator.undoPressed();
	System.out.println("小明：撤销一次后："+processor.getResult());
	calculator.undoPressed();
	System.out.println("小明：撤销两次后："+processor.getResult());
	calculator.undoPressed();
	System.out.println("小明：撤销三次后："+processor.getResult());

	System.out.println("爸爸：哈哈~好像白菜和萝卜没算错，烟这个是私房钱买的，别算进去了");
	calculator.redoPressed();
	System.out.println("小明：恢复一次操作："+processor.getResult());
	calculator.redoPressed();
	System.out.println("小明：恢复两次操作："+processor.getResult());
	
}

```

#### .2. 存储恢复式

> **通过存储恢复的方式**进行撤销和恢复撤销。我们只需要让每个命令在执行之前，先记住此时的值，恢复时，只需要将这个命令记住的值恢复就好了。

```java
/**
 * 加法命令
 */
public class AddCommand implements Command{

    /**
     * 持有真正进行运算操作的对象
     */
    private Processor processor;

    /**
     * 记录之前的值
     */
    private int previousValue;

    /**
     * 要加上的数值
     */
    private int number;

    public AddCommand(Processor processor, int number) {
        this.processor = processor;
        this.number = number;
    }

    @Override
    public void execute() {
        this.previousValue = this.processor.getResult();
        this.processor.add(number);
    }

    @Override
    public void undo() {
        this.processor.setResult(this.previousValue);
    }

}
```



### 4. 队列请求

- 对命令对象进行排队，组成工作队列，然后依次取出命令对象来执行。通常用多线程或线程池来进行命令队列的处理。
- 日志请求，就是把请求保存下来，一般是采用持久化存储的方式。这样，如果在运行请求的过程中，系统崩溃了，系统重新启动时，就可以从保存的历史记录中，获取日志请求，并重新执行。Java中实现日志请求，一般就是将对象序列化保存起来，使用时，进行反序列化操作。

```java
public class CommandQueue {
    private LinkedList<Command> commands;

    public CommandQueue(){
        commands = new LinkedList<>();
    }

    public synchronized void addCommand(Command command){
        commands.add(command);
    }

    public synchronized Command getCommand(){
        if(commands.size() != 0){
            return commands.removeLast();
        }

        return null;
    }
}
```

```java
public class RequestCommand implements Command {
    private String name;

    public RequestCommand(String name){
        this.name = name;
    }

    @Override
    public void execute() {
        System.out.println("process request " + name);
    }

    @Override
    public void undo() {
        //请求命令没有撤销功能，这里不做任何处理
    }
}
```

```java
public class CommandQueueTest {
    public static void main(String[] args) {
        //创建请求队列
        CommandQueue commandQueue = new CommandQueue();
        //创建请求命令
        for(int i=0;i<15;i++){
            RequestCommand requestCommand = new RequestCommand("request" + i);
            commandQueue.addCommand(requestCommand);
        }

        //多线程执行请求队列中的命令
        for(int i=0;i<15;i++){
                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        //从命令队列中取出命令执行
                        commandQueue.getCommand().execute();
                    }
                });
            thread.start();
        }


    }
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/commandmode/  

