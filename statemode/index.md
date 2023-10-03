# StateMode


> 状态模式（State Pattern）：它主要用来解决`对象在多种状态转换时`，需要`对外输出不同的行为的问题`。状态和行为是一一对应的，状态之间可以相互转换。
>
> - 应用场景 ：当一个事件或者对象很很多种状态，状态之间会相互依赖，对不同的状态要求有不同的行为的时候，可以考虑使用状态模式。
>
> - 当一个对象的内在状态改变时，允许改变其行为，这个对象看起来像是改变其类。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705084330703.png)

- Context 类为环境角色，用于维护State 实例，这个实例定义当前状态。
- State 是抽象状态角色，定义一个`接口封装与Context 的一个特点接口相关行为`。
- ConcreteState `具体的状态角色`，每个子类实现一个与Context 的一个状态相关行为。

### 0. java Demo0

> 糖果机有四个状态，在不同的动作下状态会发生转变。后续可能会添加其他状态，例如转动曲柄，有可能调出两颗弹。
>
> 1、定义一个State接口，在这个接口内，糖果机的每个工作都有一个对应的方法。
>
> 2、然后为机器中的每个状态实现状态类。这些类将负责在对应的状态下进行机器的行为。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705090327404.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705091701818.png)

### 1. java Demo1

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705084625384.png)

```java
package com.example.demo.state;

/**
 * 状态抽象类
 */
public abstract class State {
	// 扣除积分 - 50
	public abstract void deductMoney();
	// 是否抽中奖品
	public abstract boolean raffle();
	// 发放奖品
	public abstract void dispensePrize();
}
package com.example.demo.state;

import java.util.Random;

public class CanRaffleState extends State {
	
	RaffleActivity activity;
	public CanRaffleState(RaffleActivity activity) { 
		this.activity = activity;
	}

	//已经扣除了积分，不能再扣 
	@Override
	public void deductMoney() {
		System.out.println("已经扣取过了积分"); 
	}

	@Override
	public boolean raffle() {
		System.out.println("正在抽奖，请稍等!"); 
		Random r = new Random();
		int num = r.nextInt(10);
		// 10%中奖机会
		if(num == 0){
			// 改变活动状态为发放奖品 context 
			activity.setState(activity.getDispenseState()); 
			return true;
		}else{ 
			System.out.println("很遗憾没有抽中奖品!"); // 改变状态为不能抽奖 
			activity.setState(activity.getNoRafflleState()); 
			return false;
		}
	}

	// 不能发放奖品 
	@Override
	public void dispensePrize() {
		System.out.println("没中奖，不能发放奖品"); 
	}

}
package com.example.demo.state;

public class RaffleActivity {
	// state 表示活动当前的状态，是变化 
	State state = null;
	
	// 奖品数量 
	int count = 0;
	// 四个属性，表示四种状态
	State noRafflleState = new NoRaffleState(this); 
	State canRaffleState = new CanRaffleState(this);
	State dispenseState = new DispenseState(this); 
	State dispensOutState = new DispenseOutState(this);
	//构造器
	//1. 初始化当前的状态为 noRafflleState(即不能抽奖的状态) 
	//2. 初始化奖品的数量
	public RaffleActivity( int count) {
		this.state = getNoRafflleState();
		this.count = count; 
	}
	
	//扣分, 调用当前状态的 deductMoney 
	public void debuctMoney(){
		state.deductMoney(); 
	}
	//抽奖
	public void raffle(){
		// 如果当前的状态是抽奖成功
		if(state.raffle()){ 
			//领取奖品
			state.dispensePrize(); 
		}
	}
	public State getState() { 
		return state;
	}
	public void setState(State state) { 
		this.state = state;
	}
	//这里请大家注意，每领取一次奖品，count-- 
	public int getCount() {
		int curCount = count; 
		count--;
		return curCount; 
	}
	public void setCount(int count) { 
		this.count = count;
	}
	
	public State getNoRafflleState() { 
		return noRafflleState;
	}
	public void setNoRafflleState(State noRafflleState) { 
		this.noRafflleState = noRafflleState;
	}
	public State getCanRaffleState() { 
		return canRaffleState;
	}
	public void setCanRaffleState(State canRaffleState) { 
		this.canRaffleState = canRaffleState;
	}
	public State getDispenseState() { 
		return dispenseState;
	}
	public void setDispenseState(State dispenseState) { 
		this.dispenseState = dispenseState;
	}
	public State getDispensOutState() { 
		return dispensOutState;
	}
	public void setDispensOutState(State dispensOutState) { 
		this.dispensOutState = dispensOutState;
	}	
}
package com.example.demo.state;

/**
 * 奖品发放完毕状态
 * 说明，当我们 activity 改变成 DispenseOutState， 抽奖活动结束
 * @author zhaozhaohai
 *
 */
public class DispenseOutState extends State {

	// 初始化时传入活动引用 
	RaffleActivity activity;
	public DispenseOutState(RaffleActivity activity) { 
		this.activity = activity;
	}
	@Override
	public void deductMoney() {
		System.out.println("奖品发送完了，请下次再参加"); 
	}

	@Override
	public boolean raffle() {
		System.out.println("奖品发送完了，请下次再参加");
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void dispensePrize() {
		// TODO Auto-generated method stub
		System.out.println("奖品发送完了，请下次再参加");
	}

}
package com.example.demo.state;

/**
 * 发放奖品的状态
 * @author zhaozhaohai
 *
 */
public class DispenseState extends State {
	
	// 初始化时传入活动引用，发放奖品后改变其状态 
	RaffleActivity activity;
	public DispenseState(RaffleActivity activity) { 
		this.activity = activity;
	}

	@Override
	public void deductMoney() {
		System.out.println("不能扣除积分"); 
	}
	@Override
	public boolean raffle() {
		System.out.println("不能抽奖");
		return false; 
	}
	//发放奖品
	@Override
	public void dispensePrize() {
		if(activity.getCount() > 0){ 
			System.out.println("恭喜中奖了");
			// 改变状态为不能抽奖 
			activity.setState(activity.getNoRafflleState());
		}else{
			System.out.println("很遗憾，奖品发送完了");
			// 改变状态为奖品发送完毕, 后面我们就不可以抽奖 
			activity.setState(activity.getDispensOutState()); 
			//System.out.println("抽奖活动结束");
			//System.exit(0);
		}	
	}

}
package com.example.demo.state;

public class NoRaffleState extends State {

	// 初始化时传入活动引用，扣除积分后改变其状态 
	RaffleActivity activity;
	public NoRaffleState(RaffleActivity activity) { 
		this.activity = activity;
	}
	// 当前状态可以扣积分 , 扣除后，将状态设置成可以抽奖状态 
	@Override
	public void deductMoney() {
		System.out.println("扣除 50 积分成功，您可以抽奖了");
		activity.setState(activity.getCanRaffleState()); 
	}

	// 当前状态不能抽奖 
	@Override
	public boolean raffle() {
		System.out.println("扣了积分才能抽奖喔!");
		return false; 
	}
	// 当前状态不能发奖品 
	@Override
	public void dispensePrize() {
		System.out.println("不能发放奖品"); 
	}

}
package com.example.demo.state;

public class ClientTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// 创建活动对象，奖品有 1 个奖品
		RaffleActivity activity = new RaffleActivity(1);
		// 我们连续抽 300 次奖 
		for (int i = 0; i < 30; i++) {
			System.out.println("--------第" + (i + 1) + "次抽奖----------"); 
			// 参加抽奖，第一步点击扣除积分 
			activity.debuctMoney();
			// 第二步抽奖
			activity.raffle(); 
		}
	}

}
```

### Resource

- https://my.oschina.net/silence88?tab=newest&catalogId=5659600

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/statemode/  

