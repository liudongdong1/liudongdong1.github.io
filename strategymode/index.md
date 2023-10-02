# StrategyMode


> 1、封装变化（把可能变化的代码封装起来）
>
> 2、多用组合，少用继承（我们使用组合的方式，为客户设置了算法）
>
> 3、针对接口编程，不针对实现（对于Role类的设计完全的针对角色，和技能的实现没有关系）

#### 1.案例介绍

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210306112040518.png)

> 每个角色对应一个名字，每类角色对应一种样子，每个角色拥有一个逃跑、攻击、防御的技能。
>
> - 遵循设计的原则，找出应用中可能需要变化的部分，把它们独立出来，不要和那些不需要变化的代码混在一起。我们发现，对于每个角色的display，attack，defend，run都是有可能变化的，于是我们必须把这写独立出来。再根据另一个设计原则：针对接口（超类型）编程，而不是针对实现编程。

```java
package com.zhy.bean;
public interface IAttackBehavior
{
	void attack();
}

package com.zhy.bean;
public interface IDefendBehavior
{
	void defend();
}

package com.zhy.bean;
public interface IDisplayBehavior
{
	void display();
}

//技能一个实例
package com.zhy.bean;
public class AttackJY implements IAttackBehavior
{
 
	@Override
	public void attack()
	{
		System.out.println("九阳神功！");
	}
}

package com.zhy.bean;
 
/**
 * 游戏的角色超类
 * 
 * @author zhy
 * 
 */
public abstract class Role
{
	protected String name;
 
	protected IDefendBehavior defendBehavior;
	protected IDisplayBehavior displayBehavior;
	protected IRunBehavior runBehavior;
	protected IAttackBehavior attackBehavior;
 
	public Role setDefendBehavior(IDefendBehavior defendBehavior)
	{
		this.defendBehavior = defendBehavior;
		return this;
	}
 
	public Role setDisplayBehavior(IDisplayBehavior displayBehavior)
	{
		this.displayBehavior = displayBehavior;
		return this;
	}
 
	public Role setRunBehavior(IRunBehavior runBehavior)
	{
		this.runBehavior = runBehavior;
		return this;
	}
 
	public Role setAttackBehavior(IAttackBehavior attackBehavior)
	{
		this.attackBehavior = attackBehavior;
		return this;
	}
 
	protected void display()
	{
		displayBehavior.display();
	}
 
	protected void run()
	{
		runBehavior.run();
	}
 
	protected void attack()
	{
		attackBehavior.attack();
	}
 
	protected void defend()
	{
		defendBehavior.defend();
	}
}

//角色实例
package com.zhy.bean;
public class RoleA extends Role
{
	public RoleA(String name)
	{
		this.name = name;
	}
 
}

```



---

> 作者: liudongdong  
> URL: liudongdong1.github.io/strategymode/  

