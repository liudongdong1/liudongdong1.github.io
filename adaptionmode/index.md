# AdaptionMode


> 将一个类的接口转换成客户期望的另一个接口，适配器让原本接口不兼容的类可以相互合作。这个定义还好，说适配器的功能就是把一个接口转成另一个接口。适配器模式可分为对象适配器和类适配器两种，在**对象适配器模式**中，适配器与适配者之间是**关联**关系；在**类适配器模式**中，适配器与适配者之间是**继承**（或实现）关系。
>
> **角色：**
>
> - **Target（目标抽象类）**：目标抽象类`定义客户所需接口`，可以是一个抽象类或接口，也可以是具体类。用户通过操作 Targe来调用适配者类中方法。
> - **Adapter（适配器类）**：适配器可以`调用另一个接口`，作为一个转换器，对Adaptee和Target进行适配，适配器类是适配器模式的核心，在对象适配器中，它`通过继承Target并关联一个Adaptee对象`使二者产生联系。
> - **Adaptee（适配者类）**：适配者即`被适配的角色`，它`定义了一个已经存在的接口`，这个接口需要适配，适配者类一般是一个具体类，`包含了客户希望使用的业务方法`，在某些情况下可能没有适配者类的源代码。
>
> **优点：**
>
> - `将目标类和适配者类解耦`，通过引入一个适配器类来重用现有的适配者类，无须修改原有结构。
> - 增加了类的透明性和复用性，`将具体的业务实现过程封装在适配者类中`，`对于客户端类而言是透明的`，而且提高了适配者的复用性，同一个适配者类可以在多个不同的系统中复用。
>
> **适用场景**：
>
> - 系统需要使用一些现有的类，而这些类的接口（如方法名）不符合系统的需要，甚至没有这些类的源代码。
> - 想创建一个可以重复使用的类，用于与一些彼此之间没有太大关联的一些类，包括一些可能在将来引进的类一起工作。

- ###### 类适配器

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704165911702.png)

- **对象适配器**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704170032471.png)

```java
public class Mobile
{
	/**
	 * 充电
	 * @param power 
	 */
	public void inputPower(V5Power power)
	{
		int provideV5Power = power.provideV5Power();
		System.out.println("手机（客户端）：我需要5V电压充电，现在是-->" + provideV5Power + "V");
	}
}

package com.zhy.pattern.adapter;
/**
 * 提供5V电压的一个接口
 * @author zhy
 *
 */
public interface V5Power
{
	public int provideV5Power();
}
/**
 * 适配器，把220V电压变成5V
 * @author zhy
 *
 */
public class V5PowerAdapter implements V5Power
{
	/**
	 * 组合的方式
	 */
	private V220Power v220Power ;
	
	public V5PowerAdapter(V220Power v220Power)
	{
		this.v220Power = v220Power ;
	}
 
	@Override
	public int provideV5Power()
	{
		int power = v220Power.provideV220Power() ;
		//power经过各种操作-->5 
		System.out.println("适配器：我悄悄的适配了电压。");
		return 5 ; 
	} 
	
}

/**
 * 家用220V交流电
 * @author zhy
 *
 */
public class V220Power
{
	/**
	 * 提供220V电压
	 * @return
	 */
	public int provideV220Power()
	{
		System.out.println("我提供220V交流电压。");
		return 220 ; 
	}
}

```

### Spring案例

#### .1. SpringAOP 适配者模式

Advice`的类型有：`MethodBeforeAdvice`、`AfterReturningAdvice`、`ThrowsAdvice

在每个类型 `Advice` 都有对应的拦截器，`MethodBeforeAdviceInterceptor`、`AfterReturningAdviceInterceptor`、`ThrowsAdviceInterceptor`

- **适配者类 **

```java
public interface MethodBeforeAdvice extends BeforeAdvice {
    void before(Method var1, Object[] var2, @Nullable Object var3) throws Throwable;
}

public interface AfterReturningAdvice extends AfterAdvice {
    void afterReturning(@Nullable Object var1, Method var2, Object[] var3, @Nullable Object var4) throws Throwable;
}

public interface ThrowsAdvice extends AfterAdvice {
}

```

- **目标接口 Target**

```java
//有两个方法，一个判断 `Advice` 类型是否匹配，一个是工厂方法，创建对应类型的 `Advice` 对应的拦截器
public interface AdvisorAdapter {
    boolean supportsAdvice(Advice var1);

    MethodInterceptor getInterceptor(Advisor var1);
}
```

- **适配器类 Adapter**

```java
class MethodBeforeAdviceAdapter implements AdvisorAdapter, Serializable {
	@Override
	public boolean supportsAdvice(Advice advice) {
		return (advice instanceof MethodBeforeAdvice);
	}

	@Override
	public MethodInterceptor getInterceptor(Advisor advisor) {
		MethodBeforeAdvice advice = (MethodBeforeAdvice) advisor.getAdvice();
		return new MethodBeforeAdviceInterceptor(advice);
	}
}

@SuppressWarnings("serial")
class AfterReturningAdviceAdapter implements AdvisorAdapter, Serializable {
	@Override
	public boolean supportsAdvice(Advice advice) {
		return (advice instanceof AfterReturningAdvice);
	}
	@Override
	public MethodInterceptor getInterceptor(Advisor advisor) {
		AfterReturningAdvice advice = (AfterReturningAdvice) advisor.getAdvice();
		return new AfterReturningAdviceInterceptor(advice);
	}
}

class ThrowsAdviceAdapter implements AdvisorAdapter, Serializable {
	@Override
	public boolean supportsAdvice(Advice advice) {
		return (advice instanceof ThrowsAdvice);
	}
	@Override
	public MethodInterceptor getInterceptor(Advisor advisor) {
		return new ThrowsAdviceInterceptor(advisor.getAdvice());
	}
}

```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210704171255301.png)

#### .2. SpringJPA 适配者模式

首先定义了一个接口的 `JpaVendorAdapter`，然后不同的持久层框架都实现此接口。

jpaVendorAdapter：用于设置实现厂商JPA实现的特定属性，如设置Hibernate的是否自动生成DDL的属性generateDdl；这些属性是厂商特定的，因此最好在这里设置；目前Spring提供 `HibernateJpaVendorAdapter`、`OpenJpaVendorAdapter`、`EclipseLinkJpaVendorAdapter`、`TopLinkJpaVendorAdapter` 四个实现。其中最重要的属性是 database，用来指定使用的数据库类型，从而能**根据数据库类型来决定比如如何将数据库特定异常转换为Spring的一致性异常**，目前支持如下数据库（DB2、DERBY、H2、HSQL、INFORMIX、MYSQL、ORACLE、POSTGRESQL、SQL_SERVER、SYBASE）

```java
public interface JpaVendorAdapter
{
  // 返回一个具体的持久层提供者
  public abstract PersistenceProvider getPersistenceProvider();

  // 返回持久层提供者的包名
  public abstract String getPersistenceProviderRootPackage();

  // 返回持久层提供者的属性
  public abstract Map<String, ?> getJpaPropertyMap();

  // 返回JpaDialect
  public abstract JpaDialect getJpaDialect();

  // 返回持久层管理器工厂
  public abstract Class<? extends EntityManagerFactory> getEntityManagerFactoryInterface();

  // 返回持久层管理器
  public abstract Class<? extends EntityManager> getEntityManagerInterface();

  // 自定义回调方法
  public abstract void postProcessEntityManagerFactory(EntityManagerFactory paramEntityManagerFactory);
}

```

- 适配器实现类 HibernateJpaVendorAdapter

```java
public class HibernateJpaVendorAdapter extends AbstractJpaVendorAdapter {
    //设定持久层提供者
    private final PersistenceProvider persistenceProvider;
    //设定持久层方言
    private final JpaDialect jpaDialect;

    public HibernateJpaVendorAdapter() {
        this.persistenceProvider = new HibernatePersistence();
        this.jpaDialect = new HibernateJpaDialect();
    }

    //返回持久层方言
    public PersistenceProvider getPersistenceProvider() {
        return this.persistenceProvider;
    }

    //返回持久层提供者
    public String getPersistenceProviderRootPackage() {
        return "org.hibernate";
    }

    //返回JPA的属性
    public Map<String, Object> getJpaPropertyMap() {
        Map jpaProperties = new HashMap();

        if (getDatabasePlatform() != null) {
            jpaProperties.put("hibernate.dialect", getDatabasePlatform());
        } else if (getDatabase() != null) {
            Class databaseDialectClass = determineDatabaseDialectClass(getDatabase());
            if (databaseDialectClass != null) {
                jpaProperties.put("hibernate.dialect",
                        databaseDialectClass.getName());
            }
        }

        if (isGenerateDdl()) {
            jpaProperties.put("hibernate.hbm2ddl.auto", "update");
        }
        if (isShowSql()) {
            jpaProperties.put("hibernate.show_sql", "true");
        }

        return jpaProperties;
    }

    //设定数据库
    protected Class determineDatabaseDialectClass(Database database)     
    {                                                                                       
        switch (1.$SwitchMap$org$springframework$orm$jpa$vendor$Database[database.ordinal()]) 
        {                                                                                     
        case 1:                                                                             
          return DB2Dialect.class;                                                            
        case 2:                                                                               
          return DerbyDialect.class;                                                          
        case 3:                                                                               
          return H2Dialect.class;                                                             
        case 4:                                                                               
          return HSQLDialect.class;                                                           
        case 5:                                                                               
          return InformixDialect.class;                                                       
        case 6:                                                                               
          return MySQLDialect.class;                                                          
        case 7:                                                                               
          return Oracle9iDialect.class;                                                       
        case 8:                                                                               
          return PostgreSQLDialect.class;                                                     
        case 9:                                                                               
          return SQLServerDialect.class;                                                      
        case 10:                                                                              
          return SybaseDialect.class; }                                                       
        return null;              
    }

    //返回JPA方言
    public JpaDialect getJpaDialect() {
        return this.jpaDialect;
    }

    //返回JPA实体管理器工厂
    public Class<? extends EntityManagerFactory> getEntityManagerFactoryInterface() {
        return HibernateEntityManagerFactory.class;
    }

    //返回JPA实体管理器
    public Class<? extends EntityManager> getEntityManagerInterface() {
        return HibernateEntityManager.class;
    }
}

```

#### .3. SpringMVC 适配者模式

> `DispatcherServlet` 作为用户，`HandlerAdapter` 作为期望接口，具体的适配器实现类用于对目标类进行适配，`Controller` 作为需要适配的类。

- **适配器接口 `HandlerAdapter`**

```java
public interface HandlerAdapter {
    boolean supports(Object var1);

    ModelAndView handle(HttpServletRequest var1, HttpServletResponse var2, Object var3) throws Exception;

    long getLastModified(HttpServletRequest var1, Object var2);
}
```

- **HttpRequestHandlerAdapter**

```java
public class HttpRequestHandlerAdapter implements HandlerAdapter {
    public HttpRequestHandlerAdapter() {
    }

    public boolean supports(Object handler) {
        return handler instanceof HttpRequestHandler;
    }

    public ModelAndView handle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        ((HttpRequestHandler)handler).handleRequest(request, response);
        return null;
    }

    public long getLastModified(HttpServletRequest request, Object handler) {
        return handler instanceof LastModified ? ((LastModified)handler).getLastModified(request) : -1L;
    }
}

```

```java
public class DispatcherServlet extends FrameworkServlet {
    private List<HandlerAdapter> handlerAdapters;
    
    //初始化handlerAdapters
    private void initHandlerAdapters(ApplicationContext context) {
        //..省略...
    }
    
    // 遍历所有的 HandlerAdapters，通过 supports 判断找到匹配的适配器
    protected HandlerAdapter getHandlerAdapter(Object handler) throws ServletException {
		for (HandlerAdapter ha : this.handlerAdapters) {
			if (logger.isTraceEnabled()) {
				logger.trace("Testing handler adapter [" + ha + "]");
			}
			if (ha.supports(handler)) {
				return ha;
			}
		}
	}
	
	// 分发请求，请求需要找到匹配的适配器来处理
	protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
		HttpServletRequest processedRequest = request;
		HandlerExecutionChain mappedHandler = null;

		// Determine handler for the current request.
		mappedHandler = getHandler(processedRequest);
			
		// 确定当前请求的匹配的适配器.
		HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

		ha.getLastModified(request, mappedHandler.getHandler());
					
		mv = ha.handle(processedRequest, response, mappedHandler.getHandler());
    }
	// ...省略...
}	

```


### Resource

- https://juejin.cn/post/6844903682136342541

---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/adaptionmode/  

