# CodeRegulation


> 先进行`需求分析和系统设计`，这样可以帮助我们正确的理解功能，然后再开发代码，在开发时应该`边开发编写测试代码`，保证开发的函数功能是正确的，否则都开完在统一测试，出现问题很难快速的定位问题，开发完成后要对代码进行上线和持续的运维迭代。

> `需求分析`：定义系统/软件的`黑盒行为`，即从外部看系统可以实现什么功能
>
> `系统设计`：系统设计/软件的白盒机制，详细的说明系统的`功能、组成，以及组件间`是如何划分的
>
> 需求是系统设计的决策来源。
>
> 系统设计应遵循的约束：`计算、存储、I/O网络（资源约束）`
>
> （1）系统中`每一个组件（子系统/模块）的功能都应该是单一`的 
>
> （2）功能单一是`复用和可扩展的`基础
>
> （3）软件的`耦合度要低`
>
> （4）接口的迭代过程中应实现`前向兼容`
>
> `数据模块`的主要功能：完成对数据的封装（模块的内部变量，类的内部变量）、对外提供明确的数据访问接口（数据结构和算法属于模块内部的工作）；`过程类模块`的主要功能：本身不含数据（可从文件中读取数据），可以调用其他数据或过程类模块，主要完成操作。
>
> （1）文件头：模块的说明`（功能简介）`、`修改历史（时间、人、修改的内容）`、模块内荣的顺序应该一致
>
> （2）函数：函数描述的三要素：`功能描述@func`、`形参列表描述@params（含义、限制条件）`、`返回值的描述@return`（函数返回值的可能性应该尽量枚举）`函数的规模要尽量小，尽量控制在39行以内`，如果c或c++要控制在两屏以内。函数应该为`单入口单出口`（如果存在多线程时更要单入单出，单出指函数只有一个return）
>
> - `面向对象的实质是实现对数据的封装`，开发程序时应该以数据为重心进行考虑问题，`在过程类的模块中，与类的成员变量无关的函数，应该作为一个独立的函数放在类的外面`，不作为类的方法，这样方便复用和扩展。
> - 系统中多态和继承（最好不要超过3层）都应谨慎使用。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/naming-mindmap.png)

#### 1. 使用空行分割逻辑

> 一般代码超过30行左右，我们就在考虑，要不要把这些代码封装到一个方法中去。但是即使把这一大段代码扔到一个方法中去，在主函数里调用这个方法，也不能保证以后不会修改这个方法了。
>
> 一个方法里面的代码，逻辑也是可以分成一小块一小块的，这个时候我们在这些逻辑中间加上空行，就能告诉别人，我这个代码这里，两个空行中间的代码关联比较大。在没有添加任何注释的情况下，因为有空行来分割逻辑，倒也不是一步就吓退想要看你代码的人。
>
> - `声明语句和执行语句之间插入空行`
> - **运算符前后需要留出空格**

#### 2. 使用注释和花括号

> 如果项目组有使用文档来规范开发，那就要去遵守这个规定。但是文档也有一个问题，就是代码修改之后就要去修改文档，万一文档没有更新，接手的人反而会受到误导。

#### 3. 不用的代码和引用删除

> 需要修改代码时经常注释之前的代码，然后把自己的代码加上去。该删的时候就要删掉。要学会使用SVN或者是Git来进行版本控制。即使当前版本删掉了，回滚到之前版本依然能够找回来。
>
> 对于warning，过多没有使用的引用，或者外部包，框架，库等等也会拖慢程序的运行速度。干掉这些无用的东西既能净化我们的心灵，还能减少不易察觉的隐患。

#### 4. 不要用中文拼音做变量名可用，清晰优雅，高效

> 要明确需求，考虑全面一点，从`正确性，边界值，反例`三个角度去思考问题，定下详细的需求，掌握需求方的意图。不要过早优化，先把粗糙的成品做出来，后续慢慢优化。
>
> 具体到代码，很多时候都需要调试代码，不要一上来就断点调试，`先看一遍代码，检查代码逻辑，理一理思路`，然后采用`二分法设置断点输出日志`，快速定位问题代码。优化时，确定一个优化的基准，优化之后有对比，用数据来告诉别人优化的效果。

#### 5. 代码命名

##### 5.1. java 命名规范

| 类型(名) | 约束                                                       | 例                        |
| -------- | ---------------------------------------------------------- | ------------------------- |
| 项目     | 全部小写多个单词用中划线分隔‘-’                            | spring-cloud              |
| 包       | 全部小写                                                   | com.alibaba.fastjson      |
| 类       | 单词首字母大写                                             | Feature,FieldDeserializer |
| 变量     | 首字母小写多个单词组成时，除首个单词其他单词首字母都要大写 | password, userName        |
| 常量     | 全部大写，多个单词，用'_'分隔                              | CACHEEXPIREDTIME          |
| 方法     | 同变量                                                     | read(), getById(Long id)  |

##### 5.2. 包命名

> 【前缀】 【发起者名】【项目名】【模块名】

| 前缀       | 例                             | 含义                                                         |
| ---------- | ------------------------------ | ------------------------------------------------------------ |
| indi或onem | indi.发起者名.项目名.模块名.…… | 个体项目个人发起，但非自己独自完成可公开或私有项目，copyright主要属于发起者。 |
| pers       | pers.个人名.项目名.模块名.……   | 个人项目指个人发起，独自完成，可分享的项目copyright主要属于个人 |
| priv       | priv.个人名.项目名.模块名.……   | 私有项目，指个人发起，独自完成非公开的私人使用的项目，copyright属于个人。 |
| team       | team.团队名.项目名.模块名.……   | 团队项目，指由团队发起并由该团队开发的项目copyright属于该团队所有 |
| 顶级域名   | com.公司名.项目名.模块名.……    | 公司项目copyright由项目发起的公司所有                        |

##### 5.3. 类命名

| 属性(类)     | 约束                                    | 例                                                           |
| ------------ | --------------------------------------- | ------------------------------------------------------------ |
| 抽象         | Abstract 或Base 开头                    | BaseUserService                                              |
| 枚举         | Enum 作为后缀                           | OSType                                                       |
| 工具         | Utils作为后缀                           | StringUtils                                                  |
| 异常         | Exception结尾                           | RuntimeException                                             |
| 接口实现     | 接口名+ Impl                            | UserServiceImpl                                              |
| 领域模型相   | /DO/DTO/VO/DAO                          | 正例：UserDAO反例：UserDao                                   |
| 设计模式相关 | Builder，Factory等                      | 当使用到设计模式时要使用对应的设计模式作为后缀如ThreadFactory |
| 处理特定功能 | Handler，PredicateValidator             | 表示处理器，校验器，断言这些类工厂还有配套的方法名如handle，predicate，validate |
| 测试         | Test后缀                                | UserServiceTest表示用来测试UserService类的                   |
| MVC分层      | Controller，ServiceServiceImpl，DAO后缀 | UserManageControllerUserManageDAO                            |

##### 5.4. 方法命名

4.1 返回真伪值的方法

注：pre- prefix前缀，suf- suffix后缀，alo-alone 单独使用

| 位置 | 单词   | 意义                                                         | 例            |
| ---- | ------ | ------------------------------------------------------------ | ------------- |
| pre  | is     | 对象是否符合期待的状态                                       | isValid       |
| pre  | can    | 对象**能否执行**所期待的动作                                 | canRemove     |
| pre  | should | 调用方执行某个命令或方法是好还是不好应不应该，或者说推荐还是不推荐 | shouldMigrate |
| pre  | has    | 对象**是否持有**所期**待的数据**和属性                       | hasObservers  |
| pre  | needs  | 调用方**是否需要**执行某**个命令或**方法                     | needsMigrate  |

4.2 用来检查的方法

| 单词     | 意义                                               | 例             |
| -------- | -------------------------------------------------- | -------------- |
| ensure   | 检查是否为期待的状态不是则抛出异常或返回error code | ensureCapacity |
| validate | 检查是否为正确的状态不是则抛出异常或返回error code | validateInputs |

4.3 按需求才执行的方法

| 位置 | 单词      | 意义                                    | 例                     |
| ---- | --------- | --------------------------------------- | ---------------------- |
| suf  | IfNeeded  | 需要的时候执行不需要则什么都不做        | drawIfNeeded           |
| pre  | might     | 同上                                    | mightCreate            |
| pre  | try       | 尝试执行失败时抛出异常或是返回errorcode | tryCreate              |
| suf  | OrDefault | 尝试执行失败时返回默认值                | getOrDefault           |
| suf  | OrElse    | 尝试执行失败时返回实际参数中指定的值    | getOrElse              |
| pre  | force     | 强制尝试执行error抛出异常或是返回值     | forceCreate, forceStop |

4.4 异步相关方法

| 位置    | 单词         | 意义               | 例                    |
| ------- | ------------ | ------------------ | --------------------- |
| pre     | blocking     | 线程阻塞方法       | blockingGetUser       |
| suf     | InBackground | 执行在后台线程     | doInBackground        |
| suf     | Async        | 异步方法           | sendAsync             |
| suf     | Sync         | 同步方法           | sendSync              |
| pre/alo | schedule     | Job和Task放入队列  | schedule, scheduleJob |
| pre/alo | post         | 同上               | postJob               |
| pre/alo | execute      | 执行异步或同步方法 | execute,executeTask   |
| pre/alo | start        | 同上               | star,tstartJob        |
| pre/alo | cancel       | 停止异步方法       | cance,cancelJob       |
| pre/alo | stop         | 同上               | stop,stopJob          |

4.5 回调方法

| 位置 | 单词   | 意义                 | 例           |
| ---- | ------ | -------------------- | ------------ |
| pre  | on     | 事件发生时执行       | onCompleted  |
| pre  | before | 事件发生前执行       | beforeUpdate |
| pre  | pre    | 同上                 | preUpdate    |
| pre  | will   | 同上                 | willUpdate   |
| pre  | after  | 事件发生后执行       | afterUpdate  |
| pre  | post   | 同上                 | postUpdate   |
| pre  | did    | 同上                 | didUpdate    |
| pre  | should | 确认事件是否可以执行 | shouldUpdate |

4.6 操作对象生命周期的方法

| 单词       | 意义                   | 例              |
| ---------- | ---------------------- | --------------- |
| initialize | 初始化或延迟初始化使用 | initialize      |
| pause      | 暂停                   | onPause , pause |
| stop       | 停止                   | onStop, stop    |
| abandon    | 销毁的替代             | abandon         |
| destroy    | 同上                   | destroy         |
| dispose    | 同上                   | dispose         |

4.7 与集合操作相关的方法

| 单词     | 意义                     | 例         |
| -------- | ------------------------ | ---------- |
| contains | 是包含指定对象相同的对象 | contains   |
| add      | 添加                     | addJob     |
| append   | 添加                     | appendJob  |
| insert   | 插入到下标n              | insertJob  |
| put      | 添加与key对应的元素      | putJob     |
| remove   | 移除元素                 | removeJob  |
| enqueue  | 添加到队列的最末位       | enqueueJob |
| dequeue  | 从队列中头部取出并移除   | dequeueJob |
| push     | 添加到栈头               | pushJob    |
| pop      | 从栈头取出并移除         | popJob     |
| peek     | 从栈头取出但不移除       | peekJob    |
| find     | 寻找符合条件的某物       | findById   |

4.8 与数据相关的方法

| 单词   | 意义                                 | 例            |
| ------ | ------------------------------------ | ------------- |
| create | 新创建                               | createAccount |
| new    | 新创建                               | newAccount    |
| from   | 从既有的某物新建或是从其他的数据新建 | fromConfig    |
| to     | 转换                                 | toString      |
| update | 更新既有某物                         | updateAccount |
| load   | 读取                                 | loadAccount   |
| fetch  | 远程读取                             | fetchAccount  |
| delete | 删除                                 | deleteAccount |
| remove | 删除                                 | removeAccount |
| save   | 保存                                 | saveAccount   |
| store  | 保存                                 | storeAccount  |
| commit | 保存                                 | commitChange  |
| apply  | 保存或应用                           | applyChange   |
| clear  | 清除或是恢复到初始状态               | clearAll      |
| reset  | 清除或是恢复到初始状态               | resetAll      |

4.9 成对出现的动词

| 单词           | 意义              |
| -------------- | ----------------- |
| get获取        | set 设置          |
| add 增加       | remove 删除       |
| create 创建    | destory 移除      |
| start 启动     | stop 停止         |
| open 打开      | close 关闭        |
| read 读取      | write 写入        |
| load 载入      | save 保存         |
| create 创建    | destroy 销毁      |
| begin 开始     | end 结束          |
| backup 备份    | restore 恢复      |
| import 导入    | export 导出       |
| split 分割     | merge 合并        |
| inject 注入    | extract 提取      |
| attach 附着    | detach 脱离       |
| bind 绑定      | separate 分离     |
| view 查看      | browse 浏览       |
| edit 编辑      | modify 修改       |
| select 选取    | mark 标记         |
| copy 复制      | paste 粘贴        |
| undo 撤销      | redo 重做         |
| insert 插入    | delete 移除       |
| add 加入       | append 添加       |
| clean 清理     | clear 清除        |
| index 索引     | sort 排序         |
| find 查找      | search 搜索       |
| increase 增加  | decrease 减少     |
| play 播放      | pause 暂停        |
| launch 启动    | run 运行          |
| compile 编译   | execute 执行      |
| debug 调试     | trace 跟踪        |
| observe 观察   | listen 监听       |
| build 构建     | publish 发布      |
| input 输入     | output 输出       |
| encode 编码    | decode 解码       |
| encrypt 加密   | decrypt 解密      |
| compress 压缩  | decompress 解压缩 |
| pack 打包      | unpack 解包       |
| parse 解析     | emit 生成         |
| connect 连接   | disconnect 断开   |
| send 发送      | receive 接收      |
| download 下载  | upload 上传       |
| refresh 刷新   | synchronize 同步  |
| update 更新    | revert 复原       |
| lock 锁定      | unlock 解锁       |
| check out 签出 | check in 签入     |
| submit 提交    | commit 交付       |
| push 推        | pull 拉           |
| expand 展开    | collapse 折叠     |
| begin 起始     | end 结束          |
| start 开始     | finish 完成       |
| enter 进入     | exit 退出         |
| abort 放弃     | quit 离开         |
| obsolete 废弃  | depreciate 废旧   |
| collect 收集   | aggregate 聚集    |

#### 6. 代码注释

>注解大体上可以分为两种，一种是javadoc注解，另一种是简单注解。javadoc注解可以生成JavaAPI为外部用户提供有效的支持javadoc注解通常在使用IDEA，或者Eclipse等开发工具时都可以自动生成，也支持自定义的注解模板，仅需要对对应的字段进行解释。参与同一项目开发的同学，尽量设置成相同的注解模板。

##### 6.1. 包注解

>包注解一般在包的根目录下，名称统一为package-info.java。

```java
/**
 * 落地也质量检测
 * 1. 用来解决什么问题
 * 对广告主投放的广告落地页进行性能检测，模拟不同的系统，如Android，IOS等; 模拟不同的网络：2G，3G，4G，wifi等
 *
 * 2. 如何实现
 * 基于chrome浏览器，用chromedriver驱动浏览器，设置对应的网络，OS参数，获取到浏览器返回结果。
 *
 * 注意：网络环境配置信息{@link cn.mycookies.landingpagecheck.meta.NetWorkSpeedEnum}目前使用是常规速度，可以根据实际情况进行调整
 *
 * @author cruder
 * @time 2019/12/7 20:3 下午
 */
package cn.mycookies.landingpagecheck;
```

##### 6.2. 类注解

```java
/**
* Copyright (C), 2019-2020, Jann  balabala...
*
* 类的介绍：这是一个用来做什么事情的类，有哪些功能，用到的技术.....
*
* @author   类创建者姓名 保持对齐
* @date     创建日期 保持对齐
* @version  版本号 保持对齐
*/
```

##### 6.3. 属性注解

```java
/** 提示信息 */
private String userName;
/**
 * 密码
 */
private String password;
```

##### 6.4. 方法注解

```java
/**
  * 方法的详细说明，能干嘛，怎么实现的，注意事项...
  *
  * @param xxx      参数1的使用说明， 能否为null
  * @return 返回结果的说明， 不同情况下会返回怎样的结果
  * @throws 异常类型   注明从此类方法中抛出异常的说明
  */


/**
  * 构造方法的详细说明
  *
  * @param xxx      参数1的使用说明， 能否为null
  * @throws 异常类型   注明从此类方法中抛出异常的说明
  */
```

- https://mp.weixin.qq.com/s?__biz=Mzg2OTA0Njk0OA==&mid=2247486449&idx=1&sn=c3b502529ff991c7180281bcc22877af&chksm=cea2443af9d5cd2c1c87049ed15ccf6f88275419c7dbe542406166a703b27d0f3ecf2af901f8&token=999884676&lang=zh_CN&scene=21#wechat_redirect
- 命名工具：https://unbug.github.io/codelf/#get


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/coderegulation/  

