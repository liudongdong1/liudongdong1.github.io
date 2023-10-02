# DDD 设计理念


> 微服务架构，在集中式架构中，系统分析、设计和开发往往是独立进行的，而且各个阶段负责人可能不一样，那么就涉及到交流信息丢失的问题， 另外项目从分析到开发经历的流程很长，很容易最终开发设计与需求实现的不一样，微服务主要就是解决第二阶段的这些痛点，实现应用之间的解耦，解决单体应用扩展性的问题。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/v2-c17113de024188a9e0bf4f05a4c89a26_720w.jpg)

### 1. [概念](https://www.processon.com/mindmap/610a687c1e0853337b1b7300)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210804183341548.png)

> 面对客户的业务需求，由`领域专家与开发团队`展开充分的交流，经过`需求分析与知识提炼`，以获得清晰的问题域。通过`对问题域进行分析和建模`，`识别限界上下文`，`利用它划分相对独立的领域`，再通过上下文映射`建立它们之间的关系`，辅以`分层架构与六边形架构划分系统的逻辑边界与物理边界`，`界定领域与技术之间的界限`。之后，进入战术设计阶段，深入到限界上下文内对领域进行建模，并以领域模型指导程序设计与编码实现。若在实现过程中，发现领域模型存在重复、错位或缺失时，再进而对已有模型进行重构，甚至重新划分限界上下文。
>
> 两个不同阶段的设计目标是保持一致的，它们是一个连贯的过程，彼此之间又相互指导与规范，并最终保证一个有效的领域模型和一个富有表达力的实现同时演进。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051042.jpg)

### 2. [提炼问题阈](https://www.processon.com/mindmap/610a717a5653bb143a27de38)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051043.png)

### 3. [专注核心问题](https://www.processon.com/view/5cba8957e4b059e20a0068c8#map)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051044.png)

> 在`一个大的问题空间中会同时存在很多的小问题域`，而这些`小问题域往往只有少部分是核心领域`，其他的可能都是通用域和支撑域。核心域是我们软件的根本竞争力所在，因此也可以说是我们编写软件的原因。拿一个在线拍卖网站来说，可以见下图所示划分了核心域、支撑域和通用域：

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210510455.jpg)

### 4. [驱动模型设计](https://www.processon.com/view/5cbaa844e4b01941c8b441d2)

> `模型驱动设计专注于实现以及对于初始模型可能需要修改的约束`，`领域驱动设计则专注于语言、协作和领域知识`，他们是一个彼此互补的关系。而要实现协作，就需要使用通用语言，借助通用语言可以将分析模型和代码模型绑定在一起，并最终实现团队建模。实践UL是一个持续的过程，多个迭代后会不断对UL进行验证和改进，以便实现更好的协作。
>
> 由于时间和精力都有限，只有仅仅为核心域应用模型驱动设计和创建UL才能带来最大的价值，而不需要将这些实践应用到整个应用程序之中。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051046.png)

### 5. [**领域模型实现模式**](https://www.processon.com/view/5cbab6c5e4b06bcc13844497)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051047.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051048.jpg)

- 领域模型模式：适用于复杂问题域，领域中的概念被封装为数据和行为的对象
- 事务脚本模式：组织所有的领域逻辑来满足业务事务或用例
- 表模块模式：代表着以对象形式建模的数据，数据驱动
- 活动记录模式：类似表模块，数据驱动，关注表中的行而非表本身
- 贫血模式：类似领域模型，不包含任何行为，纯粹的一个对象状态模型，需要一个单独的服务类来实现行为

### 6. [**使用有界上下文维护领域模型的完整性**](https://www.processon.com/view/5cbad3dee4b09a3e45a3fbc6)

> 从`展现层`、`领域逻辑层`再到`持久化层`的完整代码堆栈，正应对了我们的每一个`微服务的应用程序`，也具有较高的独立性，拥有自己的数据库和一套完成的垂直切片的架构模式。`不应该局限在某一种或者两种架构模式上，而是应该量身应用`，没有复杂性业务逻辑的微服务，那就应该KISS（Keep It Simple & Stupid），否则就可以考虑DDD。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/2021051049.png)

![微软的大DEMO项目eShopOnContainers](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210510410.png)

### 7. [上下文映射](https://www.processon.com/view/5cbc3240e4b0bab909613768)

> 上下文映射用来`捕获各个有界上下文之间的技术与组织关系`，它最大的作用就是`保持模型的完整性`。在战略设计阶段，针对问题域，通过引入限界上下文和上下文映射可以对问题域进行合理的分解，识别出核心领域和子领域，并确定领域的边界以及他们之间的关系，从而维持模型的完整性。
>
> 限界上下文不仅局限于对领域模型的控制，而在于`分离关注点之后，使得整个上下文可以成为独立部署的设计单元`，这就是我们非常熟悉的“微服务”的概念；而上下文映射的诸多模式则对应了微服务之间的协作。　

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210510411.png)

### 8. [应用程序架构](https://www.processon.com/view/5cc1cbe4e4b0841b84400fc9)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210510412.png)

### 9. **[团队开始应用DDD通常会遇到的问题](https://www.processon.com/view/5cc46afbe4b08b66b9bd9513)**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210510413.png)

### 10. [应用DDD的原则、实践与模式](https://www.processon.com/view/5cc5568be4b059e20a0bc1e1)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/20210510414.png)

### Resource

- http://www.uml.org.cn/sjms/202105104.asp?artid=23938

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/ddd%E8%AE%BE%E8%AE%A1%E7%90%86%E5%BF%B5/  

