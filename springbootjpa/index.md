# springbootJPA


> SpringData：其实SpringData就是Spring提供了一个`操作数据的框架`。而`SpringData JPA只是SpringData框架下的一个基于JPA标准操作数据的模块`。
> SpringData JPA：基于`JPA的标准数据进行操作。简化操作持久层的代码。只需要编写接口就可以。`
>
> JPA是Spring Data下的子项目,JPA是Java Persistence API的简称，中文名为Java持久层API，是JDK 5.0注解或XML描述对象－关系表的映射关系，并将运行期的实体对象持久化到数据库中.

#### 1. 环境使用

- 导入jar包

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
```

- yml配置文件

```yml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mytest
    type: com.alibaba.druid.pool.DruidDataSource
    username: root
    password: root
    driver-class-name: com.mysql.jdbc.Driver //驱动
  jpa:
    hibernate:
      ddl-auto: update //自动更新
    show-sql: true  //日志中显示sql语句
```

- application.properties

```python
#项目端口的常用配置
server.port=8081

# 数据库连接的配置
spring.datasource.url=jdbc:mysql:///jpa?useSSL=false
spring.datasource.username=root
spring.datasource.password=zempty123
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver

#数据库连接池的配置，hikari 连接池的配置
spring.datasource.hikari.idle-timeout=30000
spring.datasource.hikari.connection-timeout=10000
spring.datasource.hikari.maximum-pool-size=15
spring.datasource.hikari.minimum-idle=5
spring.datasource.hikari.auto-commit=true


#通过 jpa 自动生成数据库中的表
spring.jpa.hibernate.ddl-auto=update   #这里 update操作注意
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.MySQL5InnoDBDialect
```

> `jpa:hibernate:ddl-auto: update`是hibernate的配置属性，其主要作用是：`自动创建、更新、验证数据库表结构`。该参数的几种配置如下：
> 1.·create：`每次加载hibernate时都会删除上一次的生成的表，然后根据你的model类再重新来生成新表`，哪怕两次没有任何改变也要这样执行，这就是导致数据库表数据丢失的一个重要原因。
> 2.·create-drop：`每次加载hibernate时根据model类生成表，但是sessionFactory一关闭,表就自动删除。`
> 3.·update：最常用的属性，第一次加载hibernate时根据model类会自动建立起表的结构`（前提是先建立好数据库`），以后加载hibernate时根据model类自动更新表结构，即使表结构改变了但表中的行仍然存在不会删除以前的行。要注意的是当部署到服务器后，表结构是不会被马上建立起来的，是要等应用第一次运行起来后才会。
> 4.·validate：`每次加载hibernate时，验证创建数据库表结构，只会和数据库中的表进行比较，不会创建新表，但是会插入新值。`
>
> - **初次创建时会设为create,创建好后改为validate.**

#### 2. 实体类

> 具体参考annotation_JPA, annotation_Lombok进行使用

```java
@Entity
@Getter
@Setter
@Table(name = "person")
public class Person {
    @Id
    @GeneratedValue
    private Long id;
    @Column(name = "name", length = 20)
    private String name;
    @Column(name = "agee", length = 4)
    private int age;
}
```

#### 3. Repository接口

> 这里主要负责数据库的增删查改操作

##### .1. 简单查询

- 基本查询也分为两种，一种是`spring data默认已经实现`，一种是`根据查询的方法来自动解析成SQL`。若只是简单的对单表进行crud只需要继承JpaRepository接口,传递了两个参数`:1.实体类,2.实体类中主键类型`
- 要的语法是`findXXBy`,`readAXXBy`,`queryXXBy`,`countXXBy`, `getXXBy`后面跟属性名称：

```java
package com.yizhu.repository;

import com.yizhu .entity.User;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface UserRepository extends JpaRepository<User, Long>{

    /**
     * 根据年龄查询用户信息
     * @param age
     * @return
     */
    List<User> findAllByAge(Integer age);
    /**
     * 查询所有用户信息
     * @return
     */
    @Query(value = "from User u")
    List<User> findAll();
     /**
     * 根据年龄查询用户信息
     * @param age
     * @return
     */
    @Query(value = "select * from t_user u where u.user_age = ?1", nativeQuery = true)
    List<User> findAllByAge(Integer age);
    /**
     * 根据用户性别和所属组织名称查询用户信息
     * @param userSex
     * @param orgName
     * @return
     */
    @Query(value = "select u from User u left join u.org o where u.userSex = :userSex and o.orgName = :orgName")
    List<User> findUsersBySexAndOrg(@Param("userSex") Integer userSex, @Param("orgName") String orgName);

    /**
     * 根据用户性别和所属组织名称查询用户信息
     * @param userSex
     * @param orgName
     * @return
     */
    List<User> findBySexAndOrg(@Param("sex") Integer sex, @Param("name") String name);

    /**
     * 根据用户名模糊查询
     * @return
     */
    List<User> findAllByNameLike(@Param("name") String name);
}
```

> **CrudRepository 中的save方法是相当于merge+save ，它会先判断记录是否存在，如果存在则更新，不存在则插入记录**

![JpaRepository](https://gitee.com/github-25970295/blogpictureV2/raw/master/JpaRepository.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210619103608467.png)

###### .2. 分页查询

```python
@Test
public void testPageQuery() throws Exception {
	int page=1,size=10;
	Sort sort = new Sort(Direction.DESC, "id");
    Pageable pageable = new PageRequest(page, size, sort);
    userRepository.findALL(pageable);
    userRepository.findByUserName("testName", pageable);
}
```

###### .3. `@Query`注解自定义查询sql

> 1. JPQL 形式的 sql 语句，from 后面是以类名呈现的。
> 2. 原生的 sql 语句，需要使用 nativeQuery = true 指定使用原生 sql

```java
@Modifying
@Query("update User u set u.userName = ?1 where u.id = ?2")
int modifyByIdAndUserId(String  userName, Long id);
	
@Transactional
@Modifying
@Query("delete from User where id = ?1")
void deleteByUserId(Long id);
  
@Transactional(timeout = 10)
@Query("select u from User u where u.emailAddress = ?1")
User findByEmailAddress(String emailAddress);

@Query(value="select o from UserModel o where o.name like %:nn")
public List<UserModel> findByUuidOrAge(@Param("nn") String name);

//使用的 JPQL 的 sql 形式 from 后面是类名
// ?1 代表是的是方法中的第一个参数
@Query("select s from ClassRoom s where s.name =?1")
List<ClassRoom> findClassRoom1(String name);

//这是使用正常的 sql 语句去查询
// :name 是通过 @Param 注解去确定的
@Query(nativeQuery = true,value = "select * from class_room c where c.name =:name")
List<ClassRoom> findClassRoom2(@Param("name")String name);

@Query("select t from Teacher t where t.subject = :subject")
Page<Teacher> getPage(@Param("subject") String subject, Pageable pageable);

//更新类的query语句
@Transactional(timeout = 10)
@Modifying
@Query(value="update UserModel o set o.name=:newName where o.name like %:nn")
public int findByUuidOrAge(@Param("nn") String name,@Param("newName") String newName);
//删除类语句
@Transactional(timeout = 10)
@Query(value = "delete from r_upa where user_id= ?1 and point_indecs_id in (?2)", nativeQuery = true)
@Modifying
void deleteByUserAndPointIndecs(Long uid, List<Long> hids);

//多表查询   一般新建临时类，用于存储查询结果，但不存储数据库
@Query("select h.city as city, h.name as name, avg(r.rating) as averageRating "
		- "from Hotel h left outer join h.reviews r where h.city = ?1 group by h")
Page<HotelSummary> findByCity(City city, Pageable pageable);

@Query("select h.name as name, avg(r.rating) as averageRating "
		- "from Hotel h left outer join h.reviews r  group by h")
Page<HotelSummary> findByCity(Pageable pageable);

Page<HotelSummary> hotels = this.hotelRepository.findByCity(new PageRequest(0, 10, Direction.ASC, "name"));
for(HotelSummary summay:hotels){
		System.out.println("Name" +summay.getName());
	}
```

> - 直接可以使用delete(id),依据id来删除一条数据
> - 也可以使用deleteByName(String name)时，需要添加@Transactional注解，才能使用
> - Spring Data JPA的deleteByXXXX，是先select，在整个Transaction完了之后才执行delete

###### .3. Example<s> 查询

> 按例查询（QBE）是一种用户界面友好的查询技术。 它允许动态创建查询，并且不需要编写包含字段名称的查询。 实际上，按示例查询不需要使用特定的数据库的查询语言来编写查询语句。
>
> 1. Probe: 含有对应字段的实例对象。
> 2. ExampleMatcher：ExampleMatcher携带有关如何匹配特定字段的详细信息，相当于匹配条件。
> 3. Example：由Probe和ExampleMatcher组成，用于查询。

```java
//根据用户名精确查询
@Test
public void test7() {
        // 根据用户名精确查询
        User user = new User();
        user.setUsername("张三");
        Example<User> example = Example.of(user);
        List<User> users = userDao.findAll(example);
        for (User u : users) {
            System.out.println(u);
        }
}
//查询用户名为张开头的用户
@Test
public void test8() {
        // 查询用户名为张开头的用户
        User user = new User();
        user.setUsername("张");
        ExampleMatcher matcher = ExampleMatcher.matching()
                .withMatcher("username",GenericPropertyMatchers.startsWith());
//                .withMatcher()   还可以添加多个条件
        Example<User> example = Example.of(user,matcher);
        List<User> users = userDao.findAll(example);
        for (User u : users) {
            System.out.println(u);
        }
}
```

- `ExampleMatcher.matching()` :默认是返回满足全部条件的用户
- `ExampleMatcher.matchingAny()` :只要有一个条件满足即可
- `GenericPropertyMatchers.startsWith()` : 匹配开头 等价于 like 张%
- `GenericPropertyMatchers.contains()` : 匹配中间 等价于 like %张%
- `GenericPropertyMatchers.endsWith()` : 匹配结尾 等价于 like %张
- `GenericPropertyMatchers.ignoreCase(boolean b)` : 同时还能指定是否忽略大小写

##### .4.  复杂查询

- **Specification**

```java
public interface JpaSpecificationExecutor<T> {
    Optional<T> findOne(@Nullable Specification<T> var1);
    List<T> findAll(@Nullable Specification<T> var1);
    Page<T> findAll(@Nullable Specification<T> var1, Pageable var2);
    List<T> findAll(@Nullable Specification<T> var1, Sort var2);
    long count(@Nullable Specification<T> var1);
}


public interface TeacherRepositoty extends JpaRepository<Teacher,Integer> , JpaSpecificationExecutor {
}

//Predicate toPredicate(Root<T> var1, CriteriaQuery<?> var2, CriteriaBuilder var3);
 //实例化 Specification 类  @PathVariable("subject") String subject
Specification specification = ((root, criteriaQuery, criteriaBuilder) -> {
    // 构建查询条件
    Predicate predicate = criteriaBuilder.equal(root.get("subject"), subject);
    // 使用 and 连接上一个条件
    predicate = criteriaBuilder.and(predicate, criteriaBuilder.greaterThan(root.get("age"), 21));
    return predicate;
});
//使用查询
return teacherRepositoty.findAll(specification);
```

```java
@Test
public void test10() {
        Specification<User> specification = new Specification<User>() {
            @Override
            public Predicate toPredicate(Root<User> root, CriteriaQuery<?> criteriaQuery, CriteriaBuilder criteriaBuilder) {
                // 获取比较的属性
                Path<Object> username = root.get("username");
                Path<Object> password = root.get("password");
                // 构造查询条件
                Predicate p1 = criteriaBuilder.equal(username, "张三");
                Predicate p2 = criteriaBuilder.equal(password, "222");
                // 把条件组合到一起
                return criteriaBuilder.and(p1,p2);
            }
        };
        List<User> users = userDao.findAll(specification);
        users.forEach(System.out::println);
}

//根据用户名进行模糊查询
@Test
public void test11() {
        Specification<User> specification = new Specification<User>() {
            @Override
            public Predicate toPredicate(Root<User> root, CriteriaQuery<?> criteriaQuery, CriteriaBuilder criteriaBuilder) {
                // 获取比较的属性,并指定比较参数的类型String
                Path<String> username = root.get("username");
                // 构造查询条件
                Predicate p1 = criteriaBuilder.like(username, "张%");
                // 把条件组合到一起
                return p1;
            }
        };
        // 根据id倒序排序
        Sort sort = Sort.by("id").descending();
        List<User> users = userDao.findAll(specification,sort);
        users.forEach(System.out::println);
    
```

> 现在有这样的一条 sql 语句 ： **select \* from teacher where age > 20**
>
> 1. ``Predicate`` 是用来建立 `where 后的查寻条件的相当于上述sql语句的 age > 20。`
> 2. `Root 使用来定位具体的查询字段`，比如 `root.get(“age”) ,定位 age字段，`
> 3. `CriteriaBuilder`是用来构建一个字段的范围，相当于` > ,= ,<，and …. 等等`
> 4. `CriteriaQuery `可以用来构建整个 sql 语句，可以`指定sql 语句中的 select 后的查询字段，也可以拼接 where ， groupby 和 having 等复杂语句。`

-  jpa + QueryDSL

- Projection

> 有时候不需要返回所有字段的数据，我们只需要个别字段数据，这样使用 Projection 也是不错的选择,现在的需求是我只需要 Teacher 类对应的表 teacher 中的 name 和 age 的数据，其他数据不需要。 定义一个如下的接口：

```java
public interface TeacherProjection {

    String getName();

    Integer getAge();

    @Value("#{target.name +' and age is' + target.age}")
    String getTotal();
}

public interface TeacherRepositoty extends JpaRepository<Teacher,Integer>, JpaSpecificationExecutor {

   // 返回 TeacherProjection 接口类型的数据
    @Query("select t from Teacher t ")
    List<TeacherProjection> getTeacherNameAndAge();
}
```

- 测试案例

```java
@RunWith(SpringRunner)
@ContextConfiguration("file:src/main/webapp/WEB-INF/applicationContext.xml")
class DaoTest {
    @Autowired
    CommonUserRepository commonUserRepository
    @Test
    void testCrudRepository() {
        User user = new User(username: 'yitian', nickname: '易天', registerTime: LocalDateTime.now())
        commonUserRepository.save(user)
    }
}
```

#### 4. Service

> 这里主要负责复杂的业务处理

#### 5.  Contoller

> 这里主要负责url查询获取相应的结果

```java
@RestController
@RequestMapping(value = "person")
public class PerconController {

    @Autowired
    private PersonRepository personRepository;

    @PostMapping(path = "addPerson")
    public void addPerson(Person person) {
        personRepository.save(person);
    }

    @DeleteMapping(path = "deletePerson")
    public void deletePerson(Long id) {
        personRepository.delete(id);
    }
}
```

#### Resouce

- https://docs.spring.io/spring-data/data-jpa/docs/current/reference/html/#jpa.named-parameters
- https://blog.csdn.net/fly910905/article/details/78557110


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/springbootjpa/  

