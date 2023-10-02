# Annotation_JPA


## JPA基本注解

### @Entity

- @Entity 标注用于实体类声明语句**之前**，**指出该Java 类为实体类，将映射到指定的数据库表**。如声明一个实体类 Custom，它将映射到数据库中的 `Custom` 表上。

**元数据属性说明**：

- `name`： 表名，基本上不使用该属性

```python
@Entity(name=”Custom”)
public class Customer { ... }
```

> 说明：Customer类对应数据库中的Custom表，其中name为可选，**缺省类名即表名**！

### @Table

- **当实体类与其映射的数据库表名不同名时需要使用** @Table 标注说明，该标注与 **@Entity 标注并列使用**，置于实体类声明语句之前，可写于单独语句行，也可与声明语句同行。
- @Table 标注的常用选项是 **name**，用于指明数据库的表名
- @Table标注还有一个两个选项 catalog 和 schema 用于设置表所属的数据库目录或模式，通常为数据库名。uniqueConstraints 选项用于设置约束条件，通常不须设置。

**元数据属性说明**：

- `name`：表名
- `catalog`：对应关系数据库中的catalog
- `schema`：对应关系数据库中的schema
- `UniqueConstraints`：定义一个UniqueConstraint数组，指定需要建唯一约束的列

```java
@Entity
@Table(name="my_customer")
public class Customer { ... }
```

### @UniqueConstraint

- UniqueConstraint定义在Table或SecondaryTable元数据里，用来指定建表时需要建唯一约束的列。

**元数据属性说明**：

- columnNames:定义一个字符串数组，指定设置`唯一约束的列名`

```java
@Entity
@Table(name="EMPLOYEE",
       uniqueConstraints={@UniqueConstraint(columnNames={"EMP_ID", "EMP_NAME"})}
      )
public class Employee { ... }
```

### @Id

- @Id `标注用于声明一个实体类的属性映射为数据库的主键列`
- 该属性通常置于属性声明语句之前，可与声明语句同行，也可写在单独行上。

- id值的获取方式有五种：`TABLE, SEQUENCE, IDENTITY, AUTO, NONE`
  - Oracle和DB2支持SEQUENCE，SQL Server和Sybase支持IDENTITY,MySQL支持AUTO
  - 所有的数据库都可以指定为AUTO，我们会根据不同数据库做转换
  - NONE (默认)需要用户自己指定Id的值
- @Id标注也可置于属性的**getter方法之前**。

**元数据属性说明**：

- `generate`：主键值的获取类型
- `generator`：TableGenerator的名字（当generate=GeneratorType.TABLE才需要指定该属性）

> 关于主键的生成策略有相应的注解

```java
@Entity
@Table(name = "OTASK")
public class Task {
    @Id(generate = GeneratorType.AUTO)
    public Integer getId() {
        return id;
    }
}
```

> 码声明Task的主键列id是自动增长的。(Oracle和DB2从默认的SEQUENCE取值，SQL Server和Sybase该列建成IDENTITY，mysql该列建成auto increment)

### @GeneratedValue

- `@GeneratedValue`  用于标注主键的生成策略，通过 strategy 属性指定。默认情况下，JPA 自动选择一个最适合底层数据库的主键生成策略：SqlServer 对应 identity，MySQL 对应 auto increment。
- 在 javax.persistence.GenerationType中定义了以下几种可供选择的策略：
  - **IDENTITY**：采用数据库 ID自增长的方式来自增主键字段，Oracle 不支持这种方式；
  - **AUTO**： **JPA自动选择合适的策略，是默认选项**；(MySQL使用该项有问题)
  - **SEQUENCE**：通过序列产生主键，通过 @SequenceGenerator 注解指定序列名，MySql 不支持这种方式
  - **TABLE**：通过表产生主键，框架借由表模拟序列产生主键，使用该策略可以使应用更易于数据库移植。

### @Basic

- @Basic 表示一个简单的属性到数据库表的字段的映射,对于**没有任何标注的 getXxxx() 方法,默认即为@Basic**
- `fetch: 表示该属性的读取策略`,有 `EAGER 和 LAZY `两种,分别表示主支抓取和延迟加载,**默认为 EAGER**.
- `optional`:表示`该属性是否允许为null, 默认为true`

### @Column

- **当实体的`属性与其映射的数据库表的列不同名`时需要使用@Column 标注说明**，该属性通常置于实体的属性声明语句之前，还可与 @Id 标注一起使用。
- `@Column 标注的常用属性是 name`，用于设置映射数据库表的列名。此外，该标注还包含其它多个属性，如：`unique 、nullable、length `等。
- @Column 标注的 columnDefinition 属性: **表示该字段在数据库中的实际类型**.通常 ORM 框架可以根据属性类型自动判断数据库中字段的类型,但是`对于Date类型仍无法确定数据库中字段类型究竟是DATE,TIME还是TIMESTAMP`.此外,`String的默认映射类型为VARCHAR`, 如果要将 String 类型映射到特定数据库的 BLOB 或TEXT 字段类型.
- @Column标注也可置于属性的getter方法之前

元数据属性说明：

- `name`：列名。
- `unique`： 是否唯一
- `nullable`： 是否允许为空
- `insertable`: 是否允许插入
- `updatable`：是否允许更新
- `columnDefinition`：定义建表时创建此列的DDL
- `secondaryTable`：从表名。如果此列不建在主表上（默认建在主表），该属性定义该列所在从表的名字。

```java
public class Person {
    @Column(name = "PERSONNAME", unique = true, nullable = false, updatable = true)
    private String password;

    @Column(name = "PHOTO", columnDefinition = "BLOB NOT NULL", secondaryTable="PER_PHOTO")
    private byte[] picture;
}
```

### @Transient 

- 表示`该属性并非一个到数据库表的字段的映射,ORM框架将忽略该属性`.
- 如果一个属性并非数据库表的字段映射,就务必将其标示为`@Transient`,否则,ORM框架默认其注解为`@Basic`

```java
@Transient
private String name;
```

### @Temporal

在核心的 Java API 中并没有定义 Date 类型的精度(temporal precision).  而在数据库中,表示 Date 类型的数据有 DATE, TIME, 和 TIMESTAMP 三种精度(即单纯的`日期,时间,或者两者 兼备`).**在进行属性映射时可使用@Temporal注解来调整精度**.

| 参数  | 类型         | 描述                                                         |
| ----- | ------------ | ------------------------------------------------------------ |
| value | TemporalType | 存储的类型，可选值：  `TemporalType.DATE（日期）  TemporalType.TIME（时间）  TemporalType.TIMESTAMP（日期和时间）` |

```java
@Entity(name = "user")
public class User implements Serializable { 
    @Id
    @GeneratedValue
    private Long id;
    
    @Temporal(TemporalType.DATE)
    private Date birthday;
    
    @Temporal(TemporalType.TIMESTAMP)
    private Date lastLoginTime;
    
    @Temporal(TemporalType.TIME)
    private Date tokenExpiredTime;
    
    // getters and setters
    
}
```

## 关联关系注解

### @OneToOne

`@OneToOne`：描述一个 一对一的关联

**元数据属性说明**：

- fetch：表示抓取策略，**默认为FetchType.LAZY**，延迟加载
- cascade：表示级联操作策略

### @ManyToOne

`@ManyToOne`：表示一个多对一的映射,**该注解标注的属性通常是数据库表的外键**

元数据属性说明：

- `optional`：是否允许该字段为null，该属性应该根据数据库表的外键约束来确定，**默认为true**
- `fetch`：表示抓取策略，**默认为FetchType.EAGER**
- `cascade`：表示默认的级联操作策略，可以指定为**ALL，PERSIST，MERGE，REFRESH和REMOVE**中的若干组合，默认为**无级联操作**
- `targetEntity`：表示该属性关联的实体类型。**该属性通常不必指定**，ORM框架根据属性类型自动判断targetEntity。

### @OneToMany

`@OneToMany`：描述一个 一对多的关联,该属性应该为集体类型,**在数据库中并没有实际字段**。

**元数据属性说明**：

- `fetch`：表示抓取策略,默认为**FetchType.LAZY**,因为关联的多个对象通常不必从数据库预先读取到内存
- `cascade`：表示级联操作策略,对于OneToMany类型的关联非常重要,通常该实体更新或删除时,其关联的实体也应当被更新或删除

### @ManyToMany

`@ManyToMany`：描述一个多对多的关联.**多对多关联上是两个一对多关联**,但是在ManyToMany描述中,中间表是由ORM框架自动处理

**元数据属性说明**：

- `targetEntity`：表示多对多关联的另一个实体类的全名,例如:package.Book.class
- `mappedBy`：表示多对多关联的另一个实体类的对应集合属性名称，控制权力的转移

> 两个实体间相互关联的属性必须标记为@ManyToMany,并相互指定targetEntity属性, 需要注意的是,有且只有一个实体的@ManyToMany注解需要指定mappedBy属性,指向targetEntity的集合属性名称 利用ORM工具自动生成的表除了User和Book表外,还自动生成了一个User_Book表,用于实现多对多关联

### @JoinTable

JoinTable在many-to-many关系的所有者一边定义。如果没有定义JoinTable，使用JoinTable的默认值

**元数据属性说明**：

- `table`：这个join table的Table定义。
- `joinColumns`：定义指向所有者主表的外键列，数据类型是JoinColumn数组。
- `inverseJoinColumns`：定义指向非所有者主表的外键列，数据类型是JoinColumn数组。

```
@JoinTable(
    table=@Table(name=CUST_PHONE),
    joinColumns=@JoinColumn(name="CUST_ID", referencedColumnName="ID"),
    inverseJoinColumns=@JoinColumn(name="PHONE_ID", referencedColumnName="ID")
)
复制代码
```

> 定义了一个连接表CUST和PHONE的join table。join table的表名是CUST_PHONE，包含两个外键，一个外键是CUST_ID，指向表CUST的主键ID，另一个外键是PHONE_ID，指向表PHONE的主键ID。

### @JoinColumn

如果在entity class的field上定义了关系（**one2one或one2many**等），我们通过JoinColumn来定义关系的属性。JoinColumn的大部分属性和Column类似。

**元数据属性说明**：

- `name`：列名。
- `referencedColumnName`：该列指向列的列名（建表时该列作为外键列指向关系另一端的指定列）
- `unique`： 是否唯一
- `nullable`：是否允许为空
- `insertable`：是否允许插入
- `updatable`：是否允许更新
- `columnDefinition`：定义建表时创建此列的DDL
- `secondaryTable`：从表名。如果此列不建在主表上（默认建在主表），该属性定义该列所在从表的名字

```
public class Custom {
    @OneToOne
    @JoinColumn(
        name="CUST_ID", referencedColumnName="ID", unique=true, nullable=true, updatable=true)
    public order getOrder() {
        return order;
    }

复制代码
```

> Custom和Order是一对一关系。在Order对应的映射表建一个名为CUST_ID的列，该列作为外键指向Custom对应表中名为ID的列

### @JoinColumns

如果在entity class的field上定义了关系（one2one或one2many等），并且关系存在多个JoinColumn，用JoinColumns定义多个JoinColumn的属性

**元数据属性说明**：

- `value`： 定义JoinColumn数组，指定每个JoinColumn的属性

```
public class Custom {
    @OneToOne
    @JoinColumns({
        @JoinColumn(name="CUST_ID", referencedColumnName="ID"),
        @JoinColumn(name="CUST_NAME", referencedColumnName="NAME")
    })
    public order getOrder() {
        return order;
    }

复制代码
```

> Custom和Order是一对一关系。在Order对应的映射表建两列，一列名为CUST_ID，该列作为外键指向Custom对应表中名为ID的列,另一列名为CUST_NAME，该列作为外键指向Custom对应表中名为NAME的列

### @OrderBy

在一对多，多对多关系中，有时我们希望从数据库加载出来的集合对象是按一定方式排序的，这可以通过OrderBy来实现，默认是按对象的主键升序排列。

**元数据属性说明**：

- `value`：字符串类型，指定排序方式。格式为"fieldName1 [ASC|DESC],fieldName2 [ASC|DESC],…",排序类型可以不指定，默认是ASC。

```
@Table(name = "MAPKEY_PERSON")
public class Person {
    @OneToMany(targetEntity = Book.class, cascade = CascadeType.ALL, mappedBy = "person")
    @OrderBy(name = "isbn ASC, name DESC")
    private List books = new ArrayList();
}
复制代码
```

> Person和Book之间是一对多关系。集合books按照Book的isbn升序，name降序排列

## 其他注解介绍

### @SecondaryTable

- 一个entity class可以**映射到多表**，SecondaryTable用来定义单个从表的名字，主键名字等属性。

**元数据属性说明**：

- `name`：表名
- `catalog`： 对应关系数据库中的catalog
- `schema`：对应关系数据库中的schema
- `pkJoin`： 定义一个PrimaryKeyJoinColumn数组，指定从表的主键列
- `UniqueConstraints`：定义一个UniqueConstraint数组，指定需要建唯一约束的列

```java
@Entity
@Table(name="CUSTOMER")
@SecondaryTable(name="CUST_DETAIL",pkJoin=@PrimaryKeyJoinColumn(name="CUST_ID"))
public class Customer { ... }
```

> 代码说明Customer类映射到两个表，主表名是CUSTOMER，从表名是CUST_DETAIL，从表的主键列和主表的主键列类型相同，列名为CUST_ID

### @SecondaryTables

- 当一个entity class映射到一个主表和多个从表时，用SecondaryTables来定义各个从表的属性。

**元数据属性说明**：

- value： 定义一个SecondaryTable数组，指定每个从表的属性

```java
@Table(name = "CUSTOMER")
@SecondaryTables( value = {
    @SecondaryTable(name = "CUST_NAME", pkJoin = { @PrimaryKeyJoinColumn(name = "STMO_ID", referencedColumnName = "id") }),
    @SecondaryTable(name = "CUST_ADDRESS", pkJoin = { @PrimaryKeyJoinColumn(name = "STMO_ID", referencedColumnName = "id") }) })
public class Customer {}
```

### @IdClass

当entity class`使用复合主键时`，需要定义一个类作为id class。 id class必须符合以下要求:

- 类必须声明为public，并提供一个声明为public的空构造函数。
- 必须实现Serializable接口，覆写 equals() 和 hashCode()方法。
- entity class的所有id field在id class都要定义，且类型一样。

**元数据属性说明**：

- `value`： id class的类名

```java
public class EmployeePK implements java.io.Serializable{
    String empName;
    Integer empAge;

    public EmployeePK(){}

    public boolean equals(Object obj){ ......}
    public int hashCode(){......}
}
@IdClass(value=com.acme.EmployeePK.class)
@Entity(access=FIELD)
public class Employee {
    @Id String empName;
    @Id Integer empAge;
}
```

### @MapKey

- `在一对多，多对多关系中，我们可以用Map来保存集合对象`。默认用主键值做key，如果使用复合主键，`则用id class的实例做key，如果指定了name属性，就用指定的field的值做key。`

**元数据属性说明**：

- `name`： 用来做key的field名

```java
@Table(name = "PERSON")
public class Person {

    @OneToMany(targetEntity = Book.class, cascade = CascadeType.ALL, mappedBy = "person")
    @MapKey(name = "isbn")
    private Map books = new HashMap();
}
```

> Person和Book之间是一对多关系。Person的books字段是Map类型，用Book的isbn字段的值作为Map的key

### @PrimaryKeyJoinColumn

在三种情况下会用到PrimaryKeyJoinColumn。

- 继承。
- entity class映射到一个或多个从表。从表根据主表的主键列（列名为referencedColumnName值的列），建立一个类型一样的主键列，列名由name属性定义。
- one2one关系，关系维护端的主键作为外键指向关系被维护端的主键，不再新建一个外键列。

**元数据属性说明**：

- `name`：列名。
- `referencedColumnName`：该列引用列的列名
- `columnDefinition`：定义建表时创建此列的DDL

```java
@Entity
@Table(name="CUSTOMER")
@SecondaryTable(name="CUST_DETAIL",pkJoin=@PrimaryKeyJoinColumn(name="CUST_ID"，referencedColumnName="id"))
public class Customer { 
    @Id(generate = GeneratorType.AUTO)
    public Integer getId() {
        return id;
    }
}
```

> Customer映射到两个表，主表CUSTOMER,从表CUST_DETAIL，从表需要建立主键列CUST_ID，该列和主表的主键列id除了列名不同，其他定义一样

```
@Table(name = "Employee")
public class Employee {
    @OneToOne
    @PrimaryKeyJoinColumn(name = "id", referencedColumnName="INFO_ID")
    EmployeeInfo info;
}
复制代码
```

> Employee和EmployeeInfo是一对一关系，Employee的主键列id作为外键指向EmployeeInfo的主键列INFO_ID

### @PrimaryKeyJoinColumns

如果entity class使用了复合主键，指定单个PrimaryKeyJoinColumn不能满足要求时，可以用PrimaryKeyJoinColumns来定义多个PrimaryKeyJoinColumn。

**元数据属性说明**：

- `value`： 一个PrimaryKeyJoinColumn数组，包含所有PrimaryKeyJoinColumn

```
@Entity
@IdClass(EmpPK.class)
@Table(name = "EMPLOYEE")
public class Employee {

    private int id;

    private String name;

    private String address;

    @OneToOne(cascade = CascadeType.ALL)
    @PrimaryKeyJoinColumns({
        @PrimaryKeyJoinColumn(name="id", referencedColumnName="INFO_ID"),
        @PrimaryKeyJoinColumn(name="name" , referencedColumnName="INFO_NAME")})
    EmployeeInfo info;
}

@Entity
@IdClass(EmpPK.class)
@Table(name = "EMPLOYEE_INFO")
public class EmployeeInfo {

    @Id
    @Column(name = "INFO_ID")
    private int id;

    @Id
    @Column(name = "INFO_NAME")
    private String name;
}
复制代码
```

### @Version

- Version指定实体类在乐观事务中的version属性。在实体类重新由`EntityManager管理并且加入到乐观事务中时，保证完整性`。每一个类只能有一个属性被指定为version，version属性应该映射到实体类的主表上

```java
@Version
@Column("OPTLOCK")
protected int getVersionNum() { return versionNum; }
```

> versionNum属性作为这个类的version，映射到数据库中主表的列名是OPTLOCK

### @Lob

- Lob指定一个属性作为数据库支持的大对象类型在数据库中存储。使用`LobType这个枚举来定义Lob是二进制类型还是字符类型`

**LobType枚举类型说明**：

- BLOB 二进制大对象，`Byte[]或者Serializable的类型`可以指定为BLOB。
- CLOB 字符型大对象，`char[]、Character[]或String类型`可以指定为CLOB

**元数据属性说明**：

- `fetch`： 定义这个字段是lazy loaded还是eagerly fetched。数据类型是FetchType枚举，默认为LAZY,即lazy loaded.
- `type`： 定义这个字段在数据库中的JDBC数据类型。数据类型是LobType枚举，默认为BLOB

```java
@Lob
@Column(name="PHOTO" columnDefinition="BLOB NOT NULL")
protected JPEGImage picture;

@Lob(fetch=EAGER, type=CLOB)
@Column(name="REPORT")
protected String report;   
```

### @TableGenerator

TableGenerator定义一个主键值生成器，在Id这个元数据的generate＝TABLE时，generator属性中可以使用生成器的名字。生成器可以在类、方法或者属性上定义。 生成器是为多个实体类提供连续的ID值的表，每一行为一个类提供ID值，ID值通常是整数。

**元数据属性说明**：

- `name`：生成器的唯一名字，可以被Id元数据使用。
- `table`：生成器用来存储id值的Table定义。
- `pkColumnName`：生成器表的主键名称。
- `valueColumnName`：生成器表的ID值的列名称。
- `pkColumnValue`：生成器表中的一行数据的主键值。
- `initialValue`：id值的初始值。
- `allocationSize`：id值的增量。

```java
@Entity
public class Employee {
    ...
        @TableGenerator(name="empGen",
                        table=@Table(name="ID_GEN"),
                        pkColumnName="GEN_KEY",
                        valueColumnName="GEN_VALUE",
                        pkColumnValue="EMP_ID",
                        allocationSize=1)
        @Id(generate=TABLE, generator="empGen")
        public int id;
    ...
}

@Entity 
public class Address {
    ...
        @TableGenerator(name="addressGen",
                        table=@Table(name="ID_GEN"),
                        pkColumnValue="ADDR_ID")
        @Id(generate=TABLE, generator="addressGen")
        public int id;
    ...
}
```

### @SequenceGenerator

SequenceGenerator定义一个主键值生成器，在Id这个元数据的generator属性中可以使用生成器的名字。生成器可以在类、方法或者属性上定义。生成器是数据库支持的sequence对象。

**元数据属性说明**：

- `name`：生成器的唯一名字，可以被Id元数据使用。
- `sequenceName`：数据库中，sequence对象的名称。如果不指定，会使用提供商指定的默认名称。
- `initialValue:id`：值的初始值。
- `allocationSize`：id值的增量。

```java
@SequenceGenerator(name="EMP_SEQ", allocationSize=25)	
```

> 定义了一个使用提供商默认名称的sequence生成器

### @DiscriminatorColumn

DiscriminatorColumn定义在使用SINGLE_TABLE或JOINED继承策略的表中区别不继承层次的列

**元数据属性说明**：

- name:column的名字。默认值为TYPE。
- columnDefinition:生成DDL的sql片断。
- length:String类型的column的长度，其他类型使用默认值10

```java
@Entity
@Table(name="CUST")
@Inheritance(strategy=SINGLE_TABLE,
             discriminatorType=STRING,
             discriminatorValue="CUSTOMER")
@DiscriminatorColumn(name="DISC", length=20)
public class Customer { ... }
```

## Resouce

- https://juejin.cn/post/6844904037859459085

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/annotation_jpa/  

