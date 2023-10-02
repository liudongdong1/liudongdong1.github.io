# 时间戳


> 如果需要对`时间字段进行操作(如通过时间范围查找或者排序等)`，推荐使用`bigint`，如果`时间字段不需要进行任何操作，推荐使用timestamp`，使用4个字节保存比较节省空间，但是只能记录到2038年记录的时间有限

```sql
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `time_date` datetime NOT NULL,
  `time_timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `time_long` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `time_long` (`time_long`),
  KEY `time_timestamp` (`time_timestamp`),
  KEY `time_date` (`time_date`)
) ENGINE=InnoDB AUTO_INCREMENT=500003 DEFAULT CHARSET=latin1
```

```java
@Builder
@Data
public class Users {
    /**
     * 自增唯一id
     * */
    private Long id;

    /**
     * date类型的时间
     * */
    private Date timeDate;

    /**
     * timestamp类型的时间
     * */
    private Timestamp timeTimestamp;

    /**
     * long类型的时间
     * */
    private long timeLong;
}
```

```java
@Mapper
public interface UsersMapper {
    @Insert("insert into users(time_date, time_timestamp, time_long) value(#{timeDate}, #{timeTimestamp}, #{timeLong})")
    @Options(useGeneratedKeys = true,keyProperty = "id",keyColumn = "id")
    int saveUsers(Users users);
}
```

```java
public class UsersMapperTest extends BaseTest {
    @Resource
    private UsersMapper usersMapper;

    @Test
    public void test() {
        for (int i = 0; i < 500000; i++) {
            long time = System.currentTimeMillis();
            usersMapper.saveUsers(Users.builder().timeDate(new Date(time)).timeLong(time).timeTimestamp(new Timestamp(time)).build());
        }
    }
}
```
- **sql查询速率测试**

```sql
#通过datetime类型查询： 耗时：0.171
select count(*) from users where time_date >="2018-10-21 23:32:44" and time_date <="2018-10-21 23:41:22"

#通过timestamp类型查询:  耗时：0.351
select count(*) from users where time_timestamp >= "2018-10-21 23:32:44" and time_timestamp <="2018-10-21 23:41:22"

#通过bigint类型查询: 耗时：0.130s
select count(*) from users where time_long >=1540135964091 and time_long <=1540136482372
```

- [sql分组速率测试](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247487551&idx=1&sn=18f64ba49f3f0f9d8be9d1fdef8857d9&scene=21#wechat_redirect)

```sql
#通过datetime类型分组： 耗时：0.176s
select time_date, count(*) from users group by time_date

#通过timestamp类型分组： 耗时：0.173s
select time_timestamp, count(*) from users group by time_timestamp
```

> - 结论 在InnoDB存储引擎下，通过时间分组，性能timestamp > datetime，但是相差不大

- [sql排序速率测试](https://mp.weixin.qq.com/s?__biz=MzUzMTA2NTU2Ng==&mid=2247487551&idx=1&sn=18f64ba49f3f0f9d8be9d1fdef8857d9&scene=21#wechat_redirect)

```sql
#通过datetime类型排序： 耗时：1.038s
select * from users order by time_date

#通过timestamp类型排序： 耗时：0.933s
select * from users order by time_timestamp

#通过bigint类型排序： 耗时：0.775s
select * from users order by time_long
```

> - 结论 在InnoDB存储引擎下，通过时间排序，性能bigint > timestamp > datetime

From: https://mp.weixin.qq.com/s?__biz=MzUxOTc4NjEyMw==&mid=2247512548&idx=3&sn=24a8ab60ae4c4206c02f3d9434c1d569&chksm=f9f6aa00ce812316c4ef247d597842859c4cf6a79da7a409228ad4e36115c83191880a249b53&scene=126&&sessionid=1625275581&subscene=207#rd



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/%E6%97%B6%E9%97%B4%E6%88%B3/  

