# 标准字段


|        字段名         |           类型            |                                                                                      描述                                                                                      |
| --------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **name**              | **string**                | **name**字段应该包含[相对资源名](03-资源命名.md)                                                                                                                               |
| **parent**            | **string**                | 对于资源定义和`List`/`Create`请求，**parent**字段应包含父级[相对资源名](03-资源命名.md)                                                                                        |
| **create\_time**      | **Timestamp**             | 一个实体的创建时间戳                                                                                                                                                           |
| **update\_time**      | **Timestamp**             | 一个实体的最后更新时间戳；注意update_time会被create/patch/delete等操作更新                                                                                                     |
| **delete\_time**      | **Timestamp**             | 实体的删除时间戳，仅当支持保留时。                                                                                                                                             |
| **time\_zone**        | **string**                | 时区名，它应该符合[IANA时区](http://www.iana.org/time-zones)标准，如"America/Los_Angeles"。 有关详细信息，请参阅 https://en.wikipedia.org/wiki/List_of_tz_database_time_zones. |
| **region\_code**      | **string**                | 位置的Unicode国家/地区代码（CLDR），例如“US”和“419”。 有关详细信息，请参阅 http://www.unicode.org/reports/tr35/#unicode_region_subtag。                                        |
| **language\_code**    | **string**                | BCP-47语言代码，如“en-US”或“sr-Latn”。 有关详细信息，请参阅http://www.unicode.org/reports/tr35/#Unicode_locale_identifier。                                                    |
| **display\_name**     | **string**                | 一个实体显示的名称。                                                                                                                                                           |
| **title**             | **string**                | 实体的正式名称，例如公司名称。 它应该被视为正规版本的display\_name                                                                                                             |
| **description**       | **string**                | 一个实体的详细文字描述                                                                                                                                                         |
| **filter**            | **string**                | `List`方法的标准过滤参数                                                                                                                                                       |
| **query**             | **string**                | 应用于`Search`方法的(也就是说 `:search`)**过滤**参数                                                                                                                           |
| **page\_token**       | **string**                | `List`请求的数据分页令牌                                                                                                                                                       |
| **page\_size**        | **int32**                 | `List`请求的数据分页大小                                                                                                                                                       |
| **total\_size**       | **int32**                 | 列表中的总条目数，不考虑分页                                                                                                                                                   |
| **next\_page\_token** | **string**                | `List`返回结果中下一个分页的令牌。它应该在后续请求中传递为page_token参数；空值意味着没有更多数据                                                                               |
| **request\_id**       | **string**                | 用于检测重复请求的唯一字符串id                                                                                                                                                 |
| **resume\_token**     | **string**                | 用于恢复流式传输请求的隐含令牌                                                                                                                                                 |
| **labels**            | **map\<string, string\>** | 表示云资源的标签                                                                                                                                                               |
| **deleted**           | **bool**                  | 如果资源允许取消删除，则它必须有**deleted**字段表示资源是否已被删除                                                                                                            |
| **show\_deleted**     | **bool**                  | 如果资源允许取消删除，相应的`List`方法必须有一个**show_deleted**字段，以便客户端发现已删除的资源。                                                                             |
| **validate_only**     | **bool**                  | 如果为true，则表示给定的请求仅需要被检验，而不是被执行。                                                                                                                       |


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/06-%E6%A0%87%E5%87%86%E5%AD%97%E6%AE%B5/  

