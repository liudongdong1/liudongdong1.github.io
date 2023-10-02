# GithubSearch


>- https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories
>
>- https://docs.github.com/cn/github/searching-for-information-on-github/getting-started-with-searching-on-github/understanding-the-search-syntax

#### 0. [查询大于或小于另一个值的值](https://docs.github.com/cn/github/searching-for-information-on-github/getting-started-with-searching-on-github/understanding-the-search-syntax#query-for-values-greater-or-less-than-another-value)

| 查询    | 示例                                                         |
| :------ | :----------------------------------------------------------- |
| `>*n*`  | **[cats stars:>1000](https://github.com/search?utf8=✓&q=cats+stars%3A>1000&type=Repositories)** 匹配含有 "cats" 字样、星标超过 1000 个的仓库。 |
| `>=*n*` | **[cats topics:>=5](https://github.com/search?utf8=✓&q=cats+topics%3A>%3D5&type=Repositories)** 匹配含有 "cats" 字样、有 5 个或更多主题的仓库。 |
| `<*n*`  | **[cats size:<10000](https://github.com/search?utf8=✓&q=cats+size%3A<10000&type=Code)** 匹配小于 10 KB 的文件中含有 "cats" 字样的代码。 |
| `<=*n*` | **[cats stars:<=50](https://github.com/search?utf8=✓&q=cats+stars%3A<%3D50&type=Repositories)** 匹配含有 "cats" 字样、星标不超过 50 个的仓库。 |

| 查询     | 示例                                                         |
| :------- | :----------------------------------------------------------- |
| `*n*..*` | **[cats stars:10..\*](https://github.com/search?utf8=✓&q=cats+stars%3A10..\*&type=Repositories)** 等同于 `stars:>=10` 并匹配含有 "cats" 字样、有 10 个或更多星号的仓库。 |
| `*..*n*` | **[cats stars:\*..10](https://github.com/search?utf8=✓&q=cats+stars%3A"\*..10"&type=Repositories)** 等同于 `stars:<=10` 并匹配含有 "cats" 字样、有不超过 10 个星号的仓库。 |

| 查询                                 | 示例                                                         |
| :----------------------------------- | :----------------------------------------------------------- |
| `>*YYYY*-*MM*-*DD*`                  | **[cats created:>2016-04-29](https://github.com/search?utf8=✓&q=cats+created%3A>2016-04-29&type=Issues)** 匹配含有 "cats" 字样、在 2016 年 4 月 29 日之后创建的议题。 |
| `>=*YYYY*-*MM*-*DD*`                 | **[cats created:>=2017-04-01](https://github.com/search?utf8=✓&q=cats+created%3A>%3D2017-04-01&type=Issues)** 匹配含有 "cats" 字样、在 2017 年 4 月 1 日或之后创建的议题。 |
| `<*YYYY*-*MM*-*DD*`                  | **[cats pushed:<2012-07-05](https://github.com/search?q=cats+pushed%3A<2012-07-05&type=Code&utf8=✓)** 匹配在 2012 年 7 月 5 日之前推送的仓库中含有 "cats" 字样的代码。 |
| `<=*YYYY*-*MM*-*DD*`                 | **[cats created:<=2012-07-04](https://github.com/search?utf8=✓&q=cats+created%3A<%3D2012-07-04&type=Issues)** 匹配含有 "cats" 字样、在 2012 年 7 月 4 日或之前创建的议题。 |
| `*YYYY*-*MM*-*DD*..*YYYY*-*MM*-*DD*` | **[cats pushed:2016-04-30..2016-07-04](https://github.com/search?utf8=✓&q=cats+pushed%3A2016-04-30..2016-07-04&type=Repositories)** 匹配含有 "cats" 字样、在 2016 年 4 月末到 7 月之间推送的仓库。 |
| `*YYYY*-*MM*-*DD*..*`                | **[cats created:2012-04-30..\*](https://github.com/search?utf8=✓&q=cats+created%3A2012-04-30..\*&type=Issues)** 匹配在 2012 年 4 月 30 日之后创建、含有 "cats" 字样的议题。 |
| `*..*YYYY*-*MM*-*DD*`                | **[cats created:\*..2012-07-04](https://github.com/search?utf8=✓&q=cats+created%3A\*..2012-07-04&type=Issues)** 匹配在 2012 年 7 月 4 日之前创建、含有 "cats" 字样的议题。 |

#### 1. [按仓库名称、说明或自述文件内容搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-repository-name-description-or-contents-of-the-readme-file)

| 限定符            | 示例                                                         |
| :---------------- | :----------------------------------------------------------- |
| `in:name`         | [**jquery in:name**](https://github.com/search?q=jquery+in%3Aname&type=Repositories) 匹配仓库名称中含有 "jquery" 的仓库。 |
| `in:description`  | [**jquery in:name,description**](https://github.com/search?q=jquery+in%3Aname%2Cdescription&type=Repositories) 匹配仓库名称或说明中含有 "jquery" 的仓库。 |
| `in:readme`       | [**jquery in:readme**](https://github.com/search?q=jquery+in%3Areadme&type=Repositories) 匹配仓库自述文件中提及 "jquery" 的仓库。 |
| `repo:owner/name` | [**repo:octocat/hello-world**](https://github.com/search?q=repo%3Aoctocat%2Fhello-world) 匹配特定仓库名称。 |

#### 2. [在用户或组织的仓库内搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-within-a-users-or-organizations-repositories)

| 限定符            | 示例                                                         |
| :---------------- | :----------------------------------------------------------- |
| `user:*USERNAME*` | [**user:defunkt forks:>100**](https://github.com/search?q=user%3Adefunkt+forks%3A>%3D100&type=Repositories) 匹配来自 @defunkt、拥有超过 100 复刻的仓库。 |
| `org:*ORGNAME*`   | [**org:github**](https://github.com/search?utf8=✓&q=org%3Agithub&type=Repositories) 匹配来自 GitHub 的仓库。 |

#### 3. [按仓库大小搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-repository-size)

| 限定符     | 示例                                                         |
| :--------- | :----------------------------------------------------------- |
| `size:*n*` | [**size:1000**](https://github.com/search?q=size%3A1000&type=Repositories) 匹配恰好为 1 MB 的仓库。 |
|            | [**size:>=30000**](https://github.com/search?q=size%3A>%3D30000&type=Repositories) 匹配至少为 30 MB 的仓库。 |
|            | [**size:<50**](https://github.com/search?q=size%3A<50&type=Repositories) 匹配小于 50 KB 的仓库。 |
|            | [**size:50..120**](https://github.com/search?q=size%3A50..120&type=Repositories) 匹配介于 50 KB 与 120 KB 之间的仓库。 |

#### 4. [按关注者数量搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-number-of-followers)

| 限定符          | 示例                                                         |
| :-------------- | :----------------------------------------------------------- |
| `followers:*n*` | [**node followers:>=10000**](https://github.com/search?q=node+followers%3A>%3D10000) 匹配有 10,000 或更多关注者提及文字 "node" 的仓库。 |
|                 | [**styleguide linter followers:1..10**](https://github.com/search?q=styleguide+linter+followers%3A1..10&type=Repositories) 匹配拥有 1 到 10 个关注者并且提及 "styleguide linter" 一词的的仓库。 |

#### 5. [按复刻数量搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-number-of-forks)

| 限定符      | 示例                                                         |
| :---------- | :----------------------------------------------------------- |
| `forks:*n*` | [**forks:5**](https://github.com/search?q=forks%3A5&type=Repositories) 匹配只有 5 个复刻的仓库。 |
|             | [**forks:>=205**](https://github.com/search?q=forks%3A>%3D205&type=Repositories) 匹配具有至少 205 个复刻的仓库。 |
|             | [**forks:<90**](https://github.com/search?q=forks%3A<90&type=Repositories) 匹配具有少于 90 个复刻的仓库。 |
|             | [**forks:10..20**](https://github.com/search?q=forks%3A10..20&type=Repositories) 匹配具有 10 到 20 个复刻的仓库。 |

#### 6. [按星号数量搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-number-of-stars)

| 限定符      | 示例                                                         |
| :---------- | :----------------------------------------------------------- |
| `stars:*n*` | [**stars:500**](https://github.com/search?utf8=✓&q=stars%3A500&type=Repositories) 匹配恰好具有 500 个星号的仓库。 |
|             | [**stars:10..20**](https://github.com/search?q=stars%3A10..20+size%3A<1000&type=Repositories) 匹配具有 10 到 20 个星号、小于 1000 KB 的仓库。 |
|             | [**stars:>=500 fork:true language:php**](https://github.com/search?q=stars%3A>%3D500+fork%3Atrue+language%3Aphp&type=Repositories) 匹配具有至少 500 个星号，包括复刻的星号（以 PHP 编写）的仓库。 |

#### 7.  [按仓库创建或上次更新时间搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-when-a-repository-was-created-or-last-updated)

| 限定符                 | 示例                                                         |
| :--------------------- | :----------------------------------------------------------- |
| `created:*YYYY-MM-DD*` | [**webos created:<2011-01-01**](https://github.com/search?q=webos+created%3A<2011-01-01&type=Repositories) 匹配具有 "webos" 字样、在 2011 年之前创建的仓库。 |
| `pushed:*YYYY-MM-DD*`  | [**css pushed:>2013-02-01**](https://github.com/search?utf8=✓&q=css+pushed%3A>2013-02-01&type=Repositories) 匹配具有 "css" 字样、在 2013 年 1 月之后收到推送的仓库。 |
|                        | [**case pushed:>=2013-03-06 fork:only**](https://github.com/search?q=case+pushed%3A>%3D2013-03-06+fork%3Aonly&type=Repositories) 匹配具有 "case" 字样、在 2013 年 3 月 6 日或之后收到推送并且作为复刻的仓库。 |

#### 8. [按语言搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-language)

| 限定符                | 示例                                                         |
| :-------------------- | :----------------------------------------------------------- |
| `language:*LANGUAGE*` | [**rails language:javascript**](https://github.com/search?q=rails+language%3Ajavascript&type=Repositories) 匹配具有 "rails" 字样、以 JavaScript 编写的仓库。 |

#### 9. [按主题数量搜索](https://docs.github.com/cn/github/searching-for-information-on-github/searching-on-github/searching-for-repositories#search-by-number-of-topics)

| 限定符       | 示例                                                         |
| :----------- | :----------------------------------------------------------- |
| `topics:*n*` | [**topics:5**](https://github.com/search?utf8=✓&q=topics%3A5&type=Repositories&ref=searchresults) 匹配具有五个主题的仓库。 |
|              | [**topics:>3**](https://github.com/search?utf8=✓&q=topics%3A>3&type=Repositories&ref=searchresults) 匹配超过三个主题的仓库。 |

#### 10. [大文件上传](https://docs.github.com/cn/github/managing-large-files/versioning-large-files)

> Git LFS 处理大文件的方式是存储对仓库中文件的引用，而不实际文件本身。 为满足 Git 的架构要求，Git LFS 创建了指针文件，用于对实际文件（存储在其他位置）的引用。 GitHub 在仓库中管理此指针文件。 克隆仓库时，GitHub 使用指针文件作为映射来查找大文件。

| 产品                    | 最大文件大小 |
| :---------------------- | :----------- |
| GitHub Free             | 2 GB         |
| GitHub Pro              | 2 GB         |
| GitHub Team             | 4 GB         |
| GitHub Enterprise Cloud | 5 GB         |

- 安装[Git Large File Storage](https://github.com/git-lfs/git-lfs/releases/tag/v2.13.3)， [使用教程](https://git-lfs.github.com/)
- 验证安装： git lfs install



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/githubsearch/  

