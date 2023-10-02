# HashSet


- From: https://juejin.cn/post/6870649688388730893

## 集群中普通的Hash算法 - 前世

> 这里先使用最简单的思路去看待集群中的路由访问问题

首先我们这里有三个客户端，以及有三个服务器。客户端首先访问的是一台负载均衡服务器，负载服务器不直接处理业务，而是把请求通过一定的算法，转交给内部的N台机器中的一台。 这里假设使用的是一个简单的计算方法：`机器编号=hash(ip) % 节点数量` 这里的节点数量配置为**3**

### 集群中使用简单的Hash算法

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/99d205ed54b5473d928168ad013977a0~tplv-k3u1fbpfcp-zoom-1.image) 这个时候请求机器和对应服务器的访问关系是

| 客户端   | 服务器   |
| -------- | -------- |
| 客户端 1 | Tomcat 2 |
| 客户端 2 | Tomcat 1 |
| 客户端 3 | Tomcat 3 |

### 集群中机器扩容，缩容，宕机

一切看起来都那么的美好，直到有台机器蹦了。这里假定是Tomcat3挂了 

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/ebebb8d2d87d4820b27d2f01cdaf6724~tplv-k3u1fbpfcp-zoom-1.image) 当这种情况发生的时候，负载均衡服务器，监测到了Tomcat3挂掉了，然后就调整自己的算法，`机器编号=hash(ip) % 节点数量` 这里的节点数量配置为2。这样虽然也能继续负载均衡功能，但是客户端和服务器之间的映射关系已经发生了比较大的变化了。新的映射关系已经变成这样了。

| 客户端   | 服务器   |
| -------- | -------- |
| 客户端 1 | Tomcat 1 |
| 客户端 2 | Tomcat 2 |
| 客户端 3 | Tomcat 1 |

### 集群中Hash算法所产生的问题

在真实的情况下，会拥有很多台的服务器，那么影响会很大的，`在缩容和扩容的情况下`，也会存在相同的问题。`原来用户在服务器中的会话都会丢失`。

## 一致性Hash算法 - 今世

> 目前需要解决的一个问题是：如何`在机器扩容，机器缩容，机器宕机的情况下，让受影响的用户量最少`

### 什么是一致性Hash算法？

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/666f29ac3aa3436789ef690233a94700~tplv-k3u1fbpfcp-zoom-1.image)

- **hash环**

⾸先有⼀条直线，直线开头和结尾分别定为为1和2的32次⽅减1，这相当于⼀个地址，对于这样⼀条 线，弯过来构成⼀个圆环形成闭环，这样的⼀个圆环称为hash环。

- **使用hash环**

我们把服务器的`ip或者主机名求hash值然后对应到hash环上`，那么针对客户端⽤户，也根据它的ip进⾏hash求值，对应到环上某个位 置

- **确定客户端的路由**

如何确定⼀个客户端路由到哪个服务器处理呢？ `按照逆时针⽅向找最近的服务器节点 `

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/5f7a0ca2745540ee891406d051df9571~tplv-k3u1fbpfcp-zoom-1.image) 举个例

### 一致性Hash算法之机器扩容

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/8cf985862d044cd19befb020635d6418~tplv-k3u1fbpfcp-zoom-1.image) 

这个时候，我们在增加了一台服务器，IP经过Hash之后，落在了2,3节点之间，变成了节点5。 那么此时，只有浏览器Ip Hash值落在 3-5之间，5-2之间的才收到影响，其他的所有的用户根本就不会收到影响。

### 一致性Hash算法之机器缩容

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/946914649c804e5e957ed10b63f3a6c6~tplv-k3u1fbpfcp-zoom-1.image) 当我们机器3蹦了之后，原本浏览器Ip Hash值值在 2-3之间，3-4之间的。直接转移到顺时针的第一台机器，也就是节点4。此时其他的节点之间的机器访问地址不受影响

### 一致性Hash算法之虚拟节点

#### 数据倾斜问题

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/24163450b27749288cdcaf3437f57c1c~tplv-k3u1fbpfcp-zoom-1.image) 

但是，⼀致性哈希算法在服务节点太少时，容易因为节点分部不均匀⽽造成数据倾斜问题。例如系统中只有两台服务器，其环分布如下，节点2只能负责⾮常⼩的⼀段，⼤量的客户端 请求落在了节点1上，这就是数据（请求)倾斜问题

#### 虚拟节点

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/07f33b5d9f0247f0b61c95348df83971~tplv-k3u1fbpfcp-zoom-1.image)

为了解决这种数据倾斜问题，⼀致性哈希算法引⼊了虚拟节点机制，即`对每⼀个服务节点计算多个哈希`，每个计算结果位置都放置⼀个此服务节点，称为`虚拟节点`。

具体做法可以在服务器ip或主机名的后⾯`增加编号`来实现。⽐如，可以为每台服务器计算三个虚拟节点，于是可以分别计算每台机器的Hash值

- “节点1的ip#1”
- “节点1的ip#2”
- “节点1的ip#3”
- “节点2的ip#1”
- “节点2的ip#2”
- “节点2的ip#3”

于是形成六个虚拟节点，这个6个虚拟节点，在环上分配得更加均匀了，当客户端被路由到虚拟节点时，系统找出这个虚拟节点对应的真实节点。这样每个浏览器IP进行访问的时候，也能相对均匀的分配到两台真实的服务器中。

## 仿写一致性Hash算法

### 普通Hash实现方案

```java
public class GeneralHash {

    public static void main(String[] args) {
        // 定义客户端IP
        String[] clients = new String[]{"10.78.12.3","113.25.63.1","126.12.3.8"};

        // 定义服务器数量
        int serverCount = 5;// (编号对应0，1，2)

        // hash(ip)%node_counts=index
        //根据index锁定应该路由到的tomcat服务器
        for(String client: clients) {
            int hash = Math.abs(client.hashCode());
            int index = hash%serverCount;
            System.out.println("客户端：" + client + " 被路由到服务器编号为：" + index);

        }
    }
}
```

```
客户端：10.78.12.3 被路由到服务器编号为：4
客户端：113.25.63.1 被路由到服务器编号为：0
客户端：126.12.3.8 被路由到服务器编号为：1
```

### 一致性Hash之没有虚拟节点

1. 初始化时，使用一个有序的Map来存放节点IPHash值和对应的IP值

2. 针对客户端求出IPHash值

3. 根据IP Hash值在Hash环上进行取值

   3.1 IP的Hash值不在map中间 --> 取环上的第一个值

   3.2 IP的Hash值在map中间 --> 取离自己最近的最大值

```java
public class ConsistentHashNoVirtual {

    public static void main(String[] args) {
        //step1 初始化：把服务器节点IP的哈希值对应到哈希环上
        // 定义服务器ip
        String[] tomcatServers = new String[]{"123.111.0.0","123.101.3.1","111.20.35.2","123.98.26.3"};

        //我们这里使用能够排序的Map
        SortedMap<Integer,String> hashServerMap = new TreeMap<>();

        for(String tomcatServer: tomcatServers) {
            // 求出每一个ip的hash值，对应到hash环上，存储hash值与ip的对应关系
            int serverHash = Math.abs(tomcatServer.hashCode());
            // 存储hash值与ip的对应关系
            hashServerMap.put(serverHash,tomcatServer);
        }

        //step2 针对客户端IP求出hash值
        // 定义客户端IP
        String[] clients = new String[]{"10.78.12.3","113.25.63.1","126.12.3.8"};

        for(String client : clients) {
            int clientHash = Math.abs(client.hashCode());

            //step3 针对客户端,找到能够处理当前客户端请求的服务器（哈希环上顺时针最近）
            // 根据客户端ip的哈希值去找出哪一个服务器节点能够处理（）

            // 这里最妙的就是这个tailMap的用法，向上取值
            SortedMap<Integer, String> integerStringSortedMap = hashServerMap.tailMap(clientHash);
 
            if(integerStringSortedMap.isEmpty()) {
                // 取哈希环上的顺时针第一台服务器
                Integer firstKey = hashServerMap.firstKey();
                System.out.println("==========>>>>客户端：" + client + " 被路由到服务器：" + hashServerMap.get(firstKey));
            }else{
                Integer firstKey = integerStringSortedMap.firstKey();
                System.out.println("==========>>>>客户端：" + client + " 被路由到服务器：" + hashServerMap.get(firstKey));
            }
        }
    }
}
```

```
==========>>>>客户端：10.78.12.3 被路由到服务器：111.20.35.2
==========>>>>客户端：113.25.63.1 被路由到服务器：123.98.26.3
==========>>>>客户端：126.12.3.8 被路由到服务器：111.20.35.2
```

### 一致性Hash之使用虚拟节点

> 在使用虚拟节点的情况下，我们只需要在初始化的时候,针对每个节点，都在IP后面增加 `#+编号`的方式生成虚拟节点

```
for(String tomcatServer: tomcatServers) {
  // 求出每一个ip的hash值，对应到hash环上，存储hash值与ip的对应关系
  int serverHash = Math.abs(tomcatServer.hashCode());
  // 存储hash值与ip的对应关系
  hashServerMap.put(serverHash,tomcatServer);

  // 处理虚拟节点
  for(int i = 0; i < virtaulCount; i++) {
    int virtualHash = Math.abs((tomcatServer + "#" + i).hashCode());
    hashServerMap.put(virtualHash,"----由虚拟节点"+ i  + "映射过来的请求："+ tomcatServer);
  }
}
```

完整代码如下：

```java
public class ConsistentHashWithVirtual {

    public static void main(String[] args) {
        //step1 初始化：把服务器节点IP的哈希值对应到哈希环上
        // 定义服务器ip
        String[] tomcatServers = new String[]{"123.111.0.0","123.101.3.1","111.20.35.2","123.98.26.3"};

        SortedMap<Integer,String> hashServerMap = new TreeMap<>();


        // 定义针对每个真实服务器虚拟出来几个节点
        int virtaulCount = 3;


        for(String tomcatServer: tomcatServers) {
            // 求出每一个ip的hash值，对应到hash环上，存储hash值与ip的对应关系
            int serverHash = Math.abs(tomcatServer.hashCode());
            // 存储hash值与ip的对应关系
            hashServerMap.put(serverHash,tomcatServer);

            // 处理虚拟节点
            for(int i = 0; i < virtaulCount; i++) {
                int virtualHash = Math.abs((tomcatServer + "#" + i).hashCode());
                hashServerMap.put(virtualHash,"----由虚拟节点"+ i  + "映射过来的请求："+ tomcatServer);
            }

        }


        //step2 针对客户端IP求出hash值
        // 定义客户端IP
        String[] clients = new String[]{"10.78.12.3","113.25.63.1","126.12.3.8"};
        for(String client : clients) {
            int clientHash = Math.abs(client.hashCode());
            //step3 针对客户端,找到能够处理当前客户端请求的服务器（哈希环上顺时针最近）
            // 根据客户端ip的哈希值去找出哪一个服务器节点能够处理（）
            SortedMap<Integer, String> integerStringSortedMap = hashServerMap.tailMap(clientHash);
            if(integerStringSortedMap.isEmpty()) {
                // 取哈希环上的顺时针第一台服务器
                Integer firstKey = hashServerMap.firstKey();
                System.out.println("==========>>>>客户端：" + client + " 被路由到服务器：" + hashServerMap.get(firstKey));
            }else{
                Integer firstKey = integerStringSortedMap.firstKey();
                System.out.println("==========>>>>客户端：" + client + " 被路由到服务器：" + hashServerMap.get(firstKey));
            }
        }
    }
}
```

输出结果：

```
==========>>>>客户端：10.78.12.3 被路由到服务器：111.20.35.2
==========>>>>客户端：113.25.63.1 被路由到服务器：----由虚拟节点2映射过来的请求：111.20.35.2
==========>>>>客户端：126.12.3.8 被路由到服务器：----由虚拟节点0映射过来的请求：123.101.3.1
```

## Nginx中的中使用一致性Hash算法

> `ngx_http_upstream_consistent_hash 模块是⼀个负载均衡器`，使⽤⼀个内部⼀致性hash算法来选择合适的后端节点。

该模块可以根据配置参数采取不同的⽅式将请求均匀映射到后端机器，

- consistent_hash $remote_addr：可以根据客户端`ip映射`
- consistent_hash $request_uri：根据客户端请求的`uri映射`
- consistent_hash $args：根据客户端携带的`参数进⾏映`

### 单独安装

ngx_http_upstream_consistent_hash 模块是⼀个第三⽅模块，需要我们下载安装后使⽤

1. github下载nginx⼀致性hash负载均衡模块 [github.com/replay/ngx_…](https://link.juejin.cn/?target=https%3A%2F%2Fgithub.com%2Freplay%2Fngx_http_consistent_hash)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/c17af46c79a1425eaed538531c5bf610~tplv-k3u1fbpfcp-zoom-1.image)

1. 将下载的压缩包上传到nginx服务器，并解压
2. 我们已经编译安装过nginx，此时进⼊当时nginx的源码⽬录，执⾏如下命令

```shell
./configure —add-module=/root/ngx_http_consistent_hash-master
make
make install
复制代码
```

nginx.conf⽂件中配置 负载均衡策略

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/641e24765325463fadc91e3f26443771~tplv-k3u1fbpfcp-zoom-1.image)



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/hashset/  

