# Kubernetes-yaml详解


|  属性名称  |    介绍    |
| :--------: | :--------: |
| apiVersion |  API 版本  |
|    kind    |  资源类型  |
|  metadata  | 资源元数据 |
|    spec    |  资源规格  |
|  replicas  |  副本数量  |
|  selector  | 标签选择器 |
|  template  |  Pod 模板  |
|  metadata  | Pod 元数据 |
|    spec    |  Pod 规格  |
| containers |  容器配置  |

### 1. 核心技术概念

#### .1. 复制控制器（Replication Controller，RC）

- RC是K8s集群中保证Pod高可用的API对象。通过监控运行中的Pod来保证集群中运行指定数目的Pod副本。指定的数目可以是多个也可以是1个；少于指定数目，RC就会启动运行新的Pod副本；
- 多于指定数目，RC就会杀死多余的Pod副本。即使在指定数目为1的情况下，`通过RC运行Pod也比直接运行Pod更明智，因为RC也可以发挥它高可用的能力`，保证永远有1个Pod在运行。
- RC是K8s较早期的技术概念，只适用于长期伺服型的业务类型，比如控制小机器人提供高可用的Web服务。

#### .2. 副本集（Replica Set，RS）

- RS是新一代RC，提供同样的高可用能力，区别主要在于RS后来居上，能支持更多种类的匹配模式。副本集对象一般不单独使用，而是作为Deployment的理想状态参数使用。

#### .3. 部署(Deployment)

- 部署表示用户对K8s集群的`一次更新操作`。部署是一个比RS应用模式更广的API对象，可以是`创建一个新的服务`，`更新一个新的服务`，也可以是`滚动升级一个服务`。
- 滚动升级一个服务，实际是创建一个新的RS，然后逐渐将新RS中副本数增加到理想状态，将旧RS中的副本数减小到0的复合操作；
- 这样一个复合操作用一个RS是不太好描述的，所以用一个更通用的Deployment来描述。

#### .4. 服务（Service）

-  `RC、RS和Deployment只是保证了支撑服务的微服务Pod的数量`，但是没有解决如何访问这些服务的问题。
- `一个Pod只是一个运行服务的实例，随时可能在一个节点上停止`，在另一个节点以一个新的IP启动一个新的Pod，因此不能以确定的IP和端口号提供服务。
- 要稳定地提供服务需要服务发现和负载均衡能力。服务发现完成的工作，是针对客户端访问的服务，找到对应的的后端服务实例。在K8s集群中，客户端需要访问的服务就是Service对象。
- 每个Service会对应一个集群内部有效的虚拟IP，集群内部通过虚拟IP访问一个服务。
- 在K8s集群中微服务的负载均衡是由Kube-proxy实现的。Kube-proxy是K8s集群内部的负载均衡器。
- 它是一个分布式代理服务器，在K8s的每个节点上都有一个；这一设计体现了它的伸缩性优势，需要访问服务的节点越多，提供负载均衡能力的Kube-proxy就越多，高可用节点也随之增多。
- 与之相比，我们平时在服务器端做个反向代理做负载均衡，还要进一步解决反向代理的负载均衡和高可用问题。

#### .5. 任务（Job）

- Job是K8s用来`控制批处理型任务的API对象`。批处理业务与长期伺服业务的主要区别是批处理业务的运行有头有尾，而长期伺服业务在用户不停止的情况下永远运行。
- Job管理的Pod根据用户的设置把任务成功完成就自动退出了。
- 成功完成的标志根据不同的spec.completions策略而不同：单Pod型任务有一个Pod成功就标志完成；定数成功型任务保证有N个任务全部成功；工作队列型任务根据应用确认的全局成功而标志成功。

#### .6. 后台支撑服务集（DaemonSet）

- 长期伺服型和批处理型服务的核心在业务应用，可能有些节点运行多个同类业务的Pod，
- 有些节点上又没有这类Pod运行；而后台支撑型服务的核心关注点在K8s集群中的节点（物理机或虚拟机），要保证每个节点上都有一个此类Pod运行。
- 节点可能是所有集群节点也可能是通过nodeSelector选定的一些特定节点。典型的后台支撑型服务包括，存储，日志和监控等在每个节点上支持K8s集群运行的服务。

#### .7. 存储卷（Volume）

- K8s集群中的存储卷跟Docker的存储卷有些类似，只不过Docker的存储卷作用范围为一个容器，而`K8s的存储卷的生命周期和作用范围是一个Pod`。`每个Pod中声明的存储卷由Pod中的所有容器共享`。K8s支持非常多的存储卷类型，特别的，支持多种公有云平台的存储，包括AWS，Google和Azure云；支持多种分布式存储包括GlusterFS和Ceph；也支持较容易使用的主机本地目录hostPath和NFS。K8s还支持使用Persistent Volume Claim即PVC这种逻辑存储，使用这种存储，使得存储的使用者可以忽略后台的实际存储技术（例如AWS，Google或GlusterFS和Ceph），而将有关存储实际技术的配置交给存储管理员通过Persistent Volume来配置。

- `持久存储卷（Persistent Volume，PV）`和`持久存储卷声明（Persistent Volume Claim，PVC）`
- PV和PVC使得K8s集群具备了存储的逻辑抽象能力，使得在配置Pod的逻辑里可以忽略对实际后台存储技术的配置，而把这项配置的工作交给PV的配置者，即集群的管理者。存储的PV和PVC的这种关系，跟计算的Node和Pod的关系是非常类似的；PV和Node是资源的提供者，根据集群的基础设施变化而变化，由K8s集群管理员配置；而PVC和Pod是资源的使用者，根据业务服务的需求变化而变化，有K8s集群的使用者即服务的管理员来配置。

#### .8. 节点（Node）

- K8s集群中的计算能力由Node提供，最初Node称为服务节点Minion，后来改名为Node。
- K8s集群中的Node也就等同于Mesos集群中的Slave节点，`是所有Pod运行所在的工作主机，可以是物理机也可以是虚拟机`。
- 不论是物理机还是虚拟机，`工作主机的统一特征是上面要运行kubelet管理节点上运行的容器`。

![kubernetes Container、Pod、Replicaset、Service、Deployment、Lable、Statefulset关系和区别_Kubernetes](https://s5.51cto.com/images/blog/202106/17/107c0a87915be3d36e4d96b13ffe5207.png?x-oss-process=image/watermark,size_16,text_QDUxQ1RP5Y2a5a6i,color_FFFFFF,t_100,g_se,x_10,y_10,shadow_90,type_ZmFuZ3poZW5naGVpdGk=)

### 2. 完整yaml

```yaml
apiVersion: v1        　　#必选，版本号，例如v1
kind: Pod       　　　　　　#必选，Pod
metadata:       　　　　　　#必选，元数据
  name: string        　　#必选，Pod名称
  namespace: string     　　#必选，Pod所属的命名空间
  labels:       　　　　　　#自定义标签
    - name: string      　#自定义标签名字
  annotations:        　　#自定义注释列表
    - name: string
spec:         　　　　　　　#必选，Pod中容器的详细定义,  期望的状态
  containers:       　　　　#必选，Pod中容器列表
  - name: string      　　#必选，容器名称
    image: string     　　#必选，容器的镜像名称
    imagePullPolicy: [Always | Never | IfNotPresent]  #获取镜像的策略 Alawys表示下载镜像 IfnotPresent表示优先使用本地镜像，否则下载镜像，Nerver表示仅使用本地镜像
    command: [string]     　　#容器的启动命令列表，如不指定，使用打包时使用的启动命令
    args: [string]      　　 #容器的启动命令参数列表
    workingDir: string      #容器的工作目录
    volumeMounts:     　　　　#挂载到容器内部的存储卷配置
    - name: string      　　　#引用pod定义的共享存储卷的名称，需用volumes[]部分定义的的卷名
      mountPath: string     #存储卷在容器内mount的绝对路径，应少于512字符
      readOnly: boolean     #是否为只读模式
    ports:        　　　　　　#需要暴露的端口库号列表
    - name: string      　　　#端口号名称
      containerPort: int    #容器需要监听的端口号
      hostPort: int     　　 #容器所在主机需要监听的端口号，默认与Container相同
      protocol: string      #端口协议，支持TCP和UDP，默认TCP
    env:        　　　　　　#容器运行前需设置的环境变量列表
    - name: string      　　#环境变量名称
      value: string     　　#环境变量的值
    resources:        　　#资源限制和请求的设置
      limits:       　　　　#资源限制的设置
        cpu: string     　　#Cpu的限制，单位为core数，将用于docker run --cpu-shares参数
        memory: string      #内存限制，单位可以为Mib/Gib，将用于docker run --memory参数
      requests:       　　#资源请求的设置
        cpu: string     　　#Cpu请求，容器启动的初始可用数量
        memory: string      #内存清楚，容器启动的初始可用数量
    livenessProbe:      　　#对Pod内个容器健康检查的设置，当探测无响应几次后将自动重启该容器，检查方法有exec、httpGet和tcpSocket，对一个容器只需设置其中一种方法即可
      exec:       　　　　　　#对Pod容器内检查方式设置为exec方式
        command: [string]   #exec方式需要制定的命令或脚本
      httpGet:        　　　　#对Pod内个容器健康检查方法设置为HttpGet，需要制定Path、port
        path: string
        port: number
        host: string
        scheme: string
        HttpHeaders:
        - name: string
          value: string
      tcpSocket:      　　　　　　#对Pod内个容器健康检查方式设置为tcpSocket方式
         port: number
       initialDelaySeconds: 0   #容器启动完成后首次探测的时间，单位为秒
       timeoutSeconds: 0    　　#对容器健康检查探测等待响应的超时时间，单位秒，默认1秒
       periodSeconds: 0     　　#对容器监控检查的定期探测时间设置，单位秒，默认10秒一次
       successThreshold: 0
       failureThreshold: 0
       securityContext:
         privileged: false
    restartPolicy: [Always | Never | OnFailure] #Pod的重启策略，Always表示一旦不管以何种方式终止运行，kubelet都将重启，OnFailure表示只有Pod以非0退出码退出才重启，Nerver表示不再重启该Pod
    nodeSelector: obeject   　　#设置NodeSelector表示将该Pod调度到包含这个label的node上，以key：value的格式指定
    imagePullSecrets:     　　　　#Pull镜像时使用的secret名称，以key：secretkey格式指定
    - name: string
    hostNetwork: false      　　#是否使用主机网络模式，默认为false，如果设置为true，表示使用宿主机网络
    volumes:        　　　　　　#在该pod上定义共享存储卷列表
    - name: string     　　 　　#共享存储卷名称 （volumes类型有很多种）
      emptyDir: {}      　　　　#类型为emtyDir的存储卷，与Pod同生命周期的一个临时目录。为空值
      hostPath: string      　　#类型为hostPath的存储卷，表示挂载Pod所在宿主机的目录
        path: string      　　#Pod所在宿主机的目录，将被用于同期中mount的目录
      secret:       　　　　　　#类型为secret的存储卷，挂载集群与定义的secre对象到容器内部
        scretname: string  
        items:     
        - key: string
          path: string
      configMap:      　　　　#类型为configMap的存储卷，挂载预定义的configMap对象到容器内部
        name: string
        items:
        - key: string
          path: string    
```

### 3. Deployment

```yaml
apiVersion: apps/v1   # 1.9.0 之前的版本使用 apps/v1beta2，可通过命令 kubectl api-versions 查看
kind: Deployment 	#指定创建资源的角色/类型
metadata: 	 #资源的元数据/属性
  name: nginx-deployment	#资源的名字，在同一个namespace中必须唯一
spec:
  replicas: 2 	 #副本数量2
  selector:      #定义标签选择器
    matchLabels:
      app: web-server
  template: 	 #这里Pod的定义
    metadata:
      labels: 	 #Pod的label
        app: web-server
    spec:		 # 指定该资源的内容  
      containers:  
      - name: nginx 	 #容器的名字  
        image: nginx:1.12.1  #容器的镜像地址    
        ports:  
        - containerPort: 80  #容器对外的端口
```

### 4. pod

```yaml
apiVersion: v1
kind: Pod  
metadata:  
  name: pod-redis
  labels:
    name: redis
spec: 
  containers:
  - name: pod-redis
    image: docker.io/redis  
    ports:
    - containerPort: 80	#容器对外的端口
```

### 5. service

```yaml
apiVersion: v1
kind: Pod  
metadata:  
  name: pod-redis
  labels:
    name: redis
spec: 
  containers:
  - name: pod-redis
    image: docker.io/redis  
    ports:
    - containerPort: 80	#容器对外的端口
```

### Resource

- https://www.cnblogs.com/flying1819/articles/9039529.html
- https://www.cnblogs.com/bakari/p/10509484.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes-yaml%E8%AF%A6%E8%A7%A3/  

