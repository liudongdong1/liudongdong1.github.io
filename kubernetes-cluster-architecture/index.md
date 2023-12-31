# Kubernetes-Cluster Architecture


> Kubernetes runs your workload by placing containers into Pods to run on *Nodes*. A node may be a virtual or physical machine, depending on the cluster.

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7d70666e06094bb5b54a51f661fe4823~tplv-k3u1fbpfcp-zoom-in-crop-mark:3024:0:0:0.awebp)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220810210400686.png)

Kubernetes主要由以下几个核心组件组成：

- etcd保存了整个集群的状态；
- apiserver提供了资源操作的唯一入口，并提供认证、授权、访问控制、API注册和发现等机制；
- controller manager负责维护集群的状态，比如故障检测、自动扩展、滚动更新等；
- scheduler负责资源的调度，按照预定的调度策略将Pod调度到相应的机器上；
- kubelet负责维护容器的生命周期，同时也负责Volume（CVI）和网络（CNI）的管理；
- Container runtime负责镜像管理以及Pod和容器的真正运行（CRI）；
- kube-proxy负责为Service提供cluster内部的服务发现和负载均衡；

除了核心组件，还有一些推荐的Add-ons：

- kube-dns负责为整个集群提供DNS服务
- Ingress Controller为服务提供外网入口
- Heapster提供资源监控
- Dashboard提供GUI
- Federation提供跨可用区的集群

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/98593a62139449e281185ab42a0095eetplv-k3u1fbpfcp-zoom-in-crop-mark3024000.awebp)

![img](https://bbs-img.huaweicloud.com/blogs/img/image3(269).png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220424223135943.png)

![img](https://bbs-img.huaweicloud.com/blogs/img/images_162391078762653.png)

### List-Watch机制控制器架构

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220810210559303.png)

1、`客户端提交创建请求`，可以通过`API Server的Restful API`，也可以使用`kubectl命令行工具`。支持的数据类型包括JSON和YAML。

2、`API Server处理用户请求，存储Pod数据到etcd`。

3、`调度器通过API Server查看未绑定的Pod`。尝试为Pod分配主机。

4、`过滤主机 (调度预选)`：调度器用一组规则过滤掉不符合要求的主机。比如Pod指定了所需要的资源量，那么可用资源比Pod需要的资源量少的主机会被过滤掉。

5、`主机打分(调度优选)`：对第一步筛选出的符合要求的主机进行打分，在主机打分阶段，调度器会考虑一些整体优化策略，比如把容一个Replication Controller的副本分布到不同的主机上，使用最低负载的主机等。

6、`选择主机`：选择打分最高的主机，`进行binding操作，结果存储到etcd中`。

7、kubelet根据调度结果`执行Pod创建操作`： 绑定成功后，scheduler会调用APIServer的API在etcd中创建一个boundpod对象，描述在一个工作节点上绑定运行的所有pod信息。运行在每个工作节点上的kubelet也会定期与etcd同步boundpod信息，一旦发现应该在该工作节点上运行的boundpod对象没有更新，则调用Docker API创建并启动pod内的容器。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/af449707a9ba439185a3a937452cf5b1tplv-k3u1fbpfcp-zoom-in-crop-mark3024000.awebp)

![image.png](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/92b20772e5164de5a5c9bb076cf0160btplv-k3u1fbpfcp-zoom-in-crop-mark3024000.awebp)

## 1. Node

### .1. Node status

#### 1. Address

- HostName: The hostname as reported by the node's kernel. Can be overridden via the kubelet `--hostname-override` parameter.
- ExternalIP: Typically the IP address of the node that is externally routable (available from outside the cluster).
- InternalIP: Typically the IP address of the node that is routable only within the cluster.

#### 2. Conditions

| Node Condition       | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `Ready`              | `True` if the node is healthy and ready to accept pods, `False` if the node is not healthy and is not accepting pods, and `Unknown` if the node controller has not heard from the node in the last `node-monitor-grace-period` (default is 40 seconds) |
| `DiskPressure`       | `True` if pressure exists on the disk size—that is, if the disk capacity is low; otherwise `False` |
| `MemoryPressure`     | `True` if pressure exists on the node memory—that is, if the node memory is low; otherwise `False` |
| `PIDPressure`        | `True` if pressure exists on the processes—that is, if there are too many processes on the node; otherwise `False` |
| `NetworkUnavailable` | `True` if the network for the node is not correctly configured, otherwise `False` |

#### 3. Capacity and Allocatable

- Describes the resources available on the node: CPU, memory, and the maximum number of pods that can be scheduled onto the node.

#### 4. Info

- Describes general information about the node, such as kernel version, Kubernetes version (kubelet and kube-proxy version), container runtime details, and which operating system the node uses. The kubelet gathers this information from the node and publishes it into the Kubernetes API.

## 2. Control Plane-Node Communication

### .1. Node to Control Plane

- The apiserver is configured to listen for remote connections on a secure HTTPS port (typically 443) with one or more forms of client [authentication](https://kubernetes.io/docs/reference/access-authn-authz/authentication/) enabled. 

### .2. Control Plane to node

#### 1. apiserver to kubelet

- Fetching logs for pods.
- Attaching (through kubectl) to running pods.
- Providing the kubelet's port-forwarding functionality.

#### 2. apiserver to nodes, pods, and services

-  from the apiserver to any node, pod, or service through the apiserver's proxy functionality

## 3. Cloud controller manager

### .1. Node controller

1. Update a Node object with the corresponding server's unique identifier obtained from the cloud provider API.
2. Annotating and labelling the Node object with cloud-specific information, such as the region the node is deployed into and the resources (CPU, memory, etc) that it has available.
3. Obtain the node's hostname and network addresses.
4. Verifying the node's health. In case a node becomes unresponsive, this controller checks with your cloud provider's API to see if the server has been deactivated / deleted / terminated. If the node has been deleted from the cloud, the controller deletes the Node object from your Kubernetes cluster.

### .2. Route controller

-  configuring routes in the cloud appropriately so that containers on different nodes in your Kubernetes cluster can communicate with each other.

### .3. Service controller

- integrate with cloud infrastructure components such as managed load balancers, IP addresses, network packet filtering, and target health checking. 

## 4. Container Runtime Interface (CRI)

- The Kubernetes Container Runtime Interface (CRI) defines the main [gRPC](https://grpc.io/) protocol for the communication between the [cluster components](https://kubernetes.io/docs/concepts/overview/components/#node-components) [kubelet](https://kubernetes.io/docs/reference/generated/kubelet) and [container runtime](https://kubernetes.io/docs/setup/production-environment/container-runtimes)
- The kubelet acts as a client when connecting to the container runtime via gRPC. The runtime and image service endpoints have to be available in the container runtime, which can be configured separately within the kubelet by using the `--image-service-endpoint` and `--container-runtime-endpoint` [command line flags](https://kubernetes.io/docs/reference/command-line-tools-reference/kubelet)

## 5. Framework

![](https://gitee.com/github-25970295/picture2022/raw/master/core-ecosystem.png)

![](https://gitee.com/github-25970295/picture2022/raw/master/workflow.png)

![](https://gitee.com/github-25970295/picture2022/raw/master/ports.png)

### Resource

- https://juejin.cn/post/6952331691524358174#heading-5


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes-cluster-architecture/  

