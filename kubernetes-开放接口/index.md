# Kubernetes-开放接口


> Kubernetes作为云原生应用的基础调度平台，相当于云原生的操作系统，为了便于系统的扩展，Kubernetes中开放的以下接口，可以分别对接不同的后端，来实现自己的业务逻辑：
>
> - **CRI（Container Runtime Interface）**：容器运行时接口，提供计算资源
> - **CNI（Container Network Interface）**：容器网络接口，提供网络资源
> - **CSI（Container Storage Interface**）：容器存储接口，提供存储资源

# CRI - Container Runtime Interface

> CRI中定义了**容器**和**镜像**的服务的接口，因为容器运行时与镜像的生命周期是彼此隔离的，因此需要定义两个服务。该接口使用[Protocol Buffer](https://developers.google.com/protocol-buffers/)，基于[gRPC](https://grpc.io/)，在kubernetes v1.7+版本中是在[pkg/kubelet/apis/cri/v1alpha1/runtime](https://github.com/kubernetes/kubernetes/tree/master/pkg/kubelet/apis/cri/v1alpha1/runtime)的`api.proto`中定义的。
>
> Container Runtime实现了CRI gRPC Server，包括`RuntimeService`和`ImageService`。该gRPC Server需要监听本地的Unix socket，而kubelet则作为gRPC Client运行。
>
> - **RuntimeService**：容器和Sandbox运行时管理
> - **ImageService**：提供了从镜像仓库拉取、查看、和移除镜像的RPC。
>
> 当前Linux上支持unix socket，windows上支持tcp。例如：`unix:///var/run/dockershim.sock`、 `tcp://localhost:373`，默认是`unix:///var/run/dockershim.sock`，即默认使用本地的docker作为容器运行时。

![CRI架构-图片来自kubernetes blog](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/a51f510f0b990c8babfa6e58d6e7f8f1.png)

# CNI - Container Network Interface

> CNI（Container Network Interface）是CNCF旗下的一个项目，由一组用于配置Linux容器的网络接口的规范和库组成，同时还包含了一些插件。CNI仅关心容器创建时的网络分配，和当容器被删除时释放网络资源。通过此链接浏览该项目：https://github.com/containernetworking/cni。

```go
type CNI interface {
    AddNetworkList(net *NetworkConfigList, rt *RuntimeConf) (types.Result, error)
    DelNetworkList(net *NetworkConfigList, rt *RuntimeConf) error
    AddNetwork(net *NetworkConfig, rt *RuntimeConf) (types.Result, error)
    DelNetwork(net *NetworkConfig, rt *RuntimeConf) error
}
```

## CNI插件

CNI插件必须实现一个可执行文件，这个文件可以被容器管理系统（例如rkt或Kubernetes）调用。

CNI插件负责`将网络接口插入容器网络命名空间`（例如，veth对的一端），并在主机上进行任何必要的改变（例如将veth的另一端连接到网桥）。然后将IP分配给接口，并通过调用适当的IPAM插件来设置与“IP地址管理”部分一致的路由。

#### 将容器添加到网络

参数：

- **版本**。调用者正在使用的CNI规范（容器管理系统或调用插件）的版本。
- **容器ID** 。由运行时分配的容器的唯一明文标识符。一定不能是空的。
- **网络命名空间路径**。要添加的网络名称空间的路径，即`/proc/[pid]/ns/net`或绑定挂载/链接。
- **网络配置**。描述容器可以加入的网络的JSON文档。
- **额外的参数**。这提供了一个替代机制，允许在每个容器上简单配置CNI插件。
- **容器内接口的名称**。这是应该分配给容器（网络命名空间）内创建的接口的名称；因此它必须符合Linux接口名称上的标准限制。

结果：

- **接口列表**。根据插件的不同，这可以包括沙箱（例如容器或管理程序）接口名称和/或主机接口名称，每个接口的硬件地址以及接口所在的沙箱（如果有的话）的详细信息。
- **分配给每个接口的IP配置**。分配给沙箱和/或主机接口的IPv4和/或IPv6地址，网关和路由。
- **DNS信息**。包含nameserver、domain、search domain和option的DNS信息的字典。

#### 从网络中删除容器

参数：

- **版本**。调用者正在使用的CNI规范（容器管理系统或调用插件）的版本。
- **容器ID** ，如上所述。
- **网络命名空间路径**，如上定义。
- **网络配置**，如上所述。
- **额外的参数**，如上所述。
- **上面定义的容器**内的接口的名称。

- 所有参数应与传递给相应的添加操作的参数相同。
- 删除操作应释放配置的网络中提供的containerid拥有的所有资源。

报告版本

- 参数：无。
- 结果：插件支持的CNI规范版本信息。

```
{“cniVersion”：“0.3.1”，//此输出使用的CNI规范的版本“supportedVersions”：[“0.1.0”，“0.2.0”，“0.3.0”，“0.3.1”] //此插件支持的CNI规范版本列表}
```

CNI插件的详细说明请参考：[CNI SPEC](https://github.com/containernetworking/cni/blob/master/SPEC.md)。

### IP分配

作为容器网络管理的一部分，`CNI插件需要为接口分配（并维护）IP地址`，并安装与该接口相关的所有必要路由。这给了CNI插件很大的灵活性，但也给它带来了很大的负担。众多的CNI插件需要编写相同的代码来支持用户需要的多种IP管理方案（例如dhcp、host-local）。

为了减轻负担，使IP管理策略与CNI插件类型解耦，我们`定义了IP地址管理插件（IPAM插件）。CNI插件的职责是在执行时恰当地调用IPAM插件`。 IPAM插件必须确定接口IP/subnet，网关和路由，并将此信息返回到“主”插件来应用配置。 IPAM插件可以通过协议（例如dhcp）、存储在本地文件系统上的数据、网络配置文件的“ipam”部分或上述的组合来获得信息。

#### IPAM插件

像CNI插件一样，调用IPAM插件的可执行文件。可执行文件位于预定义的路径列表中，通过`CNI_PATH`指示给CNI插件。 IPAM插件必须接收所有传入CNI插件的相同环境变量。就像CNI插件一样，IPAM插件通过stdin接收网络配置。

## 可用插件

### Main：接口创建

- **bridge**：创建网桥，并添加主机和容器到该网桥
- **ipvlan**：在容器中添加一个[ipvlan](https://www.kernel.org/doc/Documentation/networking/ipvlan.txt)接口
- **loopback**：创建一个回环接口
- **macvlan**：创建一个新的MAC地址，将所有的流量转发到容器
- **ptp**：创建veth对
- **vlan**：分配一个vlan设备

### IPAM：IP地址分配

- **dhcp**：在主机上运行守护程序，代表容器发出DHCP请求
- **host-local**：维护分配IP的本地数据库

### Meta：其它插件

- **flannel**：根据flannel的配置文件创建接口
- **tuning**：调整现有接口的sysctl参数
- **portmap**：一个基于iptables的portmapping插件。将端口从主机的地址空间映射到容器。

# CSI - Container Storage Interface

CSI 代表[容器存储接口](https://github.com/container-storage-interface/spec/blob/master/spec.md)，CSI 试图建立一个行业标准接口的规范，借助 CSI 容器编排系统（CO）可以将任意存储系统暴露给自己的容器工作负载。

- [External-attacher](https://github.com/kubernetes-csi/external-attacher): 可监听 Kubernetes VolumeAttachment 对象并`触发 ControllerPublish 和 ControllerUnPublish 操作`的 sidecar 容器，通过 CSI endpoint 触发 ；
- [External-provisioner](https://github.com/kubernetes-csi/external-provisioner): 监听 Kubernetes PersistentVolumeClaim 对象的 sidecar 容器，并触发对 CSI 端点的 `CreateVolume 和DeleteVolume` 操作；
- [Driver-registrar](https://github.com/kubernetes-csi/driver-registrar): 使用 Kubelet（将来）注册 CSI 驱动程序的 sidecar 容器，并将 `NodeId` （通过 `GetNodeID` 调用检索到 CSI endpoint）添加到 Kubernetes Node API 对象的 annotation 里面。

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes-%E5%BC%80%E6%94%BE%E6%8E%A5%E5%8F%A3/  

