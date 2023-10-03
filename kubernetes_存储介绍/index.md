# Kubernetes-存储介绍


## Docker插件机制-架构&评价

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220724085742161.png)

### Docker volumn 插件

| 名称                              | 描述                                                         | 地址                                                      |
| --------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| Azure File Storage plugin         | Lets you mount Microsoft Azure File Storage shares to Docker containers as volumes using the SMB 3.0 protocol. Learn more. | https://github.com/Azure/azurefile dockervolumedriver     |
| BeeGFS Volume Plugin              | An open source volume plugin to create persistent volumes in a BeeGFS parallel file system. | https://github.com/RedCoolBeans/ docker-volume-beegfs     |
| Blockbridge plugin                | A volume plugin that provides access to an extensible set of container-based persistent storage options. It supports single and multi-host Docker environments with features that include tenant isolation, automated provisioning, encryption, secure deletion, snapshots and QoS. | https://github.com/blockbridge/blo ckbridge-docker-volume |
| Contiv Volume Plugin              | An open source volume plugin that provides multi-tenant, persistent, distributed storage with intent based consumption. It has support for Ceph and NFS. | https://github.com/rancher/convoy                         |
| DigitalOcean Block Storage plugin | Integrates DigitalOcean’s block storage solution into the Docker ecosystem by automatically attaching a given block storage volume to a DigitalOcean droplet and making the contents of the volume available to Docker containers running on that droplet. | https://github.com/omallo/docker volume-plugin-dostorage  |
| DRBD plugin                       | A volume plugin that provides highly available storage replicated by DRBD. Data written to the docker volume is replicated in a cluster of DRBD nodes. | https://www.drbd.org/en/supporte d-projects/docker        |
| Flocker plugin                    | A volume plugin that provides multi-host portable volumes for Docker, enabling you to run databases and other stateful containers and move them around across a cluster of machines. | https://clusterhq.com/docker plugin/                      |
| Fuxi Volume Plugin                | A volume plugin that is developed as part of the OpenStack Kuryr project and implements the Docker volume plugin API by utilizing Cinder, the OpenStack block storage service. | https://github.com/openstack/fuxi                         |
| gce-docker plugin                 | A volume plugin able to attach, format and mount Google Compute persistent-disks. | https://github.com/mcuadros/gce docker                    |
| GlusterFS plugin                  | A volume plugin that provides multi-host volumes management for Docker using GlusterFS. | https://github.com/calavera/docker -volume-glusterfs      |

| Horcrux Volume Plugin        | A volume plugin that allows on-demand, version controlled access to your data. Horcrux is an open-source plugin, written in Go, and supports SCP, Minio and Amazon S3. | https://github.com/muthu-r/horcrux                           |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| HPE 3Par Volume Plugin       | A volume plugin that supports HPE 3Par and StoreVirtual iSCSI storage arrays. | https://github.com/hpe storage/python-hpedockerplugin/ -     |
| Infinit volume plugin        | A volume plugin that makes it easy to mount and manage Infinit volumes using Docker. | https://infinit.sh/documentation/dock er/volume-plugin       |
| IPFS Volume Plugin           | An open source volume plugin that allows using an ipfs filesystem as a volume. | http://github.com/vdemeester/docker -volume-ipfs             |
| Keywhiz plugin               | A plugin that provides credentials and secret management using Keywhiz as a central repository. | https://github.com/calavera/docker volume-keywhiz            |
| Local Persist Plugin         | A volume plugin that extends the default local driver’s functionality by allowing you specify a mountpoint anywhere on the host, which enables the files to always persist, even if the volume is removed via docker volume rm. | https://github.com/CWSpear/local persist                     |
| NetApp Plugin(nDVP)          | A volume plugin that provides direct integration with the Docker ecosystem for the NetApp storage portfolio. The nDVP package supports the provisioning and management of storage resources from the storage platform to Docker hosts, with a robust framework for adding additional platforms in the future. | https://github.com/NetApp/netappdv p                         |
| Netshare plugin              | A volume plugin that provides volume management for NFS 3/4, AWS EFS and CIFS file systems. | https://github.com/ContainX/docker volume-netshare           |
| Nimble Storage Volume Plugin | A volume plug-in that integrates with Nimble Storage Unified Flash Fabric arrays. The plug-in abstracts array volume capabilities to the Docker administrator to allow self-provisioning of secure multi-tenant volumes and clones. | https://connect.nimblestorage.com/co mmunity/app-integration/dockerOpenStorage Plugin |

| OpenStorage Plugin                 | A cluster-aware volume plugin that provides volume management for file and block storage solutions. It implements a vendor neutral specification for implementing extensions such as CoS, encryption, and snapshots. It has example drivers based on FUSE, NFS, NBD and EBS to name a few. | https://github.com/libopenstorage/ openstorage    |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| Portworx Volume Plugin             | A volume plugin that turns any server into a scale-out converged compute/storage node, providing container granular storage and highly available volumes across any node, using a shared-nothing storage backend that works with any docker scheduler. | https://github.com/portworx/px dev                |
| Quobyte Volume Plugin              | A volume plugin that connects Docker to Quobyte’s data center file system, a general-purpose scalable and fault-tolerant storage platform. | https://github.com/quobyte/docker -volume         |
| REX-Ray plugin                     | A volume plugin which is written in Go and provides advanced storage functionality for many platforms including VirtualBox, EC2, Google Compute Engine, OpenStack, and EMC. | https://github.com/emccode/rexray                 |
| Virtuozzo Storage and Ploop plugin | A volume plugin with support for Virtuozzo Storage distributed cloud file system as well as ploop devices. | https://github.com/virtuozzo/docke r-volume-ploop |
| VMware vSphere Storage Plugin      | Docker Volume Driver for vSphere enables customers to address persistent storage requirements for Docker containers in vSphere environments. | https://github.com/vmware/docker volume-vsphere   |

## K8S 存储能力-Volume概述

- K8S中的`普通Volume`提供了在容器中挂卷的能力，它不是独立的K8S资源对象，`不能通过k8s去管理（创建、删除等）`，只能在创建Pod时去引用。
- Pod需要设置`卷来源（ spec.volume ） 和挂载点（ spec.containers.volumeMounts ）` 两个信息后才可以使用相应的Volume。

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YMSAFuccAAGVhqcxzgU330.png)

### K8S 存储能力： In-Tree Volume Plugins

> K8S的VolumePlugin提供了`插件化扩展存储的机制`，分为内置插件（In-Tree Plugins）和外置插件（Out-of-Tree） 两种

| 名称                 | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| awsElasticBlockStore | mounts an Amazon Web Services (AWS) EBS Volume (Elastic Block Store) |
| azureDisk            | is used to mount a Microsoft Azure Data Disk into a Pod.     |
| azureFile            | is used to mount a Microsoft Azure File Volume (SMB 2.1 and 3.0) into a Pod. |
| `cephfs`             | allows an existing CephFS volume to be mounted into your pod. |
| cinder               | is used to mount OpenStack Block Storage into a pod.         |
| `configMap`          | The data stored in a ConfigMap object can be referenced in a volume of type configMap and then consumed by containerized applications running in a Pod. |
| downwardAPI          | is used to make downward API data available to applications. It mounts a directory and writes the requested data in plain text files |
| emptyDir             | is first created when a Pod is assigned to a Node, and exists as long as that Pod is running on that node. When a Pod is removed from a node for any reason, the data in the emptyDir is deleted forever. |
| fc (fibre channel)   | allows an existing fibre channel volume to be mounted in a pod |
| flocker              | allows a Flocker dataset to be mounted into a pod.           |
| gcePersistentDisk    | mounts a Google Compute Engine (GCE) Persistent Disk into your pod. |
| gitRepo              | mounts an empty directory and clones a git repository into it for your pod to use. |
| `glusterfs`          | allows a Glusterfs (an open source networked filesystem) volume to be mounted into your pod |
| `hostPath`           | mounts a file or directory from the host node’s filesystem into your pod. |
| `iscsi`              | allows an existing iSCSI (SCSI over IP) volume to be mounted into your pod |
| `local`              | represents a mounted local storage device such as a disk, partition or directory. can only be used as a statically created PersistentVolume. |

| 名称                    | 描述                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `nfs`                   | allows an existing NFS (Network File System) share to be mounted into your pod |
| `persistentVolumeClaim` | is used to mount a PersistentVolume into a pod.              |
| projected               | maps several existing volume sources into the same directory. |
| `portworxVolume`        | can be dynamically created through Kubernetes or it can also be pre-provisioned and referenced inside a Kubernetes pod. |
| quobyte                 | allows an existing Quobyte volume to be mounted into your pod. |
| rbd                     | allows a Rados Block Device volume to be mounted into your pod. |
| scaleIO                 | ScaleIO is a software-based storage platform that can use existing hardware to create clusters of scalable shared block networked storage. The ScaleIO volume plugin allows deployed pods to access existing ScaleIO volumes |
| secret                  | is used to pass sensitive information, such as passwords, to pods |
| storageos               | allows an existing StorageOS volume to be mounted into your pod. StorageOS provides block storage to containers, accessible via a file system. |
| vsphereVolume           | used to mount a vSphere VMDK Volume into your Pod.           |

### K8S 存储能力-PersistentVolume  **

> Kubernetes通过Persistent Volume子系统API对管理员和用户提供了存储资源创建和使用的抽象
>
> - `FlexVolume: 此Volume Driver允许不同厂商去开发他们自己的驱动来挂载卷到计算节点`
> - PersistentVolumeClaim： K8提供的资源抽象的Volume Driver，让用户不用关心具体的Volume的实现细节

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YMWAPt5SAAHnN6RtcJw960.png)

### K8S FlexVolume存储扩展机制

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YMeAcOhmAAIU1aMDRxU941.png)

#### Flex Volume Driver部署脚本和配置

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YMiAQ_4JAAK-YG1tcjA886.png)

#### Flex Volume CLI API

| 步骤               | 命令                                                         | 描述                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Init               | <driver executable> init                                     | 初始化驱动。在Kubelet和Controller-Manager初始化时被调用。若调用成功则需要返回一个展示对应驱动 所支持的FlexVolume能力的map，现在只包含一个必填字段attach，用于表明本驱动是否需要attach和 detach操作。为向后兼容该字段一般默认值设为true。 |
| Attach             | <driver executable> attach <json options> <node name>        | 将给定`规格的卷添加到给定的主机上`。若调用成功则返回存储设备添加到该主机的路径。 Kubelet和 Controller-Manager都需要调用该方法。 |
| Detach             | <driver executable> detach <mount device> <node name>        | `卸载给定主机上的指定卷`。 Kubelet和Controller-Manager都需要调用该方法。 |
| Wait for attach    | <driver executable> waitforattach <mount device> <json options> | 等待卷被添加到远程节点。若调用成功则将返回设备路径。 Kubelet和Controller-Manager都需要调用该方 法。 |
| Volume is Attached | <driver executable> isattached <json options> <node name>    | 检查卷是否已被添加到节点上。 Kubelet和Controller-Manager都需要调用该方法。 |
| Mount device       | <driver executable> mountdevice <mount dir> <mount device> <json options> | 将存储设备挂载到一个将被pod使用的全局路径上。 Kubelet需要调用该方法。 |
| Unmount device     | <driver executable> unmountdevice <mount device>             | 将存储设备卸载。 This is called once all bind mounts have been unmounted. Kubelet需要调用该方法。 |
| Mount              | <driver executable> mount <mount dir> <json options>         | 将卷挂载到指定目录。 Kubelet需要调用该方法。                 |
| Unmount            | <driver executable> unmount <mount dir>                      | 将卷进行卸载。 Kubelet需要调用该方法。                       |

### K8S CSI存储扩展机制

#### 相关术语

| 术语                | 含义                                                         |
| ------------------- | ------------------------------------------------------------ |
| CO                  | `容器编排系统（Container Orchestrator），使用CSI gRPC服务来与插件通信` |
| RPC                 | 远程方法调用（Remote Procedure Call）                        |
| Plugin              | `插件实现，实现CSI服务的gRPC访问端点`                        |
| SP                  | `存储提供商（Storage Provider），负责提供CSI插件实现`        |
| Volume              | 卷， CO管理的容器可使用的存储单元                            |
| Block Volume        | 块设备卷                                                     |
| Mounted Volume      | 使用指定文件系统挂载到容器的卷，并显示为容器内的一个目录     |
| Workload            | 工作负载，是CO任务调度的基本单元，可以是一个或一组容器       |
| Node                | 用户运行工作负载的主机，从插件的角度通过节点 ID来进行唯一标识 |
| In-Tree             | 内置的，存在于K8S核心代码仓库内的代码                        |
| Out-Of-Tree         | 外置的，存在于K8S核心代码仓库外的代码                        |
| `CSI Volume Plugin` | 一个新的内置卷插件，`作为一个适配器来使得外置的第三方CSI卷驱动可以被K8S所使用` |
| `CSI Volume Driver` | 一个外置的`CSI兼容的卷插件驱动，可通过K8S卷插件被K8S所使用`  |

#### 优势

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YMuAYLgxAAPBh01e6ho175.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YM6AYc2MAAOZ2lzpExk611.png)

### CSI通用架构

> CO通过gRPC与插件交互，每个SP必须实现以下两个plugin：
> • Node Plugin： 需要`运行在使用Volume的Node上`，主要负责V`olume Mount/Unmount等操作`
> • Controller Plugin：可以`运行在任何节点`上，主要负责`Volume Creation/Deletion、 Attach/Detach等操作`

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YM-AZl2pAAGOv1eyoAM156.png)

#### CO与Plugin的交互过程

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YNGAXx2hAAJgsrqp3tI355.png)

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YNOAT6hLAAMXkmm6Yd4767.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YNWAV8XtAAKAk8WL-NM148.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YNaAa6uwAAJWvv4dhwk488.png)

#### 卷生命周期

![img](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YNeABj33AADssTtyNGc569.png)

### RPC 接口集合

- Identity Service： Node Plugin和Controller Plugin都需要实现的RPC集合，
  - 身份服务RPC允许CO查询插件的功能，健康状况和其他元数据。
- Controller Service： Controller Plugin需要实现的RPC集合
  - 控制服务RPC提供卷的`创建、删除、 Attach、 Detach、查询`等功能，以及`卷快照的创建、删除、查询`等功能
- Node Service： Node Plugin需要实现的RPC集合
  - 将卷mount 到制定全局路径，在NodePublishVolume之前，每个卷/节点执行一次
  - 将卷从指定全局路径umount
  - 将卷从制定的全局路径mount到目标路径，在NodePublishVolume之前，每个卷/节点执行一次
  - 将卷从指定目标路径unmount

## K8S CSI架构

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/rBAADF80YN-ADt2aAAP1iSVfOCM122.png)

为了部署一个容器化的第三方CSI volume driver，存储提供商需要执行如下操作：

1. 创建一个实现CSI规范描述的插件功能，并通过Unix套接字来暴露gPRC访问接口的”CSI volume driver” 容器；
2. 结合使用K8S团队提供的帮助容器来部署CSI volume driver，具体需要创建如下两类K8S对象：
   1. StatefulSet：用于与K8S控制器进行交互，实例数1，包含3个容器（ CSI volume driver、 external-attacher 、 external provisioner ），需要挂载一个挂载点为/var/lib/csi/sockets/pluginproxy/的emptyDir volume
   2. DaemonSet ：包含2个容器（ CSI volumedriver、 K8S CSI Helper），挂载3个hostpath volume
3. 集群管理员为存储系统在K8S集群中部署上述StatefulSet和DaemonSet

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes_%E5%AD%98%E5%82%A8%E4%BB%8B%E7%BB%8D/  

