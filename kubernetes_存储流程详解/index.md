# Kubernetes_存储流程详解


### 1.流程概览

![](../../../../blogimgv2022/88c73171b412b2f9d996d155384647ad.png)

流程如下：

1. 用户创建了一个包含 PVC 的 Pod，该 PVC 要求使用动态存储卷；
2. Scheduler 根据 Pod 配置、节点状态、PV 配置等信息，把 `Pod 调度到一个合适的 Worker 节点上`；
3. `PV 控制器 watch 到该 Pod 使用的 PVC 处于 Pending 状态，于是调用 Volume Plugin（in-tree）创建存储卷`，并`创建 PV 对象`（out-of-tree 由 External Provisioner 来处理）；
4. `AD 控制器发现 Pod 和 PVC 处于待挂接状态`，于是调用 `Volume Plugin 挂接存储设备到目标 Worker 节点上`
5. 在 Worker 节点上，Kubelet 中的 Volume Manager 等待存储设备挂接完成，并通过 `Volume Plugin 将设备挂载到全局目录`：/var/lib/kubelet/pods/[pod uid]/volumes/kubernetes.io~iscsi/[PV name]（以 iscsi 为例）；
6. Kubelet 通过 Docker 启动 Pod 的 Containers，用 `bind mount 方式将已挂载到本地全局目录的卷映射到容器中。`

更详细的流程如下：

![img](../../../../blogimgv2022/47a28aa992e90fd5a5bb7a30b6e4a8b3.png)

### 2.流程详解

不同 K8s 版本，持久化存储流程略有区别。本文基于 Kubernetes 1.14.8 版本。从上述流程图中可看到，存储卷从创建到提供应用使用共分为三个阶段：`Provision/Delete、Attach/Detach、Mount/Unmount`。

#### provisioning volumes

![img](https://static001.infoq.cn/resource/image/fb/b7/fb17827e4e6f2266a0987e7d0dfbeeb7.png)

**PV 控制器中有两个 Worker**：

- ClaimWorker：处理 `PVC 的 add / update / delete 相关事件以及 PVC 的状态迁移`；
- VolumeWorker：`负责 PV 的状态迁移`。

**PV 状态迁移（UpdatePVStatus）**：

- PV 初始状态为 Available，当 PV 与 PVC 绑定后，状态变为 Bound；
- 与 PV 绑定的 PVC 删除后，状态变为 Released；
- 当 PV 回收策略为 Recycled 或手动删除 PV 的 .Spec.ClaimRef 后，PV 状态变为 Available；
- 当 PV 回收策略未知或 Recycle 失败或存储卷删除失败，PV 状态变为 Failed；
- 手动删除 PV 的 .Spec.ClaimRef，PV 状态变为 Available。

**PVC 状态迁移（UpdatePVCStatus）**：

- 当集群中不存在满足 PVC 条件的 PV 时，PVC 状态为 Pending。在 PV 与 PVC 绑定后，PVC 状态由 Pending 变为 Bound；
- 与 PVC 绑定的 PV 在环境中被删除，PVC 状态变为 Lost；
- 再次与一个同名 PV 绑定后，PVC 状态变为 Bound。

#### -Provision 流程

**静态存储卷流程（FindBestMatch）**：PV 控制器首先在环境中筛选一个状态为 Available 的 PV 与新 PVC 匹配。

- **DelayBinding**：
  - PV 控制器判断该 PVC 是否需要延迟绑定：查看 PVC 的 annotation 中是否包含 volume.kubernetes.io/selected-node，若存在则表示该 PVC 已经被调度器指定好了节点（属于 ProvisionVolume），故不需要延迟绑定；
  - 若 PVC 的 annotation 中不存在 volume.kubernetes.io/selected-node，同时没有 StorageClass，默认表示不需要延迟绑定；`若有 StorageClass，查看其 VolumeBindingMode 字段，若为 WaitForFirstConsumer 则需要延迟绑定`，若为 Immediate 则不需要延迟绑定；
- **FindBestMatchPVForClaim**：PV 控制器尝试找一个满足 PVC 要求的环境中现有的 PV。PV 控制器会将所有的 PV 进行一次筛选，并会从满足条件的 PV 中选择一个最佳匹配的 PV。筛选规则：
  - `VolumeMode 是否匹配`；
  - PV 是否已绑定到 PVC 上；
  -  PV 的 .Status.Phase 是否为 Available；
  -  `LabelSelector `检查，PV 与 PVC 的 label 要保持一致；
  - PV 与 PVC 的 `StorageClass` 是否一致；
  - 每次迭代更新`最小满足 PVC requested size `的 PV，并作为最终结果返回；
- **Bind**：PV 控制器对选中的 PV、PVC 进行绑定：
  - 更新 PV 的 .Spec.ClaimRef 信息为当前 PVC；
  - 更新 PV 的 .Status.Phase 为 Bound；
  -  新增 PV 的 annotation ：pv.kubernetes.io/bound-by-controller: “yes”；
  - 更新 PVC 的 .Spec.VolumeName 为 PV 名称；
  - 更新 PVC 的 .Status.Phase 为 Bound；
  - 新增 PVC 的 annotation：pv.kubernetes.io/bound-by-controller: “yes” 和 pv.kubernetes.io/bind-completed: “yes”；

**动态存储卷流程（ProvisionVolume）**：若环境中没有合适的 PV，则进入动态 Provisioning 场景：

- **Before Provisioning**：
  - PV 控制器首先判断 PVC 使用的 StorageClass 是 in-tree 还是 out-of-tree：通过查看 `StorageClass 的 Provisioner 字段是否包含 “kubernetes.io/” 前缀来判断`；
  - PV 控制器更新 PVC 的 annotation：claim.Annotations[“volume.beta.kubernetes.io/storage-provisioner”] = storageClass.Provisioner；
- **in-tree Provisioning**（internal provisioning）：
  - in-tree 的 Provioner 会实现 `ProvisionableVolumePlugin 接口的 NewProvisioner 方法，用来返回一个新的 Provisioner`；
  - PV 控制器调用 Provisioner 的 Provision 函数，该函数会返回一个 PV 对象；
  - PV 控制器创建上一步返回的 PV 对象，将其与 PVC 绑定，Spec.ClaimRef 设置为 PVC，.Status.Phase 设置为 Bound，.Spec.StorageClassName 设置为与 PVC 相同的 StorageClassName；同时新增 annotation：“pv.kubernetes.io/bound-by-controller”=“yes” 和 “pv.kubernetes.io/provisioned-by”=plugin.GetPluginName()；
- **out-of-tree Provisioning**（external provisioning）：
  - External Provisioner 检查 PVC 中的 claim.Spec.VolumeName 是否为空，不为空则直接跳过该 PVC；
  - External Provisioner 检查 PVC 中的 claim.Annotations[“volume.beta.kubernetes.io/storage-provisioner”] 是否等于自己的 Provisioner Name（External Provisioner 在启动时会传入 --provisioner 参数来确定自己的 Provisioner Name）；
  - 若 PVC 的 VolumeMode=Block，检查 External Provisioner 是否支持块设备；
  - External Provisioner 调用 Provision 函数：通过 `gRPC 调用 CSI 存储插件的 CreateVolume 接口`；
  - External Provisioner 创建一个 PV 来代表该 volume，同时将该 PV 与之前的 PVC 做绑定。

#### -**deleting volumes** 流程

- 用户删除 PVC，删除 PV 控制器改变 PV.Status.Phase 为 Released。

- 当 PV.Status.Phase == Released 时，PV 控制器首先检查 Spec.PersistentVolumeReclaimPolicy 的值，为 Retain 时直接跳过，为 Delete 时：

  - **in-tree Deleting**：
    - in-tree 的 Provioner 会实现 DeletableVolumePlugin 接口的 NewDeleter 方法，用来返回一个新的 Deleter；
    - 控制器调用 Deleter 的 Delete 函数，删除对应 volume；
    - 在 volume 删除后，PV 控制器会删除 PV 对象；

  - **out-of-tree Deleting**：
    - External Provisioner 调用 Delete 函数，通过 gRPC 调用 CSI 插件的 DeleteVolume 接口；
    - 在 volume 删除后，External Provisioner 会删除 PV 对象

#### **Attaching Volumes**

Kubelet 组件和 AD 控制器都可以做 attach/detach 操作，若 Kubelet 的启动参数中指定了 --enable-controller-attach-detach，则由 Kubelet 来做；否则默认由 AD 控制起来做。下面以 AD 控制器为例来讲解 attach/detach 操作。

![](../../../../blogimgv2022/bdcbcf0ce4b69a68a2efa5ff98f79ac7.png)



**AD 控制器中有两个核心变量**：

- DesiredStateOfWorld（DSW）：集群中`预期的数据卷挂接状态`，包含了 nodes->volumes->pods 的信息；
- ActualStateOfWorld（ASW）：集群中`实际的数据卷挂接状态`，包含了 volumes->nodes 的信息。

#### - Attaching 流程

AD 控制器根据集群中的资源信息，初始化 DSW 和 ASW。

AD 控制器内部有三个组件周期性更新 DSW 和 ASW：

- **Reconciler**。通过一个 GoRoutine 周期性运行，确保 volume 挂接 / 摘除完毕。此期间不断更新 ASW：
  - in-tree attaching：1. in-tree 的 Attacher 会实现 AttachableVolumePlugin 接口的 NewAttacher 方法，用来返回一个新的 Attacher；2. AD 控制器调用 Attacher 的 Attach 函数进行设备挂接；3. 更新 ASW。
  - out-of-tree attaching：1. 调用 in-tree 的 CSIAttacher 创建一个 VolumeAttachement（VA）对象，该对象包含了 Attacher 信息、节点名称、待挂接 PV 信息；2. External Attacher 会 watch 集群中的 VolumeAttachement 资源，发现有需要挂接的数据卷时，调用 Attach 函数，通过 gRPC 调用 CSI 插件的 ControllerPublishVolume 接口。

- **DesiredStateOfWorldPopulator**。通过一个 GoRoutine 周期性运行，主要功能是更新 DSW
  - findAndRemoveDeletedPods - 遍历所有 DSW 中的 Pods，若其已从集群中删除则从 DSW 中移除；
  - findAndAddActivePods - 遍历所有 PodLister 中的 Pods，若 DSW 中不存在该 Pod 则添加至 DSW。

- **PVC Worker**。watch PVC 的 add/update 事件，处理 PVC 相关的 Pod，并实时更新 DSW。

#### -Detaching Volumes 流程

1. 当 Pod 被删除，`AD 控制器`会 watch 到该事件。首先 AD 控制器检查 Pod 所在的 Node 资源是否包含"volumes.kubernetes.io/`keep-terminated-pod-volumes`"标签，若包含则不做操作；不包含则从 DSW 中去掉该 volume；
2. AD 控制器通过 `Reconciler `使 ActualStateOfWorld 状态向 DesiredStateOfWorld 状态靠近，当发现 ASW 中有 DSW 中不存在的 volume 时，会做 Detach 操作：

**in-tree detaching**：1. AD 控制器会实现 AttachableVolumePlugin 接口的 NewDetacher 方法，用来返回一个新的 Detacher；2. 控制器调用 Detacher 的 Detach 函数，detach 对应 volume；3. AD 控制器更新 ASW。

**out-of-tree detaching**：1. AD 控制器调用 out-tree 的 CSIAttacher 删除相关 VolumeAttachement 对象；2. External Attacher 会 watch 集群中的 VolumeAttachement（VA）资源，发现有需要摘除的数据卷时，调用 Detach 函数，通过 gRPC 调用 CSI 插件的 ControllerUnpublishVolume 接口；3. AD 控制器更新 ASW。

#### - Mounting/Unmounting Volumes

![](../../../../blogimgv2022/e8c4e7e085da874578243bb3747c91bf.png)

Volume Manager 中同样也有两个核心变量：

- DesiredStateOfWorld（DSW）：集群中预期的数据卷挂载状态，包含了 volumes->pods 的信息；
- ActualStateOfWorld（ASW）：集群中实际的数据卷挂载状态，包含了 volumes->pods 的信息。

Mounting/UnMounting 流程如下：

> 全局目录（global mount path）存在的目的：块设备在 Linux 上只能挂载一次，而在 K8s 场景中，一个 PV 可能被挂载到同一个 Node 上的多个 Pod 实例中。若块设备格式化后先挂载至 Node 上的一个临时全局目录，然后再使用 Linux 中的 bind mount 技术把这个全局目录挂载进 Pod 中对应的目录上，就可以满足要求。上述流程图中，全局目录即 /var/lib/kubelet/pods/[pod uid]/volumes/kubernetes.io~iscsi/[PV name]

VolumeManager 根据集群中的资源信息，初始化 DSW 和 ASW。

VolumeManager 内部有两个组件周期性更新 DSW 和 ASW：

- DesiredStateOfWorldPopulator：通过一个 GoRoutine 周期性运行，主要功能是更新 DSW；
- Reconciler：通过一个 GoRoutine 周期性运行，确保 volume 挂载 / 卸载完毕。此期间不断更新 ASW：

unmountVolumes：确保 Pod 删除后 volumes 被 unmount。遍历一遍所有 ASW 中的 Pod，若其不在 DSW 中（表示 Pod 被删除），此处以 VolumeMode=FileSystem 举例，则执行如下操作：

1. Remove all bind-mounts：调用 Unmounter 的 TearDown 接口（若为 out-of-tree 则调用 CSI 插件的 NodeUnpublishVolume 接口）；
2. Unmount volume：调用 DeviceUnmounter 的 UnmountDevice 函数（若为 out-of-tree 则调用 CSI 插件的 NodeUnstageVolume 接口）；
3. 更新 ASW。

mountAttachVolumes：确保 Pod 要使用的 volumes 挂载成功。遍历一遍所有 DSW 中的 Pod，若其不在 ASW 中（表示目录待挂载映射到 Pod 上），此处以 VolumeMode=FileSystem 举例，执行如下操作：

1. 等待 volume 挂接到节点上（由 External Attacher or Kubelet 本身挂接）；
2. 挂载 volume 到全局目录：调用 DeviceMounter 的 MountDevice 函数（若为 out-of-tree 则调用 CSI 插件的 NodeStageVolume 接口）；
3. 更新 ASW：该 volume 已挂载到全局目录；
4. bind-mount volume 到 Pod 上：调用 Mounter 的 SetUp 接口（若为 out-of-tree 则调用 CSI 插件的 NodePublishVolume 接口）；
5. 更新 ASW。

unmountDetachDevices：确保需要 unmount 的 volumes 被 unmount。遍历一遍所有 ASW 中的 UnmountedVolumes，若其不在 DSW 中（表示 volume 已无需使用），执行如下操作：

1. Unmount volume：调用 DeviceUnmounter 的 UnmountDevice 函数（若为 out-of-tree 则调用 CSI 插件的 NodeUnstageVolume 接口）；
2. 更新 ASW。

### Resource

- https://juejin.cn/post/7014625141720088606#heading-17

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes_%E5%AD%98%E5%82%A8%E6%B5%81%E7%A8%8B%E8%AF%A6%E8%A7%A3/  

