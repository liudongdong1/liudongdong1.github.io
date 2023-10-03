# Kubernetes-Storage


> - in-tree：代码逻辑在 K8s 官方仓库中；
> - out-of-tree：代码逻辑在 K8s 官方仓库之外，实现与 K8s 代码的解耦；
> - PV：PersistentVolume，`集群级别的资源`，由 `集群管理员 or External Provisioner 创建`。PV 的生命周期独立于使用 PV 的 Pod，PV 的 .Spec 中保存了存储设备的详细信息；
> - PVC：PersistentVolumeClaim，命名空间（namespace）级别的资源，由 `用户 or StatefulSet 控制器（根据 VolumeClaimTemplate） 创建`。PVC 类似于 Pod，`Pod 消耗 Node 资源，PVC 消耗 PV 资源`。Pod 可以请求特定级别的资源（CPU 和内存），而 PVC 可以请求特定存储卷的大小及访问模式（Access Mode）；
> - StorageClass：StorageClass 是`集群级别的资源`，由集群管理员创建。SC 为管理员提供了一种动态提供存储卷的“类”模板，SC 中的 .Spec 中详细定义了存储卷 PV 的不同服务质量级别、备份策略等等；
> - CSI：Container Storage Interface，目的是定义行业标准的“容器存储接口”，使存储供应商（SP）基于 CSI 标准开发的插件可以在不同容器编排（CO）系统中工作，CO 系统包括 Kubernetes、Mesos、Swarm 等。

### 1. configMap

> A [ConfigMap](https://kubernetes.io/docs/tasks/configure-pod-container/configure-pod-configmap/) provides a way to inject configuration data into pods. The data stored in a ConfigMap can be referenced in a volume of type `configMap` and then consumed by containerized applications running in a pod.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: configmap-pod
spec:
  containers:
    - name: test
      image: busybox:1.28
      volumeMounts:
        - name: config-vol
          mountPath: /etc/config
  volumes:
    - name: config-vol
      configMap:
        name: log-config
        items:
          - key: log_level
            path: log_level
```

> The `log-config` ConfigMap is mounted as a volume, and all contents stored in its `log_level` entry are mounted into the Pod at path `/etc/config/log_level`. Note that this path is derived from the volume's `mountPath` and the `path` keyed with `log_level`.

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernetes-storage/  

