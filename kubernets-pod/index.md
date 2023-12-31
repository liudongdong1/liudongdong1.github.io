# Kubernetes-Pod


> While Kubernetes supports more [container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes) than just Docker, [Docker](https://www.docker.com/) is the most commonly known runtime, and it helps to describe Pods using some terminology from Docker.
>
> `The shared context of a Pod is a set of Linux namespaces, cgroups, and potentially other facets of isolation` - the same things that isolate a Docker container. Within a Pod's context, the individual applications may have further sub-isolations applied.
>
> In terms of Docker concepts, `a Pod is similar to a group of Docker containers with shared namespaces and shared filesystem volumes.`

### 1. Using Pods

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
    ports:
    - containerPort: 80
```

- **Pods that run a single container**. The "one-container-per-Pod" model is the most common Kubernetes use case; in this case, you can think of a Pod as a wrapper around a single container; Kubernetes manages Pods rather than managing the containers directly.
- **Pods that run multiple containers that need to work together**. The containers in a Pod are automatically co-located and co-scheduled on the same physical or virtual machine in the cluster. The containers can share resources and dependencies, communicate with one another, and coordinate when and how they are terminated.

### 2. working with pods

#### .1. Pod and controllers

> A controller for the resource handles replication and rollout and automatic healing in case of Pod failure. 

#### .2. Pod template

> PodTemplates are specifications for creating Pods, and are included in workload resources such as [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/), [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/), and [DaemonSets](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/).

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hello
spec:
  template:
    # This is the pod template
    spec:
      containers:
      - name: hello
        image: busybox:1.28
        command: ['sh', '-c', 'echo "Hello, Kubernetes!" && sleep 3600']
      restartPolicy: OnFailure
    # The pod template ends here
```

#### 3. Pod update and replacement

> when the Pod template for a workload resource is changed, the controller creates new Pods based on the updated template instead of updating or patching the existing Pods.

- Most of the metadata about a Pod is immutable. For example, you cannot change the `namespace`, `name`, `uid`, or `creationTimestamp` fields; the `generation` field is unique. It only accepts updates that increment the field's current value.
- If the `metadata.deletionTimestamp` is set, no new entry can be added to the `metadata.finalizers` list.
- Pod updates may not change fields other than `spec.containers[*].image`, `spec.initContainers[*].image`, `spec.activeDeadlineSeconds` or `spec.tolerations`. For `spec.tolerations`, you can only add new entries.
- When updating the `spec.activeDeadlineSeconds` field, two types of updates are allowed:
  1. setting the unassigned field to a positive number;
  2. updating the field from a positive number to a smaller, non-negative number.

### 3. Resource sharing and communication

- A Pod can specify a set of shared storage [volumes](https://kubernetes.io/docs/concepts/storage/volumes/). All containers in the Pod can access the shared volumes, allowing those containers to share data.
- Volumes also allow persistent data in a Pod to survive in case one of the containers within needs to be restarted. 
- Each Pod is assigned a unique IP address for each address family. Every container in a Pod shares the network namespace, including the IP address and network ports.
- `Inside a Pod` (and **only** then), the containers that belong to the Pod can communicate with one another using `localhost`. When containers in a Pod communicate with entities `outside the Pod`, they must coordinate how they use the shared network resources (such as ports).

### 4. Container probes

A *probe* is a diagnostic performed periodically by the kubelet on a container. To perform a diagnostic, the kubelet can invoke different actions:

- `ExecAction` (performed with the help of the container runtime)
- `TCPSocketAction` (checked directly by the kubelet)
- `HTTPGetAction` (checked directly by the kubelet)

#### .1. Check mechanisms

- `exec`: Executes a specified command inside the container. The diagnostic is considered successful if the command exits with a status code of 0.
- `grpc`: Performs a remote procedure call using [gRPC](https://grpc.io/). The target should implement [gRPC health checks](https://grpc.io/grpc/core/md_doc_health-checking.html). The diagnostic is considered successful if the `status` of the response is `SERVING`.
  gRPC probes are an alpha feature and are only available if you enable the `GRPCContainerProbe` [feature gate](https://kubernetes.io/docs/reference/command-line-tools-reference/feature-gates/).
- `httpGet`: Performs an HTTP `GET` request against the Pod's IP address on a specified port and path. The diagnostic is considered successful if the response has a status code greater than or equal to 200 and less than 400.

- `tcpSocket`: Performs a TCP check against the Pod's IP address on a specified port. The diagnostic is considered successful if the port is open. If the remote system (the container) closes the connection immediately after it opens, this counts as healthy.

#### .2. Types of probe

- livenessProbe: Indicates whether the container is running. If the liveness probe fails, the kubelet kills the container, and the container is subjected to its [restart policy](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy). If a container does not provide a liveness probe, the default state is `Success`.
- readlinessProbe: Indicates whether the container is ready to respond to requests. If the readiness probe fails, the endpoints controller removes the Pod's IP address from the endpoints of all Services that match the Pod. The default state of readiness before the initial delay is `Failure`. If a container does not provide a readiness probe, the default state is `Success`.
- startupProbe: Indicates whether the application within the container is started. All other probes are disabled if a startup probe is provided, until it succeeds. If the startup probe fails, the kubelet kills the container, and the container is subjected to its [restart policy](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy). If a container does not provide a startup probe, the default state is `Success`.

### 5. Pod Lifecycle

| Value       | Description                                                  |
| :---------- | :----------------------------------------------------------- |
| `Pending`   | The Pod has been accepted by the Kubernetes cluster, but one or more of the containers has not been set up and made ready to run. This includes time a Pod spends waiting to be scheduled as well as the time spent downloading container images over the network. |
| `Running`   | The Pod has been bound to a node, and all of the containers have been created. At least one container is still running, or is in the process of starting or restarting. |
| `Succeeded` | All containers in the Pod have terminated in success, and will not be restarted. |
| `Failed`    | All containers in the Pod have terminated, and at least one container has terminated in failure. That is, the container either exited with non-zero status or was terminated by the system. |
| `Unknown`   | For some reason the state of the Pod could not be obtained. This phase typically occurs due to an error in communicating with the node where the Pod should be running. |

> When a Pod is being deleted, it is shown as `Terminating` by some kubectl commands. This `Terminating` status is not one of the Pod phases.

- The `spec` of a Pod has a `restartPolicy` field with possible values Always, OnFailure, and Never. The default value is Always.
- The `restartPolicy` applies to all containers in the Pod. `restartPolicy` only refers to restarts of the containers by the kubelet on the same node. 

### 6. Pod conditions

A Pod has a PodStatus, which has an array of [PodConditions](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.24/#podcondition-v1-core) through which the Pod has or has not passed:

- `PodScheduled`: the Pod has been scheduled to a node.
- `ContainersReady`: all containers in the Pod are ready.
- `Initialized`: all [init containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) have completed successfully.
- `Ready`: the Pod is able to serve requests and should be added to the load balancing pools of all matching Services.

| Field name           | Description                                                  |
| :------------------- | :----------------------------------------------------------- |
| `type`               | Name of this Pod condition.                                  |
| `status`             | Indicates whether that condition is applicable, with possible values "`True`", "`False`", or "`Unknown`". |
| `lastProbeTime`      | Timestamp of when the Pod condition was last probed.         |
| `lastTransitionTime` | Timestamp for when the Pod last transitioned from one status to another. |
| `reason`             | Machine-readable, UpperCamelCase text indicating the reason for the condition's last transition. |
| `message`            | Human-readable message indicating details about the last status transition. |

#### .1. Pod readiness

- Your application can inject extra feedback or signals into PodStatus: *Pod readiness*. To use this, set `readinessGates` in the Pod's `spec` to specify a list of additional conditions that the kubelet evaluates for Pod readiness.

```yaml
kind: Pod
...
spec:
  readinessGates:
    - conditionType: "www.example.com/feature-1"
status:
  conditions:
    - type: Ready                              # a built in PodCondition
      status: "False"
      lastProbeTime: null
      lastTransitionTime: 2018-01-01T00:00:00Z
    - type: "www.example.com/feature-1"        # an extra PodCondition
      status: "False"
      lastProbeTime: null
      lastTransitionTime: 2018-01-01T00:00:00Z
  containerStatuses:
    - containerID: docker://abcd...
      ready: true
...
```

### 7. Init Containers

> specialized containers that run before app containers in a [Pod](https://kubernetes.io/docs/concepts/workloads/pods/). Init containers can contain `utilities or setup scripts` not present in an app image.
>
> Init containers support all the fields and features of app containers, including `resource limits, volumes, and security settings`. 

- Init containers can contain `utilities or custom code for setup` that are not present in an app image. For example, there is no need to make an image `FROM` another image just to use a tool like `sed`, `awk`, `python`, or `dig` during setup.
- The application image builder and deployer roles can work independently without the need to jointly build a single app image.
- Init containers can `run with a different view of the filesystem than app containers in the same Pod`. Consequently, they can be given access to [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) that app containers cannot access.
- Because init containers run to completion before any app containers start, init containers `offer a mechanism to block or delay app container startup until a set of preconditions are met`. Once preconditions are met, all of the app containers in a Pod can start in parallel.
- Init containers can securely run utilities or custom code that would otherwise make an app container image less secure. By keeping unnecessary tools separate you can limit the attack surface of your app container image.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp-pod
  labels:
    app: myapp
spec:
  containers:
  - name: myapp-container
    image: busybox:1.28
    command: ['sh', '-c', 'echo The app is running! && sleep 3600']
  initContainers:
  - name: init-myservice
    image: busybox:1.28
    command: ['sh', '-c', "until nslookup myservice.$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace).svc.cluster.local; do echo waiting for myservice; sleep 2; done"]
  - name: init-mydb
    image: busybox:1.28
    command: ['sh', '-c', "until nslookup mydb.$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace).svc.cluster.local; do echo waiting for mydb; sleep 2; done"]
```

### 8. Pod topology spread constraints

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mypod
spec:
  topologySpreadConstraints:
    - maxSkew: <integer>
      topologyKey: <string>
      whenUnsatisfiable: <string>
      labelSelector: <object>
```

define one or multiple `topologySpreadConstraint` to` instruct the kube-scheduler how to place each incoming Pod in relation to the existing Pods` across your cluster.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220505151333060.png)

- **labelSelector:** 用来查找匹配的 Pod，我们能够计算出每个拓扑域中匹配该 label selector 的 Pod 数量，在上图中，假如 label selector 是 `app:foo`，那么 zone1 的匹配个数为 2， zone2 的匹配个数为 0。

- **topologyKey:** 是 Node label 的 key，如果两个 Node 的 label 同时具有该 key 并且 label 值相同，就说它们在同一个拓扑域。在上图中，指定 topologyKey 为 zone， 具有 `zone=zone1` 标签的 Node 被分在一个拓扑域，具有 `zone=zone2` 标签的 Node 被分在另一个拓扑域。

- **maxSkew:** 描述了 Pod 在不同拓扑域中*不均匀分布的最大程度*，*maxSkew 的取值必须大于 0*。每个拓扑域都有一个 skew，计算的公式是: `skew[i] = 拓扑域[i]中匹配的 Pod 个数 - min{其他拓扑域中匹配的 Pod 个数}`。在上图中，我们新建一个带有 `app=foo` 标签的 Pod：

- - 如果该 Pod 被调度到 zone1，那么 zone1 中 Node 的 skew 值变为 3，zone2 中 Node 的 skew 值变为 0 (zone1 有 3 个匹配的 Pod，zone2 有 0 个匹配的 Pod)
  - 如果该 Pod 被调度到 zone2，那么 zone1 中 Node 的 skew 值变为 1，zone2 中 Node 的 skew 值变为 0 (zone2 有 1 个匹配的 Pod，拥有全局最小匹配 Pod 数的拓扑域正是 zone2 自己)

- **whenUnsatisfiable:** 描述了如果 Pod 不满足分布约束条件该采取何种策略：

  - **DoNotSchedule** (默认) 告诉调度器不要调度该 Pod，因此也可以叫作硬策略；

  - **ScheduleAnyway** 告诉调度器根据每个 Node 的 skew 值打分排序后仍然调度，因此也可以叫作软策略。


### Resource

- https://toutiao.io/posts/hur1p6x/preview

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernets-pod/  

